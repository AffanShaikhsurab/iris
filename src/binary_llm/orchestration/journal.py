"""Canonical, tamper-evident filesystem journal for registry mutations."""

from __future__ import annotations

import os
import time
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from types import UnionType
from typing import Any, Callable, Mapping, Union, get_args, get_origin, get_type_hints

from binary_llm.domain.errors import RegistryJournalError, Retryability
from binary_llm.domain.identity import sha256_bytes
from binary_llm.domain.serialization import (
    CanonicalSerializationError,
    canonical_json_bytes,
    parse_canonical_json,
)


ZERO_HASH = "0" * 64
JOURNAL_SCHEMA_VERSION = 1


def model_from_dict(model_type: type[Any], value: Mapping[str, Any]) -> Any:
    """Construct a typed dataclass recursively from canonical JSON data."""

    if not is_dataclass(model_type):
        raise TypeError("model_type must be a dataclass type")
    hints = get_type_hints(model_type)
    names = {item.name for item in fields(model_type)}
    if set(value) != names:
        raise ValueError(f"{model_type.__name__} fields do not match schema")
    return model_type(
        **{name: _decode(hints[name], value[name]) for name in names}
    )


def _decode(annotation: Any, value: Any) -> Any:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (Union, UnionType):
        if value is None and type(None) in args:
            return None
        candidates = tuple(item for item in args if item is not type(None))
        for candidate in candidates:
            try:
                return _decode(candidate, value)
            except (TypeError, ValueError):
                continue
        raise ValueError("value does not match union")
    if origin is tuple:
        if not isinstance(value, list):
            raise TypeError("tuple field must be an array")
        item_type = args[0]
        return tuple(_decode(item_type, item) for item in value)
    if origin in (dict, Mapping) or origin is not None and issubclass(origin, Mapping):
        if not isinstance(value, dict):
            raise TypeError("mapping field must be an object")
        value_type = args[1] if len(args) == 2 else Any
        return {str(key): _decode(value_type, item) for key, item in value.items()}
    if annotation is Any:
        return value
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation(value)
    if isinstance(annotation, type) and is_dataclass(annotation):
        if not isinstance(value, dict):
            raise TypeError("dataclass field must be an object")
        return model_from_dict(annotation, value)
    if annotation in (str, int, float, bool):
        if annotation is float and isinstance(value, int) and not isinstance(value, bool):
            return float(value)
        if type(value) is not annotation:
            raise TypeError(f"expected {annotation.__name__}")
    return value


class FilesystemRegistryJournal:
    """One atomically committed canonical file per monotonic sequence number."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        stale_lock_seconds: float = 300.0,
    ) -> None:
        self.root = Path(root)
        self.entries = self.root / "entries"
        self.head_path = self.root / "HEAD.json"
        self.lock_path = self.root / "writer.lock"
        self.stale_lock_seconds = stale_lock_seconds
        self._next_sequence = 0
        self._last_hash = ZERO_HASH

    def replay(self, consume: Callable[[str, str, Mapping[str, Any], Mapping[str, Any]], None]) -> None:
        """Validate the complete chain, then consume records in sequence order."""

        paths = sorted(self.entries.glob("*.json")) if self.entries.exists() else []
        expected_sequence = 0
        previous_hash = ZERO_HASH
        seen: dict[int, str] = {}
        decoded: list[tuple[str, str, Mapping[str, Any], Mapping[str, Any]]] = []
        for path in paths:
            try:
                raw = path.read_bytes()
                envelope = parse_canonical_json(raw)
                if not isinstance(envelope, dict):
                    raise ValueError("entry must be an object")
                required = {
                    "schema_version", "sequence", "previous_hash", "kind", "record_id",
                    "record", "context", "entry_hash",
                }
                if set(envelope) != required:
                    raise ValueError("entry schema is unknown")
                if envelope["schema_version"] != JOURNAL_SCHEMA_VERSION:
                    raise ValueError("entry schema version is unknown")
                sequence = envelope["sequence"]
                if type(sequence) is not int or sequence < 0:
                    raise ValueError("entry sequence is invalid")
                expected_name = f"{sequence:020d}.json"
                if path.name != expected_name:
                    raise ValueError("entry filename and sequence differ")
                body = {key: value for key, value in envelope.items() if key != "entry_hash"}
                entry_hash = sha256_bytes(canonical_json_bytes(body))
                if envelope["entry_hash"] != entry_hash:
                    raise ValueError("entry checksum differs")
                if sequence in seen and seen[sequence] != entry_hash:
                    raise ValueError("duplicate sequence has conflicting content")
                if sequence != expected_sequence:
                    raise ValueError("journal sequence gap or reorder")
                if envelope["previous_hash"] != previous_hash:
                    raise ValueError("journal hash chain differs")
                if not isinstance(envelope["kind"], str) or not isinstance(envelope["record_id"], str):
                    raise ValueError("entry identity is invalid")
                if not isinstance(envelope["record"], dict) or not isinstance(envelope["context"], dict):
                    raise ValueError("entry record or context is invalid")
                seen[sequence] = entry_hash
                decoded.append(
                    (envelope["kind"], envelope["record_id"], envelope["record"], envelope["context"])
                )
                expected_sequence += 1
                previous_hash = entry_hash
            except (OSError, CanonicalSerializationError, TypeError, ValueError) as error:
                raise self._failure(
                    "registry journal is corrupt, truncated, reordered, or unsupported",
                    "registry_journal.replay_failed",
                    path=str(path),
                    sequence=expected_sequence,
                ) from error
        if self.head_path.exists():
            try:
                head = parse_canonical_json(self.head_path.read_bytes())
                expected_head = {
                    "schema_version": JOURNAL_SCHEMA_VERSION,
                    "entry_count": expected_sequence,
                    "last_hash": previous_hash,
                }
                if head != expected_head:
                    raise ValueError("journal head does not match committed entries")
            except (OSError, CanonicalSerializationError, TypeError, ValueError) as error:
                raise self._failure(
                    "registry journal head is corrupt or entries were truncated",
                    "registry_journal.head_mismatch",
                    sequence=expected_sequence,
                ) from error
        elif paths:
            raise self._failure(
                "registry journal entries exist without a commit head",
                "registry_journal.head_missing",
                sequence=expected_sequence,
            )
        for item in decoded:
            consume(*item)
        self._next_sequence = expected_sequence
        self._last_hash = previous_hash

    def append(
        self,
        kind: str,
        record_id: str,
        record: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> bool:
        """Atomically commit one hash-chained entry under an exclusive lock."""

        self.root.mkdir(parents=True, exist_ok=True)
        self.entries.mkdir(parents=True, exist_ok=True)
        descriptor = self._acquire_lock()
        try:
            probe = FilesystemRegistryJournal(
                self.root, stale_lock_seconds=self.stale_lock_seconds
            )
            existing: dict[tuple[str, str], tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
            probe.replay(
                lambda observed_kind, observed_id, observed_record, observed_context:
                existing.__setitem__(
                    (observed_kind, observed_id), (observed_record, observed_context)
                )
            )
            prior = existing.get((kind, record_id))
            if prior is not None:
                if prior == (dict(record), dict(context)):
                    return False
                raise self._failure(
                    "registry record identifier is already committed with different content",
                    "registry_journal.record_conflict",
                    kind=kind,
                    record_id=record_id,
                )
            body = {
                "schema_version": JOURNAL_SCHEMA_VERSION,
                "sequence": probe._next_sequence,
                "previous_hash": probe._last_hash,
                "kind": kind,
                "record_id": record_id,
                "record": dict(record),
                "context": dict(context),
            }
            envelope = {**body, "entry_hash": sha256_bytes(canonical_json_bytes(body))}
            destination = self.entries / f"{probe._next_sequence:020d}.json"
            temporary = self.entries / (
                f".{destination.name}.{os.getpid()}.{time.time_ns()}.tmp"
            )
            try:
                with temporary.open("xb") as stream:
                    stream.write(canonical_json_bytes(envelope))
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, destination)
                self._fsync_directory(self.entries)
                self._atomic_write(
                    self.head_path,
                    canonical_json_bytes(
                        {
                            "schema_version": JOURNAL_SCHEMA_VERSION,
                            "entry_count": probe._next_sequence + 1,
                            "last_hash": envelope["entry_hash"],
                        }
                    ),
                )
            finally:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
            self._next_sequence = probe._next_sequence + 1
            self._last_hash = envelope["entry_hash"]
            return True
        except RegistryJournalError:
            raise
        except OSError as error:
            raise self._failure(
                "registry journal append failed",
                "registry_journal.write_failed",
                sequence=self._next_sequence,
            ) from error
        finally:
            os.close(descriptor)
            try:
                self.lock_path.unlink()
            except FileNotFoundError:
                pass

    def _acquire_lock(self) -> int:
        try:
            return os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            age = max(0.0, time.time() - self.lock_path.stat().st_mtime)
            code = (
                "registry_journal.stale_lock"
                if age >= self.stale_lock_seconds
                else "registry_journal.writer_busy"
            )
            raise self._failure(
                "registry journal writer lock already exists",
                code,
                lock_age_seconds=age,
            ) from error

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
        try:
            with temporary.open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            FilesystemRegistryJournal._fsync_directory(path.parent)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        try:
            descriptor = os.open(directory, flags)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        except OSError:
            pass
        finally:
            os.close(descriptor)

    @staticmethod
    def _failure(message: str, code: str, **context: Any) -> RegistryJournalError:
        return RegistryJournalError(
            message,
            retryability=Retryability.AFTER_REMEDIATION,
            code=code,
            context=context,
        )
