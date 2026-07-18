"""Atomic filesystem storage for immutable artifact payloads and references.

This first durable-evidence slice provides process-local locking and atomic
same-filesystem publication. Atomic replacement plus post-write verification
gives reasonable cross-process behavior, but multi-writer recovery and orphan
collection are deliberately deferred to the next hardening slice.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from threading import RLock
from typing import Any, Callable

from binary_llm.domain.errors import ArtifactStoreError, Retryability
from binary_llm.domain.identity import sha256_bytes
from binary_llm.domain.models import ArtifactRef, SUPPORTED_SCHEMA_VERSION
from binary_llm.domain.serialization import (
    CanonicalSerializationError,
    canonical_json_bytes,
    parse_canonical_json,
)


_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ARTIFACT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_ARTIFACT_FIELDS = frozenset(
    {
        "artifact_id",
        "kind",
        "sha256",
        "bytes",
        "media_type",
        "parent_artifact_id",
        "producing_run_id",
        "producing_attempt_id",
        "schema_version",
    }
)
_REFERENCE_FIELDS = frozenset({"schema_version", "artifact", "artifact_ref_sha256"})


class RecoveryIssueKind(StrEnum):
    STALE_TEMP = "stale_temp"
    ORPHAN_OBJECT = "orphan_object"
    ORPHAN_REFERENCE = "orphan_reference"
    INVALID_REFERENCE = "invalid_reference"
    STALE_LOCK = "stale_lock"


@dataclass(frozen=True, slots=True)
class RecoveryIssue:
    kind: RecoveryIssueKind
    relative_path: str
    detail: str


@dataclass(frozen=True, slots=True)
class ArtifactRecoveryReport:
    issues: tuple[RecoveryIssue, ...]


class FilesystemArtifactStore:
    """Content-addressed payload store with append-only artifact references.

    Reference publication is the commit point: objects are published first,
    then one canonical reference record is atomically published per artifact
    ID. A crash can therefore leave an unreferenced content object, but never
    a reference to a partially written object.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        stale_after_seconds: float = 300.0,
    ) -> None:
        self._root = Path(root)
        self._lock = RLock()
        self._writer_lock_path = self._root / "writer.lock"
        self._stale_after_seconds = stale_after_seconds

    @property
    def root(self) -> Path:
        """Return the configured root for inspection and administration."""

        return self._root

    def object_path(self, digest: str) -> Path:
        """Return the deterministic sharded object path after validating digest."""

        self._validate_digest(digest)
        return self._root / "objects" / digest[:2] / digest[2:4] / digest

    def reference_path(self, artifact_id: str) -> Path:
        """Return the hashed, deterministic reference path for an artifact ID."""

        self._validate_artifact_id(artifact_id)
        key = sha256_bytes(artifact_id.encode("utf-8"))
        return self._root / "references" / key[:2] / key[2:4] / f"{key}.json"

    def put_bytes(self, artifact: ArtifactRef, payload: bytes) -> ArtifactRef:
        """Validate and idempotently publish exact payload and reference bytes."""

        self._validate_artifact(artifact)
        if not isinstance(payload, bytes):
            raise TypeError("artifact payload must be bytes")
        observed_digest = sha256_bytes(payload)
        if observed_digest != artifact.sha256:
            raise self._failure(
                "payload digest does not match artifact metadata",
                "artifact_store.payload_digest_mismatch",
                artifact.artifact_id,
                expected=artifact.sha256,
                observed=observed_digest,
            )
        if len(payload) != artifact.bytes:
            raise self._failure(
                "payload size does not match artifact metadata",
                "artifact_store.payload_size_mismatch",
                artifact.artifact_id,
                expected=artifact.bytes,
                observed=len(payload),
            )

        object_path = self.object_path(artifact.sha256)
        reference_path = self.reference_path(artifact.artifact_id)
        artifact_bytes = canonical_json_bytes(artifact)
        reference_bytes = canonical_json_bytes(
            {
                "schema_version": SUPPORTED_SCHEMA_VERSION,
                "artifact": artifact.to_dict(),
                "artifact_ref_sha256": sha256_bytes(artifact_bytes),
            }
        )
        with self._lock:
            lock_descriptor = self._acquire_writer_lock()
            try:
                return self._put_bytes_locked(
                    artifact, payload, object_path, reference_path, reference_bytes
                )
            finally:
                os.close(lock_descriptor)
                try:
                    self._writer_lock_path.unlink()
                except FileNotFoundError:
                    pass

    def _put_bytes_locked(
        self,
        artifact: ArtifactRef,
        payload: bytes,
        object_path: Path,
        reference_path: Path,
        reference_bytes: bytes,
    ) -> ArtifactRef:
        if reference_path.exists():
            existing = self._read_artifact_ref(reference_path, artifact.artifact_id)
            if existing != artifact:
                raise self._failure(
                    "artifact ID is already bound to different immutable metadata",
                    "artifact_store.reference_conflict",
                    artifact.artifact_id,
                )
        self._publish_or_verify(
            object_path,
            payload,
            lambda: self._verify_payload_file(object_path, artifact),
            artifact.artifact_id,
        )
        self._publish_or_verify(
            reference_path,
            reference_bytes,
            lambda: self._verify_reference_file(reference_path, artifact),
            artifact.artifact_id,
        )
        self._verify_payload_file(object_path, artifact)
        self._verify_reference_file(reference_path, artifact)
        return artifact

    def scan_recovery(self) -> ArtifactRecoveryReport:
        """Return a deterministic, non-mutating report of recoverable evidence."""

        issues: list[RecoveryIssue] = []
        now = time.time()
        if self._writer_lock_path.exists():
            age = max(0.0, now - self._writer_lock_path.stat().st_mtime)
            if age >= self._stale_after_seconds:
                issues.append(
                    RecoveryIssue(
                        RecoveryIssueKind.STALE_LOCK,
                        self._relative(self._writer_lock_path),
                        "stale exclusive writer lock",
                    )
                )
        for path in sorted(self._root.rglob("*.tmp")) if self._root.exists() else ():
            if "quarantine" in path.relative_to(self._root).parts:
                continue
            age = max(0.0, now - path.stat().st_mtime)
            if age >= self._stale_after_seconds:
                issues.append(
                    RecoveryIssue(
                        RecoveryIssueKind.STALE_TEMP,
                        self._relative(path),
                        "stale uncommitted temporary file",
                    )
                )
        referenced_digests: set[str] = set()
        reference_paths = (
            sorted((self._root / "references").rglob("*.json"))
            if (self._root / "references").exists()
            else []
        )
        for path in reference_paths:
            try:
                record = parse_canonical_json(path.read_bytes())
                if not isinstance(record, dict):
                    raise ValueError("reference is not an object")
                artifact_data = record.get("artifact")
                if not isinstance(artifact_data, dict):
                    raise ValueError("artifact is not an object")
                artifact_id = artifact_data.get("artifact_id")
                if not isinstance(artifact_id, str):
                    raise ValueError("artifact ID is invalid")
                artifact = self._read_artifact_ref(path, artifact_id)
                referenced_digests.add(artifact.sha256)
                if not self.object_path(artifact.sha256).is_file():
                    issues.append(
                        RecoveryIssue(
                            RecoveryIssueKind.ORPHAN_REFERENCE,
                            self._relative(path),
                            artifact.sha256,
                        )
                    )
            except (ArtifactStoreError, OSError, CanonicalSerializationError, ValueError):
                issues.append(
                    RecoveryIssue(
                        RecoveryIssueKind.INVALID_REFERENCE,
                        self._relative(path),
                        "reference cannot be verified",
                    )
                )
        object_paths = (
            sorted(path for path in (self._root / "objects").rglob("*") if path.is_file())
            if (self._root / "objects").exists()
            else []
        )
        for path in object_paths:
            if _DIGEST_PATTERN.fullmatch(path.name) and path.name not in referenced_digests:
                issues.append(
                    RecoveryIssue(
                        RecoveryIssueKind.ORPHAN_OBJECT,
                        self._relative(path),
                        path.name,
                    )
                )
        return ArtifactRecoveryReport(
            tuple(sorted(issues, key=lambda item: (item.kind.value, item.relative_path)))
        )

    def quarantine_recoverable(
        self,
        report: ArtifactRecoveryReport | None = None,
    ) -> tuple[str, ...]:
        """Atomically quarantine only stale temps, locks, and orphan evidence."""

        selected = report or self.scan_recovery()
        allowed = {
            RecoveryIssueKind.STALE_TEMP,
            RecoveryIssueKind.STALE_LOCK,
            RecoveryIssueKind.ORPHAN_OBJECT,
            RecoveryIssueKind.ORPHAN_REFERENCE,
        }
        moved: list[str] = []
        quarantine_root = self._root / "quarantine"
        for issue in selected.issues:
            if issue.kind not in allowed:
                continue
            source = self._root / Path(issue.relative_path)
            if not source.is_file():
                continue
            destination = quarantine_root / issue.kind.value / Path(issue.relative_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise self._failure(
                    "quarantine destination already exists",
                    "artifact_store.quarantine_conflict",
                    issue.relative_path,
                )
            os.replace(source, destination)
            self._fsync_directory(destination.parent)
            moved.append(issue.relative_path)
        return tuple(sorted(moved))

    def _relative(self, path: Path) -> str:
        return path.relative_to(self._root).as_posix()

    def _acquire_writer_lock(self) -> int:
        self._root.mkdir(parents=True, exist_ok=True)
        try:
            return os.open(self._writer_lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            age = max(0.0, time.time() - self._writer_lock_path.stat().st_mtime)
            code = (
                "artifact_store.stale_lock"
                if age >= self._stale_after_seconds
                else "artifact_store.writer_busy"
            )
            raise self._failure(
                "artifact store writer lock already exists",
                code,
                "writer-lock",
                lock_age_seconds=age,
            ) from error

    def artifact_ref(self, artifact_id: str) -> ArtifactRef:
        """Load and validate canonical typed metadata for one artifact ID."""

        reference_path = self.reference_path(artifact_id)
        with self._lock:
            if not reference_path.is_file():
                raise self._failure(
                    "artifact reference does not exist",
                    "artifact_store.not_found",
                    artifact_id,
                )
            return self._read_artifact_ref(reference_path, artifact_id)

    def resolve(self, artifact_id: str, expected: ArtifactRef | None = None) -> bytes:
        """Resolve bytes after exact metadata, digest, and size verification."""

        self._validate_artifact_id(artifact_id)
        if expected is not None:
            if not isinstance(expected, ArtifactRef):
                raise TypeError("expected must be an ArtifactRef or None")
            self._validate_artifact(expected)
        with self._lock:
            artifact = self.artifact_ref(artifact_id)
            if expected is not None and artifact != expected:
                raise self._failure(
                    "stored artifact metadata does not match the expected reference",
                    "artifact_store.expected_reference_mismatch",
                    artifact_id,
                )
            object_path = self.object_path(artifact.sha256)
            return self._read_verified_payload(object_path, artifact)

    @staticmethod
    def _validate_artifact_id(artifact_id: str) -> None:
        if not isinstance(artifact_id, str) or not _ARTIFACT_ID_PATTERN.fullmatch(artifact_id):
            raise ArtifactStoreError(
                "artifact ID is malformed or unsafe",
                retryability=Retryability.NEVER,
                code="artifact_store.invalid_artifact_id",
                context={"artifact_id": artifact_id if isinstance(artifact_id, str) else None},
            )

    @staticmethod
    def _validate_digest(digest: str) -> None:
        if not isinstance(digest, str) or not _DIGEST_PATTERN.fullmatch(digest):
            raise ArtifactStoreError(
                "artifact digest must be 64 lowercase hexadecimal characters",
                retryability=Retryability.NEVER,
                code="artifact_store.invalid_digest",
            )

    def _validate_artifact(self, artifact: ArtifactRef) -> None:
        if not isinstance(artifact, ArtifactRef):
            raise TypeError("artifact must be an ArtifactRef")
        self._validate_artifact_id(artifact.artifact_id)
        self._validate_digest(artifact.sha256)
        if isinstance(artifact.bytes, bool) or not isinstance(artifact.bytes, int):
            raise self._failure(
                "artifact byte count must be an integer",
                "artifact_store.invalid_reference",
                artifact.artifact_id,
            )
        for linked_id in (
            artifact.parent_artifact_id,
            artifact.producing_run_id,
            artifact.producing_attempt_id,
        ):
            if linked_id is not None and (not isinstance(linked_id, str) or not linked_id.strip()):
                raise self._failure(
                    "artifact metadata contains an invalid linked ID",
                    "artifact_store.invalid_reference",
                    artifact.artifact_id,
                )

    def _read_artifact_ref(self, path: Path, artifact_id: str) -> ArtifactRef:
        try:
            raw = path.read_bytes()
            record = parse_canonical_json(raw)
            if not isinstance(record, dict) or set(record) != _REFERENCE_FIELDS:
                raise ValueError("reference record has an unexpected schema")
            if record.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
                raise ValueError("reference record has an unsupported schema version")
            artifact_record = record.get("artifact")
            if not isinstance(artifact_record, dict) or set(artifact_record) != _ARTIFACT_FIELDS:
                raise ValueError("artifact reference has an unexpected schema")
            integrity = record.get("artifact_ref_sha256")
            self._validate_digest(integrity)
            if sha256_bytes(canonical_json_bytes(artifact_record)) != integrity:
                raise ValueError("artifact reference integrity checksum does not match")
            if isinstance(artifact_record.get("bytes"), bool) or not isinstance(
                artifact_record.get("bytes"), int
            ):
                raise ValueError("reference byte count must be an integer")
            text_fields = ("artifact_id", "kind", "sha256", "media_type")
            if any(not isinstance(artifact_record.get(name), str) for name in text_fields):
                raise ValueError("reference text field has the wrong type")
            optional_fields = (
                "parent_artifact_id",
                "producing_run_id",
                "producing_attempt_id",
            )
            if any(
                value is not None and not isinstance(value, str)
                for value in (artifact_record.get(name) for name in optional_fields)
            ):
                raise ValueError("reference optional field has the wrong type")
            artifact = ArtifactRef(**artifact_record)
            self._validate_artifact(artifact)
            if artifact.artifact_id != artifact_id:
                raise ValueError("reference artifact ID does not match its lookup key")
            return artifact
        except ArtifactStoreError:
            raise
        except (OSError, CanonicalSerializationError, TypeError, ValueError) as error:
            raise self._failure(
                "artifact reference metadata is invalid or corrupt",
                "artifact_store.invalid_reference",
                artifact_id,
            ) from error

    def _read_verified_payload(self, path: Path, artifact: ArtifactRef) -> bytes:
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        total = 0
        try:
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
                    chunks.append(chunk)
                    total += len(chunk)
        except OSError as error:
            raise self._failure(
                "artifact payload is missing or unreadable",
                "artifact_store.payload_unavailable",
                artifact.artifact_id,
            ) from error
        if total != artifact.bytes or digest.hexdigest() != artifact.sha256:
            raise self._failure(
                "artifact payload failed digest or size verification",
                "artifact_store.payload_corrupt",
                artifact.artifact_id,
                expected_digest=artifact.sha256,
                observed_digest=digest.hexdigest(),
                expected_bytes=artifact.bytes,
                observed_bytes=total,
            )
        return b"".join(chunks)

    def _verify_payload_file(self, path: Path, artifact: ArtifactRef) -> None:
        self._read_verified_payload(path, artifact)

    def _verify_reference_file(self, path: Path, artifact: ArtifactRef) -> None:
        observed = self._read_artifact_ref(path, artifact.artifact_id)
        if observed != artifact:
            raise self._failure(
                "artifact ID is already bound to different immutable metadata",
                "artifact_store.reference_conflict",
                artifact.artifact_id,
            )

    def _publish_or_verify(
        self,
        path: Path,
        data: bytes,
        verify: Callable[[], None],
        artifact_id: str,
    ) -> None:
        if path.exists():
            verify()
            return
        try:
            self._atomic_write(path, data)
            verify()
        except ArtifactStoreError:
            raise
        except OSError as error:
            raise self._failure(
                "artifact publication failed",
                "artifact_store.write_failed",
                artifact_id,
            ) from error

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
            ) as stream:
                temporary = stream.name
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            temporary = None
            FilesystemArtifactStore._fsync_directory(path.parent)
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
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
    def _failure(
        message: str,
        code: str,
        artifact_id: str,
        **context: Any,
    ) -> ArtifactStoreError:
        return ArtifactStoreError(
            message,
            retryability=Retryability.AFTER_REMEDIATION,
            code=code,
            affected_ids={"artifact_ids": (artifact_id,)},
            context=context,
        )
