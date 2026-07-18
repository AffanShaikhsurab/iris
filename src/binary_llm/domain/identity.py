"""Content-addressed identities and immutable schema evolution."""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass
from typing import Any, Mapping

from .serialization import canonical_json_bytes, parse_canonical_json


_KIND_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]*$")
_IDENTITY_SCHEMA = "binary-llm-content-identity/v1"


def sha256_bytes(data: bytes) -> str:
    """Return a lowercase SHA-256 digest for exact bytes."""

    if not isinstance(data, bytes):
        raise TypeError("SHA-256 input must be bytes")
    return hashlib.sha256(data).hexdigest()


def _validate_kind(kind: str) -> None:
    if not _KIND_PATTERN.fullmatch(kind):
        raise ValueError("identity kind must match ^[a-z][a-z0-9_.-]*$")


@dataclass(frozen=True, slots=True)
class ContentIdentity:
    """A stable identity over a canonical record and its representation."""

    kind: str
    sha256: str
    canonical_bytes: bytes

    @property
    def value(self) -> str:
        return f"{self.kind}:sha256:{self.sha256}"


def identify_content(
    record: Any,
    *,
    kind: str,
    representation_fields: Any | None = None,
) -> ContentIdentity:
    """Identify all record fields plus every supplied representation field."""

    _validate_kind(kind)
    envelope = {
        "identity_schema": _IDENTITY_SCHEMA,
        "kind": kind,
        "record": record,
        "representation": representation_fields,
    }
    identity_bytes = canonical_json_bytes(envelope)
    return ContentIdentity(kind, sha256_bytes(identity_bytes), identity_bytes)


def new_unique_id(kind: str) -> str:
    """Allocate a unique opaque run/attempt-style identifier."""

    _validate_kind(kind)
    return f"{kind}_{uuid.uuid4().hex}"


@dataclass(frozen=True, slots=True)
class CanonicalDocument:
    """An immutable canonical record whose original bytes remain available."""

    schema_version: str | int
    record_bytes: bytes
    identity: ContentIdentity
    representation_bytes: bytes | None

    @classmethod
    def create(
        cls,
        record: Mapping[str, Any],
        *,
        kind: str,
        representation_fields: Any | None = None,
    ) -> "CanonicalDocument":
        if "schema_version" not in record:
            raise ValueError("persisted records require an explicit schema_version")
        schema_version = record["schema_version"]
        if not isinstance(schema_version, (str, int)) or isinstance(schema_version, bool):
            raise ValueError("schema_version must be a string or integer")
        record_bytes = canonical_json_bytes(record)
        representation_bytes = (
            None
            if representation_fields is None
            else canonical_json_bytes(representation_fields)
        )
        return cls(
            schema_version=schema_version,
            record_bytes=record_bytes,
            identity=identify_content(
                record,
                kind=kind,
                representation_fields=representation_fields,
            ),
            representation_bytes=representation_bytes,
        )

    @property
    def content_id(self) -> str:
        return self.identity.value

    def record(self) -> dict[str, Any]:
        parsed = parse_canonical_json(self.record_bytes)
        if not isinstance(parsed, dict):
            raise ValueError("canonical document root must be an object")
        return parsed


@dataclass(frozen=True, slots=True)
class SchemaEvolution:
    """Links preserved source bytes to a separately identified new schema."""

    previous: CanonicalDocument
    current: CanonicalDocument


def evolve_document(
    previous: CanonicalDocument,
    migrated_record: Mapping[str, Any],
    *,
    representation_fields: Any | None = None,
) -> SchemaEvolution:
    """Create a new identity while retaining the exact previous document bytes."""

    if "schema_version" not in migrated_record:
        raise ValueError("migrated records require an explicit schema_version")
    if migrated_record["schema_version"] == previous.schema_version:
        raise ValueError("schema evolution must change schema_version")
    if representation_fields is None and previous.representation_bytes is not None:
        representation_fields = parse_canonical_json(previous.representation_bytes)
    current = CanonicalDocument.create(
        migrated_record,
        kind=previous.identity.kind,
        representation_fields=representation_fields,
    )
    if current.identity.sha256 == previous.identity.sha256:
        raise ValueError("schema evolution must create a new content identity")
    return SchemaEvolution(previous=previous, current=current)
