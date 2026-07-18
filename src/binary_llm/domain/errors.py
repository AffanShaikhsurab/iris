"""Stable machine-readable failures for framework boundaries."""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Iterable, Mapping

from .serialization import canonical_json_bytes, parse_canonical_json


_CODE_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]*$")


class Retryability(str, Enum):
    """Whether and under what condition an operation may be retried."""

    NEVER = "never"
    AFTER_REMEDIATION = "after_remediation"
    FROM_CHECKPOINT = "from_checkpoint"
    TRANSIENT = "transient"


def _normalize_affected_ids(
    affected_ids: Mapping[str, Iterable[str]] | None,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    normalized: list[tuple[str, tuple[str, ...]]] = []
    for id_kind, values in (affected_ids or {}).items():
        if not isinstance(id_kind, str) or not id_kind.endswith("_ids"):
            raise ValueError("affected ID keys must be strings ending in '_ids'")
        if isinstance(values, (str, bytes)):
            raise TypeError("affected ID values must be iterables of ID strings")
        unique_values = tuple(sorted(set(values)))
        if any(not isinstance(value, str) or not value for value in unique_values):
            raise ValueError("affected IDs must be non-empty strings")
        normalized.append((id_kind, unique_values))
    return tuple(sorted(normalized))


class FrameworkFailure(Exception):
    """Base class carrying a deterministic public error payload."""

    category = "framework"

    def __init__(
        self,
        message: str,
        *,
        retryability: Retryability,
        code: str | None = None,
        affected_ids: Mapping[str, Iterable[str]] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(message, str) or not message:
            raise ValueError("failure message must be a non-empty string")
        resolved_code = code or self.category
        if not _CODE_PATTERN.fullmatch(resolved_code):
            raise ValueError("failure code must be a stable lowercase machine code")
        if resolved_code != self.category and not resolved_code.startswith(f"{self.category}."):
            raise ValueError(f"failure code must be '{self.category}' or start with '{self.category}.'")
        if not isinstance(retryability, Retryability):
            raise TypeError("retryability must be a Retryability value")
        self.message = message
        self.code = resolved_code
        self.retryability = retryability
        self._affected_ids = _normalize_affected_ids(affected_ids)
        self._context_bytes = canonical_json_bytes(dict(context or {}))
        super().__init__(message)

    @property
    def affected_ids(self) -> dict[str, tuple[str, ...]]:
        return dict(self._affected_ids)

    @property
    def context(self) -> dict[str, Any]:
        value = parse_canonical_json(self._context_bytes)
        if not isinstance(value, dict):
            raise ValueError("failure context must be an object")
        return value

    def to_dict(self) -> dict[str, Any]:
        """Return a versioned payload safe for logs, APIs, and evidence."""

        return {
            "schema_version": 1,
            "error_type": type(self).__name__,
            "code": self.code,
            "message": self.message,
            "retryability": self.retryability.value,
            "affected_ids": {
                key: list(values) for key, values in self._affected_ids
            },
            "context": self.context,
        }

    def to_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())


class ManifestError(FrameworkFailure):
    category = "manifest"


class SealViolation(FrameworkFailure):
    category = "seal"


class ProvenanceViolation(FrameworkFailure):
    category = "provenance"


class AmbiguityBlocked(FrameworkFailure):
    category = "ambiguity"


class NumericalFailure(FrameworkFailure):
    category = "numerical"


class BudgetExceeded(FrameworkFailure):
    category = "budget"


class GateFailure(FrameworkFailure):
    category = "gate"


class ExportError(FrameworkFailure):
    category = "export"


class ParityError(FrameworkFailure):
    category = "parity"


class EvaluationError(FrameworkFailure):
    category = "evaluation"


class DeviceQualificationError(FrameworkFailure):
    category = "device_qualification"


class ReproductionError(FrameworkFailure):
    category = "reproduction"


class ArtifactStoreError(FrameworkFailure):
    """Artifact persistence, metadata, or payload integrity failure."""

    category = "artifact_store"


class RegistryJournalError(FrameworkFailure):
    """Registry journal integrity, locking, or replay failure."""

    category = "registry_journal"
