"""Deterministic serialization for persisted framework records."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping


class CanonicalSerializationError(ValueError):
    """Raised when a value cannot be represented by canonical JSON."""


def _normalize(value: Any, path: str = "$") -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _normalize(asdict(value), path)
    if isinstance(value, Enum):
        return _normalize(value.value, path)
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise CanonicalSerializationError(f"{path}: datetime must include a timezone")
        utc_value = value.astimezone(timezone.utc)
        return utc_value.isoformat(timespec="microseconds").replace("+00:00", "Z")
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalSerializationError(f"{path}: object keys must be strings")
            normalized[key] = _normalize(item, f"{path}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalize(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalSerializationError(f"{path}: non-finite numbers are not JSON-compatible")
        return value
    raise CanonicalSerializationError(
        f"{path}: unsupported canonical JSON value {type(value).__name__}"
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Return UTF-8 canonical JSON with deterministic keys and no whitespace."""

    normalized = _normalize(value)
    try:
        text = json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise CanonicalSerializationError(str(error)) from error
    return text.encode("utf-8")


def canonical_json(value: Any) -> str:
    """Return canonical JSON text."""

    return canonical_json_bytes(value).decode("utf-8")


def parse_canonical_json(data: bytes) -> Any:
    """Parse bytes only when they are already in the canonical encoding."""

    if not isinstance(data, bytes):
        raise TypeError("canonical JSON input must be bytes")
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CanonicalSerializationError("invalid UTF-8 canonical JSON") from error
    if canonical_json_bytes(value) != data:
        raise CanonicalSerializationError("JSON bytes are valid but not canonical")
    return value
