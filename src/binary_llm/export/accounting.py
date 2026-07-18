"""Exact, overflow-safe tensor and whole-artifact byte accounting."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from fractions import Fraction
from math import prod
from types import MappingProxyType
from typing import Mapping

from binary_llm.adapters.contracts import (
    BINARY_BODY_ROLES,
    TensorDescriptor,
    TensorRole,
    TensorScope,
)

DECIMAL_MEGABYTE = 1_000_000
TARGET_BAND_MIN_BYTES = 200 * DECIMAL_MEGABYTE
TARGET_BAND_MAX_BYTES = 300 * DECIMAL_MEGABYTE


class ArtifactPlacement(StrEnum):
    PACKED_ARTIFACT = "packed_artifact"
    REQUIRED_SIDECAR = "required_sidecar"


class TensorContentKind(StrEnum):
    PARAMETER = "parameter"
    SCALE = "scale"
    MASK = "mask"
    BUFFER = "buffer"
    EXCEPTION = "exception"


class ArtifactByteCategory(StrEnum):
    BINARY_PAYLOAD = "binary_payload"
    EMBEDDING = "embedding"
    UNTIED_HEAD = "untied_head"
    TIED_HEAD = "tied_head"
    NORMALIZATION = "normalization"
    BIAS = "bias"
    SCALE = "scale"
    MASK = "mask"
    BUFFER = "buffer"
    EXCLUDED_OTHER = "excluded_other"
    OFFSET = "offset"
    EXCEPTION = "exception"
    METADATA = "metadata"
    ALIGNMENT = "alignment"
    CHECKSUM = "checksum"
    TOKENIZER = "tokenizer"
    CONTAINER = "container"
    INDEX = "index"
    OTHER_REQUIRED = "other_required"


class TargetBandStatus(StrEnum):
    BELOW = "below"
    WITHIN = "within"
    ABOVE = "above"


def _require_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_nonnegative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class TensorLedgerEntry:
    tensor_name: str
    shape: tuple[int, ...]
    parameter_count: int
    semantic_role: TensorRole
    scope: TensorScope
    representation_id: str
    payload_bytes: int
    scale_bytes: int = 0
    offset_bytes: int = 0
    exception_bytes: int = 0
    alignment_bytes: int = 0
    metadata_bytes: int = 0
    checksum_bytes: int = 0
    file_offset: int = 0
    checksum: str = "not-yet-materialized"
    content_kind: TensorContentKind = TensorContentKind.PARAMETER
    counts_toward_original_parameters: bool = True
    tied_to: str | None = None

    def __post_init__(self) -> None:
        _require_text("tensor_name", self.tensor_name)
        _require_text("representation_id", self.representation_id)
        _require_text("checksum", self.checksum)
        if not self.shape or any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0
            for size in self.shape
        ):
            raise ValueError("shape must contain positive integers")
        _require_nonnegative_int("parameter_count", self.parameter_count)
        if self.parameter_count != prod(self.shape):
            raise ValueError("parameter_count must equal the tensor shape product")
        for name in (
            "payload_bytes", "scale_bytes", "offset_bytes", "exception_bytes",
            "alignment_bytes", "metadata_bytes", "checksum_bytes", "file_offset",
        ):
            _require_nonnegative_int(name, getattr(self, name))
        if (self.semantic_role in BINARY_BODY_ROLES) != (
            self.scope is TensorScope.BINARY_BODY
        ):
            raise ValueError("binary scope must agree with the declared semantic role")
        if self.scope is TensorScope.BINARY_BODY and self.content_kind is not TensorContentKind.PARAMETER:
            raise ValueError("binary-body entries must be parameter tensors")
        if self.content_kind is TensorContentKind.MASK and self.semantic_role is not TensorRole.BUFFER:
            raise ValueError("mask content requires the buffer semantic role")
        if self.content_kind is TensorContentKind.SCALE and self.semantic_role not in {
            TensorRole.INPUT_CHANNEL_SCALE, TensorRole.LEARNED_ROW_SCALE,
        }:
            raise ValueError("scale content requires an explicit scale semantic role")
        if self.counts_toward_original_parameters and self.content_kind is not TensorContentKind.PARAMETER:
            raise ValueError("auxiliary scales, masks, and buffers are not original parameters")
        if self.tied_to is not None:
            _require_text("tied_to", self.tied_to)
            if self.tied_to == self.tensor_name:
                raise ValueError("a tensor cannot be tied to itself")
            if self.counts_toward_original_parameters:
                raise ValueError("a tied alias cannot double-count original parameters")

    @property
    def total_bytes(self) -> int:
        return (
            self.payload_bytes + self.scale_bytes + self.offset_bytes
            + self.exception_bytes + self.alignment_bytes
            + self.metadata_bytes + self.checksum_bytes
        )

    @classmethod
    def from_descriptor(
        cls,
        descriptor: TensorDescriptor,
        *,
        payload_bytes: int,
        scale_bytes: int = 0,
        offset_bytes: int = 0,
        exception_bytes: int = 0,
        alignment_bytes: int = 0,
        metadata_bytes: int = 0,
        checksum_bytes: int = 0,
        file_offset: int = 0,
        checksum: str = "not-yet-materialized",
        content_kind: TensorContentKind = TensorContentKind.PARAMETER,
        counts_toward_original_parameters: bool = True,
    ) -> "TensorLedgerEntry":
        return cls(
            tensor_name=descriptor.name,
            shape=descriptor.shape,
            parameter_count=descriptor.parameter_count,
            semantic_role=descriptor.semantic_role,
            scope=descriptor.scope,
            representation_id=descriptor.representation_id,
            payload_bytes=payload_bytes,
            scale_bytes=scale_bytes,
            offset_bytes=offset_bytes,
            exception_bytes=exception_bytes,
            alignment_bytes=alignment_bytes,
            metadata_bytes=metadata_bytes,
            checksum_bytes=checksum_bytes,
            file_offset=file_offset,
            checksum=checksum,
            content_kind=content_kind,
            counts_toward_original_parameters=counts_toward_original_parameters,
            tied_to=descriptor.tied_to,
        )


@dataclass(frozen=True, slots=True)
class ArtifactByteEntry:
    name: str
    category: ArtifactByteCategory
    bytes: int
    placement: ArtifactPlacement

    def __post_init__(self) -> None:
        _require_text("name", self.name)
        _require_nonnegative_int("bytes", self.bytes)


@dataclass(frozen=True, slots=True)
class ArtifactFileEntry:
    path: str
    bytes: int
    checksum: str

    def __post_init__(self) -> None:
        _require_text("path", self.path)
        _require_nonnegative_int("bytes", self.bytes)
        _require_text("checksum", self.checksum)


@dataclass(frozen=True, slots=True)
class ArtifactLedger:
    artifact_id: str
    format_version: str
    source_checkpoint_id: str
    original_parameter_count: int
    tensor_entries: tuple[TensorLedgerEntry, ...]
    non_tensor_entries: tuple[ArtifactByteEntry, ...]
    file_inventory: tuple[ArtifactFileEntry, ...]
    binary_body_parameters: int
    excluded_parameters: int
    auxiliary_tensor_elements: int
    binary_body_bytes: int
    excluded_tensor_bytes: int
    byte_category_totals: Mapping[ArtifactByteCategory, int]
    ideal_binary_payload_bytes: int
    packed_artifact_bytes: int
    required_sidecar_bytes: int
    total_distributable_bytes: int
    ideal_to_packed_delta_bytes: int
    packed_to_distributable_delta_bytes: int
    ideal_to_distributable_delta_bytes: int
    effective_bits_per_original_parameter: Fraction
    target_band_status: TargetBandStatus
    measured_total_distributable_bytes: int | None
    reconciliation_delta_bytes: int | None

    def __post_init__(self) -> None:
        for name in ("artifact_id", "format_version", "source_checkpoint_id"):
            _require_text(name, getattr(self, name))
        for name in (
            "original_parameter_count", "binary_body_parameters", "excluded_parameters",
            "auxiliary_tensor_elements", "binary_body_bytes", "excluded_tensor_bytes",
            "ideal_binary_payload_bytes", "packed_artifact_bytes",
            "required_sidecar_bytes", "total_distributable_bytes",
        ):
            _require_nonnegative_int(name, getattr(self, name))
        expected_categories = set(ArtifactByteCategory)
        if set(self.byte_category_totals) != expected_categories:
            raise ValueError("byte_category_totals must enumerate every byte category")
        frozen = {
            category: self.byte_category_totals[category]
            for category in ArtifactByteCategory
        }
        for category, value in frozen.items():
            _require_nonnegative_int(f"{category.value} bytes", value)
        object.__setattr__(self, "byte_category_totals", MappingProxyType(frozen))
        if self.binary_body_parameters + self.excluded_parameters != self.original_parameter_count:
            raise ValueError("binary and excluded parameter totals must cover the original denominator")
        if self.binary_body_bytes + self.excluded_tensor_bytes != sum(
            item.total_bytes for item in self.tensor_entries
        ):
            raise ValueError("binary and excluded byte totals must cover every tensor byte")
        if sum(frozen.values()) != self.total_distributable_bytes:
            raise ValueError("byte categories must reconcile to total distributable bytes")
        if self.packed_artifact_bytes + self.required_sidecar_bytes != self.total_distributable_bytes:
            raise ValueError("packed and sidecar bytes must reconcile to total distributable bytes")
        expected_bits = Fraction(8 * self.total_distributable_bytes, self.original_parameter_count)
        if self.effective_bits_per_original_parameter != expected_bits:
            raise ValueError("effective bits per parameter must use total distributable bytes")
        expected_delta = (
            None
            if self.measured_total_distributable_bytes is None
            else self.measured_total_distributable_bytes - self.total_distributable_bytes
        )
        if self.reconciliation_delta_bytes != expected_delta:
            raise ValueError("reconciliation delta must compare measured and manifest totals")

    @property
    def target_band_200_300_decimal_mb(self) -> bool:
        return self.target_band_status is TargetBandStatus.WITHIN

    @property
    def accounting_reconciled(self) -> bool:
        return self.reconciliation_delta_bytes == 0

    @property
    def qualification_allowed(self) -> bool:
        return self.accounting_reconciled

    @property
    def effective_bits_per_parameter(self) -> Fraction:
        """Exact alias matching the manifest field named by the design."""
        return self.effective_bits_per_original_parameter

    @property
    def effective_bits_numerator(self) -> int:
        return self.effective_bits_per_original_parameter.numerator

    @property
    def effective_bits_denominator(self) -> int:
        return self.effective_bits_per_original_parameter.denominator

    def with_file_inventory(
        self, file_inventory: tuple[ArtifactFileEntry, ...]
    ) -> "ArtifactLedger":
        _require_unique_names("file paths", tuple(item.path for item in file_inventory))
        measured = sum(item.bytes for item in file_inventory)
        return replace(
            self,
            file_inventory=file_inventory,
            measured_total_distributable_bytes=measured,
            reconciliation_delta_bytes=measured - self.total_distributable_bytes,
        )


def _require_unique_names(label: str, names: tuple[str, ...]) -> None:
    if len(names) != len(set(names)):
        raise ValueError(f"{label} must be unique")


def _payload_category(entry: TensorLedgerEntry) -> ArtifactByteCategory:
    if entry.content_kind is TensorContentKind.SCALE:
        return ArtifactByteCategory.SCALE
    if entry.content_kind is TensorContentKind.MASK:
        return ArtifactByteCategory.MASK
    if entry.content_kind is TensorContentKind.BUFFER:
        return ArtifactByteCategory.BUFFER
    if entry.content_kind is TensorContentKind.EXCEPTION:
        return ArtifactByteCategory.EXCEPTION
    if entry.scope is TensorScope.BINARY_BODY:
        return ArtifactByteCategory.BINARY_PAYLOAD
    if entry.semantic_role is TensorRole.TOKEN_EMBEDDING:
        return ArtifactByteCategory.EMBEDDING
    if entry.semantic_role is TensorRole.LANGUAGE_MODEL_HEAD:
        return (
            ArtifactByteCategory.TIED_HEAD
            if entry.tied_to is not None
            else ArtifactByteCategory.UNTIED_HEAD
        )
    if entry.semantic_role is TensorRole.NORMALIZATION:
        return ArtifactByteCategory.NORMALIZATION
    if entry.semantic_role is TensorRole.BIAS:
        return ArtifactByteCategory.BIAS
    if entry.semantic_role is TensorRole.DECLARED_EXCEPTION:
        return ArtifactByteCategory.EXCEPTION
    return ArtifactByteCategory.EXCLUDED_OTHER


def _category_totals(
    tensors: tuple[TensorLedgerEntry, ...],
    non_tensors: tuple[ArtifactByteEntry, ...],
) -> Mapping[ArtifactByteCategory, int]:
    totals = {category: 0 for category in ArtifactByteCategory}
    for entry in tensors:
        totals[_payload_category(entry)] += entry.payload_bytes
        totals[ArtifactByteCategory.SCALE] += entry.scale_bytes
        totals[ArtifactByteCategory.OFFSET] += entry.offset_bytes
        totals[ArtifactByteCategory.EXCEPTION] += entry.exception_bytes
        totals[ArtifactByteCategory.ALIGNMENT] += entry.alignment_bytes
        totals[ArtifactByteCategory.METADATA] += entry.metadata_bytes
        totals[ArtifactByteCategory.CHECKSUM] += entry.checksum_bytes
    for entry in non_tensors:
        totals[entry.category] += entry.bytes
    return totals


def account_artifact(
    *,
    artifact_id: str,
    format_version: str,
    source_checkpoint_id: str,
    original_parameter_count: int,
    tensor_entries: tuple[TensorLedgerEntry, ...],
    non_tensor_entries: tuple[ArtifactByteEntry, ...] = (),
    file_inventory: tuple[ArtifactFileEntry, ...] = (),
) -> ArtifactLedger:
    """Build a complete manifest using integer sums and an exact rational bit rate."""
    _require_text("artifact_id", artifact_id)
    _require_text("format_version", format_version)
    _require_text("source_checkpoint_id", source_checkpoint_id)
    _require_nonnegative_int("original_parameter_count", original_parameter_count)
    if original_parameter_count == 0:
        raise ValueError("original_parameter_count must be positive")
    if not tensor_entries:
        raise ValueError("tensor_entries must enumerate the artifact tensors")
    _require_unique_names("tensor names", tuple(item.tensor_name for item in tensor_entries))
    _require_unique_names("non-tensor entry names", tuple(item.name for item in non_tensor_entries))
    _require_unique_names("file paths", tuple(item.path for item in file_inventory))

    original_entries = tuple(
        item for item in tensor_entries if item.counts_toward_original_parameters
    )
    counted_parameters = sum(item.parameter_count for item in original_entries)
    if counted_parameters != original_parameter_count:
        raise ValueError(
            "original parameter denominator does not match enumerated original tensors: "
            f"declared={original_parameter_count}, enumerated={counted_parameters}"
        )
    binary_body_parameters = sum(
        item.parameter_count
        for item in original_entries
        if item.scope is TensorScope.BINARY_BODY
    )
    excluded_parameters = counted_parameters - binary_body_parameters
    auxiliary_elements = sum(
        item.parameter_count
        for item in tensor_entries
        if not item.counts_toward_original_parameters
    )
    binary_body_bytes = sum(
        item.total_bytes for item in tensor_entries if item.scope is TensorScope.BINARY_BODY
    )
    excluded_tensor_bytes = sum(
        item.total_bytes for item in tensor_entries if item.scope is TensorScope.EXCLUDED
    )
    tensor_bytes = binary_body_bytes + excluded_tensor_bytes
    packed_overhead = sum(
        item.bytes
        for item in non_tensor_entries
        if item.placement is ArtifactPlacement.PACKED_ARTIFACT
    )
    sidecar_bytes = sum(
        item.bytes
        for item in non_tensor_entries
        if item.placement is ArtifactPlacement.REQUIRED_SIDECAR
    )
    packed_bytes = tensor_bytes + packed_overhead
    total_bytes = packed_bytes + sidecar_bytes
    ideal_bytes = (binary_body_parameters + 7) // 8
    measured = sum(item.bytes for item in file_inventory) if file_inventory else None
    delta = None if measured is None else measured - total_bytes
    if total_bytes < TARGET_BAND_MIN_BYTES:
        band = TargetBandStatus.BELOW
    elif total_bytes <= TARGET_BAND_MAX_BYTES:
        band = TargetBandStatus.WITHIN
    else:
        band = TargetBandStatus.ABOVE

    return ArtifactLedger(
        artifact_id=artifact_id,
        format_version=format_version,
        source_checkpoint_id=source_checkpoint_id,
        original_parameter_count=original_parameter_count,
        tensor_entries=tensor_entries,
        non_tensor_entries=non_tensor_entries,
        file_inventory=file_inventory,
        binary_body_parameters=binary_body_parameters,
        excluded_parameters=excluded_parameters,
        auxiliary_tensor_elements=auxiliary_elements,
        binary_body_bytes=binary_body_bytes,
        excluded_tensor_bytes=excluded_tensor_bytes,
        byte_category_totals=_category_totals(tensor_entries, non_tensor_entries),
        ideal_binary_payload_bytes=ideal_bytes,
        packed_artifact_bytes=packed_bytes,
        required_sidecar_bytes=sidecar_bytes,
        total_distributable_bytes=total_bytes,
        ideal_to_packed_delta_bytes=packed_bytes - ideal_bytes,
        packed_to_distributable_delta_bytes=sidecar_bytes,
        ideal_to_distributable_delta_bytes=total_bytes - ideal_bytes,
        effective_bits_per_original_parameter=Fraction(8 * total_bytes, original_parameter_count),
        target_band_status=band,
        measured_total_distributable_bytes=measured,
        reconciliation_delta_bytes=delta,
    )


ArtifactManifest = ArtifactLedger


__all__ = [
    "DECIMAL_MEGABYTE",
    "TARGET_BAND_MIN_BYTES",
    "TARGET_BAND_MAX_BYTES",
    "ArtifactByteCategory",
    "ArtifactByteEntry",
    "ArtifactFileEntry",
    "ArtifactLedger",
    "ArtifactManifest",
    "ArtifactPlacement",
    "TargetBandStatus",
    "TensorContentKind",
    "TensorLedgerEntry",
    "account_artifact",
]
