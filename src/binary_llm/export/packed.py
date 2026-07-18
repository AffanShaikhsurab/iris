"""Deterministic direct export of training-checkpoint binary tensors."""

from __future__ import annotations

import math
import os
import re
import struct
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from binary_llm.adapters.contracts import BINARY_BODY_ROLES, TensorRole, TensorScope
from binary_llm.domain import (
    ExportError,
    FormatSpec,
    Retryability,
    canonical_json_bytes,
    parse_canonical_json,
    sha256_bytes,
)
from binary_llm.math import ZeroSignRule

from .accounting import (
    ArtifactByteCategory,
    ArtifactByteEntry,
    ArtifactFileEntry,
    ArtifactLedger,
    ArtifactPlacement,
    TensorLedgerEntry,
    account_artifact,
)

MAGIC = b"BLLMPK1\0"
_HEADER_BYTES = 16
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SCALE_FORMATS = {"float16": "e", "float32": "f", "float64": "d"}
_ENDIAN_PREFIXES = {"little": "<", "big": ">"}


class SourceArtifactKind(StrEnum):
    TRAINING_CHECKPOINT = "training_checkpoint"
    QUANTIZED_DEPLOYMENT = "quantized_deployment"

@dataclass(frozen=True, slots=True)
class PackedTensorSource:
    """One final binary matrix and its already-merged inference row scales."""

    name: str
    semantic_role: TensorRole
    representation_id: str
    latent_weight: Tensor
    merged_scale: Tensor

    def __post_init__(self) -> None:
        if not self.name or not self.representation_id:
            raise ValueError("tensor name and representation_id must be explicit")
        if self.semantic_role not in BINARY_BODY_ROLES:
            raise ValueError("packed export accepts only declared Binary Body roles")
        if not isinstance(self.latent_weight, Tensor) or not self.latent_weight.is_floating_point():
            raise TypeError("latent_weight must be a floating torch.Tensor")
        if self.latent_weight.ndim != 2 or min(self.latent_weight.shape) < 1:
            raise ValueError("latent_weight must have non-empty shape [rows, columns]")
        if not isinstance(self.merged_scale, Tensor) or not self.merged_scale.is_floating_point():
            raise TypeError("merged_scale must be a floating torch.Tensor")
        if self.merged_scale.shape != (self.latent_weight.shape[0],):
            raise ValueError("merged_scale must contain one value per output row")
        if not torch.isfinite(self.latent_weight).all().item():
            raise ValueError("latent_weight must be finite")
        if not torch.isfinite(self.merged_scale).all().item():
            raise ValueError("merged_scale must be finite")
        object.__setattr__(self, "latent_weight", self.latent_weight.detach().cpu().contiguous().clone())
        object.__setattr__(self, "merged_scale", self.merged_scale.detach().cpu().contiguous().clone())


@dataclass(frozen=True, slots=True)
class IdentifiedCheckpoint:
    checkpoint_id: str
    checkpoint_sha256: str
    tensors: tuple[PackedTensorSource, ...]
    source_kind: SourceArtifactKind = SourceArtifactKind.TRAINING_CHECKPOINT

    def __post_init__(self) -> None:
        if not self.checkpoint_id:
            raise ValueError("checkpoint_id must be explicit")
        if not _SHA256_PATTERN.fullmatch(self.checkpoint_sha256):
            raise ValueError("checkpoint_sha256 must be a lowercase SHA-256 digest")
        object.__setattr__(self, "source_kind", SourceArtifactKind(self.source_kind))
        if not self.tensors:
            raise ValueError("checkpoint must enumerate at least one export tensor")
        names = tuple(item.name for item in self.tensors)
        if len(names) != len(set(names)):
            raise ValueError("checkpoint export tensor names must be unique")


@dataclass(frozen=True, slots=True)
class PackedSignRows:
    data: bytes
    rows: int
    columns: int
    row_payload_bytes: int
    row_stride_bytes: int
    row_padding_bits: int


@dataclass(frozen=True, slots=True)
class PackedArtifactRef:
    artifact_id: str
    path: Path
    sha256: str
    bytes: int
    source_checkpoint_id: str
    source_checkpoint_sha256: str
    format_spec: FormatSpec
    exporter_revision: str
    runtime_revision: str
    build_flags: tuple[str, ...]
    scale_tolerance: float
    ledger: ArtifactLedger | None = None


@dataclass(frozen=True, slots=True)
class ExportProof:
    artifact_id: str
    artifact_sha256: str
    source_checkpoint_id: str
    format_version: str
    signs_verified: int
    scales_verified: int
    bytes_verified: int
    tensor_checksums: Mapping[str, str]
    exact_byte_match: bool
    accounting_reconciled: bool


@dataclass(frozen=True, slots=True)
class PackedExportPlan:
    artifact_id: str
    checkpoint: IdentifiedCheckpoint
    format_spec: FormatSpec
    exporter_revision: str
    runtime_revision: str
    build_flags: tuple[str, ...]
    scale_tolerance: float
    metadata: Mapping[str, Any]
    container_bytes: bytes
    ledger: ArtifactLedger

def _alignment_padding(offset: int, alignment: int) -> int:
    return (-offset) % alignment


def _sign_values(weight: Tensor, zero_rule: ZeroSignRule) -> Tensor:
    zeros = weight == 0
    if zero_rule is ZeroSignRule.ERROR and zeros.any().item():
        raise ValueError("zero_sign_rule=error cannot export an exact zero")
    positive = weight > 0
    if zero_rule is ZeroSignRule.POSITIVE:
        positive = positive | zeros
    return torch.where(positive, 1, -1).to(torch.int8)


def pack_sign_rows(
    weight: Tensor,
    *,
    bit_order: str,
    zero_sign_rule: ZeroSignRule | str,
    row_alignment: int,
) -> PackedSignRows:
    """Pack +1 as one and -1 as zero, padding each row independently with zeroes."""

    if bit_order not in {"lsb0", "msb0"}:
        raise ValueError("bit_order must be lsb0 or msb0")
    if isinstance(row_alignment, bool) or not isinstance(row_alignment, int) or row_alignment < 1:
        raise ValueError("row_alignment must be a positive integer")
    if not isinstance(weight, Tensor) or weight.ndim != 2:
        raise ValueError("weight must be a rank-two tensor")
    signs = _sign_values(weight, ZeroSignRule(zero_sign_rule))
    rows, columns = signs.shape
    row_payload_bytes = (columns + 7) // 8
    row_stride_bytes = row_payload_bytes + _alignment_padding(row_payload_bytes, row_alignment)
    output = bytearray(rows * row_stride_bytes)
    for row in range(rows):
        base = row * row_stride_bytes
        for column in range(columns):
            if int(signs[row, column].item()) > 0:
                bit = column % 8
                shift = bit if bit_order == "lsb0" else 7 - bit
                output[base + column // 8] |= 1 << shift
    return PackedSignRows(
        bytes(output), rows, columns, row_payload_bytes, row_stride_bytes,
        row_payload_bytes * 8 - columns,
    )


def unpack_sign_rows(
    data: bytes,
    *,
    rows: int,
    columns: int,
    bit_order: str,
    row_stride_bytes: int,
) -> Tensor:
    """Decode row-major packed signs without interpreting padding as tensor values."""

    if bit_order not in {"lsb0", "msb0"}:
        raise ValueError("bit_order must be lsb0 or msb0")
    row_payload_bytes = (columns + 7) // 8
    if rows < 1 or columns < 1 or row_stride_bytes < row_payload_bytes:
        raise ValueError("invalid packed row dimensions")
    if len(data) != rows * row_stride_bytes:
        raise ValueError("packed sign byte count does not match row dimensions")
    result = torch.empty((rows, columns), dtype=torch.int8)
    for row in range(rows):
        base = row * row_stride_bytes
        for column in range(columns):
            bit = column % 8
            shift = bit if bit_order == "lsb0" else 7 - bit
            result[row, column] = 1 if data[base + column // 8] & (1 << shift) else -1
        if any(data[base + row_payload_bytes : base + row_stride_bytes]):
            raise ValueError("row alignment padding must contain only zero bytes")
        unused = row_payload_bytes * 8 - columns
        if unused:
            final = data[base + row_payload_bytes - 1]
            used_mask = (1 << (8 - unused)) - 1 if bit_order == "lsb0" else (0xFF << unused) & 0xFF
            if final & (~used_mask & 0xFF):
                raise ValueError("row padding bits must be zero")
    return result


def _scale_layout(dtype: str, endianness: str) -> tuple[str, int]:
    try:
        code = _SCALE_FORMATS[dtype]
        prefix = _ENDIAN_PREFIXES[endianness]
    except KeyError as error:
        raise ValueError("scale dtype/endianness is unsupported") from error
    return prefix + code, struct.calcsize(code)


def _encode_scales(scales: Tensor, dtype: str, endianness: str) -> bytes:
    layout, _ = _scale_layout(dtype, endianness)
    return b"".join(struct.pack(layout, float(value)) for value in scales.tolist())


def _decode_scales(data: bytes, count: int, dtype: str, endianness: str) -> tuple[float, ...]:
    layout, width = _scale_layout(dtype, endianness)
    if len(data) != count * width:
        raise ValueError("scale byte count does not match row count")
    return tuple(struct.unpack_from(layout, data, index * width)[0] for index in range(count))

class DeterministicPackedExporter:
    """Plan, atomically write, reopen, and exhaustively verify packed artifacts."""

    def __init__(
        self,
        *,
        exporter_revision: str,
        runtime_revision: str,
        build_flags: tuple[str, ...],
    ) -> None:
        if not exporter_revision or not runtime_revision:
            raise ValueError("exporter and runtime revisions must be explicit")
        if any(not isinstance(flag, str) or not flag for flag in build_flags):
            raise ValueError("build flags must be explicit strings")
        self.exporter_revision = exporter_revision
        self.runtime_revision = runtime_revision
        self.build_flags = tuple(build_flags)

    @staticmethod
    def _validate_format(spec: FormatSpec) -> None:
        if spec.bit_order not in {"lsb0", "msb0"}:
            raise ValueError("format bit_order must be lsb0 or msb0")
        if spec.row_order != "row_major":
            raise ValueError("packed exporter supports only row_major order")
        ZeroSignRule(spec.zero_sign_rule)
        _scale_layout(spec.scale_dtype, spec.scale_endianness)
        if spec.metadata_encoding not in {"json", "canonical_json"}:
            raise ValueError("metadata encoding must declare canonical JSON")

    def plan(
        self,
        checkpoint: IdentifiedCheckpoint,
        format_spec: FormatSpec,
        *,
        scale_tolerance: float,
        artifact_id: str | None = None,
    ) -> PackedExportPlan:
        if not isinstance(checkpoint, IdentifiedCheckpoint):
            raise TypeError("checkpoint must be an IdentifiedCheckpoint")
        if checkpoint.source_kind is not SourceArtifactKind.TRAINING_CHECKPOINT:
            raise ExportError(
                "deployment formats must be derived directly from a training checkpoint",
                retryability=Retryability.NEVER,
                code="export.non_training_source",
                affected_ids={"checkpoint_ids": (checkpoint.checkpoint_id,)},
            )
        if not isinstance(format_spec, FormatSpec):
            raise TypeError("format_spec must be a FormatSpec")
        self._validate_format(format_spec)
        if not isinstance(scale_tolerance, (int, float)) or isinstance(scale_tolerance, bool):
            raise TypeError("scale_tolerance must be a preregistered number")
        tolerance = float(scale_tolerance)
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("scale_tolerance must be finite and non-negative")

        zero_rule = ZeroSignRule(format_spec.zero_sign_rule)
        data = bytearray()
        records: list[dict[str, Any]] = []
        ledger_parts: list[dict[str, Any]] = []
        for source in sorted(checkpoint.tensors, key=lambda item: item.name):
            if source.representation_id not in format_spec.allowed_representation_ids:
                raise ExportError(
                    "tensor representation is not allowed by the format",
                    retryability=Retryability.NEVER,
                    code="export.unsupported_representation",
                    affected_ids={"tensor_ids": (source.name,)},
                    context={"representation_id": source.representation_id},
                )
            packed = pack_sign_rows(
                source.latent_weight,
                bit_order=format_spec.bit_order,
                zero_sign_rule=zero_rule,
                row_alignment=format_spec.row_alignment,
            )
            scale_bytes = _encode_scales(
                source.merged_scale, format_spec.scale_dtype, format_spec.scale_endianness
            )
            decoded_scales = _decode_scales(
                scale_bytes, packed.rows, format_spec.scale_dtype, format_spec.scale_endianness
            )
            max_error = max(
                abs(actual - float(expected))
                for actual, expected in zip(decoded_scales, source.merged_scale.tolist(), strict=True)
            )
            if max_error > tolerance:
                raise ExportError(
                    "encoded scale exceeds the preregistered precision tolerance",
                    retryability=Retryability.NEVER,
                    code="export.scale_tolerance",
                    affected_ids={"tensor_ids": (source.name,)},
                    context={"maximum_error": max_error, "scale_tolerance": tolerance},
                )

            tensor_padding = _alignment_padding(len(data), format_spec.row_alignment)
            data.extend(b"\0" * tensor_padding)
            sign_offset = len(data)
            data.extend(packed.data)
            scale_padding = _alignment_padding(len(data), format_spec.row_alignment)
            data.extend(b"\0" * scale_padding)
            scale_offset = len(data)
            data.extend(scale_bytes)
            sign_checksum = sha256_bytes(packed.data)
            scale_checksum = sha256_bytes(scale_bytes)
            checksum_offset = len(data)
            data.extend(bytes.fromhex(sign_checksum))
            data.extend(bytes.fromhex(scale_checksum))
            tensor_checksum = sha256_bytes(packed.data + scale_bytes)
            row_alignment_bytes = packed.rows * (
                packed.row_stride_bytes - packed.row_payload_bytes
            )
            records.append({
                "name": source.name,
                "semantic_role": source.semantic_role.value,
                "representation_id": source.representation_id,
                "shape": [packed.rows, packed.columns],
                "positive_bit": 1,
                "sign_offset": sign_offset,
                "sign_storage_bytes": len(packed.data),
                "row_payload_bytes": packed.row_payload_bytes,
                "row_stride_bytes": packed.row_stride_bytes,
                "row_padding_bits": packed.row_padding_bits,
                "row_alignment_padding_bytes": packed.row_stride_bytes - packed.row_payload_bytes,
                "scale_offset": scale_offset,
                "scale_bytes": len(scale_bytes),
                "scale_alignment_padding_bytes": scale_padding,
                "checksum_offset": checksum_offset,
                "checksum_bytes": 64,
                "sign_checksum_sha256": sign_checksum,
                "scale_checksum_sha256": scale_checksum,
                "tensor_checksum_sha256": tensor_checksum,
            })
            ledger_parts.append({
                "source": source,
                "packed": packed,
                "scale_bytes": scale_bytes,
                "alignment_bytes": tensor_padding + row_alignment_bytes + scale_padding,
                "sign_offset": sign_offset,
                "checksum": tensor_checksum,
            })

        identity_record = {
            "checkpoint_id": checkpoint.checkpoint_id,
            "checkpoint_sha256": checkpoint.checkpoint_sha256,
            "format": format_spec.to_dict(),
            "exporter_revision": self.exporter_revision,
            "runtime_revision": self.runtime_revision,
            "build_flags": list(self.build_flags),
            "scale_tolerance": tolerance,
            "tensor_checksums": [
                {"name": item["name"], "sha256": item["tensor_checksum_sha256"]}
                for item in records
            ],
        }
        resolved_artifact_id = artifact_id or f"packed-{sha256_bytes(canonical_json_bytes(identity_record))}"
        if not resolved_artifact_id:
            raise ValueError("artifact_id must be non-empty")
        metadata = {
            "schema_version": 1,
            "artifact_id": resolved_artifact_id,
            "source": {
                "kind": checkpoint.source_kind.value,
                "checkpoint_id": checkpoint.checkpoint_id,
                "checkpoint_sha256": checkpoint.checkpoint_sha256,
            },
            "format": format_spec.to_dict(),
            "exporter_revision": self.exporter_revision,
            "runtime_revision": self.runtime_revision,
            "build_flags": list(self.build_flags),
            "scale_tolerance": tolerance,
            "tensor_count": len(records),
            "tensors": records,
        }
        metadata_bytes = canonical_json_bytes(metadata)
        prefix_padding = _alignment_padding(
            _HEADER_BYTES + len(metadata_bytes), format_spec.row_alignment
        )
        data_offset = _HEADER_BYTES + len(metadata_bytes) + prefix_padding
        container = (
            MAGIC
            + struct.pack("<Q", len(metadata_bytes))
            + metadata_bytes
            + b"\0" * prefix_padding
            + bytes(data)
        )
        tensor_entries = tuple(
            TensorLedgerEntry(
                tensor_name=part["source"].name,
                shape=tuple(part["source"].latent_weight.shape),
                parameter_count=part["source"].latent_weight.numel(),
                semantic_role=part["source"].semantic_role,
                scope=TensorScope.BINARY_BODY,
                representation_id=part["source"].representation_id,
                payload_bytes=part["packed"].rows * part["packed"].row_payload_bytes,
                scale_bytes=len(part["scale_bytes"]),
                alignment_bytes=part["alignment_bytes"],
                checksum_bytes=64,
                file_offset=data_offset + part["sign_offset"],
                checksum=part["checksum"],
            )
            for part in ledger_parts
        )
        non_tensor_entries = (
            ArtifactByteEntry(
                "container_header", ArtifactByteCategory.CONTAINER, _HEADER_BYTES,
                ArtifactPlacement.PACKED_ARTIFACT,
            ),
            ArtifactByteEntry(
                "canonical_metadata", ArtifactByteCategory.METADATA, len(metadata_bytes),
                ArtifactPlacement.PACKED_ARTIFACT,
            ),
            ArtifactByteEntry(
                "metadata_alignment", ArtifactByteCategory.ALIGNMENT, prefix_padding,
                ArtifactPlacement.PACKED_ARTIFACT,
            ),
        )
        artifact_hash = sha256_bytes(container)
        ledger = account_artifact(
            artifact_id=resolved_artifact_id,
            format_version=format_spec.version,
            source_checkpoint_id=checkpoint.checkpoint_id,
            original_parameter_count=sum(item.latent_weight.numel() for item in checkpoint.tensors),
            tensor_entries=tensor_entries,
            non_tensor_entries=non_tensor_entries,
            file_inventory=(ArtifactFileEntry("model.bllmp", len(container), artifact_hash),),
        )
        if not ledger.qualification_allowed or ledger.total_distributable_bytes != len(container):
            raise ExportError(
                "planned packed bytes do not reconcile with the artifact ledger",
                retryability=Retryability.NEVER,
                code="export.byte_mismatch",
                affected_ids={"artifact_ids": (resolved_artifact_id,)},
            )
        return PackedExportPlan(
            resolved_artifact_id, checkpoint, format_spec, self.exporter_revision,
            self.runtime_revision, self.build_flags, tolerance, metadata,
            container, ledger,
        )

    @staticmethod
    def _quarantine(path: Path) -> Path | None:
        if not path.exists() or not path.is_file():
            return None
        content = path.read_bytes()
        directory = path.parent / "quarantine"
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{path.name}.{sha256_bytes(content)[:16]}.quarantine"
        if target.exists():
            path.unlink()
        else:
            os.replace(path, target)
        return target

    @staticmethod
    def _artifact_from_plan(path: Path, plan: PackedExportPlan) -> PackedArtifactRef:
        return PackedArtifactRef(
            artifact_id=plan.artifact_id,
            path=path,
            sha256=sha256_bytes(plan.container_bytes),
            bytes=len(plan.container_bytes),
            source_checkpoint_id=plan.checkpoint.checkpoint_id,
            source_checkpoint_sha256=plan.checkpoint.checkpoint_sha256,
            format_spec=plan.format_spec,
            exporter_revision=plan.exporter_revision,
            runtime_revision=plan.runtime_revision,
            build_flags=plan.build_flags,
            scale_tolerance=plan.scale_tolerance,
            ledger=plan.ledger,
        )

    def export(
        self, plan: PackedExportPlan, destination: str | os.PathLike[str]
    ) -> PackedArtifactRef:
        if not isinstance(plan, PackedExportPlan):
            raise TypeError("plan must be a PackedExportPlan")
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            existing = self._artifact_from_plan(path, plan)
            try:
                self.verify(existing, plan.checkpoint)
                return existing
            except ExportError:
                pass

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b", prefix=f".{path.name}.", suffix=".partial",
                dir=path.parent, delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(plan.container_bytes)
                temporary.flush()
                os.fsync(temporary.fileno())
            temporary_ref = self._artifact_from_plan(temporary_path, plan)
            self._verify_impl(temporary_ref, plan.checkpoint)
            os.replace(temporary_path, path)
            temporary_path = None
            return self._artifact_from_plan(path, plan)
        except Exception as error:
            quarantined = None if temporary_path is None else self._quarantine(temporary_path)
            if isinstance(error, ExportError):
                raise
            raise ExportError(
                "packed export did not complete and its partial output was quarantined",
                retryability=Retryability.AFTER_REMEDIATION,
                code="export.partial_output",
                affected_ids={"artifact_ids": (plan.artifact_id,)},
                context={"quarantine_path": None if quarantined is None else str(quarantined)},
            ) from error

    @staticmethod
    def _read_container(path: Path) -> tuple[bytes, Mapping[str, Any], int]:
        content = path.read_bytes()
        if len(content) < _HEADER_BYTES or content[:8] != MAGIC:
            raise ValueError("packed artifact has an invalid or partial header")
        metadata_length = struct.unpack_from("<Q", content, 8)[0]
        metadata_end = _HEADER_BYTES + metadata_length
        if metadata_end > len(content):
            raise ValueError("packed artifact metadata is truncated")
        metadata = parse_canonical_json(content[_HEADER_BYTES:metadata_end])
        if not isinstance(metadata, dict):
            raise ValueError("packed artifact metadata must be an object")
        format_record = metadata.get("format")
        if not isinstance(format_record, dict):
            raise ValueError("packed artifact format metadata is missing")
        alignment = format_record.get("row_alignment")
        if isinstance(alignment, bool) or not isinstance(alignment, int) or alignment < 1:
            raise ValueError("packed artifact alignment metadata is invalid")
        padding = _alignment_padding(metadata_end, alignment)
        if metadata_end + padding > len(content):
            raise ValueError("packed artifact metadata alignment is truncated")
        if any(content[metadata_end : metadata_end + padding]):
            raise ValueError("metadata alignment padding must contain only zero bytes")
        return content, metadata, metadata_end + padding

    def reopen(self, path: str | os.PathLike[str]) -> PackedArtifactRef:
        artifact_path = Path(path)
        content, metadata, _ = self._read_container(artifact_path)
        format_record = dict(metadata["format"])
        format_record["allowed_representation_ids"] = tuple(
            format_record["allowed_representation_ids"]
        )
        format_spec = FormatSpec(**format_record)
        source = metadata.get("source")
        if not isinstance(source, dict):
            raise ValueError("packed artifact source metadata is missing")
        build_flags = metadata.get("build_flags")
        if not isinstance(build_flags, list) or any(not isinstance(item, str) for item in build_flags):
            raise ValueError("packed artifact build flags are invalid")
        return PackedArtifactRef(
            artifact_id=str(metadata["artifact_id"]),
            path=artifact_path,
            sha256=sha256_bytes(content),
            bytes=len(content),
            source_checkpoint_id=str(source["checkpoint_id"]),
            source_checkpoint_sha256=str(source["checkpoint_sha256"]),
            format_spec=format_spec,
            exporter_revision=str(metadata["exporter_revision"]),
            runtime_revision=str(metadata["runtime_revision"]),
            build_flags=tuple(build_flags),
            scale_tolerance=float(metadata["scale_tolerance"]),
        )

    def _verify_impl(
        self, artifact: PackedArtifactRef, checkpoint: IdentifiedCheckpoint
    ) -> ExportProof:
        if checkpoint.source_kind is not SourceArtifactKind.TRAINING_CHECKPOINT:
            raise ValueError("verification source must be a training checkpoint")
        if artifact.source_checkpoint_id != checkpoint.checkpoint_id or (
            artifact.source_checkpoint_sha256 != checkpoint.checkpoint_sha256
        ):
            raise ValueError("artifact source identity does not match the training checkpoint")
        content, metadata, data_offset = self._read_container(artifact.path)
        actual_hash = sha256_bytes(content)
        if len(content) != artifact.bytes or actual_hash != artifact.sha256:
            raise ValueError("artifact reference size or SHA-256 does not match reopened bytes")
        expected = self.plan(
            checkpoint,
            artifact.format_spec,
            scale_tolerance=artifact.scale_tolerance,
            artifact_id=artifact.artifact_id,
        )
        if content != expected.container_bytes:
            limit = min(len(content), len(expected.container_bytes))
            first_difference = next(
                (index for index in range(limit) if content[index] != expected.container_bytes[index]),
                limit,
            )
            raise ExportError(
                "reopened artifact differs from deterministic checkpoint export bytes",
                retryability=Retryability.AFTER_REMEDIATION,
                code="export.byte_mismatch",
                affected_ids={"artifact_ids": (artifact.artifact_id,)},
                context={
                    "actual_bytes": len(content),
                    "expected_bytes": len(expected.container_bytes),
                    "first_difference_offset": first_difference,
                },
            )
        if metadata != expected.metadata:
            raise ValueError("reopened canonical metadata does not match the export plan")

        signs_verified = 0
        scales_verified = 0
        tensor_checksums: dict[str, str] = {}
        sources = sorted(checkpoint.tensors, key=lambda item: item.name)
        tensor_records = metadata.get("tensors")
        if not isinstance(tensor_records, list) or len(tensor_records) != len(sources):
            raise ValueError("tensor metadata does not cover the checkpoint exactly")
        for source, record in zip(sources, tensor_records, strict=True):
            if not isinstance(record, dict) or record.get("name") != source.name:
                raise ValueError("tensor metadata ordering or identity is invalid")
            rows, columns = source.latent_weight.shape
            sign_start = data_offset + int(record["sign_offset"])
            sign_end = sign_start + int(record["sign_storage_bytes"])
            sign_bytes = content[sign_start:sign_end]
            decoded_signs = unpack_sign_rows(
                sign_bytes,
                rows=rows,
                columns=columns,
                bit_order=artifact.format_spec.bit_order,
                row_stride_bytes=int(record["row_stride_bytes"]),
            )
            expected_signs = _sign_values(
                source.latent_weight, ZeroSignRule(artifact.format_spec.zero_sign_rule)
            )
            if not torch.equal(decoded_signs, expected_signs):
                raise ValueError(f"decoded signs differ for tensor {source.name}")
            scale_start = data_offset + int(record["scale_offset"])
            scale_end = scale_start + int(record["scale_bytes"])
            scale_bytes = content[scale_start:scale_end]
            decoded_scales = _decode_scales(
                scale_bytes, rows, artifact.format_spec.scale_dtype,
                artifact.format_spec.scale_endianness,
            )
            for actual, source_value in zip(
                decoded_scales, source.merged_scale.tolist(), strict=True
            ):
                if abs(actual - float(source_value)) > artifact.scale_tolerance:
                    raise ValueError(f"decoded scales differ for tensor {source.name}")
            sign_checksum = sha256_bytes(sign_bytes)
            scale_checksum = sha256_bytes(scale_bytes)
            checksum_start = data_offset + int(record["checksum_offset"])
            checksum_end = checksum_start + int(record["checksum_bytes"])
            stored_checksums = content[checksum_start:checksum_end]
            if stored_checksums != bytes.fromhex(sign_checksum) + bytes.fromhex(scale_checksum):
                raise ValueError(f"stored checksums differ for tensor {source.name}")
            if sign_checksum != record["sign_checksum_sha256"] or (
                scale_checksum != record["scale_checksum_sha256"]
            ):
                raise ValueError(f"metadata checksums differ for tensor {source.name}")
            tensor_checksum = sha256_bytes(sign_bytes + scale_bytes)
            if tensor_checksum != record["tensor_checksum_sha256"]:
                raise ValueError(f"tensor checksum differs for tensor {source.name}")
            tensor_checksums[source.name] = tensor_checksum
            signs_verified += source.latent_weight.numel()
            scales_verified += source.merged_scale.numel()

        if not expected.ledger.accounting_reconciled:
            raise ValueError("reopened artifact bytes do not reconcile with the ledger")
        if artifact.ledger is not None and artifact.ledger != expected.ledger:
            raise ValueError("artifact ledger differs from the deterministic reopened ledger")
        return ExportProof(
            artifact_id=artifact.artifact_id,
            artifact_sha256=actual_hash,
            source_checkpoint_id=checkpoint.checkpoint_id,
            format_version=artifact.format_spec.version,
            signs_verified=signs_verified,
            scales_verified=scales_verified,
            bytes_verified=len(content),
            tensor_checksums=tensor_checksums,
            exact_byte_match=True,
            accounting_reconciled=True,
        )

    def verify(
        self, artifact: PackedArtifactRef, checkpoint: IdentifiedCheckpoint
    ) -> ExportProof:
        try:
            return self._verify_impl(artifact, checkpoint)
        except Exception as error:
            quarantined = self._quarantine(artifact.path)
            if isinstance(error, ExportError):
                code = error.code
                retryability = error.retryability
                context = error.context
            else:
                code = "export.verification_failed"
                retryability = Retryability.AFTER_REMEDIATION
                context = {"cause": str(error)}
            context["quarantine_path"] = None if quarantined is None else str(quarantined)
            raise ExportError(
                "packed artifact verification failed and the output was quarantined",
                retryability=retryability,
                code=code,
                affected_ids={
                    "artifact_ids": (artifact.artifact_id,),
                    "checkpoint_ids": (checkpoint.checkpoint_id,),
                },
                context=context,
            ) from error


PackedExporter = DeterministicPackedExporter


__all__ = [
    "ExportProof",
    "IdentifiedCheckpoint",
    "PackedArtifactRef",
    "PackedExportPlan",
    "PackedExporter",
    "PackedSignRows",
    "PackedTensorSource",
    "SourceArtifactKind",
    "pack_sign_rows",
    "unpack_sign_rows",
]
