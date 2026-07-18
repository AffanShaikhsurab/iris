"""Memory-mapped scalar execution for deterministic packed binary artifacts."""

from __future__ import annotations

import hashlib
import math
import mmap
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import BinaryIO, Mapping

import torch
from torch import Tensor, nn

from binary_llm.adapters.linear import BinaryLinear
from binary_llm.domain import ParityError, Retryability, parse_canonical_json

from .packed import MAGIC, PackedArtifactRef

_HEADER_BYTES = 16
_SCALE_FORMATS = {"float16": "e", "float32": "f", "float64": "d"}
_ENDIAN_PREFIXES = {"little": "<", "big": ">"}


@dataclass(frozen=True, slots=True)
class RuntimeMemoryEvidence:
    """Logical allocations owned by one packed runtime session."""

    mapped_packed_bytes: int
    persistent_metadata_bytes: int
    persistent_scale_bytes: int
    persistent_dense_weight_bytes: int
    maximum_temporary_bytes: int
    maximum_temporary_expansion_bytes: int
    temporary_expansion_limit_bytes: int
    maximum_output_bytes: int

    @property
    def persistent_dense_copy_detected(self) -> bool:
        return self.persistent_dense_weight_bytes > 0

    @property
    def temporary_expansion_within_limit(self) -> bool:
        return self.maximum_temporary_expansion_bytes <= self.temporary_expansion_limit_bytes


@dataclass(frozen=True, slots=True)
class PackedTensorView:
    name: str
    rows: int
    columns: int
    sign_offset: int
    sign_storage_bytes: int
    row_payload_bytes: int
    row_stride_bytes: int
    scale_offset: int
    scale_bytes: int
    bit_order: str
    scale_layout: str
    scale_width: int


class RuntimeSession:
    """Owns a read-only map and never stores decoded weight or scale tensors."""

    def __init__(
        self,
        *,
        artifact: PackedArtifactRef,
        file_handle: BinaryIO,
        mapping: mmap.mmap,
        tensors: Mapping[str, PackedTensorView],
        metadata_bytes: int,
    ) -> None:
        self.artifact = artifact
        self._file_handle = file_handle
        self._mapping = mapping
        self._tensors = MappingProxyType(dict(tensors))
        self._metadata_bytes = metadata_bytes
        self._maximum_output_bytes = 0
        self._closed = False

    @property
    def tensor_names(self) -> tuple[str, ...]:
        return tuple(self._tensors)

    def tensor(self, name: str) -> PackedTensorView:
        self._ensure_open()
        try:
            return self._tensors[name]
        except KeyError as error:
            raise KeyError(f"packed tensor is not present: {name}") from error

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("packed runtime session is closed")

    def memory_report(self) -> RuntimeMemoryEvidence:
        return RuntimeMemoryEvidence(
            mapped_packed_bytes=self.artifact.bytes,
            persistent_metadata_bytes=self._metadata_bytes,
            persistent_scale_bytes=0,
            persistent_dense_weight_bytes=0,
            maximum_temporary_bytes=max(self._metadata_bytes, 64),
            maximum_temporary_expansion_bytes=0,
            temporary_expansion_limit_bytes=(
                self.artifact.format_spec.temporary_expansion_limit_bytes
            ),
            maximum_output_bytes=self._maximum_output_bytes,
        )

    def _scale(self, tensor: PackedTensorView, row: int) -> float:
        value = struct.unpack_from(
            tensor.scale_layout,
            self._mapping,
            tensor.scale_offset + row * tensor.scale_width,
        )[0]
        if not math.isfinite(value):
            raise ValueError(f"packed scale is non-finite for tensor {tensor.name}")
        return float(value)

    def linear(
        self,
        tensor_name: str,
        inputs: Tensor,
        bias: Tensor | None = None,
    ) -> Tensor:
        """Accumulate from mapped sign bits in scalar column order."""

        self._ensure_open()
        tensor = self.tensor(tensor_name)
        if not isinstance(inputs, Tensor) or not inputs.is_floating_point():
            raise TypeError("packed linear inputs must be a floating torch.Tensor")
        if inputs.device.type != "cpu":
            raise ValueError("the scalar packed reference runtime supports CPU inputs only")
        if not inputs.is_contiguous():
            raise ValueError("inputs must be contiguous to avoid an unreported temporary copy")
        if inputs.ndim < 1 or inputs.shape[-1] != tensor.columns:
            raise ValueError("packed linear input width does not match the tensor")
        if bias is not None:
            if bias.device.type != "cpu" or bias.shape != (tensor.rows,):
                raise ValueError("packed linear bias must be a CPU vector with one value per row")
            if not bias.is_floating_point() or not torch.isfinite(bias).all().item():
                raise ValueError("packed linear bias must be finite and floating point")
        if not torch.isfinite(inputs).all().item():
            raise ValueError("packed linear inputs must be finite")

        flat_inputs = inputs.view(-1, tensor.columns)
        output = torch.empty(
            (flat_inputs.shape[0], tensor.rows), dtype=inputs.dtype, device="cpu"
        )
        for vector_index in range(flat_inputs.shape[0]):
            for row in range(tensor.rows):
                accumulator = 0.0
                row_start = tensor.sign_offset + row * tensor.row_stride_bytes
                for column in range(tensor.columns):
                    packed_byte = self._mapping[row_start + column // 8]
                    bit = column % 8
                    shift = bit if tensor.bit_order == "lsb0" else 7 - bit
                    sign = 1.0 if packed_byte & (1 << shift) else -1.0
                    accumulator += float(flat_inputs[vector_index, column]) * sign
                value = accumulator * self._scale(tensor, row)
                if bias is not None:
                    value += float(bias[row])
                output[vector_index, row] = value
        result = output.view(*inputs.shape[:-1], tensor.rows)
        self._maximum_output_bytes = max(
            self._maximum_output_bytes, result.numel() * result.element_size()
        )
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._mapping.close()
        self._file_handle.close()
        self._closed = True

    def __enter__(self) -> RuntimeSession:
        self._ensure_open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def _sha256_mapped(mapping: mmap.mmap) -> str:
    digest = hashlib.sha256()
    view = memoryview(mapping)
    try:
        for offset in range(0, len(mapping), 1024 * 1024):
            digest.update(view[offset : offset + 1024 * 1024])
    finally:
        view.release()
    return digest.hexdigest()


def _sha256_range(mapping: mmap.mmap, start: int, length: int) -> str:
    digest = hashlib.sha256()
    view = memoryview(mapping)
    try:
        digest.update(view[start : start + length])
    finally:
        view.release()
    return digest.hexdigest()


def _require_int(record: Mapping[str, object], key: str, *, minimum: int = 0) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"packed tensor {key} is invalid")
    return value


class ScalarPackedRuntime:
    """Load and validate the exact artifact/runtime/build tuple using read-only mmap."""

    def __init__(self, *, runtime_revision: str, build_flags: tuple[str, ...]) -> None:
        if not runtime_revision:
            raise ValueError("runtime_revision must be explicit")
        if any(not isinstance(flag, str) or not flag for flag in build_flags):
            raise ValueError("build_flags must contain explicit strings")
        self.runtime_revision = runtime_revision
        self.build_flags = tuple(build_flags)

    def load(self, artifact: PackedArtifactRef) -> RuntimeSession:
        if not isinstance(artifact, PackedArtifactRef):
            raise TypeError("artifact must be a PackedArtifactRef")
        if artifact.runtime_revision != self.runtime_revision:
            raise ParityError(
                "packed artifact runtime revision does not match the reference runtime",
                retryability=Retryability.NEVER,
                code="parity.runtime_revision",
                affected_ids={"artifact_ids": (artifact.artifact_id,)},
            )
        if artifact.build_flags != self.build_flags:
            raise ParityError(
                "packed artifact build flags do not match the reference runtime",
                retryability=Retryability.NEVER,
                code="parity.build_flags",
                affected_ids={"artifact_ids": (artifact.artifact_id,)},
            )

        path = Path(artifact.path)
        handle = path.open("rb")
        mapping: mmap.mmap | None = None
        try:
            actual_bytes = os.fstat(handle.fileno()).st_size
            if actual_bytes != artifact.bytes:
                raise ValueError("packed artifact byte count differs from its reference")
            mapping = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
            if len(mapping) < _HEADER_BYTES or mapping[:8] != MAGIC:
                raise ValueError("packed artifact has an invalid header")
            if _sha256_mapped(mapping) != artifact.sha256:
                raise ValueError("packed artifact SHA-256 differs from its reference")
            metadata_length = struct.unpack_from("<Q", mapping, 8)[0]
            metadata_end = _HEADER_BYTES + metadata_length
            if metadata_end > len(mapping):
                raise ValueError("packed artifact metadata is truncated")
            metadata_raw = mapping[_HEADER_BYTES:metadata_end]
            metadata = parse_canonical_json(metadata_raw)
            if not isinstance(metadata, dict):
                raise ValueError("packed artifact metadata must be an object")
            if metadata.get("artifact_id") != artifact.artifact_id:
                raise ValueError("packed metadata artifact identity differs")
            if metadata.get("runtime_revision") != self.runtime_revision:
                raise ValueError("packed metadata runtime revision differs")
            if tuple(metadata.get("build_flags", ())) != self.build_flags:
                raise ValueError("packed metadata build flags differ")

            alignment = artifact.format_spec.row_alignment
            prefix_padding = (-metadata_end) % alignment
            data_offset = metadata_end + prefix_padding
            if any(mapping[metadata_end:data_offset]):
                raise ValueError("packed metadata padding must contain only zero bytes")
            tensors = self._parse_tensors(mapping, metadata, data_offset, artifact)
            return RuntimeSession(
                artifact=artifact,
                file_handle=handle,
                mapping=mapping,
                tensors=tensors,
                metadata_bytes=metadata_length,
            )
        except Exception as error:
            if mapping is not None:
                mapping.close()
            handle.close()
            if isinstance(error, ParityError):
                raise
            raise ParityError(
                "packed artifact could not be loaded by the scalar reference runtime",
                retryability=Retryability.AFTER_REMEDIATION,
                code="parity.runtime_load",
                affected_ids={"artifact_ids": (artifact.artifact_id,)},
                context={"cause": str(error)},
            ) from error

    @staticmethod
    def _parse_tensors(
        mapping: mmap.mmap,
        metadata: Mapping[str, object],
        data_offset: int,
        artifact: PackedArtifactRef,
    ) -> Mapping[str, PackedTensorView]:
        raw_records = metadata.get("tensors")
        if not isinstance(raw_records, list) or not raw_records:
            raise ValueError("packed artifact must enumerate tensors")
        try:
            scale_code = _SCALE_FORMATS[artifact.format_spec.scale_dtype]
            endian = _ENDIAN_PREFIXES[artifact.format_spec.scale_endianness]
        except KeyError as error:
            raise ValueError("packed scale layout is unsupported") from error
        scale_layout = endian + scale_code
        scale_width = struct.calcsize(scale_code)
        tensors: dict[str, PackedTensorView] = {}
        occupied: list[tuple[int, int]] = []
        for raw in raw_records:
            if not isinstance(raw, dict):
                raise ValueError("packed tensor metadata must be an object")
            name = raw.get("name")
            shape = raw.get("shape")
            if not isinstance(name, str) or not name or name in tensors:
                raise ValueError("packed tensor names must be explicit and unique")
            if (
                not isinstance(shape, list)
                or len(shape) != 2
                or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in shape)
            ):
                raise ValueError(f"packed tensor shape is invalid for {name}")
            rows, columns = shape
            sign_relative = _require_int(raw, "sign_offset")
            sign_bytes = _require_int(raw, "sign_storage_bytes", minimum=1)
            row_payload = _require_int(raw, "row_payload_bytes", minimum=1)
            row_stride = _require_int(raw, "row_stride_bytes", minimum=row_payload)
            scale_relative = _require_int(raw, "scale_offset")
            scale_bytes = _require_int(raw, "scale_bytes", minimum=1)
            checksum_relative = _require_int(raw, "checksum_offset")
            checksum_bytes = _require_int(raw, "checksum_bytes", minimum=64)
            if row_payload != (columns + 7) // 8 or sign_bytes != rows * row_stride:
                raise ValueError(f"packed row layout is inconsistent for {name}")
            if scale_bytes != rows * scale_width or checksum_bytes != 64:
                raise ValueError(f"packed scale/checksum layout is inconsistent for {name}")

            sign_offset = data_offset + sign_relative
            scale_offset = data_offset + scale_relative
            checksum_offset = data_offset + checksum_relative
            ranges = (
                (sign_offset, sign_offset + sign_bytes),
                (scale_offset, scale_offset + scale_bytes),
                (checksum_offset, checksum_offset + checksum_bytes),
            )
            if any(start < data_offset or end > len(mapping) for start, end in ranges):
                raise ValueError(f"packed tensor range is outside the artifact for {name}")
            for start, end in ranges:
                if any(start < old_end and old_start < end for old_start, old_end in occupied):
                    raise ValueError(f"packed tensor ranges overlap for {name}")
                occupied.append((start, end))

            sign_hash = _sha256_range(mapping, sign_offset, sign_bytes)
            scale_hash = _sha256_range(mapping, scale_offset, scale_bytes)
            if sign_hash != raw.get("sign_checksum_sha256") or scale_hash != raw.get(
                "scale_checksum_sha256"
            ):
                raise ValueError(f"packed tensor checksum metadata differs for {name}")
            stored = mapping[checksum_offset : checksum_offset + checksum_bytes]
            if stored != bytes.fromhex(sign_hash) + bytes.fromhex(scale_hash):
                raise ValueError(f"packed tensor stored checksums differ for {name}")
            for row in range(rows):
                row_start = sign_offset + row * row_stride
                if any(mapping[row_start + row_payload : row_start + row_stride]):
                    raise ValueError(f"packed row alignment padding is nonzero for {name}")
                unused = row_payload * 8 - columns
                if unused:
                    final = mapping[row_start + row_payload - 1]
                    if artifact.format_spec.bit_order == "lsb0":
                        used_mask = (1 << (8 - unused)) - 1
                    else:
                        used_mask = (0xFF << unused) & 0xFF
                    if final & (~used_mask & 0xFF):
                        raise ValueError(f"packed row padding bits are nonzero for {name}")

            tensors[name] = PackedTensorView(
                name=name,
                rows=rows,
                columns=columns,
                sign_offset=sign_offset,
                sign_storage_bytes=sign_bytes,
                row_payload_bytes=row_payload,
                row_stride_bytes=row_stride,
                scale_offset=scale_offset,
                scale_bytes=scale_bytes,
                bit_order=artifact.format_spec.bit_order,
                scale_layout=scale_layout,
                scale_width=scale_width,
            )
        if metadata.get("tensor_count") != len(tensors):
            raise ValueError("packed tensor count metadata differs")
        return tensors


class PackedLinear(nn.Module):
    """A model bridge whose only weight state is a packed-session tensor name."""

    def __init__(
        self,
        session: RuntimeSession,
        *,
        tensor_name: str,
        bias: Tensor | None,
    ) -> None:
        super().__init__()
        tensor = session.tensor(tensor_name)
        self.session = session
        self.tensor_name = tensor_name
        self.in_features = tensor.columns
        self.out_features = tensor.rows
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(bias.detach().cpu().clone(), requires_grad=False)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.session.linear(self.tensor_name, inputs, self.bias)


def bind_packed_linears(model: nn.Module, session: RuntimeSession) -> nn.Module:
    """Replace every BinaryLinear in-place and require exact artifact coverage."""

    replacements: list[tuple[str, BinaryLinear]] = [
        (name, module)
        for name, module in model.named_modules()
        if name and isinstance(module, BinaryLinear)
    ]
    expected_names = {module.tensor_name for _, module in replacements}
    if not replacements or expected_names != set(session.tensor_names):
        raise ValueError("model BinaryLinear inventory must exactly match packed tensors")
    for module_path, source in replacements:
        if "." in module_path:
            parent_path, attribute = module_path.rsplit(".", 1)
            parent = model.get_submodule(parent_path)
        else:
            parent, attribute = model, module_path
        setattr(
            parent,
            attribute,
            PackedLinear(session, tensor_name=source.tensor_name, bias=source.bias),
        )
    return model


__all__ = [
    "PackedLinear",
    "PackedTensorView",
    "RuntimeMemoryEvidence",
    "RuntimeSession",
    "ScalarPackedRuntime",
    "bind_packed_linears",
]
