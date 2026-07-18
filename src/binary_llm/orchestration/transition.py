"""Exact, deterministic Stage 1 to progressive-training transition."""

from __future__ import annotations

import struct
import sys
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from binary_llm.adapters import BinaryLinear, TrainingActivationMode
from binary_llm.domain import (
    ArtifactRef,
    ManifestError,
    Retryability,
    canonical_json_bytes,
    parse_canonical_json,
    sha256_bytes,
)

from .store import FilesystemArtifactStore

_MAGIC = b"BINARYLLM-STAGE1-TRANSITION\x00"
_SCHEMA_VERSION = 1
_MEDIA_TYPE = "application/vnd.binary-llm.stage1-transition.v1"
_DTYPES: dict[torch.dtype, str] = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
    torch.float64: "float64",
}
_TORCH_DTYPES = {name: dtype for dtype, name in _DTYPES.items()}


class Stage1TransitionKind(StrEnum):
    REFERENCE_EXPLICIT = "reference_explicit"
    MODIFIED_WEIGHT_ONLY = "modified_weight_only"


@dataclass(frozen=True, slots=True)
class Stage1Transition:
    artifact_ref: ArtifactRef
    payload: bytes
    source_checkpoint_id: str
    source_run_id: str
    parent_artifact_id: str | None
    kind: Stage1TransitionKind
    training_activation_mode: TrainingActivationMode
    module_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AppliedStage1Transition:
    artifact_ref: ArtifactRef
    source_checkpoint_id: str
    parent_artifact_id: str | None
    kind: Stage1TransitionKind
    model_identity: int
    application_sha256: str


def _failure(message: str, code: str, **context: Any) -> ManifestError:
    return ManifestError(
        message,
        retryability=Retryability.NEVER,
        code=f"manifest.stage1_transition.{code}",
        context=context,
    )


def _binary_modules(model: nn.Module) -> tuple[tuple[str, BinaryLinear], ...]:
    modules = tuple(
        sorted(
            (
                (name, module)
                for name, module in model.named_modules()
                if isinstance(module, BinaryLinear)
            ),
            key=lambda item: item[0],
        )
    )
    if not modules:
        raise _failure("transition requires at least one BinaryLinear", "no_modules")
    return modules


def _tensor_bytes(tensor: Tensor) -> bytes:
    if sys.byteorder != "little":
        raise _failure("transition encoding requires a little-endian host", "endian")
    value = tensor.detach().cpu().contiguous()
    if value.dtype not in _DTYPES:
        raise _failure(
            "transition tensor dtype is unsupported",
            "unsupported_dtype",
            dtype=str(value.dtype),
        )
    return value.view(torch.uint8).numpy().tobytes()


def _tensor_record(name: str, tensor: Tensor, role: str, offset: int) -> tuple[dict[str, Any], bytes]:
    data = _tensor_bytes(tensor)
    return (
        {
            "name": name,
            "role": role,
            "dtype": _DTYPES[tensor.dtype],
            "shape": list(tensor.shape),
            "order": "C",
            "byte_order": "little",
            "offset": offset,
            "bytes": len(data),
            "sha256": sha256_bytes(data),
        },
        data,
    )


def _digest(tensor: Tensor) -> str:
    return sha256_bytes(_tensor_bytes(tensor))


def _config_record(module: BinaryLinear) -> dict[str, Any]:
    config = module.config
    return {
        "scale_parameterization": config.scale_parameterization.value,
        "training_activation_mode": config.training_activation_mode.value,
        "zero_sign_rule": config.zero_sign_rule.value,
        "initial_input_scale": config.initial_input_scale,
        "scale_epsilon": config.scale_epsilon,
        "minimum_scale_magnitude": config.minimum_scale_magnitude,
        "progressive_operator": {
            "precision": config.progressive_operator.precision.value,
            "analytical_scale_gradient": (
                config.progressive_operator.analytical_scale_gradient.value
            ),
        },
    }


def build_stage1_transition(
    *,
    model: nn.Module,
    source_checkpoint_id: str,
    source_run_id: str,
    parent_artifact_id: str | None = None,
    training_config: Mapping[str, Any] | None = None,
) -> Stage1Transition:
    """Build canonical metadata plus contiguous little-endian scientific tensors."""

    if not source_checkpoint_id or not source_run_id:
        raise ValueError("source checkpoint and run IDs must be non-empty")
    modules = _binary_modules(model)
    modes = {module.config.training_activation_mode for _, module in modules}
    if len(modes) != 1:
        raise _failure("all transition modules must use one activation mode", "mixed_modes")
    mode = next(iter(modes))
    kind = (
        Stage1TransitionKind.REFERENCE_EXPLICIT
        if mode is TrainingActivationMode.EXPLICIT_ACTIVATION_TRANSFORM
        else Stage1TransitionKind.MODIFIED_WEIGHT_ONLY
    )
    tensor_records: list[dict[str, Any]] = []
    tensor_payloads: list[bytes] = []
    module_records: list[dict[str, Any]] = []
    offset = 0
    for name, module in modules:
        source = module.weight.detach()
        scale = module.input_scale().detach()
        transformed = module.transformed_weight().detach()
        for suffix, tensor, role in (
            ("scale", scale, "optimized_input_scale"),
            ("weight_tilde", transformed, "stage2_parent_latent"),
        ):
            record, data = _tensor_record(f"{name}.{suffix}", tensor, role, offset)
            tensor_records.append(record)
            tensor_payloads.append(data)
            offset += len(data)
        module_records.append(
            {
                "name": name,
                "tensor_name": module.tensor_name,
                "weight_shape": list(source.shape),
                "weight_dtype": _DTYPES.get(source.dtype),
                "source_weight_sha256": _digest(source),
                "dense_reference_weight_sha256": _digest(module.dense_reference_weight),
                "bias_present": module.bias is not None,
                "bias_sha256": None if module.bias is None else _digest(module.bias),
                "dense_reference_bias_sha256": (
                    None
                    if module.dense_reference_bias is None
                    else _digest(module.dense_reference_bias)
                ),
                "scale_tensor": f"{name}.scale",
                "transformed_tensor": f"{name}.weight_tilde",
                "config": _config_record(module),
            }
        )
    metadata = {
        "schema_version": _SCHEMA_VERSION,
        "artifact_kind": "stage1_transition",
        "transition_kind": kind.value,
        "source_checkpoint_id": source_checkpoint_id,
        "source_run_id": source_run_id,
        "parent_artifact_id": parent_artifact_id,
        "training_activation_mode": mode.value,
        "training_config": dict(training_config or {}),
        "modules": module_records,
        "tensors": tensor_records,
    }
    metadata_bytes = canonical_json_bytes(metadata)
    payload = _MAGIC + struct.pack("<Q", len(metadata_bytes)) + metadata_bytes + b"".join(
        tensor_payloads
    )
    payload_digest = sha256_bytes(payload)
    artifact = ArtifactRef(
        artifact_id=f"stage1-transition:{payload_digest}",
        kind="stage1_transition",
        sha256=payload_digest,
        bytes=len(payload),
        media_type=_MEDIA_TYPE,
        parent_artifact_id=parent_artifact_id,
        producing_run_id=source_run_id,
    )
    return Stage1Transition(
        artifact,
        payload,
        source_checkpoint_id,
        source_run_id,
        parent_artifact_id,
        kind,
        mode,
        tuple(name for name, _ in modules),
    )


def _decode(payload: bytes, expected: ArtifactRef | None = None) -> tuple[dict[str, Any], dict[str, Tensor]]:
    if not isinstance(payload, bytes):
        raise TypeError("transition payload must be bytes")
    if expected is not None and (
        expected.kind != "stage1_transition"
        or expected.media_type != _MEDIA_TYPE
        or expected.bytes != len(payload)
        or expected.sha256 != sha256_bytes(payload)
        or expected.artifact_id != f"stage1-transition:{expected.sha256}"
    ):
        raise _failure("transition ArtifactRef integrity failed", "artifact_ref")
    if not payload.startswith(_MAGIC) or len(payload) < len(_MAGIC) + 8:
        raise _failure("transition payload header is invalid", "header")
    metadata_length = struct.unpack(
        "<Q", payload[len(_MAGIC) : len(_MAGIC) + 8]
    )[0]
    metadata_start = len(_MAGIC) + 8
    metadata_end = metadata_start + metadata_length
    try:
        metadata = parse_canonical_json(payload[metadata_start:metadata_end])
    except Exception as error:
        raise _failure("transition metadata is not canonical JSON", "metadata") from error
    if not isinstance(metadata, dict) or metadata.get("schema_version") != _SCHEMA_VERSION:
        raise _failure("transition metadata schema is unsupported", "schema")
    if canonical_json_bytes(metadata) != payload[metadata_start:metadata_end]:
        raise _failure("transition metadata encoding is non-canonical", "metadata")
    tensor_area = payload[metadata_end:]
    tensors: dict[str, Tensor] = {}
    records = metadata.get("tensors")
    if not isinstance(records, list):
        raise _failure("transition tensor manifest is invalid", "tensor_manifest")
    expected_offset = 0
    for record in records:
        if not isinstance(record, dict):
            raise _failure("transition tensor record is invalid", "tensor_manifest")
        name = record.get("name")
        dtype_name = record.get("dtype")
        shape = record.get("shape")
        offset = record.get("offset")
        byte_count = record.get("bytes")
        if (
            not isinstance(name, str)
            or dtype_name not in _TORCH_DTYPES
            or not isinstance(shape, list)
            or not all(isinstance(item, int) and item >= 0 for item in shape)
            or offset != expected_offset
            or not isinstance(byte_count, int)
            or record.get("order") != "C"
            or record.get("byte_order") != "little"
        ):
            raise _failure("transition tensor metadata is invalid", "tensor_metadata")
        data = tensor_area[offset : offset + byte_count]
        if len(data) != byte_count or sha256_bytes(data) != record.get("sha256"):
            raise _failure("transition tensor digest failed", "tensor_digest", tensor=name)
        dtype = _TORCH_DTYPES[dtype_name]
        try:
            tensor = torch.frombuffer(bytearray(data), dtype=dtype).clone().reshape(shape)
        except (RuntimeError, ValueError) as error:
            raise _failure("transition tensor bytes do not match metadata", "tensor_shape") from error
        tensors[name] = tensor
        expected_offset += byte_count
    if expected_offset != len(tensor_area) or len(tensors) != len(records):
        raise _failure("transition contains trailing or duplicate tensor data", "tensor_layout")
    return metadata, tensors


def load_stage1_transition(
    store: FilesystemArtifactStore, artifact_ref: ArtifactRef
) -> Stage1Transition:
    payload = store.resolve(artifact_ref.artifact_id, expected=artifact_ref)
    metadata, _ = _decode(payload, artifact_ref)
    return Stage1Transition(
        artifact_ref=artifact_ref,
        payload=payload,
        source_checkpoint_id=str(metadata["source_checkpoint_id"]),
        source_run_id=str(metadata["source_run_id"]),
        parent_artifact_id=metadata.get("parent_artifact_id"),
        kind=Stage1TransitionKind(metadata["transition_kind"]),
        training_activation_mode=TrainingActivationMode(
            metadata["training_activation_mode"]
        ),
        module_names=tuple(item["name"] for item in metadata["modules"]),
    )


def persist_stage1_transition(
    transition: Stage1Transition, store: FilesystemArtifactStore
) -> ArtifactRef:
    _decode(transition.payload, transition.artifact_ref)
    return store.put_bytes(transition.artifact_ref, transition.payload)


def apply_stage1_transition(
    model: nn.Module,
    transition: Stage1Transition,
    *,
    expected_source_checkpoint_id: str,
    expected_parent_artifact_id: str | None = None,
    require_reference: bool = True,
) -> AppliedStage1Transition:
    """Validate the complete model first, then install W/S and identity scales."""

    metadata, tensors = _decode(transition.payload, transition.artifact_ref)
    if transition.source_checkpoint_id != metadata.get("source_checkpoint_id"):
        raise _failure("transition object disagrees with payload", "object_metadata")
    if metadata.get("source_checkpoint_id") != expected_source_checkpoint_id:
        raise _failure("transition source checkpoint is wrong", "source_checkpoint")
    if metadata.get("parent_artifact_id") != expected_parent_artifact_id:
        raise _failure("transition parent lineage is wrong", "parent_lineage")
    kind = Stage1TransitionKind(metadata["transition_kind"])
    if require_reference and kind is not Stage1TransitionKind.REFERENCE_EXPLICIT:
        raise _failure("modified Stage 1 arm cannot initialize reference Stage 2", "modified_arm")
    modules = dict(_binary_modules(model))
    records = metadata.get("modules")
    if (
        not isinstance(records, list)
        or any(not isinstance(item, dict) for item in records)
        or {item.get("name") for item in records} != set(modules)
    ):
        raise _failure("transition module set does not exactly match model", "module_set")
    pending: list[tuple[BinaryLinear, Tensor]] = []
    for record in records:
        name = record["name"]
        module = modules[name]
        scale = tensors.get(record.get("scale_tensor"))
        transformed = tensors.get(record.get("transformed_tensor"))
        if scale is None or transformed is None:
            raise _failure("transition module tensor is missing", "module_tensor", module=name)
        checks = (
            list(module.weight.shape) == record.get("weight_shape"),
            _DTYPES.get(module.weight.dtype) == record.get("weight_dtype"),
            _digest(module.weight) == record.get("source_weight_sha256"),
            _digest(module.dense_reference_weight)
            == record.get("dense_reference_weight_sha256"),
            _digest(module.input_scale()) == _digest(scale),
            (module.bias is not None) == record.get("bias_present"),
            (None if module.bias is None else _digest(module.bias))
            == record.get("bias_sha256"),
            (
                None
                if module.dense_reference_bias is None
                else _digest(module.dense_reference_bias)
            )
            == record.get("dense_reference_bias_sha256"),
            transformed.shape == module.weight.shape,
            transformed.dtype == module.weight.dtype,
            torch.equal(module.weight.detach().cpu() / scale.unsqueeze(0), transformed),
        )
        if not all(checks):
            raise _failure("transition module compatibility failed", "module_mismatch", module=name)
        try:
            module.identity_input_scale_parameter()
        except ValueError as error:
            raise _failure(
                "transition target cannot represent an exact identity scale",
                "identity_scale",
                module=name,
            ) from error
        pending.append((module, transformed))
    dense_snapshots = [module.dense_reference_weight.detach().clone() for module, _ in pending]
    with torch.no_grad():
        for module, transformed in pending:
            module.weight.copy_(transformed.to(module.weight.device))
            module.reset_input_scale_to_identity_()
    for (module, transformed), dense in zip(pending, dense_snapshots):
        if not torch.equal(module.transformed_weight().detach().cpu(), transformed):
            raise AssertionError("validated transition application was not exact")
        if not torch.equal(module.dense_reference_weight, dense):
            raise AssertionError("transition mutated the dense reference oracle")
    application_record: Mapping[str, Any] = {
        "artifact_id": transition.artifact_ref.artifact_id,
        "source_checkpoint_id": transition.source_checkpoint_id,
        "parent_artifact_id": transition.parent_artifact_id,
        "kind": kind.value,
        "modules": [item["name"] for item in records],
    }
    return AppliedStage1Transition(
        transition.artifact_ref,
        transition.source_checkpoint_id,
        transition.parent_artifact_id,
        kind,
        id(model),
        sha256_bytes(canonical_json_bytes(application_record)),
    )


__all__ = [
    "AppliedStage1Transition",
    "Stage1Transition",
    "Stage1TransitionKind",
    "apply_stage1_transition",
    "build_stage1_transition",
    "load_stage1_transition",
    "persist_stage1_transition",
]
