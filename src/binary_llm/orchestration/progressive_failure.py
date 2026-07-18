"""Canonical, diagnostic-only progressive numerical-failure artifacts."""

from __future__ import annotations

import hashlib
import struct
import sys
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from binary_llm.domain import ArtifactRef, canonical_json_bytes, parse_canonical_json, sha256_bytes
from binary_llm.math import FiniteStateDiagnostics

from .budgets import BudgetUsage
from .progressive_state import ProgressiveBoundaryCheckpoint, _state_normal_form
from .stage1 import CausalBatch
from .store import FilesystemArtifactStore

_MAGIC = b"BINARYLLM-PROGRESSIVE-FAILURE\x00"
_MEDIA_TYPE = "application/vnd.binary-llm.progressive-failure.v1"


class ProgressiveFailurePoint(StrEnum):
    PRE_BACKWARD = "pre_backward"
    POST_BACKWARD = "post_backward"
    POST_UPDATE = "post_update"


@dataclass(frozen=True, slots=True)
class ProgressiveFailureArtifact:
    """Immutable evidence for a failed partial phase; never a checkpoint."""

    artifact_ref: ArtifactRef
    payload: bytes
    failure_code: str
    failure_point: ProgressiveFailurePoint
    incomplete_phase_ordinal: int
    last_complete_checkpoint_id: str | None
    resumable: bool = False
    promotion_eligible: bool = False


def _tensor_bytes(value: Tensor) -> bytes:
    tensor = value.detach().cpu().contiguous()
    if sys.byteorder != "little":
        raise RuntimeError("failure artifact encoding requires a little-endian host")
    return tensor.reshape(-1).view(torch.uint8).numpy().tobytes()


def _tensor_record(name: str, value: Tensor, offset: int) -> tuple[dict[str, Any], bytes]:
    raw = _tensor_bytes(value)
    return (
        {
            "name": name,
            "dtype": str(value.dtype).removeprefix("torch."),
            "shape": list(value.shape),
            "byte_order": "little",
            "order": "C",
            "offset": offset,
            "bytes": len(raw),
            "sha256": sha256_bytes(raw),
        },
        raw,
    )


def _state_digest(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(_state_normal_form(value)))


def build_progressive_failure_artifact(
    *,
    run_id: str,
    attempt_id: str | None,
    stage1_parent_checkpoint_id: str,
    plan_hash: str,
    phase_ordinal: int,
    schedule_index: int,
    partition_id: str,
    partition_hash: str,
    batch: CausalBatch,
    failure_point: ProgressiveFailurePoint,
    failure_code: str,
    exception: BaseException,
    diagnostics: FiniteStateDiagnostics,
    model: nn.Module,
    optimizer: Optimizer,
    optimizer_step: int,
    budget_usage: BudgetUsage | None,
    last_complete_checkpoint: ProgressiveBoundaryCheckpoint | None,
) -> ProgressiveFailureArtifact:
    tensors: list[tuple[str, Tensor]] = [("batch.input_ids", batch.input_ids)]
    if batch.labels is not None:
        tensors.append(("batch.labels", batch.labels))
    records: list[dict[str, Any]] = []
    chunks: list[bytes] = []
    offset = 0
    for name, tensor in tensors:
        record, raw = _tensor_record(name, tensor, offset)
        records.append(record)
        chunks.append(raw)
        offset += len(raw)
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    latent_summaries = {
        name: _state_normal_form(value)
        for name, value in model_state.items()
        if name.endswith(".weight") or "scale" in name
    }
    metadata: Mapping[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "progressive_numerical_failure",
        "diagnostic_only": True,
        "resumable": False,
        "promotion_eligible": False,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "stage1_parent_checkpoint_id": stage1_parent_checkpoint_id,
        "plan_hash": plan_hash,
        "phase_ordinal": phase_ordinal,
        "schedule_index": schedule_index,
        "partition_id": partition_id,
        "partition_hash": partition_hash,
        "batch_id": batch.batch_id,
        "failure_point": failure_point.value,
        "failure_code": failure_code,
        "exception_type": type(exception).__name__,
        "last_complete_checkpoint_id": (
            None if last_complete_checkpoint is None else last_complete_checkpoint.checkpoint_id
        ),
        "last_complete_checkpoint_hashes": (
            None if last_complete_checkpoint is None else dict(last_complete_checkpoint.state_hashes)
        ),
        "optimizer_step": optimizer_step,
        "budget_usage": None if budget_usage is None else asdict(budget_usage),
        "schedule_state": (
            None if last_complete_checkpoint is None else asdict(last_complete_checkpoint.schedule_state)
        ),
        "data_cursor": (
            None if last_complete_checkpoint is None else asdict(last_complete_checkpoint.data_cursor)
        ),
        "rng_state_hash": (
            None if last_complete_checkpoint is None else last_complete_checkpoint.state_hashes["rng_state"]
        ),
        "model_state_hash": _state_digest(model_state),
        "latent_scale_summaries": latent_summaries,
        "optimizer_state_hash": _state_digest(optimizer_state),
        "optimizer_state_summary": _state_normal_form(optimizer_state),
        "finite_diagnostics": [asdict(item) for item in diagnostics.summaries],
        "batch_tensors": records,
    }
    metadata_bytes = canonical_json_bytes(metadata)
    payload = _MAGIC + struct.pack("<Q", len(metadata_bytes)) + metadata_bytes + b"".join(chunks)
    digest = sha256_bytes(payload)
    artifact_ref = ArtifactRef(
        artifact_id=f"progressive-failure:{digest}",
        kind="progressive_numerical_failure",
        sha256=digest,
        bytes=len(payload),
        media_type=_MEDIA_TYPE,
        parent_artifact_id=(
            None if last_complete_checkpoint is None else last_complete_checkpoint.checkpoint_id
        ),
        producing_run_id=run_id,
        producing_attempt_id=attempt_id,
    )
    return ProgressiveFailureArtifact(
        artifact_ref,
        payload,
        failure_code,
        failure_point,
        phase_ordinal,
        None if last_complete_checkpoint is None else last_complete_checkpoint.checkpoint_id,
    )


def verify_progressive_failure_artifact(artifact: ProgressiveFailureArtifact) -> None:
    if artifact.resumable or artifact.promotion_eligible:
        raise ValueError("failure artifacts must remain diagnostic-only")
    payload = artifact.payload
    ref = artifact.artifact_ref
    if (
        not payload.startswith(_MAGIC)
        or ref.kind != "progressive_numerical_failure"
        or ref.media_type != _MEDIA_TYPE
        or ref.bytes != len(payload)
        or ref.sha256 != sha256_bytes(payload)
        or ref.artifact_id != f"progressive-failure:{ref.sha256}"
    ):
        raise ValueError("progressive failure artifact integrity failed")
    metadata_length = struct.unpack("<Q", payload[len(_MAGIC) : len(_MAGIC) + 8])[0]
    start = len(_MAGIC) + 8
    end = start + metadata_length
    metadata = parse_canonical_json(payload[start:end])
    if (
        not isinstance(metadata, dict)
        or metadata.get("schema_version") != 1
        or not metadata.get("diagnostic_only")
        or metadata.get("resumable") is not False
        or metadata.get("promotion_eligible") is not False
    ):
        raise ValueError("progressive failure artifact metadata is invalid")
    tensor_area = payload[end:]
    expected_offset = 0
    for record in metadata.get("batch_tensors", []):
        offset = record["offset"]
        size = record["bytes"]
        raw = tensor_area[offset : offset + size]
        if offset != expected_offset or len(raw) != size or hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError("progressive failure batch tensor integrity failed")
        expected_offset += size
    if expected_offset != len(tensor_area):
        raise ValueError("progressive failure artifact has trailing bytes")


def persist_progressive_failure_artifact(
    artifact: ProgressiveFailureArtifact, store: FilesystemArtifactStore
) -> ArtifactRef:
    verify_progressive_failure_artifact(artifact)
    stored = store.put_bytes(artifact.artifact_ref, artifact.payload)
    if store.resolve(stored.artifact_id, expected=stored) != artifact.payload:
        raise ValueError("persisted progressive failure artifact did not verify")
    return stored


__all__ = [
    "ProgressiveFailureArtifact",
    "ProgressiveFailurePoint",
    "build_progressive_failure_artifact",
    "persist_progressive_failure_artifact",
    "verify_progressive_failure_artifact",
]
