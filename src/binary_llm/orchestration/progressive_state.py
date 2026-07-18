"""Deterministic progressive phase plans and atomic resumable checkpoints."""

from __future__ import annotations

import copy
import hashlib
import math
import os
import random
import tempfile
from dataclasses import asdict, dataclass, is_dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from binary_llm.domain import (
    AmbiguityEntry,
    AmbiguityStatus,
    ScaleRung,
    canonical_json_bytes,
    sha256_bytes,
)
from binary_llm.math import (
    PhaseIndexConvention,
    ProgressionScheduleConfig,
    REFERENCE_PHASE_COUNT,
    phase_indices,
    progression_parameters,
)

from .corpus import PartitionRef
from .budgets import BudgetUsage
from .registry import COMPLETE_CHECKPOINT_STATE_KEYS

_PHASE_INDEX_AMBIGUITY = "progressive.phase_index"
_CANDIDATE_CONVENTIONS = {
    "zero_based": PhaseIndexConvention.ZERO_BASED,
    "one_based": PhaseIndexConvention.ONE_BASED,
}


class ProgressivePlanKind(StrEnum):
    PRODUCTION = "production"
    DETERMINISTIC_TEST = "deterministic_test"


@dataclass(frozen=True, slots=True)
class PhaseIndexResolution:
    ambiguity_id: str
    register_version: str
    selected_candidate: str
    convention: PhaseIndexConvention
    resolution_hash: str
    evidence_refs: tuple[str, ...]
    scale_rung: ScaleRung

@dataclass(frozen=True, slots=True)
class ProgressivePhase:
    phase_ordinal: int
    schedule_index: int
    progression_parameter: float
    partition_phase_index: int
    partition_id: str
    partition_hash: str
    partition_token_count: int


@dataclass(frozen=True, slots=True)
class ProgressivePhasePlan:
    kind: ProgressivePlanKind
    schedule: ProgressionScheduleConfig
    phase_index_resolution: PhaseIndexResolution
    phases: tuple[ProgressivePhase, ...]
    source_partition_hashes: tuple[str, ...]
    source_partition_seed: int
    reference_phase_count: int = REFERENCE_PHASE_COUNT
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        if not self.phases:
            raise ValueError("a progressive phase plan must contain at least one phase")
        expected_ordinals = tuple(range(1, len(self.phases) + 1))
        if tuple(phase.phase_ordinal for phase in self.phases) != expected_ordinals:
            raise ValueError("phase ordinals must be contiguous and one-based")
        hashes = tuple(phase.partition_hash for phase in self.phases)
        if len(hashes) != len(set(hashes)):
            raise ValueError("each planned phase must bind a distinct partition hash")
        if self.kind is ProgressivePlanKind.PRODUCTION and len(self.phases) != REFERENCE_PHASE_COUNT:
            raise ValueError("production progressive plans require exactly 20 phases")
        if self.kind is ProgressivePlanKind.DETERMINISTIC_TEST and len(self.phases) >= REFERENCE_PHASE_COUNT:
            raise ValueError("deterministic test plans must be shorter than production plans")

    def identity_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind.value,
            "schedule": asdict(self.schedule),
            "phase_index_resolution": asdict(self.phase_index_resolution),
            "phases": [asdict(phase) for phase in self.phases],
            "source_partition_hashes": self.source_partition_hashes,
            "source_partition_seed": self.source_partition_seed,
            "reference_phase_count": self.reference_phase_count,
        }

    @property
    def plan_hash(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.identity_record()))

    @property
    def phase_count(self) -> int:
        return len(self.phases)


@dataclass(frozen=True, slots=True)
class ProgressiveScheduleState:
    plan_hash: str
    completed_phase_count: int
    completed_schedule_index: int | None
    completed_progression_parameter: float | None
    next_schedule_index: int | None
    next_progression_parameter: float | None


@dataclass(frozen=True, slots=True)
class ProgressiveDataCursor:
    plan_hash: str
    completed_partition_hashes: tuple[str, ...]
    next_partition_hash: str | None
    record_offset: int
    token_offset: int
    consumed_tokens: int


@dataclass(frozen=True, slots=True)
class RNGState:
    python_state: tuple[Any, ...]
    torch_cpu_state: Tensor
    torch_cuda_states: tuple[Tensor, ...]

@dataclass(frozen=True, slots=True)
class ProgressiveBoundaryCheckpoint:
    checkpoint_id: str
    parent_checkpoint_id: str
    stage1_parent_checkpoint_id: str
    plan_hash: str
    completed_phase_count: int
    optimizer_step: int
    consumed_tokens: int
    training_state: Mapping[str, Any]
    optimizer_state: Mapping[str, Any]
    rng_state: RNGState
    schedule_state: ProgressiveScheduleState
    data_cursor: ProgressiveDataCursor
    state_hashes: Mapping[str, str]
    budget_usage: BudgetUsage | None = None
    complete: bool = True
    # Schema v2 makes the immutable Stage 1 root explicit. Version 1 must not
    # be inferred for phase 2+ because its immediate parent cannot prove it.
    schema_version: int = 2

    def __post_init__(self) -> None:
        for name in (
            "checkpoint_id",
            "parent_checkpoint_id",
            "stage1_parent_checkpoint_id",
            "plan_hash",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.schema_version != 2:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        if not self.complete:
            raise ValueError("phase-boundary checkpoints must be atomically complete")
        if self.completed_phase_count < 1:
            raise ValueError("a phase-boundary checkpoint must complete at least one phase")
        if self.optimizer_step < 0 or self.consumed_tokens < 0:
            raise ValueError("optimizer_step and consumed_tokens must be non-negative")
        if set(self.state_hashes) != COMPLETE_CHECKPOINT_STATE_KEYS:
            raise ValueError("checkpoint must contain every required state hash exactly once")
        if self.schedule_state.plan_hash != self.plan_hash or self.data_cursor.plan_hash != self.plan_hash:
            raise ValueError("checkpoint schedule and cursor must identify its phase plan")
        if self.schedule_state.completed_phase_count != self.completed_phase_count:
            raise ValueError("schedule boundary does not match checkpoint boundary")
        if self.data_cursor.consumed_tokens != self.consumed_tokens:
            raise ValueError("data cursor token count does not match checkpoint token count")
        verify_progressive_checkpoint(self)


def _resolved_phase_index(
    entry: AmbiguityEntry,
    schedule: ProgressionScheduleConfig,
    scale_rung: ScaleRung,
) -> PhaseIndexResolution:
    if entry.ambiguity_id != _PHASE_INDEX_AMBIGUITY:
        raise ValueError("phase planning requires the progressive.phase_index ambiguity")
    if entry.status is not AmbiguityStatus.RESOLVED:
        raise ValueError("progressive.phase_index must be explicitly resolved")
    if scale_rung not in entry.scale_scope:
        raise ValueError("phase-index resolution does not apply to the requested scale rung")
    selected = entry.selected_candidate
    convention = _CANDIDATE_CONVENTIONS.get(selected or "")
    if convention is None:
        raise ValueError("phase-index resolution selected an unsupported candidate")
    if schedule.phase_index is not convention:
        raise ValueError("schedule convention conflicts with the resolved phase-index ambiguity")
    resolution_hash = sha256_bytes(canonical_json_bytes(entry.to_dict()))
    return PhaseIndexResolution(
        ambiguity_id=entry.ambiguity_id,
        register_version=entry.register_version,
        selected_candidate=selected or "",
        convention=convention,
        resolution_hash=resolution_hash,
        evidence_refs=entry.evidence_refs,
        scale_rung=scale_rung,
    )

def _validated_partitions(partitions: Sequence[PartitionRef]) -> tuple[PartitionRef, ...]:
    ordered = tuple(sorted(partitions, key=lambda partition: partition.phase_index))
    if len(ordered) != REFERENCE_PHASE_COUNT:
        raise ValueError("phase planning requires the frozen set of exactly 20 partitions")
    if tuple(partition.phase_index for partition in ordered) != tuple(range(REFERENCE_PHASE_COUNT)):
        raise ValueError("partition phase indices must cover 0 through 19 exactly once")
    if len({partition.partition_id for partition in ordered}) != REFERENCE_PHASE_COUNT:
        raise ValueError("partition IDs must be unique")
    if len({partition.manifest_hash for partition in ordered}) != REFERENCE_PHASE_COUNT:
        raise ValueError("partition hashes must be unique")
    if len({partition.seed for partition in ordered}) != 1:
        raise ValueError("all progressive partitions must share one data-order seed")
    for partition in ordered:
        digest = partition.manifest_hash
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("partition hashes must be lowercase SHA-256 digests")
        if partition.token_count < 0:
            raise ValueError("partition token counts must be non-negative")
    return ordered


def build_progressive_phase_plan(
    partitions: Sequence[PartitionRef],
    schedule: ProgressionScheduleConfig,
    phase_index_ambiguity: AmbiguityEntry,
    *,
    scale_rung: ScaleRung,
    test_phase_count: int | None = None,
) -> ProgressivePhasePlan:
    """Bind schedule values and partition hashes with no implicit ambiguity choice."""

    if not isinstance(schedule, ProgressionScheduleConfig):
        raise TypeError("schedule must be a ProgressionScheduleConfig")
    ordered = _validated_partitions(partitions)
    resolution = _resolved_phase_index(phase_index_ambiguity, schedule, scale_rung)
    if test_phase_count is None:
        selected_count = REFERENCE_PHASE_COUNT
        kind = ProgressivePlanKind.PRODUCTION
    else:
        if (
            isinstance(test_phase_count, bool)
            or not isinstance(test_phase_count, int)
            or not 1 <= test_phase_count < REFERENCE_PHASE_COUNT
        ):
            raise ValueError("test_phase_count must be between 1 and 19")
        selected_count = test_phase_count
        kind = ProgressivePlanKind.DETERMINISTIC_TEST

    full_indices = phase_indices(REFERENCE_PHASE_COUNT, schedule.phase_index)
    full_parameters = progression_parameters(schedule, phase_count=REFERENCE_PHASE_COUNT)
    phases = tuple(
        ProgressivePhase(
            phase_ordinal=offset + 1,
            schedule_index=full_indices[offset],
            progression_parameter=full_parameters[offset],
            partition_phase_index=partition.phase_index,
            partition_id=partition.partition_id,
            partition_hash=partition.manifest_hash,
            partition_token_count=partition.token_count,
        )
        for offset, partition in enumerate(ordered[:selected_count])
    )
    return ProgressivePhasePlan(
        kind=kind,
        schedule=schedule,
        phase_index_resolution=resolution,
        phases=phases,
        source_partition_hashes=tuple(partition.manifest_hash for partition in ordered),
        source_partition_seed=ordered[0].seed,
    )

def _state_normal_form(value: Any) -> Any:
    if isinstance(value, Tensor):
        tensor = value.detach().cpu().contiguous()
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        return {
            "type": "tensor",
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {"type": "dataclass", "value": _state_normal_form(asdict(value))}
    if isinstance(value, Mapping):
        items = [(_state_normal_form(key), _state_normal_form(item)) for key, item in value.items()]
        items.sort(key=lambda pair: canonical_json_bytes(pair[0]))
        return {"type": "mapping", "items": items}
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [_state_normal_form(item) for item in value]}
    if isinstance(value, list):
        return {"type": "list", "items": [_state_normal_form(item) for item in value]}
    if isinstance(value, bytes):
        return {"type": "bytes", "sha256": hashlib.sha256(value).hexdigest(), "length": len(value)}
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("checkpoint states cannot contain non-finite scalar values")
        return value
    raise TypeError(f"unsupported checkpoint state value: {type(value).__name__}")


def _state_hash(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(_state_normal_form(value)))


def capture_rng_state() -> RNGState:
    cuda_states: tuple[Tensor, ...] = ()
    if torch.cuda.is_available():
        cuda_states = tuple(state.cpu().clone() for state in torch.cuda.get_rng_state_all())
    return RNGState(
        python_state=copy.deepcopy(random.getstate()),
        torch_cpu_state=torch.random.get_rng_state().cpu().clone(),
        torch_cuda_states=cuda_states,
    )


def restore_rng_state(state: RNGState) -> None:
    if not isinstance(state, RNGState):
        raise TypeError("state must be an RNGState")
    random.setstate(state.python_state)
    torch.random.set_rng_state(state.torch_cpu_state.cpu())
    available_cuda_states = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if len(state.torch_cuda_states) != available_cuda_states:
        raise ValueError("checkpoint CUDA RNG state count does not match available devices")
    if state.torch_cuda_states:
        torch.cuda.set_rng_state_all(list(state.torch_cuda_states))

def _schedule_state(plan: ProgressivePhasePlan, completed: int) -> ProgressiveScheduleState:
    previous = plan.phases[completed - 1]
    following = None if completed == plan.phase_count else plan.phases[completed]
    return ProgressiveScheduleState(
        plan_hash=plan.plan_hash,
        completed_phase_count=completed,
        completed_schedule_index=previous.schedule_index,
        completed_progression_parameter=previous.progression_parameter,
        next_schedule_index=None if following is None else following.schedule_index,
        next_progression_parameter=None if following is None else following.progression_parameter,
    )


def _data_cursor(plan: ProgressivePhasePlan, completed: int) -> ProgressiveDataCursor:
    completed_phases = plan.phases[:completed]
    following = None if completed == plan.phase_count else plan.phases[completed]
    return ProgressiveDataCursor(
        plan_hash=plan.plan_hash,
        completed_partition_hashes=tuple(phase.partition_hash for phase in completed_phases),
        next_partition_hash=None if following is None else following.partition_hash,
        record_offset=0,
        token_offset=0,
        consumed_tokens=sum(phase.partition_token_count for phase in completed_phases),
    )


def capture_progressive_boundary_checkpoint(
    *,
    checkpoint_id: str,
    parent_checkpoint_id: str,
    stage1_parent_checkpoint_id: str,
    plan: ProgressivePhasePlan,
    completed_phase_count: int,
    optimizer_step: int,
    model: nn.Module | Mapping[str, Any],
    optimizer: Optimizer | Mapping[str, Any],
    budget_usage: BudgetUsage | None = None,
) -> ProgressiveBoundaryCheckpoint:
    """Snapshot every state required to resume after one complete phase."""

    if (
        isinstance(completed_phase_count, bool)
        or not isinstance(completed_phase_count, int)
        or not 1 <= completed_phase_count <= plan.phase_count
    ):
        raise ValueError("completed_phase_count must identify a completed planned phase")
    if isinstance(optimizer_step, bool) or not isinstance(optimizer_step, int) or optimizer_step < 0:
        raise ValueError("optimizer_step must be a non-negative integer")
    training_source = model.state_dict() if isinstance(model, nn.Module) else model
    optimizer_source = optimizer.state_dict() if isinstance(optimizer, Optimizer) else optimizer
    training_state = copy.deepcopy(dict(training_source))
    optimizer_state = copy.deepcopy(dict(optimizer_source))
    rng_state = capture_rng_state()
    schedule_state = _schedule_state(plan, completed_phase_count)
    data_cursor = _data_cursor(plan, completed_phase_count)
    state_hashes = {
        "training_state": _state_hash(training_state),
        "optimizer_state": _state_hash(optimizer_state),
        "schedule_state": _state_hash(
            {
                "schedule_state": schedule_state,
                "stage1_parent_checkpoint_id": stage1_parent_checkpoint_id,
                "schema_version": 2,
            }
        ),
        "rng_state": _state_hash(rng_state),
        "data_cursor": _state_hash(
            data_cursor
            if budget_usage is None
            else {"data_cursor": data_cursor, "budget_usage": budget_usage}
        ),
    }
    return ProgressiveBoundaryCheckpoint(
        checkpoint_id=checkpoint_id,
        parent_checkpoint_id=parent_checkpoint_id,
        stage1_parent_checkpoint_id=stage1_parent_checkpoint_id,
        plan_hash=plan.plan_hash,
        completed_phase_count=completed_phase_count,
        optimizer_step=optimizer_step,
        consumed_tokens=data_cursor.consumed_tokens,
        training_state=training_state,
        optimizer_state=optimizer_state,
        rng_state=rng_state,
        schedule_state=schedule_state,
        data_cursor=data_cursor,
        state_hashes=state_hashes,
        budget_usage=budget_usage,
    )

def verify_progressive_checkpoint(checkpoint: ProgressiveBoundaryCheckpoint) -> None:
    if getattr(checkpoint, "schema_version", None) != 2:
        raise ValueError(
            "unsupported progressive checkpoint schema; schema v1 has no "
            "verifiable Stage 1 root lineage"
        )
    root = getattr(checkpoint, "stage1_parent_checkpoint_id", None)
    if not isinstance(root, str) or not root.strip():
        raise ValueError("stage1_parent_checkpoint_id must be a non-empty string")
    expected = {
        "training_state": _state_hash(checkpoint.training_state),
        "optimizer_state": _state_hash(checkpoint.optimizer_state),
        "schedule_state": _state_hash(
            {
                "schedule_state": checkpoint.schedule_state,
                "stage1_parent_checkpoint_id": root,
                "schema_version": checkpoint.schema_version,
            }
        ),
        "rng_state": _state_hash(checkpoint.rng_state),
        "data_cursor": _state_hash(
            checkpoint.data_cursor
            if checkpoint.budget_usage is None
            else {
                "data_cursor": checkpoint.data_cursor,
                "budget_usage": checkpoint.budget_usage,
            }
        ),
    }
    if dict(checkpoint.state_hashes) != expected:
        raise ValueError("checkpoint state hashes do not match the captured states")


def save_progressive_checkpoint(
    path: str | os.PathLike[str], checkpoint: ProgressiveBoundaryCheckpoint
) -> Path:
    """Atomically replace ``path`` only after a complete checkpoint is serialized."""

    verify_progressive_checkpoint(checkpoint)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", prefix=f".{destination.name}.", suffix=".tmp",
            dir=destination.parent, delete=False,
        ) as temporary:
            temporary_name = temporary.name
            torch.save(checkpoint, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
        return destination
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def load_progressive_checkpoint(
    path: str | os.PathLike[str], *, plan: ProgressivePhasePlan
) -> ProgressiveBoundaryCheckpoint:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, ProgressiveBoundaryCheckpoint):
        raise ValueError("file does not contain a progressive boundary checkpoint")
    if checkpoint.plan_hash != plan.plan_hash:
        raise ValueError("checkpoint belongs to a different progressive phase plan")
    if checkpoint.completed_phase_count > plan.phase_count:
        raise ValueError("checkpoint boundary exceeds the supplied phase plan")
    verify_progressive_checkpoint(checkpoint)
    return checkpoint


def restore_progressive_checkpoint(
    checkpoint: ProgressiveBoundaryCheckpoint,
    *,
    plan: ProgressivePhasePlan,
    model: nn.Module,
    optimizer: Optimizer,
) -> None:
    """Restore model, optimizer, and all process RNG state after validation."""

    if checkpoint.plan_hash != plan.plan_hash:
        raise ValueError("checkpoint belongs to a different progressive phase plan")
    verify_progressive_checkpoint(checkpoint)
    model.load_state_dict(checkpoint.training_state)
    optimizer.load_state_dict(checkpoint.optimizer_state)
    restore_rng_state(checkpoint.rng_state)


__all__ = [
    "PhaseIndexResolution",
    "ProgressiveBoundaryCheckpoint",
    "ProgressiveDataCursor",
    "ProgressivePhase",
    "ProgressivePhasePlan",
    "ProgressivePlanKind",
    "ProgressiveScheduleState",
    "RNGState",
    "build_progressive_phase_plan",
    "capture_progressive_boundary_checkpoint",
    "capture_rng_state",
    "load_progressive_checkpoint",
    "restore_progressive_checkpoint",
    "restore_rng_state",
    "save_progressive_checkpoint",
    "verify_progressive_checkpoint",
]
