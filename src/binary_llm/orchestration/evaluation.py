"""Immutable temporal records for preregistered evaluation access."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping

from binary_llm.domain.models import EvaluationEvidence


class EvaluationSetPurpose(StrEnum):
    DEVELOPMENT = "development"
    BLIND_EXTERNAL = "blind_external"
    BLIND_PRIVATE = "blind_private"


class EvaluationPanel(StrEnum):
    EXTERNAL = "external"
    PRIVATE = "private"


class EvaluationBlindStatus(StrEnum):
    DEVELOPMENT = "development"
    BLIND = "blind"


def _require_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _instant(value: str) -> datetime:
    _require_text("timestamp", value)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamps must include a UTC offset")
    return parsed

@dataclass(frozen=True, slots=True)
class ThresholdSnapshot:
    snapshot_id: str
    evaluation_set_id: str
    version: int
    thresholds: Mapping[str, int | float]
    frozen_at: str
    previous_snapshot_id: str | None = None

    def __post_init__(self) -> None:
        _require_text("snapshot_id", self.snapshot_id)
        _require_text("evaluation_set_id", self.evaluation_set_id)
        _instant(self.frozen_at)
        if self.version < 1:
            raise ValueError("threshold snapshot version must be positive")
        if not self.thresholds:
            raise ValueError("threshold snapshots cannot be empty")
        frozen: dict[str, int | float] = {}
        for gate_id, threshold in self.thresholds.items():
            _require_text("threshold gate_id", gate_id)
            if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
                raise ValueError("threshold values must be numeric")
            frozen[gate_id] = threshold
        object.__setattr__(self, "thresholds", MappingProxyType(frozen))
        if self.version == 1 and self.previous_snapshot_id is not None:
            raise ValueError("the first threshold snapshot cannot name a predecessor")
        if self.version > 1:
            _require_text("previous_snapshot_id", self.previous_snapshot_id or "")


@dataclass(frozen=True, slots=True)
class EvaluationPreregistration:
    preregistration_id: str
    evaluation_set_id: str
    purpose: EvaluationSetPurpose
    primary_metrics: tuple[str, ...]
    confidence_procedures: tuple[str, ...]
    continuation_floor_ids: tuple[str, ...]
    release_gate_ids: tuple[str, ...]
    stop_condition_ids: tuple[str, ...]
    threshold_snapshot_id: str
    registered_at: str

    def __post_init__(self) -> None:
        for name in ("preregistration_id", "evaluation_set_id", "threshold_snapshot_id"):
            _require_text(name, getattr(self, name))
        _instant(self.registered_at)
        for name in (
            "primary_metrics", "confidence_procedures", "continuation_floor_ids",
            "release_gate_ids", "stop_condition_ids",
        ):
            values = getattr(self, name)
            if not values or any(not value.strip() for value in values):
                raise ValueError(f"{name} must contain explicit identifiers")
            if len(values) != len(set(values)):
                raise ValueError(f"{name} contains duplicate identifiers")


@dataclass(frozen=True, slots=True)
class TemporalEvaluationRecord:
    record_id: str
    preregistration_id: str
    evidence_id: str
    evaluation_set_id: str
    panel: EvaluationPanel
    blind_status: EvaluationBlindStatus
    evaluation_count: int
    threshold_snapshot_id: str
    accessed_at: str

    def __post_init__(self) -> None:
        for name in (
            "record_id", "preregistration_id", "evidence_id", "evaluation_set_id",
            "threshold_snapshot_id",
        ):
            _require_text(name, getattr(self, name))
        _instant(self.accessed_at)
        if self.evaluation_count < 1:
            raise ValueError("evaluation_count must be positive")

@dataclass(frozen=True, slots=True)
class EvaluationHistory:
    preregistration: EvaluationPreregistration
    threshold_history: tuple[ThresholdSnapshot, ...]
    records: tuple[TemporalEvaluationRecord, ...] = ()
    invalidated_decision_ids: tuple[str, ...] = ()
    requires_new_frozen_test_set: bool = False

    def __post_init__(self) -> None:
        if not self.threshold_history:
            raise ValueError("evaluation history requires a threshold snapshot")
        ids = tuple(item.snapshot_id for item in self.threshold_history)
        if len(ids) != len(set(ids)):
            raise ValueError("threshold history contains duplicate snapshots")
        record_ids = tuple(item.record_id for item in self.records)
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("evaluation history contains duplicate records")
        initial = self.threshold_history[0]
        if initial.snapshot_id != self.preregistration.threshold_snapshot_id:
            raise ValueError("preregistration must reference the initial threshold snapshot")
        if initial.evaluation_set_id != self.preregistration.evaluation_set_id:
            raise ValueError("threshold and preregistration evaluation sets differ")
        if _instant(initial.frozen_at) > _instant(self.preregistration.registered_at):
            raise ValueError("thresholds must be frozen no later than preregistration")

    @property
    def development_evaluation_count(self) -> int:
        return sum(
            item.blind_status is EvaluationBlindStatus.DEVELOPMENT
            for item in self.records
        )

    @property
    def blind_accessed(self) -> bool:
        return any(item.blind_status is EvaluationBlindStatus.BLIND for item in self.records)

    @property
    def current_thresholds(self) -> ThresholdSnapshot:
        return self.threshold_history[-1]


def start_evaluation_history(
    preregistration: EvaluationPreregistration,
    thresholds: ThresholdSnapshot,
) -> EvaluationHistory:
    return EvaluationHistory(preregistration, (thresholds,))


def record_evaluation_access(
    history: EvaluationHistory,
    evidence: EvaluationEvidence,
    *,
    record_id: str,
    panel: EvaluationPanel,
    accessed_at: str,
) -> EvaluationHistory:
    if history.requires_new_frozen_test_set:
        raise ValueError("threshold mutation requires a newly frozen evaluation set")
    preregistration = history.preregistration
    if evidence.evaluation_set_id != preregistration.evaluation_set_id:
        raise ValueError("evidence does not belong to the preregistered evaluation set")
    if _instant(accessed_at) < _instant(preregistration.registered_at):
        raise ValueError("blind outputs cannot be accessed before preregistration")
    if _instant(evidence.created_at) < _instant(preregistration.registered_at):
        raise ValueError("evaluation evidence predates preregistration")
    purpose = preregistration.purpose
    expected_panel = {
        EvaluationSetPurpose.BLIND_EXTERNAL: EvaluationPanel.EXTERNAL,
        EvaluationSetPurpose.BLIND_PRIVATE: EvaluationPanel.PRIVATE,
    }.get(purpose)
    if expected_panel is not None and panel is not expected_panel:
        raise ValueError("blind evaluation panel does not match its preregistered purpose")
    if purpose is EvaluationSetPurpose.DEVELOPMENT:
        if evidence.blind_status != EvaluationBlindStatus.DEVELOPMENT.value:
            raise ValueError("a development evaluation cannot be represented as blind")
        count = history.development_evaluation_count + 1
        if evidence.evaluation_ordinal != count:
            raise ValueError("development evaluation ordinal must increment on every access")
        status = EvaluationBlindStatus.DEVELOPMENT
    else:
        if history.records:
            raise ValueError("a blind evaluation set permits only its first access")
        if evidence.blind_status != EvaluationBlindStatus.BLIND.value:
            raise ValueError("blind evidence must retain blind status")
        if evidence.evaluation_ordinal != 1:
            raise ValueError("the first blind evaluation ordinal must be one")
        count = 1
        status = EvaluationBlindStatus.BLIND
    record = TemporalEvaluationRecord(
        record_id, preregistration.preregistration_id, evidence.evidence_id,
        evidence.evaluation_set_id, panel, status, count,
        history.current_thresholds.snapshot_id, accessed_at,
    )
    return replace(history, records=history.records + (record,))

def append_threshold_snapshot(
    history: EvaluationHistory,
    snapshot: ThresholdSnapshot,
    *,
    affected_decision_ids: tuple[str, ...] = (),
) -> EvaluationHistory:
    current = history.current_thresholds
    if snapshot.evaluation_set_id != history.preregistration.evaluation_set_id:
        raise ValueError("a new evaluation set requires a new preregistration history")
    if snapshot.version != current.version + 1:
        raise ValueError("threshold versions must be consecutive")
    if snapshot.previous_snapshot_id != current.snapshot_id:
        raise ValueError("threshold history must preserve its direct predecessor")
    if _instant(snapshot.frozen_at) < _instant(current.frozen_at):
        raise ValueError("threshold history cannot move backwards in time")
    changed = dict(snapshot.thresholds) != dict(current.thresholds)
    if history.blind_accessed and changed:
        if not affected_decision_ids:
            raise ValueError("post-access threshold changes must name invalidated decisions")
        invalidated = history.invalidated_decision_ids + affected_decision_ids
        if len(invalidated) != len(set(invalidated)):
            raise ValueError("invalidated decision identifiers must be unique")
        return replace(
            history,
            threshold_history=history.threshold_history + (snapshot,),
            invalidated_decision_ids=invalidated,
            requires_new_frozen_test_set=True,
        )
    return replace(history, threshold_history=history.threshold_history + (snapshot,))
