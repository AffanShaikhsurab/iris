"""Pure declarative gate evaluation and monotonic scale promotion."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping

from binary_llm.domain.models import (
    AmbiguityRegister,
    AmbiguityStatus,
    CheckpointBoundary,
    GateComparator,
    GateDefinition,
    GateSet,
    ScaleRung,
    StopMode,
)


class GateResultStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    MISSING = "missing"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True, slots=True)
class MetricEvidence:
    """One identified evidence record available to declarative gates."""

    kind: str
    scope: str
    metrics: Mapping[str, Any]
    evidence_refs: tuple[str, ...]
    applicable: bool = True

    def __post_init__(self) -> None:
        if not self.kind or not self.scope or not self.evidence_refs:
            raise ValueError("evidence kind, scope, and references must be explicit")
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    items: tuple[MetricEvidence, ...]


@dataclass(frozen=True, slots=True)
class GateResult:
    gate_id: str
    status: GateResultStatus
    observed: int | float | None
    threshold: int | float
    evidence_refs: tuple[str, ...]
    evaluated_at: str
    category: str
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class GateReport:
    report_id: str
    gate_set_id: str
    results: tuple[GateResult, ...]
    evaluated_at: str
    immediate_stop_gate_ids: tuple[str, ...]

    @property
    def promotion_allowed(self) -> bool:
        return all(
            item.status in (GateResultStatus.PASS, GateResultStatus.NOT_APPLICABLE)
            for item in self.results
        )

    @property
    def continuation_allowed(self) -> bool:
        return self.promotion_allowed

    @property
    def stop_requested(self) -> bool:
        return bool(self.immediate_stop_gate_ids)


def _lookup(metrics: Mapping[str, Any], path: str) -> Any:
    current: Any = metrics
    for component in path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            return None
        current = current[component]
    return current


def _compare(observed: float, comparator: GateComparator, threshold: float) -> bool:
    if comparator is GateComparator.LT:
        return observed < threshold
    if comparator is GateComparator.LE:
        return observed <= threshold
    if comparator is GateComparator.EQ:
        return observed == threshold
    if comparator is GateComparator.GE:
        return observed >= threshold
    return observed > threshold


def _matching(bundle: EvidenceBundle, gate: GateDefinition) -> tuple[MetricEvidence, ...]:
    return tuple(
        item for item in bundle.items
        if item.kind == gate.required_evidence_kind and item.scope == gate.scope
    )


def _evaluate_one(
    gate: GateDefinition,
    evidence: EvidenceBundle,
    baseline: EvidenceBundle | None,
    evaluated_at: str,
) -> GateResult:
    matches = _matching(evidence, gate)
    refs = tuple(dict.fromkeys(ref for item in matches for ref in item.evidence_refs))
    if matches and not any(item.applicable for item in matches):
        return GateResult(
            gate.gate_id, GateResultStatus.NOT_APPLICABLE, None, gate.threshold,
            refs, evaluated_at, gate.category.value,
        )
    applicable = tuple(item for item in matches if item.applicable)
    values = tuple(_lookup(item.metrics, gate.metric_path) for item in applicable)
    values = tuple(value for value in values if value is not None)
    if len(values) != 1 or isinstance(values[0], bool) if values else True:
        return GateResult(
            gate.gate_id, GateResultStatus.MISSING, None, gate.threshold,
            refs, evaluated_at, gate.category.value,
            "required evidence or a unique numeric observation is missing",
        )
    observed = values[0]
    if not isinstance(observed, (int, float)):
        return GateResult(
            gate.gate_id, GateResultStatus.MISSING, None, gate.threshold,
            refs, evaluated_at, gate.category.value, "observation is not numeric",
        )
    comparison_value = float(observed)
    if gate.baseline_relative:
        if baseline is None:
            return GateResult(
                gate.gate_id, GateResultStatus.MISSING, observed, gate.threshold,
                refs, evaluated_at, gate.category.value, "baseline evidence is missing",
            )
        baseline_matches = tuple(item for item in _matching(baseline, gate) if item.applicable)
        baseline_values = tuple(
            _lookup(item.metrics, gate.metric_path) for item in baseline_matches
        )
        baseline_values = tuple(value for value in baseline_values if value is not None)
        if len(baseline_values) != 1 or isinstance(baseline_values[0], bool) or not isinstance(
            baseline_values[0], (int, float)
        ):
            return GateResult(
                gate.gate_id, GateResultStatus.MISSING, observed, gate.threshold,
                refs, evaluated_at, gate.category.value,
                "unique numeric baseline evidence is missing",
            )
        comparison_value -= float(baseline_values[0])
    passed = math.isfinite(comparison_value) and _compare(
        comparison_value, gate.comparator, float(gate.threshold)
    )
    return GateResult(
        gate.gate_id, GateResultStatus.PASS if passed else GateResultStatus.FAIL,
        observed, gate.threshold, refs, evaluated_at, gate.category.value,
        None if passed else "observed value crossed the preregistered gate",
    )


class GateEngine:
    """Evaluate every declared gate without mutating evidence or run state."""

    def evaluate(
        self,
        gate_set: GateSet,
        evidence: EvidenceBundle,
        *,
        report_id: str,
        evaluated_at: str,
        baseline: EvidenceBundle | None = None,
    ) -> GateReport:
        if not report_id or not evaluated_at:
            raise ValueError("gate reports require an identity and evaluation time")
        results = tuple(
            _evaluate_one(gate, evidence, baseline, evaluated_at)
            for gate in gate_set.definitions
        )
        by_id = {gate.gate_id: gate for gate in gate_set.definitions}
        stopping = tuple(
            result.gate_id for result in results
            if result.status in (GateResultStatus.FAIL, GateResultStatus.MISSING)
            and by_id[result.gate_id].stop_mode is StopMode.IMMEDIATE_SAFE_BOUNDARY
        )
        return GateReport(report_id, gate_set.gate_set_id, results, evaluated_at, stopping)


class RunGateStatus(StrEnum):
    ACTIVE = "active"
    STOP_PENDING = "stop_pending"
    STOPPED = "stopped"
    COMPLETED = "completed"


@dataclass(frozen=True, slots=True)
class RunGateStateMachine:
    checkpoint_boundary: CheckpointBoundary
    status: RunGateStatus = RunGateStatus.ACTIVE
    stopping_gate_ids: tuple[str, ...] = ()

    def apply(self, report: GateReport) -> RunGateStateMachine:
        if self.status is not RunGateStatus.ACTIVE or not report.stop_requested:
            return self
        return RunGateStateMachine(
            self.checkpoint_boundary, RunGateStatus.STOP_PENDING,
            report.immediate_stop_gate_ids,
        )

    def checkpoint_completed(self, boundary: CheckpointBoundary) -> RunGateStateMachine:
        if self.status is not RunGateStatus.STOP_PENDING:
            return self
        if boundary is not self.checkpoint_boundary:
            return self
        return RunGateStateMachine(
            self.checkpoint_boundary, RunGateStatus.STOPPED, self.stopping_gate_ids
        )

    def complete(self) -> RunGateStateMachine:
        if self.status is RunGateStatus.STOP_PENDING:
            raise ValueError("a pending stop must reach its safe checkpoint boundary")
        return RunGateStateMachine(
            self.checkpoint_boundary, RunGateStatus.COMPLETED, self.stopping_gate_ids
        )


@dataclass(frozen=True, slots=True)
class AmbiguitySnapshot:
    snapshot_ref: str
    register_id: str
    register_version: str
    scale_rung: ScaleRung
    unresolved_ids: tuple[str, ...]


def snapshot_ambiguities(
    register: AmbiguityRegister, scale_rung: ScaleRung, *, snapshot_ref: str
) -> AmbiguitySnapshot:
    unresolved = tuple(
        entry.ambiguity_id for entry in register.entries
        if scale_rung in entry.scale_scope and entry.status is not AmbiguityStatus.RESOLVED
    )
    return AmbiguitySnapshot(
        snapshot_ref, register.register_id, register.version, scale_rung, unresolved
    )


@dataclass(frozen=True, slots=True)
class MaterialChange:
    change_id: str
    operator_changed: bool
    ambiguity_changed: bool

    def __post_init__(self) -> None:
        if not self.change_id or not (self.operator_changed or self.ambiguity_changed):
            raise ValueError("a material change must identify an operator or ambiguity change")


@dataclass(frozen=True, slots=True)
class ScaleValidation:
    rung: ScaleRung
    family_id: str
    operator_id: str
    ambiguity_snapshot_ref: str
    gate_report_ref: str
    evidence_ref: str
    qualified: bool
    screening: bool = False
    material_change_id: str | None = None


@dataclass(frozen=True, slots=True)
class RecoveryBudgetState:
    exhausted: bool
    release_gates_passed: bool
    held_out_improving: bool


class PromotionStatus(StrEnum):
    PROMOTE = "promote"
    BLOCK = "block"
    STOP = "stop"


class PromotionState(StrEnum):
    SCREENING = "screening"
    INTERMEDIATE_ELIGIBLE = "intermediate_eligible"
    IRIS_ELIGIBLE = "iris_eligible"
    BLOCKED = "blocked"
    STOP_PENDING = "stop_pending"


@dataclass(frozen=True, slots=True)
class PromotionRequest:
    decision_id: str
    candidate_id: str
    family_id: str
    from_rung: ScaleRung
    gate_report: GateReport
    ambiguity_snapshot: AmbiguitySnapshot
    operator_id: str
    validations: tuple[ScaleValidation, ...]
    decided_at: str
    reference_regime_reproduced: bool = False
    material_change: MaterialChange | None = None
    current_screening_evidence_ref: str | None = None
    recovery_budget: RecoveryBudgetState | None = None
    export_based: bool = False
    progressive_report: GateReport | None = None


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    decision_id: str
    candidate_id: str
    from_rung: ScaleRung
    to_rung: ScaleRung | None
    gate_report_ref: str
    ambiguity_snapshot_ref: str
    status: PromotionStatus
    state: PromotionState
    reasons: tuple[str, ...]
    decided_at: str


def _qualified_validation(
    request: PromotionRequest, rung: ScaleRung, *, screening: bool = False
) -> bool:
    return any(
        item.rung is rung
        and item.family_id == request.family_id
        and item.operator_id == request.operator_id
        and item.ambiguity_snapshot_ref == request.ambiguity_snapshot.snapshot_ref
        and item.qualified
        and (item.screening or not screening)
        for item in request.validations
    )


def _has_required_screening(request: PromotionRequest) -> bool:
    change = request.material_change
    if change is None:
        return True
    if (
        request.from_rung is ScaleRung.SMALL
        and request.current_screening_evidence_ref
        and request.gate_report.promotion_allowed
    ):
        return True
    return any(
        item.rung is ScaleRung.SMALL
        and item.family_id == request.family_id
        and item.operator_id == request.operator_id
        and item.ambiguity_snapshot_ref == request.ambiguity_snapshot.snapshot_ref
        and item.qualified
        and item.screening
        and item.material_change_id == change.change_id
        for item in request.validations
    )


class PromotionEngine:
    """Compute scale eligibility; callers cannot manually assert promotion."""

    def decide(self, request: PromotionRequest) -> PromotionDecision:
        reasons: list[str] = []
        if request.ambiguity_snapshot.scale_rung is not request.from_rung:
            reasons.append("ambiguity snapshot does not cover the candidate scale")
        if request.ambiguity_snapshot.unresolved_ids:
            reasons.append("required ambiguities remain unresolved")
        if request.gate_report.stop_requested:
            reasons.append("a failure gate requested stopping at the next safe boundary")
        elif not request.gate_report.promotion_allowed:
            reasons.append("one or more required gates failed or have missing evidence")
        if request.export_based and request.progressive_report is None:
            reasons.append("export promotion requires progressive-view gate evidence")
        elif (
            request.export_based
            and request.progressive_report is not None
            and request.progressive_report.promotion_allowed
            and not request.gate_report.promotion_allowed
        ):
            reasons.append("sign-substituted evidence cannot bypass a gate passed by the progressive view")
        budget = request.recovery_budget
        budget_stop = bool(
            budget and budget.exhausted and not budget.release_gates_passed
            and not budget.held_out_improving
        )
        if budget_stop:
            reasons.append("recovery budget is exhausted without release gates or improving held-out trend")
        if not _has_required_screening(request):
            reasons.append("material operator or ambiguity change requires new qualifying small screening evidence")

        matching_prior = tuple(
            item for item in request.validations
            if item.family_id == request.family_id and item.qualified
        )
        if (
            request.material_change is None
            and matching_prior
            and not any(
                item.operator_id == request.operator_id
                and item.ambiguity_snapshot_ref == request.ambiguity_snapshot.snapshot_ref
                for item in matching_prior
            )
        ):
            reasons.append("material configuration changed without a declared screened change")
        target: ScaleRung | None = None
        eligible_state = PromotionState.BLOCKED
        if request.from_rung is ScaleRung.SMALL:
            target = ScaleRung.INTERMEDIATE
            eligible_state = PromotionState.INTERMEDIATE_ELIGIBLE
            if not request.reference_regime_reproduced:
                reasons.append("small-scale BinaryLLM reference regime has not been reproduced")
        elif request.from_rung is ScaleRung.INTERMEDIATE:
            target = ScaleRung.IRIS
            eligible_state = PromotionState.IRIS_ELIGIBLE
            if not _qualified_validation(request, ScaleRung.SMALL):
                reasons.append("matching qualifying small-scale validation is missing")
        else:
            reasons.append("Iris is the final scale rung")
        if budget_stop or request.gate_report.stop_requested:
            status = PromotionStatus.STOP
            state = PromotionState.STOP_PENDING
        elif reasons:
            status = PromotionStatus.BLOCK
            state = PromotionState.BLOCKED
            target = None
        else:
            status = PromotionStatus.PROMOTE
            state = eligible_state
        return PromotionDecision(
            request.decision_id, request.candidate_id, request.from_rung, target,
            request.gate_report.report_id, request.ambiguity_snapshot.snapshot_ref,
            status, state, tuple(reasons), request.decided_at,
        )
