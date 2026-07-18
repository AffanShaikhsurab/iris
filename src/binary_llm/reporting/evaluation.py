"""Fail-closed evaluation reports with separate external and private panels."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping

from binary_llm.orchestration.evaluation import EvaluationPanel

if TYPE_CHECKING:
    from binary_llm.orchestration.evidence_coordinators import CapabilityPanelEvidence


class RequiredFailureSlice(StrEnum):
    MATH = "math"
    CODE = "code"
    SAFETY = "safety"
    ROUTE = "route"
    IRIS = "iris"


def _freeze_metrics(values: Mapping[str, int | float]) -> Mapping[str, int | float]:
    frozen: dict[str, int | float] = {}
    for name, value in values.items():
        if not name.strip() or isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("report metrics require named numeric values")
        frozen[name] = value
    return MappingProxyType(frozen)


@dataclass(frozen=True, slots=True)
class PanelEvaluation:
    panel: EvaluationPanel
    evidence_id: str
    per_panel_scores: Mapping[str, int | float]
    aggregates: Mapping[str, int | float]
    candidate_minus_baseline_deltas: Mapping[str, int | float]
    confidence_intervals: Mapping[str, tuple[float, float]]
    paired_failure_categories: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.evidence_id.strip():
            raise ValueError("panel evidence_id must be explicit")
        object.__setattr__(self, "per_panel_scores", _freeze_metrics(self.per_panel_scores))
        object.__setattr__(self, "aggregates", _freeze_metrics(self.aggregates))
        object.__setattr__(
            self, "candidate_minus_baseline_deltas",
            _freeze_metrics(self.candidate_minus_baseline_deltas),
        )
        intervals: dict[str, tuple[float, float]] = {}
        for name, bounds in self.confidence_intervals.items():
            if not name.strip() or len(bounds) != 2 or bounds[0] > bounds[1]:
                raise ValueError("confidence intervals require ordered lower/upper bounds")
            intervals[name] = (float(bounds[0]), float(bounds[1]))
        object.__setattr__(self, "confidence_intervals", MappingProxyType(intervals))
        if len(self.paired_failure_categories) != len(set(self.paired_failure_categories)):
            raise ValueError("paired failure categories must be unique")


@dataclass(frozen=True, slots=True)
class SliceEvaluation:
    slice_name: RequiredFailureSlice
    passed: bool | None
    evidence_refs: tuple[str, ...]
    failure_categories: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.evidence_refs) != len(set(self.evidence_refs)):
            raise ValueError("slice evidence references must be unique")
        if len(self.failure_categories) != len(set(self.failure_categories)):
            raise ValueError("slice failure categories must be unique")


@dataclass(frozen=True, slots=True)
class FailClosedEvaluationReport:
    report_id: str
    external_panel: PanelEvaluation | None
    private_panel: PanelEvaluation | None
    slices: tuple[SliceEvaluation, ...]
    aggregate_passed: bool
    held_out_capability_improved: bool
    training_loss_improved: bool
    blocking_reasons: tuple[str, ...]
    failure_categories: tuple[str, ...]
    created_at: str

    def __post_init__(self) -> None:
        if not self.report_id.strip() or not self.created_at.strip():
            raise ValueError("evaluation report identity and timestamp are required")

    @property
    def complete(self) -> bool:
        return not any(reason.startswith("missing ") for reason in self.blocking_reasons)

    @property
    def qualification_allowed(self) -> bool:
        return not self.blocking_reasons


def _panel_reasons(
    panel: PanelEvaluation | None,
    expected: EvaluationPanel,
) -> list[str]:
    if panel is None:
        return [f"missing {expected.value} evaluation panel"]
    reasons: list[str] = []
    if panel.panel is not expected:
        reasons.append(f"{expected.value} panel has the wrong panel identity")
    for label, values in (
        ("per-panel scores", panel.per_panel_scores),
        ("aggregate scores", panel.aggregates),
        ("candidate-minus-baseline deltas", panel.candidate_minus_baseline_deltas),
        ("confidence intervals", panel.confidence_intervals),
    ):
        if not values:
            reasons.append(f"missing {expected.value} {label}")
    return reasons

def build_fail_closed_evaluation_report(
    *,
    report_id: str,
    external_panel: PanelEvaluation | None,
    private_panel: PanelEvaluation | None,
    slices: tuple[SliceEvaluation, ...],
    aggregate_passed: bool,
    held_out_capability_improved: bool,
    training_loss_improved: bool,
    created_at: str,
) -> FailClosedEvaluationReport:
    reasons = _panel_reasons(external_panel, EvaluationPanel.EXTERNAL)
    reasons.extend(_panel_reasons(private_panel, EvaluationPanel.PRIVATE))
    if (
        external_panel is not None
        and private_panel is not None
        and external_panel.evidence_id == private_panel.evidence_id
    ):
        reasons.append("external and private panels must retain separate evidence")
    by_slice: dict[RequiredFailureSlice, SliceEvaluation] = {}
    duplicates: set[RequiredFailureSlice] = set()
    for item in slices:
        if item.slice_name in by_slice:
            duplicates.add(item.slice_name)
        by_slice[item.slice_name] = item
    for name in sorted(duplicates, key=str):
        reasons.append(f"duplicate required slice: {name.value}")
    for required in RequiredFailureSlice:
        item = by_slice.get(required)
        if item is None:
            reasons.append(f"missing required slice: {required.value}")
        elif item.passed is None or not item.evidence_refs:
            reasons.append(f"missing required slice evidence: {required.value}")
        elif not item.passed:
            reasons.append(f"required slice failed: {required.value}")
    if not aggregate_passed:
        reasons.append("aggregate capability evaluation failed")
    if training_loss_improved and not held_out_capability_improved:
        reasons.append("training-loss-only improvement is unsupported for promotion")
    failures: list[str] = []
    for panel in (external_panel, private_panel):
        if panel is not None:
            failures.extend(panel.paired_failure_categories)
    for item in slices:
        failures.extend(item.failure_categories)
    return FailClosedEvaluationReport(
        report_id, external_panel, private_panel, slices, aggregate_passed,
        held_out_capability_improved, training_loss_improved,
        tuple(dict.fromkeys(reasons)), tuple(dict.fromkeys(failures)), created_at,
    )


def panel_evaluation_from_capability_evidence(
    evidence: CapabilityPanelEvidence,
    *,
    evidence_id: str,
) -> PanelEvaluation:
    """Convert complete coordinator evidence without dropping any paired failure category."""

    if not evidence.complete:
        raise ValueError("capability panel evidence is incomplete")
    return PanelEvaluation(
        panel=evidence.protocol.evaluation_panel,
        evidence_id=evidence_id,
        per_panel_scores=evidence.per_panel_scores,
        aggregates=evidence.aggregates,
        candidate_minus_baseline_deltas=evidence.candidate_minus_baseline_deltas,
        confidence_intervals=evidence.confidence_intervals,
        paired_failure_categories=evidence.failure_categories,
    )
