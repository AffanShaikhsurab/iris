from __future__ import annotations

from dataclasses import replace

import pytest

from binary_llm.domain import EvaluationEvidence, ScaleRung
from binary_llm.orchestration import (
    EvaluationPanel,
    EvaluationPreregistration,
    EvaluationSetPurpose,
    GateReport,
    GateResult,
    GateResultStatus,
    ThresholdSnapshot,
    append_threshold_snapshot,
    record_evaluation_access,
    start_evaluation_history,
)
from binary_llm.reporting import (
    DistributionEligibility,
    PanelEvaluation,
    ReleaseCandidateQualification,
    ReleaseDecisionStatus,
    ReleaseGateRequirement,
    RequiredFailureSlice,
    SliceEvaluation,
    build_fail_closed_evaluation_report,
    select_release_candidate,
)

T0 = "2026-07-16T00:00:00Z"
T1 = "2026-07-16T00:01:00Z"
T2 = "2026-07-16T00:02:00Z"
T3 = "2026-07-16T00:03:00Z"


def evidence(
    evidence_id: str,
    evaluation_set_id: str,
    blind_status: str,
    ordinal: int,
    *,
    created_at: str = T2,
) -> EvaluationEvidence:
    return EvaluationEvidence(
        evidence_id=evidence_id, artifact_id="artifact", family_id="binary",
        scale_rung=ScaleRung.IRIS, evaluation_set_id=evaluation_set_id,
        evaluator_revision="eval-v1", blind_status=blind_status,
        evaluation_ordinal=ordinal, seed=7, prompt_order_hash="order-hash",
        decoding={"temperature": 0}, raw_outputs_ref=f"raw-{evidence_id}",
        per_case_ref=f"cases-{evidence_id}", aggregates={"accuracy": 0.9},
        failure_categories=(), created_at=created_at,
    )

def preregistration(
    evaluation_set_id: str,
    purpose: EvaluationSetPurpose,
) -> EvaluationPreregistration:
    return EvaluationPreregistration(
        "prereg-1", evaluation_set_id, purpose,
        ("accuracy",), ("paired-bootstrap-95",), ("continuation",),
        ("release",), ("stop",), "thresholds-v1", T1,
    )


def thresholds(
    evaluation_set_id: str,
    *,
    snapshot_id: str = "thresholds-v1",
    version: int = 1,
    value: float = 0.9,
    frozen_at: str = T0,
    previous: str | None = None,
) -> ThresholdSnapshot:
    return ThresholdSnapshot(
        snapshot_id, evaluation_set_id, version, {"accuracy": value}, frozen_at, previous
    )


def panel(kind: EvaluationPanel, evidence_id: str) -> PanelEvaluation:
    return PanelEvaluation(
        kind, evidence_id, {"math": 0.9}, {"broad": 0.9},
        {"broad": -0.01}, {"broad": (-0.02, 0.0)}, (),
    )


def slices(*, failed: RequiredFailureSlice | None = None) -> tuple[SliceEvaluation, ...]:
    return tuple(
        SliceEvaluation(
            name, name is not failed, (f"slice-{name.value}",),
            (f"{name.value}-failure",) if name is failed else (),
        )
        for name in RequiredFailureSlice
    )


def evaluation_report(*, failed: RequiredFailureSlice | None = None):
    return build_fail_closed_evaluation_report(
        report_id="evaluation-report", external_panel=panel(EvaluationPanel.EXTERNAL, "external"),
        private_panel=panel(EvaluationPanel.PRIVATE, "private"), slices=slices(failed=failed),
        aggregate_passed=True, held_out_capability_improved=True,
        training_loss_improved=True, created_at=T3,
    )


def test_development_access_count_increments_and_cannot_be_represented_as_blind() -> None:
    history = start_evaluation_history(
        preregistration("dev-set", EvaluationSetPurpose.DEVELOPMENT), thresholds("dev-set")
    )
    history = record_evaluation_access(
        history, evidence("dev-1", "dev-set", "development", 1),
        record_id="record-1", panel=EvaluationPanel.EXTERNAL, accessed_at=T3,
    )
    history = record_evaluation_access(
        history, evidence("dev-2", "dev-set", "development", 2),
        record_id="record-2", panel=EvaluationPanel.EXTERNAL, accessed_at=T3,
    )

    assert history.development_evaluation_count == 2
    assert tuple(item.evaluation_count for item in history.records) == (1, 2)
    with pytest.raises(ValueError, match="cannot be represented as blind"):
        record_evaluation_access(
            history, evidence("dev-3", "dev-set", "blind", 3),
            record_id="record-3", panel=EvaluationPanel.EXTERNAL, accessed_at=T3,
        )

def test_blind_access_requires_preregistration_and_matching_private_panel() -> None:
    history = start_evaluation_history(
        preregistration("private-set", EvaluationSetPurpose.BLIND_PRIVATE),
        thresholds("private-set"),
    )
    with pytest.raises(ValueError, match="predates preregistration"):
        record_evaluation_access(
            history, evidence("early", "private-set", "blind", 1, created_at=T0),
            record_id="early-record", panel=EvaluationPanel.PRIVATE, accessed_at=T3,
        )
    with pytest.raises(ValueError, match="panel does not match"):
        record_evaluation_access(
            history, evidence("wrong-panel", "private-set", "blind", 1),
            record_id="wrong-panel-record", panel=EvaluationPanel.EXTERNAL, accessed_at=T3,
        )


def test_post_blind_threshold_change_preserves_history_and_invalidates_decision() -> None:
    history = start_evaluation_history(
        preregistration("blind-set", EvaluationSetPurpose.BLIND_EXTERNAL),
        thresholds("blind-set"),
    )
    history = record_evaluation_access(
        history, evidence("blind-1", "blind-set", "blind", 1),
        record_id="blind-record", panel=EvaluationPanel.EXTERNAL, accessed_at=T3,
    )
    changed = thresholds(
        "blind-set", snapshot_id="thresholds-v2", version=2, value=0.8,
        frozen_at="2026-07-16T00:04:00Z", previous="thresholds-v1",
    )

    history = append_threshold_snapshot(
        history, changed, affected_decision_ids=("promotion-1",)
    )

    assert tuple(item.snapshot_id for item in history.threshold_history) == (
        "thresholds-v1", "thresholds-v2"
    )
    assert history.invalidated_decision_ids == ("promotion-1",)
    assert history.requires_new_frozen_test_set
    with pytest.raises(ValueError, match="newly frozen"):
        record_evaluation_access(
            history, evidence("again", "blind-set", "blind", 1),
            record_id="again-record", panel=EvaluationPanel.EXTERNAL, accessed_at=T3,
        )


def test_report_retains_panel_failures_and_a_passing_aggregate_cannot_hide_a_slice() -> None:
    report = evaluation_report(failed=RequiredFailureSlice.CODE)

    assert not report.qualification_allowed
    assert "required slice failed: code" in report.blocking_reasons
    assert report.failure_categories == ("code-failure",)
    assert report.external_panel is not report.private_panel


def test_report_fails_closed_on_missing_panel_fields_and_training_loss_only_gain() -> None:
    incomplete_external = PanelEvaluation(
        EvaluationPanel.EXTERNAL, "external", {}, {"broad": 1.0},
        {"broad": 0.0}, {"broad": (0.0, 0.0)}, (),
    )
    report = build_fail_closed_evaluation_report(
        report_id="incomplete", external_panel=incomplete_external,
        private_panel=None, slices=slices(), aggregate_passed=True,
        held_out_capability_improved=False, training_loss_improved=True, created_at=T3,
    )

    assert not report.complete
    assert "missing external per-panel scores" in report.blocking_reasons
    assert "missing private evaluation panel" in report.blocking_reasons
    assert "training-loss-only improvement is unsupported for promotion" in report.blocking_reasons

def gate_report(
    report_id: str,
    *,
    failed: ReleaseGateRequirement | None = None,
) -> tuple[GateReport, dict[ReleaseGateRequirement, tuple[str, ...]]]:
    mapping = {
        requirement: (f"gate-{requirement.value}",)
        for requirement in ReleaseGateRequirement
    }
    results = tuple(
        GateResult(
            mapping[requirement][0],
            GateResultStatus.FAIL if requirement is failed else GateResultStatus.PASS,
            0 if requirement is ReleaseGateRequirement.CRITICAL_SAFETY else 1,
            0 if requirement is ReleaseGateRequirement.CRITICAL_SAFETY else 1,
            (f"evidence-{requirement.value}",), T3, requirement.value,
        )
        for requirement in ReleaseGateRequirement
    )
    return GateReport(report_id, "release-gates", results, T3, ()), mapping


def qualification(
    candidate_id: str,
    size: int,
    *,
    failed: ReleaseGateRequirement | None = None,
    distribution: DistributionEligibility | None = None,
) -> ReleaseCandidateQualification:
    report, mapping = gate_report(f"gates-{candidate_id}", failed=failed)
    return ReleaseCandidateQualification(
        candidate_id, "iphone-x", size, report, mapping, evaluation_report(),
        distribution or DistributionEligibility(True, True, True),
    )


def test_release_selects_smallest_complete_qualifier_and_prefers_larger_passer() -> None:
    smaller_failure = qualification(
        "small", 100, failed=ReleaseGateRequirement.RUNTIME_PARITY
    )
    larger_passer = qualification("large", 200)
    largest_passer = qualification("largest", 300)

    decision = select_release_candidate(
        (largest_passer, smaller_failure, larger_passer), device_tier="iphone-x",
        decision_id="release-1", decided_at=T3,
    )

    assert decision.status is ReleaseDecisionStatus.RELEASE
    assert decision.selected_candidate_id == "large"
    assert decision.selected_total_distributable_bytes == 200
    assert decision.qualifying_candidate_ids == ("large", "largest")
    assert not decision.thresholds_lowered


def test_unresolved_distribution_produces_private_only_with_research_evidence() -> None:
    private = qualification(
        "private", 150,
        distribution=DistributionEligibility(False, True, False),
    )

    decision = select_release_candidate(
        (private,), device_tier="iphone-x", decision_id="private-1", decided_at=T3
    )

    assert decision.status is ReleaseDecisionStatus.PRIVATE_ONLY
    assert decision.selected_candidate_id == "private"
    assert decision.reasons == (
        "distribution terms remain unresolved",
        "required attribution remains unresolved",
    )


def test_no_complete_qualifier_reports_no_release_without_lowering_thresholds() -> None:
    failed = qualification("failed", 100, failed=ReleaseGateRequirement.SEMANTIC_QUALITY)
    incomplete = replace(qualification("incomplete", 90), artifact_complete=False)

    decision = select_release_candidate(
        (failed, incomplete), device_tier="iphone-x",
        decision_id="none-1", decided_at=T3,
    )

    assert decision.status is ReleaseDecisionStatus.NO_QUALIFYING_BINARY_RELEASE
    assert decision.selected_candidate_id is None
    assert any("gate-semantic_quality" in reason for reason in decision.reasons)
    assert any("artifact accounting" in reason for reason in decision.reasons)
    assert not decision.thresholds_lowered
