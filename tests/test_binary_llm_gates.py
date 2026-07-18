from __future__ import annotations

from dataclasses import replace

from binary_llm.domain import (
    CheckpointBoundary,
    GateCategory,
    GateComparator,
    GateDefinition,
    GateSet,
    ScaleRung,
    StopMode,
)
from binary_llm.orchestration import (
    AmbiguitySnapshot,
    EvidenceBundle,
    GateEngine,
    GateReport,
    GateResult,
    GateResultStatus,
    MaterialChange,
    MetricEvidence,
    PromotionEngine,
    PromotionRequest,
    PromotionState,
    PromotionStatus,
    RecoveryBudgetState,
    RunGateStateMachine,
    RunGateStatus,
    ScaleValidation,
)

NOW = "2026-07-16T00:00:00Z"


def gate(
    gate_id: str,
    path: str,
    *,
    kind: str = "evaluation",
    scope: str = "screen",
    comparator: GateComparator = GateComparator.GE,
    threshold: float = 0.8,
    relative: bool = False,
    stop_mode: StopMode = StopMode.IMMEDIATE_SAFE_BOUNDARY,
) -> GateDefinition:
    return GateDefinition(
        gate_id, GateCategory.CAPABILITY_REGRESSION, path, comparator, threshold,
        relative, scope, kind, stop_mode,
    )


def test_gate_engine_returns_pass_fail_missing_and_not_applicable_results() -> None:
    gates = GateSet(
        "screen-gates",
        (
            gate("passing", "quality"),
            gate("failing", "quality", threshold=0.95),
            gate("missing", "code"),
            gate("not-applicable", "latency", kind="device", scope="physical"),
        ),
    )
    evidence = EvidenceBundle(
        (
            MetricEvidence("evaluation", "screen", {"quality": 0.9}, ("eval-1",)),
            MetricEvidence("device", "physical", {}, ("device-policy",), applicable=False),
        )
    )

    report = GateEngine().evaluate(
        gates, evidence, report_id="report-1", evaluated_at=NOW
    )

    assert tuple(item.status for item in report.results) == (
        GateResultStatus.PASS,
        GateResultStatus.FAIL,
        GateResultStatus.MISSING,
        GateResultStatus.NOT_APPLICABLE,
    )
    assert report.immediate_stop_gate_ids == ("failing", "missing")
    assert not report.continuation_allowed


def test_baseline_relative_gate_uses_observed_minus_baseline_and_fails_nonfinite() -> None:
    gates = GateSet(
        "continuation",
        (
            gate("drop-floor", "parse", threshold=-0.05, relative=True),
            gate("finite", "divergence", comparator=GateComparator.LE, threshold=1.0),
        ),
    )
    candidate = EvidenceBundle(
        (MetricEvidence("evaluation", "screen", {"parse": 0.91, "divergence": float("inf")}, ("candidate",)),)
    )
    baseline = EvidenceBundle(
        (MetricEvidence("evaluation", "screen", {"parse": 0.96}, ("baseline",)),)
    )

    report = GateEngine().evaluate(
        gates, candidate, baseline=baseline, report_id="relative", evaluated_at=NOW
    )

    assert report.results[0].status is GateResultStatus.PASS
    assert report.results[1].status is GateResultStatus.FAIL


def test_failure_stop_becomes_effective_only_at_the_registered_safe_boundary() -> None:
    gates = GateSet("safety", (gate("safety", "violations", comparator=GateComparator.EQ, threshold=0),))
    evidence = EvidenceBundle(
        (MetricEvidence("evaluation", "screen", {"violations": 1}, ("safety-evidence",)),)
    )
    report = GateEngine().evaluate(
        gates, evidence, report_id="safety-report", evaluated_at=NOW
    )

    state = RunGateStateMachine(CheckpointBoundary.PROGRESSIVE_PHASE).apply(report)

    assert state.status is RunGateStatus.STOP_PENDING
    assert state.checkpoint_completed(CheckpointBoundary.OPTIMIZER_STEP).status is RunGateStatus.STOP_PENDING
    assert state.checkpoint_completed(CheckpointBoundary.PROGRESSIVE_PHASE).status is RunGateStatus.STOPPED


def passing_report(report_id: str = "pass") -> GateReport:
    return GateReport(
        report_id,
        "gates",
        (
            GateResult(
                "all", GateResultStatus.PASS, 1.0, 1.0, ("evidence",), NOW,
                GateCategory.CONTINUATION.value,
            ),
        ),
        NOW,
        (),
    )


def blocked_report(report_id: str = "blocked") -> GateReport:
    return GateReport(
        report_id,
        "gates",
        (
            GateResult(
                "capability", GateResultStatus.FAIL, 0.5, 0.8, ("evidence",), NOW,
                GateCategory.CAPABILITY_REGRESSION.value,
            ),
        ),
        NOW,
        (),
    )


def snapshot(rung: ScaleRung, ref: str = "ambiguities-v2") -> AmbiguitySnapshot:
    return AmbiguitySnapshot(ref, "register", "2", rung, ())


def request(**changes: object) -> PromotionRequest:
    base = PromotionRequest(
        decision_id="decision",
        candidate_id="candidate",
        family_id="binary",
        from_rung=ScaleRung.SMALL,
        gate_report=passing_report(),
        ambiguity_snapshot=snapshot(ScaleRung.SMALL),
        operator_id="operator-v2",
        validations=(),
        decided_at=NOW,
        reference_regime_reproduced=True,
    )
    return replace(base, **changes)


def test_scale_promotion_is_monotonic_and_requires_matching_lower_scale_validation() -> None:
    engine = PromotionEngine()
    small = engine.decide(request())
    intermediate_request = request(
        from_rung=ScaleRung.INTERMEDIATE,
        ambiguity_snapshot=snapshot(ScaleRung.INTERMEDIATE),
    )

    blocked = engine.decide(intermediate_request)
    eligible = engine.decide(
        replace(
            intermediate_request,
            validations=(
                ScaleValidation(
                    ScaleRung.SMALL, "binary", "operator-v2", "ambiguities-v2",
                    "small-report", "small-evidence", True,
                ),
            ),
        )
    )

    assert (small.status, small.to_rung, small.state) == (
        PromotionStatus.PROMOTE, ScaleRung.INTERMEDIATE, PromotionState.INTERMEDIATE_ELIGIBLE
    )
    assert blocked.status is PromotionStatus.BLOCK
    assert eligible.status is PromotionStatus.PROMOTE
    assert eligible.to_rung is ScaleRung.IRIS


def test_material_change_requires_new_matching_small_screening_evidence() -> None:
    change = MaterialChange("operator-v2-change", operator_changed=True, ambiguity_changed=False)
    changed = request(
        from_rung=ScaleRung.INTERMEDIATE,
        ambiguity_snapshot=snapshot(ScaleRung.INTERMEDIATE),
        material_change=change,
    )
    validation = ScaleValidation(
        ScaleRung.SMALL, "binary", "operator-v2", "ambiguities-v2",
        "screen-report", "new-screen", True, screening=True,
        material_change_id="operator-v2-change",
    )

    assert PromotionEngine().decide(changed).status is PromotionStatus.BLOCK
    decision = PromotionEngine().decide(replace(changed, validations=(validation,)))
    assert decision.status is PromotionStatus.PROMOTE
    assert decision.to_rung is ScaleRung.IRIS


def test_sign_view_and_exhausted_recovery_budget_cannot_bypass_gates() -> None:
    sign_decision = PromotionEngine().decide(
        request(
            gate_report=blocked_report("sign"),
            export_based=True,
            progressive_report=passing_report("progressive"),
        )
    )
    exhausted = PromotionEngine().decide(
        request(
            recovery_budget=RecoveryBudgetState(
                exhausted=True, release_gates_passed=False, held_out_improving=False
            )
        )
    )

    assert sign_decision.status is PromotionStatus.BLOCK
    assert any("sign-substituted" in reason for reason in sign_decision.reasons)
    assert exhausted.status is PromotionStatus.STOP
    assert exhausted.state is PromotionState.STOP_PENDING
