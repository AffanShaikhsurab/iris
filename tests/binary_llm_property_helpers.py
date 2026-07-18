from __future__ import annotations

from binary_llm.domain import ArtifactRef
from binary_llm.orchestration import GateReport, GateResult, GateResultStatus
from binary_llm.orchestration.evaluation import EvaluationPanel
from binary_llm.reporting import (
    DistributionEligibility,
    PanelEvaluation,
    ReleaseCandidateQualification,
    ReleaseGateRequirement,
    RequiredFailureSlice,
    SliceEvaluation,
    build_fail_closed_evaluation_report,
)

NOW = "2026-07-18T00:00:00Z"


def artifact(artifact_id: str, *, parent: str | None = None, size: int = 1) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=artifact_id,
        kind="evidence",
        sha256="a" * 64,
        bytes=size,
        media_type="application/octet-stream",
        parent_artifact_id=parent,
    )


def gate_report(
    report_id: str = "report",
    *,
    failed: ReleaseGateRequirement | None = None,
) -> tuple[GateReport, dict[ReleaseGateRequirement, tuple[str, ...]]]:
    mapping = {item: (f"gate-{item.value}",) for item in ReleaseGateRequirement}
    results = tuple(
        GateResult(
            mapping[item][0],
            GateResultStatus.FAIL if item is failed else GateResultStatus.PASS,
            0 if item is ReleaseGateRequirement.CRITICAL_SAFETY else 1,
            0 if item is ReleaseGateRequirement.CRITICAL_SAFETY else 1,
            (f"evidence-{item.value}",),
            NOW,
            item.value,
        )
        for item in ReleaseGateRequirement
    )
    return GateReport(report_id, "release-gates", results, NOW, ()), mapping


def evaluation_report(*, failed: RequiredFailureSlice | None = None):
    def panel(kind: EvaluationPanel) -> PanelEvaluation:
        return PanelEvaluation(
            kind,
            f"{kind.value}-evidence",
            {"score": 1.0},
            {"aggregate": 1.0},
            {"delta": 0.0},
            {"delta": (0.0, 0.0)},
            (),
        )

    slices = tuple(
        SliceEvaluation(
            item,
            item is not failed,
            (f"slice-{item.value}",),
            (f"{item.value}-failure",) if item is failed else (),
        )
        for item in RequiredFailureSlice
    )
    return build_fail_closed_evaluation_report(
        report_id="evaluation",
        external_panel=panel(EvaluationPanel.EXTERNAL),
        private_panel=panel(EvaluationPanel.PRIVATE),
        slices=slices,
        aggregate_passed=True,
        held_out_capability_improved=True,
        training_loss_improved=False,
        created_at=NOW,
    )


def qualification(
    candidate_id: str,
    size: int,
    *,
    tier: str = "device",
    failed: ReleaseGateRequirement | None = None,
    public: bool = True,
) -> ReleaseCandidateQualification:
    report, mapping = gate_report(candidate_id, failed=failed)
    return ReleaseCandidateQualification(
        candidate_id,
        tier,
        size,
        report,
        mapping,
        evaluation_report(),
        DistributionEligibility(public, True, True),
    )
