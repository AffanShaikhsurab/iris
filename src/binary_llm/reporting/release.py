"""Fail-closed, size-optimal binary release selection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping

from binary_llm.orchestration.gates import GateReport, GateResultStatus
from binary_llm.reporting.evaluation import FailClosedEvaluationReport


class ReleaseGateRequirement(StrEnum):
    SEMANTIC_QUALITY = "semantic_quality"
    CRITICAL_SAFETY = "critical_safety_violations"
    ARTIFACT_SIZE = "artifact_size"
    RUNTIME_PARITY = "runtime_parity"
    MEMORY = "memory"
    LATENCY = "latency"
    THERMAL = "thermal"
    PROVENANCE = "provenance"
    LICENSING = "licensing"
    DETERMINISTIC_REPRODUCTION = "deterministic_reproduction"


class ReleaseDecisionStatus(StrEnum):
    RELEASE = "release"
    PRIVATE_ONLY = "private_only"
    NO_QUALIFYING_BINARY_RELEASE = "no_qualifying_binary_release"


@dataclass(frozen=True, slots=True)
class DistributionEligibility:
    distribution_terms_resolved: bool
    data_provenance_resolved: bool
    required_attribution_resolved: bool

    @property
    def public_release_allowed(self) -> bool:
        return (
            self.distribution_terms_resolved
            and self.data_provenance_resolved
            and self.required_attribution_resolved
        )

    @property
    def blockers(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if not self.distribution_terms_resolved:
            reasons.append("distribution terms remain unresolved")
        if not self.data_provenance_resolved:
            reasons.append("data provenance remains unresolved")
        if not self.required_attribution_resolved:
            reasons.append("required attribution remains unresolved")
        return tuple(reasons)

@dataclass(frozen=True, slots=True)
class ReleaseCandidateQualification:
    candidate_id: str
    device_tier: str
    total_distributable_bytes: int
    gate_report: GateReport
    gate_ids_by_requirement: Mapping[ReleaseGateRequirement, tuple[str, ...]]
    evaluation_report: FailClosedEvaluationReport
    distribution: DistributionEligibility
    artifact_complete: bool = True

    def __post_init__(self) -> None:
        if not self.candidate_id.strip() or not self.device_tier.strip():
            raise ValueError("candidate and device-tier identifiers are required")
        if isinstance(self.total_distributable_bytes, bool) or self.total_distributable_bytes <= 0:
            raise ValueError("total distributable bytes must be positive")
        frozen: dict[ReleaseGateRequirement, tuple[str, ...]] = {}
        for requirement, gate_ids in self.gate_ids_by_requirement.items():
            if not isinstance(requirement, ReleaseGateRequirement):
                raise ValueError("release gate mappings require known requirement keys")
            if not gate_ids or any(not gate_id.strip() for gate_id in gate_ids):
                raise ValueError("release requirements must reference explicit gate identifiers")
            if len(gate_ids) != len(set(gate_ids)):
                raise ValueError("release requirement gate identifiers must be unique")
            frozen[requirement] = tuple(gate_ids)
        object.__setattr__(self, "gate_ids_by_requirement", MappingProxyType(frozen))

    @property
    def blocking_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if not self.artifact_complete:
            reasons.append("artifact accounting or required files are incomplete")
        if not self.evaluation_report.qualification_allowed:
            reasons.append("required evaluation report is incomplete or failed")
        results = {item.gate_id: item for item in self.gate_report.results}
        for requirement in ReleaseGateRequirement:
            gate_ids = self.gate_ids_by_requirement.get(requirement)
            if not gate_ids:
                reasons.append(f"missing release gate coverage: {requirement.value}")
                continue
            for gate_id in gate_ids:
                result = results.get(gate_id)
                if result is None:
                    reasons.append(f"missing release gate result: {gate_id}")
                elif result.status is not GateResultStatus.PASS:
                    reasons.append(
                        f"release gate did not pass: {gate_id} ({result.status.value})"
                    )
        mapped = {
            gate_id
            for gate_ids in self.gate_ids_by_requirement.values()
            for gate_id in gate_ids
        }
        for result in self.gate_report.results:
            if result.gate_id not in mapped:
                reasons.append(f"unclassified release gate result: {result.gate_id}")
            elif result.status is not GateResultStatus.PASS:
                continue
        return tuple(dict.fromkeys(reasons))

    @property
    def gate_qualified(self) -> bool:
        return not self.blocking_reasons

@dataclass(frozen=True, slots=True)
class ReleaseSelectionDecision:
    decision_id: str
    device_tier: str
    status: ReleaseDecisionStatus
    selected_candidate_id: str | None
    selected_total_distributable_bytes: int | None
    considered_candidate_ids: tuple[str, ...]
    qualifying_candidate_ids: tuple[str, ...]
    reasons: tuple[str, ...]
    decided_at: str
    thresholds_lowered: bool = False

    def __post_init__(self) -> None:
        if not self.decision_id.strip() or not self.device_tier.strip() or not self.decided_at.strip():
            raise ValueError("release decision identity, device tier, and timestamp are required")
        if self.thresholds_lowered:
            raise ValueError("release decisions cannot lower preregistered thresholds")
        selected = self.selected_candidate_id is not None
        if selected != (self.selected_total_distributable_bytes is not None):
            raise ValueError("selected candidate identity and byte count must appear together")
        if self.status is ReleaseDecisionStatus.NO_QUALIFYING_BINARY_RELEASE and selected:
            raise ValueError("a no-qualifying-release decision cannot select a candidate")
        if self.status is not ReleaseDecisionStatus.NO_QUALIFYING_BINARY_RELEASE and not selected:
            raise ValueError("release and private-only decisions require a selected candidate")


def select_release_candidate(
    candidates: tuple[ReleaseCandidateQualification, ...],
    *,
    device_tier: str,
    decision_id: str,
    decided_at: str,
) -> ReleaseSelectionDecision:
    if not device_tier.strip():
        raise ValueError("device_tier must be explicit")
    candidate_ids = tuple(item.candidate_id for item in candidates)
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("release candidate identifiers must be unique")
    applicable = tuple(item for item in candidates if item.device_tier == device_tier)
    considered = tuple(item.candidate_id for item in applicable)
    qualified = tuple(item for item in applicable if item.gate_qualified)
    public = tuple(item for item in qualified if item.distribution.public_release_allowed)

    def candidate_order(item: ReleaseCandidateQualification) -> tuple[int, str]:
        return item.total_distributable_bytes, item.candidate_id

    if public:
        selected = min(public, key=candidate_order)
        status = ReleaseDecisionStatus.RELEASE
        reasons = (
            "selected the smallest total distributable artifact passing every required gate",
        )
    elif qualified:
        selected = min(qualified, key=candidate_order)
        status = ReleaseDecisionStatus.PRIVATE_ONLY
        reasons = selected.distribution.blockers
    else:
        selected = None
        status = ReleaseDecisionStatus.NO_QUALIFYING_BINARY_RELEASE
        details = [
            f"{item.candidate_id}: {reason}"
            for item in applicable
            for reason in item.blocking_reasons
        ]
        reasons = tuple(details) or ("no candidate exists for the requested device tier",)
    return ReleaseSelectionDecision(
        decision_id=decision_id,
        device_tier=device_tier,
        status=status,
        selected_candidate_id=None if selected is None else selected.candidate_id,
        selected_total_distributable_bytes=(
            None if selected is None else selected.total_distributable_bytes
        ),
        considered_candidate_ids=considered,
        qualifying_candidate_ids=tuple(
            item.candidate_id for item in sorted(qualified, key=candidate_order)
        ),
        reasons=reasons,
        decided_at=decided_at,
    )
