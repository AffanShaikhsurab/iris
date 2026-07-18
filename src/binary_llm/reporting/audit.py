"""Pure completeness contract for content-addressed promotion audit reports."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from binary_llm.domain.models import ArtifactRef


REQUIRED_PROMOTION_REPORT_SECTIONS = (
    "hypotheses",
    "configurations",
    "ablations",
    "failures",
    "capability",
    "accounting",
    "parity",
    "device",
    "license",
    "risk",
)


@dataclass(frozen=True, slots=True)
class PromotionAuditReport:
    report_id: str
    candidate_id: str
    sections: Mapping[str, tuple[ArtifactRef, ...]]
    baseline_refs: tuple[ArtifactRef, ...]

    def __post_init__(self) -> None:
        if not self.report_id.strip() or not self.candidate_id.strip():
            raise ValueError("report and candidate identities are required")
        missing = set(REQUIRED_PROMOTION_REPORT_SECTIONS) - set(self.sections)
        extra = set(self.sections) - set(REQUIRED_PROMOTION_REPORT_SECTIONS)
        if missing or extra:
            raise ValueError(
                f"promotion report sections differ: missing={sorted(missing)}, extra={sorted(extra)}"
            )
        frozen: dict[str, tuple[ArtifactRef, ...]] = {}
        for section in REQUIRED_PROMOTION_REPORT_SECTIONS:
            references = tuple(self.sections[section])
            if not references:
                raise ValueError(f"promotion report section is empty: {section}")
            if len({item.artifact_id for item in references}) != len(references):
                raise ValueError(f"promotion report section has duplicate references: {section}")
            frozen[section] = references
        if not self.baseline_refs:
            raise ValueError("promotion report must retain baseline oracle references")
        baseline_ids = {item.artifact_id for item in self.baseline_refs}
        candidate_ids = {
            item.artifact_id
            for references in frozen.values()
            for item in references
        }
        if baseline_ids & candidate_ids:
            raise ValueError("baseline oracle references must remain distinct from candidate evidence")
        object.__setattr__(self, "sections", MappingProxyType(frozen))


def build_promotion_audit_report(
    *,
    report_id: str,
    candidate_id: str,
    sections: Mapping[str, tuple[ArtifactRef, ...]],
    baseline_refs: tuple[ArtifactRef, ...],
) -> PromotionAuditReport:
    """Join immutable references without claiming that release evidence exists."""

    return PromotionAuditReport(report_id, candidate_id, sections, baseline_refs)


__all__ = [
    "REQUIRED_PROMOTION_REPORT_SECTIONS",
    "PromotionAuditReport",
    "build_promotion_audit_report",
]
