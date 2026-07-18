"""Fail-closed, content-addressed small-scale acceptance evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping

from binary_llm.domain import ClaimStatus, ContentIdentity, identify_content

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SmallScaleEvidenceKind(StrEnum):
    RAW_OUTPUTS = "raw_outputs"
    CHECKPOINTS = "checkpoints"
    RESOURCE_LEDGER = "resource_ledger"
    ABLATIONS = "ablations"
    CONFIDENCE_INTERVALS = "confidence_intervals"
    CLAIM_STATUSES = "claim_statuses"
    PROGRESSIVE = "progressive"
    SIGN = "sign"
    PROMOTION_DECISION = "promotion_decision"


class SmallScalePrerequisite(StrEnum):
    DETERMINISTIC_CORE = "deterministic_math_domain_unit"
    TINY_STAGE1 = "tiny_stage1"
    TINY_PROGRESSIVE = "tiny_progressive"
    PACKING_PARITY = "packing_reference_runtime_parity"
    SEALS = "seals"
    PROVENANCE = "provenance"
    BUDGETS = "budgets"
    AMBIGUITIES = "required_ambiguity_selections"


@dataclass(frozen=True, slots=True)
class PrerequisiteEvidence:
    prerequisite: SmallScalePrerequisite
    passed: bool
    evidence_sha256: str

    def __post_init__(self) -> None:
        if not _SHA256.fullmatch(self.evidence_sha256):
            raise ValueError("prerequisite evidence must be a lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class SmallScaleAcceptanceInputs:
    experiment_id: str
    candidate_id: str
    protocol_suite_sha256: str
    prerequisites: tuple[PrerequisiteEvidence, ...]
    records: Mapping[SmallScaleEvidenceKind, Mapping[str, Any]]
    created_at: str

    def __post_init__(self) -> None:
        for name in ("experiment_id", "candidate_id", "created_at"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be empty")
        if not _SHA256.fullmatch(self.protocol_suite_sha256):
            raise ValueError("protocol suite identity must be a lowercase SHA-256")
        object.__setattr__(self, "records", MappingProxyType(dict(self.records)))


@dataclass(frozen=True, slots=True)
class SmallScaleEvidenceRecord:
    kind: SmallScaleEvidenceKind
    identity: ContentIdentity

    @property
    def content_id(self) -> str:
        return self.identity.value


@dataclass(frozen=True, slots=True)
class SmallScaleAcceptanceEvidence:
    experiment_id: str
    candidate_id: str
    protocol_suite_sha256: str
    prerequisite_sha256: Mapping[SmallScalePrerequisite, str]
    records: Mapping[SmallScaleEvidenceKind, SmallScaleEvidenceRecord]
    identity: ContentIdentity
    created_at: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "prerequisite_sha256", MappingProxyType(dict(self.prerequisite_sha256))
        )
        object.__setattr__(self, "records", MappingProxyType(dict(self.records)))

    @property
    def content_id(self) -> str:
        return self.identity.value


class SmallScaleAcceptanceHarness:
    """Capture qualifying evidence only after every cheap prerequisite passes."""

    def capture(self, inputs: SmallScaleAcceptanceInputs) -> SmallScaleAcceptanceEvidence:
        prerequisites = self._validate_prerequisites(inputs.prerequisites)
        self._validate_records(inputs.records)
        records = {
            kind: SmallScaleEvidenceRecord(
                kind,
                identify_content(
                    value,
                    kind=f"small-scale-{kind.value.replace('_', '-')}",
                    representation_fields={
                        "experiment_id": inputs.experiment_id,
                        "candidate_id": inputs.candidate_id,
                        "protocol_suite_sha256": inputs.protocol_suite_sha256,
                    },
                ),
            )
            for kind, value in sorted(inputs.records.items(), key=lambda item: item[0].value)
        }
        envelope = {
            "schema_version": 1,
            "experiment_id": inputs.experiment_id,
            "candidate_id": inputs.candidate_id,
            "protocol_suite_sha256": inputs.protocol_suite_sha256,
            "prerequisites": {
                item.value: prerequisites[item] for item in sorted(prerequisites, key=str)
            },
            "evidence_records": {
                kind.value: records[kind].content_id for kind in sorted(records, key=str)
            },
            "created_at": inputs.created_at,
        }
        identity = identify_content(envelope, kind="small-scale-acceptance-evidence")
        return SmallScaleAcceptanceEvidence(
            inputs.experiment_id,
            inputs.candidate_id,
            inputs.protocol_suite_sha256,
            prerequisites,
            records,
            identity,
            inputs.created_at,
        )

    @staticmethod
    def _validate_prerequisites(
        values: tuple[PrerequisiteEvidence, ...],
    ) -> dict[SmallScalePrerequisite, str]:
        seen: dict[SmallScalePrerequisite, PrerequisiteEvidence] = {}
        for item in values:
            if item.prerequisite in seen:
                raise ValueError(f"duplicate prerequisite: {item.prerequisite.value}")
            seen[item.prerequisite] = item
        missing = set(SmallScalePrerequisite) - set(seen)
        failed = {item.prerequisite for item in seen.values() if not item.passed}
        if missing or failed:
            details = [*(f"missing:{item.value}" for item in sorted(missing, key=str)),
                       *(f"failed:{item.value}" for item in sorted(failed, key=str))]
            raise ValueError("small-scale acceptance prerequisites did not pass: " + ", ".join(details))
        return {kind: item.evidence_sha256 for kind, item in seen.items()}

    @staticmethod
    def _validate_records(
        records: Mapping[SmallScaleEvidenceKind, Mapping[str, Any]],
    ) -> None:
        missing = set(SmallScaleEvidenceKind) - set(records)
        extra = set(records) - set(SmallScaleEvidenceKind)
        if missing or extra:
            raise ValueError(
                "acceptance evidence records must be exact; "
                f"missing={sorted(item.value for item in missing)}, extra={list(extra)}"
            )
        empty = [kind.value for kind, value in records.items() if not value]
        if empty:
            raise ValueError(f"acceptance evidence records must not be empty: {sorted(empty)}")
        claim_values = records[SmallScaleEvidenceKind.CLAIM_STATUSES]
        if not claim_values:
            raise ValueError("claim statuses must not be empty")
        for claim_id, status in claim_values.items():
            if not str(claim_id).strip():
                raise ValueError("claim IDs must not be empty")
            ClaimStatus(status)


__all__ = [
    "PrerequisiteEvidence", "SmallScaleAcceptanceEvidence",
    "SmallScaleAcceptanceHarness", "SmallScaleAcceptanceInputs",
    "SmallScaleEvidenceKind", "SmallScaleEvidenceRecord", "SmallScalePrerequisite",
]
