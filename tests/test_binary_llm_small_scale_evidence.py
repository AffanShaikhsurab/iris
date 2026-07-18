from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from binary_llm.domain import ClaimStatus
from binary_llm.orchestration import (
    PrerequisiteEvidence,
    SmallScaleAcceptanceHarness,
    SmallScaleAcceptanceInputs,
    SmallScaleEvidenceKind,
    SmallScalePrerequisite,
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _prerequisites() -> tuple[PrerequisiteEvidence, ...]:
    return tuple(
        PrerequisiteEvidence(item, True, _hash(item.value))
        for item in SmallScalePrerequisite
    )


def _records() -> dict[SmallScaleEvidenceKind, dict[str, object]]:
    return {
        SmallScaleEvidenceKind.RAW_OUTPUTS: {"artifact_refs": [_hash("raw")]},
        SmallScaleEvidenceKind.CHECKPOINTS: {"artifact_refs": [_hash("checkpoint")]},
        SmallScaleEvidenceKind.RESOURCE_LEDGER: {"tokens": 100, "steps": 10},
        SmallScaleEvidenceKind.ABLATIONS: {"no_initialization": _hash("ablation")},
        SmallScaleEvidenceKind.CONFIDENCE_INTERVALS: {"loss_delta": [-0.2, -0.1]},
        SmallScaleEvidenceKind.CLAIM_STATUSES: {"claim-1": ClaimStatus.REPRODUCED.value},
        SmallScaleEvidenceKind.PROGRESSIVE: {"evidence_ref": _hash("progressive")},
        SmallScaleEvidenceKind.SIGN: {"evidence_ref": _hash("sign")},
        SmallScaleEvidenceKind.PROMOTION_DECISION: {"status": "small_validated"},
    }


def _inputs() -> SmallScaleAcceptanceInputs:
    return SmallScaleAcceptanceInputs(
        experiment_id="small-paper-reference",
        candidate_id="candidate-sign",
        protocol_suite_sha256=_hash("protocol-suite"),
        prerequisites=_prerequisites(),
        records=_records(),
        created_at="2026-01-01T00:00:00Z",
    )


def test_harness_captures_every_required_record_with_stable_content_identities() -> None:
    harness = SmallScaleAcceptanceHarness()
    first = harness.capture(_inputs())
    reordered = dict(reversed(tuple(_records().items())))
    second = harness.capture(replace(_inputs(), records=reordered))

    assert set(first.records) == set(SmallScaleEvidenceKind)
    assert all(record.content_id.startswith(f"small-scale-{kind.value.replace('_', '-')}:sha256:")
               for kind, record in first.records.items())
    assert first.content_id == second.content_id
    assert first.prerequisite_sha256[SmallScalePrerequisite.PACKING_PARITY] == _hash(
        SmallScalePrerequisite.PACKING_PARITY.value
    )


@pytest.mark.parametrize("problem", ["missing", "failed"])
def test_harness_fails_closed_when_any_ordinary_validation_prerequisite_does_not_pass(
    problem: str,
) -> None:
    inputs = _inputs()
    prerequisites = list(inputs.prerequisites)
    target = SmallScalePrerequisite.TINY_PROGRESSIVE
    index = next(i for i, item in enumerate(prerequisites) if item.prerequisite is target)
    if problem == "missing":
        prerequisites.pop(index)
    else:
        prerequisites[index] = replace(prerequisites[index], passed=False)

    with pytest.raises(ValueError, match=rf"{problem}:{target.value}"):
        SmallScaleAcceptanceHarness().capture(
            replace(inputs, prerequisites=tuple(prerequisites))
        )


def test_harness_rejects_incomplete_records_and_invalid_claim_statuses() -> None:
    inputs = _inputs()
    incomplete = dict(inputs.records)
    incomplete.pop(SmallScaleEvidenceKind.SIGN)
    with pytest.raises(ValueError, match="missing=.*sign"):
        SmallScaleAcceptanceHarness().capture(replace(inputs, records=incomplete))

    invalid_claim = dict(inputs.records)
    invalid_claim[SmallScaleEvidenceKind.CLAIM_STATUSES] = {"claim-1": "assumed"}
    with pytest.raises(ValueError):
        SmallScaleAcceptanceHarness().capture(replace(inputs, records=invalid_claim))


def test_prerequisite_evidence_requires_content_addressed_proof() -> None:
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        PrerequisiteEvidence(SmallScalePrerequisite.SEALS, True, "not-a-hash")
