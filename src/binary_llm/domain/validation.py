"""Fail-closed validation across immutable domain records."""

from __future__ import annotations

from collections.abc import Iterable

from .ambiguities import REQUIRED_AMBIGUITY_KEYS
from .models import (
    AmbiguityRegister,
    AmbiguityStatus,
    BINARY_FAMILY,
    BUILTIN_FAMILIES,
    CandidateRecord,
    ClaimRecord,
    EvaluationEvidence,
    ExperimentFamilyKind,
    ExperimentManifest,
    GateCategory,
    GateSet,
    RepresentationSpec,
    ToleranceSet,
)

REQUIRED_FAILURE_GATE_CATEGORIES = frozenset(
    {
        GateCategory.NO_PROGRESS,
        GateCategory.NUMERICAL_INSTABILITY,
        GateCategory.CAPABILITY_REGRESSION,
        GateCategory.STORAGE,
        GateCategory.RUNTIME,
        GateCategory.PROVENANCE,
        GateCategory.COST,
        GateCategory.SAFETY,
    }
)


def _index_unique(records: Iterable[object], attribute: str, label: str) -> dict[str, object]:
    indexed: dict[str, object] = {}
    for record in records:
        identifier = getattr(record, attribute)
        if identifier in indexed:
            raise ValueError(f"duplicate {label}: {identifier}")
        indexed[identifier] = record
    return indexed


def validate_manifest(
    manifest: ExperimentManifest,
    *,
    ambiguity_register: AmbiguityRegister,
    gate_set: GateSet,
    tolerance_set: ToleranceSet,
    claims: Iterable[ClaimRecord] = (),
    representations: Iterable[RepresentationSpec] = (),
) -> None:
    """Validate all resolvable prerequisites; missing scientific choices fail."""
    if manifest.ambiguity_register_ref != ambiguity_register.register_id:
        raise ValueError("manifest ambiguity register reference does not match supplied register")
    if manifest.gate_set_ref != gate_set.gate_set_id:
        raise ValueError("manifest gate-set reference does not match supplied gate set")
    if manifest.tolerance_set_ref != tolerance_set.tolerance_set_id:
        raise ValueError("manifest tolerance-set reference does not match supplied tolerance set")

    builtin_by_kind = {family.kind: family for family in BUILTIN_FAMILIES}
    if manifest.family.kind is not ExperimentFamilyKind.HYBRID:
        expected = builtin_by_kind[manifest.family.kind]
        if manifest.family.family_id != expected.family_id:
            raise ValueError("single-family experiments must use the canonical family identity")

    claim_by_id = _index_unique(claims, "claim_id", "claim ID")
    missing_claims = set(manifest.source_claim_refs) - set(claim_by_id)
    if missing_claims:
        raise ValueError(f"unresolved source claim references: {sorted(missing_claims)}")

    ambiguity_by_id = _index_unique(ambiguity_register.entries, "ambiguity_id", "ambiguity ID")
    missing_ambiguities = set(REQUIRED_AMBIGUITY_KEYS) - set(ambiguity_by_id)
    if missing_ambiguities:
        raise ValueError(f"required ambiguity entries are missing: {sorted(missing_ambiguities)}")

    if manifest.family == BINARY_FAMILY:
        unresolved = []
        unscoped = []
        for key in REQUIRED_AMBIGUITY_KEYS:
            entry = ambiguity_by_id[key]
            if entry.status is not AmbiguityStatus.RESOLVED:
                unresolved.append(key)
            elif manifest.scale_rung not in entry.scale_scope:
                unscoped.append(key)
        if unresolved:
            raise ValueError(f"binary manifest has unresolved scientific choices: {unresolved}")
        if unscoped:
            raise ValueError(f"ambiguity resolutions do not cover scale rung: {unscoped}")

    gate_categories = {definition.category for definition in gate_set.definitions}
    missing_gates = REQUIRED_FAILURE_GATE_CATEGORIES - gate_categories
    if missing_gates:
        raise ValueError(
            "required failure-gate categories are missing: "
            f"{sorted(item.value for item in missing_gates)}"
        )

    representation_by_id = _index_unique(representations, "representation_id", "representation ID")
    if manifest.artifact_format is not None:
        missing_representations = set(manifest.artifact_format.allowed_representation_ids) - set(
            representation_by_id
        )
        if missing_representations:
            raise ValueError(f"artifact format references unknown representations: {sorted(missing_representations)}")


def validate_candidate(
    candidate: CandidateRecord,
    *,
    evidence: Iterable[EvaluationEvidence],
) -> None:
    """Prevent evidence or deployment properties from crossing family/artifact identities."""
    evidence_by_id = _index_unique(evidence, "evidence_id", "evidence ID")
    missing = set(candidate.evidence_refs) - set(evidence_by_id)
    if missing:
        raise ValueError(f"candidate references missing evidence: {sorted(missing)}")
    expected_artifact_id = (
        candidate.artifact.artifact_id if candidate.artifact is not None else candidate.source_checkpoint.artifact_id
    )
    for evidence_id in candidate.evidence_refs:
        item = evidence_by_id[evidence_id]
        if item.family_id != candidate.family.family_id:
            raise ValueError("candidate evidence belongs to a different experiment family")
        if item.scale_rung is not candidate.scale_rung:
            raise ValueError("candidate evidence belongs to a different model scale")
        if item.artifact_id != expected_artifact_id:
            raise ValueError("candidate evidence belongs to a different artifact")
    if candidate.deployable_runtime_supported and not candidate.metadata_accounting_complete:
        raise ValueError("deployable candidates require complete metadata accounting")
