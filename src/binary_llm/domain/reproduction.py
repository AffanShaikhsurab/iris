"""Pure source-setting comparison and scale-scoped claim classification."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .models import (
    ClaimRecord,
    ClaimStatus,
    ComponentAblationResult,
    ExperimentManifest,
    ReproductionAssessment,
    ReproductionClassification,
    ReproductionComponent,
    ReproductionSettings,
    ScaleRung,
    SettingDifference,
)

_MISSING = object()
_COMPONENTS = tuple(ReproductionComponent)


def _escape_pointer(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _same_value(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        return left.keys() == right.keys() and all(_same_value(left[key], right[key]) for key in left)
    if isinstance(left, tuple):
        return len(left) == len(right) and all(_same_value(a, b) for a, b in zip(left, right))
    return left == right


def _raw_differences(
    source: Mapping[str, Any],
    resolved: Mapping[str, Any],
    prefix: str,
) -> list[tuple[str, bool, bool, Any, Any]]:
    differences: list[tuple[str, bool, bool, Any, Any]] = []
    for key in sorted(set(source) | set(resolved)):
        source_value = source.get(key, _MISSING)
        resolved_value = resolved.get(key, _MISSING)
        path = f"{prefix}/{_escape_pointer(key)}"
        if isinstance(source_value, Mapping) and isinstance(resolved_value, Mapping):
            differences.extend(_raw_differences(source_value, resolved_value, path))
        elif source_value is _MISSING or resolved_value is _MISSING or not _same_value(source_value, resolved_value):
            differences.append((path, source_value is not _MISSING, resolved_value is not _MISSING,
                                None if source_value is _MISSING else source_value,
                                None if resolved_value is _MISSING else resolved_value))
    return differences


def resolved_settings_from_manifest(manifest: ExperimentManifest) -> ReproductionSettings:
    """Project a resolved manifest onto the five source-comparison components."""
    return ReproductionSettings(
        operator={
            "binary_scope": manifest.binary_scope,
            "operator_config": manifest.operator_config,
        },
        data={
            "corpus_ref": manifest.corpus_ref,
            "partition_refs": manifest.partition_refs,
            "recovery_config": manifest.recovery_config,
            "teacher_refs": manifest.teacher_refs,
        },
        schedule={"progressive_config": manifest.progressive_config},
        optimization={
            "stage1_config": manifest.stage1_config,
            "seed_set": manifest.seed_set,
            "determinism_policy": manifest.determinism_policy,
            "budget": manifest.budget.to_dict(),
        },
        evaluation={
            "frozen_evaluation_refs": manifest.frozen_evaluation_refs,
            "evaluator_protocols": manifest.evaluator_protocols,
        },
    )


def reproduction_setting_difference_paths(
    source: ReproductionSettings,
    resolved: ReproductionSettings,
) -> tuple[str, ...]:
    """Return deterministic material leaf paths without waiving traceability."""
    paths: list[str] = []
    for component in _COMPONENTS:
        paths.extend(
            difference[0]
            for difference in _raw_differences(
                getattr(source, component.value),
                getattr(resolved, component.value),
                f"/{component.value}",
            )
        )
    return tuple(paths)


def compare_reproduction_settings(
    source: ReproductionSettings,
    resolved: ReproductionSettings,
    *,
    ablation_refs: Mapping[str, str],
) -> tuple[SettingDifference, ...]:
    """Return deterministic leaf differences, requiring one ablation ref per difference."""
    raw: list[tuple[ReproductionComponent, str, bool, bool, Any, Any]] = []
    for component in _COMPONENTS:
        source_values = getattr(source, component.value)
        resolved_values = getattr(resolved, component.value)
        raw.extend(
            (component, *difference)
            for difference in _raw_differences(
                source_values, resolved_values, f"/{component.value}"
            )
        )
    paths = set(reproduction_setting_difference_paths(source, resolved))
    missing = paths - set(ablation_refs)
    extra = set(ablation_refs) - paths
    if missing:
        raise ValueError(f"source-setting differences require ablation references: {sorted(missing)}")
    if extra:
        raise ValueError(f"ablation references do not identify source-setting differences: {sorted(extra)}")
    return tuple(
        SettingDifference(
            component=component,
            path=path,
            source_present=source_present,
            resolved_present=resolved_present,
            source_value=source_value,
            resolved_value=resolved_value,
            ablation_ref=ablation_refs[path],
        )
        for component, path, source_present, resolved_present, source_value, resolved_value in raw
    )


def classify_reproduction(
    assessment_id: str,
    claim: ClaimRecord,
    source_settings: ReproductionSettings,
    resolved_settings: ReproductionSettings,
    *,
    scale_rung: ScaleRung,
    observed_result_ref: str,
    ablation_refs: Mapping[str, str],
    component_ablations: Iterable[ComponentAblationResult] = (),
) -> ReproductionAssessment:
    """Classify one claim at one scale and apply local ablation-direction evidence."""
    if scale_rung not in claim.scale_scope:
        raise ValueError("claim is not applicable to the requested model scale")
    if not claim.protocol_refs or not claim.evidence_refs:
        raise ValueError("classified claims require reproduction protocol and local evidence references")
    if observed_result_ref not in claim.evidence_refs:
        raise ValueError("observed result must identify evidence linked by the claim")

    differences = compare_reproduction_settings(
        source_settings, resolved_settings, ablation_refs=ablation_refs
    )
    applicable = tuple(
        item
        for item in component_ablations
        if item.claim_id == claim.claim_id and item.scale_rung is scale_rung
    )
    status = claim.status
    if any(item.source_direction is not item.local_direction for item in applicable):
        status = ClaimStatus.NOT_REPRODUCED

    classification = (
        ReproductionClassification.MODIFIED_REPRODUCTION
        if differences
        else ReproductionClassification.REFERENCE_REPRODUCTION
    )
    return ReproductionAssessment(
        assessment_id=assessment_id,
        claim_id=claim.claim_id,
        source_uri=claim.source_uri,
        source_version_hash=claim.source_version_hash,
        scale_rung=scale_rung,
        classification=classification,
        status=status,
        protocol_refs=claim.protocol_refs,
        evidence_refs=claim.evidence_refs,
        observed_result_ref=observed_result_ref,
        differences=differences,
        direction_results=applicable,
    )


def release_evidence_for_scale(
    assessments: Iterable[ReproductionAssessment],
    scale_rung: ScaleRung,
) -> tuple[ReproductionAssessment, ...]:
    """Return only uniquely assessed claims reproduced on the requested scale."""
    eligible: list[ReproductionAssessment] = []
    seen: set[str] = set()
    for assessment in assessments:
        if assessment.scale_rung is not scale_rung:
            continue
        if assessment.claim_id in seen:
            raise ValueError(
                f"multiple reproduction assessments for claim at scale: {assessment.claim_id}"
            )
        seen.add(assessment.claim_id)
        if assessment.status is ClaimStatus.REPRODUCED:
            eligible.append(assessment)
    return tuple(eligible)
