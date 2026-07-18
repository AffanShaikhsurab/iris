from __future__ import annotations

import json
from dataclasses import replace

import pytest

from binary_llm.domain import (
    AblationDirection,
    ClaimRecord,
    ClaimStatus,
    ComponentAblationResult,
    ReproductionClassification,
    ReproductionComponent,
    ReproductionSettings,
    ScaleRung,
    classify_reproduction,
    compare_reproduction_settings,
    release_evidence_for_scale,
)


def settings(value: str = "paper") -> ReproductionSettings:
    return ReproductionSettings(
        operator={"choice": value},
        data={"choice": value},
        schedule={"choice": value},
        optimization={"choice": value},
        evaluation={"choice": value},
    )


def claim(
    status: ClaimStatus = ClaimStatus.REPRODUCED,
    scales: tuple[ScaleRung, ...] = (ScaleRung.SMALL,),
) -> ClaimRecord:
    return ClaimRecord(
        claim_id="claim-1",
        source_uri="https://example.invalid/paper",
        source_version_hash="source-version-sha256",
        statement="The component improves the reference result.",
        category="component_ablation",
        status=status,
        scale_scope=scales,
        protocol_refs=("protocol-1",),
        evidence_refs=("evidence-1", "observed-result-1"),
        differences=(),
    )


def ablation(
    scale: ScaleRung,
    local_direction: AblationDirection,
) -> ComponentAblationResult:
    return ComponentAblationResult(
        ablation_ref=f"ablation-{scale.value}",
        claim_id="claim-1",
        component=ReproductionComponent.OPERATOR,
        scale_rung=scale,
        source_direction=AblationDirection.IMPROVEMENT,
        local_direction=local_direction,
        protocol_ref=f"ablation-protocol-{scale.value}",
        evidence_ref=f"ablation-evidence-{scale.value}",
    )


def test_identical_settings_are_a_traceable_reference_reproduction():
    assessment = classify_reproduction(
        "assessment-1",
        claim(),
        settings(),
        settings(),
        scale_rung=ScaleRung.SMALL,
        observed_result_ref="observed-result-1",
        ablation_refs={},
    )

    assert assessment.classification is ReproductionClassification.REFERENCE_REPRODUCTION
    assert assessment.status is ClaimStatus.REPRODUCED
    assert assessment.differences == ()
    assert assessment.source_version_hash == "source-version-sha256"
    assert assessment.protocol_refs == ("protocol-1",)
    assert assessment.observed_result_ref in assessment.evidence_refs
    assert json.loads(json.dumps(assessment.to_dict()))["scale_rung"] == "small"


def test_every_material_setting_difference_requires_a_corresponding_ablation_reference():
    paths = {
        "/operator/choice",
        "/data/choice",
        "/schedule/choice",
        "/optimization/choice",
        "/evaluation/choice",
    }

    with pytest.raises(ValueError, match="require ablation references"):
        compare_reproduction_settings(settings(), settings("resolved"), ablation_refs={})

    references = {path: f"ablation-{index}" for index, path in enumerate(sorted(paths))}
    assessment = classify_reproduction(
        "assessment-modified",
        claim(),
        settings(),
        settings("resolved"),
        scale_rung=ScaleRung.SMALL,
        observed_result_ref="observed-result-1",
        ablation_refs=references,
    )

    assert assessment.classification is ReproductionClassification.MODIFIED_REPRODUCTION
    assert {item.path for item in assessment.differences} == paths
    assert {item.component for item in assessment.differences} == set(ReproductionComponent)
    assert {item.ablation_ref for item in assessment.differences} == set(references.values())


def test_setting_comparison_records_added_removed_and_type_changed_values():
    source = replace(
        settings(),
        operator={"nested": {"removed": 1, "typed": True}},
    )
    resolved = replace(
        settings(),
        operator={"nested": {"added": 2, "typed": 1}},
    )
    references = {
        "/operator/nested/added": "ablation-added",
        "/operator/nested/removed": "ablation-removed",
        "/operator/nested/typed": "ablation-typed",
    }

    differences = compare_reproduction_settings(source, resolved, ablation_refs=references)

    by_path = {item.path: item for item in differences}
    assert by_path["/operator/nested/added"].source_present is False
    assert by_path["/operator/nested/removed"].resolved_present is False
    assert by_path["/operator/nested/typed"].source_value is True
    assert by_path["/operator/nested/typed"].resolved_value == 1


def test_unstated_source_settings_are_recorded_as_material_additions():
    source = ReproductionSettings(
        operator={}, data={}, schedule={}, optimization={}, evaluation={}
    )
    resolved = ReproductionSettings(
        operator={}, data={}, schedule={},
        optimization={"warmup_steps": 10}, evaluation={},
    )

    differences = compare_reproduction_settings(
        source,
        resolved,
        ablation_refs={"/optimization/warmup_steps": "ablation-warmup"},
    )

    assert len(differences) == 1
    assert differences[0].source_present is False
    assert differences[0].resolved_value == 10


def test_reproduction_settings_reject_non_mapping_components():
    with pytest.raises(TypeError, match="operator reproduction settings must be a mapping"):
        ReproductionSettings(  # type: ignore[arg-type]
            operator=("not", "a", "mapping"),
            data={}, schedule={}, optimization={}, evaluation={},
        )


def test_direction_mismatch_marks_only_the_applicable_scale_not_reproduced():
    scoped_claim = claim(scales=(ScaleRung.SMALL, ScaleRung.INTERMEDIATE))
    observations = (
        ablation(ScaleRung.SMALL, AblationDirection.DEGRADATION),
        ablation(ScaleRung.INTERMEDIATE, AblationDirection.IMPROVEMENT),
    )

    small = classify_reproduction(
        "assessment-small",
        scoped_claim,
        settings(),
        settings(),
        scale_rung=ScaleRung.SMALL,
        observed_result_ref="observed-result-1",
        ablation_refs={},
        component_ablations=observations,
    )
    intermediate = classify_reproduction(
        "assessment-intermediate",
        scoped_claim,
        settings(),
        settings(),
        scale_rung=ScaleRung.INTERMEDIATE,
        observed_result_ref="observed-result-1",
        ablation_refs={},
        component_ablations=observations,
    )

    assert small.status is ClaimStatus.NOT_REPRODUCED
    assert small.direction_results == (observations[0],)
    assert intermediate.status is ClaimStatus.REPRODUCED
    assert intermediate.direction_results == (observations[1],)


def test_release_evidence_excludes_nonreproduced_and_other_scale_claims():
    scoped_claim = claim(scales=(ScaleRung.SMALL, ScaleRung.INTERMEDIATE))
    small_failed = classify_reproduction(
        "assessment-small",
        scoped_claim,
        settings(),
        settings(),
        scale_rung=ScaleRung.SMALL,
        observed_result_ref="observed-result-1",
        ablation_refs={},
        component_ablations=(ablation(ScaleRung.SMALL, AblationDirection.DEGRADATION),),
    )
    intermediate_passed = classify_reproduction(
        "assessment-intermediate",
        scoped_claim,
        settings(),
        settings(),
        scale_rung=ScaleRung.INTERMEDIATE,
        observed_result_ref="observed-result-1",
        ablation_refs={},
        component_ablations=(ablation(ScaleRung.INTERMEDIATE, AblationDirection.IMPROVEMENT),),
    )

    assert release_evidence_for_scale(
        (small_failed, intermediate_passed), ScaleRung.SMALL
    ) == ()
    assert release_evidence_for_scale(
        (small_failed, intermediate_passed), ScaleRung.INTERMEDIATE
    ) == (intermediate_passed,)


def test_classification_fails_closed_without_claim_traceability():
    with pytest.raises(ValueError, match="protocol and local evidence"):
        classify_reproduction(
            "assessment-missing-trace",
            replace(claim(), protocol_refs=()),
            settings(),
            settings(),
            scale_rung=ScaleRung.SMALL,
            observed_result_ref="observed-result-1",
            ablation_refs={},
        )

    with pytest.raises(ValueError, match="observed result"):
        classify_reproduction(
            "assessment-unlinked-result",
            claim(),
            settings(),
            settings(),
            scale_rung=ScaleRung.SMALL,
            observed_result_ref="unlinked-result",
            ablation_refs={},
        )
