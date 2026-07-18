from __future__ import annotations

from dataclasses import replace

import pytest

from binary_llm.domain import (
    BINARYLLM_AMBIGUOUS_PATHS,
    BINARYLLM_FIDELITY_PATHS,
    BINARYLLM_PAPER_LEDGER,
    AblationClassification,
    AmbiguityBlocked,
    BinaryLLMFidelityConfig,
    FidelityFieldResolution,
    ManifestError,
    ReproductionClassification,
    SourceStatus,
    build_binaryllm_reference_config,
    build_matched_fidelity_arms,
)


def _resolutions() -> dict[str, FidelityFieldResolution]:
    values = {
        "/stage1/input_scale/parameterization": "positive_exp",
        "/stage1/input_scale/positivity": "strict_positive",
        "/stage1/input_scale/zero_avoidance": "by_parameterization",
        "/stage1/activation_transform_mode": "explicit_activation_transform",
        "/stage1/backward": "straight_through_unspecified",
        "/stage1/optimizer/kind": "AdamW",
        "/stage1/optimizer/learning_rate": 1e-4,
        "/stage2/phases/index_origin": "0..19",
        "/stage2/schedule/t_start": 0.0,
        "/stage2/schedule/t_end": 83.824,
        "/stage2/scales/analytical/gradient": "detached",
        "/numeric/compute_precision": "fp32_core",
        "/numeric/scale_storage_precision": "fp32",
        "/optimizer/betas": (0.9, 0.999),
        "/optimizer/epsilon": 1e-8,
        "/optimizer/warmup": "none",
        "/optimizer/gradient_clipping": "none",
        "/reproducibility/seed_policy": "fixed_seed_set",
        "/reproducibility/data_order": "frozen_hash_order",
        "/export/zero_sign_policy": "positive",
        "/export/rounding": "round_to_nearest_even",
        "/export/bit_order": "lsb_first",
        "/export/row_alignment": 1,
        "/export/scale_endianness": "little",
    }
    return {
        path: FidelityFieldResolution(
            path,
            values[path],
            SourceStatus.PAPER_INFERRED,
            f"ambiguity-register:{ambiguity_id}",
            ambiguity_id,
        )
        for path, ambiguity_id in BINARYLLM_AMBIGUOUS_PATHS.items()
    }


def test_reference_config_has_complete_exact_provenance_and_deterministic_hash() -> None:
    first = build_binaryllm_reference_config(_resolutions())
    second = build_binaryllm_reference_config(dict(reversed(tuple(_resolutions().items()))))

    assert set(first.values) == BINARYLLM_FIDELITY_PATHS
    assert set(first.resolution_ledger) == BINARYLLM_FIDELITY_PATHS
    assert first.config_hash == second.config_hash
    assert first.reproduction_classification is ReproductionClassification.REFERENCE_REPRODUCTION
    assert all(item.source_status is not SourceStatus.FRAMEWORK_SELECTED for item in first.fields)


def test_mutating_a_resolved_value_changes_hash_and_classification() -> None:
    reference = build_binaryllm_reference_config(_resolutions())
    fields = list(reference.fields)
    index = next(i for i, item in enumerate(fields) if item.path == "/optimizer/betas")
    fields[index] = replace(
        fields[index],
        value=(0.8, 0.95),
        source_status=SourceStatus.FRAMEWORK_SELECTED,
        source_ref="framework:optimizer-control",
    )
    modified = BinaryLLMFidelityConfig(tuple(fields))

    assert modified.config_hash != reference.config_hash
    assert modified.reproduction_classification is ReproductionClassification.MODIFIED_REPRODUCTION


def test_missing_extra_and_unresolved_paths_fail_closed() -> None:
    resolutions = _resolutions()
    resolutions.pop("/optimizer/betas")
    with pytest.raises(AmbiguityBlocked) as missing:
        build_binaryllm_reference_config(resolutions)
    assert missing.value.code == "ambiguity.fidelity_resolution"

    reference = build_binaryllm_reference_config(_resolutions())
    with pytest.raises(ManifestError) as extra:
        BinaryLLMFidelityConfig(
            (*reference.fields, FidelityFieldResolution("/unknown", 1, SourceStatus.FRAMEWORK_SELECTED, "test"))
        )
    assert extra.value.code == "manifest.fidelity_inventory"


def test_matched_arms_declare_exact_changed_paths_and_ablation_ids() -> None:
    reference = build_binaryllm_reference_config(_resolutions())
    arms = {arm.arm_id: arm for arm in build_matched_fidelity_arms(reference)}

    assert arms["stage2_body_only"].changed_factor_paths == ("/stage2/trainability/scope",)
    assert (
        reference.values["/stage2/trainability/folded_input_scale"]
        == "identity_frozen_transition_state"
    )
    reference_values = dict(reference.values)
    body_values = dict(arms["stage2_body_only"].config.values)
    assert {
        path for path in reference_values if reference_values[path] != body_values[path]
    } == {"/stage2/trainability/scope"}
    assert (
        body_values["/stage2/trainability/scope"] == "binary_body_only"
        and arms["stage2_body_only"].ablation_id
        == "binaryllm-fidelity-stage2_body_only"
    )
    assert arms["analytical_scale_only"].changed_factor_paths == ("/stage2/scales/composition",)
    assert arms["stage1_weight_only"].changed_factor_paths == (
        "/stage1/activation_transform",
        "/stage1/activation_transform_mode",
    )
    assert arms["stage1_weight_only"].classification is AblationClassification.INTERACTION_EXPERIMENT
    assert all(arm.ablation_id.startswith("binaryllm-fidelity-") for arm in arms.values())
    assert all(
        arm.expected_source_difference is ReproductionClassification.MODIFIED_REPRODUCTION
        for arm in arms.values()
    )


def test_paper_ledger_never_contains_framework_claims() -> None:
    assert all(
        fact.source_status is not SourceStatus.FRAMEWORK_SELECTED
        for fact in BINARYLLM_PAPER_LEDGER.facts
    )
