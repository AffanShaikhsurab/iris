from __future__ import annotations

from dataclasses import replace

import pytest

from binary_llm.domain import (
    REQUIRED_PRIMARY_ABLATION_FACTORS,
    AblationArm,
    AblationClassification,
    AblationDefinition,
    BootstrapPlan,
    Budget,
    CheckpointBoundary,
    EvaluationError,
    ScoreObservation,
)
from binary_llm.orchestration import (
    MatchedComparison,
    exact_enumerated_paired_distribution,
    paired_bootstrap_statistics,
    validate_matched_comparison,
    validate_primary_ablation_suite,
)


def budget(steps: int = 10) -> Budget:
    return Budget(1_000, steps, 60, 2.0, CheckpointBoundary.PROGRESSIVE_PHASE)


def arm(experiment_id: str, settings: dict[str, object], **changes: object) -> AblationArm:
    values = {
        "experiment_id": experiment_id,
        "model_revision": "model-rev",
        "data_split_id": "split-v1",
        "evaluator_revision": "eval-v1",
        "seed_set": (7, 11),
        "compute_budget": budget(),
        "settings": settings,
    }
    values.update(changes)
    return AblationArm(**values)


def definition(
    *,
    factor: str = "progressive_schedule",
    changed_fields: tuple[str, ...] = ("/schedule",),
    interaction: bool = False,
) -> AblationDefinition:
    return AblationDefinition(
        "ablation-1",
        factor,
        "candidate",
        "control",
        ("/optimizer",),
        changed_fields,
        interaction,
        ("held_out_capability",),
        BootstrapPlan(seed=123),
    )


def test_one_factor_comparison_preserves_required_matching_fields() -> None:
    candidate = arm("candidate", {"schedule": "exponential", "optimizer": "adamw"})
    control = arm("control", {"schedule": "linear", "optimizer": "adamw"})

    result = validate_matched_comparison(definition(), candidate, control)

    assert result.classification is AblationClassification.COMPONENT_ABLATION
    assert result.changed_fields == ("/schedule",)
    assert result.primary_metrics == ("held_out_capability",)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("model_revision", "other-model"),
        ("data_split_id", "other-split"),
        ("evaluator_revision", "other-evaluator"),
        ("seed_set", (99,)),
        ("compute_budget", budget(11)),
    ),
)
def test_matched_comparison_rejects_required_invariant_mismatch(
    field: str, value: object
) -> None:
    candidate = arm("candidate", {"schedule": "exponential", "optimizer": "adamw"})
    control = arm(
        "control",
        {"schedule": "linear", "optimizer": "adamw"},
        **{field: value},
    )

    with pytest.raises(ValueError, match="required invariants"):
        validate_matched_comparison(definition(), candidate, control)


def test_multiple_changes_require_interaction_classification() -> None:
    candidate = arm(
        "candidate", {"schedule": "exponential", "precision": "fp32", "optimizer": "adamw"}
    )
    control = arm(
        "control", {"schedule": "linear", "precision": "bf16", "optimizer": "adamw"}
    )
    interaction = definition(
        changed_fields=("/schedule", "/precision"), interaction=True
    )

    result = validate_matched_comparison(interaction, candidate, control)

    assert result.classification is AblationClassification.INTERACTION_EXPERIMENT
    with pytest.raises(ValueError, match="interaction experiment"):
        validate_matched_comparison(replace(interaction, interaction=False), candidate, control)


def test_comparison_rejects_undeclared_or_nonchanging_fields() -> None:
    candidate = arm(
        "candidate", {"schedule": "exponential", "optimizer": "adamw", "precision": "fp32"}
    )
    control = arm(
        "control", {"schedule": "linear", "optimizer": "adamw", "precision": "bf16"}
    )

    with pytest.raises(ValueError, match="undeclared changed fields"):
        validate_matched_comparison(definition(), candidate, control)
    with pytest.raises(ValueError, match="does not differ"):
        validate_matched_comparison(
            definition(),
            arm("candidate", {"schedule": "linear", "optimizer": "adamw"}),
            arm("control", {"schedule": "linear", "optimizer": "adamw"}),
        )


def test_primary_ablation_suite_requires_every_preregistered_factor() -> None:
    comparisons = tuple(
        MatchedComparison(
            f"ablation-{factor}",
            factor,
            f"candidate-{factor}",
            f"control-{factor}",
            AblationClassification.COMPONENT_ABLATION,
            (f"/{factor}",),
            ("quality",),
        )
        for factor in sorted(REQUIRED_PRIMARY_ABLATION_FACTORS)
    )

    assert validate_primary_ablation_suite(comparisons) == comparisons
    with pytest.raises(ValueError, match="binary_aware_initialization"):
        validate_primary_ablation_suite(
            item for item in comparisons if item.factor != "binary_aware_initialization"
        )


def scores(rows: tuple[tuple[str, str, float], ...]) -> tuple[ScoreObservation, ...]:
    return tuple(ScoreObservation(case, family, "quality", score) for case, family, score in rows)


def test_paired_bootstrap_uses_equal_weight_semantic_family_deltas() -> None:
    candidate = scores((("a1", "family-a", 2.0), ("a2", "family-a", 6.0), ("b1", "family-b", 0.0)))
    control = scores((("a1", "family-a", 1.0), ("a2", "family-a", 3.0), ("b1", "family-b", 2.0)))

    result = paired_bootstrap_statistics(candidate, control, "quality", BootstrapPlan(seed=41))

    assert tuple(item.delta for item in result.case_deltas) == (1.0, 3.0, -2.0)
    assert tuple((item.semantic_family_id, item.pair_count, item.delta) for item in result.family_deltas) == (
        ("family-a", 2, 2.0),
        ("family-b", 1, -2.0),
    )
    assert result.paired_delta == 0.0
    assert result.pair_count == 3
    assert result.semantic_family_count == 2


def test_bootstrap_is_exactly_10000_resamples_and_seed_reproducible() -> None:
    candidate = scores((("a", "family-a", 1.0), ("b", "family-b", 3.0)))
    control = scores((("a", "family-a", 0.0), ("b", "family-b", 0.0)))

    first = paired_bootstrap_statistics(candidate, control, "quality", BootstrapPlan(seed=17))
    repeated = paired_bootstrap_statistics(candidate, control, "quality", BootstrapPlan(seed=17))
    changed_seed = paired_bootstrap_statistics(candidate, control, "quality", BootstrapPlan(seed=18))

    assert first.bootstrap_resamples == len(first.bootstrap_distribution) == 10_000
    assert first.bootstrap_distribution == repeated.bootstrap_distribution
    assert first.bootstrap_distribution != changed_seed.bootstrap_distribution
    assert first.confidence_lower <= first.confidence_upper


@pytest.mark.parametrize("resamples", (0, 9_999, 10_001))
def test_bootstrap_plan_rejects_any_resample_count_other_than_10000(resamples: int) -> None:
    with pytest.raises(ValueError, match="exactly 10,000"):
        BootstrapPlan(seed=1, resamples=resamples)


def test_missing_pairs_and_family_mismatches_fail_closed() -> None:
    candidate = scores((("paired", "family-a", 1.0), ("candidate-only", "family-b", 2.0)))
    control = scores((("paired", "family-a", 0.0),))

    with pytest.raises(EvaluationError) as missing:
        paired_bootstrap_statistics(candidate, control, "quality", BootstrapPlan(seed=1))
    assert missing.value.code == "evaluation.missing_pairs"
    assert missing.value.affected_ids == {"case_ids": ("candidate-only",)}

    with pytest.raises(EvaluationError) as mismatch:
        paired_bootstrap_statistics(
            scores((("paired", "family-a", 1.0),)),
            scores((("paired", "family-b", 0.0),)),
            "quality",
            BootstrapPlan(seed=1),
        )
    assert mismatch.value.code == "evaluation.family_mismatch"


def test_single_family_is_degenerate_and_exact_enumeration_is_available_for_tiny_fixtures() -> None:
    one_candidate = scores((("a", "only-family", 1.5),))
    one_control = scores((("a", "only-family", 1.0),))
    degenerate = paired_bootstrap_statistics(
        one_candidate, one_control, "quality", BootstrapPlan(seed=7)
    )

    assert degenerate.confidence_lower == degenerate.confidence_upper == 0.5
    assert set(degenerate.bootstrap_distribution) == {0.5}

    candidate = scores((("a", "family-a", 1.0), ("b", "family-b", 3.0)))
    control = scores((("a", "family-a", 0.0), ("b", "family-b", 0.0)))
    exact = exact_enumerated_paired_distribution(candidate, control, "quality")

    assert exact == (1.0, 2.0, 2.0, 3.0)
    with pytest.raises(ValueError, match="would produce"):
        exact_enumerated_paired_distribution(candidate, control, "quality", maximum_outcomes=3)
