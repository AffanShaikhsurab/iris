from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings, strategies as st

from binary_llm.adapters import BinaryScope, TensorDescriptor, TensorRole, TensorScope
from binary_llm.domain import (
    AblationDirection,
    AmbiguityCandidate,
    AmbiguityEntry,
    AmbiguityStatus,
    BINARY_FAMILY,
    Budget,
    CandidateRecord,
    CheckpointBoundary,
    ClaimRecord,
    ClaimStatus,
    ComponentAblationResult,
    ExperimentFamilyIdentity,
    ExperimentFamilyKind,
    RepresentationKind,
    RepresentationSpec,
    ReproductionComponent,
    ReproductionSettings,
    ScaleRung,
    SourceStatus,
    identify_content,
    release_evidence_for_scale,
)
from binary_llm.domain.ablations import AblationArm, AblationDefinition, BootstrapPlan
from binary_llm.domain.reproduction import classify_reproduction
from binary_llm.math import ScaleParameterization, Stage1ScaleState, diagnose_finite_state
from binary_llm.orchestration import (
    AmbiguitySnapshot,
    MaterialChange,
    PromotionEngine,
    PromotionRequest,
    PromotionStatus,
    ScaleValidation,
    validate_matched_comparison,
)

from binary_llm_property_helpers import NOW, artifact, gate_report


FINITE = st.floats(
    min_value=-100.0,
    max_value=100.0,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)


def _claim(status: ClaimStatus, scale: ScaleRung) -> ClaimRecord:
    return ClaimRecord(
        "claim", "https://example.invalid", "paper-v1", "claim", "formula",
        status, (ScaleRung.SMALL, ScaleRung.INTERMEDIATE), ("protocol",),
        ("observed",), (),
    )


# Feature: binary-llm-conversion-framework, Property 1: Claim traceability and scale-qualified evidence
@given(
    status=st.sampled_from(list(ClaimStatus)),
    assessment_scale=st.sampled_from((ScaleRung.SMALL, ScaleRung.INTERMEDIATE)),
    release_scale=st.sampled_from((ScaleRung.SMALL, ScaleRung.INTERMEDIATE)),
)
@settings(max_examples=100)
def test_property_01_claim_traceability(status, assessment_scale, release_scale) -> None:
    claim = _claim(status, assessment_scale)
    settings_record = ReproductionSettings({}, {}, {}, {}, {})
    assessment = classify_reproduction(
        "assessment", claim, settings_record, settings_record,
        scale_rung=assessment_scale, observed_result_ref="observed", ablation_refs={},
    )
    released = release_evidence_for_scale((assessment,), release_scale)
    assert set(ClaimStatus) == {
        ClaimStatus.REPRODUCED, ClaimStatus.NOT_REPRODUCED,
        ClaimStatus.CONTRADICTED, ClaimStatus.NOT_TESTED,
    }
    assert released == ((assessment,) if (
        status is ClaimStatus.REPRODUCED and assessment_scale is release_scale
    ) else ())


# Feature: binary-llm-conversion-framework, Property 2: Experiment-family isolation
@given(
    hybrid=st.booleans(),
    component_count=st.integers(min_value=0, max_value=3),
)
@settings(max_examples=100)
def test_property_02_experiment_family_isolation(hybrid, component_count) -> None:
    components = tuple(f"family-{index}" for index in range(component_count))
    if hybrid and component_count >= 2:
        identity = ExperimentFamilyIdentity(
            "hybrid-v1", ExperimentFamilyKind.HYBRID, components
        )
        assert identity.family_id != "binary"
        assert len(identity.component_family_ids) >= 2
    elif hybrid:
        with pytest.raises(ValueError):
            ExperimentFamilyIdentity("hybrid-v1", ExperimentFamilyKind.HYBRID, components)
    else:
        with pytest.raises(ValueError):
            ExperimentFamilyIdentity("binary", ExperimentFamilyKind.BINARY, components or ("x",))


# Feature: binary-llm-conversion-framework, Property 3: Seal mismatch detection and immutable lineage
@given(
    payload=st.binary(min_size=1, max_size=64),
    mutation=st.binary(min_size=1, max_size=64),
    sideways=st.booleans(),
)
@settings(max_examples=100)
def test_property_03_seal_mismatch_and_lineage(payload, mutation, sideways) -> None:
    original = identify_content({"payload": payload.hex()}, kind="sealed-asset")
    changed = identify_content({"payload": mutation.hex()}, kind="sealed-asset")
    assert (original.sha256 == changed.sha256) == (payload == mutation)
    checkpoint = artifact("checkpoint")
    direct_parent = "other-rung" if sideways else checkpoint.artifact_id
    packed = artifact("packed", parent=direct_parent)
    representation = RepresentationSpec(
        "binary", RepresentationKind.BINARY, 1, "bits", "float16", None, None,
        False, (), "positive", {},
    )
    if sideways:
        with pytest.raises(ValueError, match="direct source checkpoint"):
            CandidateRecord(
                "candidate", BINARY_FAMILY,
                _minimal_model(), ScaleRung.SMALL, checkpoint, packed,
                representation, (), True, True,
            )
    else:
        candidate = CandidateRecord(
            "candidate", BINARY_FAMILY,
            _minimal_model(), ScaleRung.SMALL, checkpoint, packed,
            representation, (), True, True,
        )
        assert candidate.artifact.parent_artifact_id == checkpoint.artifact_id


def _minimal_model():
    from binary_llm.domain import ModelIdentity

    return ModelIdentity(
        "model", "rev", "tok", "test", 1, "config", ("tok",), ("template",),
        ("weights",), False,
    )


def _passing_promotion_report():
    return gate_report()[0]


# Feature: binary-llm-conversion-framework, Property 4: Monotonic scale promotion
@given(
    rung=st.sampled_from((ScaleRung.SMALL, ScaleRung.INTERMEDIATE)),
    lower_valid=st.booleans(),
    material_change=st.booleans(),
)
@settings(max_examples=100)
def test_property_04_monotonic_scale_promotion(rung, lower_valid, material_change) -> None:
    snapshot = AmbiguitySnapshot("ambiguities", "register", "1", rung, ())
    validations = ()
    if lower_valid:
        validations = (
            ScaleValidation(
                ScaleRung.SMALL, "binary", "operator", "ambiguities",
                "gate", "evidence", True, screening=True,
                material_change_id="change" if material_change else None,
            ),
        )
    request = PromotionRequest(
        "decision", "candidate", "binary", rung, _passing_promotion_report(),
        snapshot, "operator", validations, NOW, reference_regime_reproduced=True,
        material_change=MaterialChange("change", True, False) if material_change else None,
    )
    decision = PromotionEngine().decide(request)
    expected = (rung is ScaleRung.SMALL and not material_change) or lower_valid
    assert (decision.status is PromotionStatus.PROMOTE) is expected


# Feature: binary-llm-conversion-framework, Property 5: Exact role-based binary scope
@given(
    roles=st.lists(st.sampled_from(list(TensorRole)), min_size=1, max_size=20, unique=True),
    width=st.integers(min_value=1, max_value=16),
)
@settings(max_examples=100)
def test_property_05_exact_role_based_binary_scope(roles, width) -> None:
    descriptors = tuple(
        TensorDescriptor(
            f"tensor-{index}", (width,), "float32", width, role,
            TensorScope.BINARY_BODY if role in {
                TensorRole.ATTENTION_QUERY, TensorRole.ATTENTION_KEY,
                TensorRole.ATTENTION_VALUE, TensorRole.ATTENTION_QUERY_KEY_VALUE,
                TensorRole.ATTENTION_OUTPUT, TensorRole.FFN_GATE,
                TensorRole.FFN_UP, TensorRole.FFN_DOWN,
            } else TensorScope.EXCLUDED,
            "binary" if role.name.startswith(("ATTENTION", "FFN")) else "excluded",
        )
        for index, role in enumerate(roles)
    )
    scope = BinaryScope.from_inventory(descriptors)
    assert set(scope.binary_body).isdisjoint(scope.excluded)
    assert set(scope.binary_body) | set(scope.excluded) == set(descriptors)
    assert all(item.semantic_role.name.startswith(("ATTENTION", "FFN")) for item in scope.binary_body)


# Feature: binary-llm-conversion-framework, Property 6: Reproduction classification follows source differences
@given(
    component=st.sampled_from(list(ReproductionComponent)),
    source_value=st.integers(min_value=0, max_value=10),
    resolved_value=st.integers(min_value=0, max_value=10),
    direction_matches=st.booleans(),
)
@settings(max_examples=100)
def test_property_06_reproduction_classification(
    component, source_value, resolved_value, direction_matches
) -> None:
    source_parts = {item.value: {} for item in ReproductionComponent}
    resolved_parts = {item.value: {} for item in ReproductionComponent}
    source_parts[component.value] = {"choice": source_value}
    resolved_parts[component.value] = {"choice": resolved_value}
    source = ReproductionSettings(**source_parts)
    resolved = ReproductionSettings(**resolved_parts)
    path = f"/{component.value}/choice"
    ablations = {} if source_value == resolved_value else {path: "ablation"}
    direction = AblationDirection.IMPROVEMENT
    result = ComponentAblationResult(
        "direction", "claim", component, ScaleRung.SMALL, direction,
        direction if direction_matches else AblationDirection.DEGRADATION,
        "protocol", "observed",
    )
    assessment = classify_reproduction(
        "assessment", _claim(ClaimStatus.REPRODUCED, ScaleRung.SMALL),
        source, resolved, scale_rung=ScaleRung.SMALL,
        observed_result_ref="observed", ablation_refs=ablations,
        component_ablations=(result,),
    )
    assert bool(assessment.differences) is (source_value != resolved_value)
    assert (assessment.status is ClaimStatus.NOT_REPRODUCED) is (not direction_matches)


# Feature: binary-llm-conversion-framework, Property 7: Stage 1 trainability and scale shape
@given(
    rows=st.integers(min_value=1, max_value=8),
    columns=st.integers(min_value=1, max_value=16),
    initial=st.floats(min_value=0.1, max_value=4.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_property_07_stage1_trainability_and_scale_shape(rows, columns, initial) -> None:
    dense = torch.arange(rows * columns, dtype=torch.float32).reshape(rows, columns) + 1
    state = Stage1ScaleState(
        dense, parameterization=ScaleParameterization.POSITIVE_EXP, initial_value=initial
    )
    assert state.input_scale().shape == (columns,)
    assert torch.allclose(state.input_scale(), torch.full((columns,), initial), rtol=1e-5)
    assert torch.equal(state.dense_source, dense)
    assert tuple(name for name, _ in state.named_parameters()) == ("input_scale.raw_scale",)


# Feature: binary-llm-conversion-framework, Property 8: Non-finite state fails closed
@given(
    value=st.one_of(FINITE, st.sampled_from((float("nan"), float("inf"), float("-inf")))),
    location=st.sampled_from(("scale", "weight", "loss", "reconstruction", "gradient")),
)
@settings(max_examples=100)
def test_property_08_nonfinite_state_fails_closed(value, location) -> None:
    values = {
        "scales": torch.tensor([1.0]),
        "transformed_weights": torch.tensor([1.0]),
        "losses": torch.tensor(1.0),
        "reconstruction_errors": torch.tensor(1.0),
        "gradients": torch.tensor([1.0]),
    }
    keys = {
        "scale": "scales", "weight": "transformed_weights", "loss": "losses",
        "reconstruction": "reconstruction_errors", "gradient": "gradients",
    }
    values[keys[location]] = torch.tensor([value]) if location not in {"loss", "reconstruction"} else torch.tensor(value)
    diagnostics = diagnose_finite_state(**values)
    assert diagnostics.is_finite is torch.isfinite(torch.tensor(value)).item()
    if not diagnostics.is_finite:
        assert diagnostics.failing_fields


# Feature: binary-llm-conversion-framework, Property 9: Ambiguity resolution is explicit and scoped
@given(
    resolved=st.booleans(),
    candidates_fail=st.booleans(),
    scale=st.sampled_from(list(ScaleRung)),
)
@settings(max_examples=100)
def test_property_09_ambiguity_resolution_is_explicit_and_scoped(
    resolved, candidates_fail, scale
) -> None:
    candidates = (
        AmbiguityCandidate("a", "choice a", SourceStatus.PAPER_INFERRED),
        AmbiguityCandidate("b", "choice b", SourceStatus.FRAMEWORK_SELECTED),
    )
    status = (
        AmbiguityStatus.RESOLVED if resolved
        else AmbiguityStatus.UNRESOLVED_BLOCKING if candidates_fail
        else AmbiguityStatus.TESTING
    )
    entry = AmbiguityEntry(
        "ambiguity", "1", "statement", SourceStatus.PAPER_INFERRED, ("operator",),
        candidates, "matched", "decision", "paired-95", ("finite",), (scale,), status,
        "a" if resolved else None, ("evidence",) if resolved else (),
        ("b",) if resolved else (),
    )
    assert entry.scale_scope == (scale,)
    assert (entry.selected_candidate is not None) is resolved
    assert (entry.status is AmbiguityStatus.UNRESOLVED_BLOCKING) is (
        candidates_fail and not resolved
    )


# Feature: binary-llm-conversion-framework, Property 10: Matched comparisons change only declared factors
@given(
    changed_count=st.integers(min_value=1, max_value=3),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=100)
def test_property_10_matched_comparisons_change_only_declared_factors(
    changed_count, seed
) -> None:
    paths = tuple(f"/factor_{index}" for index in range(changed_count))
    budget = Budget(100, 10, 60, 0.0, CheckpointBoundary.PROGRESSIVE_PHASE)
    common = dict(
        model_revision="model", data_split_id="split", evaluator_revision="eval",
        seed_set=(seed,), compute_budget=budget,
    )
    control = AblationArm("control", settings={path[1:]: 0 for path in paths}, **common)
    candidate = AblationArm("candidate", settings={path[1:]: 1 for path in paths}, **common)
    definition = AblationDefinition(
        "ablation", "factor", "candidate", "control", (), paths,
        changed_count > 1, ("score",), BootstrapPlan(seed),
    )
    comparison = validate_matched_comparison(definition, candidate, control)
    assert comparison.changed_fields == paths
    assert (comparison.classification.value == "interaction_experiment") is (changed_count > 1)
