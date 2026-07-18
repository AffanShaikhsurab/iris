from __future__ import annotations

import math

import pytest
import torch
from hypothesis import given, settings, strategies as st

from binary_llm.adapters import ActiveRepresentation
from binary_llm.domain import (
    GateCategory,
    ScaleRung,
    canonical_json_bytes,
    identify_content,
    sha256_bytes,
)
from binary_llm.math import (
    AnalyticalScaleGradient,
    DualScaleState,
    OperatorPrecision,
    PhaseIndexConvention,
    ProgressionScheduleConfig,
    ProgressionScheduleKind,
    ProgressiveOperatorConfig,
    progressive,
    progressive_derivative,
    progression_parameters,
)
from binary_llm.orchestration import (
    AmbiguitySnapshot,
    AttemptExecutionContext,
    AttemptRef,
    GateReport,
    GateResult,
    GateResultStatus,
    PromotionEngine,
    PromotionRequest,
    PromotionStatus,
)
from binary_llm.orchestration.corpus import (
    CorpusDenyList,
    CorpusService,
    ProvenanceRecord,
    SplitPolicy,
)
from binary_llm.orchestration.recovery import (
    NoProgressPolicy,
    RecoveryTrainerConfig,
    RecoveryProgressPoint,
    SupervisionCoverage,
    SupervisionMode,
    TeacherRoutingConfig,
    evaluate_no_progress,
)
from binary_llm.orchestration.stage1 import Stage1OptimizerConfig, Stage1OptimizerKind

from binary_llm_property_helpers import NOW


FINITE = st.floats(
    min_value=-5.0,
    max_value=5.0,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)


# Feature: binary-llm-conversion-framework, Property 11: Progressive function and derivative match the reference mathematics
@given(
    values=st.lists(FINITE, min_size=1, max_size=16),
    t=st.floats(min_value=0.0, max_value=8.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_property_11_progressive_function_and_derivative(values, t) -> None:
    tensor = torch.tensor(values, dtype=torch.float64)
    config = ProgressiveOperatorConfig(
        OperatorPrecision.FLOAT32,
        AnalyticalScaleGradient.DETACHED,
        small_t_threshold=1e-6,
    )
    actual = progressive(tensor, t, config=config).double()
    derivative = progressive_derivative(tensor, t, config=config).double()
    if t <= config.small_t_threshold:
        expected = tensor
        expected_derivative = torch.ones_like(tensor)
    else:
        expected = torch.tanh(t * tensor) / math.tanh(t)
        expected_derivative = t * (1 - torch.tanh(t * tensor) ** 2) / math.tanh(t)
    assert torch.allclose(actual, expected, rtol=2e-5, atol=2e-6)
    assert torch.allclose(derivative, expected_derivative, rtol=2e-5, atol=2e-6)


def _record(record_id: str, tokens: int = 1, *, synthetic: bool = False) -> ProvenanceRecord:
    normalized = {"messages": [{"role": "user", "content": record_id}]}
    return ProvenanceRecord(
        record_id, sha256_bytes(canonical_json_bytes(normalized)), f"source:{record_id}",
        "terms", "training", ("normalized",), f"semantic:{record_id}",
        f"split:{record_id}", f"dedup:{record_id}", f"fuzzy:{record_id}",
        "causal_lm", "broad_text", tokens, synthetic, True, normalized,
    )


# Feature: binary-llm-conversion-framework, Property 12: Progressive partition coverage and schedule determinism
@given(
    count=st.integers(min_value=1, max_value=40),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    one_based=st.booleans(),
)
@settings(max_examples=100)
def test_property_12_partition_coverage_and_schedule_determinism(count, seed, one_based) -> None:
    records = tuple(_record(f"record-{index}") for index in range(count))
    policy = SplitPolicy(
        "policy", seed, target_allocation=(("broad_text", 1.0),),
        allocation_tolerance=0.0,
    )
    manifest = CorpusService().freeze(records, policy)
    partitions = CorpusService().progressive_partitions(manifest, seed=seed)
    flattened = tuple(item for partition in partitions for item in partition.record_ids)
    assert len(partitions) == 20
    assert len(flattened) == len(set(flattened)) == count
    assert set(flattened) == {item.record_id for item in manifest.records}
    config = ProgressionScheduleConfig(
        ProgressionScheduleKind.PAPER_EXPONENTIAL,
        PhaseIndexConvention.ONE_BASED if one_based else PhaseIndexConvention.ZERO_BASED,
    )
    assert progression_parameters(config, phase_count=20) == progression_parameters(
        config, phase_count=20
    )


# Feature: binary-llm-conversion-framework, Property 13: Dual-scale algebra and update semantics
@given(
    rows=st.integers(min_value=1, max_value=8),
    columns=st.integers(min_value=1, max_value=16),
    update=FINITE,
)
@settings(max_examples=100)
def test_property_13_dual_scale_algebra_and_update_semantics(rows, columns, update) -> None:
    latent = torch.arange(1, rows * columns + 1, dtype=torch.float32).reshape(rows, columns)
    state = DualScaleState(rows, analytical_gradient=AnalyticalScaleGradient.DETACHED)
    assert torch.equal(state.learned(), torch.ones(rows))
    before = state.analytical(latent)
    changed = latent.clone()
    changed[0, 0] = update
    after = state.analytical(changed)
    assert torch.allclose(after, changed.abs().mean(dim=1))
    assert torch.allclose(state.merged(changed), after * state.learned())
    assert state.merged(changed).shape == (rows,)
    if update != latent[0, 0].item():
        assert not torch.equal(before, after)


# Feature: binary-llm-conversion-framework, Property 14: Representation identity captures inference exceptions
@given(
    field=st.sampled_from((
        "offset", "exceptions", "scale", "embedding", "vocabulary", "alignment",
    )),
    value=st.integers(min_value=2, max_value=10_000),
)
@settings(max_examples=100)
def test_property_14_representation_identity_captures_exceptions(field, value) -> None:
    base = {"offset": 0, "exceptions": (), "scale": "float16", "embedding": "bf16",
            "vocabulary": 100, "alignment": 1}
    changed = dict(base)
    changed[field] = (f"exception-{value}",) if field == "exceptions" else (
        f"dtype-{value}" if field in {"scale", "embedding"} else value
    )
    first = identify_content({"candidate": "x"}, kind="representation", representation_fields=base)
    second = identify_content({"candidate": "x"}, kind="representation", representation_fields=changed)
    assert first.value != second.value


def _simple_report(passed: bool, report_id: str) -> GateReport:
    status = GateResultStatus.PASS if passed else GateResultStatus.FAIL
    return GateReport(
        report_id, "gates",
        (GateResult("gate", status, int(passed), 1, ("evidence",), NOW,
                    GateCategory.CONTINUATION.value),),
        NOW, (),
    )


# Feature: binary-llm-conversion-framework, Property 15: Sign substitution cannot bypass gates
@given(sign_passes=st.booleans(), progressive_passes=st.booleans())
@settings(max_examples=100)
def test_property_15_sign_substitution_cannot_bypass_gates(
    sign_passes, progressive_passes
) -> None:
    request = PromotionRequest(
        "decision", "candidate", "binary", ScaleRung.SMALL,
        _simple_report(sign_passes, "sign"),
        AmbiguitySnapshot("ambiguities", "register", "1", ScaleRung.SMALL, ()),
        "operator", (), NOW, reference_regime_reproduced=True, export_based=True,
        progressive_report=_simple_report(progressive_passes, "progressive"),
    )
    decision = PromotionEngine().decide(request)
    assert (decision.status is PromotionStatus.PROMOTE) is sign_passes
    if progressive_passes and not sign_passes:
        assert any("sign-substituted" in reason for reason in decision.reasons)


# Feature: binary-llm-conversion-framework, Property 16: Recovery keeps the target operator and routes teachers correctly
@given(
    shared_tokenizer=st.booleans(),
    behavioral=st.booleans(),
    target=st.sampled_from((ActiveRepresentation.PROGRESSIVE, ActiveRepresentation.SIGN)),
    supervised=st.integers(min_value=0, max_value=100),
    eligible=st.integers(min_value=100, max_value=200),
)
@settings(max_examples=100)
def test_property_16_recovery_operator_and_teacher_routing(
    shared_tokenizer, behavioral, target, supervised, eligible
) -> None:
    mode = (
        SupervisionMode.SHARED_TOKENIZER_SEQUENCE
        if shared_tokenizer else SupervisionMode.DIFFERENT_TOKENIZER_TEXT_EXECUTION
    )
    config = TeacherRoutingConfig(
        student_tokenizer_id="student",
        sealed_bf16_teacher_identity_id="bf16-teacher",
        behavioral_mode=mode if behavioral else SupervisionMode.NONE,
        broad_mode=SupervisionMode.NONE,
        sealed_bf16_inventory_id="inventory" if behavioral else None,
    )
    assert (config.behavioral_mode is not SupervisionMode.NONE) is behavioral
    assert "iris_tools" in config.behavioral_slices
    assert mode is not SupervisionMode.SHARED_TOKENIZER_LOGITS or shared_tokenizer
    trainer = RecoveryTrainerConfig(
        Stage1OptimizerConfig(Stage1OptimizerKind.ADAMW, 1e-3, 0.0),
        7,
        1,
        target,
        NoProgressPolicy("calibration", "frozen"),
        progression_parameter=0.5 if target is ActiveRepresentation.PROGRESSIVE else None,
    )
    coverage = SupervisionCoverage(supervised, eligible, supervised / eligible)
    assert trainer.target_representation is target
    assert trainer.target_representation is not ActiveRepresentation.DENSE_REFERENCE
    assert coverage.fraction == supervised / eligible


# Feature: binary-llm-conversion-framework, Property 17: Recovery allocation and progress policy
@given(
    calibration_gain=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    capability_gain=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    threshold=st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_property_17_recovery_allocation_and_progress_policy(
    calibration_gain, capability_gain, threshold
) -> None:
    initial = RecoveryProgressPoint(0, 0.0, 0.0)
    current = RecoveryProgressPoint(1, calibration_gain, capability_gain)
    policy = NoProgressPolicy("calibration", "frozen", threshold)
    decision = evaluate_no_progress(initial, current, policy)
    assert decision.triggered is (
        calibration_gain > threshold and capability_gain <= threshold
    )


# Feature: binary-llm-conversion-framework, Property 18: Provenance-complete, leakage-free corpora
@given(
    count=st.integers(min_value=2, max_value=20),
    denied_index=st.integers(min_value=0, max_value=19),
)
@settings(max_examples=100)
def test_property_18_provenance_complete_leakage_free_corpora(count, denied_index) -> None:
    records = tuple(_record(f"record-{index}") for index in range(count))
    denied = records[denied_index % count]
    denylist = CorpusDenyList(
        "frozen", content_hashes=(denied.content_hash,),
    )
    policy = SplitPolicy(
        "policy", 7, target_allocation=(("broad_text", 1.0),),
        frozen_evaluation=denylist,
    )
    manifest = CorpusService().freeze(records, policy)
    assert denied.record_id not in {item.record_id for item in manifest.records}
    assert manifest.provenance_complete
    assert all(item.source and item.license_or_terms and item.permitted_use for item in manifest.records)


# Feature: binary-llm-conversion-framework, Property 19: Release slices retain trustworthy support
@given(
    synthetic=st.booleans(),
    verified=st.booleans(),
    tokens=st.integers(min_value=1, max_value=100),
)
@settings(max_examples=100)
def test_property_19_release_slices_retain_trustworthy_support(synthetic, verified, tokens) -> None:
    record = _record("release", tokens, synthetic=synthetic)
    record = ProvenanceRecord(
        record.record_id, record.content_hash, record.source, record.license_or_terms,
        record.permitted_use, record.transformation_history, record.semantic_family_id,
        record.split_family_id, record.deduplication_key, record.fuzzy_cluster_id,
        record.target_type, record.capability_slice, record.token_count, synthetic,
        verified, record.normalized_record,
    )
    policy = SplitPolicy(
        "policy", 1, target_allocation=(("broad_text", 1.0),),
        release_evidence_slices=("broad_text",),
    )
    if not synthetic and verified:
        manifest = CorpusService().freeze((record,), policy)
        assert manifest.release_slice_support == (("broad_text", 1),)
    else:
        with pytest.raises(ValueError, match="lack permitted non-synthetic"):
            CorpusService().freeze((record,), policy)


# Feature: binary-llm-conversion-framework, Property 20: Run evidence is uniquely attributable and reproducible
@given(
    run_id=st.text(alphabet="abc123", min_size=1, max_size=12),
    attempt_id=st.text(alphabet="xyz789", min_size=1, max_size=12),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    nondeterministic=st.booleans(),
)
@settings(max_examples=100)
def test_property_20_run_evidence_attributable_reproducible(
    run_id, attempt_id, seed, nondeterministic
) -> None:
    reference = AttemptRef(run_id, attempt_id, "experiment")
    context = AttemptExecutionContext(
        "commit", True, {"cpu": "test"}, {"python": "3.12"}, seed, "order",
        ("kernel",) if nondeterministic else (),
        ("kernel-tolerance",) if nondeterministic else (),
    )
    assert (reference.run_id, reference.attempt_id) == (run_id, attempt_id)
    assert context.seed == seed and context.data_order_hash == "order"
    assert bool(context.preregistered_tolerance_ids) is nondeterministic
