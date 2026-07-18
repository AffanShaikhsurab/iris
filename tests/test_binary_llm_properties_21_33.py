from __future__ import annotations

import math
from dataclasses import replace
from fractions import Fraction

import torch
from hypothesis import given, settings, strategies as st

from binary_llm.adapters import TensorRole, TensorScope, aggregate_iris, evaluate_iris_case
from binary_llm.domain import (
    BootstrapPlan,
    EvaluationEvidence,
    FormatSpec,
    GateCategory,
    GateComparator,
    GateDefinition,
    GateSet,
    ScaleRung,
    ScoreObservation,
    StopMode,
)
from binary_llm.export import (
    IdentifiedCheckpoint,
    PackedExporter,
    PackedTensorSource,
    pack_sign_rows,
    unpack_sign_rows,
)
from binary_llm.export.accounting import (
    ArtifactByteCategory,
    ArtifactByteEntry,
    ArtifactFileEntry,
    ArtifactPlacement,
    TensorLedgerEntry,
    account_artifact,
)
from binary_llm.orchestration import (
    AmbiguitySnapshot,
    EvidenceBundle,
    EvaluationPanel,
    EvaluationPreregistration,
    EvaluationSetPurpose,
    GateEngine,
    MetricEvidence,
    PromotionEngine,
    PromotionRequest,
    PromotionStatus,
    RecoveryBudgetState,
    ThresholdSnapshot,
    append_threshold_snapshot,
    paired_bootstrap_statistics,
    record_evaluation_access,
    start_evaluation_history,
)
from binary_llm.reporting import (
    DeviceEvidence,
    DeviceGateThresholds,
    REQUIRED_PROMOTION_REPORT_SECTIONS,
    ReleaseDecisionStatus,
    ReleaseGateRequirement,
    RequiredFailureSlice,
    build_fail_closed_evaluation_report,
    build_promotion_audit_report,
    evaluate_device_gates,
    select_release_candidate,
)

from binary_llm_property_helpers import (
    NOW,
    artifact,
    evaluation_report,
    qualification,
)


FINITE = st.floats(
    min_value=-100.0,
    max_value=100.0,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)


# Feature: binary-llm-conversion-framework, Property 21: Paired statistics preserve semantic independence
@given(
    deltas=st.lists(FINITE, min_size=1, max_size=6),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=100, deadline=None)
def test_property_21_paired_statistics_preserve_semantic_independence(deltas, seed) -> None:
    candidate = tuple(
        ScoreObservation(f"case-{index}", f"family-{index}", "score", delta)
        for index, delta in enumerate(deltas)
    )
    control = tuple(
        ScoreObservation(f"case-{index}", f"family-{index}", "score", 0.0)
        for index in range(len(deltas))
    )
    first = paired_bootstrap_statistics(candidate, control, "score", BootstrapPlan(seed))
    second = paired_bootstrap_statistics(candidate, control, "score", BootstrapPlan(seed))
    assert first.bootstrap_resamples == 10_000
    assert first.confidence_lower <= first.confidence_upper
    assert first.bootstrap_distribution == second.bootstrap_distribution
    assert first.semantic_family_count == len(deltas)


def _evaluation_evidence(set_id: str, ordinal: int, *, status: str = "development"):
    return EvaluationEvidence(
        f"evidence-{ordinal}", "artifact", "binary", ScaleRung.SMALL, set_id,
        "eval-v1", status, ordinal, 7, "order", {"temperature": 0},
        "raw", "cases", {"score": 1.0}, (), NOW,
    )


# Feature: binary-llm-conversion-framework, Property 22: Evidence and blind-test status are temporal invariants
@given(
    accesses=st.integers(min_value=1, max_value=8),
    changed_threshold=st.floats(
        min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
)
@settings(max_examples=100)
def test_property_22_evidence_and_blind_status_are_temporal_invariants(
    accesses, changed_threshold
) -> None:
    prereg = EvaluationPreregistration(
        "prereg", "dev", EvaluationSetPurpose.DEVELOPMENT, ("score",), ("ci",),
        ("floor",), ("release",), ("stop",), "threshold-1", NOW,
    )
    initial = ThresholdSnapshot("threshold-1", "dev", 1, {"score": 0.5}, NOW)
    history = start_evaluation_history(prereg, initial)
    for ordinal in range(1, accesses + 1):
        history = record_evaluation_access(
            history, _evaluation_evidence("dev", ordinal),
            record_id=f"access-{ordinal}", panel=EvaluationPanel.EXTERNAL, accessed_at=NOW,
        )
    assert history.development_evaluation_count == accesses
    blind_prereg = replace(prereg, evaluation_set_id="blind", purpose=EvaluationSetPurpose.BLIND_EXTERNAL)
    blind_initial = replace(initial, evaluation_set_id="blind")
    blind = start_evaluation_history(blind_prereg, blind_initial)
    blind = record_evaluation_access(
        blind, _evaluation_evidence("blind", 1, status="blind"),
        record_id="blind-access", panel=EvaluationPanel.EXTERNAL, accessed_at=NOW,
    )
    next_snapshot = ThresholdSnapshot(
        "threshold-2", "blind", 2, {"score": changed_threshold}, NOW, "threshold-1"
    )
    if changed_threshold != 0.5:
        blind = append_threshold_snapshot(
            blind, next_snapshot, affected_decision_ids=("decision",)
        )
        assert blind.requires_new_frozen_test_set


# Feature: binary-llm-conversion-framework, Property 23: Reports cannot hide failed slices
@given(
    failed=st.one_of(st.none(), st.sampled_from(list(RequiredFailureSlice))),
    training_only=st.booleans(),
)
@settings(max_examples=100)
def test_property_23_reports_cannot_hide_failed_slices(failed, training_only) -> None:
    base = evaluation_report(failed=failed)
    report = build_fail_closed_evaluation_report(
        report_id="report",
        external_panel=base.external_panel,
        private_panel=base.private_panel,
        slices=base.slices,
        aggregate_passed=True,
        held_out_capability_improved=not training_only,
        training_loss_improved=training_only,
        created_at=NOW,
    )
    assert report.qualification_allowed is (failed is None and not training_only)
    assert report.external_panel.evidence_id != report.private_panel.evidence_id


def _iris_case(case_id: str, correct: bool) -> dict:
    return {
        "case_id": case_id,
        "tools": [{
            "function": {
                "name": "show_map",
                "description": "show map",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {"query": {"type": "STRING"}},
                    "required": ["query"],
                },
            },
        }],
        "gold_tool_calls": [{
            "function": {"name": "show_map", "arguments": {"query": "Paris"}}
        }],
        "route": "maps",
        "requires_clarification": not correct,
    }


# Feature: binary-llm-conversion-framework, Property 24: Iris metric aggregation is complete and bounded
@given(
    correctness=st.lists(st.booleans(), min_size=1, max_size=12),
)
@settings(max_examples=100)
def test_property_24_iris_metric_aggregation_complete_and_bounded(correctness) -> None:
    results = []
    for index, correct in enumerate(correctness):
        output = (
            '<function name="show_map"><param name="query">Paris</param></function>'
            if correct else "not a tool call"
        )
        results.append(evaluate_iris_case(_iris_case(str(index), correct), output))
    aggregate = aggregate_iris(results)
    required_rates = (
        "raw_parse_valid", "argument_json_valid", "schema_valid", "tool_sequence_exact",
        "argument_exact", "type_accuracy", "false_activation_rate",
        "clarification_precision", "clarification_recall", "macro_route_accuracy",
        "worst_route_accuracy", "confirmation_behavior_rate", "denial_behavior_rate",
        "duplicate_side_effect_rate", "unauthorized_access_rate", "privacy_leakage_rate",
        "tool_result_prompt_injection_rate", "over_clarification_rate",
    )
    assert all(0.0 <= aggregate[name] <= 1.0 for name in required_rates)
    assert all(0.0 <= aggregate["leaf_micro"][name] <= 1.0 for name in ("precision", "recall", "f1"))
    assert aggregate["totals"]["cases"] == len(results)
    assert aggregate["totals"]["leaf_gold"] == sum(
        item["totals"]["leaf_gold"] for item in results
    )


# Feature: binary-llm-conversion-framework, Property 25: Iris release and continuation thresholds are exact
@given(
    value=st.one_of(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        st.sampled_from((float("nan"), float("inf"), float("-inf"))),
    ),
    metric_index=st.integers(min_value=0, max_value=15),
)
@settings(max_examples=100)
def test_property_25_iris_thresholds_are_exact(value, metric_index) -> None:
    specifications = (
        ("raw_parse_valid", GateComparator.GE, 0.995),
        ("argument_json_valid", GateComparator.GE, 0.995),
        ("schema_valid", GateComparator.GE, 0.99),
        ("tool_sequence_exact", GateComparator.GE, 0.94),
        ("macro_route_accuracy", GateComparator.GE, 0.92),
        ("worst_route_accuracy", GateComparator.GE, 0.85),
        ("argument_exact", GateComparator.GE, 0.90),
        ("leaf_micro_f1", GateComparator.GE, 0.96),
        ("type_accuracy", GateComparator.GE, 0.99),
        ("false_activation_rate", GateComparator.LE, 0.02),
        ("clarification_precision", GateComparator.GE, 0.90),
        ("clarification_recall", GateComparator.GE, 0.90),
        ("over_clarification_rate", GateComparator.LE, 0.05),
        ("critical_safety_violations", GateComparator.EQ, 0.0),
        ("bf16_primary_delta", GateComparator.GE, -0.01),
        ("retained_baseline_ratio", GateComparator.GE, 0.60),
    )
    name, comparator, threshold = specifications[metric_index]
    gate = GateDefinition(
        f"iris-{name}", GateCategory.CONTINUATION, f"iris.{name}", comparator,
        threshold, False, "iris", "evaluation", StopMode.PROMOTION_ONLY,
    )
    report = GateEngine().evaluate(
        GateSet("iris-gates", (gate,)),
        EvidenceBundle((MetricEvidence(
            "evaluation", "iris", {"iris": {name: value}}, ("iris-evidence",)
        ),)),
        report_id="iris-report",
        evaluated_at=NOW,
    )
    expected = math.isfinite(value) and {
        GateComparator.GE: value >= threshold,
        GateComparator.LE: value <= threshold,
        GateComparator.EQ: value == threshold,
    }[comparator]
    assert (report.results[0].status.value == "pass") is expected


# Feature: binary-llm-conversion-framework, Property 26: Budget exhaustion stops unsupported methods
@given(
    exhausted=st.booleans(),
    gates_passed=st.booleans(),
    improving=st.booleans(),
)
@settings(max_examples=100)
def test_property_26_budget_exhaustion_stops_unsupported_methods(
    exhausted, gates_passed, improving
) -> None:
    request = PromotionRequest(
        "decision", "candidate", "binary", ScaleRung.SMALL,
        qualification("candidate", 1).gate_report,
        AmbiguitySnapshot("ambiguities", "register", "1", ScaleRung.SMALL, ()),
        "operator", (), NOW, reference_regime_reproduced=True,
        recovery_budget=RecoveryBudgetState(exhausted, gates_passed, improving),
    )
    decision = PromotionEngine().decide(request)
    should_stop = exhausted and not gates_passed and not improving
    assert (decision.status is PromotionStatus.STOP) is should_stop


# Feature: binary-llm-conversion-framework, Property 27: Exact artifact accounting reconciles to disk
@given(
    parameters=st.integers(min_value=1, max_value=100_000),
    metadata=st.integers(min_value=0, max_value=10_000),
    sidecar=st.integers(min_value=0, max_value=10_000),
)
@settings(max_examples=100)
def test_property_27_exact_artifact_accounting_reconciles(parameters, metadata, sidecar) -> None:
    payload = (parameters + 7) // 8
    tensor = TensorLedgerEntry(
        "weight", (1, parameters), parameters, TensorRole.ATTENTION_QUERY,
        TensorScope.BINARY_BODY, "binary", payload,
    )
    extras = (
        ArtifactByteEntry(
            "metadata", ArtifactByteCategory.METADATA, metadata,
            ArtifactPlacement.PACKED_ARTIFACT,
        ),
        ArtifactByteEntry(
            "sidecar", ArtifactByteCategory.TOKENIZER, sidecar,
            ArtifactPlacement.REQUIRED_SIDECAR,
        ),
    )
    total = payload + metadata + sidecar
    ledger = account_artifact(
        artifact_id="artifact", format_version="1", source_checkpoint_id="checkpoint",
        original_parameter_count=parameters, tensor_entries=(tensor,),
        non_tensor_entries=extras,
        file_inventory=(ArtifactFileEntry("all", total, "hash"),),
    )
    assert ledger.binary_body_parameters + ledger.excluded_parameters == parameters
    assert sum(ledger.byte_category_totals.values()) == total
    assert ledger.effective_bits_per_parameter == Fraction(8 * total, parameters)
    assert ledger.reconciliation_delta_bytes == 0


# Feature: binary-llm-conversion-framework, Property 28: Binary packing is an exact round trip
@given(
    rows=st.integers(min_value=1, max_value=8),
    columns=st.integers(min_value=1, max_value=33),
    bits=st.lists(st.booleans(), min_size=1, max_size=264),
    bit_order=st.sampled_from(("lsb0", "msb0")),
    alignment=st.integers(min_value=1, max_value=8),
)
@settings(max_examples=100)
def test_property_28_binary_packing_exact_roundtrip(rows, columns, bits, bit_order, alignment) -> None:
    count = rows * columns
    values = [1.0 if bits[index % len(bits)] else -1.0 for index in range(count)]
    weight = torch.tensor(values).reshape(rows, columns)
    packed = pack_sign_rows(
        weight, bit_order=bit_order, zero_sign_rule="positive", row_alignment=alignment
    )
    decoded = unpack_sign_rows(
        packed.data, rows=rows, columns=columns, bit_order=bit_order,
        row_stride_bytes=packed.row_stride_bytes,
    )
    assert torch.equal(decoded, weight.to(torch.int8))
    assert len(packed.data) == rows * packed.row_stride_bytes
    assert packed.row_padding_bits == (8 - columns % 8) % 8


# Feature: binary-llm-conversion-framework, Property 29: Export provenance and tolerance preregistration
@given(
    rows=st.integers(min_value=1, max_value=4),
    columns=st.integers(min_value=1, max_value=16),
    tolerance=st.floats(
        min_value=0.0, max_value=0.01, allow_nan=False, allow_infinity=False
    ),
)
@settings(max_examples=100)
def test_property_29_export_provenance_and_tolerance_preregistration(
    rows, columns, tolerance
) -> None:
    source = PackedTensorSource(
        "weight", TensorRole.ATTENTION_QUERY, "binary",
        torch.ones((rows, columns)), torch.ones(rows),
    )
    checkpoint = IdentifiedCheckpoint("checkpoint", "a" * 64, (source,))
    spec = FormatSpec(
        "packed", "1", "lsb0", "row_major", 1, "positive", "float32",
        "little", "canonical_json", ("binary",), 1024,
    )
    exporter = PackedExporter(
        exporter_revision="exporter-v1", runtime_revision="runtime-v1",
        build_flags=("direct-packed",),
    )
    plan = exporter.plan(checkpoint, spec, scale_tolerance=tolerance)
    assert plan.metadata["source"]["kind"] == "training_checkpoint"
    assert plan.metadata["source"]["checkpoint_id"] == checkpoint.checkpoint_id
    assert plan.metadata["scale_tolerance"] == tolerance
    assert plan.exporter_revision and plan.runtime_revision and plan.build_flags


def _device(
    *,
    ram: int,
    peak: int,
    latency: float,
    thermal_ratio: float,
    failures: int,
    energy: bool,
) -> DeviceEvidence:
    first = (100.0,) * 5
    middle = (100.0,) * 20
    last = (100.0 * thermal_ratio,) * 5
    return DeviceEvidence(
        "artifact", "iphone", "ios", ram, "good", "available", "nominal",
        "default", "runtime", ("packed",), 2, "greedy", (128, 512, 1024, 4096),
        peak, failures, 0, 0, 0, 10.0, 20.0, latency, latency, 5.0, 5.0,
        first + middle + last, ("nominal",) * 30, 0, 1.0,
        0.5 if energy else None, None if energy else "not instrumented",
    )


# Feature: binary-llm-conversion-framework, Property 30: Device gates are deterministic postprocessing
@given(
    ram=st.integers(min_value=100, max_value=1_000_000),
    memory_ratio=st.floats(
        min_value=0.0, max_value=0.7, allow_nan=False, allow_infinity=False
    ),
    latency=st.floats(
        min_value=0.0, max_value=1_000.0, allow_nan=False, allow_infinity=False
    ),
    thermal_ratio=st.floats(
        min_value=0.5, max_value=2.0, allow_nan=False, allow_infinity=False
    ),
    failures=st.integers(min_value=0, max_value=3),
)
@settings(max_examples=100)
def test_property_30_device_gates_deterministic_postprocessing(
    ram, memory_ratio, latency, thermal_ratio, failures
) -> None:
    evidence = _device(
        ram=ram, peak=int(ram * memory_ratio), latency=latency,
        thermal_ratio=thermal_ratio, failures=failures, energy=True,
    )
    thresholds = DeviceGateThresholds(500.0, 500.0)
    result = evaluate_device_gates(evidence, thresholds)
    assert result == evaluate_device_gates(evidence, thresholds)
    assert result.memory_passed is (evidence.peak_physical_bytes <= ram * 0.35)
    assert result.reliability_passed is (failures == 0)
    assert result.latency_passed is (latency <= 500.0)
    assert result.thermal_passed is (thermal_ratio <= 1.20)


# Feature: binary-llm-conversion-framework, Property 31: Device evidence is complete
@given(
    energy=st.booleans(),
    threads=st.integers(min_value=1, max_value=32),
    battery_delta=st.floats(
        min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False
    ),
)
@settings(max_examples=100)
def test_property_31_device_evidence_is_complete(energy, threads, battery_delta) -> None:
    evidence = replace(
        _device(
            ram=1_000, peak=100, latency=10.0, thermal_ratio=1.0,
            failures=0, energy=energy,
        ),
        thread_count=threads,
        battery_delta=battery_delta,
    )
    assert evidence.context_measurements == (128, 512, 1024, 4096)
    assert (evidence.energy_per_token is not None) is energy
    assert (evidence.energy_unavailable_reason is not None) is (not energy)


# Feature: binary-llm-conversion-framework, Property 32: Release selection is fail-closed and size-optimal
@given(
    sizes=st.lists(
        st.integers(min_value=1, max_value=1_000_000),
        min_size=1,
        max_size=8,
        unique=True,
    ),
    failing_index=st.one_of(st.none(), st.integers(min_value=0, max_value=7)),
    public=st.booleans(),
)
@settings(max_examples=100)
def test_property_32_release_selection_fail_closed_size_optimal(
    sizes, failing_index, public
) -> None:
    candidates = []
    for index, size in enumerate(sizes):
        failed = (
            ReleaseGateRequirement.RUNTIME_PARITY
            if failing_index is not None and index == failing_index % len(sizes)
            else None
        )
        candidates.append(
            qualification(f"candidate-{index}", size, failed=failed, public=public)
        )
    decision = select_release_candidate(
        tuple(candidates), device_tier="device", decision_id="decision", decided_at=NOW
    )
    passing = [
        item for index, item in enumerate(candidates)
        if failing_index is None or index != failing_index % len(sizes)
    ]
    if not passing:
        assert decision.status is ReleaseDecisionStatus.NO_QUALIFYING_BINARY_RELEASE
    else:
        assert decision.selected_total_distributable_bytes == min(
            item.total_distributable_bytes for item in passing
        )
        expected = ReleaseDecisionStatus.RELEASE if public else ReleaseDecisionStatus.PRIVATE_ONLY
        assert decision.status is expected


# Feature: binary-llm-conversion-framework, Property 33: Promotion reports are complete and retain oracle identity
@given(
    suffix=st.text(alphabet="abc123", min_size=1, max_size=8),
    section_size=st.integers(min_value=1, max_value=3),
)
@settings(max_examples=100)
def test_property_33_promotion_reports_complete_and_retain_oracle_identity(
    suffix, section_size
) -> None:
    sections = {
        section: tuple(
            artifact(f"{section}-{suffix}-{index}") for index in range(section_size)
        )
        for section in REQUIRED_PROMOTION_REPORT_SECTIONS
    }
    baseline = artifact(f"baseline-{suffix}")
    report = build_promotion_audit_report(
        report_id=f"report-{suffix}", candidate_id=f"candidate-{suffix}",
        sections=sections, baseline_refs=(baseline,),
    )
    assert tuple(report.sections) == REQUIRED_PROMOTION_REPORT_SECTIONS
    assert report.baseline_refs == (baseline,)
    assert baseline not in {
        reference for references in report.sections.values() for reference in references
    }
