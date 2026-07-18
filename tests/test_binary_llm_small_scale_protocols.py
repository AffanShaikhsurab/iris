from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from binary_llm.adapters import LocalSmallModelRequest, PYTHIA_70M, SMOLLM_135M
from binary_llm.domain import (
    AmbiguityCandidate,
    AmbiguityRegister,
    AmbiguityStatus,
    ArtifactRef,
    Budget,
    BINARYLLM_AMBIGUOUS_PATHS,
    BINARYLLM_PAPER_REFERENCE_SETTINGS,
    CheckpointBoundary,
    GateComparator,
    GateDefinition,
    GateSet,
    ModelIdentity,
    ScaleRung,
    SealedBaselineRef,
    SourceStatus,
    FidelityFieldResolution,
    StopMode,
    ToleranceDefinition,
    ToleranceSet,
    compare_reproduction_settings,
    resolved_settings_from_manifest,
    seed_ambiguity_register,
    build_binaryllm_reference_config,
)
from binary_llm.domain.validation import REQUIRED_FAILURE_GATE_CATEGORIES
from binary_llm.orchestration import (
    SmallScaleArmKind,
    SmallScaleProtocolInputs,
    SmallScaleProtocolKind,
    allocate_small_scale_model,
    build_small_scale_protocol_suite,
    preflight_small_scale_protocol,
    protocol_catalog,
    validate_serialized_small_scale_suite,
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _artifact(label: str) -> ArtifactRef:
    return ArtifactRef(label, "model", _hash(label), 1, "application/octet-stream")


def _model(kind: SmallScaleProtocolKind) -> ModelIdentity:
    registration = PYTHIA_70M if kind is SmallScaleProtocolKind.SCREENING else SMOLLM_135M
    return ModelIdentity(
        model_id=registration.model_id,
        revision=_hash(f"{kind}-model-revision"),
        tokenizer_revision=_hash(f"{kind}-tokenizer-revision"),
        architecture=registration.architecture,
        parameter_count=70_000_000 if kind is SmallScaleProtocolKind.SCREENING else 135_000_000,
        config_hash=_hash(f"{kind}-config"),
        tokenizer_hashes=(_hash(f"{kind}-tokenizer"),),
        template_hashes=(_hash(f"{kind}-template"),),
        weight_hashes=(_hash(f"{kind}-weights"),),
        tied_weights=registration.tied_weights,
    )


def _resolved_register() -> AmbiguityRegister:
    register_id = _hash("ambiguity-register")
    seeded = seed_ambiguity_register(register_id, "1")
    entries = []
    for entry in seeded.entries:
        candidates = entry.candidates or (
            AmbiguityCandidate("choice-a", "Candidate A", SourceStatus.FRAMEWORK_SELECTED),
            AmbiguityCandidate("choice-b", "Candidate B", SourceStatus.FRAMEWORK_SELECTED),
        )
        entries.append(
            replace(
                entry,
                candidates=candidates,
                matched_protocol="matched-small-scale",
                decision_rule="finite and above continuation floors",
                confidence_method="paired-bootstrap",
                required_floors=("finite", "continuation"),
                scale_scope=(ScaleRung.SMALL,),
                status=AmbiguityStatus.RESOLVED,
                selected_candidate=candidates[0].candidate_id,
                evidence_refs=(_hash(entry.ambiguity_id),),
                rejected_candidates=tuple(item.candidate_id for item in candidates[1:]),
            )
        )
    return AmbiguityRegister(register_id, "1", tuple(entries))


def _gates() -> GateSet:
    return GateSet(
        _hash("gate-set"),
        tuple(
            GateDefinition(
                gate_id=f"gate-{category.value}",
                category=category,
                metric_path="metrics.value",
                comparator=GateComparator.LE,
                threshold=1.0,
                baseline_relative=False,
                scope="small-scale",
                required_evidence_kind="evaluation",
                stop_mode=StopMode.IMMEDIATE_SAFE_BOUNDARY,
            )
            for category in REQUIRED_FAILURE_GATE_CATEGORIES
        ),
    )


def _tolerances() -> ToleranceSet:
    return ToleranceSet(
        _hash("tolerance-set"),
        (ToleranceDefinition("exact", "tokens", "small-scale", None, None, True),),
    )


def _inputs(kind: SmallScaleProtocolKind) -> SmallScaleProtocolInputs:
    model = _model(kind)
    baseline = SealedBaselineRef(
        baseline_id=f"{kind.value}-dense",
        role="base",
        model=model,
        artifact_refs=(_artifact(f"{kind}-baseline"),),
        evaluator_revision=_hash(f"{kind}-baseline-evaluator"),
        baseline_output_refs=(_artifact(f"{kind}-baseline-output"),),
        sealed_at="2026-01-01T00:00:00Z",
    )
    budget = (
        Budget(100_000, 100, 3_600, 10.0, CheckpointBoundary.PROGRESSIVE_PHASE)
        if kind is SmallScaleProtocolKind.SCREENING
        else Budget(10_000_000, 10_000, 86_400, 500.0, CheckpointBoundary.PROGRESSIVE_PHASE)
    )
    return SmallScaleProtocolInputs(
        kind=kind,
        model=model,
        parent_checkpoint=_artifact(f"{kind}-parent"),
        baseline_refs=(baseline,),
        binary_scope=("attention.q", "attention.k", "attention.v", "attention.o", "ffn.up", "ffn.down"),
        ambiguity_register=_resolved_register(),
        fidelity_config=build_binaryllm_reference_config(
            {
                path: FidelityFieldResolution(
                    path,
                    f"resolved:{ambiguity_id}",
                    SourceStatus.PAPER_INFERRED,
                    f"ambiguity-register:{ambiguity_id}",
                    ambiguity_id,
                )
                for path, ambiguity_id in BINARYLLM_AMBIGUOUS_PATHS.items()
            }
        ),
        gate_set=_gates(),
        tolerance_set=_tolerances(),
        claims=(),
        corpus_hash=_hash(f"{kind}-corpus"),
        partition_hashes=tuple(_hash(f"{kind}-partition-{index}") for index in range(20)),
        frozen_evaluation_hashes=(_hash(f"{kind}-frozen-eval"),),
        evaluator_protocol_hashes={name: _hash(f"{kind}-{name}") for name in ("perplexity", "paper_zero_shot", "sign", "dense_control")},
        seed_set=(7, 11),
        data_order_hash=_hash(f"{kind}-data-order"),
        budget=budget,
        source_revision=_hash("source-revision"),
        dependency_lock_hash=_hash("dependency-lock"),
        container_digest=_hash("container"),
        command_prefix=("python", "scripts/run-binary-llm-small.py"),
        sanitized_environment={
            "offline": True,
            "clean_tree": True,
            "hardware_inventory": "test-cpu",
            "compiler_inventory": "python-test",
        },
        preregistered_at="2026-01-01T00:00:00Z",
        screening_evidence_hash=(
            _hash("qualifying-screening-evidence")
            if kind is SmallScaleProtocolKind.PAPER_REFERENCE
            else None
        ),
    )


def test_full_reference_suite_preregisters_all_matched_evidence_arms() -> None:
    inputs = _inputs(SmallScaleProtocolKind.PAPER_REFERENCE)
    suite = build_small_scale_protocol_suite(inputs)
    report = preflight_small_scale_protocol(suite, inputs)

    assert report.compute_allowed
    assert {item.arm for item in suite.arms} == set(SmallScaleArmKind)
    reference = next(item for item in suite.arms if item.arm is SmallScaleArmKind.REFERENCE_DUAL)
    body_only = next(
        item for item in suite.arms if item.arm is SmallScaleArmKind.STAGE2_BODY_ONLY
    )
    assert reference.manifest.stage1_config["steps"] == 50
    assert reference.manifest.progressive_config["phase_count"] == 20
    assert reference.manifest.progressive_config["final_views"] == ("progressive", "sign")
    assert reference.manifest.operator_config["scale_variant"] == "dual"
    assert reference.manifest.progressive_config["trainability_arm"] == "all_model_parameters"
    assert body_only.manifest.progressive_config["trainability_arm"] == "binary_body_only"
    assert {
        key
        for key in reference.manifest.progressive_config
        if reference.manifest.progressive_config[key]
        != body_only.manifest.progressive_config[key]
    } == {"trainability_arm"}
    assert len(reference.manifest.partition_refs) == 20
    assert reference.source_classification.value == "modified_reproduction"
    assert reference.source_differences
    assert all(item.source_differences for item in suite.arms)
    assert all(
        difference.ablation_ref.startswith("source-difference:binaryllm-paper:")
        for item in suite.arms
        for difference in item.source_differences
    )
    recorded_refs = {
        difference.path: difference.ablation_ref
        for difference in reference.source_differences
    }
    assert compare_reproduction_settings(
        BINARYLLM_PAPER_REFERENCE_SETTINGS,
        resolved_settings_from_manifest(reference.manifest),
        ablation_refs=recorded_refs,
    ) == reference.source_differences
    assert validate_serialized_small_scale_suite(suite.to_dict()) is None


def test_screening_suite_is_bounded_and_uses_the_registered_70m_model() -> None:
    inputs = _inputs(SmallScaleProtocolKind.SCREENING)
    suite = build_small_scale_protocol_suite(inputs)
    reference = next(item for item in suite.arms if item.arm is SmallScaleArmKind.REFERENCE_DUAL)

    assert reference.manifest.model.model_id == PYTHIA_70M.model_id
    assert reference.manifest.stage1_config["steps"] == 5
    assert reference.manifest.budget.max_tokens == 100_000
    excessive = replace(inputs, budget=replace(inputs.budget, max_tokens=2_000_001))
    with pytest.raises(ValueError, match="bounded ceiling"):
        build_small_scale_protocol_suite(excessive)


def test_unresolved_or_non_hash_prerequisites_block_protocol_before_allocation() -> None:
    inputs = _inputs(SmallScaleProtocolKind.SCREENING)
    with pytest.raises(ValueError, match="resolved lowercase SHA-256"):
        build_small_scale_protocol_suite(replace(inputs, data_order_hash="unresolved"))

    open_register = seed_ambiguity_register(_hash("open-register"), "1")
    with pytest.raises(ValueError, match="unresolved"):
        build_small_scale_protocol_suite(replace(inputs, ambiguity_register=open_register))


def test_allocator_is_not_called_when_preflight_state_has_changed(tmp_path) -> None:
    inputs = _inputs(SmallScaleProtocolKind.SCREENING)
    suite = build_small_scale_protocol_suite(inputs)
    first = suite.arms[0]
    altered_manifest = replace(
        first.manifest,
        determinism_policy={"deterministic": True, "data_order_hash": _hash("changed-order")},
    )
    altered = replace(suite, arms=(replace(first, manifest=altered_manifest), *suite.arms[1:]))
    allocated: list[bool] = []

    def allocator(_request):
        allocated.append(True)
        return object()

    request = LocalSmallModelRequest(tmp_path, tmp_path, inputs.model)
    with pytest.raises(ValueError, match="data order"):
        allocate_small_scale_model(altered, inputs, request, allocator)
    assert allocated == []


def test_allocator_runs_only_after_complete_preflight(tmp_path) -> None:
    inputs = _inputs(SmallScaleProtocolKind.SCREENING)
    suite = build_small_scale_protocol_suite(inputs)
    allocated: list[LocalSmallModelRequest] = []
    request = LocalSmallModelRequest(tmp_path, tmp_path, inputs.model)

    result = allocate_small_scale_model(suite, inputs, request, lambda item: allocated.append(item) or "loaded")

    assert result == "loaded"
    assert allocated == [request]


def test_command_catalog_exposes_every_preallocation_requirement() -> None:
    catalog = protocol_catalog()
    requirements = set(catalog["preallocation_requirements"])

    assert set(catalog["arms"]) == {item.value for item in SmallScaleArmKind}
    assert {"resolved_hashes", "ambiguity_choices", "seed_set", "data_order_hash"} <= requirements
    assert {"max_tokens", "max_optimizer_steps", "max_wall_seconds", "max_billable_cost"} <= requirements
    assert {"failure_gates", "progressive_phase_checkpoint_boundary", "source_difference_classification"} <= requirements
