from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace

import pytest

from binary_llm.domain import (
    AmbiguityCandidate,
    AmbiguityRegister,
    AmbiguityStatus,
    ArtifactRef,
    BINARY_FAMILY,
    BUILTIN_FAMILIES,
    Budget,
    CandidateRecord,
    CheckpointBoundary,
    ClaimRecord,
    ClaimStatus,
    EvaluationEvidence,
    ExperimentFamilyIdentity,
    ExperimentFamilyKind,
    ExperimentManifest,
    FormatSpec,
    GateComparator,
    GateDefinition,
    GateSet,
    ModelIdentity,
    REQUIRED_AMBIGUITY_KEYS,
    REQUIRED_FAILURE_GATE_CATEGORIES,
    RepresentationKind,
    RepresentationSpec,
    ScaleRung,
    SealedBaselineRef,
    SourceStatus,
    StopMode,
    ToleranceDefinition,
    ToleranceSet,
    seed_ambiguity_register,
    validate_candidate,
    validate_manifest,
    resolved_settings_from_manifest,
)


def model_identity() -> ModelIdentity:
    return ModelIdentity(
        model_id="model", revision="rev", tokenizer_revision="tok-rev",
        architecture="test", parameter_count=100, config_hash="config",
        tokenizer_hashes=("tokenizer",), template_hashes=("template",),
        weight_hashes=("weights",), tied_weights=False,
    )


def artifact(artifact_id: str, parent: str | None = None) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=artifact_id, kind="checkpoint", sha256="a" * 64, bytes=1,
        media_type="application/octet-stream", parent_artifact_id=parent,
    )


def representation() -> RepresentationSpec:
    return RepresentationSpec(
        representation_id="binary-v1", kind=RepresentationKind.BINARY, bit_width=1,
        payload_dtype="packed_bits", scale_dtype="float16", offset_dtype=None,
        exception_dtype=None, has_inference_offset=False, higher_precision_exceptions=(),
        zero_sign_rule="positive", metadata={"packing": {"bit_order": "lsb0"}},
    )


def format_spec() -> FormatSpec:
    return FormatSpec(
        format_id="packed-v1", version="1", bit_order="lsb0", row_order="row_major",
        row_alignment=1, zero_sign_rule="positive", scale_dtype="float16",
        scale_endianness="little", metadata_encoding="json",
        allowed_representation_ids=("binary-v1",), temporary_expansion_limit_bytes=1024,
    )


def gate_set() -> GateSet:
    definitions = tuple(
        GateDefinition(
            gate_id=f"gate-{category.value}", category=category, metric_path="metrics.value",
            comparator=GateComparator.LE, threshold=1.0, baseline_relative=False,
            scope="experiment", required_evidence_kind="evaluation",
            stop_mode=StopMode.IMMEDIATE_SAFE_BOUNDARY,
        )
        for category in REQUIRED_FAILURE_GATE_CATEGORIES
    )
    return GateSet("required-gates", definitions)


def tolerance_set() -> ToleranceSet:
    return ToleranceSet(
        "tolerances",
        (ToleranceDefinition("exact-output", "outputs.tokens", "runtime", None, None, True),),
    )


def resolved_register() -> AmbiguityRegister:
    seeded = seed_ambiguity_register("ambiguities", "1")
    entries = []
    for entry in seeded.entries:
        candidates = entry.candidates or (
            AmbiguityCandidate("choice-a", "Preregistered candidate A.", SourceStatus.FRAMEWORK_SELECTED),
            AmbiguityCandidate("choice-b", "Preregistered candidate B.", SourceStatus.FRAMEWORK_SELECTED),
        )
        selected = candidates[0].candidate_id
        entries.append(
            replace(
                entry, candidates=candidates, matched_protocol="protocol-1",
                decision_rule="Pass stability and continuation floors.",
                confidence_method="paired-bootstrap", required_floors=("finite", "continuation"),
                scale_scope=(ScaleRung.SMALL,), status=AmbiguityStatus.RESOLVED,
                selected_candidate=selected, evidence_refs=(f"evidence-{entry.ambiguity_id}",),
                rejected_candidates=tuple(item.candidate_id for item in candidates[1:]),
            )
        )
    return AmbiguityRegister("ambiguities", "1", tuple(entries))


def claim() -> ClaimRecord:
    return ClaimRecord(
        claim_id="claim-1", source_uri="https://example.invalid/paper",
        source_version_hash="paper-hash", statement="A paper claim.", category="formula",
        status=ClaimStatus.NOT_TESTED, scale_scope=(ScaleRung.SMALL,),
        protocol_refs=("protocol-1",), evidence_refs=(), differences=(),
    )


def manifest() -> ExperimentManifest:
    checkpoint = artifact("checkpoint")
    baseline_artifact = artifact("baseline-model")
    baseline_output = artifact("baseline-output")
    baseline = SealedBaselineRef(
        baseline_id="bf16", role="bf16", model=model_identity(),
        artifact_refs=(baseline_artifact,), evaluator_revision="eval-1",
        baseline_output_refs=(baseline_output,), sealed_at="2026-01-01T00:00:00Z",
    )
    return ExperimentManifest(
        experiment_id="experiment", family=BINARY_FAMILY, hypothesis_refs=("hypothesis",),
        source_claim_refs=("claim-1",), model=model_identity(), scale_rung=ScaleRung.SMALL,
        parent_checkpoint=checkpoint, baseline_refs=(baseline,), binary_scope=("attention.q",),
        operator_config={"operator": "progressive"}, ambiguity_register_ref="ambiguities",
        stage1_config={"steps": 50}, progressive_config={"phases": 20},
        recovery_config=None, teacher_refs=(), corpus_ref="corpus",
        partition_refs=("partition",), frozen_evaluation_refs=("eval-set",),
        evaluator_protocols=("protocol-1",), seed_set=(7,),
        determinism_policy={"data_order": "seeded"},
        budget=Budget(100, 50, 60, 1.0, CheckpointBoundary.PROGRESSIVE_PHASE),
        gate_set_ref="required-gates", tolerance_set_ref="tolerances",
        artifact_format=format_spec(), device_protocol=None, source_revision="source-rev",
        dependency_lock_hash="lock", container_digest="container", command=("run",),
        sanitized_environment={"offline": True}, preregistered_at="2026-01-01T00:00:00Z",
    )


def test_seed_register_contains_every_required_unresolved_choice():
    register = seed_ambiguity_register("register", "v1")

    assert tuple(item.ambiguity_id for item in register.entries) == REQUIRED_AMBIGUITY_KEYS
    assert all(item.status is AmbiguityStatus.OPEN for item in register.entries)
    assert all(item.selected_candidate is None for item in register.entries)
    assert all(item.source_status is SourceStatus.PAPER_INFERRED for item in register.entries)


def test_family_and_source_status_identities_are_explicit_and_stable():
    assert tuple(family.family_id for family in BUILTIN_FAMILIES) == (
        "binary", "ternary", "structured_pruning_q4", "distilled_student_q4_or_qat"
    )
    assert {item.value for item in SourceStatus} == {
        "paper_specified", "paper_inferred", "framework_selected"
    }


def test_models_are_frozen_and_convert_nested_state_to_json():
    spec = representation()

    with pytest.raises(FrozenInstanceError):
        spec.bit_width = 2  # type: ignore[misc]
    with pytest.raises(TypeError):
        spec.metadata["packing"] = {}  # type: ignore[index]

    encoded = json.dumps(spec.to_dict(), sort_keys=True)
    assert '"bit_width": 1' in encoded
    assert spec.to_dict()["metadata"]["packing"]["bit_order"] == "lsb0"


def test_representation_requires_explicit_binary_exceptions_and_offset_storage():
    with pytest.raises(ValueError, match="one bit"):
        replace(representation(), bit_width=2)
    with pytest.raises(ValueError, match="offset"):
        replace(representation(), has_inference_offset=True)
    with pytest.raises(ValueError, match="exception dtype"):
        replace(representation(), higher_precision_exceptions=("lm_head",))


def test_hybrid_identity_requires_a_new_name_and_component_controls():
    with pytest.raises(ValueError, match="distinct family_id"):
        ExperimentFamilyIdentity(
            "hybrid", ExperimentFamilyKind.HYBRID, ("binary", "ternary")
        )

    hybrid = ExperimentFamilyIdentity(
        "binary-plus-ternary-v1", ExperimentFamilyKind.HYBRID, ("binary", "ternary")
    )
    with pytest.raises(ValueError, match="one distinct control"):
        replace(
            manifest(), family=hybrid, binary_scope=(), operator_config={}, stage1_config={},
            progressive_config={}, component_control_experiment_ids=("binary-control",),
        )


def test_manifest_validation_fails_closed_on_unresolved_ambiguities():
    with pytest.raises(ValueError, match="unresolved scientific choices"):
        validate_manifest(
            manifest(), ambiguity_register=seed_ambiguity_register("ambiguities", "1"),
            gate_set=gate_set(), tolerance_set=tolerance_set(), claims=(claim(),),
            representations=(representation(),),
        )


def test_manifest_validation_requires_every_failure_gate_category():
    incomplete = GateSet("required-gates", gate_set().definitions[:-1])

    with pytest.raises(ValueError, match="required failure-gate categories"):
        validate_manifest(
            manifest(), ambiguity_register=resolved_register(), gate_set=incomplete,
            tolerance_set=tolerance_set(), claims=(claim(),),
            representations=(representation(),),
        )


def test_fully_explicit_manifest_validates_and_is_json_compatible():
    item = manifest()

    validate_manifest(
        item, ambiguity_register=resolved_register(), gate_set=gate_set(),
        tolerance_set=tolerance_set(), claims=(claim(),), representations=(representation(),),
    )
    assert json.loads(json.dumps(item.to_dict()))["family"]["family_id"] == "binary"


def test_manifest_projection_includes_material_data_and_optimization_settings():
    projected = resolved_settings_from_manifest(manifest())

    assert projected.data["teacher_refs"] == ()
    assert projected.optimization["determinism_policy"] == {"data_order": "seeded"}
    assert projected.optimization["budget"]["max_optimizer_steps"] == 50


def test_candidate_validation_rejects_cross_family_evidence():
    checkpoint = artifact("checkpoint")
    packed = artifact("packed", parent="checkpoint")
    candidate = CandidateRecord(
        candidate_id="candidate", family=BINARY_FAMILY, model=model_identity(),
        scale_rung=ScaleRung.SMALL, source_checkpoint=checkpoint, artifact=packed,
        representation=representation(), evidence_refs=("evidence",),
        deployable_runtime_supported=False, metadata_accounting_complete=True,
    )
    evidence = EvaluationEvidence(
        evidence_id="evidence", artifact_id="packed", family_id="ternary",
        scale_rung=ScaleRung.SMALL, evaluation_set_id="set", evaluator_revision="eval",
        blind_status="development", evaluation_ordinal=1, seed=1,
        prompt_order_hash="order", decoding={"temperature": 0}, raw_outputs_ref="raw",
        per_case_ref="cases", aggregates={"accuracy": 1.0}, failure_categories=(),
        created_at="2026-01-01T00:00:00Z",
    )

    with pytest.raises(ValueError, match="different experiment family"):
        validate_candidate(candidate, evidence=(evidence,))


def test_duplicate_ids_and_unsupported_schema_versions_are_rejected():
    definition = gate_set().definitions[0]
    with pytest.raises(ValueError, match="duplicate identifiers"):
        GateSet("duplicates", (definition, definition))
    with pytest.raises(ValueError, match="unsupported schema_version"):
        replace(model_identity(), schema_version=999)
