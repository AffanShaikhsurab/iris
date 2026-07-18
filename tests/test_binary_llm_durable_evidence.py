from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from binary_llm.adapters import ActiveRepresentation
from binary_llm.domain import ArtifactStoreError, ProvenanceViolation, canonical_json_bytes, sha256_bytes
from binary_llm.orchestration import (
    BinaryForwardEvent,
    CausalBatch,
    CorpusService,
    FilesystemArtifactStore,
    INITIAL_TOKEN_ALLOCATION,
    NoProgressPolicy,
    RecoveryCorpusMix,
    RecoveryCorpusPlan,
    RecoveryRunResult,
    RecoveryRunStatus,
    RecoveryTrainerConfig,
    SplitPolicy,
    Stage1OptimizerConfig,
    SupervisionMode,
    TeacherRouter,
    TeacherRoutingConfig,
    build_frozen_corpus_bundle,
    build_recovery_run_evidence,
    load_frozen_corpus_bundle,
    load_recovery_run_evidence,
    persist_frozen_corpus_bundle,
    persist_recovery_run_evidence,
    validate_recovery_token_mix,
)
from binary_llm.orchestration.corpus import ProvenanceRecord


def _records(prefix: str = "record") -> tuple[ProvenanceRecord, ...]:
    records = []
    for index, (capability_slice, fraction) in enumerate(INITIAL_TOKEN_ALLOCATION):
        normalized = {
            "messages": [{"role": "user", "content": f"{prefix}-{capability_slice}"}],
            "source_index": index,
        }
        records.append(ProvenanceRecord(
            record_id=f"{prefix}-{capability_slice}",
            content_hash=sha256_bytes(canonical_json_bytes(normalized)),
            source=f"fixture:{prefix}:{index}",
            source_revision="fixture-v1",
            license_or_terms="fixture-license",
            permitted_use="test-only",
            transformation_history=("fixture-normalization-v1",),
            semantic_family_id=f"semantic:{prefix}:{index}",
            split_family_id=f"split:{prefix}:{index}",
            deduplication_key=f"dedup:{prefix}:{index}",
            fuzzy_cluster_id=f"fuzzy:{prefix}:{index}",
            target_type="causal_lm",
            capability_slice=capability_slice,
            token_count=int(fraction * 100),
            synthetic=False,
            executable_or_human_verified=True,
            normalized_record=normalized,
        ))
    return tuple(records)


def _policy() -> SplitPolicy:
    return SplitPolicy(
        "durable-fixture-policy",
        47,
        target_allocation=INITIAL_TOKEN_ALLOCATION,
        allocation_tolerance=0.0,
    )


def _plan() -> RecoveryCorpusPlan:
    return RecoveryCorpusPlan(
        RecoveryCorpusMix(
            "initial",
            INITIAL_TOKEN_ALLOCATION,
            preregistered=True,
            initial=True,
        ),
        (
            RecoveryCorpusMix(
                "alternative",
                (
                    ("broad_text", 0.20),
                    ("math", 0.20),
                    ("code", 0.15),
                    ("iris_tools", 0.15),
                    ("conversation", 0.10),
                    ("safety_transitions", 0.10),
                    ("adversarial", 0.10),
                ),
                preregistered=True,
            ),
        ),
    )


def _config() -> RecoveryTrainerConfig:
    return RecoveryTrainerConfig(
        optimizer=Stage1OptimizerConfig("sgd", learning_rate=0.01, weight_decay=0.0),
        seed=9,
        max_optimizer_steps=1,
        target_representation=ActiveRepresentation.SIGN,
        no_progress_policy=NoProgressPolicy("calibration", "frozen-capability"),
    )


def _router() -> TeacherRouter:
    return TeacherRouter(
        TeacherRoutingConfig(
            student_tokenizer_id="student",
            sealed_bf16_teacher_identity_id="unused-behavioral-teacher",
            behavioral_mode=SupervisionMode.NONE,
            broad_mode=SupervisionMode.NONE,
        )
    )


def _batches(records: tuple[ProvenanceRecord, ...]) -> dict[str, CausalBatch]:
    return {
        record.record_id: CausalBatch(
            f"batch:{record.record_id}",
            torch.tensor([[index + 1, index + 2, index + 3]], dtype=torch.long),
        )
        for index, record in enumerate(records)
    }


def _stopped_result(records: tuple[ProvenanceRecord, ...]) -> RecoveryRunResult:
    return RecoveryRunResult(
        status=RecoveryRunStatus.STOPPED_BUDGET,
        completed_steps=0,
        mix_evidence=validate_recovery_token_mix(records, _plan().initial_mix),
        progress=(),
        steps=(),
        forward_events=(),
        teacher_provenance=(),
        no_progress_decision=None,
    )


def test_corpus_bundle_is_deterministic_persisted_and_freshly_resolved(tmp_path) -> None:
    records = _records()
    manifest = CorpusService().freeze(records, _policy(), version="fixture-v1")
    first = build_frozen_corpus_bundle(manifest, _policy())
    second = build_frozen_corpus_bundle(manifest, _policy())

    assert first.payload == second.payload
    assert first.bundle_id == second.bundle_id
    store = FilesystemArtifactStore(tmp_path / "artifacts")
    persist_frozen_corpus_bundle(first, store)
    loaded = load_frozen_corpus_bundle(
        FilesystemArtifactStore(tmp_path / "artifacts"), first.artifact_ref
    )

    assert loaded.manifest == manifest
    assert loaded.split_policy == _policy()
    assert len(loaded.manifest.partition_refs) == 20
    assert {
        item for partition in loaded.manifest.partition_refs for item in partition.record_ids
    } == {item.record_id for item in records}
    assert all(
        item.content_hash == sha256_bytes(canonical_json_bytes(item.normalized_record))
        for item in loaded.manifest.records
    )


def test_corpus_bundle_missing_and_tampered_payload_fail_closed(tmp_path) -> None:
    manifest = CorpusService().freeze(_records(), _policy())
    bundle = build_frozen_corpus_bundle(manifest, _policy())
    store = FilesystemArtifactStore(tmp_path / "artifacts")

    with pytest.raises(ArtifactStoreError) as missing:
        load_frozen_corpus_bundle(store, bundle.artifact_ref)
    assert missing.value.code == "artifact_store.not_found"

    persist_frozen_corpus_bundle(bundle, store)
    path = store.object_path(bundle.artifact_ref.sha256)
    path.write_bytes(bundle.payload[:-1] + bytes([bundle.payload[-1] ^ 1]))
    with pytest.raises(ArtifactStoreError) as tampered:
        load_frozen_corpus_bundle(store, bundle.artifact_ref)
    assert tampered.value.code == "artifact_store.payload_corrupt"


def test_recovery_evidence_round_trips_exact_batches_and_is_non_promotional(tmp_path) -> None:
    records = _records()
    manifest = CorpusService().freeze(records, _policy())
    bundle = build_frozen_corpus_bundle(manifest, _policy())
    store = FilesystemArtifactStore(tmp_path / "artifacts")
    persist_frozen_corpus_bundle(bundle, store)
    batches = _batches(records)
    evidence = build_recovery_run_evidence(
        run_id="run-fixture",
        attempt_id=None,
        result=_stopped_result(records),
        config=_config(),
        corpus_bundle=bundle,
        corpus_plan=_plan(),
        selected_mix_id="initial",
        records=records,
        batches=batches,
        teacher_router=_router(),
        parent_artifact_id=bundle.bundle_id,
    )
    repeated = build_recovery_run_evidence(
        run_id="run-fixture",
        attempt_id=None,
        result=_stopped_result(records),
        config=_config(),
        corpus_bundle=bundle,
        corpus_plan=_plan(),
        selected_mix_id="initial",
        records=records,
        batches=batches,
        teacher_router=_router(),
        parent_artifact_id=bundle.bundle_id,
    )

    assert evidence.payload == repeated.payload
    assert evidence.evidence_id == repeated.evidence_id
    persist_recovery_run_evidence(evidence, store)
    loaded = load_recovery_run_evidence(
        FilesystemArtifactStore(tmp_path / "artifacts"), evidence.artifact_ref
    )
    assert loaded.promotion_eligible is False
    assert loaded.metadata["result"]["status"] == RecoveryRunStatus.STOPPED_BUDGET.value
    assert loaded.metadata["scientific_promotion_implied"] is False
    for record in records:
        assert torch.equal(
            loaded.batches[record.record_id].input_ids,
            batches[record.record_id].input_ids,
        )


def test_recovery_evidence_rejects_record_batch_mix_and_representation_substitution(
    tmp_path,
) -> None:
    records = _records()
    bundle = build_frozen_corpus_bundle(
        CorpusService().freeze(records, _policy()), _policy()
    )
    store = FilesystemArtifactStore(tmp_path / "artifacts")
    persist_frozen_corpus_bundle(bundle, store)
    common = {
        "run_id": "run-fixture",
        "attempt_id": None,
        "result": _stopped_result(records),
        "config": _config(),
        "corpus_bundle": bundle,
        "corpus_plan": _plan(),
        "selected_mix_id": "initial",
        "records": records,
        "batches": _batches(records),
        "teacher_router": _router(),
        "parent_artifact_id": bundle.bundle_id,
    }

    substituted = replace(
        records[0],
        normalized_record={"messages": [{"role": "user", "content": "substitute"}]},
        content_hash=sha256_bytes(canonical_json_bytes(
            {"messages": [{"role": "user", "content": "substitute"}]}
        )),
    )
    with pytest.raises(ProvenanceViolation) as record_error:
        build_recovery_run_evidence(**{**common, "records": (substituted,) + records[1:]})
    assert record_error.value.code in {
        "provenance.evidence.mix_evidence",
        "provenance.evidence.record_substitution",
    }

    with pytest.raises(ProvenanceViolation) as batch_error:
        build_recovery_run_evidence(**{
            **common,
            "batches": {key: value for key, value in _batches(records).items() if key != records[0].record_id},
        })
    assert batch_error.value.code == "provenance.evidence.mix_substitution"

    invalid_result = replace(
        _stopped_result(records),
        completed_steps=1,
        forward_events=(
            BinaryForwardEvent(
                0, "recovery_training", (ActiveRepresentation.PROGRESSIVE,)
            ),
        ),
    )
    with pytest.raises(ProvenanceViolation) as representation_error:
        build_recovery_run_evidence(**{**common, "result": invalid_result})
    assert representation_error.value.code == "provenance.evidence.representation"
