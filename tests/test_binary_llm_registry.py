from __future__ import annotations

from dataclasses import replace

import pytest

from binary_llm.domain import (
    ArtifactRef,
    BINARY_FAMILY,
    Budget,
    CheckpointBoundary,
    ExperimentManifest,
    FormatSpec,
    ManifestError,
    ModelIdentity,
    ReproductionError,
    ScaleRung,
    SealedBaselineRef,
    ToleranceDefinition,
)
from binary_llm.orchestration import (
    COMPLETE_CHECKPOINT_STATE_KEYS,
    AppendOnlyExperimentRegistry,
    AttemptExecutionContext,
    AttemptStatus,
    ReproductionOutcome,
    ReproductionProtocol,
    ResourceDelta,
    RunEvent,
    FilesystemArtifactStore,
    FilesystemRegistryJournal,
)


NOW = "2026-07-16T00:00:00Z"


def _artifact(
    artifact_id: str,
    *,
    kind: str = "checkpoint",
    parent: str | None = None,
    run_id: str | None = None,
    attempt_id: str | None = None,
    digest: str = "a" * 64,
) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=artifact_id,
        kind=kind,
        sha256=digest,
        bytes=10,
        media_type="application/octet-stream",
        parent_artifact_id=parent,
        producing_run_id=run_id,
        producing_attempt_id=attempt_id,
    )


def _manifest() -> ExperimentManifest:
    model = ModelIdentity(
        model_id="model",
        revision="revision",
        tokenizer_revision="tokenizer-revision",
        architecture="test",
        parameter_count=100,
        config_hash="config-hash",
        tokenizer_hashes=("tokenizer-hash",),
        template_hashes=("template-hash",),
        weight_hashes=("weight-hash",),
        tied_weights=False,
    )
    root = _artifact("root-checkpoint")
    baseline_artifact = _artifact("bf16-artifact", kind="baseline")
    baseline_output = _artifact("bf16-output", kind="baseline-output")
    baseline = SealedBaselineRef(
        baseline_id="bf16",
        role="bf16",
        model=model,
        artifact_refs=(baseline_artifact,),
        evaluator_revision="evaluator-revision",
        baseline_output_refs=(baseline_output,),
        sealed_at=NOW,
    )
    format_spec = FormatSpec(
        format_id="packed-v1",
        version="1",
        bit_order="lsb0",
        row_order="row_major",
        row_alignment=1,
        zero_sign_rule="positive",
        scale_dtype="float16",
        scale_endianness="little",
        metadata_encoding="json",
        allowed_representation_ids=("binary-v1",),
        temporary_expansion_limit_bytes=1024,
    )
    return ExperimentManifest(
        experiment_id="experiment",
        family=BINARY_FAMILY,
        hypothesis_refs=("hypothesis",),
        source_claim_refs=(),
        model=model,
        scale_rung=ScaleRung.SMALL,
        parent_checkpoint=root,
        baseline_refs=(baseline,),
        binary_scope=("attention.q",),
        operator_config={"operator": "progressive"},
        ambiguity_register_ref="ambiguities",
        stage1_config={"steps": 50},
        progressive_config={"phases": 20},
        recovery_config=None,
        teacher_refs=(),
        corpus_ref="corpus",
        partition_refs=("partition",),
        frozen_evaluation_refs=("frozen-eval",),
        evaluator_protocols=("protocol",),
        seed_set=(7,),
        determinism_policy={"data_order": "seeded"},
        budget=Budget(100, 50, 60, 1.0, CheckpointBoundary.PROGRESSIVE_PHASE),
        gate_set_ref="gates",
        tolerance_set_ref="tolerances",
        artifact_format=format_spec,
        device_protocol=None,
        source_revision="source-commit",
        dependency_lock_hash="lock-hash",
        container_digest="container-digest",
        command=("python", "run.py"),
        sanitized_environment={"offline": True},
        preregistered_at=NOW,
    )


def _context(*, tolerance_ids: tuple[str, ...] = ()) -> AttemptExecutionContext:
    operations = ("nondeterministic-kernel",) if tolerance_ids else ()
    return AttemptExecutionContext(
        source_commit="source-commit",
        clean_tree=True,
        hardware_inventory={"gpu": "test"},
        compiler_inventory={"cuda": "test"},
        seed=7,
        data_order_hash="data-order-hash",
        nondeterministic_operations=operations,
        preregistered_tolerance_ids=tolerance_ids,
    )


def _registry() -> AppendOnlyExperimentRegistry:
    counters: dict[str, int] = {}

    def allocate(kind: str) -> str:
        counters[kind] = counters.get(kind, 0) + 1
        return f"{kind}_{counters[kind]}"

    return AppendOnlyExperimentRegistry(id_allocator=allocate, clock=lambda: NOW)


def _produced_artifact(
    artifact_id: str,
    attempt,
    *,
    parent: str = "root-checkpoint",
    kind: str = "checkpoint",
    digest: str = "b" * 64,
) -> ArtifactRef:
    return _artifact(
        artifact_id,
        kind=kind,
        parent=parent,
        run_id=attempt.run_id,
        attempt_id=attempt.attempt_id,
        digest=digest,
    )


def test_unique_allocation_idempotent_writes_lineage_and_supersession():
    registry = _registry()
    experiment = registry.register(_manifest())
    assert registry.register(_manifest()) == experiment

    first = registry.begin_attempt(experiment, _context())
    second = registry.begin_attempt(experiment, _context())
    assert first.run_id != second.run_id
    assert first.attempt_id != second.attempt_id

    event = RunEvent("event-1", "metric", {"loss": 1.0}, NOW)
    original_ref = registry.append_event(first, event)
    assert registry.append_event(first, event) == original_ref
    with pytest.raises(ManifestError, match="different content"):
        registry.append_event(first, replace(event, payload={"loss": 2.0}))

    child = _produced_artifact("child", first)
    attachment_ref = registry.attach_artifact(first, child)
    assert registry.attach_artifact(first, child) == attachment_ref

    sideways = _produced_artifact("sideways", second, parent="child")
    with pytest.raises(ManifestError, match="sideways"):
        registry.attach_artifact(second, sideways)

    correction = RunEvent("event-2", "metric-correction", {"loss": 0.9}, NOW)
    correction_ref = registry.append_event(first, correction)
    supersession = registry.supersede(
        original_ref,
        correction_ref,
        reason="corrected evaluator input",
        supersession_id="supersession-1",
        created_at=NOW,
    )
    assert registry.supersede(
        original_ref,
        correction_ref,
        reason="corrected evaluator input",
    ) == supersession
    assert registry.events(first) == (event, correction)


def test_complete_checkpoint_resume_retention_and_cumulative_resource_ledger():
    registry = _registry()
    experiment = registry.register(_manifest())
    first = registry.begin_attempt(experiment, _context())

    partial = _produced_artifact("partial-checkpoint", first)
    registry.publish_checkpoint(
        first, partial, phase=1, complete=False, state_hashes={}
    )
    with pytest.raises(ValueError, match="missing state hashes"):
        registry.publish_checkpoint(
            first,
            _produced_artifact("invalid-complete", first),
            phase=1,
            complete=True,
            state_hashes={"training_state": "hash"},
        )
    assert "invalid-complete" not in {item.artifact_id for item in registry.artifacts()}

    complete = _produced_artifact("complete-checkpoint", first)
    state_hashes = {name: f"{name}-hash" for name in COMPLETE_CHECKPOINT_STATE_KEYS}
    registry.publish_checkpoint(
        first, complete, phase=1, complete=True, state_hashes=state_hashes
    )
    registry.record_resources(
        first,
        ResourceDelta("ledger-1", first.attempt_id, 10.0, 8.0, 100, 5, 1.25, "USD", NOW),
    )
    registry.finish_attempt(first, AttemptStatus.INTERRUPTED, failure_label="worker_loss")

    with pytest.raises(ManifestError, match="complete checkpoint"):
        registry.begin_attempt(
            experiment, _context(), resume_checkpoint_id="partial-checkpoint"
        )
    resumed = registry.begin_attempt(
        experiment, _context(), resume_checkpoint_id="complete-checkpoint"
    )
    assert resumed.run_id == first.run_id
    assert resumed.attempt_id != first.attempt_id
    assert registry.attempts()[1].resume_checkpoint_id == "complete-checkpoint"
    assert registry.attempts()[1].current_phase == 1

    registry.record_resources(
        resumed,
        ResourceDelta("ledger-2", resumed.attempt_id, 4.0, 3.0, 40, 2, 0.75, "USD", NOW),
    )
    registry.finish_attempt(resumed, AttemptStatus.FAILED, failure_label="numerical_failure")

    ledger = registry.cumulative_resource_ledger(resumed)
    assert ledger.wall_seconds == 14.0
    assert ledger.accelerator_seconds == 11.0
    assert ledger.consumed_tokens == 140
    assert ledger.optimizer_steps == 7
    assert ledger.billable_cost == 2.0
    assert ledger.interrupted_attempts == 1
    assert ledger.attempt_ids == (first.attempt_id, resumed.attempt_id)
    assert [item.status for item in registry.terminal_statuses()] == [
        AttemptStatus.INTERRUPTED,
        AttemptStatus.FAILED,
    ]


def test_reproduction_checks_inputs_before_execution_and_blocks_failed_results():
    registry = _registry()
    experiment = registry.register(_manifest())
    attempt = registry.begin_attempt(experiment, _context(tolerance_ids=("loss-tolerance",)))
    protocol = ReproductionProtocol(
        protocol_id="reproduce-1",
        original_attempt_id=attempt.attempt_id,
        expected_input_hashes={"manifest": "manifest-hash", "dataset": "dataset-hash"},
        expected_artifact_hashes={"checkpoint": "checkpoint-hash"},
        expected_metrics={"held_out_loss": 1.0},
        tolerances=(
            ToleranceDefinition(
                "loss-tolerance", "held_out_loss", "reproduction", 0.1, None, False
            ),
        ),
        preregistered_at=NOW,
    )
    calls: list[str] = []

    with pytest.raises(ReproductionError) as raised:
        registry.validate_reproduction(
            protocol,
            {"manifest": "changed", "dataset": "dataset-hash"},
            lambda: calls.append("executed") or ReproductionOutcome({}, {}),
            validation_id="input-drift",
        )
    assert raised.value.code == "reproduction.input_hash_mismatch"
    assert calls == []
    assert registry.reproductions()[0].executed is False

    failed = registry.validate_reproduction(
        protocol,
        {"manifest": "manifest-hash", "dataset": "dataset-hash"},
        lambda: ReproductionOutcome(
            {"checkpoint": "different-checkpoint"}, {"held_out_loss": 1.2}
        ),
        validation_id="failed-rerun",
    )
    assert failed.executed and not failed.passed
    assert {item["kind"] for item in failed.mismatches} == {"artifact_hash", "metric"}
    assert not registry.release_use_allowed(attempt)

    passed = registry.validate_reproduction(
        protocol,
        {"manifest": "manifest-hash", "dataset": "dataset-hash"},
        lambda: ReproductionOutcome(
            {"checkpoint": "checkpoint-hash"}, {"held_out_loss": 1.05}
        ),
        validation_id="passing-rerun",
    )
    assert passed.passed
    registry.supersede(
        registry.record_ref("reproduction-validation", "failed-rerun"),
        registry.record_ref("reproduction-validation", "passing-rerun"),
        reason="resolved with matching rerun",
        supersession_id="resolve-reproduction",
        created_at=NOW,
    )
    assert registry.release_use_allowed(attempt)


def test_manifest_and_attempt_identity_conflicts_fail_closed():
    registry = _registry()
    experiment = registry.register(_manifest())

    with pytest.raises(ManifestError, match="different manifest"):
        registry.register(replace(_manifest(), source_revision="different"))

    registry.begin_attempt(experiment, _context(), run_id="run-fixed", attempt_id="attempt-fixed")
    with pytest.raises(ManifestError, match="already been allocated"):
        registry.begin_attempt(experiment, _context(), run_id="run-fixed")
    with pytest.raises(ManifestError, match="already been allocated"):
        registry.begin_attempt(experiment, _context(), attempt_id="attempt-fixed")


def test_durable_journal_replays_all_record_kinds_and_continues(tmp_path):
    journal_root = tmp_path / "journal"
    store = FilesystemArtifactStore(tmp_path / "artifacts")
    registry = AppendOnlyExperimentRegistry(
        journal=FilesystemRegistryJournal(journal_root),
        artifact_store=store,
        id_allocator=lambda kind: f"{kind}-allocated",
        clock=lambda: NOW,
    )
    experiment = registry.register(_manifest())
    attempt = registry.begin_attempt(
        experiment, _context(tolerance_ids=("loss-tolerance",)),
        run_id="run-durable", attempt_id="attempt-durable",
    )
    event = RunEvent("event-durable", "metric", {"loss": 1.0}, NOW)
    event_ref = registry.append_event(attempt, event)
    payload = b"durable checkpoint"
    artifact = replace(
        _produced_artifact("checkpoint-durable", attempt),
        sha256=__import__("hashlib").sha256(payload).hexdigest(),
        bytes=len(payload),
    )
    registry.publish_checkpoint(
        attempt,
        artifact,
        phase=2,
        complete=True,
        state_hashes={name: f"{name}-hash" for name in COMPLETE_CHECKPOINT_STATE_KEYS},
        payload=payload,
    )
    registry.record_resources(
        attempt,
        ResourceDelta("ledger-durable", attempt.attempt_id, 2.0, 1.0, 20, 2, 0.5, "USD", NOW),
    )
    correction = RunEvent("event-correction", "metric", {"loss": 0.9}, NOW)
    correction_ref = registry.append_event(attempt, correction)
    registry.supersede(
        event_ref, correction_ref, reason="correction",
        supersession_id="supersession-durable", created_at=NOW,
    )
    protocol = ReproductionProtocol(
        "protocol-durable", attempt.attempt_id, {"input": "hash"}, {},
        {"loss": 1.0},
        (ToleranceDefinition("loss-tolerance", "loss", "reproduction", 0.1, None, False),),
        NOW,
    )
    registry.validate_reproduction(
        protocol, {"input": "hash"},
        lambda: ReproductionOutcome({}, {"loss": 1.0}),
        validation_id="validation-durable", validated_at=NOW,
    )
    registry.finish_attempt(
        attempt, AttemptStatus.INTERRUPTED, failure_label="worker_loss", ended_at=NOW
    )

    reopened = AppendOnlyExperimentRegistry(
        journal=FilesystemRegistryJournal(journal_root),
        artifact_store=FilesystemArtifactStore(tmp_path / "artifacts"),
        id_allocator=lambda kind: f"{kind}-continued",
        clock=lambda: NOW,
    )
    assert reopened.register(_manifest()) == experiment
    assert reopened.events(attempt) == (event, correction)
    assert reopened.checkpoints()[0].checkpoint_id == artifact.artifact_id
    assert reopened.cumulative_resource_ledger(attempt).consumed_tokens == 20
    assert reopened.reproductions()[0].passed
    assert store.resolve(artifact.artifact_id, artifact) == payload
    resumed = reopened.begin_attempt(
        experiment, _context(), resume_checkpoint_id=artifact.artifact_id,
        attempt_id="attempt-continued",
    )
    assert resumed.run_id == attempt.run_id
    assert len(tuple((journal_root / "entries").glob("*.json"))) >= 1


@pytest.mark.parametrize("mutation", ["tamper", "truncate", "gap", "reorder", "unknown"])
def test_durable_journal_corruption_fails_closed(tmp_path, mutation):
    from binary_llm.domain import RegistryJournalError, canonical_json_bytes

    root = tmp_path / "journal"
    registry = AppendOnlyExperimentRegistry(
        journal=FilesystemRegistryJournal(root), clock=lambda: NOW
    )
    registry.register(_manifest())
    paths = sorted((root / "entries").glob("*.json"))
    target = paths[-1]
    if mutation == "tamper":
        target.write_bytes(target.read_bytes().replace(b'"kind":"artifact"', b'"kind":"artifacX"'))
    elif mutation == "truncate":
        target.write_bytes(target.read_bytes()[:20])
    elif mutation == "gap":
        target.rename(target.with_name("00000000000000000999.json"))
    elif mutation == "reorder":
        target.write_bytes(target.read_bytes().replace(
            f'"sequence":{len(paths) - 1}'.encode(), b'"sequence":0'
        ))
    else:
        data = __import__("json").loads(target.read_text())
        body = {**data, "kind": "unknown-kind"}
        body.pop("entry_hash")
        data = {
            **body,
            "entry_hash": __import__("hashlib").sha256(canonical_json_bytes(body)).hexdigest(),
        }
        target.write_bytes(canonical_json_bytes(data))
    with pytest.raises(RegistryJournalError):
        AppendOnlyExperimentRegistry(journal=FilesystemRegistryJournal(root))


def test_two_registry_instances_exact_retry_and_conflict(tmp_path):
    root = tmp_path / "journal"
    first = AppendOnlyExperimentRegistry(journal=FilesystemRegistryJournal(root))
    second = AppendOnlyExperimentRegistry(journal=FilesystemRegistryJournal(root))
    assert first.register(_manifest()) == second.register(_manifest())
    with pytest.raises(ManifestError) as raised:
        second.register(replace(_manifest(), source_revision="different"))
    assert raised.value.code == "manifest.registry_experiment_conflict"


def test_registry_writer_lock_contention_and_stale_diagnosis(tmp_path):
    from binary_llm.domain import RegistryJournalError

    root = tmp_path / "journal"
    journal = FilesystemRegistryJournal(root, stale_lock_seconds=60)
    root.mkdir(parents=True)
    journal.lock_path.write_bytes(b"busy")
    with pytest.raises(RegistryJournalError) as raised:
        AppendOnlyExperimentRegistry(journal=journal).register(_manifest())
    assert raised.value.code == "registry_journal.writer_busy"

    stale = FilesystemRegistryJournal(root, stale_lock_seconds=0)
    with pytest.raises(RegistryJournalError) as raised:
        AppendOnlyExperimentRegistry(journal=stale).register(_manifest())
    assert raised.value.code == "registry_journal.stale_lock"
