from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from binary_llm.adapters import (
    ActiveRepresentation,
    BinaryLinear,
    BinaryLinearConfig,
    ConfiguredBinaryLinearFactory,
    TinyCausalLMConfig,
    TinyCausalModelAdapter,
    TrainingActivationMode,
    build_seeded_tiny_causal_lm,
)
from binary_llm.domain import (
    AmbiguityStatus,
    Budget,
    CheckpointBoundary,
    ScaleRung,
    ProgressiveTrainabilityArm,
    seed_ambiguity_register,
)
from binary_llm.math import (
    AnalyticalScaleGradient,
    OperatorPrecision,
    ProgressiveOperatorConfig,
    ProgressionScheduleConfig,
    ZeroSignRule,
    analytical_row_scales,
    progressive,
)
from binary_llm.orchestration import (
    CausalBatch,
    BudgetUsage,
    GateReport,
    GateResult,
    GateResultStatus,
    ProgressiveRunStatus,
    ProgressiveTrainerBackend,
    ProgressiveTrainerConfig,
    FilesystemArtifactStore,
    load_progressive_checkpoint,
    save_progressive_checkpoint,
    verify_progressive_failure_artifact,
    Stage1OptimizerConfig,
    apply_stage1_transition,
    build_stage1_transition,
    build_progressive_phase_plan,
)
from binary_llm.orchestration.corpus import PartitionRef


def _plan(phase_count: int):
    partitions = tuple(
        PartitionRef(
            phase_index=index,
            partition_id=f"partition-{index}",
            manifest_hash=f"{index + 1:064x}",
            record_ids=(f"record-{index}",),
            content_hashes=(f"{index + 101:064x}",),
            semantic_family_ids=(f"semantic-{index}",),
            split_family_ids=(f"split-{index}",),
            group_ids=(f"group-{index}",),
            token_count=index + 8,
            seed=19,
        )
        for index in range(20)
    )
    entry = next(
        item
        for item in seed_ambiguity_register("ambiguities", "v1").entries
        if item.ambiguity_id == "progressive.phase_index"
    )
    resolution = replace(
        entry,
        matched_protocol="matched-phase-index-screening",
        decision_rule="Select the finite continuation-qualified convention.",
        confidence_method="paired-bootstrap",
        required_floors=("finite", "continuation"),
        scale_scope=(ScaleRung.SMALL,),
        status=AmbiguityStatus.RESOLVED,
        selected_candidate="zero_based",
        evidence_refs=("phase-index-evidence",),
        rejected_candidates=("one_based",),
    )
    return build_progressive_phase_plan(
        partitions,
        ProgressionScheduleConfig("paper_exponential", "0..19"),
        resolution,
        scale_rung=ScaleRung.SMALL,
        test_phase_count=phase_count,
    )


def _model():
    adapter = TinyCausalModelAdapter()
    model = build_seeded_tiny_causal_lm(
        seed=23,
        config=TinyCausalLMConfig(
            vocab_size=13,
            hidden_size=4,
            intermediate_size=6,
            max_sequence_length=5,
        ),
    )
    adapter.replace_linears(
        model,
        ConfiguredBinaryLinearFactory(
            BinaryLinearConfig(
                scale_parameterization="positive_exp",
                training_activation_mode=TrainingActivationMode.EXPLICIT_ACTIVATION_TRANSFORM,
                progressive_operator=ProgressiveOperatorConfig(
                    precision=OperatorPrecision.FLOAT32,
                    analytical_scale_gradient=AnalyticalScaleGradient.DETACHED,
                ),
                zero_sign_rule=ZeroSignRule.POSITIVE,
            )
        ),
    )
    return adapter, model


def _trainer(
    arm: ProgressiveTrainabilityArm = ProgressiveTrainabilityArm.ALL_MODEL_PARAMETERS,
) -> ProgressiveTrainerBackend:
    return ProgressiveTrainerBackend(
        ProgressiveTrainerConfig(
            optimizer=Stage1OptimizerConfig(
                kind="adamw", learning_rate=0.01, weight_decay=0.0
            ),
            seed=29,
            trainability_arm=arm,
        )
    )


def _transition(model):
    transition = build_stage1_transition(
        model=model,
        source_checkpoint_id="stage1-complete",
        source_run_id="stage1-run",
    )
    return apply_stage1_transition(
        model,
        transition,
        expected_source_checkpoint_id="stage1-complete",
    )


def _batches(plan):
    source = (
        CausalBatch("train-a", torch.tensor([[1, 2, 3, 4]], dtype=torch.long)),
        CausalBatch("train-b", torch.tensor([[4, 3, 2, 1]], dtype=torch.long)),
    )
    return {phase.partition_hash: source for phase in plan.phases}


def _held_out():
    return (CausalBatch("held", torch.tensor([[2, 4, 1, 3]], dtype=torch.long)),)


def _capability(model, phase, representation):
    assert representation in (
        ActiveRepresentation.PROGRESSIVE,
        ActiveRepresentation.SIGN,
    )
    return {"token_accuracy": float(phase.phase_ordinal) / 10.0}


def _final_gate(evaluation):
    passed = evaluation.identity.representation is ActiveRepresentation.PROGRESSIVE
    result = GateResult(
        gate_id="continuation",
        status=GateResultStatus.PASS if passed else GateResultStatus.FAIL,
        observed=1.0 if passed else 0.0,
        threshold=1.0,
        evidence_refs=(evaluation.identity.identity_id,),
        evaluated_at="2026-01-01T00:00:00Z",
        category="capability",
        reason=None if passed else "sign view failed",
    )
    return GateReport(
        report_id=f"report-{evaluation.identity.representation.value}",
        gate_set_id="final-gates",
        results=(result,),
        evaluated_at="2026-01-01T00:00:00Z",
        immediate_stop_gate_ids=(),
    )


def test_fresh_progressive_run_requires_applied_transition() -> None:
    plan = _plan(1)
    adapter, model = _model()
    with pytest.raises(ValueError, match="verified applied"):
        _trainer().run(
            run_id="missing-transition",
            parent_checkpoint_id="stage1-complete",
            model=model,
            adapter=adapter,
            plan=plan,
            phase_batches=_batches(plan),
            held_out_batches=_held_out(),
            capability_evaluator=_capability,
        )


def test_progressive_backend_updates_dual_scale_state_and_emits_distinct_final_views(
    tmp_path,
) -> None:
    plan = _plan(1)
    adapter, model = _model()
    applied = _transition(model)
    learned_before = {
        name: module.dual_scale.learned.value.detach().clone()
        for name, module in model.named_modules()
        if isinstance(module, BinaryLinear)
    }

    result = _trainer().run(
        run_id="progressive-one",
        parent_checkpoint_id="stage1-complete",
        model=model,
        adapter=adapter,
        plan=plan,
        phase_batches=_batches(plan),
        held_out_batches=_held_out(),
        capability_evaluator=_capability,
        applied_transition=applied,
        final_gate_evaluator=_final_gate,
        checkpoint_directory=tmp_path,
    )

    assert result.status is ProgressiveRunStatus.COMPLETED
    assert result.completed_phase_count == 1
    assert result.optimizer_steps == 2
    assert len(result.phase_metrics) == len(result.checkpoints) == 1
    assert result.checkpoints[0].complete
    assert result.checkpoints[0].parent_checkpoint_id == "stage1-complete"
    assert result.checkpoints[0].stage1_parent_checkpoint_id == "stage1-complete"
    assert result.checkpoint_paths[0].is_file()
    metric = result.phase_metrics[0]
    assert metric.partition_hash == plan.phases[0].partition_hash
    assert metric.batch_ids == ("train-a", "train-b")
    assert metric.analytical_recomputations == 2
    assert metric.finite_state.is_finite
    assert metric.gradient_metrics.nonfinite_count == 0
    assert 0 <= metric.saturation_fraction <= 1
    assert 0 <= metric.sign_flip_rate <= 1
    assert metric.capability_metrics == {"token_accuracy": 0.1}

    modules = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, BinaryLinear)
    }
    assert any(
        not torch.equal(module.dual_scale.learned.value, learned_before[name])
        for name, module in modules.items()
    )
    for module in modules.values():
        latent = module.transformed_weight()
        analytical = analytical_row_scales(
            latent, gradient=AnalyticalScaleGradient.DETACHED
        )
        expected = (
            module.dual_scale.learned().unsqueeze(1)
            * analytical.unsqueeze(1)
            * progressive(
                latent / analytical.unsqueeze(1),
                plan.phases[0].progression_parameter,
                config=module.config.progressive_operator,
            )
        )
        torch.testing.assert_close(module.effective_weight(), expected)

    final = result.final_evidence
    assert final is not None
    assert final.progressive.identity.representation is ActiveRepresentation.PROGRESSIVE
    assert final.sign_substituted.identity.representation is ActiveRepresentation.SIGN
    assert (
        final.progressive.identity.identity_id
        != final.sign_substituted.identity.identity_id
    )
    assert not final.sign_export_eligible
    assert "sign substitution crossed" in " ".join(final.rejection_reasons)
    assert adapter.active_representation(model) is ActiveRepresentation.PROGRESSIVE


def test_existing_stop_request_takes_effect_only_after_the_next_complete_boundary() -> (
    None
):
    plan = _plan(2)
    adapter, model = _model()
    applied = _transition(model)

    result = _trainer().run(
        run_id="progressive-stop",
        parent_checkpoint_id="stage1-complete",
        model=model,
        adapter=adapter,
        plan=plan,
        phase_batches=_batches(plan),
        held_out_batches=_held_out(),
        capability_evaluator=_capability,
        applied_transition=applied,
        stop_requested=True,
    )

    assert result.status is ProgressiveRunStatus.STOPPED_GATE
    assert result.completed_phase_count == 1
    assert result.optimizer_steps == 2
    assert len(result.phase_metrics) == len(result.checkpoints) == 1
    assert result.checkpoints[0].complete
    assert result.checkpoints[0].schedule_state.next_schedule_index == 1
    assert result.final_evidence is None


def test_progressive_budget_never_publishes_partial_phase_and_resumes_cumulatively() -> (
    None
):
    plan = _plan(2)
    adapter, model = _model()
    applied = _transition(model)
    budget = Budget(6, 10, 100, 1.0, CheckpointBoundary.PROGRESSIVE_PHASE)
    first = _trainer().run(
        run_id="progressive-budget",
        parent_checkpoint_id="stage1-complete",
        model=model,
        adapter=adapter,
        plan=plan,
        phase_batches=_batches(plan),
        held_out_batches=_held_out(),
        capability_evaluator=_capability,
        applied_transition=applied,
        stop_requested=True,
        budget=budget,
        monotonic_clock=lambda: 0.0,
    )
    assert first.completed_phase_count == 1
    assert first.checkpoints[0].budget_usage is not None
    assert first.checkpoints[0].budget_usage.consumed_tokens == 6

    adapter, model = _model()
    remaining = {
        plan.phases[1].partition_hash: _batches(plan)[plan.phases[1].partition_hash]
    }
    resumed = _trainer().run(
        run_id="progressive-budget",
        parent_checkpoint_id="stage1-complete",
        model=model,
        adapter=adapter,
        plan=plan,
        phase_batches=remaining,
        held_out_batches=_held_out(),
        capability_evaluator=_capability,
        resume_checkpoint=first.checkpoints[0],
        budget=Budget(8, 10, 100, 1.0, CheckpointBoundary.PROGRESSIVE_PHASE),
        monotonic_clock=lambda: 0.0,
    )
    assert resumed.status is ProgressiveRunStatus.STOPPED_BUDGET
    assert resumed.completed_phase_count == 1
    assert resumed.incomplete_phase_ordinal == 2
    assert resumed.checkpoints == ()
    assert resumed.last_complete_checkpoint == first.checkpoints[0]
    assert resumed.budget_usage is not None
    assert resumed.budget_usage.consumed_tokens == 6
    assert resumed.budget_usage.interrupted_attempts == 0
    assert resumed.phase_metrics == ()


def test_progressive_resume_retains_stopped_partial_work_without_resuming_its_state() -> (
    None
):
    plan = _plan(2)
    adapter, model = _model()
    applied = _transition(model)
    first = _trainer().run(
        run_id="progressive-partial-accounting",
        parent_checkpoint_id="stage1-complete",
        model=model,
        adapter=adapter,
        plan=plan,
        phase_batches=_batches(plan),
        held_out_batches=_held_out(),
        capability_evaluator=_capability,
        applied_transition=applied,
        stop_requested=True,
        budget=Budget(100, 100, 100, 10.0, CheckpointBoundary.PROGRESSIVE_PHASE),
        monotonic_clock=lambda: 0.0,
    )
    checkpoint = first.checkpoints[0]
    stopped_usage = BudgetUsage(
        consumed_tokens=7,
        optimizer_steps=3,
        wall_seconds=4.0,
        accelerator_seconds=2.0,
        billable_cost=0.4,
        interrupted_attempts=1,
    )

    adapter, model = _model()
    resumed = _trainer().run(
        run_id="progressive-partial-accounting",
        parent_checkpoint_id="stage1-complete",
        model=model,
        adapter=adapter,
        plan=plan,
        phase_batches={
            plan.phases[1].partition_hash: _batches(plan)[plan.phases[1].partition_hash]
        },
        held_out_batches=_held_out(),
        capability_evaluator=_capability,
        resume_checkpoint=checkpoint,
        prior_budget_usage=stopped_usage,
        budget=Budget(10, 100, 100, 10.0, CheckpointBoundary.PROGRESSIVE_PHASE),
        monotonic_clock=lambda: 0.0,
    )

    assert resumed.status is ProgressiveRunStatus.STOPPED_BUDGET
    assert resumed.completed_phase_count == 1
    assert resumed.optimizer_steps == checkpoint.optimizer_step + 1
    assert resumed.checkpoints == ()
    assert resumed.checkpoint_paths == ()
    assert resumed.phase_metrics == ()
    assert resumed.last_complete_checkpoint is checkpoint
    assert resumed.incomplete_phase_ordinal == 2
    assert resumed.budget_usage == BudgetUsage(
        consumed_tokens=10,
        optimizer_steps=4,
        wall_seconds=4.0,
        accelerator_seconds=2.0,
        billable_cost=0.4,
        interrupted_attempts=2,
    )
    assert resumed.budget_failure is not None
    assert (
        resumed.budget_failure.context["usage"]
        == resumed.budget_usage.evidence_values()
    )


def test_progressive_pre_step_rejection_after_complete_phase_is_not_interrupted() -> (
    None
):
    plan = _plan(2)
    adapter, model = _model()
    applied = _transition(model)

    result = _trainer().run(
        run_id="progressive-pre-step-rejection",
        parent_checkpoint_id="stage1-complete",
        model=model,
        adapter=adapter,
        plan=plan,
        phase_batches=_batches(plan),
        held_out_batches=_held_out(),
        capability_evaluator=_capability,
        applied_transition=applied,
        budget=Budget(6, 100, 100, 10.0, CheckpointBoundary.PROGRESSIVE_PHASE),
        monotonic_clock=lambda: 0.0,
    )

    assert result.status is ProgressiveRunStatus.STOPPED_BUDGET
    assert result.completed_phase_count == 1
    assert result.optimizer_steps == 2
    assert len(result.checkpoints) == len(result.phase_metrics) == 1
    assert result.last_complete_checkpoint is result.checkpoints[0]
    assert result.incomplete_phase_ordinal == 2
    assert result.budget_crossing is not None
    assert result.budget_crossing.phase == "pre_step"
    assert result.budget_usage is not None
    assert result.budget_usage.interrupted_attempts == 0


@pytest.mark.parametrize(
    "prior_usage",
    (
        BudgetUsage(consumed_tokens=5, optimizer_steps=2),
        BudgetUsage(consumed_tokens=6, optimizer_steps=1),
    ),
)
def test_progressive_resume_rejects_budget_reconciliation_bypass(prior_usage) -> None:
    plan = _plan(2)
    adapter, model = _model()
    applied = _transition(model)
    first = _trainer().run(
        run_id="progressive-reconciliation",
        parent_checkpoint_id="stage1-complete",
        model=model,
        adapter=adapter,
        plan=plan,
        phase_batches=_batches(plan),
        held_out_batches=_held_out(),
        capability_evaluator=_capability,
        applied_transition=applied,
        stop_requested=True,
        budget=Budget(100, 100, 100, 10.0, CheckpointBoundary.PROGRESSIVE_PHASE),
        monotonic_clock=lambda: 0.0,
    )

    adapter, model = _model()
    with pytest.raises(ValueError, match="component-wise include checkpoint usage"):
        _trainer().run(
            run_id="progressive-reconciliation",
            parent_checkpoint_id="stage1-complete",
            model=model,
            adapter=adapter,
            plan=plan,
            phase_batches={
                plan.phases[1].partition_hash: _batches(plan)[
                    plan.phases[1].partition_hash
                ]
            },
            held_out_batches=_held_out(),
            capability_evaluator=_capability,
            resume_checkpoint=first.checkpoints[0],
            prior_budget_usage=prior_usage,
            budget=Budget(100, 100, 100, 10.0, CheckpointBoundary.PROGRESSIVE_PHASE),
            monotonic_clock=lambda: 0.0,
        )


def test_phase_two_resume_validates_immutable_stage1_root() -> None:
    plan = _plan(3)
    adapter, model = _model()
    phase_one = _trainer().run(
        run_id="progressive-root",
        parent_checkpoint_id="stage1-complete",
        model=model,
        adapter=adapter,
        plan=plan,
        phase_batches=_batches(plan),
        held_out_batches=_held_out(),
        capability_evaluator=_capability,
        applied_transition=_transition(model),
        stop_requested=True,
    )

    adapter, model = _model()
    phase_two = _trainer().run(
        run_id="progressive-root",
        parent_checkpoint_id="stage1-complete",
        model=model,
        adapter=adapter,
        plan=plan,
        phase_batches={
            phase.partition_hash: _batches(plan)[phase.partition_hash]
            for phase in plan.phases[1:]
        },
        held_out_batches=_held_out(),
        capability_evaluator=_capability,
        resume_checkpoint=phase_one.checkpoints[0],
        stop_requested=True,
    )
    checkpoint = phase_two.checkpoints[0]
    assert checkpoint.completed_phase_count == 2
    assert checkpoint.parent_checkpoint_id == phase_one.checkpoints[0].checkpoint_id
    assert checkpoint.stage1_parent_checkpoint_id == "stage1-complete"

    adapter, model = _model()
    remaining = {
        plan.phases[2].partition_hash: _batches(plan)[plan.phases[2].partition_hash]
    }
    with pytest.raises(ValueError, match="Stage 1 parent lineage"):
        _trainer().run(
            run_id="progressive-root",
            parent_checkpoint_id="different-stage1",
            model=model,
            adapter=adapter,
            plan=plan,
            phase_batches=remaining,
            held_out_batches=_held_out(),
            capability_evaluator=_capability,
            resume_checkpoint=checkpoint,
        )

    adapter, model = _model()
    resumed = _trainer().run(
        run_id="progressive-root",
        parent_checkpoint_id="stage1-complete",
        model=model,
        adapter=adapter,
        plan=plan,
        phase_batches=remaining,
        held_out_batches=_held_out(),
        capability_evaluator=_capability,
        resume_checkpoint=checkpoint,
    )
    assert resumed.status is ProgressiveRunStatus.COMPLETED
    assert resumed.checkpoints[0].stage1_parent_checkpoint_id == "stage1-complete"
    assert resumed.checkpoints[0].parent_checkpoint_id == checkpoint.checkpoint_id


def test_trainability_arms_inventory_and_updates_are_exact() -> None:
    plan = _plan(1)
    outcomes = {}
    for arm in ProgressiveTrainabilityArm:
        adapter, model = _model()
        result = _trainer(arm).run(
            run_id=f"trainability-{arm.value}",
            parent_checkpoint_id="stage1-complete",
            model=model,
            adapter=adapter,
            plan=plan,
            phase_batches=_batches(plan),
            held_out_batches=_held_out(),
            capability_evaluator=_capability,
            applied_transition=_transition(model),
        )
        inventory = result.parameter_inventory
        assert inventory is not None
        assert len(inventory.records) == len({record.name for record in inventory.records})
        assert all(
            record.optimizer_inclusion_count == (1 if record.trainable else 0)
            for record in inventory.records
        )
        folded = [
            record
            for record in inventory.records
            if record.role == "folded_stage1_input_scale"
        ]
        assert folded and all(not record.trainable and not record.changed for record in folded)
        outcomes[arm] = {record.role: record for record in inventory.records}

    all_parameters = outcomes[ProgressiveTrainabilityArm.ALL_MODEL_PARAMETERS]
    body_only = outcomes[ProgressiveTrainabilityArm.BINARY_BODY_ONLY]
    for role in ("embedding", "normalization", "language_model_head", "bias"):
        assert all_parameters[role].trainable
        assert all_parameters[role].changed
        assert not body_only[role].trainable
        assert not body_only[role].changed
    for role in ("binary_latent_weight", "learned_row_scale"):
        assert all_parameters[role].trainable and all_parameters[role].changed
        assert body_only[role].trainable and body_only[role].changed


def test_uninterrupted_and_disk_resumed_runs_are_exact(tmp_path) -> None:
    plan = _plan(2)
    adapter, uninterrupted_model = _model()
    uninterrupted = _trainer().run(
        run_id="exact-resume",
        parent_checkpoint_id="stage1-complete",
        model=uninterrupted_model,
        adapter=adapter,
        plan=plan,
        phase_batches=_batches(plan),
        held_out_batches=_held_out(),
        capability_evaluator=_capability,
        applied_transition=_transition(uninterrupted_model),
    )

    adapter, phase_one_model = _model()
    phase_one = _trainer().run(
        run_id="exact-resume",
        parent_checkpoint_id="stage1-complete",
        model=phase_one_model,
        adapter=adapter,
        plan=plan,
        phase_batches=_batches(plan),
        held_out_batches=_held_out(),
        capability_evaluator=_capability,
        applied_transition=_transition(phase_one_model),
        stop_requested=True,
    )
    path = save_progressive_checkpoint(tmp_path / "phase-one.pt", phase_one.checkpoints[0])
    loaded = load_progressive_checkpoint(path, plan=plan)
    adapter, resumed_model = _model()
    resumed = _trainer().run(
        run_id="exact-resume",
        parent_checkpoint_id="stage1-complete",
        model=resumed_model,
        adapter=adapter,
        plan=plan,
        phase_batches={plan.phases[1].partition_hash: _batches(plan)[plan.phases[1].partition_hash]},
        held_out_batches=_held_out(),
        capability_evaluator=_capability,
        resume_checkpoint=loaded,
    )

    expected = uninterrupted.checkpoints[-1]
    actual = resumed.checkpoints[-1]
    assert actual.state_hashes == expected.state_hashes
    assert actual.training_state.keys() == expected.training_state.keys()
    assert all(
        torch.equal(actual.training_state[name], expected.training_state[name])
        for name in actual.training_state
    )
    assert actual.optimizer_step == expected.optimizer_step
    assert actual.consumed_tokens == expected.consumed_tokens
    assert actual.schedule_state == expected.schedule_state
    assert actual.data_cursor == expected.data_cursor
    assert resumed.phase_metrics[0] == uninterrupted.phase_metrics[1]
    assert resumed.final_evidence == uninterrupted.final_evidence


def test_nonfinite_failure_is_diagnostic_only_and_durable(tmp_path, monkeypatch) -> None:
    plan = _plan(1)
    adapter, model = _model()
    original = torch.optim.AdamW.step

    def corrupt_after_step(optimizer, *args, **kwargs):
        result = original(optimizer, *args, **kwargs)
        with torch.no_grad():
            optimizer.param_groups[0]["params"][0].fill_(float("nan"))
        return result

    monkeypatch.setattr(torch.optim.AdamW, "step", corrupt_after_step)
    store = FilesystemArtifactStore(tmp_path / "artifacts")
    result = _trainer().run(
        run_id="nonfinite",
        attempt_id="attempt-1",
        parent_checkpoint_id="stage1-complete",
        model=model,
        adapter=adapter,
        plan=plan,
        phase_batches=_batches(plan),
        held_out_batches=_held_out(),
        capability_evaluator=_capability,
        applied_transition=_transition(model),
        artifact_store=store,
    )

    assert result.status is ProgressiveRunStatus.STOPPED_NONFINITE
    assert result.checkpoints == ()
    assert result.last_complete_checkpoint is None
    assert result.incomplete_phase_ordinal == 1
    assert result.final_evidence is None
    artifact = result.failure_artifact
    assert artifact is not None
    assert not artifact.resumable and not artifact.promotion_eligible
    verify_progressive_failure_artifact(artifact)
    assert store.resolve(artifact.artifact_ref.artifact_id) == artifact.payload
    with pytest.raises(ValueError, match="integrity"):
        verify_progressive_failure_artifact(
            replace(artifact, payload=artifact.payload[:-1] + b"x")
        )
