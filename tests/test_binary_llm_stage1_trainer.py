from __future__ import annotations

import math

import torch

from binary_llm.adapters import (
    BinaryLinear,
    BinaryLinearConfig,
    ConfiguredBinaryLinearFactory,
    TinyCausalLMConfig,
    TinyCausalModelAdapter,
    TrainingActivationMode,
    build_seeded_tiny_causal_lm,
)
from binary_llm.domain import Budget, CheckpointBoundary
from binary_llm.math import (
    AnalyticalScaleGradient,
    OperatorPrecision,
    ProgressiveOperatorConfig,
    ZeroSignRule,
)
from binary_llm.orchestration import (
    REFERENCE_STAGE1_STEPS,
    CausalBatch,
    Stage1OptimizerConfig,
    Stage1RunStatus,
    Stage1TrainerBackend,
    Stage1TrainerConfig,
)


def _converted_model(seed: int = 31):
    adapter = TinyCausalModelAdapter()
    model = build_seeded_tiny_causal_lm(
        seed=seed,
        config=TinyCausalLMConfig(
            vocab_size=13,
            hidden_size=4,
            intermediate_size=6,
            max_sequence_length=5,
        ),
    )
    config = BinaryLinearConfig(
        scale_parameterization="positive_exp",
        training_activation_mode=TrainingActivationMode.EXPLICIT_ACTIVATION_TRANSFORM,
        progressive_operator=ProgressiveOperatorConfig(
            precision=OperatorPrecision.FLOAT32,
            analytical_scale_gradient=AnalyticalScaleGradient.DETACHED,
        ),
        zero_sign_rule=ZeroSignRule.POSITIVE,
    )
    adapter.replace_linears(model, ConfiguredBinaryLinearFactory(config))
    return adapter, model


def _trainer() -> Stage1TrainerBackend:
    return Stage1TrainerBackend(
        Stage1TrainerConfig(
            optimizer=Stage1OptimizerConfig(
                kind="adamw",
                learning_rate=0.02,
                weight_decay=0.0,
            ),
            seed=7,
        )
    )


def _batches() -> tuple[tuple[CausalBatch, ...], tuple[CausalBatch, ...]]:
    train = (
        CausalBatch("train-a", torch.tensor([[1, 2, 3, 4]], dtype=torch.long)),
        CausalBatch("train-b", torch.tensor([[4, 3, 2, 1]], dtype=torch.long)),
    )
    held_out = (
        CausalBatch("held-out", torch.tensor([[2, 4, 1, 3]], dtype=torch.long)),
    )
    return train, held_out


def test_stage1_defaults_to_exact_reference_steps_and_restricts_test_override() -> None:
    assert Stage1TrainerConfig.resolve_steps() == REFERENCE_STAGE1_STEPS == 50
    assert Stage1TrainerConfig.resolve_steps(3) == 3
    for invalid in (0, 50, -1, True):
        try:
            Stage1TrainerConfig.resolve_steps(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid test override: {invalid!r}")


def test_stage1_short_run_optimizes_only_input_scales_and_emits_control_evidence() -> None:
    adapter, model = _converted_model()
    train, held_out = _batches()
    frozen_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not name.endswith("input_scale.raw_scale")
    }
    scales_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name.endswith("input_scale.raw_scale")
    }

    result = _trainer().run(
        run_id="short-stage1",
        model=model,
        adapter=adapter,
        train_batches=train,
        held_out_batches=held_out,
        test_step_override=3,
    )

    assert result.status is Stage1RunStatus.COMPLETED
    assert result.requested_steps == result.completed_steps == 3
    assert result.dense_parameters_unchanged
    assert not result.stopped_nonfinite and result.failure_batch is None
    assert result.diagnostics.finite_state.is_finite
    assert all(
        math.isfinite(value)
        for value in (
            result.diagnostics.initial_loss,
            result.diagnostics.final_loss,
            result.diagnostics.held_out_loss,
            result.diagnostics.binary_reconstruction_error,
        )
    )

    assert result.diagnostics.scale_min is not None
    assert result.diagnostics.scale_max is not None
    assert len(result.gradient_diagnostics) == len(adapter.declared_linear_roles)
    assert all(item.nonfinite_count == 0 for item in result.gradient_diagnostics)
    assert tuple(result.comparison.control.train_batch_ids) == ("train-a", "train-b")
    assert tuple(result.comparison.control.held_out_batch_ids) == ("held-out",)
    assert result.comparison.control.planned_steps == 3
    assert result.comparison.control.performed_steps == 0
    assert result.comparison.control.diagnostics.initial_loss == result.comparison.control.diagnostics.final_loss
    assert result.comparison.differing_factor == "binary_aware_initialization"
    assert "optimizer_step_budget" in result.comparison.matched_fields
    assert result.checkpoint.checkpoint_id == "short-stage1:stage1:3:complete"
    assert result.checkpoint.completed_steps == 3
    assert result.checkpoint.optimizer_state["state"]
    assert result.transition is not None
    assert result.transition.source_checkpoint_id == result.checkpoint.checkpoint_id

    current = dict(model.named_parameters())
    for name, expected in frozen_before.items():
        torch.testing.assert_close(current[name], expected, rtol=0, atol=0)
    assert any(
        not torch.equal(current[name], before) for name, before in scales_before.items()
    )
    for _, module in model.named_modules():
        if isinstance(module, BinaryLinear):
            assert module.input_scale.raw_scale.shape == (module.in_features,)
            assert module.input_scale.raw_scale.requires_grad
            assert not module.weight.requires_grad


def test_stage1_nonfinite_state_stops_and_preserves_checkpoint_and_batch() -> None:
    adapter, model = _converted_model()
    train, held_out = _batches()
    first_binary = next(
        module for module in model.modules() if isinstance(module, BinaryLinear)
    )
    with torch.no_grad():
        first_binary.input_scale.raw_scale[0] = float("inf")

    result = _trainer().run(
        run_id="failed-stage1",
        model=model,
        adapter=adapter,
        train_batches=train,
        held_out_batches=held_out,
        test_step_override=2,
    )

    assert result.status is Stage1RunStatus.STOPPED_NONFINITE
    assert result.completed_steps == 0
    assert result.failure_batch is not None
    assert result.failure_batch.batch_id == "train-a"
    torch.testing.assert_close(result.failure_batch.input_ids, train[0].input_ids)
    assert result.checkpoint.checkpoint_id == "failed-stage1:stage1:0:nonfinite"
    assert result.checkpoint.completed_steps == 0
    assert result.transition is None
    assert not result.diagnostics.finite_state.is_finite
    assert result.diagnostics.finite_state.total_nonfinite_count > 0
    assert any(
        name.startswith(("scale.", "loss.", "transformed_weight."))
        for name in result.diagnostics.finite_state.failing_fields
    )
    saved_scale = result.checkpoint.model_state[
        "block.attention.q_proj.input_scale.raw_scale"
    ]
    assert torch.isinf(saved_scale[0])


def test_stage1_budget_rejects_before_step_and_stops_after_wall_crossing() -> None:
    adapter, model = _converted_model()
    train, held_out = _batches()
    token_budget = Budget(2, 10, 10, 1.0, CheckpointBoundary.OPTIMIZER_STEP)
    rejected = _trainer().run(
        run_id="token-budget",
        model=model,
        adapter=adapter,
        train_batches=train,
        held_out_batches=held_out,
        test_step_override=2,
        budget=token_budget,
        monotonic_clock=lambda: 0.0,
    )
    assert rejected.status is Stage1RunStatus.STOPPED_BUDGET
    assert rejected.completed_steps == 0
    assert rejected.transition is None
    assert rejected.budget_crossing is not None
    assert rejected.budget_crossing.dimension.value == "tokens"
    assert rejected.budget_failure is not None
    assert rejected.budget_failure.context["phase"] == "pre_step"

    adapter, model = _converted_model()
    times = iter((0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0))
    wall_budget = Budget(100, 10, 1, 1.0, CheckpointBoundary.OPTIMIZER_STEP)
    stopped = _trainer().run(
        run_id="wall-budget",
        model=model,
        adapter=adapter,
        train_batches=train,
        held_out_batches=held_out,
        test_step_override=2,
        budget=wall_budget,
        monotonic_clock=lambda: next(times),
    )
    assert stopped.status is Stage1RunStatus.STOPPED_BUDGET
    assert stopped.completed_steps == 1
    assert stopped.transition is None
    assert stopped.checkpoint.completed_steps == 1
    assert stopped.budget_usage is not None
    assert stopped.budget_usage.consumed_tokens == 3
