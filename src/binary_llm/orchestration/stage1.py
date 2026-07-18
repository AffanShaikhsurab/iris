"""Fail-closed PyTorch backend for BinaryLLM binary-aware initialization."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Callable, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from binary_llm.adapters import ActiveRepresentation, BinaryLinear, ModelAdapter
from binary_llm.domain import Budget, BudgetExceeded
from binary_llm.math import (
    FiniteStateDiagnostics,
    ScaleRole,
    Stage1Diagnostics,
    binary_reconstruction_error,
    diagnose_finite_state,
    named_trainable_scales,
    stage1_diagnostics,
)
from .budgets import (
    BudgetCrossing,
    BudgetMonitor,
    BudgetUsage,
    causal_training_tokens,
)
from .store import FilesystemArtifactStore
from .transition import (
    Stage1Transition,
    build_stage1_transition,
    persist_stage1_transition,
)

REFERENCE_STAGE1_STEPS = 50


class Stage1OptimizerKind(StrEnum):
    ADAMW = "adamw"
    SGD = "sgd"


class Stage1RunStatus(StrEnum):
    COMPLETED = "completed"
    STOPPED_NONFINITE = "stopped_nonfinite"
    STOPPED_BUDGET = "stopped_budget"


@dataclass(frozen=True, slots=True)
class Stage1OptimizerConfig:
    kind: Stage1OptimizerKind | str
    learning_rate: float
    weight_decay: float
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    momentum: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", Stage1OptimizerKind(self.kind))
        for name in ("learning_rate", "weight_decay", "beta1", "beta2", "epsilon", "momentum"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.epsilon <= 0:
            raise ValueError("learning_rate and epsilon must be positive; weight_decay must be non-negative")
        if not 0 <= self.momentum < 1:
            raise ValueError("momentum must be in [0, 1)")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("Adam betas must be in [0, 1)")


@dataclass(frozen=True, slots=True)
class Stage1TrainerConfig:
    optimizer: Stage1OptimizerConfig
    seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.optimizer, Stage1OptimizerConfig):
            raise TypeError("optimizer must be Stage1OptimizerConfig")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")

    @staticmethod
    def resolve_steps(test_step_override: int | None = None) -> int:
        if test_step_override is None:
            return REFERENCE_STAGE1_STEPS
        if (
            not isinstance(test_step_override, int)
            or isinstance(test_step_override, bool)
            or not 1 <= test_step_override < REFERENCE_STAGE1_STEPS
        ):
            raise ValueError("test_step_override must be between 1 and 49")
        return test_step_override


@dataclass(frozen=True, slots=True)
class CausalBatch:
    batch_id: str
    input_ids: Tensor
    labels: Tensor | None = None

    def __post_init__(self) -> None:
        if not self.batch_id:
            raise ValueError("batch_id must be non-empty")
        if self.input_ids.ndim != 2 or self.input_ids.shape[1] < 2:
            raise ValueError("input_ids must have shape [batch, sequence>=2]")
        if self.input_ids.dtype != torch.long:
            raise TypeError("input_ids must use torch.long")
        if self.labels is not None:
            if self.labels.shape != self.input_ids.shape or self.labels.dtype != torch.long:
                raise ValueError("labels must be torch.long and match input_ids shape")

    def snapshot(self) -> CausalBatch:
        return CausalBatch(
            self.batch_id,
            self.input_ids.detach().cpu().clone(),
            None if self.labels is None else self.labels.detach().cpu().clone(),
        )


@dataclass(frozen=True, slots=True)
class Stage1GradientDiagnostics:
    name: str
    element_count: int
    nonfinite_count: int
    finite_min: float | None
    finite_max: float | None
    l2_norm: float | None


@dataclass(frozen=True, slots=True)
class Stage1Checkpoint:
    checkpoint_id: str
    completed_steps: int
    model_state: Mapping[str, Tensor]
    optimizer_state: Mapping[str, object]
    cpu_rng_state: Tensor


@dataclass(frozen=True, slots=True)
class NoInitializationControl:
    diagnostics: Stage1Diagnostics
    planned_steps: int
    performed_steps: int
    train_batch_ids: tuple[str, ...]
    held_out_batch_ids: tuple[str, ...]
    seed: int


@dataclass(frozen=True, slots=True)
class Stage1ControlComparison:
    control: NoInitializationControl
    final_loss_delta: float
    held_out_loss_delta: float
    reconstruction_error_delta: float
    matched_fields: tuple[str, ...]
    differing_factor: str = "binary_aware_initialization"


@dataclass(frozen=True, slots=True)
class Stage1RunResult:
    status: Stage1RunStatus
    requested_steps: int
    completed_steps: int
    diagnostics: Stage1Diagnostics
    gradient_diagnostics: tuple[Stage1GradientDiagnostics, ...]
    checkpoint: Stage1Checkpoint
    comparison: Stage1ControlComparison
    trainable_scale_names: tuple[str, ...]
    frozen_parameter_names: tuple[str, ...]
    dense_parameters_unchanged: bool
    transition: Stage1Transition | None = None
    failure_batch: CausalBatch | None = None
    budget_usage: BudgetUsage | None = None
    budget_crossing: BudgetCrossing | None = None
    budget_failure: BudgetExceeded | None = None

    @property
    def stopped_nonfinite(self) -> bool:
        return self.status is Stage1RunStatus.STOPPED_NONFINITE


def causal_language_model_loss(model: nn.Module, batch: CausalBatch) -> Tensor:
    """Compute end-to-end next-token cross entropy without repairing labels."""

    logits = model(batch.input_ids)
    if logits.ndim != 3 or logits.shape[:2] != batch.input_ids.shape:
        raise ValueError("causal model must return [batch, sequence, vocabulary] logits")
    labels = batch.input_ids if batch.labels is None else batch.labels
    return nn.functional.cross_entropy(
        logits[:, :-1, :].contiguous().view(-1, logits.shape[-1]),
        labels[:, 1:].contiguous().view(-1),
        ignore_index=-100,
    )


def _binary_modules(model: nn.Module) -> tuple[tuple[str, BinaryLinear], ...]:
    modules = tuple(
        (name, module) for name, module in model.named_modules() if isinstance(module, BinaryLinear)
    )
    if not modules:
        raise ValueError("Stage 1 requires at least one BinaryLinear")
    return modules


def _operator_state(model: nn.Module) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    scales: dict[str, Tensor] = {}
    transformed: dict[str, Tensor] = {}
    for name, module in _binary_modules(model):
        scales[name] = module.input_scale()
        transformed[name] = module.transformed_weight()
    return scales, transformed


def _gradients(model: nn.Module) -> dict[str, Tensor]:
    return {
        item.name: item.parameter.grad
        for item in named_trainable_scales(model)
        if item.parameter.grad is not None
    }


def _reconstruction_error(model: nn.Module) -> Tensor:
    references: list[Tensor] = []
    candidates: list[Tensor] = []
    for _, module in _binary_modules(model):
        references.append(module.dense_reference_weight.reshape(-1))
        candidates.append(module.effective_weight().reshape(-1))
    return binary_reconstruction_error(torch.cat(references), torch.cat(candidates))


def _average_loss(model: nn.Module, batches: Sequence[CausalBatch]) -> Tensor:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        losses = torch.stack([causal_language_model_loss(model, batch) for batch in batches])
    model.train(was_training)
    return losses.mean()


def _gradient_diagnostics(gradients: Mapping[str, Tensor]) -> tuple[Stage1GradientDiagnostics, ...]:
    reports: list[Stage1GradientDiagnostics] = []
    for name in sorted(gradients):
        gradient = gradients[name].detach()
        finite = gradient[torch.isfinite(gradient)]
        reports.append(
            Stage1GradientDiagnostics(
                name=name,
                element_count=gradient.numel(),
                nonfinite_count=int((~torch.isfinite(gradient)).sum().item()),
                finite_min=None if finite.numel() == 0 else float(finite.min().item()),
                finite_max=None if finite.numel() == 0 else float(finite.max().item()),
                l2_norm=None if finite.numel() != gradient.numel() else float(torch.linalg.vector_norm(gradient).item()),
            )
        )
    return tuple(reports)


def _make_optimizer(config: Stage1OptimizerConfig, parameters: Sequence[nn.Parameter]) -> Optimizer:
    if config.kind is Stage1OptimizerKind.ADAMW:
        return torch.optim.AdamW(
            parameters,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.epsilon,
            weight_decay=config.weight_decay,
        )
    return torch.optim.SGD(
        parameters,
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )


def _checkpoint(
    run_id: str,
    suffix: str,
    completed_steps: int,
    model: nn.Module,
    optimizer: Optimizer,
) -> Stage1Checkpoint:
    return Stage1Checkpoint(
        checkpoint_id=f"{run_id}:stage1:{completed_steps}:{suffix}",
        completed_steps=completed_steps,
        model_state={name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
        optimizer_state=copy.deepcopy(optimizer.state_dict()),
        cpu_rng_state=torch.random.get_rng_state().clone(),
    )


def _build_diagnostics(
    model: nn.Module,
    train_batches: Sequence[CausalBatch],
    held_out_batches: Sequence[CausalBatch],
    initial_loss: Tensor,
    gradients: Mapping[str, Tensor],
) -> Stage1Diagnostics:
    scales, transformed = _operator_state(model)
    return stage1_diagnostics(
        scales=scales,
        transformed_weights=transformed,
        initial_loss=initial_loss,
        final_loss=_average_loss(model, train_batches),
        held_out_loss=_average_loss(model, held_out_batches),
        binary_reconstruction_error=_reconstruction_error(model),
        gradients=gradients,
    )


def _control(
    model: nn.Module,
    train_batches: Sequence[CausalBatch],
    held_out_batches: Sequence[CausalBatch],
    requested_steps: int,
    seed: int,
) -> NoInitializationControl:
    initial_loss = _average_loss(model, train_batches)
    return NoInitializationControl(
        diagnostics=_build_diagnostics(
            model, train_batches, held_out_batches, initial_loss, gradients={}
        ),
        planned_steps=requested_steps,
        performed_steps=0,
        train_batch_ids=tuple(batch.batch_id for batch in train_batches),
        held_out_batch_ids=tuple(batch.batch_id for batch in held_out_batches),
        seed=seed,
    )


def _finite_state(
    model: nn.Module,
    loss: Tensor,
    gradients: Mapping[str, Tensor],
) -> FiniteStateDiagnostics:
    scales, transformed = _operator_state(model)
    reconstruction = _reconstruction_error(model)
    return diagnose_finite_state(
        scales=scales,
        transformed_weights=transformed,
        losses={"causal": loss},
        reconstruction_errors={"binary": reconstruction},
        gradients=gradients,
    )


def _comparison(
    diagnostics: Stage1Diagnostics,
    control: NoInitializationControl,
) -> Stage1ControlComparison:
    return Stage1ControlComparison(
        control=control,
        final_loss_delta=diagnostics.final_loss - control.diagnostics.final_loss,
        held_out_loss_delta=diagnostics.held_out_loss - control.diagnostics.held_out_loss,
        reconstruction_error_delta=(
            diagnostics.binary_reconstruction_error
            - control.diagnostics.binary_reconstruction_error
        ),
        matched_fields=(
            "model_state",
            "binary_operator",
            "train_batches",
            "held_out_batches",
            "seed",
            "optimizer_step_budget",
        ),
    )


class Stage1TrainerBackend:
    """Optimize only declared input scales under the active binary causal model."""

    def __init__(self, config: Stage1TrainerConfig) -> None:
        if not isinstance(config, Stage1TrainerConfig):
            raise TypeError("config must be Stage1TrainerConfig")
        self.config = config

    def run(
        self,
        *,
        run_id: str,
        model: nn.Module,
        adapter: ModelAdapter[nn.Module],
        train_batches: Sequence[CausalBatch],
        held_out_batches: Sequence[CausalBatch],
        test_step_override: int | None = None,
        budget: Budget | None = None,
        prior_budget_usage: BudgetUsage | None = None,
        monotonic_clock: Callable[[], float] | None = None,
        cost_meter: Callable[[], float] | None = None,
        artifact_store: FilesystemArtifactStore | None = None,
        parent_artifact_id: str | None = None,
    ) -> Stage1RunResult:
        if not run_id:
            raise ValueError("run_id must be non-empty")
        if not train_batches or not held_out_batches:
            raise ValueError("train_batches and held_out_batches must be non-empty")
        requested_steps = self.config.resolve_steps(test_step_override)
        adapter.set_active_representation(model, ActiveRepresentation.TRAINING)
        scales = named_trainable_scales(model)
        if not scales or any(item.role is not ScaleRole.STAGE1_INPUT for item in scales):
            raise ValueError("only declared Stage 1 input scales may be trainable")
        invalid_scale_shapes = tuple(
            name
            for name, module in _binary_modules(model)
            if module.input_scale.raw_scale.shape != (module.in_features,)
        )
        if invalid_scale_shapes:
            raise ValueError(
                f"Stage 1 requires one scale per input channel: {invalid_scale_shapes}"
            )
        scale_ids = {id(item.parameter) for item in scales}
        undeclared = tuple(
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and id(parameter) not in scale_ids
        )
        if undeclared:
            raise ValueError(f"undeclared optimizer-visible parameters: {undeclared}")
        trainable_names = tuple(item.name for item in scales)
        frozen = {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
            if id(parameter) not in scale_ids
        }
        control_model = copy.deepcopy(model)
        control = _control(
            control_model,
            train_batches,
            held_out_batches,
            requested_steps,
            self.config.seed,
        )
        optimizer = _make_optimizer(
            self.config.optimizer, [item.parameter for item in scales]
        )
        initial_loss = _average_loss(model, train_batches)
        completed_steps = 0
        last_gradients: dict[str, Tensor] = {}
        budget_monitor = (
            None
            if budget is None
            else BudgetMonitor(
                budget,
                prior_usage=prior_budget_usage,
                **({} if monotonic_clock is None else {"clock": monotonic_clock}),
                cost_meter=cost_meter,
            )
        )

        def finish(
            status: Stage1RunStatus,
            suffix: str,
            failure_batch: CausalBatch | None,
            budget_crossing: BudgetCrossing | None = None,
        ) -> Stage1RunResult:
            diagnostics = _build_diagnostics(
                model,
                train_batches,
                held_out_batches,
                initial_loss,
                last_gradients,
            )
            unchanged = all(
                torch.equal(parameter.detach().cpu(), frozen[name])
                for name, parameter in model.named_parameters()
                if name in frozen
            )
            checkpoint = _checkpoint(run_id, suffix, completed_steps, model, optimizer)
            transition = (
                build_stage1_transition(
                    model=model,
                    source_checkpoint_id=checkpoint.checkpoint_id,
                    source_run_id=run_id,
                    parent_artifact_id=parent_artifact_id,
                    training_config={
                        "optimizer_kind": self.config.optimizer.kind.value,
                        "learning_rate": self.config.optimizer.learning_rate,
                        "weight_decay": self.config.optimizer.weight_decay,
                        "seed": self.config.seed,
                        "completed_steps": completed_steps,
                    },
                )
                if status is Stage1RunStatus.COMPLETED
                else None
            )
            if transition is not None and artifact_store is not None:
                persist_stage1_transition(transition, artifact_store)
            return Stage1RunResult(
                status=status,
                requested_steps=requested_steps,
                completed_steps=completed_steps,
                diagnostics=diagnostics,
                gradient_diagnostics=_gradient_diagnostics(last_gradients),
                checkpoint=checkpoint,
                comparison=_comparison(diagnostics, control),
                trainable_scale_names=trainable_names,
                frozen_parameter_names=tuple(frozen),
                dense_parameters_unchanged=unchanged,
                transition=transition,
                failure_batch=None if failure_batch is None else failure_batch.snapshot(),
                budget_usage=None if budget_monitor is None else budget_monitor.usage,
                budget_crossing=budget_crossing,
                budget_failure=(
                    None
                    if budget_monitor is None or budget_crossing is None
                    else budget_monitor.failure(budget_crossing)
                ),
            )

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.config.seed)
            initial_state = _finite_state(model, initial_loss, gradients={})
            if not initial_state.is_finite:
                return finish(
                    Stage1RunStatus.STOPPED_NONFINITE,
                    "nonfinite",
                    train_batches[0],
                )
            model.train()
            for step in range(requested_steps):
                batch = train_batches[step % len(train_batches)]
                batch_tokens = causal_training_tokens(batch.input_ids, batch.labels)
                if budget_monitor is not None:
                    crossing = budget_monitor.crossing(
                        phase="pre_step",
                        proposed_tokens=batch_tokens,
                        proposed_steps=1,
                    )
                    if crossing is not None:
                        return finish(
                            Stage1RunStatus.STOPPED_BUDGET,
                            "budget",
                            batch,
                            crossing,
                        )
                optimizer.zero_grad(set_to_none=True)
                loss = causal_language_model_loss(model, batch)
                pre_backward = _finite_state(model, loss, gradients={})
                if not pre_backward.is_finite:
                    return finish(Stage1RunStatus.STOPPED_NONFINITE, "nonfinite", batch)
                loss.backward()
                last_gradients = _gradients(model)
                post_backward = _finite_state(model, loss, last_gradients)
                if not post_backward.is_finite:
                    return finish(Stage1RunStatus.STOPPED_NONFINITE, "nonfinite", batch)
                optimizer.step()
                completed_steps += 1
                if budget_monitor is not None:
                    budget_monitor.record_step(batch_tokens)
                post_update = _finite_state(model, loss, last_gradients)
                if not post_update.is_finite:
                    return finish(Stage1RunStatus.STOPPED_NONFINITE, "nonfinite", batch)
                if budget_monitor is not None:
                    crossing = budget_monitor.crossing(phase="post_step")
                    if crossing is not None:
                        return finish(
                            Stage1RunStatus.STOPPED_BUDGET,
                            "budget",
                            batch,
                            crossing,
                        )
            return finish(Stage1RunStatus.COMPLETED, "complete", None)


__all__ = [
    "REFERENCE_STAGE1_STEPS",
    "CausalBatch",
    "NoInitializationControl",
    "Stage1Checkpoint",
    "Stage1ControlComparison",
    "Stage1GradientDiagnostics",
    "Stage1OptimizerConfig",
    "Stage1OptimizerKind",
    "Stage1RunResult",
    "Stage1RunStatus",
    "Stage1TrainerBackend",
    "Stage1TrainerConfig",
    "causal_language_model_loss",
]
