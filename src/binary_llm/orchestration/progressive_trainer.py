"""Resumable phase-boundary trainer for consistent progressive BinaryLLM."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from binary_llm.adapters import ActiveRepresentation, BinaryLinear, ModelAdapter
from binary_llm.domain import (
    Budget,
    BudgetExceeded,
    NumericalFailure,
    canonical_json_bytes,
    sha256_bytes,
)
from binary_llm.domain.fidelity import ProgressiveTrainabilityArm
from binary_llm.math import (
    FiniteStateDiagnostics,
    analytical_row_scales,
    binary_sign,
    diagnose_finite_state,
)

from .gates import GateReport
from .budgets import (
    BudgetCrossing,
    BudgetMonitor,
    BudgetUsage,
    causal_training_tokens,
)
from .progressive_state import (
    ProgressiveBoundaryCheckpoint,
    ProgressivePhase,
    ProgressivePhasePlan,
    restore_progressive_checkpoint,
    save_progressive_checkpoint,
    capture_progressive_boundary_checkpoint,
)
from .progressive_failure import (
    ProgressiveFailureArtifact,
    ProgressiveFailurePoint,
    build_progressive_failure_artifact,
    persist_progressive_failure_artifact,
)
from .store import FilesystemArtifactStore
from .stage1 import (
    CausalBatch,
    Stage1OptimizerConfig,
    Stage1OptimizerKind,
    causal_language_model_loss,
)
from .transition import AppliedStage1Transition, Stage1TransitionKind


class ProgressiveRunStatus(StrEnum):
    COMPLETED = "completed"
    STOPPED_GATE = "stopped_gate"
    STOPPED_BUDGET = "stopped_budget"
    STOPPED_NONFINITE = "stopped_nonfinite"


@dataclass(frozen=True, slots=True)
class ProgressiveParameterRecord:
    name: str
    aliases: tuple[str, ...]
    role: str
    element_count: int
    trainable: bool
    optimizer_inclusion_count: int
    before_digest: str
    after_digest: str
    changed: bool


@dataclass(frozen=True, slots=True)
class ProgressiveParameterInventory:
    arm: ProgressiveTrainabilityArm
    records: tuple[ProgressiveParameterRecord, ...]
    inventory_digest: str
    stage1_input_scale_state: str = "folded_into_w_tilde_identity_frozen"


@dataclass(frozen=True, slots=True)
class ProgressiveTrainerConfig:
    optimizer: Stage1OptimizerConfig
    seed: int
    trainability_arm: ProgressiveTrainabilityArm = (
        ProgressiveTrainabilityArm.ALL_MODEL_PARAMETERS
    )
    saturation_threshold: float = 0.99

    def __post_init__(self) -> None:
        if not isinstance(self.optimizer, Stage1OptimizerConfig):
            raise TypeError("optimizer must be Stage1OptimizerConfig")
        object.__setattr__(
            self,
            "trainability_arm",
            ProgressiveTrainabilityArm(self.trainability_arm),
        )
        if (
            not isinstance(self.seed, int)
            or isinstance(self.seed, bool)
            or self.seed < 0
        ):
            raise ValueError("seed must be a non-negative integer")
        if (
            not math.isfinite(self.saturation_threshold)
            or not 0 < self.saturation_threshold <= 1
        ):
            raise ValueError("saturation_threshold must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class DistributionMetrics:
    count: int
    minimum: float
    maximum: float
    mean: float
    standard_deviation: float


@dataclass(frozen=True, slots=True)
class ProgressiveGradientMetrics:
    parameter_count: int
    nonfinite_count: int
    minimum: float | None
    maximum: float | None
    l2_norm: float | None


@dataclass(frozen=True, slots=True)
class ProgressivePhaseMetrics:
    phase_ordinal: int
    schedule_index: int
    progression_parameter: float
    partition_id: str
    partition_hash: str
    batch_ids: tuple[str, ...]
    optimizer_step_start: int
    optimizer_step_end: int
    training_loss: float
    held_out_loss: float
    perplexity: float
    capability_metrics: Mapping[str, float]
    analytical_scales: DistributionMetrics
    learned_scales: DistributionMetrics
    merged_scales: DistributionMetrics
    saturation_fraction: float
    sign_flip_rate: float
    gradient_metrics: ProgressiveGradientMetrics
    analytical_recomputations: int
    finite_state: FiniteStateDiagnostics
    checkpoint_id: str


@dataclass(frozen=True, slots=True)
class ProgressiveViewIdentity:
    identity_id: str
    checkpoint_id: str
    plan_hash: str
    representation: ActiveRepresentation
    progression_parameter: float | None


@dataclass(frozen=True, slots=True)
class ProgressiveViewEvaluation:
    identity: ProgressiveViewIdentity
    held_out_loss: float
    perplexity: float
    capability_metrics: Mapping[str, float]
    analytical_scales: DistributionMetrics
    learned_scales: DistributionMetrics
    merged_scales: DistributionMetrics
    finite_state: FiniteStateDiagnostics


@dataclass(frozen=True, slots=True)
class FinalProgressiveEvidence:
    progressive: ProgressiveViewEvaluation
    sign_substituted: ProgressiveViewEvaluation
    progressive_gate_report: GateReport | None
    sign_gate_report: GateReport | None
    sign_export_eligible: bool
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProgressiveRunResult:
    status: ProgressiveRunStatus
    completed_phase_count: int
    optimizer_steps: int
    phase_metrics: tuple[ProgressivePhaseMetrics, ...]
    checkpoints: tuple[ProgressiveBoundaryCheckpoint, ...]
    checkpoint_paths: tuple[Path, ...]
    stopping_gate_ids: tuple[str, ...]
    final_evidence: FinalProgressiveEvidence | None
    budget_usage: BudgetUsage | None = None
    budget_crossing: BudgetCrossing | None = None
    budget_failure: BudgetExceeded | None = None
    last_complete_checkpoint: ProgressiveBoundaryCheckpoint | None = None
    incomplete_phase_ordinal: int | None = None
    parameter_inventory: ProgressiveParameterInventory | None = None
    failure_artifact: ProgressiveFailureArtifact | None = None


CapabilityEvaluator = Callable[
    [nn.Module, ProgressivePhase, ActiveRepresentation], Mapping[str, float]
]
PhaseGateEvaluator = Callable[
    [ProgressivePhaseMetrics, ProgressiveBoundaryCheckpoint], GateReport | None
]
FinalGateEvaluator = Callable[[ProgressiveViewEvaluation], GateReport]
ProgressiveFaultInjector = Callable[[ProgressivePhase, CausalBatch, nn.Module], None]


def _binary_modules(model: nn.Module) -> tuple[tuple[str, BinaryLinear], ...]:
    modules = tuple(
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, BinaryLinear)
    )
    if not modules:
        raise ValueError("progressive training requires at least one BinaryLinear")
    return modules


def _named_parameters_with_aliases(
    model: nn.Module,
) -> tuple[tuple[str, tuple[str, ...], nn.Parameter], ...]:
    grouped: dict[int, tuple[nn.Parameter, list[str]]] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        entry = grouped.setdefault(id(parameter), (parameter, []))
        entry[1].append(name)
    return tuple(
        (names[0], tuple(names[1:]), parameter)
        for parameter, names in grouped.values()
    )


def _parameter_digest(parameter: nn.Parameter) -> str:
    value = parameter.detach().cpu().contiguous()
    return sha256_bytes(value.reshape(-1).view(torch.uint8).numpy().tobytes())


def _parameter_roles(model: nn.Module) -> dict[int, str]:
    roles: dict[int, str] = {}
    for _, module in _binary_modules(model):
        roles[id(module.weight)] = "binary_latent_weight"
        roles[id(module.dual_scale.learned.value)] = "learned_row_scale"
        roles[id(module.input_scale.raw_scale)] = "folded_stage1_input_scale"
        if module.bias is not None:
            roles[id(module.bias)] = "bias"
    return roles


def _configure_progressive_parameters(
    model: nn.Module, arm: ProgressiveTrainabilityArm
) -> tuple[tuple[nn.Parameter, ...], dict[int, tuple[str, tuple[str, ...], str, int, str]]]:
    named = _named_parameters_with_aliases(model)
    roles = _parameter_roles(model)
    body_ids = {
        id(parameter)
        for _, module in _binary_modules(model)
        for parameter in (module.weight, module.dual_scale.learned.value)
    }
    folded_ids = {
        id(module.input_scale.raw_scale) for _, module in _binary_modules(model)
    }
    selected: list[nn.Parameter] = []
    before: dict[int, tuple[str, tuple[str, ...], str, int, str]] = {}
    for name, aliases, parameter in named:
        parameter_id = id(parameter)
        eligible = parameter_id not in folded_ids
        trainable = eligible and (
            arm is ProgressiveTrainabilityArm.ALL_MODEL_PARAMETERS
            or parameter_id in body_ids
        )
        parameter.requires_grad_(trainable)
        if trainable:
            selected.append(parameter)
        role = roles.get(parameter_id)
        if role is None:
            if "embed" in name:
                role = "embedding"
            elif "norm" in name:
                role = "normalization"
            elif "head" in name:
                role = "language_model_head"
            elif name.endswith(".bias"):
                role = "bias"
            else:
                role = "model_parameter"
        before[parameter_id] = (
            name,
            aliases,
            role,
            parameter.numel(),
            _parameter_digest(parameter),
        )
    if any(parameter.requires_grad for _, _, parameter in named if id(parameter) in folded_ids):
        raise AssertionError("folded Stage 1 input scales must remain frozen")
    return tuple(selected), before


def _parameter_inventory(
    model: nn.Module,
    optimizer: Optimizer,
    arm: ProgressiveTrainabilityArm,
    before: Mapping[int, tuple[str, tuple[str, ...], str, int, str]],
) -> ProgressiveParameterInventory:
    inclusion: dict[int, int] = {}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            inclusion[id(parameter)] = inclusion.get(id(parameter), 0) + 1
    current = {id(parameter): parameter for parameter in model.parameters()}
    records = tuple(
        ProgressiveParameterRecord(
            name=name,
            aliases=aliases,
            role=role,
            element_count=count,
            trainable=current[parameter_id].requires_grad,
            optimizer_inclusion_count=inclusion.get(parameter_id, 0),
            before_digest=before_digest,
            after_digest=_parameter_digest(current[parameter_id]),
            changed=before_digest != _parameter_digest(current[parameter_id]),
        )
        for parameter_id, (name, aliases, role, count, before_digest) in before.items()
    )
    if any(
        record.optimizer_inclusion_count != (1 if record.trainable else 0)
        for record in records
    ):
        raise AssertionError("every trainable parameter must enter the optimizer exactly once")
    record = {
        "arm": arm.value,
        "stage1_input_scale_state": "folded_into_w_tilde_identity_frozen",
        "records": [
            {
                "name": item.name,
                "aliases": item.aliases,
                "role": item.role,
                "element_count": item.element_count,
                "trainable": item.trainable,
                "optimizer_inclusion_count": item.optimizer_inclusion_count,
                "before_digest": item.before_digest,
                "after_digest": item.after_digest,
                "changed": item.changed,
            }
            for item in records
        ],
    }
    return ProgressiveParameterInventory(
        arm,
        records,
        sha256_bytes(canonical_json_bytes(record)),
    )


def _make_optimizer(
    config: Stage1OptimizerConfig, parameters: Sequence[nn.Parameter]
) -> Optimizer:
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


def _operator_tensors(
    model: nn.Module,
) -> tuple[
    dict[str, Tensor],
    dict[str, Tensor],
    dict[str, Tensor],
    dict[str, Tensor],
    dict[str, Tensor],
]:
    transformed: dict[str, Tensor] = {}
    analytical: dict[str, Tensor] = {}
    learned: dict[str, Tensor] = {}
    merged: dict[str, Tensor] = {}
    normalized: dict[str, Tensor] = {}
    for name, module in _binary_modules(model):
        latent = module.transformed_weight()
        row_scale = analytical_row_scales(
            latent,
            gradient=module.config.progressive_operator.analytical_scale_gradient,
        )
        if torch.any(row_scale == 0).item():
            raise ValueError(
                f"progressive representation cannot normalize zero row: {name}"
            )
        learned_scale = module.dual_scale.learned()
        transformed[name] = latent
        analytical[name] = row_scale
        learned[name] = learned_scale
        merged[name] = row_scale * learned_scale
        normalized[name] = latent / row_scale.unsqueeze(1)
    return transformed, analytical, learned, merged, normalized


def _distribution(values: Mapping[str, Tensor]) -> DistributionMetrics:
    flattened = torch.cat(
        [value.detach().float().reshape(-1) for value in values.values()]
    )
    if flattened.numel() == 0 or not torch.isfinite(flattened).all().item():
        raise ValueError("distribution values must be finite and non-empty")
    return DistributionMetrics(
        count=flattened.numel(),
        minimum=float(flattened.min().item()),
        maximum=float(flattened.max().item()),
        mean=float(flattened.mean().item()),
        standard_deviation=float(flattened.std(unbiased=False).item()),
    )


def _gradients(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: parameter.grad
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    }


def _gradient_metrics(gradients: Mapping[str, Tensor]) -> ProgressiveGradientMetrics:
    if not gradients:
        return ProgressiveGradientMetrics(0, 0, None, None, None)
    flattened = torch.cat(
        [gradient.detach().float().reshape(-1) for gradient in gradients.values()]
    )
    finite = flattened[torch.isfinite(flattened)]
    all_finite = finite.numel() == flattened.numel()
    return ProgressiveGradientMetrics(
        parameter_count=flattened.numel(),
        nonfinite_count=flattened.numel() - finite.numel(),
        minimum=None if finite.numel() == 0 else float(finite.min().item()),
        maximum=None if finite.numel() == 0 else float(finite.max().item()),
        l2_norm=(
            float(torch.linalg.vector_norm(finite).item()) if all_finite else None
        ),
    )


def _finite_state(
    model: nn.Module,
    *,
    loss: Tensor | float,
    gradients: Mapping[str, Tensor],
) -> FiniteStateDiagnostics:
    transformed, analytical, learned, merged, _ = _operator_tensors(model)
    scales = {
        **{f"analytical.{name}": value for name, value in analytical.items()},
        **{f"learned.{name}": value for name, value in learned.items()},
        **{f"merged.{name}": value for name, value in merged.items()},
    }
    return diagnose_finite_state(
        scales=scales,
        transformed_weights=transformed,
        losses={"causal": loss},
        gradients=gradients,
    )


def _average_loss(model: nn.Module, batches: Sequence[CausalBatch]) -> Tensor:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        losses = torch.stack(
            [causal_language_model_loss(model, batch) for batch in batches]
        )
    model.train(was_training)
    return losses.mean()


def _perplexity(loss: float) -> float:
    try:
        value = math.exp(loss)
    except OverflowError as exc:
        raise ValueError("held-out loss produces non-finite perplexity") from exc
    if not math.isfinite(value):
        raise ValueError("held-out loss produces non-finite perplexity")
    return value


def _capabilities(
    evaluator: CapabilityEvaluator,
    model: nn.Module,
    phase: ProgressivePhase,
    representation: ActiveRepresentation,
) -> dict[str, float]:
    raw = evaluator(model, phase, representation)
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("capability evaluator must return non-empty metric evidence")
    metrics: dict[str, float] = {}
    for name, value in raw.items():
        if not isinstance(name, str) or not name:
            raise ValueError("capability metric names must be non-empty strings")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("capability metrics must be numeric")
        resolved = float(value)
        if not math.isfinite(resolved):
            raise ValueError("capability metrics must be finite")
        metrics[name] = resolved
    return metrics


def _signs(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: binary_sign(module.transformed_weight(), module.config.zero_sign_rule)
        .detach()
        .clone()
        for name, module in _binary_modules(model)
    }


def _sign_flip_rate(before: Mapping[str, Tensor], after: Mapping[str, Tensor]) -> float:
    changed = 0
    total = 0
    for name in before:
        changed += int((before[name] != after[name]).sum().item())
        total += before[name].numel()
    return changed / total if total else 0.0


def _saturation_fraction(model: nn.Module, threshold: float) -> float:
    values: list[Tensor] = []
    with torch.no_grad():
        for _, module in _binary_modules(model):
            latent = module.transformed_weight()
            analytical = module.dual_scale.analytical(latent)
            normalized = latent / analytical.unsqueeze(1)
            assert module.progression_parameter is not None
            progressive_values = module.effective_weight() / (
                analytical * module.dual_scale.learned()
            ).unsqueeze(1)
            if progressive_values.shape != normalized.shape:
                raise AssertionError("progressive operator changed the latent shape")
            values.append(progressive_values.abs().reshape(-1))
    flattened = torch.cat(values)
    return float((flattened >= threshold).float().mean().item())


def _view_identity(
    checkpoint: ProgressiveBoundaryCheckpoint,
    representation: ActiveRepresentation,
    progression_parameter: float | None,
) -> ProgressiveViewIdentity:
    record = {
        "checkpoint_id": checkpoint.checkpoint_id,
        "plan_hash": checkpoint.plan_hash,
        "state_hashes": dict(checkpoint.state_hashes),
        "representation": representation.value,
        "progression_parameter": progression_parameter,
    }
    return ProgressiveViewIdentity(
        identity_id=sha256_bytes(canonical_json_bytes(record)),
        checkpoint_id=checkpoint.checkpoint_id,
        plan_hash=checkpoint.plan_hash,
        representation=representation,
        progression_parameter=progression_parameter,
    )


def _evaluate_view(
    *,
    model: nn.Module,
    phase: ProgressivePhase,
    checkpoint: ProgressiveBoundaryCheckpoint,
    representation: ActiveRepresentation,
    progression_parameter: float | None,
    held_out_batches: Sequence[CausalBatch],
    capability_evaluator: CapabilityEvaluator,
) -> ProgressiveViewEvaluation:
    held_out = _average_loss(model, held_out_batches)
    transformed, analytical, learned, merged, _ = _operator_tensors(model)
    finite = diagnose_finite_state(
        scales={
            **{f"analytical.{name}": value for name, value in analytical.items()},
            **{f"learned.{name}": value for name, value in learned.items()},
            **{f"merged.{name}": value for name, value in merged.items()},
        },
        transformed_weights=transformed,
        losses={"held_out": held_out},
    )
    finite.raise_if_nonfinite(
        checkpoint_id=checkpoint.checkpoint_id,
        batch_id=held_out_batches[0].batch_id,
    )
    loss_value = float(held_out.item())
    return ProgressiveViewEvaluation(
        identity=_view_identity(checkpoint, representation, progression_parameter),
        held_out_loss=loss_value,
        perplexity=_perplexity(loss_value),
        capability_metrics=_capabilities(
            capability_evaluator, model, phase, representation
        ),
        analytical_scales=_distribution(analytical),
        learned_scales=_distribution(learned),
        merged_scales=_distribution(merged),
        finite_state=finite,
    )


def _final_evidence(
    *,
    model: nn.Module,
    adapter: ModelAdapter[nn.Module],
    plan: ProgressivePhasePlan,
    checkpoint: ProgressiveBoundaryCheckpoint,
    held_out_batches: Sequence[CausalBatch],
    capability_evaluator: CapabilityEvaluator,
    gate_evaluator: FinalGateEvaluator | None,
) -> FinalProgressiveEvidence:
    final_phase = plan.phases[-1]
    t = final_phase.progression_parameter
    adapter.set_active_representation(
        model, ActiveRepresentation.PROGRESSIVE, progression_parameter=t
    )
    progressive_view = _evaluate_view(
        model=model,
        phase=final_phase,
        checkpoint=checkpoint,
        representation=ActiveRepresentation.PROGRESSIVE,
        progression_parameter=t,
        held_out_batches=held_out_batches,
        capability_evaluator=capability_evaluator,
    )
    adapter.set_active_representation(model, ActiveRepresentation.SIGN)
    sign_view = _evaluate_view(
        model=model,
        phase=final_phase,
        checkpoint=checkpoint,
        representation=ActiveRepresentation.SIGN,
        progression_parameter=None,
        held_out_batches=held_out_batches,
        capability_evaluator=capability_evaluator,
    )
    adapter.set_active_representation(
        model, ActiveRepresentation.PROGRESSIVE, progression_parameter=t
    )

    if progressive_view.identity.identity_id == sign_view.identity.identity_id:
        raise AssertionError(
            "progressive and sign-substituted identities must be distinct"
        )
    if gate_evaluator is None:
        return FinalProgressiveEvidence(
            progressive_view,
            sign_view,
            None,
            None,
            False,
            ("final progressive/sign gate evidence is missing",),
        )
    progressive_gate = gate_evaluator(progressive_view)
    sign_gate = gate_evaluator(sign_view)
    if not isinstance(progressive_gate, GateReport) or not isinstance(
        sign_gate, GateReport
    ):
        raise TypeError("final gate evaluator must return GateReport")
    reasons: list[str] = []
    if not progressive_gate.promotion_allowed:
        reasons.append("final progressive view failed or lacked required gate evidence")
    if not sign_gate.promotion_allowed:
        reasons.append(
            "final sign-substituted view failed or lacked required gate evidence"
        )
        if progressive_gate.promotion_allowed:
            reasons.append(
                "sign substitution crossed a gate passed by the progressive view"
            )
    return FinalProgressiveEvidence(
        progressive_view,
        sign_view,
        progressive_gate,
        sign_gate,
        not reasons,
        tuple(reasons),
    )


class ProgressiveTrainerBackend:
    """Run exact progressive updates and expose only complete phase boundaries."""

    def __init__(self, config: ProgressiveTrainerConfig) -> None:
        if not isinstance(config, ProgressiveTrainerConfig):
            raise TypeError("config must be ProgressiveTrainerConfig")
        self.config = config

    def run(
        self,
        *,
        run_id: str,
        parent_checkpoint_id: str,
        model: nn.Module,
        adapter: ModelAdapter[nn.Module],
        plan: ProgressivePhasePlan,
        phase_batches: Mapping[str, Sequence[CausalBatch]],
        held_out_batches: Sequence[CausalBatch],
        capability_evaluator: CapabilityEvaluator,
        applied_transition: AppliedStage1Transition | None = None,
        resume_checkpoint: ProgressiveBoundaryCheckpoint | None = None,
        stop_requested: bool = False,
        phase_gate_evaluator: PhaseGateEvaluator | None = None,
        final_gate_evaluator: FinalGateEvaluator | None = None,
        checkpoint_directory: str | Path | None = None,
        budget: Budget | None = None,
        prior_budget_usage: BudgetUsage | None = None,
        monotonic_clock: Callable[[], float] | None = None,
        cost_meter: Callable[[], float] | None = None,
        attempt_id: str | None = None,
        artifact_store: FilesystemArtifactStore | None = None,
        fault_injector: ProgressiveFaultInjector | None = None,
    ) -> ProgressiveRunResult:
        if not run_id or not parent_checkpoint_id:
            raise ValueError("run and parent checkpoint identities must be non-empty")
        if not isinstance(plan, ProgressivePhasePlan):
            raise TypeError("plan must be ProgressivePhasePlan")
        if not held_out_batches:
            raise ValueError("held_out_batches must be non-empty")
        if resume_checkpoint is None:
            if not isinstance(applied_transition, AppliedStage1Transition):
                raise ValueError(
                    "fresh progressive training requires a verified applied Stage 1 transition"
                )
            if applied_transition.model_identity != id(model):
                raise ValueError("Stage 1 transition was not applied to this model")
            if applied_transition.source_checkpoint_id != parent_checkpoint_id:
                raise ValueError("Stage 1 transition checkpoint lineage does not match")
            if applied_transition.kind is not Stage1TransitionKind.REFERENCE_EXPLICIT:
                raise ValueError("reference progressive training requires an explicit Stage 1 transition")
            stage1_parent_checkpoint_id = applied_transition.source_checkpoint_id
        elif applied_transition is not None:
            raise ValueError("resume is governed by its checkpoint; do not reapply a Stage 1 transition")
        elif resume_checkpoint.stage1_parent_checkpoint_id != parent_checkpoint_id:
            raise ValueError("resume checkpoint Stage 1 parent lineage does not match")
        else:
            stage1_parent_checkpoint_id = resume_checkpoint.stage1_parent_checkpoint_id
        first_phase_index = (
            0 if resume_checkpoint is None else resume_checkpoint.completed_phase_count
        )
        if first_phase_index > plan.phase_count:
            raise ValueError("resume checkpoint exceeds the phase plan")
        if (
            resume_checkpoint is not None
            and resume_checkpoint.plan_hash != plan.plan_hash
        ):
            raise ValueError("resume checkpoint belongs to a different phase plan")
        required_hashes = {
            phase.partition_hash for phase in plan.phases[first_phase_index:]
        }
        if set(phase_batches) != required_hashes:
            raise ValueError(
                "phase batches must identify every remaining planned partition exactly once"
            )
        if any(not batches for batches in phase_batches.values()):
            raise ValueError(
                "every remaining partition must contain at least one causal batch"
            )
        active_phase = plan.phases[min(first_phase_index, plan.phase_count - 1)]
        adapter.set_active_representation(
            model,
            ActiveRepresentation.PROGRESSIVE,
            progression_parameter=active_phase.progression_parameter,
        )
        parameters, _ = _configure_progressive_parameters(
            model, self.config.trainability_arm
        )
        if resume_checkpoint is None:
            for _, module in _binary_modules(model):
                if not torch.equal(
                    module.dual_scale.learned.value.detach(),
                    torch.ones_like(module.dual_scale.learned.value),
                ):
                    raise ValueError(
                        "reference progressive learned scales must start at one"
                    )
        optimizer = _make_optimizer(self.config.optimizer, parameters)

        checkpoints: list[ProgressiveBoundaryCheckpoint] = []
        checkpoint_paths: list[Path] = []
        metrics: list[ProgressivePhaseMetrics] = []
        stopping_gate_ids: tuple[str, ...] = ()
        optimizer_steps = (
            0 if resume_checkpoint is None else resume_checkpoint.optimizer_step
        )
        current_parent = (
            parent_checkpoint_id
            if resume_checkpoint is None
            else resume_checkpoint.checkpoint_id
        )
        final_checkpoint = resume_checkpoint
        status = ProgressiveRunStatus.COMPLETED
        budget_crossing: BudgetCrossing | None = None
        incomplete_phase_ordinal: int | None = None
        failure_artifact: ProgressiveFailureArtifact | None = None
        checkpoint_usage = (
            None if resume_checkpoint is None else resume_checkpoint.budget_usage
        )
        if (
            prior_budget_usage is not None
            and checkpoint_usage is not None
            and not prior_budget_usage.dominates(checkpoint_usage)
        ):
            raise ValueError(
                "prior budget usage must component-wise include checkpoint usage"
            )
        inherited_usage = prior_budget_usage or checkpoint_usage
        budget_monitor = (
            None
            if budget is None
            else BudgetMonitor(
                budget,
                prior_usage=inherited_usage,
                **({} if monotonic_clock is None else {"clock": monotonic_clock}),
                cost_meter=cost_meter,
            )
        )

        rng_devices = (
            list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        )
        with torch.random.fork_rng(devices=rng_devices):
            if resume_checkpoint is None:
                torch.manual_seed(self.config.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(self.config.seed)
            else:
                restore_progressive_checkpoint(
                    resume_checkpoint, plan=plan, model=model, optimizer=optimizer
                )
            _, parameter_state_before = _configure_progressive_parameters(
                model, self.config.trainability_arm
            )
            for phase in plan.phases[first_phase_index:]:
                incomplete_phase_updates = 0
                adapter.set_active_representation(
                    model,
                    ActiveRepresentation.PROGRESSIVE,
                    progression_parameter=phase.progression_parameter,
                )
                _configure_progressive_parameters(model, self.config.trainability_arm)
                phase_start_step = optimizer_steps
                signs_before = _signs(model)
                training_losses: list[float] = []
                last_gradients: dict[str, Tensor] = {}
                model.train()
                for batch in phase_batches[phase.partition_hash]:
                    batch_tokens = causal_training_tokens(batch.input_ids, batch.labels)
                    if budget_monitor is not None:
                        budget_crossing = budget_monitor.crossing(
                            phase="pre_step",
                            proposed_tokens=batch_tokens,
                            proposed_steps=1,
                        )
                        if budget_crossing is not None:
                            status = ProgressiveRunStatus.STOPPED_BUDGET
                            incomplete_phase_ordinal = phase.phase_ordinal
                            break
                    optimizer.zero_grad(set_to_none=True)
                    failure_point = ProgressiveFailurePoint.PRE_BACKWARD
                    diagnostics = diagnose_finite_state(
                        scales={}, transformed_weights={}, losses={}
                    )
                    try:
                        loss = causal_language_model_loss(model, batch)
                        diagnostics = _finite_state(model, loss=loss, gradients={})
                        diagnostics.raise_if_nonfinite(
                            checkpoint_id=current_parent, batch_id=batch.batch_id
                        )
                        failure_point = ProgressiveFailurePoint.POST_BACKWARD
                        loss.backward()
                        last_gradients = _gradients(model)
                        diagnostics = _finite_state(
                            model, loss=loss, gradients=last_gradients
                        )
                        diagnostics.raise_if_nonfinite(
                            checkpoint_id=current_parent, batch_id=batch.batch_id
                        )
                        failure_point = ProgressiveFailurePoint.POST_UPDATE
                        optimizer.step()
                        optimizer_steps += 1
                        incomplete_phase_updates += 1
                        if fault_injector is not None:
                            fault_injector(phase, batch, model)
                        if budget_monitor is not None:
                            budget_monitor.record_step(batch_tokens)
                        diagnostics = _finite_state(
                            model, loss=loss, gradients=last_gradients
                        )
                        diagnostics.raise_if_nonfinite(
                            checkpoint_id=current_parent, batch_id=batch.batch_id
                        )
                    except NumericalFailure as error:
                        status = ProgressiveRunStatus.STOPPED_NONFINITE
                        incomplete_phase_ordinal = phase.phase_ordinal
                        failure_artifact = build_progressive_failure_artifact(
                            run_id=run_id,
                            attempt_id=attempt_id,
                            stage1_parent_checkpoint_id=stage1_parent_checkpoint_id,
                            plan_hash=plan.plan_hash,
                            phase_ordinal=phase.phase_ordinal,
                            schedule_index=phase.schedule_index,
                            partition_id=phase.partition_id,
                            partition_hash=phase.partition_hash,
                            batch=batch,
                            failure_point=failure_point,
                            failure_code=error.code,
                            exception=error,
                            diagnostics=diagnostics,
                            model=model,
                            optimizer=optimizer,
                            optimizer_step=optimizer_steps,
                            budget_usage=(
                                None if budget_monitor is None else budget_monitor.usage
                            ),
                            last_complete_checkpoint=final_checkpoint,
                        )
                        if artifact_store is not None:
                            persist_progressive_failure_artifact(
                                failure_artifact, artifact_store
                            )
                        if budget_monitor is not None:
                            budget_monitor.record_interrupted_attempt()
                        break
                    training_losses.append(float(loss.detach().item()))
                    if budget_monitor is not None:
                        budget_crossing = budget_monitor.crossing(phase="post_step")
                        if budget_crossing is not None:
                            status = ProgressiveRunStatus.STOPPED_BUDGET
                            incomplete_phase_ordinal = phase.phase_ordinal
                            break

                if status in (
                    ProgressiveRunStatus.STOPPED_BUDGET,
                    ProgressiveRunStatus.STOPPED_NONFINITE,
                ):
                    # A rejection before this attempt updates anything is not an
                    # interruption; discarded partial-phase work is exactly one.
                    if (
                        status is ProgressiveRunStatus.STOPPED_BUDGET
                        and budget_monitor is not None
                        and incomplete_phase_updates
                    ):
                        budget_monitor.record_interrupted_attempt()
                    break

                held_out = _average_loss(model, held_out_batches)
                transformed, analytical, learned, merged, _ = _operator_tensors(model)
                finite = diagnose_finite_state(
                    scales={
                        **{
                            f"analytical.{name}": value
                            for name, value in analytical.items()
                        },
                        **{f"learned.{name}": value for name, value in learned.items()},
                        **{f"merged.{name}": value for name, value in merged.items()},
                    },
                    transformed_weights=transformed,
                    losses={
                        "training": sum(training_losses) / len(training_losses),
                        "held_out": held_out,
                    },
                    gradients=last_gradients,
                )
                finite.raise_if_nonfinite(
                    checkpoint_id=current_parent,
                    batch_id=held_out_batches[0].batch_id,
                )
                checkpoint_id = (
                    f"{run_id}:progressive:phase-{phase.phase_ordinal}:complete"
                )
                phase_metrics = ProgressivePhaseMetrics(
                    phase_ordinal=phase.phase_ordinal,
                    schedule_index=phase.schedule_index,
                    progression_parameter=phase.progression_parameter,
                    partition_id=phase.partition_id,
                    partition_hash=phase.partition_hash,
                    batch_ids=tuple(
                        batch.batch_id for batch in phase_batches[phase.partition_hash]
                    ),
                    optimizer_step_start=phase_start_step,
                    optimizer_step_end=optimizer_steps,
                    training_loss=sum(training_losses) / len(training_losses),
                    held_out_loss=float(held_out.item()),
                    perplexity=_perplexity(float(held_out.item())),
                    capability_metrics=_capabilities(
                        capability_evaluator,
                        model,
                        phase,
                        ActiveRepresentation.PROGRESSIVE,
                    ),
                    analytical_scales=_distribution(analytical),
                    learned_scales=_distribution(learned),
                    merged_scales=_distribution(merged),
                    saturation_fraction=_saturation_fraction(
                        model, self.config.saturation_threshold
                    ),
                    sign_flip_rate=_sign_flip_rate(signs_before, _signs(model)),
                    gradient_metrics=_gradient_metrics(last_gradients),
                    analytical_recomputations=(optimizer_steps - phase_start_step),
                    finite_state=finite,
                    checkpoint_id=checkpoint_id,
                )
                checkpoint = capture_progressive_boundary_checkpoint(
                    checkpoint_id=checkpoint_id,
                    parent_checkpoint_id=current_parent,
                    stage1_parent_checkpoint_id=stage1_parent_checkpoint_id,
                    plan=plan,
                    completed_phase_count=phase.phase_ordinal,
                    optimizer_step=optimizer_steps,
                    model=model,
                    optimizer=optimizer,
                    budget_usage=(
                        None if budget_monitor is None else budget_monitor.usage
                    ),
                )

                checkpoints.append(checkpoint)
                metrics.append(phase_metrics)
                final_checkpoint = checkpoint
                current_parent = checkpoint.checkpoint_id
                if checkpoint_directory is not None:
                    path = Path(checkpoint_directory) / (
                        f"phase-{phase.phase_ordinal:02d}.pt"
                    )
                    checkpoint_paths.append(
                        save_progressive_checkpoint(path, checkpoint)
                    )
                report = (
                    None
                    if phase_gate_evaluator is None
                    else phase_gate_evaluator(phase_metrics, checkpoint)
                )
                if report is not None and not isinstance(report, GateReport):
                    raise TypeError(
                        "phase gate evaluator must return GateReport or None"
                    )
                boundary_stop = stop_requested or bool(report and report.stop_requested)
                if boundary_stop:
                    status = ProgressiveRunStatus.STOPPED_GATE
                    stopping_gate_ids = (
                        () if report is None else report.immediate_stop_gate_ids
                    )
                    break

            completed_phase_count = (
                first_phase_index
                if final_checkpoint is None
                else final_checkpoint.completed_phase_count
            )
            final_evidence = None
            if status is ProgressiveRunStatus.COMPLETED:
                if (
                    completed_phase_count != plan.phase_count
                    or final_checkpoint is None
                ):
                    raise AssertionError(
                        "completed run did not reach the final phase boundary"
                    )
                final_evidence = _final_evidence(
                    model=model,
                    adapter=adapter,
                    plan=plan,
                    checkpoint=final_checkpoint,
                    held_out_batches=held_out_batches,
                    capability_evaluator=capability_evaluator,
                    gate_evaluator=final_gate_evaluator,
                )

        _configure_progressive_parameters(model, self.config.trainability_arm)
        parameter_inventory = _parameter_inventory(
            model,
            optimizer,
            self.config.trainability_arm,
            parameter_state_before,
        )
        return ProgressiveRunResult(
            status=status,
            completed_phase_count=completed_phase_count,
            optimizer_steps=optimizer_steps,
            phase_metrics=tuple(metrics),
            checkpoints=tuple(checkpoints),
            checkpoint_paths=tuple(checkpoint_paths),
            stopping_gate_ids=stopping_gate_ids,
            final_evidence=final_evidence,
            budget_usage=None if budget_monitor is None else budget_monitor.usage,
            budget_crossing=budget_crossing,
            budget_failure=(
                None
                if budget_monitor is None or budget_crossing is None
                else budget_monitor.failure(budget_crossing)
            ),
            last_complete_checkpoint=final_checkpoint,
            incomplete_phase_ordinal=incomplete_phase_ordinal,
            parameter_inventory=parameter_inventory,
            failure_artifact=failure_artifact,
        )


__all__ = [
    "CapabilityEvaluator",
    "DistributionMetrics",
    "FinalGateEvaluator",
    "FinalProgressiveEvidence",
    "PhaseGateEvaluator",
    "ProgressiveGradientMetrics",
    "ProgressiveFaultInjector",
    "ProgressiveParameterInventory",
    "ProgressiveParameterRecord",
    "ProgressivePhaseMetrics",
    "ProgressiveRunResult",
    "ProgressiveRunStatus",
    "ProgressiveTrainerBackend",
    "ProgressiveTrainerConfig",
    "ProgressiveViewEvaluation",
    "ProgressiveViewIdentity",
]
