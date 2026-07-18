"""Stage 1 and dual-scaling primitives for BinaryLLM operators."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from binary_llm.domain.errors import NumericalFailure, Retryability


class ScaleParameterization(str, Enum):
    """Explicit Stage 1 input-channel scale interpretations."""

    POSITIVE_EXP = "positive_exp"
    POSITIVE_SOFTPLUS = "positive_softplus"
    SIGNED_CLAMP = "signed_clamp"


class AnalyticalScaleGradient(str, Enum):
    """Whether gradients pass through the analytical row statistic."""

    DETACHED = "detached"
    DIFFERENTIABLE = "differentiable"


class ZeroSignRule(str, Enum):
    """Declared treatment of exact zeros in a binary sign view."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    ERROR = "error"


class ScaleRole(str, Enum):
    """Optimizer-visible scale roles."""

    STAGE1_INPUT = "stage1_input"
    LEARNED_ROW = "learned_row"


def _enum_value(value: str | Enum, enum_type: type[Enum], field: str) -> Enum:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        choices = ", ".join(item.value for item in enum_type)
        raise ValueError(f"{field} must be one of: {choices}") from exc


def _require_floating_tensor(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must use a floating dtype")


def _require_matrix(value: Tensor, name: str) -> None:
    _require_floating_tensor(value, name)
    if value.ndim != 2 or value.shape[0] < 1 or value.shape[1] < 1:
        raise ValueError(f"{name} must have non-empty shape [out_features, in_features]")


def _softplus_inverse(value: float) -> float:
    return value + math.log(-math.expm1(-value))


class _DeclaredScaleModule(nn.Module):
    def declared_scale_parameters(self) -> tuple[tuple[str, ScaleRole, nn.Parameter], ...]:
        raise NotImplementedError


class InputChannelScale(_DeclaredScaleModule):
    """One explicitly parameterized trainable scale per input channel."""

    def __init__(
        self,
        in_features: int,
        *,
        parameterization: ScaleParameterization | str,
        initial_value: float = 1.0,
        epsilon: float = 1e-6,
        minimum_magnitude: float = 1e-6,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(in_features, int) or isinstance(in_features, bool) or in_features < 1:
            raise ValueError("in_features must be a positive integer")
        if not math.isfinite(initial_value):
            raise ValueError("initial_value must be finite")
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        if not math.isfinite(minimum_magnitude) or minimum_magnitude <= 0:
            raise ValueError("minimum_magnitude must be finite and positive")
        resolved = _enum_value(parameterization, ScaleParameterization, "parameterization")
        assert isinstance(resolved, ScaleParameterization)
        if resolved is ScaleParameterization.POSITIVE_EXP:
            if initial_value <= 0:
                raise ValueError("positive_exp requires initial_value > 0")
            raw_initial = math.log(initial_value)
        elif resolved is ScaleParameterization.POSITIVE_SOFTPLUS:
            if initial_value <= epsilon:
                raise ValueError("positive_softplus requires initial_value > epsilon")
            raw_initial = _softplus_inverse(initial_value - epsilon)
        else:
            if abs(initial_value) < minimum_magnitude:
                raise ValueError("signed_clamp cannot represent an initializer below its magnitude floor")
            raw_initial = initial_value
        self.parameterization = resolved
        self.initial_value = float(initial_value)
        self.epsilon = float(epsilon)
        self.minimum_magnitude = float(minimum_magnitude)
        self.raw_scale = nn.Parameter(
            torch.full((in_features,), raw_initial, dtype=dtype, device=device)
        )

    def forward(self) -> Tensor:
        if self.parameterization is ScaleParameterization.POSITIVE_EXP:
            return torch.exp(self.raw_scale)
        if self.parameterization is ScaleParameterization.POSITIVE_SOFTPLUS:
            return F.softplus(self.raw_scale) + self.epsilon
        signs = torch.where(self.raw_scale < 0, -torch.ones_like(self.raw_scale), torch.ones_like(self.raw_scale))
        return signs * self.raw_scale.abs().clamp_min(self.minimum_magnitude)

    def declared_scale_parameters(self) -> tuple[tuple[str, ScaleRole, nn.Parameter], ...]:
        return (("raw_scale", ScaleRole.STAGE1_INPUT, self.raw_scale),)


class LearnableRowScale(_DeclaredScaleModule):
    """One learnable output-row scale, initialized exactly to one."""

    def __init__(
        self,
        out_features: int,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(out_features, int) or isinstance(out_features, bool) or out_features < 1:
            raise ValueError("out_features must be a positive integer")
        self.value = nn.Parameter(torch.ones(out_features, dtype=dtype, device=device))

    def forward(self) -> Tensor:
        return self.value

    def declared_scale_parameters(self) -> tuple[tuple[str, ScaleRole, nn.Parameter], ...]:
        return (("value", ScaleRole.LEARNED_ROW, self.value),)


def transformed_weight(dense_weight: Tensor, input_scales: Tensor) -> Tensor:
    """Return ``W / s`` without mutating the dense source or scale vector."""

    _require_matrix(dense_weight, "dense_weight")
    _require_floating_tensor(input_scales, "input_scales")
    if input_scales.ndim != 1 or input_scales.shape[0] != dense_weight.shape[1]:
        raise ValueError("input_scales must have shape [in_features]")
    if torch.any(input_scales == 0).item():
        raise ValueError("input_scales must not contain zero")
    return dense_weight / input_scales.unsqueeze(0)


def binary_sign(values: Tensor, zero_rule: ZeroSignRule | str) -> Tensor:
    """Create a sign view while retaining NaNs for fail-closed diagnostics."""

    _require_floating_tensor(values, "values")
    resolved = _enum_value(zero_rule, ZeroSignRule, "zero_rule")
    assert isinstance(resolved, ZeroSignRule)
    zero_mask = values == 0
    if resolved is ZeroSignRule.ERROR and torch.any(zero_mask).item():
        raise ValueError("zero values are forbidden by the declared sign rule")
    result = torch.sign(values)
    result = torch.where(torch.isnan(values), values, result)
    if resolved is not ZeroSignRule.ERROR:
        fill = 1.0 if resolved is ZeroSignRule.POSITIVE else -1.0
        result = torch.where(zero_mask, torch.full_like(result, fill), result)
    return result


def analytical_row_scales(
    latent_weight: Tensor,
    *,
    gradient: AnalyticalScaleGradient | str,
) -> Tensor:
    """Recompute each current row's exact mean absolute value."""

    _require_matrix(latent_weight, "latent_weight")
    resolved = _enum_value(gradient, AnalyticalScaleGradient, "gradient")
    scale = latent_weight.abs().mean(dim=1)
    return scale.detach() if resolved is AnalyticalScaleGradient.DETACHED else scale


def merged_inference_scales(
    latent_weight: Tensor,
    learned_row_scales: Tensor,
    *,
    gradient: AnalyticalScaleGradient | str,
) -> Tensor:
    """Merge analytical and learned scales into one value per output row."""

    analytical = analytical_row_scales(latent_weight, gradient=gradient)
    _require_floating_tensor(learned_row_scales, "learned_row_scales")
    if learned_row_scales.ndim != 1 or learned_row_scales.shape != analytical.shape:
        raise ValueError("learned_row_scales must have shape [out_features]")
    return analytical * learned_row_scales


def dual_scaled_sign_weight(
    latent_weight: Tensor,
    learned_row_scales: Tensor,
    *,
    gradient: AnalyticalScaleGradient | str,
    zero_rule: ZeroSignRule | str,
) -> Tensor:
    """Apply the merged dual scale to a final binary sign view."""

    merged = merged_inference_scales(
        latent_weight,
        learned_row_scales,
        gradient=gradient,
    )
    return merged.unsqueeze(1) * binary_sign(latent_weight, zero_rule)


class Stage1ScaleState(nn.Module):
    """Frozen dense source plus the only optimizer-visible Stage 1 scales."""

    def __init__(
        self,
        dense_weight: Tensor,
        *,
        parameterization: ScaleParameterization | str,
        initial_value: float = 1.0,
        epsilon: float = 1e-6,
        minimum_magnitude: float = 1e-6,
    ) -> None:
        super().__init__()
        _require_matrix(dense_weight, "dense_weight")
        self.register_buffer(
            "_dense_source",
            dense_weight.detach().clone(),
            persistent=True,
        )
        self.input_scale = InputChannelScale(
            dense_weight.shape[1],
            parameterization=parameterization,
            initial_value=initial_value,
            epsilon=epsilon,
            minimum_magnitude=minimum_magnitude,
            dtype=dense_weight.dtype,
            device=dense_weight.device,
        )

    @property
    def dense_source(self) -> Tensor:
        """Return a defensive snapshot; callers cannot mutate the stored source."""

        return self._dense_source.detach().clone()

    def transformed_weight(self) -> Tensor:
        return transformed_weight(self._dense_source, self.input_scale())

    def binary_weight(self, *, zero_rule: ZeroSignRule | str) -> Tensor:
        return binary_sign(self.transformed_weight(), zero_rule)


class DualScaleState(nn.Module):
    """Learned row state with analytical scales recomputed on every call."""

    def __init__(
        self,
        out_features: int,
        *,
        analytical_gradient: AnalyticalScaleGradient | str,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        resolved = _enum_value(analytical_gradient, AnalyticalScaleGradient, "analytical_gradient")
        assert isinstance(resolved, AnalyticalScaleGradient)
        self.analytical_gradient = resolved
        self.learned = LearnableRowScale(out_features, dtype=dtype, device=device)

    def analytical(self, latent_weight: Tensor) -> Tensor:
        return analytical_row_scales(latent_weight, gradient=self.analytical_gradient)

    def merged(self, latent_weight: Tensor) -> Tensor:
        return merged_inference_scales(
            latent_weight,
            self.learned(),
            gradient=self.analytical_gradient,
        )

    def sign_weight(self, latent_weight: Tensor, *, zero_rule: ZeroSignRule | str) -> Tensor:
        return dual_scaled_sign_weight(
            latent_weight,
            self.learned(),
            gradient=self.analytical_gradient,
            zero_rule=zero_rule,
        )


@dataclass(frozen=True, slots=True)
class NamedTrainableScale:
    """A declared optimizer-visible scale parameter and its semantic role."""

    name: str
    role: ScaleRole
    parameter: nn.Parameter


@dataclass(frozen=True, slots=True)
class ScaleTrainabilityReport:
    """Detect undeclared trainables and accidentally frozen declared scales."""

    declared_trainable: tuple[str, ...]
    undeclared_trainable: tuple[str, ...]
    frozen_declared: tuple[str, ...]

    @property
    def only_declared_scales_trainable(self) -> bool:
        return not self.undeclared_trainable and not self.frozen_declared


def _declared_scales(module: nn.Module) -> tuple[NamedTrainableScale, ...]:
    declared: list[NamedTrainableScale] = []
    seen: set[int] = set()
    for module_name, child in module.named_modules():
        if not isinstance(child, _DeclaredScaleModule):
            continue
        for local_name, role, parameter in child.declared_scale_parameters():
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            prefix = f"{module_name}." if module_name else ""
            declared.append(NamedTrainableScale(f"{prefix}{local_name}", role, parameter))
    return tuple(declared)


def named_trainable_scales(module: nn.Module) -> tuple[NamedTrainableScale, ...]:
    """Identify trainable scales by declaration, never by parameter-name heuristics."""

    if not isinstance(module, nn.Module):
        raise TypeError("module must be a torch.nn.Module")
    return tuple(item for item in _declared_scales(module) if item.parameter.requires_grad)


def scale_trainability_report(module: nn.Module) -> ScaleTrainabilityReport:
    """Report whether exactly the declared scale parameters are trainable."""

    if not isinstance(module, nn.Module):
        raise TypeError("module must be a torch.nn.Module")
    declared = _declared_scales(module)
    declared_ids = {id(item.parameter) for item in declared}
    return ScaleTrainabilityReport(
        declared_trainable=tuple(item.name for item in declared if item.parameter.requires_grad),
        undeclared_trainable=tuple(
            name
            for name, parameter in module.named_parameters()
            if parameter.requires_grad and id(parameter) not in declared_ids
        ),
        frozen_declared=tuple(item.name for item in declared if not item.parameter.requires_grad),
    )


@dataclass(frozen=True, slots=True)
class TensorFiniteSummary:
    """Finite-value counts and extrema for one named tensor or scalar."""

    name: str
    element_count: int
    nonfinite_count: int
    finite_min: float | None
    finite_max: float | None

    @property
    def is_finite(self) -> bool:
        return self.nonfinite_count == 0


def _summary(name: str, value: Tensor | float) -> TensorFiniteSummary:
    tensor = value.detach() if isinstance(value, Tensor) else torch.as_tensor(value, dtype=torch.float64)
    _require_floating_tensor(tensor, name)
    finite_mask = torch.isfinite(tensor)
    finite_values = tensor[finite_mask]
    return TensorFiniteSummary(
        name=name,
        element_count=tensor.numel(),
        nonfinite_count=int((~finite_mask).sum().item()),
        finite_min=None if finite_values.numel() == 0 else float(finite_values.min().item()),
        finite_max=None if finite_values.numel() == 0 else float(finite_values.max().item()),
    )


def _named_values(
    category: str,
    values: Tensor | float | Mapping[str, Tensor | float] | None,
) -> tuple[tuple[str, Tensor | float], ...]:
    if values is None:
        return ()
    if isinstance(values, Mapping):
        if any(not isinstance(name, str) or not name for name in values):
            raise ValueError(f"{category} names must be non-empty strings")
        return tuple((f"{category}.{name}", values[name]) for name in sorted(values))
    return ((category, values),)


@dataclass(frozen=True, slots=True)
class FiniteStateDiagnostics:
    """Complete fail-closed summaries for all numerically sensitive Stage 1 state."""

    summaries: tuple[TensorFiniteSummary, ...]

    @property
    def is_finite(self) -> bool:
        return all(item.is_finite for item in self.summaries)

    @property
    def total_nonfinite_count(self) -> int:
        return sum(item.nonfinite_count for item in self.summaries)

    @property
    def failing_fields(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.summaries if not item.is_finite)

    @property
    def nonfinite_counts(self) -> dict[str, int]:
        return {item.name: item.nonfinite_count for item in self.summaries}

    def extrema(self, prefix: str) -> tuple[float | None, float | None]:
        matching = [item for item in self.summaries if item.name == prefix or item.name.startswith(f"{prefix}.")]
        minima = [item.finite_min for item in matching if item.finite_min is not None]
        maxima = [item.finite_max for item in matching if item.finite_max is not None]
        return (min(minima) if minima else None, max(maxima) if maxima else None)

    def raise_if_nonfinite(self, *, checkpoint_id: str, batch_id: str) -> None:
        """Stop with stable references to the failing checkpoint and diagnostic batch."""

        if self.is_finite:
            return
        if not checkpoint_id or not batch_id:
            raise ValueError("non-finite failures require checkpoint_id and batch_id")
        scale_min, scale_max = self.extrema("scale")
        raise NumericalFailure(
            "non-finite BinaryLLM operator state detected",
            retryability=Retryability.FROM_CHECKPOINT,
            code="numerical.nonfinite_state",
            affected_ids={"checkpoint_ids": (checkpoint_id,), "batch_ids": (batch_id,)},
            context={
                "failing_fields": list(self.failing_fields),
                "nonfinite_counts": self.nonfinite_counts,
                "scale_finite_min": scale_min,
                "scale_finite_max": scale_max,
                "total_nonfinite_count": self.total_nonfinite_count,
            },
        )


def diagnose_finite_state(
    *,
    scales: Tensor | Mapping[str, Tensor],
    transformed_weights: Tensor | Mapping[str, Tensor],
    losses: Tensor | float | Mapping[str, Tensor | float] | None = None,
    reconstruction_errors: Tensor | float | Mapping[str, Tensor | float] | None = None,
    gradients: Tensor | Mapping[str, Tensor] | None = None,
) -> FiniteStateDiagnostics:
    """Inspect scales, transformed weights, losses, reconstruction, and gradients."""

    named = (
        _named_values("scale", scales)
        + _named_values("transformed_weight", transformed_weights)
        + _named_values("loss", losses)
        + _named_values("reconstruction_error", reconstruction_errors)
        + _named_values("gradient", gradients)
    )
    return FiniteStateDiagnostics(tuple(_summary(name, value) for name, value in named))


def _scalar_value(value: Tensor | float, name: str) -> float:
    tensor = value.detach() if isinstance(value, Tensor) else torch.as_tensor(value, dtype=torch.float64)
    _require_floating_tensor(tensor, name)
    if tensor.numel() != 1:
        raise ValueError(f"{name} must be scalar")
    return float(tensor.item())


def _json_number(value: float) -> float | str:
    if math.isnan(value):
        return "nan"
    if value == math.inf:
        return "positive_infinity"
    if value == -math.inf:
        return "negative_infinity"
    return value


@dataclass(frozen=True, slots=True)
class Stage1Diagnostics:
    """Required Stage 1 metrics plus full finite-state evidence."""

    initial_loss: float
    final_loss: float
    held_out_loss: float
    binary_reconstruction_error: float
    scale_min: float | None
    scale_max: float | None
    finite_state: FiniteStateDiagnostics

    def to_dict(self) -> dict[str, object]:
        return {
            "initial_loss": _json_number(self.initial_loss),
            "final_loss": _json_number(self.final_loss),
            "held_out_loss": _json_number(self.held_out_loss),
            "binary_reconstruction_error": _json_number(self.binary_reconstruction_error),
            "scale_min": self.scale_min,
            "scale_max": self.scale_max,
            "nonfinite_counts": self.finite_state.nonfinite_counts,
            "total_nonfinite_count": self.finite_state.total_nonfinite_count,
            "failing_fields": list(self.finite_state.failing_fields),
            "is_finite": self.finite_state.is_finite,
        }


def stage1_diagnostics(
    *,
    scales: Tensor | Mapping[str, Tensor],
    transformed_weights: Tensor | Mapping[str, Tensor],
    initial_loss: Tensor | float,
    final_loss: Tensor | float,
    held_out_loss: Tensor | float,
    binary_reconstruction_error: Tensor | float,
    gradients: Tensor | Mapping[str, Tensor] | None,
) -> Stage1Diagnostics:
    """Build the complete initialization diagnostic schema required by the trainer."""

    finite_state = diagnose_finite_state(
        scales=scales,
        transformed_weights=transformed_weights,
        losses={"initial": initial_loss, "final": final_loss, "held_out": held_out_loss},
        reconstruction_errors={"binary": binary_reconstruction_error},
        gradients=gradients,
    )
    scale_min, scale_max = finite_state.extrema("scale")
    return Stage1Diagnostics(
        initial_loss=_scalar_value(initial_loss, "initial_loss"),
        final_loss=_scalar_value(final_loss, "final_loss"),
        held_out_loss=_scalar_value(held_out_loss, "held_out_loss"),
        binary_reconstruction_error=_scalar_value(binary_reconstruction_error, "binary_reconstruction_error"),
        scale_min=scale_min,
        scale_max=scale_max,
        finite_state=finite_state,
    )


def binary_reconstruction_error(reference: Tensor, candidate: Tensor) -> Tensor:
    """Return declared mean-squared binary reconstruction error."""

    _require_floating_tensor(reference, "reference")
    _require_floating_tensor(candidate, "candidate")
    if reference.shape != candidate.shape or reference.numel() == 0:
        raise ValueError("reference and candidate must have the same non-empty shape")
    return torch.mean(torch.square(reference - candidate))
