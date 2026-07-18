"""Stable consistent-progressive BinaryLLM operator and registered schedules."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor

from .scales import AnalyticalScaleGradient

REFERENCE_PHASE_COUNT = 20
PAPER_SCHEDULE_SCALE = 1.3
PAPER_SCHEDULE_RATE = 0.22


class OperatorPrecision(str, Enum):
    """Manifest-declared precision for the numerically sensitive operator core."""

    BFLOAT16 = "bf16"
    FLOAT32 = "fp32"


class PhaseIndexConvention(str, Enum):
    """Explicit interpretation of the paper's twenty phase indices."""

    ZERO_BASED = "0..19"
    ONE_BASED = "1..20"


class ProgressionScheduleKind(str, Enum):
    """Registered reference and matched alternative progression schedules."""

    PAPER_EXPONENTIAL = "paper_exponential"
    LINEAR_ENDPOINT_MATCHED = "linear_endpoint_matched"
    COSINE_ENDPOINT_MATCHED = "cosine_endpoint_matched"


@dataclass(frozen=True, slots=True)
class ProgressiveOperatorConfig:
    """Every material numerical and analytical-scale choice for the operator."""

    precision: OperatorPrecision | str
    analytical_scale_gradient: AnalyticalScaleGradient | str
    small_t_threshold: float = 1e-4

    def __post_init__(self) -> None:
        object.__setattr__(self, "precision", _resolve_enum(self.precision, OperatorPrecision, "precision"))
        object.__setattr__(
            self,
            "analytical_scale_gradient",
            _resolve_enum(
                self.analytical_scale_gradient,
                AnalyticalScaleGradient,
                "analytical_scale_gradient",
            ),
        )
        if not math.isfinite(self.small_t_threshold) or self.small_t_threshold <= 0:
            raise ValueError("small_t_threshold must be finite and positive")


@dataclass(frozen=True, slots=True)
class ProgressionScheduleConfig:
    """A schedule cannot be evaluated without an explicit indexing convention."""

    schedule: ProgressionScheduleKind | str
    phase_index: PhaseIndexConvention | str

    def __post_init__(self) -> None:
        object.__setattr__(self, "schedule", _resolve_enum(self.schedule, ProgressionScheduleKind, "schedule"))
        object.__setattr__(
            self,
            "phase_index",
            _resolve_enum(self.phase_index, PhaseIndexConvention, "phase_index"),
        )


def _resolve_enum(value: object, enum_type: type[Enum], field: str) -> Enum:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        choices = ", ".join(item.value for item in enum_type)
        raise ValueError(f"{field} must be one of: {choices}") from exc


def _require_floating_tensor(value: Tensor) -> None:
    if not isinstance(value, Tensor):
        raise TypeError("x must be a torch.Tensor")
    if not value.is_floating_point():
        raise TypeError("x must use a floating dtype")


def _require_progression_parameter(t: float) -> float:
    if isinstance(t, bool) or not isinstance(t, (int, float)):
        raise TypeError("t must be a real scalar")
    resolved = float(t)
    if not math.isfinite(resolved) or resolved < 0:
        raise ValueError("t must be finite and non-negative")
    return resolved


def _compute_dtype(precision: OperatorPrecision) -> torch.dtype:
    if precision is OperatorPrecision.BFLOAT16:
        return torch.bfloat16
    return torch.float32


def _tanhc(value: Tensor) -> Tensor:
    """Stable tanh(value) / value, including its value at zero."""

    squared = value.square()
    series = 1.0 - squared / 3.0 + 2.0 * squared.square() / 15.0
    safe_denominator = torch.where(value == 0, torch.ones_like(value), value)
    quotient = torch.tanh(value) / safe_denominator
    return torch.where(value.abs() < 1e-3, series, quotient)


def _forward_in_compute_precision(x: Tensor, t: float, small_t_threshold: float) -> Tensor:
    if t == 0.0:
        return x.clone()
    t_tensor = x.new_tensor(t)
    tx = t_tensor * x
    if t <= small_t_threshold:
        return x * _tanhc(tx) / _tanhc(t_tensor)
    return torch.tanh(tx) / torch.tanh(t_tensor)


def _derivative_in_compute_precision(x: Tensor, t: float, small_t_threshold: float) -> Tensor:
    if t == 0.0:
        return torch.ones_like(x)
    t_tensor = x.new_tensor(t)
    tanh_tx = torch.tanh(t_tensor * x)
    numerator = 1.0 - tanh_tx.square()
    if t <= small_t_threshold:
        return numerator / _tanhc(t_tensor)
    return t_tensor * numerator / torch.tanh(t_tensor)


class _ProgressiveAutograd(torch.autograd.Function):
    """Custom autograd path using the paper's analytical derivative, never an STE."""

    @staticmethod
    def forward(ctx: object, x: Tensor, t: float, compute_dtype: torch.dtype, threshold: float) -> Tensor:
        x_compute = x.to(dtype=compute_dtype)
        ctx.save_for_backward(x_compute)  # type: ignore[attr-defined]
        ctx.t = t  # type: ignore[attr-defined]
        ctx.compute_dtype = compute_dtype  # type: ignore[attr-defined]
        ctx.input_dtype = x.dtype  # type: ignore[attr-defined]
        ctx.threshold = threshold  # type: ignore[attr-defined]
        result = _forward_in_compute_precision(x_compute, t, threshold)
        return result.to(dtype=x.dtype)

    @staticmethod
    def backward(ctx: object, grad_output: Tensor) -> tuple[Tensor, None, None, None]:
        (x_compute,) = ctx.saved_tensors  # type: ignore[attr-defined]
        derivative = _derivative_in_compute_precision(
            x_compute,
            ctx.t,  # type: ignore[attr-defined]
            ctx.threshold,  # type: ignore[attr-defined]
        )
        grad_x = grad_output.to(dtype=ctx.compute_dtype) * derivative  # type: ignore[attr-defined]
        return grad_x.to(dtype=ctx.input_dtype), None, None, None  # type: ignore[attr-defined]


def progressive(x: Tensor, t: float, *, config: ProgressiveOperatorConfig) -> Tensor:
    """Apply ``tanh(t*x)/tanh(t)`` with its continuous branch at ``t=0``."""

    _require_floating_tensor(x)
    if not isinstance(config, ProgressiveOperatorConfig):
        raise TypeError("config must be a ProgressiveOperatorConfig")
    resolved_t = _require_progression_parameter(t)
    return _ProgressiveAutograd.apply(
        x,
        resolved_t,
        _compute_dtype(config.precision),
        config.small_t_threshold,
    )


def progressive_derivative(x: Tensor, t: float, *, config: ProgressiveOperatorConfig) -> Tensor:
    """Evaluate the exact analytical derivative used by ``progressive`` backward."""

    _require_floating_tensor(x)
    if not isinstance(config, ProgressiveOperatorConfig):
        raise TypeError("config must be a ProgressiveOperatorConfig")
    resolved_t = _require_progression_parameter(t)
    compute = x.to(dtype=_compute_dtype(config.precision))
    derivative = _derivative_in_compute_precision(compute, resolved_t, config.small_t_threshold)
    return derivative.to(dtype=x.dtype)


def paper_exponential_parameter(phase_index: int) -> float:
    """Return the paper value ``1.3 * exp(0.22*c) - 1.3`` for one index."""

    if not isinstance(phase_index, int) or isinstance(phase_index, bool) or phase_index < 0:
        raise ValueError("phase_index must be a non-negative integer")
    try:
        value = PAPER_SCHEDULE_SCALE * math.exp(PAPER_SCHEDULE_RATE * phase_index) - PAPER_SCHEDULE_SCALE
    except OverflowError as exc:
        raise ValueError("phase_index produces a non-finite progression parameter") from exc
    if not math.isfinite(value):
        raise ValueError("phase_index produces a non-finite progression parameter")
    return value


def phase_indices(
    phase_count: int,
    convention: PhaseIndexConvention | str,
) -> tuple[int, ...]:
    """Return explicit zero- or one-based phase indices."""

    if not isinstance(phase_count, int) or isinstance(phase_count, bool) or phase_count < 1:
        raise ValueError("phase_count must be a positive integer")
    resolved = _resolve_enum(convention, PhaseIndexConvention, "phase_index")
    start = 0 if resolved is PhaseIndexConvention.ZERO_BASED else 1
    return tuple(range(start, start + phase_count))


def progression_parameters(
    config: ProgressionScheduleConfig,
    *,
    phase_count: int = REFERENCE_PHASE_COUNT,
) -> tuple[float, ...]:
    """Evaluate a registered schedule; alternatives match paper endpoints."""

    if not isinstance(config, ProgressionScheduleConfig):
        raise TypeError("config must be a ProgressionScheduleConfig")
    indices = phase_indices(phase_count, config.phase_index)
    paper_values = tuple(paper_exponential_parameter(index) for index in indices)
    if config.schedule is ProgressionScheduleKind.PAPER_EXPONENTIAL:
        return paper_values
    if phase_count == 1:
        return (paper_values[0],)
    start, end = paper_values[0], paper_values[-1]
    values: list[float] = []
    for offset in range(phase_count):
        fraction = offset / (phase_count - 1)
        if config.schedule is ProgressionScheduleKind.COSINE_ENDPOINT_MATCHED:
            fraction = (1.0 - math.cos(math.pi * fraction)) / 2.0
        values.append(start + (end - start) * fraction)
    return tuple(values)


def registered_schedule_kinds() -> tuple[ProgressionScheduleKind, ...]:
    """Return every schedule eligible for preregistered matched comparison."""

    return tuple(ProgressionScheduleKind)


__all__ = [
    "AnalyticalScaleGradient",
    "OperatorPrecision",
    "PAPER_SCHEDULE_RATE",
    "PAPER_SCHEDULE_SCALE",
    "PhaseIndexConvention",
    "ProgressionScheduleConfig",
    "ProgressionScheduleKind",
    "ProgressiveOperatorConfig",
    "REFERENCE_PHASE_COUNT",
    "paper_exponential_parameter",
    "phase_indices",
    "progression_parameters",
    "progressive",
    "progressive_derivative",
    "registered_schedule_kinds",
]
