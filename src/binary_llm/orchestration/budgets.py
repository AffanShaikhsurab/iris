"""Shared deterministic runtime-budget accounting for training backends."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Callable

from torch import Tensor

from binary_llm.domain import Budget, BudgetExceeded, Retryability


class BudgetDimension(StrEnum):
    TOKENS = "tokens"
    OPTIMIZER_STEPS = "optimizer_steps"
    WALL_SECONDS = "wall_seconds"
    BILLABLE_COST = "billable_cost"


@dataclass(frozen=True, slots=True)
class BudgetUsage:
    consumed_tokens: int = 0
    optimizer_steps: int = 0
    wall_seconds: float = 0.0
    accelerator_seconds: float | None = None
    billable_cost: float = 0.0
    interrupted_attempts: int = 0

    def __post_init__(self) -> None:
        numeric = (self.wall_seconds, self.billable_cost)
        if self.accelerator_seconds is not None:
            numeric += (self.accelerator_seconds,)
        if any(not math.isfinite(value) or value < 0 for value in numeric):
            raise ValueError("runtime usage must be finite and non-negative")
        if min(self.consumed_tokens, self.optimizer_steps, self.interrupted_attempts) < 0:
            raise ValueError("runtime usage counts must be non-negative")

    def resource_values(self) -> dict[str, int | float]:
        """Return fields directly compatible with registry resource evidence."""

        return {
            "wall_seconds": self.wall_seconds,
            "accelerator_seconds": self.accelerator_seconds or 0.0,
            "consumed_tokens": self.consumed_tokens,
            "optimizer_steps": self.optimizer_steps,
            "billable_cost": self.billable_cost,
        }

    def evidence_values(self) -> dict[str, int | float | None]:
        """Return complete cumulative usage, including non-ledger evidence fields."""

        return {
            "wall_seconds": self.wall_seconds,
            "accelerator_seconds": self.accelerator_seconds,
            "consumed_tokens": self.consumed_tokens,
            "optimizer_steps": self.optimizer_steps,
            "billable_cost": self.billable_cost,
            "interrupted_attempts": self.interrupted_attempts,
        }

    def dominates(self, other: BudgetUsage) -> bool:
        """Whether this cumulative observation includes every component of ``other``."""

        accelerator_includes = (
            other.accelerator_seconds is None
            or (
                self.accelerator_seconds is not None
                and self.accelerator_seconds >= other.accelerator_seconds
            )
        )
        return (
            self.consumed_tokens >= other.consumed_tokens
            and self.optimizer_steps >= other.optimizer_steps
            and self.wall_seconds >= other.wall_seconds
            and accelerator_includes
            and self.billable_cost >= other.billable_cost
            and self.interrupted_attempts >= other.interrupted_attempts
        )


@dataclass(frozen=True, slots=True)
class BudgetExceedance:
    dimension: BudgetDimension
    observed: int | float
    limit: int | float


@dataclass(frozen=True, slots=True)
class BudgetCrossing:
    exceedances: tuple[BudgetExceedance, ...]
    phase: str

    def __post_init__(self) -> None:
        if not self.exceedances:
            raise ValueError("budget crossing requires at least one exceeded dimension")
        dimensions = tuple(item.dimension for item in self.exceedances)
        if len(dimensions) != len(set(dimensions)):
            raise ValueError("budget crossing dimensions must be unique")

    @property
    def primary(self) -> BudgetExceedance:
        return self.exceedances[0]

    @property
    def dimension(self) -> BudgetDimension:
        return self.primary.dimension

    @property
    def observed(self) -> int | float:
        return self.primary.observed

    @property
    def limit(self) -> int | float:
        return self.primary.limit


def causal_training_tokens(input_ids: Tensor, labels: Tensor | None = None) -> int:
    """Count shifted next-token targets, excluding ignored labels."""

    resolved = input_ids if labels is None else labels
    if resolved.ndim != 2 or resolved.shape != input_ids.shape:
        raise ValueError("labels must match two-dimensional input_ids")
    return int((resolved[:, 1:] != -100).sum().item())


class BudgetMonitor:
    """Accumulate observed usage without estimating unobserved cost."""

    def __init__(
        self,
        budget: Budget,
        *,
        prior_usage: BudgetUsage | None = None,
        clock: Callable[[], float] = time.monotonic,
        cost_meter: Callable[[], float] | None = None,
    ) -> None:
        if not isinstance(budget, Budget):
            raise TypeError("budget must be Budget")
        self.budget = budget
        self._prior = prior_usage or BudgetUsage()
        self._clock = clock
        self._started = float(clock())
        self._tokens = self._prior.consumed_tokens
        self._steps = self._prior.optimizer_steps
        self._accelerator = self._prior.accelerator_seconds
        self._cost = self._prior.billable_cost
        self._interrupted = self._prior.interrupted_attempts
        self._cost_meter = cost_meter
        self._meter_origin = None if cost_meter is None else self._read_meter()

    def _read_meter(self) -> float:
        assert self._cost_meter is not None
        value = float(self._cost_meter())
        if not math.isfinite(value) or value < 0:
            raise ValueError("billable-cost observations must be finite and non-negative")
        return value

    @property
    def usage(self) -> BudgetUsage:
        elapsed = float(self._clock()) - self._started
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("monotonic clock moved backwards or returned a non-finite value")
        cost = self._cost
        if self._cost_meter is not None:
            assert self._meter_origin is not None
            current = self._read_meter()
            if current < self._meter_origin:
                raise ValueError("billable-cost meter moved backwards")
            cost += current - self._meter_origin
        return BudgetUsage(
            consumed_tokens=self._tokens,
            optimizer_steps=self._steps,
            wall_seconds=self._prior.wall_seconds + elapsed,
            accelerator_seconds=self._accelerator,
            billable_cost=cost,
            interrupted_attempts=self._interrupted,
        )

    def observe_cost(self, amount: float) -> BudgetUsage:
        if self._cost_meter is not None:
            raise ValueError("explicit observations cannot be mixed with a cost meter")
        if not math.isfinite(amount) or amount < 0:
            raise ValueError("billable-cost observations must be finite and non-negative")
        self._cost += amount
        return self.usage

    def observe_accelerator_seconds(self, amount: float) -> BudgetUsage:
        if not math.isfinite(amount) or amount < 0:
            raise ValueError("accelerator seconds must be finite and non-negative")
        self._accelerator = (self._accelerator or 0.0) + amount
        return self.usage

    def record_step(self, consumed_tokens: int) -> BudgetUsage:
        if isinstance(consumed_tokens, bool) or consumed_tokens < 0:
            raise ValueError("consumed_tokens must be a non-negative integer")
        self._tokens += consumed_tokens
        self._steps += 1
        return self.usage

    def record_interrupted_attempt(self) -> BudgetUsage:
        self._interrupted += 1
        return self.usage

    def crossing(
        self,
        *,
        phase: str,
        proposed_tokens: int = 0,
        proposed_steps: int = 0,
    ) -> BudgetCrossing | None:
        usage = self.usage
        checks = (
            (
                BudgetDimension.TOKENS,
                usage.consumed_tokens + proposed_tokens,
                self.budget.max_tokens,
            ),
            (
                BudgetDimension.OPTIMIZER_STEPS,
                usage.optimizer_steps + proposed_steps,
                self.budget.max_optimizer_steps,
            ),
            (BudgetDimension.WALL_SECONDS, usage.wall_seconds, self.budget.max_wall_seconds),
            (BudgetDimension.BILLABLE_COST, usage.billable_cost, self.budget.max_billable_cost),
        )
        exceedances = tuple(
            BudgetExceedance(dimension, observed, limit)
            for dimension, observed, limit in checks
            if observed > limit
        )
        return None if not exceedances else BudgetCrossing(exceedances, phase)

    def failure(self, crossing: BudgetCrossing) -> BudgetExceeded:
        primary = crossing.primary
        return BudgetExceeded(
            f"{primary.dimension.value} budget exceeded",
            retryability=Retryability.FROM_CHECKPOINT,
            code=f"budget.{primary.dimension.value}_exceeded",
            context={
                "crossed_dimension": primary.dimension.value,
                "crossed_dimensions": [
                    {
                        "dimension": item.dimension.value,
                        "observed": item.observed,
                        "limit": item.limit,
                    }
                    for item in crossing.exceedances
                ],
                "observed": primary.observed,
                "limit": primary.limit,
                "phase": crossing.phase,
                "usage": self.usage.evidence_values(),
            },
        )


__all__ = [
    "BudgetCrossing",
    "BudgetDimension",
    "BudgetExceedance",
    "BudgetMonitor",
    "BudgetUsage",
    "causal_training_tokens",
]
