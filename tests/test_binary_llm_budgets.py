from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from binary_llm.domain import Budget, CheckpointBoundary
from binary_llm.orchestration import (
    BudgetDimension,
    BudgetMonitor,
    BudgetUsage,
    causal_training_tokens,
)


@dataclass
class FakeClock:
    value: float = 0.0

    def __call__(self) -> float:
        return self.value


def _budget(**changes) -> Budget:
    values = {
        "max_tokens": 6,
        "max_optimizer_steps": 2,
        "max_wall_seconds": 5,
        "max_billable_cost": 1.0,
        "checkpoint_boundary": CheckpointBoundary.OPTIMIZER_STEP,
    }
    values.update(changes)
    return Budget(**values)


def test_causal_tokens_use_shifted_labels_and_ignore_mask() -> None:
    inputs = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    labels = torch.tensor([[1, 2, -100, 4], [4, -100, 2, -100]])

    assert causal_training_tokens(inputs) == 6
    assert causal_training_tokens(inputs, labels) == 3


def test_budget_boundaries_are_inclusive_and_crossings_are_typed() -> None:
    clock = FakeClock()
    cost = FakeClock()
    monitor = BudgetMonitor(
        _budget(),
        prior_usage=BudgetUsage(consumed_tokens=3, optimizer_steps=1),
        clock=clock,
        cost_meter=cost,
    )

    assert (
        monitor.crossing(phase="pre_step", proposed_tokens=3, proposed_steps=1) is None
    )
    crossing = monitor.crossing(phase="pre_step", proposed_tokens=4, proposed_steps=1)
    assert crossing is not None
    assert crossing.dimension is BudgetDimension.TOKENS

    monitor.record_step(3)
    clock.value = 5
    cost.value = 1
    assert monitor.crossing(phase="post_step") is None
    clock.value = 5.01
    crossing = monitor.crossing(phase="post_step")
    assert crossing is not None
    failure = monitor.failure(crossing)
    assert failure.code == "budget.wall_seconds_exceeded"
    assert failure.context["crossed_dimension"] == "wall_seconds"
    assert failure.context["phase"] == "post_step"
    assert failure.context["usage"]["consumed_tokens"] == 6


def test_crossing_reports_every_simultaneously_exceeded_dimension_in_order() -> None:
    clock = FakeClock()
    cost = FakeClock()
    monitor = BudgetMonitor(
        _budget(),
        prior_usage=BudgetUsage(consumed_tokens=7, optimizer_steps=3),
        clock=clock,
        cost_meter=cost,
    )
    clock.value = 6.0
    cost.value = 2.0

    crossing = monitor.crossing(phase="post_step")

    assert crossing is not None
    assert crossing.dimension is BudgetDimension.TOKENS
    assert tuple(item.dimension for item in crossing.exceedances) == (
        BudgetDimension.TOKENS,
        BudgetDimension.OPTIMIZER_STEPS,
        BudgetDimension.WALL_SECONDS,
        BudgetDimension.BILLABLE_COST,
    )
    failure = monitor.failure(crossing)
    assert failure.context["crossed_dimensions"] == [
        {"dimension": "tokens", "observed": 7, "limit": 6},
        {"dimension": "optimizer_steps", "observed": 3, "limit": 2},
        {"dimension": "wall_seconds", "observed": 6.0, "limit": 5},
        {"dimension": "billable_cost", "observed": 2.0, "limit": 1.0},
    ]
    assert failure.context["usage"] == {
        "wall_seconds": 6.0,
        "accelerator_seconds": None,
        "consumed_tokens": 7,
        "optimizer_steps": 3,
        "billable_cost": 2.0,
        "interrupted_attempts": 0,
    }


def test_resumed_cost_meter_adds_only_fresh_attempt_delta() -> None:
    meter = FakeClock(100.0)
    monitor = BudgetMonitor(
        _budget(max_billable_cost=10.0),
        prior_usage=BudgetUsage(billable_cost=4.0),
        clock=FakeClock(),
        cost_meter=meter,
    )

    assert monitor.usage.billable_cost == 4.0
    meter.value = 101.5
    assert monitor.usage.billable_cost == pytest.approx(5.5)


def test_budget_usage_dominance_is_component_wise() -> None:
    checkpoint = BudgetUsage(
        consumed_tokens=6,
        optimizer_steps=2,
        wall_seconds=4.0,
        accelerator_seconds=3.0,
        billable_cost=0.5,
        interrupted_attempts=1,
    )
    assert checkpoint.dominates(checkpoint)
    assert BudgetUsage(
        consumed_tokens=7,
        optimizer_steps=3,
        wall_seconds=5.0,
        accelerator_seconds=4.0,
        billable_cost=0.6,
        interrupted_attempts=2,
    ).dominates(checkpoint)
    for lower in (
        BudgetUsage(5, 2, 4.0, 3.0, 0.5, 1),
        BudgetUsage(6, 1, 4.0, 3.0, 0.5, 1),
        BudgetUsage(6, 2, 3.0, 3.0, 0.5, 1),
        BudgetUsage(6, 2, 4.0, 2.0, 0.5, 1),
        BudgetUsage(6, 2, 4.0, 3.0, 0.4, 1),
        BudgetUsage(6, 2, 4.0, 3.0, 0.5, 0),
    ):
        assert not lower.dominates(checkpoint)


def test_explicit_cost_observations_are_cumulative_without_a_cost_model() -> None:
    monitor = BudgetMonitor(_budget(), clock=FakeClock())
    monitor.observe_cost(0.4)
    monitor.observe_cost(0.6)
    assert monitor.usage.billable_cost == pytest.approx(1.0)
    assert monitor.crossing(phase="post_step") is None
    monitor.observe_cost(0.01)
    crossing = monitor.crossing(phase="post_step")
    assert crossing is not None
    assert crossing.dimension is BudgetDimension.BILLABLE_COST
