"""Paired semantic-family bootstrap statistics for primary ablation metrics."""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from statistics import fmean
from typing import Iterable

from binary_llm.domain import EvaluationError, Retryability
from binary_llm.domain.ablations import BootstrapPlan, ScoreObservation


@dataclass(frozen=True, slots=True)
class PairedCaseDelta:
    case_id: str
    semantic_family_id: str
    candidate_score: float
    control_score: float
    delta: float


@dataclass(frozen=True, slots=True)
class SemanticFamilyDelta:
    semantic_family_id: str
    pair_count: int
    delta: float


@dataclass(frozen=True, slots=True)
class PairedStatistics:
    metric_id: str
    paired_delta: float
    confidence_lower: float
    confidence_upper: float
    confidence_level: float
    seed: int
    bootstrap_resamples: int
    pair_count: int
    semantic_family_count: int
    case_deltas: tuple[PairedCaseDelta, ...]
    family_deltas: tuple[SemanticFamilyDelta, ...]
    bootstrap_distribution: tuple[float, ...]


def _failure(message: str, code: str, case_ids: Iterable[str] = ()) -> EvaluationError:
    return EvaluationError(
        message,
        retryability=Retryability.AFTER_REMEDIATION,
        code=f"evaluation.{code}",
        affected_ids={"case_ids": tuple(case_ids)},
    )


def _metric_index(
    observations: Iterable[ScoreObservation], metric_id: str, arm: str
) -> dict[str, ScoreObservation]:
    selected = tuple(item for item in observations if item.metric_id == metric_id)
    if not selected:
        raise _failure(f"{arm} has no observations for metric {metric_id}", "missing_metric")
    index: dict[str, ScoreObservation] = {}
    duplicates: list[str] = []
    for item in selected:
        if item.case_id in index:
            duplicates.append(item.case_id)
        index[item.case_id] = item
    if duplicates:
        raise _failure(f"{arm} contains duplicate case observations", "duplicate_pairs", duplicates)
    return index


def _prepare_deltas(
    candidate_scores: Iterable[ScoreObservation],
    control_scores: Iterable[ScoreObservation],
    metric_id: str,
) -> tuple[tuple[PairedCaseDelta, ...], tuple[SemanticFamilyDelta, ...]]:
    candidate = _metric_index(candidate_scores, metric_id, "candidate")
    control = _metric_index(control_scores, metric_id, "control")
    missing_control = tuple(sorted(set(candidate) - set(control)))
    missing_candidate = tuple(sorted(set(control) - set(candidate)))
    if missing_control or missing_candidate:
        missing = tuple(sorted(set(missing_control) | set(missing_candidate)))
        raise _failure(
            "paired statistics require candidate and control observations for every case",
            "missing_pairs",
            missing,
        )

    case_deltas: list[PairedCaseDelta] = []
    grouped: dict[str, list[float]] = {}
    for case_id in sorted(candidate):
        candidate_item = candidate[case_id]
        control_item = control[case_id]
        if candidate_item.semantic_family_id != control_item.semantic_family_id:
            raise _failure(
                f"semantic-family mismatch for paired case {case_id}",
                "family_mismatch",
                (case_id,),
            )
        delta = float(candidate_item.score) - float(control_item.score)
        if not math.isfinite(delta):
            raise _failure(f"paired delta is non-finite for case {case_id}", "nonfinite_delta", (case_id,))
        family_id = candidate_item.semantic_family_id
        case_deltas.append(
            PairedCaseDelta(
                case_id,
                family_id,
                float(candidate_item.score),
                float(control_item.score),
                delta,
            )
        )
        grouped.setdefault(family_id, []).append(delta)

    family_deltas = tuple(
        SemanticFamilyDelta(family_id, len(values), fmean(values))
        for family_id, values in sorted(grouped.items())
    )
    return tuple(case_deltas), family_deltas


def _percentile(sorted_values: tuple[float, ...], probability: float) -> float:
    if not sorted_values:
        raise ValueError("a percentile requires at least one value")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] + fraction * (sorted_values[upper] - sorted_values[lower])


def confidence_interval(
    distribution: Iterable[float], confidence_level: float = 0.95
) -> tuple[float, float]:
    """Return a linearly interpolated equal-tail percentile interval."""

    values = tuple(sorted(float(value) for value in distribution))
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("confidence distributions must be non-empty and finite")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")
    tail = (1.0 - confidence_level) / 2.0
    return _percentile(values, tail), _percentile(values, 1.0 - tail)


def paired_bootstrap_statistics(
    candidate_scores: Iterable[ScoreObservation],
    control_scores: Iterable[ScoreObservation],
    metric_id: str,
    plan: BootstrapPlan,
) -> PairedStatistics:
    """Compute candidate-minus-control deltas and exactly 10,000 grouped resamples."""

    case_deltas, family_records = _prepare_deltas(candidate_scores, control_scores, metric_id)
    family_deltas = tuple(item.delta for item in family_records)
    family_count = len(family_deltas)
    generator = random.Random(plan.seed)
    distribution = tuple(
        fmean(family_deltas[generator.randrange(family_count)] for _ in range(family_count))
        for _ in range(plan.resamples)
    )
    lower, upper = confidence_interval(distribution, plan.confidence_level)
    return PairedStatistics(
        metric_id,
        fmean(family_deltas),
        lower,
        upper,
        plan.confidence_level,
        plan.seed,
        len(distribution),
        len(case_deltas),
        family_count,
        case_deltas,
        family_records,
        distribution,
    )


def exact_enumerated_paired_distribution(
    candidate_scores: Iterable[ScoreObservation],
    control_scores: Iterable[ScoreObservation],
    metric_id: str,
    *,
    maximum_outcomes: int = 100_000,
) -> tuple[float, ...]:
    """Enumerate every grouped bootstrap outcome for a deliberately tiny fixture."""

    if maximum_outcomes < 1:
        raise ValueError("maximum_outcomes must be positive")
    _, family_records = _prepare_deltas(candidate_scores, control_scores, metric_id)
    family_deltas = tuple(item.delta for item in family_records)
    outcome_count = len(family_deltas) ** len(family_deltas)
    if outcome_count > maximum_outcomes:
        raise ValueError(
            f"exact enumeration would produce {outcome_count} outcomes; limit is {maximum_outcomes}"
        )
    return tuple(
        fmean(outcome)
        for outcome in itertools.product(family_deltas, repeat=len(family_deltas))
    )
