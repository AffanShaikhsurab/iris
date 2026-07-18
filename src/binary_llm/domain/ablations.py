"""Immutable contracts for matched ablations and paired evaluation scores."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping

from .models import Budget, CanonicalModel, JsonValue, SUPPORTED_SCHEMA_VERSION

PAIRED_BOOTSTRAP_RESAMPLES = 10_000
PAIRED_CONFIDENCE_LEVEL = 0.95

REQUIRED_PRIMARY_ABLATION_FACTORS = frozenset(
    {
        "binary_aware_initialization",
        "progressive_schedule",
        "forward_backward_consistency",
        "analytical_scale",
        "learnable_scale",
        "teacher_guidance",
        "corpus_mix",
        "recovery_budget",
    }
)


def _require_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_schema(version: int) -> None:
    if version != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {version}")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool, StrEnum)):
        return value
    raise TypeError(f"value is not JSON-compatible: {type(value).__name__}")


class AblationClassification(StrEnum):
    COMPONENT_ABLATION = "component_ablation"
    INTERACTION_EXPERIMENT = "interaction_experiment"


@dataclass(frozen=True, slots=True)
class BootstrapPlan(CanonicalModel):
    seed: int
    resamples: int = PAIRED_BOOTSTRAP_RESAMPLES
    confidence_level: float = PAIRED_CONFIDENCE_LEVEL
    unit: str = "semantic_family"
    schema_version: int = SUPPORTED_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema(self.schema_version)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("bootstrap seed must be an integer")
        if self.resamples != PAIRED_BOOTSTRAP_RESAMPLES:
            raise ValueError("paired bootstrap plans require exactly 10,000 resamples")
        if self.confidence_level != PAIRED_CONFIDENCE_LEVEL:
            raise ValueError("paired bootstrap plans require a 95% confidence level")
        if self.unit != "semantic_family":
            raise ValueError("paired bootstrap plans must resample semantic-family groups")


@dataclass(frozen=True, slots=True)
class AblationDefinition(CanonicalModel):
    ablation_id: str
    factor: str
    candidate_experiment: str
    control_experiment: str
    invariant_fields: tuple[str, ...]
    changed_fields: tuple[str, ...]
    interaction: bool
    primary_metrics: tuple[str, ...]
    bootstrap_plan: BootstrapPlan
    schema_version: int = SUPPORTED_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema(self.schema_version)
        for name in ("ablation_id", "factor", "candidate_experiment", "control_experiment"):
            _require_text(name, getattr(self, name))
        if self.candidate_experiment == self.control_experiment:
            raise ValueError("candidate and control experiments must be distinct")
        if not self.changed_fields:
            raise ValueError("an ablation must declare at least one changed field")
        if not self.primary_metrics:
            raise ValueError("an ablation must preregister primary metrics")
        for name, values in (
            ("invariant_fields", self.invariant_fields),
            ("changed_fields", self.changed_fields),
            ("primary_metrics", self.primary_metrics),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{name} contains duplicates")
            if any(not value.startswith("/") for value in values if name != "primary_metrics"):
                raise ValueError(f"{name} must use JSON-pointer paths")
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(f"{name} must contain non-empty strings")
        if set(self.invariant_fields) & set(self.changed_fields):
            raise ValueError("invariant and changed fields must be disjoint")
        for left in self.changed_fields:
            for right in self.changed_fields:
                if left != right and right.startswith(f"{left}/"):
                    raise ValueError("changed fields cannot contain overlapping paths")


@dataclass(frozen=True, slots=True)
class AblationArm(CanonicalModel):
    experiment_id: str
    model_revision: str
    data_split_id: str
    evaluator_revision: str
    seed_set: tuple[int, ...]
    compute_budget: Budget
    settings: Mapping[str, JsonValue]
    schema_version: int = SUPPORTED_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema(self.schema_version)
        for name in ("experiment_id", "model_revision", "data_split_id", "evaluator_revision"):
            _require_text(name, getattr(self, name))
        if not self.seed_set or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in self.seed_set):
            raise ValueError("seed_set must contain integer seeds")
        if len(self.seed_set) != len(set(self.seed_set)):
            raise ValueError("seed_set contains duplicates")
        object.__setattr__(self, "settings", _freeze(self.settings))


@dataclass(frozen=True, slots=True)
class ScoreObservation(CanonicalModel):
    case_id: str
    semantic_family_id: str
    metric_id: str
    score: float
    schema_version: int = SUPPORTED_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema(self.schema_version)
        for name in ("case_id", "semantic_family_id", "metric_id"):
            _require_text(name, getattr(self, name))
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise TypeError("score must be numeric")
        if not math.isfinite(float(self.score)):
            raise ValueError("score must be finite")
