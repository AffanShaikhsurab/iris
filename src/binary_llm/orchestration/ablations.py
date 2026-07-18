"""Pure matched-ablation validation and interaction classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from binary_llm.domain.ablations import (
    REQUIRED_PRIMARY_ABLATION_FACTORS,
    AblationArm,
    AblationClassification,
    AblationDefinition,
)

_MISSING = object()


def _tokens(path: str) -> tuple[str, ...]:
    if not path.startswith("/"):
        raise ValueError("field paths must be JSON pointers")
    if path == "/":
        return ("",)
    return tuple(token.replace("~1", "/").replace("~0", "~") for token in path[1:].split("/"))


def _lookup(settings: Mapping[str, Any], path: str) -> Any:
    value: Any = settings
    for token in _tokens(path):
        if not isinstance(value, Mapping) or token not in value:
            return _MISSING
        value = value[token]
    return value


def _escape(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def _leaf_differences(candidate: Any, control: Any, path: str = "") -> set[str]:
    if isinstance(candidate, Mapping) and isinstance(control, Mapping):
        differences: set[str] = set()
        for key in set(candidate) | set(control):
            child = f"{path}/{_escape(str(key))}"
            differences.update(
                _leaf_differences(candidate.get(key, _MISSING), control.get(key, _MISSING), child)
            )
        return differences
    return set() if candidate == control else {path or "/"}


def _covered(path: str, declared: str) -> bool:
    return path == declared or path.startswith(f"{declared}/")


@dataclass(frozen=True, slots=True)
class MatchedComparison:
    ablation_id: str
    factor: str
    candidate_experiment: str
    control_experiment: str
    classification: AblationClassification
    changed_fields: tuple[str, ...]
    primary_metrics: tuple[str, ...]


def validate_matched_comparison(
    definition: AblationDefinition,
    candidate: AblationArm,
    control: AblationArm,
) -> MatchedComparison:
    """Validate matching, declared changes, and the component/interaction label."""

    if candidate.experiment_id != definition.candidate_experiment:
        raise ValueError("candidate arm does not match the ablation definition")
    if control.experiment_id != definition.control_experiment:
        raise ValueError("control arm does not match the ablation definition")
    required_matches = {
        "model revision": (candidate.model_revision, control.model_revision),
        "data split": (candidate.data_split_id, control.data_split_id),
        "evaluator": (candidate.evaluator_revision, control.evaluator_revision),
        "seed set": (candidate.seed_set, control.seed_set),
        "compute budget": (candidate.compute_budget, control.compute_budget),
    }
    mismatches = tuple(name for name, pair in required_matches.items() if pair[0] != pair[1])
    if mismatches:
        raise ValueError(f"matched comparison differs in required invariants: {', '.join(mismatches)}")

    for path in definition.invariant_fields:
        candidate_value = _lookup(candidate.settings, path)
        control_value = _lookup(control.settings, path)
        if candidate_value is _MISSING or control_value is _MISSING:
            raise ValueError(f"declared invariant field is missing: {path}")
        if candidate_value != control_value:
            raise ValueError(f"declared invariant field differs: {path}")

    for path in definition.changed_fields:
        candidate_value = _lookup(candidate.settings, path)
        control_value = _lookup(control.settings, path)
        if candidate_value is _MISSING or control_value is _MISSING:
            raise ValueError(f"declared changed field is missing: {path}")
        if candidate_value == control_value:
            raise ValueError(f"declared changed field does not differ: {path}")

    actual_leaves = _leaf_differences(candidate.settings, control.settings)
    undeclared = tuple(
        sorted(path for path in actual_leaves if not any(_covered(path, item) for item in definition.changed_fields))
    )
    if undeclared:
        raise ValueError(f"comparison has undeclared changed fields: {', '.join(undeclared)}")

    classification = (
        AblationClassification.COMPONENT_ABLATION
        if len(definition.changed_fields) == 1
        else AblationClassification.INTERACTION_EXPERIMENT
    )
    expected_interaction = classification is AblationClassification.INTERACTION_EXPERIMENT
    if definition.interaction != expected_interaction:
        label = "interaction experiment" if expected_interaction else "component ablation"
        raise ValueError(f"comparison must be labeled as an {label}")
    return MatchedComparison(
        definition.ablation_id,
        definition.factor,
        candidate.experiment_id,
        control.experiment_id,
        classification,
        definition.changed_fields,
        definition.primary_metrics,
    )


def validate_primary_ablation_suite(
    comparisons: Iterable[MatchedComparison],
) -> tuple[MatchedComparison, ...]:
    """Require one valid component comparison for every preregistered primary factor."""

    items = tuple(comparisons)
    primary = {
        item.factor
        for item in items
        if item.classification is AblationClassification.COMPONENT_ABLATION
    }
    missing = tuple(sorted(REQUIRED_PRIMARY_ABLATION_FACTORS - primary))
    if missing:
        raise ValueError(f"primary ablation suite is missing factors: {', '.join(missing)}")
    return items
