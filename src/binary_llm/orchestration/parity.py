"""Fail-closed packed-runtime trace capture and parity coordination."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import torch
from torch import Tensor, nn

from binary_llm.domain import (
    ParityError,
    Retryability,
    ToleranceDefinition,
    ToleranceSet,
)
from binary_llm.export import PackedArtifactRef, RuntimeMemoryEvidence

BINARY_OUTPUT_METRIC = "runtime.binary_linear_outputs"
SELECTED_LAYER_METRIC = "runtime.selected_layer_outputs"
LOGIT_METRIC = "runtime.logits"
GREEDY_TOKEN_METRIC = "runtime.greedy_tokens"
TOOL_DECISION_METRIC = "runtime.tool_decisions"
_REQUIRED_METRICS = (
    BINARY_OUTPUT_METRIC,
    SELECTED_LAYER_METRIC,
    LOGIT_METRIC,
    GREEDY_TOKEN_METRIC,
    TOOL_DECISION_METRIC,
)


@dataclass(frozen=True, slots=True)
class ParityTrace:
    binary_linear_outputs: Mapping[str, Tensor]
    selected_layer_outputs: Mapping[str, Tensor]
    logits: Tensor
    greedy_tokens: tuple[tuple[int, ...], ...]
    tool_decisions: tuple[str, ...]

    def __post_init__(self) -> None:
        binary = _freeze_tensor_mapping("binary linear", self.binary_linear_outputs)
        layers = _freeze_tensor_mapping("selected layer", self.selected_layer_outputs)
        _validate_tensor("logits", self.logits)
        if not self.greedy_tokens or any(not row for row in self.greedy_tokens):
            raise ValueError("parity traces require non-empty greedy token sequences")
        if not self.tool_decisions or any(not isinstance(item, str) for item in self.tool_decisions):
            raise ValueError("parity traces require explicit tool decision strings")
        object.__setattr__(self, "binary_linear_outputs", binary)
        object.__setattr__(self, "selected_layer_outputs", layers)


def _validate_tensor(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} output must be a floating torch.Tensor")
    if value.numel() < 1 or not torch.isfinite(value).all().item():
        raise ValueError(f"{name} output must be non-empty and finite")


def _freeze_tensor_mapping(name: str, values: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
    if not values:
        raise ValueError(f"parity traces require at least one {name} output")
    frozen: dict[str, Tensor] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} output names must be explicit")
        _validate_tensor(f"{name} {key}", value)
        frozen[key] = value
    return MappingProxyType(frozen)


@dataclass(frozen=True, slots=True)
class NumericalParityStats:
    element_count: int
    maximum_absolute_error: float
    maximum_relative_error: float
    absolute_tolerance: float
    relative_tolerance: float
    passed: bool


@dataclass(frozen=True, slots=True)
class ParityEvidence:
    checkpoint_id: str
    packed_artifact_id: str
    training_operator_revision: str
    exporter_revision: str
    runtime_revision: str
    tolerance_set_ref: str
    decoded_signs_exact: bool
    scale_error_max: float
    binary_linear_error_stats: Mapping[str, NumericalParityStats]
    layer_error_stats: Mapping[str, NumericalParityStats]
    logit_error_stats: NumericalParityStats
    greedy_sequences_exact: bool
    iris_decisions_exact: bool
    persistent_dense_copy_detected: bool
    memory: RuntimeMemoryEvidence
    traces_ref: tuple[str, ...]
    failures: tuple[str, ...]
    passed: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "binary_linear_error_stats",
            MappingProxyType(dict(self.binary_linear_error_stats)),
        )
        object.__setattr__(
            self, "layer_error_stats", MappingProxyType(dict(self.layer_error_stats))
        )


def _module_at(model: nn.Module, path: str) -> nn.Module:
    try:
        return model.get_submodule(path)
    except AttributeError as error:
        raise ValueError(f"trace module path is missing: {path}") from error


def _greedy_generate(model: nn.Module, input_ids: Tensor, steps: int) -> Tensor:
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("greedy_steps must be a positive integer")
    generated = input_ids.detach().cpu().contiguous().clone()
    with torch.no_grad():
        for _ in range(steps):
            logits = model(generated)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=1)
    return generated


def capture_parity_trace(
    model: nn.Module,
    input_ids: Tensor,
    *,
    binary_linear_names: Sequence[str],
    selected_layer_names: Sequence[str],
    greedy_steps: int,
    tool_decision: Callable[[Tensor], Sequence[str]],
) -> ParityTrace:
    """Capture fixed-prompt numeric and deterministic outputs without repairing text."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch module")
    if not isinstance(input_ids, Tensor) or input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    if not binary_linear_names or not selected_layer_names:
        raise ValueError("binary and selected layer trace inventories must be non-empty")
    binary_outputs: dict[str, Tensor] = {}
    layer_outputs: dict[str, Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def hook_into(target: dict[str, Tensor], name: str) -> Callable[..., None]:
        def capture(module: nn.Module, arguments: tuple[object, ...], output: object) -> None:
            if not isinstance(output, Tensor):
                raise TypeError(f"trace output must be a tensor for {name}")
            target[name] = output.detach().cpu().clone()
        return capture

    try:
        for name in binary_linear_names:
            handles.append(_module_at(model, name).register_forward_hook(hook_into(binary_outputs, name)))
        for name in selected_layer_names:
            handles.append(_module_at(model, name).register_forward_hook(hook_into(layer_outputs, name)))
        with torch.no_grad():
            logits = model(input_ids).detach().cpu().clone()
    finally:
        for handle in handles:
            handle.remove()
    generated = _greedy_generate(model, input_ids, greedy_steps)
    decisions = tuple(tool_decision(generated.detach().cpu()))
    return ParityTrace(
        binary_linear_outputs=binary_outputs,
        selected_layer_outputs=layer_outputs,
        logits=logits,
        greedy_tokens=tuple(tuple(int(token) for token in row) for row in generated.tolist()),
        tool_decisions=decisions,
    )


def _registered_tolerances(tolerances: ToleranceSet) -> Mapping[str, ToleranceDefinition]:
    if not isinstance(tolerances, ToleranceSet):
        raise TypeError("tolerances must be a preregistered ToleranceSet")
    by_metric: dict[str, ToleranceDefinition] = {}
    for definition in tolerances.definitions:
        if definition.metric_path in by_metric:
            raise ParityError(
                "parity metric has multiple preregistered tolerance definitions",
                retryability=Retryability.NEVER,
                code="parity.duplicate_tolerance",
                affected_ids={"tolerance_set_ids": (tolerances.tolerance_set_id,)},
                context={"metric_path": definition.metric_path},
            )
        by_metric[definition.metric_path] = definition
    missing = tuple(metric for metric in _REQUIRED_METRICS if metric not in by_metric)
    if missing:
        raise ParityError(
            "runtime parity tolerances must be preregistered for every comparison level",
            retryability=Retryability.NEVER,
            code="parity.missing_tolerance",
            affected_ids={"tolerance_set_ids": (tolerances.tolerance_set_id,)},
            context={"missing_metric_paths": list(missing)},
        )
    for metric in (BINARY_OUTPUT_METRIC, SELECTED_LAYER_METRIC, LOGIT_METRIC):
        if by_metric[metric].exact:
            raise ValueError(f"{metric} requires numeric absolute/relative tolerances")
    for metric in (GREEDY_TOKEN_METRIC, TOOL_DECISION_METRIC):
        if not by_metric[metric].exact:
            raise ValueError(f"{metric} must be preregistered as exact")
    return MappingProxyType(by_metric)


def _numeric_stats(
    expected: Tensor,
    actual: Tensor,
    tolerance: ToleranceDefinition,
) -> NumericalParityStats:
    if expected.shape != actual.shape:
        return NumericalParityStats(
            0, math.inf, math.inf,
            float(tolerance.absolute or 0.0), float(tolerance.relative or 0.0), False,
        )
    expected64 = expected.detach().cpu().to(torch.float64)
    actual64 = actual.detach().cpu().to(torch.float64)
    difference = torch.abs(actual64 - expected64)
    absolute = float(difference.max().item()) if difference.numel() else 0.0
    denominator = torch.abs(expected64)
    relative_values = torch.where(
        denominator > 0,
        difference / denominator,
        torch.where(difference == 0, torch.zeros_like(difference), torch.full_like(difference, math.inf)),
    )
    relative = float(relative_values.max().item()) if difference.numel() else 0.0
    atol = float(tolerance.absolute or 0.0)
    rtol = float(tolerance.relative or 0.0)
    passed = bool(torch.all(difference <= atol + rtol * denominator).item())
    return NumericalParityStats(expected.numel(), absolute, relative, atol, rtol, passed)


def _compare_mapping(
    expected: Mapping[str, Tensor],
    actual: Mapping[str, Tensor],
    tolerance: ToleranceDefinition,
) -> Mapping[str, NumericalParityStats]:
    names = sorted(set(expected) | set(actual))
    stats: dict[str, NumericalParityStats] = {}
    for name in names:
        if name not in expected or name not in actual:
            stats[name] = NumericalParityStats(
                0, math.inf, math.inf,
                float(tolerance.absolute or 0.0), float(tolerance.relative or 0.0), False,
            )
        else:
            stats[name] = _numeric_stats(expected[name], actual[name], tolerance)
    return MappingProxyType(stats)


class ParityCoordinator:
    """Compare all preregistered levels and retain complete failure evidence."""

    def coordinate(
        self,
        *,
        artifact: PackedArtifactRef,
        training_operator_revision: str,
        tolerances: ToleranceSet,
        training_trace: ParityTrace,
        packed_trace: ParityTrace,
        memory: RuntimeMemoryEvidence,
        decoded_signs_exact: bool,
        scale_error_max: float,
        traces_ref: tuple[str, ...],
    ) -> ParityEvidence:
        if not training_operator_revision or not traces_ref or any(not item for item in traces_ref):
            raise ValueError("training operator revision and trace references must be explicit")
        if not isinstance(scale_error_max, (int, float)) or isinstance(scale_error_max, bool):
            raise TypeError("scale_error_max must be numeric")
        scale_error = float(scale_error_max)
        if not math.isfinite(scale_error) or scale_error < 0:
            raise ValueError("scale_error_max must be finite and non-negative")
        registered = _registered_tolerances(tolerances)
        binary_stats = _compare_mapping(
            training_trace.binary_linear_outputs,
            packed_trace.binary_linear_outputs,
            registered[BINARY_OUTPUT_METRIC],
        )
        layer_stats = _compare_mapping(
            training_trace.selected_layer_outputs,
            packed_trace.selected_layer_outputs,
            registered[SELECTED_LAYER_METRIC],
        )
        logit_stats = _numeric_stats(
            training_trace.logits, packed_trace.logits, registered[LOGIT_METRIC]
        )
        greedy_exact = training_trace.greedy_tokens == packed_trace.greedy_tokens
        decisions_exact = training_trace.tool_decisions == packed_trace.tool_decisions

        failures: list[str] = []
        if not decoded_signs_exact:
            failures.append("decoded_signs_not_exact")
        if scale_error > artifact.scale_tolerance:
            failures.append("scale_error_exceeded")
        failures.extend(
            f"binary_linear:{name}"
            for name, stats in binary_stats.items() if not stats.passed
        )
        failures.extend(
            f"selected_layer:{name}"
            for name, stats in layer_stats.items() if not stats.passed
        )
        if not logit_stats.passed:
            failures.append("logits")
        if not greedy_exact:
            failures.append("greedy_tokens")
        if not decisions_exact:
            failures.append("tool_decisions")
        if memory.persistent_dense_copy_detected:
            failures.append("persistent_dense_weight_copy")
        if not memory.temporary_expansion_within_limit:
            failures.append("temporary_expansion_limit")
        return ParityEvidence(
            checkpoint_id=artifact.source_checkpoint_id,
            packed_artifact_id=artifact.artifact_id,
            training_operator_revision=training_operator_revision,
            exporter_revision=artifact.exporter_revision,
            runtime_revision=artifact.runtime_revision,
            tolerance_set_ref=tolerances.tolerance_set_id,
            decoded_signs_exact=decoded_signs_exact,
            scale_error_max=scale_error,
            binary_linear_error_stats=binary_stats,
            layer_error_stats=layer_stats,
            logit_error_stats=logit_stats,
            greedy_sequences_exact=greedy_exact,
            iris_decisions_exact=decisions_exact,
            persistent_dense_copy_detected=memory.persistent_dense_copy_detected,
            memory=memory,
            traces_ref=traces_ref,
            failures=tuple(failures),
            passed=not failures,
        )


__all__ = [
    "BINARY_OUTPUT_METRIC",
    "GREEDY_TOKEN_METRIC",
    "LOGIT_METRIC",
    "SELECTED_LAYER_METRIC",
    "TOOL_DECISION_METRIC",
    "NumericalParityStats",
    "ParityCoordinator",
    "ParityEvidence",
    "ParityTrace",
    "capture_parity_trace",
]
