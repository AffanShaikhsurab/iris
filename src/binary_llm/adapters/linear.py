"""PyTorch binary linear with explicit training and evaluation views."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from binary_llm.math import (
    DualScaleState,
    InputChannelScale,
    ProgressiveOperatorConfig,
    ScaleParameterization,
    ZeroSignRule,
    analytical_row_scales,
    binary_sign,
    dual_scaled_sign_weight,
    progressive,
    transformed_weight,
)

from .contracts import ActiveRepresentation, BINARY_BODY_ROLES, TensorRole


class TrainingActivationMode(StrEnum):
    """Explicit Stage 1 interpretation of the inverse activation transform."""

    EXPLICIT_ACTIVATION_TRANSFORM = "explicit_activation_transform"
    WEIGHT_ONLY = "weight_only"


@dataclass(frozen=True, slots=True)
class BinaryLinearConfig:
    scale_parameterization: ScaleParameterization | str
    training_activation_mode: TrainingActivationMode | str
    progressive_operator: ProgressiveOperatorConfig
    zero_sign_rule: ZeroSignRule | str
    initial_input_scale: float = 1.0
    scale_epsilon: float = 1e-6
    minimum_scale_magnitude: float = 1e-6

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "scale_parameterization",
            ScaleParameterization(self.scale_parameterization),
        )
        object.__setattr__(
            self,
            "training_activation_mode",
            TrainingActivationMode(self.training_activation_mode),
        )
        object.__setattr__(self, "zero_sign_rule", ZeroSignRule(self.zero_sign_rule))
        if not isinstance(self.progressive_operator, ProgressiveOperatorConfig):
            raise TypeError("progressive_operator must be a ProgressiveOperatorConfig")
        for name in ("initial_input_scale", "scale_epsilon", "minimum_scale_magnitude"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.scale_epsilon <= 0 or self.minimum_scale_magnitude <= 0:
            raise ValueError("scale floors must be positive")


class BinaryLinear(nn.Module):
    """A declared transformer linear with four observable representations."""

    def __init__(
        self,
        source: nn.Linear,
        *,
        tensor_name: str,
        semantic_role: TensorRole,
        config: BinaryLinearConfig,
    ) -> None:
        super().__init__()
        if not isinstance(source, nn.Linear):
            raise TypeError("source must be torch.nn.Linear")
        if not tensor_name.endswith(".weight"):
            raise ValueError("tensor_name must identify the source weight")
        if semantic_role not in BINARY_BODY_ROLES:
            raise ValueError("BinaryLinear requires a declared attention or FFN role")
        if not isinstance(config, BinaryLinearConfig):
            raise TypeError("config must be a BinaryLinearConfig")
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.tensor_name = tensor_name
        self.semantic_role = semantic_role
        self.config = config
        self.weight = source.weight
        self.bias = source.bias
        self.register_buffer("dense_reference_weight", source.weight.detach().clone())
        self.register_buffer(
            "dense_reference_bias",
            None if source.bias is None else source.bias.detach().clone(),
        )
        self.input_scale = InputChannelScale(
            self.in_features,
            parameterization=config.scale_parameterization,
            initial_value=config.initial_input_scale,
            epsilon=config.scale_epsilon,
            minimum_magnitude=config.minimum_scale_magnitude,
            dtype=source.weight.dtype,
            device=source.weight.device,
        )
        self.dual_scale = DualScaleState(
            self.out_features,
            analytical_gradient=config.progressive_operator.analytical_scale_gradient,
            dtype=source.weight.dtype,
            device=source.weight.device,
        )
        self._active_representation = ActiveRepresentation.TRAINING
        self._progression_parameter: float | None = None
        self.set_active_representation(ActiveRepresentation.TRAINING)

    @property
    def active_representation(self) -> ActiveRepresentation:
        return self._active_representation

    @property
    def progression_parameter(self) -> float | None:
        return self._progression_parameter

    @classmethod
    def from_linear(
        cls,
        source: nn.Linear,
        *,
        tensor_name: str,
        semantic_role: TensorRole,
        config: BinaryLinearConfig,
    ) -> BinaryLinear:
        return cls(
            source,
            tensor_name=tensor_name,
            semantic_role=semantic_role,
            config=config,
        )

    def set_active_representation(
        self,
        representation: ActiveRepresentation | str,
        *,
        progression_parameter: float | None = None,
    ) -> None:
        resolved = ActiveRepresentation(representation)
        if resolved is ActiveRepresentation.PROGRESSIVE:
            if progression_parameter is None:
                raise ValueError("progressive representation requires progression_parameter")
            if not math.isfinite(progression_parameter) or progression_parameter < 0:
                raise ValueError("progression_parameter must be finite and non-negative")
        elif progression_parameter is not None:
            raise ValueError("progression_parameter is only valid for progressive representation")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        if resolved is ActiveRepresentation.TRAINING:
            self.input_scale.raw_scale.requires_grad_(True)
        elif resolved is ActiveRepresentation.PROGRESSIVE:
            self.weight.requires_grad_(True)
            self.dual_scale.learned.value.requires_grad_(True)
        self._active_representation = resolved
        self._progression_parameter = (
            float(progression_parameter)
            if resolved is ActiveRepresentation.PROGRESSIVE
            else None
        )

    def transformed_weight(self) -> Tensor:
        return transformed_weight(self.weight, self.input_scale())

    def identity_input_scale_parameter(self) -> Tensor:
        """Return raw parameter bytes that evaluate to exact unit scales."""

        parameterization = self.input_scale.parameterization
        if parameterization is ScaleParameterization.POSITIVE_EXP:
            raw = torch.zeros_like(self.input_scale.raw_scale)
        elif parameterization is ScaleParameterization.POSITIVE_SOFTPLUS:
            target = torch.ones_like(self.input_scale.raw_scale)
            shifted = target - self.input_scale.epsilon
            if torch.any(shifted <= 0).item():
                raise ValueError("input scale parameterization cannot represent identity")
            raw = shifted + torch.log(-torch.expm1(-shifted))
        else:
            raw = torch.ones_like(self.input_scale.raw_scale)
        if parameterization is ScaleParameterization.POSITIVE_EXP:
            represented = torch.exp(raw)
        elif parameterization is ScaleParameterization.POSITIVE_SOFTPLUS:
            represented = F.softplus(raw) + self.input_scale.epsilon
        else:
            represented = raw.abs().clamp_min(
                self.input_scale.minimum_magnitude
            )
        if not torch.equal(represented, torch.ones_like(represented)):
            raise ValueError("input scale parameterization cannot represent exact identity")
        return raw

    def reset_input_scale_to_identity_(self) -> None:
        """Reset the Stage 1 transform without replacing its parameter."""

        with torch.no_grad():
            self.input_scale.raw_scale.copy_(self.identity_input_scale_parameter())
            if not torch.equal(
                self.input_scale(), torch.ones_like(self.input_scale.raw_scale)
            ):
                raise ValueError("input scale parameterization did not reset exactly to identity")

    def _training_weight(self) -> Tensor:
        latent = self.transformed_weight()
        binary = binary_sign(latent, self.config.zero_sign_rule)
        return latent + (binary - latent).detach()

    def _progressive_weight(self) -> Tensor:
        latent = self.transformed_weight()
        analytical = analytical_row_scales(
            latent,
            gradient=self.config.progressive_operator.analytical_scale_gradient,
        )
        if torch.any(analytical == 0).item():
            raise ValueError("progressive representation cannot normalize a zero row")
        normalized = latent / analytical.unsqueeze(1)
        assert self._progression_parameter is not None
        values = progressive(
            normalized,
            self._progression_parameter,
            config=self.config.progressive_operator,
        )
        return self.dual_scale.learned().unsqueeze(1) * analytical.unsqueeze(1) * values

    def effective_weight(self) -> Tensor:
        if self._active_representation is ActiveRepresentation.DENSE_REFERENCE:
            return self.dense_reference_weight
        if self._active_representation is ActiveRepresentation.TRAINING:
            return self._training_weight()
        if self._active_representation is ActiveRepresentation.PROGRESSIVE:
            return self._progressive_weight()
        return dual_scaled_sign_weight(
            self.transformed_weight(),
            self.dual_scale.learned(),
            gradient=self.config.progressive_operator.analytical_scale_gradient,
            zero_rule=self.config.zero_sign_rule,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.shape[-1] != self.in_features:
            raise ValueError("BinaryLinear input width does not match in_features")
        effective_inputs = inputs
        if (
            self._active_representation is ActiveRepresentation.TRAINING
            and self.config.training_activation_mode
            is TrainingActivationMode.EXPLICIT_ACTIVATION_TRANSFORM
        ):
            effective_inputs = inputs * self.input_scale()
        bias = (
            self.dense_reference_bias
            if self._active_representation is ActiveRepresentation.DENSE_REFERENCE
            else self.bias
        )
        return F.linear(effective_inputs, self.effective_weight(), bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, role={self.semantic_role.value}, "
            f"representation={self.active_representation.value}"
        )


@dataclass(frozen=True, slots=True)
class ConfiguredBinaryLinearFactory:
    config: BinaryLinearConfig

    def __call__(
        self,
        source: nn.Linear,
        *,
        tensor_name: str,
        semantic_role: TensorRole,
    ) -> BinaryLinear:
        return BinaryLinear.from_linear(
            source,
            tensor_name=tensor_name,
            semantic_role=semantic_role,
            config=self.config,
        )


__all__ = [
    "BinaryLinear",
    "BinaryLinearConfig",
    "ConfiguredBinaryLinearFactory",
    "TrainingActivationMode",
]
