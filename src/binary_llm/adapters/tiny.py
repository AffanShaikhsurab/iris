"""Deterministic CPU-sized causal model and exact role-based adapter."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .contracts import (
    ActiveRepresentation,
    BinaryLinearFactory,
    BinaryScope,
    NamedActiveRepresentation,
    TensorDescriptor,
    TensorRole,
    TensorScope,
)
from .linear import BinaryLinear


@dataclass(frozen=True, slots=True)
class TinyCausalLMConfig:
    vocab_size: int = 32
    hidden_size: int = 16
    intermediate_size: int = 32
    max_sequence_length: int = 16

    def __post_init__(self) -> None:
        for name in (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "max_sequence_length",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 2:
                raise ValueError(f"{name} must be an integer of at least two")


class TinyCausalSelfAttention(nn.Module):
    def __init__(self, config: TinyCausalLMConfig) -> None:
        super().__init__()
        width = config.hidden_size
        self.q_proj = nn.Linear(width, width, bias=True)
        self.k_proj = nn.Linear(width, width, bias=True)
        self.v_proj = nn.Linear(width, width, bias=True)
        self.out_proj = nn.Linear(width, width, bias=True)
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(
                    config.max_sequence_length,
                    config.max_sequence_length,
                    dtype=torch.bool,
                ),
                diagonal=1,
            ),
        )

    def forward(self, hidden: Tensor) -> Tensor:
        sequence_length = hidden.shape[1]
        query = self.q_proj(hidden)
        key = self.k_proj(hidden)
        value = self.v_proj(hidden)
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(hidden.shape[-1])
        scores = scores.masked_fill(
            self.causal_mask[:sequence_length, :sequence_length],
            torch.finfo(scores.dtype).min,
        )
        probabilities = torch.softmax(scores, dim=-1)
        return self.out_proj(torch.matmul(probabilities, value))


class TinyFeedForward(nn.Module):
    def __init__(self, config: TinyCausalLMConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class TinyTransformerBlock(nn.Module):
    def __init__(self, config: TinyCausalLMConfig) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(config.hidden_size)
        self.attention = TinyCausalSelfAttention(config)
        self.post_attention_norm = nn.LayerNorm(config.hidden_size)
        self.feed_forward = TinyFeedForward(config)

    def forward(self, hidden: Tensor) -> Tensor:
        hidden = hidden + self.attention(self.input_norm(hidden))
        return hidden + self.feed_forward(self.post_attention_norm(hidden))


class TinyCausalLM(nn.Module):
    """One-block causal LM used only for deterministic operator integration."""

    def __init__(self, config: TinyCausalLMConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.block = TinyTransformerBlock(config)
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: Tensor) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.shape[1] > self.config.max_sequence_length:
            raise ValueError("input sequence exceeds max_sequence_length")
        hidden = self.token_embedding(input_ids)
        hidden = self.block(hidden)
        return self.lm_head(self.final_norm(hidden))


def build_seeded_tiny_causal_lm(
    *,
    seed: int,
    config: TinyCausalLMConfig | None = None,
) -> TinyCausalLM:
    """Build deterministically on CPU without changing the caller's RNG stream."""

    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    resolved_config = config or TinyCausalLMConfig()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return TinyCausalLM(resolved_config).cpu()


_DECLARED_LINEAR_ROLES: Mapping[str, TensorRole] = MappingProxyType(
    {
        "block.attention.q_proj": TensorRole.ATTENTION_QUERY,
        "block.attention.k_proj": TensorRole.ATTENTION_KEY,
        "block.attention.v_proj": TensorRole.ATTENTION_VALUE,
        "block.attention.out_proj": TensorRole.ATTENTION_OUTPUT,
        "block.feed_forward.gate_proj": TensorRole.FFN_GATE,
        "block.feed_forward.up_proj": TensorRole.FFN_UP,
        "block.feed_forward.down_proj": TensorRole.FFN_DOWN,
    }
)

_EXCLUDED_PARAMETER_ROLES: Mapping[str, TensorRole] = MappingProxyType(
    {
        "token_embedding.weight": TensorRole.TOKEN_EMBEDDING,
        "block.input_norm.weight": TensorRole.NORMALIZATION,
        "block.input_norm.bias": TensorRole.BIAS,
        "block.post_attention_norm.weight": TensorRole.NORMALIZATION,
        "block.post_attention_norm.bias": TensorRole.BIAS,
        "final_norm.weight": TensorRole.NORMALIZATION,
        "final_norm.bias": TensorRole.BIAS,
        "lm_head.weight": TensorRole.LANGUAGE_MODEL_HEAD,
        **{
            f"{module_name}.bias": TensorRole.BIAS
            for module_name in _DECLARED_LINEAR_ROLES
        },
    }
)

_EXCLUDED_BUFFER_ROLES: Mapping[str, TensorRole] = MappingProxyType(
    {"block.attention.causal_mask": TensorRole.BUFFER}
)


def _module_at(model: nn.Module, path: str) -> nn.Module:
    current: nn.Module = model
    for component in path.split("."):
        child = getattr(current, component, None)
        if not isinstance(child, nn.Module):
            raise ValueError(f"declared module path is missing: {path}")
        current = child
    return current


def _replace_module(model: nn.Module, path: str, replacement: nn.Module) -> None:
    parent_path, attribute = path.rsplit(".", 1)
    parent = _module_at(model, parent_path)
    setattr(parent, attribute, replacement)


class TinyCausalModelAdapter:
    """Exact-path adapter exercising the same contracts as real model adapters."""

    declared_linear_roles = _DECLARED_LINEAR_ROLES

    @staticmethod
    def _dynamic_operator_roles(model: TinyCausalLM) -> dict[str, TensorRole]:
        roles: dict[str, TensorRole] = {}
        for module_name, module in model.named_modules():
            if not isinstance(module, BinaryLinear):
                continue
            roles[f"{module_name}.dense_reference_weight"] = TensorRole.DENSE_REFERENCE
            if module.dense_reference_bias is not None:
                roles[f"{module_name}.dense_reference_bias"] = TensorRole.DENSE_REFERENCE
            roles[f"{module_name}.input_scale.raw_scale"] = TensorRole.INPUT_CHANNEL_SCALE
            roles[f"{module_name}.dual_scale.learned.value"] = TensorRole.LEARNED_ROW_SCALE
        return roles

    def enumerate_tensors(self, model: TinyCausalLM) -> tuple[TensorDescriptor, ...]:
        if not isinstance(model, TinyCausalLM):
            raise TypeError("model must be TinyCausalLM")
        roles: dict[str, TensorRole] = {
            f"{module_name}.weight": role
            for module_name, role in self.declared_linear_roles.items()
        }
        roles.update(_EXCLUDED_PARAMETER_ROLES)
        roles.update(_EXCLUDED_BUFFER_ROLES)
        roles.update(self._dynamic_operator_roles(model))
        tensors = tuple(model.named_parameters()) + tuple(model.named_buffers())
        actual_names = tuple(name for name, _ in tensors)
        missing = tuple(sorted(set(actual_names) - set(roles)))
        stale = tuple(sorted(set(roles) - set(actual_names)))
        if missing or stale:
            raise ValueError(
                f"tensor role inventory mismatch: missing_roles={missing}, stale_roles={stale}"
            )
        descriptors: list[TensorDescriptor] = []
        for name, tensor in tensors:
            role = roles[name]
            scope = (
                TensorScope.BINARY_BODY
                if role in self.declared_linear_roles.values()
                else TensorScope.EXCLUDED
            )
            descriptors.append(
                TensorDescriptor(
                    name=name,
                    shape=tuple(tensor.shape),
                    dtype=str(tensor.dtype).removeprefix("torch."),
                    parameter_count=tensor.numel(),
                    semantic_role=role,
                    scope=scope,
                    representation_id=(
                        "binary-body-candidate"
                        if scope is TensorScope.BINARY_BODY
                        else f"excluded-{str(tensor.dtype).removeprefix('torch.')}"
                    ),
                )
            )
        if len(actual_names) != len(set(actual_names)):
            raise ValueError("model exposes duplicate tensor names")
        return tuple(descriptors)

    def binary_body(self, tensors: tuple[TensorDescriptor, ...]) -> BinaryScope:
        if not isinstance(tensors, tuple):
            raise TypeError("tensors must be a tuple")
        return BinaryScope.from_inventory(tensors)

    def replace_linears(
        self,
        model: TinyCausalLM,
        factory: BinaryLinearFactory,
    ) -> TinyCausalLM:
        self.enumerate_tensors(model)
        for module_name, role in self.declared_linear_roles.items():
            source = _module_at(model, module_name)
            if not isinstance(source, nn.Linear):
                raise ValueError(f"declared linear is not replaceable: {module_name}")
            replacement = factory(
                source,
                tensor_name=f"{module_name}.weight",
                semantic_role=role,
            )
            if not isinstance(replacement, BinaryLinear):
                raise TypeError("factory must return BinaryLinear")
            _replace_module(model, module_name, replacement)
        self.set_active_representation(model, ActiveRepresentation.TRAINING)
        return model

    def set_active_representation(
        self,
        model: TinyCausalLM,
        representation: ActiveRepresentation,
        *,
        progression_parameter: float | None = None,
    ) -> None:
        resolved = ActiveRepresentation(representation)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for module_name in self.declared_linear_roles:
            module = _module_at(model, module_name)
            if not isinstance(module, BinaryLinear):
                raise ValueError("all declared linears must be replaced before selecting a view")
            module.set_active_representation(
                resolved,
                progression_parameter=progression_parameter,
            )

    def active_representations(
        self,
        model: TinyCausalLM,
    ) -> tuple[NamedActiveRepresentation, ...]:
        active: list[NamedActiveRepresentation] = []
        for module_name, role in self.declared_linear_roles.items():
            module = _module_at(model, module_name)
            if not isinstance(module, BinaryLinear):
                raise ValueError("all declared linears must be replaced before observing views")
            active.append(
                NamedActiveRepresentation(
                    tensor_name=f"{module_name}.weight",
                    semantic_role=role,
                    representation=module.active_representation,
                )
            )
        return tuple(active)

    def active_representation(self, model: TinyCausalLM) -> ActiveRepresentation:
        representations = {
            item.representation for item in self.active_representations(model)
        }
        if len(representations) != 1:
            raise ValueError("model has mixed active representations")
        return next(iter(representations))


__all__ = [
    "TinyCausalLM",
    "TinyCausalLMConfig",
    "TinyCausalModelAdapter",
    "build_seeded_tiny_causal_lm",
]
