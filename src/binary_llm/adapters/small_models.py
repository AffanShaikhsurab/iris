"""Explicit, offline-only adapters for registered small-scale model architectures."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from torch import nn

from binary_llm.domain import ModelIdentity

from .contracts import (
    ActiveRepresentation,
    BINARY_BODY_ROLES,
    BinaryLinearFactory,
    BinaryScope,
    NamedActiveRepresentation,
    TensorDescriptor,
    TensorRole,
    TensorScope,
)
from .linear import BinaryLinear


class SmallModelUse(StrEnum):
    CHEAP_SCREENING = "cheap_screening"
    PAPER_COMPATIBLE_REFERENCE = "paper_compatible_reference"


@dataclass(frozen=True, slots=True)
class SmallModelRegistration:
    model_id: str
    tokenizer_id: str
    architecture: str
    model_type: str
    scale_label: str
    use: SmallModelUse
    tied_weights: bool
    tokenizer_revision_policy: str = "explicit_sealed_revision"
    activations_are_excluded: bool = True


PYTHIA_70M = SmallModelRegistration(
    model_id="EleutherAI/pythia-70m-deduped",
    tokenizer_id="EleutherAI/pythia-70m-deduped",
    architecture="GPTNeoXForCausalLM",
    model_type="gpt_neox",
    scale_label="70m",
    use=SmallModelUse.CHEAP_SCREENING,
    tied_weights=False,
)

SMOLLM_135M = SmallModelRegistration(
    model_id="HuggingFaceTB/SmolLM-135M",
    tokenizer_id="HuggingFaceTB/SmolLM-135M",
    architecture="LlamaForCausalLM",
    model_type="llama",
    scale_label="135m",
    use=SmallModelUse.PAPER_COMPATIBLE_REFERENCE,
    tied_weights=True,
)

@dataclass(frozen=True, slots=True)
class LocalSmallModelRequest:
    model_directory: Path
    tokenizer_directory: Path
    identity: ModelIdentity

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_directory", Path(self.model_directory))
        object.__setattr__(self, "tokenizer_directory", Path(self.tokenizer_directory))
        if not isinstance(self.identity, ModelIdentity):
            raise TypeError("identity must be a sealed ModelIdentity")


@dataclass(frozen=True, slots=True)
class LoadedSmallModel:
    model: nn.Module
    tokenizer: Any
    identity: ModelIdentity
    registration: SmallModelRegistration


def _module_at(model: nn.Module, path: str) -> nn.Module:
    current = model
    for component in path.split("."):
        child = getattr(current, component, None)
        if not isinstance(child, nn.Module):
            raise ValueError(f"declared module path is missing: {path}")
        current = child
    return current


def _replace_module(model: nn.Module, path: str, replacement: nn.Module) -> None:
    parent_path, attribute = path.rsplit(".", 1)
    setattr(_module_at(model, parent_path), attribute, replacement)


def _named_parameters_with_aliases(model: nn.Module):
    return tuple(model.named_parameters(remove_duplicate=False))


def _named_buffers_with_aliases(model: nn.Module):
    return tuple(model.named_buffers(remove_duplicate=False))


class _ExplicitSmallModelAdapter:
    registration: SmallModelRegistration

    def _architecture_type(self) -> type[nn.Module]:
        raise NotImplementedError

    def _linear_roles(self, model: nn.Module) -> Mapping[str, TensorRole]:
        raise NotImplementedError

    def _excluded_parameter_roles(self, model: nn.Module) -> Mapping[str, TensorRole]:
        raise NotImplementedError

    def _allowed_buffer_names(self, model: nn.Module) -> frozenset[str]:
        raise NotImplementedError

    def _embedding_and_head_paths(self) -> tuple[str, str]:
        raise NotImplementedError

    def _validate_model(self, model: nn.Module) -> None:
        expected_type = self._architecture_type()
        if type(model) is not expected_type:
            raise TypeError(f"model must be exactly {self.registration.architecture}")
        config = model.config
        if getattr(config, "model_type", None) != self.registration.model_type:
            raise ValueError("model config type does not match adapter registration")
        if bool(getattr(config, "tie_word_embeddings", False)) is not self.registration.tied_weights:
            raise ValueError("model tied-weight configuration does not match registration")
        embedding_path, head_path = self._embedding_and_head_paths()
        embedding = _module_at(model, embedding_path)
        head = _module_at(model, head_path)
        actually_tied = embedding.weight is head.weight
        if actually_tied is not self.registration.tied_weights:
            raise ValueError("loaded embedding/head alias does not match registered tied-weight fact")

    def _dynamic_operator_roles(self, model: nn.Module) -> dict[str, TensorRole]:
        roles: dict[str, TensorRole] = {}
        for module_name in self._linear_roles(model):
            module = _module_at(model, module_name)
            if not isinstance(module, BinaryLinear):
                continue
            roles[f"{module_name}.dense_reference_weight"] = TensorRole.DENSE_REFERENCE
            if module.dense_reference_bias is not None:
                roles[f"{module_name}.dense_reference_bias"] = TensorRole.DENSE_REFERENCE
            roles[f"{module_name}.input_scale.raw_scale"] = TensorRole.INPUT_CHANNEL_SCALE
            roles[f"{module_name}.dual_scale.learned.value"] = TensorRole.LEARNED_ROW_SCALE
        return roles

    def enumerate_tensors(self, model: nn.Module) -> tuple[TensorDescriptor, ...]:
        self._validate_model(model)
        linear_roles = self._linear_roles(model)
        roles = {f"{path}.weight": role for path, role in linear_roles.items()}
        roles.update(self._excluded_parameter_roles(model))
        roles.update(self._dynamic_operator_roles(model))
        parameters = _named_parameters_with_aliases(model)
        buffers = _named_buffers_with_aliases(model)
        for name, _ in buffers:
            if name in roles:
                continue
            if name not in self._allowed_buffer_names(model):
                raise ValueError(f"undeclared architecture buffer: {name}")
            roles[name] = TensorRole.BUFFER
        actual_names = tuple(name for name, _ in parameters + buffers)
        missing = tuple(sorted(set(actual_names) - set(roles)))
        stale = tuple(sorted(set(roles) - set(actual_names)))
        if missing or stale:
            raise ValueError(
                f"tensor role inventory mismatch: missing_roles={missing}, stale_roles={stale}"
            )
        embedding_path, head_path = self._embedding_and_head_paths()
        embedding_name = f"{embedding_path}.weight"
        head_name = f"{head_path}.weight"
        descriptors: list[TensorDescriptor] = []
        for name, tensor in parameters + buffers:
            role = roles[name]
            scope = TensorScope.BINARY_BODY if role in BINARY_BODY_ROLES else TensorScope.EXCLUDED
            tied_to = (
                embedding_name
                if self.registration.tied_weights and name == head_name
                else None
            )
            dtype = str(tensor.dtype).removeprefix("torch.")
            descriptors.append(
                TensorDescriptor(
                    name=name,
                    shape=tuple(tensor.shape),
                    dtype=dtype,
                    parameter_count=tensor.numel(),
                    semantic_role=role,
                    scope=scope,
                    representation_id=(
                        "binary-body-candidate"
                        if scope is TensorScope.BINARY_BODY
                        else f"excluded-{dtype}"
                    ),
                    tied_to=tied_to,
                )
            )
        if len(actual_names) != len(set(actual_names)):
            aliases = [item for item in descriptors if item.tied_to is not None]
            if len(actual_names) - len(set(actual_names)) != len(aliases):
                raise ValueError("model exposes undeclared duplicate tensor names")
        return tuple(descriptors)

    def binary_body(self, tensors: tuple[TensorDescriptor, ...]) -> BinaryScope:
        if not isinstance(tensors, tuple):
            raise TypeError("tensors must be a tuple")
        return BinaryScope.from_inventory(tensors)

    def replace_linears(self, model: nn.Module, factory: BinaryLinearFactory) -> nn.Module:
        self.enumerate_tensors(model)
        for module_name, role in self._linear_roles(model).items():
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
        model: nn.Module,
        representation: ActiveRepresentation,
        *,
        progression_parameter: float | None = None,
    ) -> None:
        self._validate_model(model)
        resolved = ActiveRepresentation(representation)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for module_name in self._linear_roles(model):
            module = _module_at(model, module_name)
            if not isinstance(module, BinaryLinear):
                raise ValueError("all declared linears must be replaced before selecting a view")
            module.set_active_representation(resolved, progression_parameter=progression_parameter)

    def active_representations(
        self, model: nn.Module
    ) -> tuple[NamedActiveRepresentation, ...]:
        self._validate_model(model)
        active: list[NamedActiveRepresentation] = []
        for module_name, role in self._linear_roles(model).items():
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

    def active_representation(self, model: nn.Module) -> ActiveRepresentation:
        representations = {item.representation for item in self.active_representations(model)}
        if len(representations) != 1:
            raise ValueError("model has mixed active representations")
        return next(iter(representations))

    def _validate_identity(self, identity: ModelIdentity) -> None:
        expected = self.registration
        if identity.model_id != expected.model_id:
            raise ValueError("sealed model identity does not match adapter registration")
        if identity.architecture != expected.architecture:
            raise ValueError("sealed architecture does not match adapter registration")
        if identity.tied_weights is not expected.tied_weights:
            raise ValueError("sealed tied-weight fact does not match adapter registration")
        if not identity.tokenizer_revision:
            raise ValueError("tokenizer revision must be explicit")

    def load_local(self, request: LocalSmallModelRequest) -> LoadedSmallModel:
        """Load sealed directories without hub resolution or remote model code."""

        if not isinstance(request, LocalSmallModelRequest):
            raise TypeError("request must be LocalSmallModelRequest")
        self._validate_identity(request.identity)
        model_directory = request.model_directory.resolve(strict=True)
        tokenizer_directory = request.tokenizer_directory.resolve(strict=True)
        if not model_directory.is_dir() or not tokenizer_directory.is_dir():
            raise ValueError("model and tokenizer paths must be local directories")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(
            str(model_directory),
            local_files_only=True,
            trust_remote_code=False,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_directory),
            local_files_only=True,
            trust_remote_code=False,
        )
        self._validate_model(model)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count != request.identity.parameter_count:
            raise ValueError("loaded parameter count does not match sealed identity")
        model.eval()
        return LoadedSmallModel(model, tokenizer, request.identity, self.registration)


def _layer_count(model: nn.Module) -> int:
    count = getattr(model.config, "num_hidden_layers", None)
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("num_hidden_layers must be a positive integer")
    return count


def _roles_with_declared_biases(
    model: nn.Module,
    linear_roles: Mapping[str, TensorRole],
    base_roles: dict[str, TensorRole],
) -> Mapping[str, TensorRole]:
    for module_name in linear_roles:
        module = _module_at(model, module_name)
        if getattr(module, "bias", None) is not None:
            base_roles[f"{module_name}.bias"] = TensorRole.BIAS
    return MappingProxyType(base_roles)

class Pythia70MAdapter(_ExplicitSmallModelAdapter):
    """GPT-NeoX adapter with an explicit fused Q/K/V physical tensor role."""

    registration = PYTHIA_70M

    def _architecture_type(self) -> type[nn.Module]:
        from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXForCausalLM

        return GPTNeoXForCausalLM

    def _linear_roles(self, model: nn.Module) -> Mapping[str, TensorRole]:
        roles: dict[str, TensorRole] = {}
        for layer in range(_layer_count(model)):
            prefix = f"gpt_neox.layers.{layer}"
            roles[f"{prefix}.attention.query_key_value"] = TensorRole.ATTENTION_QUERY_KEY_VALUE
            roles[f"{prefix}.attention.dense"] = TensorRole.ATTENTION_OUTPUT
            roles[f"{prefix}.mlp.dense_h_to_4h"] = TensorRole.FFN_UP
            roles[f"{prefix}.mlp.dense_4h_to_h"] = TensorRole.FFN_DOWN
        return MappingProxyType(roles)

    def _excluded_parameter_roles(self, model: nn.Module) -> Mapping[str, TensorRole]:
        roles = {
            "gpt_neox.embed_in.weight": TensorRole.TOKEN_EMBEDDING,
            "gpt_neox.final_layer_norm.weight": TensorRole.NORMALIZATION,
            "gpt_neox.final_layer_norm.bias": TensorRole.BIAS,
            "embed_out.weight": TensorRole.LANGUAGE_MODEL_HEAD,
        }
        for layer in range(_layer_count(model)):
            prefix = f"gpt_neox.layers.{layer}"
            roles[f"{prefix}.input_layernorm.weight"] = TensorRole.NORMALIZATION
            roles[f"{prefix}.input_layernorm.bias"] = TensorRole.BIAS
            roles[f"{prefix}.post_attention_layernorm.weight"] = TensorRole.NORMALIZATION
            roles[f"{prefix}.post_attention_layernorm.bias"] = TensorRole.BIAS
        return _roles_with_declared_biases(model, self._linear_roles(model), roles)

    def _allowed_buffer_names(self, model: nn.Module) -> frozenset[str]:
        names = {"gpt_neox.rotary_emb.inv_freq"}
        for layer in range(_layer_count(model)):
            prefix = f"gpt_neox.layers.{layer}.attention"
            names.update(
                {
                    f"{prefix}.bias",
                    f"{prefix}.masked_bias",
                    f"{prefix}.rotary_emb.inv_freq",
                }
            )
        return frozenset(names)

    def _embedding_and_head_paths(self) -> tuple[str, str]:
        return "gpt_neox.embed_in", "embed_out"


class SmolLM135MAdapter(_ExplicitSmallModelAdapter):
    """Llama-family adapter registered as the 135M reference reproduction target."""

    registration = SMOLLM_135M

    def _architecture_type(self) -> type[nn.Module]:
        from transformers.models.llama.modeling_llama import LlamaForCausalLM

        return LlamaForCausalLM

    def _linear_roles(self, model: nn.Module) -> Mapping[str, TensorRole]:
        roles: dict[str, TensorRole] = {}
        for layer in range(_layer_count(model)):
            prefix = f"model.layers.{layer}"
            roles[f"{prefix}.self_attn.q_proj"] = TensorRole.ATTENTION_QUERY
            roles[f"{prefix}.self_attn.k_proj"] = TensorRole.ATTENTION_KEY
            roles[f"{prefix}.self_attn.v_proj"] = TensorRole.ATTENTION_VALUE
            roles[f"{prefix}.self_attn.o_proj"] = TensorRole.ATTENTION_OUTPUT
            roles[f"{prefix}.mlp.gate_proj"] = TensorRole.FFN_GATE
            roles[f"{prefix}.mlp.up_proj"] = TensorRole.FFN_UP
            roles[f"{prefix}.mlp.down_proj"] = TensorRole.FFN_DOWN
        return MappingProxyType(roles)

    def _excluded_parameter_roles(self, model: nn.Module) -> Mapping[str, TensorRole]:
        roles = {
            "model.embed_tokens.weight": TensorRole.TOKEN_EMBEDDING,
            "model.norm.weight": TensorRole.NORMALIZATION,
            "lm_head.weight": TensorRole.LANGUAGE_MODEL_HEAD,
        }
        for layer in range(_layer_count(model)):
            prefix = f"model.layers.{layer}"
            roles[f"{prefix}.input_layernorm.weight"] = TensorRole.NORMALIZATION
            roles[f"{prefix}.post_attention_layernorm.weight"] = TensorRole.NORMALIZATION
        return _roles_with_declared_biases(model, self._linear_roles(model), roles)

    def _allowed_buffer_names(self, model: nn.Module) -> frozenset[str]:
        names = {"model.rotary_emb.inv_freq"}
        for layer in range(_layer_count(model)):
            names.add(f"model.layers.{layer}.self_attn.rotary_emb.inv_freq")
        return frozenset(names)

    def _embedding_and_head_paths(self) -> tuple[str, str]:
        return "model.embed_tokens", "lm_head"


__all__ = [
    "LoadedSmallModel",
    "LocalSmallModelRequest",
    "PYTHIA_70M",
    "Pythia70MAdapter",
    "SMOLLM_135M",
    "SmallModelRegistration",
    "SmallModelUse",
    "SmolLM135MAdapter",
]
