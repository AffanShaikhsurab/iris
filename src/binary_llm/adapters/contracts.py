"""Typed model-adapter contracts and exact tensor-role inventories."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, Protocol, TypeVar, runtime_checkable

from torch import nn


class TensorScope(StrEnum):
    BINARY_BODY = "binary_body"
    EXCLUDED = "excluded"


class TensorRole(StrEnum):
    ATTENTION_QUERY = "attention_query"
    ATTENTION_KEY = "attention_key"
    ATTENTION_VALUE = "attention_value"
    ATTENTION_QUERY_KEY_VALUE = "attention_query_key_value"
    ATTENTION_OUTPUT = "attention_output"
    FFN_GATE = "ffn_gate"
    FFN_UP = "ffn_up"
    FFN_DOWN = "ffn_down"
    TOKEN_EMBEDDING = "token_embedding"
    LANGUAGE_MODEL_HEAD = "language_model_head"
    NORMALIZATION = "normalization"
    BIAS = "bias"
    BUFFER = "buffer"
    INPUT_CHANNEL_SCALE = "input_channel_scale"
    LEARNED_ROW_SCALE = "learned_row_scale"
    DENSE_REFERENCE = "dense_reference"
    DECLARED_EXCEPTION = "declared_exception"


BINARY_BODY_ROLES = frozenset(
    {
        TensorRole.ATTENTION_QUERY,
        TensorRole.ATTENTION_KEY,
        TensorRole.ATTENTION_VALUE,
        TensorRole.ATTENTION_QUERY_KEY_VALUE,
        TensorRole.ATTENTION_OUTPUT,
        TensorRole.FFN_GATE,
        TensorRole.FFN_UP,
        TensorRole.FFN_DOWN,
    }
)


class ActiveRepresentation(StrEnum):
    TRAINING = "training"
    PROGRESSIVE = "progressive"
    SIGN = "sign"
    DENSE_REFERENCE = "dense_reference"


@dataclass(frozen=True, slots=True)
class TensorDescriptor:
    name: str
    shape: tuple[int, ...]
    dtype: str
    parameter_count: int
    semantic_role: TensorRole
    scope: TensorScope
    representation_id: str
    tied_to: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.dtype or not self.representation_id:
            raise ValueError("tensor name, dtype, and representation_id must be explicit")
        if not self.shape or any(size < 1 for size in self.shape):
            raise ValueError("tensor shape must be non-empty and positive")
        expected_count = 1
        for size in self.shape:
            expected_count *= size
        if self.parameter_count != expected_count:
            raise ValueError("parameter_count must equal the tensor shape product")
        if (self.semantic_role in BINARY_BODY_ROLES) is not (
            self.scope is TensorScope.BINARY_BODY
        ):
            raise ValueError("only declared attention and FFN weights belong to binary_body")
        if self.tied_to == self.name:
            raise ValueError("a tensor cannot be tied to itself")


@dataclass(frozen=True, slots=True)
class BinaryScope:
    binary_body: tuple[TensorDescriptor, ...]
    excluded: tuple[TensorDescriptor, ...]

    def __post_init__(self) -> None:
        descriptors = self.binary_body + self.excluded
        names = tuple(item.name for item in descriptors)
        if len(names) != len(set(names)):
            raise ValueError("tensor inventory contains duplicate names")
        if any(item.scope is not TensorScope.BINARY_BODY for item in self.binary_body):
            raise ValueError("binary_body contains an excluded tensor")
        if any(item.scope is not TensorScope.EXCLUDED for item in self.excluded):
            raise ValueError("excluded inventory contains a binary tensor")

    @classmethod
    def from_inventory(cls, tensors: tuple[TensorDescriptor, ...]) -> BinaryScope:
        return cls(
            tuple(item for item in tensors if item.scope is TensorScope.BINARY_BODY),
            tuple(item for item in tensors if item.scope is TensorScope.EXCLUDED),
        )


@dataclass(frozen=True, slots=True)
class NamedActiveRepresentation:
    tensor_name: str
    semantic_role: TensorRole
    representation: ActiveRepresentation


@runtime_checkable
class BinaryLinearFactory(Protocol):
    def __call__(
        self,
        source: nn.Linear,
        *,
        tensor_name: str,
        semantic_role: TensorRole,
    ) -> nn.Module: ...


ModelT = TypeVar("ModelT", bound=nn.Module)


@runtime_checkable
class ModelAdapter(Protocol, Generic[ModelT]):
    def enumerate_tensors(self, model: ModelT) -> tuple[TensorDescriptor, ...]: ...

    def binary_body(self, tensors: tuple[TensorDescriptor, ...]) -> BinaryScope: ...

    def replace_linears(self, model: ModelT, factory: BinaryLinearFactory) -> ModelT: ...

    def set_active_representation(
        self,
        model: ModelT,
        representation: ActiveRepresentation,
        *,
        progression_parameter: float | None = None,
    ) -> None: ...

    def active_representations(
        self, model: ModelT
    ) -> tuple[NamedActiveRepresentation, ...]: ...


__all__ = [
    "ActiveRepresentation",
    "BINARY_BODY_ROLES",
    "BinaryLinearFactory",
    "BinaryScope",
    "ModelAdapter",
    "NamedActiveRepresentation",
    "TensorDescriptor",
    "TensorRole",
    "TensorScope",
]
