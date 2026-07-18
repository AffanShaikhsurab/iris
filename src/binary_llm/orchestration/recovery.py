"""Binary-active recovery, teacher routing, and no-progress enforcement."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from binary_llm.adapters import ActiveRepresentation, ModelAdapter
from binary_llm.domain import (
    ArtifactRef,
    Budget,
    BudgetExceeded,
    canonical_json_bytes,
    sha256_bytes,
)
from binary_llm.domain.errors import ArtifactStoreError, Retryability, SealViolation
from binary_llm.domain.identity import identify_content

from .budgets import (
    BudgetCrossing,
    BudgetMonitor,
    BudgetUsage,
    causal_training_tokens,
)
from .corpus import INITIAL_TOKEN_ALLOCATION, ProvenanceRecord
from .sealing import (
    SealVerificationReport,
    SealedAssetCategory,
    SealedInventory,
)
from .stage1 import CausalBatch, Stage1OptimizerConfig, Stage1OptimizerKind


class TeacherRole(StrEnum):
    BEHAVIORAL = "behavioral"
    BROAD = "broad"


class SupervisionMode(StrEnum):
    NONE = "none"
    SHARED_TOKENIZER_LOGITS = "shared_tokenizer_logits"
    SHARED_TOKENIZER_SEQUENCE = "shared_tokenizer_sequence"
    DIFFERENT_TOKENIZER_TEXT_EXECUTION = "different_tokenizer_text_execution"
    DIFFERENT_TOKENIZER_TEXT_RUBRIC = "different_tokenizer_text_rubric"


class LogitSupervisionKind(StrEnum):
    FULL = "full"
    TOP_K = "top_k"


class RecoveryRunStatus(StrEnum):
    COMPLETED = "completed"
    STOPPED_NO_PROGRESS = "stopped_no_progress"
    STOPPED_BUDGET = "stopped_budget"


def _require_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    canonical_json_bytes(value)
    return MappingProxyType(dict(value))


def _content_id(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _tensor_digest(value: Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    metadata = canonical_json_bytes(
        {"shape": tuple(tensor.shape), "dtype": str(tensor.dtype)}
    )
    raw = tensor.view(torch.uint8).numpy().tobytes()
    return sha256_bytes(metadata + raw)


@dataclass(frozen=True, slots=True)
class TeacherIdentity:
    teacher_id: str
    role: TeacherRole | str
    model_id: str
    revision: str
    tokenizer_id: str
    license_or_terms: str
    permitted_use: str
    generation_settings: Mapping[str, Any] = field(repr=False, compare=True)
    artifact_hash: str = ""
    sealed: bool = False
    baseline_role: str | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", TeacherRole(self.role))
        for name in (
            "teacher_id", "model_id", "revision", "tokenizer_id",
            "license_or_terms", "permitted_use", "artifact_hash",
        ):
            _require_text(name, getattr(self, name))
        if len(self.artifact_hash) != 64 or any(
            character not in "0123456789abcdef" for character in self.artifact_hash
        ):
            raise ValueError("artifact_hash must be a lowercase SHA-256 digest")
        if self.schema_version != 1:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        if not self.generation_settings:
            raise ValueError("generation_settings must be explicit")
        object.__setattr__(
            self, "generation_settings", _freeze_mapping(self.generation_settings)
        )
        if self.baseline_role is not None:
            _require_text("baseline_role", self.baseline_role)

    @property
    def identity_id(self) -> str:
        return _content_id(
            {
                "schema_version": self.schema_version,
                "teacher_id": self.teacher_id,
                "role": self.role.value,
                "model_id": self.model_id,
                "revision": self.revision,
                "tokenizer_id": self.tokenizer_id,
                "license_or_terms": self.license_or_terms,
                "permitted_use": self.permitted_use,
                "generation_settings": self.generation_settings,
                "artifact_hash": self.artifact_hash,
                "sealed": self.sealed,
                "baseline_role": self.baseline_role,
            }
        )


class TeacherArtifactStore(Protocol):
    """Minimal durable-store contract required to authorize a BF16 teacher."""

    def resolve(self, artifact_id: str, expected: ArtifactRef | None = None) -> bytes: ...


def aggregate_teacher_weight_artifacts(artifacts: Sequence[ArtifactRef]) -> str:
    """Bind ``TeacherIdentity.artifact_hash`` to an ordered-independent weight set.

    The digest is SHA-256 over canonical JSON containing every exact immutable
    ``ArtifactRef``, sorted by artifact ID. It is not any individual file digest.
    """

    references = tuple(sorted(artifacts, key=lambda item: item.artifact_id))
    if not references:
        raise ValueError("teacher weight artifacts must not be empty")
    if len({item.artifact_id for item in references}) != len(references):
        raise ValueError("teacher weight artifact IDs must be unique")
    return sha256_bytes(
        canonical_json_bytes(
            {
                "kind": "bf16-teacher-weight-set",
                "schema_version": 1,
                "artifacts": [item.to_dict() for item in references],
            }
        )
    )


_VERIFIED_BINDING_TOKEN = object()


@dataclass(frozen=True, slots=True)
class VerifiedTeacherBinding:
    """Immutable evidence that one teacher resolves to one verified BF16 oracle."""

    inventory_id: str
    baseline_id: str
    baseline_content_id: str
    teacher_identity_id: str
    weight_artifacts: tuple[ArtifactRef, ...]
    tokenizer_artifacts: tuple[ArtifactRef, ...]
    template_artifacts: tuple[ArtifactRef, ...]
    configuration_artifacts: tuple[ArtifactRef, ...]
    weight_aggregate_sha256: str
    schema_version: int = 1
    _resolver_token: object = field(
        default=None, repr=False, compare=False, hash=False
    )

    def __post_init__(self) -> None:
        if self._resolver_token is not _VERIFIED_BINDING_TOKEN:
            raise TypeError(
                "VerifiedTeacherBinding instances must be created by "
                "resolve_verified_teacher_binding"
            )
        for name in (
            "inventory_id",
            "baseline_id",
            "baseline_content_id",
            "teacher_identity_id",
            "weight_aggregate_sha256",
        ):
            _require_text(name, getattr(self, name))
        if self.schema_version != 1:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        if self.weight_aggregate_sha256 != aggregate_teacher_weight_artifacts(
            self.weight_artifacts
        ):
            raise ValueError("weight aggregate does not match exact artifact references")
        for references in (
            self.weight_artifacts,
            self.tokenizer_artifacts,
            self.template_artifacts,
            self.configuration_artifacts,
        ):
            if not references:
                raise ValueError("verified teacher artifact categories must not be empty")
            if references != tuple(sorted(references, key=lambda item: item.artifact_id)):
                raise ValueError("verified teacher artifacts must use deterministic order")

    @property
    def binding_id(self) -> str:
        return _content_id(
            {
                "schema_version": self.schema_version,
                "inventory_id": self.inventory_id,
                "baseline_id": self.baseline_id,
                "baseline_content_id": self.baseline_content_id,
                "teacher_identity_id": self.teacher_identity_id,
                "weight_artifacts": [item.to_dict() for item in self.weight_artifacts],
                "tokenizer_artifacts": [
                    item.to_dict() for item in self.tokenizer_artifacts
                ],
                "template_artifacts": [
                    item.to_dict() for item in self.template_artifacts
                ],
                "configuration_artifacts": [
                    item.to_dict() for item in self.configuration_artifacts
                ],
                "weight_aggregate_sha256": self.weight_aggregate_sha256,
            }
        )


def _teacher_seal_failure(
    message: str,
    code: str,
    *,
    inventory_id: str,
    baseline_id: str | None = None,
    teacher_identity_id: str | None = None,
    artifact_ids: Sequence[str] = (),
    context: Mapping[str, Any] | None = None,
) -> SealViolation:
    affected: dict[str, tuple[str, ...]] = {"inventory_ids": (inventory_id,)}
    if baseline_id:
        affected["baseline_ids"] = (baseline_id,)
    if teacher_identity_id:
        affected["teacher_identity_ids"] = (teacher_identity_id,)
    if artifact_ids:
        affected["artifact_ids"] = tuple(artifact_ids)
    return SealViolation(
        message,
        retryability=Retryability.AFTER_REMEDIATION,
        code=code,
        affected_ids=affected,
        context=context,
    )


def resolve_verified_teacher_binding(
    identity: TeacherIdentity,
    inventory: SealedInventory,
    report: SealVerificationReport,
    store: TeacherArtifactStore,
) -> VerifiedTeacherBinding:
    """Resolve and authorize an exact behavioral teacher from sealed evidence."""

    bf16_oracles = tuple(oracle for oracle in inventory.oracles if oracle.role == "bf16")
    baseline_id = bf16_oracles[0].baseline_id if len(bf16_oracles) == 1 else None
    failure_args = {
        "inventory_id": inventory.inventory_id,
        "baseline_id": baseline_id,
        "teacher_identity_id": identity.identity_id,
    }
    expected_asset_ids = tuple(item.asset.artifact_id for item in inventory.assets)
    if (
        not report.compute_allowed
        or report.inventory_id != inventory.inventory_id
        or report.oracle_identities != inventory.oracle_identities
        or set(report.verified_asset_ids) != set(expected_asset_ids)
        or len(report.verified_asset_ids) != len(expected_asset_ids)
    ):
        raise _teacher_seal_failure(
            "teacher authorization requires the complete matching seal verification report",
            "seal.teacher_report_mismatch",
            **failure_args,
        )
    if len(bf16_oracles) != 1:
        raise _teacher_seal_failure(
            "teacher authorization requires exactly one BF16 oracle",
            "seal.teacher_oracle_ambiguous",
            **failure_args,
        )
    oracle = bf16_oracles[0]
    baseline_content_id = identify_content(
        oracle.to_dict(), kind="sealed-baseline"
    ).value
    if (
        identity.role is not TeacherRole.BEHAVIORAL
        or identity.baseline_role != "bf16"
        or not identity.sealed
    ):
        raise _teacher_seal_failure(
            "teacher identity is not an explicitly sealed behavioral BF16 identity",
            "seal.teacher_identity_role_mismatch",
            **failure_args,
        )
    role_assets = tuple(
        asset for asset in inventory.assets if "bf16" in asset.oracle_roles
    )

    def category_refs(category: SealedAssetCategory) -> tuple[ArtifactRef, ...]:
        return tuple(
            sorted(
                (asset.asset for asset in role_assets if asset.category is category),
                key=lambda item: item.artifact_id,
            )
        )

    weights = category_refs(SealedAssetCategory.WEIGHTS)
    tokenizers = category_refs(SealedAssetCategory.TOKENIZERS)
    templates = category_refs(SealedAssetCategory.TEMPLATES)
    configurations = category_refs(SealedAssetCategory.CONFIGURATIONS)
    aggregate = aggregate_teacher_weight_artifacts(weights)
    expected_weight_refs = tuple(sorted(oracle.artifact_refs, key=lambda item: item.artifact_id))
    if (
        oracle.model.model_id != identity.model_id
        or oracle.model.revision != identity.revision
        or aggregate != identity.artifact_hash
        or expected_weight_refs != weights
    ):
        raise _teacher_seal_failure(
            "teacher identity is not exactly bound to the BF16 oracle",
            "seal.teacher_oracle_identity_mismatch",
            **failure_args,
            context={
                "expected_model_id": oracle.model.model_id,
                "expected_revision": oracle.model.revision,
                "expected_weight_aggregate_sha256": aggregate,
            },
        )
    required = weights + tokenizers + templates + configurations
    try:
        for reference in required:
            store.resolve(reference.artifact_id, expected=reference)
    except ArtifactStoreError as error:
        raise _teacher_seal_failure(
            "a required BF16 teacher artifact did not resolve exactly",
            "seal.teacher_artifact_resolution_failed",
            **failure_args,
            artifact_ids=(reference.artifact_id,),
            context={"store_error": error.to_dict()},
        ) from error
    return VerifiedTeacherBinding(
        inventory_id=inventory.inventory_id,
        baseline_id=oracle.baseline_id,
        baseline_content_id=baseline_content_id,
        teacher_identity_id=identity.identity_id,
        weight_artifacts=weights,
        tokenizer_artifacts=tokenizers,
        template_artifacts=templates,
        configuration_artifacts=configurations,
        weight_aggregate_sha256=aggregate,
        _resolver_token=_VERIFIED_BINDING_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class TeacherOutput:
    record_id: str
    mode: SupervisionMode | str
    logits: Tensor | None = field(default=None, repr=False, compare=False)
    sequence_ids: Tensor | None = field(default=None, repr=False, compare=False)
    text: str | None = None
    logit_kind: LogitSupervisionKind | str | None = None
    top_k: int | None = None

    def __post_init__(self) -> None:
        _require_text("record_id", self.record_id)
        object.__setattr__(self, "mode", SupervisionMode(self.mode))
        if self.logit_kind is not None:
            object.__setattr__(self, "logit_kind", LogitSupervisionKind(self.logit_kind))
        payloads = sum(value is not None for value in (self.logits, self.sequence_ids, self.text))
        if payloads != 1:
            raise ValueError("teacher output must contain exactly one supervision payload")
        if self.mode is SupervisionMode.SHARED_TOKENIZER_LOGITS:
            if self.logits is None or self.logits.ndim != 3:
                raise ValueError("logit supervision requires [batch, sequence, vocabulary] logits")
            if self.logit_kind is None:
                raise ValueError("logit supervision kind must be explicit")
            if self.logit_kind is LogitSupervisionKind.TOP_K:
                if self.top_k is None or isinstance(self.top_k, bool) or self.top_k < 1:
                    raise ValueError("top-k logit supervision requires a positive top_k")
                if self.top_k > self.logits.shape[-1]:
                    raise ValueError("top_k exceeds the teacher vocabulary")
            elif self.top_k is not None:
                raise ValueError("top_k is only valid for top-k logit supervision")
        elif self.mode is SupervisionMode.SHARED_TOKENIZER_SEQUENCE:
            if self.sequence_ids is None or self.sequence_ids.ndim != 2:
                raise ValueError("sequence supervision requires [batch, sequence] token IDs")
            if self.sequence_ids.dtype != torch.long:
                raise TypeError("teacher sequence IDs must use torch.long")
            if self.logit_kind is not None or self.top_k is not None:
                raise ValueError("sequence supervision cannot declare logit settings")
        elif self.mode in (
            SupervisionMode.DIFFERENT_TOKENIZER_TEXT_EXECUTION,
            SupervisionMode.DIFFERENT_TOKENIZER_TEXT_RUBRIC,
        ):
            _require_text("text", self.text or "")
            if self.logit_kind is not None or self.top_k is not None:
                raise ValueError("text supervision cannot declare logit settings")
        else:
            raise ValueError("a teacher cannot emit no-teacher supervision")

    @property
    def output_hash(self) -> str:
        payload = (
            _tensor_digest(self.logits)
            if self.logits is not None
            else _tensor_digest(self.sequence_ids)
            if self.sequence_ids is not None
            else sha256_bytes((self.text or "").encode("utf-8"))
        )
        return _content_id(
            {
                "record_id": self.record_id,
                "mode": self.mode.value,
                "payload_hash": payload,
                "logit_kind": None if self.logit_kind is None else self.logit_kind.value,
                "top_k": self.top_k,
            }
        )


class TeacherPort(Protocol):
    def identity(self) -> TeacherIdentity: ...

    def supervise(
        self,
        record: ProvenanceRecord,
        batch: CausalBatch,
        mode: SupervisionMode,
    ) -> TeacherOutput: ...


@dataclass(frozen=True, slots=True)
class TeacherRecordProvenance:
    record_id: str
    teacher_identity_id: str
    teacher_id: str
    role: TeacherRole
    revision: str
    tokenizer_id: str
    license_or_terms: str
    permitted_use: str
    generation_settings: Mapping[str, Any] = field(repr=False, compare=True)
    supervision_mode: SupervisionMode
    output_hash: str

    def __post_init__(self) -> None:
        for name in (
            "record_id", "teacher_identity_id", "teacher_id", "revision",
            "tokenizer_id", "license_or_terms", "permitted_use", "output_hash",
        ):
            _require_text(name, getattr(self, name))
        object.__setattr__(
            self, "generation_settings", _freeze_mapping(self.generation_settings)
        )


@dataclass(frozen=True, slots=True)
class SupervisionCoverage:
    supervised_positions: int
    eligible_positions: int
    fraction: float
    probability_mass: float | None = None

    def __post_init__(self) -> None:
        if self.supervised_positions < 0 or self.eligible_positions < 0:
            raise ValueError("coverage counts must be non-negative")
        if self.supervised_positions > self.eligible_positions:
            raise ValueError("supervised positions cannot exceed eligible positions")
        expected = (
            0.0
            if self.eligible_positions == 0
            else self.supervised_positions / self.eligible_positions
        )
        if not math.isclose(self.fraction, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("coverage fraction does not match coverage counts")
        if self.probability_mass is not None and not 0.0 <= self.probability_mass <= 1.0:
            raise ValueError("probability mass must be between zero and one")


@dataclass(frozen=True, slots=True)
class RoutedSupervision:
    output: TeacherOutput
    provenance: TeacherRecordProvenance


@dataclass(frozen=True, slots=True)
class TeacherRoutingConfig:
    student_tokenizer_id: str
    sealed_bf16_teacher_identity_id: str
    behavioral_mode: SupervisionMode | str = SupervisionMode.NONE
    broad_mode: SupervisionMode | str = SupervisionMode.NONE
    sealed_bf16_inventory_id: str | None = None
    behavioral_slices: tuple[str, ...] = (
        "iris_tools",
        "safety_transitions",
        "adversarial",
    )

    def __post_init__(self) -> None:
        _require_text("student_tokenizer_id", self.student_tokenizer_id)
        _require_text(
            "sealed_bf16_teacher_identity_id", self.sealed_bf16_teacher_identity_id
        )
        object.__setattr__(self, "behavioral_mode", SupervisionMode(self.behavioral_mode))
        object.__setattr__(self, "broad_mode", SupervisionMode(self.broad_mode))
        if self.behavioral_mode is not SupervisionMode.NONE:
            if self.sealed_bf16_inventory_id is None:
                raise ValueError(
                    "enabled behavioral guidance requires a sealed BF16 inventory ID"
                )
            _require_text(
                "sealed_bf16_inventory_id", self.sealed_bf16_inventory_id
            )
        elif self.sealed_bf16_inventory_id is not None:
            _require_text(
                "sealed_bf16_inventory_id", self.sealed_bf16_inventory_id
            )
        if not self.behavioral_slices or len(self.behavioral_slices) != len(
            set(self.behavioral_slices)
        ):
            raise ValueError("behavioral_slices must be unique and non-empty")


class TeacherRouter:
    """Route behavioral records only to the sealed BF16 teacher."""

    def __init__(
        self,
        config: TeacherRoutingConfig,
        *,
        behavioral_teacher: TeacherPort | None = None,
        broad_teacher: TeacherPort | None = None,
        behavioral_binding: VerifiedTeacherBinding | None = None,
    ) -> None:
        if not isinstance(config, TeacherRoutingConfig):
            raise TypeError("config must be TeacherRoutingConfig")
        self.config = config
        self.behavioral_teacher = behavioral_teacher
        self.broad_teacher = broad_teacher
        self.behavioral_binding = behavioral_binding
        self._validate_port(
            behavioral_teacher, config.behavioral_mode, TeacherRole.BEHAVIORAL
        )
        self._validate_port(broad_teacher, config.broad_mode, TeacherRole.BROAD)
        if config.behavioral_mode is not SupervisionMode.NONE:
            self._require_behavioral_binding()

    def _require_behavioral_binding(self) -> TeacherIdentity:
        identity = (
            None if self.behavioral_teacher is None else self.behavioral_teacher.identity()
        )
        binding = self.behavioral_binding
        if (
            identity is None
            or binding is None
            or binding.inventory_id != self.config.sealed_bf16_inventory_id
            or identity.identity_id != self.config.sealed_bf16_teacher_identity_id
            or identity.identity_id != binding.teacher_identity_id
            or identity.role is not TeacherRole.BEHAVIORAL
            or not identity.sealed
            or identity.baseline_role != "bf16"
            or identity.artifact_hash != binding.weight_aggregate_sha256
        ):
            raise _teacher_seal_failure(
                "behavioral routing requires the configured verified BF16 teacher binding",
                "seal.teacher_binding_mismatch",
                inventory_id="unbound" if binding is None else binding.inventory_id,
                baseline_id=None if binding is None else binding.baseline_id,
                teacher_identity_id=None if identity is None else identity.identity_id,
            )
        return identity

    def _validate_port(
        self,
        port: TeacherPort | None,
        mode: SupervisionMode,
        expected_role: TeacherRole,
    ) -> None:
        if mode is SupervisionMode.NONE:
            if port is not None:
                raise ValueError("a no-teacher route cannot have a teacher port")
            return
        if port is None:
            raise ValueError("enabled teacher guidance requires a teacher port")
        identity = port.identity()
        if identity.role is not expected_role:
            raise ValueError(f"teacher role must be {expected_role.value}")
        shared = identity.tokenizer_id == self.config.student_tokenizer_id
        token_mode = mode in (
            SupervisionMode.SHARED_TOKENIZER_LOGITS,
            SupervisionMode.SHARED_TOKENIZER_SEQUENCE,
        )
        if shared != token_mode:
            raise ValueError(
                "shared tokenizers require token supervision and different tokenizers require text supervision"
            )

    def supervise(
        self, record: ProvenanceRecord, batch: CausalBatch
    ) -> RoutedSupervision | None:
        behavioral = record.capability_slice in self.config.behavioral_slices
        port = self.behavioral_teacher if behavioral else self.broad_teacher
        mode = self.config.behavioral_mode if behavioral else self.config.broad_mode
        if mode is SupervisionMode.NONE:
            return None
        if port is None:
            raise RuntimeError("validated teacher route is unexpectedly missing")
        identity = port.identity()
        if behavioral:
            identity = self._require_behavioral_binding()
        output = port.supervise(record, batch, mode)
        if output.record_id != record.record_id or output.mode is not mode:
            raise ValueError("teacher output does not match the routed record and mode")
        if record.teacher_identity is not None and record.teacher_identity != identity.identity_id:
            raise ValueError("record teacher identity conflicts with routed teacher")
        if record.generation_settings is not None and dict(record.generation_settings) != dict(
            identity.generation_settings
        ):
            raise ValueError("record generation settings conflict with routed teacher")
        provenance = TeacherRecordProvenance(
            record_id=record.record_id,
            teacher_identity_id=identity.identity_id,
            teacher_id=identity.teacher_id,
            role=identity.role,
            revision=identity.revision,
            tokenizer_id=identity.tokenizer_id,
            license_or_terms=identity.license_or_terms,
            permitted_use=identity.permitted_use,
            generation_settings=identity.generation_settings,
            supervision_mode=mode,
            output_hash=output.output_hash,
        )
        return RoutedSupervision(output, provenance)


@dataclass(frozen=True, slots=True)
class RecoveryCorpusMix:
    mix_id: str
    allocation: tuple[tuple[str, float], ...]
    preregistered: bool
    initial: bool = False

    def __post_init__(self) -> None:
        _require_text("mix_id", self.mix_id)
        names = tuple(name for name, _ in self.allocation)
        if not names or len(names) != len(set(names)):
            raise ValueError("mix allocation requires unique capability slices")
        if any(
            not isinstance(fraction, (int, float))
            or isinstance(fraction, bool)
            or not math.isfinite(fraction)
            or fraction < 0
            for _, fraction in self.allocation
        ):
            raise ValueError("mix fractions must be finite and non-negative")
        if not math.isclose(
            sum(fraction for _, fraction in self.allocation),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("mix fractions must sum to one")
        if self.initial and self.allocation != INITIAL_TOKEN_ALLOCATION:
            raise ValueError("initial recovery mix must use the registered 30/15/10/20/10/10/5 allocation")
        if not self.preregistered:
            raise ValueError("recovery mixes must be preregistered")


@dataclass(frozen=True, slots=True)
class RecoveryCorpusPlan:
    initial_mix: RecoveryCorpusMix
    alternative_mixes: tuple[RecoveryCorpusMix, ...]

    def __post_init__(self) -> None:
        if not self.initial_mix.initial:
            raise ValueError("initial_mix must be marked as the registered initial allocation")
        if not self.alternative_mixes:
            raise ValueError("at least one preregistered alternative corpus mix is required")
        mixes = (self.initial_mix,) + self.alternative_mixes
        ids = tuple(mix.mix_id for mix in mixes)
        if len(ids) != len(set(ids)):
            raise ValueError("recovery mix IDs must be unique")
        if any(mix.initial for mix in self.alternative_mixes):
            raise ValueError("alternative mixes cannot be marked initial")
        if any(mix.allocation == self.initial_mix.allocation for mix in self.alternative_mixes):
            raise ValueError("alternative allocation must differ from the initial allocation")
        initial_slices = tuple(name for name, _ in self.initial_mix.allocation)
        if any(tuple(name for name, _ in mix.allocation) != initial_slices for mix in mixes):
            raise ValueError("all recovery mixes must cover the same ordered capability slices")

    def select(self, mix_id: str) -> RecoveryCorpusMix:
        for mix in (self.initial_mix,) + self.alternative_mixes:
            if mix.mix_id == mix_id:
                return mix
        raise ValueError(f"unknown recovery mix: {mix_id}")


@dataclass(frozen=True, slots=True)
class TokenMixSliceEvidence:
    capability_slice: str
    target_fraction: float
    target_tokens: int
    actual_tokens: int
    actual_fraction: float


@dataclass(frozen=True, slots=True)
class TokenMixEvidence:
    mix_id: str
    total_tokens: int
    record_count: int
    slices: tuple[TokenMixSliceEvidence, ...]


def _rounded_token_targets(
    total_tokens: int, allocation: tuple[tuple[str, float], ...]
) -> dict[str, int]:
    raw = [(name, fraction * total_tokens) for name, fraction in allocation]
    targets = {name: math.floor(value) for name, value in raw}
    remainder = total_tokens - sum(targets.values())
    ranked = sorted(
        enumerate(raw), key=lambda item: (-(item[1][1] - math.floor(item[1][1])), item[0])
    )
    for index, _ in ranked[:remainder]:
        targets[allocation[index][0]] += 1
    return targets


def validate_recovery_token_mix(
    records: Sequence[ProvenanceRecord], mix: RecoveryCorpusMix
) -> TokenMixEvidence:
    if not records:
        raise ValueError("recovery corpus records must be non-empty")
    allowed = {name for name, _ in mix.allocation}
    unknown = tuple(sorted({record.capability_slice for record in records} - allowed))
    if unknown:
        raise ValueError(f"recovery corpus contains undeclared capability slices: {unknown}")
    total_tokens = sum(record.token_count for record in records)
    targets = _rounded_token_targets(total_tokens, mix.allocation)
    actual = {name: 0 for name in allowed}
    for record in records:
        actual[record.capability_slice] += record.token_count
    mismatches = {
        name: (actual[name], targets[name])
        for name in allowed
        if actual[name] != targets[name]
    }
    if mismatches:
        raise ValueError(
            f"recovery corpus token allocation does not match preregistered mix: {mismatches}"
        )
    return TokenMixEvidence(
        mix_id=mix.mix_id,
        total_tokens=total_tokens,
        record_count=len(records),
        slices=tuple(
            TokenMixSliceEvidence(
                capability_slice=name,
                target_fraction=fraction,
                target_tokens=targets[name],
                actual_tokens=actual[name],
                actual_fraction=actual[name] / total_tokens,
            )
            for name, fraction in mix.allocation
        ),
    )


@dataclass(frozen=True, slots=True)
class NoProgressPolicy:
    calibration_metric: str
    frozen_capability_metric: str
    minimum_improvement: float = 0.0

    def __post_init__(self) -> None:
        _require_text("calibration_metric", self.calibration_metric)
        _require_text("frozen_capability_metric", self.frozen_capability_metric)
        if not math.isfinite(self.minimum_improvement) or self.minimum_improvement < 0:
            raise ValueError("minimum_improvement must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class RecoveryProgressPoint:
    optimizer_step: int
    calibration_score: float
    frozen_capability_score: float

    def __post_init__(self) -> None:
        if self.optimizer_step < 0:
            raise ValueError("optimizer_step must be non-negative")
        if not math.isfinite(self.calibration_score) or not math.isfinite(
            self.frozen_capability_score
        ):
            raise ValueError("recovery progress scores must be finite")


@dataclass(frozen=True, slots=True)
class NoProgressDecision:
    triggered: bool
    calibration_improvement: float
    frozen_capability_improvement: float
    reason: str | None


def evaluate_no_progress(
    initial: RecoveryProgressPoint,
    current: RecoveryProgressPoint,
    policy: NoProgressPolicy,
) -> NoProgressDecision:
    if current.optimizer_step <= initial.optimizer_step:
        raise ValueError("current progress point must follow the initial point")
    calibration = current.calibration_score - initial.calibration_score
    capability = current.frozen_capability_score - initial.frozen_capability_score
    threshold = policy.minimum_improvement
    triggered = calibration > threshold and capability <= threshold
    return NoProgressDecision(
        triggered=triggered,
        calibration_improvement=calibration,
        frozen_capability_improvement=capability,
        reason=(
            "calibration improved without Frozen_Evaluation_Set capability improvement"
            if triggered
            else None
        ),
    )


@dataclass(frozen=True, slots=True)
class BinaryForwardEvent:
    forward_index: int
    purpose: str
    representations: tuple[ActiveRepresentation, ...]


class BinaryActiveForward:
    """The only candidate-forward entry point exposed by recovery orchestration."""

    def __init__(
        self,
        model: nn.Module,
        adapter: ModelAdapter[nn.Module],
        target_representation: ActiveRepresentation,
    ) -> None:
        if target_representation is ActiveRepresentation.DENSE_REFERENCE:
            raise ValueError("recovery cannot target the dense-reference representation")
        self.model = model
        self.adapter = adapter
        self.target_representation = target_representation
        self._events: list[BinaryForwardEvent] = []

    @property
    def events(self) -> tuple[BinaryForwardEvent, ...]:
        return tuple(self._events)

    def _representations(self) -> tuple[ActiveRepresentation, ...]:
        active = tuple(
            item.representation for item in self.adapter.active_representations(self.model)
        )
        if not active or any(item is not self.target_representation for item in active):
            raise RuntimeError("recovery forward attempted without the target binary operator")
        return active

    def __call__(self, input_ids: Tensor, *, purpose: str) -> Tensor:
        _require_text("purpose", purpose)
        before = self._representations()
        output = self.model(input_ids)
        after = self._representations()
        if before != after:
            raise RuntimeError("active representation changed during recovery forward")
        self._events.append(BinaryForwardEvent(len(self._events), purpose, before))
        return output


@dataclass(frozen=True, slots=True)
class TextLevelEvaluation:
    record_id: str
    mode: SupervisionMode
    score: float
    passed: bool
    evaluator_id: str

    def __post_init__(self) -> None:
        _require_text("record_id", self.record_id)
        _require_text("evaluator_id", self.evaluator_id)
        if self.mode not in (
            SupervisionMode.DIFFERENT_TOKENIZER_TEXT_EXECUTION,
            SupervisionMode.DIFFERENT_TOKENIZER_TEXT_RUBRIC,
        ):
            raise ValueError("text-level evaluation requires a text supervision mode")
        if not math.isfinite(self.score):
            raise ValueError("text-level score must be finite")


class TextLevelEvaluator(Protocol):
    def __call__(
        self,
        supervision: RoutedSupervision,
        forward: BinaryActiveForward,
        batch: CausalBatch,
    ) -> TextLevelEvaluation: ...


MetricEvaluator = Callable[[BinaryActiveForward], float]


@dataclass(frozen=True, slots=True)
class RecoveryTrainerConfig:
    optimizer: Stage1OptimizerConfig
    seed: int
    max_optimizer_steps: int
    target_representation: ActiveRepresentation | str
    no_progress_policy: NoProgressPolicy
    progression_parameter: float | None = None
    causal_loss_weight: float = 1.0
    teacher_loss_weight: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.optimizer, Stage1OptimizerConfig):
            raise TypeError("optimizer must be Stage1OptimizerConfig")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if (
            not isinstance(self.max_optimizer_steps, int)
            or isinstance(self.max_optimizer_steps, bool)
            or self.max_optimizer_steps < 1
        ):
            raise ValueError("max_optimizer_steps must be a positive integer")
        object.__setattr__(
            self, "target_representation", ActiveRepresentation(self.target_representation)
        )
        if self.target_representation is ActiveRepresentation.DENSE_REFERENCE:
            raise ValueError("dense-reference recovery is prohibited")
        if self.target_representation is ActiveRepresentation.PROGRESSIVE:
            if self.progression_parameter is None or not math.isfinite(
                self.progression_parameter
            ) or self.progression_parameter < 0:
                raise ValueError("progressive recovery requires a non-negative progression parameter")
        elif self.progression_parameter is not None:
            raise ValueError("progression_parameter is only valid for progressive recovery")
        for name in ("causal_loss_weight", "teacher_loss_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.causal_loss_weight == 0 and self.teacher_loss_weight == 0:
            raise ValueError("at least one recovery loss weight must be positive")


@dataclass(frozen=True, slots=True)
class RecoveryStepEvidence:
    optimizer_step: int
    record_id: str
    causal_loss: float
    teacher_loss: float | None
    total_loss: float
    supervision_mode: SupervisionMode
    coverage: SupervisionCoverage
    teacher_identity_id: str | None
    text_evaluation: TextLevelEvaluation | None


@dataclass(frozen=True, slots=True)
class RecoveryRunResult:
    status: RecoveryRunStatus
    completed_steps: int
    mix_evidence: TokenMixEvidence
    progress: tuple[RecoveryProgressPoint, ...]
    steps: tuple[RecoveryStepEvidence, ...]
    forward_events: tuple[BinaryForwardEvent, ...]
    teacher_provenance: tuple[TeacherRecordProvenance, ...]
    no_progress_decision: NoProgressDecision | None
    budget_usage: BudgetUsage | None = None
    budget_crossing: BudgetCrossing | None = None
    budget_failure: BudgetExceeded | None = None


def _make_optimizer(
    config: Stage1OptimizerConfig, parameters: Sequence[nn.Parameter]
) -> Optimizer:
    if config.kind is Stage1OptimizerKind.ADAMW:
        return torch.optim.AdamW(
            parameters,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.epsilon,
            weight_decay=config.weight_decay,
        )
    return torch.optim.SGD(
        parameters,
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )


def _causal_loss_from_logits(logits: Tensor, batch: CausalBatch) -> Tensor:
    if logits.ndim != 3 or logits.shape[:2] != batch.input_ids.shape:
        raise ValueError("candidate must return [batch, sequence, vocabulary] logits")
    labels = batch.input_ids if batch.labels is None else batch.labels
    return nn.functional.cross_entropy(
        logits[:, :-1, :].contiguous().view(-1, logits.shape[-1]),
        labels[:, 1:].contiguous().view(-1),
        ignore_index=-100,
    )


def _teacher_loss_and_coverage(
    student_logits: Tensor,
    batch: CausalBatch,
    routed: RoutedSupervision,
) -> tuple[Tensor | None, SupervisionCoverage]:
    output = routed.output
    eligible = batch.input_ids.shape[0] * (batch.input_ids.shape[1] - 1)
    if output.mode is SupervisionMode.SHARED_TOKENIZER_LOGITS:
        teacher_logits = output.logits
        assert teacher_logits is not None
        if teacher_logits.shape != student_logits.shape:
            raise ValueError("shared-tokenizer teacher and candidate logits must have equal shape")
        teacher = teacher_logits[:, :-1, :].detach()
        student = student_logits[:, :-1, :]
        if output.logit_kind is LogitSupervisionKind.FULL:
            targets = torch.softmax(teacher, dim=-1)
            loss = nn.functional.kl_div(
                torch.log_softmax(student, dim=-1), targets, reduction="batchmean"
            )
            mass = 1.0
        else:
            assert output.top_k is not None
            probabilities = torch.softmax(teacher, dim=-1)
            values, indices = torch.topk(probabilities, output.top_k, dim=-1)
            mass = float(values.sum(dim=-1).mean().item())
            targets = values / values.sum(dim=-1, keepdim=True)
            student_selected = torch.gather(student, -1, indices)
            loss = nn.functional.kl_div(
                torch.log_softmax(student_selected, dim=-1),
                targets,
                reduction="batchmean",
            )
        return loss, SupervisionCoverage(eligible, eligible, 1.0, mass)
    if output.mode is SupervisionMode.SHARED_TOKENIZER_SEQUENCE:
        labels = output.sequence_ids
        assert labels is not None
        if labels.shape != batch.input_ids.shape:
            raise ValueError("teacher sequence IDs must match the candidate batch shape")
        shifted = labels[:, 1:].contiguous()
        supervised = int((shifted != -100).sum().item())
        loss = nn.functional.cross_entropy(
            student_logits[:, :-1, :].contiguous().view(-1, student_logits.shape[-1]),
            shifted.view(-1),
            ignore_index=-100,
        )
        return loss, SupervisionCoverage(
            supervised,
            eligible,
            0.0 if eligible == 0 else supervised / eligible,
        )
    return None, SupervisionCoverage(1, 1, 1.0)


class RecoveryTrainerBackend:
    """Train under guarded target-native forwards and auditable supervision."""

    def __init__(self, config: RecoveryTrainerConfig) -> None:
        if not isinstance(config, RecoveryTrainerConfig):
            raise TypeError("config must be RecoveryTrainerConfig")
        self.config = config

    def run(
        self,
        *,
        run_id: str,
        model: nn.Module,
        adapter: ModelAdapter[nn.Module],
        records: Sequence[ProvenanceRecord],
        batches: Mapping[str, CausalBatch],
        corpus_plan: RecoveryCorpusPlan,
        selected_mix_id: str,
        teacher_router: TeacherRouter,
        calibration_evaluator: MetricEvaluator,
        frozen_capability_evaluator: MetricEvaluator,
        text_level_evaluator: TextLevelEvaluator | None = None,
        budget: Budget | None = None,
        prior_budget_usage: BudgetUsage | None = None,
        monotonic_clock: Callable[[], float] | None = None,
        cost_meter: Callable[[], float] | None = None,
    ) -> RecoveryRunResult:
        _require_text("run_id", run_id)
        selected_mix = corpus_plan.select(selected_mix_id)
        mix_evidence = validate_recovery_token_mix(records, selected_mix)
        record_ids = tuple(record.record_id for record in records)
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("recovery record IDs must be unique")
        if set(batches) != set(record_ids):
            raise ValueError("recovery batches must exactly match recovery record IDs")
        adapter.set_active_representation(
            model,
            self.config.target_representation,
            progression_parameter=self.config.progression_parameter,
        )
        forward = BinaryActiveForward(
            model, adapter, self.config.target_representation
        )
        parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
        if not parameters:
            raise ValueError("target recovery representation exposes no trainable parameters")
        optimizer = _make_optimizer(self.config.optimizer, parameters)

        def metric_point(step: int) -> RecoveryProgressPoint:
            calibration = float(calibration_evaluator(forward))
            capability = float(frozen_capability_evaluator(forward))
            return RecoveryProgressPoint(step, calibration, capability)

        progress = [metric_point(0)]
        step_evidence: list[RecoveryStepEvidence] = []
        provenance_by_record: dict[str, TeacherRecordProvenance] = {}
        no_progress: NoProgressDecision | None = None
        status = RecoveryRunStatus.COMPLETED
        budget_crossing: BudgetCrossing | None = None
        budget_monitor = (
            None
            if budget is None
            else BudgetMonitor(
                budget,
                prior_usage=prior_budget_usage,
                **({} if monotonic_clock is None else {"clock": monotonic_clock}),
                cost_meter=cost_meter,
            )
        )

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.config.seed)
            model.train()
            for index in range(self.config.max_optimizer_steps):
                record = records[index % len(records)]
                batch = batches[record.record_id]
                batch_tokens = causal_training_tokens(batch.input_ids, batch.labels)
                if budget_monitor is not None:
                    budget_crossing = budget_monitor.crossing(
                        phase="pre_step",
                        proposed_tokens=batch_tokens,
                        proposed_steps=1,
                    )
                    if budget_crossing is not None:
                        status = RecoveryRunStatus.STOPPED_BUDGET
                        break
                optimizer.zero_grad(set_to_none=True)
                logits = forward(batch.input_ids, purpose="recovery_training")
                causal = _causal_loss_from_logits(logits, batch)
                routed = teacher_router.supervise(record, batch)
                teacher_loss: Tensor | None = None
                text_evaluation: TextLevelEvaluation | None = None
                if routed is None:
                    coverage = SupervisionCoverage(0, 0, 0.0)
                    mode = SupervisionMode.NONE
                    teacher_identity_id = None
                else:
                    teacher_loss, coverage = _teacher_loss_and_coverage(
                        logits, batch, routed
                    )
                    mode = routed.output.mode
                    teacher_identity_id = routed.provenance.teacher_identity_id
                    previous = provenance_by_record.get(record.record_id)
                    if previous is not None and previous != routed.provenance:
                        raise ValueError("teacher provenance changed within a recovery run")
                    provenance_by_record[record.record_id] = routed.provenance
                    if mode in (
                        SupervisionMode.DIFFERENT_TOKENIZER_TEXT_EXECUTION,
                        SupervisionMode.DIFFERENT_TOKENIZER_TEXT_RUBRIC,
                    ):
                        if text_level_evaluator is None:
                            raise ValueError(
                                "different-tokenizer supervision requires execution checks or a registered rubric"
                            )
                        text_evaluation = text_level_evaluator(routed, forward, batch)
                        if (
                            text_evaluation.record_id != record.record_id
                            or text_evaluation.mode is not mode
                        ):
                            raise ValueError("text-level evidence does not match routed supervision")
                total = self.config.causal_loss_weight * causal
                if teacher_loss is not None:
                    total = total + self.config.teacher_loss_weight * teacher_loss
                if not torch.isfinite(total).item():
                    raise ValueError("recovery loss became non-finite")
                total.backward()
                if any(
                    parameter.grad is not None
                    and not torch.isfinite(parameter.grad).all().item()
                    for parameter in parameters
                ):
                    raise ValueError("recovery gradient became non-finite")
                optimizer.step()
                completed = index + 1
                if budget_monitor is not None:
                    budget_monitor.record_step(batch_tokens)
                step_evidence.append(
                    RecoveryStepEvidence(
                        optimizer_step=completed,
                        record_id=record.record_id,
                        causal_loss=float(causal.detach().item()),
                        teacher_loss=(
                            None
                            if teacher_loss is None
                            else float(teacher_loss.detach().item())
                        ),
                        total_loss=float(total.detach().item()),
                        supervision_mode=mode,
                        coverage=coverage,
                        teacher_identity_id=teacher_identity_id,
                        text_evaluation=text_evaluation,
                    )
                )
                current = metric_point(completed)
                progress.append(current)
                if budget_monitor is not None:
                    budget_crossing = budget_monitor.crossing(phase="post_step")
                    if budget_crossing is not None:
                        status = RecoveryRunStatus.STOPPED_BUDGET
                        break
                decision = evaluate_no_progress(
                    progress[0], current, self.config.no_progress_policy
                )
                if decision.triggered:
                    no_progress = decision
                    status = RecoveryRunStatus.STOPPED_NO_PROGRESS
                    break

        return RecoveryRunResult(
            status=status,
            completed_steps=len(step_evidence),
            mix_evidence=mix_evidence,
            progress=tuple(progress),
            steps=tuple(step_evidence,
            ),
            forward_events=forward.events,
            teacher_provenance=tuple(
                provenance_by_record[record_id]
                for record_id in record_ids
                if record_id in provenance_by_record
            ),
            no_progress_decision=no_progress,
            budget_usage=None if budget_monitor is None else budget_monitor.usage,
            budget_crossing=budget_crossing,
            budget_failure=(
                None
                if budget_monitor is None or budget_crossing is None
                else budget_monitor.failure(budget_crossing)
            ),
        )


__all__ = [
    "BinaryActiveForward",
    "BinaryForwardEvent",
    "LogitSupervisionKind",
    "MetricEvaluator",
    "NoProgressDecision",
    "NoProgressPolicy",
    "RecoveryCorpusMix",
    "RecoveryCorpusPlan",
    "RecoveryProgressPoint",
    "RecoveryRunResult",
    "RecoveryRunStatus",
    "RecoveryStepEvidence",
    "RecoveryTrainerBackend",
    "RecoveryTrainerConfig",
    "RoutedSupervision",
    "SupervisionCoverage",
    "SupervisionMode",
    "TeacherIdentity",
    "TeacherArtifactStore",
    "TeacherOutput",
    "TeacherPort",
    "TeacherRecordProvenance",
    "TeacherRole",
    "TeacherRouter",
    "TeacherRoutingConfig",
    "VerifiedTeacherBinding",
    "TextLevelEvaluation",
    "TextLevelEvaluator",
    "TokenMixEvidence",
    "TokenMixSliceEvidence",
    "evaluate_no_progress",
    "aggregate_teacher_weight_artifacts",
    "resolve_verified_teacher_binding",
    "validate_recovery_token_mix",
]
