"""Immutable, JSON-compatible domain records for binary-LLM experiments."""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping, TypeAlias

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | tuple["JsonValue", ...] | Mapping[str, "JsonValue"]
SUPPORTED_SCHEMA_VERSION = 1


def _require_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_schema(version: int) -> None:
    if version != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {version}")


def _require_unique(name: str, values: tuple[str, ...]) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{name} contains duplicate identifiers")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (str, int, float, bool, type(None), StrEnum)):
        return value
    raise TypeError(f"value is not JSON-compatible: {type(value).__name__}")


def _json(value: Any) -> Any:
    if isinstance(value, CanonicalModel):
        return value.to_dict()
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, Mapping):
        return {key: _json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json(item) for item in value]
    return value


class CanonicalModel:
    """Mixin exposing records as ordinary JSON-compatible dictionaries."""

    def to_dict(self) -> dict[str, Any]:
        return {field.name: _json(getattr(self, field.name)) for field in fields(self)}


class SourceStatus(StrEnum):
    PAPER_SPECIFIED = "paper_specified"
    PAPER_INFERRED = "paper_inferred"
    FRAMEWORK_SELECTED = "framework_selected"


class ClaimStatus(StrEnum):
    REPRODUCED = "reproduced"
    NOT_REPRODUCED = "not_reproduced"
    CONTRADICTED = "contradicted"
    NOT_TESTED = "not_tested"


class ReproductionComponent(StrEnum):
    OPERATOR = "operator"
    DATA = "data"
    SCHEDULE = "schedule"
    OPTIMIZATION = "optimization"
    EVALUATION = "evaluation"


class ReproductionClassification(StrEnum):
    REFERENCE_REPRODUCTION = "reference_reproduction"
    MODIFIED_REPRODUCTION = "modified_reproduction"


class AblationDirection(StrEnum):
    IMPROVEMENT = "improvement"
    DEGRADATION = "degradation"
    NO_EFFECT = "no_effect"


class ExperimentFamilyKind(StrEnum):
    BINARY = "binary"
    TERNARY = "ternary"
    STRUCTURED_PRUNING_Q4 = "structured_pruning_q4"
    DISTILLED_STUDENT_Q4_OR_QAT = "distilled_student_q4_or_qat"
    HYBRID = "hybrid"


class ScaleRung(StrEnum):
    SMALL = "small"
    INTERMEDIATE = "intermediate"
    IRIS = "iris"


class AmbiguityStatus(StrEnum):
    OPEN = "open"
    TESTING = "testing"
    RESOLVED = "resolved"
    UNRESOLVED_BLOCKING = "unresolved_blocking"
    SUPERSEDED = "superseded"


class CheckpointBoundary(StrEnum):
    OPTIMIZER_STEP = "optimizer_step"
    PROGRESSIVE_PHASE = "progressive_phase"


class GateCategory(StrEnum):
    CONTINUATION = "continuation"
    NO_PROGRESS = "no_progress"
    NUMERICAL_INSTABILITY = "numerical_instability"
    CAPABILITY_REGRESSION = "capability_regression"
    STORAGE = "storage"
    RUNTIME = "runtime"
    PROVENANCE = "provenance"
    COST = "cost"
    SAFETY = "safety"
    RECOVERY_BUDGET = "recovery_budget"
    SCALE_RUNG_ELIGIBILITY = "scale_rung_eligibility"


class GateComparator(StrEnum):
    LT = "lt"
    LE = "le"
    EQ = "eq"
    GE = "ge"
    GT = "gt"


class StopMode(StrEnum):
    IMMEDIATE_SAFE_BOUNDARY = "immediate_safe_boundary"
    PROMOTION_ONLY = "promotion_only"


class RepresentationKind(StrEnum):
    BINARY = "binary"
    TERNARY = "ternary"
    Q4 = "q4"
    Q8 = "q8"
    BF16 = "bf16"
    FP32 = "fp32"
    REDUCED_VOCABULARY = "reduced_vocabulary"
    CUSTOM = "custom"


@dataclass(frozen=True, slots=True)
class ExperimentFamilyIdentity(CanonicalModel):
    family_id: str
    kind: ExperimentFamilyKind
    component_family_ids: tuple[str, ...] = ()
    schema_version: int = SUPPORTED_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema(self.schema_version)
        _require_text("family_id", self.family_id)
        _require_unique("component_family_ids", self.component_family_ids)
        if self.kind is ExperimentFamilyKind.HYBRID:
            if len(self.component_family_ids) < 2 or self.family_id == self.kind.value:
                raise ValueError("hybrids require a distinct family_id and at least two components")
        elif self.component_family_ids:
            raise ValueError("single-family identities cannot declare components")


BINARY_FAMILY = ExperimentFamilyIdentity("binary", ExperimentFamilyKind.BINARY)
TERNARY_FAMILY = ExperimentFamilyIdentity("ternary", ExperimentFamilyKind.TERNARY)
STRUCTURED_PRUNING_Q4_FAMILY = ExperimentFamilyIdentity(
    "structured_pruning_q4", ExperimentFamilyKind.STRUCTURED_PRUNING_Q4
)
DISTILLED_STUDENT_Q4_OR_QAT_FAMILY = ExperimentFamilyIdentity(
    "distilled_student_q4_or_qat", ExperimentFamilyKind.DISTILLED_STUDENT_Q4_OR_QAT
)
BUILTIN_FAMILIES = (
    BINARY_FAMILY,
    TERNARY_FAMILY,
    STRUCTURED_PRUNING_Q4_FAMILY,
    DISTILLED_STUDENT_Q4_OR_QAT_FAMILY,
)


@dataclass(frozen=True, slots=True)
class ModelIdentity(CanonicalModel):
    model_id: str
    revision: str
    tokenizer_revision: str
    architecture: str
    parameter_count: int
    config_hash: str
    tokenizer_hashes: tuple[str, ...]
    template_hashes: tuple[str, ...]
    weight_hashes: tuple[str, ...]
    tied_weights: bool
    schema_version: int = SUPPORTED_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema(self.schema_version)
        for name in ("model_id", "revision", "tokenizer_revision", "architecture", "config_hash"):
            _require_text(name, getattr(self, name))
        if self.parameter_count <= 0:
            raise ValueError("parameter_count must be positive")
        for name in ("tokenizer_hashes", "template_hashes", "weight_hashes"):
            values = getattr(self, name)
            if not values:
                raise ValueError(f"{name} must not be empty")
            _require_unique(name, values)


@dataclass(frozen=True, slots=True)
class ArtifactRef(CanonicalModel):
    artifact_id: str
    kind: str
    sha256: str
    bytes: int
    media_type: str
    parent_artifact_id: str | None = None
    producing_run_id: str | None = None
    producing_attempt_id: str | None = None
    schema_version: int = SUPPORTED_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema(self.schema_version)
        for name in ("artifact_id", "kind", "sha256", "media_type"):
            _require_text(name, getattr(self, name))
        if self.bytes < 0:
            raise ValueError("bytes must be non-negative")
        if self.producing_attempt_id and not self.producing_run_id:
            raise ValueError("producing_attempt_id requires producing_run_id")


@dataclass(frozen=True, slots=True)
class SealedBaselineRef(CanonicalModel):
    baseline_id: str
    role: str
    model: ModelIdentity
    artifact_refs: tuple[ArtifactRef, ...]
    evaluator_revision: str
    baseline_output_refs: tuple[ArtifactRef, ...]
    sealed_at: str
    schema_version: int = SUPPORTED_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema(self.schema_version)
        _require_text("baseline_id", self.baseline_id)
        if self.role not in {"bf16", "q4", "base", "adapter"}:
            raise ValueError("baseline role must be bf16, q4, base, or adapter")
        if not self.artifact_refs or not self.baseline_output_refs:
            raise ValueError("sealed baselines require artifacts and baseline outputs")
        _require_unique("artifact_refs", tuple(item.artifact_id for item in self.artifact_refs))
        _require_unique("baseline_output_refs", tuple(item.artifact_id for item in self.baseline_output_refs))
        _require_text("evaluator_revision", self.evaluator_revision)
        _require_text("sealed_at", self.sealed_at)


@dataclass(frozen=True, slots=True)
class ClaimRecord(CanonicalModel):
    claim_id: str
    source_uri: str
    source_version_hash: str
    statement: str
    category: str
    status: ClaimStatus
    scale_scope: tuple[ScaleRung, ...]
    protocol_refs: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    differences: tuple[str, ...]
    schema_version: int = SUPPORTED_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema(self.schema_version)
        for name in ("claim_id", "source_uri", "source_version_hash", "statement", "category"):
            _require_text(name, getattr(self, name))
        _require_unique("scale_scope", tuple(item.value for item in self.scale_scope))
        _require_unique("protocol_refs", self.protocol_refs)
        _require_unique("evidence_refs", self.evidence_refs)
        _require_unique("differences", self.differences)


@dataclass(frozen=True, slots=True)
class ReproductionSettings(CanonicalModel):
    operator: Mapping[str, JsonValue]
    data: Mapping[str, JsonValue]
    schedule: Mapping[str, JsonValue]
    optimization: Mapping[str, JsonValue]
    evaluation: Mapping[str, JsonValue]
    schema_version: int = SUPPORTED_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema(self.schema_version)
        for field_name in ("operator", "data", "schedule", "optimization", "evaluation"):
            value = getattr(self, field_name)
            if not isinstance(value, Mapping):
                raise TypeError(f"{field_name} reproduction settings must be a mapping")
            object.__setattr__(self, field_name, _freeze(value))


@dataclass(frozen=True, slots=True)
class SettingDifference(CanonicalModel):
    component: ReproductionComponent
    path: str
    source_present: bool
    resolved_present: bool
    source_value: JsonValue
    resolved_value: JsonValue
    ablation_ref: str
    schema_version: int = SUPPORTED_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema(self.schema_version)
        _require_text("path", self.path)
        _require_text("ablation_ref", self.ablation_ref)
        if not self.path.startswith(f"/{self.component.value}/"):
            raise ValueError("difference path must be rooted in its reproduction component")
        if self.source_present:
            object.__setattr__(self, "source_value", _freeze(self.source_value))
        elif self.source_value is not None:
            raise ValueError("absent source values must be null")
        if self.resolved_present:
            object.__setattr__(self, "resolved_value", _freeze(self.resolved_value))
        elif self.resolved_value is not None:
            raise ValueError("absent resolved values must be null")


@dataclass(frozen=True, slots=True)
class ComponentAblationResult(CanonicalModel):
    ablation_ref: str
    claim_id: str
    component: ReproductionComponent
    scale_rung: ScaleRung
    source_direction: AblationDirection
    local_direction: AblationDirection
    protocol_ref: str
    evidence_ref: str
    schema_version: int = SUPPORTED_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema(self.schema_version)
        for name in ("ablation_ref", "claim_id", "protocol_ref", "evidence_ref"):
            _require_text(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class ReproductionAssessment(CanonicalModel):
    assessment_id: str
    claim_id: str
    source_uri: str
    source_version_hash: str
    scale_rung: ScaleRung
    classification: ReproductionClassification
    status: ClaimStatus
    protocol_refs: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    observed_result_ref: str
    differences: tuple[SettingDifference, ...]
    direction_results: tuple[ComponentAblationResult, ...]
    schema_version: int = SUPPORTED_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema(self.schema_version)
        for name in (
            "assessment_id", "claim_id", "source_uri", "source_version_hash",
            "observed_result_ref",
        ):
            _require_text(name, getattr(self, name))
        if not self.protocol_refs or not self.evidence_refs:
            raise ValueError("reproduction assessments require protocol and local evidence references")
        _require_unique("protocol_refs", self.protocol_refs)
        _require_unique("evidence_refs", self.evidence_refs)
        _require_unique("difference paths", tuple(item.path for item in self.differences))
        _require_unique(
            "direction ablation references",
            tuple(item.ablation_ref for item in self.direction_results),
        )
        if self.observed_result_ref not in self.evidence_refs:
            raise ValueError("observed_result_ref must identify local evidence")
        modified = self.classification is ReproductionClassification.MODIFIED_REPRODUCTION
        if modified != bool(self.differences):
            raise ValueError("modified reproduction classification must match source differences")
        if any(
            item.claim_id != self.claim_id or item.scale_rung is not self.scale_rung
            for item in self.direction_results
        ):
            raise ValueError("direction results must match the assessed claim and scale")
        if any(
            item.source_direction is not item.local_direction for item in self.direction_results
        ) and self.status is not ClaimStatus.NOT_REPRODUCED:
            raise ValueError("direction mismatches must mark the claim not reproduced")


@dataclass(frozen=True, slots=True)
class Budget(CanonicalModel):
    max_tokens: int
    max_optimizer_steps: int
    max_wall_seconds: int
    max_billable_cost: float
    checkpoint_boundary: CheckpointBoundary
    schema_version: int = SUPPORTED_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema(self.schema_version)
        if min(self.max_tokens, self.max_optimizer_steps, self.max_wall_seconds) <= 0:
            raise ValueError("token, optimizer-step, and wall-time budgets must be positive")
        if self.max_billable_cost < 0:
            raise ValueError("billable-cost budget must be non-negative")


@dataclass(frozen=True, slots=True)
class GateDefinition(CanonicalModel):
    gate_id: str
    category: GateCategory
    metric_path: str
    comparator: GateComparator
    threshold: int | float
    baseline_relative: bool
    scope: str
    required_evidence_kind: str
    stop_mode: StopMode
    missing_policy: str = "fail"
    schema_version: int = SUPPORTED_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema(self.schema_version)
        for name in ("gate_id", "metric_path", "scope", "required_evidence_kind"):
            _require_text(name, getattr(self, name))
        if self.missing_policy != "fail":
            raise ValueError("gate missing_policy must be fail")
        if isinstance(self.threshold, bool):
            raise ValueError("gate threshold must be numeric")


@dataclass(frozen=True, slots=True)
class GateSet(CanonicalModel):
    gate_set_id: str
    definitions: tuple[GateDefinition, ...]
    schema_version: int = SUPPORTED_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema(self.schema_version)
        _require_text("gate_set_id", self.gate_set_id)
        if not self.definitions:
            raise ValueError("gate sets must not be empty")
        _require_unique("gate definitions", tuple(item.gate_id for item in self.definitions))


@dataclass(frozen=True, slots=True)
class ToleranceDefinition(CanonicalModel):
    tolerance_id: str
    metric_path: str
    scope: str
    absolute: float | None
    relative: float | None
    exact: bool
    schema_version: int = SUPPORTED_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema(self.schema_version)
        for name in ("tolerance_id", "metric_path", "scope"):
            _require_text(name, getattr(self, name))
        if self.exact and (self.absolute is not None or self.relative is not None):
            raise ValueError("exact tolerances cannot also declare numeric tolerances")
        if not self.exact and self.absolute is None and self.relative is None:
            raise ValueError("a non-exact tolerance requires an absolute or relative bound")
        if any(value is not None and value < 0 for value in (self.absolute, self.relative)):
            raise ValueError("tolerances must be non-negative")


@dataclass(frozen=True, slots=True)
class ToleranceSet(CanonicalModel):
    tolerance_set_id: str
    definitions: tuple[ToleranceDefinition, ...]
    schema_version: int = SUPPORTED_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema(self.schema_version)
        _require_text("tolerance_set_id", self.tolerance_set_id)
        if not self.definitions:
            raise ValueError("tolerance sets must not be empty")
        _require_unique("tolerance definitions", tuple(item.tolerance_id for item in self.definitions))


@dataclass(frozen=True, slots=True)
class AmbiguityCandidate(CanonicalModel):
    candidate_id: str
    description: str
    source_status: SourceStatus
    schema_version: int = SUPPORTED_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema(self.schema_version)
        _require_text("candidate_id", self.candidate_id)
        _require_text("description", self.description)


@dataclass(frozen=True, slots=True)
class AmbiguityEntry(CanonicalModel):
    ambiguity_id: str
    register_version: str
    statement: str
    source_status: SourceStatus
    affected_components: tuple[str, ...]
    candidates: tuple[AmbiguityCandidate, ...]
    matched_protocol: str | None
    decision_rule: str | None
    confidence_method: str | None
    required_floors: tuple[str, ...]
    scale_scope: tuple[ScaleRung, ...]
    status: AmbiguityStatus
    selected_candidate: str | None
    evidence_refs: tuple[str, ...]
    rejected_candidates: tuple[str, ...]
    supersedes: str | None = None
    schema_version: int = SUPPORTED_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema(self.schema_version)
        for name in ("ambiguity_id", "register_version", "statement"):
            _require_text(name, getattr(self, name))
        if not self.affected_components:
            raise ValueError("ambiguities require affected components")
        candidate_ids = tuple(item.candidate_id for item in self.candidates)
        _require_unique("ambiguity candidates", candidate_ids)
        _require_unique("scale_scope", tuple(item.value for item in self.scale_scope))
        _require_unique("evidence_refs", self.evidence_refs)
        _require_unique("rejected_candidates", self.rejected_candidates)
        if self.selected_candidate is not None and self.selected_candidate not in candidate_ids:
            raise ValueError("selected_candidate must identify a declared candidate")
        if any(item not in candidate_ids for item in self.rejected_candidates):
            raise ValueError("rejected_candidates must identify declared candidates")
        if self.status is AmbiguityStatus.RESOLVED:
            required = (self.selected_candidate, self.matched_protocol, self.decision_rule, self.confidence_method)
            if not all(required) or not self.required_floors or not self.scale_scope or not self.evidence_refs:
                raise ValueError("resolved ambiguities require an explicit decision, floors, and scoped evidence")
            expected_rejections = set(candidate_ids) - {self.selected_candidate}
            if set(self.rejected_candidates) != expected_rejections:
                raise ValueError("resolved ambiguities must record every rejected alternative")
        elif self.selected_candidate is not None:
            raise ValueError("only resolved ambiguities may select a candidate")


@dataclass(frozen=True, slots=True)
class AmbiguityRegister(CanonicalModel):
    register_id: str
    version: str
    entries: tuple[AmbiguityEntry, ...]
    schema_version: int = SUPPORTED_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema(self.schema_version)
        _require_text("register_id", self.register_id)
        _require_text("version", self.version)
        _require_unique("ambiguity entries", tuple(item.ambiguity_id for item in self.entries))
        if any(item.register_version != self.version for item in self.entries):
            raise ValueError("every ambiguity entry must match the register version")


@dataclass(frozen=True, slots=True)
class RepresentationSpec(CanonicalModel):
    representation_id: str
    kind: RepresentationKind
    bit_width: int
    payload_dtype: str
    scale_dtype: str | None
    offset_dtype: str | None
    exception_dtype: str | None
    has_inference_offset: bool
    higher_precision_exceptions: tuple[str, ...]
    zero_sign_rule: str | None
    metadata: Mapping[str, JsonValue]
    schema_version: int = SUPPORTED_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema(self.schema_version)
        _require_text("representation_id", self.representation_id)
        _require_text("payload_dtype", self.payload_dtype)
        if self.bit_width <= 0:
            raise ValueError("bit_width must be positive")
        if self.kind is RepresentationKind.BINARY:
            if self.bit_width != 1 or not self.scale_dtype or not self.zero_sign_rule:
                raise ValueError("binary representations require one bit, scale dtype, and zero-sign rule")
        if self.has_inference_offset != (self.offset_dtype is not None):
            raise ValueError("inference-offset flag and offset dtype must agree")
        if self.higher_precision_exceptions and not self.exception_dtype:
            raise ValueError("higher-precision exceptions require an exception dtype")
        _require_unique("higher_precision_exceptions", self.higher_precision_exceptions)
        object.__setattr__(self, "metadata", _freeze(self.metadata))


@dataclass(frozen=True, slots=True)
class FormatSpec(CanonicalModel):
    format_id: str
    version: str
    bit_order: str
    row_order: str
    row_alignment: int
    zero_sign_rule: str
    scale_dtype: str
    scale_endianness: str
    metadata_encoding: str
    allowed_representation_ids: tuple[str, ...]
    temporary_expansion_limit_bytes: int
    schema_version: int = SUPPORTED_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema(self.schema_version)
        for name in (
            "format_id", "version", "bit_order", "row_order", "zero_sign_rule",
            "scale_dtype", "scale_endianness", "metadata_encoding",
        ):
            _require_text(name, getattr(self, name))
        if self.row_alignment <= 0 or self.temporary_expansion_limit_bytes < 0:
            raise ValueError("alignment must be positive and expansion limit non-negative")
        if not self.allowed_representation_ids:
            raise ValueError("format must explicitly allow at least one representation")
        _require_unique("allowed_representation_ids", self.allowed_representation_ids)


@dataclass(frozen=True, slots=True)
class EvaluationEvidence(CanonicalModel):
    evidence_id: str
    artifact_id: str
    family_id: str
    scale_rung: ScaleRung
    evaluation_set_id: str
    evaluator_revision: str
    blind_status: str
    evaluation_ordinal: int
    seed: int
    prompt_order_hash: str
    decoding: Mapping[str, JsonValue]
    raw_outputs_ref: str
    per_case_ref: str
    aggregates: Mapping[str, JsonValue]
    failure_categories: tuple[str, ...]
    created_at: str
    paired_deltas: Mapping[str, JsonValue] | None = None
    confidence_intervals: Mapping[str, JsonValue] | None = None
    schema_version: int = SUPPORTED_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema(self.schema_version)
        for name in (
            "evidence_id", "artifact_id", "family_id", "evaluation_set_id",
            "evaluator_revision", "blind_status", "prompt_order_hash",
            "raw_outputs_ref", "per_case_ref", "created_at",
        ):
            _require_text(name, getattr(self, name))
        if self.evaluation_ordinal < 1:
            raise ValueError("evaluation_ordinal must be positive")
        object.__setattr__(self, "decoding", _freeze(self.decoding))
        object.__setattr__(self, "aggregates", _freeze(self.aggregates))
        if self.paired_deltas is not None:
            object.__setattr__(self, "paired_deltas", _freeze(self.paired_deltas))
        if self.confidence_intervals is not None:
            object.__setattr__(self, "confidence_intervals", _freeze(self.confidence_intervals))


@dataclass(frozen=True, slots=True)
class CandidateRecord(CanonicalModel):
    candidate_id: str
    family: ExperimentFamilyIdentity
    model: ModelIdentity
    scale_rung: ScaleRung
    source_checkpoint: ArtifactRef
    artifact: ArtifactRef | None
    representation: RepresentationSpec
    evidence_refs: tuple[str, ...]
    deployable_runtime_supported: bool
    metadata_accounting_complete: bool
    schema_version: int = SUPPORTED_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema(self.schema_version)
        _require_text("candidate_id", self.candidate_id)
        _require_unique("evidence_refs", self.evidence_refs)
        if self.artifact and self.artifact.parent_artifact_id != self.source_checkpoint.artifact_id:
            raise ValueError("candidate artifact must name its direct source checkpoint")


@dataclass(frozen=True, slots=True)
class ExperimentManifest(CanonicalModel):
    experiment_id: str
    family: ExperimentFamilyIdentity
    hypothesis_refs: tuple[str, ...]
    source_claim_refs: tuple[str, ...]
    model: ModelIdentity
    scale_rung: ScaleRung
    parent_checkpoint: ArtifactRef
    baseline_refs: tuple[SealedBaselineRef, ...]
    binary_scope: tuple[str, ...]
    operator_config: Mapping[str, JsonValue]
    ambiguity_register_ref: str
    stage1_config: Mapping[str, JsonValue]
    progressive_config: Mapping[str, JsonValue]
    recovery_config: Mapping[str, JsonValue] | None
    teacher_refs: tuple[str, ...]
    corpus_ref: str
    partition_refs: tuple[str, ...]
    frozen_evaluation_refs: tuple[str, ...]
    evaluator_protocols: tuple[str, ...]
    seed_set: tuple[int, ...]
    determinism_policy: Mapping[str, JsonValue]
    budget: Budget
    gate_set_ref: str
    tolerance_set_ref: str
    artifact_format: FormatSpec | None
    device_protocol: str | None
    source_revision: str
    dependency_lock_hash: str
    container_digest: str
    command: tuple[str, ...]
    sanitized_environment: Mapping[str, JsonValue]
    preregistered_at: str
    component_control_experiment_ids: tuple[str, ...] = ()
    schema_version: int = SUPPORTED_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema(self.schema_version)
        for name in (
            "experiment_id", "ambiguity_register_ref", "corpus_ref", "gate_set_ref",
            "tolerance_set_ref", "source_revision", "dependency_lock_hash",
            "container_digest", "preregistered_at",
        ):
            _require_text(name, getattr(self, name))
        for name in (
            "hypothesis_refs", "baseline_refs", "partition_refs", "frozen_evaluation_refs",
            "evaluator_protocols", "seed_set", "command",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")
        _require_unique("source_claim_refs", self.source_claim_refs)
        _require_unique("partition_refs", self.partition_refs)
        _require_unique("frozen_evaluation_refs", self.frozen_evaluation_refs)
        _require_unique("evaluator_protocols", self.evaluator_protocols)
        _require_unique("seed_set", tuple(str(item) for item in self.seed_set))
        baseline_ids = tuple(item.baseline_id for item in self.baseline_refs)
        _require_unique("baseline_refs", baseline_ids)
        if self.family.kind is ExperimentFamilyKind.BINARY:
            if not self.binary_scope or not self.operator_config or not self.stage1_config or not self.progressive_config:
                raise ValueError("binary experiments require explicit scope and operator/stage configurations")
        elif self.binary_scope:
            raise ValueError("non-binary families cannot claim a binary scope")
        if self.family.kind is ExperimentFamilyKind.HYBRID:
            controls = set(self.component_control_experiment_ids)
            if len(controls) != len(self.family.component_family_ids):
                raise ValueError("hybrids require one distinct control experiment per component family")
        elif self.component_control_experiment_ids:
            raise ValueError("single-family manifests cannot declare hybrid component controls")
        for field_name in (
            "operator_config", "stage1_config", "progressive_config", "determinism_policy",
            "sanitized_environment",
        ):
            object.__setattr__(self, field_name, _freeze(getattr(self, field_name)))
        if self.recovery_config is not None:
            object.__setattr__(self, "recovery_config", _freeze(self.recovery_config))
