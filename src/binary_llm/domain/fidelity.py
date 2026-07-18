"""Canonical executable BinaryLLM fidelity configuration and matched arms."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping

from .ablations import AblationClassification
from .errors import AmbiguityBlocked, ManifestError, Retryability
from .identity import ContentIdentity, identify_content
from .models import CanonicalModel, ReproductionClassification, SourceStatus
from .paper import BINARYLLM_PAPER_LEDGER


class ProgressiveTrainabilityArm(StrEnum):
    ALL_MODEL_PARAMETERS = "all_model_parameters"
    BINARY_BODY_ONLY = "binary_body_only"


@dataclass(frozen=True, slots=True)
class FidelityFieldResolution(CanonicalModel):
    path: str
    value: Any
    source_status: SourceStatus
    source_ref: str
    ambiguity_id: str | None = None

    def __post_init__(self) -> None:
        if not self.path.startswith("/"):
            raise ValueError("fidelity field paths must be JSON pointers")
        if not self.source_ref:
            raise ValueError("fidelity field resolutions require a source reference")
        if self.source_status is SourceStatus.PAPER_SPECIFIED and self.ambiguity_id is not None:
            raise ValueError("paper-specified fields cannot cite an ambiguity")
        if self.source_status is SourceStatus.PAPER_INFERRED and not self.ambiguity_id:
            raise ValueError("paper-inferred fields must cite an ambiguity")


_PAPER_FIELDS: dict[str, tuple[Any, str]] = {
    "/stage1/input_scale/shape": ("per_input_channel", "stage1.input_channel_scale"),
    "/stage1/input_scale/initial_value": (1.0, "stage1.input_channel_scale"),
    "/stage1/weight_transform": ("W*S_t^-1", "stage1.inverse_activation"),
    "/stage1/activation_transform": ("S_t*A", "stage1.inverse_activation"),
    "/stage1/optimizer/trainable_scope": ("input_channel_scales_only", "stage1.input_channel_scale"),
    "/stage1/steps": (50, "stage1.steps"),
    "/stage1/objective": ("autoregressive_loss", "stage1.binary_quantization"),
    "/stage1/quantization": ("1_bit_on_scaled_weight", "stage1.binary_quantization"),
    "/transition/weight_transform": ("W_tilde=W/S_t_star", "stage1.output"),
    "/stage2/progressive/forward": ("tanh(t*x)/tanh(t)", "operator.progressive_function"),
    "/stage2/progressive/backward": ("analytical_derivative", "operator.progressive_derivative"),
    "/stage2/progressive/consistency": ("consistent", "operator.progressive_derivative"),
    "/stage2/phases/count": (20, "stage2.twenty_phases"),
    "/stage2/schedule/kind": ("paper_exponential", "schedule.exponential"),
    "/stage2/schedule/formula": ("1.3*exp(0.22*c)-1.3", "schedule.exponential"),
    "/stage2/partitions/binding": ("one_data_chunk_per_training_phase", "stage2.twenty_phases"),
    "/stage2/trainability/scope": (
        ProgressiveTrainabilityArm.ALL_MODEL_PARAMETERS.value,
        "stage2.all_parameters_trainable",
    ),
    "/stage2/trainability/folded_input_scale": (
        "identity_frozen_transition_state",
        "stage1.output",
    ),
    "/stage2/scales/analytical/definition": ("mean_absolute_current_latent_row", "operator.dual_scaling"),
    "/stage2/scales/analytical/recompute": ("every_update", "operator.dual_scaling"),
    "/stage2/scales/learned/initial_value": (1.0, "operator.dual_scaling"),
    "/stage2/scales/composition": ("dual_product", "operator.dual_scaling"),
    "/optimizer/kind": ("AdamW", "optimization.reported"),
    "/optimizer/learning_rate/initial": (1e-4, "optimization.reported"),
    "/optimizer/learning_rate/final": (2e-6, "optimization.reported"),
    "/optimizer/learning_rate/schedule": ("cosine", "optimization.reported"),
    "/optimizer/weight_decay": (0.1, "optimization.reported"),
    "/data/dataset": ("RedPajama", "data.redpajama"),
    "/data/sampling": ("random", "data.redpajama"),
    "/data/sequence_length": (2048, "optimization.reported"),
    "/data/batch_size": (128, "optimization.reported"),
    "/evaluation/perplexity": (("WikiText-2", "C4", "PTB"), "evaluation.paper_panel"),
    "/evaluation/zero_shot": (
        ("BoolQ", "PIQA", "HellaSwag", "WinoGrande", "ARC-Easy", "ARC-Challenge", "OpenBookQA"),
        "evaluation.paper_panel",
    ),
    "/evaluation/sign_view": ("required_after_final_phase", "operator.final_sign"),
    "/export/scale_merge": ("S_l*S_a", "operator.final_sign"),
    "/export/inference_operator": ("merged_scale_times_sign", "operator.final_sign"),
    "/export/offset": (False, "operator.final_sign"),
}

_AMBIGUOUS_FIELDS: dict[str, str] = {
    "/stage1/input_scale/parameterization": "stage1.scale_parameterization",
    "/stage1/input_scale/positivity": "stage1.scale_parameterization",
    "/stage1/input_scale/zero_avoidance": "stage1.scale_parameterization",
    "/stage1/activation_transform_mode": "stage1.inverse_activation",
    "/stage1/backward": "stage1.loss_and_batching",
    "/stage1/optimizer/kind": "optimizer",
    "/stage1/optimizer/learning_rate": "optimizer",
    "/stage2/phases/index_origin": "progressive.phase_index",
    "/stage2/schedule/t_start": "progressive.phase_index",
    "/stage2/schedule/t_end": "progressive.phase_index",
    "/stage2/scales/analytical/gradient": "progressive.analytical_scale_gradient",
    "/numeric/compute_precision": "numeric.precision",
    "/numeric/scale_storage_precision": "numeric.precision",
    "/optimizer/betas": "optimizer",
    "/optimizer/epsilon": "optimizer",
    "/optimizer/warmup": "optimizer",
    "/optimizer/gradient_clipping": "optimizer",
    "/reproducibility/seed_policy": "reproducibility",
    "/reproducibility/data_order": "reproducibility",
    "/export/zero_sign_policy": "export",
    "/export/rounding": "export",
    "/export/bit_order": "export",
    "/export/row_alignment": "export",
    "/export/scale_endianness": "export",
}

BINARYLLM_FIDELITY_PATHS = frozenset((*_PAPER_FIELDS, *_AMBIGUOUS_FIELDS))
BINARYLLM_AMBIGUOUS_PATHS: Mapping[str, str] = MappingProxyType(_AMBIGUOUS_FIELDS)


@dataclass(frozen=True, slots=True)
class BinaryLLMFidelityConfig(CanonicalModel):
    fields: tuple[FidelityFieldResolution, ...]

    def __post_init__(self) -> None:
        paths = tuple(item.path for item in self.fields)
        actual = set(paths)
        missing = BINARYLLM_FIDELITY_PATHS - actual
        extra = actual - BINARYLLM_FIDELITY_PATHS
        duplicates = sorted({path for path in paths if paths.count(path) > 1})
        if missing or extra or duplicates:
            raise ManifestError(
                "BinaryLLM fidelity field inventory is not exact",
                retryability=Retryability.AFTER_REMEDIATION,
                code="manifest.fidelity_inventory",
                context={"missing_paths": sorted(missing), "unknown_paths": sorted(extra), "duplicate_paths": duplicates},
            )
        object.__setattr__(self, "fields", tuple(sorted(self.fields, key=lambda item: item.path)))

    @property
    def values(self) -> Mapping[str, Any]:
        return MappingProxyType({item.path: item.value for item in self.fields})

    @property
    def resolution_ledger(self) -> Mapping[str, FidelityFieldResolution]:
        return MappingProxyType({item.path: item for item in self.fields})

    @property
    def identity(self) -> ContentIdentity:
        return identify_content(
            self.to_dict(),
            kind="binaryllm-fidelity-config",
            representation_fields={"paper_sha256": BINARYLLM_PAPER_LEDGER.source_sha256},
        )

    @property
    def config_hash(self) -> str:
        return self.identity.sha256

    @property
    def reproduction_classification(self) -> ReproductionClassification:
        specified = {path: value for path, (value, _fact) in _PAPER_FIELDS.items()}
        modified = any(self.values[path] != value for path, value in specified.items()) or any(
            item.source_status is SourceStatus.FRAMEWORK_SELECTED for item in self.fields
        )
        return (
            ReproductionClassification.MODIFIED_REPRODUCTION
            if modified
            else ReproductionClassification.REFERENCE_REPRODUCTION
        )


def build_binaryllm_reference_config(
    ambiguity_resolutions: Mapping[str, FidelityFieldResolution],
) -> BinaryLLMFidelityConfig:
    """Build only from the paper ledger plus explicit ambiguity resolutions."""

    missing = set(_AMBIGUOUS_FIELDS) - set(ambiguity_resolutions)
    extra = set(ambiguity_resolutions) - set(_AMBIGUOUS_FIELDS)
    if missing or extra:
        raise AmbiguityBlocked(
            "BinaryLLM reference configuration has unresolved or unknown choices",
            retryability=Retryability.AFTER_REMEDIATION,
            code="ambiguity.fidelity_resolution",
            context={"missing_paths": sorted(missing), "unknown_paths": sorted(extra)},
        )
    fields = [
        FidelityFieldResolution(
            path,
            value,
            SourceStatus.PAPER_SPECIFIED,
            f"paper:{BINARYLLM_PAPER_LEDGER.source_sha256}:{fact_id}",
        )
        for path, (value, fact_id) in _PAPER_FIELDS.items()
    ]
    for path, ambiguity_id in _AMBIGUOUS_FIELDS.items():
        resolution = ambiguity_resolutions[path]
        if resolution.path != path or resolution.ambiguity_id != ambiguity_id:
            raise AmbiguityBlocked(
                "BinaryLLM ambiguity resolution is bound to the wrong field or ambiguity",
                retryability=Retryability.AFTER_REMEDIATION,
                code="ambiguity.fidelity_binding",
                context={"path": path, "expected_ambiguity_id": ambiguity_id},
            )
        if resolution.source_status is SourceStatus.PAPER_SPECIFIED:
            raise AmbiguityBlocked(
                "an underspecified paper choice cannot be classified as paper-specified",
                retryability=Retryability.AFTER_REMEDIATION,
                code="ambiguity.false_paper_claim",
                context={"path": path},
            )
        fields.append(resolution)
    return BinaryLLMFidelityConfig(tuple(fields))


@dataclass(frozen=True, slots=True)
class FidelityAblationArm(CanonicalModel):
    arm_id: str
    ablation_id: str
    config: BinaryLLMFidelityConfig
    changed_factor_paths: tuple[str, ...]
    expected_source_difference: ReproductionClassification
    classification: AblationClassification


_ARM_MUTATIONS: dict[str, dict[str, Any]] = {
    "stage1_weight_only": {"/stage1/activation_transform_mode": "weight_only", "/stage1/activation_transform": "discarded"},
    "phase_index_alternative": {"/stage2/phases/index_origin": "alternative"},
    "inconsistent_progression": {"/stage2/progressive/backward": "straight_through", "/stage2/progressive/consistency": "inconsistent"},
    "analytical_scale_only": {"/stage2/scales/composition": "analytical_only"},
    "learned_scale_only": {"/stage2/scales/composition": "learned_only"},
    "stage2_body_only": {
        "/stage2/trainability/scope": ProgressiveTrainabilityArm.BINARY_BODY_ONLY.value
    },
    "analytical_scale_gradient_alternative": {"/stage2/scales/analytical/gradient": "alternative"},
    "zero_sign_policy_alternative": {"/export/zero_sign_policy": "alternative"},
}


def build_matched_fidelity_arms(reference: BinaryLLMFidelityConfig) -> tuple[FidelityAblationArm, ...]:
    """Generate preregistered one-factor arms or explicit interactions."""

    arms: list[FidelityAblationArm] = []
    for arm_id, mutations in _ARM_MUTATIONS.items():
        ledger = dict(reference.resolution_ledger)
        for path, value in mutations.items():
            old = ledger[path]
            ledger[path] = FidelityFieldResolution(
                path, value, SourceStatus.FRAMEWORK_SELECTED, f"ablation:{arm_id}", old.ambiguity_id
            )
        config = BinaryLLMFidelityConfig(tuple(ledger.values()))
        changed = tuple(sorted(mutations))
        classification = (
            AblationClassification.COMPONENT_ABLATION
            if len(changed) == 1
            else AblationClassification.INTERACTION_EXPERIMENT
        )
        arms.append(
            FidelityAblationArm(
                arm_id,
                f"binaryllm-fidelity-{arm_id}",
                config,
                changed,
                config.reproduction_classification,
                classification,
            )
        )
    return tuple(arms)


__all__ = [
    "BINARYLLM_AMBIGUOUS_PATHS",
    "BINARYLLM_FIDELITY_PATHS",
    "BinaryLLMFidelityConfig",
    "FidelityAblationArm",
    "FidelityFieldResolution",
    "ProgressiveTrainabilityArm",
    "build_binaryllm_reference_config",
    "build_matched_fidelity_arms",
]
