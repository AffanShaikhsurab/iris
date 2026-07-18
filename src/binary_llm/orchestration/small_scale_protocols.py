"""Preregistered, fail-closed protocols for opt-in small-model reproduction."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeVar

from binary_llm.adapters.small_models import (
    LocalSmallModelRequest,
    PYTHIA_70M,
    Pythia70MAdapter,
    SMOLLM_135M,
    SmolLM135MAdapter,
)
from binary_llm.domain.models import (
    BINARY_FAMILY,
    ArtifactRef,
    Budget,
    CheckpointBoundary,
    ClaimRecord,
    ExperimentManifest,
    GateSet,
    ModelIdentity,
    ReproductionClassification,
    ScaleRung,
    SealedBaselineRef,
    SettingDifference,
    StopMode,
    ToleranceSet,
    AmbiguityRegister,
)
from binary_llm.domain.fidelity import (
    BinaryLLMFidelityConfig,
    ProgressiveTrainabilityArm,
)
from binary_llm.domain.paper import (
    BINARYLLM_PAPER_REFERENCE_SETTINGS,
    BINARYLLM_PAPER_SHA256,
)
from binary_llm.domain.reproduction import (
    compare_reproduction_settings,
    reproduction_setting_difference_paths,
    resolved_settings_from_manifest,
)
from binary_llm.domain.validation import REQUIRED_FAILURE_GATE_CATEGORIES, validate_manifest

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_REVISION = re.compile(r"^[0-9a-f]{40,64}$")
_REQUIRED_EVALUATORS = frozenset({"perplexity", "paper_zero_shot", "sign", "dense_control"})


class SmallScaleProtocolKind(StrEnum):
    SCREENING = "screening"
    PAPER_REFERENCE = "paper_reference"


class SmallScaleArmKind(StrEnum):
    REFERENCE_DUAL = "reference_dual"
    NO_INITIALIZATION = "no_initialization"
    ALTERNATIVE_SCHEDULE = "alternative_schedule"
    INCONSISTENT_BACKWARD = "inconsistent_backward"
    ANALYTICAL_ONLY = "analytical_only"
    LEARNED_ONLY = "learned_only"
    DENSE_CONTROL = "dense_control"
    STAGE2_BODY_ONLY = "stage2_body_only"


_REQUIRED_ARMS = frozenset(SmallScaleArmKind)
_SCREENING_CEILINGS = Budget(
    max_tokens=2_000_000,
    max_optimizer_steps=500,
    max_wall_seconds=14_400,
    max_billable_cost=50.0,
    checkpoint_boundary=CheckpointBoundary.PROGRESSIVE_PHASE,
)


@dataclass(frozen=True, slots=True)
class SmallScaleProtocolInputs:
    kind: SmallScaleProtocolKind
    model: ModelIdentity
    parent_checkpoint: ArtifactRef
    baseline_refs: tuple[SealedBaselineRef, ...]
    binary_scope: tuple[str, ...]
    ambiguity_register: AmbiguityRegister
    fidelity_config: BinaryLLMFidelityConfig
    gate_set: GateSet
    tolerance_set: ToleranceSet
    claims: tuple[ClaimRecord, ...]
    corpus_hash: str
    partition_hashes: tuple[str, ...]
    frozen_evaluation_hashes: tuple[str, ...]
    evaluator_protocol_hashes: Mapping[str, str]
    seed_set: tuple[int, ...]
    data_order_hash: str
    budget: Budget
    source_revision: str
    dependency_lock_hash: str
    container_digest: str
    command_prefix: tuple[str, ...]
    sanitized_environment: Mapping[str, Any]
    preregistered_at: str
    screening_evidence_hash: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "evaluator_protocol_hashes", dict(self.evaluator_protocol_hashes))
        object.__setattr__(self, "sanitized_environment", dict(self.sanitized_environment))
        if len(self.partition_hashes) != 20:
            raise ValueError("small-scale protocols require exactly 20 progressive partitions")
        if not self.seed_set:
            raise ValueError("small-scale protocols require an explicit seed set")
        if not self.command_prefix:
            raise ValueError("small-scale protocols require an explicit command")


@dataclass(frozen=True, slots=True)
class SmallScaleProtocolArm:
    arm: SmallScaleArmKind
    manifest: ExperimentManifest
    source_classification: ReproductionClassification
    source_differences: tuple[SettingDifference, ...]
    sign_evaluation_protocol: str
    dense_control_protocol: str
    perplexity_protocol: str
    paper_zero_shot_protocol: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm.value,
            "manifest": self.manifest.to_dict(),
            "source_classification": self.source_classification.value,
            "source_differences": [item.to_dict() for item in self.source_differences],
            "evidence_protocols": {
                "sign": self.sign_evaluation_protocol,
                "dense_control": self.dense_control_protocol,
                "perplexity": self.perplexity_protocol,
                "paper_zero_shot": self.paper_zero_shot_protocol,
            },
        }


@dataclass(frozen=True, slots=True)
class SmallScaleProtocolSuite:
    kind: SmallScaleProtocolKind
    arms: tuple[SmallScaleProtocolArm, ...]
    ambiguity_choices: Mapping[str, str]
    resolved_hashes: tuple[str, ...]
    screening_evidence_hash: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "ambiguity_choices", dict(self.ambiguity_choices))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": self.kind.value,
            "ambiguity_choices": dict(self.ambiguity_choices),
            "resolved_hashes": list(self.resolved_hashes),
            "screening_evidence_hash": self.screening_evidence_hash,
            "arms": [arm.to_dict() for arm in self.arms],
        }


@dataclass(frozen=True, slots=True)
class SmallScalePreflightReport:
    kind: SmallScaleProtocolKind
    manifest_count: int
    resolved_hash_count: int
    source_classifications: tuple[ReproductionClassification, ...]
    compute_allowed: bool = True


def _plain_hash(value: str) -> str:
    return value.removeprefix("sha256:")


def _require_sha256(label: str, value: str) -> str:
    normalized = _plain_hash(value)
    if not _SHA256.fullmatch(normalized):
        raise ValueError(f"{label} must be a resolved lowercase SHA-256")
    return normalized


def _ambiguity_choices(register: AmbiguityRegister) -> dict[str, str]:
    choices: dict[str, str] = {}
    for entry in register.entries:
        if entry.selected_candidate is None:
            raise ValueError(f"ambiguity choice is unresolved: {entry.ambiguity_id}")
        choices[entry.ambiguity_id] = entry.selected_candidate
    return choices


def _resolved_hashes(inputs: SmallScaleProtocolInputs) -> tuple[str, ...]:
    values: list[tuple[str, str]] = [
        ("model config", inputs.model.config_hash),
        ("parent checkpoint", inputs.parent_checkpoint.sha256),
        ("corpus", inputs.corpus_hash),
        ("data order", inputs.data_order_hash),
        ("ambiguity register", inputs.ambiguity_register.register_id),
        ("fidelity config", inputs.fidelity_config.config_hash),
        ("gate set", inputs.gate_set.gate_set_id),
        ("tolerance set", inputs.tolerance_set.tolerance_set_id),
        ("dependency lock", inputs.dependency_lock_hash),
        ("container", inputs.container_digest),
    ]
    values.extend(("model weight", value) for value in inputs.model.weight_hashes)
    values.extend(("tokenizer", value) for value in inputs.model.tokenizer_hashes)
    values.extend(("template", value) for value in inputs.model.template_hashes)
    values.extend(("partition", value) for value in inputs.partition_hashes)
    values.extend(("frozen evaluation", value) for value in inputs.frozen_evaluation_hashes)
    values.extend((f"evaluator {name}", value) for name, value in inputs.evaluator_protocol_hashes.items())
    for baseline in inputs.baseline_refs:
        values.append(("baseline evaluator", baseline.evaluator_revision))
        values.extend(("baseline artifact", ref.sha256) for ref in baseline.artifact_refs)
        values.extend(("baseline output", ref.sha256) for ref in baseline.baseline_output_refs)
    if inputs.kind is SmallScaleProtocolKind.PAPER_REFERENCE:
        if inputs.screening_evidence_hash is None:
            raise ValueError("paper reference allocation requires qualifying screening evidence")
        values.append(("screening evidence", inputs.screening_evidence_hash))
    return tuple(_require_sha256(label, value) for label, value in values)


def _validate_input_identity(inputs: SmallScaleProtocolInputs) -> None:
    expected = PYTHIA_70M if inputs.kind is SmallScaleProtocolKind.SCREENING else SMOLLM_135M
    if inputs.model.model_id != expected.model_id or inputs.model.architecture != expected.architecture:
        raise ValueError(f"{inputs.kind.value} requires the registered {expected.scale_label} model")
    if inputs.model.tied_weights is not expected.tied_weights:
        raise ValueError("model tied-weight identity does not match the registered protocol")
    if not _GIT_REVISION.fullmatch(inputs.model.revision):
        raise ValueError("model revision must be a resolved 40-64 character hexadecimal revision")
    if not _GIT_REVISION.fullmatch(inputs.model.tokenizer_revision):
        raise ValueError("tokenizer revision must be a resolved 40-64 character hexadecimal revision")
    missing_environment = {
        "clean_tree", "hardware_inventory", "compiler_inventory", "offline"
    } - set(inputs.sanitized_environment)
    if missing_environment:
        raise ValueError(f"run environment inventory is incomplete: {sorted(missing_environment)}")
    missing = _REQUIRED_EVALUATORS - set(inputs.evaluator_protocol_hashes)
    extra = set(inputs.evaluator_protocol_hashes) - _REQUIRED_EVALUATORS
    if missing or extra:
        raise ValueError(f"evaluator protocols must be exactly {sorted(_REQUIRED_EVALUATORS)}")


def _validate_budget(kind: SmallScaleProtocolKind, budget: Budget) -> None:
    if budget.checkpoint_boundary is not CheckpointBoundary.PROGRESSIVE_PHASE:
        raise ValueError("small-scale runs must stop and checkpoint at progressive-phase boundaries")
    if kind is SmallScaleProtocolKind.SCREENING:
        limits = (
            ("tokens", budget.max_tokens, _SCREENING_CEILINGS.max_tokens),
            ("optimizer steps", budget.max_optimizer_steps, _SCREENING_CEILINGS.max_optimizer_steps),
            ("wall time", budget.max_wall_seconds, _SCREENING_CEILINGS.max_wall_seconds),
            ("billable cost", budget.max_billable_cost, _SCREENING_CEILINGS.max_billable_cost),
        )
        exceeded = [name for name, actual, maximum in limits if actual > maximum]
        if exceeded:
            raise ValueError(f"screening budget exceeds bounded ceiling: {', '.join(exceeded)}")


def _base_configs(
    inputs: SmallScaleProtocolInputs, choices: Mapping[str, str]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    stage1_steps = 5 if inputs.kind is SmallScaleProtocolKind.SCREENING else 50
    operator = {
        "mode": "binary_progressive",
        "forward": "paper_progressive",
        "backward": "analytical_derivative",
        "scale_variant": "dual",
        "analytical_recompute": "every_update",
        "learned_scale_initial": 1.0,
        "ambiguity_choices": dict(choices),
        "scale_evidence": ["quality", "scale_bytes", "artifact_bytes", "distribution", "non_finite"],
        "fidelity_config_hash": inputs.fidelity_config.config_hash,
    }
    stage1 = {
        "enabled": True,
        "steps": stage1_steps,
        "paper_reference_steps": 50,
        "original_dense_parameters": "frozen",
        "optimizer_visibility": "input_channel_scales_only",
    }
    progressive = {
        "phase_count": 20,
        "schedule": "paper_exponential",
        "schedule_formula": "1.3*exp(0.22*c)-1.3",
        "phase_index": choices["progressive.phase_index"],
        "partition_binding": "one_hash_per_phase",
        "final_views": ["progressive", "sign"],
        "checkpoint_boundary": "progressive_phase",
        "trainability_arm": ProgressiveTrainabilityArm.ALL_MODEL_PARAMETERS.value,
    }
    return operator, stage1, progressive


def _arm_configs(
    arm: SmallScaleArmKind,
    operator: dict[str, Any],
    stage1: dict[str, Any],
    progressive: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str | None]:
    operator = {**operator}
    stage1 = {**stage1}
    progressive = {**progressive}
    changed_path: str | None = None
    if arm is SmallScaleArmKind.NO_INITIALIZATION:
        stage1["enabled"] = False
        changed_path = "/optimization/stage1_config/enabled"
    elif arm is SmallScaleArmKind.ALTERNATIVE_SCHEDULE:
        progressive["schedule"] = "linear_matched_endpoints"
        changed_path = "/schedule/progressive_config/schedule"
    elif arm is SmallScaleArmKind.INCONSISTENT_BACKWARD:
        operator["backward"] = "straight_through_control"
        changed_path = "/operator/operator_config/backward"
    elif arm is SmallScaleArmKind.ANALYTICAL_ONLY:
        operator["scale_variant"] = "analytical_only"
        changed_path = "/operator/operator_config/scale_variant"
    elif arm is SmallScaleArmKind.LEARNED_ONLY:
        operator["scale_variant"] = "learned_only"
        changed_path = "/operator/operator_config/scale_variant"
    elif arm is SmallScaleArmKind.DENSE_CONTROL:
        operator["mode"] = "dense_control"
        changed_path = "/operator/operator_config/mode"
    elif arm is SmallScaleArmKind.STAGE2_BODY_ONLY:
        progressive["trainability_arm"] = (
            ProgressiveTrainabilityArm.BINARY_BODY_ONLY.value
        )
        changed_path = "/schedule/progressive_config/trainability_arm"
    return operator, stage1, progressive, changed_path


def _manifest(
    inputs: SmallScaleProtocolInputs,
    arm: SmallScaleArmKind,
    choices: Mapping[str, str],
) -> tuple[ExperimentManifest, str | None]:
    operator, stage1, progressive = _base_configs(inputs, choices)
    operator, stage1, progressive, changed_path = _arm_configs(
        arm, operator, stage1, progressive
    )
    evaluators = inputs.evaluator_protocol_hashes
    manifest = ExperimentManifest(
        experiment_id=f"small-{inputs.kind.value}-{arm.value}",
        family=BINARY_FAMILY,
        hypothesis_refs=(f"binaryllm-small-{arm.value}",),
        source_claim_refs=tuple(claim.claim_id for claim in inputs.claims),
        model=inputs.model,
        scale_rung=ScaleRung.SMALL,
        parent_checkpoint=inputs.parent_checkpoint,
        baseline_refs=inputs.baseline_refs,
        binary_scope=inputs.binary_scope,
        operator_config=operator,
        ambiguity_register_ref=inputs.ambiguity_register.register_id,
        stage1_config=stage1,
        progressive_config=progressive,
        recovery_config=None,
        teacher_refs=(),
        corpus_ref=inputs.corpus_hash,
        partition_refs=inputs.partition_hashes,
        frozen_evaluation_refs=inputs.frozen_evaluation_hashes,
        evaluator_protocols=tuple(evaluators[name] for name in sorted(evaluators)),
        seed_set=inputs.seed_set,
        determinism_policy={"deterministic": True, "data_order_hash": inputs.data_order_hash},
        budget=inputs.budget,
        gate_set_ref=inputs.gate_set.gate_set_id,
        tolerance_set_ref=inputs.tolerance_set.tolerance_set_id,
        artifact_format=None,
        device_protocol=None,
        source_revision=inputs.source_revision,
        dependency_lock_hash=inputs.dependency_lock_hash,
        container_digest=inputs.container_digest,
        command=(*inputs.command_prefix, "--arm", arm.value),
        sanitized_environment=inputs.sanitized_environment,
        preregistered_at=inputs.preregistered_at,
    )
    return manifest, changed_path


def _paper_difference_refs(manifest: ExperimentManifest) -> dict[str, str]:
    resolved = resolved_settings_from_manifest(manifest)
    return {
        path: f"source-difference:binaryllm-paper:{BINARYLLM_PAPER_SHA256}:{path}"
        for path in reproduction_setting_difference_paths(
            BINARYLLM_PAPER_REFERENCE_SETTINGS, resolved
        )
    }


def build_small_scale_protocol_suite(inputs: SmallScaleProtocolInputs) -> SmallScaleProtocolSuite:
    """Build all matched screening or paper-reference arms without loading a model."""
    _validate_input_identity(inputs)
    _validate_budget(inputs.kind, inputs.budget)
    hashes = _resolved_hashes(inputs)
    choices = _ambiguity_choices(inputs.ambiguity_register)
    built = tuple((arm, *_manifest(inputs, arm, choices)) for arm in SmallScaleArmKind)
    protocol_arms: list[SmallScaleProtocolArm] = []
    for arm, manifest, _changed_path in built:
        resolved = resolved_settings_from_manifest(manifest)
        differences = compare_reproduction_settings(
            BINARYLLM_PAPER_REFERENCE_SETTINGS,
            resolved,
            ablation_refs=_paper_difference_refs(manifest),
        )
        classification = (
            ReproductionClassification.MODIFIED_REPRODUCTION
            if differences
            else ReproductionClassification.REFERENCE_REPRODUCTION
        )
        evaluators = inputs.evaluator_protocol_hashes
        protocol_arms.append(
            SmallScaleProtocolArm(
                arm=arm,
                manifest=manifest,
                source_classification=classification,
                source_differences=differences,
                sign_evaluation_protocol=evaluators["sign"],
                dense_control_protocol=evaluators["dense_control"],
                perplexity_protocol=evaluators["perplexity"],
                paper_zero_shot_protocol=evaluators["paper_zero_shot"],
            )
        )
    suite = SmallScaleProtocolSuite(
        inputs.kind,
        tuple(protocol_arms),
        choices,
        hashes,
        inputs.screening_evidence_hash,
    )
    preflight_small_scale_protocol(suite, inputs)
    return suite


def preflight_small_scale_protocol(
    suite: SmallScaleProtocolSuite,
    inputs: SmallScaleProtocolInputs,
) -> SmallScalePreflightReport:
    """Verify every scientific and resource prerequisite before model allocation."""
    _validate_input_identity(inputs)
    _validate_budget(suite.kind, inputs.budget)
    expected_hashes = _resolved_hashes(inputs)
    if suite.resolved_hashes != expected_hashes:
        raise ValueError("protocol resolved-hash inventory does not match current inputs")
    if suite.screening_evidence_hash != inputs.screening_evidence_hash:
        raise ValueError("protocol screening evidence does not match current inputs")
    if suite.ambiguity_choices != _ambiguity_choices(inputs.ambiguity_register):
        raise ValueError("protocol ambiguity choices do not match the resolved register")
    if set(inputs.fidelity_config.resolution_ledger) != set(inputs.fidelity_config.values):
        raise ValueError("fidelity configuration resolution ledger is incomplete")
    arms = {item.arm for item in suite.arms}
    if arms != _REQUIRED_ARMS or len(suite.arms) != len(_REQUIRED_ARMS):
        raise ValueError("protocol suite must contain every preregistered arm exactly once")
    unsafe_gates = tuple(
        definition.gate_id
        for definition in inputs.gate_set.definitions
        if definition.category in REQUIRED_FAILURE_GATE_CATEGORIES
        and definition.stop_mode is not StopMode.IMMEDIATE_SAFE_BOUNDARY
    )
    if unsafe_gates:
        raise ValueError(f"failure gates must stop at the next safe boundary: {unsafe_gates}")
    for item in suite.arms:
        validate_manifest(
            item.manifest,
            ambiguity_register=inputs.ambiguity_register,
            gate_set=inputs.gate_set,
            tolerance_set=inputs.tolerance_set,
            claims=inputs.claims,
        )
        if item.manifest.model != inputs.model:
            raise ValueError("protocol arm model identity differs from the preflight identity")
        if item.manifest.operator_config.get("fidelity_config_hash") != inputs.fidelity_config.config_hash:
            raise ValueError("protocol arm is not bound to the preregistered fidelity configuration")
        if item.manifest.seed_set != inputs.seed_set:
            raise ValueError("matched protocol arms must use the preregistered seed set")
        if item.manifest.determinism_policy.get("data_order_hash") != inputs.data_order_hash:
            raise ValueError("matched protocol arms must use the preregistered data order")
        if len(item.manifest.partition_refs) != 20:
            raise ValueError("every protocol arm must bind all 20 progressive partitions")
        if item.manifest.budget != inputs.budget:
            raise ValueError("matched protocol arms must use the preregistered four-part budget")
        actual_differences = compare_reproduction_settings(
            BINARYLLM_PAPER_REFERENCE_SETTINGS,
            resolved_settings_from_manifest(item.manifest),
            ablation_refs=_paper_difference_refs(item.manifest),
        )
        if actual_differences != item.source_differences:
            raise ValueError("recorded source differences do not match the resolved manifest")
        expected = bool(item.source_differences)
        modified = item.source_classification is ReproductionClassification.MODIFIED_REPRODUCTION
        if expected != modified:
            raise ValueError("source-difference classification is inconsistent")
    reference = next(item for item in suite.arms if item.arm is SmallScaleArmKind.REFERENCE_DUAL)
    expected_steps = 5 if suite.kind is SmallScaleProtocolKind.SCREENING else 50
    if reference.manifest.stage1_config["steps"] != expected_steps:
        raise ValueError("reference Stage 1 step count does not match protocol kind")
    if reference.manifest.progressive_config["phase_count"] != 20:
        raise ValueError("reference protocol must preregister exactly 20 progressive phases")
    if (
        reference.manifest.progressive_config["trainability_arm"]
        != ProgressiveTrainabilityArm.ALL_MODEL_PARAMETERS.value
    ):
        raise ValueError("reference protocol requires all-model-parameter Stage 2 trainability")
    body_only = next(
        item for item in suite.arms if item.arm is SmallScaleArmKind.STAGE2_BODY_ONLY
    )
    expected_body_path = "/schedule/progressive_config"
    body_paths = {difference.path for difference in body_only.source_differences}
    differing_progressive_keys = {
        key
        for key in reference.manifest.progressive_config
        if reference.manifest.progressive_config[key]
        != body_only.manifest.progressive_config[key]
    }
    if (
        body_only.manifest.progressive_config["trainability_arm"]
        != ProgressiveTrainabilityArm.BINARY_BODY_ONLY.value
        or body_only.source_classification
        is not ReproductionClassification.MODIFIED_REPRODUCTION
        or expected_body_path not in body_paths
        or differing_progressive_keys != {"trainability_arm"}
    ):
        raise ValueError("body-only arm must be an exact classified trainability ablation")
    return SmallScalePreflightReport(
        suite.kind,
        len(suite.arms),
        len(suite.resolved_hashes),
        tuple(item.source_classification for item in suite.arms),
    )


_Allocated = TypeVar("_Allocated")


def allocate_small_scale_model(
    suite: SmallScaleProtocolSuite,
    inputs: SmallScaleProtocolInputs,
    request: LocalSmallModelRequest,
    allocator: Callable[[LocalSmallModelRequest], _Allocated] | None = None,
) -> _Allocated:
    """Run complete protocol preflight, then and only then invoke the model allocator."""
    preflight_small_scale_protocol(suite, inputs)
    if request.identity != inputs.model:
        raise ValueError("allocation request identity differs from the preregistered model")
    adapter = Pythia70MAdapter() if suite.kind is SmallScaleProtocolKind.SCREENING else SmolLM135MAdapter()
    load = allocator or adapter.load_local
    return load(request)


def protocol_catalog() -> dict[str, Any]:
    """Return the command-visible preregistration contract, never unresolved defaults."""
    return {
        "schema_version": 1,
        "protocols": {
            "screening": {
                "model": PYTHIA_70M.model_id,
                "stage1_steps": 5,
                "progressive_phases": 20,
                "budget_ceilings": _SCREENING_CEILINGS.to_dict(),
            },
            "paper_reference": {
                "model": SMOLLM_135M.model_id,
                "stage1_steps": 50,
                "progressive_phases": 20,
            },
        },
        "arms": [item.value for item in SmallScaleArmKind],
        "required_evidence": sorted(_REQUIRED_EVALUATORS),
        "preallocation_requirements": [
            "resolved_hashes", "qualifying_screening_evidence_for_reference",
            "ambiguity_choices", "seed_set", "data_order_hash",
            "fidelity_config_hash", "complete_field_provenance",
            "source_commit", "clean_tree", "hardware_inventory", "compiler_inventory",
            "max_tokens", "max_optimizer_steps", "max_wall_seconds", "max_billable_cost",
            "failure_gates", "progressive_phase_checkpoint_boundary",
            "source_difference_classification",
        ],
    }


def validate_serialized_small_scale_suite(value: Mapping[str, Any]) -> None:
    """Fail-closed structural validation for command-produced canonical suite JSON."""
    if value.get("schema_version") != 1:
        raise ValueError("unsupported small-scale protocol schema")
    kind = SmallScaleProtocolKind(value.get("kind"))
    choices = value.get("ambiguity_choices")
    hashes = value.get("resolved_hashes")
    arms = value.get("arms")
    if not isinstance(choices, dict) or not choices or any(not key or not selected for key, selected in choices.items()):
        raise ValueError("serialized protocol requires explicit ambiguity choices")
    if not isinstance(hashes, list) or not hashes:
        raise ValueError("serialized protocol requires a resolved hash inventory")
    for value_hash in hashes:
        _require_sha256("serialized protocol hash", value_hash)
    if not isinstance(arms, list) or {item.get("arm") for item in arms} != {item.value for item in _REQUIRED_ARMS}:
        raise ValueError("serialized protocol does not contain every required arm")
    for arm in arms:
        manifest = arm.get("manifest", {})
        budget = manifest.get("budget", {})
        if len(manifest.get("partition_refs", [])) != 20:
            raise ValueError("serialized arm must bind exactly 20 partitions")
        if budget.get("checkpoint_boundary") != CheckpointBoundary.PROGRESSIVE_PHASE.value:
            raise ValueError("serialized arm checkpoint boundary is unsafe")
        if any(not budget.get(name) for name in ("max_tokens", "max_optimizer_steps", "max_wall_seconds")):
            raise ValueError("serialized arm has an incomplete resource budget")
        if budget.get("max_billable_cost") is None or budget["max_billable_cost"] < 0:
            raise ValueError("serialized arm has an incomplete cost budget")
        policy = manifest.get("determinism_policy", {})
        _require_sha256("serialized data order", policy.get("data_order_hash", ""))
        if not manifest.get("seed_set"):
            raise ValueError("serialized arm requires a seed set")
        ReproductionClassification(arm.get("source_classification"))
    if kind is SmallScaleProtocolKind.PAPER_REFERENCE:
        if not value.get("screening_evidence_hash"):
            raise ValueError("serialized paper reference requires qualifying screening evidence")
        _require_sha256("serialized screening evidence", value["screening_evidence_hash"])
        reference = next(item for item in arms if item["arm"] == SmallScaleArmKind.REFERENCE_DUAL.value)
        if reference["manifest"].get("stage1_config", {}).get("steps") != 50:
            raise ValueError("paper reference must use exactly 50 Stage 1 steps")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("catalog", help="print the preregistered small-scale protocol catalog")
    validate_parser = subparsers.add_parser("validate", help="validate a generated protocol suite before allocation")
    validate_parser.add_argument("--suite", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "catalog":
        print(json.dumps(protocol_catalog(), sort_keys=True))
        return 0
    value = json.loads(args.suite.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("protocol suite JSON must contain an object")
    validate_serialized_small_scale_suite(value)
    print(json.dumps({"compute_allowed": True, "suite": str(args.suite)}, sort_keys=True))
    return 0


__all__ = [
    "SmallScaleArmKind",
    "SmallScalePreflightReport",
    "SmallScaleProtocolArm",
    "SmallScaleProtocolInputs",
    "SmallScaleProtocolKind",
    "SmallScaleProtocolSuite",
    "allocate_small_scale_model",
    "build_small_scale_protocol_suite",
    "main",
    "preflight_small_scale_protocol",
    "protocol_catalog",
    "validate_serialized_small_scale_suite",
]


if __name__ == "__main__":
    raise SystemExit(main())
