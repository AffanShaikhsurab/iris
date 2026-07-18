from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
import torch

from binary_llm.adapters import (
    ActiveRepresentation,
    BinaryLinearConfig,
    ConfiguredBinaryLinearFactory,
    TinyCausalLMConfig,
    TinyCausalModelAdapter,
    TrainingActivationMode,
    build_seeded_tiny_causal_lm,
)
from binary_llm.domain import (
    Budget,
    CheckpointBoundary,
    ArtifactRef,
    ModelIdentity,
    SealViolation,
    SealedBaselineRef,
    canonical_json_bytes,
    sha256_bytes,
)
from binary_llm.math import (
    AnalyticalScaleGradient,
    OperatorPrecision,
    ProgressiveOperatorConfig,
    ZeroSignRule,
)
from binary_llm.orchestration import (
    CausalBatch,
    LogitSupervisionKind,
    NoProgressPolicy,
    RecoveryCorpusMix,
    RecoveryCorpusPlan,
    RecoveryRunStatus,
    RecoveryTrainerBackend,
    RecoveryTrainerConfig,
    SealedAsset,
    SealedAssetCategory,
    SealedInventory,
    Stage1OptimizerConfig,
    SupervisionMode,
    TeacherIdentity,
    TeacherOutput,
    TeacherRole,
    TeacherRouter,
    TeacherRoutingConfig,
    TextLevelEvaluation,
    FilesystemArtifactStore,
    aggregate_teacher_weight_artifacts,
    resolve_verified_teacher_binding,
    verify_sealed_inventory,
    validate_recovery_token_mix,
)
from binary_llm.orchestration.corpus import INITIAL_TOKEN_ALLOCATION, ProvenanceRecord


def _record(
    record_id: str,
    capability_slice: str,
    token_count: int,
    *,
    teacher_identity: str | None = None,
    generation_settings=None,
) -> ProvenanceRecord:
    normalized = {"messages": [{"role": "user", "content": record_id}]}
    return ProvenanceRecord(
        record_id=record_id,
        content_hash=sha256_bytes(canonical_json_bytes(normalized)),
        source=f"source:{record_id}",
        license_or_terms="test-terms",
        permitted_use="research-training",
        transformation_history=("normalized-v1",),
        semantic_family_id=f"semantic:{record_id}",
        split_family_id=f"split:{record_id}",
        deduplication_key=f"dedup:{record_id}",
        fuzzy_cluster_id=f"fuzzy:{record_id}",
        target_type="causal_lm",
        capability_slice=capability_slice,
        token_count=token_count,
        synthetic=False,
        executable_or_human_verified=True,
        normalized_record=normalized,
        teacher_identity=teacher_identity,
        generation_settings=generation_settings,
    )


def _initial_records() -> tuple[ProvenanceRecord, ...]:
    return tuple(
        _record(f"record-{name}", name, int(fraction * 100))
        for name, fraction in INITIAL_TOKEN_ALLOCATION
    )


def _plan() -> RecoveryCorpusPlan:
    initial = RecoveryCorpusMix(
        "initial-30-15-10-20-10-10-5",
        INITIAL_TOKEN_ALLOCATION,
        preregistered=True,
        initial=True,
    )
    alternative = RecoveryCorpusMix(
        "alternative-balanced",
        (
            ("broad_text", 0.20),
            ("math", 0.20),
            ("code", 0.15),
            ("iris_tools", 0.15),
            ("conversation", 0.10),
            ("safety_transitions", 0.10),
            ("adversarial", 0.10),
        ),
        preregistered=True,
    )
    return RecoveryCorpusPlan(initial, (alternative,))


def _identity(
    role: TeacherRole,
    *,
    tokenizer_id: str = "student-tokenizer",
    sealed: bool = False,
    baseline_role: str | None = None,
    model_id: str | None = None,
    revision: str = "revision-1",
    artifact_hash: str | None = None,
) -> TeacherIdentity:
    return TeacherIdentity(
        teacher_id=f"{role.value}-teacher",
        role=role,
        model_id=model_id or f"model:{role.value}",
        revision=revision,
        tokenizer_id=tokenizer_id,
        license_or_terms="teacher-terms-v1",
        permitted_use="synthetic-training-and-evaluation",
        generation_settings={"temperature": 0.0, "max_new_tokens": 32},
        artifact_hash=artifact_hash or (
            ("a" if role is TeacherRole.BEHAVIORAL else "b") * 64
        ),
        sealed=sealed,
        baseline_role=baseline_role,
    )


def _verified_behavioral_teacher(tmp_path: Path):
    source = tmp_path / "sealed"
    store_root = tmp_path / "store"
    assets: list[SealedAsset] = []
    oracles: list[SealedBaselineRef] = []
    payloads: dict[str, bytes] = {}
    for role in ("base", "adapter", "bf16", "q4"):
        role_assets: dict[SealedAssetCategory, SealedAsset] = {}
        for category in (
            SealedAssetCategory.WEIGHTS,
            SealedAssetCategory.TOKENIZERS,
            SealedAssetCategory.TEMPLATES,
            SealedAssetCategory.CONFIGURATIONS,
            SealedAssetCategory.BASELINE_OUTPUTS,
            SealedAssetCategory.EVALUATORS,
        ):
            artifact_id = f"{role}-{category.value}"
            payload = f"{role}:{category.value}".encode()
            reference = ArtifactRef(
                artifact_id,
                category.value,
                hashlib.sha256(payload).hexdigest(),
                len(payload),
                "application/octet-stream",
            )
            relative_path = f"{category.value}/{artifact_id}.bin"
            path = source / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            sealed = SealedAsset(reference, category, relative_path, (role,))
            assets.append(sealed)
            role_assets[category] = sealed
            payloads[artifact_id] = payload
        weight = role_assets[SealedAssetCategory.WEIGHTS].asset
        tokenizer = role_assets[SealedAssetCategory.TOKENIZERS].asset
        template = role_assets[SealedAssetCategory.TEMPLATES].asset
        config = role_assets[SealedAssetCategory.CONFIGURATIONS].asset
        output = role_assets[SealedAssetCategory.BASELINE_OUTPUTS].asset
        evaluator = role_assets[SealedAssetCategory.EVALUATORS].asset
        oracles.append(
            SealedBaselineRef(
                f"baseline-{role}",
                role,
                ModelIdentity(
                    f"model-{role}",
                    "sealed-revision",
                    "tokenizer-revision",
                    "test",
                    100,
                    config.sha256,
                    (tokenizer.sha256,),
                    (template.sha256,),
                    (weight.sha256,),
                    False,
                ),
                (weight,),
                evaluator.sha256,
                (output,),
                "2026-01-01T00:00:00Z",
            )
        )
    dataset_payload = b"dataset"
    dataset_ref = ArtifactRef(
        "dataset",
        "datasets",
        hashlib.sha256(dataset_payload).hexdigest(),
        len(dataset_payload),
        "application/octet-stream",
    )
    dataset_path = source / "datasets/dataset.bin"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_bytes(dataset_payload)
    assets.append(
        SealedAsset(
            dataset_ref,
            SealedAssetCategory.DATASETS,
            "datasets/dataset.bin",
            ("base", "adapter", "bf16", "q4"),
        )
    )
    inventory = SealedInventory("teacher-inventory", tuple(assets), tuple(oracles))
    report = verify_sealed_inventory(inventory, source)
    store = FilesystemArtifactStore(store_root)
    bf16_assets = tuple(asset for asset in assets if "bf16" in asset.oracle_roles)
    for asset in bf16_assets:
        if asset.category in {
            SealedAssetCategory.WEIGHTS,
            SealedAssetCategory.TOKENIZERS,
            SealedAssetCategory.TEMPLATES,
            SealedAssetCategory.CONFIGURATIONS,
        }:
            store.put_bytes(asset.asset, payloads[asset.asset.artifact_id])
    weights = tuple(
        asset.asset
        for asset in bf16_assets
        if asset.category is SealedAssetCategory.WEIGHTS
    )
    identity = _identity(
        TeacherRole.BEHAVIORAL,
        sealed=True,
        baseline_role="bf16",
        model_id="model-bf16",
        revision="sealed-revision",
        artifact_hash=aggregate_teacher_weight_artifacts(weights),
    )
    binding = resolve_verified_teacher_binding(identity, inventory, report, store)
    return identity, binding, inventory, report, store


@dataclass
class _Teacher:
    teacher_identity: TeacherIdentity
    vocabulary_size: int = 13

    def identity(self) -> TeacherIdentity:
        return self.teacher_identity

    def supervise(
        self,
        record: ProvenanceRecord,
        batch: CausalBatch,
        mode: SupervisionMode,
    ) -> TeacherOutput:
        if mode is SupervisionMode.SHARED_TOKENIZER_LOGITS:
            shape = (*batch.input_ids.shape, self.vocabulary_size)
            return TeacherOutput(
                record.record_id,
                mode,
                logits=torch.zeros(shape),
                logit_kind=LogitSupervisionKind.TOP_K,
                top_k=3,
            )
        if mode is SupervisionMode.SHARED_TOKENIZER_SEQUENCE:
            return TeacherOutput(
                record.record_id,
                mode,
                sequence_ids=batch.input_ids.detach().clone(),
            )
        return TeacherOutput(record.record_id, mode, text=f"teacher:{record.record_id}")


def _model():
    adapter = TinyCausalModelAdapter()
    model = build_seeded_tiny_causal_lm(
        seed=41,
        config=TinyCausalLMConfig(
            vocab_size=13,
            hidden_size=4,
            intermediate_size=6,
            max_sequence_length=5,
        ),
    )
    adapter.replace_linears(
        model,
        ConfiguredBinaryLinearFactory(
            BinaryLinearConfig(
                scale_parameterization="positive_exp",
                training_activation_mode=TrainingActivationMode.WEIGHT_ONLY,
                progressive_operator=ProgressiveOperatorConfig(
                    precision=OperatorPrecision.FLOAT32,
                    analytical_scale_gradient=AnalyticalScaleGradient.DETACHED,
                ),
                zero_sign_rule=ZeroSignRule.POSITIVE,
            )
        ),
    )
    return adapter, model


def _trainer(steps: int = 1) -> RecoveryTrainerBackend:
    return RecoveryTrainerBackend(
        RecoveryTrainerConfig(
            optimizer=Stage1OptimizerConfig(
                kind="adamw", learning_rate=0.005, weight_decay=0.0
            ),
            seed=43,
            max_optimizer_steps=steps,
            target_representation=ActiveRepresentation.TRAINING,
            no_progress_policy=NoProgressPolicy(
                "calibration", "frozen-capability"
            ),
        )
    )


def _batches(records: tuple[ProvenanceRecord, ...]):
    tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    return {
        record.record_id: CausalBatch(record.record_id, tokens.clone())
        for record in records
    }


class _Metric:
    def __init__(self, scores: tuple[float, ...]) -> None:
        self.scores = scores
        self.calls = 0

    def __call__(self, forward) -> float:
        forward(torch.tensor([[1, 2, 3, 4]], dtype=torch.long), purpose="metric")
        score = self.scores[min(self.calls, len(self.scores) - 1)]
        self.calls += 1
        return score


def test_registered_recovery_mix_is_enforced_by_tokens_not_record_counts() -> None:
    records = _initial_records()
    evidence = validate_recovery_token_mix(records, _plan().initial_mix)

    assert evidence.total_tokens == 100
    assert evidence.record_count == 7
    assert {item.capability_slice: item.actual_tokens for item in evidence.slices} == {
        name: int(fraction * 100) for name, fraction in INITIAL_TOKEN_ALLOCATION
    }
    assert len({record.token_count for record in records}) > 1

    row_balanced = tuple(
        _record(f"equal-{index}", name, 10)
        for index, (name, _) in enumerate(INITIAL_TOKEN_ALLOCATION)
    )
    with pytest.raises(ValueError, match="token allocation"):
        validate_recovery_token_mix(row_balanced, _plan().initial_mix)


def test_teacher_router_uses_only_sealed_bf16_for_behavior_and_records_provenance(
    tmp_path: Path,
) -> None:
    bf16, binding, _, _, _ = _verified_behavioral_teacher(tmp_path)
    broad = _identity(TeacherRole.BROAD)
    router = TeacherRouter(
        TeacherRoutingConfig(
            student_tokenizer_id="student-tokenizer",
            sealed_bf16_teacher_identity_id=bf16.identity_id,
            sealed_bf16_inventory_id=binding.inventory_id,
            behavioral_mode=SupervisionMode.SHARED_TOKENIZER_SEQUENCE,
            broad_mode=SupervisionMode.SHARED_TOKENIZER_LOGITS,
        ),
        behavioral_teacher=_Teacher(bf16),
        broad_teacher=_Teacher(broad),
        behavioral_binding=binding,
    )
    batch = CausalBatch("batch", torch.tensor([[1, 2, 3]], dtype=torch.long))

    behavior = router.supervise(_record("tool", "iris_tools", 3), batch)
    broad_result = router.supervise(_record("math", "math", 3), batch)

    assert behavior is not None and broad_result is not None
    assert behavior.provenance.teacher_identity_id == bf16.identity_id
    assert behavior.provenance.role is TeacherRole.BEHAVIORAL
    assert behavior.output.mode is SupervisionMode.SHARED_TOKENIZER_SEQUENCE
    assert broad_result.provenance.teacher_identity_id == broad.identity_id
    assert broad_result.output.mode is SupervisionMode.SHARED_TOKENIZER_LOGITS
    assert broad_result.output.logit_kind is LogitSupervisionKind.TOP_K
    assert dict(behavior.provenance.generation_settings) == dict(bf16.generation_settings)

    unsealed = _identity(TeacherRole.BEHAVIORAL)
    with pytest.raises(SealViolation) as raised:
        TeacherRouter(
            TeacherRoutingConfig(
                "student-tokenizer",
                bf16.identity_id,
                sealed_bf16_inventory_id=binding.inventory_id,
                behavioral_mode=SupervisionMode.SHARED_TOKENIZER_SEQUENCE,
            ),
            behavioral_teacher=_Teacher(unsealed),
        )
    assert raised.value.code == "seal.teacher_binding_mismatch"


def test_teacher_binding_is_durable_deterministic_and_fails_closed(tmp_path: Path) -> None:
    identity, binding, inventory, report, store = _verified_behavioral_teacher(tmp_path)

    reopened = FilesystemArtifactStore(store.root)
    equivalent = resolve_verified_teacher_binding(identity, inventory, report, reopened)
    assert equivalent == binding
    assert equivalent.binding_id == binding.binding_id
    assert binding.weight_aggregate_sha256 == identity.artifact_hash

    with pytest.raises(SealViolation) as raised:
        TeacherRouter(
            TeacherRoutingConfig(
                "student-tokenizer",
                identity.identity_id,
                sealed_bf16_inventory_id=binding.inventory_id,
                behavioral_mode=SupervisionMode.SHARED_TOKENIZER_SEQUENCE,
            ),
            behavioral_teacher=_Teacher(identity),
        )
    assert raised.value.code == "seal.teacher_binding_mismatch"

    failed_report = replace(report, verified_asset_ids=report.verified_asset_ids[:-1])
    with pytest.raises(SealViolation) as raised:
        resolve_verified_teacher_binding(identity, inventory, failed_report, reopened)
    assert raised.value.code == "seal.teacher_report_mismatch"

    for changed in (
        replace(identity, baseline_role="q4"),
        replace(identity, model_id="wrong-model"),
        replace(identity, revision="wrong-revision"),
        replace(identity, artifact_hash="0" * 64),
    ):
        with pytest.raises(SealViolation):
            resolve_verified_teacher_binding(changed, inventory, report, reopened)

    with pytest.raises(SealViolation) as raised:
        TeacherRouter(
            TeacherRoutingConfig(
                "student-tokenizer",
                identity.identity_id,
                sealed_bf16_inventory_id="another-inventory",
                behavioral_mode=SupervisionMode.SHARED_TOKENIZER_SEQUENCE,
            ),
            behavioral_teacher=_Teacher(identity),
            behavioral_binding=binding,
        )
    assert raised.value.code == "seal.teacher_binding_mismatch"


@pytest.mark.parametrize("failure_kind", ["missing", "corrupt", "wrong_metadata"])
def test_teacher_binding_wraps_store_integrity_failures(
    tmp_path: Path, failure_kind: str
) -> None:
    identity, binding, inventory, report, store = _verified_behavioral_teacher(tmp_path)
    target = binding.tokenizer_artifacts[0]
    if failure_kind == "missing":
        store.object_path(target.sha256).unlink()
    elif failure_kind == "corrupt":
        store.object_path(target.sha256).write_bytes(b"corrupt")
    else:
        store.reference_path(target.artifact_id).write_bytes(
            canonical_json_bytes(target.to_dict())
        )

    with pytest.raises(SealViolation) as raised:
        resolve_verified_teacher_binding(identity, inventory, report, store)
    assert raised.value.code == "seal.teacher_artifact_resolution_failed"
    assert raised.value.affected_ids["artifact_ids"] == (target.artifact_id,)


def test_teacher_router_rechecks_identity_before_accepting_output(tmp_path: Path) -> None:
    identity, binding, _, _, _ = _verified_behavioral_teacher(tmp_path)
    teacher = _Teacher(identity)
    router = TeacherRouter(
        TeacherRoutingConfig(
            "student-tokenizer",
            identity.identity_id,
            sealed_bf16_inventory_id=binding.inventory_id,
            behavioral_mode=SupervisionMode.SHARED_TOKENIZER_SEQUENCE,
        ),
        behavioral_teacher=teacher,
        behavioral_binding=binding,
    )
    teacher.teacher_identity = replace(identity, teacher_id="substituted")

    with pytest.raises(SealViolation) as raised:
        router.supervise(
            _record("tool-substitution", "iris_tools", 3),
            CausalBatch("batch", torch.tensor([[1, 2, 3]], dtype=torch.long)),
        )
    assert raised.value.code == "seal.teacher_binding_mismatch"


def test_different_tokenizer_forbids_token_loss_and_requires_text_level_checks() -> None:
    broad = _identity(TeacherRole.BROAD, tokenizer_id="other-tokenizer")
    with pytest.raises(ValueError, match="different tokenizers require text"):
        TeacherRouter(
            TeacherRoutingConfig(
                "student-tokenizer",
                "unused-sealed-bf16",
                broad_mode=SupervisionMode.SHARED_TOKENIZER_LOGITS,
            ),
            broad_teacher=_Teacher(broad),
        )

    router = TeacherRouter(
        TeacherRoutingConfig(
            "student-tokenizer",
            "unused-sealed-bf16",
            broad_mode=SupervisionMode.DIFFERENT_TOKENIZER_TEXT_EXECUTION,
        ),
        broad_teacher=_Teacher(broad),
    )
    records = _initial_records()
    adapter, model = _model()

    with pytest.raises(ValueError, match="execution checks or a registered rubric"):
        _trainer().run(
            run_id="missing-text-evaluator",
            model=model,
            adapter=adapter,
            records=records,
            batches=_batches(records),
            corpus_plan=_plan(),
            selected_mix_id=_plan().initial_mix.mix_id,
            teacher_router=router,
            calibration_evaluator=_Metric((0.0, 0.0)),
            frozen_capability_evaluator=_Metric((0.0, 0.1)),
        )

    adapter, model = _model()

    def execution_check(supervision, forward, batch):
        forward(batch.input_ids, purpose="text_execution_check")
        return TextLevelEvaluation(
            record_id=supervision.output.record_id,
            mode=supervision.output.mode,
            score=1.0,
            passed=True,
            evaluator_id="sandbox-execution-v1",
        )

    result = _trainer().run(
        run_id="text-level-recovery",
        model=model,
        adapter=adapter,
        records=records,
        batches=_batches(records),
        corpus_plan=_plan(),
        selected_mix_id=_plan().initial_mix.mix_id,
        teacher_router=router,
        calibration_evaluator=_Metric((0.0, 0.0)),
        frozen_capability_evaluator=_Metric((0.0, 0.1)),
        text_level_evaluator=execution_check,
    )

    assert result.status is RecoveryRunStatus.COMPLETED
    assert result.steps[0].teacher_loss is None
    assert result.steps[0].text_evaluation is not None
    assert result.steps[0].text_evaluation.evaluator_id == "sandbox-execution-v1"
    assert result.teacher_provenance[0].tokenizer_id == "other-tokenizer"


def test_no_teacher_recovery_keeps_binary_operator_active_and_stops_on_no_progress() -> None:
    records = _initial_records()
    adapter, model = _model()
    router = TeacherRouter(
        TeacherRoutingConfig(
            "student-tokenizer",
            "unused-sealed-bf16",
        )
    )

    result = _trainer(steps=3).run(
        run_id="no-teacher-control",
        model=model,
        adapter=adapter,
        records=records,
        batches=_batches(records),
        corpus_plan=_plan(),
        selected_mix_id=_plan().initial_mix.mix_id,
        teacher_router=router,
        calibration_evaluator=_Metric((0.2, 0.3)),
        frozen_capability_evaluator=_Metric((0.4, 0.4)),
    )

    assert result.status is RecoveryRunStatus.STOPPED_NO_PROGRESS
    assert result.completed_steps == 1
    assert result.no_progress_decision is not None
    assert result.no_progress_decision.triggered
    assert result.teacher_provenance == ()
    assert result.steps[0].supervision_mode is SupervisionMode.NONE
    assert result.steps[0].coverage.eligible_positions == 0
    assert result.forward_events
    assert all(
        event.representations
        and set(event.representations) == {ActiveRepresentation.TRAINING}
        for event in result.forward_events
    )
    assert adapter.active_representation(model) is ActiveRepresentation.TRAINING


def test_shared_tokenizer_logit_and_sequence_recovery_report_coverage_and_provenance(
    tmp_path: Path,
) -> None:
    records = _initial_records()
    bf16, binding, _, _, _ = _verified_behavioral_teacher(tmp_path)
    broad = _identity(TeacherRole.BROAD)
    router = TeacherRouter(
        TeacherRoutingConfig(
            "student-tokenizer",
            bf16.identity_id,
            sealed_bf16_inventory_id=binding.inventory_id,
            behavioral_mode=SupervisionMode.SHARED_TOKENIZER_SEQUENCE,
            broad_mode=SupervisionMode.SHARED_TOKENIZER_LOGITS,
        ),
        behavioral_teacher=_Teacher(bf16),
        broad_teacher=_Teacher(broad),
        behavioral_binding=binding,
    )
    adapter, model = _model()

    result = _trainer(steps=4).run(
        run_id="shared-tokenizer-recovery",
        model=model,
        adapter=adapter,
        records=records,
        batches=_batches(records),
        corpus_plan=_plan(),
        selected_mix_id=_plan().initial_mix.mix_id,
        teacher_router=router,
        calibration_evaluator=_Metric((0.2, 0.2, 0.2, 0.2, 0.2)),
        frozen_capability_evaluator=_Metric((0.4, 0.5, 0.6, 0.7, 0.8)),
    )

    assert result.status is RecoveryRunStatus.COMPLETED
    assert result.completed_steps == 4
    assert tuple(step.supervision_mode for step in result.steps) == (
        SupervisionMode.SHARED_TOKENIZER_LOGITS,
        SupervisionMode.SHARED_TOKENIZER_LOGITS,
        SupervisionMode.SHARED_TOKENIZER_LOGITS,
        SupervisionMode.SHARED_TOKENIZER_SEQUENCE,
    )
    assert all(step.teacher_loss is not None for step in result.steps)
    assert all(step.coverage.fraction == 1.0 for step in result.steps)
    assert result.steps[0].coverage.probability_mass == pytest.approx(3 / 13)
    assert result.steps[-1].teacher_identity_id == bf16.identity_id
    assert {item.teacher_identity_id for item in result.teacher_provenance} == {
        broad.identity_id,
        bf16.identity_id,
    }


def test_recovery_uses_shared_pre_step_budget_boundary() -> None:
    records = _initial_records()
    adapter, model = _model()
    result = _trainer(steps=2).run(
        run_id="recovery-budget",
        model=model,
        adapter=adapter,
        records=records,
        batches=_batches(records),
        corpus_plan=_plan(),
        selected_mix_id=_plan().initial_mix.mix_id,
        teacher_router=TeacherRouter(
            TeacherRoutingConfig("student-tokenizer", "unused-sealed-bf16")
        ),
        calibration_evaluator=_Metric((0.2,)),
        frozen_capability_evaluator=_Metric((0.4,)),
        budget=Budget(2, 10, 100, 1.0, CheckpointBoundary.OPTIMIZER_STEP),
        monotonic_clock=lambda: 0.0,
    )

    assert result.status is RecoveryRunStatus.STOPPED_BUDGET
    assert result.completed_steps == 0
    assert result.budget_crossing is not None
    assert result.budget_crossing.dimension.value == "tokens"
    assert result.budget_failure is not None
    assert result.budget_failure.context["phase"] == "pre_step"
