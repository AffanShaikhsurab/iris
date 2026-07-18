"""Durable whole-tiny-model vertical slice across the production contracts."""

from __future__ import annotations

import io
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import Tensor, nn

from binary_llm.adapters import ActiveRepresentation, BinaryLinear, ModelAdapter
from binary_llm.domain import (
    ArtifactRef,
    Budget,
    ExperimentManifest,
    FormatSpec,
    Retryability,
    ToleranceSet,
    canonical_json_bytes,
    parse_canonical_json,
    sha256_bytes,
)
from binary_llm.domain.errors import ArtifactStoreError
from binary_llm.export import (
    IdentifiedCheckpoint,
    PackedExporter,
    PackedTensorSource,
    ScalarPackedRuntime,
    bind_packed_linears,
)
from binary_llm.math import analytical_row_scales

from .evidence_store import FrozenCorpusBundle
from .budgets import BudgetUsage
from .parity import ParityCoordinator, capture_parity_trace
from .progressive_state import ProgressiveBoundaryCheckpoint
from .progressive_trainer import (
    CapabilityEvaluator,
    FinalGateEvaluator,
    ProgressiveFaultInjector,
    ProgressiveRunResult,
    ProgressiveRunStatus,
    ProgressiveTrainerBackend,
)
from .registry import (
    AppendOnlyExperimentRegistry,
    AttemptExecutionContext,
    AttemptRef,
    AttemptStatus,
    ResourceDelta,
)
from .sealing import SealedInventory, verify_sealed_inventory
from .stage1 import CausalBatch, Stage1RunStatus, Stage1TrainerBackend
from .store import FilesystemArtifactStore
from .transition import apply_stage1_transition

_REPORT_MEDIA = "application/vnd.binary-llm.tiny-vertical-slice.v1+json"
_JSON_MEDIA = "application/vnd.binary-llm.evidence.v1+json"
_TORCH_MEDIA = "application/vnd.binary-llm.torch-checkpoint.v1"


def _plain(value: Any) -> Any:
    if isinstance(value, Tensor):
        tensor = value.detach().cpu().contiguous()
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        return {
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": list(tensor.shape),
            "sha256": sha256_bytes(raw),
            "bytes": len(raw),
        }
    if isinstance(value, ArtifactRef):
        return value.to_dict()
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _plain(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _torch_bytes(value: Any) -> bytes:
    stream = io.BytesIO()
    torch.save(value, stream)
    return stream.getvalue()


@dataclass(frozen=True, slots=True)
class TinyPipelineConfig:
    run_id: str
    first_attempt_id: str
    resume_attempt_id: str
    stage1_steps: int
    exporter_revision: str
    runtime_revision: str
    build_flags: tuple[str, ...]
    scale_tolerance: float
    training_operator_revision: str
    resource_currency: str = "USD"


@dataclass(frozen=True, slots=True)
class TinyPipelineInputs:
    manifest: ExperimentManifest
    execution_context: AttemptExecutionContext
    corpus_bundle: FrozenCorpusBundle
    sealed_inventory: SealedInventory
    sealed_root: Path
    model_factory: Callable[[], tuple[ModelAdapter[nn.Module], nn.Module]]
    stage1_trainer: Stage1TrainerBackend
    progressive_trainer: ProgressiveTrainerBackend
    progressive_plan: Any
    stage1_train_batches: Sequence[CausalBatch]
    progressive_phase_batches: Mapping[str, Sequence[CausalBatch]]
    held_out_batches: Sequence[CausalBatch]
    capability_evaluator: CapabilityEvaluator
    final_gate_evaluator: FinalGateEvaluator
    format_spec: FormatSpec
    tolerances: ToleranceSet
    parity_input_ids: Tensor
    selected_layer_names: tuple[str, ...]
    greedy_steps: int
    tool_decision: Callable[[Tensor], Sequence[str]]
    stage1_budget: Budget | None = None
    progressive_budget: Budget | None = None


@dataclass(frozen=True, slots=True)
class TinyVerticalSliceReport:
    report_ref: ArtifactRef
    artifact_refs: tuple[ArtifactRef, ...]
    run_id: str
    attempt_ids: tuple[str, ...]
    final_checkpoint_id: str
    packed_artifact_id: str
    scientific_reproduction: bool = False
    promotion_eligible: bool = False


@dataclass(frozen=True, slots=True)
class TinyPipelineResult:
    report: TinyVerticalSliceReport | None
    first_attempt: AttemptRef
    resumed_attempt: AttemptRef | None
    artifacts: tuple[ArtifactRef, ...]
    interrupted: ProgressiveRunResult | None
    completed: ProgressiveRunResult | None


class TinyPipelineReportResolver:
    """Resolve a report and every reference, failing closed on the exact bad ID."""

    def __init__(self, store: FilesystemArtifactStore) -> None:
        self.store = store

    def resolve(self, report_ref: ArtifactRef) -> Mapping[str, Any]:
        payload = self.store.resolve(report_ref.artifact_id, expected=report_ref)
        report = parse_canonical_json(payload)
        if not isinstance(report, dict) or report.get("artifact_kind") != "tiny_vertical_slice_report":
            raise ArtifactStoreError(
                "tiny vertical-slice report schema is invalid",
                retryability=Retryability.NEVER,
                code="artifact_store.invalid_tiny_report",
                affected_ids={"artifact_ids": (report_ref.artifact_id,)},
            )
        raw_refs = report.get("artifact_refs")
        if not isinstance(raw_refs, list):
            raise ArtifactStoreError(
                "tiny vertical-slice report references are invalid",
                retryability=Retryability.NEVER,
                code="artifact_store.invalid_tiny_report",
                affected_ids={"artifact_ids": (report_ref.artifact_id,)},
            )
        for raw in raw_refs:
            reference = ArtifactRef(**raw)
            self.store.resolve(reference.artifact_id, expected=reference)
        return report


class TinyPipelineOrchestrator:
    """Run a deterministic architecture proof, never a promotion or science claim."""

    def __init__(
        self,
        *,
        config: TinyPipelineConfig,
        store: FilesystemArtifactStore,
        registry: AppendOnlyExperimentRegistry,
        output_directory: str | Path,
    ) -> None:
        self.config = config
        self.store = store
        self.registry = registry
        self.output_directory = Path(output_directory)

    def _persist(
        self,
        attempt: AttemptRef,
        *,
        kind: str,
        value: Any,
        parent: str,
        media_type: str = _JSON_MEDIA,
        payload: bytes | None = None,
        artifact_id: str | None = None,
    ) -> ArtifactRef:
        content = payload if payload is not None else canonical_json_bytes(_plain(value))
        digest = sha256_bytes(content)
        reference = ArtifactRef(
            artifact_id=artifact_id or f"{kind}:{digest}",
            kind=kind,
            sha256=digest,
            bytes=len(content),
            media_type=media_type,
            parent_artifact_id=parent,
            producing_run_id=attempt.run_id,
            producing_attempt_id=attempt.attempt_id,
        )
        self.registry.attach_artifact(attempt, reference, payload=content)
        return reference

    def _checkpoint(
        self,
        attempt: AttemptRef,
        checkpoint: Any,
        *,
        parent: str,
        phase: int,
    ) -> ArtifactRef:
        payload = _torch_bytes(checkpoint)
        reference = self._persist(
            attempt,
            kind="training_checkpoint",
            value=None,
            parent=parent,
            media_type=_TORCH_MEDIA,
            payload=payload,
            artifact_id=checkpoint.checkpoint_id,
        )
        state_hashes = getattr(checkpoint, "state_hashes", None)
        if state_hashes is None:
            digest = sha256_bytes(payload)
            state_hashes = {name: digest for name in (
                "training_state", "optimizer_state", "schedule_state", "rng_state", "data_cursor"
            )}
        self.registry.publish_checkpoint(
            attempt, reference, phase=phase, complete=True, state_hashes=state_hashes
        )
        return reference

    @staticmethod
    def _checkpoint_source(checkpoint: ProgressiveBoundaryCheckpoint, model: nn.Module) -> IdentifiedCheckpoint:
        sources = []
        for module in model.modules():
            if isinstance(module, BinaryLinear):
                latent = module.transformed_weight().detach()
                sources.append(
                    PackedTensorSource(
                        name=module.tensor_name,
                        semantic_role=module.semantic_role,
                        representation_id="binary-v1",
                        latent_weight=latent,
                        merged_scale=(
                            analytical_row_scales(
                                latent,
                                gradient=module.config.progressive_operator.analytical_scale_gradient,
                            )
                            * module.dual_scale.learned().detach()
                        ),
                    )
                )
        digest = sha256_bytes(canonical_json_bytes(dict(checkpoint.state_hashes)))
        return IdentifiedCheckpoint(checkpoint.checkpoint_id, digest, tuple(sources))

    @staticmethod
    def _excluded_state(model: nn.Module) -> dict[str, Tensor]:
        binary_ids = {
            id(parameter)
            for module in model.modules()
            if isinstance(module, BinaryLinear)
            for parameter in (module.weight, module.dual_scale.learned.value, module.input_scale.raw_scale)
        }
        return {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
            if not any(
                name == candidate
                for candidate, parameter in model.named_parameters()
                if id(parameter) in binary_ids
            )
            and not (
                name.endswith("dense_reference_weight")
                or name.endswith("input_scale.raw_scale")
                or name.endswith("dual_scale.learned.value")
            )
        }

    def _record_usage(self, attempt: AttemptRef, usage: Any, suffix: str) -> None:
        if usage is None:
            return
        self.registry.record_resources(
            attempt,
            ResourceDelta(
                f"{attempt.attempt_id}:{suffix}",
                attempt.attempt_id,
                usage.wall_seconds,
                usage.accelerator_seconds,
                usage.consumed_tokens,
                usage.optimizer_steps,
                usage.billable_cost,
                self.config.resource_currency,
                "1970-01-01T00:00:00Z",
            ),
        )

    @staticmethod
    def _usage_delta(total: BudgetUsage | None, prior: BudgetUsage | None) -> BudgetUsage | None:
        if total is None:
            return None
        if prior is None:
            return total
        if not total.dominates(prior):
            raise ValueError("resumed budget usage cannot precede retained usage")
        accelerator = (
            None
            if total.accelerator_seconds is None
            else total.accelerator_seconds - (prior.accelerator_seconds or 0.0)
        )
        values = {
            "consumed_tokens": total.consumed_tokens - prior.consumed_tokens,
            "optimizer_steps": total.optimizer_steps - prior.optimizer_steps,
            "wall_seconds": total.wall_seconds - prior.wall_seconds,
            "accelerator_seconds": accelerator,
            "billable_cost": total.billable_cost - prior.billable_cost,
            "interrupted_attempts": total.interrupted_attempts - prior.interrupted_attempts,
        }
        return BudgetUsage(**values)

    def run(
        self,
        inputs: TinyPipelineInputs,
        *,
        fault_injector: ProgressiveFaultInjector | None = None,
    ) -> TinyPipelineResult:
        experiment = self.registry.register(inputs.manifest)
        first = self.registry.begin_attempt(
            experiment,
            inputs.execution_context,
            run_id=self.config.run_id,
            attempt_id=self.config.first_attempt_id,
        )
        refs: list[ArtifactRef] = []
        root = inputs.manifest.parent_checkpoint.artifact_id
        seal = verify_sealed_inventory(inputs.sealed_inventory, inputs.sealed_root)
        seal.require_compute_allowed()
        seal_ref = self._persist(first, kind="seal_verification", value=seal, parent=root)
        refs.append(seal_ref)
        corpus_ref = self._persist(
            first,
            kind="frozen_corpus_bundle",
            value=None,
            parent=seal_ref.artifact_id,
            payload=inputs.corpus_bundle.payload,
            media_type=inputs.corpus_bundle.artifact_ref.media_type,
        )
        refs.append(corpus_ref)

        adapter, model = inputs.model_factory()
        stage1 = inputs.stage1_trainer.run(
            run_id=first.run_id,
            model=model,
            adapter=adapter,
            train_batches=inputs.stage1_train_batches,
            held_out_batches=inputs.held_out_batches,
            test_step_override=self.config.stage1_steps,
            budget=inputs.stage1_budget,
            parent_artifact_id=None,
        )
        stage_ref = self._checkpoint(first, stage1.checkpoint, parent=corpus_ref.artifact_id, phase=0)
        refs.append(stage_ref)
        stage_evidence_ref = self._persist(
            first, kind="stage1_evaluation", value=stage1, parent=stage_ref.artifact_id
        )
        refs.append(stage_evidence_ref)
        self._record_usage(first, stage1.budget_usage, "stage1")
        if stage1.status is not Stage1RunStatus.COMPLETED or stage1.transition is None:
            self.registry.finish_attempt(first, AttemptStatus.FAILED, failure_label=stage1.status.value)
            return TinyPipelineResult(None, first, None, tuple(refs), None, None)

        transition_ref = replace(
            stage1.transition.artifact_ref,
            parent_artifact_id=stage_ref.artifact_id,
            producing_run_id=first.run_id,
            producing_attempt_id=first.attempt_id,
        )
        transition = replace(stage1.transition, artifact_ref=transition_ref)
        self.registry.attach_artifact(first, transition_ref, payload=transition.payload)
        refs.append(transition_ref)
        applied = apply_stage1_transition(
            model,
            transition,
            expected_source_checkpoint_id=stage1.checkpoint.checkpoint_id,
        )

        interrupted = inputs.progressive_trainer.run(
            run_id=first.run_id,
            attempt_id=first.attempt_id,
            parent_checkpoint_id=stage1.checkpoint.checkpoint_id,
            model=model,
            adapter=adapter,
            plan=inputs.progressive_plan,
            phase_batches=inputs.progressive_phase_batches,
            held_out_batches=inputs.held_out_batches,
            capability_evaluator=inputs.capability_evaluator,
            final_gate_evaluator=inputs.final_gate_evaluator,
            applied_transition=applied,
            budget=inputs.progressive_budget,
            artifact_store=None,
            fault_injector=fault_injector,
        )
        parent = transition_ref.artifact_id
        for checkpoint in interrupted.checkpoints:
            checkpoint_ref = self._checkpoint(
                first, checkpoint, parent=parent, phase=checkpoint.completed_phase_count
            )
            refs.append(checkpoint_ref)
            parent = checkpoint_ref.artifact_id
        if interrupted.failure_artifact is not None:
            failure_ref = replace(
                interrupted.failure_artifact.artifact_ref,
                parent_artifact_id=parent,
                producing_run_id=first.run_id,
                producing_attempt_id=first.attempt_id,
            )
            self.registry.attach_artifact(
                first, failure_ref, payload=interrupted.failure_artifact.payload
            )
            refs.append(failure_ref)
        self._record_usage(first, interrupted.budget_usage, "progressive")

        if interrupted.status is ProgressiveRunStatus.STOPPED_BUDGET:
            budget_ref = self._persist(
                first,
                kind="budget_failure",
                value=interrupted.budget_failure,
                parent=parent,
            )
            refs.append(budget_ref)
            self.registry.finish_attempt(first, AttemptStatus.FAILED, failure_label="budget")
            return TinyPipelineResult(None, first, None, tuple(refs), interrupted, None)
        if interrupted.status is not ProgressiveRunStatus.STOPPED_NONFINITE:
            raise ValueError("fault-injected run must stop nonfinite before resume")
        checkpoint = interrupted.last_complete_checkpoint
        if checkpoint is None:
            raise ValueError("fault injection must occur after a complete phase boundary")
        checkpoint_bytes = self.store.resolve(checkpoint.checkpoint_id)
        checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, ProgressiveBoundaryCheckpoint):
            raise ValueError("durable phase boundary did not decode as a checkpoint")
        self.registry.finish_attempt(first, AttemptStatus.INTERRUPTED, failure_label="nonfinite")

        resumed_attempt = self.registry.begin_attempt(
            experiment,
            inputs.execution_context,
            resume_checkpoint_id=checkpoint.checkpoint_id,
            attempt_id=self.config.resume_attempt_id,
        )
        resumed_adapter, resumed_model = inputs.model_factory()
        remaining = {
            phase.partition_hash: inputs.progressive_phase_batches[phase.partition_hash]
            for phase in inputs.progressive_plan.phases[checkpoint.completed_phase_count:]
        }
        completed = inputs.progressive_trainer.run(
            run_id=resumed_attempt.run_id,
            attempt_id=resumed_attempt.attempt_id,
            parent_checkpoint_id=stage1.checkpoint.checkpoint_id,
            model=resumed_model,
            adapter=resumed_adapter,
            plan=inputs.progressive_plan,
            phase_batches=remaining,
            held_out_batches=inputs.held_out_batches,
            capability_evaluator=inputs.capability_evaluator,
            final_gate_evaluator=inputs.final_gate_evaluator,
            resume_checkpoint=checkpoint,
            budget=inputs.progressive_budget,
            prior_budget_usage=interrupted.budget_usage,
        )
        if completed.status is not ProgressiveRunStatus.COMPLETED or completed.final_evidence is None:
            self.registry.finish_attempt(
                resumed_attempt, AttemptStatus.FAILED, failure_label=completed.status.value
            )
            return TinyPipelineResult(None, first, resumed_attempt, tuple(refs), interrupted, completed)
        final_checkpoint = completed.checkpoints[-1]
        final_ref = self._checkpoint(
            resumed_attempt,
            final_checkpoint,
            parent=checkpoint.checkpoint_id,
            phase=final_checkpoint.completed_phase_count,
        )
        refs.append(final_ref)
        evaluation_ref = self._persist(
            resumed_attempt,
            kind="raw_progressive_sign_evaluations",
            value={
                "source_checkpoint_id": final_checkpoint.checkpoint_id,
                "progressive": completed.final_evidence.progressive,
                "sign": completed.final_evidence.sign_substituted,
            },
            parent=final_ref.artifact_id,
        )
        refs.append(evaluation_ref)
        gate_ref = self._persist(
            resumed_attempt,
            kind="gate_records",
            value={
                "source_evaluation_id": evaluation_ref.artifact_id,
                "progressive": completed.final_evidence.progressive_gate_report,
                "sign": completed.final_evidence.sign_gate_report,
            },
            parent=evaluation_ref.artifact_id,
        )
        refs.append(gate_ref)

        resumed_adapter.set_active_representation(resumed_model, ActiveRepresentation.SIGN)
        source = self._checkpoint_source(final_checkpoint, resumed_model)
        exporter = PackedExporter(
            exporter_revision=self.config.exporter_revision,
            runtime_revision=self.config.runtime_revision,
            build_flags=self.config.build_flags,
        )
        plan = exporter.plan(
            source, inputs.format_spec, scale_tolerance=self.config.scale_tolerance
        )
        self.output_directory.mkdir(parents=True, exist_ok=True)
        packed = exporter.export(plan, self.output_directory / "model.bllmp")
        packed_ref = self._persist(
            resumed_attempt,
            kind="packed_model",
            value=None,
            parent=gate_ref.artifact_id,
            payload=packed.path.read_bytes(),
            media_type="application/vnd.binary-llm.packed.v1",
            artifact_id=packed.artifact_id,
        )
        refs.append(packed_ref)
        sidecar_state = self._excluded_state(resumed_model)
        sidecar_payload = _torch_bytes(sidecar_state)
        sidecar_path = self.output_directory / "excluded-state.pt"
        sidecar_path.write_bytes(sidecar_payload)
        sidecar_ref = self._persist(
            resumed_attempt,
            kind="required_sidecar",
            value=None,
            parent=packed_ref.artifact_id,
            payload=sidecar_payload,
            media_type=_TORCH_MEDIA,
        )
        refs.append(sidecar_ref)
        descriptors = resumed_adapter.enumerate_tensors(resumed_model)
        accounting_ref = self._persist(
            resumed_attempt,
            kind="whole_tiny_model_accounting",
            value={
                "scope": "whole-tiny-model",
                "source_checkpoint_id": final_checkpoint.checkpoint_id,
                "aliases": [
                    {"name": item.name, "tied_to": item.tied_to}
                    for item in descriptors if item.tied_to is not None
                ],
                "tensors": descriptors,
                "packed_ledger": plan.ledger,
                "files": [
                    {"path": "model.bllmp", "bytes": len(plan.container_bytes), "sha256": packed.sha256},
                    {"path": "excluded-state.pt", "bytes": len(sidecar_payload), "sha256": sha256_bytes(sidecar_payload)},
                ],
                "unexplained_bytes": 0,
            },
            parent=sidecar_ref.artifact_id,
        )
        refs.append(accounting_ref)

        binary_names = tuple(
            name for name, module in resumed_model.named_modules() if isinstance(module, BinaryLinear)
        )
        training_trace = capture_parity_trace(
            resumed_model,
            inputs.parity_input_ids,
            binary_linear_names=binary_names,
            selected_layer_names=inputs.selected_layer_names,
            greedy_steps=inputs.greedy_steps,
            tool_decision=inputs.tool_decision,
        )
        _, packed_model = inputs.model_factory()
        packed_model.load_state_dict(final_checkpoint.training_state)
        packed_model.load_state_dict(sidecar_state, strict=False)
        resumed_adapter.set_active_representation(packed_model, ActiveRepresentation.SIGN)
        runtime_artifact = replace(packed, path=self.output_directory / "model.bllmp")
        runtime = ScalarPackedRuntime(
            runtime_revision=self.config.runtime_revision, build_flags=self.config.build_flags
        )
        with runtime.load(runtime_artifact) as session:
            bind_packed_linears(packed_model, session)
            packed_trace = capture_parity_trace(
                packed_model,
                inputs.parity_input_ids,
                binary_linear_names=binary_names,
                selected_layer_names=inputs.selected_layer_names,
                greedy_steps=inputs.greedy_steps,
                tool_decision=inputs.tool_decision,
            )
            parity = ParityCoordinator().coordinate(
                artifact=runtime_artifact,
                training_operator_revision=self.config.training_operator_revision,
                tolerances=inputs.tolerances,
                training_trace=training_trace,
                packed_trace=packed_trace,
                memory=session.memory_report(),
                decoded_signs_exact=True,
                scale_error_max=0.0,
                traces_ref=(evaluation_ref.artifact_id, packed_ref.artifact_id),
            )
        parity_ref = self._persist(
            resumed_attempt,
            kind="runtime_parity",
            value=parity,
            parent=accounting_ref.artifact_id,
        )
        refs.append(parity_ref)
        self._record_usage(
            resumed_attempt,
            self._usage_delta(completed.budget_usage, interrupted.budget_usage),
            "progressive",
        )
        ledger = self.registry.cumulative_resource_ledger(resumed_attempt)
        ledger_ref = self._persist(
            resumed_attempt,
            kind="budget_ledger",
            value=ledger,
            parent=parity_ref.artifact_id,
        )
        refs.append(ledger_ref)
        report_payload = canonical_json_bytes(
            {
                "schema_version": 1,
                "artifact_kind": "tiny_vertical_slice_report",
                "scope": "whole-tiny-model",
                "fixture_evidence_only": True,
                "scientific_reproduction": False,
                "promotion_eligible": False,
                "run_id": resumed_attempt.run_id,
                "attempt_ids": [first.attempt_id, resumed_attempt.attempt_id],
                "statuses": ["interrupted_nonfinite", "completed"],
                "final_checkpoint_id": final_checkpoint.checkpoint_id,
                "packed_artifact_id": packed_ref.artifact_id,
                "protocol_ids": {
                    "gate_set": inputs.manifest.gate_set_ref,
                    "tolerance_set": inputs.tolerances.tolerance_set_id,
                },
                "budget_reconciliation": _plain(ledger),
                "accounting": {
                    "accounting_artifact_id": accounting_ref.artifact_id,
                    "unexplained_bytes": 0,
                },
                "artifact_refs": [item.to_dict() for item in refs],
            }
        )
        report_ref = self._persist(
            resumed_attempt,
            kind="tiny_vertical_slice_report",
            value=None,
            parent=ledger_ref.artifact_id,
            payload=report_payload,
            media_type=_REPORT_MEDIA,
        )
        self.registry.finish_attempt(resumed_attempt, AttemptStatus.COMPLETED)
        report = TinyVerticalSliceReport(
            report_ref,
            tuple(refs),
            resumed_attempt.run_id,
            (first.attempt_id, resumed_attempt.attempt_id),
            final_checkpoint.checkpoint_id,
            packed_ref.artifact_id,
        )
        return TinyPipelineResult(report, first, resumed_attempt, tuple((*refs, report_ref)), interrupted, completed)


__all__ = [
    "TinyPipelineConfig",
    "TinyPipelineInputs",
    "TinyPipelineOrchestrator",
    "TinyPipelineReportResolver",
    "TinyPipelineResult",
    "TinyVerticalSliceReport",
]
