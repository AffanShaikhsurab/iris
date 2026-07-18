"""Append-only experiment registry, resumable attempts, and reproduction checks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from threading import RLock
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Mapping

from binary_llm.domain.errors import ManifestError, ReproductionError, Retryability
from binary_llm.domain.identity import identify_content, new_unique_id
from binary_llm.domain.models import (
    ArtifactRef,
    CanonicalModel,
    ExperimentManifest,
    JsonValue,
    SUPPORTED_SCHEMA_VERSION,
    ToleranceDefinition,
)
from binary_llm.orchestration.journal import FilesystemRegistryJournal, model_from_dict

if TYPE_CHECKING:
    from binary_llm.orchestration.store import FilesystemArtifactStore


COMPLETE_CHECKPOINT_STATE_KEYS = frozenset(
    {"training_state", "optimizer_state", "schedule_state", "rng_state", "data_cursor"}
)


def _text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _schema(version: int) -> None:
    if version != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {version}")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (str, int, float, bool, type(None), StrEnum)):
        return value
    raise TypeError(f"value is not JSON-compatible: {type(value).__name__}")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


class AttemptStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


TERMINAL_ATTEMPT_STATUSES = frozenset(
    {AttemptStatus.COMPLETED, AttemptStatus.FAILED, AttemptStatus.INTERRUPTED}
)


@dataclass(frozen=True, slots=True)
class ExperimentRef(CanonicalModel):
    experiment_id: str
    manifest_hash: str
    schema_version: int = SUPPORTED_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        _text("experiment_id", self.experiment_id)
        _text("manifest_hash", self.manifest_hash)


@dataclass(frozen=True, slots=True)
class AttemptRef(CanonicalModel):
    run_id: str
    attempt_id: str
    experiment_id: str
    schema_version: int = SUPPORTED_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        for name in ("run_id", "attempt_id", "experiment_id"):
            _text(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class RegistryRecordRef(CanonicalModel):
    kind: str
    record_id: str
    content_hash: str
    schema_version: int = SUPPORTED_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        for name in ("kind", "record_id", "content_hash"):
            _text(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class AttemptExecutionContext(CanonicalModel):
    source_commit: str
    clean_tree: bool
    hardware_inventory: Mapping[str, JsonValue]
    compiler_inventory: Mapping[str, JsonValue]
    seed: int
    data_order_hash: str
    nondeterministic_operations: tuple[str, ...] = ()
    preregistered_tolerance_ids: tuple[str, ...] = ()
    schema_version: int = SUPPORTED_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        _text("source_commit", self.source_commit)
        _text("data_order_hash", self.data_order_hash)
        if len(self.nondeterministic_operations) != len(set(self.nondeterministic_operations)):
            raise ValueError("nondeterministic operation names must be unique")
        if len(self.preregistered_tolerance_ids) != len(set(self.preregistered_tolerance_ids)):
            raise ValueError("preregistered tolerance IDs must be unique")
        if self.nondeterministic_operations and not self.preregistered_tolerance_ids:
            raise ValueError("nondeterministic operations require preregistered tolerances")
        for value in self.nondeterministic_operations + self.preregistered_tolerance_ids:
            _text("operation or tolerance ID", value)
        object.__setattr__(self, "hardware_inventory", _freeze(self.hardware_inventory))
        object.__setattr__(self, "compiler_inventory", _freeze(self.compiler_inventory))


@dataclass(frozen=True, slots=True)
class RunAttempt(CanonicalModel):
    run_id: str
    attempt_id: str
    experiment_id: str
    status: AttemptStatus
    started_at: str
    source_commit: str
    clean_tree: bool
    dependency_lock_hash: str
    container_digest: str
    hardware_inventory: Mapping[str, JsonValue]
    compiler_inventory: Mapping[str, JsonValue]
    resolved_manifest_hash: str
    command: tuple[str, ...]
    environment: Mapping[str, JsonValue]
    seed: int
    data_order_hash: str
    current_phase: int
    resume_checkpoint_id: str | None
    nondeterministic_operations: tuple[str, ...]
    preregistered_tolerance_ids: tuple[str, ...]
    schema_version: int = SUPPORTED_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        if self.status is not AttemptStatus.RUNNING:
            raise ValueError("new attempt records must have running status")
        for name in (
            "run_id", "attempt_id", "experiment_id", "started_at", "source_commit",
            "dependency_lock_hash", "container_digest", "resolved_manifest_hash",
            "data_order_hash",
        ):
            _text(name, getattr(self, name))
        if self.current_phase < 0:
            raise ValueError("current_phase must be non-negative")
        object.__setattr__(self, "hardware_inventory", _freeze(self.hardware_inventory))
        object.__setattr__(self, "compiler_inventory", _freeze(self.compiler_inventory))
        object.__setattr__(self, "environment", _freeze(self.environment))


@dataclass(frozen=True, slots=True)
class AttemptStatusRecord(CanonicalModel):
    attempt_id: str
    status: AttemptStatus
    ended_at: str
    failure_label: str | None = None
    schema_version: int = SUPPORTED_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        _text("attempt_id", self.attempt_id)
        _text("ended_at", self.ended_at)
        if self.status not in TERMINAL_ATTEMPT_STATUSES:
            raise ValueError("attempt status record must be terminal")
        if self.status in {AttemptStatus.FAILED, AttemptStatus.INTERRUPTED}:
            _text("failure_label", self.failure_label or "")
        elif self.failure_label is not None:
            raise ValueError("completed attempts cannot have a failure label")


@dataclass(frozen=True, slots=True)
class RunEvent(CanonicalModel):
    event_id: str
    event_type: str
    payload: Mapping[str, JsonValue]
    occurred_at: str
    schema_version: int = SUPPORTED_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        for name in ("event_id", "event_type", "occurred_at"):
            _text(name, getattr(self, name))
        object.__setattr__(self, "payload", _freeze(self.payload))


@dataclass(frozen=True, slots=True)
class ArtifactAttachment(CanonicalModel):
    attempt_id: str
    artifact: ArtifactRef
    attached_at: str
    schema_version: int = SUPPORTED_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        _text("attempt_id", self.attempt_id)
        _text("attached_at", self.attached_at)


@dataclass(frozen=True, slots=True)
class CheckpointPublication(CanonicalModel):
    checkpoint_id: str
    attempt_id: str
    phase: int
    complete: bool
    state_hashes: Mapping[str, str]
    published_at: str
    schema_version: int = SUPPORTED_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        for name in ("checkpoint_id", "attempt_id", "published_at"):
            _text(name, getattr(self, name))
        if self.phase < 0:
            raise ValueError("checkpoint phase must be non-negative")
        frozen_hashes = _freeze(self.state_hashes)
        if self.complete:
            missing = COMPLETE_CHECKPOINT_STATE_KEYS - set(frozen_hashes)
            if missing:
                raise ValueError(f"complete checkpoint is missing state hashes: {sorted(missing)}")
        for name, digest in frozen_hashes.items():
            _text(f"state hash {name}", digest)
        object.__setattr__(self, "state_hashes", frozen_hashes)


@dataclass(frozen=True, slots=True)
class ResourceDelta(CanonicalModel):
    ledger_entry_id: str
    attempt_id: str
    wall_seconds: float
    accelerator_seconds: float
    consumed_tokens: int
    optimizer_steps: int
    billable_cost: float
    cost_currency: str
    recorded_at: str
    schema_version: int = SUPPORTED_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        for name in ("ledger_entry_id", "attempt_id", "cost_currency", "recorded_at"):
            _text(name, getattr(self, name))
        numeric = (self.wall_seconds, self.accelerator_seconds, self.billable_cost)
        if any(not math.isfinite(value) or value < 0 for value in numeric):
            raise ValueError("resource values must be finite and non-negative")
        if self.consumed_tokens < 0 or self.optimizer_steps < 0:
            raise ValueError("resource counts must be non-negative")


@dataclass(frozen=True, slots=True)
class ResourceLedger(CanonicalModel):
    run_id: str
    wall_seconds: float
    accelerator_seconds: float
    consumed_tokens: int
    optimizer_steps: int
    interrupted_attempts: int
    billable_cost: float
    cost_currency: str
    attempt_ids: tuple[str, ...]
    schema_version: int = SUPPORTED_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        _text("run_id", self.run_id)
        _text("cost_currency", self.cost_currency)


@dataclass(frozen=True, slots=True)
class SupersessionRecord(CanonicalModel):
    supersession_id: str
    old: RegistryRecordRef
    new: RegistryRecordRef
    reason: str
    created_at: str
    schema_version: int = SUPPORTED_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        _text("supersession_id", self.supersession_id)
        _text("reason", self.reason)
        _text("created_at", self.created_at)
        if self.old.kind != self.new.kind:
            raise ValueError("supersession records must have the same kind")
        if self.old.record_id == self.new.record_id:
            raise ValueError("a record cannot supersede itself")


@dataclass(frozen=True, slots=True)
class ReproductionProtocol(CanonicalModel):
    protocol_id: str
    original_attempt_id: str
    expected_input_hashes: Mapping[str, str]
    expected_artifact_hashes: Mapping[str, str]
    expected_metrics: Mapping[str, float]
    tolerances: tuple[ToleranceDefinition, ...]
    preregistered_at: str
    schema_version: int = SUPPORTED_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        for name in ("protocol_id", "original_attempt_id", "preregistered_at"):
            _text(name, getattr(self, name))
        if not self.expected_input_hashes:
            raise ValueError("reproduction requires preserved input hashes")
        if not self.expected_artifact_hashes and not self.expected_metrics:
            raise ValueError("reproduction requires an artifact or metric expectation")
        tolerance_paths = tuple(item.metric_path for item in self.tolerances)
        if len(tolerance_paths) != len(set(tolerance_paths)):
            raise ValueError("reproduction tolerance metric paths must be unique")
        if set(tolerance_paths) != set(self.expected_metrics):
            raise ValueError("every expected reproduction metric requires one tolerance")
        for mapping in (self.expected_input_hashes, self.expected_artifact_hashes):
            for name, digest in mapping.items():
                _text("hash name", name)
                _text("hash", digest)
        if any(not math.isfinite(value) for value in self.expected_metrics.values()):
            raise ValueError("expected metrics must be finite")
        object.__setattr__(self, "expected_input_hashes", _freeze(self.expected_input_hashes))
        object.__setattr__(self, "expected_artifact_hashes", _freeze(self.expected_artifact_hashes))
        object.__setattr__(self, "expected_metrics", _freeze(self.expected_metrics))


@dataclass(frozen=True, slots=True)
class ReproductionOutcome(CanonicalModel):
    artifact_hashes: Mapping[str, str]
    metrics: Mapping[str, float]
    schema_version: int = SUPPORTED_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        if any(not math.isfinite(value) for value in self.metrics.values()):
            raise ValueError("reproduced metrics must be finite")
        object.__setattr__(self, "artifact_hashes", _freeze(self.artifact_hashes))
        object.__setattr__(self, "metrics", _freeze(self.metrics))


@dataclass(frozen=True, slots=True)
class ReproductionValidation(CanonicalModel):
    validation_id: str
    protocol_id: str
    original_attempt_id: str
    executed: bool
    passed: bool
    mismatches: tuple[Mapping[str, JsonValue], ...]
    validated_at: str
    schema_version: int = SUPPORTED_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        for name in ("validation_id", "protocol_id", "original_attempt_id", "validated_at"):
            _text(name, getattr(self, name))
        object.__setattr__(self, "mismatches", tuple(_freeze(item) for item in self.mismatches))


class AppendOnlyExperimentRegistry:
    """Thread-safe registry optionally backed by a durable append-only journal."""

    def __init__(
        self,
        *,
        id_allocator: Callable[[str], str] = new_unique_id,
        clock: Callable[[], str] = _now,
        journal: FilesystemRegistryJournal | None = None,
        artifact_store: FilesystemArtifactStore | None = None,
    ) -> None:
        self._id_allocator = id_allocator
        self._clock = clock
        self._journal = journal
        self._artifact_store = artifact_store
        self._lock = RLock()
        self._records: dict[tuple[str, str], CanonicalModel] = {}
        self._refs: dict[tuple[str, str], RegistryRecordRef] = {}
        self._sequence: dict[tuple[str, str], int] = {}
        self._next_sequence = 0
        self._manifests: dict[str, ExperimentManifest] = {}
        self._experiment_refs: dict[str, ExperimentRef] = {}
        self._attempts: dict[str, RunAttempt] = {}
        self._terminal_statuses: dict[str, AttemptStatusRecord] = {}
        self._run_experiment: dict[str, str] = {}
        self._run_attempt_ids: dict[str, list[str]] = {}
        self._events_by_attempt: dict[str, list[str]] = {}
        self._artifacts: dict[str, ArtifactRef] = {}
        self._artifact_experiments: dict[str, set[str]] = {}
        self._attachments: dict[tuple[str, str], ArtifactAttachment] = {}
        self._checkpoints: dict[str, CheckpointPublication] = {}
        self._resource_entries: dict[str, ResourceDelta] = {}
        self._resource_ids_by_run: dict[str, list[str]] = {}
        self._supersessions: dict[str, SupersessionRecord] = {}
        self._superseded_by: dict[tuple[str, str], str] = {}
        self._reproductions: dict[str, ReproductionValidation] = {}
        self._event_owners: dict[str, str] = {}
        if self._journal is not None:
            self._replay()

    def _conflict(
        self, message: str, *, code: str, affected_ids: Mapping[str, tuple[str, ...]]
    ) -> ManifestError:
        return ManifestError(
            message,
            retryability=Retryability.NEVER,
            code=code,
            affected_ids=affected_ids,
        )

    def _append_record(
        self,
        kind: str,
        record_id: str,
        record: CanonicalModel,
        *,
        context: Mapping[str, JsonValue] | None = None,
    ) -> tuple[RegistryRecordRef, bool]:
        identity = identify_content(record.to_dict(), kind=f"registry-{kind}")
        key = (kind, record_id)
        existing = self._refs.get(key)
        if existing is not None:
            if existing.content_hash == identity.sha256:
                return existing, False
            raise self._conflict(
                "append-only record identifier already has different content",
                code="manifest.registry_record_conflict",
                affected_ids={f"{kind.replace('-', '_')}_ids": (record_id,)},
            )
        reference = RegistryRecordRef(kind, record_id, identity.sha256)
        if self._journal is not None:
            self._journal.append(kind, record_id, record.to_dict(), context or {})
        self._records[key] = record
        self._refs[key] = reference
        self._sequence[key] = self._next_sequence
        self._next_sequence += 1
        return reference, True

    def _replay(self) -> None:
        model_types: dict[str, type[CanonicalModel]] = {
            "manifest": ExperimentManifest,
            "attempt": RunAttempt,
            "event": RunEvent,
            "artifact": ArtifactRef,
            "attachment": ArtifactAttachment,
            "checkpoint": CheckpointPublication,
            "attempt-status": AttemptStatusRecord,
            "resource-delta": ResourceDelta,
            "supersession": SupersessionRecord,
            "reproduction-protocol": ReproductionProtocol,
            "reproduction-validation": ReproductionValidation,
        }
        replayed: list[tuple[str, str, CanonicalModel, Mapping[str, Any]]] = []

        def consume(
            kind: str,
            record_id: str,
            record_data: Mapping[str, Any],
            context: Mapping[str, Any],
        ) -> None:
            model_type = model_types.get(kind)
            if model_type is None:
                raise self._journal_failure(
                    "registry journal contains an unknown event schema",
                    "registry_journal.unknown_record_kind",
                    kind=kind,
                )
            try:
                record = model_from_dict(model_type, record_data)
            except (TypeError, ValueError) as error:
                raise self._journal_failure(
                    "registry journal record cannot be decoded",
                    "registry_journal.invalid_record",
                    kind=kind,
                    record_id=record_id,
                ) from error
            identity = identify_content(record.to_dict(), kind=f"registry-{kind}")
            key = (kind, record_id)
            existing = self._refs.get(key)
            if existing is not None:
                if existing.content_hash != identity.sha256:
                    raise self._journal_failure(
                        "duplicate registry identity has conflicting content",
                        "registry_journal.record_conflict",
                        kind=kind,
                        record_id=record_id,
                    )
                return
            reference = RegistryRecordRef(kind, record_id, identity.sha256)
            self._records[key] = record
            self._refs[key] = reference
            self._sequence[key] = self._next_sequence
            self._next_sequence += 1
            replayed.append((kind, record_id, record, context))

        self._journal.replay(consume)
        self._rebuild_indexes(replayed)

    @staticmethod
    def _journal_failure(message: str, code: str, **context: Any) -> Exception:
        from binary_llm.domain.errors import RegistryJournalError

        return RegistryJournalError(
            message,
            retryability=Retryability.NEVER,
            code=code,
            context=context,
        )

    def _rebuild_indexes(
        self,
        replayed: list[tuple[str, str, CanonicalModel, Mapping[str, Any]]],
    ) -> None:
        for kind, record_id, record, context in replayed:
            if kind == "manifest":
                assert isinstance(record, ExperimentManifest)
                identity = identify_content(record.to_dict(), kind="experiment-manifest")
                self._manifests[record.experiment_id] = record
                self._experiment_refs[record.experiment_id] = ExperimentRef(
                    record.experiment_id, identity.sha256
                )
                for artifact in (
                    record.parent_checkpoint,
                    *(item for baseline in record.baseline_refs for item in (
                        *baseline.artifact_refs, *baseline.baseline_output_refs
                    )),
                ):
                    self._artifact_experiments.setdefault(artifact.artifact_id, set()).add(
                        record.experiment_id
                    )
            elif kind == "attempt":
                assert isinstance(record, RunAttempt)
                if record.experiment_id not in self._manifests:
                    raise self._journal_failure(
                        "attempt references an unknown experiment",
                        "registry_journal.invalid_owner_reference",
                        attempt_id=record.attempt_id,
                    )
                self._attempts[record.attempt_id] = record
                self._run_experiment.setdefault(record.run_id, record.experiment_id)
                if self._run_experiment[record.run_id] != record.experiment_id:
                    raise self._journal_failure(
                        "run references conflicting experiments",
                        "registry_journal.invalid_owner_reference",
                        run_id=record.run_id,
                    )
                self._run_attempt_ids.setdefault(record.run_id, []).append(record.attempt_id)
                self._events_by_attempt[record.attempt_id] = []
                self._resource_ids_by_run.setdefault(record.run_id, [])
            elif kind == "event":
                owner = context.get("attempt_id")
                if not isinstance(owner, str) or owner not in self._attempts:
                    raise self._journal_failure(
                        "event references an unknown owning attempt",
                        "registry_journal.invalid_owner_reference",
                        event_id=record_id,
                    )
                self._events_by_attempt[owner].append(record_id)
                self._event_owners[record_id] = owner
            elif kind == "artifact":
                assert isinstance(record, ArtifactRef)
                if record.parent_artifact_id is not None and record.parent_artifact_id not in self._artifacts:
                    raise self._journal_failure(
                        "artifact references an unknown parent",
                        "registry_journal.invalid_parent_reference",
                        artifact_id=record.artifact_id,
                    )
                if record.producing_attempt_id is not None:
                    attempt = self._attempts.get(record.producing_attempt_id)
                    if attempt is None or attempt.run_id != record.producing_run_id:
                        raise self._journal_failure(
                            "artifact references an unknown producer",
                            "registry_journal.invalid_producer_reference",
                            artifact_id=record.artifact_id,
                        )
                self._artifacts[record.artifact_id] = record
            elif kind == "attachment":
                assert isinstance(record, ArtifactAttachment)
                attempt = self._attempts.get(record.attempt_id)
                artifact = self._artifacts.get(record.artifact.artifact_id)
                if attempt is None or artifact != record.artifact:
                    raise self._journal_failure(
                        "attachment references an unknown owner or artifact",
                        "registry_journal.invalid_owner_reference",
                        artifact_id=record.artifact.artifact_id,
                    )
                key = (record.attempt_id, record.artifact.artifact_id)
                self._attachments[key] = record
                self._artifact_experiments.setdefault(record.artifact.artifact_id, set()).add(
                    attempt.experiment_id
                )
            elif kind == "checkpoint":
                assert isinstance(record, CheckpointPublication)
                if (
                    record.attempt_id not in self._attempts
                    or (record.attempt_id, record.checkpoint_id) not in self._attachments
                ):
                    raise self._journal_failure(
                        "checkpoint references an unknown attempt or attachment",
                        "registry_journal.invalid_owner_reference",
                        checkpoint_id=record.checkpoint_id,
                    )
                self._checkpoints[record.checkpoint_id] = record
            elif kind == "attempt-status":
                assert isinstance(record, AttemptStatusRecord)
                if record.attempt_id not in self._attempts:
                    raise self._journal_failure(
                        "status references an unknown attempt",
                        "registry_journal.invalid_owner_reference",
                        attempt_id=record.attempt_id,
                    )
                self._terminal_statuses[record.attempt_id] = record
            elif kind == "resource-delta":
                assert isinstance(record, ResourceDelta)
                attempt = self._attempts.get(record.attempt_id)
                if attempt is None:
                    raise self._journal_failure(
                        "resource delta references an unknown attempt",
                        "registry_journal.invalid_owner_reference",
                        resource_id=record.ledger_entry_id,
                    )
                self._resource_entries[record.ledger_entry_id] = record
                self._resource_ids_by_run[attempt.run_id].append(record.ledger_entry_id)
            elif kind == "supersession":
                assert isinstance(record, SupersessionRecord)
                for reference in (record.old, record.new):
                    if self._refs.get((reference.kind, reference.record_id)) != reference:
                        raise self._journal_failure(
                            "supersession references an unknown record",
                            "registry_journal.invalid_parent_reference",
                            supersession_id=record.supersession_id,
                        )
                self._supersessions[record.supersession_id] = record
                self._superseded_by[(record.old.kind, record.old.record_id)] = record.supersession_id
            elif kind == "reproduction-protocol":
                assert isinstance(record, ReproductionProtocol)
                if record.original_attempt_id not in self._attempts:
                    raise self._journal_failure(
                        "reproduction protocol references an unknown attempt",
                        "registry_journal.invalid_owner_reference",
                        protocol_id=record.protocol_id,
                    )
            elif kind == "reproduction-validation":
                assert isinstance(record, ReproductionValidation)
                if (
                    record.original_attempt_id not in self._attempts
                    or ("reproduction-protocol", record.protocol_id) not in self._refs
                ):
                    raise self._journal_failure(
                        "reproduction validation references unknown records",
                        "registry_journal.invalid_owner_reference",
                        validation_id=record.validation_id,
                    )
                self._reproductions[record.validation_id] = record

    def _require_experiment(self, reference: ExperimentRef) -> ExperimentManifest:
        existing = self._experiment_refs.get(reference.experiment_id)
        if existing != reference:
            raise self._conflict(
                "experiment reference does not identify the registered manifest",
                code="manifest.registry_unknown_experiment",
                affected_ids={"experiment_ids": (reference.experiment_id,)},
            )
        return self._manifests[reference.experiment_id]

    def _require_attempt(self, reference: AttemptRef) -> RunAttempt:
        existing = self._attempts.get(reference.attempt_id)
        if existing is None or (
            existing.run_id != reference.run_id
            or existing.experiment_id != reference.experiment_id
        ):
            raise self._conflict(
                "attempt reference does not identify a registered attempt",
                code="manifest.registry_unknown_attempt",
                affected_ids={"attempt_ids": (reference.attempt_id,)},
            )
        return existing

    def _register_root_artifact(self, experiment_id: str, artifact: ArtifactRef) -> None:
        reference, created = self._append_record("artifact", artifact.artifact_id, artifact)
        if created:
            self._artifacts[artifact.artifact_id] = artifact
        else:
            del reference
        self._artifact_experiments.setdefault(artifact.artifact_id, set()).add(experiment_id)

    def register(self, manifest: ExperimentManifest) -> ExperimentRef:
        """Register an immutable manifest, idempotently by experiment ID and content."""

        with self._lock:
            identity = identify_content(manifest.to_dict(), kind="experiment-manifest")
            reference = ExperimentRef(manifest.experiment_id, identity.sha256)
            existing = self._experiment_refs.get(manifest.experiment_id)
            if existing is not None:
                if existing == reference:
                    return existing
                raise self._conflict(
                    "experiment ID is already bound to a different manifest",
                    code="manifest.registry_experiment_conflict",
                    affected_ids={"experiment_ids": (manifest.experiment_id,)},
                )
            self._append_record("manifest", manifest.experiment_id, manifest)
            self._manifests[manifest.experiment_id] = manifest
            self._experiment_refs[manifest.experiment_id] = reference
            self._register_root_artifact(manifest.experiment_id, manifest.parent_checkpoint)
            for baseline in manifest.baseline_refs:
                for artifact in baseline.artifact_refs + baseline.baseline_output_refs:
                    self._register_root_artifact(manifest.experiment_id, artifact)
            return reference

    def begin_attempt(
        self,
        experiment: ExperimentRef,
        context: AttemptExecutionContext,
        *,
        resume_checkpoint_id: str | None = None,
        run_id: str | None = None,
        attempt_id: str | None = None,
    ) -> AttemptRef:
        """Allocate a unique attempt, preserving the full execution environment."""

        with self._lock:
            manifest = self._require_experiment(experiment)
            if context.seed not in manifest.seed_set:
                raise self._conflict(
                    "attempt seed is not present in the registered manifest",
                    code="manifest.registry_unregistered_seed",
                    affected_ids={"experiment_ids": (manifest.experiment_id,)},
                )
            current_phase = 0
            if resume_checkpoint_id is not None:
                checkpoint = self._checkpoints.get(resume_checkpoint_id)
                if checkpoint is None or not checkpoint.complete:
                    raise self._conflict(
                        "attempts may resume only from a published complete checkpoint",
                        code="manifest.registry_incomplete_resume_checkpoint",
                        affected_ids={"checkpoint_ids": (resume_checkpoint_id,)},
                    )
                source_attempt = self._attempts[checkpoint.attempt_id]
                if source_attempt.experiment_id != manifest.experiment_id:
                    raise self._conflict(
                        "resume checkpoint belongs to another experiment",
                        code="manifest.registry_cross_experiment_resume",
                        affected_ids={"checkpoint_ids": (resume_checkpoint_id,)},
                    )
                source_status = self._terminal_statuses.get(source_attempt.attempt_id)
                if source_status is None or source_status.status not in {
                    AttemptStatus.FAILED, AttemptStatus.INTERRUPTED
                }:
                    raise self._conflict(
                        "resume requires a failed or interrupted source attempt",
                        code="manifest.registry_invalid_resume_source",
                        affected_ids={"attempt_ids": (source_attempt.attempt_id,)},
                    )
                resolved_run_id = source_attempt.run_id
                if run_id is not None and run_id != resolved_run_id:
                    raise self._conflict(
                        "resume must retain the checkpoint run ID",
                        code="manifest.registry_resume_run_mismatch",
                        affected_ids={"run_ids": (run_id, resolved_run_id)},
                    )
                current_phase = checkpoint.phase
            else:
                resolved_run_id = run_id or self._id_allocator("run")
                if resolved_run_id in self._run_experiment:
                    raise self._conflict(
                        "run ID has already been allocated",
                        code="manifest.registry_duplicate_run",
                        affected_ids={"run_ids": (resolved_run_id,)},
                    )
            resolved_attempt_id = attempt_id or self._id_allocator("attempt")
            if resolved_attempt_id in self._attempts:
                raise self._conflict(
                    "attempt ID has already been allocated",
                    code="manifest.registry_duplicate_attempt",
                    affected_ids={"attempt_ids": (resolved_attempt_id,)},
                )
            record = RunAttempt(
                run_id=resolved_run_id,
                attempt_id=resolved_attempt_id,
                experiment_id=manifest.experiment_id,
                status=AttemptStatus.RUNNING,
                started_at=self._clock(),
                source_commit=context.source_commit,
                clean_tree=context.clean_tree,
                dependency_lock_hash=manifest.dependency_lock_hash,
                container_digest=manifest.container_digest,
                hardware_inventory=context.hardware_inventory,
                compiler_inventory=context.compiler_inventory,
                resolved_manifest_hash=experiment.manifest_hash,
                command=manifest.command,
                environment=manifest.sanitized_environment,
                seed=context.seed,
                data_order_hash=context.data_order_hash,
                current_phase=current_phase,
                resume_checkpoint_id=resume_checkpoint_id,
                nondeterministic_operations=context.nondeterministic_operations,
                preregistered_tolerance_ids=context.preregistered_tolerance_ids,
            )
            self._append_record("attempt", resolved_attempt_id, record)
            self._attempts[resolved_attempt_id] = record
            self._run_experiment.setdefault(resolved_run_id, manifest.experiment_id)
            self._run_attempt_ids.setdefault(resolved_run_id, []).append(resolved_attempt_id)
            self._events_by_attempt[resolved_attempt_id] = []
            self._resource_ids_by_run.setdefault(resolved_run_id, [])
            return AttemptRef(resolved_run_id, resolved_attempt_id, manifest.experiment_id)


    def append_event(self, attempt: AttemptRef, event: RunEvent) -> RegistryRecordRef:
        """Append an event once; an exact retry is a no-op."""

        with self._lock:
            self._require_attempt(attempt)
            reference, created = self._append_record(
                "event",
                event.event_id,
                event,
                context={"attempt_id": attempt.attempt_id},
            )
            if created:
                self._events_by_attempt[attempt.attempt_id].append(event.event_id)
                self._event_owners[event.event_id] = attempt.attempt_id
            elif event.event_id not in self._events_by_attempt[attempt.attempt_id]:
                raise self._conflict(
                    "event ID is already owned by another attempt",
                    code="manifest.registry_event_owner_conflict",
                    affected_ids={"event_ids": (event.event_id,)},
                )
            return reference

    def attach_artifact(
        self,
        attempt: AttemptRef,
        artifact: ArtifactRef,
        *,
        attached_at: str | None = None,
        payload: bytes | None = None,
    ) -> RegistryRecordRef:
        """Attach an artifact and optionally publish its verified payload first."""

        with self._lock:
            record = self._require_attempt(attempt)
            attachment_key = (attempt.attempt_id, artifact.artifact_id)
            existing_attachment = self._attachments.get(attachment_key)
            if existing_attachment is not None:
                existing_artifact = self._artifacts[artifact.artifact_id]
                if existing_artifact == artifact:
                    return self._refs[("attachment", f"{attempt.attempt_id}:{artifact.artifact_id}")]
                raise self._conflict(
                    "artifact attachment retry changed artifact content",
                    code="manifest.registry_attachment_conflict",
                    affected_ids={"artifact_ids": (artifact.artifact_id,)},
                )
            if artifact.producing_run_id != attempt.run_id or (
                artifact.producing_attempt_id != attempt.attempt_id
            ):
                raise self._conflict(
                    "artifact producer must exactly match the owning attempt",
                    code="manifest.registry_artifact_producer_mismatch",
                    affected_ids={
                        "artifact_ids": (artifact.artifact_id,),
                        "attempt_ids": (attempt.attempt_id,),
                    },
                )
            parent_id = artifact.parent_artifact_id
            if parent_id is None or parent_id not in self._artifacts:
                raise self._conflict(
                    "produced artifacts require a registered direct parent",
                    code="manifest.registry_missing_direct_parent",
                    affected_ids={"artifact_ids": (artifact.artifact_id,)},
                )
            if record.experiment_id not in self._artifact_experiments.get(parent_id, set()):
                raise self._conflict(
                    "artifact parent belongs to another experiment",
                    code="manifest.registry_cross_experiment_parent",
                    affected_ids={"artifact_ids": (artifact.artifact_id, parent_id)},
                )
            parent = self._artifacts[parent_id]
            manifest = self._manifests[record.experiment_id]
            valid_parent = (
                parent_id == manifest.parent_checkpoint.artifact_id
                or parent.producing_attempt_id == attempt.attempt_id
                or parent_id == record.resume_checkpoint_id
            )
            if not valid_parent:
                raise self._conflict(
                    "artifact lineage may not flow sideways between attempts",
                    code="manifest.registry_sideways_lineage",
                    affected_ids={"artifact_ids": (artifact.artifact_id, parent_id)},
                )
            if payload is not None:
                if self._artifact_store is None:
                    raise self._conflict(
                        "artifact payload requires a configured filesystem artifact store",
                        code="manifest.registry_artifact_store_required",
                        affected_ids={"artifact_ids": (artifact.artifact_id,)},
                    )
                self._artifact_store.put_bytes(artifact, payload)
            artifact_ref, created = self._append_record("artifact", artifact.artifact_id, artifact)
            if created:
                self._artifacts[artifact.artifact_id] = artifact
            attachment = ArtifactAttachment(
                attempt.attempt_id, artifact, attached_at or self._clock()
            )
            attachment_id = f"{attempt.attempt_id}:{artifact.artifact_id}"
            reference, _ = self._append_record("attachment", attachment_id, attachment)
            self._attachments[attachment_key] = attachment
            self._artifact_experiments.setdefault(artifact.artifact_id, set()).add(
                record.experiment_id
            )
            del artifact_ref
            return reference

    def publish_checkpoint(
        self,
        attempt: AttemptRef,
        artifact: ArtifactRef,
        *,
        phase: int,
        complete: bool,
        state_hashes: Mapping[str, str],
        published_at: str | None = None,
        payload: bytes | None = None,
    ) -> RegistryRecordRef:
        """Publish a checkpoint record; only complete publications become resumable."""

        with self._lock:
            self._require_attempt(attempt)
            if attempt.attempt_id in self._terminal_statuses:
                raise self._conflict(
                    "cannot publish a checkpoint after attempt termination",
                    code="manifest.registry_checkpoint_after_termination",
                    affected_ids={"attempt_ids": (attempt.attempt_id,)},
                )
            if "checkpoint" not in artifact.kind:
                raise ValueError("checkpoint artifacts must use a checkpoint kind")
            resolved_published_at = published_at or self._clock()
            publication = CheckpointPublication(
                checkpoint_id=artifact.artifact_id,
                attempt_id=attempt.attempt_id,
                phase=phase,
                complete=complete,
                state_hashes=state_hashes,
                published_at=resolved_published_at,
            )
            self.attach_artifact(
                attempt,
                artifact,
                attached_at=resolved_published_at,
                payload=payload,
            )
            reference, created = self._append_record(
                "checkpoint", artifact.artifact_id, publication
            )
            if created:
                self._checkpoints[artifact.artifact_id] = publication
            return reference

    def finish_attempt(
        self,
        attempt: AttemptRef,
        status: AttemptStatus,
        *,
        failure_label: str | None = None,
        ended_at: str | None = None,
    ) -> RegistryRecordRef:
        """Append a terminal status without replacing the original attempt record."""

        with self._lock:
            self._require_attempt(attempt)
            existing = self._terminal_statuses.get(attempt.attempt_id)
            if existing is not None:
                candidate = AttemptStatusRecord(
                    attempt.attempt_id, status, ended_at or existing.ended_at, failure_label
                )
                reference, _ = self._append_record(
                    "attempt-status", attempt.attempt_id, candidate
                )
                return reference
            terminal = AttemptStatusRecord(
                attempt.attempt_id, status, ended_at or self._clock(), failure_label
            )
            reference, _ = self._append_record(
                "attempt-status", attempt.attempt_id, terminal
            )
            self._terminal_statuses[attempt.attempt_id] = terminal
            return reference

    def record_resources(self, attempt: AttemptRef, delta: ResourceDelta) -> RegistryRecordRef:
        """Append an idempotent resource delta for cumulative run accounting."""

        with self._lock:
            self._require_attempt(attempt)
            if delta.attempt_id != attempt.attempt_id:
                raise self._conflict(
                    "resource delta belongs to another attempt",
                    code="manifest.registry_resource_owner_mismatch",
                    affected_ids={"attempt_ids": (attempt.attempt_id, delta.attempt_id)},
                )
            run_entry_ids = self._resource_ids_by_run[attempt.run_id]
            currencies = {
                self._resource_entries[item].cost_currency for item in run_entry_ids
            }
            if currencies and delta.cost_currency not in currencies:
                raise self._conflict(
                    "resource ledger cannot combine currencies",
                    code="manifest.registry_resource_currency_mismatch",
                    affected_ids={"run_ids": (attempt.run_id,)},
                )
            reference, created = self._append_record(
                "resource-delta", delta.ledger_entry_id, delta
            )
            if created:
                self._resource_entries[delta.ledger_entry_id] = delta
                run_entry_ids.append(delta.ledger_entry_id)
            elif delta.ledger_entry_id not in run_entry_ids:
                raise self._conflict(
                    "resource entry ID is already owned by another run",
                    code="manifest.registry_resource_run_conflict",
                    affected_ids={"resource_delta_ids": (delta.ledger_entry_id,)},
                )
            return reference

    def cumulative_resource_ledger(self, run_or_attempt: str | AttemptRef) -> ResourceLedger:
        """Return totals across every retained attempt in a run."""

        with self._lock:
            run_id = run_or_attempt.run_id if isinstance(run_or_attempt, AttemptRef) else run_or_attempt
            if run_id not in self._run_experiment:
                raise self._conflict(
                    "unknown run ID",
                    code="manifest.registry_unknown_run",
                    affected_ids={"run_ids": (run_id,)},
                )
            entries = [self._resource_entries[item] for item in self._resource_ids_by_run[run_id]]
            attempt_ids = tuple(self._run_attempt_ids[run_id])
            interrupted = sum(
                self._terminal_statuses.get(item) is not None
                and self._terminal_statuses[item].status is AttemptStatus.INTERRUPTED
                for item in attempt_ids
            )
            return ResourceLedger(
                run_id=run_id,
                wall_seconds=math.fsum(item.wall_seconds for item in entries),
                accelerator_seconds=math.fsum(item.accelerator_seconds for item in entries),
                consumed_tokens=sum(item.consumed_tokens for item in entries),
                optimizer_steps=sum(item.optimizer_steps for item in entries),
                interrupted_attempts=interrupted,
                billable_cost=math.fsum(item.billable_cost for item in entries),
                cost_currency=entries[0].cost_currency if entries else "UNSPECIFIED",
                attempt_ids=attempt_ids,
            )


    def supersede(
        self,
        old: RegistryRecordRef,
        new: RegistryRecordRef,
        *,
        reason: str,
        supersession_id: str | None = None,
        created_at: str | None = None,
    ) -> RegistryRecordRef:
        """Append a correction edge while preserving both original records."""

        with self._lock:
            for reference in (old, new):
                if self._refs.get((reference.kind, reference.record_id)) != reference:
                    raise self._conflict(
                        "supersession references an unknown registry record",
                        code="manifest.registry_unknown_supersession_record",
                        affected_ids={"registry_record_ids": (reference.record_id,)},
                    )
            if self._sequence[(old.kind, old.record_id)] >= self._sequence[(new.kind, new.record_id)]:
                raise self._conflict(
                    "superseding record must have been appended after the original",
                    code="manifest.registry_invalid_supersession_order",
                    affected_ids={"registry_record_ids": (old.record_id, new.record_id)},
                )
            existing_id = self._superseded_by.get((old.kind, old.record_id))
            if existing_id is not None:
                existing = self._supersessions[existing_id]
                if existing.new == new and existing.reason == reason:
                    return self._refs[("supersession", existing_id)]
                raise self._conflict(
                    "record already has a different superseding correction",
                    code="manifest.registry_supersession_conflict",
                    affected_ids={"registry_record_ids": (old.record_id,)},
                )
            record = SupersessionRecord(
                supersession_id or self._id_allocator("supersession"),
                old,
                new,
                reason,
                created_at or self._clock(),
            )
            reference, _ = self._append_record(
                "supersession", record.supersession_id, record
            )
            self._supersessions[record.supersession_id] = record
            self._superseded_by[(old.kind, old.record_id)] = record.supersession_id
            return reference

    @staticmethod
    def _mapping_mismatches(
        expected: Mapping[str, str], observed: Mapping[str, str], *, kind: str
    ) -> list[Mapping[str, JsonValue]]:
        mismatches: list[Mapping[str, JsonValue]] = []
        for name in sorted(set(expected) | set(observed)):
            expected_value = expected.get(name)
            observed_value = observed.get(name)
            if expected_value != observed_value:
                mismatches.append(
                    {
                        "kind": kind,
                        "name": name,
                        "expected": expected_value,
                        "observed": observed_value,
                    }
                )
        return mismatches

    @staticmethod
    def _metric_mismatches(
        protocol: ReproductionProtocol, observed: Mapping[str, float]
    ) -> list[Mapping[str, JsonValue]]:
        tolerances = {item.metric_path: item for item in protocol.tolerances}
        mismatches: list[Mapping[str, JsonValue]] = []
        for name in sorted(set(protocol.expected_metrics) | set(observed)):
            expected = protocol.expected_metrics.get(name)
            actual = observed.get(name)
            if expected is None or actual is None or not math.isfinite(actual):
                mismatches.append(
                    {"kind": "metric", "name": name, "expected": expected, "observed": actual}
                )
                continue
            tolerance = tolerances[name]
            difference = abs(actual - expected)
            if tolerance.exact:
                passed = actual == expected
                allowed = 0.0
            else:
                bounds = []
                if tolerance.absolute is not None:
                    bounds.append(tolerance.absolute)
                if tolerance.relative is not None:
                    bounds.append(tolerance.relative * abs(expected))
                allowed = max(bounds)
                passed = difference <= allowed
            if not passed:
                mismatches.append(
                    {
                        "kind": "metric",
                        "name": name,
                        "expected": expected,
                        "observed": actual,
                        "difference": difference,
                        "allowed": allowed,
                        "tolerance_id": tolerance.tolerance_id,
                    }
                )
        return mismatches

    def validate_reproduction(
        self,
        protocol: ReproductionProtocol,
        observed_input_hashes: Mapping[str, str],
        execute: Callable[[], ReproductionOutcome],
        *,
        validation_id: str | None = None,
        validated_at: str | None = None,
    ) -> ReproductionValidation:
        """Verify inputs first, execute once, then compare exact hashes and tolerances."""

        with self._lock:
            original = self._attempts.get(protocol.original_attempt_id)
            if original is None:
                raise self._conflict(
                    "reproduction protocol references an unknown attempt",
                    code="manifest.registry_unknown_reproduction_attempt",
                    affected_ids={"attempt_ids": (protocol.original_attempt_id,)},
                )
            tolerance_ids = {item.tolerance_id for item in protocol.tolerances}
            if not tolerance_ids.issubset(set(original.preregistered_tolerance_ids)):
                raise ReproductionError(
                    "reproduction uses a tolerance that was not preregistered",
                    retryability=Retryability.NEVER,
                    code="reproduction.unregistered_tolerance",
                    affected_ids={"attempt_ids": (protocol.original_attempt_id,)},
                    context={"tolerance_ids": sorted(tolerance_ids)},
                )
            self._append_record("reproduction-protocol", protocol.protocol_id, protocol)
            resolved_validation_id = validation_id or self._id_allocator("reproduction")
            input_mismatches = self._mapping_mismatches(
                protocol.expected_input_hashes,
                observed_input_hashes,
                kind="input_hash",
            )
            if input_mismatches:
                validation = ReproductionValidation(
                    resolved_validation_id,
                    protocol.protocol_id,
                    protocol.original_attempt_id,
                    False,
                    False,
                    tuple(input_mismatches),
                    validated_at or self._clock(),
                )
                self._append_reproduction(validation)
                raise ReproductionError(
                    "reproduction input hashes differ from the preserved protocol",
                    retryability=Retryability.AFTER_REMEDIATION,
                    code="reproduction.input_hash_mismatch",
                    affected_ids={
                        "attempt_ids": (protocol.original_attempt_id,),
                        "reproduction_validation_ids": (resolved_validation_id,),
                    },
                    context={"mismatches": input_mismatches},
                )
            outcome = execute()
            if not isinstance(outcome, ReproductionOutcome):
                raise TypeError("reproduction executor must return ReproductionOutcome")
            mismatches = self._mapping_mismatches(
                protocol.expected_artifact_hashes,
                outcome.artifact_hashes,
                kind="artifact_hash",
            )
            mismatches.extend(self._metric_mismatches(protocol, outcome.metrics))
            validation = ReproductionValidation(
                resolved_validation_id,
                protocol.protocol_id,
                protocol.original_attempt_id,
                True,
                not mismatches,
                tuple(mismatches),
                validated_at or self._clock(),
            )
            self._append_reproduction(validation)
            return validation

    def _append_reproduction(
        self, validation: ReproductionValidation
    ) -> RegistryRecordRef:
        reference, created = self._append_record(
            "reproduction-validation", validation.validation_id, validation
        )
        if created:
            self._reproductions[validation.validation_id] = validation
        return reference

    def release_use_allowed(self, attempt: AttemptRef) -> bool:
        """Return false while an executed failed reproduction remains unsuperseded."""

        with self._lock:
            self._require_attempt(attempt)
            for validation_id, validation in self._reproductions.items():
                if validation.original_attempt_id != attempt.attempt_id or not validation.executed:
                    continue
                key = ("reproduction-validation", validation_id)
                if not validation.passed and key not in self._superseded_by:
                    return False
            return True

    def record_ref(self, kind: str, record_id: str) -> RegistryRecordRef:
        with self._lock:
            reference = self._refs.get((kind, record_id))
            if reference is None:
                raise KeyError((kind, record_id))
            return reference

    def attempt_status(self, attempt: AttemptRef) -> AttemptStatus:
        with self._lock:
            self._require_attempt(attempt)
            terminal = self._terminal_statuses.get(attempt.attempt_id)
            return terminal.status if terminal is not None else AttemptStatus.RUNNING

    def attempts(self, experiment_id: str | None = None) -> tuple[RunAttempt, ...]:
        with self._lock:
            records = tuple(self._attempts.values())
            if experiment_id is None:
                return records
            return tuple(item for item in records if item.experiment_id == experiment_id)

    def terminal_statuses(self) -> tuple[AttemptStatusRecord, ...]:
        with self._lock:
            return tuple(self._terminal_statuses.values())

    def events(self, attempt: AttemptRef) -> tuple[RunEvent, ...]:
        with self._lock:
            self._require_attempt(attempt)
            return tuple(
                self._records[("event", event_id)]  # type: ignore[misc]
                for event_id in self._events_by_attempt[attempt.attempt_id]
            )

    def checkpoints(self) -> tuple[CheckpointPublication, ...]:
        with self._lock:
            return tuple(self._checkpoints.values())

    def artifacts(self) -> tuple[ArtifactRef, ...]:
        with self._lock:
            return tuple(self._artifacts.values())

    def reproductions(self) -> tuple[ReproductionValidation, ...]:
        with self._lock:
            return tuple(self._reproductions.values())
