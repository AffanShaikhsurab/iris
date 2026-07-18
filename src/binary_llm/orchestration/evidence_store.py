"""Durable, content-addressed corpus and recovery evidence chains."""

from __future__ import annotations

import struct
import sys
from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from binary_llm.adapters import ActiveRepresentation
from binary_llm.domain import (
    ArtifactRef,
    ArtifactStoreError,
    ProvenanceViolation,
    Retryability,
    canonical_json_bytes,
    parse_canonical_json,
    sha256_bytes,
)

from .corpus import (
    CorpusDenyList,
    CorpusManifest,
    DeduplicationReport,
    DuplicateRemoval,
    IsolationExclusion,
    IsolationReport,
    PartitionRef,
    ProvenanceRecord,
    SplitPolicy,
    TokenAllocation,
)
from .recovery import (
    RecoveryCorpusPlan,
    RecoveryRunResult,
    RecoveryRunStatus,
    RecoveryTrainerConfig,
    SupervisionMode,
    TeacherRouter,
    validate_recovery_token_mix,
)
from .registry import AppendOnlyExperimentRegistry, AttemptRef, RegistryRecordRef
from .stage1 import CausalBatch
from .store import FilesystemArtifactStore

_CORPUS_MAGIC = b"BINARYLLM-CORPUS-BUNDLE\x00"
_RECOVERY_MAGIC = b"BINARYLLM-RECOVERY-EVIDENCE\x00"
_CORPUS_MEDIA = "application/vnd.binary-llm.corpus-bundle.v1"
_RECOVERY_MEDIA = "application/vnd.binary-llm.recovery-evidence.v1"
_DTYPES = {
    torch.bool: "bool",
    torch.uint8: "uint8",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
    torch.float64: "float64",
}
_TORCH_DTYPES = {name: dtype for dtype, name in _DTYPES.items()}


def _failure(message: str, code: str, *ids: str, **context: Any) -> ProvenanceViolation:
    return ProvenanceViolation(
        message,
        retryability=Retryability.NEVER,
        code=f"provenance.evidence.{code}",
        affected_ids={"artifact_ids": ids} if ids else None,
        context=context,
    )


def _frozen(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _frozen(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_frozen(item) for item in value)
    return value


def _plain(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _plain(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _denylist(value: CorpusDenyList) -> dict[str, Any]:
    return {
        "denylist_id": value.denylist_id,
        "content_hashes": list(value.content_hashes),
        "semantic_family_ids": list(value.semantic_family_ids),
        "split_family_ids": list(value.split_family_ids),
        "deduplication_keys": list(value.deduplication_keys),
        "fuzzy_cluster_ids": list(value.fuzzy_cluster_ids),
        "schema_version": value.schema_version,
        "manifest_hash": value.manifest_hash,
    }


def _policy(value: SplitPolicy) -> dict[str, Any]:
    return {
        "policy_id": value.policy_id,
        "seed": value.seed,
        "purpose": value.purpose,
        "target_allocation": [list(item) for item in value.target_allocation],
        "release_evidence_slices": list(value.release_evidence_slices),
        "allocation_tolerance": value.allocation_tolerance,
        "phase_count": value.phase_count,
        "frozen_evaluation": _denylist(value.frozen_evaluation),
        "public_benchmark": _denylist(value.public_benchmark),
        "schema_version": value.schema_version,
    }


def _record(value: ProvenanceRecord) -> dict[str, Any]:
    return {
        "record_id": value.record_id,
        "content_hash": value.content_hash,
        "source": value.source,
        "license_or_terms": value.license_or_terms,
        "permitted_use": value.permitted_use,
        "transformation_history": list(value.transformation_history),
        "semantic_family_id": value.semantic_family_id,
        "split_family_id": value.split_family_id,
        "deduplication_key": value.deduplication_key,
        "fuzzy_cluster_id": value.fuzzy_cluster_id,
        "target_type": value.target_type,
        "capability_slice": value.capability_slice,
        "token_count": value.token_count,
        "synthetic": value.synthetic,
        "executable_or_human_verified": value.executable_or_human_verified,
        "normalized_record": dict(value.normalized_record),
        "source_revision": value.source_revision,
        "teacher_identity": value.teacher_identity,
        "generation_settings": (
            None if value.generation_settings is None else dict(value.generation_settings)
        ),
        "source_document_id": value.source_document_id,
        "tool_template_id": value.tool_template_id,
        "synthetic_sibling_id": value.synthetic_sibling_id,
        "schema_version": value.schema_version,
    }


def _dedup(value: DeduplicationReport) -> dict[str, Any]:
    return {
        "method": value.method,
        "input_record_count": value.input_record_count,
        "output_record_count": value.output_record_count,
        "removals": [asdict(item) for item in value.removals],
        "report_hash": value.report_hash,
    }


def _manifest(value: CorpusManifest) -> dict[str, Any]:
    return {
        "corpus_id": value.corpus_id,
        "version": value.version,
        "records_ref": value.records_ref,
        "records": [_record(item) for item in value.records],
        "allocation_by_training_tokens": [
            asdict(item) for item in value.allocation_by_training_tokens
        ],
        "exact_dedup_report": _dedup(value.exact_dedup_report),
        "fuzzy_dedup_report": _dedup(value.fuzzy_dedup_report),
        "isolation_report": {
            "isolated": value.isolation_report.isolated,
            "checked_record_count": value.isolation_report.checked_record_count,
            "exclusions": [asdict(item) for item in value.isolation_report.exclusions],
            "report_hash": value.isolation_report.report_hash,
        },
        "split_policy_hash": value.split_policy_hash,
        "frozen_eval_denylist_hash": value.frozen_eval_denylist_hash,
        "public_benchmark_denylist_hash": value.public_benchmark_denylist_hash,
        "partition_refs": [asdict(item) for item in value.partition_refs],
        "total_tokens": value.total_tokens,
        "provenance_complete": value.provenance_complete,
        "release_slice_support": [list(item) for item in value.release_slice_support],
        "group_memberships": [
            [group_id, list(record_ids)]
            for group_id, record_ids in value.group_memberships
        ],
        "schema_version": value.schema_version,
    }


@dataclass(frozen=True, slots=True)
class FrozenCorpusBundle:
    artifact_ref: ArtifactRef
    payload: bytes
    manifest: CorpusManifest
    split_policy: SplitPolicy

    @property
    def bundle_id(self) -> str:
        return self.artifact_ref.artifact_id


def build_frozen_corpus_bundle(
    manifest: CorpusManifest,
    split_policy: SplitPolicy,
    *,
    parent_artifact_id: str | None = None,
    producing_run_id: str | None = None,
    producing_attempt_id: str | None = None,
) -> FrozenCorpusBundle:
    """Build one canonical bundle containing every accepted record exactly once."""

    if manifest.split_policy_hash != sha256_bytes(canonical_json_bytes({
        "policy_id": split_policy.policy_id,
        "seed": split_policy.seed,
        "purpose": split_policy.purpose,
        "target_allocation": split_policy.target_allocation,
        "release_evidence_slices": split_policy.release_evidence_slices,
        "allocation_tolerance": split_policy.allocation_tolerance,
        "phase_count": split_policy.phase_count,
        "frozen_evaluation_hash": split_policy.frozen_evaluation.manifest_hash,
        "public_benchmark_hash": split_policy.public_benchmark.manifest_hash,
        "schema_version": split_policy.schema_version,
    })):
        raise _failure("split policy does not match corpus manifest", "policy_mismatch")
    if (
        manifest.frozen_eval_denylist_hash != split_policy.frozen_evaluation.manifest_hash
        or manifest.public_benchmark_denylist_hash
        != split_policy.public_benchmark.manifest_hash
    ):
        raise _failure("deny-list identities do not match corpus manifest", "denylist_mismatch")
    metadata = {
        "schema_version": 1,
        "artifact_kind": "frozen_corpus_bundle",
        "manifest": _manifest(manifest),
        "split_policy": _policy(split_policy),
        "scientific_execution": False,
    }
    body = canonical_json_bytes(metadata)
    payload = _CORPUS_MAGIC + struct.pack("<Q", len(body)) + body
    digest = sha256_bytes(payload)
    ref = ArtifactRef(
        artifact_id=f"corpus-bundle:{digest}",
        kind="frozen_corpus_bundle",
        sha256=digest,
        bytes=len(payload),
        media_type=_CORPUS_MEDIA,
        parent_artifact_id=parent_artifact_id,
        producing_run_id=producing_run_id,
        producing_attempt_id=producing_attempt_id,
    )
    return FrozenCorpusBundle(ref, payload, manifest, split_policy)


def persist_frozen_corpus_bundle(
    bundle: FrozenCorpusBundle, store: FilesystemArtifactStore
) -> ArtifactRef:
    _decode_corpus(bundle.payload, bundle.artifact_ref)
    return store.put_bytes(bundle.artifact_ref, bundle.payload)


def _decode_envelope(
    payload: bytes, expected: ArtifactRef, magic: bytes, media_type: str, kind: str
) -> tuple[dict[str, Any], bytes]:
    if (
        expected.kind != kind
        or expected.media_type != media_type
        or expected.sha256 != sha256_bytes(payload)
        or expected.bytes != len(payload)
        or expected.artifact_id != f"{'corpus-bundle' if kind == 'frozen_corpus_bundle' else 'recovery-evidence'}:{expected.sha256}"
    ):
        raise _failure("artifact reference does not match evidence bytes", "reference_mismatch", expected.artifact_id)
    if not payload.startswith(magic) or len(payload) < len(magic) + 8:
        raise _failure("evidence envelope header is invalid", "invalid_header", expected.artifact_id)
    size = struct.unpack("<Q", payload[len(magic):len(magic) + 8])[0]
    start = len(magic) + 8
    end = start + size
    if end > len(payload):
        raise _failure("evidence metadata is truncated", "truncated", expected.artifact_id)
    try:
        metadata = parse_canonical_json(payload[start:end])
    except Exception as error:
        raise _failure("evidence metadata is not canonical", "invalid_metadata", expected.artifact_id) from error
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
        raise _failure("evidence schema is invalid", "invalid_schema", expected.artifact_id)
    return metadata, payload[end:]


def _decode_denylist(data: Mapping[str, Any]) -> CorpusDenyList:
    value = CorpusDenyList(
        denylist_id=data["denylist_id"],
        content_hashes=tuple(data["content_hashes"]),
        semantic_family_ids=tuple(data["semantic_family_ids"]),
        split_family_ids=tuple(data["split_family_ids"]),
        deduplication_keys=tuple(data["deduplication_keys"]),
        fuzzy_cluster_ids=tuple(data["fuzzy_cluster_ids"]),
        schema_version=data["schema_version"],
    )
    if value.manifest_hash != data["manifest_hash"]:
        raise _failure("deny-list content hash is inconsistent", "denylist_tampered")
    return value


def _decode_report(data: Mapping[str, Any]) -> DeduplicationReport:
    report = DeduplicationReport(
        method=data["method"],
        input_record_count=data["input_record_count"],
        output_record_count=data["output_record_count"],
        removals=tuple(DuplicateRemoval(**item) for item in data["removals"]),
        report_hash=data["report_hash"],
    )
    expected = sha256_bytes(canonical_json_bytes({
        "method": report.method,
        "input_record_count": report.input_record_count,
        "output_record_count": report.output_record_count,
        "removals": [
            {
                "dropped_record_id": item.dropped_record_id,
                "retained_record_id": item.retained_record_id,
                "reason": item.reason,
            }
            for item in report.removals
        ],
    }))
    if report.report_hash != expected:
        raise _failure("deduplication report hash is inconsistent", "dedup_tampered")
    return report


def _decode_corpus(
    payload: bytes, expected: ArtifactRef
) -> tuple[CorpusManifest, SplitPolicy]:
    metadata, trailing = _decode_envelope(
        payload, expected, _CORPUS_MAGIC, _CORPUS_MEDIA, "frozen_corpus_bundle"
    )
    if trailing or metadata.get("artifact_kind") != "frozen_corpus_bundle":
        raise _failure("corpus bundle contains invalid trailing data", "corpus_layout", expected.artifact_id)
    try:
        policy_data = metadata["split_policy"]
        policy = SplitPolicy(
            policy_id=policy_data["policy_id"],
            seed=policy_data["seed"],
            purpose=policy_data["purpose"],
            target_allocation=tuple(tuple(item) for item in policy_data["target_allocation"]),
            release_evidence_slices=tuple(policy_data["release_evidence_slices"]),
            allocation_tolerance=policy_data["allocation_tolerance"],
            phase_count=policy_data["phase_count"],
            frozen_evaluation=_decode_denylist(policy_data["frozen_evaluation"]),
            public_benchmark=_decode_denylist(policy_data["public_benchmark"]),
            schema_version=policy_data["schema_version"],
        )
        data = metadata["manifest"]
        records = tuple(ProvenanceRecord(**{
            **item,
            "transformation_history": tuple(item["transformation_history"]),
        }) for item in data["records"])
        isolation = data["isolation_report"]
        manifest = CorpusManifest(
            corpus_id=data["corpus_id"],
            version=data["version"],
            records_ref=data["records_ref"],
            records=records,
            allocation_by_training_tokens=tuple(
                TokenAllocation(**item) for item in data["allocation_by_training_tokens"]
            ),
            exact_dedup_report=_decode_report(data["exact_dedup_report"]),
            fuzzy_dedup_report=_decode_report(data["fuzzy_dedup_report"]),
            isolation_report=IsolationReport(
                isolated=isolation["isolated"],
                checked_record_count=isolation["checked_record_count"],
                exclusions=tuple(IsolationExclusion(
                    record_id=item["record_id"],
                    denylist_id=item["denylist_id"],
                    reasons=tuple(item["reasons"]),
                ) for item in isolation["exclusions"]),
                report_hash=isolation["report_hash"],
            ),
            split_policy_hash=data["split_policy_hash"],
            frozen_eval_denylist_hash=data["frozen_eval_denylist_hash"],
            public_benchmark_denylist_hash=data["public_benchmark_denylist_hash"],
            partition_refs=tuple(PartitionRef(**{
                **item,
                "record_ids": tuple(item["record_ids"]),
                "content_hashes": tuple(item["content_hashes"]),
                "semantic_family_ids": tuple(item["semantic_family_ids"]),
                "split_family_ids": tuple(item["split_family_ids"]),
                "group_ids": tuple(item["group_ids"]),
            }) for item in data["partition_refs"]),
            total_tokens=data["total_tokens"],
            provenance_complete=data["provenance_complete"],
            release_slice_support=tuple(tuple(item) for item in data["release_slice_support"]),
            group_memberships=tuple(
                (item[0], tuple(item[1])) for item in data["group_memberships"]
            ),
            schema_version=data["schema_version"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise _failure("corpus bundle cannot be reconstructed", "corpus_invalid", expected.artifact_id) from error
    records_ref = sha256_bytes(canonical_json_bytes([_record(item) for item in manifest.records]))
    isolation_hash = sha256_bytes(canonical_json_bytes({
        "checked_record_count": manifest.isolation_report.checked_record_count,
        "exclusions": [
            [item.record_id, item.denylist_id, list(item.reasons)]
            for item in manifest.isolation_report.exclusions
        ],
    }))
    partition_hashes = []
    records_by_id = {item.record_id: item for item in manifest.records}
    memberships = dict(manifest.group_memberships)
    for partition in manifest.partition_refs:
        partition_hash = sha256_bytes(canonical_json_bytes({
            "phase_index": partition.phase_index,
            "seed": partition.seed,
            "group_ids": list(partition.group_ids),
            "records": [
                [
                    record_id,
                    records_by_id[record_id].content_hash,
                    records_by_id[record_id].token_count,
                ]
                for record_id in partition.record_ids
            ],
        }))
        if (
            partition.manifest_hash != partition_hash
            or partition.partition_id
            != f"partition-{partition.phase_index:02d}-{partition_hash[:16]}"
            or partition.content_hashes
            != tuple(records_by_id[item].content_hash for item in partition.record_ids)
            or partition.token_count
            != sum(records_by_id[item].token_count for item in partition.record_ids)
            or tuple(
                sorted(item for group_id in partition.group_ids for item in memberships[group_id])
            )
            != tuple(sorted(partition.record_ids))
        ):
            raise _failure("partition evidence is internally inconsistent", "partition_tampered", expected.artifact_id)
        partition_hashes.append(partition_hash)
    corpus_id = sha256_bytes(canonical_json_bytes({
        "version": manifest.version,
        "records_ref": records_ref,
        "split_policy_hash": manifest.split_policy_hash,
        "exact_dedup_report": manifest.exact_dedup_report.report_hash,
        "fuzzy_dedup_report": manifest.fuzzy_dedup_report.report_hash,
        "isolation_report": isolation_hash,
        "partition_hashes": partition_hashes,
    }))
    if (
        manifest.records_ref != records_ref
        or manifest.isolation_report.report_hash != isolation_hash
        or manifest.corpus_id != corpus_id
    ):
        raise _failure("corpus manifest identity is inconsistent", "manifest_tampered", expected.artifact_id)
    rebuilt = build_frozen_corpus_bundle(
        manifest,
        policy,
        parent_artifact_id=expected.parent_artifact_id,
        producing_run_id=expected.producing_run_id,
        producing_attempt_id=expected.producing_attempt_id,
    )
    if rebuilt.payload != payload or rebuilt.artifact_ref != expected:
        raise _failure("corpus bundle identities are internally inconsistent", "corpus_identity", expected.artifact_id)
    return manifest, policy


def load_frozen_corpus_bundle(
    store: FilesystemArtifactStore, artifact_ref: ArtifactRef
) -> FrozenCorpusBundle:
    try:
        payload = store.resolve(artifact_ref.artifact_id, expected=artifact_ref)
    except ArtifactStoreError:
        raise
    manifest, policy = _decode_corpus(payload, artifact_ref)
    return FrozenCorpusBundle(artifact_ref, payload, manifest, policy)


def _tensor_bytes(value: Tensor) -> bytes:
    if sys.byteorder != "little":
        raise _failure("tensor evidence requires a little-endian host", "endian")
    tensor = value.detach().cpu().contiguous()
    if tensor.dtype not in _DTYPES:
        raise _failure("batch tensor dtype is unsupported", "tensor_dtype", dtype=str(tensor.dtype))
    return tensor.view(torch.uint8).numpy().tobytes()


def _trainer_config(value: RecoveryTrainerConfig) -> dict[str, Any]:
    optimizer = value.optimizer
    return {
        "optimizer": {
            "kind": optimizer.kind.value,
            "learning_rate": optimizer.learning_rate,
            "beta1": optimizer.beta1,
            "beta2": optimizer.beta2,
            "epsilon": optimizer.epsilon,
            "weight_decay": optimizer.weight_decay,
            "momentum": optimizer.momentum,
        },
        "seed": value.seed,
        "max_optimizer_steps": value.max_optimizer_steps,
        "target_representation": value.target_representation.value,
        "progression_parameter": value.progression_parameter,
        "causal_loss_weight": value.causal_loss_weight,
        "teacher_loss_weight": value.teacher_loss_weight,
        "no_progress_policy": asdict(value.no_progress_policy),
    }


def _result(value: RecoveryRunResult) -> dict[str, Any]:
    return {
        "status": value.status.value,
        "completed_steps": value.completed_steps,
        "mix_evidence": asdict(value.mix_evidence),
        "progress": [asdict(item) for item in value.progress],
        "steps": [asdict(item) for item in value.steps],
        "forward_events": [asdict(item) for item in value.forward_events],
        "teacher_provenance": [_plain(item) for item in value.teacher_provenance],
        "no_progress_decision": (
            None if value.no_progress_decision is None else asdict(value.no_progress_decision)
        ),
        "budget_usage": None if value.budget_usage is None else asdict(value.budget_usage),
        "budget_crossing": (
            None if value.budget_crossing is None else asdict(value.budget_crossing)
        ),
        "budget_failure": (
            None if value.budget_failure is None else value.budget_failure.to_dict()
        ),
    }


@dataclass(frozen=True, slots=True)
class RecoveryRunEvidence:
    artifact_ref: ArtifactRef
    payload: bytes
    metadata: Mapping[str, Any]
    batches: Mapping[str, CausalBatch]
    corpus_bundle: FrozenCorpusBundle
    promotion_eligible: bool = False

    @property
    def evidence_id(self) -> str:
        return self.artifact_ref.artifact_id


def build_recovery_run_evidence(
    *,
    run_id: str,
    attempt_id: str | None,
    result: RecoveryRunResult,
    config: RecoveryTrainerConfig,
    corpus_bundle: FrozenCorpusBundle,
    corpus_plan: RecoveryCorpusPlan,
    selected_mix_id: str,
    records: Sequence[ProvenanceRecord],
    batches: Mapping[str, CausalBatch],
    teacher_router: TeacherRouter,
    parent_artifact_id: str | None,
    checkpoint_artifact_id: str | None = None,
) -> RecoveryRunEvidence:
    """Bind a completed or stopped recovery result to exact durable inputs."""

    selected_mix = corpus_plan.select(selected_mix_id)
    observed_mix = validate_recovery_token_mix(records, selected_mix)
    if observed_mix != result.mix_evidence:
        raise _failure("result token mix differs from selected corpus records", "mix_evidence")
    corpus_records = {item.record_id: item for item in corpus_bundle.manifest.records}
    if any(corpus_records.get(item.record_id) != item for item in records):
        raise _failure("recovery records are not exact corpus-bundle members", "record_substitution")
    record_ids = tuple(item.record_id for item in records)
    if set(record_ids) != set(batches) or result.mix_evidence.mix_id != selected_mix_id:
        raise _failure("recovery batch or selected mix identity is inconsistent", "mix_substitution")
    if tuple(item.capability_slice for item in records) and (
        result.mix_evidence.record_count != len(records)
        or result.mix_evidence.total_tokens != sum(item.token_count for item in records)
    ):
        raise _failure("recovery mix evidence does not describe selected records", "mix_evidence")
    events = result.forward_events
    if tuple(item.forward_index for item in events) != tuple(range(len(events))) or any(
        not item.representations
        or any(rep is not config.target_representation for rep in item.representations)
        for item in events
    ):
        raise _failure("binary-active forward evidence is incomplete or substituted", "representation")
    if sum(item.purpose == "recovery_training" for item in events) != result.completed_steps:
        raise _failure("candidate training forwards do not match completed steps", "forward_count")
    routed = {item.record_id: item for item in result.teacher_provenance}
    for item in result.steps:
        provenance = routed.get(item.record_id)
        if item.teacher_identity_id is None:
            if item.supervision_mode is not SupervisionMode.NONE:
                raise _failure("teacher-free step declares supervision", "teacher_route")
        elif (
            provenance is None
            or provenance.teacher_identity_id != item.teacher_identity_id
            or provenance.supervision_mode is not item.supervision_mode
        ):
            raise _failure("teacher provenance does not match routed step", "teacher_route")
    for record_id, provenance in routed.items():
        record = corpus_records.get(record_id)
        if record is None:
            raise _failure("teacher provenance references a foreign record", "teacher_route")
        behavioral = record.capability_slice in teacher_router.config.behavioral_slices
        port = (
            teacher_router.behavioral_teacher
            if behavioral
            else teacher_router.broad_teacher
        )
        mode = (
            teacher_router.config.behavioral_mode
            if behavioral
            else teacher_router.config.broad_mode
        )
        if port is None:
            raise _failure("teacher provenance exists for a disabled route", "teacher_route")
        identity = port.identity()
        if (
            provenance.teacher_identity_id != identity.identity_id
            or provenance.teacher_id != identity.teacher_id
            or provenance.role is not identity.role
            or provenance.revision != identity.revision
            or provenance.tokenizer_id != identity.tokenizer_id
            or provenance.license_or_terms != identity.license_or_terms
            or provenance.permitted_use != identity.permitted_use
            or dict(provenance.generation_settings) != dict(identity.generation_settings)
            or provenance.supervision_mode is not mode
        ):
            raise _failure("teacher provenance differs from the routed identity", "teacher_route")
    binding = teacher_router.behavioral_binding
    behavioral_enabled = teacher_router.config.behavioral_mode is not SupervisionMode.NONE
    if behavioral_enabled and binding is None:
        raise _failure("behavioral recovery lacks verified teacher binding", "teacher_binding")
    if binding is not None and (
        binding.teacher_identity_id != teacher_router.config.sealed_bf16_teacher_identity_id
        or binding.inventory_id != teacher_router.config.sealed_bf16_inventory_id
    ):
        raise _failure("verified teacher binding does not match routing", "teacher_binding")

    tensor_records: list[dict[str, Any]] = []
    chunks: list[bytes] = []
    offset = 0
    batch_hashes: dict[str, str] = {}
    for record_id in record_ids:
        batch = batches[record_id]
        if batch.batch_id.strip() == "":
            raise _failure("batch ID is empty", "batch_invalid")
        descriptors = []
        for field_name, tensor in (("input_ids", batch.input_ids), ("labels", batch.labels)):
            if tensor is None:
                continue
            raw = _tensor_bytes(tensor)
            descriptor = {
                "name": f"{record_id}.{field_name}",
                "record_id": record_id,
                "field": field_name,
                "batch_id": batch.batch_id,
                "dtype": _DTYPES[tensor.dtype],
                "shape": list(tensor.shape),
                "byte_order": "little",
                "order": "C",
                "offset": offset,
                "bytes": len(raw),
                "sha256": sha256_bytes(raw),
            }
            tensor_records.append(descriptor)
            descriptors.append(descriptor)
            chunks.append(raw)
            offset += len(raw)
        batch_hashes[record_id] = sha256_bytes(canonical_json_bytes(descriptors))
    plan_data = {
        "initial_mix": asdict(corpus_plan.initial_mix),
        "alternative_mixes": [asdict(item) for item in corpus_plan.alternative_mixes],
        "selected_mix_id": selected_mix_id,
        "selected_mix": asdict(selected_mix),
    }
    binding_data = None if binding is None else {
        "binding_id": binding.binding_id,
        "inventory_id": binding.inventory_id,
        "baseline_id": binding.baseline_id,
        "baseline_content_id": binding.baseline_content_id,
        "teacher_identity_id": binding.teacher_identity_id,
        "weight_artifacts": [item.to_dict() for item in binding.weight_artifacts],
        "tokenizer_artifacts": [item.to_dict() for item in binding.tokenizer_artifacts],
        "template_artifacts": [item.to_dict() for item in binding.template_artifacts],
        "configuration_artifacts": [item.to_dict() for item in binding.configuration_artifacts],
        "weight_aggregate_sha256": binding.weight_aggregate_sha256,
    }
    metadata = {
        "schema_version": 1,
        "artifact_kind": "recovery_run_evidence",
        "run_id": run_id,
        "attempt_id": attempt_id,
        "parent_artifact_id": parent_artifact_id,
        "checkpoint_artifact_id": checkpoint_artifact_id,
        "corpus_bundle_ref": corpus_bundle.artifact_ref.to_dict(),
        "corpus_id": corpus_bundle.manifest.corpus_id,
        "records_ref": corpus_bundle.manifest.records_ref,
        "record_ids": list(record_ids),
        "record_content_hashes": [corpus_records[item].content_hash for item in record_ids],
        "batch_hashes": batch_hashes,
        "batch_tensors": tensor_records,
        "trainer_config": _trainer_config(config),
        "config_identity": sha256_bytes(canonical_json_bytes(_trainer_config(config))),
        "target_representation": config.target_representation.value,
        "corpus_plan": plan_data,
        "teacher_routing": {
            "student_tokenizer_id": teacher_router.config.student_tokenizer_id,
            "sealed_bf16_teacher_identity_id": teacher_router.config.sealed_bf16_teacher_identity_id,
            "sealed_bf16_inventory_id": teacher_router.config.sealed_bf16_inventory_id,
            "behavioral_mode": teacher_router.config.behavioral_mode.value,
            "broad_mode": teacher_router.config.broad_mode.value,
            "behavioral_slices": list(teacher_router.config.behavioral_slices),
        },
        "verified_teacher_binding": binding_data,
        "result": _result(result),
        "stop_reason": (
            None
            if result.status is RecoveryRunStatus.COMPLETED
            else "no_progress"
            if result.status is RecoveryRunStatus.STOPPED_NO_PROGRESS
            else "budget"
        ),
        "promotion_eligible": False,
        "scientific_promotion_implied": False,
    }
    body = canonical_json_bytes(metadata)
    payload = _RECOVERY_MAGIC + struct.pack("<Q", len(body)) + body + b"".join(chunks)
    digest = sha256_bytes(payload)
    ref = ArtifactRef(
        artifact_id=f"recovery-evidence:{digest}",
        kind="recovery_run_evidence",
        sha256=digest,
        bytes=len(payload),
        media_type=_RECOVERY_MEDIA,
        parent_artifact_id=parent_artifact_id,
        producing_run_id=run_id,
        producing_attempt_id=attempt_id,
    )
    return RecoveryRunEvidence(
        ref, payload, _frozen(metadata), MappingProxyType(dict(batches)),
        corpus_bundle, False
    )


def _artifact_ref(data: Mapping[str, Any]) -> ArtifactRef:
    return ArtifactRef(**dict(data))


def _decode_recovery(
    payload: bytes,
    expected: ArtifactRef,
    store: FilesystemArtifactStore,
) -> RecoveryRunEvidence:
    metadata, tensor_area = _decode_envelope(
        payload, expected, _RECOVERY_MAGIC, _RECOVERY_MEDIA, "recovery_run_evidence"
    )
    if (
        metadata.get("artifact_kind") != "recovery_run_evidence"
        or metadata.get("promotion_eligible") is not False
        or metadata.get("scientific_promotion_implied") is not False
        or metadata.get("run_id") != expected.producing_run_id
        or metadata.get("attempt_id") != expected.producing_attempt_id
        or metadata.get("parent_artifact_id") != expected.parent_artifact_id
    ):
        raise _failure("recovery evidence ownership or eligibility is invalid", "recovery_metadata", expected.artifact_id)
    corpus_ref = _artifact_ref(metadata["corpus_bundle_ref"])
    corpus = load_frozen_corpus_bundle(store, corpus_ref)
    if (
        metadata.get("corpus_id") != corpus.manifest.corpus_id
        or metadata.get("records_ref") != corpus.manifest.records_ref
    ):
        raise _failure("recovery evidence references a different corpus manifest", "cross_manifest", expected.artifact_id)
    corpus_by_id = {item.record_id: item for item in corpus.manifest.records}
    record_ids = metadata.get("record_ids")
    content_hashes = metadata.get("record_content_hashes")
    if (
        not isinstance(record_ids, list)
        or not isinstance(content_hashes, list)
        or len(record_ids) != len(set(record_ids))
        or [corpus_by_id.get(item).content_hash if item in corpus_by_id else None for item in record_ids]
        != content_hashes
    ):
        raise _failure("recovery records do not resolve in selected corpus", "record_substitution", expected.artifact_id)
    tensors: dict[str, dict[str, Tensor]] = {item: {} for item in record_ids}
    expected_offset = 0
    descriptors: dict[str, list[Mapping[str, Any]]] = {item: [] for item in record_ids}
    for item in metadata.get("batch_tensors", []):
        if (
            not isinstance(item, dict)
            or item.get("record_id") not in tensors
            or item.get("field") not in {"input_ids", "labels"}
            or item.get("dtype") not in _TORCH_DTYPES
            or item.get("offset") != expected_offset
            or item.get("byte_order") != "little"
            or item.get("order") != "C"
        ):
            raise _failure("recovery tensor manifest is invalid", "tensor_manifest", expected.artifact_id)
        size = item.get("bytes")
        shape = item.get("shape")
        if not isinstance(size, int) or not isinstance(shape, list):
            raise _failure("recovery tensor metadata is invalid", "tensor_metadata", expected.artifact_id)
        raw = tensor_area[expected_offset:expected_offset + size]
        if len(raw) != size or sha256_bytes(raw) != item.get("sha256"):
            raise _failure("recovery tensor failed integrity verification", "tensor_digest", expected.artifact_id)
        try:
            tensor = torch.frombuffer(
                bytearray(raw), dtype=_TORCH_DTYPES[item["dtype"]]
            ).clone().reshape(shape)
        except (RuntimeError, ValueError) as error:
            raise _failure("recovery tensor shape is invalid", "tensor_shape", expected.artifact_id) from error
        tensors[item["record_id"]][item["field"]] = tensor
        descriptors[item["record_id"]].append(item)
        expected_offset += size
    if expected_offset != len(tensor_area):
        raise _failure("recovery evidence has trailing tensor bytes", "tensor_layout", expected.artifact_id)
    batches: dict[str, CausalBatch] = {}
    for record_id in record_ids:
        fields = tensors[record_id]
        if "input_ids" not in fields:
            raise _failure("recovery batch input tensor is missing", "batch_missing", expected.artifact_id)
        expected_hash = metadata["batch_hashes"].get(record_id)
        if sha256_bytes(canonical_json_bytes(descriptors[record_id])) != expected_hash:
            raise _failure("recovery batch identity is inconsistent", "batch_substitution", expected.artifact_id)
        descriptor = descriptors[record_id][0]
        batches[record_id] = CausalBatch(
            batch_id=descriptor["batch_id"],
            input_ids=fields["input_ids"],
            labels=fields.get("labels"),
        )
    routing = metadata["teacher_routing"]
    binding = metadata.get("verified_teacher_binding")
    if routing["behavioral_mode"] != SupervisionMode.NONE.value and binding is None:
        raise _failure("behavioral evidence omits verified binding", "teacher_binding", expected.artifact_id)
    if binding is not None:
        if (
            binding["teacher_identity_id"] != routing["sealed_bf16_teacher_identity_id"]
            or binding["inventory_id"] != routing["sealed_bf16_inventory_id"]
        ):
            raise _failure("teacher binding and route disagree", "teacher_binding", expected.artifact_id)
        for category in (
            "weight_artifacts", "tokenizer_artifacts",
            "template_artifacts", "configuration_artifacts",
        ):
            for artifact_data in binding[category]:
                ref = _artifact_ref(artifact_data)
                store.resolve(ref.artifact_id, expected=ref)
    result = metadata["result"]
    if (
        result["mix_evidence"]["mix_id"] != metadata["corpus_plan"]["selected_mix_id"]
        or result["completed_steps"] != len(result["steps"])
        or result["status"] not in {item.value for item in RecoveryRunStatus}
    ):
        raise _failure("recovery result summary is inconsistent", "result_invalid", expected.artifact_id)
    target = ActiveRepresentation(metadata["target_representation"])
    events = result["forward_events"]
    if [item["forward_index"] for item in events] != list(range(len(events))) or any(
        not item["representations"]
        or any(ActiveRepresentation(rep) is not target for rep in item["representations"])
        for item in events
    ):
        raise _failure("recovery representation evidence was substituted", "representation", expected.artifact_id)
    return RecoveryRunEvidence(
        expected,
        payload,
        _frozen(metadata),
        MappingProxyType(batches),
        corpus,
        False,
    )


def persist_recovery_run_evidence(
    evidence: RecoveryRunEvidence,
    store: FilesystemArtifactStore,
    *,
    registry: AppendOnlyExperimentRegistry | None = None,
    attempt: AttemptRef | None = None,
) -> ArtifactRef | RegistryRecordRef:
    """Persist evidence and optionally attach it with registry lineage checks."""

    _decode_recovery(evidence.payload, evidence.artifact_ref, store)
    if registry is None:
        if attempt is not None:
            raise ValueError("attempt requires a registry")
        return store.put_bytes(evidence.artifact_ref, evidence.payload)
    if attempt is None:
        raise ValueError("registry attachment requires an attempt")
    return registry.attach_artifact(
        attempt, evidence.artifact_ref, payload=evidence.payload
    )


def load_recovery_run_evidence(
    store: FilesystemArtifactStore, artifact_ref: ArtifactRef
) -> RecoveryRunEvidence:
    payload = store.resolve(artifact_ref.artifact_id, expected=artifact_ref)
    return _decode_recovery(payload, artifact_ref, store)


__all__ = [
    "FrozenCorpusBundle",
    "RecoveryRunEvidence",
    "build_frozen_corpus_bundle",
    "build_recovery_run_evidence",
    "load_frozen_corpus_bundle",
    "load_recovery_run_evidence",
    "persist_frozen_corpus_bundle",
    "persist_recovery_run_evidence",
]
