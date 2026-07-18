"""Provenance-safe corpus freezing and deterministic progressive partitioning."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from binary_llm.domain import canonical_json_bytes, sha256_bytes

PHASE_COUNT = 20
INITIAL_TOKEN_ALLOCATION: tuple[tuple[str, float], ...] = (
    ("broad_text", 0.30),
    ("math", 0.15),
    ("code", 0.10),
    ("iris_tools", 0.20),
    ("conversation", 0.10),
    ("safety_transitions", 0.10),
    ("adversarial", 0.05),
)
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_WORD = re.compile(r"[\w]+", re.UNICODE)


def _require_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _hash(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _content_hash(content: Mapping[str, Any]) -> str:
    return _hash(content)


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({
            key: _freeze_value(item) for key, item in value.items()
        })
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    return value


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return _freeze_value(value)


@dataclass(frozen=True, slots=True)
class ProvenanceRecord:
    """A normalized training record with complete source and usage provenance."""

    record_id: str
    content_hash: str
    source: str
    license_or_terms: str
    permitted_use: str
    transformation_history: tuple[str, ...]
    semantic_family_id: str
    split_family_id: str
    deduplication_key: str
    fuzzy_cluster_id: str
    target_type: str
    capability_slice: str
    token_count: int
    synthetic: bool
    executable_or_human_verified: bool
    normalized_record: Mapping[str, Any] = field(repr=False, compare=True)
    source_revision: str | None = None
    teacher_identity: str | None = None
    generation_settings: Mapping[str, Any] | None = field(default=None, repr=False)
    source_document_id: str | None = None
    tool_template_id: str | None = None
    synthetic_sibling_id: str | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        for name in (
            "record_id", "source", "license_or_terms", "permitted_use",
            "semantic_family_id", "split_family_id", "deduplication_key",
            "fuzzy_cluster_id", "target_type", "capability_slice",
        ):
            _require_text(name, getattr(self, name))
        if self.schema_version != 1:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        if not _HEX_64.fullmatch(self.content_hash):
            raise ValueError("content_hash must be a lowercase SHA-256 digest")
        if self.content_hash != _content_hash(self.normalized_record):
            raise ValueError("content_hash does not match normalized_record")
        if not self.transformation_history or any(
            not isinstance(item, str) or not item.strip()
            for item in self.transformation_history
        ):
            raise ValueError("transformation_history must contain non-empty entries")
        if (
            isinstance(self.token_count, bool)
            or not isinstance(self.token_count, int)
            or self.token_count <= 0
        ):
            raise ValueError("token_count must be a positive integer")
        if not isinstance(self.synthetic, bool) or not isinstance(
            self.executable_or_human_verified, bool
        ):
            raise TypeError("synthetic and verification flags must be booleans")
        if self.teacher_identity is not None:
            _require_text("teacher_identity", self.teacher_identity)
            if self.generation_settings is None:
                raise ValueError("teacher-generated records require generation_settings")
        for name in (
            "source_revision", "source_document_id", "tool_template_id",
            "synthetic_sibling_id",
        ):
            value = getattr(self, name)
            if value is not None:
                _require_text(name, value)
        object.__setattr__(self, "normalized_record", _freeze_mapping(self.normalized_record))
        if self.generation_settings is not None:
            object.__setattr__(self, "generation_settings", _freeze_mapping(self.generation_settings))

    @property
    def trustworthy_release_support(self) -> bool:
        return not self.synthetic and self.executable_or_human_verified

    def relation_keys(self) -> tuple[str, ...]:
        values = [
            f"semantic:{self.semantic_family_id}",
            f"split:{self.split_family_id}",
        ]
        for prefix, value in (
            ("source_document", self.source_document_id),
            ("tool_template", self.tool_template_id),
            ("synthetic_sibling", self.synthetic_sibling_id),
        ):
            if value is not None:
                values.append(f"{prefix}:{value}")
        return tuple(values)


@dataclass(frozen=True, slots=True)
class CorpusDenyList:
    """Content and family fingerprints that may never enter a frozen corpus."""

    denylist_id: str
    content_hashes: tuple[str, ...] = ()
    semantic_family_ids: tuple[str, ...] = ()
    split_family_ids: tuple[str, ...] = ()
    deduplication_keys: tuple[str, ...] = ()
    fuzzy_cluster_ids: tuple[str, ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        _require_text("denylist_id", self.denylist_id)
        if self.schema_version != 1:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        for name in (
            "content_hashes", "semantic_family_ids", "split_family_ids",
            "deduplication_keys", "fuzzy_cluster_ids",
        ):
            values = getattr(self, name)
            if len(values) != len(set(values)):
                raise ValueError(f"{name} contains duplicate fingerprints")
            if any(not isinstance(value, str) or not value for value in values):
                raise ValueError(f"{name} must contain non-empty strings")
        if any(not _HEX_64.fullmatch(value) for value in self.content_hashes):
            raise ValueError("content_hashes must contain lowercase SHA-256 digests")

    @property
    def manifest_hash(self) -> str:
        return _hash({
            "schema_version": self.schema_version,
            "denylist_id": self.denylist_id,
            "content_hashes": sorted(self.content_hashes),
            "semantic_family_ids": sorted(self.semantic_family_ids),
            "split_family_ids": sorted(self.split_family_ids),
            "deduplication_keys": sorted(self.deduplication_keys),
            "fuzzy_cluster_ids": sorted(self.fuzzy_cluster_ids),
        })

    def reasons(self, record: ProvenanceRecord) -> tuple[str, ...]:
        matches: list[str] = []
        for name, value, denied in (
            ("content_hash", record.content_hash, self.content_hashes),
            ("semantic_family", record.semantic_family_id, self.semantic_family_ids),
            ("split_family", record.split_family_id, self.split_family_ids),
            ("deduplication_key", record.deduplication_key, self.deduplication_keys),
            ("fuzzy_cluster", record.fuzzy_cluster_id, self.fuzzy_cluster_ids),
        ):
            if value in denied:
                matches.append(name)
        return tuple(matches)


EMPTY_FROZEN_EVALUATION_DENYLIST = CorpusDenyList("empty-frozen-evaluation")
EMPTY_PUBLIC_BENCHMARK_DENYLIST = CorpusDenyList("empty-public-benchmark")


@dataclass(frozen=True, slots=True)
class SplitPolicy:
    """Frozen choices governing deduplication, isolation, allocation, and order."""

    policy_id: str
    seed: int
    purpose: str = "recovery"
    target_allocation: tuple[tuple[str, float], ...] = INITIAL_TOKEN_ALLOCATION
    release_evidence_slices: tuple[str, ...] = ()
    allocation_tolerance: float = 0.01
    phase_count: int = PHASE_COUNT
    frozen_evaluation: CorpusDenyList = EMPTY_FROZEN_EVALUATION_DENYLIST
    public_benchmark: CorpusDenyList = EMPTY_PUBLIC_BENCHMARK_DENYLIST
    schema_version: int = 1

    def __post_init__(self) -> None:
        _require_text("policy_id", self.policy_id)
        _require_text("purpose", self.purpose)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if self.phase_count != PHASE_COUNT:
            raise ValueError("progressive corpus policy requires exactly 20 phases")
        if self.schema_version != 1:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        names = [name for name, _ in self.target_allocation]
        if not names or len(names) != len(set(names)):
            raise ValueError("target_allocation requires unique capability slices")
        if any(not isinstance(value, (int, float)) or value < 0 for _, value in self.target_allocation):
            raise ValueError("target allocation shares must be non-negative numbers")
        if abs(sum(value for _, value in self.target_allocation) - 1.0) > 1e-12:
            raise ValueError("target allocation shares must sum to one")
        if (
            isinstance(self.allocation_tolerance, bool)
            or not isinstance(self.allocation_tolerance, (int, float))
            or not 0.0 <= self.allocation_tolerance <= 1.0
        ):
            raise ValueError("allocation_tolerance must be between zero and one")
        if len(self.release_evidence_slices) != len(set(self.release_evidence_slices)):
            raise ValueError("release_evidence_slices contains duplicates")


@dataclass(frozen=True, slots=True)
class DuplicateRemoval:
    dropped_record_id: str
    retained_record_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class DeduplicationReport:
    method: str
    input_record_count: int
    output_record_count: int
    removals: tuple[DuplicateRemoval, ...]
    report_hash: str


@dataclass(frozen=True, slots=True)
class IsolationExclusion:
    record_id: str
    denylist_id: str
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class IsolationReport:
    isolated: bool
    checked_record_count: int
    exclusions: tuple[IsolationExclusion, ...]
    report_hash: str


@dataclass(frozen=True, slots=True)
class TokenAllocation:
    capability_slice: str
    target_fraction: float
    training_tokens: int
    actual_fraction: float


@dataclass(frozen=True, slots=True)
class PartitionRef:
    phase_index: int
    partition_id: str
    manifest_hash: str
    record_ids: tuple[str, ...]
    content_hashes: tuple[str, ...]
    semantic_family_ids: tuple[str, ...]
    split_family_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    token_count: int
    seed: int
    schema_version: int = 1


@dataclass(frozen=True, slots=True)
class CorpusManifest:
    corpus_id: str
    version: str
    records_ref: str
    records: tuple[ProvenanceRecord, ...]
    allocation_by_training_tokens: tuple[TokenAllocation, ...]
    exact_dedup_report: DeduplicationReport
    fuzzy_dedup_report: DeduplicationReport
    isolation_report: IsolationReport
    split_policy_hash: str
    frozen_eval_denylist_hash: str
    public_benchmark_denylist_hash: str
    partition_refs: tuple[PartitionRef, ...]
    total_tokens: int
    provenance_complete: bool
    release_slice_support: tuple[tuple[str, int], ...]
    group_memberships: tuple[tuple[str, tuple[str, ...]], ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        if len(self.partition_refs) != PHASE_COUNT:
            raise ValueError("a corpus manifest requires exactly 20 phase partitions")
        record_ids = {record.record_id for record in self.records}
        partition_ids = [
            record_id
            for partition in self.partition_refs
            for record_id in partition.record_ids
        ]
        if len(partition_ids) != len(set(partition_ids)):
            raise ValueError("phase partitions overlap")
        if set(partition_ids) != record_ids:
            raise ValueError("phase partitions do not cover the frozen corpus")
        if self.total_tokens != sum(record.token_count for record in self.records):
            raise ValueError("total_tokens does not match frozen records")


@dataclass(frozen=True, slots=True)
class _Group:
    group_id: str
    records: tuple[ProvenanceRecord, ...]

    @property
    def token_count(self) -> int:
        return sum(record.token_count for record in self.records)


def _record_evidence(record: ProvenanceRecord) -> Mapping[str, Any]:
    return {
        "record_id": record.record_id,
        "content_hash": record.content_hash,
        "source": record.source,
        "source_revision": record.source_revision,
        "license_or_terms": record.license_or_terms,
        "permitted_use": record.permitted_use,
        "transformation_history": record.transformation_history,
        "semantic_family_id": record.semantic_family_id,
        "split_family_id": record.split_family_id,
        "deduplication_key": record.deduplication_key,
        "fuzzy_cluster_id": record.fuzzy_cluster_id,
        "teacher_identity": record.teacher_identity,
        "generation_settings": record.generation_settings,
        "target_type": record.target_type,
        "capability_slice": record.capability_slice,
        "token_count": record.token_count,
        "synthetic": record.synthetic,
        "executable_or_human_verified": record.executable_or_human_verified,
        "normalized_record": record.normalized_record,
        "source_document_id": record.source_document_id,
        "tool_template_id": record.tool_template_id,
        "synthetic_sibling_id": record.synthetic_sibling_id,
        "schema_version": record.schema_version,
    }


def _policy_evidence(policy: SplitPolicy) -> Mapping[str, Any]:
    return {
        "policy_id": policy.policy_id,
        "seed": policy.seed,
        "purpose": policy.purpose,
        "target_allocation": policy.target_allocation,
        "release_evidence_slices": policy.release_evidence_slices,
        "allocation_tolerance": policy.allocation_tolerance,
        "phase_count": policy.phase_count,
        "frozen_evaluation_hash": policy.frozen_evaluation.manifest_hash,
        "public_benchmark_hash": policy.public_benchmark.manifest_hash,
        "schema_version": policy.schema_version,
    }


def _report(method: str, before: int, records: tuple[ProvenanceRecord, ...],
            removals: tuple[DuplicateRemoval, ...]) -> DeduplicationReport:
    payload = {
        "method": method,
        "input_record_count": before,
        "output_record_count": len(records),
        "removals": tuple(
            {
                "dropped_record_id": item.dropped_record_id,
                "retained_record_id": item.retained_record_id,
                "reason": item.reason,
            }
            for item in removals
        ),
    }
    return DeduplicationReport(
        method=method,
        input_record_count=before,
        output_record_count=len(records),
        removals=removals,
        report_hash=_hash(payload),
    )


def _deduplicate_exact(
    records: tuple[ProvenanceRecord, ...],
) -> tuple[
    tuple[ProvenanceRecord, ...],
    DeduplicationReport,
    Mapping[str, str],
]:
    parent = list(range(len(records)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = root(left), root(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    by_content: dict[str, int] = {}
    by_key: dict[str, int] = {}
    for index, record in enumerate(records):
        for owners, identifier in (
            (by_content, record.content_hash),
            (by_key, record.deduplication_key),
        ):
            previous = owners.setdefault(identifier, index)
            union(index, previous)

    members: dict[int, list[ProvenanceRecord]] = {}
    for index, record in enumerate(records):
        members.setdefault(root(index), []).append(record)

    retained: list[ProvenanceRecord] = []
    removals: list[DuplicateRemoval] = []
    keeper_by_record: dict[str, str] = {}
    for component in members.values():
        ordered = sorted(component, key=lambda item: item.record_id)
        keeper = ordered[0]
        retained.append(keeper)
        for record in ordered:
            keeper_by_record[record.record_id] = keeper.record_id
        for record in ordered[1:]:
            reasons = sorted({
                reason
                for other in ordered
                for reason in (
                    (
                        f"content_hash:{record.content_hash}"
                        if record.content_hash == other.content_hash else None
                    ),
                    (
                        f"deduplication_key:{record.deduplication_key}"
                        if record.deduplication_key == other.deduplication_key else None
                    ),
                )
                if reason is not None
            })
            removals.append(DuplicateRemoval(
                record.record_id, keeper.record_id, "+".join(reasons)
            ))
    result = tuple(sorted(retained, key=lambda item: item.record_id))
    ordered_removals = tuple(sorted(removals, key=lambda item: item.dropped_record_id))
    return (
        result,
        _report("exact", len(records), result, ordered_removals),
        keeper_by_record,
    )


def _deduplicate_fuzzy(
    records: tuple[ProvenanceRecord, ...],
    original_records: tuple[ProvenanceRecord, ...],
    exact_keeper_by_record: Mapping[str, str],
) -> tuple[
    tuple[ProvenanceRecord, ...],
    DeduplicationReport,
    Mapping[str, str],
]:
    by_id = {record.record_id: record for record in records}
    cluster_keepers: dict[str, set[str]] = {}
    for record in original_records:
        cluster_keepers.setdefault(record.fuzzy_cluster_id, set()).add(
            exact_keeper_by_record[record.record_id]
        )

    parent = {record.record_id: record.record_id for record in records}

    def root(record_id: str) -> str:
        while parent[record_id] != record_id:
            parent[record_id] = parent[parent[record_id]]
            record_id = parent[record_id]
        return record_id

    def union(left: str, right: str) -> None:
        left_root, right_root = root(left), root(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for keeper_ids in cluster_keepers.values():
        ordered = sorted(keeper_ids)
        for keeper_id in ordered[1:]:
            union(ordered[0], keeper_id)

    components: dict[str, list[str]] = {}
    for record_id in parent:
        components.setdefault(root(record_id), []).append(record_id)

    retained: list[ProvenanceRecord] = []
    removals: list[DuplicateRemoval] = []
    final_keeper_by_exact_keeper: dict[str, str] = {}
    for member_ids in components.values():
        ordered_ids = sorted(member_ids)
        keeper_id = ordered_ids[0]
        retained.append(by_id[keeper_id])
        for record_id in ordered_ids:
            final_keeper_by_exact_keeper[record_id] = keeper_id
        for record_id in ordered_ids[1:]:
            shared_clusters = sorted(
                cluster_id
                for cluster_id, keeper_ids in cluster_keepers.items()
                if record_id in keeper_ids
                and any(other in keeper_ids for other in ordered_ids if other != record_id)
            )
            removals.append(DuplicateRemoval(
                record_id,
                keeper_id,
                "+".join(
                    f"explicit_fuzzy_cluster:{cluster_id}"
                    for cluster_id in shared_clusters
                ),
            ))
    result = tuple(sorted(retained, key=lambda item: item.record_id))
    ordered_removals = tuple(sorted(removals, key=lambda item: item.dropped_record_id))
    return (
        result,
        _report(
            "explicit_fuzzy_cluster", len(records), result, ordered_removals
        ),
        final_keeper_by_exact_keeper,
    )


def _groups(
    records: tuple[ProvenanceRecord, ...],
    *,
    retained_record_ids: frozenset[str] | None = None,
) -> tuple[_Group, ...]:
    parent = list(range(len(records)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = root(left), root(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    owner: dict[str, int] = {}
    for index, record in enumerate(records):
        keys = (
            *record.relation_keys(),
            f"exact_content:{record.content_hash}",
            f"exact_dedup:{record.deduplication_key}",
            f"explicit_fuzzy_cluster:{record.fuzzy_cluster_id}",
        )
        for key in keys:
            previous = owner.setdefault(key, index)
            union(index, previous)

    members: dict[int, list[ProvenanceRecord]] = {}
    for index, record in enumerate(records):
        if retained_record_ids is None or record.record_id in retained_record_ids:
            members.setdefault(root(index), []).append(record)
    groups = []
    for component_root, grouped_records in members.items():
        ordered = tuple(sorted(grouped_records, key=lambda item: item.record_id))
        group_id = _hash({
            "record_ids": tuple(item.record_id for item in ordered),
            "component_record_ids": tuple(sorted(
                record.record_id
                for index, record in enumerate(records)
                if root(index) == component_root
            )),
        })
        groups.append(_Group(group_id, ordered))
    return tuple(sorted(groups, key=lambda item: item.group_id))


def _groups_from_memberships(
    records: tuple[ProvenanceRecord, ...],
    memberships: tuple[tuple[str, tuple[str, ...]], ...],
) -> tuple[_Group, ...]:
    records_by_id = {record.record_id: record for record in records}
    assigned = [
        record_id
        for _, record_ids in memberships
        for record_id in record_ids
    ]
    if (
        len(assigned) != len(set(assigned))
        or set(assigned) != set(records_by_id)
    ):
        raise ValueError("stored group memberships must cover each record exactly once")
    return tuple(
        _Group(
            group_id,
            tuple(records_by_id[record_id] for record_id in record_ids),
        )
        for group_id, record_ids in memberships
    )


def _partition_groups(
    groups: tuple[_Group, ...], *, seed: int, phase_count: int
) -> tuple[PartitionRef, ...]:
    buckets: list[list[_Group]] = [[] for _ in range(phase_count)]
    token_totals = [0] * phase_count
    ordered = sorted(
        groups,
        key=lambda group: (
            _hash({"seed": seed, "group_id": group.group_id}),
            group.group_id,
        ),
    )
    for group in ordered:
        phase_index = min(
            range(phase_count),
            key=lambda index: (token_totals[index], index),
        )
        buckets[phase_index].append(group)
        token_totals[phase_index] += group.token_count

    partitions = []
    for phase_index, bucket in enumerate(buckets):
        bucket = sorted(bucket, key=lambda item: item.group_id)
        records = tuple(
            sorted(
                (record for group in bucket for record in group.records),
                key=lambda item: item.record_id,
            )
        )
        payload = {
            "phase_index": phase_index,
            "seed": seed,
            "group_ids": tuple(group.group_id for group in bucket),
            "records": tuple(
                (record.record_id, record.content_hash, record.token_count)
                for record in records
            ),
        }
        manifest_hash = _hash(payload)
        partitions.append(PartitionRef(
            phase_index=phase_index,
            partition_id=f"partition-{phase_index:02d}-{manifest_hash[:16]}",
            manifest_hash=manifest_hash,
            record_ids=tuple(record.record_id for record in records),
            content_hashes=tuple(record.content_hash for record in records),
            semantic_family_ids=tuple(sorted({
                record.semantic_family_id for record in records
            })),
            split_family_ids=tuple(sorted({
                record.split_family_id for record in records
            })),
            group_ids=tuple(group.group_id for group in bucket),
            token_count=sum(record.token_count for record in records),
            seed=seed,
        ))
    return tuple(partitions)


class CorpusService:
    """Pure, deterministic, fail-closed corpus freezing service."""

    def freeze(
        self,
        records: Iterable[ProvenanceRecord],
        policy: SplitPolicy,
        *,
        version: str = "1",
    ) -> CorpusManifest:
        if not isinstance(policy, SplitPolicy):
            raise TypeError("policy must be a SplitPolicy")
        _require_text("version", version)
        materialized = tuple(records)
        if not materialized:
            raise ValueError("cannot freeze an empty corpus")
        if any(not isinstance(record, ProvenanceRecord) for record in materialized):
            raise TypeError("all corpus records must be ProvenanceRecord values")
        ordered = tuple(sorted(materialized, key=lambda item: item.record_id))
        if len({record.record_id for record in ordered}) != len(ordered):
            raise ValueError("record_id values must be unique")

        exclusions = []
        isolated_records = []
        for record in ordered:
            for denylist in (policy.frozen_evaluation, policy.public_benchmark):
                reasons = denylist.reasons(record)
                if reasons:
                    exclusions.append(IsolationExclusion(
                        record.record_id, denylist.denylist_id, reasons
                    ))
            if not any(item.record_id == record.record_id for item in exclusions):
                isolated_records.append(record)
        isolation_payload = {
            "checked_record_count": len(ordered),
            "exclusions": tuple(
                (item.record_id, item.denylist_id, item.reasons)
                for item in exclusions
            ),
        }
        isolation_report = IsolationReport(
            isolated=True,
            checked_record_count=len(ordered),
            exclusions=tuple(exclusions),
            report_hash=_hash(isolation_payload),
        )
        if not isolated_records:
            raise ValueError("deny lists excluded every corpus record")

        isolated_tuple = tuple(isolated_records)
        exact_records, exact_report, exact_keepers = _deduplicate_exact(
            isolated_tuple
        )
        kept_records, fuzzy_report, fuzzy_keepers = _deduplicate_fuzzy(
            exact_records, isolated_tuple, exact_keepers
        )
        if not kept_records:
            raise ValueError("deduplication removed every corpus record")

        total_tokens = sum(record.token_count for record in kept_records)
        targets = dict(policy.target_allocation)
        actual_slices = {record.capability_slice for record in kept_records}
        unknown = actual_slices - set(targets)
        if unknown:
            raise ValueError(
                f"capability slices have no registered target allocation: {sorted(unknown)}"
            )
        allocations = []
        for capability_slice, target in policy.target_allocation:
            tokens = sum(
                record.token_count for record in kept_records
                if record.capability_slice == capability_slice
            )
            actual = tokens / total_tokens
            if abs(actual - target) > policy.allocation_tolerance:
                raise ValueError(
                    f"token allocation for {capability_slice!r} is {actual:.12g}; "
                    f"target {target:.12g} exceeds tolerance "
                    f"{policy.allocation_tolerance:.12g}"
                )
            allocations.append(TokenAllocation(
                capability_slice, target, tokens, actual
            ))

        support = tuple(
            (
                capability_slice,
                sum(
                    record.trustworthy_release_support
                    for record in kept_records
                    if record.capability_slice == capability_slice
                ),
            )
            for capability_slice in policy.release_evidence_slices
        )
        unsupported = [name for name, count in support if count == 0]
        if unsupported:
            raise ValueError(
                "release capability slices lack permitted non-synthetic "
                f"human/executable support: {unsupported}"
            )

        grouped = _groups(
            isolated_tuple,
            retained_record_ids=frozenset(
                fuzzy_keepers[exact_keepers[record.record_id]]
                for record in isolated_tuple
            ),
        )
        if {
            record.record_id for group in grouped for record in group.records
        } != {record.record_id for record in kept_records}:
            raise ValueError("deduplication/group closure produced inconsistent keepers")
        partitions = _partition_groups(
            grouped, seed=policy.seed, phase_count=policy.phase_count
        )
        records_ref = _hash(tuple(_record_evidence(record) for record in kept_records))
        split_policy_hash = _hash(_policy_evidence(policy))
        identity_payload = {
            "version": version,
            "records_ref": records_ref,
            "split_policy_hash": split_policy_hash,
            "exact_dedup_report": exact_report.report_hash,
            "fuzzy_dedup_report": fuzzy_report.report_hash,
            "isolation_report": isolation_report.report_hash,
            "partition_hashes": tuple(
                partition.manifest_hash for partition in partitions
            ),
        }
        corpus_id = _hash(identity_payload)
        return CorpusManifest(
            corpus_id=corpus_id,
            version=version,
            records_ref=records_ref,
            records=kept_records,
            allocation_by_training_tokens=tuple(allocations),
            exact_dedup_report=exact_report,
            fuzzy_dedup_report=fuzzy_report,
            isolation_report=isolation_report,
            split_policy_hash=split_policy_hash,
            frozen_eval_denylist_hash=policy.frozen_evaluation.manifest_hash,
            public_benchmark_denylist_hash=policy.public_benchmark.manifest_hash,
            partition_refs=partitions,
            total_tokens=total_tokens,
            provenance_complete=True,
            release_slice_support=support,
            group_memberships=tuple(
                (
                    group.group_id,
                    tuple(record.record_id for record in group.records),
                )
                for group in grouped
            ),
        )

    def progressive_partitions(
        self,
        corpus: CorpusManifest,
        phases: int = PHASE_COUNT,
        seed: int | None = None,
    ) -> tuple[PartitionRef, ...]:
        if not isinstance(corpus, CorpusManifest):
            raise TypeError("corpus must be a CorpusManifest")
        if phases != PHASE_COUNT:
            raise ValueError("progressive partitioning requires exactly 20 phases")
        selected_seed = (
            corpus.partition_refs[0].seed if seed is None else seed
        )
        if isinstance(selected_seed, bool) or not isinstance(selected_seed, int):
            raise TypeError("seed must be an integer")
        groups = (
            _groups_from_memberships(corpus.records, corpus.group_memberships)
            if corpus.group_memberships
            else _groups(corpus.records)
        )
        return _partition_groups(groups, seed=selected_seed, phase_count=phases)

    def assert_isolated(
        self, corpus: CorpusManifest, frozen_eval: CorpusDenyList
    ) -> IsolationReport:
        if not isinstance(corpus, CorpusManifest):
            raise TypeError("corpus must be a CorpusManifest")
        if not isinstance(frozen_eval, CorpusDenyList):
            raise TypeError("frozen_eval must be a CorpusDenyList")
        exclusions = tuple(
            IsolationExclusion(
                record.record_id, frozen_eval.denylist_id, frozen_eval.reasons(record)
            )
            for record in corpus.records
            if frozen_eval.reasons(record)
        )
        payload = {
            "checked_record_count": len(corpus.records),
            "denylist_hash": frozen_eval.manifest_hash,
            "exclusions": tuple(
                (item.record_id, item.denylist_id, item.reasons)
                for item in exclusions
            ),
        }
        return IsolationReport(
            isolated=not exclusions,
            checked_record_count=len(corpus.records),
            exclusions=exclusions,
            report_hash=_hash(payload),
        )
