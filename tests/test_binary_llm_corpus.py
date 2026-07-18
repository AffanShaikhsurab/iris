from __future__ import annotations

from dataclasses import replace

import pytest

from binary_llm.domain import canonical_json_bytes, sha256_bytes
from binary_llm.orchestration.corpus import (
    CorpusDenyList,
    CorpusService,
    ProvenanceRecord,
    SplitPolicy,
)


def _record(
    record_id: str,
    *,
    tokens: int = 10,
    capability_slice: str = "broad_text",
    content: str | None = None,
    semantic: str | None = None,
    split: str | None = None,
    dedup: str | None = None,
    fuzzy: str | None = None,
    source_document: str | None = None,
    tool_template: str | None = None,
    synthetic_sibling: str | None = None,
    synthetic: bool = False,
    verified: bool = True,
) -> ProvenanceRecord:
    normalized = {
        "messages": [{"role": "user", "content": content or record_id}],
        "metadata": {"record_id": record_id},
    }
    return ProvenanceRecord(
        record_id=record_id,
        content_hash=sha256_bytes(canonical_json_bytes(normalized)),
        source=f"source:{record_id}",
        license_or_terms="test-license",
        permitted_use="research-training",
        transformation_history=("normalized-v1",),
        semantic_family_id=semantic or f"semantic:{record_id}",
        split_family_id=split or f"split:{record_id}",
        deduplication_key=dedup or f"dedup:{record_id}",
        fuzzy_cluster_id=fuzzy or f"fuzzy:{record_id}",
        target_type="causal_lm",
        capability_slice=capability_slice,
        token_count=tokens,
        synthetic=synthetic,
        executable_or_human_verified=verified,
        normalized_record=normalized,
        source_document_id=source_document,
        tool_template_id=tool_template,
        synthetic_sibling_id=synthetic_sibling,
    )


def _policy(
    *,
    seed: int = 17,
    allocation=(("broad_text", 1.0),),
    tolerance: float = 0.0,
    release_slices: tuple[str, ...] = (),
    frozen: CorpusDenyList | None = None,
    public: CorpusDenyList | None = None,
) -> SplitPolicy:
    kwargs = {}
    if frozen is not None:
        kwargs["frozen_evaluation"] = frozen
    if public is not None:
        kwargs["public_benchmark"] = public
    return SplitPolicy(
        "test-policy",
        seed,
        target_allocation=allocation,
        allocation_tolerance=tolerance,
        release_evidence_slices=release_slices,
        **kwargs,
    )


def _partition_for(manifest, record_id: str) -> int:
    return next(
        partition.phase_index
        for partition in manifest.partition_refs
        if record_id in partition.record_ids
    )


def test_freeze_is_deterministic_content_addressed_and_defensively_immutable() -> None:
    mutable_content = {"messages": [{"role": "user", "content": "original"}]}
    record = ProvenanceRecord(
        record_id="record",
        content_hash=sha256_bytes(canonical_json_bytes(mutable_content)),
        source="source",
        license_or_terms="terms",
        permitted_use="training",
        transformation_history=("normalized",),
        semantic_family_id="semantic",
        split_family_id="split",
        deduplication_key="dedup",
        fuzzy_cluster_id="fuzzy",
        target_type="causal_lm",
        capability_slice="broad_text",
        token_count=10,
        synthetic=False,
        executable_or_human_verified=True,
        normalized_record=mutable_content,
    )
    mutable_content["messages"][0]["content"] = "mutated"
    service = CorpusService()
    first = service.freeze([record], _policy())
    second = service.freeze(iter([record]), _policy())

    assert first == second
    assert first.corpus_id == second.corpus_id
    assert first.records_ref == second.records_ref
    assert first.records[0].normalized_record["messages"][0]["content"] == "original"
    with pytest.raises(TypeError):
        first.records[0].normalized_record["new"] = "value"


def test_exact_and_explicit_fuzzy_dedup_preserve_auditable_decisions() -> None:
    exact_first = _record("a", dedup="shared-dedup")
    exact_second = _record("b", dedup="shared-dedup")
    fuzzy_first = _record("c", fuzzy="shared-fuzzy")
    fuzzy_second = _record("d", fuzzy="shared-fuzzy")

    manifest = CorpusService().freeze(
        [fuzzy_second, exact_second, fuzzy_first, exact_first], _policy()
    )

    assert tuple(record.record_id for record in manifest.records) == ("a", "c")
    assert manifest.exact_dedup_report.removals[0].dropped_record_id == "b"
    assert manifest.exact_dedup_report.removals[0].retained_record_id == "a"
    assert "deduplication_key:shared-dedup" in (
        manifest.exact_dedup_report.removals[0].reason
    )
    assert manifest.fuzzy_dedup_report.removals[0].dropped_record_id == "d"
    assert manifest.fuzzy_dedup_report.removals[0].reason == (
        "explicit_fuzzy_cluster:shared-fuzzy"
    )


def test_exact_dedup_uses_transitive_components_across_both_identifiers() -> None:
    first = _record("a", dedup="dedup-a")
    bridge = replace(first, record_id="b", deduplication_key="dedup-bc")
    last = _record("c", dedup="dedup-bc")

    manifest = CorpusService().freeze([last, bridge, first], _policy())

    assert tuple(record.record_id for record in manifest.records) == ("a",)
    assert {
        removal.dropped_record_id: removal.retained_record_id
        for removal in manifest.exact_dedup_report.removals
    } == {"b": "a", "c": "a"}
    assert "content_hash:" in manifest.exact_dedup_report.removals[0].reason
    assert "deduplication_key:dedup-bc" in (
        manifest.exact_dedup_report.removals[1].reason
    )


@pytest.mark.parametrize("duplicate_kind", ("exact", "fuzzy"))
def test_dropped_duplicate_bridge_preserves_partition_relation_closure(
    duplicate_kind: str,
) -> None:
    left = _record("a-left", source_document="left-document")
    if duplicate_kind == "exact":
        bridge = replace(
            left,
            record_id="b-bridge",
            source_document_id=None,
            tool_template_id="right-template",
        )
    else:
        bridge = _record(
            "b-bridge",
            fuzzy=left.fuzzy_cluster_id,
            tool_template="right-template",
        )
    right = _record("c-right", tool_template="right-template")

    service = CorpusService()
    manifest = service.freeze([right, bridge, left], _policy())
    repartitioned = service.progressive_partitions(manifest, seed=manifest.partition_refs[0].seed)

    assert tuple(record.record_id for record in manifest.records) == (
        "a-left",
        "c-right",
    )
    assert _partition_for(manifest, "a-left") == _partition_for(
        manifest, "c-right"
    )
    repartition_by_record = {
        record_id: partition.phase_index
        for partition in repartitioned
        for record_id in partition.record_ids
    }
    assert repartition_by_record["a-left"] == repartition_by_record["c-right"]


def test_connected_relations_never_cross_partitions() -> None:
    records = (
        _record("a", source_document="document"),
        _record("b", source_document="document", tool_template="template"),
        _record("c", tool_template="template", synthetic_sibling="siblings"),
        _record("d", synthetic_sibling="siblings"),
        _record("e", semantic="semantic:e", split="shared-split"),
        _record("f", semantic="semantic:f", split="shared-split"),
    )

    manifest = CorpusService().freeze(records, _policy())

    assert len({_partition_for(manifest, item) for item in ("a", "b", "c", "d")}) == 1
    assert _partition_for(manifest, "e") == _partition_for(manifest, "f")


def test_deny_lists_remove_exact_and_semantic_sibling_leakage_before_freeze() -> None:
    exact = _record("exact")
    semantic = _record("semantic", semantic="blocked-family")
    public = _record("public", fuzzy="public-cluster")
    safe = _record("safe")
    frozen = CorpusDenyList(
        "frozen",
        content_hashes=(exact.content_hash,),
        semantic_family_ids=("blocked-family",),
    )
    benchmark = CorpusDenyList(
        "benchmark", fuzzy_cluster_ids=("public-cluster",)
    )

    manifest = CorpusService().freeze(
        [exact, semantic, public, safe],
        _policy(frozen=frozen, public=benchmark),
    )

    assert tuple(record.record_id for record in manifest.records) == ("safe",)
    assert {
        (item.record_id, item.denylist_id)
        for item in manifest.isolation_report.exclusions
    } == {
        ("exact", "frozen"),
        ("semantic", "frozen"),
        ("public", "benchmark"),
    }
    assert CorpusService().assert_isolated(manifest, frozen).isolated


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("source", "", "source"),
        ("license_or_terms", "", "license_or_terms"),
        ("permitted_use", "", "permitted_use"),
        ("semantic_family_id", "", "semantic_family_id"),
        ("split_family_id", "", "split_family_id"),
        ("deduplication_key", "", "deduplication_key"),
        ("capability_slice", "", "capability_slice"),
        ("target_type", "", "target_type"),
        ("token_count", 0, "positive integer"),
        ("token_count", 1.5, "positive integer"),
    ),
)
def test_incomplete_or_invalid_provenance_fails_closed(field, value, message) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        replace(_record("record"), **{field: value})


def test_release_slices_require_non_synthetic_verified_support() -> None:
    unsupported = _record("synthetic", synthetic=True, verified=True)
    with pytest.raises(ValueError, match="lack permitted non-synthetic"):
        CorpusService().freeze(
            [unsupported], _policy(release_slices=("broad_text",))
        )

    supported = _record("executable", synthetic=False, verified=True)
    manifest = CorpusService().freeze(
        [unsupported, supported],
        _policy(release_slices=("broad_text",)),
    )
    assert manifest.release_slice_support == (("broad_text", 1),)
    preserved = {record.record_id: record for record in manifest.records}
    assert preserved["synthetic"].synthetic
    assert preserved["synthetic"].executable_or_human_verified


def test_allocation_uses_actual_tokens_not_row_counts_and_honors_tolerance() -> None:
    policy = _policy(
        allocation=(("broad_text", 0.75), ("math", 0.25)),
        tolerance=0.0,
    )
    records = (
        _record("broad", tokens=30),
        _record("math-1", tokens=5, capability_slice="math"),
        _record("math-2", tokens=5, capability_slice="math"),
    )
    manifest = CorpusService().freeze(records, policy)
    assert {
        item.capability_slice: item.training_tokens
        for item in manifest.allocation_by_training_tokens
    } == {"broad_text": 30, "math": 10}

    row_balanced = (
        _record("broad-equal", tokens=10),
        _record("math-equal", tokens=10, capability_slice="math"),
    )
    with pytest.raises(ValueError, match="token allocation"):
        CorpusService().freeze(row_balanced, policy)


def test_progressive_partitions_cover_once_are_disjoint_and_keep_groups_whole() -> None:
    records = tuple(
        _record(
            f"record-{index:02d}",
            semantic=f"semantic-{index // 2}",
            split=f"split-{index // 2}",
        )
        for index in range(42)
    )
    service = CorpusService()
    manifest = service.freeze(reversed(records), _policy(seed=101))
    partitions = service.progressive_partitions(manifest, phases=20, seed=101)

    assert len(partitions) == 20
    all_ids = [
        record_id for partition in partitions for record_id in partition.record_ids
    ]
    assert len(all_ids) == len(set(all_ids)) == len(records)
    assert set(all_ids) == {record.record_id for record in records}
    assert len({partition.manifest_hash for partition in partitions}) == 20
    for index in range(0, len(records), 2):
        assert _partition_for(manifest, f"record-{index:02d}") == _partition_for(
            manifest, f"record-{index + 1:02d}"
        )
