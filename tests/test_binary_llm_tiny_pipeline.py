from __future__ import annotations

from dataclasses import replace

import pytest

from binary_llm.domain import ArtifactRef, ArtifactStoreError, canonical_json_bytes, sha256_bytes
from binary_llm.orchestration import (
    BudgetUsage,
    FilesystemArtifactStore,
    TinyPipelineOrchestrator,
    TinyPipelineReportResolver,
)


def _put(store: FilesystemArtifactStore, artifact_id: str, payload: bytes) -> ArtifactRef:
    reference = ArtifactRef(
        artifact_id=artifact_id,
        kind="test_evidence",
        sha256=sha256_bytes(payload),
        bytes=len(payload),
        media_type="application/octet-stream",
    )
    store.put_bytes(reference, payload)
    return reference


def _report(store: FilesystemArtifactStore, refs: tuple[ArtifactRef, ...]) -> ArtifactRef:
    payload = canonical_json_bytes(
        {
            "schema_version": 1,
            "artifact_kind": "tiny_vertical_slice_report",
            "scientific_reproduction": False,
            "promotion_eligible": False,
            "artifact_refs": [item.to_dict() for item in refs],
        }
    )
    reference = ArtifactRef(
        artifact_id=f"tiny_vertical_slice_report:{sha256_bytes(payload)}",
        kind="tiny_vertical_slice_report",
        sha256=sha256_bytes(payload),
        bytes=len(payload),
        media_type="application/vnd.binary-llm.tiny-vertical-slice.v1+json",
    )
    store.put_bytes(reference, payload)
    return reference


def test_report_identity_and_fresh_root_resolution_are_deterministic(tmp_path) -> None:
    report_ids = []
    for name in ("one", "two"):
        store = FilesystemArtifactStore(tmp_path / name)
        evidence = _put(store, "evidence", b"same exact evidence")
        report = _report(store, (evidence,))
        resolved = TinyPipelineReportResolver(
            FilesystemArtifactStore(tmp_path / name)
        ).resolve(report)
        assert resolved["scientific_reproduction"] is False
        assert resolved["promotion_eligible"] is False
        report_ids.append(report.artifact_id)
    assert report_ids[0] == report_ids[1]


def test_report_resolver_fails_closed_with_exact_corrupt_artifact_id(tmp_path) -> None:
    store = FilesystemArtifactStore(tmp_path)
    evidence = _put(store, "affected-evidence", b"evidence")
    report = _report(store, (evidence,))
    store.object_path(evidence.sha256).write_bytes(b"corrupt")

    with pytest.raises(ArtifactStoreError) as captured:
        TinyPipelineReportResolver(FilesystemArtifactStore(tmp_path)).resolve(report)

    assert captured.value.code == "artifact_store.payload_corrupt"
    assert captured.value.affected_ids == {"artifact_ids": ("affected-evidence",)}


def test_budget_resume_delta_prevents_repeated_usage_records() -> None:
    prior = BudgetUsage(
        consumed_tokens=12,
        optimizer_steps=2,
        wall_seconds=1.0,
        accelerator_seconds=0.5,
        billable_cost=0.25,
        interrupted_attempts=1,
    )
    total = replace(
        prior,
        consumed_tokens=18,
        optimizer_steps=3,
        wall_seconds=1.5,
        accelerator_seconds=0.75,
        billable_cost=0.4,
    )
    assert TinyPipelineOrchestrator._usage_delta(total, prior) == BudgetUsage(
        consumed_tokens=6,
        optimizer_steps=1,
        wall_seconds=0.5,
        accelerator_seconds=0.25,
        billable_cost=0.15000000000000002,
        interrupted_attempts=0,
    )


def test_budget_delta_rejects_usage_rollback() -> None:
    with pytest.raises(ValueError, match="cannot precede"):
        TinyPipelineOrchestrator._usage_delta(
            BudgetUsage(consumed_tokens=1),
            BudgetUsage(consumed_tokens=2),
        )
