from __future__ import annotations

from dataclasses import replace

import pytest

from binary_llm.domain import ArtifactRef, ArtifactStoreError, canonical_json_bytes, sha256_bytes
from binary_llm.orchestration import FilesystemArtifactStore, RecoveryIssueKind


def _artifact(
    artifact_id: str,
    payload: bytes,
    *,
    kind: str = "evidence",
    parent: str | None = None,
    run_id: str | None = None,
    attempt_id: str | None = None,
) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=artifact_id,
        kind=kind,
        sha256=sha256_bytes(payload),
        bytes=len(payload),
        media_type="application/x-test",
        parent_artifact_id=parent,
        producing_run_id=run_id,
        producing_attempt_id=attempt_id,
    )


@pytest.mark.parametrize("payload", [b"durable evidence", b""])
def test_put_resolve_round_trip_and_deterministic_locations(tmp_path, payload):
    store = FilesystemArtifactStore(tmp_path)
    artifact = _artifact("artifact-1", payload, kind="metrics")

    assert store.put_bytes(artifact, payload) == artifact
    assert store.resolve(artifact.artifact_id) == payload
    assert store.artifact_ref(artifact.artifact_id) == artifact
    assert store.object_path(artifact.sha256) == (
        tmp_path / "objects" / artifact.sha256[:2] / artifact.sha256[2:4] / artifact.sha256
    )
    reference_key = sha256_bytes(artifact.artifact_id.encode())
    assert store.reference_path(artifact.artifact_id) == (
        tmp_path / "references" / reference_key[:2] / reference_key[2:4] / f"{reference_key}.json"
    )


def test_duplicate_write_is_idempotent_and_fresh_instance_resolves(tmp_path):
    payload = b"same bytes"
    artifact = _artifact("artifact-reopen", payload)
    first = FilesystemArtifactStore(tmp_path)

    first.put_bytes(artifact, payload)
    object_stat = first.object_path(artifact.sha256).stat()
    reference_stat = first.reference_path(artifact.artifact_id).stat()
    first.put_bytes(artifact, payload)

    assert first.object_path(artifact.sha256).stat().st_mtime_ns == object_stat.st_mtime_ns
    assert first.reference_path(artifact.artifact_id).stat().st_mtime_ns == reference_stat.st_mtime_ns
    assert FilesystemArtifactStore(tmp_path).resolve(artifact.artifact_id) == payload


def test_payload_is_checked_before_any_publication(tmp_path):
    store = FilesystemArtifactStore(tmp_path)
    payload = b"actual"

    wrong_hash = replace(_artifact("wrong-hash", payload), sha256="0" * 64)
    with pytest.raises(ArtifactStoreError) as raised:
        store.put_bytes(wrong_hash, payload)
    assert raised.value.code == "artifact_store.payload_digest_mismatch"
    assert not store.object_path(wrong_hash.sha256).exists()
    assert not store.reference_path(wrong_hash.artifact_id).exists()

    wrong_size = replace(_artifact("wrong-size", payload), bytes=len(payload) + 1)
    with pytest.raises(ArtifactStoreError) as raised:
        store.put_bytes(wrong_size, payload)
    assert raised.value.code == "artifact_store.payload_size_mismatch"
    assert not store.object_path(wrong_size.sha256).exists()
    assert not store.reference_path(wrong_size.artifact_id).exists()


def test_same_id_changed_metadata_fails_closed(tmp_path):
    store = FilesystemArtifactStore(tmp_path)
    payload = b"content"
    artifact = _artifact("fixed-id", payload)
    store.put_bytes(artifact, payload)

    with pytest.raises(ArtifactStoreError) as raised:
        store.put_bytes(replace(artifact, media_type="text/plain"), payload)
    assert raised.value.code == "artifact_store.reference_conflict"
    assert store.artifact_ref(artifact.artifact_id) == artifact


def test_same_payload_is_deduplicated_across_artifact_ids(tmp_path):
    store = FilesystemArtifactStore(tmp_path)
    payload = b"shared content"
    first = _artifact("artifact-a", payload)
    second = _artifact("artifact-b", payload, parent="artifact-a")

    store.put_bytes(first, payload)
    store.put_bytes(second, payload)

    assert store.object_path(first.sha256) == store.object_path(second.sha256)
    assert len(tuple((tmp_path / "objects").rglob(first.sha256))) == 1
    assert store.resolve(first.artifact_id) == store.resolve(second.artifact_id) == payload


def test_payload_corruption_is_detected_as_typed_failure(tmp_path):
    store = FilesystemArtifactStore(tmp_path)
    payload = b"protected"
    artifact = _artifact("artifact-corrupt", payload)
    store.put_bytes(artifact, payload)
    store.object_path(artifact.sha256).write_bytes(b"tampered")

    with pytest.raises(ArtifactStoreError) as raised:
        store.resolve(artifact.artifact_id)
    assert raised.value.code == "artifact_store.payload_corrupt"


@pytest.mark.parametrize(
    "mutator",
    [
        lambda artifact: canonical_json_bytes({**artifact.to_dict(), "kind": "tampered"}),
        lambda artifact: canonical_json_bytes(artifact) + b"\n",
        lambda artifact: canonical_json_bytes(
            {key: value for key, value in artifact.to_dict().items() if key != "media_type"}
        ),
        lambda artifact: b"{not-json",
    ],
)
def test_reference_tampering_and_noncanonical_records_fail_typed(tmp_path, mutator):
    store = FilesystemArtifactStore(tmp_path)
    payload = b"protected"
    artifact = _artifact("artifact-metadata", payload)
    store.put_bytes(artifact, payload)
    store.reference_path(artifact.artifact_id).write_bytes(mutator(artifact))

    with pytest.raises(ArtifactStoreError) as raised:
        store.resolve(artifact.artifact_id, expected=artifact)
    assert raised.value.code in {
        "artifact_store.invalid_reference",
        "artifact_store.expected_reference_mismatch",
    }


def test_exact_expected_reference_checks_owner_and_lineage(tmp_path):
    store = FilesystemArtifactStore(tmp_path)
    payload = b"owned"
    artifact = _artifact(
        "artifact-owned",
        payload,
        parent="parent-a",
        run_id="run-a",
        attempt_id="attempt-a",
    )
    store.put_bytes(artifact, payload)
    assert store.resolve(artifact.artifact_id, expected=artifact) == payload

    for mismatch in (
        replace(artifact, parent_artifact_id="parent-b"),
        replace(artifact, producing_run_id="run-b"),
        replace(artifact, producing_attempt_id="attempt-b"),
    ):
        with pytest.raises(ArtifactStoreError) as raised:
            store.resolve(artifact.artifact_id, expected=mismatch)
        assert raised.value.code == "artifact_store.expected_reference_mismatch"


@pytest.mark.parametrize("artifact_id", ["../escape", r"..\escape", "/absolute", "", "has space"])
def test_unsafe_artifact_ids_are_rejected_before_filesystem_access(tmp_path, artifact_id):
    store = FilesystemArtifactStore(tmp_path)
    with pytest.raises(ArtifactStoreError) as raised:
        store.reference_path(artifact_id)
    assert raised.value.code == "artifact_store.invalid_artifact_id"
    assert not tuple(tmp_path.iterdir())


@pytest.mark.parametrize("digest", ["a" * 63, "A" * 64, "g" * 64, "../" + "a" * 61])
def test_malformed_digests_are_rejected_before_filesystem_access(tmp_path, digest):
    store = FilesystemArtifactStore(tmp_path)
    with pytest.raises(ArtifactStoreError) as raised:
        store.object_path(digest)
    assert raised.value.code == "artifact_store.invalid_digest"
    assert not tuple(tmp_path.iterdir())


def test_failed_atomic_replace_publishes_nothing_and_retry_succeeds(
    tmp_path, monkeypatch
):
    store = FilesystemArtifactStore(tmp_path)
    payload = b"retryable"
    artifact = _artifact("artifact-retry", payload)
    real_replace = __import__("os").replace

    def fail_replace(source, destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr("binary_llm.orchestration.store.os.replace", fail_replace)
    with pytest.raises(ArtifactStoreError) as raised:
        store.put_bytes(artifact, payload)
    assert raised.value.code == "artifact_store.write_failed"
    assert not store.object_path(artifact.sha256).exists()
    assert not store.reference_path(artifact.artifact_id).exists()

    monkeypatch.setattr("binary_llm.orchestration.store.os.replace", real_replace)
    store.put_bytes(artifact, payload)
    assert store.resolve(artifact.artifact_id) == payload


def test_recovery_scan_reports_and_quarantines_stale_temp_and_orphan_object(tmp_path):
    store = FilesystemArtifactStore(tmp_path, stale_after_seconds=0)
    temp = tmp_path / "objects" / ".interrupted.tmp"
    temp.parent.mkdir(parents=True)
    temp.write_bytes(b"partial")
    orphan_digest = sha256_bytes(b"orphan")
    orphan = store.object_path(orphan_digest)
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(b"orphan")

    report = store.scan_recovery()
    assert {(item.kind, item.relative_path) for item in report.issues} == {
        (RecoveryIssueKind.STALE_TEMP, temp.relative_to(tmp_path).as_posix()),
        (RecoveryIssueKind.ORPHAN_OBJECT, orphan.relative_to(tmp_path).as_posix()),
    }
    moved = store.quarantine_recoverable(report)
    assert moved == tuple(sorted(moved))
    assert not temp.exists()
    assert not orphan.exists()
    assert store.scan_recovery().issues == ()


def test_recovery_reports_orphan_reference_without_silent_deletion(tmp_path):
    store = FilesystemArtifactStore(tmp_path)
    payload = b"referenced"
    artifact = _artifact("orphan-reference", payload)
    store.put_bytes(artifact, payload)
    store.object_path(artifact.sha256).unlink()

    report = store.scan_recovery()
    assert [item.kind for item in report.issues] == [RecoveryIssueKind.ORPHAN_REFERENCE]
    assert store.reference_path(artifact.artifact_id).exists()
    store.quarantine_recoverable(report)
    assert not store.reference_path(artifact.artifact_id).exists()


def test_two_store_instances_contend_via_exclusive_lock(tmp_path):
    first = FilesystemArtifactStore(tmp_path)
    second = FilesystemArtifactStore(tmp_path)
    payload = b"contention"
    artifact = _artifact("contended", payload)
    first._writer_lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = __import__("os").open(
        first._writer_lock_path,
        __import__("os").O_CREAT | __import__("os").O_EXCL | __import__("os").O_WRONLY,
    )
    try:
        with pytest.raises(ArtifactStoreError) as raised:
            second.put_bytes(artifact, payload)
        assert raised.value.code == "artifact_store.writer_busy"
    finally:
        __import__("os").close(descriptor)
        first._writer_lock_path.unlink()
    assert second.put_bytes(artifact, payload) == artifact


def test_store_stale_lock_is_reported_and_requires_explicit_quarantine(tmp_path):
    store = FilesystemArtifactStore(tmp_path, stale_after_seconds=0)
    store._writer_lock_path.parent.mkdir(parents=True, exist_ok=True)
    store._writer_lock_path.write_bytes(b"stale")

    report = store.scan_recovery()
    assert [item.kind for item in report.issues] == [RecoveryIssueKind.STALE_LOCK]
    payload = b"blocked"
    with pytest.raises(ArtifactStoreError) as raised:
        store.put_bytes(_artifact("blocked", payload), payload)
    assert raised.value.code == "artifact_store.stale_lock"
    assert store.quarantine_recoverable(report) == ("writer.lock",)
    assert store.put_bytes(_artifact("unblocked", payload), payload).artifact_id == "unblocked"
