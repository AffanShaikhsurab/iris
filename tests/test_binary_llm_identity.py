from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from binary_llm.domain import (
    CanonicalDocument,
    CanonicalSerializationError,
    ExportError,
    Retryability,
    canonical_json,
    canonical_json_bytes,
    evolve_document,
    identify_content,
    new_unique_id,
    parse_canonical_json,
)


def test_canonical_json_is_stable_across_key_order_and_normalizes_utc():
    instant = datetime(2026, 7, 16, 12, 30, tzinfo=timezone(timedelta(hours=2)))
    left = {"schema_version": 1, "nested": {"z": 2, "a": 1}, "at": instant}
    right = {"at": instant, "nested": {"a": 1, "z": 2}, "schema_version": 1}

    expected = (
        '{"at":"2026-07-16T10:30:00.000000Z","nested":{"a":1,"z":2},'
        '"schema_version":1}'
    )
    assert canonical_json(left) == expected
    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert parse_canonical_json(expected.encode("utf-8"))["schema_version"] == 1


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), {"not", "json"}, {1: "non-string-key"}],
)
def test_canonical_json_rejects_values_without_stable_json_meaning(value):
    with pytest.raises(CanonicalSerializationError):
        canonical_json_bytes({"schema_version": 1, "value": value})


def test_content_identity_covers_all_representation_and_export_fields():
    record = {"schema_version": 1, "candidate": "candidate-a"}
    representation = {
        "body": "binary",
        "offsets": False,
        "exception_tensors": [],
        "scale_dtype": "float16",
        "embedding_dtype": "q4",
        "head_dtype": "q8",
        "vocabulary_size": 130_560,
        "format_version": "1",
        "exporter_revision": "abc123",
        "runtime_revision": "def456",
        "build_flags": ["scalar-reference"],
    }

    baseline = identify_content(record, kind="candidate", representation_fields=representation)
    reordered = identify_content(
        dict(reversed(list(record.items()))),
        kind="candidate",
        representation_fields=dict(reversed(list(representation.items()))),
    )
    changed = identify_content(
        record,
        kind="candidate",
        representation_fields={**representation, "exception_tensors": ["lm_head.weight"]},
    )

    assert baseline.value.startswith("candidate:sha256:")
    assert baseline == reordered
    assert baseline.sha256 != changed.sha256


def test_schema_evolution_preserves_old_bytes_and_creates_new_identity():
    representation = {"body": "binary", "scale_dtype": "float16"}
    old = CanonicalDocument.create(
        {"schema_version": 1, "artifact_id": "artifact-a", "value": 7},
        kind="artifact",
        representation_fields=representation,
    )
    old_bytes = old.record_bytes
    old_identity = old.content_id

    evolution = evolve_document(
        old,
        {
            "schema_version": 2,
            "artifact_id": "artifact-a",
            "value": 7,
            "new_required_field": "explicit",
        },
    )

    assert evolution.previous is old
    assert evolution.previous.record_bytes == old_bytes
    assert evolution.previous.content_id == old_identity
    assert evolution.current.schema_version == 2
    assert evolution.current.content_id != old_identity
    assert evolution.current.representation_bytes == old.representation_bytes


def test_schema_evolution_requires_an_explicit_new_version():
    old = CanonicalDocument.create(
        {"schema_version": "1", "artifact_id": "artifact-a"},
        kind="artifact",
    )

    with pytest.raises(ValueError, match="must change schema_version"):
        evolve_document(old, {"schema_version": "1", "artifact_id": "artifact-a"})


def test_typed_failure_has_stable_retryability_and_affected_ids():
    failure = ExportError(
        "decoded signs differ",
        retryability=Retryability.AFTER_REMEDIATION,
        code="export.sign_decode_mismatch",
        affected_ids={
            "run_ids": ["run-b", "run-a", "run-a"],
            "artifact_ids": ["artifact-a"],
        },
        context={"tensor": "layers.0.mlp.down_proj.weight", "mismatch_count": 3},
    )

    assert failure.to_dict() == {
        "schema_version": 1,
        "error_type": "ExportError",
        "code": "export.sign_decode_mismatch",
        "message": "decoded signs differ",
        "retryability": "after_remediation",
        "affected_ids": {
            "artifact_ids": ["artifact-a"],
            "run_ids": ["run-a", "run-b"],
        },
        "context": {
            "mismatch_count": 3,
            "tensor": "layers.0.mlp.down_proj.weight",
        },
    }
    assert failure.to_json_bytes() == canonical_json_bytes(failure.to_dict())


def test_typed_failure_rejects_unstable_or_wrong_category_codes():
    with pytest.raises(ValueError, match="start with 'export\\.'"):
        ExportError(
            "bad export",
            retryability=Retryability.NEVER,
            code="manifest.invalid",
        )


def test_run_style_identifiers_are_unique_and_type_prefixed():
    first = new_unique_id("run")
    second = new_unique_id("run")

    assert first.startswith("run_")
    assert len(first) == len("run_") + 32
    assert first != second
