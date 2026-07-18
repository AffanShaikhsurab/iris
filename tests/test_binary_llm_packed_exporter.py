from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from binary_llm.adapters import TensorRole
from binary_llm.domain import ExportError, FormatSpec
from binary_llm.export import (
    IdentifiedCheckpoint,
    PackedExporter,
    PackedTensorSource,
    SourceArtifactKind,
    pack_sign_rows,
    unpack_sign_rows,
)


def format_spec(
    *, bit_order: str = "lsb0", scale_dtype: str = "float32",
    scale_endianness: str = "little", row_alignment: int = 4,
) -> FormatSpec:
    return FormatSpec(
        format_id="binary-packed-test",
        version="1",
        bit_order=bit_order,
        row_order="row_major",
        row_alignment=row_alignment,
        zero_sign_rule="positive",
        scale_dtype=scale_dtype,
        scale_endianness=scale_endianness,
        metadata_encoding="canonical_json",
        allowed_representation_ids=("binary-v1",),
        temporary_expansion_limit_bytes=64,
    )


def checkpoint(
    *, source_kind: SourceArtifactKind = SourceArtifactKind.TRAINING_CHECKPOINT,
) -> IdentifiedCheckpoint:
    return IdentifiedCheckpoint(
        checkpoint_id="phase-20-complete",
        checkpoint_sha256="a" * 64,
        source_kind=source_kind,
        tensors=(
            PackedTensorSource(
                name="block.attention.q_proj.weight",
                semantic_role=TensorRole.ATTENTION_QUERY,
                representation_id="binary-v1",
                latent_weight=torch.tensor(
                    [[1.0, -2.0, 0.0, 4.0, -5.0], [-1.0, 2.0, -3.0, -4.0, 5.0]],
                    dtype=torch.float32,
                ),
                merged_scale=torch.tensor([0.5, 1.25], dtype=torch.float32),
            ),
        ),
    )


def exporter() -> PackedExporter:
    return PackedExporter(
        exporter_revision="packed-exporter-test-r1",
        runtime_revision="scalar-runtime-test-r1",
        build_flags=("deterministic", "cpu"),
    )

def test_golden_row_major_bit_order_and_padding_vectors() -> None:
    weight = checkpoint().tensors[0].latent_weight
    lsb = pack_sign_rows(
        weight, bit_order="lsb0", zero_sign_rule="positive", row_alignment=2
    )
    msb = pack_sign_rows(
        weight, bit_order="msb0", zero_sign_rule="positive", row_alignment=2
    )

    assert lsb.data == bytes((0x0D, 0x00, 0x12, 0x00))
    assert msb.data == bytes((0xB0, 0x00, 0x48, 0x00))
    assert lsb.row_padding_bits == 3
    assert lsb.row_payload_bytes == 1
    assert lsb.row_stride_bytes == 2
    torch.testing.assert_close(
        unpack_sign_rows(
            lsb.data, rows=2, columns=5, bit_order="lsb0", row_stride_bytes=2
        ),
        torch.tensor([[1, -1, 1, 1, -1], [-1, 1, -1, -1, 1]], dtype=torch.int8),
    )


def test_export_is_deterministic_and_reopen_verifies_every_value_and_byte(tmp_path) -> None:
    source = checkpoint()
    service = exporter()
    plan = service.plan(
        source, format_spec(), scale_tolerance=0.0, artifact_id="packed-candidate"
    )
    first = service.export(plan, tmp_path / "first.bllmp")
    second = service.export(plan, tmp_path / "second.bllmp")

    assert first.path.read_bytes() == second.path.read_bytes() == plan.container_bytes
    assert first.sha256 == second.sha256
    assert first.bytes == plan.ledger.total_distributable_bytes
    assert plan.ledger.accounting_reconciled
    assert plan.ledger.source_checkpoint_id == source.checkpoint_id

    reopened = service.reopen(first.path)
    proof = service.verify(reopened, source)
    assert proof.exact_byte_match
    assert proof.accounting_reconciled
    assert proof.signs_verified == 10
    assert proof.scales_verified == 2
    assert proof.bytes_verified == first.bytes
    assert proof.artifact_sha256 == first.sha256
    assert set(proof.tensor_checksums) == {"block.attention.q_proj.weight"}

    metadata = plan.metadata
    assert metadata["format"]["bit_order"] == "lsb0"
    assert metadata["format"]["row_order"] == "row_major"
    assert metadata["format"]["row_alignment"] == 4
    assert metadata["format"]["scale_dtype"] == "float32"
    assert metadata["format"]["scale_endianness"] == "little"
    assert metadata["format"]["version"] == "1"
    assert metadata["source"]["checkpoint_sha256"] == "a" * 64
    tensor = metadata["tensors"][0]
    assert tensor["row_padding_bits"] == 3
    assert tensor["checksum_bytes"] == 64
    assert len(tensor["sign_checksum_sha256"]) == 64
    assert len(tensor["scale_checksum_sha256"]) == 64

def test_big_endian_float16_scales_require_preregistered_precision_tolerance(tmp_path) -> None:
    source = checkpoint()
    service = exporter()
    spec = format_spec(scale_dtype="float16", scale_endianness="big")
    exact_source = replace(
        source,
        tensors=(
            replace(
                source.tensors[0],
                merged_scale=torch.tensor([0.5, 1.25], dtype=torch.float32),
            ),
        ),
    )
    plan = service.plan(exact_source, spec, scale_tolerance=0.0)
    artifact = service.export(plan, tmp_path / "big-endian.bllmp")
    assert service.verify(artifact, exact_source).scales_verified == 2

    inexact_source = replace(
        source,
        tensors=(
            replace(
                source.tensors[0],
                merged_scale=torch.tensor([0.1, 1.2], dtype=torch.float32),
            ),
        ),
    )
    with pytest.raises(ExportError, match="precision tolerance"):
        service.plan(inexact_source, spec, scale_tolerance=0.0)
    assert service.plan(inexact_source, spec, scale_tolerance=0.001).scale_tolerance == 0.001


def test_corrupt_and_partial_outputs_are_quarantined(tmp_path) -> None:
    source = checkpoint()
    service = exporter()
    plan = service.plan(source, format_spec(), scale_tolerance=0.0)
    destination = tmp_path / "model.bllmp"
    destination.write_bytes(b"partial")

    artifact = service.export(plan, destination)
    quarantined = list((tmp_path / "quarantine").glob("*.quarantine"))
    assert artifact.path.is_file()
    assert any(item.read_bytes() == b"partial" for item in quarantined)

    damaged = bytearray(artifact.path.read_bytes())
    damaged[-1] ^= 0x01
    artifact.path.write_bytes(damaged)
    with pytest.raises(ExportError, match="quarantined") as captured:
        service.verify(artifact, source)
    assert captured.value.code in {"export.byte_mismatch", "export.verification_failed"}
    assert not artifact.path.exists()
    assert len(list((tmp_path / "quarantine").glob("*.quarantine"))) == 2


def test_export_rejects_another_quantized_deployment_as_source() -> None:
    with pytest.raises(ExportError, match="directly from a training checkpoint") as captured:
        exporter().plan(
            checkpoint(source_kind=SourceArtifactKind.QUANTIZED_DEPLOYMENT),
            format_spec(),
            scale_tolerance=0.0,
        )
    assert captured.value.code == "export.non_training_source"


def test_zero_error_rule_fails_before_any_output_is_written(tmp_path) -> None:
    spec = replace(format_spec(), zero_sign_rule="error")
    with pytest.raises(ValueError, match="exact zero"):
        exporter().plan(checkpoint(), spec, scale_tolerance=0.0)
    assert not tuple(tmp_path.iterdir())
