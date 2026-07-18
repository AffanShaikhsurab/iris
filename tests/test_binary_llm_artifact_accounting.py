from __future__ import annotations

from fractions import Fraction

import pytest

from binary_llm.adapters import TensorRole, TensorScope
from binary_llm.export import (
    TARGET_BAND_MAX_BYTES,
    TARGET_BAND_MIN_BYTES,
    ArtifactByteCategory,
    ArtifactByteEntry,
    ArtifactFileEntry,
    ArtifactPlacement,
    TargetBandStatus,
    TensorContentKind,
    TensorLedgerEntry,
    account_artifact,
)


def tensor(
    name: str,
    shape: tuple[int, ...],
    role: TensorRole,
    scope: TensorScope,
    payload: int,
    **kwargs: object,
) -> TensorLedgerEntry:
    return TensorLedgerEntry(
        tensor_name=name,
        shape=shape,
        parameter_count=shape[0] if len(shape) == 1 else shape[0] * shape[1],
        semantic_role=role,
        scope=scope,
        representation_id=str(kwargs.pop("representation_id", "test-representation")),
        payload_bytes=payload,
        **kwargs,
    )


def complete_tensor_inventory() -> tuple[TensorLedgerEntry, ...]:
    return (
        tensor(
            "block.q.weight", (2, 5), TensorRole.ATTENTION_QUERY,
            TensorScope.BINARY_BODY, 2, scale_bytes=4, metadata_bytes=3,
            alignment_bytes=1, checksum_bytes=2,
        ),
        tensor(
            "token_embedding.weight", (3, 4), TensorRole.TOKEN_EMBEDDING,
            TensorScope.EXCLUDED, 24, representation_id="embedding-bf16",
        ),
        tensor(
            "lm_head.weight", (3, 4), TensorRole.LANGUAGE_MODEL_HEAD,
            TensorScope.EXCLUDED, 6, representation_id="head-q4",
        ),
        tensor(
            "norm.weight", (3,), TensorRole.NORMALIZATION,
            TensorScope.EXCLUDED, 6,
        ),
        tensor(
            "block.q.bias", (3,), TensorRole.BIAS,
            TensorScope.EXCLUDED, 6,
        ),
        tensor(
            "block.q.scale", (2,), TensorRole.LEARNED_ROW_SCALE,
            TensorScope.EXCLUDED, 0, scale_bytes=4,
            content_kind=TensorContentKind.SCALE,
            counts_toward_original_parameters=False,
        ),
        tensor(
            "block.causal_mask", (2, 2), TensorRole.BUFFER,
            TensorScope.EXCLUDED, 4, content_kind=TensorContentKind.MASK,
            counts_toward_original_parameters=False,
        ),
    )


def complete_non_tensor_inventory() -> tuple[ArtifactByteEntry, ...]:
    packed = ArtifactPlacement.PACKED_ARTIFACT
    sidecar = ArtifactPlacement.REQUIRED_SIDECAR
    return (
        ArtifactByteEntry("container header", ArtifactByteCategory.CONTAINER, 7, packed),
        ArtifactByteEntry("tensor table", ArtifactByteCategory.METADATA, 5, packed),
        ArtifactByteEntry("padding", ArtifactByteCategory.ALIGNMENT, 3, packed),
        ArtifactByteEntry("tensor index", ArtifactByteCategory.INDEX, 2, packed),
        ArtifactByteEntry("manifest checksum", ArtifactByteCategory.CHECKSUM, 4, packed),
        ArtifactByteEntry("tokenizer", ArtifactByteCategory.TOKENIZER, 11, sidecar),
        ArtifactByteEntry("runtime mask", ArtifactByteCategory.MASK, 13, sidecar),
        ArtifactByteEntry("license", ArtifactByteCategory.OTHER_REQUIRED, 1, sidecar),
    )


def test_whole_artifact_ledger_enumerates_and_reconciles_every_category() -> None:
    files = (
        ArtifactFileEntry("model.bin", 83, "a" * 64),
        ArtifactFileEntry("tokenizer.json", 11, "b" * 64),
        ArtifactFileEntry("mask.bin", 13, "c" * 64),
        ArtifactFileEntry("LICENSE", 1, "d" * 64),
    )
    ledger = account_artifact(
        artifact_id="candidate-head-q4",
        format_version="1",
        source_checkpoint_id="checkpoint-1",
        original_parameter_count=40,
        tensor_entries=complete_tensor_inventory(),
        non_tensor_entries=complete_non_tensor_inventory(),
        file_inventory=files,
    )

    assert ledger.binary_body_parameters == 10
    assert ledger.excluded_parameters == 30
    assert ledger.auxiliary_tensor_elements == 6
    assert ledger.binary_body_bytes == 12
    assert ledger.excluded_tensor_bytes == 50
    assert ledger.ideal_binary_payload_bytes == 2
    assert ledger.packed_artifact_bytes == 83
    assert ledger.required_sidecar_bytes == 25
    assert ledger.total_distributable_bytes == 108
    assert ledger.ideal_to_packed_delta_bytes == 81
    assert ledger.packed_to_distributable_delta_bytes == 25
    assert ledger.ideal_to_distributable_delta_bytes == 106
    assert ledger.effective_bits_per_original_parameter == Fraction(108, 5)
    assert ledger.effective_bits_numerator == 108
    assert ledger.effective_bits_denominator == 5
    assert ledger.target_band_status is TargetBandStatus.BELOW
    assert not ledger.target_band_200_300_decimal_mb
    assert ledger.reconciliation_delta_bytes == 0
    assert ledger.qualification_allowed

    assert set(ledger.byte_category_totals) == set(ArtifactByteCategory)
    assert sum(ledger.byte_category_totals.values()) == ledger.total_distributable_bytes
    assert ledger.byte_category_totals[ArtifactByteCategory.BINARY_PAYLOAD] == 2
    assert ledger.byte_category_totals[ArtifactByteCategory.EMBEDDING] == 24
    assert ledger.byte_category_totals[ArtifactByteCategory.UNTIED_HEAD] == 6
    assert ledger.byte_category_totals[ArtifactByteCategory.NORMALIZATION] == 6
    assert ledger.byte_category_totals[ArtifactByteCategory.BIAS] == 6
    assert ledger.byte_category_totals[ArtifactByteCategory.SCALE] == 8
    assert ledger.byte_category_totals[ArtifactByteCategory.METADATA] == 8
    assert ledger.byte_category_totals[ArtifactByteCategory.ALIGNMENT] == 4
    assert ledger.byte_category_totals[ArtifactByteCategory.MASK] == 17
    assert ledger.byte_category_totals[ArtifactByteCategory.CHECKSUM] == 6
    assert ledger.byte_category_totals[ArtifactByteCategory.TOKENIZER] == 11
    assert ledger.byte_category_totals[ArtifactByteCategory.CONTAINER] == 7


def one_byte_binary_tensor() -> TensorLedgerEntry:
    return tensor(
        "q.weight", (1, 8), TensorRole.ATTENTION_QUERY,
        TensorScope.BINARY_BODY, 1,
    )


@pytest.mark.parametrize(
    ("total_bytes", "expected"),
    (
        (TARGET_BAND_MIN_BYTES - 1, TargetBandStatus.BELOW),
        (TARGET_BAND_MIN_BYTES, TargetBandStatus.WITHIN),
        (TARGET_BAND_MAX_BYTES, TargetBandStatus.WITHIN),
        (TARGET_BAND_MAX_BYTES + 1, TargetBandStatus.ABOVE),
    ),
)
def test_target_band_uses_inclusive_decimal_megabyte_boundaries(
    total_bytes: int, expected: TargetBandStatus
) -> None:
    ledger = account_artifact(
        artifact_id=f"candidate-{total_bytes}",
        format_version="1",
        source_checkpoint_id="checkpoint",
        original_parameter_count=8,
        tensor_entries=(one_byte_binary_tensor(),),
        non_tensor_entries=(
            ArtifactByteEntry(
                "container", ArtifactByteCategory.CONTAINER, total_bytes - 1,
                ArtifactPlacement.PACKED_ARTIFACT,
            ),
        ),
        file_inventory=(ArtifactFileEntry("model.bin", total_bytes, "sum"),),
    )

    assert ledger.total_distributable_bytes == total_bytes
    assert ledger.target_band_status is expected
    assert ledger.target_band_200_300_decimal_mb is (expected is TargetBandStatus.WITHIN)
    assert ledger.effective_bits_per_original_parameter == Fraction(8 * total_bytes, 8)


def test_measured_file_mismatch_fails_accounting_qualification() -> None:
    ledger = account_artifact(
        artifact_id="candidate",
        format_version="1",
        source_checkpoint_id="checkpoint",
        original_parameter_count=8,
        tensor_entries=(one_byte_binary_tensor(),),
    )
    assert ledger.reconciliation_delta_bytes is None
    assert not ledger.qualification_allowed

    mismatched = ledger.with_file_inventory(
        (ArtifactFileEntry("model.bin", 2, "measured"),)
    )
    assert mismatched.reconciliation_delta_bytes == 1
    assert not mismatched.accounting_reconciled
    assert not mismatched.qualification_allowed

    reconciled = ledger.with_file_inventory(
        (ArtifactFileEntry("model.bin", 1, "measured"),)
    )
    assert reconciled.reconciliation_delta_bytes == 0
    assert reconciled.qualification_allowed


def test_original_parameter_denominator_must_match_enumerated_unique_parameters() -> None:
    with pytest.raises(ValueError, match="denominator does not match"):
        account_artifact(
            artifact_id="candidate",
            format_version="1",
            source_checkpoint_id="checkpoint",
            original_parameter_count=9,
            tensor_entries=(one_byte_binary_tensor(),),
        )


def test_tied_head_alias_is_explicit_and_does_not_double_count_parameters() -> None:
    embedding = tensor(
        "embedding.weight", (2, 4), TensorRole.TOKEN_EMBEDDING,
        TensorScope.EXCLUDED, 16,
    )
    tied_head = tensor(
        "lm_head.weight", (2, 4), TensorRole.LANGUAGE_MODEL_HEAD,
        TensorScope.EXCLUDED, 0, tied_to="embedding.weight",
        counts_toward_original_parameters=False,
    )
    ledger = account_artifact(
        artifact_id="tied-candidate",
        format_version="1",
        source_checkpoint_id="checkpoint",
        original_parameter_count=16,
        tensor_entries=(one_byte_binary_tensor(), embedding, tied_head),
        file_inventory=(ArtifactFileEntry("model.bin", 17, "sum"),),
    )

    assert ledger.binary_body_parameters == 8
    assert ledger.excluded_parameters == 8
    assert ledger.byte_category_totals[ArtifactByteCategory.EMBEDDING] == 16
    assert ledger.byte_category_totals[ArtifactByteCategory.TIED_HEAD] == 0
    assert ledger.byte_category_totals[ArtifactByteCategory.UNTIED_HEAD] == 0


def test_duplicate_tensor_and_file_names_are_rejected() -> None:
    entry = one_byte_binary_tensor()
    with pytest.raises(ValueError, match="tensor names must be unique"):
        account_artifact(
            artifact_id="candidate",
            format_version="1",
            source_checkpoint_id="checkpoint",
            original_parameter_count=16,
            tensor_entries=(entry, entry),
        )
    ledger = account_artifact(
        artifact_id="candidate",
        format_version="1",
        source_checkpoint_id="checkpoint",
        original_parameter_count=8,
        tensor_entries=(entry,),
    )
    repeated = ArtifactFileEntry("model.bin", 1, "sum")
    with pytest.raises(ValueError, match="file paths must be unique"):
        ledger.with_file_inventory((repeated, repeated))
