from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

from binary_llm.domain import (
    BINARYLLM_PAPER_LEDGER,
    BINARYLLM_PAPER_PATH,
    BINARYLLM_PAPER_REFERENCE_SETTINGS,
    BINARYLLM_PAPER_SHA256,
    PaperFactKind,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_ledger_binds_exact_preserved_paper_bytes() -> None:
    paper_bytes = (REPOSITORY_ROOT / BINARYLLM_PAPER_PATH).read_bytes()

    assert hashlib.sha256(paper_bytes).hexdigest() == BINARYLLM_PAPER_SHA256
    assert BINARYLLM_PAPER_LEDGER.source_sha256 == BINARYLLM_PAPER_SHA256


def test_fact_ids_are_unique_and_key_source_records_are_retrievable() -> None:
    facts = BINARYLLM_PAPER_LEDGER.facts

    assert len({fact.fact_id for fact in facts}) == len(facts)
    assert BINARYLLM_PAPER_LEDGER.fact("stage1.steps").statement.endswith(
        "50 optimization steps."
    )
    assert BINARYLLM_PAPER_LEDGER.fact("ambiguity.phase_index").kind is PaperFactKind.AMBIGUITY
    assert (
        BINARYLLM_PAPER_LEDGER.fact("source_issue.runtime_precision").kind
        is PaperFactKind.SOURCE_ISSUE
    )
    assert BINARYLLM_PAPER_LEDGER.facts_by_kind(PaperFactKind.AMBIGUITY)


def test_ledger_identity_is_deterministic_and_source_content_sensitive() -> None:
    assert BINARYLLM_PAPER_LEDGER.identity.value == BINARYLLM_PAPER_LEDGER.identity.value

    changed_fact = replace(
        BINARYLLM_PAPER_LEDGER.facts[0],
        statement=BINARYLLM_PAPER_LEDGER.facts[0].statement + " Source-bound change.",
    )
    changed_ledger = replace(
        BINARYLLM_PAPER_LEDGER,
        facts=(changed_fact, *BINARYLLM_PAPER_LEDGER.facts[1:]),
    )

    assert changed_ledger.identity.value != BINARYLLM_PAPER_LEDGER.identity.value


def test_reference_settings_capture_exact_stated_training_and_evaluation_facts() -> None:
    settings = BINARYLLM_PAPER_REFERENCE_SETTINGS

    assert settings.optimization["stage1_steps"] == 50
    assert settings.optimization["stage1_trainable"] == "input_channel_scales_only"
    assert settings.optimization["stage1_weight_transform"] == "W*S_t^-1"
    assert settings.optimization["stage1_input_transform"] == "S_t*A"
    assert settings.optimization["stage1_transition"] == "W_tilde=W/S_t_star"
    assert settings.optimization["stage2_trainable"] == "all_parameters"
    assert settings.schedule == {
        "phase_count": 20,
        "formula": "1.3*exp(0.22*c)-1.3",
        "phase_index_origin": "paper_ambiguous",
    }
    assert settings.data == {
        "dataset": "RedPajama",
        "partition_count": 20,
        "partition_correspondence": "one_data_chunk_per_training_phase",
    }
    assert settings.optimization["optimizer"] == "AdamW"
    assert settings.optimization["sequence_length"] == 2048
    assert settings.optimization["batch_size"] == 128
    assert settings.optimization["learning_rate_initial"] == 1e-4
    assert settings.optimization["learning_rate_final"] == 2e-6
    assert settings.optimization["learning_rate_schedule"] == "cosine"
    assert settings.optimization["weight_decay"] == 0.1
    assert settings.evaluation["perplexity"] == ("WikiText-2", "C4", "PTB")
    assert settings.evaluation["zero_shot"] == (
        "BoolQ",
        "PIQA",
        "HellaSwag",
        "WinoGrande",
        "ARC-Easy",
        "ARC-Challenge",
        "OpenBookQA",
    )


def test_framework_only_claims_are_absent_from_paper_ledger() -> None:
    serialized = str(BINARYLLM_PAPER_LEDGER.to_dict()).lower()

    assert "iris" not in serialized
    assert "iphone" not in serialized
    assert "true-1-bit runtime" not in serialized
    assert "packed runtime" not in serialized
