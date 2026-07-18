"""Authoritative BinaryLLM paper claims and protocol facts.

This module contains only facts stated by the preserved paper conversion. It does
not resolve paper ambiguities and does not include Iris-specific framework choices.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .identity import ContentIdentity, identify_content
from .models import CanonicalModel, ReproductionSettings, SourceStatus

BINARYLLM_PAPER_PATH = "docs/research-papers/2508.06974v2-binaryLLM.md"
BINARYLLM_PAPER_SHA256 = "8e346e97f4d172b93c04d3ad240eb32d599a9f8924290ffd0f486661bafeb8f6"
BINARYLLM_PAPER_TITLE = (
    "Rethinking 1-bit Optimization Leveraging Pre-trained Large Language Models"
)


class PaperFactKind(StrEnum):
    EQUATION = "equation"
    TENSOR_SCOPE = "tensor_scope"
    TRAINING_PROTOCOL = "training_protocol"
    EVALUATION_PROTOCOL = "evaluation_protocol"
    REPORTED_RESULT = "reported_result"
    AMBIGUITY = "ambiguity"
    SOURCE_ISSUE = "source_issue"


@dataclass(frozen=True, slots=True)
class PaperFact(CanonicalModel):
    fact_id: str
    kind: PaperFactKind
    statement: str
    source_locator: str
    source_status: SourceStatus

    def __post_init__(self) -> None:
        for name in ("fact_id", "statement", "source_locator"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class PaperProtocolLedger(CanonicalModel):
    source_path: str
    source_sha256: str
    source_title: str
    facts: tuple[PaperFact, ...]
    reference_settings: ReproductionSettings

    def __post_init__(self) -> None:
        if self.source_path != BINARYLLM_PAPER_PATH:
            raise ValueError("paper ledger must identify the authoritative local source")
        if self.source_sha256 != BINARYLLM_PAPER_SHA256:
            raise ValueError("paper ledger source hash does not match the preserved paper")
        if self.source_title != BINARYLLM_PAPER_TITLE:
            raise ValueError("paper ledger title does not match the preserved paper")
        fact_ids = tuple(item.fact_id for item in self.facts)
        if not fact_ids or len(fact_ids) != len(set(fact_ids)):
            raise ValueError("paper facts must be non-empty and uniquely identified")
        if any(
            item.source_status is SourceStatus.FRAMEWORK_SELECTED
            for item in self.facts
        ):
            raise ValueError("paper ledger facts cannot contain framework-selected claims")

    @property
    def identity(self) -> ContentIdentity:
        return identify_content(
            self.to_dict(),
            kind="binaryllm-paper-protocol",
            representation_fields={"source_sha256": self.source_sha256},
        )

    def fact(self, fact_id: str) -> PaperFact:
        matches = tuple(item for item in self.facts if item.fact_id == fact_id)
        if len(matches) != 1:
            raise KeyError(f"unknown BinaryLLM paper fact: {fact_id}")
        return matches[0]

    def facts_by_kind(self, kind: PaperFactKind) -> tuple[PaperFact, ...]:
        return tuple(item for item in self.facts if item.kind is kind)


def _fact(
    fact_id: str,
    kind: PaperFactKind,
    statement: str,
    source_locator: str,
    *,
    status: SourceStatus = SourceStatus.PAPER_SPECIFIED,
) -> PaperFact:
    return PaperFact(fact_id, kind, statement, source_locator, status)


BINARYLLM_PAPER_FACTS = (
    _fact(
        "operator.binary_values",
        PaperFactKind.EQUATION,
        "Transformer-block linear weights use binary values {-1,+1}.",
        "Figure 1; Equation 1",
    ),
    _fact(
        "operator.progressive_function",
        PaperFactKind.EQUATION,
        "F(x,t)=tanh(t*x)/tanh(t).",
        "Equation 4",
    ),
    _fact(
        "operator.progressive_derivative",
        PaperFactKind.EQUATION,
        "dF/dx=t*(1-tanh(t*x)^2)/tanh(t), used instead of STE in Stage 2.",
        "Equations 5-6",
    ),
    _fact(
        "operator.dual_scaling",
        PaperFactKind.EQUATION,
        "Stage 2 uses S_l*S_a*F(W_tilde/S_a,t), with S_l initialized to one.",
        "Equation 9; Appendix Equation 12",
    ),
    _fact(
        "operator.final_sign",
        PaperFactKind.EQUATION,
        "Inference replaces the progressive function with S_l*S_a*Sign(W).",
        "Equation 10",
    ),
    _fact(
        "scope.transformer_linears",
        PaperFactKind.TENSOR_SCOPE,
        "All linear layers in transformer blocks are binarized.",
        "Figure 1 caption; Section 4.2",
    ),
    _fact(
        "scope.excluded_embedding_head",
        PaperFactKind.TENSOR_SCOPE,
        "Embedding and output-head parameters are excluded from reported average bit width.",
        "Section 4.2",
    ),
    _fact(
        "stage1.input_channel_scale",
        PaperFactKind.TRAINING_PROTOCOL,
        "Stage 1 learns per-input-channel scales initialized to one while other parameters are frozen.",
        "Appendix A, Stage 1",
    ),
    _fact(
        "stage1.inverse_activation",
        PaperFactKind.TRAINING_PROTOCOL,
        "Stage 1 applies W*S_t^-1 and the inverse input transform S_t*A.",
        "Appendix A, Stage 1",
    ),
    _fact(
        "stage1.binary_quantization",
        PaperFactKind.TRAINING_PROTOCOL,
        "Stage 1 performs 1-bit quantization on the scaled weight and optimizes the scales with autoregressive loss.",
        "Appendix A, Stage 1; Appendix Equation 11",
    ),
    _fact(
        "stage1.steps",
        PaperFactKind.TRAINING_PROTOCOL,
        "Binary-aware initialization runs for 50 optimization steps.",
        "Appendix A, Stage 1",
    ),
    _fact(
        "stage1.output",
        PaperFactKind.TRAINING_PROTOCOL,
        "The Stage 2 latent checkpoint is W_tilde=W/S_t*.",
        "Appendix Equation 11",
    ),
    _fact(
        "stage2.twenty_phases",
        PaperFactKind.TRAINING_PROTOCOL,
        "Training data and training time are divided into 20 corresponding chunks/phases.",
        "Appendix A, Stage 2",
    ),
    _fact(
        "stage2.all_parameters_trainable",
        PaperFactKind.TRAINING_PROTOCOL,
        "The paper states that all parameters are learnable during Stage 2.",
        "Appendix A, Stage 2",
    ),
    _fact(
        "schedule.exponential",
        PaperFactKind.TRAINING_PROTOCOL,
        "The recommended schedule is t(c)=1.3*exp(0.22*c)-1.3.",
        "Appendix A, Stage 2",
    ),
    _fact(
        "optimization.reported",
        PaperFactKind.TRAINING_PROTOCOL,
        "Reported training uses AdamW, sequence length 2048, batch size 128, cosine learning rate 1e-4 to 2e-6, and weight decay 0.1.",
        "Section 4.1, Implementation Details",
    ),
    _fact(
        "data.redpajama",
        PaperFactKind.TRAINING_PROTOCOL,
        "Training data is randomly sampled from RedPajama.",
        "Section 4.1, Implementation Details",
    ),
    _fact(
        "evaluation.paper_panel",
        PaperFactKind.EVALUATION_PROTOCOL,
        "Evaluation uses WikiText-2, C4, PTB, BoolQ, PIQA, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge, and OpenBookQA.",
        "Section 4.1, Benchmarks",
    ),
    _fact(
        "ambiguity.phase_index",
        PaperFactKind.AMBIGUITY,
        "The chunk-number origin is not stated; c=0 gives t=0.",
        "Appendix A; Appendix F",
        status=SourceStatus.PAPER_INFERRED,
    ),
    _fact(
        "ambiguity.stage1_backward",
        PaperFactKind.AMBIGUITY,
        "The Stage 1 gradient estimator through binary weights is not specified.",
        "Appendix A, Stage 1",
        status=SourceStatus.PAPER_INFERRED,
    ),
    _fact(
        "source_issue.smol_135m_tables",
        PaperFactKind.SOURCE_ISSUE,
        "Table 2 and Table 3 report different SmolLM-135M BinaryLLM values and must remain protocol-distinct.",
        "Tables 2-3",
        status=SourceStatus.PAPER_INFERRED,
    ),
    _fact(
        "source_issue.runtime_precision",
        PaperFactKind.SOURCE_ISSUE,
        "The hardware experiment uses a BitNet-style deployment and states that actual deployed precision is 1.58 bits, not demonstrated direct true-1-bit execution.",
        "Appendix D",
        status=SourceStatus.PAPER_INFERRED,
    ),
)


BINARYLLM_PAPER_REFERENCE_SETTINGS = ReproductionSettings(
    operator={
        "binary_values": (-1, 1),
        "binary_scope": "all_transformer_block_linears",
        "excluded_from_reported_average_bit_width": ("embedding", "output_head"),
        "progressive_forward": "tanh(t*x)/tanh(t)",
        "progressive_backward": "analytical_derivative",
        "analytical_scale": "mean_absolute_current_latent_row",
        "learned_scale_initial": 1.0,
        "scale_variant": "dual_product",
        "final_operator": "merged_scale_times_sign",
        "inference_offset": False,
    },
    data={
        "dataset": "RedPajama",
        "partition_count": 20,
        "partition_correspondence": "one_data_chunk_per_training_phase",
    },
    schedule={
        "phase_count": 20,
        "formula": "1.3*exp(0.22*c)-1.3",
        "phase_index_origin": "paper_ambiguous",
    },
    optimization={
        "stage1_steps": 50,
        "stage1_trainable": "input_channel_scales_only",
        "stage1_scale_initial": 1.0,
        "stage1_weight_transform": "W*S_t^-1",
        "stage1_input_transform": "S_t*A",
        "stage1_objective": "autoregressive_loss",
        "stage1_quantization": "1_bit_on_scaled_weight",
        "stage1_transition": "W_tilde=W/S_t_star",
        "stage2_trainable": "all_parameters",
        "optimizer": "AdamW",
        "sequence_length": 2048,
        "batch_size": 128,
        "learning_rate_initial": 1e-4,
        "learning_rate_final": 2e-6,
        "learning_rate_schedule": "cosine",
        "weight_decay": 0.1,
    },
    evaluation={
        "perplexity": ("WikiText-2", "C4", "PTB"),
        "zero_shot": (
            "BoolQ",
            "PIQA",
            "HellaSwag",
            "WinoGrande",
            "ARC-Easy",
            "ARC-Challenge",
            "OpenBookQA",
        ),
    },
)


BINARYLLM_PAPER_LEDGER = PaperProtocolLedger(
    BINARYLLM_PAPER_PATH,
    BINARYLLM_PAPER_SHA256,
    BINARYLLM_PAPER_TITLE,
    BINARYLLM_PAPER_FACTS,
    BINARYLLM_PAPER_REFERENCE_SETTINGS,
)


__all__ = [
    "BINARYLLM_PAPER_FACTS",
    "BINARYLLM_PAPER_LEDGER",
    "BINARYLLM_PAPER_PATH",
    "BINARYLLM_PAPER_REFERENCE_SETTINGS",
    "BINARYLLM_PAPER_SHA256",
    "BINARYLLM_PAPER_TITLE",
    "PaperFact",
    "PaperFactKind",
    "PaperProtocolLedger",
]
