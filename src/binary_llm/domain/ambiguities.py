"""Required ambiguity-register seed data; no scientific choice is selected here."""

from __future__ import annotations

from .models import (
    AmbiguityCandidate,
    AmbiguityEntry,
    AmbiguityRegister,
    AmbiguityStatus,
    SourceStatus,
)

REQUIRED_AMBIGUITY_KEYS = (
    "stage1.scale_parameterization",
    "stage1.inverse_activation",
    "stage1.loss_and_batching",
    "progressive.phase_index",
    "progressive.analytical_scale_gradient",
    "progressive.stage2_fold",
    "numeric.precision",
    "optimizer",
    "reproducibility",
    "export",
)


def _candidate(candidate_id: str, description: str, *, framework: bool = False) -> AmbiguityCandidate:
    status = SourceStatus.FRAMEWORK_SELECTED if framework else SourceStatus.PAPER_INFERRED
    return AmbiguityCandidate(candidate_id, description, status)


def _entry(
    ambiguity_id: str,
    version: str,
    statement: str,
    components: tuple[str, ...],
    candidates: tuple[AmbiguityCandidate, ...],
) -> AmbiguityEntry:
    return AmbiguityEntry(
        ambiguity_id=ambiguity_id,
        register_version=version,
        statement=statement,
        source_status=SourceStatus.PAPER_INFERRED,
        affected_components=components,
        candidates=candidates,
        matched_protocol=None,
        decision_rule=None,
        confidence_method=None,
        required_floors=(),
        scale_scope=(),
        status=AmbiguityStatus.OPEN,
        selected_candidate=None,
        evidence_refs=(),
        rejected_candidates=(),
    )


def seed_ambiguity_register(register_id: str, version: str) -> AmbiguityRegister:
    """Create the required initial register without resolving any entry."""
    entries = (
        _entry(
            "stage1.scale_parameterization", version,
            "How Stage 1 input scales remain positive and avoid zero.",
            ("stage1", "binary_operator"),
            (
                _candidate("positive_exp", "Exponential positive parameterization."),
                _candidate("positive_softplus", "Softplus plus a registered epsilon."),
                _candidate("signed_clamp", "Signed scale with a magnitude floor.", framework=True),
            ),
        ),
        _entry(
            "stage1.inverse_activation", version,
            "How inverse activation scaling is retained or folded.",
            ("stage1", "stage2_transition"),
            (
                _candidate("weight_only_transition", "Use transformed weights and discard activation factor."),
                _candidate("explicit_activation_transform", "Retain and prove an explicit activation fold."),
            ),
        ),
        _entry(
            "stage1.loss_and_batching", version,
            "Which exact causal objective and batch protocol reproduce Stage 1.",
            ("stage1", "data"),
            (),
        ),
        _entry(
            "progressive.phase_index", version,
            "Whether the twenty phases use indices zero through nineteen or one through twenty.",
            ("progressive_schedule",),
            (
                _candidate("zero_based", "Use 0..19 with the continuous t=0 limit."),
                _candidate("one_based", "Use 1..20."),
            ),
        ),
        _entry(
            "progressive.analytical_scale_gradient", version,
            "Whether analytical-scale recomputation participates in gradients.",
            ("dual_scaling",),
            (
                _candidate("detached", "Treat recomputed analytical scale as detached."),
                _candidate("differentiable", "Differentiate through analytical-scale recomputation."),
            ),
        ),
        _entry(
            "progressive.stage2_fold", version,
            "How Stage 1 transformed weights enter Stage 2.",
            ("stage2_transition",),
            (
                _candidate("transformed_latent", "Preserve only the transformed latent weights."),
                _candidate("explicit_fold", "Use an explicit equivalent fold plan."),
            ),
        ),
        _entry(
            "numeric.precision", version,
            "Which compute precision is used for numerically sensitive operators.",
            ("stage1", "progressive", "export"),
            (
                _candidate("bf16", "Use BF16 operator computation."),
                _candidate("fp32_core", "Use an FP32 core with a declared mixed-precision shell.", framework=True),
            ),
        ),
        _entry(
            "optimizer", version,
            "Which optimizer parameters, warmup, and gradient clipping are used.",
            ("stage1", "progressive", "recovery"),
            (),
        ),
        _entry(
            "reproducibility", version,
            "Which seed policy, deterministic operations, and data ordering are used.",
            ("orchestration", "data"),
            (),
        ),
        _entry(
            "export", version,
            "Which zero rule, scale precision, rounding, bit order, alignment, and expansion limit apply.",
            ("export", "runtime"),
            (),
        ),
    )
    return AmbiguityRegister(register_id=register_id, version=version, entries=entries)
