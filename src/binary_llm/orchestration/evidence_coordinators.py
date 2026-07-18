"""Auditable coordinators for perplexity, capability, code, and rubric evidence."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from enum import StrEnum
from statistics import fmean
from types import MappingProxyType
from typing import Mapping, Protocol, Sequence

from binary_llm.domain.ablations import BootstrapPlan, ScoreObservation
from binary_llm.orchestration.evaluation import EvaluationPanel
from binary_llm.orchestration.statistics import paired_bootstrap_statistics

PAPER_ZERO_SHOT_BENCHMARKS = (
    "boolq", "piqa", "hellaswag", "winogrande", "arc_easy",
    "arc_challenge", "openbookqa",
)
BROAD_BENCHMARKS = (
    "mmlu", "arc_challenge", "hellaswag", "winogrande",
    "truthfulqa", "gsm8k", "code_execution",
)
RUBRIC_PANELS = (
    "instruction_following", "summarization_faithfulness",
    "conversational_quality", "refusal_safety", "long_form_consistency",
)


def _text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _unique(name: str, values: Sequence[str]) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must be unique")


def _score(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("scores must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("scores must be finite values between zero and one")
    return result


@dataclass(frozen=True, slots=True)
class DecodingProtocol:
    temperature: float
    do_sample: bool
    max_new_tokens: int
    seed: int
    top_p: float = 1.0
    top_k: int = 0

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        if not 0.0 < self.top_p <= 1.0 or self.top_k < 0:
            raise ValueError("top_p and top_k are outside their valid ranges")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")

    @property
    def deterministic(self) -> bool:
        return self.temperature == 0.0 and not self.do_sample


@dataclass(frozen=True, slots=True)
class TokenizationSequenceProtocol:
    protocol_id: str
    tokenizer_id: str
    tokenizer_revision: str
    tokenizer_hash: str
    sequence_length: int
    stride: int
    add_special_tokens: bool
    bos_token_id: int | None
    eos_token_id: int | None
    truncation: str
    padding: str
    loss_mask_policy: str

    def __post_init__(self) -> None:
        for name in (
            "protocol_id", "tokenizer_id", "tokenizer_revision", "tokenizer_hash",
            "truncation", "padding", "loss_mask_policy",
        ):
            _text(name, getattr(self, name))
        if self.sequence_length < 2:
            raise ValueError("sequence_length must permit next-token prediction")
        if self.stride < 1 or self.stride > self.sequence_length:
            raise ValueError("stride must be in [1, sequence_length]")


@dataclass(frozen=True, slots=True)
class PerplexityCaseResult:
    case_id: str
    corpus_id: str
    token_ids_hash: str
    sequence_boundaries_hash: str
    predicted_token_count: int
    negative_log_likelihood: float

    def __post_init__(self) -> None:
        for name in ("case_id", "corpus_id", "token_ids_hash", "sequence_boundaries_hash"):
            _text(name, getattr(self, name))
        if self.predicted_token_count < 1:
            raise ValueError("perplexity cases require predicted tokens")
        if not math.isfinite(self.negative_log_likelihood) or self.negative_log_likelihood < 0:
            raise ValueError("negative log likelihood must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class PerplexityRun:
    artifact_id: str
    evaluation_set_id: str
    protocol: TokenizationSequenceProtocol
    cases: tuple[PerplexityCaseResult, ...]
    raw_outputs_ref: str

    def __post_init__(self) -> None:
        for name in ("artifact_id", "evaluation_set_id", "raw_outputs_ref"):
            _text(name, getattr(self, name))
        if not self.cases:
            raise ValueError("perplexity runs require cases")
        _unique("perplexity case ids", tuple(item.case_id for item in self.cases))


@dataclass(frozen=True, slots=True)
class PerplexityEvidence:
    candidate_artifact_id: str
    dense_artifact_id: str
    evaluation_set_id: str
    protocol: TokenizationSequenceProtocol
    candidate_perplexity: float
    dense_perplexity: float
    candidate_minus_dense: float
    candidate_by_corpus: Mapping[str, float]
    dense_by_corpus: Mapping[str, float]
    paired_case_ids: tuple[str, ...]
    raw_output_refs: tuple[str, str]


def _perplexity(cases: Sequence[PerplexityCaseResult]) -> float:
    token_count = sum(item.predicted_token_count for item in cases)
    mean_loss = sum(item.negative_log_likelihood for item in cases) / token_count
    try:
        value = math.exp(mean_loss)
    except OverflowError as exc:
        raise ValueError("perplexity is non-finite") from exc
    if not math.isfinite(value):
        raise ValueError("perplexity is non-finite")
    return value


def coordinate_perplexity(
    candidate: PerplexityRun,
    dense: PerplexityRun,
    *,
    protocol: TokenizationSequenceProtocol,
) -> PerplexityEvidence:
    """Compare candidate and dense runs only under byte-identical sequence protocols."""

    if candidate.artifact_id == dense.artifact_id:
        raise ValueError("candidate and dense artifacts must be distinct")
    if candidate.evaluation_set_id != dense.evaluation_set_id:
        raise ValueError("perplexity runs must use the same evaluation set")
    if candidate.protocol != protocol or dense.protocol != protocol:
        raise ValueError("candidate and dense tokenization/sequence protocols must be identical")
    candidate_index = {item.case_id: item for item in candidate.cases}
    dense_index = {item.case_id: item for item in dense.cases}
    if set(candidate_index) != set(dense_index):
        raise ValueError("candidate and dense perplexity cases must be exactly paired")
    for case_id in candidate_index:
        left, right = candidate_index[case_id], dense_index[case_id]
        if (
            left.corpus_id, left.token_ids_hash, left.sequence_boundaries_hash,
            left.predicted_token_count,
        ) != (
            right.corpus_id, right.token_ids_hash, right.sequence_boundaries_hash,
            right.predicted_token_count,
        ):
            raise ValueError(f"tokenization or sequence mismatch for perplexity case {case_id}")
    corpora = sorted({item.corpus_id for item in candidate.cases})
    candidate_by_corpus = {
        corpus: _perplexity(tuple(item for item in candidate.cases if item.corpus_id == corpus))
        for corpus in corpora
    }
    dense_by_corpus = {
        corpus: _perplexity(tuple(item for item in dense.cases if item.corpus_id == corpus))
        for corpus in corpora
    }
    candidate_value = _perplexity(candidate.cases)
    dense_value = _perplexity(dense.cases)
    return PerplexityEvidence(
        candidate.artifact_id, dense.artifact_id, candidate.evaluation_set_id, protocol,
        candidate_value, dense_value, candidate_value - dense_value,
        MappingProxyType(candidate_by_corpus), MappingProxyType(dense_by_corpus),
        tuple(sorted(candidate_index)), (candidate.raw_outputs_ref, dense.raw_outputs_ref),
    )


class CapabilityPanelKind(StrEnum):
    PAPER = "paper"
    BROAD = "broad"


class PairedFailureStatus(StrEnum):
    BOTH_PASS = "both_pass"
    CANDIDATE_ONLY = "candidate_only_failure"
    BASELINE_ONLY = "baseline_only_failure"
    BOTH_FAIL = "both_failure"


@dataclass(frozen=True, slots=True)
class CapabilityPanelProtocol:
    protocol_id: str
    kind: CapabilityPanelKind
    evaluation_panel: EvaluationPanel
    evaluator_revision: str
    evaluation_set_id: str
    required_benchmarks: tuple[str, ...]
    deterministic_benchmarks: tuple[str, ...]
    decoding: DecodingProtocol
    bootstrap_seed: int

    def __post_init__(self) -> None:
        for name in ("protocol_id", "evaluator_revision", "evaluation_set_id"):
            _text(name, getattr(self, name))
        _unique("required benchmarks", self.required_benchmarks)
        _unique("deterministic benchmarks", self.deterministic_benchmarks)
        if not set(self.deterministic_benchmarks) <= set(self.required_benchmarks):
            raise ValueError("deterministic benchmarks must belong to the panel")
        required = (
            set(PAPER_ZERO_SHOT_BENCHMARKS)
            if self.kind is CapabilityPanelKind.PAPER else set(BROAD_BENCHMARKS)
        )
        missing = required - set(self.required_benchmarks)
        if missing:
            raise ValueError(f"{self.kind.value} panel is missing required benchmarks: {sorted(missing)}")
        if not required <= set(self.deterministic_benchmarks):
            raise ValueError("deterministically scoreable required benchmarks must use deterministic decoding")
        if self.deterministic_benchmarks and not self.decoding.deterministic:
            raise ValueError("deterministic benchmarks require greedy non-sampling decoding")


@dataclass(frozen=True, slots=True)
class CapabilityCaseResult:
    artifact_id: str
    protocol_id: str
    evaluator_revision: str
    benchmark_id: str
    case_id: str
    semantic_family_id: str
    prompt_ref: str
    raw_prompt: str
    raw_output: str
    score: float
    passed: bool
    decoding: DecodingProtocol
    failure_categories: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "artifact_id", "protocol_id", "evaluator_revision", "benchmark_id",
            "case_id", "semantic_family_id", "prompt_ref", "raw_prompt",
        ):
            _text(name, getattr(self, name))
        object.__setattr__(self, "score", _score(self.score))
        _unique("failure categories", self.failure_categories)
        if not self.passed and not self.failure_categories:
            raise ValueError("failed capability cases require a failure category")


@dataclass(frozen=True, slots=True)
class PairedFailureRecord:
    benchmark_id: str
    case_id: str
    semantic_family_id: str
    status: PairedFailureStatus
    candidate_failure_categories: tuple[str, ...]
    baseline_failure_categories: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CapabilityPanelEvidence:
    protocol: CapabilityPanelProtocol
    candidate_artifact_id: str
    baseline_artifact_id: str
    per_panel_scores: Mapping[str, float]
    baseline_per_panel_scores: Mapping[str, float]
    aggregates: Mapping[str, float]
    candidate_minus_baseline_deltas: Mapping[str, float]
    confidence_intervals: Mapping[str, tuple[float, float]]
    paired_failures: tuple[PairedFailureRecord, ...]
    failure_categories: tuple[str, ...]
    candidate_raw_outputs: tuple[tuple[str, str, str], ...]
    baseline_raw_outputs: tuple[tuple[str, str, str], ...]

    @property
    def complete(self) -> bool:
        expected = set(self.protocol.required_benchmarks)
        return (
            expected <= set(self.per_panel_scores)
            and expected <= set(self.baseline_per_panel_scores)
            and len(self.paired_failures) == len(self.candidate_raw_outputs)
            and len(self.paired_failures) == len(self.baseline_raw_outputs)
        )


def _pair_status(candidate_passed: bool, baseline_passed: bool) -> PairedFailureStatus:
    if candidate_passed and baseline_passed:
        return PairedFailureStatus.BOTH_PASS
    if not candidate_passed and baseline_passed:
        return PairedFailureStatus.CANDIDATE_ONLY
    if candidate_passed and not baseline_passed:
        return PairedFailureStatus.BASELINE_ONLY
    return PairedFailureStatus.BOTH_FAIL


def coordinate_capability_panel(
    protocol: CapabilityPanelProtocol,
    candidate_artifact_id: str,
    baseline_artifact_id: str,
    candidate_results: Sequence[CapabilityCaseResult],
    baseline_results: Sequence[CapabilityCaseResult],
) -> CapabilityPanelEvidence:
    """Build complete paper or broad evidence from exactly paired raw case results."""

    if candidate_artifact_id == baseline_artifact_id:
        raise ValueError("candidate and baseline artifacts must be distinct")
    if not candidate_results or not baseline_results:
        raise ValueError("capability panels require candidate and baseline results")

    def index(
        arm: str, artifact_id: str, results: Sequence[CapabilityCaseResult]
    ) -> dict[tuple[str, str], CapabilityCaseResult]:
        indexed: dict[tuple[str, str], CapabilityCaseResult] = {}
        for item in results:
            if item.artifact_id != artifact_id:
                raise ValueError(f"{arm} result is attributed to the wrong artifact")
            if item.protocol_id != protocol.protocol_id or item.evaluator_revision != protocol.evaluator_revision:
                raise ValueError(f"{arm} result does not match the frozen evaluator protocol")
            if item.benchmark_id in protocol.deterministic_benchmarks:
                if item.decoding != protocol.decoding or not item.decoding.deterministic:
                    raise ValueError(f"{item.benchmark_id} must retain deterministic decoding")
            key = (item.benchmark_id, item.case_id)
            if key in indexed:
                raise ValueError(f"duplicate {arm} panel case: {key}")
            indexed[key] = item
        return indexed

    candidate_index = index("candidate", candidate_artifact_id, candidate_results)
    baseline_index = index("baseline", baseline_artifact_id, baseline_results)
    if set(candidate_index) != set(baseline_index):
        raise ValueError("candidate and baseline panel cases must be exactly paired")
    observed = {benchmark for benchmark, _ in candidate_index}
    missing = set(protocol.required_benchmarks) - observed
    if missing:
        raise ValueError(f"panel is missing required benchmark results: {sorted(missing)}")

    pair_records: list[PairedFailureRecord] = []
    candidate_observations: list[ScoreObservation] = []
    baseline_observations: list[ScoreObservation] = []
    candidate_raw: list[tuple[str, str, str]] = []
    baseline_raw: list[tuple[str, str, str]] = []
    for benchmark_id, case_id in sorted(candidate_index):
        candidate = candidate_index[(benchmark_id, case_id)]
        baseline = baseline_index[(benchmark_id, case_id)]
        if (
            candidate.semantic_family_id != baseline.semantic_family_id
            or candidate.prompt_ref != baseline.prompt_ref
            or candidate.raw_prompt != baseline.raw_prompt
        ):
            raise ValueError(f"prompt or semantic-family mismatch for paired case {benchmark_id}/{case_id}")
        status = _pair_status(candidate.passed, baseline.passed)
        pair_records.append(PairedFailureRecord(
            benchmark_id, case_id, candidate.semantic_family_id, status,
            candidate.failure_categories, baseline.failure_categories,
        ))
        paired_case_id = f"{benchmark_id}/{case_id}"
        candidate_observations.append(ScoreObservation(
            paired_case_id, candidate.semantic_family_id, "panel_macro", candidate.score
        ))
        baseline_observations.append(ScoreObservation(
            paired_case_id, baseline.semantic_family_id, "panel_macro", baseline.score
        ))
        candidate_raw.append((benchmark_id, case_id, candidate.raw_output))
        baseline_raw.append((benchmark_id, case_id, baseline.raw_output))

    plan = BootstrapPlan(seed=protocol.bootstrap_seed)
    candidate_scores: dict[str, float] = {}
    baseline_scores: dict[str, float] = {}
    deltas: dict[str, float] = {}
    intervals: dict[str, tuple[float, float]] = {}
    for benchmark_id in protocol.required_benchmarks:
        candidate_subset = tuple(
            ScoreObservation(item.case_id, item.semantic_family_id, benchmark_id, item.score)
            for item in candidate_results if item.benchmark_id == benchmark_id
        )
        baseline_subset = tuple(
            ScoreObservation(item.case_id, item.semantic_family_id, benchmark_id, item.score)
            for item in baseline_results if item.benchmark_id == benchmark_id
        )
        candidate_scores[benchmark_id] = fmean(item.score for item in candidate_subset)
        baseline_scores[benchmark_id] = fmean(item.score for item in baseline_subset)
        stats = paired_bootstrap_statistics(candidate_subset, baseline_subset, benchmark_id, plan)
        deltas[benchmark_id] = stats.paired_delta
        intervals[benchmark_id] = (stats.confidence_lower, stats.confidence_upper)
    aggregate_stats = paired_bootstrap_statistics(
        candidate_observations, baseline_observations, "panel_macro", plan
    )
    candidate_macro = fmean(candidate_scores.values())
    baseline_macro = fmean(baseline_scores.values())
    deltas["macro"] = candidate_macro - baseline_macro
    intervals["macro"] = (
        aggregate_stats.confidence_lower, aggregate_stats.confidence_upper
    )
    failure_categories = tuple(
        f"{item.benchmark_id}:{item.case_id}:{item.status.value}"
        for item in pair_records if item.status is not PairedFailureStatus.BOTH_PASS
    )
    return CapabilityPanelEvidence(
        protocol, candidate_artifact_id, baseline_artifact_id,
        MappingProxyType(candidate_scores), MappingProxyType(baseline_scores),
        MappingProxyType({"macro": candidate_macro, "baseline_macro": baseline_macro}),
        MappingProxyType(deltas), MappingProxyType(intervals), tuple(pair_records),
        failure_categories, tuple(candidate_raw), tuple(baseline_raw),
    )


class PerplexityCoordinator:
    def coordinate(
        self,
        candidate: PerplexityRun,
        dense: PerplexityRun,
        *,
        protocol: TokenizationSequenceProtocol,
    ) -> PerplexityEvidence:
        return coordinate_perplexity(candidate, dense, protocol=protocol)


class PaperPanelCoordinator:
    def coordinate(
        self,
        protocol: CapabilityPanelProtocol,
        candidate_artifact_id: str,
        dense_artifact_id: str,
        candidate_results: Sequence[CapabilityCaseResult],
        dense_results: Sequence[CapabilityCaseResult],
    ) -> CapabilityPanelEvidence:
        if protocol.kind is not CapabilityPanelKind.PAPER:
            raise ValueError("paper-panel coordinator requires a paper protocol")
        return coordinate_capability_panel(
            protocol, candidate_artifact_id, dense_artifact_id,
            candidate_results, dense_results,
        )


class BroadPanelCoordinator:
    def coordinate(
        self,
        protocol: CapabilityPanelProtocol,
        candidate_artifact_id: str,
        baseline_artifact_id: str,
        candidate_results: Sequence[CapabilityCaseResult],
        baseline_results: Sequence[CapabilityCaseResult],
    ) -> CapabilityPanelEvidence:
        if protocol.kind is not CapabilityPanelKind.BROAD:
            raise ValueError("broad-panel coordinator requires a broad protocol")
        return coordinate_capability_panel(
            protocol, candidate_artifact_id, baseline_artifact_id,
            candidate_results, baseline_results,
        )


@dataclass(frozen=True, slots=True)
class SandboxPolicy:
    policy_id: str
    executor_revision: str
    image_digest: str
    isolation_backend: str
    network_access: bool
    read_only_root: bool
    cpu_seconds: int
    wall_seconds: int
    memory_bytes: int
    max_output_bytes: int
    allowed_languages: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("policy_id", "executor_revision", "image_digest", "isolation_backend"):
            _text(name, getattr(self, name))
        if self.network_access:
            raise ValueError("code-evaluation sandboxes must deny network access")
        if not self.read_only_root:
            raise ValueError("code-evaluation sandboxes require a read-only root")
        if min(self.cpu_seconds, self.wall_seconds, self.memory_bytes, self.max_output_bytes) < 1:
            raise ValueError("sandbox resource limits must be positive")
        if self.wall_seconds < self.cpu_seconds:
            raise ValueError("sandbox wall limit cannot be below its CPU limit")
        if not self.allowed_languages:
            raise ValueError("sandbox must explicitly allow at least one language")
        _unique("allowed languages", self.allowed_languages)


@dataclass(frozen=True, slots=True)
class CodeCase:
    case_id: str
    semantic_family_id: str
    language: str
    prompt_ref: str
    tests_ref: str

    def __post_init__(self) -> None:
        for name in ("case_id", "semantic_family_id", "language", "prompt_ref", "tests_ref"):
            _text(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class CodeSubmission:
    case_id: str
    source: str
    raw_output_ref: str

    def __post_init__(self) -> None:
        _text("case_id", self.case_id)
        _text("raw_output_ref", self.raw_output_ref)


@dataclass(frozen=True, slots=True)
class SandboxExecutionResult:
    case_id: str
    policy_id: str
    executor_revision: str
    image_digest: str
    isolated: bool
    network_observed: bool
    timed_out: bool
    exit_code: int | None
    tests_passed: int
    tests_total: int
    stdout_ref: str
    stderr_ref: str
    failure_categories: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "case_id", "policy_id", "executor_revision", "image_digest",
            "stdout_ref", "stderr_ref",
        ):
            _text(name, getattr(self, name))
        if self.tests_total < 1 or not 0 <= self.tests_passed <= self.tests_total:
            raise ValueError("sandbox test counts are invalid")
        _unique("sandbox failure categories", self.failure_categories)

    @property
    def passed(self) -> bool:
        return (
            self.isolated and not self.network_observed and not self.timed_out
            and self.exit_code == 0 and self.tests_passed == self.tests_total
            and not self.failure_categories
        )


class SandboxExecutor(Protocol):
    def execute(
        self,
        case: CodeCase,
        submission: CodeSubmission,
        policy: SandboxPolicy,
    ) -> SandboxExecutionResult: ...


@dataclass(frozen=True, slots=True)
class CodeExecutionEvidence:
    artifact_id: str
    policy: SandboxPolicy
    pass_rate: float
    results: tuple[SandboxExecutionResult, ...]
    failure_categories: tuple[str, ...]
    raw_output_refs: tuple[str, ...]


class CodeEvidenceCoordinator:
    """Run untrusted submissions only through a policy-attesting sandbox port."""

    def evaluate(
        self,
        *,
        artifact_id: str,
        cases: Sequence[CodeCase],
        submissions: Sequence[CodeSubmission],
        policy: SandboxPolicy,
        executor: SandboxExecutor,
    ) -> CodeExecutionEvidence:
        _text("artifact_id", artifact_id)
        if not cases:
            raise ValueError("code execution panels require cases")
        case_index = {item.case_id: item for item in cases}
        submission_index = {item.case_id: item for item in submissions}
        if len(case_index) != len(cases) or len(submission_index) != len(submissions):
            raise ValueError("code cases and submissions must have unique ids")
        if set(case_index) != set(submission_index):
            raise ValueError("every code case requires exactly one submission")
        results: list[SandboxExecutionResult] = []
        failure_categories: list[str] = []
        for case_id in sorted(case_index):
            case = case_index[case_id]
            if case.language not in policy.allowed_languages:
                raise ValueError(f"language is not permitted by the sandbox policy: {case.language}")
            result = executor.execute(case, submission_index[case_id], policy)
            if result.case_id != case_id:
                raise ValueError("sandbox result is attributed to the wrong case")
            if (
                result.policy_id != policy.policy_id
                or result.executor_revision != policy.executor_revision
                or result.image_digest != policy.image_digest
            ):
                raise ValueError("sandbox result does not attest the frozen execution policy")
            if not result.isolated or result.network_observed:
                raise ValueError("sandbox isolation or network denial was not enforced")
            results.append(result)
            if not result.passed:
                labels = result.failure_categories or (
                    "timeout" if result.timed_out else "execution_or_test_failure",
                )
                failure_categories.extend(f"code_execution:{case_id}:{label}" for label in labels)
        return CodeExecutionEvidence(
            artifact_id, policy, sum(item.passed for item in results) / len(results),
            tuple(results), tuple(failure_categories),
            tuple(submission_index[item].raw_output_ref for item in sorted(submission_index)),
        )


class RubricPreference(StrEnum):
    RESPONSE_A = "response_a"
    RESPONSE_B = "response_b"
    TIE = "tie"


@dataclass(frozen=True, slots=True)
class RubricProtocol:
    protocol_id: str
    rubric_revision: str
    evaluation_set_id: str
    evaluation_panel: EvaluationPanel
    required_panels: tuple[str, ...]
    seed: int
    judge_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("protocol_id", "rubric_revision", "evaluation_set_id"):
            _text(name, getattr(self, name))
        _unique("rubric panels", self.required_panels)
        _unique("rubric judges", self.judge_ids)
        if set(RUBRIC_PANELS) - set(self.required_panels):
            raise ValueError("rubric protocol is missing a required release panel")
        if not self.judge_ids or any(not item.strip() for item in self.judge_ids):
            raise ValueError("rubric protocol requires explicit judge identities")


@dataclass(frozen=True, slots=True)
class RubricCase:
    case_id: str
    panel_id: str
    semantic_family_id: str
    prompt_ref: str
    prompt: str
    candidate_output: str
    baseline_output: str

    def __post_init__(self) -> None:
        for name in ("case_id", "panel_id", "semantic_family_id", "prompt_ref", "prompt"):
            _text(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class BlindedRubricPair:
    pair_id: str
    panel_id: str
    prompt_ref: str
    prompt: str
    response_a: str
    response_b: str
    rubric_revision: str


@dataclass(frozen=True, slots=True)
class RubricJudgment:
    pair_id: str
    judge_id: str
    preference: RubricPreference
    rationale_ref: str
    response_a_score: float
    response_b_score: float
    failure_categories: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("pair_id", "judge_id", "rationale_ref"):
            _text(name, getattr(self, name))
        object.__setattr__(self, "response_a_score", _score(self.response_a_score))
        object.__setattr__(self, "response_b_score", _score(self.response_b_score))
        _unique("rubric failure categories", self.failure_categories)


class RubricJudge(Protocol):
    @property
    def judge_id(self) -> str: ...

    def judge(self, pair: BlindedRubricPair) -> RubricJudgment: ...


@dataclass(frozen=True, slots=True)
class RubricPairAudit:
    pair_id: str
    case_id: str
    panel_id: str
    semantic_family_id: str
    candidate_label: RubricPreference
    judgments: tuple[RubricJudgment, ...]
    disagreement: bool
    candidate_wins: int
    baseline_wins: int
    ties: int


@dataclass(frozen=True, slots=True)
class RubricEvidence:
    protocol: RubricProtocol
    candidate_artifact_id: str
    baseline_artifact_id: str
    panel_scores: Mapping[str, float]
    baseline_panel_scores: Mapping[str, float]
    candidate_minus_baseline_deltas: Mapping[str, float]
    pair_audits: tuple[RubricPairAudit, ...]
    disagreements: tuple[RubricPairAudit, ...]
    failure_categories: tuple[str, ...]
    randomized_order_hash: str

    @property
    def complete(self) -> bool:
        return (
            set(self.protocol.required_panels) <= set(self.panel_scores)
            and bool(self.pair_audits)
            and all(item.judgments for item in self.pair_audits)
        )


class RubricEvidenceCoordinator:
    """Blind identities before judging and retain every individual disagreement."""

    def evaluate(
        self,
        *,
        protocol: RubricProtocol,
        candidate_artifact_id: str,
        baseline_artifact_id: str,
        cases: Sequence[RubricCase],
        judges: Sequence[RubricJudge],
    ) -> RubricEvidence:
        for name, value in (
            ("candidate_artifact_id", candidate_artifact_id),
            ("baseline_artifact_id", baseline_artifact_id),
        ):
            _text(name, value)
        if candidate_artifact_id == baseline_artifact_id:
            raise ValueError("candidate and rubric baseline artifacts must be distinct")
        if not cases:
            raise ValueError("rubric panels require cases")
        case_ids = tuple(item.case_id for item in cases)
        _unique("rubric case ids", case_ids)
        observed_panels = {item.panel_id for item in cases}
        missing_panels = set(protocol.required_panels) - observed_panels
        if missing_panels:
            raise ValueError(f"rubric evidence is missing required panels: {sorted(missing_panels)}")
        judge_index = {item.judge_id: item for item in judges}
        if len(judge_index) != len(judges) or set(judge_index) != set(protocol.judge_ids):
            raise ValueError("judges must exactly match preregistered blinded judge identities")

        generator = random.Random(protocol.seed)
        audits: list[RubricPairAudit] = []
        order_manifest: list[str] = []
        failure_categories: list[str] = []
        for case in sorted(cases, key=lambda item: item.case_id):
            candidate_first = bool(generator.getrandbits(1))
            pair_id = hashlib.sha256(
                f"{protocol.protocol_id}\0{protocol.seed}\0{case.case_id}".encode("utf-8")
            ).hexdigest()
            if candidate_first:
                response_a, response_b = case.candidate_output, case.baseline_output
                candidate_label = RubricPreference.RESPONSE_A
            else:
                response_a, response_b = case.baseline_output, case.candidate_output
                candidate_label = RubricPreference.RESPONSE_B
            pair = BlindedRubricPair(
                pair_id, case.panel_id, case.prompt_ref, case.prompt,
                response_a, response_b, protocol.rubric_revision,
            )
            judgments: list[RubricJudgment] = []
            for judge_id in protocol.judge_ids:
                judgment = judge_index[judge_id].judge(pair)
                if judgment.pair_id != pair_id or judgment.judge_id != judge_id:
                    raise ValueError("rubric judgment attribution does not match its blind pair")
                judgments.append(judgment)
                failure_categories.extend(
                    f"rubric:{case.panel_id}:{case.case_id}:{category}"
                    for category in judgment.failure_categories
                )
            preferences = {item.preference for item in judgments}
            disagreement = len(preferences) > 1
            candidate_wins = sum(item.preference is candidate_label for item in judgments)
            baseline_label = (
                RubricPreference.RESPONSE_B
                if candidate_label is RubricPreference.RESPONSE_A
                else RubricPreference.RESPONSE_A
            )
            baseline_wins = sum(item.preference is baseline_label for item in judgments)
            ties = sum(item.preference is RubricPreference.TIE for item in judgments)
            audits.append(RubricPairAudit(
                pair_id, case.case_id, case.panel_id, case.semantic_family_id,
                candidate_label, tuple(judgments), disagreement,
                candidate_wins, baseline_wins, ties,
            ))
            order_manifest.append(f"{case.case_id}:{candidate_label.value}")

        panel_scores: dict[str, float] = {}
        baseline_scores: dict[str, float] = {}
        deltas: dict[str, float] = {}
        for panel_id in protocol.required_panels:
            selected = tuple(item for item in audits if item.panel_id == panel_id)
            denominator = sum(len(item.judgments) for item in selected)
            candidate_score = (
                sum(item.candidate_wins + 0.5 * item.ties for item in selected) / denominator
            )
            baseline_score = (
                sum(item.baseline_wins + 0.5 * item.ties for item in selected) / denominator
            )
            panel_scores[panel_id] = candidate_score
            baseline_scores[panel_id] = baseline_score
            deltas[panel_id] = candidate_score - baseline_score
        order_hash = hashlib.sha256("\n".join(order_manifest).encode("utf-8")).hexdigest()
        disagreement_records = tuple(item for item in audits if item.disagreement)
        return RubricEvidence(
            protocol, candidate_artifact_id, baseline_artifact_id,
            MappingProxyType(panel_scores), MappingProxyType(baseline_scores),
            MappingProxyType(deltas), tuple(audits), disagreement_records,
            tuple(dict.fromkeys(failure_categories)), order_hash,
        )
