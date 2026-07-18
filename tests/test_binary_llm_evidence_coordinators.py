from __future__ import annotations

from dataclasses import replace

import pytest

from binary_llm.orchestration import (
    BROAD_BENCHMARKS,
    PAPER_ZERO_SHOT_BENCHMARKS,
    RUBRIC_PANELS,
    BroadPanelCoordinator,
    CapabilityCaseResult,
    CapabilityPanelKind,
    CapabilityPanelProtocol,
    CodeCase,
    CodeEvidenceCoordinator,
    CodeSubmission,
    DecodingProtocol,
    EvaluationPanel,
    PairedFailureStatus,
    PaperPanelCoordinator,
    PerplexityCaseResult,
    PerplexityCoordinator,
    PerplexityRun,
    RubricCase,
    RubricEvidenceCoordinator,
    RubricJudgment,
    RubricPreference,
    RubricProtocol,
    SandboxExecutionResult,
    SandboxPolicy,
    TokenizationSequenceProtocol,
)
from binary_llm.reporting import panel_evaluation_from_capability_evidence


def decoding() -> DecodingProtocol:
    return DecodingProtocol(0.0, False, 64, 7)


def token_protocol() -> TokenizationSequenceProtocol:
    return TokenizationSequenceProtocol(
        "tok-protocol", "tokenizer", "revision", "sha256", 128, 64,
        True, 1, 2, "right", "none", "next_token_non_padding",
    )


def perplexity_run(
    artifact_id: str,
    *,
    protocol: TokenizationSequenceProtocol | None = None,
    token_hash: str = "tokens",
    loss: float = 4.0,
) -> PerplexityRun:
    return PerplexityRun(
        artifact_id, "held-out", protocol or token_protocol(),
        (
            PerplexityCaseResult("wiki", "wikitext2", token_hash, "bounds", 2, loss),
            PerplexityCaseResult("ptb", "ptb", "ptb-tokens", "ptb-bounds", 2, 2.0),
        ),
        f"raw-{artifact_id}",
    )


def test_perplexity_requires_identical_tokenization_sequences_and_reports_dense_pair() -> None:
    evidence = PerplexityCoordinator().coordinate(
        perplexity_run("candidate"), perplexity_run("dense", loss=2.0),
        protocol=token_protocol(),
    )

    assert evidence.candidate_perplexity == pytest.approx(4.4816890703)
    assert evidence.dense_perplexity == pytest.approx(2.7182818284)
    assert set(evidence.candidate_by_corpus) == {"ptb", "wikitext2"}
    assert evidence.raw_output_refs == ("raw-candidate", "raw-dense")

    with pytest.raises(ValueError, match="tokenization or sequence mismatch"):
        PerplexityCoordinator().coordinate(
            perplexity_run("candidate"), perplexity_run("dense", token_hash="different"),
            protocol=token_protocol(),
        )


def panel_protocol(kind: CapabilityPanelKind) -> CapabilityPanelProtocol:
    benchmarks = (
        PAPER_ZERO_SHOT_BENCHMARKS
        if kind is CapabilityPanelKind.PAPER else BROAD_BENCHMARKS
    )
    return CapabilityPanelProtocol(
        f"{kind.value}-protocol", kind, EvaluationPanel.EXTERNAL, "eval-v1",
        f"{kind.value}-set", benchmarks, benchmarks, decoding(), 19,
    )


def panel_results(
    artifact_id: str,
    protocol: CapabilityPanelProtocol,
    *,
    fail_benchmark: str | None = None,
) -> tuple[CapabilityCaseResult, ...]:
    return tuple(
        CapabilityCaseResult(
            artifact_id, protocol.protocol_id, protocol.evaluator_revision,
            benchmark, f"{benchmark}-1", f"family-{benchmark}",
            f"prompt-{benchmark}", f"Question for {benchmark}",
            f"Output from {artifact_id}", 0.0 if benchmark == fail_benchmark else 1.0,
            benchmark != fail_benchmark, decoding(),
            ("wrong_answer",) if benchmark == fail_benchmark else (),
        )
        for benchmark in protocol.required_benchmarks
    )


def test_paper_panel_pairs_dense_results_and_retains_every_failure() -> None:
    protocol = panel_protocol(CapabilityPanelKind.PAPER)
    evidence = PaperPanelCoordinator().coordinate(
        protocol, "candidate", "dense",
        panel_results("candidate", protocol, fail_benchmark="boolq"),
        panel_results("dense", protocol),
    )

    assert evidence.complete
    assert set(evidence.per_panel_scores) == set(PAPER_ZERO_SHOT_BENCHMARKS)
    assert len(evidence.paired_failures) == len(PAPER_ZERO_SHOT_BENCHMARKS)
    failed = next(item for item in evidence.paired_failures if item.benchmark_id == "boolq")
    assert failed.status is PairedFailureStatus.CANDIDATE_ONLY
    assert failed.candidate_failure_categories == ("wrong_answer",)
    assert evidence.failure_categories == (
        "boolq:boolq-1:candidate_only_failure",
    )
    report_panel = panel_evaluation_from_capability_evidence(
        evidence, evidence_id="paper-evidence"
    )
    assert report_panel.paired_failure_categories == evidence.failure_categories
    assert report_panel.panel is EvaluationPanel.EXTERNAL


def test_broad_panel_is_complete_and_deterministic_decoding_cannot_drift() -> None:
    protocol = panel_protocol(CapabilityPanelKind.BROAD)
    evidence = BroadPanelCoordinator().coordinate(
        protocol, "candidate", "bf16",
        panel_results("candidate", protocol), panel_results("bf16", protocol),
    )
    assert evidence.complete
    assert set(evidence.per_panel_scores) == set(BROAD_BENCHMARKS)
    drifted = list(panel_results("candidate", protocol))
    drifted[0] = replace(drifted[0], decoding=DecodingProtocol(0.7, True, 64, 7))
    with pytest.raises(ValueError, match="deterministic decoding"):
        BroadPanelCoordinator().coordinate(
            protocol, "candidate", "bf16", drifted, panel_results("bf16", protocol)
        )


def sandbox_policy() -> SandboxPolicy:
    return SandboxPolicy(
        "sandbox-v1", "executor-v1", "sha256:image", "container",
        False, True, 1, 2, 64_000_000, 1_000_000, ("python",),
    )


class Executor:
    def __init__(self, *, isolated: bool = True, exit_code: int = 0) -> None:
        self.isolated = isolated
        self.exit_code = exit_code

    def execute(
        self, case: CodeCase, submission: CodeSubmission, policy: SandboxPolicy
    ) -> SandboxExecutionResult:
        return SandboxExecutionResult(
            case.case_id, policy.policy_id, policy.executor_revision, policy.image_digest,
            self.isolated, False, False, self.exit_code,
            int(self.exit_code == 0), 1, f"stdout-{case.case_id}", f"stderr-{case.case_id}",
        )


def test_code_checks_require_sandbox_attestation_and_retain_per_case_failures() -> None:
    cases = (
        CodeCase("pass", "family-a", "python", "prompt-pass", "tests-pass"),
        CodeCase("fail", "family-b", "python", "prompt-fail", "tests-fail"),
    )
    submissions = (
        CodeSubmission("pass", "print(1)", "raw-pass"),
        CodeSubmission("fail", "raise RuntimeError", "raw-fail"),
    )

    class MixedExecutor(Executor):
        def execute(self, case, submission, policy):
            self.exit_code = 1 if case.case_id == "fail" else 0
            return super().execute(case, submission, policy)

    evidence = CodeEvidenceCoordinator().evaluate(
        artifact_id="candidate", cases=cases, submissions=submissions,
        policy=sandbox_policy(), executor=MixedExecutor(),
    )
    assert evidence.pass_rate == 0.5
    assert evidence.failure_categories == (
        "code_execution:fail:execution_or_test_failure",
    )
    assert len(evidence.results) == 2

    with pytest.raises(ValueError, match="isolation"):
        CodeEvidenceCoordinator().evaluate(
            artifact_id="candidate", cases=cases[:1], submissions=submissions[:1],
            policy=sandbox_policy(), executor=Executor(isolated=False),
        )
    with pytest.raises(ValueError, match="deny network"):
        replace(sandbox_policy(), network_access=True)


class Judge:
    def __init__(self, judge_id: str, preference: RubricPreference) -> None:
        self._judge_id = judge_id
        self.preference = preference
        self.seen = []

    @property
    def judge_id(self) -> str:
        return self._judge_id

    def judge(self, pair):
        self.seen.append(pair)
        return RubricJudgment(
            pair.pair_id, self.judge_id, self.preference,
            f"rationale-{self.judge_id}-{pair.pair_id}", 0.8, 0.2,
        )


def rubric_cases() -> tuple[RubricCase, ...]:
    return tuple(
        RubricCase(
            f"case-{panel}", panel, f"family-{panel}", f"prompt-{panel}",
            f"Prompt for {panel}", f"candidate {panel}", f"baseline {panel}",
        )
        for panel in RUBRIC_PANELS
    )


def test_rubric_pairing_is_blinded_seeded_and_preserves_judge_disagreement() -> None:
    protocol = RubricProtocol(
        "rubric-v1", "rubric-revision", "broad-set", EvaluationPanel.EXTERNAL,
        RUBRIC_PANELS, 31, ("judge-a", "judge-b"),
    )
    judge_a = Judge("judge-a", RubricPreference.RESPONSE_A)
    judge_b = Judge("judge-b", RubricPreference.RESPONSE_B)
    coordinator = RubricEvidenceCoordinator()
    evidence = coordinator.evaluate(
        protocol=protocol, candidate_artifact_id="candidate", baseline_artifact_id="bf16",
        cases=rubric_cases(), judges=(judge_a, judge_b),
    )

    assert evidence.complete
    assert len(evidence.disagreements) == len(RUBRIC_PANELS)
    assert all(len(item.judgments) == 2 for item in evidence.pair_audits)
    assert all(not hasattr(pair, "artifact_id") for pair in judge_a.seen)
    assert all(not hasattr(pair, "candidate_label") for pair in judge_a.seen)
    repeated = coordinator.evaluate(
        protocol=protocol, candidate_artifact_id="candidate", baseline_artifact_id="bf16",
        cases=tuple(reversed(rubric_cases())),
        judges=(Judge("judge-a", RubricPreference.RESPONSE_A),
                Judge("judge-b", RubricPreference.RESPONSE_B)),
    )
    assert repeated.randomized_order_hash == evidence.randomized_order_hash
    assert tuple(item.candidate_label for item in repeated.pair_audits) == tuple(
        item.candidate_label for item in evidence.pair_audits
    )
