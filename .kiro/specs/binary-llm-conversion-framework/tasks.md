# BinaryLLM Conversion Framework — Evidence-Aware Roadmap

Read first: [status](status.md), [requirements](requirements.md), [design](design.md), and the
[authoritative paper](../../../docs/research-papers/2508.06974v2-binaryLLM.md), SHA-256
`8e346e97f4d172b93c04d3ad240eb32d599a9f8924290ffd0f486661bafeb8f6`.

The paper is the sole authority for BinaryLLM claims. Requirements/design add framework
obligations, not paper facts. This file requires no conversation context.

## Conventions

- `[x]`: verified complete for the exact stated scope.
- `[~]`: partial or verified only at narrower scope.
- `[ ]`: pending.
- `[ ] BLOCKED:` prohibited until named prerequisites exist.

Top-level completion requires every mandatory subtask, output, test, and evidence gate.
Implementation, test verification, scientific execution, promotion, and release are distinct.
All test tasks—including all 33 Hypothesis properties—are mandatory before scientific
promotion; none is optional or skippable.

Current baseline: 304 BinaryLLM tests passed in 32.81s and the scoped Ruff check passed on
2026-07-18; 33/33 required properties; zero model-scale, whole-model, or device runs. The CLI only has `catalog` and
`validate`. The worktree is heavily dirty/untracked; never discard unrelated work.

Every task below states purpose, evidence, files/symbols, outputs, verification, exit criteria,
traceability, prerequisites, and non-goals. On completion, update [status](status.md) with date,
exact commands/results, files, and durable evidence IDs.

## Wave 0 — Close deterministic foundations

- [~] 1. Domain truth and static-quality closure
  - **Purpose:** make paper facts, paper ambiguities, and framework extensions unambiguous.
  - **Evidence:** canonical records/serialization/identity/errors and
    `domain/paper.py::BINARYLLM_PAPER_LEDGER` exist; paper/domain/identity/reproduction example
    tests pass. The 18 final-audit static findings were fixed without suppressions; the scoped
    Ruff check and all 209 BinaryLLM tests passed on 2026-07-18.
  - **Files/symbols:** `src/binary_llm/domain/{paper,models,identity,serialization,
    ambiguities,reproduction,validation,errors}.py`, `pyproject.toml`, corresponding tests.
  - **Outputs:** hash-bound paper ledger; complete source classification; clean scoped static
    report; no silent scientific defaults.
  - **Verification:** paper/domain tests; configured Ruff/type checks; Properties 1, 2, 6, 9,
    14, 20 under Task 8.
  - **Exit:** zero scoped static findings; preserved paper bytes verify; every material
    difference has an ablation reference.
  - **Traceability:** Requirements 1, 6, 11, 19; Design Properties 1, 2, 6, 9, 14, 20.
  - **Prerequisites:** none. **Non-goal:** no local reproduction claim.
  - [x] 1.1 Canonical domain models, identities, schema evolution, and typed failures are
    example-tested.
  - [x] 1.2 Paper ledger/reference classification, path, and hash are directly tested.
  - [~] 1.3 Validation exists; generated property coverage remains absent.
  - [x] 1.4 All 18 static findings were fixed and documented without blanket suppression
    (`All checks passed!`; `209 passed in 7.90s`).

- [~] 2. Seals, lineage, durable registry, and evidence resolver
  - **Purpose:** make evidence independently reopenable and attributable across processes.
  - **Evidence:** `SealedInventoryVerifier` and `AppendOnlyExperimentRegistry` verify fixtures,
    lineage, complete checkpoints, supersession, resources, and reruns. The
    `FilesystemArtifactStore` now durably resolves exact payloads and canonical immutable
    `ArtifactRef` metadata across fresh instances; registry state is still only in memory.
  - **Files/symbols:** `orchestration/{sealing,registry}.py`,
    `ArtifactRef`, new store/resolver modules and integration tests.
  - **Outputs:** atomic filesystem content-addressed store; append-only log; reopen/reindex;
    owner/parent/producer/hash validation; corruption and interrupted-write diagnostics.
  - **Verification:** fresh-process resolution, category mutation, idempotency/concurrency,
    orphan/sideways-lineage rejection; Properties 3, 20, 33.
  - **Exit:** every manifest, checkpoint, raw output, metric, packed file, and report resolves
    by hash after restart; corruption fails closed.
  - **Traceability:** Requirements 2, 11, 12.7, 21.6–21.7; Properties 3, 20, 33.
  - **Prerequisites:** Task 1. **Non-goal:** in-memory records are not durable evidence.
  - [x] 2.1 Fixture seals and independent baseline identities are example-tested.
  - [x] 2.2 In-memory append-only registry semantics are example-tested.
  - [x] 2.3 Atomic content-addressed artifact payload/reference storage, restart resolution,
    exact owner/lineage expectations, and corruption failures are verified.
  - [x] 2.4 Hash-chained journal replay, process-boundary continuation, corruption/gap/reorder
    rejection, serialized writer contention, and explicit orphan/temp/stale-lock recovery are
    verified (`38` focused tests; `243` full-suite tests).
  - [x] 2.5 Behavioral teacher identity is authorized only through a deterministic
    `VerifiedTeacherBinding` created from the complete matching seal report, exact BF16 oracle,
    and durable resolution of BF16 weights/tokenizer/template/configuration artifacts.
    Caller flags alone and post-construction identity substitution fail closed
    (`48` focused tests; `253` full-suite tests).

- [~] 3. Math core and executable paper-fidelity configuration
  - **Purpose:** make every operator/optimizer choice explicit and source-classified.
  - **Evidence:** progressive function/analytical backward, schedules, input scales, dual
    scales, signs, diagnostics, and tensor scope have example tests.
  - **Files/symbols:** `domain/paper.py::BINARYLLM_PAPER_REFERENCE_SETTINGS`,
    `math/{progressive,scales}.py`, `adapters/linear.py::BinaryLinear`,
    `orchestration/small_scale_protocols.py`.
  - **Outputs:** immutable config covering Stage 1 inverse activation/gradient/positivity,
    transition, Stage 2 trainability, scale gradients, phase origin, precision, AdamW details,
    warmup/clipping, seed/order, and export semantics; every field classified.
  - **Verification:** no-default/config-hash tests, numerical differential tests, ambiguity arm
    generation; Properties 5, 7–14.
  - **Exit:** allocation fails on any omitted material choice; exact paper and modified arms
    are queryable and classified.
  - **Traceability:** Requirements 4–8; Properties 5, 7–14.
  - **Prerequisites:** Tasks 1–2. **Non-goal:** plausible choices are not paper facts.
  - [x] 3.1 Progressive equations/schedules are example-tested.
  - [x] 3.2 Scale/sign/scope primitives are example-tested.
  - [x] 3.3 Complete executable paper-fidelity config with exact field-path provenance,
    deterministic paper-bound identity, and fail-closed missing/extra/ambiguity validation.
  - [x] 3.4 Matched ambiguity and modified-reproduction arms declare exact changed paths,
    ablation IDs, and interaction classifications; small-scale preflight binds the config hash.

- [~] 4. Gates, statistics, blind evidence, and release policy
  - **Purpose:** make stopping and promotion fail closed.
  - **Evidence:** gate/promotion state, safe boundaries, matched comparisons, 10,000 grouped
    bootstrap samples, blind history, failure slices, and size-optimal policy have examples.
  - **Files/symbols:** `orchestration/{gates,statistics}.py`, `domain/ablations.py`,
    `reporting/{evaluation,release}.py`.
  - **Outputs:** durable gate snapshots/results, threshold history, paired evidence, complete
    reasons.
  - **Verification:** exact boundaries, missing evidence, post-blind mutation, report
    completeness; Properties 4, 10, 15, 21–26, 30, 32.
  - **Exit:** missing evidence fails; no aggregate hides a slice; post-blind tuning invalidates
    the decision.
  - **Traceability:** Requirements 3, 7.7, 12–15, 20–21.
  - **Prerequisites:** Task 2 for promotion. **Non-goal:** policy tests prove no candidate.
  - [x] 4.1 Pure policy/statistics primitives are example-tested.
  - [ ] 4.2 Persist inputs, results, and temporal history.
  - [ ] 4.3 Complete mandatory generated properties in Task 8.

## Wave 1 — Exact training fidelity and failure recovery

- [x] 5. Stage 1 inverse activation and transition artifact
  - **Purpose:** implement the paper’s Stage 1 and prove its Stage 2 parent.
  - **Evidence:** tiny trainer freezes dense weights, optimizes scales, supports 50 steps,
    diagnostics, budgets, no-init control, and injected failure. Exact `W*S_t^-1` with
    `S_t*A` is algebra-tested, and deterministic durable `W_tilde=W/S_t*` artifacts are
    validated before atomic application.
  - **Files/symbols:** `adapters/linear.py::{TrainingActivationMode,BinaryLinear}`,
    `orchestration/stage1.py::{Stage1TrainerBackend,Stage1Checkpoint}`, new transition type.
  - **Outputs:** exact inverse-activation arm; weight-only modified arm; dense-equivalence
    proof; content-addressed transition containing transformed tensors, parent/config,
    diagnostics, and hashes.
  - **Verification:** transformed equality, activation equivalence, 50-step/short tests,
    non-finite persistence; Properties 7, 8, 10.
  - **Exit:** Stage 2 only accepts verified transition; exact/modified arms are distinct.
  - **Traceability:** Requirements 5, 6, 11. **Prerequisites:** Tasks 2–3.
  - **Non-goal:** do not discard inverse activation because weight-only runs.
  - [x] 5.1 Tiny scale-only trainer/control is example-tested.
  - [x] 5.2 Exact inverse-activation algebra is verified; weight-only is a distinct modified
    transition kind and cannot initialize reference Stage 2.
  - [x] 5.3 Deterministic `W_tilde=W/S_t*` artifacts emit/reload through the durable store,
    validate all module tensors/config/lineage before mutation, reset Stage 1 scales to exact
    identity, and are mandatory for fresh Stage 2. Checkpoint schema v2 retains and validates
    the immutable Stage 1 root on every resume (`26` focused tests; `263` full-suite tests).

- [x] 6. Stage 2 trainability, exact resume, and numerical diagnostics
  - **Purpose:** separate literal paper Stage 2 from the current body-only modification.
  - **Evidence:** phase plans, checkpoint schema v2, budgets, progressive/sign views, literal
    all-parameter and modified body-only arms, exact two-phase disk resume, and durable
    diagnostic-only numerical failures are tested.
  - **Files/symbols:** `orchestration/{progressive_trainer,progressive_state}.py`, adapters,
    failure types/tests.
  - **Outputs:** all-parameter reference arm; body-only modified arm; matched ablation; exact
    uninterrupted/resumed evidence; diagnostic artifact with batch, model/latent, optimizer,
    RNG, schedule, cursor, tensor/gradient summaries, usage, exception, hashes.
  - **Verification:** parameter-delta inventory; two-phase exact/toleranced comparison;
    injected NaN/Inf at forward/loss/gradient/scale; restart resume.
  - **Exit:** literal arm matches paper statement; modified arm labeled; no repeated/skipped
    data/cost; partial failure state is diagnostic-only.
  - **Traceability:** Requirements 3, 6–8, 11–12, 15; Properties 8, 10, 12–13, 15, 20, 26.
  - **Prerequisites:** Tasks 2, 5. **Non-goal:** no dense-then-quantize recovery.
  - [x] 6.1 Phase/checkpoint/body-only/budget primitives are example-tested.
  - [x] 6.2 All-parameter reference and binary-body-only modified arms have deterministic
    alias-aware parameter inventories and exact fidelity/ablation classification.
  - [x] 6.3 Uninterrupted and checkpoint-v2 disk-resumed two-phase CPU runs are exactly
    equivalent across model/optimizer hashes, RNG, cursor, budgets, metrics, and final views.
  - [x] 6.4 Non-finite partial phases emit durable diagnostic-only content-addressed artifacts,
    retain only the last complete checkpoint, and cannot resume or promote
    (`23` focused acceptance tests; `266` canonical full-suite tests).

- [x] 7. Corpus freezing, recovery, and teacher integrity
  - **Purpose:** guarantee provenance/isolation and binary-active recovery.
  - **Evidence:** content-addressed corpus manifests, transitive dedup/grouping, deny lists,
    token allocation/partitions, binary-active recovery, verified teacher routing, shared
    budgets, and durable fixture corpus/recovery evidence are example-tested.
  - **Files/symbols:** `orchestration/corpus.py::CorpusService`,
    `orchestration/recovery.py::{TeacherRouter,RecoveryTrainerBackend}`, data adapters.
  - **Outputs:** durable corpus/dedup/isolation reports; sealed teacher resolution; initial and
    alternate token mixes; no-progress evidence; recovery checkpoint lineage.
  - **Verification:** leakage/sibling/fuzzy cases, token accounting, missing terms, teacher
    substitution rejection, binary-forward audit; Properties 16–19.
  - **Exit:** every record has provenance; frozen siblings cannot enter; every teacher resolves
    to a seal; target operator stays active.
  - **Traceability:** Requirements 9–10; Properties 16–19.
  - **Prerequisites:** Tasks 2, 6. **Non-goal:** synthetic-only release support is invalid.
  - [x] 7.1 Corpus freezing improvements are example-tested.
  - [x] 7.2 Shared runtime budget enforcement is tested across training backends.
  - [x] 7.3 Tiny recovery/routing and resolver-created sealed BF16 teacher binding are tested.
  - [x] 7.4 Deterministic corpus bundles and recovery evidence persist/reopen exact records,
    all 20 partitions, batch tensors, teacher bindings/provenance, binary-active forwards,
    budgets/stops, and registry lineage. Substitution fails closed and fixture evidence is
    explicitly non-promotional (`47` focused tests; `270` canonical full-suite tests).

## Wave 2 — Mandatory proof suite and durable tiny slice

- [x] 8. Implement all 33 Hypothesis properties
  - **Purpose:** prove every universal invariant before compute promotion.
  - **Evidence:** Hypothesis `6.156.2` and deterministic profile are configured; exactly 33
    tagged properties use `@given` and `max_examples=100`, enforced by source inventory.
  - **Files:** dedicated property test modules under `tests/`; pure domain/math/orchestration/
    export/reporting functions as needed.
  - **Outputs:** one test per Design Property 1–33, exact comment tag
    `# Feature: binary-llm-conversion-framework, Property N: ...`,
    `@settings(max_examples=100)` or stronger, deterministic CI, minimized persisted failures.
  - **Verification:** property-only command plus full suite; inventory script proves exactly
    33 unique tags and minimum examples.
  - **Exit:** 33/33 pass; no `assume` abuse, broad suppression, trivial generators, or mocks
    presented as runtime/device evidence.
  - **Traceability:** Design Properties 1–33 and their listed requirement mappings.
  - **Prerequisites:** Tasks 1–7 APIs stable. **Non-goal:** randomized GPU/model runs.
  - [x] 8.1 Properties 1–10: claims, family, seals, promotion, scope, reproduction, Stage 1,
    non-finite, ambiguities, matched comparisons.
  - [x] 8.2 Properties 11–20: math, partitions, scales, identity, sign gates, recovery,
    provenance, release support, run evidence.
  - [x] 8.3 Properties 21–33: statistics, blind history, reports, Iris, budgets, accounting,
    packing, export, devices, release selection/report (`34` property/inventory tests;
    `304` canonical full-suite tests). Device/report properties are pure contracts only.

- [ ] 9. Complete tiny durable end-to-end vertical slice
  - **Purpose:** prove architecture integration before Pythia.
  - **Evidence:** components pass isolated tiny/example tests; no single durable run joins them.
  - **Files/symbols:** tiny adapter; orchestrator/CLI; Tasks 2, 5–8 components; integration tests.
  - **Outputs:** manifest -> seal -> Stage 1 -> transition -> two progressive phases ->
    failure/resume -> progressive/sign evaluation -> gates -> whole-tiny-model accounting ->
    export -> packed runtime parity -> durable report, all hash-resolvable.
  - **Verification:** run twice from clean temp roots; uninterrupted/resumed equality; corruption
    and budget/failure injection; zero-byte reconciliation; exact token/tool parity.
  - **Exit:** fresh process resolves complete chain; no metric copied across artifacts; every
    failed attempt retained; full tests/static checks clean.
  - **Traceability:** Requirements 2–17, 20–21; all relevant Properties.
  - **Prerequisites:** Tasks 1–8. **Non-goal:** tiny results are not scientific reproduction.
  - [ ] 9.1 Implement orchestration and durable report.
  - [ ] 9.2 Add happy-path, resume, failure, and corruption acceptance tests.

## Wave 3 — Whole-model tooling and executable small-scale workflow

- [~] 10. Whole-model export, accounting, and parity
  - **Purpose:** extend tensor/tiny primitives to every tensor and required sidecar.
  - **Evidence:** tensor ledger, deterministic packing, scalar packed linear runtime, and tiny
    parity are tested. No real whole-model inventory/export/runtime evidence exists.
  - **Files/symbols:** `export/{accounting,packed,runtime}.py`, `orchestration/parity.py`,
    model adapters, format manifests.
  - **Outputs:** complete tensor/alias inventory; binary/excluded representations; tokenizer/
    metadata/container bytes; deterministic whole-model packed file; direct packed execution;
    allocation trace; layer/logit/token/tool parity.
  - **Verification:** non-byte rows, scale dtypes, tied aliases, actual-file zero reconciliation,
    no persistent dense body, fixed-prompt parity; Properties 27–29.
  - **Exit:** every parameter/byte appears once; actual bytes equal manifest; runtime consumes
    packed weights directly; artifact derives from training checkpoint.
  - **Traceability:** Requirements 16–17; Properties 27–29.
  - **Prerequisites:** Task 9. **Non-goal:** container compression/persistent expansion is not
    packed inference.
  - [x] 10.1 Tensor/tiny accounting, packing, and parity primitives are example-tested.
  - [ ] 10.2 Implement and verify whole-model inventory/export/runtime/parity.

- [~] 11. Executable small-scale CLI and protocols
  - **Purpose:** turn preregistration objects into a resumable scientific executor.
  - **Evidence:** Pythia/SmolLM adapters, protocol suite/preflight, evidence schemas, and CLI
    `catalog`/`validate` exist; there is no training executor.
  - **Files/symbols:** `scripts/run-binary-llm-small.py`,
    `orchestration/small_scale_protocols.py::main`, new orchestration/command modules.
  - **Outputs:** commands `preflight`, `run`, `resume`, `evaluate`, `export`, `report`; canonical
    config input; dry-run plan; noninteractive IDs; safe interruption; durable outputs.
  - **Verification:** CLI subprocess tests for every command, malformed/missing seals, resume,
    budget stops, and no-allocation preflight failures.
  - **Exit:** one command family executes Task 9 flow and model-scale equivalent; every command
    emits resolvable IDs; no ad hoc defaults.
  - **Traceability:** Requirements 3–17, 20–21.
  - **Prerequisites:** Tasks 9–10. **Non-goal:** `catalog`/`validate` is not execution.
  - [x] 11.1 Explicit small-model inventories and protocol/preflight builders are tested.
  - [ ] 11.2 Implement six required executable commands and subprocess tests.

## Wave 4 — Scientific scale ladder

- [ ] BLOCKED: 12. Execute Pythia screening, then SmolLM-135M reproduction
  - **Purpose:** obtain first local BinaryLLM scientific evidence in the required order.
  - **Evidence:** zero runs. Only adapters/protocol builders exist.
  - **Files/artifacts:** CLI, sealed Pythia-70M and SmolLM-135M revisions, frozen RedPajama
    corpus/20 partitions, paper panel, manifests, checkpoints, raw outputs, resource ledgers.
  - **Outputs:** Pythia reference/ambiguity/scale/schedule/no-init/all-vs-body arms; gate report;
    only after pass, SmolLM paper-reference and matched ablations with perplexity and exact
    paper zero-shot panel.
  - **Verification:** input hashes, budgets, 20 phases, 50-step SmolLM Stage 1, paired 95% CIs,
    progressive/sign results, rerun tolerances.
  - **Exit:** durable claim status is reproduced/not reproduced/contradicted/not tested;
    unsuccessful runs retained; promotion only on preregistered gates.
  - **Traceability:** Requirements 3–8, 11–13, 15, 20.
  - **Prerequisites:** Tasks 1–11 clean. **Non-goal:** never copy paper metrics.
  - [ ] 12.1 Execute bounded Pythia screening.
  - [ ] BLOCKED: 12.2 Execute SmolLM only after exact Pythia method passes.

- [ ] BLOCKED: 13. Intermediate and Iris gates
  - **Purpose:** enforce monotonic promotion to 430M–500M and then pinned Iris.
  - **Evidence:** zero runs; no substantive executable intermediate/Iris model pipeline.
  - **Files/artifacts:** explicit intermediate adapter; MiniCPM5/Iris adapter and seal resolver;
    base/adapter/BF16/Q4 oracles; frozen broad/private panels; recovery/export evidence.
  - **Outputs:** qualifying intermediate result; then Iris continuation/recovery, whole-model
    variants, semantic/safety/accounting/parity evidence.
  - **Verification:** unchanged method/config lineage; all lower-rung gates; exact Iris metrics
    and zero critical violations; deterministic reproduction.
  - **Exit:** each rung has durable promotion decision; changes return to Pythia screening.
  - **Traceability:** Requirements 2–4, 9–18, 20–21.
  - **Prerequisites:** Task 12 SmolLM promotion before intermediate; intermediate promotion
    before Iris. **Non-goal:** protocol schemas are not model runs.
  - [ ] 13.1 Implement/select explicit intermediate adapter and execute acceptance.
  - [ ] BLOCKED: 13.2 Implement/execute Iris only after intermediate pass.

- [ ] BLOCKED: 14. Isolated controls
  - **Purpose:** compare binary against ternary, pruning, and distillation without label leakage.
  - **Evidence:** family/policy records exist; no control executors or runs.
  - **Files/artifacts:** independent plugins; 430M–500M pruning Q4; 270M–500M distilled/QAT;
    shared evaluator/accounting protocols; separate reports.
  - **Outputs:** isolated manifests, runs, bytes/compute/quality evidence, hybrid component
    controls, deployable-Pareto eligibility.
  - **Verification:** family isolation, one-factor/shared-baseline checks, unsupported-runtime
    exclusion; Properties 2, 10, 27, 32.
  - **Exit:** no control is labeled binary; every family reports its own evidence.
  - **Traceability:** Requirement 19; Properties 2, 10, 27, 32.
  - **Prerequisites:** Task 13 scale evidence and Task 10 runtime support as applicable.
  - **Non-goal:** do not mix family metrics into a BinaryLLM claim.

## Wave 5 — Devices and release

- [ ] BLOCKED: 15. Device evidence and physical qualification
  - **Purpose:** prove the exact artifact/runtime/build works in complete applications.
  - **Evidence:** zero device runs; schemas/pure gate ideas do not qualify devices.
  - **Files/artifacts:** separately approved target runtime/device agent; signed protocol;
    iPhone X, iPhone 11, newer control traces.
  - **Outputs:** memory breakdown at 128/512/1024/4096 context, cold load, TTFT, throughput,
    p95 latency, 30-request thermal loop, reliability, offline network, battery/energy, exact
    artifact/runtime/build binding.
  - **Verification:** raw trace validation and deterministic postprocessing; Properties 30–31.
  - **Exit:** all physical thresholds pass; no simulator/desktop inference; persistent dense
    expansion fails.
  - **Traceability:** Requirement 18; Properties 30–31.
  - **Prerequisites:** qualifying whole-model Iris artifact and approved direct runtime.
  - **Non-goal:** paper CPU results or simulator traces are not device evidence.

- [ ] BLOCKED: 16. Final audit, report, and release selection
  - **Purpose:** issue one honest, size-optimal, fail-closed decision.
  - **Evidence:** report helpers exist; no complete content-addressed audit report or candidate.
  - **Files/symbols:** `reporting/*`, durable store, gate/release selector, report schema.
  - **Outputs:** hypotheses/configs/ambiguities/ablations/failures/resources/capability/
    accounting/parity/device/licenses/reproduction/risks; `release`, `private_only`, or
    `no_qualifying_binary_release`; retained BF16/Q4 oracles.
  - **Verification:** fresh-process reference resolution; completeness/property 33;
    size-optimal selection/property 32; no metric relabeling.
  - **Exit:** every report fact resolves by hash; all gates pass or release fails; unresolved
    terms force private-only.
  - **Traceability:** Requirement 21; Properties 32–33.
  - **Prerequisites:** Tasks 12–15 and all 33 properties.
  - **Non-goal:** never lower a threshold to create a release.

## Dependency graph

```text
Wave 0: 1 static/domain ─┬─> 2 durable evidence
                        ├─> 3 fidelity config
                        └─> 4 gates/statistics
Wave 1: 2+3 -> 5 Stage 1 transition -> 6 Stage 2/resume/failure
        2+6 -> 7 corpus/recovery/teacher
Wave 2: 1..7 -> 8 all 33 properties -> 9 durable tiny vertical slice
Wave 3: 9 -> 10 whole-model export/parity -> 11 executable CLI
Wave 4: 11 -> 12 Pythia -> SmolLM -> 13 intermediate -> Iris -> 14 controls
Wave 5: qualifying whole-model Iris -> 15 devices -> 16 audit/release
```

The tiny durable slice is mandatory before Pythia. Pythia is mandatory before SmolLM.
SmolLM is mandatory before intermediate. Intermediate is mandatory before Iris.

## Stop conditions and forbidden shortcuts

- Stop at the next safe boundary on non-finite state, budget crossing, seal/provenance
  mismatch, failed continuation floor, or unresolved blocking ambiguity.
- No threshold tuning after blind access; freeze a new set.
- No dense-training-then-quantize recovery; target representation stays active.
- No metrics copied across artifacts, attempts, scales, formats, or runtime builds.
- No packed claim if the runtime persistently expands the whole binary body.
- No scale-promotion bypass and no paper-reported value claimed as local evidence.
- No model allocation before complete hashes, ambiguity choices, four-part budget, gates,
  seeds/order, and evaluator protocols resolve.

## Standard validation commands (Windows PowerShell)

```powershell
Set-Location "C:\Users\affan\Fun Projects\siri"
$files = Get-ChildItem -Path "tests" -File -Filter "test_binary_llm*.py" |
    Sort-Object FullName |
    ForEach-Object { $_.FullName }
if (-not $files) { throw "No BinaryLLM tests found" }
$env:HYPOTHESIS_STORAGE_DIRECTORY = Join-Path $env:TEMP "siri-binary-llm-hypothesis"
python -m pytest -p no:cacheprovider @files
python -m ruff check --no-cache src/binary_llm scripts/run-binary-llm-small.py @files
git diff --check -- ".kiro/specs/binary-llm-conversion-framework/status.md" ".kiro/specs/binary-llm-conversion-framework/tasks.md"
```

Record actual output and elapsed time in [status](status.md). Do not rewrite the current
`304 passed in 32.81s` result unless the full command is rerun; preserve earlier results as
historical evidence.
