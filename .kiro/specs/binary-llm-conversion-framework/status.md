# BinaryLLM Conversion Framework — Agent Handoff Status

**Status date:** 2026-07-18 (UTC+05:30)
**Read this file first:** it records the evidence boundary between implemented framework code and work that has actually been scientifically executed.

## Authority and epistemic rules

- The sole authority for BinaryLLM paper claims is
  [`docs/research-papers/2508.06974v2-binaryLLM.md`](../../../docs/research-papers/2508.06974v2-binaryLLM.md).
- Preserved SHA-256:
  `8e346e97f4d172b93c04d3ad240eb32d599a9f8924290ffd0f486661bafeb8f6`.
- The local [requirements](requirements.md) define framework obligations. The local
  [design](design.md) defines intended architecture. The executable
  [task roadmap](tasks.md) records current implementation work.
- Implementation notes, prior chats, reports, and external papers may help locate a question,
  but they must not override or add facts to the authoritative paper.

Every statement must be classified along both of these axes:

1. **Origin**
   - **Paper fact:** explicitly stated by the preserved paper.
   - **Paper ambiguity/inference:** absent, inconsistent, or underspecified in the paper.
   - **Framework extension:** selected by this project for auditability, Iris, deployment, or
     reproducibility. It is not a BinaryLLM claim.
2. **Evidence maturity**
   - **Code implemented:** a symbol exists.
   - **Test verified:** a deterministic local test exercises the stated scope.
   - **Scientifically executed:** a registered run on the declared model/data/protocol emitted
     durable evidence.
   - **Promoted/released:** all prerequisite scale, semantic, accounting, runtime, provenance,
     reproducibility, and device gates passed for the exact artifact.

Never infer a later state from an earlier one. Passing unit tests is not a model reproduction;
a protocol builder is not an executor; an in-memory registry is not durable evidence; packing
a tiny tensor is not whole-model export; and paper-reported metrics are not local results.

## Current verdict

The repository contains a substantial deterministic research-framework prototype. It is **not**
a completed BinaryLLM reproduction, whole-model conversion pipeline, or release candidate.

- Last verified BinaryLLM suite: **304 passed in 32.81s** on 2026-07-18.
- Model-scale scientific runs: **zero**.
- Required Hypothesis properties implemented: **33 of 33**.
- Current scoped Ruff findings: **zero** (`All checks passed!` on 2026-07-18).
- Pythia screening runs: **zero**.
- SmolLM-135M paper-reference runs: **zero**.
- Intermediate (430M–500M) runs: **zero**.
- Iris runs: **zero**.
- Whole-model conversion/export/runtime runs: **zero**.
- Physical-device qualification runs: **zero**.

## Architecture and key symbols

The intended flow is:

`paper/requirements -> immutable manifest -> seal/preflight -> trainer -> evaluation -> gates -> export -> packed runtime -> device evidence -> audit/release`

Current implementation map:

- `src/binary_llm/domain/`
  - `paper.py`: `BINARYLLM_PAPER_LEDGER`, paper path/hash, facts, and reference settings.
  - `models.py`: canonical experiment, claim, ambiguity, gate, tolerance, representation,
    candidate, and manifest records.
  - `identity.py`, `serialization.py`: canonical serialization and SHA-256 identities.
  - `ambiguities.py`, `reproduction.py`, `validation.py`, `errors.py`: ambiguity seed,
    source-difference classification, validation, and typed failures.
- `src/binary_llm/math/`
  - `progressive.py`: paper progressive function, custom analytical backward, schedules.
  - `scales.py`: Stage 1 input scales, dual scales, sign policy, diagnostics.
- `src/binary_llm/adapters/`
  - `linear.py`: binary linear training/progressive/sign views.
  - `tiny.py`: deterministic tiny causal model.
  - `small_models.py`: explicit Pythia/SmolLM inventories and local loader contracts.
  - `iris_evaluation.py`: additive Iris evidence adapter; not an Iris model executor.
- `src/binary_llm/orchestration/`
  - `budgets.py`: shared token/step/wall/cost accounting.
  - `stage1.py`: tiny Stage 1 training backend.
  - `progressive_state.py`, `progressive_trainer.py`: phase plans and resumable checkpoints.
  - `corpus.py`: provenance, deduplication, isolation, token allocation, partitions.
  - `registry.py`: append-only **in-memory** registry contracts.
  - `sealing.py`: sealed inventory verification.
  - `gates.py`, `ablations.py`, `statistics.py`: gates, matched comparisons, bootstrap logic.
  - `recovery.py`: binary-active recovery and teacher-routing contracts.
  - `evaluation.py`, `evidence_coordinators.py`, `small_scale_evidence.py`: evidence records
    and evaluation coordinators.
  - `small_scale_protocols.py`: Pythia/SmolLM protocol construction and preflight.
- `src/binary_llm/export/`
  - `accounting.py`: tensor/file byte ledger primitives.
  - `packed.py`: deterministic tensor packing and verification.
  - `runtime.py`: scalar reference packed-linear runtime.
- `src/binary_llm/reporting/`
  - evaluation/release decision helpers; not a complete durable audit-report pipeline.
- `scripts/run-binary-llm-small.py`
  - thin CLI wrapper. Current commands are only `catalog` and `validate`; it does not train,
    resume, evaluate, export, or report.
- `tests/test_binary_llm*.py`
  - 304 example/integration/property tests at the latest full verification. All 33 required
    Design Properties have exact tags and at least 100 generated examples.

## Verified implementation

The following are implemented and covered by direct deterministic tests for their stated,
mostly in-memory or tiny-model scope:

- Package boundaries, exact pinned Hypothesis dependency/profile configuration, canonical
  models, serialization, identity, schema evolution, and typed failures.
- Paper ledger/reference classification bound to the preserved path and SHA-256
  (`test_binary_llm_paper.py`, `test_binary_llm_reproduction.py`).
- Sealed fixture inventory mismatch detection and independent oracle identities
  (`test_binary_llm_sealing.py`).
- Append-only registry semantics, attempt identity, lineage, checkpoint completeness,
  supersession, and deterministic-reproduction validation in memory
  (`test_binary_llm_registry.py`).
- Atomic filesystem content-addressed artifact storage with exact payload SHA-256/size
  verification, canonical checksummed `ArtifactRef` metadata, idempotent publication,
  fresh-instance resolution, exact expected owner/lineage checks, and typed corruption
  failures (`orchestration/store.py`, `test_binary_llm_store.py`). This persists artifact
  payloads and references, not the in-memory registry event graph.
- Canonical hash-chained filesystem registry journal with monotonic sequencing, durable head
  commits, fresh-process replay/continuation, owner/parent/producer validation, verified
  artifact-payload integration, exclusive writer-lock diagnosis, and explicit artifact-store
  recovery/quarantine (`orchestration/journal.py`, expanded registry/store tests). Portable
  lock files serialize writers; crashed locks require explicit quarantine and are never
  silently stolen.
- Progressive forward/analytical backward, zero limit, schedules, scale math, finite-state
  diagnostics, and explicit tensor scope
  (`test_binary_llm_progressive.py`, `test_binary_llm_scales.py`,
  `test_binary_llm_tiny_adapter.py`).
- Canonical executable fidelity configuration with exact material field-path inventory,
  paper/inference/framework provenance, deterministic paper-bound identity, fail-closed
  ambiguity resolution, matched modified/interaction arms, and small-scale preflight binding
  (`domain/fidelity.py`, `test_binary_llm_fidelity.py`).
- Declarative gates, monotonic promotion logic, matched ablation checks, grouped bootstrap,
  blind-evaluation history, and release-selection policy
  (`test_binary_llm_gates.py`, `test_binary_llm_ablation_statistics.py`,
  `test_binary_llm_evaluation_release.py`).
- Tiny Stage 1 training with scale-only optimization and failure capture
  (`test_binary_llm_stage1_trainer.py`).
- Exact Stage 1 reference algebra `Sign(W/S_t) * (S_t*A)` with the weight-only
  interpretation retained as an explicitly modified arm, plus deterministic
  content-addressed `W_tilde=W/S_t*` transition artifacts. Transition application validates
  the complete model before atomically installing `W_tilde`, resets input scales to exact
  identity to prevent double division, preserves dense-reference buffers/biases, and is
  mandatory for fresh progressive training (`orchestration/transition.py`,
  `test_binary_llm_stage1_transition.py`).
- Corpus freezing improvements: content-addressed manifests, transitive exact/fuzzy relation
  handling, provenance rejection, deny-list isolation, semantic-family grouping, token
  allocation, and disjoint phase coverage (`test_binary_llm_corpus.py`).
- Deterministic durable corpus bundles bind exact accepted records, provenance, licenses,
  deny lists, dedup/isolation reports, group closure, allocation, and all 20 partition
  memberships. Recovery evidence binds exact batch tensors, selected mix/config, teacher
  binding/provenance, binary-active forwards, steps/progress/budgets, stop reason, and lineage;
  fresh stores resolve the complete chain while all fixture evidence remains explicitly
  non-promotional (`orchestration/evidence_store.py`,
  `test_binary_llm_durable_evidence.py`).
- Progressive phase state/checkpoint round trip and safe-boundary behavior. Checkpoint schema
  v2 carries the immutable Stage 1 root on every phase independently of the immediate parent,
  and every resume validates that root (`test_binary_llm_progressive_state.py`,
  `test_binary_llm_progressive_trainer.py`).
- Explicit Stage 2 trainability arms: the paper-reference arm optimizes every unique eligible
  model parameter exactly once while folded Stage 1 input scales remain identity/frozen
  transition state; the binary-body-only arm is an exact modified ablation. Deterministic
  parameter inventories record aliases, roles, optimizer inclusion, and before/after digests.
- Exact deterministic CPU equivalence between uninterrupted two-phase training and a
  checkpoint-v2 disk round trip into a fresh model/optimizer, including model/optimizer state
  hashes, RNG, schedule, data cursor, budgets, metrics, and final progressive/sign evidence.
- Diagnostic-only content-addressed progressive numerical-failure artifacts preserve the
  failing batch, state summaries/hashes, finite diagnostics, failure boundary, Stage 1 root,
  plan/partition, budget, and last complete checkpoint without publishing partial work
  (`orchestration/progressive_failure.py`).
- Shared runtime budget enforcement across Stage 1, progressive, and recovery backends
  (`test_binary_llm_budgets.py` plus backend tests).
- Tensor-level accounting, deterministic packed export, scalar packed runtime, and tiny-model
  parity (`test_binary_llm_artifact_accounting.py`, `test_binary_llm_packed_exporter.py`,
  `test_binary_llm_packed_runtime.py`).
- Binary-active recovery and teacher routing on tiny fixtures, including resolver-created
  behavioral-teacher authorization bound to a complete matching seal report, the exact BF16
  oracle identity, and durable resolution of all BF16 weight/tokenizer/template/configuration
  artifacts (`orchestration/recovery.py`, `test_binary_llm_recovery.py`). Caller-supplied
  `sealed=True` is not sufficient; identity substitution is checked again before supervision.
- Iris scoring additions and generic evidence coordinators on fixtures
  (`test_binary_llm_iris_evaluation_adapter.py`,
  `test_binary_llm_evidence_coordinators.py`).
- Explicit small-model inventory/loader contracts, preregistered protocol construction,
  preallocation checks, and acceptance-evidence schemas
  (`test_binary_llm_small_model_adapters.py`,
  `test_binary_llm_small_scale_protocols.py`,
  `test_binary_llm_small_scale_evidence.py`).

## Partial or overstated areas

- **Registry/evidence:** durable journal replay now covers manifests, attempts, events,
  attachments, checkpoints, statuses, resource deltas, supersessions, and reproduction
  records, while the artifact store resolves verified bytes. Remaining limitations are
  portable lock-file rather than OS advisory locking, best-effort directory `fsync` on
  unsupported Windows/filesystem APIs, serialized rather than lock-free writers, and
  deliberately manual quarantine for crashed locks or orphaned evidence.
- All **33/33** Design Properties have exactly one tagged Hypothesis test with
  `max_examples=100`, meaningful bounded generators, and a source-inventory test enforcing
  numbering, exact titles, decorators, uniqueness, and minimum examples. Device Properties
  30–31 prove only pure schema/gate behavior, not physical-device evidence; Property 33 proves
  only the report completeness contract.
- **Tiny vertical slice:** components exist separately, but one durable flow from manifest and
  seal through Stage 1, transition, two progressive phases, failure/resume, evaluation, gates,
  export, packed runtime, registry resolution, and report is absent.
- **Accounting/export/runtime:** tested for tensors and a tiny model, not every tensor and
  sidecar of a real whole model. No whole-model byte reconciliation or whole-model parity
  evidence exists.
- **Small-scale path:** adapters and protocol/preflight builders exist, but the CLI is a
  validator, not an executor.
- **Evaluation:** protocol/evidence coordinators exist; actual lm-eval, perplexity, code,
  rubric, private Iris, and whole-model generation runs have not occurred.
- **Teacher sealing:** the authorization boundary is verified, but artifacts are fully read
  and hashed when `VerifiedTeacherBinding` is created rather than before every generated
  token. Store mutation after binding requires explicit rebinding or another integrity
  boundary before the run.
- **Intermediate, Iris, controls, devices, report:** some data models or policy helpers exist,
  but substantive executable implementations and scientific evidence do not.

## Missing areas

1. Resolve the 18 static-analysis findings and make the scoped static check clean.
2. Build an executable paper-fidelity configuration with no silent defaults.
3. Emit and verify the exact Stage 1 transition artifact.
4. Implement literal all-parameter Stage 2 and body-only modified arms with matched evidence.
5. Prove exact two-phase resume equivalence.
6. Preserve durable progressive numerical-failure diagnostics.
7. Integrate remaining training/evaluation/report producers with the durable journal and
   artifact resolver so every emitted evidence reference is persisted rather than merely
   represented by a durable-capable API.
8. Complete the durable tiny end-to-end slice.
10. Implement whole-model inventory, export, direct packed execution, and parity.
11. Expand the CLI to `preflight`, `run`, `resume`, `evaluate`, `export`, and `report`.
12. Execute Pythia screening, then SmolLM-135M reference reproduction.
13. Implement and execute intermediate and Iris gates in order.
14. Implement isolated ternary/pruning/distillation controls.
15. Build a separately approved target runtime and gather physical-device evidence.
16. Produce the final content-addressed audit report and fail-closed release decision.

## Scientific execution ledger

- Tiny synthetic unit/integration execution: **yes**.
- Durable tiny end-to-end run: **no**.
- Pythia-70M screening: **no**.
- SmolLM-135M paper-reference reproduction: **no**.
- 430M–500M intermediate: **no**.
- MiniCPM5/Iris: **no**.
- Whole-model packed export/direct runtime: **no**.
- Ternary/pruning/distillation control runs: **no**.
- iPhone X / iPhone 11 / newer-control qualification: **no**.
- Release candidate: **none**.

## Validation baseline

Exact last full test command (Windows PowerShell):

```powershell
Set-Location "C:\Users\affan\Fun Projects\siri"
$files = Get-ChildItem -Path "tests" -File -Filter "test_binary_llm*.py" |
    Sort-Object FullName |
    ForEach-Object { $_.FullName }
if (-not $files) { throw "No BinaryLLM tests found" }
$env:HYPOTHESIS_STORAGE_DIRECTORY = Join-Path $env:TEMP "siri-binary-llm-hypothesis"
python -m pytest -p no:cacheprovider @files
```

Last verified result: `304 passed in 32.81s` on 2026-07-18 (35 test files; exit code 0; no
failures, errors, or reported skips; shell elapsed 40.893s). The temporary Hypothesis directory
and disabled cache provider prevented test-cache writes in the repository. Pytest emitted one
configuration warning because the explicit `norecursedirs` setting replaces its default
`.hypothesis` ignore; this is a collection-configuration cleanup item, not a failed property.

The preceding results were `209 passed in 7.02s`, `209 passed in 7.90s`, and
`231 passed in 8.17s`, `243 passed in 8.48s`, `248 passed in 44.57s`, and
`253 passed in 9.42s`, `263 passed in 10.04s`, `266 passed in 11.04s`, and
`270 passed in 11.08s`; retain them as historical evidence rather than treating any as the
current run. Do not rewrite the current result unless the command is rerun.
Read-only AST parsing reported `AST syntax OK: 72 files`, covering 53
`src/binary_llm/**/*.py` files, the command script, and 18 unique physical BinaryLLM test
files. The final audit originally reported 18 Ruff errors. They were closed on 2026-07-18:

- multiple `E402` import-order findings in `src/binary_llm/orchestration/__init__.py`;
- `F401` unused imports in `adapters/linear.py`, `small_scale_protocols.py`, and two tests;
- `F821` undefined forward type name `CapabilityPanelEvidence` in
  `reporting/evaluation.py`; and
- `E731` assigned lambda in `reporting/release.py`.

The exact scoped Ruff command below now returns `All checks passed!`. The fixes reorganized
package imports before `__all__` construction, removed unused imports, added a type-check-only
capability-evidence import, and replaced the assigned lambda with an equivalent named local
ordering function. No suppression was added and the complete BinaryLLM suite remained green.
Future agents must capture the exact current output before editing; do not assume it remains
clean.

Recommended read-only/scoped checks:

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

If the repository’s configured static command differs, record the exact command and output
here before changing code. Do not add dependencies merely to validate these Markdown files.

## Critical fidelity and release risks

- Stage 1 inverse activation and the exact Stage 1 -> Stage 2 transition remain unresolved.
- Paper all-parameter Stage 2 differs from current body-only trainability.
- A declared protocol is not a scientific executor.
- In-memory evidence is not durable or independently resolvable.
- Tiny tensor packing is not whole-model accounting/export.
- Progressive numerical failures lack a guaranteed durable diagnostic artifact.
- All 33 required property tests are absent.
- Teacher sealing depends on the caller rather than an end-to-end sealed resolver.
- No local result may inherit paper metrics or another artifact’s metrics.
- The paper’s hardware section used a BitNet-style 1.58-bit deployment; it is not evidence of
  this framework’s direct true-1-bit runtime.

## Smallest honest next milestone

**Complete the deterministic durable tiny vertical slice before any model-scale run.**

Ordered roadmap:

1. Make static analysis clean and freeze an executable paper-fidelity config.
2. Implement the exact Stage 1 transition artifact and both Stage 2 trainability arms.
3. Add exact resume equivalence and durable numerical-failure artifacts.
4. Add the durable content-addressed store/resolver.
5. Implement all 33 required properties.
6. Run the complete tiny manifest-to-report path, including whole-tiny-model packing/parity.
7. Add the executable small-scale CLI.
8. Execute bounded Pythia screening.
9. Only after a passing registered Pythia decision, execute SmolLM-135M reproduction.
10. Only after small-scale promotion, consider intermediate; then Iris; then controls,
    whole-model target runtime, devices, and release.

## Do-not-start gates

- **Pythia:** do not start until static checks, all 33 properties, exact transition/trainability
  semantics, durable evidence, and the complete tiny slice pass.
- **SmolLM-135M:** do not start until a registered Pythia screening result passes the
  preregistered continuation gates for the exact operator/configuration.
- **Intermediate:** do not start until the SmolLM reference and mandatory matched ablations
  produce qualifying durable small-scale evidence.
- **Iris:** do not start until the unchanged method passes intermediate promotion and all
  pinned Iris/base/adapter/BF16/Q4/data/evaluator seals verify.
- **Release:** do not start release selection until semantic, safety, provenance, accounting,
  whole-model packed parity, deterministic reproduction, and device evidence are complete.
- **Devices:** do not claim qualification from schemas, simulators, desktop runs, or paper
  numbers. Start only with a whole-model packed artifact and approved direct target runtime.

## Stop conditions and forbidden shortcuts

- Stop at the next safe boundary for non-finite state, budget crossing, seal/provenance
  mismatch, failed continuation floor, or unresolved blocking ambiguity.
- Never tune a threshold after blind-set access; freeze a new set instead.
- Never recover by dense training followed by final quantization; the target operator must
  remain active.
- Never copy metrics across checkpoints, artifacts, runs, scales, formats, or runtime builds.
- Never claim packed execution if the runtime persistently expands the whole binary body.
- Never bypass `tiny -> Pythia -> SmolLM -> intermediate -> Iris`.
- Never call paper-reported values locally reproduced.

## Worktree safety

This repository is heavily dirty and contains many untracked files. The BinaryLLM package,
tests, scripts, and spec documents may themselves be untracked. Never discard, reset, clean,
rename, or overwrite unrelated changes. Never state that all current files are committed.
Before each slice, inspect scoped status and edit only the files authorized by the user.

## Updating this handoff

After every implementation or scientific slice:

1. Update **Status date** with timezone.
2. Record the exact command, result, and elapsed time for tests/static checks actually run.
3. Move an item only to the evidence maturity actually reached.
4. Name the exact files/symbols and content-addressed artifacts added.
5. Record scientific runs separately with model revision, manifest/evidence IDs, budgets,
   result, and stop reason.
6. Add newly discovered ambiguities and risks; do not silently resolve them.
7. Update [tasks.md](tasks.md) markers and acceptance evidence in the same slice.
8. Preserve historical facts (for example, `209 passed in 7.02s`) as dated results rather
   than replacing them with assumptions.
