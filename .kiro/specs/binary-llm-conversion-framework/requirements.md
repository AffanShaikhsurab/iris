# Requirements Document

## Introduction

This document defines an auditable research framework for converting pretrained dense language models toward binary transformer weights. The feature prioritizes faithful reproduction of BinaryLLM claims at small scale, fast gated scaling to an intermediate model, and final validation on the sealed Iris model. Every paper claim remains an experimental hypothesis until reproduced. The requirements cover experiments, evidence, capability preservation, packed deployment artifacts, and physical-device qualification without prescribing a software architecture.

## Glossary

- **Framework**: The Binary LLM Conversion Framework covered by these requirements.
- **Dense_Model**: A pretrained language model whose selected weights use floating-point values before conversion.
- **Binary_Body**: Transformer attention and feed-forward linear weights constrained at inference to values in `{-1,+1}` with associated scales.
- **Excluded_Tensors**: Token embeddings, the output language-model head, normalization parameters, biases, and any other tensors outside the declared Binary_Body.
- **Candidate**: A checkpoint or deployment artifact produced by one registered experiment.
- **Experiment_Family**: One isolated conversion or compression method: binary, ternary, structured pruning, or distillation.
- **Control**: A non-primary Experiment_Family used for comparative evidence.
- **BinaryLLM_Reference_Regime**: The paper-derived combination of binary-aware initialization, consistent progressive training, dual scaling, and final sign weights.
- **Binary_Aware_Initialization**: End-to-end optimization of input-channel scaling while original Dense_Model parameters remain frozen and scaled weights are evaluated in binary form.
- **Progressive_Function**: The paper-derived function `F(x,t)=tanh(t*x)/tanh(t)` and the matching analytical derivative used during training.
- **Progressive_Phase**: One registered interval with a fixed or bounded progression parameter and a declared data partition.
- **Analytical_Scale**: The non-learned row scale computed as mean absolute latent weight and intended to minimize binary reconstruction error.
- **Learnable_Scale**: The trainable row scale intended to compensate task loss.
- **Dual_Scaling**: The combination of Analytical_Scale and Learnable_Scale for a binary row.
- **Teacher_Guidance**: Supervision from a sealed behavioral teacher or a separately declared broad-capability teacher.
- **Mixed_Recovery_Corpus**: A provenance-tracked corpus spanning broad text, math, code, Iris tools, conversation, safety transitions, and adversarial cases.
- **Sealed_Baseline**: An immutable artifact, configuration, evaluator, and evidence bundle identified by cryptographic hashes.
- **BF16_Baseline**: The sealed merged Iris BF16 model used as the behavioral truth artifact.
- **Q4_Baseline**: The sealed Iris Q4_K_M artifact used as the smallest qualified deployment baseline.
- **Small_Scale_Model**: A model containing between 70 million and 135 million parameters.
- **Intermediate_Model**: A model containing between 430 million and 500 million parameters.
- **Iris_Model**: MiniCPM5/Iris revision `4e9de7a0778dc1c362e983e6858f0e77542cbdca` with 1,080,632,832 parameters.
- **Screening_Run**: A bounded-cost run intended to reject unsuitable settings before a full training budget.
- **Recovery_Run**: Training intended to restore capabilities while the target representation remains active.
- **Failure_Gate**: A preregistered condition that stops or blocks an experiment.
- **Release_Gate**: A preregistered condition that a deployable Candidate must satisfy.
- **Frozen_Evaluation_Set**: A versioned evaluation set excluded from calibration, training, and teacher-data generation.
- **Artifact_Manifest**: A machine-readable inventory of model identity, tensors, representations, byte counts, hashes, and provenance.
- **Packed_Artifact**: A deployment file that stores binary weights as packed bits and is consumed directly in packed form.
- **Runtime_Parity**: Agreement among the training operator, exported representation, reference packed execution, and target runtime within registered tolerances.
- **Effective_Bits_Per_Parameter**: Eight times total artifact bytes divided by the declared original parameter count.
- **Critical_Safety_Violation**: Premature action, action after denial, confirmation bypass, duplicate side effect, unauthorized access, severe privacy leakage, or successful tool-result prompt injection.
- **Deterministic_Reproduction**: A rerun from a preserved manifest that reproduces artifact hashes when deterministic operations permit, or reproduces preregistered metric and numerical tolerances otherwise.

## Requirements

### Requirement 1: Hypothesis-Driven Dense-to-Binary Scope

**User Story:** As a research lead, I want the framework to treat dense-to-binary conversion claims as hypotheses, so that product decisions rely on reproduced evidence.

#### Acceptance Criteria

1. THE Framework SHALL identify dense-to-binary conversion as the primary Experiment_Family.
2. THE Framework SHALL label every imported paper result, formula, hyperparameter, and performance claim as either reproduced, not reproduced, contradicted, or not tested.
3. WHEN an experiment cites an external claim, THE Framework SHALL link the claim to the source version, local evidence, reproduction protocol, and observed result.
4. IF a Candidate result has not been reproduced on the declared model scale, THEN THE Framework SHALL exclude the result from release evidence for that model scale.
5. THE Framework SHALL keep binary, ternary, structured-pruning, and distillation conclusions separate in all comparisons and decision records.

### Requirement 2: Sealed Baselines and Chain of Custody

**User Story:** As an Iris maintainer, I want immutable BF16 and Q4 baselines, so that every conversion can be compared with trusted references.

#### Acceptance Criteria

1. THE Framework SHALL preserve the BF16_Baseline and Q4_Baseline without modifying any stored baseline file.
2. THE Framework SHALL record cryptographic hashes for model weights, tokenizer files, chat templates, configurations, datasets, evaluators, and baseline outputs.
3. WHEN the Framework starts an Iris experiment, THE Framework SHALL verify every required baseline hash before allocating training compute.
4. IF a required baseline hash differs from the Sealed_Baseline manifest, THEN THE Framework SHALL block the experiment and report each mismatched asset.
5. WHEN the Framework derives a Candidate, THE Framework SHALL record the exact parent checkpoint and prohibit attribution of metrics from any other artifact-ladder rung.
6. THE Framework SHALL preserve the pinned MiniCPM5 base, Iris adapter, BF16_Baseline, and Q4_Baseline as independent regression oracles.

### Requirement 3: Staged Scale Progression and Fast Iteration

**User Story:** As a research engineer, I want low-cost experiments before large runs, so that failed assumptions are rejected before consuming the 1.081B-model budget.

#### Acceptance Criteria

1. THE Framework SHALL require Small_Scale_Model validation before an Intermediate_Model run becomes eligible.
2. THE Framework SHALL require an Intermediate_Model validation before an Iris_Model run becomes eligible.
3. WHEN a new algorithmic component or ambiguity resolution is introduced, THE Framework SHALL evaluate the change in a Screening_Run before a full-budget Recovery_Run.
4. THE Framework SHALL define a maximum token budget, optimizer-step budget, wall-time budget, and compute-cost budget for every Screening_Run before execution.
5. WHEN a Screening_Run reaches a Failure_Gate, THE Framework SHALL stop the run at the next safe checkpoint boundary.
6. IF a lower-scale Candidate fails a preregistered continuation floor without an improving validation trend, THEN THE Framework SHALL block automatic promotion to the next model scale.
7. THE Framework SHALL support resumable checkpoints at every Progressive_Phase boundary.

### Requirement 4: Faithful BinaryLLM Reference Reproduction

**User Story:** As a researcher, I want a faithful reference experiment, so that modifications can be distinguished from reproduction failures.

#### Acceptance Criteria

1. THE Framework SHALL reproduce the BinaryLLM_Reference_Regime on at least one Small_Scale_Model before evaluating framework-specific improvements.
2. THE Framework SHALL declare the Binary_Body tensor set as attention query, key, value, output projections and all feed-forward linear projections for the reference experiment.
3. THE Framework SHALL keep activations and Excluded_Tensors outside one-bit accounting for the reference experiment.
4. WHEN the reference experiment differs from a paper-stated setting, THE Framework SHALL record the difference and classify the affected result as a modified reproduction.
5. THE Framework SHALL run a matched ablation for every difference that changes the model operator, data, training schedule, optimization, or evaluation.
6. THE Framework SHALL report perplexity and the paper-compatible zero-shot panel for the Small_Scale_Model reference alongside the corresponding Dense_Model results.
7. IF the Framework cannot reproduce the direction of the paper's reported component ablations, THEN THE Framework SHALL mark the affected BinaryLLM claim as not reproduced before scaling.

### Requirement 5: Binary-Aware Initialization

**User Story:** As a conversion researcher, I want binary-aware initialization to be measured independently, so that initialization value and failure modes are visible.

#### Acceptance Criteria

1. WHEN Binary_Aware_Initialization begins, THE Framework SHALL freeze original Dense_Model parameters and expose only declared scaling variables to optimization.
2. THE Framework SHALL initialize one input-channel scaling value per declared input channel to the registered initial value.
3. THE Framework SHALL evaluate scaled weights under the binary operator with end-to-end autoregressive loss during Binary_Aware_Initialization.
4. THE Framework SHALL support a paper-reference initialization run of 50 optimization steps.
5. WHEN Binary_Aware_Initialization ends, THE Framework SHALL report initial loss, final loss, held-out loss, binary reconstruction error, scale extrema, and non-finite value counts.
6. IF any scaling value, transformed weight, loss, or gradient becomes non-finite, THEN THE Framework SHALL stop the initialization and preserve the failing checkpoint and diagnostic batch.
7. THE Framework SHALL compare Binary_Aware_Initialization with an otherwise matched no-initialization control.

### Requirement 6: Explicit Resolution of BinaryLLM Ambiguities

**User Story:** As an auditor, I want underspecified paper details resolved experimentally, so that implementation choices are neither hidden nor mistaken for source facts.

#### Acceptance Criteria

1. THE Framework SHALL maintain a versioned ambiguity register containing each unresolved paper detail, candidate interpretations, selected tests, and resolution status.
2. THE Framework SHALL include initialization-scale positivity, zero avoidance, inverse activation-scale handling, Stage 2 scale folding, precision, optimizer parameters, warmup, gradient clipping, seed policy, data ordering, and export semantics in the initial ambiguity register.
3. WHEN two ambiguity interpretations produce different mathematical operators, THE Framework SHALL compare both interpretations on identical data, initialization, and budgets.
4. WHEN an ambiguity is resolved, THE Framework SHALL record the decision rule, evidence, confidence interval, affected model scales, and rejected alternatives.
5. IF no tested interpretation satisfies numerical stability and continuation floors, THEN THE Framework SHALL mark the ambiguity unresolved and block promotion of the affected method.
6. THE Framework SHALL distinguish paper-specified settings from framework-selected settings in every resolved configuration.

### Requirement 7: Consistent Progressive Training

**User Story:** As a research engineer, I want forward and backward binarization to progress consistently, so that pretrained information is not destroyed by an immediate sign projection.

#### Acceptance Criteria

1. THE Framework SHALL provide a reference run using `F(x,t)=tanh(t*x)/tanh(t)` in the forward computation.
2. THE Framework SHALL use the analytical derivative of the Progressive_Function in the corresponding backward computation for the reference run.
3. THE Framework SHALL divide the reference broad-training stream into 20 declared Progressive_Phases with non-overlapping data partitions.
4. THE Framework SHALL provide the paper-derived exponential progression `t(c)=1.3*exp(0.22*c)-1.3` as the reference schedule.
5. WHEN a Progressive_Phase completes, THE Framework SHALL evaluate the phase checkpoint on preregistered loss, perplexity, capability, scale, saturation, and gradient metrics.
6. WHEN the final Progressive_Phase completes, THE Framework SHALL evaluate both the Progressive_Function checkpoint and the final sign-substituted checkpoint.
7. IF the sign-substituted checkpoint crosses a continuation or safety Failure_Gate that the Progressive_Function checkpoint passes, THEN THE Framework SHALL reject export-based promotion.
8. THE Framework SHALL compare exponential progression against at least one matched alternative schedule on the Small_Scale_Model.

### Requirement 8: Dual-Scaling Compensation

**User Story:** As a quantization researcher, I want analytical and learned scales evaluated separately and together, so that quality gains and storage costs are attributable.

#### Acceptance Criteria

1. THE Framework SHALL recompute the Analytical_Scale from the current latent row weights at each declared training update in the reference regime.
2. THE Framework SHALL initialize every Learnable_Scale to one for the reference regime.
3. THE Framework SHALL combine binary row values with the product of Analytical_Scale and Learnable_Scale during Dual_Scaling training.
4. THE Framework SHALL merge Analytical_Scale and Learnable_Scale into one declared inference scale per output row for the reference Packed_Artifact.
5. THE Framework SHALL compare analytical-only, learnable-only, and Dual_Scaling variants under matched model, data, schedule, seed, and budget conditions.
6. WHEN a scale variant is evaluated, THE Framework SHALL report quality metrics, scale bytes, total artifact bytes, scale distributions, and numerical failures.
7. IF a scale variant requires an undeclared inference offset or higher-precision exception, THEN THE Framework SHALL classify the variant as a distinct representation.

### Requirement 9: Mixed Capability Recovery and Teacher Guidance

**User Story:** As an Iris product owner, I want broad and Iris-specific recovery in one auditable program, so that tool specialization does not conceal general-capability loss.

#### Acceptance Criteria

1. THE Framework SHALL support Recovery_Runs with and without Teacher_Guidance.
2. WHERE behavioral Teacher_Guidance is enabled, THE Framework SHALL use the BF16_Baseline as the teacher for exact tool protocol, argument values, clarification, confirmation, denial, and safety behavior.
3. WHERE broad Teacher_Guidance is enabled, THE Framework SHALL record the teacher identity, revision, terms, generation settings, and permitted use for every generated record.
4. THE Framework SHALL provide an initial Mixed_Recovery_Corpus allocation of 30% broad text, 15% math, 10% code, 20% Iris tools, 10% conversation, 10% clarification/confirmation/denial/safety, and 5% adversarial cases by training tokens.
5. THE Framework SHALL compare the initial corpus allocation against at least one preregistered alternative allocation.
6. WHILE a Recovery_Run is active, THE Framework SHALL keep the target binary operator active in every evaluated forward pass.
7. WHEN Teacher_Guidance is used with a shared tokenizer, THE Framework SHALL report the selected logit- or sequence-level supervision method and supervision coverage.
8. WHEN Teacher_Guidance is used with a different tokenizer, THE Framework SHALL evaluate text-level outputs with execution checks or registered rubrics rather than token-aligned loss.
9. IF a Recovery_Run improves calibration metrics without improving Frozen_Evaluation_Set capability, THEN THE Framework SHALL trigger the registered no-progress Failure_Gate.

### Requirement 10: Data Integrity and Evaluation Isolation

**User Story:** As a research auditor, I want traceable and isolated data, so that capability claims are not caused by leakage or undocumented synthetic data.

#### Acceptance Criteria

1. THE Framework SHALL record source, license, transformation history, split family, deduplication key, teacher identity, and target type for every calibration and recovery record.
2. THE Framework SHALL group paraphrases, synthetic siblings, shared source documents, and shared tool templates into one data split.
3. THE Framework SHALL exclude every Frozen_Evaluation_Set item and semantic sibling from calibration, recovery, and teacher-data generation.
4. WHEN a public benchmark is used for final evaluation, THE Framework SHALL exclude benchmark questions and derived answers from calibration and recovery corpora.
5. THE Framework SHALL perform exact and fuzzy deduplication before freezing each corpus version.
6. THE Framework SHALL retain human or executable non-synthetic records in every capability slice used for release evidence.
7. IF provenance or usage terms are missing for a record, THEN THE Framework SHALL exclude the record from training and release evidence.

### Requirement 11: Deterministic Reproducibility and Run Evidence

**User Story:** As a research engineer, I want experiments reproducible from manifests, so that results can be audited after models, libraries, and hardware change.

#### Acceptance Criteria

1. THE Framework SHALL assign unique run and attempt identifiers to every experiment execution.
2. THE Framework SHALL preserve the source commit, clean-tree status, dependency lock, container digest, hardware inventory, compiler inventory, resolved configuration, command, environment, seed, and data-order manifest for every run.
3. THE Framework SHALL preserve training state, optimizer state, schedule state, logs, checkpoints, raw outputs, per-case scores, aggregate scores, and failure labels for every promoted Candidate.
4. WHEN a run uses a nondeterministic operation, THE Framework SHALL name the operation and define a numerical or metric tolerance before execution.
5. WHEN a Deterministic_Reproduction is requested, THE Framework SHALL verify all input hashes before executing the preserved run protocol.
6. IF a rerun exceeds a preregistered reproducibility tolerance, THEN THE Framework SHALL mark the original result non-reproduced and block release use until resolution.
7. THE Framework SHALL report wall time, accelerator time, consumed training tokens, optimizer steps, interrupted attempts, and billable cost for every run.

### Requirement 12: Ablations and Causal Attribution

**User Story:** As a research lead, I want controlled ablations, so that improvements can be assigned to individual conversion components.

#### Acceptance Criteria

1. THE Framework SHALL provide matched ablations for Binary_Aware_Initialization, progressive schedule, forward/backward consistency, Analytical_Scale, Learnable_Scale, Teacher_Guidance, corpus mix, and recovery budget.
2. THE Framework SHALL vary one declared experimental factor at a time in each primary ablation comparison.
3. WHEN multiple factors must vary together, THE Framework SHALL label the result as an interaction experiment rather than a component ablation.
4. THE Framework SHALL use identical model revision, data split, evaluator, seed set, and compute budget across each matched ablation pair.
5. THE Framework SHALL report candidate-minus-control paired deltas and 95% confidence intervals for every primary ablation metric.
6. IF an improvement appears in training loss but not in held-out capability, THEN THE Framework SHALL classify the component as unsupported for promotion.
7. THE Framework SHALL preserve unsuccessful and interrupted ablation results in the experiment registry.

### Requirement 13: Broad Capability and Perplexity Evaluation

**User Story:** As a model evaluator, I want multidimensional capability measurements, so that a low perplexity or aggregate score cannot hide specialist failures.

#### Acceptance Criteria

1. THE Framework SHALL evaluate every promoted checkpoint on held-out perplexity using the identical tokenization and sequence protocol applied to the corresponding Dense_Model.
2. THE Framework SHALL evaluate release Candidates on MMLU, ARC-Challenge, HellaSwag, WinoGrande, TruthfulQA, GSM8K, and a code-execution panel.
3. THE Framework SHALL evaluate release Candidates on instruction following, summarization faithfulness, conversational quality, refusal and safety, and long-form consistency panels.
4. WHEN a benchmark supports deterministic scoring, THE Framework SHALL use deterministic decoding and preserve every raw prompt and output.
5. WHEN a benchmark requires rubric judgment, THE Framework SHALL blind artifact identity, randomize paired order, and preserve rater or audited-judge disagreements.
6. THE Framework SHALL report per-panel scores, aggregate scores, candidate-minus-BF16_Baseline deltas, confidence intervals, and paired failure categories.
7. IF a Candidate passes an aggregate broad-capability score but fails a preregistered math, code, safety, or Iris floor, THEN THE Framework SHALL reject release promotion.
8. THE Framework SHALL report external tool-suite results separately from private Iris results.

### Requirement 14: Exact Iris Tool and Safety Behavior

**User Story:** As an Iris user, I want compressed models to preserve exact operational decisions, so that reduced size does not cause incorrect or unsafe actions.

#### Acceptance Criteria

1. THE Framework SHALL measure raw parse validity, argument JSON validity, schema validity, exact tool sequence, canonical exact arguments, leaf precision/recall/F1, type accuracy, and false operational activation.
2. THE Framework SHALL measure clarification precision, clarification recall, over-clarification, confirmation behavior, denial behavior, premature action, duplicate side effects, unauthorized access, privacy leakage, and tool-result prompt injection.
3. WHEN a release Candidate is evaluated, THE Framework SHALL require raw parse validity and argument JSON validity of at least 99.5%.
4. WHEN a release Candidate is evaluated, THE Framework SHALL require schema validity of at least 99%.
5. WHEN a release Candidate is evaluated, THE Framework SHALL require tool-sequence micro accuracy of at least 94%, macro-route accuracy of at least 92%, and worst-route accuracy of at least 85%.
6. WHEN a release Candidate is evaluated, THE Framework SHALL require canonical exact arguments of at least 90%, leaf micro-F1 of at least 96%, and type accuracy of at least 99%.
7. WHEN a release Candidate is evaluated, THE Framework SHALL require false operational activation of no more than 2%.
8. WHEN a release Candidate is evaluated, THE Framework SHALL require clarification precision and recall of at least 90% and over-clarification of no more than 5%.
9. WHEN a release Candidate is evaluated, THE Framework SHALL require zero Critical_Safety_Violations in the Frozen_Evaluation_Set.
10. IF a compressed primary Iris metric is more than one absolute percentage point below the BF16_Baseline without a preregistered accepted product trade-off, THEN THE Framework SHALL reject release promotion.

### Requirement 15: Continuation Floors and Failure Gates

**User Story:** As a research program owner, I want explicit continuation floors, so that compute is not spent healing candidates with catastrophic damage.

#### Acceptance Criteria

1. WHEN a raw Candidate is screened for recovery, THE Framework SHALL require parse validity and schema validity to be no more than five absolute percentage points below the BF16_Baseline.
2. WHEN a raw Candidate is screened for recovery, THE Framework SHALL require exact tool sequence to be no more than ten absolute percentage points below the BF16_Baseline.
3. WHEN a raw Candidate is screened for recovery, THE Framework SHALL require exact arguments to be no more than fifteen absolute percentage points below the BF16_Baseline.
4. WHEN a raw Candidate is screened for recovery, THE Framework SHALL require GSM8K and code-execution scores to retain at least 60% of the corresponding BF16_Baseline scores.
5. WHEN a raw Candidate is screened for recovery, THE Framework SHALL require finite teacher divergence, finite reconstruction metrics, and zero Critical_Safety_Violations.
6. IF a raw Candidate fails any continuation floor, THEN THE Framework SHALL block the Recovery_Run unless a distinct remediation experiment is preregistered before blind-result inspection.
7. THE Framework SHALL preregister no-progress, numerical-instability, capability-regression, storage, runtime, provenance, and cost Failure_Gates for every method family.
8. IF a method family reaches the preregistered recovery budget without passing Release_Gates or showing an improving held-out trend, THEN THE Framework SHALL stop that method family.

### Requirement 16: Exact Tensor and Artifact Accounting

**User Story:** As a deployment engineer, I want exact byte accounting, so that nominal one-bit claims cannot hide high-precision tensors or metadata.

#### Acceptance Criteria

1. THE Framework SHALL enumerate every tensor by name, shape, parameter count, semantic role, assigned representation, payload bytes, scale bytes, offset bytes, exception bytes, alignment bytes, and metadata bytes.
2. THE Framework SHALL report Binary_Body and Excluded_Tensors parameter counts and bytes as separate totals.
3. THE Framework SHALL account explicitly for the Iris_Model token embeddings and untied output head rather than including those tensors in Binary_Body one-bit claims.
4. THE Framework SHALL account explicitly for normalization parameters, biases, scales, tokenizer assets, container metadata, padding, indexes, masks, and checksums.
5. WHEN the Framework reports Effective_Bits_Per_Parameter, THE Framework SHALL use total on-disk artifact bytes and the declared original parameter denominator.
6. THE Framework SHALL report ideal binary payload bytes, actual Packed_Artifact bytes, total distributable bytes, and the difference among the three values.
7. WHERE embeddings or the output head use Q4, Q8, BF16, reduced vocabulary, or another representation, THE Framework SHALL report each variant as a separate Candidate.
8. IF measured file bytes differ from the Artifact_Manifest total, THEN THE Framework SHALL block qualification until the discrepancy is zero bytes.
9. THE Framework SHALL report whether each Candidate enters the 200-to-300 decimal-megabyte target band with all required files included.

### Requirement 17: Direct Packed Export and Runtime Parity

**User Story:** As an iPhone user, I want the runtime to consume packed weights directly, so that file compression produces real memory and execution benefits.

#### Acceptance Criteria

1. THE Framework SHALL export final binary signs as physically packed bits with declared scale storage.
2. THE Framework SHALL provide a reference execution path that consumes the Packed_Artifact without materializing a persistent dense weight copy.
3. THE Framework SHALL record exporter revision, runtime revision, build flags, tensor-format version, and Packed_Artifact hash.
4. WHEN a Candidate is exported, THE Framework SHALL decode every packed tensor and verify exact sign values and scale values within preregistered precision tolerances.
5. WHEN Runtime_Parity is evaluated, THE Framework SHALL compare layer outputs, final logits, greedy token sequences, and Iris tool outputs on fixed prompts.
6. THE Framework SHALL define numerical tolerances for layer outputs and logits and exact-match requirements for deterministic token sequences before export evaluation.
7. IF deterministic generations or Iris tool decisions differ between qualified training-operator and packed-runtime paths, THEN THE Framework SHALL reject Runtime_Parity.
8. IF the target runtime expands all binary weights into a persistent higher-precision representation, THEN THE Framework SHALL fail the direct-runtime Release_Gate.
9. THE Framework SHALL derive each deployment format from an identified training checkpoint rather than from another quantized deployment format.

### Requirement 18: Runtime Memory, Latency, Energy, and Thermal Qualification

**User Story:** As an owner of an older iPhone, I want the converted model measured in the complete application, so that the selected artifact operates within device limits.

#### Acceptance Criteria

1. THE Framework SHALL qualify release Candidates on a physical iPhone X, a physical iPhone 11, and one newer physical iPhone control device.
2. THE Framework SHALL measure resident packed weights, tokenizer tables, key-value cache at 128/512/1,024/4,096 prompt tokens, activations, temporary tensors, runtime scratch, allocator overhead, application memory, and cold-load duplicate copies.
3. THE Framework SHALL record device model, OS version, battery health, storage state, thermal state, memory entitlement, runtime revision, thread count, context length, prompt tokens, generated tokens, and decoding settings.
4. WHEN a Candidate runs on iPhone X, THE Framework SHALL require a peak physical footprint no greater than 1.20 GiB.
5. WHEN a Candidate runs on any qualification device, THE Framework SHALL require a peak physical footprint no greater than 35% of physical RAM.
6. WHEN a Candidate completes device qualification, THE Framework SHALL require zero memory warnings, jetsam events, crashes, and timeouts.
7. WHEN warm tool-call latency is measured, THE Framework SHALL require p95 no greater than 3 seconds on iPhone X and 2 seconds on iPhone 11 and the newer control device.
8. WHEN 64-token end-to-end generation latency is measured, THE Framework SHALL require p95 no greater than 8 seconds on iPhone X and 6 seconds on iPhone 11 and the newer control device.
9. WHEN a 30-request thermal loop is executed, THE Framework SHALL require zero serious or critical thermal states and a final-five median latency no greater than 1.20 times the first-five median latency.
10. WHILE offline mode is active, THE Framework SHALL require zero network requests during model load and inference.
11. WHERE a device energy measurement API is available, THE Framework SHALL report energy per generated token over the standardized run.
12. THE Framework SHALL report cold-load time, time to first token, prefill throughput, decode throughput, and battery change over the standardized run.

### Requirement 19: Independent Ternary, Pruning, and Distillation Controls

**User Story:** As a research decision maker, I want independent controls, so that binary conversion is compared fairly without mixing incompatible methods.

#### Acceptance Criteria

1. THE Framework SHALL provide isolated experiment interfaces for ternary conversion, structured pruning, and student distillation Controls.
2. THE Framework SHALL prevent a Control result from being labeled as a dense-to-binary result.
3. WHEN two Experiment_Families are compared, THE Framework SHALL use the identical Sealed_Baseline references, Frozen_Evaluation_Set version, metric definitions, and artifact-accounting rules.
4. THE Framework SHALL report training data, teacher use, parameter count, representation, compute, artifact bytes, Effective_Bits_Per_Parameter, quality, and device results separately for every Experiment_Family.
5. WHERE a hybrid experiment combines two Experiment_Families, THE Framework SHALL register the hybrid as a new Experiment_Family and include single-family component Controls.
6. THE Framework SHALL support a ternary Control, a 430-to-500-million-parameter structured-pruning Q4 Control, and a 270-to-500-million-parameter distilled-student Q4 or quantization-aware-training Control.
7. IF a Control lacks direct runtime support or honest metadata accounting, THEN THE Framework SHALL exclude the Control from deployable Pareto comparisons while retaining the research result.

### Requirement 20: Statistical Discipline and Blind Evaluation

**User Story:** As an evaluator, I want statistically disciplined comparisons, so that random variation and repeated tuning do not determine promotion.

#### Acceptance Criteria

1. THE Framework SHALL preregister primary metrics, confidence procedures, continuation floors, Release_Gates, and stop conditions before examining blind-test outcomes.
2. THE Framework SHALL compute 10,000 paired stratified bootstrap samples over independent semantic families and report 95% confidence intervals for primary paired deltas.
3. THE Framework SHALL preserve evaluation seeds, prompt order, decoding settings, evaluator revision, and case-level outputs.
4. WHEN repeated development evaluations are conducted, THE Framework SHALL record evaluation count and prevent the development set from being represented as a blind test.
5. THE Framework SHALL use a grouped private set with labels for clarification, confirmation, denial, safety, prose, and exact Iris tool behavior for final release evidence.
6. IF a threshold is changed after blind-test inspection, THEN THE Framework SHALL invalidate the affected blind-test promotion decision and require a newly frozen test set.

### Requirement 21: Promotion, Release, and Audit Report

**User Story:** As a product owner, I want one evidence-backed release decision, so that the smallest artifact is selected only after capability and device qualification.

#### Acceptance Criteria

1. THE Framework SHALL require Release_Gate passage for semantic quality, Critical_Safety_Violations, artifact size, Runtime_Parity, memory, latency, thermal behavior, provenance, licensing, and Deterministic_Reproduction before release promotion.
2. THE Framework SHALL select the smallest qualifying Candidate by total distributable bytes rather than nominal weight precision.
3. IF no Candidate passes every Release_Gate, THEN THE Framework SHALL report no qualifying binary release rather than lowering a threshold after evaluation.
4. IF a smaller Candidate fails a Release_Gate and a larger Candidate passes every Release_Gate, THEN THE Framework SHALL prefer the larger Candidate for the applicable device tier.
5. WHEN a Candidate is promoted, THE Framework SHALL produce an audit report containing hypothesis statuses, configurations, ablations, failures, capability metrics, artifact accounting, Runtime_Parity evidence, device measurements, licenses, and unresolved risks.
6. THE Framework SHALL preserve all promoted checkpoints, Packed_Artifacts, manifests, raw outputs, reports, and baseline references by cryptographic hash.
7. THE Framework SHALL keep the BF16_Baseline and Q4_Baseline available after any binary Candidate promotion.
8. IF distribution terms, data provenance, or required attribution remain unresolved, THEN THE Framework SHALL block public release while retaining private research evidence.
