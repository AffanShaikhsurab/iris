# Iris SageMaker notebooks

## `iris-sagemaker-minicpm5.ipynb`

Thin orchestration notebook for fine-tuning `openbmb/MiniCPM5-1B` on Iris tool-use
data in Amazon SageMaker AI. It validates data, creates deterministic pilot
subsets, stages the exact model revision, runs the all-row CPU template/mask
preflight, publishes immutable S3 inputs, builds a dry-run `CreateTrainingJob`
request, submits only after an explicit confirmation flag, inspects the job, and
optionally indexes the completed run in managed MLflow.

It does **not** train in the notebook kernel. Training runs in a separate sealed
SageMaker Training Job whose entrypoint is `python -m iris_training.train`.

### Before you run
- Read [`../docs/sagemaker-research/08-notebook-walkthrough.md`](../docs/sagemaker-research/08-notebook-walkthrough.md)
  and the plan of record in
  [`../docs/sagemaker-research/07-container-notebook-reference-architecture.md`](../docs/sagemaker-research/07-container-notebook-reference-architecture.md).
- Run in current regular SageMaker Studio → JupyterLab → private CPU space
  `iris-orchestrator` (`ml.t3.medium`), from the repository root at a clean commit.
- Complete the AWS setup (Region/quota, KMS, versioned SSE-KMS S3, narrow IAM role,
  ECR image built and pinned by digest) first.

### Safety
- Dry-run by default. The submission cell asserts `CONFIRM_SUBMIT is True`.
- Immutable upload is behind `UPLOAD_IMMUTABLE_INPUTS`. Full-model staging is behind
  `STAGE_FULL_MODEL`. MLflow indexing is behind `INDEX_IN_MLFLOW`.
- The training request keeps `EnableNetworkIsolation=True`; the container has no
  outbound network access.

### Training profiles
- `configs/iris-sft-pilot.yaml` — deterministic 20-step LoRA pilot (default).
- `configs/iris-sft.yaml` — full LoRA baseline (3 epochs).
- `configs/iris-sft-full.yaml` — full-parameter tuning; requires `ml.g6e.2xlarge`.
