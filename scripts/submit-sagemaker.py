#!/usr/bin/env python3
"""Build a SageMaker CreateTrainingJob request; dry-run unless --submit is explicit."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import secrets
from pathlib import Path
from typing import Any

MODEL_REVISION = "4e9de7a0778dc1c362e983e6858f0e77542cbdca"
DIGEST_IMAGE = re.compile(r"^[0-9]+\.dkr\.ecr\.[a-z0-9-]+\.amazonaws\.com/[a-z0-9._/-]+@sha256:[0-9a-f]{64}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
IDENTITY = re.compile(r"^[A-Za-z0-9-]{1,63}$")
CONFIG_NAMES = ("iris-sft.yaml", "iris-sft-pilot.yaml", "iris-sft-full.yaml")
METRIC_DEFINITIONS = [
    {"Name": "train:loss", "Regex": r"IRIS_METRIC train_loss=([-+0-9.eE]+);"},
    {"Name": "eval:loss", "Regex": r"IRIS_METRIC eval_loss=([-+0-9.eE]+);"},
    {"Name": "train:learning_rate", "Regex": r"IRIS_METRIC learning_rate=([-+0-9.eE]+);"},
    {"Name": "train:tokens_sec", "Regex": r"IRIS_METRIC tokens_sec=([-+0-9.eE]+);"},
    {"Name": "train:grad_norm", "Regex": r"IRIS_METRIC grad_norm=([-+0-9.eE]+);"},
    {"Name": "eval:json_valid", "Regex": r"IRIS_METRIC json_valid=([-+0-9.eE]+);"},
    {"Name": "eval:route_accuracy", "Regex": r"IRIS_METRIC route_accuracy=([-+0-9.eE]+);"},
]


def _s3(uri: str, bucket: str) -> str:
    if not uri.startswith(f"s3://{bucket}/") or any(part in uri.lower() for part in ("latest", "mutable")):
        raise ValueError(f"input must use immutable s3://{bucket}/ prefix: {uri}")
    return uri


def _channel(name: str, uri: str) -> dict[str, Any]:
    return {
        "ChannelName": name,
        "DataSource": {"S3DataSource": {
            "S3DataType": "S3Prefix", "S3Uri": uri, "S3DataDistributionType": "FullyReplicated"
        }},
        "InputMode": "File",
    }


def build_request(args: argparse.Namespace) -> dict[str, Any]:
    if not DIGEST_IMAGE.fullmatch(args.image):
        raise ValueError("--image must be a private ECR URI pinned with @sha256:<64 hex>")
    if not args.role_arn.startswith("arn:aws:iam::") or ":role/" not in args.role_arn:
        raise ValueError("--role-arn must be a pre-existing IAM role ARN")
    if not args.kms_key_arn.startswith("arn:aws:kms:"):
        raise ValueError("--kms-key-arn must be a pre-existing KMS key ARN")
    if not SHA256.fullmatch(args.dataset_sha) or not SHA256.fullmatch(args.source_revision):
        raise ValueError("dataset and source revisions must be full SHA-256 values")
    if args.model_revision != MODEL_REVISION:
        raise ValueError("model revision must equal the reviewed MiniCPM5 commit")
    if not 1 <= args.max_runtime <= 86400:
        raise ValueError("max runtime must be between 1 and 86400 seconds")
    run_id = args.run_id or (
        dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + f"-{args.source_revision[:7]}-{args.dataset_sha[:12]}"
    )
    if not IDENTITY.fullmatch(run_id):
        raise ValueError("run ID must be 1-63 alphanumeric/hyphen characters")
    attempt_id = args.attempt_id or f"{run_id[:51]}-a001-{secrets.token_hex(3)}"
    if not IDENTITY.fullmatch(attempt_id):
        raise ValueError("attempt ID must be 1-63 alphanumeric/hyphen characters")
    if args.config_name == "iris-sft-full.yaml" and args.instance_type != "ml.g6e.2xlarge":
        raise ValueError("the full-tuning recipe requires the 48 GB ml.g6e.2xlarge profile")
    prefix = f"s3://{args.bucket}/iris-sft/runs/{run_id}"
    attempt_prefix = f"{prefix}/attempts/{attempt_id}"
    image_digest = args.image.rsplit("@", 1)[1]
    return {
        "TrainingJobName": attempt_id,
        "RoleArn": args.role_arn,
        "AlgorithmSpecification": {
            "TrainingImage": args.image,
            "TrainingInputMode": "File",
            "ContainerEntrypoint": ["python", "-m", "iris_training.train"],
            "MetricDefinitions": METRIC_DEFINITIONS,
        },
        "InputDataConfig": [
            _channel("train", _s3(args.train_s3, args.bucket)),
            _channel("validation", _s3(args.eval_s3, args.bucket)),
            _channel("model", _s3(args.model_s3, args.bucket)),
        ],
        "OutputDataConfig": {"S3OutputPath": f"{attempt_prefix}/output/", "KmsKeyId": args.kms_key_arn},
        "CheckpointConfig": {"S3Uri": f"{prefix}/resume/checkpoints/", "LocalPath": "/opt/ml/checkpoints"},
        "TensorBoardOutputConfig": {
            "S3OutputPath": f"{attempt_prefix}/tensorboard/",
            "LocalPath": "/opt/ml/output/tensorboard",
        },
        "ResourceConfig": {
            "InstanceType": args.instance_type, "InstanceCount": 1,
            "VolumeSizeInGB": args.volume_gb, "VolumeKmsKeyId": args.kms_key_arn,
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": args.max_runtime},
        "EnableNetworkIsolation": True,
        "EnableInterContainerTrafficEncryption": True,
        "Environment": {
            "IRIS_RUN_ID": run_id,
            "IRIS_ATTEMPT_ID": attempt_id,
            "IRIS_IMAGE_DIGEST": image_digest,
            "IRIS_SOURCE_REVISION": args.source_revision,
            "IRIS_CONFIG": f"/opt/iris/configs/{args.config_name}",
        },
        "Tags": [
            {"Key": "Project", "Value": "iris-sft"},
            {"Key": "RunId", "Value": run_id},
            {"Key": "AttemptId", "Value": attempt_id},
            {"Key": "DatasetSha256", "Value": args.dataset_sha},
            {"Key": "ModelRevision", "Value": args.model_revision},
            {"Key": "SourceRevision", "Value": args.source_revision},
        ],
    }
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role-arn", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--kms-key-arn", required=True)
    parser.add_argument("--train-s3", required=True)
    parser.add_argument("--eval-s3", required=True)
    parser.add_argument("--model-s3", required=True)
    parser.add_argument("--dataset-sha", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument(
        "--instance-type",
        choices=("ml.g6.2xlarge", "ml.g5.2xlarge", "ml.g6e.2xlarge"),
        default="ml.g6.2xlarge",
    )
    parser.add_argument("--volume-gb", type=int, default=100)
    parser.add_argument("--max-runtime", type=int, default=86400)
    parser.add_argument("--config-name", choices=CONFIG_NAMES, default="iris-sft-pilot.yaml")
    parser.add_argument("--run-id")
    parser.add_argument("--attempt-id")
    parser.add_argument("--region")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--submit", action="store_true", help="Actually call CreateTrainingJob")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    request = build_request(args)
    rendered = json.dumps(request, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not args.submit:
        print("DRY RUN: no AWS API call made; pass --submit explicitly to create the job.")
        return 0
    import boto3

    response = boto3.client("sagemaker", region_name=args.region).create_training_job(**request)
    print(json.dumps({"TrainingJobArn": response["TrainingJobArn"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
