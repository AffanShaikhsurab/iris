from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "submit_sagemaker", Path("scripts/submit-sagemaker.py")
)
submit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(submit)

IMAGE = "111122223333.dkr.ecr.us-east-1.amazonaws.com/iris-sft-train@sha256:" + "a" * 64
KMS = "arn:aws:kms:us-east-1:111122223333:key/abc"


def make(**overrides):
    base = dict(
        role_arn="arn:aws:iam::111122223333:role/IrisSageMakerTrainingRole",
        bucket="iris-bkt",
        image=IMAGE,
        kms_key_arn=KMS,
        train_s3="s3://iris-bkt/iris-sft/datasets/deadbeef/train/",
        eval_s3="s3://iris-bkt/iris-sft/datasets/deadbeef/validation/",
        model_s3="s3://iris-bkt/iris-sft/base-models/minicpm5-1b/rev/",
        dataset_sha="0" * 64,
        source_revision="1" * 64,
        model_revision=submit.MODEL_REVISION,
        instance_type="ml.g6.2xlarge",
        volume_gb=100,
        max_runtime=7200,
        config_name="iris-sft-pilot.yaml",
        run_id="testrun",
        attempt_id="testrun-a001",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_request_has_entrypoint_metrics_tensorboard_and_identity():
    request = submit.build_request(make())
    algorithm = request["AlgorithmSpecification"]
    assert algorithm["ContainerEntrypoint"] == ["python", "-m", "iris_training.train"]
    assert [item["Name"] for item in algorithm["MetricDefinitions"]] == [
        "train:loss", "eval:loss", "train:learning_rate", "train:tokens_sec",
        "train:grad_norm", "eval:json_valid", "eval:route_accuracy",
    ]
    assert request["TrainingJobName"] == "testrun-a001"
    assert request["EnableNetworkIsolation"] is True
    assert request["Environment"]["IRIS_RUN_ID"] == "testrun"
    assert request["Environment"]["IRIS_ATTEMPT_ID"] == "testrun-a001"
    assert request["Environment"]["IRIS_CONFIG"] == "/opt/iris/configs/iris-sft-pilot.yaml"
    assert request["TensorBoardOutputConfig"]["S3OutputPath"].endswith(
        "/runs/testrun/attempts/testrun-a001/tensorboard/"
    )
    assert request["CheckpointConfig"]["S3Uri"].endswith("/runs/testrun/resume/checkpoints/")
    assert request["OutputDataConfig"]["S3OutputPath"].endswith(
        "/runs/testrun/attempts/testrun-a001/output/"
    )


def test_full_config_requires_the_48gb_instance():
    with pytest.raises(ValueError, match="full-tuning recipe requires"):
        submit.build_request(make(config_name="iris-sft-full.yaml", instance_type="ml.g6.2xlarge"))
    request = submit.build_request(make(config_name="iris-sft-full.yaml", instance_type="ml.g6e.2xlarge"))
    assert request["ResourceConfig"]["InstanceType"] == "ml.g6e.2xlarge"


def test_mutable_or_wrong_bucket_inputs_are_rejected():
    with pytest.raises(ValueError, match="immutable"):
        submit.build_request(make(train_s3="s3://iris-bkt/iris-sft/datasets/latest/train/"))
    with pytest.raises(ValueError, match="immutable"):
        submit.build_request(make(eval_s3="s3://other-bucket/iris-sft/x/validation/"))


def test_image_must_be_digest_pinned_private_ecr():
    with pytest.raises(ValueError, match="@sha256"):
        submit.build_request(make(image="111122223333.dkr.ecr.us-east-1.amazonaws.com/iris-sft-train:latest"))
