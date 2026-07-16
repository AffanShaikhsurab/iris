"""One-GPU, sealed SageMaker LoRA, explicit QLoRA, or full-parameter trainer."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from . import MODEL_ID, MODEL_REVISION, UPSTREAM_TEMPLATE_SHA256

DEFAULT_CONFIG = Path(os.environ.get("IRIS_CONFIG", "/opt/iris/configs/iris-sft.yaml"))
TRAINING_TEMPLATE_SHA256 = "c3a3f93fc404b41bd0e03aa4296ecfbe6c75be1526f902f82606abfb75cf710d"


class ConfigError(ValueError):
    pass


def load_config(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConfigError("configuration must be a mapping")
    validate_config(value)
    return value


def _require(config: dict[str, Any], dotted: str, expected: Any) -> None:
    value: Any = config
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            raise ConfigError(f"missing {dotted}")
        value = value[key]
    if value != expected:
        raise ConfigError(f"{dotted} must be {expected!r}, got {value!r}")


def validate_config(config: dict[str, Any]) -> None:
    exact = {
        "schema_version": 1,
        "model.id": MODEL_ID,
        "model.revision": MODEL_REVISION,
        "model.tokenizer_revision": MODEL_REVISION,
        "model.trust_remote_code": False,
        "model.allow_hub_download": False,
        "model.mode": "non-thinking",
        "model.max_length": 4096,
        "model.attention": "sdpa",
        "model.dtype": "bfloat16",
        "model.upstream_template_sha256": UPSTREAM_TEMPLATE_SHA256,
        "model.training_template_sha256": TRAINING_TEMPLATE_SHA256,
        "train.packing": False,
        "train.assistant_only_loss": True,
    }
    for dotted, expected in exact.items():
        _require(config, dotted, expected)
    if not isinstance(config.get("train", {}).get("gradient_checkpointing"), bool):
        raise ConfigError("train.gradient_checkpointing must be true or false")
    method = config["peft"].get("method")
    qlora = config["peft"].get("qlora", {}).get("enabled") is True
    if method not in {"lora", "qlora", "full"}:
        raise ConfigError("peft.method must be lora, qlora, or full")
    if qlora != (method == "qlora"):
        raise ConfigError("QLoRA requires both peft.method=qlora and qlora.enabled=true")
    train = config["train"]
    if method == "full":
        profile = (None, None, None, 1, 16, 1, 0.00001)
    elif method == "lora":
        profile = (16, 32, "all-linear", 2, 8, 3, 0.0001)
    else:
        profile = (32, 64, "all-linear", 1, 16, 3, 0.0001)
    actual = (
        config["peft"].get("rank"), config["peft"].get("alpha"),
        config["peft"].get("target_modules"), train.get("per_device_train_batch_size"),
        train.get("gradient_accumulation_steps"), train.get("epochs"),
        train.get("learning_rate"),
    )
    if actual != profile:
        raise ConfigError(
            f"{method} profile must be rank/alpha/targets/microbatch/accumulation/epochs/lr={profile}"
        )
    if method != "full" and config["peft"].get("dropout") != 0.05:
        raise ConfigError("LoRA dropout must be 0.05")
    max_steps = train.get("max_steps", -1)
    if not isinstance(max_steps, int) or max_steps == 0 or max_steps < -1:
        raise ConfigError("max_steps must be -1 or a positive integer")
    runtime = train.get("max_runtime_seconds")
    if not isinstance(runtime, int) or not 1 <= runtime <= 86400:
        raise ConfigError("max_runtime_seconds must be between 1 and 86400")
def training_plan(config: dict[str, Any]) -> dict[str, Any]:
    validate_config(config)
    train = config["train"]
    return {
        "method": config["peft"]["method"],
        "effective_batch": train["per_device_train_batch_size"] * train["gradient_accumulation_steps"],
        "epochs": train["epochs"],
        "max_length": config["model"]["max_length"],
        "assistant_only_loss": train["assistant_only_loss"],
        "packing": train["packing"],
        "one_gpu": True,
        "offline": not config["model"]["allow_hub_download"],
    }


def _one_jsonl(directory: Path) -> Path:
    files = sorted(directory.rglob("*.jsonl"))
    if len(files) != 1:
        raise ConfigError(f"expected exactly one JSONL in {directory}, found {len(files)}")
    return files[0]


def _verify_local_model(model_dir: Path) -> None:
    from .artifacts import verify_model_manifest

    try:
        verify_model_manifest(model_dir, require_weights=True)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ConfigError(str(exc)) from exc


def _latest_checkpoint(directory: Path) -> str | None:
    checkpoints = [path for path in directory.glob("checkpoint-*") if path.is_dir()]
    if not checkpoints:
        return None
    return str(max(checkpoints, key=lambda path: int(path.name.rsplit("-", 1)[-1])))


def _dataset_rows(path: Path, tokenizer: Any, template: str, max_length: int) -> list[dict[str, list[int]]]:
    from .data import read_jsonl, render_training_example

    rows: list[dict[str, list[int]]] = []
    for line, raw in read_jsonl(path):
        item = render_training_example(
            tokenizer, raw, max_length=max_length, chat_template=template, location=f"{path}:{line}"
        )
        rows.append({"input_ids": item.input_ids, "labels": item.labels})
    return rows


def run(config_path: Path) -> None:
    config = load_config(config_path)
    run_id = os.environ.get("IRIS_RUN_ID", "local")
    attempt_id = os.environ.get("IRIS_ATTEMPT_ID", "local")
    is_main = int(os.environ.get("RANK", "0")) == 0
    if is_main:
        print(f"IRIS_EVENT phase=bootstrap status=start run_id={run_id} attempt_id={attempt_id}", flush=True)
    paths = {key: Path(value) for key, value in config["paths"].items()}
    model_dir = paths["model_channel"]
    _verify_local_model(model_dir)

    import torch
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer,
        TrainerCallback, TrainingArguments,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("Iris training requires a visible CUDA GPU")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1 and torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Single-process runs require exactly one visible GPU; set CUDA_VISIBLE_DEVICES=0 "
            "for one GPU, or launch with torchrun --nproc_per_node=<N> for multi-GPU DDP"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    template_path = Path(config["model"]["training_template"])
    template = template_path.read_text(encoding="utf-8")
    train_file = _one_jsonl(paths["train_channel"])
    eval_file = _one_jsonl(paths["eval_channel"])

    from .preflight import build_report
    report = build_report(tokenizer, [train_file, eval_file], template, max_length=4096)
    if not report["zero_rejected"]:
        raise RuntimeError("all-row preflight rejected data; GPU model loading refused")
    if is_main:
        report_path = Path(config["data"]["preflight_report"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if is_main:
        print(
            "IRIS_EVENT phase=data_preflight status=pass "
            f"rows_train={report['files'][train_file.name]['accepted']} "
            f"rows_eval={report['files'][eval_file.name]['accepted']}",
            flush=True,
        )
    method = config["peft"]["method"]
    model_kwargs: dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": False,
        "attn_implementation": "sdpa",
        "torch_dtype": torch.bfloat16,
    }
    if method == "qlora":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForCausalLM.from_pretrained(model_dir, **model_kwargs)
    if method in {"lora", "qlora"}:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if method == "qlora":
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        peft_config = config["peft"]
        model = get_peft_model(
            model,
            LoraConfig(
                task_type="CAUSAL_LM",
                r=peft_config["rank"],
                lora_alpha=peft_config["alpha"],
                lora_dropout=peft_config["dropout"],
                target_modules=peft_config["target_modules"],
                bias="none",
            ),
        )
    model.config.use_cache = False
    if config["train"]["gradient_checkpointing"]:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    train_rows = _dataset_rows(train_file, tokenizer, template, 4096)
    eval_rows = _dataset_rows(eval_file, tokenizer, template, 4096)

    class ListDataset(torch.utils.data.Dataset):
        def __init__(self, values: list[dict[str, list[int]]]) -> None:
            self.values = values

        def __len__(self) -> int:
            return len(self.values)

        def __getitem__(self, index: int) -> dict[str, list[int]]:
            return self.values[index]

    def collate(features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        width = max(len(item["input_ids"]) for item in features)
        ids, labels, attention = [], [], []
        for item in features:
            padding = width - len(item["input_ids"])
            ids.append(item["input_ids"] + [tokenizer.pad_token_id] * padding)
            labels.append(item["labels"] + [-100] * padding)
            attention.append([1] * len(item["input_ids"]) + [0] * padding)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }

    class IrisMetricsCallback(TrainerCallback):
        metric_names = {
            "loss": "train_loss",
            "eval_loss": "eval_loss",
            "learning_rate": "learning_rate",
            "grad_norm": "grad_norm",
            "tokens_per_second": "tokens_sec",
        }

        def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
            if not state.is_world_process_zero:
                return
            for source, target in self.metric_names.items():
                value = (logs or {}).get(source)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    print(f"IRIS_METRIC {target}={float(value):.12g}; step={state.global_step}", flush=True)

    train = config["train"]
    checkpoint_dir = paths["checkpoints"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = Path(os.environ.get("IRIS_TENSORBOARD_DIR", "/opt/ml/output/tensorboard"))
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    arguments = TrainingArguments(
        output_dir=str(checkpoint_dir),
        logging_dir=str(tensorboard_dir),
        num_train_epochs=train["epochs"],
        max_steps=train.get("max_steps", -1),
        learning_rate=train["learning_rate"],
        per_device_train_batch_size=train["per_device_train_batch_size"],
        per_device_eval_batch_size=train["per_device_eval_batch_size"],
        gradient_accumulation_steps=train["gradient_accumulation_steps"],
        gradient_checkpointing=train["gradient_checkpointing"],
        group_by_length=True,
        bf16=True,
        tf32=True,
        eval_strategy="steps",
        eval_steps=train["eval_steps"],
        save_steps=train["save_steps"],
        logging_steps=train["logging_steps"],
        save_total_limit=train["save_total_limit"],
        warmup_ratio=train["warmup_ratio"],
        weight_decay=train["weight_decay"],
        max_grad_norm=train["max_grad_norm"],
        lr_scheduler_type=train["lr_scheduler_type"],
        seed=train["seed"],
        data_seed=train["seed"],
        save_safetensors=True,
        report_to=[],
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=ListDataset(train_rows),
        eval_dataset=ListDataset(eval_rows),
        data_collator=collate,
        callbacks=[IrisMetricsCallback()],
    )
    if is_main:
        print("IRIS_EVENT phase=train status=start step=0", flush=True)
    resume = _latest_checkpoint(checkpoint_dir) if os.environ.get("IRIS_RESUME") == "1" else None
    result = trainer.train(resume_from_checkpoint=resume)
    output_dir = paths["output"]
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output_dir)
        from .artifacts import sha256_file, write_manifest
        manifest_path = write_manifest(
            output_dir,
            config_path=config_path,
            training_template=template_path,
            dataset_hashes={"train": sha256_file(train_file), "validation": sha256_file(eval_file)},
            extra={"metrics": result.metrics, "plan": training_plan(config)},
        )
        print(
            f"IRIS_EVENT phase=finalize status=pass artifact_sha256={sha256_file(manifest_path)}",
            flush=True,
        )
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> int:
    run(parse_args().config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
