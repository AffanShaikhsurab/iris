from __future__ import annotations

import builtins
from pathlib import Path

import pytest

from iris_training.train import ConfigError, load_config, training_plan

CONFIG = Path("configs/iris-sft.yaml")


def test_training_config_is_exact_finite_offline_lora_plan():
    config = load_config(CONFIG)
    plan = training_plan(config)
    assert plan == {
        "method": "lora", "effective_batch": 16, "epochs": 3, "max_length": 4096,
        "assistant_only_loss": True, "packing": False, "one_gpu": True, "offline": True,
    }
    assert config["train"]["learning_rate"] == 1e-4
    assert config["peft"] | {"unused": None}


def test_config_and_module_smoke_without_importing_torch(monkeypatch):
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "torch" or name.startswith("transformers") or name.startswith("peft"):
            raise AssertionError(f"heavy import at config/test time: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    config = load_config(CONFIG)
    assert training_plan(config)["effective_batch"] == 16


def test_assistant_loss_and_qlora_cannot_be_enabled_implicitly():
    config = load_config(CONFIG)
    config["train"]["assistant_only_loss"] = False
    with pytest.raises(ConfigError, match="assistant_only_loss"):
        training_plan(config)
    config = load_config(CONFIG)
    config["peft"]["qlora"]["enabled"] = True
    with pytest.raises(ConfigError, match="QLoRA requires"):
        training_plan(config)


def test_container_is_digest_parameterized_and_offline():
    dockerfile = Path("Dockerfile.train").read_text(encoding="utf-8")
    assert "ARG BASE_IMAGE" in dockerfile and "FROM ${BASE_IMAGE}" in dockerfile
    assert "HF_HUB_OFFLINE=1" in dockerfile and "TRANSFORMERS_OFFLINE=1" in dockerfile
    assert "requirements.lock" in dockerfile and "--require-hashes" in dockerfile
