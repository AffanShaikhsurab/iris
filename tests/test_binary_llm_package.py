from __future__ import annotations

import importlib
import sys
import tomllib
from pathlib import Path

import pytest
from hypothesis import settings
from hypothesis.database import DirectoryBasedExampleDatabase


BOUNDARIES = ("domain", "math", "orchestration", "adapters", "export", "reporting")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_binary_llm_declares_all_public_package_boundaries():
    package = importlib.import_module("binary_llm")

    assert len(package.__all__) == len(BOUNDARIES)
    assert set(package.__all__) == set(BOUNDARIES)


@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_binary_llm_package_boundary_imports_without_iris_training(boundary: str):
    sys.modules.pop("iris_training", None)

    module = importlib.import_module(f"binary_llm.{boundary}")

    assert module.__name__ == f"binary_llm.{boundary}"
    assert "iris_training" not in sys.modules


def test_hypothesis_ci_profile_is_deterministic_and_persists_failures():
    profile = settings.get_profile("ci")
    project = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pytest_options = project["tool"]["pytest"]["ini_options"]

    assert profile.max_examples >= 100
    assert profile.deadline is None
    assert isinstance(profile.database, DirectoryBasedExampleDatabase)
    assert Path(profile.database.path) == Path(".hypothesis/examples")
    assert "--hypothesis-profile=ci" in pytest_options["addopts"]
    assert "--hypothesis-seed=0" in pytest_options["addopts"]


def test_hypothesis_is_an_exactly_pinned_development_dependency():
    project = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    development_dependencies = project["project"]["optional-dependencies"]["dev"]

    assert development_dependencies == ["hypothesis==6.156.2"]
