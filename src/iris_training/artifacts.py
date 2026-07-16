"""Deterministic artifact hashing and training manifest helpers."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

from . import MODEL_ID, MODEL_REVISION, UPSTREAM_TEMPLATE_SHA256

PACKAGES = (
    "accelerate", "bitsandbytes", "datasets", "huggingface-hub", "peft",
    "safetensors", "tokenizers", "torch", "transformers",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_inventory(root: Path, excluded: Iterable[str] = ("manifest.json",)) -> list[dict[str, Any]]:
    ignored = set(excluded)
    return [
        {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in ignored
    ]


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def write_manifest(
    output_dir: Path,
    *,
    config_path: Path,
    training_template: Path,
    dataset_hashes: dict[str, str],
    extra: dict[str, Any] | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    copied_template = output_dir / "training_chat_template.jinja"
    shutil.copyfile(training_template, copied_template)
    copied_config = output_dir / "resolved-config.yaml"
    shutil.copyfile(config_path, copied_config)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "trust_remote_code": False,
        },
        "templates": {
            "inference_retained_with_tokenizer": True,
            "upstream_sha256": UPSTREAM_TEMPLATE_SHA256,
            "training_sha256": sha256_file(copied_template),
        },
        "datasets": dict(sorted(dataset_hashes.items())),
        "packages": package_versions(),
        "run": {
            "image_digest": os.environ.get("IRIS_IMAGE_DIGEST", "unknown"),
            "run_id": os.environ.get("IRIS_RUN_ID", "local"),
            "attempt_id": os.environ.get("IRIS_ATTEMPT_ID", "local"),
            "source_revision": os.environ.get("IRIS_SOURCE_REVISION", "unknown"),
        },
    }
    if extra:
        manifest["training"] = extra
    manifest["files"] = file_inventory(output_dir)
    target = output_dir / "manifest.json"
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def verify_model_manifest(root: Path, *, require_weights: bool) -> dict[str, Any]:
    """Verify exact revision, complete inventory, paths, sizes, and hashes."""
    root = root.resolve()
    manifest_path = root / "model-manifest.json"
    if not manifest_path.is_file():
        raise ValueError("sealed model channel must contain model-manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model_id") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError("model channel manifest does not match the exact configured revision")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("model manifest must include a non-empty hashed file inventory")
    recorded: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError("model manifest contains an invalid file entry")
        relative = entry["path"]
        candidate = (root / relative).resolve()
        checksum = entry.get("sha256")
        if relative in recorded or root not in candidate.parents or not candidate.is_file():
            raise ValueError(f"model manifest path is duplicate, missing, or unsafe: {relative}")
        if not isinstance(checksum, str) or len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
            raise ValueError(f"model manifest checksum is invalid: {relative}")
        if entry.get("bytes") != candidate.stat().st_size or sha256_file(candidate) != checksum:
            raise ValueError(f"model file size/hash mismatch: {relative}")
        recorded.add(relative)
    actual = {
        path.relative_to(root).as_posix() for path in root.rglob("*")
        if path.is_file() and path.name != "model-manifest.json"
    }
    if recorded != actual:
        raise ValueError(f"model manifest inventory mismatch: missing={sorted(actual - recorded)}, extra={sorted(recorded - actual)}")
    required = {"chat_template.jinja", "config.json", "tokenizer.json", "tokenizer_config.json"}
    if not required <= recorded:
        raise ValueError(f"model manifest lacks tokenizer files: {sorted(required - recorded)}")
    kind = manifest.get("snapshot_kind")
    if kind not in {"tokenizer-only", "full"}:
        raise ValueError("model manifest has an invalid snapshot_kind")
    weights = [name for name in recorded if name.endswith(".safetensors")]
    if kind == "full" and not weights:
        raise ValueError("full model channel contains no safetensors weights")
    if kind == "tokenizer-only" and weights:
        raise ValueError("tokenizer-only channel unexpectedly contains model weights")
    if require_weights and kind != "full":
        raise ValueError("trainable model channel must be a full snapshot")
    indexes = [name for name in recorded if name.endswith(".safetensors.index.json")]
    if len(indexes) > 1:
        raise ValueError("model channel contains multiple safetensors indexes")
    if indexes:
        index = json.loads((root / indexes[0]).read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("safetensors index has no weight_map")
        shards = set(weight_map.values())
        if not all(isinstance(name, str) and name.endswith(".safetensors") for name in shards):
            raise ValueError("safetensors index contains an invalid shard path")
        if not shards <= recorded:
            raise ValueError(f"safetensors index references missing shards: {sorted(shards - recorded)}")
    return manifest