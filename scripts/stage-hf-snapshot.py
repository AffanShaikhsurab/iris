#!/usr/bin/env python3
"""Stage an immutable Hugging Face snapshot and write a hashed manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

MODEL_ID = "openbmb/MiniCPM5-1B"
REVISION = "4e9de7a0778dc1c362e983e6858f0e77542cbdca"
TOKENIZER_FILES = {
    "chat_template.jinja", "config.json", "generation_config.json",
    "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
}
EXCLUDED_FILES = {".gitattributes", "README.md", "README-cn.md"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--full-model", action="store_true")
    mode.add_argument("--tokenizer-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import HfApi, snapshot_download
    info = HfApi().model_info(MODEL_ID, revision=REVISION)
    if info.sha != REVISION:
        raise SystemExit(f"resolved revision {info.sha!r} differs from {REVISION}")
    available = {item.rfilename for item in info.siblings}
    selected = sorted(available - EXCLUDED_FILES if args.full_model else TOKENIZER_FILES)
    if not TOKENIZER_FILES <= set(selected):
        raise SystemExit(f"snapshot is missing tokenizer files: {sorted(TOKENIZER_FILES - set(selected))}")
    if args.full_model and not any(name.endswith(".safetensors") for name in selected):
        raise SystemExit("full snapshot contains no safetensors weights")
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        with tempfile.TemporaryDirectory() as cache:
            source = Path(snapshot_download(MODEL_ID, revision=REVISION, allow_patterns=selected, cache_dir=cache))
            for name in selected:
                target = staging / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / name, target)
        files = [
            {"path": name, "bytes": (staging / name).stat().st_size, "sha256": digest(staging / name)}
            for name in selected
        ]
        manifest = {
            "schema_version": 1,
            "model_id": MODEL_ID,
            "revision": REVISION,
            "snapshot_kind": "full" if args.full_model else "tokenizer-only",
            "files": files,
        }
        (staging / "model-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from iris_training.artifacts import verify_model_manifest
        verify_model_manifest(staging, require_weights=args.full_model)
        staging.replace(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
