"""All-row deterministic preflight for schema, templates, masks, and lengths."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from . import MODEL_ID, MODEL_REVISION, UPSTREAM_TEMPLATE_SHA256
from .data import DataError, compact_json, normalize_row, read_jsonl, render_training_example

TRAINING_TEMPLATE_SHA256 = "c3a3f93fc404b41bd0e03aa4296ecfbe6c75be1526f902f82606abfb75cf710d"
UPSTREAM_TEMPLATE_URL = (
    "https://huggingface.co/openbmb/MiniCPM5-1B/blob/"
    f"{MODEL_REVISION}/chat_template.jinja"
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def _redacted(row: dict[str, Any], file: Path, line: int, length: int, supervised: int) -> dict[str, Any]:
    normalized = normalize_row(row, f"{file}:{line}")
    names = [
        call["function"]["name"]
        for message in normalized["messages"]
        for call in message.get("tool_calls", [])
    ]
    return {
        "file": file.name,
        "line": line,
        "message_roles": [message["role"] for message in normalized["messages"]],
        "target_tools": names,
        "tokens": length,
        "supervised_tokens": supervised,
    }
def build_report(
    tokenizer: Any,
    files: list[Path],
    template: str,
    *,
    max_length: int,
) -> dict[str, Any]:
    upstream = getattr(tokenizer, "chat_template", None)
    if not isinstance(upstream, str) or sha256_bytes(upstream.encode()) != UPSTREAM_TEMPLATE_SHA256:
        raise DataError("pinned tokenizer inference template is missing or has an unexpected hash")
    if sha256_bytes(template.encode()) != TRAINING_TEMPLATE_SHA256:
        raise DataError("vendored training template hash differs from its reviewed generation-span patch")
    report: dict[str, Any] = {
        "schema_version": 1,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "trust_remote_code": False},
        "template": {
            "upstream_url": UPSTREAM_TEMPLATE_URL,
            "upstream_sha256": UPSTREAM_TEMPLATE_SHA256,
            "training_sha256": TRAINING_TEMPLATE_SHA256,
            "patch": "one generation block confined to every rendered assistant branch",
            "inference_template_retained": True,
        },
        "files": {},
        "totals": {"rows": 0, "accepted": 0, "rejected": 0, "parallel_rows": 0, "observations": 0},
        "redacted_examples": [],
        "errors": [],
    }
    special_examples: dict[str, dict[str, Any]] = {}
    for path in files:
        lengths: list[int] = []
        supervised_lengths: list[int] = []
        normalized_digest = hashlib.sha256()
        file_rows = file_ok = file_parallel = file_observations = 0
        for line_number, row in read_jsonl(path):
            file_rows += 1
            try:
                canonical = normalize_row(row, f"{path}:{line_number}")
                normalized_digest.update((compact_json(canonical) + "\n").encode())
                rendered = render_training_example(
                    tokenizer,
                    row,
                    max_length=max_length,
                    chat_template=template,
                    location=f"{path}:{line_number}",
                )
                length, supervised = len(rendered.input_ids), sum(rendered.assistant_mask)
                lengths.append(length)
                supervised_lengths.append(supervised)
                file_ok += 1
                file_parallel += int(rendered.parallel)
                file_observations += rendered.observation_count
                example = _redacted(row, path, line_number, length, supervised)
                if len(report["redacted_examples"]) < 10:
                    report["redacted_examples"].append(example)
                if rendered.parallel and "parallel" not in special_examples:
                    special_examples["parallel"] = example
                if rendered.observation_count and "observation" not in special_examples:
                    special_examples["observation"] = example
            except Exception as exc:
                report["errors"].append(
                    {"file": path.name, "line": line_number, "type": type(exc).__name__, "message": str(exc)}
                )
        rejected = file_rows - file_ok
        stats: dict[str, Any] = {
            "source_sha256": sha256_file(path),
            "normalized_sha256": normalized_digest.hexdigest(),
            "rows": file_rows,
            "accepted": file_ok,
            "rejected": rejected,
            "parallel_rows": file_parallel,
            "observations": file_observations,
        }
        if lengths:
            stats["lengths"] = {
                "p50": percentile(lengths, 0.50),
                "p95": percentile(lengths, 0.95),
                "p99": percentile(lengths, 0.99),
                "max": max(lengths),
                "supervised_tokens": sum(supervised_lengths),
            }
        report["files"][path.name] = stats
        report["totals"]["rows"] += file_rows
        report["totals"]["accepted"] += file_ok
        report["totals"]["rejected"] += rejected
        report["totals"]["parallel_rows"] += file_parallel
        report["totals"]["observations"] += file_observations
    for slot, key in enumerate(("parallel", "observation"), start=8):
        example = special_examples.get(key)
        if example and example not in report["redacted_examples"]:
            if len(report["redacted_examples"]) <= slot:
                report["redacted_examples"].append(example)
            else:
                report["redacted_examples"][slot] = example
    report["redacted_examples"] = report["redacted_examples"][:10]
    report["zero_rejected"] = report["totals"]["rejected"] == 0
    return report
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="Pre-staged pinned model channel")
    parser.add_argument("--files", type=Path, nargs="+", required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from .artifacts import verify_model_manifest
    from transformers import AutoTokenizer

    verify_model_manifest(args.model_dir, require_weights=False)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    template = args.template.read_text(encoding="utf-8")
    report = build_report(tokenizer, args.files, template, max_length=args.max_length)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    if not report["zero_rejected"]:
        raise SystemExit(f"preflight rejected {report['totals']['rejected']} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
