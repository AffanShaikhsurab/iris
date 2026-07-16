#!/usr/bin/env python3
"""Measure which vocabulary tokens Iris actually uses, to scope embedding pruning.

Non-destructive: renders every dataset row through the pinned tokenizer + training
template, counts token usage, unions in the tokens that MUST be kept regardless of
frequency (all special/added tokens plus every single-byte fallback token so any
future input still encodes), and reports coverage and the parameter savings a
vocabulary trim would yield. Writes a keep-list for a later, separate pruning step.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iris_training.data import normalize_row, read_jsonl, to_model_boundary


def _always_keep(tokenizer: Any) -> set[int]:
    """Special/added tokens plus every token that decodes to a single byte/char."""
    keep: set[int] = set()
    for tid in getattr(tokenizer, "all_special_ids", []) or []:
        if isinstance(tid, int):
            keep.add(tid)
    added = getattr(tokenizer, "added_tokens_decoder", {}) or {}
    keep.update(int(tid) for tid in added)
    vocab = tokenizer.get_vocab()
    for token, tid in vocab.items():
        # byte-fallback tokens like <0x41> and single-character tokens keep the
        # tokenizer total-coverage guarantee even after a trim.
        if token.startswith("<0x") and token.endswith(">"):
            keep.add(int(tid))
        else:
            decoded = tokenizer.decode([tid], skip_special_tokens=False)
            if len(decoded) == 1:
                keep.add(int(tid))
    return keep


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="Pinned tokenizer/model dir")
    parser.add_argument("--files", type=Path, nargs="+", required=True, help="Dataset JSONL files")
    parser.add_argument("--template", type=Path, required=True, help="Training chat template")
    parser.add_argument("--output", type=Path, required=True, help="Keep-list + report JSON")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=False, use_fast=True
    )
    template = args.template.read_text(encoding="utf-8")
    vocab_size = len(tokenizer.get_vocab())

    usage: Counter[int] = Counter()
    rows = 0
    for path in args.files:
        for line_number, raw in read_jsonl(path):
            rows += 1
            row = normalize_row(raw, f"{path}:{line_number}")
            boundary = to_model_boundary(row)
            rendered = tokenizer.apply_chat_template(
                boundary["messages"], tools=boundary["tools"], enable_thinking=False,
                chat_template=template, tokenize=True, add_generation_prompt=False,
            )
            ids = rendered["input_ids"] if isinstance(rendered, dict) else rendered
            usage.update(int(t) for t in ids)

    always = _always_keep(tokenizer)
    used = set(usage)
    keep = sorted(used | always)
    dropped = vocab_size - len(keep)
    hidden = getattr(tokenizer, "model_max_length", None)  # informational only
    report = {
        "vocab_size": vocab_size,
        "rows": rows,
        "used_tokens": len(used),
        "always_keep_tokens": len(always),
        "keep_tokens": len(keep),
        "droppable_tokens": dropped,
        "keep_fraction": round(len(keep) / vocab_size, 4),
        "top_tokens": [
            {"id": tid, "token": tokenizer.decode([tid]), "count": count}
            for tid, count in usage.most_common(20)
        ],
        "note": (
            "Embedding+head params scale with vocab. Dropping N tokens saves about "
            "N * hidden * 2 params (untied). Keep-list guarantees total input coverage "
            "via byte-fallback + special tokens; a broad English token set should be "
            "unioned in before actual pruning to protect general-language quality."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"report": report, "keep_ids": keep}, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
