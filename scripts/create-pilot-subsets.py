"""Create deterministic, coverage-preserving Iris pilot subsets and a hash manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from iris_training.data import normalize_row


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stratum(row: dict[str, Any]) -> str:
    calls = [
        message["tool_calls"]
        for message in row["messages"]
        if message["role"] == "assistant"
    ]
    names = sorted({call["function"]["name"] for turn in calls for call in turn})
    parallel = any(len(turn) > 1 for turn in calls)
    observations = sum(message["role"] == "tool" for message in row["messages"])
    return f"tools={'+'.join(names)}|parallel={int(parallel)}|observation={int(observations > 0)}"


def _rank(seed: int, canonical: str) -> str:
    return hashlib.sha256(f"{seed}\0{canonical}".encode()).hexdigest()

def _allocate(counts: dict[str, int], target: int) -> dict[str, int]:
    if target < len(counts):
        raise ValueError(f"target {target} cannot cover all {len(counts)} strata")
    if target > sum(counts.values()):
        raise ValueError(f"target {target} exceeds {sum(counts.values())} unique rows")
    allocation = {name: 1 for name in counts}
    remaining = target - len(counts)
    capacities = {name: count - 1 for name, count in counts.items()}
    total_capacity = sum(capacities.values())
    ideals = {
        name: (remaining * capacity / total_capacity if total_capacity else 0.0)
        for name, capacity in capacities.items()
    }
    for name, ideal in ideals.items():
        allocation[name] += min(capacities[name], math.floor(ideal))
    left = target - sum(allocation.values())
    order = sorted(
        counts,
        key=lambda name: (-(ideals[name] - math.floor(ideals[name])), name),
    )
    while left:
        progressed = False
        for name in order:
            if allocation[name] < counts[name]:
                allocation[name] += 1
                left -= 1
                progressed = True
                if not left:
                    break
        if not progressed:
            raise RuntimeError("unable to complete bounded pilot allocation")
    return allocation


def _select(source: Path, target: int, seed: int) -> tuple[list[str], dict[str, Any]]:
    strata: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    seen: set[str] = set()
    source_rows = 0
    duplicate_rows = 0
    normalized_digest = hashlib.sha256()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            source_rows += 1
            raw = json.loads(line)
            normalized = normalize_row(raw, f"{source}:{line_number}")
            canonical = _json(normalized)
            fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
            if fingerprint in seen:
                duplicate_rows += 1
                continue
            seen.add(fingerprint)
            normalized_digest.update(canonical.encode() + b"\n")
            strata[_stratum(normalized)].append((_rank(seed, canonical), line_number, line.rstrip("\r\n")))
    counts = {name: len(rows) for name, rows in strata.items()}
    allocation = _allocate(counts, target)
    chosen = [item for name, rows in strata.items() for item in sorted(rows)[: allocation[name]]]
    chosen.sort(key=lambda item: item[1])
    report = {
        "source_path": source.as_posix(),
        "source_sha256": _sha256(source),
        "source_rows": source_rows,
        "unique_normalized_rows": len(seen),
        "duplicate_normalized_rows_omitted": duplicate_rows,
        "source_normalized_sha256": normalized_digest.hexdigest(),
        "selected_source_lines": [line_number for _, line_number, _ in chosen],
        "strata": {
            name: {"available": counts[name], "selected": allocation[name]}
            for name in sorted(counts)
        },
    }
    return [line for _, _, line in chosen], report

def _write_subset(rows: list[str], output: Path) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    normalized = hashlib.sha256()
    for line_number, line in enumerate(rows, 1):
        canonical = _json(normalize_row(json.loads(line), f"{output}:{line_number}"))
        normalized.update(canonical.encode() + b"\n")
    return {
        "path": output.as_posix(),
        "rows": len(rows),
        "sha256": _sha256(output),
        "normalized_sha256": normalized.hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=1024)
    parser.add_argument("--eval-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    train_rows, train_source = _select(args.train, args.train_size, args.seed)
    eval_rows, eval_source = _select(args.eval, args.eval_size, args.seed)
    train_output = args.output_dir / "train.jsonl"
    eval_output = args.output_dir / "eval.jsonl"
    manifest = {
        "schema_version": 1,
        "algorithm": "canonical-sha256-stratified-largest-remainder-v1",
        "seed": args.seed,
        "train": {"source": train_source, "output": _write_subset(train_rows, train_output)},
        "eval": {"source": eval_source, "output": _write_subset(eval_rows, eval_output)},
    }
    manifest_path = args.output_dir / "pilot-manifest.json"
    manifest_path.write_text(_json(manifest) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"manifest": manifest_path.as_posix(), "sha256": _sha256(manifest_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
