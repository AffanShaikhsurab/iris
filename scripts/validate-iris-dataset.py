#!/usr/bin/env python3
"""Validate and merge the native-format Iris dataset.

The retained google/mobile-actions rows intentionally keep their original
seven-tool catalog. Iris extension rows use the canonical 32-tool catalog.
Both profiles are valid in the merged native dataset; rows are never reshaped.

Usage:
    python scripts/validate-iris-dataset.py
    python scripts/validate-iris-dataset.py --no-write
    python scripts/validate-iris-dataset.py --output path/to/iris.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DEFAULT = ROOT / "tmp" / "mobile-actions.jsonl"
EXTENSION_DEFAULT = ROOT / "data" / "iris-dataset" / "extension"
OUTPUT_DEFAULT = ROOT / "data" / "iris-dataset" / "iris.jsonl"

BASE_TOOLS = {
    "create_calendar_event",
    "create_contact",
    "open_wifi_settings",
    "turn_off_flashlight",
    "turn_on_flashlight",
    "send_email",
    "show_map",
}

EXPECTED_EXTENSION_FILES = {
    "01_answers_search_drafting.jsonl": 160,
    "02_calendar_reminders_tasks.jsonl": 180,
    "03_gmail_memory.jsonl": 180,
    "04_navigation_environment.jsonl": 160,
    "05_chains_safety.jsonl": 160,
}

EXPECTED_EXTENSION_TOOLS = {
    "final_answer", "ask_user", "web_search", "answer_search",
    "summarize_provided_text", "draft_reply", "draft_message",
    "create_note", "create_reminder", "quick_journal", "calendar_lookup",
    "calendar_add", "reminders_lookup", "weather_summary",
    "current_location_summary", "device_status", "open_search",
    "open_destination", "maps_search", "nearby_search", "tasks_list",
    "tasks_add", "tasks_complete", "gmail_search", "gmail_read",
    "draft_email", "send_email", "memory_read", "memory_append",
    "memory_list", "memory_status", "create_calendar_event",
}

ALLOWED_OPEN_SEARCH_TARGETS = {"google", "youtube", "reddit", "perplexity", "maps"}
ALLOWED_OPEN_DESTINATION_TARGETS = {"chatgpt", "perplexity", "calendar"}
ALLOWED_MEMORY_TOPICS = {"index", "profile", "preferences", "log", "notes", "journal"}


class ValidationError(Exception):
    pass


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def catalog_map(tools: Any, location: str) -> dict[str, dict[str, Any]]:
    if not isinstance(tools, list) or not tools:
        raise ValidationError(f"{location}: tools must be a non-empty array")
    result: dict[str, dict[str, Any]] = {}
    for index, wrapper in enumerate(tools):
        if not isinstance(wrapper, dict) or set(wrapper) != {"function"}:
            raise ValidationError(f"{location}: tools[{index}] must contain only function")
        fn = wrapper["function"]
        if not isinstance(fn, dict):
            raise ValidationError(f"{location}: tools[{index}].function must be an object")
        if set(fn) != {"name", "description", "parameters"}:
            raise ValidationError(f"{location}: tools[{index}].function has unexpected keys")
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            raise ValidationError(f"{location}: tools[{index}] has an invalid name")
        if name in result:
            raise ValidationError(f"{location}: duplicate tool name {name!r}")
        params = fn.get("parameters")
        if not isinstance(params, dict) or set(params) not in ({"type", "properties"}, {"type", "properties", "required"}):
            raise ValidationError(f"{location}: {name} has invalid parameter keys")
        if params.get("type") != "OBJECT" or not isinstance(params.get("properties"), dict):
            raise ValidationError(f"{location}: {name} must use an uppercase OBJECT schema")
        required = params.get("required", [])
        if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
            raise ValidationError(f"{location}: {name}.required must be a string array")
        properties = params["properties"]
        if set(required) - set(properties):
            raise ValidationError(f"{location}: {name}.required references unknown properties")
        for prop, schema in properties.items():
            if not isinstance(prop, str) or not isinstance(schema, dict):
                raise ValidationError(f"{location}: {name}.{prop} has an invalid property schema")
            prop_type = schema.get("type")
            if not isinstance(prop_type, str) or not prop_type.isupper():
                raise ValidationError(f"{location}: {name}.{prop} must use an uppercase type")
        result[name] = fn
    return result


def catalog_signature(catalog: dict[str, dict[str, Any]]) -> str:
    return compact_json(catalog)


def row_fingerprint(row: dict[str, Any]) -> str:
    payload = {key: value for key, value in row.items() if key != "metadata"}
    return hashlib.sha256(compact_json(payload).encode("utf-8")).hexdigest()


def validate_row(row: Any, path: Path, line_number: int, expected_catalog: dict[str, dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], Counter[str]]:
    location = f"{path}:{line_number}"
    if not isinstance(row, dict) or set(row) != {"metadata", "tools", "messages"}:
        raise ValidationError(f"{location}: row must contain exactly metadata, tools, messages")
    if row["metadata"] not in {"train", "eval"}:
        raise ValidationError(f"{location}: metadata must be exactly train or eval")

    actual_catalog = catalog_map(row["tools"], location)
    if catalog_signature(actual_catalog) != catalog_signature(expected_catalog):
        raise ValidationError(f"{location}: tool catalog differs from its dataset profile")

    messages = row["messages"]
    if not isinstance(messages, list) or len(messages) < 3:
        raise ValidationError(f"{location}: messages must contain at least developer, user, assistant")
    if messages[0].get("role") != "developer" or messages[1].get("role") != "user":
        raise ValidationError(f"{location}: first messages must be developer then user")

    route_counts: Counter[str] = Counter()
    assistant_count = 0
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict) or "role" not in message:
            raise ValidationError(f"{location}: messages[{message_index}] is invalid")
        role = message["role"]
        if role not in {"developer", "user", "assistant", "tool"}:
            raise ValidationError(f"{location}: unsupported message role {role!r}")
        if role in {"developer", "user", "tool"}:
            if set(message) != {"role", "content"} or not isinstance(message["content"], str):
                raise ValidationError(f"{location}: {role} messages must contain string content only")
            continue

        assistant_count += 1
        if set(message) != {"role", "tool_calls"} or not isinstance(message["tool_calls"], list) or not message["tool_calls"]:
            raise ValidationError(f"{location}: assistant messages must contain non-empty tool_calls only")
        if len(message["tool_calls"]) > 3:
            raise ValidationError(f"{location}: an assistant turn contains more than three tool calls")
        for call_index, call in enumerate(message["tool_calls"]):
            if not isinstance(call, dict) or set(call) != {"function"} or not isinstance(call["function"], dict):
                raise ValidationError(f"{location}: tool_calls[{call_index}] must contain only function")
            fn_call = call["function"]
            if set(fn_call) != {"name", "arguments"}:
                raise ValidationError(f"{location}: tool_calls[{call_index}] must have name and arguments only")
            name = fn_call["name"]
            args = fn_call["arguments"]
            if name not in actual_catalog:
                raise ValidationError(f"{location}: tool call selects a tool absent from its catalog: {name}")
            if not isinstance(args, dict):
                raise ValidationError(f"{location}: {name} arguments must be an object, not a string")
            params = actual_catalog[name]["parameters"]
            properties = params["properties"]
            required = params.get("required", [])
            if set(args) - set(properties):
                raise ValidationError(f"{location}: {name} arguments contain unknown keys")
            if set(required) - set(args):
                raise ValidationError(f"{location}: {name} is missing required arguments")
            if not all(isinstance(value, str) for value in args.values()):
                raise ValidationError(f"{location}: {name} arguments must be string-valued under this catalog")
            if name == "final_answer" and not args.get("answer", "").strip():
                raise ValidationError(f"{location}: final_answer.answer must be non-empty")
            if name == "ask_user" and not args.get("question", "").strip():
                raise ValidationError(f"{location}: ask_user.question must be non-empty")
            if name == "send_email" and "confirm" in properties and args.get("confirm") != "yes":
                raise ValidationError(f"{location}: send_email must use confirm='yes' when called")
            if name == "open_search" and args.get("target") not in ALLOWED_OPEN_SEARCH_TARGETS:
                raise ValidationError(f"{location}: open_search target is not allowlisted")
            if name == "open_destination" and args.get("target") not in ALLOWED_OPEN_DESTINATION_TARGETS:
                raise ValidationError(f"{location}: open_destination target is not allowlisted")
            if name in {"memory_read", "memory_append"} and args.get("topic", "log") not in ALLOWED_MEMORY_TOPICS:
                raise ValidationError(f"{location}: memory topic is not allowlisted")
            route_counts[name] += 1

    if assistant_count == 0:
        raise ValidationError(f"{location}: row has no assistant turn")
    return actual_catalog, route_counts


def read_jsonl(path: Path, expected_catalog: dict[str, dict[str, Any]] | None = None) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], Counter[str]]:
    rows: list[dict[str, Any]] = []
    profile_catalog = expected_catalog
    counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                raise ValidationError(f"{path}:{line_number}: blank JSONL lines are not allowed")
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if profile_catalog is None:
                profile_catalog = catalog_map(row.get("tools"), f"{path}:{line_number}")
            _, row_counts = validate_row(row, path, line_number, profile_catalog)
            counts.update(row_counts)
            rows.append(row)
    if profile_catalog is None:
        raise ValidationError(f"{path}: file is empty")
    return rows, profile_catalog, counts


def validate_collection(rows: list[dict[str, Any]], label: str) -> tuple[Counter[str], Counter[str]]:
    split_counts = Counter(row["metadata"] for row in rows)
    fingerprints: dict[str, str] = {}
    duplicate_count = 0
    for index, row in enumerate(rows, 1):
        fingerprint = row_fingerprint(row)
        if fingerprint in fingerprints:
            duplicate_count += 1
        else:
            fingerprints[fingerprint] = f"{label}:{index}"
    if duplicate_count:
        raise ValidationError(f"{label}: {duplicate_count} exact duplicate rows found")
    return split_counts, Counter()


def validate_merged_output(path: Path, source_catalog: dict[str, dict[str, Any]], extension_catalog: dict[str, dict[str, Any]], expected_rows: int) -> Counter[str]:
    rows = 0
    split_counts: Counter[str] = Counter()
    profiles = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                raise ValidationError(f"{path}:{line_number}: blank JSONL line was written")
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"{path}:{line_number}: merged artifact is invalid JSON: {exc.msg}") from exc
            actual = catalog_map(row.get("tools"), f"{path}:{line_number}")
            if catalog_signature(actual) == catalog_signature(source_catalog):
                expected = source_catalog
                profiles["source"] += 1
            elif catalog_signature(actual) == catalog_signature(extension_catalog):
                expected = extension_catalog
                profiles["extension"] += 1
            else:
                raise ValidationError(f"{path}:{line_number}: merged artifact has an unknown catalog profile")
            validate_row(row, path, line_number, expected)
            rows += 1
            split_counts[row["metadata"]] += 1
    if rows != expected_rows or profiles != Counter({"source": 9654, "extension": 840}):
        raise ValidationError(f"merged artifact profile/count mismatch: rows={rows}, profiles={dict(profiles)}")
    if split_counts != Counter({"train": 9449, "eval": 1045}):
        raise ValidationError(f"merged artifact split mismatch: {dict(split_counts)}")
    return split_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_DEFAULT)
    parser.add_argument("--extension-dir", type=Path, default=EXTENSION_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--no-write", action="store_true", help="Validate without writing the merged JSONL")
    parser.add_argument(
        "--upload-dir",
        type=Path,
        default=None,
        help="Also write an upload-ready folder (iris.jsonl + train.jsonl + eval.jsonl).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        source_rows, source_catalog, source_routes = read_jsonl(args.source)
        if set(source_catalog) != BASE_TOOLS:
            raise ValidationError(
                f"source catalog mismatch: expected {sorted(BASE_TOOLS)}, got {sorted(source_catalog)}"
            )
        source_splits, _ = validate_collection(source_rows, "source")
        if len(source_rows) != 9654 or source_splits != Counter({"train": 8693, "eval": 961}):
            raise ValidationError(
                f"source counts changed: rows={len(source_rows)}, splits={dict(source_splits)}; "
                "expected 9654 rows with 8693 train / 961 eval"
            )

        extension_rows: list[dict[str, Any]] = []
        extension_catalog: dict[str, dict[str, Any]] | None = None
        extension_splits = Counter()
        extension_routes = Counter()
        for filename, expected_count in EXPECTED_EXTENSION_FILES.items():
            path = args.extension_dir / filename
            if not path.is_file():
                raise ValidationError(f"missing extension shard: {path}")
            rows, shard_catalog, shard_routes = read_jsonl(path, extension_catalog)
            if extension_catalog is None:
                extension_catalog = shard_catalog
            if len(rows) != expected_count:
                raise ValidationError(f"{path}: expected {expected_count} rows, found {len(rows)}")
            split_counts, _ = validate_collection(rows, filename)
            expected_split = Counter({"train": round(expected_count * 0.9), "eval": expected_count - round(expected_count * 0.9)})
            if split_counts != expected_split:
                raise ValidationError(f"{path}: expected splits {dict(expected_split)}, found {dict(split_counts)}")
            extension_rows.extend(rows)
            extension_splits.update(split_counts)
            extension_routes.update(shard_routes)

        if extension_catalog is None or set(extension_catalog) != EXPECTED_EXTENSION_TOOLS:
            actual = sorted(extension_catalog or {})
            raise ValidationError(f"extension catalog mismatch: expected 32 tools, got {actual}")
        extension_all_splits, _ = validate_collection(extension_rows, "extension")
        if len(extension_rows) != 840 or extension_all_splits != Counter({"train": 756, "eval": 84}):
            raise ValidationError(f"extension totals changed: rows={len(extension_rows)}, splits={dict(extension_all_splits)}")

        all_rows = source_rows + extension_rows
        global_fingerprints: set[str] = set()
        for row in all_rows:
            fingerprint = row_fingerprint(row)
            if fingerprint in global_fingerprints:
                raise ValidationError("exact duplicate detected across source and extension rows")
            global_fingerprints.add(fingerprint)

        all_routes = source_routes + extension_routes
        missing_routes = sorted(set(extension_catalog) - set(all_routes))
        if missing_routes:
            raise ValidationError(f"routes never selected anywhere in the merged dataset: {missing_routes}")
        leakage = defaultdict(set)
        for row in all_rows:
            leakage[row_fingerprint(row)].add(row["metadata"])
        split_leakage = sum(1 for splits in leakage.values() if len(splits) > 1)
        if split_leakage:
            raise ValidationError(f"{split_leakage} fingerprints appear in both train and eval")

        if not args.no_write:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", encoding="utf-8", newline="\n") as handle:
                for row in all_rows:
                    handle.write(compact_json(row) + "\n")
            validate_merged_output(args.output, source_catalog, extension_catalog, len(all_rows))

        upload_report = ""
        if args.upload_dir is not None:
            args.upload_dir.mkdir(parents=True, exist_ok=True)
            train_rows = [row for row in all_rows if row["metadata"] == "train"]
            eval_rows = [row for row in all_rows if row["metadata"] == "eval"]
            targets = {
                "iris.jsonl": all_rows,
                "train.jsonl": train_rows,
                "eval.jsonl": eval_rows,
            }
            for filename, subset in targets.items():
                with (args.upload_dir / filename).open("w", encoding="utf-8", newline="\n") as handle:
                    for row in subset:
                        handle.write(compact_json(row) + "\n")
            attribution = (
                "# Iris dataset (upload bundle)\n\n"
                "Native mobile-actions-format function-calling data for Iris.\n\n"
                "- `iris.jsonl` - full merged dataset (all rows).\n"
                "- `train.jsonl` - rows with metadata=train.\n"
                "- `eval.jsonl` - rows with metadata=eval.\n\n"
                f"Rows: {len(all_rows)} total, {len(train_rows)} train, {len(eval_rows)} eval.\n"
                f"Composition: {len(source_rows)} retained google/mobile-actions rows + "
                f"{len(extension_rows)} Iris-authored extension rows.\n\n"
                "## Attribution and license\n\n"
                "The retained rows come from `google/mobile-actions`, reported as "
                "CC-BY-4.0. Attribution to google/mobile-actions is required when "
                "redistributing this data or a model trained on it. Extension rows "
                "are Iris-authored; verify any teacher-model terms before "
                "redistribution.\n"
            )
            (args.upload_dir / "README.md").write_text(attribution, encoding="utf-8")
            upload_report = (
                f"\n  upload:    {args.upload_dir} "
                f"(iris.jsonl={len(all_rows)}, train.jsonl={len(train_rows)}, eval.jsonl={len(eval_rows)}, README.md)"
            )

        print("OK: native Iris dataset validation passed")
        print(f"  source:    {len(source_rows)} rows ({source_splits['train']} train / {source_splits['eval']} eval), catalog={len(source_catalog)}")
        print(f"  extension: {len(extension_rows)} rows ({extension_splits['train']} train / {extension_splits['eval']} eval), catalog={len(extension_catalog)}")
        print(f"  merged:    {len(all_rows)} rows ({source_splits['train'] + extension_splits['train']} train / {source_splits['eval'] + extension_splits['eval']} eval)")
        print(f"  catalogs:  source={','.join(sorted(source_catalog))}; extension={len(extension_catalog)} canonical tools")
        print(f"  route calls: {len(all_routes)} distinct tools; all catalog tools covered")
        print(f"  output:    {'not written (--no-write)' if args.no_write else args.output}{upload_report}")
        return 0
    except (OSError, ValidationError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
