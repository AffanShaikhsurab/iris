#!/usr/bin/env python3
"""Convert the canonical Iris JSONL into Fireworks SFT function-call JSONL.

The source remains unchanged. Every record and tool is retained; native
mobile-actions fields are normalized to Fireworks' OpenAI-compatible format.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DEFAULT = ROOT / "data" / "iris-dataset" / "iris.jsonl"
OUTPUT_DEFAULT = ROOT / "data" / "iris-dataset" / "fireworks"
JSON_TYPES = {"array", "boolean", "integer", "null", "number", "object", "string"}


class ConversionError(ValueError):
    pass


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def normalize_schema(value: Any) -> Any:
    if isinstance(value, dict):
        result = {key: normalize_schema(item) for key, item in value.items()}
        if isinstance(result.get("type"), str):
            result["type"] = result["type"].lower()
        return result
    if isinstance(value, list):
        return [normalize_schema(item) for item in value]
    return value


def convert_tools(tools: Any, location: str) -> list[dict[str, Any]]:
    if not isinstance(tools, list) or not tools:
        raise ConversionError(f"{location}: tools must be a non-empty list")
    converted = []
    for index, wrapper in enumerate(tools):
        if not isinstance(wrapper, dict) or set(wrapper) != {"function"}:
            raise ConversionError(f"{location}: tools[{index}] must contain only function")
        function = normalize_schema(wrapper["function"])
        converted.append({"type": "function", "function": function})
    return converted
def convert_tool_calls(calls: Any, location: str) -> list[dict[str, Any]]:
    if not isinstance(calls, list) or not calls:
        raise ConversionError(f"{location}: assistant tool_calls must be non-empty")
    converted = []
    for index, wrapper in enumerate(calls):
        if not isinstance(wrapper, dict) or set(wrapper) != {"function"}:
            raise ConversionError(f"{location}: tool_calls[{index}] must contain only function")
        function = wrapper["function"]
        if not isinstance(function, dict) or set(function) != {"name", "arguments"}:
            raise ConversionError(f"{location}: tool_calls[{index}].function is invalid")
        if not isinstance(function["arguments"], dict):
            raise ConversionError(f"{location}: tool-call arguments must be an object")
        converted.append({
            "type": "function",
            "function": {
                "name": function["name"],
                "arguments": compact_json(function["arguments"]),
            },
        })
    return converted


def convert_messages(messages: Any, location: str) -> list[dict[str, Any]]:
    if not isinstance(messages, list) or len(messages) < 3:
        raise ConversionError(f"{location}: messages must contain at least three entries")
    converted: list[dict[str, Any]] = []
    previous_tool_names: list[str] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or "role" not in message:
            raise ConversionError(f"{location}: messages[{index}] is invalid")
        role = message["role"]
        if role == "developer":
            converted.append({"role": "system", "content": message["content"]})
        elif role in {"system", "user"}:
            converted.append({"role": role, "content": message["content"]})
        elif role == "assistant":
            calls = convert_tool_calls(message.get("tool_calls"), f"{location}:messages[{index}]")
            converted.append({"role": "assistant", "content": "", "tool_calls": calls})
            previous_tool_names = [call["function"]["name"] for call in calls]
        elif role == "tool":
            if len(previous_tool_names) != 1:
                raise ConversionError(f"{location}: tool result must follow exactly one tool call")
            content = message.get("content")
            if not isinstance(content, str):
                raise ConversionError(f"{location}: tool result content must be a string")
            converted.append({
                "role": "user",
                "content": f"Tool result from {previous_tool_names[0]}:\n{content}",
            })
            previous_tool_names = []
        else:
            raise ConversionError(f"{location}: unsupported role {role!r}")
    return converted
def convert_row(row: Any, location: str) -> tuple[str, dict[str, Any]]:
    if not isinstance(row, dict) or set(row) != {"metadata", "tools", "messages"}:
        raise ConversionError(f"{location}: expected metadata, tools, and messages")
    split = row["metadata"]
    if split not in {"train", "eval"}:
        raise ConversionError(f"{location}: metadata must be train or eval")
    converted = {
        "tools": convert_tools(row["tools"], location),
        "messages": convert_messages(row["messages"], location),
    }
    validate_converted_row(converted, location)
    return split, converted


def validate_converted_row(row: dict[str, Any], location: str) -> None:
    tool_names: set[str] = set()
    for index, wrapper in enumerate(row["tools"]):
        if wrapper.get("type") != "function" or set(wrapper) != {"type", "function"}:
            raise ConversionError(f"{location}: converted tool {index} has an invalid wrapper")
        function = wrapper["function"]
        name = function.get("name")
        if not isinstance(name, str) or not name or name in tool_names:
            raise ConversionError(f"{location}: converted tool name {name!r} is invalid")
        tool_names.add(name)
        schema = function.get("parameters")
        if not isinstance(schema, dict) or schema.get("type") != "object":
            raise ConversionError(f"{location}: {name} parameters must use type object")
        for schema_value in walk_schema_types(schema):
            if schema_value not in JSON_TYPES:
                raise ConversionError(f"{location}: unsupported JSON Schema type {schema_value!r}")

    roles = [message.get("role") for message in row["messages"]]
    if not roles or roles[0] != "system" or any(role not in {"system", "user", "assistant"} for role in roles):
        raise ConversionError(f"{location}: converted roles are invalid: {roles}")
    for message in row["messages"]:
        if message["role"] != "assistant":
            if set(message) != {"role", "content"} or not isinstance(message["content"], str):
                raise ConversionError(f"{location}: context message is invalid")
            continue
        if set(message) != {"role", "content", "tool_calls"} or message["content"] != "":
            raise ConversionError(f"{location}: assistant message must have empty content and tool_calls")
        for call in message["tool_calls"]:
            function = call["function"]
            if function["name"] not in tool_names:
                raise ConversionError(f"{location}: call selects an unavailable tool")
            arguments = json.loads(function["arguments"])
            if not isinstance(arguments, dict):
                raise ConversionError(f"{location}: serialized arguments are not an object")


def walk_schema_types(value: Any):
    if isinstance(value, dict):
        if "type" in value and isinstance(value["type"], str):
            yield value["type"]
        for item in value.values():
            yield from walk_schema_types(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk_schema_types(item)
def prepare(source: Path, output_dir: Path) -> Counter[str]:
    if not source.is_file():
        raise ConversionError(f"source dataset not found: {source}")
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = {split: output_dir / f".{split}.jsonl.tmp" for split in ("train", "eval")}
    final = {split: output_dir / f"{split}.jsonl" for split in ("train", "eval")}
    counts: Counter[str] = Counter()
    profile_counts: Counter[int] = Counter()
    role_counts: Counter[str] = Counter()
    handles = {split: path.open("w", encoding="utf-8", newline="\n") for split, path in temporary.items()}
    try:
        with source.open(encoding="utf-8") as source_file:
            for line_number, line in enumerate(source_file, 1):
                if not line.strip():
                    continue
                try:
                    native_row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ConversionError(f"{source}:{line_number}: invalid JSON: {error}") from error
                split, converted = convert_row(native_row, f"{source}:{line_number}")
                handles[split].write(compact_json(converted) + "\n")
                counts[split] += 1
                profile_counts[len(converted["tools"])] += 1
                role_counts.update(message["role"] for message in converted["messages"])
    finally:
        for handle in handles.values():
            handle.close()
    if not counts["train"] or not counts["eval"]:
        raise ConversionError(f"both splits must be non-empty; found {dict(counts)}")
    for split in ("train", "eval"):
        temporary[split].replace(final[split])
    counts["total"] = counts["train"] + counts["eval"]
    print(f"OK: wrote {counts['total']} Fireworks examples to {output_dir}")
    print(f"    splits: train={counts['train']}, eval={counts['eval']}")
    print(f"    tool catalog sizes: {dict(sorted(profile_counts.items()))}")
    print(f"    converted message roles: {dict(role_counts)}")
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        prepare(args.source, args.output_dir)
    except (ConversionError, OSError) as error:
        print(f"FAIL: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
