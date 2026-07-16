"""Canonical Iris normalization, model-boundary conversion, and mask rendering."""
from __future__ import annotations

import copy
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

JSON = dict[str, Any]
JSON_TYPES = {"ARRAY", "BOOLEAN", "INTEGER", "NULL", "NUMBER", "OBJECT", "STRING"}


class DataError(ValueError):
    """Raised when a row cannot be safely used for supervised training."""


@dataclass(frozen=True)
class RenderedExample:
    input_ids: list[int]
    labels: list[int]
    assistant_mask: list[int]
    supervised_text: str
    tool_call_count: int
    parallel: bool
    observation_count: int


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _schema_case(value: Any, upper: bool) -> Any:
    if isinstance(value, dict):
        result = {key: _schema_case(item, upper) for key, item in value.items()}
        kind = result.get("type")
        if isinstance(kind, str) and kind.upper() in JSON_TYPES:
            result["type"] = kind.upper() if upper else kind.lower()
        return result
    if isinstance(value, list):
        return [_schema_case(item, upper) for item in value]
    return value


def _function(wrapper: Any, location: str) -> JSON:
    if not isinstance(wrapper, dict) or not isinstance(wrapper.get("function"), dict):
        raise DataError(f"{location}: expected a function wrapper")
    return copy.deepcopy(wrapper["function"])
def _canonical_tools(raw_tools: Any, location: str) -> list[JSON]:
    if not isinstance(raw_tools, list) or not raw_tools:
        raise DataError(f"{location}: tools must be a non-empty array")
    tools: list[JSON] = []
    names: set[str] = set()
    for index, wrapper in enumerate(raw_tools):
        fn = _function(wrapper, f"{location}.tools[{index}]")
        name, parameters = fn.get("name"), fn.get("parameters")
        if not isinstance(name, str) or not name or name in names:
            raise DataError(f"{location}: invalid or duplicate tool name {name!r}")
        if not isinstance(parameters, dict):
            raise DataError(f"{location}: {name} has no parameter schema")
        fn["parameters"] = _schema_case(parameters, upper=True)
        if fn["parameters"].get("type") != "OBJECT":
            raise DataError(f"{location}: {name} parameters must be an object")
        tools.append({"function": fn})
        names.add(name)
    return tools


def _arguments(value: Any, location: str) -> JSON:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise DataError(f"{location}: arguments are not valid JSON") from exc
    if not isinstance(value, dict):
        raise DataError(f"{location}: arguments must be an object")
    return copy.deepcopy(value)


def _validate_value(value: Any, schema: Mapping[str, Any], location: str) -> None:
    kind = str(schema.get("type", "")).upper()
    checks = {
        "STRING": lambda x: isinstance(x, str),
        "BOOLEAN": lambda x: isinstance(x, bool),
        "INTEGER": lambda x: isinstance(x, int) and not isinstance(x, bool),
        "NUMBER": lambda x: isinstance(x, (int, float)) and not isinstance(x, bool),
        "NULL": lambda x: x is None,
        "ARRAY": lambda x: isinstance(x, list),
        "OBJECT": lambda x: isinstance(x, dict),
    }
    if kind not in checks or not checks[kind](value):
        raise DataError(f"{location}: expected JSON type {kind or 'declared'}")
    if "enum" in schema and value not in schema["enum"]:
        raise DataError(f"{location}: value is not in enum")
    if kind == "ARRAY" and isinstance(schema.get("items"), dict):
        for index, item in enumerate(value):
            _validate_value(item, schema["items"], f"{location}[{index}]")
    if kind == "OBJECT":
        _validate_arguments(value, schema, location)


def _validate_arguments(arguments: JSON, schema: Mapping[str, Any], location: str) -> None:
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    if not isinstance(properties, dict) or not isinstance(required, list):
        raise DataError(f"{location}: malformed object schema")
    missing = set(required) - set(arguments)
    extra = set(arguments) - set(properties)
    if missing:
        raise DataError(f"{location}: missing required keys {sorted(missing)}")
    if extra:
        raise DataError(f"{location}: extra keys {sorted(extra)}")
    for key, value in arguments.items():
        if not isinstance(properties[key], dict):
            raise DataError(f"{location}.{key}: malformed property schema")
        _validate_value(value, properties[key], f"{location}.{key}")
def normalize_row(raw: Any, location: str = "row") -> JSON:
    """Normalize native or Fireworks JSONL into one native canonical form."""
    if not isinstance(raw, dict) or not isinstance(raw.get("messages"), list):
        raise DataError(f"{location}: expected an object with messages")
    tools = _canonical_tools(raw.get("tools"), location)
    fireworks = any(isinstance(item, dict) and item.get("type") == "function" for item in raw.get("tools", []))
    catalog = {item["function"]["name"]: item["function"] for item in tools}
    messages: list[JSON] = []
    assistant_count = 0
    for index, original in enumerate(raw["messages"]):
        where = f"{location}.messages[{index}]"
        if not isinstance(original, dict):
            raise DataError(f"{where}: message must be an object")
        role = original.get("role")
        content = original.get("content", "")
        if (
            fireworks and role == "user" and isinstance(content, str)
            and content.startswith("Tool result from ") and ":\n" in content
            and messages and messages[-1]["role"] == "assistant"
        ):
            role, content = "tool", content.split(":\n", 1)[1]
        if role not in {"developer", "system", "user", "assistant", "tool"}:
            raise DataError(f"{where}: unsupported role {role!r}")
        message: JSON = {"role": role, "content": content}
        if not isinstance(message["content"], str):
            raise DataError(f"{where}: content must be a string")
        calls = original.get("tool_calls")
        if role == "assistant":
            assistant_count += 1
            if not isinstance(calls, list) or not calls:
                raise DataError(f"{where}: assistant must have non-empty tool_calls")
            normalized_calls: list[JSON] = []
            for call_index, wrapper in enumerate(calls):
                call_where = f"{where}.tool_calls[{call_index}]"
                fn = _function(wrapper, call_where)
                name = fn.get("name")
                if name not in catalog:
                    raise DataError(f"{call_where}: unknown tool {name!r}")
                arguments = _arguments(fn.get("arguments"), call_where)
                _validate_arguments(arguments, catalog[name]["parameters"], call_where)
                normalized_calls.append({"function": {"name": name, "arguments": arguments}})
            message["tool_calls"] = normalized_calls
        elif calls is not None:
            raise DataError(f"{where}: only assistant messages may contain tool_calls")
        messages.append(message)
    if not messages or messages[0]["role"] not in {"developer", "system"}:
        raise DataError(f"{location}: first message must be developer or system")
    if len(messages) < 3 or messages[1]["role"] != "user":
        raise DataError(f"{location}: second message must be user")
    for index, message in enumerate(messages[1:], 1):
        previous = messages[index - 1]["role"]
        if message["role"] in {"developer", "system"}:
            raise DataError(f"{location}: system/developer role is only valid first")
        if message["role"] == "tool" and previous not in {"assistant", "tool"}:
            raise DataError(f"{location}: tool observation must follow assistant/tool")
        if message["role"] == "assistant" and previous not in {"user", "tool"}:
            raise DataError(f"{location}: assistant must follow user/tool")
    if assistant_count == 0 or not any(message["role"] == "user" for message in messages):
        raise DataError(f"{location}: row needs user and assistant messages")
    return {"metadata": raw.get("metadata"), "tools": tools, "messages": messages}


def to_model_boundary(row: JSON) -> JSON:
    """Convert only at rendering time to HF/OpenAI lowercase schema wrappers."""
    messages: list[JSON] = []
    for original in row["messages"]:
        message = copy.deepcopy(original)
        if message["role"] == "developer":
            message["role"] = "system"
        for call in message.get("tool_calls", []):
            call["type"] = "function"
        messages.append(message)
    tools = [
        {"type": "function", "function": _schema_case(item["function"], upper=False)}
        for item in row["tools"]
    ]
    return {"messages": messages, "tools": tools}


def validate_training_template(template: str) -> None:
    if template.count("{% generation %}") + template.count("{%- generation %}") != 1:
        raise DataError("training template must contain exactly one generation block")
    if template.count("{% endgeneration %}") + template.count("{%- endgeneration %}") != 1:
        raise DataError("training template must contain exactly one endgeneration block")
    assistant = template.find('message.role == "assistant"')
    generation = max(template.find("{% generation %}"), template.find("{%- generation %}"))
    tool = template.find('message.role == "tool"', assistant)
    end = max(template.find("{% endgeneration %}"), template.find("{%- endgeneration %}"))
    if not (0 <= assistant < generation < end < tool):
        raise DataError("generation span is not confined to the assistant branch")
def _flat(values: Any, name: str) -> list[int]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, list) and len(values) == 1 and isinstance(values[0], list):
        values = values[0]
    if not isinstance(values, list) or not all(isinstance(value, (int, bool)) for value in values):
        raise DataError(f"renderer returned invalid {name}")
    return [int(value) for value in values]


def _target_markers(row: JSON) -> Iterable[str]:
    for message in row["messages"]:
        for call in message.get("tool_calls", []):
            fn = call["function"]
            yield f'<function name="{fn["name"]}">'
            for key, value in fn["arguments"].items():
                yield f'<param name="{key}">'
                if value not in ("", None):
                    yield str(value)


def render_training_example(
    tokenizer: Any,
    raw_row: Any,
    *,
    max_length: int,
    chat_template: str,
    location: str = "row",
) -> RenderedExample:
    """Render one row and fail closed if exact assistant-only labels are unavailable."""
    validate_training_template(chat_template)
    row = normalize_row(raw_row, location)
    boundary = to_model_boundary(row)
    rendered = tokenizer.apply_chat_template(
        boundary["messages"],
        tools=boundary["tools"],
        enable_thinking=False,
        chat_template=chat_template,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        add_generation_prompt=False,
    )
    if not isinstance(rendered, Mapping):
        raise DataError(f"{location}: tokenizer did not return a mapping")
    input_ids = _flat(rendered.get("input_ids"), "input_ids")
    raw_mask = rendered.get("assistant_masks", rendered.get("assistant_mask"))
    if raw_mask is None:
        raise DataError(f"{location}: missing assistant mask; refusing full-sequence loss")
    mask = _flat(raw_mask, "assistant mask")
    if len(input_ids) != len(mask) or not any(mask) or any(value not in {0, 1} for value in mask):
        raise DataError(f"{location}: empty or invalid assistant mask")
    if len(input_ids) > max_length:
        raise DataError(
            f"{location}: length {len(input_ids)} exceeds {max_length}; assistant truncation is forbidden"
        )
    supervised_ids = [token for token, keep in zip(input_ids, mask) if keep]
    supervised_text = tokenizer.decode(supervised_ids, skip_special_tokens=False)
    structural_markers = Counter(
        marker for marker in _target_markers(row) if marker.startswith("<function name=")
    )
    for marker, expected_count in structural_markers.items():
        actual_count = supervised_text.count(marker)
        if actual_count != expected_count:
            raise DataError(
                f"{location}: supervised tokens contain {actual_count} copies of "
                f"target marker {marker!r}; expected exactly {expected_count}"
            )
    markers = Counter(_target_markers(row))
    for marker, expected_count in markers.items():
        if supervised_text.count(marker) < expected_count:
            raise DataError(f"{location}: supervised tokens omit target marker {marker!r}")
    labels = [token if keep else -100 for token, keep in zip(input_ids, mask)]
    calls_per_turn = [len(message.get("tool_calls", [])) for message in row["messages"]]
    return RenderedExample(
        input_ids=input_ids,
        labels=labels,
        assistant_mask=mask,
        supervised_text=supervised_text,
        tool_call_count=sum(calls_per_turn),
        parallel=any(count > 1 for count in calls_per_turn),
        observation_count=sum(message["role"] == "tool" for message in row["messages"]),
    )


def read_jsonl(path: Path) -> Iterable[tuple[int, JSON]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise DataError(f"{path}:{line_number}: blank line")
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataError(f"{path}:{line_number}: invalid JSON") from exc
