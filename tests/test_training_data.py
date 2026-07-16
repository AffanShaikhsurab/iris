from __future__ import annotations

import json
from pathlib import Path

import pytest

from iris_training.data import DataError, normalize_row, render_training_example, to_model_boundary

TEMPLATE = Path("src/iris_training/templates/minicpm5_training.jinja").read_text(encoding="utf-8")


def tool(name="show_map"):
    return {"function": {"name": name, "description": "show it", "parameters": {
        "type": "OBJECT", "properties": {"query": {"type": "STRING"}}, "required": ["query"]
    }}}


def native_row():
    return {
        "metadata": "train", "tools": [tool()],
        "messages": [
            {"role": "developer", "content": "policy"},
            {"role": "user", "content": "map two places"},
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "show_map", "arguments": {"query": "A"}}},
                {"function": {"name": "show_map", "arguments": {"query": "B"}}},
            ]},
            {"role": "tool", "content": "A found"},
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "show_map", "arguments": {"query": "C"}}}
            ]},
        ],
    }


class FakeTokenizer:
    def __init__(self):
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.kwargs = kwargs
        targets = []
        for message in messages:
            for call in message.get("tool_calls", []):
                fn = call["function"]
                params = "".join(f'<param name="{k}">{v}</param>' for k, v in fn["arguments"].items())
                targets.append(f'<function name="{fn["name"]}">{params}</function>')
        prompt, target = "PROMPT|", "\n".join(targets)
        text = prompt + target
        return {"input_ids": [ord(c) for c in text], "assistant_masks": [0] * len(prompt) + [1] * len(target)}

    def decode(self, values, skip_special_tokens=False):
        return "".join(chr(value) for value in values)


def test_native_and_fireworks_use_one_normalizer_and_boundary():
    native = native_row()
    fireworks = json.loads(json.dumps(native))
    fireworks["tools"] = [{"type": "function", "function": {
        **tool()["function"], "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]
    }}}]
    for message in fireworks["messages"]:
        for call in message.get("tool_calls", []):
            call["type"] = "function"
            call["function"]["arguments"] = json.dumps(call["function"]["arguments"])
        if message["role"] == "developer":
            message["role"] = "system"
        elif message["role"] == "tool":
            message["role"] = "user"
            message["content"] = "Tool result from show_map:\n" + message["content"]
    for row in (native, fireworks):
        normalized = normalize_row(row)
        assert normalized["messages"][2]["tool_calls"][0]["function"]["arguments"] == {"query": "A"}
        assert normalized["messages"][3] == {"role": "tool", "content": "A found"}
        boundary = to_model_boundary(normalized)
        assert boundary["messages"][0]["role"] == "system"
        assert boundary["tools"][0]["type"] == "function"
        assert boundary["tools"][0]["function"]["parameters"]["type"] == "object"


def test_renderer_produces_exact_assistant_only_labels_and_preserves_chains():
    tokenizer = FakeTokenizer()
    rendered = render_training_example(tokenizer, native_row(), max_length=4096, chat_template=TEMPLATE)
    assert rendered.parallel and rendered.observation_count == 1 and rendered.tool_call_count == 3
    assert all(label == -100 for label in rendered.labels[:7])
    assert all(label >= 0 for label in rendered.labels[7:])
    assert tokenizer.kwargs["enable_thinking"] is False
    assert tokenizer.kwargs["tools"][0]["function"]["parameters"]["type"] == "object"


def test_renderer_refuses_missing_mask_schema_errors_and_duplicate_calls():
    tokenizer = FakeTokenizer()
    tokenizer.apply_chat_template = lambda *args, **kwargs: {"input_ids": [1, 2]}
    with pytest.raises(DataError, match="missing assistant mask"):
        render_training_example(tokenizer, native_row(), max_length=4096, chat_template=TEMPLATE)
    tokenizer = FakeTokenizer()
    original = tokenizer.apply_chat_template

    def duplicate_calls(*args, **kwargs):
        rendered = original(*args, **kwargs)
        supervised = tokenizer.decode(rendered["input_ids"][7:]) * 2
        return {
            "input_ids": [ord(c) for c in "PROMPT|" + supervised],
            "assistant_masks": [0] * 7 + [1] * len(supervised),
        }

    tokenizer.apply_chat_template = duplicate_calls
    with pytest.raises(DataError, match="expected exactly"):
        render_training_example(tokenizer, native_row(), max_length=4096, chat_template=TEMPLATE)
    bad = native_row()
    bad["messages"][2]["tool_calls"][0]["function"]["arguments"] = {"extra": "x"}
    with pytest.raises(DataError, match="missing required keys"):
        normalize_row(bad)
