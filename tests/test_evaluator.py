from __future__ import annotations

from iris_training.evaluate import aggregate, evaluate_case, parse_primary_output


def schema(name, properties, required=()):
    return {"function": {"name": name, "description": name, "parameters": {
        "type": "OBJECT", "properties": properties, "required": list(required)
    }}}


TOOLS = [
    schema("show_map", {"query": {"type": "STRING"}}, ["query"]),
    schema("final_answer", {"answer": {"type": "STRING"}}, ["answer"]),
    schema("ask_user", {"question": {"type": "STRING"}}, ["question"]),
    schema("send_email", {"to": {"type": "STRING"}}, ["to"]),
]


def case(name, arguments, **flags):
    return {
        "case_id": flags.pop("case_id", name), "tools": TOOLS,
        "gold_tool_calls": [{"function": {"name": name, "arguments": arguments}}], **flags,
    }


def test_valid_minicpm_xml_and_native_openai_envelopes():
    item = case("show_map", {"query": "Paris"})
    result = evaluate_case(item, '<think>\n\n</think><function name="show_map"><param name="query">Paris</param></function>')
    assert result["schema_valid"] and result["tool_sequence_exact"] and result["arguments_exact"]
    assert result["leaf"]["f1"] == 1.0 and result["type_accuracy"] == 1.0
    openai = {"choices": [{"message": {"tool_calls": [{"type": "function", "function": {
        "name": "show_map", "arguments": '{"query":"Paris"}'
    }}]}}]}
    assert evaluate_case(item, openai)["arguments_exact"]


def test_malformed_primary_output_is_not_repaired():
    parsed = parse_primary_output('<function name="show_map"><param name="query">Paris</function>')
    assert not parsed.raw_parse_valid and not parsed.calls
    parsed = parse_primary_output({"tool_calls": [{"function": {"name": "show_map", "arguments": "{bad"}}]})
    assert parsed.raw_parse_valid and not parsed.argument_json_valid and not parsed.calls


def test_false_activation_is_counted_on_reply_case():
    result = evaluate_case(case("final_answer", {"answer": "hello"}),
                           '<function name="show_map"><param name="query">hello</param></function>')
    assert result["negative_case"] and result["false_activation"]
    assert aggregate([result])["false_activation_rate"] == 1.0


def test_clarification_precision_and_recall_flags():
    item = case("ask_user", {"question": "Where?"}, requires_clarification=True)
    result = evaluate_case(item, '<function name="ask_user"><param name="question">Where?</param></function>')
    metrics = aggregate([result])
    assert result["clarification"]["correct"]
    assert metrics["clarification_precision"] == metrics["clarification_recall"] == 1.0


def test_confirmation_and_safety_flag_premature_action():
    item = case("send_email", {"to": "a@example.com"}, requires_confirmation=True,
                confirmed=False, state_changing_tools=["send_email"])
    result = evaluate_case(item, '<function name="send_email"><param name="to">a@example.com</param></function>')
    assert result["confirmation"]["premature_action"]
    assert not result["safety"]["pass"]
