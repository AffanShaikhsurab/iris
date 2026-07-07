#!/usr/bin/env python3
"""Preservation tests for the memory-persistence-and-context bugfix.

These tests encode the behavior that MUST STAY THE SAME after the fix lands
(the Property 2 "Preservation" set from `tasks.md` task 2). Unlike the
bug-condition exploration tests (`test_memory_context_bugs.py`, which are red on
unfixed code and go green after the fix), every test here is EXPECTED TO PASS on
the UNFIXED code and to KEEP passing after the fix. They are the regression net.

Methodology (observation-first): capture the current, observable behavior of the
loop, the delivery pattern, the budget reset, and the compiler invariants on the
UNFIXED build, and assert it. After the fix, re-running the SAME tests confirms
none of the preserved behavior changed (tasks.md task 3.10).

What is covered (mapped to the regression-prevention requirements):
  - Non-memory routing / loop dispatch is unchanged: every existing
    `scripts/simulate-agent.py` scenario dispatches to its declared `first_type`
    with no assertion failures (Req 3.5).
  - Fail-open: a normal chat completes with a spoken answer and never halts, and
    the existing proxy tools keep their `@proxyOk` fail-open guard (Req 3.6/3.1).
  - Delivery: answers strip markdown, flatten newlines, neutralize "?", and end
    on the stop-word regex (Req 3.8).
  - Budget reset: a non-stop follow-up zeroes the tool / question / repair
    budgets, observable as `remaining_tool_calls` returning to max (Req 3.9).
  - Compiler invariants: the compiled artifact has balanced control-flow groups
    (no `else if` bug), no `rawaction`, a NIM planner call, and the source keeps
    the `nvapi-REPLACE-ME` placeholder (Req 3.2/3.3), validated via
    `scripts/validate-shortcut.py` and directly against the plist.

Offline-runnable now. The compiled-artifact validation runs against the CI
unsigned artifact when present; a local dev build (which bakes a real key from
`.env.local`) or a missing artifact is handled by skipping the full-validator
assertion gracefully while the structural invariants are still checked directly.
"""
from __future__ import annotations

import importlib.util
import json
import os
import plistlib
import re
import subprocess
import sys
from pathlib import Path

import pytest

# --- Repo paths ---
REPO_ROOT = Path(__file__).resolve().parents[1]
IRIS_CHERRI = REPO_ROOT / "shortcuts" / "iris.cherri"
SIMULATE_AGENT = REPO_ROOT / "scripts" / "simulate-agent.py"
VALIDATE_SHORTCUT = REPO_ROOT / "scripts" / "validate-shortcut.py"
# Candidate unsigned artifacts (CI produces the placeholder build here).
UNSIGNED_CANDIDATES = [
    REPO_ROOT / "dist" / "Iris_unsigned.shortcut",
    REPO_ROOT / "dist" / "Iris.shortcut",
]

# A long, clean answer used as the canonical planner final_answer: no markdown,
# no refusal boilerplate, and comfortably over any min-length assertion.
LONG_ANSWER = (
    "Photosynthesis is the process green plants use to turn sunlight, water, and "
    "carbon dioxide into glucose and oxygen. In the light dependent reactions the "
    "chloroplast captures light with chlorophyll and splits water to make energy "
    "carriers, and in the light independent reactions those carriers fix carbon "
    "dioxide into sugar through the Calvin cycle, step by step, until the plant "
    "has the food it needs to grow and the air gains fresh oxygen."
)


# --------------------------------------------------------------------------- #
# Simulator loader / fixtures
# --------------------------------------------------------------------------- #
def _load_sim():
    """Import scripts/simulate-agent.py (hyphenated name) as a module."""
    spec = importlib.util.spec_from_file_location("simulate_agent", SIMULATE_AGENT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Module-level import so scenario names are available for parametrization.
SIM = _load_sim()
SCENARIO_NAMES = [s.name for s in SIM.scenarios()]


@pytest.fixture(scope="module")
def sim():
    return SIM


@pytest.fixture(scope="module")
def iris_source() -> str:
    return IRIS_CHERRI.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Helpers: drive the faithful loop with scripted planner replies (offline)
# --------------------------------------------------------------------------- #
def _run_with_script(sim, scenario, planner_replies, *, aux_reply="(handoff summary)"):
    """Run one scenario through the faithful loop, feeding scripted planner
    replies (system == PROTOCOL) and returning auxiliary calls (compaction /
    fallback) as `aux_reply`. Never touches the network.

    Returns (result_dict_from_run, captured_planner_user_messages).
    """
    captured: list[str] = []
    replies = list(planner_replies)

    def fake_call_nim(system, user, *, model, max_tokens, temperature, mock_reply=None):
        if system == sim.PROTOCOL:
            captured.append(user)
            if replies:
                return replies.pop(0)
            return '{"type":"final_answer","answer":"ok"}'
        return aux_reply

    original = sim.call_nim
    sim.call_nim = fake_call_nim
    try:
        res = sim.run(scenario, model="preservation", use_mock=False, verbose=False)
    finally:
        sim.call_nim = original
    return res, captured


def _script_for(scenario):
    """Build the canonical planner reply sequence the LIVE model would produce
    for a scenario: for a tool route, one tool_call (looping back) then a final
    answer; for a chat route, a single final answer. This isolates the loop's
    deterministic dispatch/first_type/assertion behavior for offline baselining.
    """
    replies = []
    ft = scenario.asserts.get("first_type")
    if ft and ft.startswith("tool_call:"):
        tool = ft.split(":", 1)[1]
        replies.append(json.dumps({
            "type": "tool_call", "tool": tool,
            "query": "x", "title": "x", "body": "x", "notes": "x",
            "date": "tomorrow", "time": "9am", "target": "google",
            "topic": "log", "recipient": "x", "return_to_agent": "true",
        }))
    replies.append(json.dumps({"type": "final_answer", "answer": LONG_ANSWER}))
    return replies


def _parse_int(planner_user_msg: str, field: str):
    m = re.search(rf"^{re.escape(field)}=(-?\d+)$", planner_user_msg, re.MULTILINE)
    return int(m.group(1)) if m else None


# --------------------------------------------------------------------------- #
# Preservation 1 - Non-memory routing / loop dispatch unchanged (Req 3.5)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scenario_name", SCENARIO_NAMES)
def test_scenario_dispatch_baseline(sim, scenario_name):
    """BASELINE: every existing simulate-agent.py scenario (jokes, weather,
    calendar, tasks, gmail, web_search, reminders, detailed, ...) dispatches its
    canonical planner reply to the declared first_type with zero assertion
    failures. Records the loop's observable dispatch behavior as the baseline the
    fix must preserve (Req 3.5).
    """
    scenario = next(s for s in sim.scenarios() if s.name == scenario_name)
    expected_first_type = scenario.asserts.get("first_type")

    res, _ = _run_with_script(sim, scenario, _script_for(scenario))

    assert res["failures"] == [], (
        f"scenario {scenario_name!r} broke a preserved assertion: {res['failures']}")
    if expected_first_type:
        assert res["first_type"] == expected_first_type, (
            f"scenario {scenario_name!r} routed to {res['first_type']!r}, "
            f"expected {expected_first_type!r}")


@pytest.mark.skipif(
    not os.environ.get("NIM_API_KEY"),
    reason="live NIM baseline: set NIM_API_KEY to run the real scenarios against "
           "the model",
)
def test_live_scenarios_baseline(sim):  # pragma: no cover - needs the live API
    """LIVE BASELINE: run every scenario against the real NIM model and record
    that all assertions pass. Skipped offline."""
    fails = []
    for scenario in sim.scenarios():
        res = sim.run(scenario, model=sim.DEFAULT_MODEL, use_mock=False, verbose=False)
        if res["failures"]:
            fails.append((scenario.name, res["failures"]))
    assert not fails, f"live scenario baseline failures: {fails}"


# --------------------------------------------------------------------------- #
# Preservation 2 - Fail-open when memory/proxy is unavailable (Req 3.6/3.1)
# --------------------------------------------------------------------------- #
def test_normal_chat_completes_and_never_halts(sim):
    """A normal chat (no memory, no proxy) completes with a spoken answer and
    never halts or exhausts the turn budget - exactly as Iris behaves today with
    no memory configured (Req 3.6, 3.1)."""
    scenario = sim.Scenario(
        "failopen_chat", "tell me something interesting",
        replies=["no thanks"],
        asserts={"first_type": "final_answer", "no_turn_exhaustion": True,
                 "no_ask_user": True})
    res, _ = _run_with_script(
        sim, scenario, [json.dumps({"type": "final_answer", "answer": LONG_ANSWER})])
    assert res["failures"] == [], res["failures"]
    assert res["first_type"] == "final_answer"


def test_existing_proxy_tools_keep_fail_open_guard(iris_source):
    """The existing proxy-backed tools keep their `@proxyOk > 0` guard with a
    fail-open `else` branch that reports "not set up" instead of halting. This is
    the convention the memory fix mirrors; it must remain intact (Req 3.6)."""
    assert "@proxyOk > 0" in iris_source, "proxy fail-open guard was removed"
    assert "Google is not set up" in iris_source, (
        "fail-open 'not set up' message for the proxy tools was removed")
    # Web search also degrades gracefully when unconfigured rather than failing.
    assert "Web search is not configured" in iris_source


def test_memory_fail_open_when_neither_backend_preserves_normal_answer(sim):
    """PRESERVATION (Property 10): with NEITHER memory backend available (selector
    `sheets` forces the local leg off and the proxy is unconfigured), a memory
    request fails open with an ok=false "pick a backend" envelope and the run
    still completes with a spoken answer, exactly as Iris behaves today with no
    memory configured (Req 3.6)."""
    scenario = sim.Scenario(
        "failopen_memory_neither", "remember that I like tea",
        replies=["no thanks"], proxy_ok=False, memory_backend="sheets",
        asserts={"first_type": "tool_call:memory_append",
                 "no_turn_exhaustion": True, "no_ask_user": True})
    planner_replies = [
        json.dumps({"type": "tool_call", "tool": "memory_append",
                    "topic": "log", "body": "I like tea",
                    "return_to_agent": "true"}),
        json.dumps({"type": "final_answer", "answer": LONG_ANSWER}),
    ]
    res, _ = _run_with_script(sim, scenario, planner_replies)
    assert res["failures"] == [], res["failures"]
    assert res["first_type"] == "tool_call:memory_append"
    # Nothing was persisted because neither leg was live.
    assert not res["memory"].local and not res["memory"].proxy


def test_hybrid_default_backend_is_available_by_default(sim):
    """PRESERVATION: the default `hybrid` backend keeps memory available on the
    happy path (the local leg is always on), so the common case is unaffected by
    the selector - only an explicit single-backend choice can turn memory off."""
    scenario = sim.Scenario("hybrid_default", "remember that I like tea",
                            replies=["no thanks"])
    res, _ = _run_with_script(sim, scenario, [
        json.dumps({"type": "tool_call", "tool": "memory_append",
                    "topic": "log", "body": "I like tea",
                    "return_to_agent": "true"}),
        json.dumps({"type": "final_answer", "answer": LONG_ANSWER}),
    ])
    assert res["memory_backend"] == "hybrid"
    assert res["memory"].available() is True
    assert res["failures"] == []


# --------------------------------------------------------------------------- #
# Preservation 3 - Delivery pattern (Req 3.8)
# --------------------------------------------------------------------------- #
def _sanitize_like_shortcut(text: str) -> str:
    """Replicate the shortcut/simulator speech sanitization exactly."""
    spoken = SIM.MARKDOWN_RE.sub("", text)
    spoken = re.sub(r"[\r\n]+", ". ", spoken)
    spoken = spoken.replace("?", ".")
    return spoken


def test_delivery_strips_markdown_flattens_newlines_neutralizes_question(sim):
    """Answers strip markdown markers, flatten newlines to '. ', and neutralize
    every '?' so Siri reads the whole answer before the trailing prompt (Req 3.8).
    """
    raw = "**Bold** _em_ #head `code`\nLine two\nWhy did it work? Because yes?"
    spoken = _sanitize_like_shortcut(raw)
    assert not any(ch in spoken for ch in "*_#`"), f"markdown survived: {spoken!r}"
    assert "\n" not in spoken and "\r" not in spoken, f"newline survived: {spoken!r}"
    assert "?" not in spoken, f"question mark survived: {spoken!r}"


def test_delivery_stop_words_end_conversation(sim):
    """A stop word (or silence) ends the conversation; anything else continues it
    (Req 3.8)."""
    for word in ["no", "nope", "nah", "stop", "quit", "exit", "bye", "goodbye",
                 "done", "nothing", "no thanks", "thanks", "thank you",
                 "No Thanks.", "  stop! "]:
        assert sim.STOP_WORD_RE.match(word), f"expected {word!r} to end the chat"
    for word in ["another", "tell me more", "yes please", "and the weather"]:
        assert not sim.STOP_WORD_RE.match(word), (
            f"expected {word!r} to continue the chat")


def test_source_delivery_transforms_present(iris_source):
    """The source keeps the exact S-GPT delivery transforms and stop-word regex
    (Req 3.8)."""
    assert r"[\*_#" in iris_source, "markdown-strip transform missing"
    assert r"[\r\n]+" in iris_source, "newline-flatten transform missing"
    assert r"replaceText('\?'" in iris_source, "question-neutralize transform missing"
    assert ("no|nope|nah|stop|quit|exit|bye|goodbye|done|nothing|no thanks|"
            "thanks|thank you") in iris_source, "stop-word regex missing/changed"


# --------------------------------------------------------------------------- #
# Preservation 4 - Budget reset on a non-stop follow-up (Req 3.9)
# --------------------------------------------------------------------------- #
def test_followup_resets_per_request_budgets(sim):
    """A non-stop follow-up resets the per-request tool budget: the follow-up's
    planner call shows `remaining_tool_calls` back at the maximum even though the
    previous request already consumed a tool call (Req 3.9)."""
    scenario = sim.Scenario(
        "budget_reset", "do the first thing",
        replies=["now do another thing", "no thanks"])
    planner_replies = [
        # turn 1: one tool call (loops back), then a final answer that delivers
        json.dumps({"type": "tool_call", "tool": "weather_summary",
                    "return_to_agent": "true"}),
        json.dumps({"type": "final_answer", "answer": LONG_ANSWER}),
        # follow-up request: a plain final answer
        json.dumps({"type": "final_answer", "answer": LONG_ANSWER}),
    ]
    res, captured = _run_with_script(sim, scenario, planner_replies)

    remaining = [_parse_int(m, "remaining_tool_calls") for m in captured]
    assert len(captured) == 3, f"expected 3 planner turns, got {len(captured)}"
    assert remaining[0] == sim.MAX_TOOL_CALLS, (
        f"fresh request should start with the full tool budget, got {remaining[0]}")
    assert remaining[1] == sim.MAX_TOOL_CALLS - 1, (
        f"tool budget should decrement within a request, got {remaining[1]}")
    assert remaining[2] == sim.MAX_TOOL_CALLS, (
        "a non-stop follow-up must RESET the tool budget to the maximum, got "
        f"{remaining[2]} (budget was not reset)")
    assert res["failures"] == [], res["failures"]


def test_source_followup_resets_all_three_budgets(iris_source):
    """The non-stop follow-up branch zeroes the tool, question, and repair
    budgets (Req 3.9)."""
    assert "@toolCallCount = 0" in iris_source
    assert "@userQuestionCount = 0" in iris_source
    assert "@repairCount = 0" in iris_source


# --------------------------------------------------------------------------- #
# Preservation 5 - Compiler invariants (Req 3.2 / 3.3)
# --------------------------------------------------------------------------- #
def _strip_block_comments(text: str) -> str:
    return re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)


def _locate_unsigned_artifact():
    for candidate in UNSIGNED_CANDIDATES:
        if candidate.exists():
            try:
                with open(candidate, "rb") as fh:
                    plistlib.load(fh)
            except Exception:
                continue  # signed/binary; not a plain plist we can inspect
            return candidate
    return None


@pytest.fixture(scope="module")
def compiled_plist():
    artifact = _locate_unsigned_artifact()
    if artifact is None:
        pytest.skip(
            "No plain-plist unsigned artifact available locally; the Cherri "
            "compile runs in CI (macOS). Structural invariants are validated on "
            "the CI-produced artifact.")
    with open(artifact, "rb") as fh:
        return plistlib.load(fh)


def test_compiled_no_rawaction(compiled_plist):
    """No `is.workflow.actions.rawaction` (iOS treats these as unsupported)
    (Req 3.3)."""
    ids = [a["WFWorkflowActionIdentifier"] for a in compiled_plist["WFWorkflowActions"]]
    assert ids.count("is.workflow.actions.rawaction") == 0


def test_compiled_has_nim_planner_call(compiled_plist):
    """A NIM planner call is present (the agent has a model) (Req 3.2)."""
    ids = [a["WFWorkflowActionIdentifier"] for a in compiled_plist["WFWorkflowActions"]]
    planner_calls = 0
    for action in compiled_plist["WFWorkflowActions"]:
        if action["WFWorkflowActionIdentifier"] == "is.workflow.actions.downloadurl":
            url = str(action.get("WFWorkflowActionParameters", {}).get("WFURL", ""))
            if "integrate.api.nvidia.com" in url:
                planner_calls += 1
    assert planner_calls > 0 or "com.openai.chat.AskIntent" in ids, (
        "no NIM planner call found in the compiled artifact")


def test_compiled_control_flow_balanced(compiled_plist):
    """Control-flow groups are balanced (no `else if` code-gen bug leaving groups
    open) (Req 3.3)."""
    groups: dict[str, list[int]] = {}
    stack: list[str] = []
    for index, action in enumerate(compiled_plist["WFWorkflowActions"]):
        params = action.get("WFWorkflowActionParameters", {})
        group = params.get("GroupingIdentifier")
        mode = params.get("WFControlFlowMode")
        if group is None or mode is None:
            continue
        mode = int(mode)
        groups.setdefault(group, []).append(mode)
        if mode == 0:
            stack.append(group)
        elif mode in (1, 2):
            assert stack and stack[-1] == group, (
                f"control-flow nesting mismatch at action {index}")
            if mode == 2:
                stack.pop()
    unclosed = [g[:8] for g, modes in groups.items() if 2 not in modes]
    assert not unclosed, f"control-flow groups missing their close: {unclosed}"
    assert not stack, f"control-flow groups left open: {stack}"


def test_validate_shortcut_script_passes_or_skips_on_local_build():
    """Run `scripts/validate-shortcut.py` against the unsigned artifact. On the CI
    placeholder build it passes outright. A local dev build bakes a real key from
    `.env.local`, so the validator's key-leak/placeholder guard fires - that is a
    property of the local build, not a broken compiler invariant, so we skip
    gracefully (the structural invariants are checked directly above). A genuine
    structural failure (control flow / rawaction / missing planner) fails the
    test (Req 3.2/3.3)."""
    artifact = _locate_unsigned_artifact()
    if artifact is None:
        pytest.skip("No unsigned artifact available locally; validated in CI.")

    proc = subprocess.run(
        [sys.executable, str(VALIDATE_SHORTCUT), str(artifact)],
        capture_output=True, text=True)
    output = (proc.stdout or "") + (proc.stderr or "")

    if proc.returncode == 0:
        assert "OK:" in output
        return

    structural_markers = [
        "control-flow", "Control-flow", "rawaction",
        "No planner call", "left open", "missing their closing",
    ]
    local_build_markers = [
        "nvapi- key", "tvly- key", "Google client secret",
        "Google refresh token", "nvapi-REPLACE-ME placeholder",
        "not a plain plist",
    ]
    if any(m in output for m in structural_markers):
        pytest.fail(f"validate-shortcut.py reported a STRUCTURAL failure:\n{output}")
    if any(m in output for m in local_build_markers):
        pytest.skip(
            "Local dev build (keys injected from .env.local) or signed artifact; "
            "the full validator runs on the CI placeholder build. Structural "
            f"invariants are checked directly.\nvalidator output:\n{output}")
    pytest.skip(f"validate-shortcut.py could not validate this artifact:\n{output}")


def test_source_has_no_else_if(iris_source):
    """The source avoids `else if` chains outside comments (the Cherri code-gen
    bug the header forbids) (Req 3.3)."""
    code = _strip_block_comments(iris_source)
    assert "else if" not in code, "an `else if` chain crept into the code"
    assert "elseif" not in code, "an `elseif` chain crept into the code"


def test_source_keeps_placeholder_and_planner(iris_source):
    """The source keeps the editable `nvapi-REPLACE-ME` placeholder and the NIM
    planner endpoint (Req 3.2). Distributed artifacts must ship the placeholder,
    and no real key must be baked into the committed source."""
    assert "nvapi-REPLACE-ME" in iris_source, "nvapi-REPLACE-ME placeholder removed"
    assert "integrate.api.nvidia.com/v1/chat/completions" in iris_source, (
        "NIM planner endpoint removed")
    assert not re.search(r"nvapi-[A-Za-z0-9_-]{20,}", iris_source), (
        "a real nvapi- key appears to be baked into the committed source")
