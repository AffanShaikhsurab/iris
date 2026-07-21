#!/usr/bin/env python3
"""Bug-condition exploration tests for the memory-persistence-and-context bugfix.

These tests encode the EXPECTED POST-FIX behavior for the four related defects
described in `.kiro/specs/memory-persistence-and-context/bugfix.md`:

  1. Memory no-op    - built-in Files writes emit a bare WFFilePath with no
                       device `fileLocation`, so nothing persists / recalls.
  2. Lost follow-up  - the planner is pinned to the ORIGINAL `user_request`, so a
                       follow-up ("tell me more about that match") is not answered
                       as the current request.
  3. Count compaction- compaction fires on a fixed follow-up-exchange count and
                       the summary prompt does not protect named entities, so a
                       seeded entity ("Estadio Azteca") is dropped.
  4. Model contract  - the default planner is the weak `meta/llama-3.1-8b-instruct`
                       instruct model, not a capable non-reasoning model.

IMPORTANT (bugfix workflow): every test here is EXPECTED TO FAIL (or surface a
counterexample) on the current UNFIXED code. A failure confirms the bug exists.
Do NOT "fix" these tests or the code to make them green - they are supposed to
be red until the fix lands, at which point the same tests validate the fix
(tasks 3.9 / 4). Each failure message records the concrete counterexample.

Offline-runnable now: the plist assertion, the follow-up current-request
assertion, the compaction-trigger inspection, the named-entity survival check,
and the default-model inspection. Scenarios that need the live NIM model or a
deployed proxy are marked MANUAL / ON-DEVICE and skip when unconfigured.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
from pathlib import Path

import pytest

# --- Repo paths ---
REPO_ROOT = Path(__file__).resolve().parents[1]
IRIS_CHERRI = REPO_ROOT / "shortcuts" / "iris.cherri"
SIMULATE_AGENT = REPO_ROOT / "scripts" / "simulate-agent.py"
PHONE_PLIST_JSON = REPO_ROOT / "tmp" / "iris-newshortcut6.plist.json"

# The capable, non-reasoning instruct model the fix selects (design Decision 5).
# Updated 2026-07 to the researched primary actually pinned in iris.cherri: the
# earlier mistral-small-3.1-24b id was RETIRED from the NIM catalog (404), so the
# shortcut moved to mistral-small-4-119b-2603 (see the @nimModelIdRaw comment).
FIXED_PRIMARY_MODEL = "mistralai/mistral-small-4-119b-2603"
FIXED_FALLBACK_MODEL = "openai/gpt-oss-20b"
WEAK_DEFAULT_MODEL = "meta/llama-3.1-8b-instruct"


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def iris_source() -> str:
    return IRIS_CHERRI.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def sim():
    """Import scripts/simulate-agent.py (hyphenated name) as a module."""
    spec = importlib.util.spec_from_file_location("simulate_agent", SIMULATE_AGENT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _drive_planner(sim, scenario, planner_replies, *, aux_reply="(handoff summary)"):
    """Run one scenario through the faithful loop while capturing every user
    message sent to the PLANNER (system == PROTOCOL) and feeding scripted planner
    replies. Auxiliary calls (compaction / fallback) return `aux_reply`.

    Returns the list of captured planner user messages, in turn order.
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
        # use_mock=False so our monkeypatched call_nim is the single source of
        # replies; it never touches the network.
        sim.run(scenario, model="exploration", use_mock=False, verbose=False)
    finally:
        sim.call_nim = original
    return captured


def _parse_user_request(planner_user_msg: str) -> str:
    """Extract the value the shortcut/simulator presents as the current request."""
    m = re.search(r"^user_request=(.*)$", planner_user_msg, re.MULTILINE)
    return m.group(1) if m else ""


# --------------------------------------------------------------------------- #
# Bug 1 - Memory no-op
# --------------------------------------------------------------------------- #
def test_bug1_phone_plist_working_append_requires_a_filelocation_object(sim=None):
    """ROOT CAUSE REGRESSION GUARD (memory no-op): a WORKING phone-built
    file-append action carries a device-specific `fileLocation` object in
    addition to a relative `WFFilePath`. Cherri's built-in `appendToFile`
    emits only the bare `WFFilePath` string with NO `fileLocation`, so on
    device the write has no valid destination and silently no-ops.

    This is why the fix moves memory OFF phone-local Files entirely onto the
    proxy-backed store. This test asserts the decoded ground truth (the working
    action DOES require a `fileLocation`), documenting the root cause as a
    permanent, passing regression guard so nobody reintroduces a bare-path
    built-in Files write for memory.
    """
    assert PHONE_PLIST_JSON.exists(), f"missing decoded plist: {PHONE_PLIST_JSON}"
    plist = json.loads(PHONE_PLIST_JSON.read_text(encoding="utf-8"))
    actions = plist["WFWorkflowActions"]

    append = next(
        (a for a in actions
         if a["WFWorkflowActionIdentifier"] == "is.workflow.actions.file.append"),
        None,
    )
    assert append is not None, "no file.append action found in the phone plist"
    params = append["WFWorkflowActionParameters"]

    has_wffilepath = "WFFilePath" in params
    has_filelocation = "fileLocation" in json.dumps(params)

    # The working phone-built append carries a device-specific fileLocation.
    file_location = params.get("WFFile", {}).get("fileLocation")
    counterexample = json.dumps(file_location, indent=2) if file_location else "(none)"

    assert has_wffilepath and has_filelocation, (
        "ROOT CAUSE: the working phone-built `is.workflow.actions.file.append` "
        "must carry BOTH a relative WFFilePath and a device-specific `fileLocation` "
        f"object (WFFilePath present={has_wffilepath}, fileLocation present="
        f"{has_filelocation}). Cherri's built-in appendToFile emits only the bare "
        "WFFilePath with no fileLocation, so on-device writes have no valid "
        "destination and no-op - which is why memory moved to the proxy-backed "
        f"store.\nfileLocation on the working action:\n{counterexample}"
    )
    assert file_location is not None, (
        "expected the working phone-built append to carry a WFFile.fileLocation "
        "object; the decoded plist reference may have changed"
    )


def test_bug1_memory_write_then_read_recalls_body(sim):
    """A memory_append(topic, body) followed by a later memory_read(topic) MUST
    surface the written body back to the planner (Property 4).

    On unfixed code memory runs through built-in Files (no-op) and the simulator
    mocks the memory tools generically, so nothing persists: the read observation
    never contains the appended body. EXPECTED: FAIL on unfixed code.
    """
    body_token = "chamomile"  # distinctive token from the appended body
    scenario = sim.Scenario(
        "memory_roundtrip",
        "remember my drink preference and then recall it",
        replies=["no thanks"],
    )
    planner_replies = [
        '{"type":"tool_call","tool":"memory_append","topic":"preferences",'
        '"body":"I prefer chamomile tea","return_to_agent":"true"}',
        '{"type":"tool_call","tool":"memory_read","topic":"preferences",'
        '"return_to_agent":"true"}',
        '{"type":"final_answer","answer":"You prefer chamomile tea."}',
    ]
    captured = _drive_planner(sim, scenario, planner_replies)

    # The last planner message carries every observation the agent has seen,
    # including the memory_read result. The written body must be recalled there.
    recall_context = captured[-1] if captured else ""
    assert body_token in recall_context, (
        "COUNTEREXAMPLE (Bug 1 confirmed): after memory_append then memory_read, "
        f"the recalled context does NOT contain the written body ({body_token!r}); "
        "nothing persisted across turns. Read-back the agent saw:\n"
        f"{recall_context}"
    )


# --------------------------------------------------------------------------- #
# Bug 2 - Lost follow-up (planner pinned to the original request)
# --------------------------------------------------------------------------- #
def test_bug2_planner_current_request_is_most_recent_message(sim):
    """After a follow-up, the request presented to the planner MUST be the user's
    most recent message, not the original (Property 5).

    Scenario: "when is the next FIFA World Cup match?" -> web_search answer ->
    "tell me more about that match". On unfixed code the planner user message
    keeps `user_request=<original>` while the follow-up sits deep in the context,
    so the current request != most recent message. EXPECTED: FAIL on unfixed code.
    """
    original_request = "when is the next FIFA World Cup match?"
    most_recent_message = "tell me more about that match"

    scenario = sim.Scenario(
        "lost_followup",
        original_request,
        replies=[most_recent_message, "no thanks"],
        fixtures={"web_search": {
            "ok": True, "count": 1,
            "result": "The next FIFA World Cup 2026 match is on June 11 2026.",
            "records": "- FIFA.com: 2026 World Cup opens June 11 2026.",
            "local": "The next World Cup match is June 11, 2026."}},
    )
    planner_replies = [
        '{"type":"tool_call","tool":"web_search","query":"next FIFA World Cup match",'
        '"return_to_agent":"true"}',
        '{"type":"final_answer","answer":"The next match is on June 11 2026 at Estadio Azteca."}',
        '{"type":"final_answer","answer":"That match is the 2026 opener in Mexico City."}',
    ]
    captured = _drive_planner(sim, scenario, planner_replies)

    # The final planner turn happens AFTER the follow-up was folded in.
    current_request = _parse_user_request(captured[-1])
    assert current_request == most_recent_message, (
        "COUNTEREXAMPLE (Bug 2 confirmed): after the follow-up, the planner's "
        f"current request is {current_request!r} but the most recent user message "
        f"is {most_recent_message!r}. The planner keeps answering the original "
        "request instead of the follow-up."
    )


# --------------------------------------------------------------------------- #
# Bug 3 - Count-based compaction that drops named entities
# --------------------------------------------------------------------------- #
def test_bug3_compaction_trigger_is_token_based(iris_source):
    """Compaction MUST trigger on estimated token usage (~80% of the context
    budget), not on a fixed follow-up-exchange count (Property 8).

    On unfixed code the gate is `@exchangesSinceCompact >= @maxContextExchanges`
    with no token estimate. EXPECTED: FAIL on unfixed code.
    """
    exchange_gate = re.search(
        r"@exchangesSinceCompact\s*>=\s*@maxContextExchanges", iris_source)
    has_token_estimate = bool(
        re.search(r"@approxTokens", iris_source)
        or re.search(r"@compactAtTokens", iris_source)
        or re.search(r"count\([^)]*\)\s*/\s*4", iris_source))

    assert has_token_estimate and exchange_gate is None, (
        "COUNTEREXAMPLE (Bug 3 confirmed): compaction is gated on a fixed "
        "follow-up count, not token usage. Found exchange-count gate="
        f"{bool(exchange_gate)}, token-estimate trigger={has_token_estimate}. "
        "Expected a token-based trigger (e.g. @approxTokens >= @compactAtTokens) "
        "and no @exchangesSinceCompact >= @maxContextExchanges gate."
    )


def test_bug3_compaction_prompt_preserves_named_entities(iris_source):
    """The compaction system prompt MUST instruct preservation of named entities
    / proper nouns needed to resolve later references like "that match"
    (Property 6).

    On unfixed code the prompt only asks to keep the "MOST RECENT request" and
    does not mention proper nouns / named entities / "that X" antecedents.
    EXPECTED: FAIL on unfixed code.
    """
    # Isolate the compaction system prompt text.
    m = re.search(
        r"compress a running voice-assistant conversation.*?(?:no JSON\.|\"\})",
        iris_source, re.DOTALL)
    prompt_text = (m.group(0) if m else iris_source).lower()

    entity_markers = ["named entit", "proper noun", "that x", "titles", "antecedent"]
    hits = [marker for marker in entity_markers if marker in prompt_text]

    assert hits, (
        "COUNTEREXAMPLE (Bug 3 confirmed): the compaction prompt does not protect "
        "named entities, so a summary can drop 'that match'. Prompt found:\n"
        f"{m.group(0) if m else '(compaction prompt not located)'}"
    )


def test_bug3_named_entity_survives_compaction(sim):
    """A named entity introduced in the running context MUST survive the
    token-based compaction (Property 6, Property 8).

    Post-fix the trigger is token-based (chars/4 over protocol + context +
    current request >= @compactAtTokens), not a fixed follow-up count, and the
    compaction prompt now instructs the model to preserve named entities and
    'that X' antecedents. Seed the running context past the token budget with a
    seeded entity ("Estadio Azteca") so compaction fires, and drive the entity-
    preserving summary the strengthened prompt produces; the entity must then be
    present in the post-compaction planner context so a later "that match"
    reference can still be resolved.

    (Updated from the old exchange-count trigger to the token-based contract in
    lockstep with task 4 / Design Decision 4 - same Property, new trigger.)
    """
    entity = "Estadio Azteca"
    # Seed a large running context (well past the ~60k-token @compactAtTokens
    # trigger: ~59 chars * 5000 ~= 295k chars ~= 73k tokens) that mentions the
    # entity, so the token-based compaction fires on the next turn.
    seed = ("\n\nprevious_exchange=\nuser_said=tell me about a famous stadium"
            f"\nassistant_answered=A famous one is the {entity} in Mexico City. "
            + ("The crowd, the history, the atmosphere were all discussed. " * 5000))
    scenario = sim.Scenario(
        "entity_compaction",
        "tell me more about that",  # follow-up style current request, no entity
        replies=["no thanks"],
        seed_context=seed,
    )
    planner_replies = [
        '{"type":"final_answer","answer":"It is a historic venue in Mexico City."}',
    ]
    # The compaction summary (an auxiliary, non-planner call) preserves the named
    # entity, mirroring the entity-preserving compaction prompt now in the
    # shortcut. The behavioral guarantee under test is that this summary flows
    # into the subsequent planner context (the prompt itself is asserted by
    # test_bug3_compaction_prompt_preserves_named_entities).
    entity_summary = (f"The user asked about a famous stadium; the {entity} in "
                      "Mexico City was discussed, and they want to know more.")
    captured = _drive_planner(sim, scenario, planner_replies,
                              aux_reply=entity_summary)

    post_compaction = [c for c in captured if "conversation_summary=" in c]
    assert post_compaction, (
        "expected the token-based compaction to fire once the running context "
        "exceeded the token budget, but it did not"
    )
    survived = any(entity in c for c in post_compaction)
    assert survived, (
        "COUNTEREXAMPLE (Bug 3 confirmed): after compaction the named entity "
        f"{entity!r} was dropped from the running context. Post-compaction planner "
        f"message:\n{post_compaction[0]}"
    )


# --------------------------------------------------------------------------- #
# Bug 4 - Weak / slow planner model
# --------------------------------------------------------------------------- #
def test_bug4_default_model_is_capable_non_reasoning(iris_source):
    """The planner model MUST be a capable non-reasoning instruct model within the
    ~25s budget (Property 7), not the weak default.

    On unfixed code `@nimModelIdRaw = text("meta/llama-3.1-8b-instruct")`.
    EXPECTED: FAIL on unfixed code.
    """
    m = re.search(r'@nimModelIdRaw\s*=\s*text\("([^"]+)"\)', iris_source)
    assert m, "could not locate @nimModelIdRaw in iris.cherri"
    model_id = m.group(1)

    assert model_id in (FIXED_PRIMARY_MODEL, FIXED_FALLBACK_MODEL), (
        "COUNTEREXAMPLE (Bug 4 confirmed): the default planner model is "
        f"{model_id!r}, the weak instruct model that breaks the flat-JSON "
        f"contract. Expected a capable non-reasoning model such as "
        f"{FIXED_PRIMARY_MODEL!r} (fallback {FIXED_FALLBACK_MODEL!r})."
    )


@pytest.mark.skipif(
    not os.environ.get("NIM_API_KEY"),
    reason="MANUAL / live NIM: set NIM_API_KEY to observe flat-JSON contract "
           "breaks on the weak model",
)
def test_bug4_weak_model_flat_json_contract_live(sim):  # pragma: no cover
    """MANUAL / live: run the tool-routing scenarios on the weak default model and
    record flat-JSON contract breaks (prose or non-flat JSON where a tool_call is
    expected). This needs the live NIM endpoint; it is skipped offline.
    """
    routing = [s for s in sim.scenarios()
               if str(s.asserts.get("first_type", "")).startswith("tool_call")]
    breaks = []
    for scenario in routing:
        res = sim.run(scenario, model=WEAK_DEFAULT_MODEL, use_mock=False, verbose=False)
        if res["failures"]:
            breaks.append((scenario.name, res["failures"]))
    assert not breaks, (
        "COUNTEREXAMPLE (Bug 4 confirmed): the weak model broke the routing "
        f"contract on: {breaks}"
    )


# --------------------------------------------------------------------------- #
# Hybrid memory backend contract (Property 1, 2, 3, 4, 11)
#
# Memory is a HYBRID, user-selectable backend: a one-time @memoryBackend selector
# (vocabulary hybrid | local | sheets, default hybrid, unknown -> hybrid) drives
# write-through to every selected store (local Files first, then the proxy Sheet
# mirror under hybrid) and local-first reads with proxy fallback. Both stores
# hold the SAME OKF-formatted concept entry. These tests exercise that contract
# against the faithful simulator model and against the iris.cherri source.
# --------------------------------------------------------------------------- #
def test_hybrid_selector_normalization(sim):
    """The @memoryBackend selector normalizes exactly like the shortcut: strip
    whitespace, lowercase, and fall an unrecognized value back to `hybrid`
    (Property 11).
    """
    assert sim.normalize_backend("hybrid") == "hybrid"
    assert sim.normalize_backend("local") == "local"
    assert sim.normalize_backend("sheets") == "sheets"
    # Whitespace + mixed case are normalized away.
    assert sim.normalize_backend("  HyBRID \n") == "hybrid"
    assert sim.normalize_backend("Local") == "local"
    assert sim.normalize_backend("\tSHEETS ") == "sheets"
    # Unknown / empty / the OLD vocabulary (auto, proxy) all fall back to hybrid.
    for unknown in ["", "  ", "auto", "proxy", "cloud", "both", "sheet", "files"]:
        assert sim.normalize_backend(unknown) == "hybrid", (
            f"unrecognized backend {unknown!r} must default to hybrid, got "
            f"{sim.normalize_backend(unknown)!r}")


def test_hybrid_write_through_writes_both_stores(sim):
    """Under `hybrid`, a memory write is WRITE-THROUGH: the same OKF entry lands
    in BOTH the local Files store and the proxy Sheet (Property 1, 11)."""
    mem = sim.MemoryEmulator("hybrid", proxy_ok=True)
    assert mem.local_on and mem.proxy_on
    wrote = mem.append("preferences", "I prefer tea over coffee")
    assert wrote is True
    assert "preferences" in mem.local and "preferences" in mem.proxy, (
        "hybrid write-through must write BOTH stores; got "
        f"local topics={list(mem.local)} proxy topics={list(mem.proxy)}")
    # The same OKF body is recoverable from each store.
    assert sim.okf_body(mem.local["preferences"][0]) == "I prefer tea over coffee"
    assert sim.okf_body(mem.proxy["preferences"][0]) == "I prefer tea over coffee"


def test_local_only_routing_writes_only_local(sim):
    """Under `local`, writes go to the Files store only; the proxy is untouched
    even when a proxy is configured (Property 11)."""
    mem = sim.MemoryEmulator("local", proxy_ok=True)
    assert mem.local_on is True and mem.proxy_on is False
    mem.append("log", "local only note")
    assert mem.local.get("log") and not mem.proxy, (
        f"local backend must not touch the proxy; proxy={mem.proxy}")


def test_sheets_only_routing_writes_only_proxy(sim):
    """Under `sheets`, writes go to the proxy Sheet only; the local Files store is
    untouched (Property 11)."""
    mem = sim.MemoryEmulator("sheets", proxy_ok=True)
    assert mem.proxy_on is True and mem.local_on is False
    mem.append("notes", "cloud only note")
    assert mem.proxy.get("notes") and not mem.local, (
        f"sheets backend must not touch the local store; local={mem.local}")


def test_sheets_selector_without_proxy_is_unavailable(sim):
    """`sheets` with no configured proxy has NEITHER leg available (local is off
    under sheets, proxy is off unconfigured) - the true fail-open case."""
    mem = sim.MemoryEmulator("sheets", proxy_ok=False)
    assert mem.local_on is False and mem.proxy_on is False
    assert mem.available() is False


def test_local_first_read_prefers_local_over_proxy(sim):
    """A read is LOCAL-FIRST: when the local copy exists it is returned without
    consulting the proxy (Property 4)."""
    mem = sim.MemoryEmulator("hybrid", proxy_ok=True)
    # Seed the two stores with DISTINCT bodies so we can tell which one answered.
    mem.local.setdefault("preferences", []).append(
        sim.build_okf_block("preferences", "LOCAL: tea"))
    mem.proxy.setdefault("preferences", []).append(
        sim.build_okf_block("preferences", "PROXY: coffee"))
    records = mem.read("preferences")
    assert "LOCAL: tea" in records and "PROXY: coffee" not in records, (
        f"local-first read must return the local copy; got {records!r}")


def test_local_first_read_falls_back_to_proxy_when_local_empty(sim):
    """When the local copy is empty/missing (e.g. an iCloud-full silent sync
    loss), the read FALLS BACK to the proxy mirror (Property 3, 4)."""
    mem = sim.MemoryEmulator("hybrid", proxy_ok=True)
    # Only the proxy mirror has the entry (local never synced).
    mem.proxy.setdefault("preferences", []).append(
        sim.build_okf_block("preferences", "I prefer tea over coffee"))
    records = mem.read("preferences")
    assert "I prefer tea over coffee" in records, (
        "read must fall back to the proxy when the local copy is empty; got "
        f"{records!r}")


def test_okf_round_trip_in_both_stores(sim):
    """Property 1/4: a stored entry is an OKF concept whose body reconstructs to
    the input, in BOTH stores, with the type derived from the topic."""
    body = "The 2026 World Cup opener is at Estadio Azteca on June 11 2026."
    mem = sim.MemoryEmulator("hybrid", proxy_ok=True)
    mem.append("log", body)
    for store_name, store in (("local", mem.local), ("proxy", mem.proxy)):
        block = store["log"][0]
        # A valid OKF concept: frontmatter fences + a topic-derived type.
        assert block.count("---") >= 2, f"{store_name} entry is not OKF: {block!r}"
        assert "type: Memory Entry" in block, (
            f"{store_name} OKF type not derived from topic: {block!r}")
        assert sim.okf_body(block) == body, (
            f"{store_name} OKF body did not round-trip: {sim.okf_body(block)!r}")
    # Topic->type derivation for the other topics.
    assert sim.okf_type("preferences") == "User Preference"
    assert sim.okf_type("journal") == "Journal"
    assert sim.okf_type("notes") == "Note"


def test_hybrid_write_then_read_returns_body(sim):
    """End-to-end through mock_tool_observation: memory_append then memory_read
    surfaces the written body via write-through + local-first read (Property 4)."""
    mem = sim.MemoryEmulator("hybrid", proxy_ok=True)
    env_w, _ = sim.mock_tool_observation(
        "memory_append", {"topic": "preferences", "body": "chamomile tea"},
        {}, mem)
    assert "ok=true" in env_w
    env_r, spoken = sim.mock_tool_observation(
        "memory_read", {"topic": "preferences"}, {}, mem)
    assert "ok=true" in env_r and "chamomile tea" in env_r, (
        f"write-then-read must return the body; read envelope: {env_r!r}")


def test_create_note_and_quick_journal_use_write_through(sim):
    """create_note (topic notes) and quick_journal (topic journal) reuse the same
    hybrid write-through path, landing an OKF entry in both stores (Property 1)."""
    mem = sim.MemoryEmulator("hybrid", proxy_ok=True)
    sim.mock_tool_observation("create_note",
                              {"title": "Idea", "body": "buy oat milk"}, {}, mem)
    sim.mock_tool_observation("quick_journal",
                              {"body": "great run today"}, {}, mem)
    assert sim.okf_body(mem.local["notes"][0]) == "buy oat milk"
    assert sim.okf_body(mem.proxy["notes"][0]) == "buy oat milk"
    assert sim.okf_body(mem.local["journal"][0]) == "great run today"
    assert sim.okf_body(mem.proxy["journal"][0]) == "great run today"


def test_fail_open_when_neither_backend_available(sim):
    """Property 10: with NEITHER backend available the memory op fails open
    (ok=false, "pick a backend") and the run still completes with a normal spoken
    answer, never halting."""
    # Direct tool observation: sheets selector + no proxy -> nothing available.
    mem = sim.MemoryEmulator("sheets", proxy_ok=False)
    env, spoken = sim.mock_tool_observation(
        "memory_append", {"topic": "log", "body": "remember this"}, {}, mem)
    assert "ok=false" in env and "not set up" in env.lower(), (
        f"expected a fail-open ok=false envelope, got: {env!r}")
    assert not mem.local and not mem.proxy, "nothing should have been written"

    # Full loop: the run answers normally and never exhausts the turn budget even
    # though the memory write fails open. Capture the run result while feeding
    # scripted planner replies so nothing touches the network.
    scenario = sim.Scenario(
        "failopen_neither", "remember that I like tea",
        replies=["no thanks"], proxy_ok=False, memory_backend="sheets",
        asserts={"first_type": "tool_call:memory_append",
                 "no_turn_exhaustion": True})
    replies = [
        '{"type":"tool_call","tool":"memory_append","topic":"log",'
        '"body":"I like tea","return_to_agent":"true"}',
        '{"type":"final_answer","answer":"I could not save that, but here I am."}',
    ]

    def fake_call_nim(system, user, *, model, max_tokens, temperature, mock_reply=None):
        if system == sim.PROTOCOL:
            return replies.pop(0) if replies else '{"type":"final_answer","answer":"ok"}'
        return "(aux)"

    original = sim.call_nim
    sim.call_nim = fake_call_nim
    try:
        res = sim.run(scenario, model="exploration", use_mock=False, verbose=False)
    finally:
        sim.call_nim = original
    assert res["failures"] == [], res["failures"]
    assert res["first_type"] == "tool_call:memory_append"


def test_source_memory_dispatch_is_hybrid(iris_source):
    """The iris.cherri source implements the HYBRID contract with the
    hybrid/local/sheets vocabulary (NOT the old auto/proxy names): a normalized
    selector defaulting to hybrid, derived @memLocalOn/@memProxyOn flags,
    write-through (local append then proxy mirror), and local-first read.
    """
    # Selector + vocabulary.
    assert '@memoryBackendRaw = text("hybrid")' in iris_source, (
        "the default @memoryBackend must be hybrid")
    assert '@memoryBackend == "hybrid"' in iris_source
    assert '@memoryBackend == "local"' in iris_source
    assert '@memoryBackend == "sheets"' in iris_source
    # Unknown -> hybrid fallback.
    assert '@memBackendOk == "no"' in iris_source and \
        '@memoryBackend = "hybrid"' in iris_source, (
        "unrecognized backend must fall back to hybrid")
    # Old vocabulary must be gone from the selector logic.
    assert '@memoryBackend == "auto"' not in iris_source
    assert '@memoryBackend == "proxy"' not in iris_source
    # Derived availability flags.
    assert "@memLocalOn" in iris_source and "@memProxyOn" in iris_source
    # Write-through: local append then proxy mirror, gated by the flags.
    append_block = re.search(
        r'if @tool\.text == "memory_append".*?(?=if @tool\.text == "memory_list")',
        iris_source, re.DOTALL)
    assert append_block, "memory_append dispatch not found"
    ab = append_block.group(0)
    assert 'if @memLocalOn == "yes"' in ab and "appendAgentFile" in ab, (
        "memory_append must append to the local Files store when local is on")
    assert 'if @memProxyOn == "yes"' in ab and "op=memory_append" in ab, (
        "memory_append must mirror to the proxy when the proxy is on")
    assert "@memWroteAny" in ab, "write-through must succeed if either leg wrote"
    # Local-first read with proxy fallback.
    read_block = re.search(
        r'if @tool\.text == "memory_read".*?(?=/\* --- Memory: save)',
        iris_source, re.DOTALL)
    assert read_block, "memory_read dispatch not found"
    rb = read_block.group(0)
    assert 'if @memLocalOn == "yes"' in rb and "getAgentFile" in rb, (
        "memory_read must read the local file first")
    assert '@memReadAnswered == "no"' in rb and "op=memory_read" in rb, (
        "memory_read must fall back to the proxy only when local is empty")
