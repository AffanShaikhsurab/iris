#!/usr/bin/env python3
"""Regression guards for the on-device CRASH class that the pre-existing tests
missed: a memory/proxy operation halting the whole Shortcut.

Root cause (confirmed 2026-07): `Get Dictionary from Input` HALTS the entire
Shortcut on non-JSON input (docs/shortcut-runtime-flow.md). Two compounding bugs
fed it, so asking Iris to STORE something crashed:

  1. The @proxyOk / @tavilyOk validators MATCHED their own shipped `REPLACE-ME`
     placeholders, so on the default `hybrid` backend @memProxyOn became "yes"
     with no real proxy.
  2. Every memory-store path then did downloadURL(...REPLACE-ME/exec...) -> got a
     Google HTML 404 -> getDictionary(@proxyResp) -> HALT.

These tests operate DIRECTLY on shortcuts/iris.cherri (the shipped artifact), not
on the Python simulator, because the crash lived in the real
downloadURL->getDictionary path that the simulator mocks away. They FAIL on the
pre-fix source and PASS after the fix.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
IRIS_CHERRI = REPO_ROOT / "shortcuts" / "iris.cherri"

# The exact strings the shortcut ships with (build-shortcuts.sh replaces these
# from .env.local only for a local build; CI/distributed builds keep them).
PLACEHOLDER_PROXY_URL = "https://script.google.com/macros/s/REPLACE-ME/exec"
PLACEHOLDER_TAVILY_KEY = "tvly-REPLACE-ME"
REAL_PROXY_URL = ("https://script.google.com/macros/s/"
                  "AKfycbx9Qw3rTt7bZ2mN8kLpVsDf1234567890abcdEFghIJ/exec")
REAL_TAVILY_KEY = "tvly-dev-abcdef0123456789ABCDEF"


@pytest.fixture(scope="module")
def source() -> str:
    return IRIS_CHERRI.read_text(encoding="utf-8")


def _extract_regex(source: str, var: str) -> str:
    """Pull the regex literal out of `@<var> = matchText('<regex>', ...)`."""
    m = re.search(rf"@{var}\s*=\s*matchText\('(.+?)',", source)
    assert m, f"could not find @{var} matchText(...) in iris.cherri"
    return m.group(1)


# --------------------------------------------------------------------------- #
# Bug 2 - config validators must REJECT the shipped placeholders
# --------------------------------------------------------------------------- #
def test_proxy_validator_rejects_placeholder(source):
    """@proxyOk must be 0 for the shipped placeholder URL, else @memProxyOn turns
    on with no real proxy and every memory write hits a dead URL."""
    pattern = _extract_regex(source, "proxyMatches")
    rx = re.compile(pattern)
    assert not rx.search(PLACEHOLDER_PROXY_URL), (
        "REGRESSION: the proxy validator MATCHES the shipped placeholder "
        f"{PLACEHOLDER_PROXY_URL!r} (pattern {pattern!r}). This makes @memProxyOn "
        "'yes' by default, so a memory write POSTs to a dead REPLACE-ME URL and "
        "getDictionary() on the HTML 404 halts the shortcut.")
    assert rx.search(REAL_PROXY_URL), (
        f"proxy validator must still accept a real /exec URL; pattern {pattern!r} "
        f"rejected {REAL_PROXY_URL!r}")


def test_tavily_validator_rejects_placeholder(source):
    pattern = _extract_regex(source, "tavilyMatches")
    rx = re.compile(pattern)
    assert not rx.search(PLACEHOLDER_TAVILY_KEY), (
        "REGRESSION: the tavily validator MATCHES the shipped placeholder "
        f"{PLACEHOLDER_TAVILY_KEY!r} (pattern {pattern!r}); web_search would fire "
        "against a bogus key.")
    assert rx.search(REAL_TAVILY_KEY), (
        f"tavily validator must still accept a real key; pattern {pattern!r} "
        f"rejected {REAL_TAVILY_KEY!r}")


# --------------------------------------------------------------------------- #
# Bug 1/3 - no proxy response may reach getDictionary() unguarded
# --------------------------------------------------------------------------- #
# The raw HTTP response variables. getDictionary() must NEVER be called directly
# on any of these; it must only ever parse a JSON object first EXTRACTED from
# them (a *Json var), or the store paths must not parse the response at all.
RAW_RESPONSE_VARS = ["proxyResp", "bootResp", "mTasksResp"]


@pytest.mark.parametrize("var", RAW_RESPONSE_VARS)
def test_no_unguarded_getdictionary_on_proxy_response(source, var):
    """A raw proxy/download response fed straight to getDictionary() halts the
    run on any non-JSON body (HTML error/redirect/empty). Every such call must be
    guarded by a JSON-extraction matchText first."""
    bad = f"getDictionary(@{var})"
    assert bad not in source, (
        f"REGRESSION: found unguarded {bad} in iris.cherri. A non-JSON body "
        "(Apps Script HTML error/redirect/empty) fed to getDictionary HALTS the "
        "shortcut. Extract a JSON object with matchText('(?s)\\{.*\\}', ...) "
        "first and parse that, or drop the parse (fire-and-forget).")


def test_store_paths_are_fire_and_forget(source):
    """memory_append / create_note / quick_journal only MIRROR to the proxy; the
    ack is unused, so their proxy leg must be fire-and-forget (no getDictionary).
    Extract each `if @memProxyOn == "yes" { ... }` proxy leg that POSTs
    op=memory_append and assert it does NOT parse the response."""
    lines = source.splitlines()
    store_urls = [i for i, ln in enumerate(lines) if "op=memory_append" in ln]
    assert store_urls, "no op=memory_append proxy legs found (test anchor drifted)"
    for i in store_urls:
        window = "\n".join(lines[i + 1:i + 7])  # the leg body after the URL
        assert "getDictionary(@" not in window, (  # actual call, not a comment
            "REGRESSION: a memory-store proxy leg parses its response with "
            "getDictionary (halts on non-JSON). Make it fire-and-forget like the "
            f"morning-persist path. Offending leg near line {i + 1}:\n{window}")
    # Intention marker the fix adds to each of the three interactive store legs.
    assert source.count("Fire-and-forget") >= 3, (
        "expected the three store proxy legs to be documented as fire-and-forget")


def test_guarded_getdictionary_forms_present(source):
    """Sanity: the guarded (extract-then-parse) forms the fix introduces are
    present, so the test above is guarding real code rather than a renamed var."""
    for guarded in ["getDictionary(@bootJson)", "getDictionary(@proxyJson)",
                    "getDictionary(@mTasksJson)"]:
        assert guarded in source, f"expected guarded form {guarded} in iris.cherri"


# The proxy-response JSON vars introduced by the fix (each produced by a
# matchText('(?s)\{.*\}', ...) extraction). The planner route (@routeJson) is
# guarded separately via @jsonMatches and is out of scope here.
PROXY_JSON_VARS = ["proxyJson", "bootJson", "mTasksJson",
                   "readJson", "listJson", "statJson"]


@pytest.mark.parametrize("var", PROXY_JSON_VARS)
def test_every_proxy_json_parse_has_extraction_guard(source, var):
    """Each proxy-family `getDictionary(@<x>Json)` must be preceded by the
    matching matchText JSON extraction that produced @<x>JsonMatches."""
    if f"getDictionary(@{var})" not in source:
        pytest.skip(f"@{var} not used")
    assert f"getFirstItem(@{var}Matches)" in source, (
        f"getDictionary(@{var}) has no matching JSON-extraction guard "
        f"(@{var}Matches) - it could be parsing an unvalidated body.")


# --------------------------------------------------------------------------- #
# Replica stays in sync with the fixed shortcut
# --------------------------------------------------------------------------- #
def test_replica_validators_match_shortcut(source):
    """scripts/iris_replica.py must use the SAME (fixed) validator regexes as the
    shortcut, so the faithful simulator reflects on-device configuration gating."""
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location(
        "iris_replica", REPO_ROOT / "scripts" / "iris_replica.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["iris_replica"] = mod
    spec.loader.exec_module(mod)

    assert mod.PROXY_OK_RE.pattern == _extract_regex(source, "proxyMatches"), (
        "iris_replica.PROXY_OK_RE drifted from the shortcut's @proxyMatches regex")
    assert mod.TAVILY_OK_RE.pattern == _extract_regex(source, "tavilyMatches"), (
        "iris_replica.TAVILY_OK_RE drifted from the shortcut's @tavilyMatches regex")

    # And the resolved config gates the placeholder OFF (the crash precondition).
    cfg = mod.Config(proxy_url=PLACEHOLDER_PROXY_URL,
                     tavily_key=PLACEHOLDER_TAVILY_KEY)
    assert cfg.proxy_ok is False and cfg.tavily_ok is False
    assert cfg.mem_proxy_on is False  # hybrid default, but proxy is off
    cfg2 = mod.Config(proxy_url=REAL_PROXY_URL, tavily_key=REAL_TAVILY_KEY)
    assert cfg2.proxy_ok is True and cfg2.tavily_ok is True


# --------------------------------------------------------------------------- #
# NIM / Tavily response parses must not halt on error bodies
# (getFirstItem on empty choices, or getDictionary on non-JSON)
# --------------------------------------------------------------------------- #
def test_every_choices_read_is_count_guarded(source):
    """Every `@X = getValue(@…, "choices")` must have a matching `count(@X)`
    guard, so a well-formed API error body (no choices) fails open instead of
    halting at getFirstItem."""
    choices_vars = re.findall(r'@(\w+) = getValue\(@\w+, "choices"\)', source)
    assert choices_vars, "no choices reads found (anchor drifted)"
    for v in choices_vars:
        assert f"count(@{v})" in source, (
            f"REGRESSION: `@{v} = getValue(...,'choices')` has no `count(@{v})` "
            "guard; getFirstItem on an empty choices array HALTS the run when the "
            "API returns a well-formed error body (retired model id / 401 / 429).")


def test_llm_responses_extract_json_before_getdictionary(source):
    """NIM and Tavily HTTP responses must be JSON-extracted before getDictionary
    (a non-JSON gateway/HTML body would HALT). After the fix, getDictionary runs
    on an extracted @…Json var, never on the raw @…Response."""
    for raw in ["nimResponse", "morningResponse", "compactResponse",
                "tavilyResponse"]:
        assert f"getDictionary(@{raw})" not in source, (
            f"REGRESSION: unguarded getDictionary(@{raw}) — a non-JSON body would "
            "halt the run. Extract a JSON object first.")
    # The guarded forms the fix introduces:
    for guarded in ["getDictionary(@nimJson)", "getDictionary(@tavilyJson)"]:
        assert guarded in source, f"expected guarded {guarded}"


def test_final_answer_missing_answer_does_not_speak_raw_json(source):
    """A final_answer route with no `answer` field must not deliver the raw route
    JSON to the user; the fix substitutes a clean re-ask."""
    assert "Sorry, I did not catch that." in source, (
        "final_answer empty-answer guard (clean re-ask) missing")


# --------------------------------------------------------------------------- #
# Open/navigation tools deliver directly (latency: no wasted loop-back turn)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tool", [
    # open/navigation (app switches away)
    "open_search", "open_destination", "maps_search", "nearby_search",
    # terminal action/data tools (bench_latency.py: loop-back is the cutoff risk)
    "weather_summary", "current_location_summary", "device_status",
    "create_reminder", "calendar_add", "tasks_add", "tasks_complete",
    "memory_append"])
def test_terminal_tools_are_direct_deliver(source, tool):
    assert re.search(rf'== "{tool}"\s*\{{ @speakRecords = "yes" \}}', source), (
        f"{tool} should be in the @speakRecords direct-deliver set to avoid a "
        "second planner round-trip (the dominant Siri-cutoff risk).")


def test_replica_speak_direct_matches_shortcut(source):
    """The replica's SPEAK_DIRECT set must equal the shortcut's @speakRecords set,
    so latency benchmarks on the replica reflect the shipped round-trip shape."""
    shortcut_set = set(re.findall(r'@tool\.text == "(\w+)"\s*\{ @speakRecords = "yes" \}',
                                  source))
    R = _load_replica()
    assert R.SPEAK_DIRECT == shortcut_set, (
        "replica SPEAK_DIRECT drifted from the shortcut @speakRecords set:\n"
        f"  only in replica:  {R.SPEAK_DIRECT - shortcut_set}\n"
        f"  only in shortcut: {shortcut_set - R.SPEAK_DIRECT}")


# --------------------------------------------------------------------------- #
# Replica behavioral parity with the fixed shortcut
# --------------------------------------------------------------------------- #
def _load_replica():
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location(
        "iris_replica", REPO_ROOT / "scripts" / "iris_replica.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["iris_replica"] = mod
    spec.loader.exec_module(mod)
    return mod


def _local_config(R, **over):
    fields = dict(nim_api_key="nvapi-" + "x" * 24, tavily_key="",
                  proxy_url="", proxy_secret="")
    fields.update(over)
    return R.Config(**fields)


def test_replica_open_tool_delivers_directly():
    R = _load_replica()
    cfg = _local_config(R)
    model = R.ModelClient(cfg, scripted=[
        '{"type":"tool_call","tool":"open_search","target":"google","query":"pizza"}'])
    eng = R.IrisReplica(cfg, R.DeviceState(), model)
    res = eng.run_chat("search google for pizza", ["no thanks"])
    assert res["first_type"] == "tool_call:open_search", res["first_type"]
    assert len(model.calls) == 1, "open_search must deliver directly (1 call)"


def test_replica_final_answer_without_answer_field():
    R = _load_replica()
    cfg = _local_config(R)
    model = R.ModelClient(cfg, scripted=['{"type":"final_answer"}'])
    eng = R.IrisReplica(cfg, R.DeviceState(), model)
    res = eng.run_chat("hi", ["no thanks"])
    assert "did not catch" in res["final_text"].lower(), res["final_text"]


def test_replica_empty_model_reply_is_graceful_not_crash():
    """An empty/failed model response (what the guarded shortcut yields on an API
    error) must produce the spoken error, never an exception."""
    R = _load_replica()
    cfg = _local_config(R)
    model = R.ModelClient(cfg, scripted=[""])  # simulate API failure
    eng = R.IrisReplica(cfg, R.DeviceState(), model)
    res = eng.run_chat("hello", ["no thanks"])
    assert res["final_text"] == R.NIM_ERROR_TEXT, res["final_text"]
    assert res["outcome"] == "finish"
