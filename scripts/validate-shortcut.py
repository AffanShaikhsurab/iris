#!/usr/bin/env python3
"""Structural validator for compiled Iris shortcuts.

Cherri v2.3.0 has a code-generation bug: `else if` chains emit conditional
groups whose closing action (WFControlFlowMode 2) is never generated, which
iOS misexecutes. The source avoids `else if`, and this validator fails the
build if unbalanced control flow ever appears again.

Usage: python3 scripts/validate-shortcut.py <unsigned .shortcut file>
Signed (.aea) shortcuts cannot be parsed; pass the unsigned artifact.
"""

import plistlib
import re
import sys
from collections import Counter


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def main(path: str) -> None:
    with open(path, "rb") as fh:
        try:
            plist = plistlib.load(fh)
        except plistlib.InvalidFileException:
            fail(f"{path} is not a plain plist. Pass the unsigned artifact.")

    actions = plist["WFWorkflowActions"]
    identifiers = [a["WFWorkflowActionIdentifier"] for a in actions]

    raw_count = identifiers.count("is.workflow.actions.rawaction")
    if raw_count:
        fail(
            f"{raw_count} is.workflow.actions.rawaction action(s) found. "
            "iOS treats these as unsupported; use Cherri custom actions."
        )

    planner_calls = 0
    for action in actions:
        if action["WFWorkflowActionIdentifier"] == "is.workflow.actions.downloadurl":
            url = str(action.get("WFWorkflowActionParameters", {}).get("WFURL", ""))
            if "integrate.api.nvidia.com" in url:
                planner_calls += 1
    if planner_calls == 0 and "com.openai.chat.AskIntent" not in identifiers:
        fail(
            "No planner call found (neither an integrate.api.nvidia.com request "
            "nor a com.openai.chat.AskIntent action); the agent has no model."
        )

    groups: dict[str, list[int]] = {}
    stack: list[str] = []
    for index, action in enumerate(actions):
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
            if not stack or stack[-1] != group:
                fail(
                    f"Control-flow nesting mismatch at action {index} "
                    f"({action['WFWorkflowActionIdentifier']}): "
                    f"mode {mode} for group {group[:8]} does not match "
                    f"innermost open group."
                )
            if mode == 2:
                stack.pop()

    unclosed = [g[:8] for g, modes in groups.items() if 2 not in modes]
    if unclosed:
        fail(
            f"{len(unclosed)} control-flow group(s) missing their closing "
            f"action (WFControlFlowMode 2): {unclosed}. This is the Cherri "
            "`else if` bug; remove `else if` from the source."
        )
    if stack:
        fail(f"{len(stack)} control-flow group(s) left open at end of shortcut.")

    # A real key is nvapi- followed by a long token. Substring matching would
    # false-positive on the first-run guard's own validation regex.
    key_leak = sum(1 for a in actions if re.search(r"nvapi-[A-Za-z0-9_-]{20,}", str(a)))
    if key_leak:
        fail(f"{key_leak} action(s) appear to contain a real nvapi- key. Never bake keys into the source.")

    placeholder = sum(1 for a in actions if "nvapi-REPLACE-ME" in str(a))
    if placeholder == 0:
        fail(
            "No nvapi-REPLACE-ME placeholder Text action found. The key must "
            "live in an editable Text action (import questions are broken on "
            "iOS 18.5+), and distributed artifacts must ship the placeholder."
        )

    # Same protection for the optional Tavily web-search key.
    tavily_leak = sum(
        1 for a in actions
        if re.search(r"tvly-[A-Za-z0-9_-]{10,}", str(a))
        and "tvly-REPLACE-ME" not in str(a)
    )
    if tavily_leak:
        fail(f"{tavily_leak} action(s) appear to contain a real tvly- key. Never bake keys into the source.")

    # Google OAuth secrets must never be baked into the source either.
    google_secret_leak = sum(
        1 for a in actions if re.search(r"GOCSPX-[A-Za-z0-9_\-]{10,}", str(a))
    )
    if google_secret_leak:
        fail(f"{google_secret_leak} action(s) appear to contain a real Google client secret (GOCSPX-). Never bake secrets into the source.")

    google_token_leak = sum(
        1 for a in actions
        if re.search(r"1//[A-Za-z0-9_\-]{20,}", str(a))
        and "google-refresh-token-REPLACE-ME" not in str(a)
    )
    if google_token_leak:
        fail(f"{google_token_leak} action(s) appear to contain a real Google refresh token (1//...). Never bake secrets into the source.")

    shapes = Counter(tuple(sorted(m)) for m in groups.values())
    print(f"OK: {len(actions)} actions, {len(groups)} control-flow groups, all balanced.")
    print(f"    group shapes: {dict(shapes)}")
    print(f"    NIM planner calls: {planner_calls}, AskIntent calls: {identifiers.count('com.openai.chat.AskIntent')}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1])
