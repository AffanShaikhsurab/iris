---
type: Architecture
title: Shortcut Runtime Flow And Loop Failure Notes
summary: Documents the real Apple Shortcuts execution model, Agent Router loop model, and known failure modes found during iPhone testing.
status: active-debug-reference
tags:
  - apple-shortcuts
  - agent-router
  - runtime-flow
  - debugging
related:
  - ../shortcut-syntax-reference.md
  - ./routes.md
  - ./device-testing.md
---

# Shortcut Runtime Flow And Loop Failure Notes

This file records the runtime model that Agent Router must follow. Read this
before changing `shortcuts/agent_router.cherri`.

## Critical Compiler Rule: Never Use `else if`

Cherri v2.3.0 (latest release as of 2026-07) compiles `else if` chains
incorrectly: only the innermost conditional group receives its closing action
(`WFControlFlowMode: 2`). Every outer group in the chain is emitted with modes
0/1 and never closed, which corrupts the control-flow nesting that iOS relies
on. Decoding old builds showed 28 of 54 groups unclosed; on device this makes
everything after a taken branch get skipped or misattributed to the wrong
block. This was the root cause of the "loop compiles but never works" symptom.

Rules for `shortcuts/agent_router.cherri`:

- Never write `else if`. Plain `if { } else { }` and nested blocks compile
  correctly.
- For dispatch, use flat guarded ifs plus a handled flag, or
  default-then-override assignment.
- Every build must pass `scripts/validate-shortcut.py`, which decodes the
  unsigned plist and fails on unbalanced or misnested control-flow groups and
  on `is.workflow.actions.rawaction`.

## Siri Is The Voice Interface (Ask-for-Input pattern)

When Agent Router is started by voice ("Hey Siri, Agent Router"), the only
reliable conversational primitive is **Ask for Input** (`prompt()`): Siri
reads the prompt aloud in her own voice, then listens and returns the reply
as dictated text. Everything else is unreliable:

- **Show Result is NOT spoken on an iPhone with a screen.** The voice-only
  dialog is spoken only on screenless surfaces (HomePod, CarPlay, AirPods).
  Mid-loop `show()` calls also risk halting flow. Use `show()` at most once,
  as the final on-screen record.
- Do NOT use `makeSpokenAudio()`/`playSound()` (robotic device TTS that
  fights Siri) or `speak()` (Siri prefixes "Your shortcut says..." and
  overrides the voice; also blocks locked-device runs).
- `Show Alert`/`confirm()` treats silence as cancel under Siri.

Therefore answers are delivered with the S-GPT community pattern: embed the
answer text in the prompt of the NEXT Ask for Input — "<answer> Anything
else?". Siri speaks the answer and immediately listens. A stop word ("no",
"stop", "thanks", or silence/cancel) ends the run; any other reply becomes a
follow-up request appended to the loop context, giving multi-turn
conversation with preserved context.

**Spoken-answer truncation (fixed).** An Ask for Input prompt is itself a
question, and Siri stops reading it aloud at the FIRST "?" in the text. So any
answer containing a question mark — a joke's setup ("Why did X? Because Y.")
especially — was cut off at that "?", and the user heard only the first few
words before Siri started listening. Fix: before embedding, neutralize every
"?" inside the answer to "." so the only question mark in the spoken prompt is
the trailing "Anything else?". Also flatten newlines to ". " (Siri mangles or
stops on line breaks) and strip Markdown. Answer length is adaptive, NOT hard-capped:
the protocol tells the model to keep simple answers to a sentence or two but
give a full, detailed answer when the user asks for detail or explanation, and
`max_tokens` is 600 so detail is not cut off by the token limit. (An earlier
build capped everything at <40 words; that was wrong — a detailed question must
get a detailed answer.) Separately, iOS 18.4+ has a TTS bug that can truncate
very LONG spoken text; there is no in-shortcut fix for that beyond the model
self-limiting, and the full answer is still shown on screen at the end. Do NOT
switch to a `speak()`/Speak Text action to dodge the "?" issue: it prepends
"Your shortcut says…", can use the wrong voice, and blocks locked-device runs.

## Works Manually But Fails Under Siri ("something went wrong")

If a run succeeds when launched manually from the Shortcuts app but Siri says
"something went wrong", the cause is almost never the code — it is an
interactive gate that a hands-free/locked Siri run cannot clear. The first
`Get Contents of URL` (the NIM planner call) and every `Ask for Input` are both
interactive gates. Two things to check, in order:

1. **First-run network permission.** The first time a newly imported or
   rebuilt shortcut runs `Get Contents of URL`, iOS shows a one-time
   "Allow to connect to integrate.api.nvidia.com?" modal. A manual run lets you
   tap Allow; a hands-free Siri run cannot, so it aborts. **Fix: after every
   re-import/rebuild, run the shortcut once manually and tap Allow.** The grant
   is remembered per-shortcut, so subsequent Siri runs work. A rebuild resets
   both this grant and the pasted key.
2. **Ask for Input needs an unlocked device.** Under Siri, interactive input
   forces an unlock; on a truly locked screen (or HomePod) it fails. Enable
   Settings → Face ID & Passcode → Allow Access When Locked → Siri, and unlock
   when Siri asks. Distinguishing test: if it fails even while unlocked and
   in-hand, it is #1 (network grant); if it only fails while locked, it is #2.

Rather than exercising each route by hand, use the built-in **permission
primer**: run Agent Router manually (unlocked) and answer the first prompt with
"setup". It fires every permission prompt at once (network, Calendar, Reminders,
Location, Weather, Files, Notes) so you tap Always Allow once each; hands-free
Siri runs then never popup. There is NO bulk "allow all" in iOS — consent is
per-action, per-shortcut, first-use, and resets on re-import. See
`docs/configuration.md` → "Permissions: one-time primer".

## Critical Setup Rule: No Import Questions

Cherri `#question` compiles to Apple's WFImportQuestions ("Setup" step), and
since iOS 18.5 that mechanism is broken: the setup flow runs, but the entered
value is never written into the bound action. Worse, Cherri emits the bound
Text action with an EMPTY value (the default lives only in the question), so
on a broken OS the API key ends up empty — every call 401s, and Siri reports
"Something went wrong". The key and model id therefore live in two plain
editable Text actions at the top of the shortcut (the S-GPT pattern), with a
first-run guard: if the key does not match `^nvapi-[A-Za-z0-9_-]{20,}$` after
whitespace stripping, the shortcut speaks setup instructions and stops.
`scripts/validate-shortcut.py` fails the build if the `nvapi-REPLACE-ME`
placeholder Text action is missing or a real-looking key appears.

## Critical Timing Rule: Keep Every Model Call Fast

iOS kills App Intents after 30 seconds and gives network actions roughly 25
seconds; the Siri voice path is empirically less patient than either. A slow
model call is killed by the system and surfaces as Siri's generic
"Something went wrong". Consequences for this repo:

- The planner is now a direct `jsonRequest` to
  `https://integrate.api.nvidia.com/v1/chat/completions` (NVIDIA NIM), which
  avoids the ChatGPT app's App Intent overhead and session failures.
- `Get Contents of URL` has a fixed, non-configurable timeout around 25
  seconds; a timed-out request ABORTS the whole run (Shortcuts has no
  try/catch), so the only defense is making the call fast. The default model
  is a small dense instruct model (`meta/llama-3.1-8b-instruct`); popular big
  models on NIM's free tier are frequently overloaded (30+ second responses).
  Never use reasoning/thinking models — some hang on NIM entirely, and
  thinking output breaks the flat-JSON contract. See `docs/configuration.md`.
- The protocol prompt re-sent every turn must stay compact (~1.5 KB today),
  and `max_tokens` is capped per call. Do not fatten either.
- The protocol instructs the model to answer in under 60 words.
- Models emit Markdown; the Shortcut strips `*_#` markers before text is
  spoken.
- Cherri note: `jsonRequest()` requires literal dictionaries for its body and
  headers arguments; interpolate variables inside the literal.
- An empty/failed API response is delivered to the user as an error message
  instead of looping.

## Critical Runtime Rule: Guard Get Dictionary From Input

`Get Dictionary from Input` does not return an error value on invalid JSON; it
halts the entire Shortcut. Model output must never reach `getDictionary()`
unchecked. Agent Router first regex-extracts the JSON object from the reply,
then regex-validates that it is one flat JSON object, and only then parses it.
A reply with no JSON at all is treated as a plain prose answer and spoken.

## Shortcuts Primitives We Can Rely On

Apple's Shortcuts app executes actions sequentially from top to bottom. There is
no separate graph scheduler. Visual connector lines are only magic-variable
references; named variables can make later blocks look disconnected even though
the Shortcut still continues.

Important primitives:

- **Ask for Input**: shows a dialog, returns the typed/dictated answer, and can
  be stored as a variable.
- **Set Variable**: stores an action output for later use by name.
- **Magic Variables**: reference a previous action's output using token
  attachments in the plist.
- **If**: branches based on conditions.
- **Repeat**: runs actions multiple times. In plist form it uses
  `is.workflow.actions.repeat.count` for both the opening and closing actions.
- **Show Result / spoken audio**: user-facing output. Showing raw route JSON or
  tool envelopes means the control layer leaked implementation text.
- **Stop and Output**: the cleanest hard-stop primitive for future builds. Cherri
  support should be validated before relying on it.
- **App Intent actions**: may prompt for permissions or fail if the target app is
  not installed, signed in, or reachable.

## App Intent Session Risk

The ChatGPT app action is an external dependency. Web reports and device testing
show it can fail with helper-communication/session problems or require the user
to open/sign in to the ChatGPT app. Agent Router cannot fully fix this inside
Shortcuts.

Operational rule:

- Open the ChatGPT app manually and confirm the account is signed in before
  testing Agent Router.
- If the ChatGPT app logs out or the App Intent fails, simplify the Shortcut
  will not help. The dependency has failed before the agent loop can reason.
- A future robust fallback should use a direct API call route, but that requires
  key management and is outside the current app-only design.

## Current Proven Startup Invariant

Generated shortcuts must preserve this first meaningful order:

```text
Ask for Text -> Set Variable request -> com.openai.chat.AskIntent
```

No Files, Calendar, Reminders, Weather, memory, or permission-dependent action
may run before the first ChatGPT action.

## Why The Loop Got Stuck

The restored loop can ask ChatGPT for route JSON on every turn. If the model
keeps returning:

- `ask_user` for a casual request like "say hi",
- a raw route JSON object as user-facing content,
- a raw `tool=...` envelope,
- invalid JSON that repair converts back into another route,

then the Shortcut can keep showing dialogs or processing route text until the
repeat budget ends. The user experiences this as repeated JSON popups or a loop.

The fundamental mistake is letting planner-control JSON become the user
conversation surface. Route JSON should be internal. User-facing output should
only be:

- `answer` from a valid `final_answer`,
- a local fallback created by a tool branch,
- a direct fallback answer after invalid JSON,
- a clear unsupported/tool-error explanation.

## Current Loop Contract (implemented)

The loop is a bounded state machine with exactly one ChatGPT call per turn:

1. Prompt the user, then enter `repeat turn for 8`.
2. Each turn sends the compact protocol + original request + accumulated
   observations + remaining budgets to ChatGPT (the app action is stateless,
   so everything is re-sent every time).
3. The reply is regex-extracted and regex-validated before `getDictionary()`.
   No JSON at all means the reply is delivered as a plain answer. Invalid
   JSON gets one repair turn, then fails closed into a plain-text answer.
4. `final_answer`: deliver the answer through the Ask-for-Input pattern
   ("<answer> ... Anything else?"). A stop word or silence finishes; any
   other reply is appended to context as a follow-up request and the loop
   continues.
5. `ask_user`: while the question budget (2) remains, `prompt()` the question
   (Siri asks it aloud and takes a dictated reply), append the answer as an
   observation, and continue. After that, fail closed into a direct answer.
6. `tool_call`: while the tool budget (3) remains, run only an allowlisted
   tool. Every tool produces a structured `toolResult` envelope for the agent
   and a human `localFinalText` fallback. `return_to_agent=false` delivers
   the local text directly; otherwise the observation loops back and the
   agent may chain further tools within the budget.
7. Unknown tools return an `ok=false` observation so the agent can adapt.
8. On finish, `show()` leaves the final text on screen as a record, then
   `stop()`. If the 8-turn budget is exhausted, show the latest local
   fallback or a direct plain-text summary.

The `unsupported` type was folded into `final_answer` (the protocol tells the
model to explain limits in the answer), and the `speak` field was removed —
both to keep the per-turn prompt small and fast.

## Practical Rule For Tool Results

Never speak or show raw envelopes like:

```text
tool=calendar_lookup
ok=true
records=...
```

Those are observations for the planner only. The Shortcut must speak/show
`localFinalText` if the planner fails to convert them into a natural answer.

## Reference Files

- `shortcut-syntax-reference.md`: decoded phone-created shortcut syntax and
  action identifiers.
- `docs/reference-shortcuts/icloud-new-shortcut-5/actions.json`: durable parsed
  reference shortcut action list.
- `docs/routes.md`: route registry and expected behavior.
- `scripts/simulate-agent.py`: faithful offline simulator — hits the real NIM
  API with the exact protocol/message, replicates the JSON guard, budgets,
  dispatch and delivery, mocks native tools, and asserts on route selection.
