# Architecture

Iris is built around a constrained planner/executor loop.

## Planner

The planner is a hosted NVIDIA NIM model, called through the OpenAI-compatible
endpoint `POST https://integrate.api.nvidia.com/v1/chat/completions` with an
`nvapi-` key pasted once into an editable Text action. The API is stateless, so
a compact protocol (sent as the system message) plus the accumulated
conversation state (sent as the user message) is re-sent on every turn.

Each turn the model must return exactly one flat, single-line JSON object of one
of three types: `final_answer`, `ask_user`, or `tool_call`. There is no separate
`unsupported` type and no `speak` field; the model explains limitations inside
`final_answer`, and progress/answers are spoken by Siri through the
Ask-for-Input pattern. Tool calls carry a tool name plus top-level string
arguments such as `target`, `title`, `notes`, `query`, `date`, `time`, `list`,
`body`, and `recipient`.

This intentionally does not use Apple's `Use Model` / `askllm` action because
that path depends on Apple Intelligence support. Iris is meant for
devices that do not have Apple Intelligence. An earlier version used the ChatGPT
app's own `com.openai.chat.AskIntent` App Intent; that no-key backend shape is
preserved in `notes.md` and `shortcut-syntax-reference.md`.

The model is not trusted to execute commands. It only recommends one predeclared
tool from `docs/routes.md` per loop iteration. If the tool is unknown or the
structured output cannot be parsed, the shortcut falls back to a direct answer
instead of running an invented action.

## Executor

The executor is the generated Apple Shortcut. It can only run actions that were
compiled into the `.shortcut` file. This is what makes the system auditable:
adding a new capability means adding a route, tests, and documentation.

All user-facing speech goes through Siri via `Ask for Input`: Siri reads a
prompt aloud and takes the reply by dictation. The final answer is embedded in
the next prompt ("<answer> ... Anything else?") and shown on screen once at the
end as a record. There is no separate `speak` progress field and no custom
text-to-speech; see `docs/shortcut-runtime-flow.md` for the reasoning.

## Daily Workflow Packs

The route registry is organized around daily jobs instead of app names:

- Capture: notes, reminders, quick journal entries, and saved summaries.
- Search and open: allowlisted web/app destinations and map searches.
- Calendar and planning: upcoming events, reminders, daily briefings, and meeting
  preparation.
- Communication: draft-only email/message/reply flows that require user review.
- Reading and web: summaries of text, email, or pages the user explicitly shares.
- Utilities: weather, location, and device details.

This keeps the model focused on user intent while the Shortcut remains a small,
auditable executor.

## Bounded Loop

Iris uses a bounded repeat loop. Each iteration sends the original
request, current budgets, and accumulated observations to the model. The model
can finish, ask the user for one missing detail, or call one allowed tool. Tool
and user responses are appended to a text loop context and sent back on the next
iteration.

The budgets are per user request: 3 native tool calls, 2 user clarification
prompts, and 2 repair turns. Each fails closed into a final answer rather than
looping. A non-stop "Anything else?" follow-up starts a new request and resets
these per-request budgets, so a normal multi-turn conversation is never cut off.
The outer repeat loop is a large hard cap (50 total model calls) that exists
only to stop a pathological runaway; a normal conversation never reaches it.

### Context compaction

Because each model call is stateless, the whole loop context is re-sent every
turn. Over a long conversation that context grows and slows every call toward
the ~25s iOS timeout (and degrades accuracy). To bound it, compaction fires on
**estimated token usage** rather than a fixed exchange count. Each turn the
Shortcut estimates the outgoing prompt size — `count(@protocol) +
count(@loopContext) + count(@currentRequest)`, divided by 4 as a cheap
chars-to-tokens heuristic — and when that reaches `@compactAtTokens` (60000, the
latency-safe working budget) it makes one extra model call that summarizes the
running conversation into a short handoff, then replaces the raw log with that
summary. The 60k budget is set from measurement, not from the model's 128k
window: benchmarking (`tmp/bench_context.py`, `docs/context-engineering-plan.md`
§0.2) showed the pinned fast models answer a ~59k-token prompt in ~2s — well
inside the ~25s iOS budget — while high-variance models are excluded.
The summary prompt is instructed to preserve named entities and specifics
(proper nouns, titles, numbers, the most recent request, any "that X"
antecedent) so follow-up references survive. Compaction runs only on a follow-up
boundary, never mid-request, so no in-flight tool observation is lost, and if the
summary call fails the raw context is kept unchanged. This mirrors the
compaction/handoff pattern used by coding agents.

The planner is led by the most recent user message: the user-message template
sends `user_request={@currentRequest}`, which is repointed to each new follow-up
while the full prior conversation is retained in `@loopContext` as history. This
lets a follow-up like "tell me more about that match" be answered as a follow-up
rather than re-answering the original request.

Tool routes return data to the planner using a line-based result envelope. For
example, a calendar lookup can retrieve upcoming events, then send those events
plus the original user request and prior observations back to the model for
summarization. This preserves context across turns, because each model call is
stateless and receives the accumulated loop context.

The current envelope shape is:

```text
tool=<route>
ok=<true|false>
count=<number>
result=<short summary>
records=<structured records or snippets>
error=<empty or recoverable error>
```

`return_to_agent` controls whether the observation loops back. It defaults to
loopback; a tool call that explicitly sets `return_to_agent=false` finishes with
the tool's local human-readable text and stops. Because each model call is
stateless, the loop context carrying prior observations is what preserves
context across turns.

## OKF Memory Layer

Iris memory is **hybrid**: the same OKF-formatted concept entry is written to a
**phone-local Files store** (per-topic OKF Markdown under `Shortcuts/IrisOKF/`)
and/or a server-side append-only **Google Sheet** (`Iris Memory`, tab `memory`,
columns `timestamp | topic | type | title | tags | body`) reached through the
Apps Script proxy. A one-time `@memoryBackend` selector (an editable Text config
action, values `hybrid` default | `local` | `sheets`, unrecognized → `hybrid`)
chooses the backend; it is not a per-run spoken prompt (that would gate the
hands-free happy path). It derives two flags, `@memLocalOn` (`local`|`hybrid`)
and `@memProxyOn` (`sheets`|`hybrid` AND a configured proxy).

On-device testing confirmed the phone-local Files writes **do** persist — they
land in iCloud Drive under `Shortcuts/`, so the earlier "generated shortcuts
cannot emit `fileLocation` so writes no-op" conclusion was wrong and is retracted
(see `docs/memory-system-design.md` §0). The prior failures were a missing parent
folder before the first write and the user checking the wrong Files location,
both fixed by a `createFolder` seed/self-heal and an on-device write-location
probe. The proxy Sheet mirror remains because a phone-local write can silently
fail to sync when iCloud Drive is full; the Sheet is immune to iCloud quota.
Memory remains a predeclared-tool surface, not arbitrary file access; the model
passes a `topic` and the storage layer resolves the local path / sheet row.

The memory tools implemented in `shortcuts/iris.cherri` are hybrid:

- `memory_read(topic)`: return the recent bodies saved under a topic, reading
  local-first with proxy fallback.
- `memory_append(topic, body)`: write-through — append the OKF entry to the
  local file first, then mirror it to the sheet row (append-only); succeed if
  either leg wrote.
- `memory_list`: return the last N `topic: body` rows across all topics,
  local-first with proxy fallback.
- `memory_status`: report whether memory has content, local-first with proxy
  fallback.

`topic` is constrained to the allowlist `index profile preferences log notes
journal` (default `log`), and both stores self-heal on the first write. The
model-facing surface stays `memory_append(topic, body)` only — `type`/`title`/
`tags` are derived by the storage layer, keeping the planner prompt compact.
`create_note` and `quick_journal` reuse the same write-through. A startup
`memory_summary` bootstrap builds a compact (≤10000 char, ~2.1-2.4k tokens)
profile + preferences summary (local-first with proxy fallback) and injects it as
a **standing** `user_profile=` line on **every** planner turn (context
engineering, Phase 2) —
not a one-shot line in the loop context, so it survives compaction and is not
diluted by later observations. It is labelled data, never instructions.
Everything is fail-open: when no selected backend is available, Iris runs and
answers normally with no memory and never halts.

The Shortcut treats memory content as user data, not instructions. The local
file text (OKF Markdown, never JSON) is read as opaque text and never parsed;
only the outer proxy JSON envelope is coerced with `getDictionary()` — the
`records` text is forwarded to the planner opaquely. Relevant snippets may be
sent to the model, so the runtime avoids dumping the whole store into prompts.

## Siri Invocation

The phrase "Hey Siri, Iris" starts the shortcut by name. Siri does not
grant this shortcut the same private context that Apple Intelligence uses. The
shortcut can work with:

- Text the user dictates.
- Text or URLs shared into the shortcut.
- Clipboard input when the user allows it.
- Native Shortcuts actions available on that device.
- Hybrid OKF memory (phone-local Files and/or the proxy Google Sheet) that the
  user created or approved.

It cannot automatically inspect all apps, Gmail, emails, messages, or screen
contents unless Apple or the third-party app exposes those inputs through
Shortcuts actions and the user grants permission.

## Safety Tiers

Iris exposes low- and medium-risk routes first:

- Low risk: answer, summarize, draft, show, copy, or open an allowlisted
  destination.
- Medium risk: create a note/reminder, open a compose sheet, or fetch
  Calendar/Reminders/Weather/Location data with permission.
- High risk: auto-send messages, run SSH, open arbitrary x-callback URLs, run
  arbitrary shortcuts, delete content, control payments, or mutate smart-home
  devices.

High-risk capabilities should stay unavailable unless they are hardcoded,
allowlisted, and require explicit confirmation.

## Generation Backend

The Shortcut source is written in Cherri at `shortcuts/iris.cherri`.
Cherri is a Go-based compiler for Apple Shortcuts. It parses `.cherri` files,
type-checks Shortcut actions, and emits `.shortcut` artifacts.

This project does not maintain a Python generator. Cherri is the source of truth
for Shortcut logic, and GitHub Actions is the canonical build path when local
Windows support is unavailable or unreliable.
