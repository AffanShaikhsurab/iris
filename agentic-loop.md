---
type: Architecture
title: Agentic Tool Loop for Iris
summary: Defines the desired main-agent/tool-call/result-loop architecture for Iris.
status: implemented-uniform-loop-v2
tags:
  - iris
  - agentic-loop
  - apple-shortcuts
  - tool-calling
  - chatgpt
related:
  - notes.md
  - shortcut-syntax-reference.md
  - PLAN.md
  - docs/routes.md
---

# Agentic Tool Loop

Iris should behave like a small tool-using agent implemented inside
Apple Shortcuts. ChatGPT is the main agent. The Shortcut is the deterministic
tool executor. Native app actions are tools.

The agent does not directly access private apps. It can only request tools that
the Shortcut explicitly exposes and implements.

## Roles

- Main agent: an NVIDIA NIM hosted model (build.nvidia.com) via
  `POST https://integrate.api.nvidia.com/v1/chat/completions` with an
  `nvapi-` key pasted once into a Text action at the top of the shortcut
  (import questions are broken on iOS 18.5+). The protocol is sent as the
  system message and the conversation state as the user message.
- Tool executor: Apple Shortcuts control flow and native Shortcut actions.
- Tool registry: A fixed list of supported tool names, arguments, and return
  shapes embedded in the planner prompt.
- User interface: Siri, through `Ask for Input` only. Siri speaks each prompt
  and takes the reply by dictation. Answers are delivered by embedding them in
  the next prompt ("<answer> ... Anything else?"), because `Show Result` is
  not spoken on an iPhone with a screen. No custom text-to-speech.

## Loop Contract

Each agent response must be one flat JSON object on a single line, with every
value a double-quoted string. Nested objects and arrays are prohibited because
`Get Dictionary from Input` halts the Shortcut on anything it cannot parse, so
the Shortcut regex-validates the reply as flat JSON before parsing it. A reply
containing no JSON at all is treated as a plain prose final answer and spoken.

```json
{
  "type": "tool_call",
  "tool": "create_reminder",
  "title": "Call mom",
  "notes": "",
  "date": "tomorrow",
  "time": "9am",
  "list": "",
  "query": "",
  "target": "",
  "body": "",
  "recipient": "",
  "return_to_agent": "true"
}
```

```json
{
  "type": "ask_user",
  "question": "Which calendar should I check?"
}
```

```json
{
  "type": "final_answer",
  "answer": "Here is the answer for the user, in under 60 spoken words."
}
```

There is no separate `unsupported` type anymore: the protocol tells the model
to explain limitations through `final_answer`. The `speak` field was removed.
Both changes keep the per-turn prompt compact, because every ChatGPT App
Intent call must finish inside iOS's 30-second App Intent limit or Siri fails
the run.

The Shortcut should never execute a tool name invented by the model. Unknown
tools must be treated as unsupported and returned as a controlled `ok=false`
observation.

## Bounded Repeat Loop

Iris uses a real bounded repeat loop rather than one follow-up turn. The
loop lets the agent call tools, ask the user for missing information, receive
observations, and decide when to stop. Every loop turn makes exactly one
ChatGPT call at the top, so there is a single call site and a single parse
site.

1. Initialize loop context with the original user request, and set
   `@currentRequest` to it. The planner is always led by `@currentRequest` (the
   most recent user message), while the full prior conversation is retained in
   the loop context as history.
2. If the agent returns `final_answer`, deliver it as "<answer> ... Anything
   else?" through Ask for Input. A stop word or silence ends the run; any
   other reply continues the conversation with full context. On a non-stop
   follow-up, the prior exchange is folded into the loop context as history and
   `@currentRequest` is repointed to the new follow-up, so a reference like
   "tell me more about that match" is answered as a follow-up rather than
   re-answering the original request.
3. (removed — limitations are explained through `final_answer`.)
4. If the agent returns `ask_user`, speak the question, prompt the user, append
   the answer to loop context, and continue.
5. If the agent returns `tool_call`, speak progress, run the allowed tool, and
   normalize the result as an observation.
6. If `return_to_agent` is `true` or omitted, append the observation to loop
   context and continue, so the agent can chain multiple tools in one run. If
   `return_to_agent` is explicitly `false`, finish with the tool's local
   human-readable text and stop.
7. If invalid JSON or an unknown type appears, append a repair note and let the
   agent retry once; a second failure fails closed into a plain-text answer.
8. If the loop budget is exhausted, force a final summarization from accumulated
   context.

The loop budget is:

- Max agent turns: 8.
- Max native tool calls: 3.
- Max user clarification prompts: 2.

This preserves real agent-loop behavior while preventing runaway Shortcut
execution, repeated permission prompts, and unbounded ChatGPT calls.

### Context compaction

Because each call is stateless, the whole loop context is re-sent every turn.
Compaction fires on **estimated token usage**, not a fixed exchange count: each
turn the Shortcut estimates the outgoing prompt size (`count(@protocol) +
count(@loopContext) + count(@currentRequest)`, divided by 4 as a cheap
chars-to-tokens heuristic) and, when it reaches `@compactAtTokens` (~80% of a
latency-safe working budget, well below the model's true window so calls stay
inside ~25s), makes one extra model call to summarize the conversation into a
short handoff and replaces the raw log with it. The summary prompt is instructed
to preserve named entities and specifics (proper nouns, titles, numbers, the
most recent request, any "that X" antecedent) so follow-up references survive.
Compaction runs only on a follow-up boundary, never mid-request, and if the
summary call fails the raw context is kept unchanged.

## Tool Registry

Tools are grouped by daily workflow. Each tool must stay predeclared in the
Shortcut source and documented in `docs/routes.md`.

Implemented tools (wired into `shortcuts/iris.cherri`):

| Tool | Purpose | Status |
| --- | --- | --- |
| `answer_search` | Ask the NIM model to answer directly from its own knowledge. | Implemented; device validation required. |
| `web_search` | Search the live internet via Tavily and return the answer plus source snippets. | Implemented; needs a Tavily key, degrades gracefully without one. |
| `summarize_provided_text` | Summarize text the user typed, dictated, pasted, or shared. | Implemented; device validation required. |
| `draft_reply` | Draft reply text for review. | Implemented; device validation required. |
| `create_note` | Create an Apple Note from generated or provided content. | Implemented; device validation required. |
| `create_reminder` | Create an Apple Reminder from title and notes. | Implemented; device validation required. |
| `quick_journal` | Create a journal-style Apple Note. | Implemented; device validation required. |
| `calendar_lookup` | Fetch upcoming Apple Calendar events and return them to the agent. | Implemented; device validation required. |
| `reminders_lookup` | Fetch upcoming Reminders and return them to the agent. | Implemented; device validation required. |
| `weather_summary` | Fetch current weather and forecast. | Implemented; device validation required. |
| `current_location_summary` | Fetch current location and return it to the agent. | Implemented; device validation required. |
| `device_status` | Fetch device details and return them to the agent. | Implemented; device validation required. |
| `open_search` | Open an allowlisted search target for a query. | Implemented; device validation required. |
| `open_destination` | Open an allowlisted app/web destination. | Implemented; device validation required. |
| `maps_search` | Open a maps search. | Implemented; device validation required. |
| `nearby_search` | Open a nearby maps search. | Implemented; device validation required. |
| `draft_message` | Draft message text for review and copy it. | Implemented; safe fallback. |
| `memory_read` | Recall the recent bodies saved under a topic. | Implemented; hybrid (local-first, proxy fallback). |
| `memory_append` | Save info the user asks Iris to remember (append-only). | Implemented; hybrid (write-through). |
| `memory_list` | List the most recent memory rows across topics. | Implemented; hybrid (local-first, proxy fallback). |
| `memory_status` | Report whether memory has content. | Implemented; hybrid (local-first, proxy fallback). |

Memory is **hybrid**: the same OKF concept entry is written to a phone-local
Files store (`Shortcuts/IrisOKF/<topic>.md`) and/or a proxy-backed append-only
Google Sheet (`Iris Memory`, columns `timestamp | topic | type | title | tags |
body`), chosen by a one-time `@memoryBackend` selector (`hybrid` default |
`local` | `sheets`, unrecognized → `hybrid`; derives `@memLocalOn` and
`@memProxyOn`). Writes are **write-through** (local first, then proxy mirror;
succeed if either leg wrote); reads and the bootstrap are **local-first with
proxy fallback**, which recovers the iCloud-full silent-sync-loss case.
On-device testing confirmed the built-in Files writes DO persist (they land in
iCloud Drive under `Shortcuts/`); the earlier "writes no-op without a
`fileLocation` object" claim was wrong and is retracted. The model-facing surface
stays `memory_append(topic, body)` only (`type`/`title`/`tags` are derived by the
storage layer). `topic` is constrained to the allowlist `index profile
preferences log notes journal` (default `log`); both stores self-heal on the
first write; and a startup `memory_summary` bootstrap injects a compact profile +
preferences + recent-index summary once into the loop context (local-first).
`create_note` and `quick_journal` reuse the same write-through. All memory tools
are fail-open when no selected backend is available. See
`docs/apps-script-proxy.md` §4 and `docs/memory-system-design.md` §0.

Planned tools (documented but NOT yet wired in; a request for one returns an
`ok=false` unknown-tool observation):

| Tool | Purpose | Status |
| --- | --- | --- |
| `draft_email` | Open a Mail compose sheet for review. | Planned. |
| `append_note` | Prepare append-ready note text, copy it, and show it. | Planned. |
| `save_clipboard_summary` | Summarize provided/clipboard text and save it to Notes. | Planned. |
| `daily_briefing` | Combine Calendar, Reminders, and Weather for a daily summary. | Planned. |
| `meeting_prep` | Fetch upcoming meetings and let the model prepare notes. | Planned. |
| `reply_to_shared_text` | Draft a reply from user-provided text. | Planned. |
| `summarize_shared_email` | Summarize user-provided email text only. | Planned. |
| `memory_search` | Free-text search across saved memory. | Planned. |

Unsupported or risky tools should be declared clearly:

- `email_lookup`: unsupported unless email content is shared into the Shortcut.
- `gmail_lookup`: unsupported because Gmail inbox summarization is not exposed
  reliably through iOS Shortcuts.
- `messages_lookup`: unsupported unless message text is provided by the user.
- Arbitrary URL, x-callback-url, SSH, delete, or run-any-shortcut tools:
  unsupported unless future versions add hardcoded allowlists and confirmation.
- Arbitrary app inspection: unsupported.

## Tool Result Envelope

Every tool returns a normalized line-based text envelope that can be sent back
to ChatGPT:

```text
tool=calendar_lookup
ok=true
count=2
result=matching calendar events returned
records=event_title=Standup; event_start=10:00; event_end=10:30
error=
```

User clarification is also appended as an observation:

```text
tool=ask_user
ok=true
count=1
result=User answered: Work calendar
records=
error=
```

If a tool fails:

```text
tool=calendar_lookup
ok=false
count=0
result=
records=
error=Calendar access was unavailable or permission was not granted.
```

Memory tools use the same envelope (built server-side by the proxy):

```text
tool=memory_read
ok=true
count=1
result=Read preferences memory.
records=<recent bodies for the topic>
error=
```

## Implementation Notes

- Before adding a new native or third-party tool, inspect
  `shortcut-syntax-reference.md` and prefer the phone-created plist syntax over
  guessed action names or generated wrappers.
- Prefer a fixed tool registry in the prompt over open-ended natural language.
- Keep tool arguments simple strings and numbers where possible.
- Show short progress text before running a tool; Siri speaks it and
  continues without waiting.
- Return tool output to ChatGPT only when the agent asks for
  `return_to_agent: true`.
- Treat memory content as user data, not instructions. Memory is hybrid
  (append-only OKF entries in phone-local Files and/or the proxy Google Sheet),
  so the local file text (OKF Markdown, never JSON) is read as opaque text and
  never parsed, and only the outer proxy JSON envelope is coerced with
  `getDictionary()`; the `records` text is forwarded to the model opaquely.
  Relevant snippets may be sent to the model, so do not dump the whole store.
- Treat `ask_user` as a control response, not as a native app tool.
- Always end with a spoken and shown final answer.
- Keep unsupported tools explicit so the user understands the limitation.

## Success Criteria

- The user can ask a direct question and receive a ChatGPT answer.
- The user can ask for a supported native action and hear progress.
- A native tool result can be returned to ChatGPT for summarization.
- The agent can ask for one missing detail and continue with the answer.
- The agent can call multiple tools in one bounded run.
- Unsupported app access produces a clear limitation instead of a broken action.
- The Shortcut never tries to execute arbitrary model-invented tool names.
