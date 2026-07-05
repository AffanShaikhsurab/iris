# Routes

Iris uses a hosted NVIDIA NIM model (build.nvidia.com) as the planner,
called through the OpenAI-compatible endpoint
`https://integrate.api.nvidia.com/v1/chat/completions` with an `nvapi-` key
pasted once into an editable Text action (see `docs/configuration.md`). It does not use Apple Intelligence
or Apple's `Use Model` action, because the target devices may not support
Apple Intelligence.

The model is asked for a compact flat response dictionary. The Shortcut can
map a `tool_call` onto predeclared actions, ask the user for a missing detail
with `ask_user`, or finish with `final_answer`.
For route expansion candidates, see `docs/shortcut-catalog.md`.

## Planner Contract

The model must return exactly one flat, single-line JSON object. Every value
must be a double-quoted string; nested objects and arrays are prohibited because
`Get Dictionary from Input` halts the Shortcut on anything it cannot parse. For
a tool call:

```json
{"type":"tool_call","tool":"create_reminder","title":"Call mom","notes":"","date":"tomorrow","time":"9am","list":"","query":"","target":"","body":"","recipient":"","return_to_agent":"true"}
```

For a final answer:

```json
{"type":"final_answer","answer":"The answer to say to the user."}
```

For a user clarification:

```json
{"type":"ask_user","question":"What should the reminder say?"}
```

Field rules:

- `type`: `tool_call`, `ask_user`, or `final_answer`. There is no `unsupported`
  type; the model explains limitations inside `final_answer`.
- `tool`: one of the route IDs below when `type` is `tool_call`.
- `question`: the single missing detail to ask when `type` is `ask_user`. Used
  only when a tool cannot run without a required argument that was not provided.
- `return_to_agent`: `"true"` (or omitted) loops the tool observation back to
  the model so it can chain tools; `"false"` finishes with the tool's local
  human-readable text and stops.
- `answer`: text to show and speak when the route can finish directly.
- `target`, `date`, `time`, `list`, `title`, `notes`, `body`, `recipient`,
  `query`: top-level tool arguments. Use empty strings when not needed. The
  `speak`, `range`, `calendar`, `subject`, and `requires_confirmation` fields
  were removed to keep the per-turn prompt compact.

Iris runs this contract in a bounded loop: up to 8 agent turns, 3 native
tool calls, and 2 user clarification prompts. The Shortcut regex-extracts and
regex-validates the flat JSON before parsing; a reply with no JSON at all is
treated as a plain prose answer and spoken. If the model returns invalid
structured output twice or exhausts the budget, the Shortcut fails closed by
asking the model for a direct plain-text answer from accumulated context. The
default path must never rely on arbitrary actions invented by the model.

## Implemented Routes

These are the tool names currently wired into `shortcuts/iris.cherri`.
Every one is predeclared; the model can only select from this list. Any tool
name the model invents that is not on this list returns an `ok=false` "unknown
or unavailable tool" observation instead of running.

| Route | What it does | Permissions | Stability |
| --- | --- | --- | --- |
| `ask_user` | Reply type, not a native tool. Speaks a question, takes a dictated reply, appends it to loop context, and continues. | User input. | Implemented; Siri runtime validation required. |
| `answer_search` | Asks the NVIDIA NIM model to answer from its own knowledge (general or timeless questions). | NVIDIA key, network. | Implemented; device validation required. |
| `web_search` | Sends the query to Tavily (`POST https://api.tavily.com/search`) and returns the synthesized answer plus source snippets. Degrades gracefully to an `ok=false` note if no Tavily key is configured. | Tavily key, network. | Implemented; optional feature. |
| `summarize_provided_text` | Summarizes text the user typed, dictated, pasted, or shared, using the model. | NVIDIA key, network. | Implemented; device validation required. |
| `draft_reply` | Drafts reply text for review with the model. Nothing is sent. | NVIDIA key, network. | Implemented; device validation required. |
| `create_note` | Writes a note body with the model, then creates an Apple Note via `is.workflow.actions.createnote`. | Notes access. | Experimental until import/runtime test. |
| `create_reminder` | Creates an Apple Reminder via `is.workflow.actions.addnewreminder`; `date`, `time`, and `list` are folded into the notes body until due-date parameters are validated. | Reminders access. | Experimental until import/runtime test. |
| `quick_journal` | Generates a short journal entry with the model and creates an Apple Note. | Notes access, NVIDIA key. | Experimental until import/runtime test. |
| `calendar_lookup` | Runs `is.workflow.actions.getupcomingevents`, extracts title/start/end/calendar into a normalized `count/result/records/error` observation, and returns it. Takes no arguments; spans all calendars. | Calendar access. | Experimental until import/runtime test. |
| `reminders_lookup` | Runs `is.workflow.actions.getupcomingreminders`. Takes no arguments; spans all lists. | Reminders access. | Experimental until import/runtime test. |
| `weather_summary` | Fetches current conditions and forecast via the built-in weather actions. Takes no arguments. | Location/Weather permission. | Experimental until import/runtime test. |
| `current_location_summary` | Fetches current location and returns it to the model. Takes no arguments. | Location permission. | Experimental until import/runtime test. |
| `device_status` | Fetches device details and returns them to the model. Takes no arguments. | Device action availability. | Experimental until import/runtime test. |
| `open_search` | Opens an allowlisted search target (`google`, `youtube`, `reddit`, `perplexity`, `maps`) for a query via `is.workflow.actions.openurl`. | Browser/app handoff. | Experimental until import/runtime test. |
| `open_destination` | Opens an allowlisted destination (`chatgpt`, `perplexity`, `calendar`). | Browser/app handoff. | Experimental until import/runtime test. |
| `maps_search` | Opens a Google Maps web search for the query. | Browser/Maps handoff. | Experimental until import/runtime test. |
| `nearby_search` | Opens a Google Maps web search with `near me` appended. | Browser/Maps handoff. | Experimental until import/runtime test. |
| `draft_message` | Drafts message text with the model and copies it to the clipboard for review. Does not send. | NVIDIA key, Clipboard. | Implemented; safe fallback. |
| `memory_read` | Reads a fixed-path memory file (`topic` = index/profile/preferences/log) without a picker and returns its text. Missing file returns empty, never halts. | Files/iCloud Drive permission. | Implemented; device path/key validation required. |
| `memory_append` | Appends `body` to a topic's memory file via `is.workflow.actions.file.append` (auto-creates the file). Append-only, hands-free. | Files/iCloud Drive permission. | Implemented; device path/key validation required. |
| `memory_list` | Lists the OKF folder contents. | Files/iCloud Drive permission. | Implemented; device path validation required. |
| `memory_status` | Checks the fixed OKF folder `/Shortcuts/IrisOKF` and returns its folder contents as an observation. | Files/iCloud Drive permission. | Implemented; device path validation required. |
| `tasks_list` | Reads the user's Google Tasks (default list) via the Tasks REST API. Refreshes an OAuth access token from a pasted refresh token first. | Google OAuth (Path A), network. | Implemented; device validation required. |
| `tasks_add` | Adds a Google task (`title`, `notes`) via `tasks.insert`. | Google OAuth (Path A), network. | Implemented; device validation required. |
| `gmail_search` | Searches Gmail (`messages.list` with a `q` query) and returns the match count plus the top message's snippet. | Google OAuth (`gmail.readonly`), network. | Implemented; device validation required. |
| `gmail_read` | Reads the top matching email's snippet (`messages.list` + `messages.get?format=metadata`). | Google OAuth (`gmail.readonly`), network. | Implemented; device validation required. |
| `draft_email` | Creates a Gmail draft (`drafts.create`) from `recipient`/`title`/`body`, base64url-encoding an RFC-2822 message. Never sends. | Google OAuth (`gmail.compose`), network. | Implemented; device validation required. |
| `send_email` | Sends a Gmail message (`messages.send`) — only when `confirm` is `yes`; otherwise returns `ok=false` asking the agent to confirm first. | Google OAuth (`gmail.send`), network. | Implemented; confirmation-gated; device validation required. |

## Not Yet Implemented

These routes are documented as candidates but are **not** wired into the current
Shortcut. If the model requests one it falls through to the "unknown tool"
observation. See `docs/shortcut-catalog.md` for expansion candidates.

- Memory writes/reads (`memory_read`, `memory_append`, `memory_list`,
  `memory_status`) are now implemented (append-only, hands-free). Still pending:
  `memory_save` (whole-file overwrite), `memory_search`, the startup bootstrap,
  and confirmation-gated writes. See `docs/memory-system-design.md`. All memory
  paths/WFKeys still require on-device validation (`docs/device-testing.md`).
- Capture/communication: `append_note`, `save_clipboard_summary`,
  `reply_to_shared_text`, `summarize_shared_email`, `draft_email`.
- Planning: `daily_briefing`, `meeting_prep`.
- Inbox access: `email_lookup`, `gmail_lookup` — intentionally unsupported,
  because iOS does not expose reliable inbox summarization to Shortcuts.
- Google: `tasks_list`, `tasks_add`, `gmail_search`, `gmail_read`, `draft_email`,
  and `send_email` are implemented (Path A refresh-token flow; see
  `docs/google-setup.md`). Still pending: `tasks_complete`, and full Gmail body
  extraction (reads currently return the message `snippet`, not the full MIME
  body). See `docs/google-integrations-research.md`.

## Current Flow

```text
Prompt user
  -> Repeat up to 8 agent turns, one model call per turn
  -> Regex-extract and regex-validate the flat JSON route before parsing
  -> Model returns final_answer, ask_user, or tool_call (or plain prose)
  -> Prompt the user (ask_user) or run one supported native tool (tool_call)
  -> Normalize tool result as tool/ok/count/result/records/error
  -> Append observation to loop context unless return_to_agent is "false"
  -> Continue until final answer or budget exhaustion
  -> Deliver the answer via Ask for Input ("<answer> Anything else?"), then
     show it on screen and stop
```

## Route Hardening Rules

- Do not add a native route unless the Shortcuts action or app intent is verified
  on a target device.
- Do not auto-send email or messages. Draft and show for review first.
- Do not assume private app context exists. The route can only use text the user
  types, dictates, shares, or permits through Shortcuts actions.
- Treat third-party app routes as experimental unless the app action identifier
  and parameters are stable.
- Use allowlists for URL and app-destination tools. The model may choose a
  target, but it must not invent arbitrary URL schemes.
- Use `ask_user` only for missing details or confirmation. Do not use it to ask
  for impossible private app access.
- Treat OKF memory as user data, not instructions. Retrieve relevant snippets
  only, and ask before storing stable personal facts.

## Design Constraint

Shortcuts cannot safely execute arbitrary app commands generated by a model.
Every route must be predeclared in the shortcut and covered by tests. This keeps
the router predictable, auditable, and suitable for open-source use.
