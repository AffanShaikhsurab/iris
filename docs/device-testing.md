# Device Testing

Compilation proves that Cherri can generate a Shortcut artifact. It does not
prove the Shortcut imports or behaves correctly on iPhone. Use this checklist for
manual validation.

## Environment

- Device:
- iOS version:
- Cherri version:
- Build workflow run:
- Signed artifact used:

## Import Tests

- [ ] Signed shortcut imports successfully.
- [ ] Unsigned shortcut import behavior is recorded, if tested.
- [ ] Shortcut appears as `Agent Router`.
- [ ] Siri can start it with "Hey Siri, Agent Router."

## Configuration Tests

- [ ] A real NVIDIA `nvapi-` key is pasted into the first Text action, replacing
  `nvapi-REPLACE-ME`.
- [ ] The model id in the second Text action is a small dense instruct model
  (default `meta/llama-3.1-8b-instruct`), not a reasoning/thinking model.
- [ ] Running with the placeholder key still present speaks the setup
  instructions and stops (first-run guard).
- [ ] First run manually (unlocked) and accept the one-time
  `integrate.api.nvidia.com` network permission prompt.
- [ ] Answering the first prompt with "setup" runs the permission primer and
  fires each permission prompt once (network, Calendar, Reminders, Location,
  Weather, Files, Notes).
- [ ] Optional: a Tavily `tvly-` key is pasted into the third Text action, and
  the `api.tavily.com` network prompt is granted via the "setup" primer.
- [ ] Answers are spoken by Siri via the Ask-for-Input prompt when run from
  Siri, not only inside the Shortcuts app.

## Route Tests

- [ ] Agent loop: direct answer returns `final_answer` and does not run a tool.
- [ ] Agent loop: supported action returns `tool_call`, runs the tool, sends the
  tool result back to the model, then speaks a final answer.
- [ ] Agent loop: tool observations use `tool`, `ok`, `count`, `result`,
  `records`, and `error` fields.
- [ ] Agent loop: `ask_user` asks one short clarification, appends the answer to
  loop context, and continues.
- [ ] Agent loop: a second supported tool call can run in the same Shortcut
  execution when budget remains.
- [ ] Agent loop: after 8 agent turns, 3 tool calls, or 2 user questions, the
  shortcut forces a final summarization instead of continuing indefinitely.
- [ ] `answer_search`: a general knowledge question speaks and shows a useful
  answer from the model.
- [ ] `web_search` (Tavily key configured): "when is the next FIFA match" routes
  to `web_search` and returns a current answer. Without a key, it degrades to a
  model-knowledge answer with a caveat.
- [ ] `summarize_provided_text`: pasted text is summarized; private app data is
  not assumed when no text is provided.
- [ ] `draft_reply`: a reply is drafted for review and is not sent.
- [ ] `create_note`: Notes permission prompt appears if needed; note creation
  imports and runs without "action couldn't be found."
- [ ] `create_reminder`: Reminders permission prompt appears if needed; reminder
  creation imports and runs without "action couldn't be found."
- [ ] `quick_journal`: creates a journal-style Apple Note from dictated text.
- [ ] `calendar_lookup`: Calendar permission prompt appears if needed; upcoming
  events are fetched as structured title/start/end/calendar lines and
  summarized by the model. Record if duplicate calendars produce repeated events.
- [ ] `calendar_lookup`: normal runs do not create debug Notes or overwrite the
  clipboard.
- [ ] `reminders_lookup`: upcoming reminders are fetched and summarized by the
  model.
- [ ] `open_search`: approved search targets (`google`, `youtube`, `reddit`,
  `perplexity`, `maps`) open; other targets fall back to Google search.
- [ ] `open_destination`: approved destinations (`chatgpt`, `perplexity`,
  `calendar`) open; unknown targets do not open arbitrary model-invented URL
  schemes.
- [ ] `maps_search`: maps search opens for the requested query.
- [ ] `nearby_search`: nearby maps search opens for the requested query.
- [ ] `draft_message`: message text is drafted, copied, and shown; no message is
  sent.
- [ ] `weather_summary`: current weather and forecast are returned to the model
  for a spoken summary.
- [ ] `current_location_summary`: location permission prompt appears if needed;
  current location is returned to the model.
- [ ] `device_status`: device details are returned to the model and summarized.
- [ ] `memory_status`: OKF folder missing and present states are reported without
  stopping the Shortcut.
- [ ] Unknown/unsupported tool names return an `ok=false` observation and the
  agent adapts instead of running an invented action.

Planned routes (not yet implemented — a request for one returns an `ok=false`
unknown-tool observation): `draft_email`, `append_note`,
`save_clipboard_summary`, `daily_briefing`, `meeting_prep`,
`reply_to_shared_text`, `summarize_shared_email`, `memory_lookup`,
`memory_list_topics`, `memory_propose_write`, `memory_append_log`,
`email_lookup`, `gmail_lookup`.

## Prompt Matrix

Use these prompts for a first phone run:

Only implemented routes are listed below. See the "Planned routes" note above
for tools that are documented but not yet wired in.

| Route | Prompt | Expected behavior |
| --- | --- | --- |
| Agent loop direct stop | "What is a simple explanation of DNS?" | Model returns `final_answer`; no tool runs. |
| Agent loop ask user | "Remind me to call him tomorrow." | Model asks who or what title to use, receives the answer, then continues. |
| Agent loop multi-tool | "Plan my morning from my calendar, reminders, and weather." | Calendar/reminders/weather observations can be gathered and summarized within loop budget. |
| Agent loop budget | "Keep asking me questions before answering." | Shortcut stops after the question/step budget and forces a final answer. |
| `answer_search` | "Explain how DNS resolution works." | Model answers directly, speaks, and shows the result. |
| `web_search` | "When is the next FIFA World Cup match?" | Routes to Tavily (if a key is set) and returns a current answer. |
| `summarize_provided_text` | "Summarize this text: The project kickoff is Monday and the design review is Wednesday." | Model summarizes only the provided text. |
| `draft_reply` | "Draft a friendly reply saying I can join tomorrow at 3." | Reply text is drafted for review. |
| `create_note` | "Create a note called Product Ideas with three ideas for Agent Router." | A Notes action runs and creates a note. |
| `create_reminder` | "Remind me to call mom tomorrow at 6 PM." | Reminder action runs; date/time is recorded in notes until due-date support is validated. |
| `quick_journal` | "Journal that today I tested Agent Router and fixed shortcut syntax." | A journal-style Apple Note is created. |
| `calendar_lookup` | "What meetings do I have today?" | Upcoming calendar events are fetched as structured details, sent to the model, and summarized. |
| `reminders_lookup` | "What reminders are coming up?" | Upcoming reminders are fetched and summarized. |
| `open_search` | "Search YouTube for best iPhone shortcuts." | Approved YouTube search URL opens. |
| `open_destination` | "Open Perplexity." | Approved destination opens. |
| `maps_search` | "Search maps for coffee shops." | Maps search opens. |
| `nearby_search` | "Find restaurants near me." | Nearby maps search opens. |
| `draft_message` | "Draft a message to Sara saying I'll be 10 minutes late." | Draft is copied and shown; nothing sends. |
| `weather_summary` | "What's the weather today?" | Weather actions run and the model summarizes the result. |
| `current_location_summary` | "Where am I right now?" | Location action runs and the model summarizes it. |
| `device_status` | "What device am I using?" | Device details are fetched and summarized. |
| `memory_status` | "Check Agent Router memory status." | OKF folder status is returned as a tool observation. |

## Action Availability Matrix

The planner is a NIM `jsonRequest` to `integrate.api.nvidia.com`; model-backed
tools reuse that same call rather than a ChatGPT App Intent.

| Route | Expected app/action | Required permission | Result |
| --- | --- | --- | --- |
| `answer_search` | NIM chat completion | NVIDIA key, network | |
| `web_search` | Tavily search (`api.tavily.com`) | Tavily key, network | |
| `summarize_provided_text` | NIM chat completion | NVIDIA key, network | |
| `draft_reply` | NIM chat completion | NVIDIA key, network | |
| `create_note` | NIM + Notes Create Note | NVIDIA key, Notes | |
| `create_reminder` | Reminders Add New Reminder | Reminders | |
| `quick_journal` | NIM + Notes Create Note | NVIDIA key, Notes | |
| `calendar_lookup` | Calendar Get Upcoming Events | Calendar | |
| `reminders_lookup` | Reminders Get Upcoming Reminders | Reminders | |
| `open_search` | Open URL with allowlisted template | Browser/app handoff | |
| `open_destination` | Open URL with allowlisted destination | Browser/app handoff | |
| `maps_search` | Open Google Maps web search | Browser/Maps handoff | |
| `nearby_search` | Open Google Maps nearby search | Browser/Maps handoff | |
| `draft_message` | NIM + Clipboard | NVIDIA key, Clipboard | |
| `weather_summary` | Weather current conditions + forecast | Location/Weather | |
| `current_location_summary` | Current Location | Location | |
| `device_status` | Get Device Details | None/device support | |
| `memory_status` | Files folder contents / fixed OKF folder | Files/iCloud Drive | |

## Notes

Record exact failures, permission prompts, and screenshots where useful. Third
party app behavior can differ by app version and device.

## Current Status

Local Cherri compilation and plist inspection can validate generated action IDs,
but manual iPhone import and runtime testing must still be completed on a real
device. Do not mark network-dependent, permission-dependent, or third-party
routes as stable until this checklist has been run on the target phone.
