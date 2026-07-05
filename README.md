# Shortcut Agent Router

Shortcut Agent Router is an open-source proof of concept for building an
AI-routed Apple Shortcut from source code. The Shortcut is written in
[Cherri](https://cherrilang.org/), a language that compiles directly to Apple
Shortcuts.

The project is designed around a simple rule: the AI can choose from supported
routes, but every native action must be predeclared in the Shortcut source.

## What It Does

The Agent Router shortcut is model-first, using a hosted NVIDIA NIM model as the
planner:

- Ask any general question and get a spoken answer.
- Ask for current information: with an optional Tavily web-search key, the agent
  looks facts up on the live internet (news, prices, scores, "latest"/"current").
- Ask it to summarize text that you type, dictate, paste, or share.
- Ask for draft replies or messages for review; nothing is ever auto-sent.
- Route supported productivity tasks to predeclared native actions (notes,
  reminders, calendar, reminders lookup, weather, location, maps, and more).
- Hear progress and final answers spoken by Siri.

The planner is a hosted NVIDIA NIM model, called through the OpenAI-compatible
endpoint `https://integrate.api.nvidia.com/v1/chat/completions`. It requires a
free `nvapi-` key pasted once into an editable Text action (see
`docs/configuration.md`). Web search is optional and stays disabled until you
add a free Tavily key.

This shortcut does not use Apple's `Use Model` action or Apple Intelligence, so
it can run on older iPhones that do not support Apple Intelligence. An earlier
version used the ChatGPT app's `Ask ChatGPT` App Intent; that no-key backend is
preserved in `notes.md` if you want to use it again.

## How The Shortcut Works

1. You invoke the shortcut by name, for example: "Hey Siri, Agent Router."
2. The shortcut asks what you want to do.
3. It enters a bounded agent loop with accumulated context, making exactly one
   model call per turn.
4. The model can answer, ask one missing detail, or call one predeclared tool per
   loop step. Limitations are explained inside the answer rather than through a
   separate reply type.
5. The shortcut runs only supported native routes and appends observations back
   into the loop context.
6. It speaks and shows the final answer, or forces a final summary when the loop
   budget is exhausted.

Tool observations use a normalized line-based envelope with `tool`, `ok`,
`count`, `result`, `records`, and `error` fields so the next model turn can
reason over native action results.

For requests like “summarize my emails” or “check my calendar,” the shortcut can
only work with content you provide, share into it, or expose through verified
Shortcuts actions on your device. It does not have Apple Intelligence-level
private access to Gmail, Mail, Messages, or the current screen.

For durable context, Agent Router has an experimental hook for an optional
OKF-style folder at `Shortcuts/AgentRouterOKF/`. Only a folder-status check
(`memory_status`) is implemented today; deeper retrieval and writes are still
planned. Anything eventually inserted into a model prompt is sent to the
provider, so do not store sensitive information there. See
`docs/okf-knowledge-base.md`.

## Quickstart

Install Cherri with Homebrew on macOS or Linux:

```bash
brew tap electrikmilk/cherri
brew install electrikmilk/cherri/cherri
```

Or build/install it with Go:

```bash
go install github.com/electrikmilk/cherri@latest
```

Compile the shortcut:

```bash
mkdir -p dist
cherri shortcuts/agent_router.cherri --debug --output "dist/Agent Router.shortcut"
```

In Cherri `v2.3.0`, non-macOS compilation may sign through HubSign by default.
If the generated file starts with `AEA1`, it is already in signed/package format.
On Windows, use the WSL + HubSign command sequence in `docs/signing.md`; native
Windows Cherri binaries are not currently published.
On macOS, if you have an unsigned `.shortcut`, a signed artifact can be created
with:

```bash
shortcuts sign --mode anyone \
  --input "dist/Agent Router.shortcut" \
  --output "dist/Agent Router.signed.shortcut"
```

The repository includes GitHub Actions workflows that install Cherri, compile the
shortcut, inspect whether it is already package-formatted, attempt Apple signing
when needed, and upload artifacts/logs.

## Validation

Use the simulator to exercise the real agent loop without an iPhone. It sends
the exact protocol and per-turn message the shortcut sends to the live NVIDIA
NIM endpoint, replicates the JSON guard, budgets, dispatch, and delivery, mocks
the native tool outputs, and asserts on route selection (e.g. "list my
calendars" must call `calendar_lookup`, "say hi" must not ask a question):

```bash
NIM_API_KEY=nvapi-... python scripts/simulate-agent.py            # run + assert scenarios
NIM_API_KEY=nvapi-... python scripts/simulate-agent.py --once "what's the weather" -v
python scripts/simulate-agent.py --mock --model-reply '{"type":"final_answer","answer":"hi"}'
```

It exits nonzero if any assertion fails, so it doubles as a regression test for
the planner prompt.

The CI workflow compiles `shortcuts/agent_router.cherri` on Linux. The build
workflow compiles on macOS, attempts `shortcuts sign`, and uploads the unsigned
shortcut, signed shortcut if available, and signing logs.

Manual iPhone validation is still required. See `docs/device-testing.md`.
Before changing the agent loop or Shortcut control flow, read
`docs/shortcut-runtime-flow.md`; it records the actual Shortcuts runtime model,
known ChatGPT App Intent session risks, and loop failure modes found on device.

## Security Model

Do not commit API keys, personal prompts, webhook URLs, contacts, or private
automation details.

The shortcut sends your typed, dictated, or shared text to the NVIDIA NIM model
provider, and (when web search is enabled) your query to Tavily. Do not route
sensitive personal data unless you understand and accept those providers'
privacy terms.

See `docs/security.md` and `docs/configuration.md`.

## Project Status

This is an alpha proof of concept. The Shortcut source now lives in
`shortcuts/agent_router.cherri`. Windows support depends on Cherri compiler
availability; if local Windows compilation is rough, use the GitHub Actions
workflow as the canonical build path.

## License

MIT. See `LICENSE`.
