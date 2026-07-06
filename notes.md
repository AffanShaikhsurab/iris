---
type: Technical Note
title: Working ChatGPT App Intent for Iris
summary: Phone-created Shortcut export confirms the native ChatGPT AskIntent shape supported on the current test device.
status: validated-on-device
tags:
  - apple-shortcuts
  - chatgpt
  - app-intents
  - cherri
  - iris
sources:
  - https://www.icloud.com/shortcuts/6b92d6285d1c4d1bad8462747b4e6e04
  - https://github.com/GoogleCloudPlatform/knowledge-catalog/tree/main/okf
---

# Working ChatGPT App Intent

The current test phone successfully supports the ChatGPT app's native Shortcuts
action. A Shortcut created directly on the phone and shared through iCloud was
downloaded and decoded as the repo's reference implementation.

The exported shortcut is named `New Shortcut 4` and contains two actions:

- `com.openai.chat.AskIntent`
- `com.openai.chat.OpenNewChatInAppShortcutIntent`

## Confirmed AskIntent Shape

The supported `Ask ChatGPT` action is a direct App Intent action. It does not
use Apple Intelligence, Apple's `Use Model` action, or the App Intent execution
wrapper.

```json
{
  "WFWorkflowActionIdentifier": "com.openai.chat.AskIntent",
  "WFWorkflowActionParameters": {
    "AppIntentDescriptor": {
      "TeamIdentifier": "2DC432GLL2",
      "BundleIdentifier": "com.openai.chat",
      "Name": "ChatGPT",
      "AppIntentIdentifier": "AskIntent"
    },
    "prompt": {
      "Value": {
        "string": "\ufffc",
        "attachmentsByRange": {
          "{0, 1}": {
            "Type": "Ask"
          }
        }
      },
      "WFSerializationType": "WFTextTokenString"
    },
    "UUID": "1CAA9AB8-B268-4FC0-A50C-6B5C0580BF3E"
  }
}
```

## Implementation Decision

`rawAction("com.openai.chat.AskIntent", ...)` in Cherri compiled into
`is.workflow.actions.rawaction`, which iPhone treated as an unsupported action.
The fix is to define a real Cherri custom action for ChatGPT:

```cherri
action 'com.openai.chat.AskIntent' askChatGPTApp(text prompt: 'prompt'): text {
  "AppIntentDescriptor": {
    "TeamIdentifier": "2DC432GLL2",
    "BundleIdentifier": "com.openai.chat",
    "Name": "ChatGPT",
    "AppIntentIdentifier": "AskIntent"
  },
  "ShowWhenRun": false
}
```

Iris should call `askChatGPTApp(...)` anywhere it needs ChatGPT app
output.

## Constraints

- This route requires the ChatGPT iOS app to be installed.
- The user must be signed in to ChatGPT.
- `Ask ChatGPT` must appear in the Shortcuts action picker.
- It does not require an OpenAI, OpenRouter, NVIDIA NIM, or other API key.
- It does not provide Apple Intelligence-level private context access to Mail,
  Gmail, Messages, the screen, or arbitrary apps.

## Follow-Up Risk

The phone export represents `AppIntentDescriptor` as a plain plist dictionary.
Cherri custom action definitions currently fix the top-level action identifier,
but nested dictionary serialization should be verified on-device after every
compiler/build change. If the action regresses, add a post-build plist patcher
that rewrites ChatGPT actions to exactly match the phone-exported plist shape
before signing.


---

# Status 2026-07-06: Google Apps Script proxy confirmed WORKING end-to-end

Google integration (Tasks + Gmail + Calendar) via the Apps Script proxy is now
verified working on device and from direct API tests. Recorded here so the
working configuration and the fixes that got us there are not lost.

## What works
- `tasks_list` returns real tasks (verified: 8 open tasks, clean titles).
- Gmail verified: `gmail_search` (returns real emails), `gmail_read`,
  `draft_email` (draft created), and `send_email` confirm-gate (refuses without
  `confirm=yes`).
- Calendar routed through the proxy (`calendar_list` / `calendar_add`) plus a
  local `getUpcomingEvents` fallback.

## The configuration that made it work
1. **GET, not POST.** Apps Script `/exec` POST 302-redirects to
   `script.googleusercontent.com`; iOS follows it as a bodyless GET and drops
   the JSON body -> proxy errors -> model hallucinates "need permission". Fix:
   the shortcut calls the proxy with GET + query params (secret, op, args),
   which survive the redirect. Proxy has a shared `handle()` used by both
   `doGet(e.parameter)` and `doPost`.
2. **Deploy a NEW VERSION.** Editing/saving Code.gs is not enough; `/exec`
   serves the last *deployed* version. Deploy > Manage deployments > New version.
   Symptom of a stale deploy: `/exec?...&op=tasks_list` returns the health
   message "Iris proxy is deployed." instead of data.
3. **Secret must match.** `IRIS_PROXY_SECRET` (baked into the shortcut from
   `.env.local`) must equal `SHARED_SECRET` in the deployed Code.gs. It is NOT
   the deployment id from the URL. Secret was rotated to a strong random value.
4. **Use the `/exec` URL, never `/dev`.** `/dev` requires a Google login (returns
   a sign-in page) and cannot be called by Shortcuts.

## Read/list answers now speak the actual items
Read/list tools (tasks_list, gmail_search, gmail_read, calendar_lookup,
reminders_lookup, memory_read, memory_list) deliver their `records` directly
instead of returning to the planner, which used to over-compress them into a
bare count. Proxy record formats made voice-friendly (task titles only; Gmail
"From Sender: Subject").

## Files
- `shortcuts/iris.cherri` (GET-based proxy calls, speak-records delivery)
- `docs/apps-script-proxy.md` §4 (full current Code.gs with doGet dispatch +
  calendar handlers)
- `.env.local` (IRIS_PROXY_URL + rotated IRIS_PROXY_SECRET; gitignored)
