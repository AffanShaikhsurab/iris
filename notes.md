---
type: Technical Note
title: Working ChatGPT App Intent for Agent Router
summary: Phone-created Shortcut export confirms the native ChatGPT AskIntent shape supported on the current test device.
status: validated-on-device
tags:
  - apple-shortcuts
  - chatgpt
  - app-intents
  - cherri
  - agent-router
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

Agent Router should call `askChatGPTApp(...)` anywhere it needs ChatGPT app
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
