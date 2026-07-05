---
type: Reference
title: Phone-Created Shortcut Syntax Reference
summary: Captures exact plist action identifiers, App Intent descriptors, and token attachment shapes from a broad shortcut created on the test iPhone.
status: decoded-reference
tags:
  - apple-shortcuts
  - syntax-reference
  - app-intents
  - action-identifiers
  - agent-router
sources:
  - https://www.icloud.com/shortcuts/feabeb61fb5a43609b265beef85f1df4
  - tmp/icloud-shortcut-feabeb61fb5a43609b265beef85f1df4.plist.json
  - tmp/icloud-shortcut-feabeb61fb5a43609b265beef85f1df4-meta.json
related:
  - notes.md
  - agentic-loop.md
  - docs/routes.md
---

# Phone-Created Shortcut Syntax Reference

This note records the exact syntax emitted by the Shortcuts app on the current
test iPhone for a broad, manually-created reference shortcut. The shortcut was
intentionally assembled in a rough order to collect useful actions and supported
app intents. Treat it as a syntax fixture, not as a polished workflow.

The exported shortcut is named `New Shortcut 5`, has 40 actions, and decodes as
a normal unsigned Shortcut plist. The decoded fixture is saved at
`tmp/icloud-shortcut-feabeb61fb5a43609b265beef85f1df4.plist.json`.

## How Future Agents Should Use This

- Prefer action identifiers and parameter keys observed in this file over guessed
  names from Cherri, blogs, or model memory.
- For third-party App Intents, use a real Cherri custom action with
  `action 'identifier' name(...)` so the compiled shortcut gets the correct
  `WFWorkflowActionIdentifier`.
- Do not use Cherri `rawAction(...)` for these App Intent identifiers unless the
  compiled plist is inspected afterward. Previous builds showed `rawAction(...)`
  can lower to `is.workflow.actions.rawaction`, which iOS treats as unsupported.
- Keep the manually decoded plist in `tmp/` as the source of truth when adding
  new tool actions to Agent Router.
- Re-test on device after adding any action that mutates data, opens private app
  state, or depends on a third-party app being installed.

## Action Identifier Catalog

The reference contains these actions in order:

| Area | Action identifier | Notes |
| --- | --- | --- |
| Gmail/Mail-style send | `is.workflow.actions.sendmessage` | Can include `IntentAppDefinition` for `com.google.Gmail`. |
| Google Maps | `com.google.Maps.RestaurantsIntent` | Direct App Intent action. |
| Google Maps | `com.google.Maps.SearchIntent` | Direct App Intent action. |
| Messages | `com.apple.MobileSMS.MessageEntity` | Direct App Intent/entity action for selecting a message. |
| Messages | `is.workflow.actions.sendmessage` | Built-in send message action with Messages descriptor. |
| WhatsApp | `net.whatsapp.WhatsApp.send` | Direct WhatsApp send action, minimal parameters in fixture. |
| Reminders | `com.apple.reminders.TTRSearchRemindersAppIntent` | Direct App Intent action. |
| Reminders | `com.apple.reminders.TTRCreateListAppIntent` | Direct App Intent action. |
| Reminders | `is.workflow.actions.properties.reminders` | Built-in reminder properties action. |
| Reminders | `is.workflow.actions.getupcomingreminders` | Built-in upcoming reminders action. |
| Reminders | `is.workflow.actions.filter.reminders` | Built-in reminder filter action. |
| WhatsApp Meta AI | `net.whatsapp.WhatsApp.OpenMetaAIIntent` | Direct App Intent action. |
| WhatsApp Meta AI | `net.whatsapp.WhatsApp.EndMetaAICallIntent` | Direct App Intent action. |
| Weather | `is.workflow.actions.weather.currentconditions` | Built-in current weather action. |
| Weather | `is.workflow.actions.weather.forecast` | Built-in weather forecast action. |
| Notes | `is.workflow.actions.filter.notes` | Built-in filter action with Notes entity descriptor. |
| Notes | `is.workflow.actions.appendnote` | Built-in append-to-note action with Notes descriptor. |
| Photos | `com.apple.mobileslideshow.PhotosSearchAssistantIntent` | Direct Photos search assistant App Intent. |
| Web | `is.workflow.actions.searchweb` | Built-in web search action. |
| Safari/Web Page | `is.workflow.actions.runjavascriptonwebpage` | Built-in JavaScript on web page action. |
| Safari/Web Page | `is.workflow.actions.getwebpagecontents` | Built-in page contents action. |
| Chrome | `com.google.chrome.ios.SearchInChromeIntent` | Direct Chrome App Intent with `searchPhrase`. |
| Chrome | `com.google.chrome.ios.OpenLensIntent` | Direct Chrome Lens App Intent. |
| Files | `is.workflow.actions.file.getfoldercontents` | Built-in folder contents action. |
| Device | `is.workflow.actions.getdevicedetails` | Built-in device details action. |
| Location | `is.workflow.actions.getcurrentlocation` | Built-in current location action. |
| Network | `is.workflow.actions.downloadurl` | Built-in HTTP/download action. |
| Control Flow | `is.workflow.actions.repeat.count` | Repeat start/end use the same identifier and grouping ID. |
| Shortcuts | `is.workflow.actions.getmyworkflows` | Built-in get shortcuts action. |
| Shortcuts | `is.workflow.actions.runworkflow` | Built-in run shortcut action. |
| SSH | `is.workflow.actions.runsshscript` | Built-in SSH script action. |
| URL Schemes | `is.workflow.actions.openxcallbackurl` | Built-in x-callback-url action. |
| Data | `is.workflow.actions.list` | Built-in list action. |
| Data | `is.workflow.actions.setstoredcontent` | Built-in stored content setter. |
| Data | `is.workflow.actions.getstoredcontent` | Built-in stored content getter. |
| Data | `is.workflow.actions.deletestoredcontent` | Built-in stored content delete action. |

## App Intent Descriptor Shapes

Direct App Intent actions keep their app identity in `AppIntentDescriptor`. This
is the pattern to copy into Cherri custom action definitions:

```json
{
  "WFWorkflowActionIdentifier": "com.google.Maps.SearchIntent",
  "WFWorkflowActionParameters": {
    "UUID": "840F5856-9CDE-415F-B639-488278682F90",
    "AppIntentDescriptor": {
      "TeamIdentifier": "EQHXZ8M8AV",
      "BundleIdentifier": "com.google.Maps",
      "Name": "Google Maps",
      "AppIntentIdentifier": "SearchIntent"
    }
  }
}
```

Apple first-party App Intents use `TeamIdentifier: "0000000000"` in this export:

```json
{
  "WFWorkflowActionIdentifier": "com.apple.reminders.TTRSearchRemindersAppIntent",
  "WFWorkflowActionParameters": {
    "UUID": "29E10ECA-AFE9-4DCB-AD51-215F78557B57",
    "AppIntentDescriptor": {
      "TeamIdentifier": "0000000000",
      "BundleIdentifier": "com.apple.reminders",
      "Name": "Reminders",
      "AppIntentIdentifier": "TTRSearchRemindersAppIntent"
    }
  }
}
```

Some built-in actions also include an `AppIntentDescriptor`, especially when they
operate on app entities:

```json
{
  "WFWorkflowActionIdentifier": "is.workflow.actions.appendnote",
  "WFWorkflowActionParameters": {
    "AppIntentDescriptor": {
      "TeamIdentifier": "0000000000",
      "BundleIdentifier": "com.apple.mobilenotes",
      "Name": "Notes",
      "AppIntentIdentifier": "AppendToNoteLinkAction"
    }
  }
}
```

## Intent App Definition Shape

`is.workflow.actions.sendmessage` can be specialized to a target app with
`IntentAppDefinition`. The Gmail fixture only includes the app definition and no
message fields:

```json
{
  "WFWorkflowActionIdentifier": "is.workflow.actions.sendmessage",
  "WFWorkflowActionParameters": {
    "IntentAppDefinition": {
      "BundleIdentifier": "com.google.Gmail",
      "Name": "Gmail",
      "TeamIdentifier": "EQHXZ8M8AV"
    }
  }
}
```

The Messages version combines `IntentAppDefinition`, `AppIntentDescriptor`, and
text-token input:

```json
{
  "WFWorkflowActionIdentifier": "is.workflow.actions.sendmessage",
  "WFWorkflowActionParameters": {
    "IntentAppDefinition": {
      "BundleIdentifier": "com.apple.MobileSMS"
    },
    "AppIntentDescriptor": {
      "TeamIdentifier": "0000000000",
      "BundleIdentifier": "com.apple.MobileSMS",
      "Name": "Messages",
      "AppIntentIdentifier": "SendMessageIntent"
    }
  }
}
```

## Token Attachment Patterns

Shortcuts represents connections between actions with text tokens and action
output attachments. This fixture has both `WFTextTokenString` and
`WFTextTokenAttachment` forms.

String-token input, used by Messages, Notes, Web Page Contents, Chrome search,
stored content, and x-callback URL:

```json
{
  "WFInput": {
    "Value": {
      "string": "\ufffc",
      "attachmentsByRange": {
        "{0, 1}": {
          "OutputUUID": "92D2A4F7-5991-4891-BC67-155881D173A7",
          "Type": "ActionOutput",
          "OutputName": "Note"
        }
      }
    },
    "WFSerializationType": "WFTextTokenString"
  }
}
```

Direct attachment input, used by folder contents, run shortcut, and SSH script:

```json
{
  "WFInput": {
    "Value": {
      "OutputUUID": "85A14EE9-A12D-4BF7-AF6D-9208B88764B9",
      "Type": "ActionOutput",
      "OutputName": "My Shortcuts"
    },
    "WFSerializationType": "WFTextTokenAttachment"
  }
}
```

## Control Flow Pattern

Repeat blocks use `is.workflow.actions.repeat.count` for both the opening and
closing actions. The opening action has `WFControlFlowMode: 0`; the closing
action has `WFControlFlowMode: 2`. Both share the same `GroupingIdentifier`.

```json
{
  "WFWorkflowActionIdentifier": "is.workflow.actions.repeat.count",
  "WFWorkflowActionParameters": {
    "GroupingIdentifier": "6651F8FD-D89E-42B2-8D1E-FEEFE1E3CD26",
    "WFControlFlowMode": 0
  }
}
```

```json
{
  "WFWorkflowActionIdentifier": "is.workflow.actions.repeat.count",
  "WFWorkflowActionParameters": {
    "WFControlFlowMode": 2,
    "GroupingIdentifier": "6651F8FD-D89E-42B2-8D1E-FEEFE1E3CD26",
    "UUID": "7224ED8D-9B0F-4808-AC12-8E30286F4863"
  }
}
```

## Agent Router Implications

The most useful near-term additions for Agent Router are:

- `open_search` and `open_destination`, using known URL schemes or proven direct
  App Intent identifiers like Chrome Lens, Chrome search, Google Maps search,
  and WhatsApp Meta AI.
- `notes_lookup` and `append_note`, using the observed Notes descriptors and
  token input shapes.
- `reminders_lookup`, using `TTRSearchRemindersAppIntent`,
  `getupcomingreminders`, and `filter.reminders`.
- `web_page_summary`, using `runjavascriptonwebpage` and
  `getwebpagecontents` only when the shortcut is run from Safari/share-sheet
  context.
- `run_shortcut_tool`, using `getmyworkflows` and `runworkflow` only with a
  fixed allowlist, never with arbitrary model-selected shortcut names.

Actions that should remain experimental until device-tested:

- Gmail via `is.workflow.actions.sendmessage` with `IntentAppDefinition`.
- WhatsApp send and Meta AI actions.
- Photos search assistant.
- SSH script and x-callback URL, because they can become high-risk execution
  tools if exposed directly to the model.
