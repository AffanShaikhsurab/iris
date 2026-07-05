# Shortcut Catalog Candidates

This file curates the best reusable patterns from Matthew Cassinelli-style
shortcut catalogs for Iris. The goal is not to copy every shortcut. The
goal is to identify high-value routes that are common, safe, and realistic to
build with Apple Shortcuts.

## Selection Rules

- Prefer routes that many users ask Siri for naturally.
- Prefer built-in Apple actions over fragile third-party app intents.
- Prefer "open/draft/show/ask for confirmation" over silent mutation.
- Prefer actions that can return data to the model for summarization.
- Exclude niche account-specific links unless they demonstrate a reusable
  pattern.

## Best First-Class Route Candidates

| Candidate | User request examples | Why it matters | Implementation shape | Status |
| --- | --- | --- | --- | --- |
| Web/current answer | "What are today's FIFA matches?", "What's happening in AI today?" | This is the most common Siri replacement use case. | `web_search` (Tavily) for current facts, `answer_search` (model knowledge) otherwise. | Implemented. |
| Summarize shared text or URL | "Summarize this article/email/page." | Works around private app limits by using explicit user-provided input. | Prompt input -> model -> speak/show. | Implemented for provided text; share-sheet input is future work. |
| Draft reply | "Draft a polite reply to this." | Useful and safe because it does not send automatically. | Model draft -> show draft. | Implemented. |
| Draft email | "Draft an email to Alex about tomorrow." | High-value productivity route with built-in Mail support. | Select contact -> Mail compose/draft sheet -> user reviews. | Planned (not yet implemented). |
| Calendar summary | "What's next on my calendar?", "Summarize today's meetings." | A core assistant workflow with built-in Calendar actions. | Get upcoming events -> model loopback summary -> speak/show. | Implemented as experimental. |
| Create reminder | "Remind me to call Mom." | Common Siri task, low risk, built-in Reminders support. | Add New Reminder with title/notes. | Implemented as experimental. |
| Create note | "Make a note about this idea." | Common capture workflow. | Create Note with body/title; show compose where possible. | Implemented as experimental. |
| Search web/app | "Search YouTube for X", "Search Google for X." | Useful launcher pattern for opening a search in a target site/app. | Encode query -> open URL/app deep link. | Implemented (`open_search`). |
| Open app destination | "Open Notion AI", "Open Perplexity", "Open ChatGPT." | Common launcher pattern from shortcut catalogs. | Route to `openURL(...)` or app bundle/deep link. | Implemented (`open_destination`; `chatgpt`, `perplexity`, `calendar`). |
| Messages conversation | "Open chat with Mom", "Open family thread." | Useful, but contact and app-intent support varies. | Open conversation or draft message; never auto-send. | Future route after device validation. |

## Strong Expansion Patterns

### Search And Open

These are catalog shortcuts like Search Google, YouTube Search, Search Reddit,
Search Amazon, Search X, Search Bluesky, Search Fandango, and Open Perplexity.

Iris should implement this as one generic route:

```json
{
  "action": "open_search",
  "target": "youtube",
  "query": "best WWDC shortcuts"
}
```

The shortcut can map approved targets to URL templates. This is safer than
letting the model invent arbitrary URLs.

### Open Known Destination

Catalog examples include Open Notion AI, Open Perplexity, Open Google Lens,
Open Health Summary, Open App Store Search, Open YouTube Subscriptions, Open
Reminders Today, and Open Calendar/Fantastical views.

Iris should implement this as a registry:

```json
{
  "action": "open_destination",
  "target": "notion_ai"
}
```

Each destination needs a documented URL scheme or universal link before being
added.

### Capture

Catalog examples include Create voice memo, Activate Monologue, Start Recording
Note, Log journal, Create note, Email myself, and Add reminder.

Best Iris route family:

- `create_note`
- `create_reminder`
- `draft_email`
- `voice_memo` after action support is verified

### Calendar And Meetings

Catalog examples include Join my next meeting, Show today's agenda, Show
tomorrow's schedule, Change calendar set, and Leave for theater.

Best Iris route family:

- `calendar_lookup`
- `join_next_meeting`
- `travel_to_event`

Only add `join_next_meeting` after we can reliably extract URLs from calendar
events on the target device.

### Media And Podcasts

Catalog examples include play a podcast, play a station, play Replay playlist,
change playback destination, and skip/seek playback.

These are useful but not core to the first productivity assistant. Add after the
productivity routes are stable.

### Device And Settings

Catalog examples include Set Focus Mode, Toggle Personal Hotspot, Toggle Silent
Mode, Translate, Sound Recognition, Background Sounds, and Control Center.

These can be powerful, but some are device/iOS-version-specific. Treat them as a
separate route pack after device testing.

## Routes To Avoid For Now

- Account-specific website dashboards such as WordPress admin pages, personal
  X lists, Notion workspace pages, or subscription settings.
- Purchase/payment/account routes like Amazon cart, bank accounts, QuickBooks,
  Wallet cards, or subscription management.
- Smart-home routes that affect physical devices until there is a confirmation
  layer.
- Social posting/cross-posting routes until drafts and user confirmation are
  reliable.
- Gmail inbox summarization through native Shortcuts. Gmail support is too
  limited for that route.

## Recommended Next Route Pack

After the current Iris imports and runs successfully on iPhone, add only
these routes next:

1. `open_search`: Google, YouTube, Perplexity, X, Reddit.
2. `open_destination`: ChatGPT, Perplexity, Gemini, Notion AI, Health Summary,
   Reminders Today.
3. `join_next_meeting`: only if Calendar event URL extraction works on device.
4. `voice_memo`: only if Voice Memos action identifiers import and run.
5. `message_draft`: draft-only Messages route with user confirmation.
