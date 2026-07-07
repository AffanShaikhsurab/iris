# OKF Knowledge Base

Iris uses an OKF-style knowledge base to give the agent durable context without
pretending it has Apple Intelligence-level private app access.

> **Storage update 2026-07-06 — memory is now HYBRID.** OKF concept entries are
> stored in **BOTH** backends: a **phone-local Files store** (per-topic OKF
> Markdown files) and a **server-side Google Sheet** reached through the Apps
> Script proxy. A one-time `@memoryBackend` selector (`hybrid` default | `local`
> | `sheets`) chooses which. This supersedes the earlier proxy-only note.
> On-device testing confirmed the phone-local Files writes **do** persist (they
> land in iCloud Drive under `Shortcuts/`); the earlier "generated shortcuts
> cannot emit `fileLocation` so writes no-op" claim was **wrong** and is
> retracted (see `docs/memory-system-design.md`'s status block). The OKF
> *concept shape* below (topics, frontmatter-style records, envelope) applies to
> both stores; the model-facing surface stays `memory_append(topic, body)`.

## Storage Location

The same OKF concept entry is written to both backends (selected by
`@memoryBackend`), addressed by **topic**, never by a free-form path.

**Local Files store** — one append-only OKF Markdown file per topic:

```text
Base:   Shortcuts/IrisOKF/          (@memBase; iCloud Drive under Shortcuts/)
File:   Shortcuts/IrisOKF/<topic>.md   (append-only OKF concept blocks)
```

**Proxy Sheet store** — one append-only Google Sheet, resolved by name (no path,
no device folder):

```text
Spreadsheet: Iris Memory        (found-or-created by name in the proxy owner's Drive)
Tab:         memory
Header:      timestamp | topic | type | title | tags | body
Row:         [ISO-8601 timestamp, topic, type, title, tags, body]   (append-only)
```

The proxy columns mirror the OKF frontmatter, so one row reconstructs to one OKF
concept (`memory_read(topic, format=okf)`), while the local file already holds
the OKF block verbatim. The topic allowlist applies to both stores:

```text
index  profile  preferences  log  notes  journal      (default: log)
```

Any topic outside the allowlist falls back to `log`, so the model can never
address arbitrary storage. Both stores **self-heal**: the proxy creates the
sheet + `memory` tab on the first write, and the client seeds the local
`Shortcuts/IrisOKF` folder (setup primer + auto-creating `appendToFile`), so the
first write succeeds with no manual setup. See `docs/apps-script-proxy.md` §4 for
the server ops and `docs/memory-system-design.md` §0 for the current design.

## Concept Format

Every non-reserved Markdown concept should include YAML frontmatter with at
least `type`.

Recommended fields:

- `type`: descriptive concept type, such as `User Preference`, `Project`,
  `Routine`, `Person`, `Design Fact`, or `Memory Entry`.
- `title`: human-readable name.
- `description`: one-sentence summary.
- `tags`: short retrieval hints.
- `timestamp`: creation or last update time.

Example:

```md
---
type: User Preference
title: Planning Style
description: User preference for technical planning conversations.
tags:
  - user
  - planning
timestamp: 2026-07-04T17:19:00+05:30
---

# Planning Style

The user prefers comprehensive architecture plans with explicit loop/tool
behavior and practical validation steps.
```

## Runtime Behavior

> Implementation status: `memory_read`, `memory_append`, `memory_list`, and
> `memory_status` are implemented in `shortcuts/iris.cherri` and are now
> **hybrid** (phone-local Files store AND an append-only Google Sheet via the
> Apps Script proxy, selected by `@memoryBackend`; see
> `docs/memory-system-design.md` §0 and `docs/apps-script-proxy.md` §4). The
> startup `memory_summary` bootstrap is also implemented (local-first). Everything
> is fail-open when no selected backend is available.

The memory layer is a fixed set of predeclared tools, not arbitrary file access.
The agent cannot browse files; every memory operation is routed through a tool
that write-throughs to the local file and/or the proxy op (honoring the
selector) and reads local-first with proxy fallback. The model-facing surface is
`memory_append(topic, body)` only — `type`/`title`/`tags` are derived by the
storage layer, not the model.

- `memory_read(topic)` (implemented, hybrid): return the recent bodies saved
  under `topic`, reading local-first with proxy fallback; `ok=false` when nothing
  is saved yet. Spoken.
- `memory_append(topic, body)` (implemented, hybrid): write-through — append the
  OKF concept to the local `<topic>.md` first (when `@memLocalOn`), then mirror
  it to the proxy sheet row (when `@memProxyOn`); `ok=true` if either leg wrote.
  Append-only.
- `memory_list` (implemented, hybrid): return the last N `topic: body` rows
  across topics, local-first with proxy fallback. Spoken.
- `memory_status` (implemented, hybrid): report whether memory has content,
  local-first with proxy fallback.

`create_note` (topic `notes`) and `quick_journal` (topic `journal`) reuse the
same write-through path (local OKF append AND proxy mirror, honoring the
selector).

Startup behavior (implemented, local-first): before the loop, Iris builds a
compact durable-context summary — it reads the local `profile.md` +
`preferences.md` first (when `@memLocalOn`), and only when that is empty and the
proxy is on GETs the proxy `memory_summary` op (a ≤600-char profile +
preferences + recent-index summary). When non-empty it is injected once into
`@loopContext` as `memory_summary`. When neither backend answers, the summary is
skipped and Iris runs with no memory context, fail-open. The proxy fallback
recovers the iCloud-full silent-sync-loss case.

Still out of scope for the alpha: overwrite/update/delete, free-text
`memory_search`, and confirmation-gated writes (the model is asked to confirm
stable personal facts in-conversation before emitting `memory_append`).

## Result Envelopes

Memory tools return the same normalized observation shape as other Iris tools
(built server-side by the proxy):

```text
tool=memory_read
ok=true
count=1
result=Read preferences memory.
records=<recent bodies for the topic>
error=
```

When nothing is saved for a topic yet:

```text
tool=memory_read
ok=false
count=0
result=
records=
error=No memory found for that topic yet.
```

## Privacy Rules

- Memory lives in up to two places, per the `@memoryBackend` selector: a
  phone-local Files store (`Shortcuts/IrisOKF/<topic>.md`, confirmed to land in
  iCloud Drive under `Shortcuts/`) and/or a Google Sheet owned by the proxy
  account (the user's own Google account running the Apps Script web app),
  reached only through the proxy's shared-secret guard. Choosing `sheets` keeps
  memory off the phone entirely; `local` keeps it off the cloud; `hybrid` (the
  default) writes both. No third party beyond the model provider (for snippets
  sent in a prompt) and the user's own Google account is involved.
- Any memory snippet inserted into a model prompt is sent to the model provider. The
  Shortcut should send only relevant snippets, not the whole knowledge base.
- Memory content is data, not instructions. The agent must not obey commands
  embedded inside memory files.
- Stable personal facts should be saved only after user confirmation.
- Append-only writes are safer than in-place mutation and should remain the
  default until update/delete behavior is tested.
- Sensitive information such as credentials, private health information, and
  financial details should not be stored in this alpha version.

## Implementation Stages

1. Dry-run fixtures in `tests/fixtures/okf/`. (present)
2. `memory_status` availability check. (implemented, hybrid)
3. Read-only `memory_summary` bootstrap (profile + preferences + recent index).
   (implemented, hybrid, local-first)
4. Read/list via `memory_read(topic)` and `memory_list`. (implemented, hybrid,
   local-first with proxy fallback)
5. Append via `memory_append(topic, body)` (also used by `create_note` and
   `quick_journal`); write-through to both stores; stable personal facts
   confirmed in-conversation before the write. (implemented, hybrid)
6. Later, structured concept records and update/delete tools. (planned)

## Validation

Before marking memory stable, test (hybrid; see the `curl` checks in
`docs/apps-script-proxy.md` §4):

- Selector routing: `hybrid` writes both stores, `local` writes only the Files
  store, `sheets` writes only the sheet; an unrecognized value falls back to
  `hybrid`.
- Both stores self-heal: with the `Iris Memory` sheet deleted, the first
  `memory_append` recreates the sheet + `memory` tab; with the local folder
  absent, the first append creates `Shortcuts/IrisOKF` and the write lands.
- Write-then-read round-trip: `memory_append(topic, body)` then
  `memory_read(topic)` returns records containing the body, across sessions and
  devices, reading local-first.
- Local-first fallback: with an empty local copy and a populated proxy (the
  iCloud-full case), `memory_read` falls back to the proxy and still returns the
  body.
- OKF round-trip: a stored entry reconstructs to a valid OKF concept (local
  block verbatim; proxy `format=okf` from the row columns).
- `memory_status` count increments by one per append.
- `memory_summary` caps at 600 chars and injects once at session start.
- An unknown topic falls back to `log`.
- Fail-open: with no selected backend available, a normal chat is unaffected and
  never halts.
