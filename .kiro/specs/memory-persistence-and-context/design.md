# Memory Persistence and Context Bugfix Design

## Overview

Iris is a hands-free voice agent compiled from `shortcuts/iris.cherri` by Cherri,
with a stateless NVIDIA NIM model as the planner. Four related defects make it
unreliable: memory never persists, in-conversation context is lost, the planner
model is weak-or-slow, and compaction fires on the wrong signal. This design
fixes all four with the smallest change that is provably correct and hands-free.

The core strategy for memory is a **HYBRID, user-selectable backend** that
writes OKF-formatted entries to a phone-local Files store, a server-side Google
Sheet (via the Apps Script proxy), or both. This **supersedes the earlier
proxy-only Decision 1** (see the marked note in Decision 1 below).

> **Root-cause correction (on-device testing).** The earlier design claimed
> Cherri's built-in `appendToFile`/`getFile` "silently no-op" because they emit
> a bare `WFFilePath` with no `fileLocation` object. **That conclusion was wrong
> — on-device testing refuted it.** With a fixed text path, the built-in
> `appendToFile`/`getFile`/`createFolder` **do** store data — the file landed in
> **iCloud Drive under the `Shortcuts` folder**. The writes were never lost. The
> two *real* prior failures were: **(a) the parent folder was not created before
> the first `appendToFile`** (a first write into a missing folder produced
> nothing), and **(b) the user was checking the wrong Files location** (looking
> under On My iPhone / a different folder than where the data actually landed).
> Neither is "an impossible location object"; the narrative below is corrected
> accordingly. So phone-local Files is a viable backend after all.
>
> **Where writes land is still partly UNVERIFIED, so we probe it.** The confirmed
> behavior is that writes land in **iCloud Drive/Shortcuts**. For true
> iCloud-quota immunity the *ideal* local target is **On My iPhone (Local
> Storage)** (the reference plist pins `WFFileLocationType = "LocalStorage"`), but
> whether Cherri's built-in file actions — which emit only a text `WFFilePath`
> and no `fileLocation` object — can pin the service to On My iPhone rather than
> iCloud Drive is **not verified**. The design therefore adds an explicit
> **on-device probe** (write a marker, then confirm where it lands) and
> reconciles `@memBase` to the confirmed location before shipping. See the probe
> in Decision 1c and the Testing Strategy.
>
> The remaining *real* risk for the phone-local leg is the **iCloud Drive full /
> not paid** case: when writes land in iCloud Drive, a local write can silently
> fail to sync, which breaks cross-device access and iCloud backup even though the
> write appeared to succeed. (If the probe confirms writes can be pinned to On My
> iPhone / Local Storage, this risk drops to device-loss only.) Either way the
> proxy (Google Sheet) is immune to iCloud quota and gives durability plus
> cross-device recall, so the hybrid keeps a proxy mirror regardless of the probe
> outcome.

Because neither risk can be *detected at runtime* (Shortcuts has no try/catch and
file actions return no error value), the hybrid is designed to be safe **without
error-detection**:

- **Write = write-through** to every backend that is available/configured: append
  the OKF entry to the local Files store **and** POST/GET it to the proxy. If
  only one backend is available, write only that one.
- **Read / bootstrap = local-first, proxy fallback**: read the local file; if it
  is empty or missing, read the proxy. This makes the iCloud-full case safe —
  even if the local copy silently failed to sync, the proxy mirror exists and the
  read falls back to it.
- A reliable **backend selector** (`@memoryBackend`, an editable Text config
  action, values `hybrid`/`local`/`sheets`) lets the user force a single backend;
  `hybrid` (default) does both as above, `local` uses only phone-local Files, and
  `sheets` uses only the Google Sheet proxy. It is a **one-time config toggle**
  (a Text action edited once in the Shortcuts editor, exactly like the API key),
  **not** a per-run spoken prompt — asking every run would break hands-free use,
  add latency, and cannot persist a choice. If neither backend is available,
  memory fails open (no memory), exactly as today (Req 3.6).

The proxy path reuses a code path already wired, network-primed, and
envelope-shaped in `iris.cherri`; the local path reuses Cherri's built-in Files
actions now that they are confirmed working. Both stores hold the **same
OKF-formatted concept entries** (`docs/okf-knowledge-base.md`), so recall is
uniform regardless of which backend answers.

The remaining three fixes are in-shortcut: make the planner answer the user's
**most recent** message (with full history retained), compact on **estimated
token usage** instead of a fixed exchange count while preserving named entities,
and switch the default model to a **capable, non-reasoning, latency-safe** NIM
model chosen from the current catalog.

All fixes respect the Cherri constraints in the file header: no `else if` (flat
guarded `if`s / default-then-override), `jsonRequest`/literal-dict bodies,
never `getDictionary()` on non-JSON, coerce JSON with `getDictionary()` +
`getValue()` per level, and keep the protocol compact and every call fast.

## Glossary

- **Bug_Condition (C)**: The condition that triggers a defect — the agent is
  asked to persist/recall memory, or the user sends a follow-up referencing an
  earlier turn, or the running context grows past the model's usable window, or
  the planner model is a weak/slow/reasoning model.
- **Property (P)**: The desired behavior — memory is durably written and later
  recalled; the planner answers the most recent message with history preserved;
  compaction triggers on estimated token usage and keeps follow-up entities;
  the planner is capable, non-reasoning, and finishes within the ~25s budget.
- **Preservation**: All non-triggering inputs (hands-free happy path, flat-JSON
  parsing, non-memory tools, delivery pattern, budget resets) behave exactly as
  today.
- **Proxy**: The Google Apps Script web app (`docs/apps-script-proxy.md`) Iris
  calls via GET with query params (`secret`, `op`, args), which returns the
  `tool=/ok=/count=/result=/records=/error=` envelope already shaped.
- **Proxy memory store**: A Google Sheet owned by the proxy account, holding
  append-only rows that mirror the OKF frontmatter
  (`timestamp | topic | type | title | tags | body`). Server-side; no device
  path; immune to iCloud quota; durable and cross-device.
- **Local memory store**: An append-only OKF Markdown file per topic under
  `{@memBase}/<topic>.md`, written/read by Cherri's built-in
  `appendToFile`/`getFile`/`createFolder` at a fixed text path (confirmed working
  on device). Writes are confirmed to land in **iCloud Drive/Shortcuts**; whether
  they can instead be pinned to **On My iPhone (Local Storage)** — the ideal
  iCloud-quota-immune target — is unverified and is settled by the **on-device
  probe** (Decision 1c), which reconciles `@memBase` to the confirmed location.
  Fast and offline, but while writes land in iCloud Drive a write can silently
  fail to sync when iCloud Drive is full/unpaid.
- **On-device write-location probe**: A one-time manual check (part of the setup
  primer) that writes a marker via `appendAgentFile` and confirms where it lands
  (iCloud Drive vs On My iPhone / Local Storage), so `@memBase` can be reconciled
  to the real location before shipping. See Decision 1c and Testing Strategy.
- **Backend selector (`@memoryBackend`)**: A one-time editable Text config action
  near the API-key config, normalized (whitespace-stripped, lowercased) from
  `@memoryBackendRaw`. Values: `hybrid` (default — write local-first then mirror
  to the proxy, read local-first with proxy fallback), `local` (Files only),
  `sheets` (Sheet/proxy only). Not a per-run spoken prompt.
- **Write-through (hybrid write)**: A memory write that writes to the phone-local
  Files store **first** (instant, iCloud-quota-proof) and **then** mirrors to the
  proxy Sheet when the proxy is configured, under `hybrid`. Each leg fails open
  independently: the write succeeds if EITHER leg succeeds, and the run never
  halts. No runtime error-detection is required.
- **Local-first read**: A read/bootstrap that reads the local file first and only
  falls back to the proxy when the local result is empty/missing.
- **OKF concept entry**: The stored record shape from `docs/okf-knowledge-base.md`
  — YAML frontmatter (`type`, `title`, `tags`, `timestamp`) followed by the body
  text. The storage layer wraps each `memory_append(topic, body)` into an OKF
  concept; the model-facing protocol stays `memory_append(topic, body)` only.
- **@loopContext**: The accumulated conversation state re-sent as the user
  message every turn (the API is stateless).
- **@currentRequest**: The message the planner is currently answering. Today it
  is pinned to the original request; this design repoints it to the most recent
  user message.
- **Context length / token budget**: The model's usable context window; used as
  the basis for the token-based compaction trigger.

## Bug Details

### Bug Condition

The memory defect manifests whenever a memory operation (`memory_append`,
`memory_read`, `memory_list`, `memory_status`, and `create_note`/`quick_journal`)
is backed by a **single fragile store**: memory that is written to only one place
is not durable and cross-device. On-device testing corrected the earlier
diagnosis — a fixed-path built-in Files write DOES persist (it landed in iCloud
Drive under `Shortcuts/`) — so the real failure is (a) proxy-only recall having
no local copy, and (b) local-only recall silently losing data when iCloud Drive
is full and the write never syncs to other devices/backup. Neither condition is
detectable at runtime (no try/catch, file actions return no error value), so the
fix is structural: write-through to both stores and read local-first with proxy
fallback. The context defect
manifests whenever the user sends a follow-up that references an earlier turn:
the planner is still led by the original `user_request` while the follow-up sits
far down `@loopContext`. The compaction defect manifests whenever the running
conversation grows: compaction fires on a fixed follow-up count
(`@maxContextExchanges`), not on how full the context window is. The model
defect manifests whenever the planner is the weak default (breaks flat JSON) or
a swapped-in large/reasoning model (exceeds ~25s or emits `<think>`).

**Formal Specification:**
```
FUNCTION isBugCondition(input)
  INPUT: input of type AgentTurn
  OUTPUT: boolean

  memoryBug  := input.op IN ['memory_append','memory_read','memory_list',
                             'memory_status','create_note','quick_journal']
                AND ( NOT persistedDurably(input)          // not written anywhere recoverable
                      OR NOT recallableCrossDevice(input)  // single backend, no mirror
                      OR silentSyncLossUnrecovered(input) )// iCloud-full write lost with no fallback

  contextBug := input.isFollowUp
                AND plannerCurrentRequest(input) != input.mostRecentUserMessage

  compactBug := contextGrew(input)
                AND compactionTrigger(input) == "fixed_exchange_count"

  modelBug   := plannerModel(input) IS weakInstruct
                OR plannerModel(input) IS reasoningModel
                OR plannerLatency(input) > 25_seconds

  RETURN memoryBug OR contextBug OR compactBug OR modelBug
END FUNCTION
```

### Examples

- "Remember that I prefer tea over coffee." → under the old proxy-only design the
  local copy is never written, and under a naive local-only design an iCloud-full
  phone silently fails to sync, so a *different* device later reads nothing. Next
  session "what do I prefer to drink?" → `memory_read` returns empty. (Expected:
  the preference is written through to both stores as an OKF entry, and recall
  reads local-first with proxy fallback so it survives across turns, sessions,
  and devices even when local sync silently fails.)
- "When is the next match?" (web_search answers) then "tell me more about that
  match." → the planner keeps answering "when is the next match" because
  `user_request` is still the original. (Expected: it answers the follow-up
  about *that* match.)
- A long chat hits 4 follow-ups → compaction fires and drops the match name, so
  a later "that match" reference can no longer be resolved. (Expected:
  compaction preserves the named entity.)
- Default `meta/llama-3.1-8b-instruct` returns prose or malformed JSON on a
  tool request; swapping to a 70B/reasoning model times out past ~25s or emits
  `<think>` that breaks the flat-JSON contract. (Expected: a capable,
  non-reasoning model that reliably returns flat JSON within budget.)

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- Hands-free Siri runs stay picker-free and gate-free on the happy path (3.1).
- Planner replies are still required to be one flat single-line JSON object,
  regex-validated before `getDictionary()`; memory text never reaches
  `getDictionary()` (3.2).
- Control flow stays free of `else if`; flat guarded `if` / default-then-override
  only (3.3).
- Every planner and tool call stays within the ~25s iOS network budget (3.4).
- Non-memory tools (`web_search`, `calendar_lookup`, `reminders_lookup`,
  `weather_summary`, `tasks_*`, `gmail_*`, drafting, `open_*`) behave exactly as
  today (3.5).
- When the memory store is unreachable/unconfigured, Iris runs and answers
  normally, fail-open, no halt (3.6).
- Retrieved memory is treated as opaque user data, not instructions; only
  relevant snippets go to the model (3.7).
- Final answers still use the Ask-for-Input S-GPT pattern (neutralize "?",
  flatten newlines, strip markdown) and end on a stop word/silence (3.8).
- A non-stop follow-up still resets the per-request budgets (3.9).

**Scope:**
All inputs that do NOT match `isBugCondition` are completely unaffected: casual
chat, jokes, one-shot answers, and every existing non-memory tool route follow
the current code path unchanged. The correct behavior for the bug-triggering
inputs is defined in the Correctness Properties below.

## Hypothesized Root Cause

1. **Misdiagnosis, not a no-op (corrected primary memory root cause).** The
   earlier hypothesis — that Cherri's built-in `appendToFile`/`getFile` emit a
   bare `WFFilePath` with no `fileLocation` object and therefore silently no-op —
   was **wrong in its conclusion, and refuted by on-device testing**. With a
   fixed text path the built-ins **do** persist: the data landed in iCloud Drive
   under the `Shortcuts` folder. The two *actual* prior failures were:
   **(a) the parent folder was never created before the first `appendToFile`**,
   so a first write into a missing folder produced nothing (fixed by a
   `createFolder`-on-setup seed plus a self-heal `createFolder` before the first
   append); and **(b) the user was checking the wrong Files location**, so the
   data that did land looked "missing" (a discovery/verification failure, not a
   write failure). The `crossDeviceItemID`/`fileLocation` object is NOT required
   for a text-path write to succeed. Phone-local Files is a viable backend.

   **Open, probe-settled question:** writes are confirmed to land in **iCloud
   Drive/Shortcuts**; whether the built-ins can be pinned to **On My iPhone
   (Local Storage)** for full iCloud-quota immunity is **unverified**, because the
   built-ins emit only a text `WFFilePath` and no `fileLocation` object to select
   the service. The design resolves this with an explicit on-device probe
   (Decision 1c) that reconciles `@memBase` to wherever writes actually land.

2. **Single-backend fragility under iCloud quota (the real remaining risk).** A
   phone-local write can **silently fail to sync** when iCloud Drive is full or
   unpaid — the write appears to succeed locally but never propagates, so
   cross-device recall and iCloud backup break with no error surfaced. Shortcuts
   has no try/catch and file actions return no error value, so this cannot be
   detected at runtime. A local-only backend therefore cannot guarantee durable,
   cross-device memory on its own; a proxy mirror plus a local-first read with
   proxy fallback removes the risk without needing error-detection.

3. **No user control over backend / no self-healing folder.** There was no
   reliable way for the user to force local-only or cloud-only, and the local
   folder was only ever created in the manual "setup" branch, so a first local
   write before setup could miss its parent folder. A persisted Text-action
   selector plus a `createFolder` seed in the setup primer fixes both.

4. **Planner led by the original request.** The user message template hardcodes
   `user_request={@request}`; follow-ups are appended as `user_followup=` deep in
   `@loopContext`, so recency is inverted and reference resolution fails.

5. **Compaction on exchange count.** `@exchangesSinceCompact >= @maxContextExchanges`
   ignores actual context size, and the summary prompt does not protect named
   entities, so it both fires at the wrong time and can drop follow-up specifics.

6. **Model choice.** The default is a weak 8B instruct; capable models on the
   free tier are either too slow (70B) or reasoning models that emit `<think>`
   and break flat JSON. No middle option was selected.

## Correctness Properties

Property 1: Bug Condition — Memory persists via OKF write-through

_For any_ agent turn where the user asks Iris to save information (memory write),
the fixed system SHALL persist it as an append-only **OKF concept entry** to
every backend selected by `@memoryBackend`: under `hybrid` it writes the local
Files store (`{@memBase}/<topic>.md`) FIRST and THEN mirrors to the proxy memory
store (Google Sheet) when the proxy is configured; under `local` or `sheets` it
writes only that one; each leg fails open independently so the write succeeds if
EITHER leg succeeds; so the information survives across turns, sessions, and
shortcut rebuilds so long as at least one selected backend is available.

**Validates: Requirements 2.1**

Property 2: Bug Condition — Both stores self-heal so the first write succeeds

_For any_ memory write where the backing store does not yet exist, the fixed
system SHALL create it on demand before writing: the proxy side finds-or-creates
the `Iris Memory` sheet and `memory` tab, and the client seeds the local
`Shortcuts/IrisOKF` folder (setup primer + auto-creating `appendToFile`), so the
first write succeeds with no prior manual setup run.

**Validates: Requirements 2.2**

Property 3: Bug Condition — Reliable addressing with sync-loss recovery

_For any_ memory operation, the fixed system SHALL address storage by a fixed
`topic` (never a device-specific identifier), and SHALL make a silent
local-sync failure recoverable: because writes are mirrored to the proxy under
`hybrid`, a read that finds the local copy empty/missing falls back to the proxy
copy, so an iCloud-full phone never loses cross-device recall.

**Validates: Requirements 2.3**

Property 4: Bug Condition — Write-then-read returns the written value (local-first, proxy fallback)

_For any_ topic, a `memory_append(topic, body)` followed by a later
`memory_read(topic)` (same session or a later session, same or different device)
SHALL return content that includes the previously written `body`, reading
local-first and falling back to the proxy when the local copy is empty/missing.

**Validates: Requirements 2.4**

Property 5: Bug Condition — Planner answers the most recent message

_For any_ follow-up turn, the fixed system SHALL present the user's most recent
message as the current request to the planner while keeping the full prior
conversation available as context, so the model answers the follow-up rather
than the original request.

**Validates: Requirements 2.5**

Property 6: Bug Condition — Compaction preserves follow-up entities

_For any_ compaction of the running context, the fixed system SHALL preserve the
named entities and specifics (proper nouns, titles, numbers, the most recent
request) needed to resolve later references such as "that match".

**Validates: Requirements 2.6**

Property 7: Bug Condition — Capable, non-reasoning, latency-safe model

_For any_ planner call, the fixed system SHALL use a NIM model that is a
non-reasoning instruct model selected from the current catalog and fast enough
to respond within the ~25s iOS budget, and SHALL NOT use a reasoning/thinking
model.

**Validates: Requirements 2.7**

Property 8: Bug Condition — Token-based compaction trigger

_For any_ growth of the running context, the fixed system SHALL trigger
compaction when the estimated token usage of the outgoing planner call reaches
approximately 80% of the configured model context budget, rather than on a fixed
follow-up-exchange count.

**Validates: Requirements 2.8**

Property 9: Preservation — Non-triggering behavior is unchanged

_For any_ input where the bug condition does NOT hold (hands-free happy path,
flat-JSON parsing and the `getDictionary()` guard, absence of `else if`, ~25s
call budget, non-memory tools, opaque memory handling, S-GPT delivery, and
per-request budget resets), the fixed system SHALL produce the same behavior as
the original system.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.7, 3.8, 3.9**

Property 10: Preservation — Fail-open when memory is unavailable

_For any_ turn where no selected backend is available (the proxy is unreachable/
unconfigured AND the local Files store is unavailable, per the `@memoryBackend`
selection), the fixed system SHALL continue to run and answer normally (no halt),
exactly as Iris behaves today with no memory.

**Validates: Requirements 3.6**

Property 11: Bug Condition — Backend selector honors hybrid/local/sheets

_For any_ value of `@memoryBackend` normalized from the one-time editable Text
config (`hybrid`, `local`, or `sheets`), the fixed system SHALL route memory
writes and reads accordingly: `hybrid` writes local-first then mirrors to the
proxy (each leg fail-open) and reads local-first with proxy fallback; `local`
uses only the Files store; `sheets` uses only the Sheet; and an unrecognized
value SHALL default to `hybrid`.

**Validates: Requirements 2.1, 2.3**

## Fix Implementation

### Decision 1 — Memory backend: HYBRID, user-selectable, OKF in both stores

> **CHANGE MARKER — this supersedes the previous proxy-only Decision 1.** The
> earlier decision moved memory to a proxy-backed Google Sheet *only*, on the
> (now refuted) grounds that phone-local Files always no-ops. On-device testing
> confirmed the built-in Files write **works**; the real risk is silent
> iCloud-sync loss when iCloud Drive is full. Decision 1 is therefore rewritten
> as a **hybrid**: a one-time backend selector (`hybrid`/`local`/`sheets`),
> write-through to both stores under `hybrid` (local first, then proxy mirror),
> local-first read with proxy fallback, and OKF-formatted storage in BOTH stores.
> Decisions 2–5 are unchanged except Decision 2's read source (now local-first,
> proxy fallback).

**File**: `docs/apps-script-proxy.md` (proxy/server) and `shortcuts/iris.cherri`
(client + local Files). Supabase remains documented as an alternative proxy
backend but is not designed in detail.

#### 1a — Backend selector (reliable, not a per-run prompt)

Add an editable Text config action next to the API-key config (the same S-GPT
pattern used for the NVIDIA key, Tavily key, and proxy URL/secret), normalized
exactly like the other config values:

```cherri
/* MEMORY BACKEND: hybrid (default), local, or sheets. Edit this Text box ONCE in
   the Shortcuts editor to force a single backend; leave "hybrid" to use both.
   - hybrid : write phone-local Files FIRST, THEN mirror to the proxy Sheet;
              read local-first with proxy fallback (recommended default).
   - local  : phone-local Files only (fast/offline; can lose cross-device sync
              if iCloud Drive is full).
   - sheets : Google Sheet via the Apps Script proxy only (durable, cross-device,
              immune to iCloud quota).
   A per-run VOICE prompt is deliberately NOT used: under hands-free/locked Siri
   the only reliable conversational primitive is Ask for Input, and asking the
   backend every run would be a picker-like gate on the happy path (violates
   Req 3.1), add latency, and cannot persist a choice. A Text action is the
   persistence surface that already works reliably in this shortcut (it is how
   the API key survives), so the one-time selector lives there. */
@memoryBackendRaw = text("hybrid")
@memoryBackendLower = replaceText('\s+', "", "{@memoryBackendRaw}", false, true)
@memoryBackend = lowercase("{@memoryBackendLower}")
/* Default-then-override: an unrecognized value falls back to hybrid (Prop 11). */
@memBackendOk = "no"
if @memoryBackend == "hybrid" { @memBackendOk = "yes" }
if @memoryBackend == "local"  { @memBackendOk = "yes" }
if @memoryBackend == "sheets" { @memBackendOk = "yes" }
if @memBackendOk == "no" { @memoryBackend = "hybrid" }
```

Two derived availability flags decide what each dispatch actually does, honoring
the selector, `@proxyOk`, and whether the local Files path is usable:

```cherri
/* Is the proxy (Sheet) leg usable this run? sheets or hybrid, AND a real proxy
   URL. The selector value is `sheets`; the transport is still the Apps Script
   proxy, so the flag name stays @memProxyOn. Nested flat ifs, never else-if. */
@memProxyOn = "no"
if @memoryBackend == "sheets" { if @proxyOk > 0 { @memProxyOn = "yes" } }
if @memoryBackend == "hybrid" { if @proxyOk > 0 { @memProxyOn = "yes" } }
/* Is the local Files store usable this run? local or hybrid. (Files actions are
   confirmed working; there is no runtime availability probe, so local is treated
   as available whenever selected — a silent sync loss is covered by the Sheet
   mirror under hybrid, per Prop 3.) */
@memLocalOn = "no"
if @memoryBackend == "local"  { @memLocalOn = "yes" }
if @memoryBackend == "hybrid" { @memLocalOn = "yes" }
```

The **setup primer** speaks one extra line telling the user they can edit the
backend box; default `hybrid` needs no action (see the primer changes in 1d).

#### 1b — OKF concept entry shape (stored in BOTH stores)

Each `memory_append(topic, body)` is wrapped by the **storage layer** (not the
model) into an OKF concept entry, per `docs/okf-knowledge-base.md`:

```md
---
type: <derived from topic, e.g. log→Memory Entry, preferences→User Preference,
       profile→Profile, notes→Note, journal→Journal, index→Index, projects→Project>
title: <optional: first line of body, or empty>
tags: <optional: empty by default>
timestamp: <ISO-8601, added by the shortcut/proxy, never by the model>
---

<body text>
```

**Model-facing surface stays lean.** The protocol tool is still
`memory_append(topic, body)` **only** — no `title`/`tags`/`type` are added to the
protocol, keeping the planner prompt compact for the ~25s budget. The storage
layer derives `type` from `topic`, stamps `timestamp`, and leaves `title`/`tags`
optional (title = first line of body or empty; tags empty). This is how memory
"follows OKF" (the on-disk/on-sheet record is a valid OKF concept) while the
agent protocol carries only two fields.

**Server-side storage shape (proxy Sheet).** One append-only sheet, resolved by
name so no ID is pasted anywhere, with columns mirroring the OKF frontmatter:

- Spreadsheet: found/created by name `Iris Memory` in the owner's Drive; a tab
  `memory` with header row `timestamp | topic | type | title | tags | body`.
- A write appends one row `[ISO-8601 now, topic, type, title, tags, body]`.
  Append-only (no overwrite). One row reconstructs to one OKF concept.
- A read filters rows by `topic` (case-insensitive), returns the most recent N
  (cap `MAX_MEMORY_ROWS = 20`) entries. `memory_read` can reconstruct an
  OKF-formatted concept from the row columns, or return just the `body` values
  for voice; the bodies are joined newest-last for speaking.
- `topic` is constrained server-side to an allowlist
  (`index`, `profile`, `preferences`, `log`, `notes`, `journal`); any other
  value falls back to `log`. This keeps the model from addressing arbitrary
  storage and matches the "fixed predeclared tools" rule.

**Local storage shape (Files).** The OKF concept block is **appended** to
`Shortcuts/IrisOKF/<topic>.md` (append-only). `getFile` reads the file back as
**opaque text** — it is Markdown with YAML frontmatter, never JSON, so it is
**never** passed to `getDictionary()` (Req 3.2, 3.7). The shortcut supplies the
`timestamp`/`type` when it builds the block; the model supplies only `body`.

**New proxy ops (doGet/handle dispatch).** Add to `handle(req)` in
`docs/apps-script-proxy.md` §4, alongside the existing `if (op === ...)` lines:

```javascript
if (op === 'memory_append')  { return memoryAppend(req); }
if (op === 'memory_read')    { return memoryRead(req); }
if (op === 'memory_list')    { return memoryList(req); }
if (op === 'memory_status')  { return memoryStatus(req); }
if (op === 'memory_summary') { return memorySummary(req); }
```

**Apps Script handler additions** (documented in `docs/apps-script-proxy.md`;
self-heals the store, so Property 2 holds). The columns mirror the OKF
frontmatter (`timestamp | topic | type | title | tags | body`) so a row
reconstructs to an OKF concept, while the model still passes only `topic`+`body`:

```javascript
/* ---------- Memory (append-only Google Sheet, OKF-shaped rows) ---------- */

var MEMORY_SHEET_NAME = 'Iris Memory';
var MEMORY_TAB = 'memory';
var MEMORY_HEADER = ['timestamp', 'topic', 'type', 'title', 'tags', 'body'];
var MEMORY_TOPICS = ['index','profile','preferences','log','notes','journal'];
var MAX_MEMORY_ROWS = 20;

// Derive the OKF `type` from the topic (server-side; the model never sends it).
function _okfType(topic) {
  var map = {
    log: 'Memory Entry', preferences: 'User Preference', profile: 'Profile',
    notes: 'Note', journal: 'Journal', index: 'Index'
  };
  return map[topic] || 'Memory Entry';
}

// Find-or-create the spreadsheet and tab. Self-healing: the first write
// creates everything, so no manual setup run is needed (Req 2.2, 2.3).
function _memorySheet() {
  var files = DriveApp.getFilesByName(MEMORY_SHEET_NAME);
  var ss = files.hasNext() ? SpreadsheetApp.open(files.next())
                           : SpreadsheetApp.create(MEMORY_SHEET_NAME);
  var sh = ss.getSheetByName(MEMORY_TAB);
  if (!sh) {
    sh = ss.insertSheet(MEMORY_TAB);
    sh.appendRow(MEMORY_HEADER);
  }
  return sh;
}

function _normTopic(t) {
  t = String(t || 'log').trim().toLowerCase();
  return MEMORY_TOPICS.indexOf(t) === -1 ? 'log' : t;
}

// Reconstruct an OKF concept block from a row [ts, topic, type, title, tags, body].
function _rowToOkf(r) {
  return '---\ntype: ' + (r[2] || 'Memory Entry') +
         '\ntitle: ' + (r[3] || '') +
         '\ntags: ' + (r[4] || '') +
         '\ntimestamp: ' + (r[0] || '') +
         '\n---\n' + (r[5] || '');
}

function memoryAppend(req) {
  var topic = _normTopic(req.topic);
  var body = String(req.body || '').trim();
  if (!body) { return _env('memory_append', false, 0, '', '', 'A body to remember is required.'); }
  var type = _okfType(topic);
  // title = optional first line of body; tags optional (empty). Timestamp server-side.
  var title = body.split('\n')[0].slice(0, 80);
  var tags = String(req.tags || '');
  _memorySheet().appendRow([new Date().toISOString(), topic, type, title, tags, body]);
  return _env('memory_append', true, 1, 'Saved to ' + topic + ' memory.', body, '');
}

// Return the most recent MAX_MEMORY_ROWS rows for a topic (full row arrays).
function _readTopicRows(topic) {
  var sh = _memorySheet();
  var values = sh.getDataRange().getValues(); // includes header
  var rows = [];
  for (var i = 1; i < values.length; i++) {
    if (String(values[i][1]).toLowerCase() === topic) { rows.push(values[i]); }
  }
  return rows.slice(-MAX_MEMORY_ROWS);
}

function _readTopic(topic) {
  return _readTopicRows(topic).map(function (r) { return r[5]; }); // bodies
}

function memoryRead(req) {
  var topic = _normTopic(req.topic);
  var rows = _readTopicRows(topic);
  if (!rows.length) {
    return _env('memory_read', false, 0, '', '', 'No memory found for that topic yet.');
  }
  // Default: bodies joined newest-last for voice (records stays plain text so
  // the Shortcut never getDictionary()s it). Set req.format='okf' to return the
  // reconstructed OKF concept blocks instead (e.g. for a non-voice client).
  var records;
  if (String(req.format || '') === 'okf') {
    records = rows.map(_rowToOkf).join('\n\n');
  } else {
    records = rows.map(function (r) { return r[5]; }).join('; ');
  }
  return _env('memory_read', true, rows.length, 'Read ' + topic + ' memory.', records, '');
}

function memoryList(req) {
  var sh = _memorySheet();
  var values = sh.getDataRange().getValues();
  var rows = [];
  for (var i = Math.max(1, values.length - MAX_MEMORY_ROWS); i < values.length; i++) {
    rows.push(values[i][1] + ': ' + values[i][5]); // topic: body
  }
  if (!rows.length) { return _env('memory_list', true, 0, 'Memory is empty.', '', ''); }
  return _env('memory_list', true, rows.length, 'Read memory.', rows.join('; '), '');
}

function memoryStatus(req) {
  var sh = _memorySheet();
  var n = Math.max(0, sh.getLastRow() - 1); // minus header
  if (n === 0) { return _env('memory_status', true, 0, 'Memory is set up but empty.', '', ''); }
  return _env('memory_status', true, n, 'Memory is available.', '', '');
}

// Compact bootstrap summary for turn-one injection: profile + preferences +
// a few recent index/log lines, hard-capped so it never fattens every call.
function memorySummary(req) {
  var parts = []
    .concat(_readTopic('profile'))
    .concat(_readTopic('preferences'))
    .concat(_readTopic('index').slice(-3));
  var text = parts.join('; ');
  if (text.length > 600) { text = text.slice(0, 600); }
  return _env('memory_summary', true, parts.length, 'Memory summary.', text, '');
}
```

> The envelope stays the standard six keys. `memory_read` puts voice bodies in
> `records` by default and can reconstruct OKF concept blocks (`_rowToOkf`) into
> `records` when `format=okf` is passed — the row columns are the OKF fields, so
> a row round-trips to a concept. Keep the existing `MAX_MEMORY_ROWS` cap and the
> 600-char `memory_summary` cap.

Manifest note (`docs/apps-script-proxy.md` §3): add the Sheets/Drive scopes the
handlers need, e.g. `https://www.googleapis.com/auth/spreadsheets` and
`https://www.googleapis.com/auth/drive`. The owner authorizes these once at the
same consent step as Tasks/Gmail; no on-device token.

#### 1c — Local Files custom actions (reintroduced, confirmed working)

Reinstate the built-in Files custom actions for the local path (device-confirmed
to persist at a fixed text path). Declare them alongside the other custom actions
using Cherri's `action` form — never `rawAction()`:

```cherri
/* --- Local memory (Files) custom actions — fixed text path, no picker.
   Confirmed on device: with a fixed WFFilePath the built-ins persist to iCloud
   Drive under Shortcuts/. Content is OKF Markdown (YAML frontmatter + body),
   treated as OPAQUE TEXT — never getDictionary()'d (Req 3.2, 3.7). */
action 'is.workflow.actions.file.createfolder' createAgentFolder(text path: 'WFFilePath'): text
action 'is.workflow.actions.file.append' appendAgentFile(text content: 'WFInput', text path: 'WFFilePath'): text {
  "WFAppendFileWriteMode": "Append",
  "WFFileAppendNewLine": true
}
action 'is.workflow.actions.documentpicker.open' getAgentFile(text path: 'WFGetFilePath'): text {
  "WFShowFilePicker": false,
  "WFFileErrorIfNotFound": false
}
```

`@memBase = "Shortcuts/IrisOKF"` is the fixed local KB root; the per-topic path
is resolved with flat default-then-override `if`s (no `else if`, Req 3.3):

```cherri
@memBase = "Shortcuts/IrisOKF"
@memPath = "{@memBase}/log.md"                 /* safe default */
if @topic.text == "index"       { @memPath = "{@memBase}/index.md" }
if @topic.text == "profile"     { @memPath = "{@memBase}/profile.md" }
if @topic.text == "preferences" { @memPath = "{@memBase}/preferences.md" }
if @topic.text == "log"         { @memPath = "{@memBase}/log.md" }
if @topic.text == "notes"       { @memPath = "{@memBase}/notes.md" }
if @topic.text == "journal"     { @memPath = "{@memBase}/journal.md" }
/* Build the OKF concept block the shortcut appends locally (type derived from
   topic; timestamp stamped here; title/tags optional). */
@okfType = "Memory Entry"
if @topic.text == "preferences" { @okfType = "User Preference" }
if @topic.text == "profile"     { @okfType = "Profile" }
if @topic.text == "notes"       { @okfType = "Note" }
if @topic.text == "journal"     { @okfType = "Journal" }
if @topic.text == "index"       { @okfType = "Index" }
@okfBlock = "\n---\ntype: {@okfType}\ntitle: \ntags: \ntimestamp: {@nowRaw}\n---\n{@body}\n"
```

**On-device write-location probe (settles where local writes land).** The
built-in Files actions emit only a text `WFFilePath` and no `fileLocation`
object, so which Files service they target is not selectable from Cherri and is
**unverified**: confirmed behavior is that writes land in **iCloud Drive under
`Shortcuts/`**, but **On My iPhone (Local Storage)** — the quota-immune target
the reference plist pins with `WFFileLocationType = "LocalStorage"` — cannot be
confirmed from the emitted action alone. Rather than guess, the setup primer runs
a one-time **probe**: write a marker to `{@memBase}/probe.md`, then read it back
and tell the user exactly where to look, so `@memBase` can be reconciled to the
real location before shipping.

```cherri
/* PROBE (setup only): write a marker, read it back, and tell the user where to
   look so we can confirm iCloud Drive vs On My iPhone / Local Storage. Uses the
   same fixed-text-path built-ins as the real local leg; content is opaque text,
   never getDictionary()'d (Req 3.2). */
createAgentFolder("{@memBase}")
@probeMarker = "\n---\ntype: Index\ntitle: probe\ntags: \ntimestamp: {@nowRaw}\n---\nIris write-location probe.\n"
@probeWrite = appendAgentFile("{@probeMarker}", "{@memBase}/probe.md")
@probeRead = getAgentFile("{@memBase}/probe.md")
@probeChars = count("{@probeRead}")
@probeMsg = "I saved a test note. Open the Files app and check BOTH On My iPhone and iCloud Drive under Shortcuts, IrisOKF, probe. Tell me which one it landed in so memory is pointed at the right place."
if @probeChars == 0 { @probeMsg = "The test note did not read back. Memory will still work through the cloud Sheet; check that the memory backend box is hybrid or local." }
@probeAck = prompt("{@probeMsg}")
```

Reconciliation is a one-line config edit, not runtime logic: if the probe shows
writes land in On My iPhone / Local Storage, `@memBase` is already correct and
the phone-local leg is quota-immune; if they land in iCloud Drive, `@memBase`
stays `Shortcuts/IrisOKF` and the hybrid Sheet mirror covers the iCloud-full
silent-sync risk (Prop 3). Either way the probe outcome is **low-risk** because
the durable leg does not depend on it. The probe writes to a dedicated
`probe.md` so it never pollutes a real topic file.

#### 1d — Client dispatch shape (hybrid)

Each memory dispatch honors `@memoryBackend` via the derived `@memLocalOn` /
`@memProxyOn` flags (1a), using flat guarded `if`s (never `else if`) and
literal-dict `downloadURL`. The proxy `records` string is plain text forwarded
into the envelope and is **never** passed to `getDictionary()` (only the outer
JSON envelope is); the local file text is likewise opaque (Req 3.2, 3.7).

**`memory_append` — write-through.** Append to the local file when local is on,
AND POST to the proxy when proxy is on; the envelope reports success if either
backend was written:

```cherri
if @tool.text == "memory_append" {
  @toolHandled = "yes"
  @memWroteAny = "no"
  /* Local write-through (append-only OKF block; folder auto-created). */
  if @memLocalOn == "yes" {
    @memMkdir = createAgentFolder("{@memBase}")
    @memLocalWrite = appendAgentFile("{@okfBlock}", "{@memPath}")
    @memWroteAny = "yes"
  }
  /* Proxy write-through. */
  @pOk = "false"
  if @memProxyOn == "yes" {
    @encTopic = urlEncode("{@topic}")
    @encBody = urlEncode("{@body}")
    @proxyCallUrl = "{@irisProxyUrl}?secret={@proxySecretEnc}&op=memory_append&topic={@encTopic}&body={@encBody}"
    @proxyResp = downloadURL("{@proxyCallUrl}", {"Accept": "application/json"})
    @proxyDict = getDictionary(@proxyResp)
    @pOk = getValue(@proxyDict, "ok")
    @memWroteAny = "yes"
  }
  if @memWroteAny == "yes" {
    @toolResult = "tool=memory_append\nok=true\ncount=1\nresult=Saved to {@topic} memory.\nrecords={@body}\nerror="
    @localFinalText = "Saved that to your memory."
  } else {
    @toolResult = "tool=memory_append\nok=false\ncount=0\nresult=\nrecords=\nerror=Memory is not set up. Pick a backend: set the memory backend box to local, or deploy the Apps Script proxy and paste its URL and secret in."
    @localFinalText = "Memory is not set up yet, so I could not save that."
  }
}
```

**`memory_read` / `memory_list` / `memory_status` — local-first, proxy
fallback.** Read the local file first; only when it is empty/missing do we call
the proxy:

```cherri
if @tool.text == "memory_read" {
  @toolHandled = "yes"
  @memRecords = ""
  @memChars = 0
  /* Local-first. */
  if @memLocalOn == "yes" {
    @memText = getAgentFile("{@memPath}")       /* opaque text; never getDictionary() */
    @memChars = count("{@memText}")
    if @memChars > 0 { @memRecords = "{@memText}" }
  }
  /* Proxy fallback only when local yielded nothing. */
  if @memChars == 0 {
    if @memProxyOn == "yes" {
      @encTopic = urlEncode("{@topic}")
      @proxyCallUrl = "{@irisProxyUrl}?secret={@proxySecretEnc}&op=memory_read&topic={@encTopic}"
      @proxyResp = downloadURL("{@proxyCallUrl}", {"Accept": "application/json"})
      @proxyDict = getDictionary(@proxyResp)
      @pRecords = getValue(@proxyDict, "records")
      @memRecords = "{@pRecords}"
      @memChars = count("{@memRecords}")
    }
  }
  if @memChars > 0 {
    @toolResult = "tool=memory_read\nok=true\ncount=1\nresult=Read {@topic} memory.\nrecords={@memRecords}\nerror="
    @localFinalText = "Here is what I have saved about {@topic}: {@memRecords}"
  } else {
    @toolResult = "tool=memory_read\nok=false\ncount=0\nresult=\nrecords=\nerror=No memory found for that topic yet."
    @localFinalText = "I do not have anything saved about {@topic} yet."
  }
}
```

`memory_list` and `memory_status` follow the same local-first/proxy-fallback
shape (list reads the per-topic files or the proxy `memory_list`; status reports
available/empty from whichever backend answered). `memory_read` / `memory_list`
stay in the existing `@speakRecords = "yes"` set so retrieved items are spoken.

**`create_note` (topic=notes) and `quick_journal` (topic=journal) — write-through.**
Both reuse the same write-through path as `memory_append` (local append of the
OKF block AND proxy POST, honoring the selector). When neither backend is
available they fail-open with an `ok=false` "not set up" envelope, keeping the
run alive (Req 3.6). These two tools are not in the 3.5 unchanged list, so
routing them through the hybrid store is in scope and required by Req 2.1.

**Setup primer changes (1d).** Reintroduce the local seeding in the setup primer
(now that Files writes are confirmed): `createAgentFolder("{@memBase}")` and a
seed `appendAgentFile` per topic file, so the first real local write never hits a
missing folder and the first Files-permission dialog is answered during the one
manual unlocked run (Req 3.1 on later hands-free runs). The primer still warms
the proxy via the existing `tasks_list` GET, and runs the write-location probe
(1c). The primer also **speaks one line**: "Memory uses both your phone and the
cloud by default. To use only one, edit the memory backend box to local or
sheets." Default `hybrid` needs no user action.

**Supabase alternative (documented, not designed):** a Supabase REST table
`iris_memory(id, created_at, topic, body)` reached with `jsonRequest` + an
`apikey` header would satisfy the same properties. It adds a second host to
prime and a key to manage, so the proxy (already wired and primed) is preferred.

### Decision 2 — Startup memory bootstrap (local-first, proxy fallback)

Before the `repeat` loop (after the setup-primer block, before the first planner
`jsonRequest`), read a compact summary once and inject it, fail-open. Consistent
with Decision 1's hybrid read, the bootstrap is **local-first with proxy
fallback**: read the local profile/preferences files first, and only call the
proxy `memory_summary` when local yielded nothing (which also covers the
iCloud-full silent-sync case, since the proxy mirror still has the data):

```cherri
@memorySummary = ""
/* Local-first: read the small local bootstrap files if local is on. */
if @memLocalOn == "yes" {
  @bootProfile = getAgentFile("{@memBase}/profile.md")     /* opaque text */
  @bootPrefs = getAgentFile("{@memBase}/preferences.md")
  @bootLocal = "{@bootProfile} {@bootPrefs}"
  if count("{@bootLocal}") > 1 {
    @memorySummary = "{@bootLocal}"
  }
}
/* Proxy fallback only when local produced nothing. */
if count("{@memorySummary}") == 0 {
  if @memProxyOn == "yes" {
    @bootUrl = "{@irisProxyUrl}?secret={@proxySecretEnc}&op=memory_summary"
    @bootResp = downloadURL("{@bootUrl}", {"Accept": "application/json"})
    @bootDict = getDictionary(@bootResp)
    @bootRecords = getValue(@bootDict, "records")
    if @bootRecords {
      @memorySummary = "{@bootRecords}"
    }
  }
}
if @memorySummary {
  @loopContext += "\n\nmemory_summary={@memorySummary}"
}
```

The proxy caps its summary at 600 chars, so it does not fatten every call toward
the ~25s budget (Req 3.4); when the local branch answers, keep the injected text
similarly small (the bootstrap files are the compact profile/preferences OKF
concepts). If neither backend is available or both return empty, `@loopContext`
is unchanged and Iris behaves exactly as today (Req 3.6). The summary is injected
as `memory_summary=` data, which the protocol already treats as observation data,
not instructions (Req 3.7). The local file text is opaque and is never passed to
`getDictionary()` (Req 3.2).

### Decision 3 — In-session context: answer the most recent message

Introduce `@currentRequest`, initialized to the first `@request`, and repoint
the planner user message from `user_request={@request}` to
`user_request={@currentRequest}`. On a non-stop follow-up, fold the prior
exchange into `@loopContext` as history and set the current request to the
follow-up:

```cherri
/* at init */
@currentRequest = "{@request}"

/* planner user message (unchanged except the variable) */
{"role": "user", "content": "CONVERSATION\ncurrent_datetime={@nowRaw}\nuser_request={@currentRequest}{@loopContext}\nremaining_tool_calls={@remainingToolCalls}\nremaining_user_questions={@remainingUserQuestions}\nReply with one flat single-line JSON object now."}

/* on a non-stop follow-up, in the delivery block */
@toolCallCount = 0
@userQuestionCount = 0
@repairCount = 0
@loopContext += "\n\nprevious_exchange=\nuser_said={@currentRequest}\nassistant_answered={@finalText}"
@currentRequest = "{@followup}"
```

The full prior conversation stays in `@loopContext` (every message appended), but
the model is now led by the latest message, so "tell me more about that match"
is answered as a follow-up (Req 2.5). The compaction system prompt already
re-sends `@loopContext`, so history remains available.

### Decision 4 — Token-based compaction

Replace the exchange-count trigger with an estimated-token trigger. Cherri's
`count()` on a text value returns its character count; a `chars / 4` heuristic is
the standard cheap token estimate. Add constants near the other config:

```cherri
/* Effective context budget in tokens for the chosen model. The chosen model's
   true window is 128000, but a 128k prompt cannot return within iOS's ~25s
   network budget on the free tier, so we operate within a latency-safe working
   window and trigger compaction at ~80% of it. Raise if latency ever allows. */
@modelContextTokens = 128000
@contextTokenBudget = 12000
@compactAtTokens = 9600            /* ~80% of the working budget (Req 2.8, 3.4) */
```

Estimate tokens each turn and trigger on that instead of `@exchangesSinceCompact`:

```cherri
@approxChars = count("{@protocol}") + count("{@loopContext}") + count("{@currentRequest}")
@approxTokens = @approxChars / 4
if @approxTokens >= @compactAtTokens {
  /* ...existing compaction jsonRequest, then... */
  @loopContext = "\n\nconversation_summary={@compactSummary}"
}
```

`@maxContextExchanges` / `@exchangesSinceCompact` are removed. The trigger is now
token-based and honors Req 2.8 while the working-window sizing keeps every call
inside the ~25s budget (Req 3.4). The tension between "80% of the model's context
length" and the latency budget is resolved by treating the *effective* context
length as the latency-bounded working window rather than the theoretical 128k.

The compaction system prompt is strengthened to preserve follow-up entities
(Req 2.6):

> "You compress a running voice-assistant conversation into a short handoff.
> Preserve the user's overall goal, the MOST RECENT request, and every named
> entity or specific needed to resolve later references — proper nouns, titles,
> people, places, dates, numbers, and any 'that X' antecedent. Plain text, under
> 90 words, no markdown, no JSON."

### Decision 5 — Planner model (live NVIDIA NIM research, 2026)

Requirement: capable at instruction-following AND fast enough for the ~25s iOS
budget on the free tier, and **not** a reasoning/thinking model (reasoning models
emit `<think>` and break the flat-JSON contract; sources below confirm the
Nemotron Super/Nano families are reasoning models).

Catalog findings (build.nvidia.com / integrate.api.nvidia.com):
- [`meta/llama-3.1-8b-instruct`](https://build.nvidia.com/meta/llama-3_1-8b-instruct) —
  current default; fast but weak at the flat-JSON contract. Reject as primary.
- [`meta/llama-3.3-70b-instruct`](https://build.nvidia.com/meta/llama-3_3-70b-instruct/playground) —
  strong instruction following, 128K context, but frequently overloaded on the
  free tier (30s+), busting the ~25s budget. Reject as primary.
- [`nvidia/llama-3.3-nemotron-super-49b-v1.5`](https://build.nvidia.com/nvidia/llama-3_3-nemotron-super-49b-v1_5) and
  [`nvidia/nemotron-3-nano-30b-a3b`](https://build.nvidia.com/nvidia/nemotron-3-nano-30b-a3b.md) —
  explicitly **reasoning models** ("generates a reasoning trace"). Reject
  (violates Req 2.7).
- [`mistralai/mistral-small-3.1-24b-instruct-2503`](https://build.nvidia.com/mistralai/mistral-small-3_1-24b-instruct-2503.md) —
  24B dense **instruct** (non-reasoning), 128K context, strong system-prompt
  adherence; Mistral reports ~150 tokens/s inference. Best capability/speed
  balance on the free tier.
- [`qwen/qwen2.5-7b-instruct`](https://build.nvidia.com/qwen/qwen2_5-7b-instruct.md) —
  7B **instruct** (non-reasoning), up to 128K context, notable instruction-
  following improvement over Qwen2; very fast.

Recommendation:
- **Primary: `mistralai/mistral-small-3.1-24b-instruct-2503`.** Materially
  stronger flat-JSON/instruction-following than the 8B, far faster and less
  overloaded than the 70B, and non-reasoning. It is the smallest model that
  reliably holds the route contract while staying inside ~25s.
- **Fallback: `qwen/qwen2.5-7b-instruct`.** If Mistral Small is slow/overloaded
  in a given region, this 7B instruct is faster, still beats the 8B Llama on
  instruction following, and is non-reasoning. Swap by pasting its id into the
  model Text box.

Set the model id and the context constant for Decision 4:

```cherri
@nimModelIdRaw = text("mistralai/mistral-small-3.1-24b-instruct-2503")
/* Fallback if overloaded: qwen/qwen2.5-7b-instruct (both 128k, non-reasoning). */
@modelContextTokens = 128000   /* chosen model's true window; see Decision 4 */
```

The MODEL CHOICE RULE comment block is updated to name the new primary/fallback
and keep the "never a reasoning/thinking model" rule.

_Sources rephrased for licensing compliance; see the linked NVIDIA/Mistral model
cards above._

## Testing Strategy

### Validation Approach

Two phases: first surface counterexamples that demonstrate each defect on the
UNFIXED code, then verify the fix works and preserves existing behavior. Because
the target is a compiled Shortcut, validation runs on three surfaces already in
the repo: `scripts/validate-shortcut.py` (structural/compiler invariants),
`scripts/simulate-agent.py` (faithful offline loop against the real NIM API),
and a small `clasp`/`curl` check of the proxy `memory_*` ops.

### Exploratory Bug Condition Checking

**Goal**: Surface counterexamples BEFORE the fix to confirm the root causes. If
refuted, re-hypothesize.

**Test Plan**: Exercise each defect on the current build and observe failure.

**Test Cases**:
1. **Memory not durable/cross-device**: On the pre-hybrid build, run "remember X"
   then a later "what did I ask you to remember." Under proxy-only, confirm no
   local copy exists; under a naive local-only build with iCloud Drive full,
   confirm a *second device* recalls nothing (silent sync loss). Both surface the
   single-backend fragility the hybrid fixes (will fail on unfixed code). Also
   confirm the corrected root cause: a fixed-path built-in `appendToFile` DOES
   land a file in iCloud Drive under `Shortcuts/` (refuting the old "no-op"
   hypothesis).
2. **Follow-up reference**: Run "when is the next match" then "tell me more about
   that match" through `simulate-agent.py`; observe the planner re-answering the
   original request (will fail on unfixed code).
3. **Compaction drop**: Drive a long conversation past 4 follow-ups and confirm
   the summary can drop a named entity; confirm the trigger is exchange-count,
   not token usage (will fail on unfixed code).
4. **Model contract**: Run the tool-routing scenarios on `meta/llama-3.1-8b-instruct`
   and observe flat-JSON contract breaks; swap a 70B/reasoning model and observe
   >25s timeouts or `<think>` output (will fail on unfixed code).

**Expected Counterexamples**:
- Writes never appear in Files; reads return empty.
- Planner answers the original request, not the follow-up.
- Compaction fires on count and loses "that match".
- 8B breaks JSON; 70B/reasoning models bust the budget or contract.

### Fix Checking

**Goal**: For all inputs where the bug condition holds, the fixed system
produces the expected behavior.

**Pseudocode:**
```
FOR ALL input WHERE isBugCondition(input) DO
  result := iris_fixed(input)
  ASSERT expectedBehavior(result)
  // memory: append writes through to selected backends; read is local-first
  //         with proxy fallback and returns the written body; entries are
  //         OKF-formatted in both stores; selector honors hybrid/local/sheets
  //         (Prop 1,2,3,4,11)
  // context: planner sees mostRecentUserMessage as current request (Prop 5)
  // compaction: summary retains named entities; trigger is token-based (Prop 6,8)
  // model: reply is flat JSON within ~25s, model is non-reasoning (Prop 7)
END FOR
```

### Preservation Checking

**Goal**: For all inputs where the bug condition does NOT hold, the fixed system
produces the same result as the original.

**Pseudocode:**
```
FOR ALL input WHERE NOT isBugCondition(input) DO
  ASSERT iris_original(input) == iris_fixed(input)
END FOR
```

**Testing Approach**: Property-based testing is recommended for preservation
because it generates many inputs across the domain, catches edge cases manual
tests miss, and gives strong guarantees that non-memory routes are unchanged.
The existing `simulate-agent.py` scenario harness is the generator surface: its
routing scenarios (jokes, weather, calendar, tasks, gmail, web_search,
reminders, detailed answers) are the non-triggering inputs whose first-route and
delivery must be identical before and after the fix.

**Test Plan**: Capture current behavior on the UNFIXED code for non-memory
routes and the delivery pattern, then assert the fixed code matches.

**Test Cases**:
1. **Non-memory routing unchanged**: All existing `simulate-agent.py` scenarios
   keep the same `first_type` and pass their asserts (Req 3.5).
2. **Fail-open**: With neither backend available (`@memProxyOn == "no"` AND
   `@memLocalOn == "no"` — e.g. proxy unconfigured and backend forced to `sheets`,
   or Files unavailable and forced to `local`), a "remember X" turn returns an
   `ok=false` "not set up" observation and Iris still answers normally; a normal
   chat is unaffected (Req 3.6).
3. **Delivery pattern**: Answers still strip markdown, flatten newlines,
   neutralize "?", and end on the stop-word regex (Req 3.8).
4. **Budget reset**: A non-stop follow-up still zeroes tool/question/repair
   budgets (Req 3.9).
5. **Compiler invariants**: `validate-shortcut.py` passes — no `else if`
   (balanced control-flow groups), no `rawaction`, `nvapi-REPLACE-ME`
   placeholder present, NIM planner call present (Req 3.3, and the build gate).

### Unit Tests

- Proxy `memory_*` ops (Apps Script): `memoryAppend` then `memoryRead` returns
  the body; the appended row has the six OKF columns
  (`timestamp | topic | type | title | tags | body`) with `type` derived from
  topic and a server-stamped timestamp; `memoryRead(format=okf)` reconstructs a
  valid OKF concept block; `memoryStatus` counts rows; `memorySummary` caps at
  600 chars; unknown topic falls back to `log`; sheet/tab auto-created on first
  call.
- Local Files store: appending the OKF block to `Shortcuts/IrisOKF/<topic>.md`
  produces frontmatter (`type/title/tags/timestamp`) + body; `getFile` reads it
  back as opaque text; folder auto-created via `createAgentFolder`.
- Backend selector normalization: `@memoryBackendRaw` values (with whitespace /
  mixed case, and an unrecognized value) normalize to one of `hybrid/local/sheets`
  with unrecognized → `hybrid`; `@memLocalOn`/`@memProxyOn` derive correctly for
  each selector value crossed with `@proxyOk` 0/1.
- Token estimator: `chars/4` over protocol + context + request crosses
  `@compactAtTokens` at the expected size; small conversations do not trigger.
- Topic path resolution: each allowlisted topic maps correctly; default is `log`.

### Property-Based Tests

- **OKF round-trip (both stores)**: for any topic in the allowlist and any
  non-empty body, the stored entry (local file block and proxy row) is a valid
  OKF concept whose body reconstructs to the input body (Prop 1, 4).
- **Hybrid write-through**: for `@memoryBackend=hybrid` with both backends
  available, a write appends the local file FIRST and THEN the proxy sheet, and
  succeeds if either leg succeeds (Prop 1).
- **Local-only / sheets-only**: for `@memoryBackend=local` only the local file is
  written; for `sheets` only the sheet is written; the other store is untouched
  (Prop 11).
- **Local-first read with proxy fallback**: for any topic, when the local copy is
  empty/missing but the proxy has the entry, `read(topic)` returns the proxy
  value; when the local copy is present it is used without a proxy call (Prop 3,
  4).
- **Selector routing**: for any normalized `@memoryBackend`, writes/reads route
  per Property 11 and an unrecognized value behaves as `hybrid` (Prop 11).
- **Preservation**: for any non-memory scenario in the harness, fixed and
  original produce the same route and delivery (Prop 9).
- **Fail-open**: for any turn with neither backend available, the run completes
  with a spoken answer and never halts (Prop 10).
- **Compaction recency**: for any generated multi-turn conversation, the current
  request presented to the planner equals the most recent user message (Prop 5),
  and post-compaction context still contains the seeded named entity (Prop 6).

### Integration Tests

- **On-device write-location probe (manual, settles `@memBase`)**: run the setup
  primer on a real device, let the probe write `{@memBase}/probe.md`, then open
  the Files app and confirm which service it landed in — On My iPhone (Local
  Storage) vs iCloud Drive under `Shortcuts/IrisOKF/`. Confirm the probe reads
  the marker back (`@probeChars > 0`), reconcile `@memBase` to the confirmed
  location, and record the result in the device-test report. This both refutes
  the old "built-in file write silently no-ops" claim (a fixed-path write DOES
  land) and verifies the exact service, which Cherri cannot select from the
  emitted action. Low-risk: whatever it shows, the Sheet mirror covers
  durability (Prop 3).
- **Hybrid write-both / read-local-first routing**: with `@memoryBackend=hybrid`
  and both backends available, confirm a single "remember X" writes the local
  file FIRST and the proxy sheet SECOND (both contain the same OKF block), and
  that a follow-up `memory_read` is served from the local file with NO proxy
  network call (inspect for the absence of the `memory_read` GET); then empty the
  local copy and confirm the next read falls back to the proxy.
- **Hybrid write-through, cross-backend recall**: deploy the proxy; with
  `@memoryBackend=hybrid` run "remember I prefer tea", confirm the entry lands in
  BOTH the local `preferences.md` and the proxy sheet as OKF, relaunch Iris (new
  session), ask "what do I prefer to drink", and confirm recall.
- **Local-first read with proxy fallback (iCloud-full case)**: simulate an empty
  local store (or a device where local never synced) with the proxy populated;
  confirm `memory_read` falls back to the proxy and recalls the value.
- **Local-only and sheets-only**: force each backend via the Text box and confirm
  writes/reads use only that store and still recall correctly.
- Full follow-up flow: "when is the next match" → web_search answer → "tell me
  more about that match" resolves to the same match.
- Long conversation crosses the token threshold, compaction fires once on a
  follow-up boundary, and a later "that match" still resolves.
- Whole flow runs from a voice trigger with no picker/popup, on the chosen
  primary model, within the ~25s budget per call.
