# Implementation Plan

## Overview

Bugfix workflow: explore (tests that fail on unfixed code) → preserve (tests
that pass on unfixed code) → implement the fix → verify. Tasks are ordered by
dependency: the memory backend prerequisites (selector + local Files actions +
server proxy ops) land first (the client dispatch depends on all three), then
the in-shortcut client edits, then validation and docs.

> **Design revision (hybrid memory).** Decision 1 was rewritten from
> **proxy-only** to a **HYBRID, user-selectable backend** (`design.md` Decision 1
> rewrite). On-device testing refuted the old "built-in Files always no-op"
> root cause: with a fixed text path the built-ins DO persist to iCloud Drive
> under `Shortcuts/`. The real remaining risk is a silent iCloud-sync loss when
> iCloud Drive is full. So memory now: (a) reads a reliable `@memoryBackend`
> selector (`auto`/`local`/`proxy`), (b) **write-through** to every available
> backend, (c) reads **local-first with proxy fallback**, and (d) stores the
> **same OKF-formatted concept entry** in both stores. The already-implemented
> in-shortcut fixes (recent-message context, token compaction, model swap) are
> unchanged by the hybrid; the memory tasks (server ops, client dispatch,
> bootstrap) are re-scoped, and two new prerequisite tasks are added (backend
> selector; reintroduced local Files custom actions).

Cherri constraints (from `shortcuts/iris.cherri` header) apply to every
client edit: no `else if` (flat guarded `if` / default-then-override),
`jsonRequest`/`downloadURL` with LITERAL dict bodies/headers, never
`getDictionary()` on non-JSON (only the outer envelope; local OKF file text and
proxy `records` are opaque), coerce JSON with `getDictionary()` + `getValue()`
per level, keep the protocol compact and every call fast (~25s budget).

> **Build note:** the Cherri → `.shortcut` compile runs in CI (GitHub Actions,
> `.github/workflows/build-shortcut.yml`); a local Windows build is not
> available. Structural validation (`scripts/validate-shortcut.py`) runs against
> the CI-produced unsigned artifact. Tasks that need a deployed proxy or a
> hands-free Siri run are marked **[MANUAL / ON-DEVICE]**.

## Tasks

- [x] 1. Write bug-condition exploration tests (BEFORE any fix)
  - **Property 1: Bug Condition** - Single-backend memory fragility, lost follow-up, count-based compaction, weak/slow model
  - **CRITICAL**: These tests MUST FAIL (or surface the counterexample) on the UNFIXED code - failure confirms the bugs exist. DO NOT fix the test or the code when they fail.
  - **NOTE**: These encode the expected post-fix behavior; they validate the fix when they pass later.
  - **GOAL**: Surface concrete counterexamples for each defect.
  - **Scoped PBT approach** (deterministic reproduction): scope each property to a concrete failing case.
  - Memory (corrected root cause): the phone-plist assertion in `tests/test_memory_context_bugs.py` was updated to a **passing root-cause regression guard** - it decodes `tmp/iris-newshortcut6.plist.json` and asserts a WORKING phone-built `file.append` carries BOTH a relative `WFFilePath` AND a device `fileLocation` object (documenting why a bare-path built-in write is fragile). Keep this test as-is (passes). The memory bug is now single-backend fragility: add a `simulate-agent.py` scenario that does `memory_append(topic,body)` then a later `memory_read(topic)` and asserts the read records contain `body`; this fails on unfixed code (nothing recallable across a fresh store / no mirror). (isBugCondition: `op IN memory_* AND (NOT persistedDurably OR NOT recallableCrossDevice OR silentSyncLossUnrecovered)`)
  - Lost follow-up: add a `simulate-agent.py` scenario "when is the next match" → web_search → "tell me more about that match" and assert the current request presented to the planner equals the most recent user message (fails on unfixed code - `user_request` stays the original). (isBugCondition: `isFollowUp AND plannerCurrentRequest != mostRecentUserMessage`)
  - Count-based compaction: drive a conversation past 4 follow-ups; assert the compaction trigger is token-based and that a seeded named entity survives the summary (fails on unfixed code - trigger is `@exchangesSinceCompact`, entity can drop). (isBugCondition: `contextGrew AND compactionTrigger == fixed_exchange_count`)
  - Model contract: run the tool-routing scenarios on `meta/llama-3.1-8b-instruct` and record flat-JSON contract breaks; note a reasoning/70B model busts ~25s or emits `<think>` (fails on unfixed code). (isBugCondition: `plannerModel IS weakInstruct OR reasoningModel OR latency > 25s`)
  - Run on UNFIXED code. **EXPECTED OUTCOME**: tests FAIL / counterexamples recorded (except the plist regression guard, which passes and documents the root cause). Document each counterexample.
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8_

- [x] 2. Write preservation tests (BEFORE any fix)
  - **Property 2: Preservation** - Non-memory routes, fail-open, delivery pattern, budget reset, compiler invariants unchanged
  - **IMPORTANT**: Follow observation-first methodology - capture behavior on UNFIXED code, then assert the fixed code matches.
  - Non-memory routing: run every existing `scripts/simulate-agent.py` scenario (jokes, weather, calendar, tasks, gmail, web_search, reminders, detailed) and record each `first_type` + passing asserts as the baseline (Req 3.5).
  - Fail-open: record that with NEITHER backend available (proxy unconfigured AND local unavailable/forced off) a normal chat is unaffected and completes with a spoken answer, never halts (Req 3.6, 3.1).
  - Delivery: record that answers strip markdown, flatten newlines, neutralize "?", and end on the stop-word regex (Req 3.8).
  - Budget reset: record that a non-stop follow-up zeroes tool/question/repair budgets (Req 3.9).
  - Compiler invariants: run `python scripts/validate-shortcut.py <unsigned artifact>` on the current CI build and record it passes (balanced groups / no `else if`, no `rawaction`, `nvapi-REPLACE-ME` placeholder present, NIM planner call present) (Req 3.2, 3.3).
  - Run on UNFIXED code. **EXPECTED OUTCOME**: tests PASS (this is the baseline to preserve).
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9_

- [ ] 3. Fix: hybrid memory (selector + write-through + local-first read + OKF) + recent-message context + token compaction + model swap

  > **Numbering note:** 3.5–3.8 keep their original IDs and remain complete
  > (unchanged by the hybrid). 3.1–3.4 are re-scoped for the hybrid. 3.11 and
  > 3.12 are NEW prerequisite tasks added for the hybrid; per the Task Dependency
  > Graph they run in the same early wave as 3.1 (before the client edits in 3.3
  > and the bootstrap in 3.4).

  - [x] 3.1 Rework the proxy `memory_*` handlers to OKF columns in the Apps Script proxy (Design Decision 1b, server side)
    - In `docs/apps-script-proxy.md` §4, change the memory constants/columns to the OKF frontmatter shape: `MEMORY_HEADER = ['timestamp','topic','type','title','tags','body']` (was `timestamp|topic|body`), keep `MEMORY_SHEET_NAME='Iris Memory'`, `MEMORY_TAB='memory'`, `MEMORY_TOPICS=['index','profile','preferences','log','notes','journal']`, `MAX_MEMORY_ROWS=20`.
    - Add `_okfType(topic)` (topic→OKF type map: log→Memory Entry, preferences→User Preference, profile→Profile, notes→Note, journal→Journal, index→Index) and `_rowToOkf(r)` (reconstruct an OKF concept block from a `[ts,topic,type,title,tags,body]` row). Update `_memorySheet()` to write the six-column header (self-healing so the first write succeeds), and add `_readTopicRows(topic)` returning full row arrays; keep `_readTopic(topic)` returning bodies.
    - `memoryAppend`: append `[ISO-now, topic, _okfType(topic), title, tags, body]` where `title` = optional first line of body (≤80 chars), `tags` optional/empty, timestamp server-stamped; require non-empty body. `memoryRead`: default returns bodies joined newest-last (plain text `records`), and when `req.format === 'okf'` returns `rows.map(_rowToOkf).join('\n\n')` reconstructed OKF blocks; `ok=false` when none. `memoryList` uses `topic: body` (body now column index 5). `memoryStatus` = row count minus header. `memorySummary` = profile + preferences + last 3 index bodies, hard-cap 600 chars. All keep the `_env(tool,ok,count,result,records,error)` envelope.
    - Keep the `handle(req)` dispatch lines (`memory_append`, `memory_read`, `memory_list`, `memory_status`, `memory_summary`) reachable via both `doGet` query params and `doPost`; keep the `MAX_RESULTS`/`MAX_MEMORY_ROWS` caps and the 600-char summary cap. Keep the §3 manifest Sheets/Drive scopes (`spreadsheets`, `drive`). Update the §4 prose + the curl validation block in `docs/apps-script-proxy.md` to show the OKF columns and the `format=okf` read.
    - _Bug_Condition: isBugCondition(input) where op IN memory_* AND (NOT recallableCrossDevice OR silentSyncLossUnrecovered) - the proxy is the durable, iCloud-quota-immune mirror_
    - _Expected_Behavior: durable append-only OKF-shaped rows; self-healing store; row round-trips to an OKF concept via _rowToOkf; format=okf reconstruction; body still recalled_
    - _Preservation: existing tasks_/gmail_/calendar_ ops and the six-key envelope unchanged (Req 3.5); model-facing surface stays memory_append(topic,body) with no new args_
    - _Requirements: 2.1, 2.4_

  - [x] 3.11 Add the memory backend selector config in `shortcuts/iris.cherri` (Design Decision 1a) **[NEW - hybrid prerequisite]**
    - Add an editable Text config action next to the API-key/proxy config: `@memoryBackendRaw = text("auto")`; normalize exactly like the other config values - `@memoryBackendLower = replaceText('\s+', "", "{@memoryBackendRaw}", false, true)` then `@memoryBackend = lowercase("{@memoryBackendLower}")`.
    - Default-then-override validation (no `else if`, Req 3.3): set `@memBackendOk = "no"`, then three flat `if @memoryBackend == "auto"/"local"/"proxy" { @memBackendOk = "yes" }`, then `if @memBackendOk == "no" { @memoryBackend = "auto" }` so an unrecognized value falls back to `auto` (Prop 11).
    - Derive the two availability flags used by every memory dispatch: `@memProxyOn` = yes when (`@memoryBackend == "proxy"` OR `"auto"`) AND `@proxyOk > 0` (nested flat `if`s, never `else if`); `@memLocalOn` = yes when `@memoryBackend == "local"` OR `"auto"` (local Files are confirmed working, no runtime probe; a silent sync loss is covered by the proxy mirror under `auto`).
    - Comment why a per-run VOICE prompt is deliberately NOT used (hands-free Siri only reliably supports Ask for Input; asking every run is a picker-like gate violating Req 3.1 and cannot persist a choice) - a Text action is the persistence surface that already works.
    - _Bug_Condition: isBugCondition where the user needs to force/confirm a backend and no reliable selector exists (single-backend fragility, no user control)_
    - _Expected_Behavior: @memoryBackend normalizes to auto/local/proxy with unknown→auto; @memLocalOn/@memProxyOn derive from the selector crossed with @proxyOk, routing writes/reads per Property 11_
    - _Preservation: config-block pattern matches the existing S-GPT Text-action config; no `else if`; hands-free happy path ungated (Req 3.1, 3.3)_
    - _Requirements: 2.1, 2.3_

  - [x] 3.12 Reintroduce the local Files custom actions + topic→path + OKF block in `shortcuts/iris.cherri` (Design Decision 1c) **[NEW - hybrid prerequisite]**
    - Declare the built-in Files custom actions with Cherri's `action` form (never `rawAction()`): `createAgentFolder` (`is.workflow.actions.file.createfolder`, `WFFilePath`), `appendAgentFile` (`is.workflow.actions.file.append`, `WFInput`+`WFFilePath`, params `WFAppendFileWriteMode=Append`, `WFFileAppendNewLine=true`), `getAgentFile` (`is.workflow.actions.documentpicker.open`, `WFGetFilePath`, params `WFShowFilePicker=false`, `WFFileErrorIfNotFound=false`). Replace the "Memory is proxy-backed / no Files actions" comment block with a note that the local path uses fixed-text-path built-ins (confirmed persisting to iCloud Drive under `Shortcuts/`) and that file content is OPAQUE OKF Markdown, never `getDictionary()`'d (Req 3.2, 3.7).
    - Reintroduce `@memBase = "Shortcuts/IrisOKF"` and resolve the per-topic path with flat default-then-override `if`s (no `else if`): `@memPath = "{@memBase}/log.md"` default, then one `if @topic.text == "index"/"profile"/"preferences"/"log"/"notes"/"journal" { @memPath = ... }` per topic.
    - Build the OKF concept block the client appends locally: derive `@okfType` from topic via flat `if`s (default "Memory Entry"; preferences→User Preference, profile→Profile, notes→Note, journal→Journal, index→Index), then `@okfBlock = "\n---\ntype: {@okfType}\ntitle: \ntags: \ntimestamp: {@nowRaw}\n---\n{@body}\n"` (timestamp stamped by the shortcut, title/tags optional; model supplies only `body`).
    - _Bug_Condition: isBugCondition where op IN memory_* AND the local backend was removed (no fast/offline store, no self-healing folder before first write)_
    - _Expected_Behavior: fixed-path Files actions available for write-through/local-first read; topic→path + OKF-block resolution mirror the proxy OKF columns so both stores hold the same concept_
    - _Preservation: no `else if`; no `rawAction`; local file text stays opaque and never reaches `getDictionary()` (Req 3.2, 3.3, 3.7)_
    - _Requirements: 2.1, 2.2, 2.4_

  - [ ] 3.2 Validate the proxy `memory_*` ops (curl / unit check) **[MANUAL / ON-DEVICE: requires a deployed proxy]**
    - After deploying the updated script (new version on the existing `/exec` deployment), run `curl -L` GETs: `memory_append(topic=notes,body=...)` → `memory_read(topic=notes)` returns records containing the body; `memory_read(topic=notes,format=okf)` returns a reconstructed OKF concept block (frontmatter `type/title/tags/timestamp` + body).
    - Assert the appended row has the six OKF columns (`timestamp|topic|type|title|tags|body`) with `type` derived from the topic and a server-stamped timestamp; `memory_status` count increments by one per append; `memory_summary` caps at 600 chars; an unknown topic (e.g. `topic=xyz`) falls back to `log`; the sheet + `memory` tab are auto-created on the first call (delete the sheet, call `memory_append`, confirm it reappears with the six-column header).
    - _Bug_Condition: single-backend fragility - proxy mirror must be durable and OKF-shaped_
    - _Expected_Behavior: append→read round-trip; OKF row + format=okf reconstruction; self-heal; capped summary; topic fallback_
    - _Requirements: 2.1, 2.2, 2.4_

  - [x] 3.3 Rework the client memory tools to the HYBRID dispatch in `shortcuts/iris.cherri` (Design Decision 1d, client side)
    - `memory_append` = **write-through** guarded by the selector flags (3.11): when `@memLocalOn == "yes"` do `createAgentFolder("{@memBase}")` then `appendAgentFile("{@okfBlock}", "{@memPath}")` (append-only OKF block); when `@memProxyOn == "yes"` build `@proxyCallUrl = "{@irisProxyUrl}?secret={@proxySecretEnc}&op=memory_append&topic={@encTopic}&body={@encBody}"` with `urlEncode` and `downloadURL` GET, reading `ok` via `getDictionary()`+`getValue()` on the outer envelope only. Track `@memWroteAny`; report `ok=true` when either backend was written, else an `ok=false` "not set up - pick a backend" envelope with a fail-open `@localFinalText` (Req 3.6).
    - `memory_read` / `memory_list` / `memory_status` = **local-first, proxy fallback**: when `@memLocalOn == "yes"` read the local file with `getAgentFile("{@memPath}")` (OPAQUE text; `@memChars = count(...)`, never `getDictionary()`); only when the local result is empty (`@memChars == 0`) AND `@memProxyOn == "yes"` call the proxy op and read `records` from the envelope. Build the `tool=.../ok=/count=/result=/records=/error=` observation from whichever backend answered. Keep `memory_read` / `memory_list` in the `@speakRecords = "yes"` set so retrieved items are spoken.
    - `create_note` (topic=notes) and `quick_journal` (topic=journal) reuse the same write-through path (local OKF append AND proxy POST, honoring the selector); fail-open when neither backend is available.
    - Setup primer: reintroduce local seeding - `createAgentFolder("{@memBase}")` + a seed `appendAgentFile` per topic file so the first real write never hits a missing folder and the Files-permission dialog is answered during the one manual unlocked run; keep the existing `tasks_list` GET that warms the proxy host. The primer also **speaks one line**: "Memory uses both your phone and the cloud by default. To use only one, edit the memory backend box to local or proxy." Never pass local file text or `records` to `getDictionary()` (Req 3.2, 3.7).
    - _Bug_Condition: isBugCondition where op IN [memory_append,memory_read,memory_list,memory_status,create_note,quick_journal] AND single-backend fragility (no mirror / silent sync loss)_
    - _Expected_Behavior: write-through persists to every selected backend as an OKF entry; read is local-first with proxy fallback so an iCloud-full silent sync loss is recovered; write-then-read returns the body; fail-open when neither backend is available_
    - _Preservation: tasks_/gmail_ blocks and the six-key envelope contract unchanged (Req 3.5); getDictionary guard preserved on the outer envelope only, opaque memory text (Req 3.2, 3.7); no `else if` (Req 3.3)_
    - _Requirements: 2.1, 2.2, 2.4, 3.6_

  - [x] 3.4 Rework the startup memory bootstrap to local-first in `shortcuts/iris.cherri` (Design Decision 2)
    - Before the `repeat` loop (after the setup-primer block, before the first planner `jsonRequest`), init `@memorySummary = ""`. **Local-first:** when `@memLocalOn == "yes"` read `getAgentFile("{@memBase}/profile.md")` and `.../preferences.md` (opaque text), join, and if non-empty set `@memorySummary`. **Proxy fallback:** only when `count("{@memorySummary}") == 0` AND `@memProxyOn == "yes"`, `downloadURL` the `memory_summary` op and read `records`. If `@memorySummary` is non-empty, append `\n\nmemory_summary={@memorySummary}` into `@loopContext` once. Fail-open: when neither backend answers, leave `@loopContext` unchanged.
    - The 600-char server cap (and keeping the local bootstrap files compact) prevents fattening every call toward ~25s (Req 3.4); the summary is injected as observation data, not instructions (Req 3.7); local file text is opaque and never `getDictionary()`'d (Req 3.2).
    - _Bug_Condition: memory recall unavailable at session start; local-only recall lost on iCloud-full silent sync failure_
    - _Expected_Behavior: prior profile/preferences injected once, local-first with proxy fallback (covers the iCloud-full case), fail-open_
    - _Requirements: 2.4, 3.4, 3.6, 3.7_

  - [x] 3.5 Answer the most recent message: introduce `@currentRequest` (Design Decision 3)
    - Add `@currentRequest = "{@request}"` at init.
    - Repoint the planner user message from `user_request={@request}` to `user_request={@currentRequest}`.
    - In the delivery block, on a non-stop follow-up (replacing today's `@loopContext += "...assistant_answer=...user_followup=..."`), fold the prior exchange into `@loopContext` as `previous_exchange=\nuser_said={@currentRequest}\nassistant_answered={@finalText}` and set `@currentRequest = "{@followup}"`. Keep zeroing `@toolCallCount`/`@userQuestionCount`/`@repairCount` (Req 3.9).
    - Full prior conversation stays in `@loopContext`; the planner is led by the latest message so "tell me more about that match" is answered as a follow-up.
    - _Bug_Condition: isFollowUp AND plannerCurrentRequest != mostRecentUserMessage_
    - _Expected_Behavior: planner answers most recent message with full history retained_
    - _Preservation: budget reset on non-stop follow-up unchanged (Req 3.9); delivery pattern unchanged (Req 3.8)_
    - _Requirements: 2.5, 3.9_

  - [x] 3.6 Token-based compaction in `shortcuts/iris.cherri` (Design Decision 4)
    - Add config constants: `@modelContextTokens = 128000`, `@contextTokenBudget = 12000`, `@compactAtTokens = 9600` (~80% of the latency-safe working window).
    - Replace the `@exchangesSinceCompact >= @maxContextExchanges` trigger with `@approxChars = count("{@protocol}") + count("{@loopContext}") + count("{@currentRequest}")`; `@approxTokens = @approxChars / 4`; `if @approxTokens >= @compactAtTokens { ...existing compaction jsonRequest... }`.
    - Remove `@maxContextExchanges` and `@exchangesSinceCompact` (declaration, the follow-up increments, and the reset).
    - Strengthen the compaction system prompt to preserve named entities: "...preserve the user's overall goal, the MOST RECENT request, and every named entity or specific needed to resolve later references - proper nouns, titles, people, places, dates, numbers, and any 'that X' antecedent. Plain text, under 90 words, no markdown, no JSON."
    - _Bug_Condition: contextGrew AND compactionTrigger == fixed_exchange_count_
    - _Expected_Behavior: compaction triggers on estimated token usage (~80% budget) and preserves follow-up entities_
    - _Preservation: compaction still runs only on a follow-up boundary; ~25s budget honored (Req 3.4)_
    - _Requirements: 2.6, 2.8, 3.4_

  - [x] 3.7 Swap the planner model in `shortcuts/iris.cherri` (Design Decision 5)
    - Set `@nimModelIdRaw = text("mistralai/mistral-small-3.1-24b-instruct-2503")`.
    - Set `@modelContextTokens = 128000` (chosen model's true window; see 3.6).
    - Update the MODEL CHOICE RULE comment to name the primary (`mistralai/mistral-small-3.1-24b-instruct-2503`) and fallback (`qwen/qwen2.5-7b-instruct`, both 128k, non-reasoning), keeping the "never a reasoning/thinking model" rule.
    - _Bug_Condition: plannerModel IS weakInstruct OR reasoningModel OR latency > 25s_
    - _Expected_Behavior: capable non-reasoning instruct model that holds the flat-JSON contract within ~25s_
    - _Requirements: 2.7, 3.4_

  - [x] 3.8 Update the protocol tool-registry text (only if the memory arg surface changed)
    - In the `@protocol` string in `shortcuts/iris.cherri`, keep the memory tool registry compact and accurate: `memory_read(topic)`, `memory_append(topic,body)`, `memory_list(no args)`, `memory_status(no args)`, topic allowlist `index profile preferences log notes journal` (default `log`). The hybrid keeps the model-facing surface at `memory_append(topic,body)` with NO new args (title/tags/type are derived server-side and client-side, not by the model), so no protocol change is required beyond confirming the allowlist is already `index profile preferences log notes journal`.
    - Keep the identical copy in `scripts/simulate-agent.py` `PROTOCOL` in sync (its sync check fails loudly on drift).
    - _Bug_Condition: N/A (registry consistency for memory ops)_
    - _Expected_Behavior: registry matches the hybrid memory arg surface (topic,body only), protocol stays compact_
    - _Preservation: all other tool registry lines unchanged (Req 3.5); flat-JSON contract intact (Req 3.2)_
    - _Requirements: 2.1, 3.2, 3.5_

  - [ ] 3.9 Verify the bug-condition exploration tests now pass
    - **Property 1: Expected Behavior** - Memory persists via hybrid write-through + local-first read, follow-up answered, token compaction preserves entities, model holds contract
    - **IMPORTANT**: Re-run the SAME tests from task 1 - do NOT write new tests.
    - Run the memory round-trip (write-through + local-first read + proxy fallback + OKF round-trip), follow-up, compaction, and model scenarios from task 1 (memory round-trip and proxy checks are **[MANUAL / ON-DEVICE]** where they need a deployed proxy and/or a device; the follow-up and compaction scenarios run offline in `simulate-agent.py`; the plist regression guard stays passing).
    - **EXPECTED OUTCOME**: tests PASS (confirms each bug is fixed, including hybrid persistence/recall).
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8_

  - [ ] 3.10 Verify the preservation tests still pass
    - **Property 2: Preservation** - Non-memory routes, fail-open (neither backend), delivery, budget reset, compiler invariants unchanged
    - **IMPORTANT**: Re-run the SAME tests from task 2 - do NOT write new tests.
    - Run all `simulate-agent.py` non-memory scenarios (same `first_type` + asserts), the fail-open case (neither backend available), delivery, budget-reset, and `scripts/validate-shortcut.py` on the CI artifact (no `else if`, no `rawaction` despite the reintroduced Files `action` declarations, placeholder present, planner call present).
    - **EXPECTED OUTCOME**: tests PASS (no regressions).
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9_

- [ ] 4. Extend validation harness and run invariants
  - Extend `scripts/simulate-agent.py` to cover the hybrid: (a) `@memoryBackend=auto` write-through asserts BOTH stores are written; (b) `local`-only writes/reads use only the emulated Files store; (c) `proxy`-only uses only the emulated Sheet; (d) local-first read with proxy fallback when the local copy is empty; (e) OKF-format round-trip (a stored entry reconstructs to the input body); (f) selector normalization (whitespace/mixed case, unknown→auto). Keep the existing follow-up scenario (current request equals the most recent user message), the memory round-trip (append→read returns body), the fail-open assertion (neither backend → normal answer, no halt), and the token-based compaction + named-entity survival scenario. Keep the emulated per-run memory store, the selector/`@memLocalOn`/`@memProxyOn` mirrors, and keep `PROTOCOL` in sync with the shortcut (the arg surface is still `memory_append(topic,body)`).
  - Run `scripts/validate-shortcut.py` against the CI-produced unsigned artifact and confirm all structural invariants pass (Req 3.3, and the build gate), including that the reintroduced Files `action` declarations do not emit `rawaction`.
  - Note: the Cherri compile itself happens in CI (GitHub Actions); a local Windows build is unavailable, so run the compile-dependent validation on the CI artifact.
  - _Requirements: 2.1, 2.3, 2.4, 2.5, 2.6, 2.8, 3.3, 3.5, 3.6_

- [ ] 5. Sync documentation
  - Update `docs/memory-system-design.md` to describe the HYBRID store: the `@memoryBackend` selector (auto/local/proxy), write-through to both backends, local-first read with proxy fallback, and OKF-formatted entries in BOTH the phone-local Files store and the proxy Google Sheet (replacing the proxy-only description).
  - Update `docs/okf-knowledge-base.md` to reflect that OKF concept entries are stored in both backends (local `Shortcuts/IrisOKF/<topic>.md` OKF blocks and the proxy sheet's OKF columns `timestamp|topic|type|title|tags|body`), with the model-facing surface still `memory_append(topic,body)`.
  - Update `docs/architecture.md` and `agentic-loop.md` to reflect the hybrid memory (selector, write-through, local-first read, OKF in both stores), the `@currentRequest` recent-message context, and token-based compaction (replacing the proxy-only / exchange-count descriptions).
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.8_

- [ ] 6. Checkpoint - ensure all tests pass
  - Confirm Property 1 (Expected Behavior) and Property 2 (Preservation) both pass, `validate-shortcut.py` is green on the CI artifact, and the offline `simulate-agent.py` suite (auto/local/proxy, write-through, local-first fallback, OKF round-trip, selector normalization, follow-up, compaction, fail-open) passes. **[MANUAL / ON-DEVICE]** confirm end-to-end on device: with `@memoryBackend=auto`, deploy the proxy, run "remember I prefer tea", confirm the entry lands in BOTH the local `preferences.md` and the proxy sheet as OKF; relaunch Iris (new session), ask "what do I prefer to drink" and confirm cross-session recall; simulate an empty local store with the proxy populated and confirm the local-first read falls back to the proxy (iCloud-full case); force `local` and `proxy` via the Text box and confirm each uses only that store; run the full flow hands-free via Siri with no picker/popup within budget. Ask the user if questions arise.

- [ ] 7. Hybrid backend + OKF (supersedes the proxy-only memory backend)

  Design change: memory moves from proxy-only to a HYBRID (phone-local Files +
  Google Sheet proxy) backend with a one-time `@memoryBackend` selector
  (`hybrid`/`local`/`sheets`, default `hybrid`), OKF-formatted entries in BOTH
  stores, local-first read/bootstrap with proxy fallback, and an on-device
  write-location probe. Tasks 3.1/3.3/3.4 (proxy-only memory) are reused and
  extended, not undone. The non-memory fixes (3.5 `@currentRequest`, 3.6 token
  compaction, 3.7 model swap) are unchanged.

  - [x] 7.1 Extend the proxy memory ops to OKF-shaped rows (Decision 1b, server)
    - In `docs/apps-script-proxy.md`: change the memory sheet header to
      `timestamp | topic | type | title | tags | body`; add `_okfType(topic)`,
      `_rowToOkf(row)`, `_readTopicRows(topic)`; update `memoryAppend` to stamp
      `type`/`title`/`tags`/`timestamp` server-side; update `memoryRead` to
      support `format=okf` (reconstructed concept) with the voice `body` join as
      default. Update the curl/unit note.
    - _Expected_Behavior: a row round-trips to a valid OKF concept; model still sends only topic+body_
    - _Requirements: 2.1, 2.4_

  - [x] 7.2 Add the `@memoryBackend` selector + derived flags (Decision 1a, client)
    - In `shortcuts/iris.cherri`, add editable Text `@memoryBackendRaw` (default
      `hybrid`), normalize to `@memoryBackend` (lowercased, unrecognized→hybrid via
      flat guarded ifs), and derive `@memProxyOn` (sheets|hybrid AND `@proxyOk>0`)
      and `@memLocalOn` (local|hybrid).
    - _Requirements: 2.1, 2.3_

  - [x] 7.3 Reintroduce local Files custom actions + path/OKF builders (Decision 1c)
    - Declare `createAgentFolder`/`appendAgentFile`/`getAgentFile` (built-ins, no
      rawAction). Add `@memBase = "Shortcuts/IrisOKF"`, per-topic `@memPath`
      resolution, and the `@okfBlock` builder (type from topic, timestamp stamped,
      title/tags optional). Local content is opaque text; never `getDictionary()`.
    - _Requirements: 2.1, 2.2, 2.4, 3.2, 3.7_

  - [x] 7.4 Rework client memory dispatch to hybrid (Decision 1d)
    - `memory_append`: write-through — local append FIRST when `@memLocalOn`, then
      proxy mirror when `@memProxyOn`; succeed if either leg succeeds; fail-open
      "not set up" only when neither. `memory_read`/`memory_list`/`memory_status`:
      local-first, proxy fallback only when local is empty. Repoint `create_note`
      (notes) and `quick_journal` (journal) through the same write-through. Keep
      `memory_read`/`memory_list` in the `@speakRecords="yes"` set. Flat guarded
      ifs; literal-dict `downloadURL`; records never `getDictionary()`'d.
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 3.2, 3.6, 3.7_

  - [x] 7.5 Rework startup bootstrap to local-first, proxy fallback (Decision 2)
    - Read local `profile.md`/`preferences.md` first when `@memLocalOn`; only call
      the proxy `memory_summary` when local produced nothing and `@memProxyOn`.
      Inject once into `@loopContext`; fail-open.
    - _Requirements: 2.4, 3.4, 3.6, 3.7_

  - [x] 7.6 Setup primer: local seeding + write-location probe + backend hint (Decision 1c/1d)
    - `createAgentFolder(@memBase)` + seed per-topic files; run the probe (write
      `{@memBase}/probe.md`, read back, tell the user to check iCloud Drive vs On
      My iPhone under Shortcuts/IrisOKF); speak the one-line backend hint. Keep the
      existing proxy/tasks_list warm-up.
    - **[MANUAL / ON-DEVICE]** for the probe verification.
    - _Requirements: 2.2, 3.1, 3.6_

  - [x] 7.7 Update tests + simulator for the hybrid backend
    - Extend `tests/test_memory_context_bugs.py`, `tests/test_memory_context_preservation.py`,
      and `scripts/simulate-agent.py` for: selector normalization/routing
      (hybrid/local/sheets, unrecognized→hybrid), hybrid write-through (local then
      proxy), local-first read with proxy fallback, OKF round-trip in both stores,
      and fail-open when neither backend is available. Keep all prior tests green.
      Re-run the suite.
    - _Requirements: 2.1, 2.3, 2.4, 3.5, 3.6_

  - [x] 7.8 Update docs for the hybrid backend
    - `docs/memory-system-design.md`, `docs/okf-knowledge-base.md`,
      `docs/architecture.md`, `agentic-loop.md`: describe the hybrid backend, the
      `@memoryBackend` selector, OKF-in-both-stores, local-first read, and the
      corrected root-cause narrative (built-in Files writes DO land in iCloud
      Drive; prior failures were missing folder + wrong location, not an
      impossible location object).
    - _Requirements: 2.1, 2.3_

  - [ ] 7.9 Checkpoint — hybrid end-to-end **[MANUAL / ON-DEVICE]**
    - Redeploy the proxy (OKF columns), run the probe, confirm iCloud-vs-local
      landing and reconcile `@memBase`; verify hybrid write-both + read-local-first
      and cross-session recall hands-free with no picker.
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

## Task Dependency Graph

Waves group tasks that can proceed once their predecessors are complete. Within
a wave, tasks are independent of each other.

```json
{
  "waves": [
    {
      "wave": 1,
      "name": "Baseline on unfixed code",
      "tasks": ["1", "2"],
      "dependsOn": []
    },
    {
      "wave": 2,
      "name": "Memory prerequisites: backend selector, local Files actions, server OKF ops",
      "tasks": ["3.11", "3.12", "3.1"],
      "dependsOn": ["1", "2"]
    },
    {
      "wave": 3,
      "name": "Proxy deploy validation (manual/on-device)",
      "tasks": ["3.2"],
      "dependsOn": ["3.1"]
    },
    {
      "wave": 4,
      "name": "Client hybrid memory + bootstrap + context + compaction + model",
      "tasks": ["3.3", "3.4", "3.5", "3.6", "3.7"],
      "dependsOn": ["3.1", "3.11", "3.12"]
    },
    {
      "wave": 5,
      "name": "Protocol registry sync",
      "tasks": ["3.8"],
      "dependsOn": ["3.3", "3.4", "3.5", "3.6", "3.7"]
    },
    {
      "wave": 6,
      "name": "Verify fix and preservation",
      "tasks": ["3.9", "3.10"],
      "dependsOn": ["3.8"]
    },
    {
      "wave": 7,
      "name": "Validation harness and invariants",
      "tasks": ["4"],
      "dependsOn": ["3.9", "3.10"]
    },
    {
      "wave": 8,
      "name": "Docs sync",
      "tasks": ["5"],
      "dependsOn": ["4"]
    },
    {
      "wave": 9,
      "name": "Checkpoint (manual/on-device end-to-end)",
      "tasks": ["6"],
      "dependsOn": ["5"]
    }
  ],
  "notes": [
    "3.11 (backend selector) and 3.12 (local Files actions) are NEW hybrid prerequisites; together with 3.1 (server OKF ops) they form wave 2 and block the client dispatch (3.3) and bootstrap (3.4).",
    "3.1 (server ops) also blocks 3.2 (deploy validation).",
    "3.3 (client hybrid dispatch) depends on the selector flags (@memLocalOn/@memProxyOn from 3.11), the local Files actions + OKF block (3.12), and the server OKF ops (3.1). 3.4 (bootstrap) depends on 3.11 + 3.12.",
    "3.5-3.7 are already complete and unchanged by the hybrid; 3.5 (@currentRequest) precedes 3.6 (the token estimator counts @currentRequest) and 3.7 (model swap) sets @modelContextTokens used by 3.6.",
    "3.8 depends on the final memory arg surface from 3.3 (which stays memory_append(topic,body), so it is a confirmation).",
    "MANUAL / ON-DEVICE: 3.2 (proxy deploy + curl), the proxy/device-dependent half of 3.9, and 6 (hybrid recall + Siri hands-free run) require a deployed proxy and/or a physical device."
  ]
}
```

## Notes

- **Bug condition methodology**: Property 1 (tasks 1, 3.9) covers the four
  bug-condition families (single-backend memory fragility, lost follow-up,
  count-based compaction, weak/slow model); Property 2 (tasks 2, 3.10) covers
  preservation of all non-triggering behavior. The exploration tests must FAIL on
  unfixed code and PASS after the fix (the plist test is a passing root-cause
  regression guard); the preservation tests must PASS both before and after.
- **Hybrid memory**: `@memoryBackend` (auto/local/proxy) drives write-through to
  every available backend and local-first reads with proxy fallback; both stores
  hold the same OKF concept entry. The model-facing surface stays
  `memory_append(topic,body)` only.
- **Cherri constraints**: no `else if`, literal-dict `jsonRequest`/`downloadURL`
  bodies, Files custom actions declared via `action` (never `rawAction`), never
  `getDictionary()` on the opaque local OKF text or the plain-text proxy
  `records` field, keep the protocol compact and every call within the ~25s iOS
  budget.
- **Build**: local Windows cannot run the Cherri compiler; the compile and the
  compiled-artifact validation run in CI (GitHub Actions). Structural checks use
  the CI unsigned artifact.
- **Manual/on-device**: deploying the Apps Script proxy (new version on the
  existing `/exec` deployment), the `curl` round-trip checks (including
  `format=okf`), cross-session and cross-backend recall (auto write-through,
  local-first fallback, forced local/proxy), and the hands-free Siri run require
  a deployed proxy and/or a device.
- **Secrets**: never bake `nvapi-`/`tvly-` keys or the proxy secret into source;
  `validate-shortcut.py` enforces the `nvapi-REPLACE-ME` placeholder.
