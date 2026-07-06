---
type: Design
title: Iris Phone-Local Memory System Design
summary: A working, hands-free memory layer for Iris built on device-validated Files actions, mapped to Cherri custom actions and the bounded tool loop.
status: design-ready-pending-device-validation
tags:
  - iris
  - memory
  - okf
  - apple-shortcuts
  - files
  - non-interactive
related:
  - shortcuts/iris.cherri
  - docs/okf-knowledge-base.md
  - agentic-loop.md
  - shortcut-syntax-reference.md
  - tmp/iris-memory-ref/shortcut.json
sources:
  - https://matthewcassinelli.com/actions/get-file/
  - https://matthewcassinelli.com/actions/save-file/
  - https://matthewcassinelli.com/actions/append-to-text-file/
  - tmp/iris-memory-ref/shortcut.json
---

# Iris Phone-Local Memory System Design

> **UPDATE 2026-07-06 — IMPLEMENTED, root cause fixed.** The earlier approach in
> this doc (hand-rolled `action '...'` custom definitions with community-guessed
> WFKeys) was the bug: keys like `WFShowFilePicker`, `WFFileAppendNewLine`, and
> `WFDestinationPath` were wrong/ineffective, so the path never got set and
> **nothing was ever stored** on device. The path also had a bad leading slash
> (`/Shortcuts/IrisOKF`).
>
> **Fix:** `iris.cherri` now uses **Cherri's built-in file actions** via
> `#include 'actions/documents'` — `createFolder(path)`, `appendToFile(path, &text)`,
> `getFile(path)` — which emit the correct, maintainer-validated WFKeys
> (`WFFilePath`, `WFGetFilePath`, `WFAppendFileWriteMode`). The base path is
> `Shortcuts/IrisOKF` (NO leading slash; resolves under the iCloud Drive
> Shortcuts sandbox). The setup primer runs `createFolder` then seeds the
> readable topic files (log/index/profile/preferences) so `getFile` never halts
> on a missing file. `appendToFile`'s text argument MUST be a variable, not a
> string literal. Verified in the compiled plist: correct keys present, bad keys
> absent. Sections below are retained for history.

This is a design document only. It does not modify `shortcuts/iris.cherri` and
nothing here is compiled. It specifies a **working, hands-free** phone-local
memory layer for Iris, replacing the aspirational (and never-implemented) OKF
tool set in `docs/okf-knowledge-base.md` with concrete, device-verified action
shapes and Cherri custom-action definitions that follow the existing pattern in
`iris.cherri`.

The single most important requirement: memory must work **under Siri, hands
free, with no picker popups**. The reference shortcut the user captured
(`tmp/iris-memory-ref/shortcut.json`) uses the *interactive* picker variants,
which prompt the user. Section 2 documents the *non-interactive* parameter
variants that make read/write silent.

Content licensing note: action parameter details below were confirmed against
Matthew Cassinelli's Shortcuts action directory and rephrased for compliance;
exact `WFKey` strings that are not visible in the reference plist are marked
**must validate on device**.

---

## 1. What the reference shortcut actually contains (interactive variants)

Extracted verbatim from `tmp/iris-memory-ref/shortcut.json`. This is a rough
syntax fixture (a chained demo), not a working flow, so treat it as a source of
**identifiers and parameter keys**, not control flow.

| Order | Action identifier | Parameters present in fixture | Meaning |
| --- | --- | --- | --- |
| 1 | `is.workflow.actions.file.getfoldercontents` | `UUID` only (no `WFFolder`) | List a folder; with no `WFFolder` it prompts / uses default |
| 2 | `com.apple.DocumentsApp.SearchFile` | `AppIntentDescriptor` (Team `0000000000`, Bundle `com.apple.DocumentsApp`, Name `Files`, Intent `SearchFile`) | Files app "Search" App Intent |
| 3 | `is.workflow.actions.properties.files` | `UUID` only | Get details/properties of files |
| 4 | `is.workflow.actions.file.getfoldercontents` | `WFFolder` = attachment to action 3 output ("Details of Files") | List folder from a piped value |
| 5 | `is.workflow.actions.documentpicker.save` | `WFInput` = attachment to action 4 ("Folder Contents") | **Save File** — no `WFAskWhereToSave`, so it PROMPTS |
| 6 | `is.workflow.actions.documentpicker.open` | `WFFile` = attachment to action 5 ("Saved File") | **Get File** — no `WFShowFilePicker:false`, so it PROMPTS |
| 7 | `is.workflow.actions.file.append` | `WFInput` = text token attaching action 6 ("File") | **Append to Text File** — no `WFFilePath`, so path is unset |
| 8 | `is.workflow.actions.previewdocument` | `WFInput` = attachment to action 7 | Preview (interactive) |

Key takeaways:

- The fixture confirms the **identifiers** we need: `documentpicker.save`,
  `documentpicker.open`, `file.append`, `file.getfoldercontents`,
  `com.apple.DocumentsApp.SearchFile`, `properties.files`.
- Every file action in the fixture is in its **interactive** form: `save`
  without `WFAskWhereToSave`, `open` with `WFFile` (an input attachment) instead
  of a fixed path, and `append` without a `WFFilePath`. Wiring these into Iris
  as-is is exactly what would make Siri pop a document picker and stall a
  hands-free run. Section 2 fixes that.
- The `documentpicker.open` in the fixture uses `WFFile` (it opens an
  already-produced file object). For a **fixed-path** read we do NOT use
  `WFFile`; we use `WFShowFilePicker:false` + a path key (below).

---

## 2. Non-interactive (hands-free) parameter variants — CRITICAL

These are the parameter variants that let memory run silently under Siri. Toggle
labels are confirmed from the Shortcuts action directory; the exact `WFKey`
strings for the fields hidden behind those toggles are the widely-used community
key names and are marked **must validate on device** where the reference plist
does not show them.

### 2.1 Read a file at a FIXED path without a picker

Action: **Get File from Folder** — `is.workflow.actions.documentpicker.open`.
Confirmed toggles: "Show Document Picker" (turn OFF to specify a path),
"Select Multiple". The action *takes no input* and *returns a File*.

| Purpose | WFKey | Value | Confidence |
| --- | --- | --- | --- |
| Turn the picker off (silent) | `WFShowFilePicker` | `false` | High (toggle confirmed; key widely documented) |
| Fixed path to read | `WFGetFilePath` | e.g. `Shortcuts/IrisOKF/index.md` | **Must validate on device** (path field key) |
| Don't halt if file missing | `WFFileErrorIfNotFound` | `false` | **Must validate on device** |

Notes:
- When `WFShowFilePicker` is `false`, do **not** pass `WFFile`; the path field
  drives the read. (The reference fixture's `WFFile` form is the interactive
  "open this produced file" variant and is not what we want.)
- Setting error-if-not-found to `false` is essential: a first-run empty KB must
  return nothing, not throw a runtime error that Siri reports as
  "Something went wrong".
- The returned value is a **file**; coerce to text before use. Never run
  `getDictionary()` on it (see root cause 3.3).

### 2.2 Save a file to a FIXED path without prompting

Action: **Save File** — `is.workflow.actions.documentpicker.save`. Confirmed
toggle: "Ask Where To Save" (turn OFF to specify a destination path in the
Shortcuts folder). Input: the content/file to save.

| Purpose | WFKey | Value | Confidence |
| --- | --- | --- | --- |
| Content to write | `WFInput` | text token | High (confirmed in fixture) |
| Don't prompt for location | `WFAskWhereToSave` | `false` | High (toggle confirmed) |
| Fixed destination path | `WFDestinationPath` | e.g. `Shortcuts/IrisOKF/log.md` | **Must validate on device** — alt key seen in the wild: `WFSaveFileDestinationPath` |
| Overwrite existing file | `WFOverwriteIfExists` | `true` | **Must validate on device** — alt key: `WFSaveFileOverwrite` |

Notes:
- Save File **overwrites**; it is a whole-file write. For append semantics use
  2.3, or read-modify-save (2.1 → concat → 2.2) as a fallback.
- Because there are two candidate key names for both the path and the overwrite
  toggle, device validation must confirm which pair the installed iOS build
  emits (decode a phone-built shortcut and compare).

### 2.3 Append text to a file at a fixed path (preferred write primitive)

Action: **Append to Text File** — `is.workflow.actions.file.append`. Confirmed
fields: Append/Prepend mode, Service, **File Path** (text, e.g. `/example.txt`),
"Make New Line" toggle. Input: Text. **Crucially, the action auto-creates the
file if it does not exist** (per Apple's action notes) — so it doubles as a safe
"create if missing" primitive.

| Purpose | WFKey | Value | Confidence |
| --- | --- | --- | --- |
| Text to append | `WFInput` | text token | High (confirmed in fixture) |
| Fixed path (auto-created if absent) | `WFFilePath` | e.g. `Shortcuts/IrisOKF/log.md` | **Must validate on device** (path field key) |
| Append vs prepend | `WFAppendFileWriteMode` | `"Append"` | **Must validate on device** |
| Add a newline first | `WFFileAppendNewLine` | `true` | **Must validate on device** (alt: `WFFileAppendNewline`) |

This is the **recommended write path** for Iris memory: it is append-only (the
safety default OKF asks for), it self-heals a missing file/folder file, and it
carries a fixed path so it never prompts.

### 2.4 Listing a folder — DO NOT use a text path (device-confirmed failure)

Action: `is.workflow.actions.file.getfoldercontents`, param `WFFolder`. This was
originally assumed to accept a text path, but on device it **requires a folder
OBJECT** (a value piped from another action). Passing a text path such as
`/Shortcuts/IrisOKF` throws the runtime error **"File Is Not a Folder — Please
pass a folder to the Get Folder Contents action instead of a standard file"**,
which halts the run. This was the root cause of on-device memory failure: the
permission primer and `memory_status`/`memory_list` all called it with a text
path, so the primer halted before seeding the folder and nothing persisted.

**Fix (implemented):** stop calling `getfoldercontents` with a text path. Ensure
the folder exists with **Create Folder** (`is.workflow.actions.file.createfolder`,
param `WFFilePath`), which *does* take a text path and runs hands-free; and
implement `memory_status`/`memory_list` by reading the log file with Get File
(2.1) instead of listing the folder. To enumerate a folder later, obtain a
folder object upstream and pipe it into `WFFolder`.

### 2.5 Search Files (App Intent) — experimental only

Action: `com.apple.DocumentsApp.SearchFile` carries an `AppIntentDescriptor`
block (Bundle `com.apple.DocumentsApp`, Intent `SearchFile`). Two problems make
this **experimental**:

1. It is a third-party-style App Intent whose search-term parameter key is not
   visible in the fixture (**must validate on device**).
2. Cherri custom actions emit the `WFWorkflowActionIdentifier` but do **not**
   reliably emit the `AppIntentDescriptor` block; without it iOS may treat the
   action as unsupported (same failure class flagged for `rawAction` in
   `shortcut-syntax-reference.md`).

**Decision:** implement `memory_search` on top of `getfoldercontents` + name
matching (Section 5), not on `SearchFile`. Keep `SearchFile` as a documented
future option once a phone-built descriptor is decoded.

---

## 3. Why the previous attempt failed (root-cause diagnosis)

Grounded in the runtime rules in the `iris.cherri` header comments,
`agentic-loop.md`, and `okf-knowledge-base.md`.

### 3.1 Only `memory_status` was ever implemented
`iris.cherri` wires exactly one memory tool — `memory_status` — which calls
`getAgentFolderContents("/Shortcuts/IrisOKF")` and reports a count. There is no
read, no write, no append, no bootstrap. `okf-knowledge-base.md` itself flags
`memory_lookup`, `memory_list_topics`, `memory_propose_write`, and
`memory_append_log` as **planned**. So "memory" never persisted or recalled
anything; it only counted folder entries. That is the primary reason it "did not
work": the capability was documented but absent.

### 3.2 Interactive picker blocks hands-free Siri
The reference the user captured uses `documentpicker.save`/`.open` in their
prompting form. Under Siri, `iris.cherri`'s own conversation model note says the
only reliable primitive is Ask for Input; a document-picker popup mid-run stalls
or silently fails a hands-free/locked invocation. Any earlier build that used
the picker variants would appear to "hang" or do nothing when triggered by
voice. Fix: the fixed-path, no-prompt variants in Section 2.

### 3.3 `getDictionary()` halting on non-JSON memory files
The header COMPILER/RUNTIME rules are explicit: **Get Dictionary from Input
HALTS the Shortcut on invalid JSON**, which is exactly why the agent route text
is regex-validated before parsing. OKF files are **Markdown with YAML
frontmatter**, not JSON. Any earlier attempt that read a memory file and piped
it through `getDictionary()` (the same reflex used for API responses) would halt
the run on the first `#` heading. Memory content must be treated as **opaque
text**, never parsed as a dictionary.

### 3.4 Fixed iCloud-vs-local path ambiguity
`okf-knowledge-base.md` already warns the fixed paths "must be validated on
device". There are two ambiguities:
- **Base**: the append action's own docs use a path relative to the service root
  (`/Public/notes.txt`), while `getfoldercontents` is called with
  `/Shortcuts/IrisOKF`. Is the base the iCloud Drive root or the `Shortcuts`
  folder? A wrong base silently reads/writes the wrong place or nothing.
- **Location**: if iCloud Drive is disabled, the `Shortcuts` folder is
  **On My iPhone** (local), and the same path string resolves differently.
A build that assumed one base would read an empty/absent file, get nothing back,
and look "broken".

### 3.5 Files permission first-run gate
The `iris.cherri` permission-primer comment states iOS consent is per-action,
per-shortcut, first-use, and **cannot be scripted away**. The very first file
read/write/append fires a Files permission dialog. Under hands-free/locked Siri
that dialog cannot be answered, so the first memory operation fails. Any earlier
attempt that skipped a **manual** priming run would hit this gate on the first
real use.

### 3.6 Whole-file overwrite races / data loss
If write was attempted with Save File (overwrite) instead of append, any
read-modify-write that lost the read (see 3.3/3.4) would overwrite the KB with
partial content. Append-only avoids this entirely.

**Summary:** the feature was mostly unimplemented (3.1), and the parts that were
sketched used interactive actions (3.2), risked halting on non-JSON (3.3),
depended on an unvalidated path base (3.4), and would trip the first-run
permission gate under voice (3.5).

---

## 4. Cherri custom action definitions (ready to paste)

These follow the exact existing pattern in `iris.cherri` (compare
`action 'is.workflow.actions.addnewreminder' addAgentReminder(text title: 'WFCalendarItemTitle', text ?notes: 'WFCalendarItemNotes'): text`
and `getAgentFolderContents`). Place them alongside the other custom actions,
after `getAgentFolderContents`. Do **not** use `rawAction()`.

```cherri
/* --- Memory (Files) custom actions ---
   Non-interactive, fixed-path variants so memory runs hands-free under Siri.
   Parameter WFKeys behind picker toggles are community-standard names and are
   marked "must validate on device" in docs/memory-system-design.md. Decode a
   phone-built Save File / Get File / Append shortcut and reconcile the keys
   before shipping. */

/* Read a file at a fixed path, no picker. WFShowFilePicker off + path field.
   error-if-not-found off so a missing file returns empty instead of halting. */
action 'is.workflow.actions.documentpicker.open' getAgentFile(text path: 'WFGetFilePath'): text {
  "WFShowFilePicker": false,
  "WFFileErrorIfNotFound": false
}

/* Overwrite-save a whole file at a fixed path, no prompt. */
action 'is.workflow.actions.documentpicker.save' saveAgentFile(text content: 'WFInput', text path: 'WFDestinationPath'): text {
  "WFAskWhereToSave": false,
  "WFOverwriteIfExists": true
}

/* Append text to a fixed path; auto-creates the file if absent. Preferred
   write primitive (append-only, self-healing, never prompts). */
action 'is.workflow.actions.file.append' appendAgentFile(text content: 'WFInput', text path: 'WFFilePath'): text {
  "WFAppendFileWriteMode": "Append",
  "WFFileAppendNewLine": true
}

/* List a folder at a fixed path. (Already present as getAgentFolderContents;
   reuse it — shown here for completeness, do not redeclare.) */
/* action 'is.workflow.actions.file.getfoldercontents' getAgentFolderContents(text folder: 'WFFolder'): text */

/* Get details/properties of files (name, path, size...). Optional, supports a
   richer memory_list. Property name key must be validated on device. */
action 'is.workflow.actions.properties.files' getAgentFileDetails(text input: 'WFInput'): text
```

Caveats to verify when these compile:
- Confirm Cherri accepts a mapped parameter **and** a default body dict on the
  same action (the reminder action uses only params; `getupcomingevents` uses
  only a body). If mixing is rejected, move the toggles into the call site by
  splitting into two actions, or hardcode via a body-only action plus a piped
  path token.
- Confirm boolean literals (`true`/`false`) are accepted in the action body (the
  existing bodies use strings and numbers). If not, use `1`/`0`.
- `com.apple.DocumentsApp.SearchFile` is intentionally **not** declared here
  (Section 2.5): Cherri will not emit its `AppIntentDescriptor`.

---

## 5. Agent-facing memory tool registry

All memory tools use the standard Iris tool-result envelope
(`tool=/ok=/count=/result=/records=/error=`) and set `@localFinalText` for the
`return_to_agent:false` path, exactly like the existing tools. Keep
`memory_status` as-is. Add `memory_save`, `memory_append`, `memory_read`,
`memory_list`, `memory_search`.

Fixed base path constant (validate in 7.1): treat `Shortcuts/IrisOKF/` as the
KB root. Filenames are constrained by the Shortcut, never free-form from the
model, so the model cannot read/write arbitrary files.

### 5.1 Protocol/registry additions
Add to the `@protocol` tool list (keep it terse — every sentence slows the
call):

```
memory_read(topic), memory_list(no args), memory_search(query),
memory_append(topic,body), memory_save(topic,body), memory_status(no args)
```

Constrain `topic` to a small allowlist the Shortcut maps to real filenames:
`index`, `profile`, `preferences`, `log`, `projects`, `people`, `facts`,
`routines`. The model passes a topic word; the Shortcut resolves the path. This
keeps arbitrary file access impossible and matches the "fixed set of predeclared
tools, not arbitrary file access" rule in `okf-knowledge-base.md`.

### 5.2 Path resolution helper (sketch)
Resolve the model's `topic` to a fixed path with flat guarded `if`s (never
`else if`, per the COMPILER RULE). Default first, then override:

```cherri
@memBase = "Shortcuts/IrisOKF"
@memPath = "{@memBase}/log.md"          /* safe default */
@memTopicOk = "no"
if @topic.text == "index"       { @memPath = "{@memBase}/index.md"        @memTopicOk = "yes" }
if @topic.text == "profile"     { @memPath = "{@memBase}/profile.md"      @memTopicOk = "yes" }
if @topic.text == "preferences" { @memPath = "{@memBase}/preferences.md"  @memTopicOk = "yes" }
if @topic.text == "log"         { @memPath = "{@memBase}/log.md"          @memTopicOk = "yes" }
if @topic.text == "projects"    { @memPath = "{@memBase}/projects/iris.md" @memTopicOk = "yes" }
if @topic.text == "facts"       { @memPath = "{@memBase}/facts/loop-reliability.md" @memTopicOk = "yes" }
```

### 5.3 `memory_read`
- Arguments: `topic`.
- Dispatch sketch (place in the tool dispatch chain, flat `if` like the others):

```cherri
if @tool.text == "memory_read" {
  @toolHandled = "yes"
  @memPath = "Shortcuts/IrisOKF/index.md"     /* resolve via 5.2 */
  /* ...topic resolution ifs... */
  @memText = getAgentFile("{@memPath}")        /* WFShowFilePicker:false, no halt if missing */
  @memChars = count(@memText)                  /* NOTE: treat as opaque text; never getDictionary() */
  if @memChars > 0 {
    @toolResult = "tool=memory_read\nok=true\ncount=1\nresult=Read {@topic} memory.\nrecords={@memText}\nerror="
    @localFinalText = "Here is what I have saved about {@topic}: {@memText}"
  } else {
    @toolResult = "tool=memory_read\nok=false\ncount=0\nresult=\nrecords=\nerror=No memory found for that topic yet."
    @localFinalText = "I do not have anything saved about {@topic} yet."
  }
}
```
- Envelope (found): `tool=memory_read / ok=true / count=1 / result=Read <topic> memory. / records=<file text> / error=`
- Envelope (empty): `ok=false / count=0 / error=No memory found for that topic yet.`
- localFinalText: "Here is what I have saved about <topic>: <text>" / "I do not have anything saved about <topic> yet."

### 5.4 `memory_append` (primary write)
- Arguments: `topic`, `body`.
- Dispatch sketch:

```cherri
if @tool.text == "memory_append" {
  @toolHandled = "yes"
  @memPath = "Shortcuts/IrisOKF/log.md"        /* resolve via 5.2 */
  /* ...topic resolution ifs... */
  @memEntry = "\n\n---\ntimestamp: {@currentDate}\ntype: Memory Entry\n\n{@body}"
  @memWrite = appendAgentFile("{@memEntry}", "{@memPath}")   /* auto-creates if missing */
  @toolResult = "tool=memory_append\nok=true\ncount=1\nresult=Saved to {@topic} memory.\nrecords={@body}\nerror="
  @localFinalText = "Saved that to your {@topic} memory."
}
```
- Envelope: `tool=memory_append / ok=true / count=1 / result=Saved to <topic> memory. / records=<body> / error=`
- localFinalText: "Saved that to your <topic> memory."
- Safety: append-only; the model supplies `body`, the Shortcut supplies the
  timestamp/frontmatter. Per OKF privacy rules, stable personal facts should be
  confirmed with the user first — do this by having the agent `ask_user` before
  emitting the `memory_append` tool_call (control stays in the loop, not a new
  action). Sensitive categories (credentials, health, financial) stay out of the
  alpha.

### 5.5 `memory_save` (whole-file overwrite — guarded)
- Arguments: `topic`, `body`.
- Same shape as append but calls `saveAgentFile(content, path)` and **replaces**
  the file. Use only for regenerating `index.md`/`profile.md`. Because it is
  destructive, prefer append; keep `memory_save` behind an explicit user
  confirmation via `ask_user`.
- Envelope: `tool=memory_save / ok=true / count=1 / result=Rewrote <topic> memory. / records=<body> / error=`
- localFinalText: "I updated your <topic> memory."

### 5.6 `memory_list`
- Arguments: none.
- Dispatch (implemented): `@memListText = getAgentFile("/Shortcuts/IrisOKF/log.md")`
  then branch on truthiness. Does **not** use `getfoldercontents` (see 2.4).
- Envelope (has content): `tool=memory_list / ok=true / count=1 / result=Read memory log. / records=<log text> / error=`
- Envelope (empty): `ok=true / count=0 / result=Memory log is empty.`
- localFinalText: "Here is what is in your memory log: <text>" / "Your memory log is empty so far."

### 5.7 `memory_search`
- Arguments: `query`.
- Implementation: list the folder (and known subfolders) with
  `getAgentFolderContents`, then match `query` against file names / listed text
  with `matchText` (the same regex primitive the loop already uses). Do **not**
  use `com.apple.DocumentsApp.SearchFile` (Section 2.5). Optionally read the top
  matching file with `getAgentFile` to return a snippet.
- Envelope (hit): `tool=memory_search / ok=true / count=<n> / result=Found <n> memory matches. / records=<matched names/snippets> / error=`
- Envelope (miss): `ok=false / count=0 / error=No memory matched that query.`
- localFinalText: "I found <n> things in memory about <query>." / "I did not find anything in memory about <query>."

### 5.8 `memory_status` (implemented)
Reads the log file with Get File (`getAgentFile("/Shortcuts/IrisOKF/log.md")`)
and reports whether memory has content — it does **not** list the folder (see
2.4, which caused the "File Is Not a Folder" halt). Envelope:
`tool=memory_status / ok=true / count=1 / result=Memory is available. / records=<log text> / error=`
when the log has content, else `count=0 / result=Memory is set up but empty.`

### 5.9 Unknown-tool behavior
The existing `@toolHandled == "no"` fallthrough already returns a controlled
`ok=false` unknown-tool envelope, so the model can never invoke a memory
operation the Shortcut did not predeclare. No change needed.

---

## 6. Startup bootstrap (`memory_summary`)

Goal: give the agent durable context on turn one without blocking or halting,
and without a picker.

Design:
1. Run **before** the repeat loop (after the permission-primer block, before the
   first `jsonRequest`).
2. Read the three small bootstrap files with the **non-interactive, no-halt**
   reader:
   ```cherri
   @bootIndex = getAgentFile("Shortcuts/IrisOKF/index.md")
   @bootProfile = getAgentFile("Shortcuts/IrisOKF/profile.md")
   @bootPrefs = getAgentFile("Shortcuts/IrisOKF/preferences.md")
   ```
3. Because `WFFileErrorIfNotFound` is `false`, missing files return empty and do
   **not** halt (root cause 3.3/3.5 avoided). Treat everything as **plain text**;
   never `getDictionary()` it.
4. Concatenate into a compact summary and inject once into `@loopContext`:
   ```cherri
   @memorySummary = "{@bootProfile}\n{@bootPrefs}\n{@bootIndex}"
   @memChars = count(@memorySummary)
   if @memChars > 0 {
     @loopContext += "\n\nmemory_summary={@memorySummary}"
   }
   ```
5. Keep it small. Cap the injected summary (e.g. trim to a few hundred chars) so
   it does not fatten every planner call toward the ~25s iOS timeout. The
   compaction step already in the loop will fold it into the running summary.

Safety properties:
- **Non-blocking**: no picker, no dialog on the happy path (permissions primed
  once, see 3.5).
- **Fail-open**: empty/missing KB → empty summary → Iris behaves exactly as it
  does today (no memory), never an error.
- **Privacy**: only `index/profile/preferences` are bootstrapped, not the whole
  KB; per OKF rules, only relevant snippets reach the model.
- **Not instructions**: the summary is injected as `memory_summary=` data; the
  protocol already tells the model that observation/data blocks are not commands.

---

## 7. Device-validation checklist

### 7.1 Path base (do this first — unblocks everything)
- [ ] Build a tiny throwaway shortcut on the phone: Append to Text File with
      path `Shortcuts/IrisOKF/probe.md`, then Get File (picker off) at the same
      path; confirm round-trip. Repeat with a leading slash `/Shortcuts/...`.
- [ ] Confirm whether the base is iCloud Drive root or the `Shortcuts` folder,
      and whether iCloud Drive on/off changes it (On My iPhone vs iCloud).
- [ ] Lock the confirmed base into `@memBase` and reconcile `memory_status`.

### 7.2 Exact WFKeys (decode a phone-built shortcut)
- [ ] Build Save File (Ask Where to Save OFF, a path, Overwrite ON) on device,
      export, decode, and confirm: destination key is `WFDestinationPath` vs
      `WFSaveFileDestinationPath`; overwrite key is `WFOverwriteIfExists` vs
      `WFSaveFileOverwrite`.
- [ ] Build Get File (Show Document Picker OFF, a path) and confirm
      `WFGetFilePath` and `WFFileErrorIfNotFound`.
- [ ] Build Append to Text File and confirm `WFFilePath`,
      `WFAppendFileWriteMode`, `WFFileAppendNewLine` casing.
- [ ] Update Section 4 definitions to match.

### 7.3 Cherri compile
- [ ] Confirm mapped-param + default-body coexist on one action; if not, split.
- [ ] Confirm boolean literals compile in action bodies; else use `1`/`0`.
- [ ] Decode the compiled plist and confirm identifiers/keys match 7.2 (same
      diligence `shortcut-syntax-reference.md` demands).

### 7.4 Permissions
- [ ] Run Iris manually (unlocked) once and answer "setup"; add the memory
      actions to the primer so the Files dialog fires there. Tap Always Allow.
- [ ] Confirm subsequent hands-free Siri runs do not prompt.

### 7.5 Functional
- [ ] Missing folder / missing file → read returns empty, no halt.
- [ ] `memory_append` creates `log.md` when absent, then appends.
- [ ] `memory_read` returns saved text; never runs `getDictionary()` on it.
- [ ] `memory_save` overwrites only the intended file.
- [ ] `memory_list` / `memory_search` return folder-based results.
- [ ] Bootstrap injects `memory_summary` and the first answer reflects it.
- [ ] Whole flow runs end-to-end from a **voice** trigger with no popup.

---

## 8. Open risks

1. **WFKey ambiguity** (save destination/overwrite; append path/mode/newline).
   Two candidate names each; wrong choice = silent no-op. Mitigated by 7.2.
2. **Path base / iCloud-vs-local** (root cause 3.4). Highest-risk unknown;
   validate before anything else (7.1).
3. **Cherri param+body mixing** may not compile; fallback is split actions or
   piped path tokens.
4. **SearchFile App Intent** likely won't carry its descriptor from Cherri;
   `memory_search` deliberately avoids it, but a future descriptor-based search
   needs a decoded phone build.
5. **Permission gate under locked Siri** (3.5). If the user skips manual
   priming, the first memory op fails; UX must nudge the one-time setup run.
6. **Prompt bloat**: adding five tools + a bootstrap summary lengthens every
   planner call toward the ~25s timeout. Keep the registry terse and cap the
   summary; lean on existing compaction.
7. **Append newline/encoding**: `.md` is fine as UTF-8 text, but confirm the
   appended token doesn't inject the object-replacement char (`\ufffc`) seen in
   the fixture's text-token attachment.
8. **No update/delete** in alpha by design (append-only is safer). Structured
   concept-file create/update stays a later stage per OKF.
