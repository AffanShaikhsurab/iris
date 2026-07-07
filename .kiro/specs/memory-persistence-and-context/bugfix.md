# Bugfix Requirements Document

## Introduction

Iris is a hands-free voice assistant implemented as an Apple Shortcut (compiled
from `shortcuts/iris.cherri` via Cherri) whose planner is a stateless NVIDIA NIM
model called once per turn over an OpenAI-compatible endpoint. Several closely
related defects make the assistant unreliable in exactly the ways that matter
for a conversational agent:

- **Memory never persists.** Across roughly 45 attempts, nothing the assistant
  is asked to "remember" is ever written into the phone's Files store, and
  nothing is ever recalled on a later turn or in a later session.
- **In-conversation context is lost.** Within a single conversation the
  assistant forgets earlier turns — asking about a match and then saying "tell
  me more about that match" fails, because the follow-up is buried while the
  original request keeps leading every planner call.
- **The planner model is weak and slow.** The default small model breaks the
  flat-JSON contract and mishandles context, while more capable models routinely
  exceed the ~25s iOS network budget and abort the run.
- **Compaction triggers on the wrong signal.** Context is compacted on a fixed
  follow-up-exchange count instead of on how full the model's context window
  actually is.

Root-cause evidence for the memory defect comes from decoding the user's own
phone-built shortcut (`tmp/iris-newshortcut6.plist.json`): every working file
action carries an explicit `fileLocation` object (`WFFileLocationType =
"LocalStorage"`, `fileProviderDomainID = "com.apple.FileProvider.LocalStorage"`,
a `relativeSubpath` such as `shortcuts/iris`, plus a relative `WFFilePath`).
Cherri's built-in `appendToFile`/`getFile`/`createFolder` emit only the
`WFFilePath` text string and **no** `fileLocation` object, and `iris.cherri`
assumes an iCloud Drive base (`Shortcuts/IrisOKF`) while the device resolves to
On My iPhone / LocalStorage. So writes have no valid destination and silently
no-op. The `crossDeviceItemID` in a real location object is device-specific and
non-portable, which makes reliable phone-local Files persistence genuinely
fragile in a generated shortcut and strongly favors an external, proxy-backed
store (Plan B). The whole system must keep running hands-free under Siri (no
pickers, no `getDictionary()` on non-JSON, no `else if`, and fast planner calls).

## Bug Analysis

### Current Behavior (Defect)

What currently happens when each bug is triggered.

**Memory never persists**

1.1 WHEN the agent runs a memory write (`memory_append`, `create_note`, or `quick_journal`) via the built-in `appendToFile` THEN the system emits only a `WFFilePath` text string with no `fileLocation` object, so iOS has no valid destination and the write silently no-ops with nothing appearing in the Files folder.

1.2 WHEN a memory write or read runs before the manual "setup" primer has created the folder THEN the system fails silently because `appendToFile` does not create parent folders and the folder is only ever created in the setup branch.

1.3 WHEN the shortcut addresses `@memBase = "Shortcuts/IrisOKF"` assuming an iCloud Drive base THEN the device resolves file storage to On My iPhone / LocalStorage, so the path does not match where a file would actually live and reads/writes address an invisible or wrong location.

1.4 WHEN a later turn calls `memory_read`/`memory_list` to recall previously "saved" information THEN the system returns empty and reports it has nothing saved, because nothing was ever persisted (reproduced across ~45 attempts).

**In-conversation context lost within a session**

1.5 WHEN the user asks a follow-up that refers to an earlier turn (e.g. "tell me more about that match") THEN the system still leads every planner call with the original `user_request={@request}` while the follow-up is appended far down `@loopContext`, so the model keeps answering the original request and the follow-up reference is lost.

1.6 WHEN a conversation reaches `@maxContextExchanges` (4) follow-ups THEN the system replaces the raw running context with a short summary that can drop the specific detail a later follow-up depends on.

**Planner model quality and speed**

1.7 WHEN the default planner model `meta/llama-3.1-8b-instruct` is used THEN the system frequently receives replies that break the flat-JSON contract and mishandle context, yet swapping in a larger, more capable NIM model routinely exceeds the ~25s iOS network timeout and aborts the whole run.

**Compaction trigger**

1.8 WHEN the loop context grows THEN the system triggers compaction on a fixed follow-up-exchange count (`@maxContextExchanges`) rather than on actual token usage, so it fires without regard to how full the model's context window is.

### Expected Behavior (Correct)

What should happen instead. Each clause corresponds to the same-numbered defect above.

**Memory persists reliably**

2.1 WHEN the agent saves information THEN the system SHALL persist it to a durable external store that reliably survives across turns and sessions; because phone-local Files persistence is evidenced to be fragile (built-in file actions omit the required `fileLocation` object and the `crossDeviceItemID` is device-specific and non-portable), the system SHALL use a proxy-backed external store as the primary memory backend — the existing Google Apps Script proxy writing to a Sheet/Drive doc as the leading option, with Supabase as an alternative.

2.2 WHEN a memory write occurs and the backing store or container does not yet exist THEN the system SHALL ensure the target exists (create or self-heal) so the first write succeeds without requiring a prior manual setup run.

2.3 WHEN memory is stored THEN the system SHALL address a single reliable location that does not depend on an unvalidated iCloud-vs-local base path or a device-specific identifier.

2.4 WHEN a later turn recalls saved information THEN the system SHALL return the previously saved content, so that a write followed by a read of the same topic/key returns the written value within the same session and across sessions.

**In-conversation context is preserved**

2.5 WHEN the user asks a follow-up that refers to an earlier turn THEN the system SHALL keep the full running conversation available to the planner (every message appended) and present the user's most recent message as the current request, so the model answers the follow-up rather than the original request.

2.6 WHEN the running context must be reduced THEN the system SHALL compact in a way that preserves the specific entities and details needed to resolve follow-up references (e.g. "that match").

**Better and faster planner model**

2.7 WHEN choosing the planner model THEN the system SHALL use a model that is both more capable at instruction-following AND fast enough to reliably respond within the ~25s iOS budget, selected from the current NVIDIA NIM catalog with latency verified, and SHALL NOT use reasoning/thinking models that break the flat-JSON contract.

**Token-based compaction**

2.8 WHEN the loop context grows THEN the system SHALL trigger compaction when token usage reaches approximately 80% of the model's context length, rather than on a fixed follow-up-exchange count.

### Unchanged Behavior (Regression Prevention)

Existing behavior that must be preserved for inputs that do not trigger the bugs.

3.1 WHEN Iris is invoked hands-free by Siri THEN the system SHALL CONTINUE TO run with no pickers, dialogs, or interactive gates on the happy path.

3.2 WHEN a planner reply is parsed THEN the system SHALL CONTINUE TO require one flat single-line JSON object, regex-validate it before `getDictionary()`, and never pass non-JSON (such as memory text) to `getDictionary()`.

3.3 WHEN control flow is compiled THEN the system SHALL CONTINUE TO avoid `else if` chains, using flat guarded `if` blocks or default-then-override assignment.

3.4 WHEN a planner or tool call is made THEN the system SHALL CONTINUE TO keep each call fast enough to finish within the ~25s iOS network budget.

3.5 WHEN a non-memory tool runs (e.g. `web_search`, `calendar_lookup`, `reminders_lookup`, `weather_summary`, `tasks_*`, `gmail_*`, drafting, and `open_*` tools) THEN the system SHALL CONTINUE TO behave exactly as it does today.

3.6 WHEN the external memory store is unreachable or not configured THEN the system SHALL CONTINUE TO run and answer normally (fail-open, no halt), exactly as it behaves today with no memory.

3.7 WHEN memory content is retrieved THEN the system SHALL CONTINUE TO treat it as opaque user data rather than instructions, and send only relevant snippets to the model.

3.8 WHEN a final answer is delivered THEN the system SHALL CONTINUE TO use the Ask-for-Input S-GPT pattern (neutralizing "?", flattening newlines, stripping markdown) and end on a stop word or silence.

3.9 WHEN the user gives a non-stop follow-up reply THEN the system SHALL CONTINUE TO reset the per-request budgets (tool calls, user questions, repair turns).
