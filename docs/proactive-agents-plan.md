---
type: Implementation Plan
title: Proactive Iris — the Six-Role Agent Model + Automation Engine
summary: Comprehensive, buildable plan to turn Iris into a proactive AI chief of staff using iOS Time-of-Day Automations and a six-role agent model implemented as scheduled protocol variants that hand off through shared state (the OKF store / Sheet). Grounded in verified iOS automation behavior — proactive runs are UNATTENDED "compute + persist + notify" jobs (no Ask-for-Input), and the voice conversation only happens on attended runs. Supersedes the "morning automation speaks via Ask-for-Input" assumption in docs/chief-of-staff-vision.md.
status: proposal
tags:
  - iris
  - proactivity
  - automations
  - multi-agent
  - chief-of-staff
  - okf
related:
  - docs/chief-of-staff-vision.md
  - docs/shortcut-runtime-flow.md
  - docs/architecture.md
  - agentic-loop.md
  - docs/memory-system-design.md
  - docs/okf-knowledge-base.md
  - docs/apps-script-proxy.md
sources:
  - https://support.apple.com/en-us/guide/shortcuts/apd2175adcab/ios
  - https://www.idownloadblog.com/2022/02/01/run-shortcuts-automations-without-notifications-tutorial/
  - https://talk.automators.fm/t/why-do-some-time-triggered-shortcuts-run-on-a-locked-iphone-and-others-fail/18608
  - https://discussions.apple.com/thread/256078223
  - https://www.healthyapps.dev/blog/export-apple-health-data-to-rest-apis-mqtt-brokers-and-home-assistant
  - https://apple.stackexchange.com/q/404600
---

# Proactive Iris — the Six-Role Agent Model + Automation Engine

> Scope: the two things that matter — **(1) the agent model** and **(2) making
> Iris proactive** — and making the base flow rock-solid. The external-agent /
> plugin route (GPT Actions, connectors) is explicitly **out of scope** for now.
>
> This plan is grounded in the hard runtime rules in
> `docs/shortcut-runtime-flow.md` (no `else if`, ~25s network budget, one model
> call per turn, `getDictionary()` guard, fail-open) **and** in verified iOS
> automation behavior (§2). Where it contradicts the earlier
> `docs/chief-of-staff-vision.md`, this document wins. Vendor-behavior claims are
> rephrased for compliance with sources linked inline.

---

## Implementation status (updated)

- **Phase 1 (base flow)** — tracked in `.kiro/specs/memory-persistence-and-context`.
- **Phase 2 (mode plumbing + two-phase pattern)** — IMPLEMENTED in
  `shortcuts/iris.cherri`: `@mode` is read from `ShortcutInput` (default `chat`,
  chat path unchanged), and a `showAgentNotification`
  (`is.workflow.actions.notification`) custom action was added for unattended
  delivery.
- **Phase 3 (morning briefing)** — IMPLEMENTED in `shortcuts/iris.cherri` as a
  self-contained `morning` Phase-A branch (compute → persist to `log` →
  `showAgentNotification`, then `stop()`), placed before the interactive prompt
  so it never calls `Ask for Input`.
- **Pending validation (repo norm):** CI compile + on-device checks — that an
  automation passing `morning` lands in `ShortcutInput`, the notification WFKeys
  are correct, and unattended (locked) network calls succeed after priming.
- **Not yet built:** knowledge-graph topics (Phase 4), evening/Coach (Phase 5),
  weekly/Strategy (Phase 6). `evening`/`weekly` inputs currently fall back to
  `chat` on purpose until built.

---

## Part 0 — The single most important design decision

A Time-of-Day automation fires when the phone is typically **locked, in a
pocket, unattended**. Two verified facts collide:

1. A Time-of-Day automation **can run a shortcut with no tap** — since iOS 14 you
   turn off "Ask Before Running" / choose "Run Immediately," and it runs in the
   background without a notification-to-tap
   ([apple.stackexchange](https://apple.stackexchange.com/q/404600);
   [iDownloadBlog](https://www.idownloadblog.com/2022/02/01/run-shortcuts-automations-without-notifications-tutorial/)).
2. But any **interactive** action — `Ask for Input` (Iris's entire voice UI),
   Siri dictation, and even `Speak Text` in some cases — **fails or stalls when
   the device is locked/unattended**. Shortcuts that need interaction or an
   unlocked-only capability simply don't complete in that state
   ([automators.fm](https://talk.automators.fm/t/why-do-some-time-triggered-shortcuts-run-on-a-locked-iphone-and-others-fail/18608);
   [Apple discussions](https://discussions.apple.com/thread/256078223)). This is
   the same "Ask for Input needs an unlocked device" failure already documented
   in `docs/shortcut-runtime-flow.md`.

**Therefore every proactive run must be split into two phases:**

- **Phase A — Unattended (automation-triggered): compute → persist → notify.**
  No `Ask for Input`, no reliance on Siri speech. Gather data, call the planner
  once, **write the result to the OKF store / Sheet**, and deliver via
  **`Show Notification`** (the reliable unattended surface —
  [Apple: Show Notification](https://support.apple.com/en-us/guide/shortcut/apd2175adcab/ios)).
- **Phase B — Attended (user taps the notification, or says "Hey Siri, Iris"):**
  the normal voice loop runs, reads the **already-computed** briefing/reflection
  from the store, and has the spoken conversation. Now the device is unlocked, so
  `Ask for Input` works.

This is not a limitation to apologize for — it is the correct architecture. The
**Sheet becomes the handoff bus** between the unattended "thinking" phase and the
attended "talking" phase, and between the six roles.

```text
07:30  Automation fires (phone locked)
        │  Phase A (unattended)
        ▼
   gather Tasks+Calendar → planner call → write "today's plan" to OKF
        │
        ▼
   Show Notification: "Good morning. Your 1 thing today: ship the beta. 2 more + 3 emails. Tap to hear it."
        │  user taps (device unlocks)
        ▼
   Phase B (attended): Iris opens, reads today's plan from OKF, speaks it via Ask-for-Input, takes questions
```

---

## Part 1 — The six-role agent model (as scheduled protocol variants)

### 1.1 Why not a literal multi-agent swarm

A literal "six agents talking to each other in one run" is impossible here: each
model call must finish inside ~25s or Siri kills the run, and Iris makes exactly
**one** model call per turn (`docs/shortcut-runtime-flow.md`). So the six agents
are **roles**, each = a distinct **protocol prompt** + a distinct **entry point /
schedule**, all sharing one **source of truth** (the OKF store / Sheet). Roles
hand off through **durable state**, not a live conversation. This is the only
design that survives the runtime and stays auditable.

### 1.2 The roles

| # | Role | Trigger | Phase A (unattended) | Phase B (attended) | Reuses today |
| --- | --- | --- | --- | --- | --- |
| 1 | **Memory** | always | The OKF store itself (Files + Sheet). Read/written by every role. | same | Hybrid memory + proxy |
| 2 | **Planner** | 07:30 auto | Read goals+priorities+yesterday's plan, read Tasks+Calendar, pick top 1–3, write "today's plan" to `log`, notify. | Speak the plan, take questions. | Calendar/Tasks lookups, loop |
| 3 | **Observer** | 07:30 + 21:30 | Diff yesterday's plan vs what got done (Tasks/Calendar/Gmail), write "what slipped" to `log`. | Mention slips in the spoken briefing. | `tasks_list`, `gmail_search`, `calendar_lookup` |
| 4 | **Coach** | 21:30 auto | Detect the top slipped item, write a pending coaching question to `log`; notify "1 quick reflection?". | Ask the question(s), store answer under `reflections`. | The Ask-for-Input loop |
| 5 | **Strategy** | Sun 18:00 | Review projects vs goals vs `reflections`, write a "stop/keep" recommendation to `log`; notify. | Present the recommendation, act on the decision (mark project dropped). | Loop + memory read |
| 6 | **Execution** | on demand + within any run | Draft emails, add tasks, set reminders, schedule — **never auto-send**. | same | All of `docs/routes.md` |

**Memory and Execution already exist.** The build is really **Planner, Observer,
Coach, Strategy** as scheduled protocol variants, plus the two-phase pattern and
a richer knowledge graph (Part 4).

### 1.3 One shortcut, many modes (recommended) vs many shortcuts

Two ways to ship the roles. Recommendation: **one `iris.cherri`, a `@mode`
variable**, because Cherri has no imports and the base loop/guards/config should
not be duplicated.

- **Recommended — mode parameter.** The main shortcut reads an input/parameter
  into `@mode` (`chat` default | `morning` | `evening` | `weekly`). Each
  automation runs the same shortcut passing its mode. A small guard block near
  the top selects the protocol prompt and the phase (unattended vs attended) with
  **flat guarded `if`s** (never `else if`, per the compiler rule). The
  Ask-for-Input happy path stays exactly as today for `chat`.
- **Alternative — wrapper shortcuts.** "Iris Morning/Evening/Weekly" each just
  call the main shortcut with the mode text. More shortcuts to install, but each
  automation is dead simple. Use only if passing input from an automation proves
  awkward on-device.

Either way, **the protocol prompt swaps by mode** — that is what makes a role a
role. Keep every mode's prompt compact (the ~25s budget applies to all of them).

### 1.4 How a role "hands off" to the next

Roles never call each other. They read and append to the OKF `log` topic (and the
new topics in Part 4):

```text
Planner (07:30)   → appends: plan.today = [ship beta, reply recruiters, gym]
Observer (21:30)  → reads plan.today, diffs against tasks_list → appends: slipped = [ship beta]
Coach   (21:30)   → reads slipped[0] → appends: pending_reflection = "why not the beta?"
(user, attended)  → answers → appends: reflections/<date> = "felt too big"
Strategy (Sunday) → reads projects + reflections → appends: recommendation = "split beta into 3 tasks; keep"
```

The Sheet is the shared memory bus. Every write is append-only OKF (Part 4), so
nothing is destructively overwritten and the whole chain is auditable.

---

## Part 2 — The proactivity engine (verified iOS behavior)

### 2.1 What Time-of-Day automations can and cannot do

Grounded facts (rephrased from sources):

- **They can run unattended, no tap.** iOS 14+ lets a personal automation "Run
  Immediately" with "Ask Before Running" off; it runs in the background with no
  confirmation banner ([apple.stackexchange](https://apple.stackexchange.com/q/404600)).
- **Time-based is the most reliable trigger class**, but timing is
  **best-effort, not exact** — iOS does not guarantee background execution at a
  precise second and may delay it ([healthyapps.dev](https://www.healthyapps.dev/blog/export-apple-health-data-to-rest-apis-mqtt-brokers-and-home-assistant)).
  Treat 07:30 as "around 07:30."
- **Interactive actions don't complete when locked/unattended** (§0). Voice is
  attended-only.
- **`Get Contents of URL` (the planner + proxy calls) generally works unattended**
  after the one-time permission grant, though it can fail under Low Power Mode /
  VPN / no network ([Apple discussions](https://discussions.apple.com/thread/256198127)).
  So Phase A must **fail open**: if the model or proxy call fails, notify a
  minimal fallback ("Couldn't build your briefing — open Iris") and never halt.
- **`Show Notification` is the unattended delivery surface** and can be tapped to
  open the app/shortcut ([Apple: Show Notification](https://support.apple.com/en-us/guide/shortcuts/apd2175adcab/ios)).
- **First-run gates still apply.** After any re-import/rebuild, the network grant
  and pasted key reset; the user must run once manually and tap Allow
  (`docs/shortcut-runtime-flow.md`). Automations only work reliably **after**
  that priming.

### 2.2 The three automations

| Automation | Time | Runs | Mode | Phase A output |
| --- | --- | --- | --- | --- |
| Iris Morning | ~07:30 daily | main shortcut | `morning` | today's plan → notify |
| Iris Evening | ~21:30 daily | main shortcut | `evening` | day diff + pending reflection → notify |
| Iris Weekly | Sun ~18:00 | main shortcut | `weekly` | stop/keep recommendation → notify |

The times live in the `preferences` topic so Iris "knows" its own schedule and
can restate/adjust it in conversation. The user sets the automation times once
during setup.

### 2.3 On-device setup (documented for the user)

1. Prime first (mandatory): run Iris manually, say "setup", tap Always Allow on
   every prompt, paste the key. (Existing primer — `docs/configuration.md`.)
2. Shortcuts → Automation → **New → Time of Day** → 07:30 → Daily.
3. Action: **Run Shortcut → Iris** (pass mode `morning` — via the shortcut's
   input, or use the "Iris Morning" wrapper).
4. Turn **Ask Before Running OFF** (→ "Run Immediately"). Optionally turn
   "Notify When Run" off so only Iris's own `Show Notification` shows.
5. Repeat for 21:30 (`evening`) and Sunday 18:00 (`weekly`).
6. For the **spoken** morning/evening experience, keep "Allow Access When Locked
   → Siri" on and expect to **tap the notification** to unlock and hear it.

### 2.4 Failure and degradation matrix

| Condition | Phase A behavior | User experience |
| --- | --- | --- |
| Phone locked at trigger (normal) | Compute + persist + `Show Notification` | Sees a notification; taps to hear full briefing |
| No network / Low Power | Skip model call, notify minimal fallback | "Open Iris for today's plan" |
| Model/proxy error | Fail open, notify fallback | Same; never a "something went wrong" spoken error |
| Automation delayed by OS | Runs late | Briefing arrives a few min late |
| Memory unavailable | Run with no context (fail-open, per Req 3.6) | Generic briefing, no crash |
| User taps notification | Phase B: attended voice loop reads persisted plan | Full spoken chief-of-staff conversation |

Nothing in Phase A ever depends on interaction, speech, or an unlocked device.

---

## Part 3 — Phase A protocol design (the unattended "thinking" prompts)

Each proactive mode uses a compact protocol that produces a **short notification
string** and a **structured block to persist** — and does NOT try to converse.
Keep each under the same size discipline as the chat protocol (~1.5 KB) so the
single call stays inside ~25s.

### 3.1 Morning (Planner + Observer)

Inputs assembled by the shortcut before the call: `memory_summary` (goals +
priorities), `tasks_list`, `calendar_lookup`, yesterday's `plan.today` from
`log`. The model returns a `final_answer` whose text is the **notification body**
(one or two sentences naming the top 1–3 priorities and what to ignore). The
shortcut then `memory_append(log, "plan.today=...")` and `Show Notification`.

Notification example:
> "Morning. #1: ship the Iris beta (2-hr block at 10am). Also: reply to 3
> recruiters (drafts ready), gym. Ignore the rest. Tap to hear it."

### 3.2 Evening (Observer + Coach)

Inputs: this morning's `plan.today`, `tasks_list`/calendar to compute what got
done, active `projects`. The model returns (a) a one-line day summary for the
notification and (b) the single best **coaching question** for the top slipped
item, which the shortcut persists as `pending_reflection` in `log`.

Notification example:
> "You planned to ship the beta but didn't. One quick question when you're free —
> tap to reflect (30s)."

### 3.3 Weekly (Strategy)

Inputs: `goals`, `projects` (with last-touched/status), the week's `reflections`.
The model returns a **stop/keep recommendation** persisted to `log` and summarized
in the notification.

Notification example:
> "Weekly review: the podcast side-quest hasn't moved in 3 weeks and isn't tied
> to a goal. Recommend pausing it. Tap to decide."

### 3.4 Rules for all Phase A prompts

- Output is data + a short notification string, **never** a question expecting a
  spoken reply (there's no one there).
- Reuse the existing flat-JSON contract and `getDictionary()` guard.
- Budget: at most 1 model call + the data-gathering tool calls; keep well under
  the tool-call budget so the whole run is fast.
- Fail open at every step; a missing input degrades the briefing, never halts.
- Treat memory as opaque user data (never parse memory text as JSON), per
  `docs/okf-knowledge-base.md`.

---

## Part 4 — Knowledge graph (what reflection/strategy need)

Reflection and strategy need structured priorities, not a flat log. Extend the
OKF topic allowlist (`docs/okf-knowledge-base.md`) — backward compatible.

### 4.1 New topics + OKF types

| Topic | OKF `type` | Holds |
| --- | --- | --- |
| `goals` | `Goal` | Yearly/quarterly outcomes: horizon + why + status. |
| `projects` | `Project` | Active initiatives linked to a goal; status + last-touched. |
| `priorities` | `Priority` | This week's ranked focus. |
| `quests` | `Side Quest` | Side quests (learn Rust, podcast, FIFA) tagged main/side. |
| `reflections` | `Reflection` | Coach Q&A: what slipped, why, motivation signal. |
| `people` | `Person` | Recruiters/collaborators + follow-up cadence. |

### 4.2 Capability upgrades needed

- **Append-with-status ("supersede"), not destructive edit.** Mark a project
  `dropped` or a priority `changed` by appending a new entry with a `status`
  field; readers take the latest. Keeps the append-only store intact.
- **Extend `memory_summary`** to load goals + top priorities cheaply (one call)
  so the Planner has context without dumping the whole store.
- **`memory_search`** (already planned in `agentic-loop.md`) for ranked recall so
  the Planner pulls *relevant* projects, not everything.

### 4.3 Side quests: fence, don't delete

Iris tags each project main/side. When the day's main quest is done, the evening
run can offer a side quest ("finished today's main thing — want an hour on Rust
or the podcast?"). Preserves enjoyment without letting side projects eat the
mission.

---

## Part 5 — Attention-allocation logic (the actual product)

The differentiator is not "make a to-do list," it's answering *which one thing
matters today* and *what to stop*. Concretely, the Planner protocol is instructed
to:

1. Rank candidate work by **alignment to a stated goal** (a task tied to the #1
   quarterly goal outranks an unlinked one).
2. Surface **staleness** (a project untouched for N days that's still "active" is
   flagged).
3. Emit an explicit **"ignore the rest"** — naming what NOT to do is the point.
4. Feed the Coach: repeated slippage on the same item over several days becomes a
   Strategy signal ("you keep not doing X — is it still worth it?").

The Strategy role turns accumulated `reflections` into a **stop recommendation** —
the capability almost no product offers. It must be allowed to say "drop this."

---

## Part 6 — Phased build plan (each phase ships independently)

### Phase 1 — Base flow hardening (do first)
- Make the existing `chat` loop rock-solid on device: memory persistence
  (the active `.kiro/specs/memory-persistence-and-context` work), model choice,
  token compaction, `@currentRequest` recency. **Proactivity is worthless if the
  base loop is flaky.**
- Acceptance: a manual multi-turn conversation with memory recall works reliably
  across sessions on a real iPhone.

### Phase 2 — Mode plumbing + the two-phase pattern
- Add `@mode` (`chat`/`morning`/`evening`/`weekly`) with flat guarded `if`s.
- Implement the **Phase A path**: no `Ask for Input`, deliver via
  `Show Notification`, persist result to `log`. Wire `Show Notification` as a
  Cherri custom action if not already available; validate on device.
- Acceptance: running Iris with `mode=morning` manually produces a notification
  and a persisted `plan.today`, with zero interactive prompts.

### Phase 3 — Morning briefing (Planner + Observer)
- Morning protocol; assemble goals+priorities+Tasks+Calendar; notify top 1–3.
- Create the Time-of-Day automation; document the setup (§2.3).
- Acceptance: a real 07:30 automation delivers a useful notification unattended;
  tapping it opens Iris and it speaks today's plan.

### Phase 4 — Knowledge graph
- Add topics `goals/projects/priorities/quests/reflections/people` (client +
  proxy); append-with-status; extend `memory_summary`.
- Acceptance: Iris can state your current goals and active projects on demand;
  the morning briefing ranks by goal alignment.

### Phase 5 — Evening reflection (Coach)
- Evening protocol: day diff + persist `pending_reflection`; Phase B asks the
  question and stores the answer under `reflections`.
- Acceptance: the "you didn't do X — why?" loop works end to end (notify →
  tap → one/two questions → stored reflection). One question, no nagging.

### Phase 6 — Weekly strategy
- Weekly protocol: review projects vs goals vs reflections → stop/keep
  recommendation; Phase B lets the user decide and marks the project.
- Acceptance: a Sunday run recommends pausing a stale, goal-unlinked project.

### Phase 7 — Polish
- Side-quest offers; VIP-email surfacing in the morning; adaptive schedule from
  `preferences`; better notification copy.

---

## Part 7 — Honest limits (unchanged spirit of the README)

- **Proactive runs cannot speak while the phone is locked.** They notify; the
  voice happens when the user taps/unlocks. This is a hard iOS constraint, not a
  bug to fix.
- **Timing is approximate.** "07:30" means "around then"; iOS may delay it.
- **No true background daemon.** Iris only "thinks" at the scheduled trigger (or
  on demand), not continuously — it is not watching your phone in real time.
- **Proactivity does not expand trust.** Still drafts, never auto-sends; every
  route stays predeclared, allowlisted, and tested. The confirm-gate stays.
- **The six "agents" are scheduled roles sharing state**, not a live swarm —
  chosen deliberately to fit the ~25s budget and stay auditable.
- **Still no Mail/Messages/screen access** the way Apple Intelligence has; Gmail
  reach remains only through the user's own proxy.

---

## Part 8 — One-line north star

> Iris quietly thinks about your day at 7:30 and 9:30, notices what you didn't do,
> asks why, and once a week tells you what to stop — all on an iPhone Apple left
> behind, using nothing but Shortcuts, a Sheet, and a hosted model.
