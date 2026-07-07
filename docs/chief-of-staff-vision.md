---
type: Vision / Design
title: From Voice Assistant to AI Chief of Staff
summary: A product and architecture plan to evolve Iris from a reactive, invoke-only Siri agent into a proactive, reflective "AI Chief of Staff" — using Shortcuts Automations for proactivity, a six-role agent model, an expanded OKF knowledge graph, and Google Sheets as a shared source of truth for external agents (ChatGPT / Gemini / Claude).
status: proposal
tags:
  - iris
  - product-vision
  - chief-of-staff
  - proactivity
  - automations
  - multi-agent
  - okf
related:
  - README.md
  - PLAN.md
  - docs/architecture.md
  - agentic-loop.md
  - docs/memory-system-design.md
  - docs/apps-script-proxy.md
  - docs/okf-knowledge-base.md
  - docs/routes.md
---

# From Voice Assistant to AI Chief of Staff

> Status: proposal / north-star. Nothing here is wired into `shortcuts/iris.cherri`
> yet. This document reframes the product, maps the ambition onto Iris's *real*
> runtime constraints, and lays out a phased, buildable path. Every proposal is
> checked against the hard limits already documented in
> `docs/shortcut-runtime-flow.md` (the ~25s iOS network budget, no `else if`,
> hands-free-under-Siri, one model call per turn).

## 1. The reframe: this is not a productivity app

The most useful way to think about where Iris is going is not "a smarter Siri"
or "another task manager." It is an **AI Chief of Staff** — a system that thinks
about the user's life the way a human executive assistant would.

Almost every existing tool solves **one layer** of the problem:

- Calendar planners (Reclaim, Motion, Morgen) optimize *schedules*, not
  *strategy*. They never ask "why are we even working on this project?"
- Workflow builders (Make/Zapier + LLM) *route* data between Gmail, Calendar,
  and Notion, but they don't reflect on motivation.
- Agent frameworks (AutoGPT, CrewAI, BabyAGI) *execute* multi-step tasks but
  require heavy setup and offer no long-term coaching.
- ChatGPT Agent is the closest general foundation, but it doesn't become a
  durable, opinionated *chief of staff* that owns your priorities over months.

The whole loop a chief of staff runs looks like this:

```text
Life data (Gmail, Calendar, Notes, Tasks, Docs, job apps, messages)
        ↓
Understand long-term priorities
        ↓
Observe what happened today
        ↓
Notice what DIDN'T happen
        ↓
Ask why  →  reason about motivation  →  challenge the priority
        ↓
Re-plan  →  update tasks  →  repeat every day
```

That is fundamentally different from "AI creates a to-do list." The real problem
Iris should solve is **attention allocation**, not task capture:

> Out of the 500 possible things I could do, which ONE deserves my attention
> today — and which projects should I *stop*?

The single hardest, least-solved capability in the whole space is **reflection**:

```text
Iris:  Yesterday you didn't work on the launch. Why?
User:  I wasn't excited about it.
Iris:  Why?
User:  I don't think this project matters anymore.
Iris:  Let's compare it against your yearly priorities.
       ...Based on those, I'd recommend dropping it. Want to?
```

That exchange is coaching, not automation. No mainstream product does it well.
It is Iris's opportunity.

## 2. Where Iris is today (honest baseline)

Iris already has three of the pieces most projects struggle to build. It is
missing the two that define a chief of staff.

| Chief-of-staff capability | Iris today | Gap |
| --- | --- | --- |
| **Durable memory** | Hybrid OKF store: phone-local Files + Google Sheet via the Apps Script proxy (`docs/memory-system-design.md`). Append-only, topic-scoped, fail-open. | Schema is shallow (log/notes/journal). No goals/projects/priorities model, no update/delete, no ranked recall. |
| **Execution** | Bounded planner→tool loop (`agentic-loop.md`): reminders, notes, calendar, Gmail search/read/draft, Tasks, drafts, web search, maps. Auditable, predeclared routes. | Reactive only. |
| **Real Google reach** | Apps Script proxy confirmed working end-to-end (`notes.md`): Tasks, Gmail, Calendar, memory Sheet — running *as the user*, token-free on device. | Underused: only invoked when the user speaks. |
| **Proactivity** | **None.** Iris runs only when the user says "Hey Siri, Iris." | No scheduler, no background trigger, no daily/weekly initiative. |
| **Reflection / strategy** | **None.** The loop answers the current request and stops. | No observer, no coach, no weekly project review, no "drop this project" recommendation. |

The two gaps — **proactivity** and **reflection** — are the entire distance
between "smart Siri" and "chief of staff." Everything below targets those two.

## 3. The six-role agent model — mapped to Iris's real runtime

The vision calls for six agents. That is the right *conceptual* decomposition,
but a literal six-agents-orchestrated-in-one-turn design is impossible under
Iris's runtime: every model call must finish inside ~25s or Siri kills the run
(`docs/shortcut-runtime-flow.md`), and the shortcut makes exactly **one** model
call per turn.

So we implement the six agents as **roles, not simultaneous processes**. Each
role is:

1. a distinct **protocol prompt** (system message) that reuses the existing
   bounded loop, and
2. triggered at a **different time / entry point** (voice, morning automation,
   evening automation, weekly automation), and
3. sharing one **source of truth** — the Google Sheet / OKF knowledge graph — so
   the roles hand off to each other *through durable state*, not through a live
   multi-agent conversation.

This is the only design that fits Shortcuts. It is also genuinely good
architecture: the Sheet becomes the "shared memory bus" between roles.

| Role | What it does | How it runs in Iris | Reuses today |
| --- | --- | --- | --- |
| **1. Memory Agent** | Knows everything; never forgets. Goals, projects, priorities, people, quests, daily log. | Expanded OKF schema in the hybrid store (§5). | Hybrid memory + proxy Sheet. |
| **2. Planner** | Every morning: "Given your priorities, today matters because… these 3 things move your goals. Ignore the rest." | A **morning automation** runs a `daily_briefing` protocol (§4). | Calendar/Tasks/Reminders lookups; the loop. |
| **3. Observer** | Notices meetings, unfinished tasks, ignored emails, recurring distractions — without being asked. | Runs inside the morning/evening automation: reads Tasks + Calendar + Gmail via the proxy, diffs against yesterday's plan, writes findings to the `log` topic. | `tasks_list`, `gmail_search`, `calendar_lookup`. |
| **4. Coach** | Instead of "task overdue," asks "Why? What made it hard? Is it too big? Still aligned?" | An **evening automation** opens a short reflective dialogue (Ask-for-Input), stores answers under `reflections`. | The Ask-for-Input conversation loop. |
| **5. Strategy Agent** | Weekly: "Which project no longer deserves attention?" Actually recommends stopping things. | A **weekly automation** runs a `weekly_review` protocol over the projects + reflections topics. | The loop + memory read. |
| **6. Execution Agent** | Actually does the work: drafts emails, creates tasks, schedules, follows up, sets reminders. | The **existing** bounded tool loop, unchanged. | All of `docs/routes.md`. |

Key insight: **Execution and Memory already exist.** The build is really about
adding **Planner, Observer, Coach, and Strategy** as scheduled protocol variants
plus a richer knowledge graph.

## 4. The proactivity engine — Shortcuts Automations

This is the linchpin the user identified, and it is correct: **iOS Shortcuts
Automations are how a Shortcuts-based agent becomes proactive.** iOS supports
**Time of Day** automations that run a shortcut automatically. On iOS 17+ many
automations can be set to "Run Immediately" with no confirmation tap, which is
what makes a hands-free morning briefing possible.

### 4.1 The three scheduled entry points

```text
07:30  Morning automation  → run "Iris Morning"  → Planner + Observer
21:30  Evening automation  → run "Iris Evening"  → Coach + Observer (day diff)
Sun 18:00  Weekly automation → run "Iris Weekly" → Strategy Agent
```

Each is a **separate shortcut** (or one shortcut branching on an input
parameter) with its own protocol prompt, all reading/writing the same OKF store.
The user sets the times once during setup; the times themselves can be stored in
the `preferences` topic so Iris "knows" its own schedule.

### 4.2 What the morning briefing actually does (Planner + Observer)

Concrete, buildable with today's tools:

1. Bootstrap memory (`memory_summary`) → load goals, active projects,
   priorities, and yesterday's plan.
2. Observe: `tasks_list` + `calendar_lookup` + optionally `gmail_search` for
   unread-from-VIP → build today's real state.
3. Diff: compare yesterday's plan (stored in `log`) against what got done →
   surface what *didn't* happen.
4. Plan: the model selects the **top 1–3 things** that move the stated goals,
   and explicitly says what to ignore.
5. Deliver: Siri speaks the briefing via the Ask-for-Input pattern; the plan is
   written back to the `log` topic so the evening run and tomorrow's run can
   diff against it.

Output feels like:

> "Good morning. Your #1 priority this quarter is shipping the Iris beta.
> Yesterday you didn't touch it. You have a clear 2-hour block at 10am — that's
> the one thing that matters today. You also have 3 recruiter emails; I've
> drafted replies for review. Everything else can wait."

### 4.3 What the evening reflection does (Coach)

The evening automation is where the **reflection** differentiator lives:

1. Diff the morning plan against `tasks_list` / calendar → find what slipped.
2. For the top slipped item, ask **one** coaching question ("You planned to work
   on the beta but didn't — what got in the way?").
3. Take the dictated answer, ask at most one follow-up ("Is it that the task is
   too big, or that it no longer feels important?").
4. Store the exchange under a new `reflections` topic. Do **not** nag — one or
   two questions, then stop.

Over weeks, `reflections` becomes the raw material the Strategy Agent reasons
over.

### 4.4 Runtime constraints this must respect

- **Automations run the same shortcut**, so all the rules in
  `docs/shortcut-runtime-flow.md` still apply: one model call per turn, ~25s
  budget, no `else if`, regex-guard before `getDictionary()`, fail-open.
- **A scheduled run may fire while the phone is locked.** Any interactive
  Ask-for-Input needs "Allow When Locked" or must degrade to a notification.
  Realistic fallback: if the run can't get voice input (locked, no unlock), it
  writes the briefing to a note / sends a notification instead of speaking, and
  never halts.
- **"Run Immediately" is best-effort.** Time-of-day automations can be delayed
  or skipped by the OS. Treat the briefing as opportunistic, not guaranteed;
  the Sheet remains the durable record either way.
- **No new high-risk capability.** The proactive runs still only *draft* (never
  auto-send) and only touch predeclared routes. Proactivity changes *when* Iris
  acts, not *what* it is trusted to do.

## 5. The knowledge graph — expanding OKF for a chief of staff

Reflection and strategy need structured priorities, not a flat log. The OKF
format (`docs/okf-knowledge-base.md`) already supports typed concepts with
frontmatter — we extend the **topic allowlist** and define new concept `type`s.
This is backward compatible: existing topics (`index profile preferences log
notes journal`) keep working.

### 5.1 New topics

| Topic | OKF `type` | Holds |
| --- | --- | --- |
| `goals` | `Goal` | Yearly / quarterly outcomes with a horizon and a "why". |
| `projects` | `Project` | Active initiatives, each linked to a goal, with status + last-touched. |
| `priorities` | `Priority` | Ranked focus for the current week. |
| `quests` | `Side Quest` | Ali-Abdaal-style side quests (learn Rust, podcast, FIFA, read a book), tagged Main/Side. |
| `reflections` | `Reflection` | Coach Q&A: what slipped, why, motivation signal. |
| `people` | `Person` | Recruiters, collaborators, follow-up cadence. |

### 5.2 Example concepts

```md
---
type: Goal
title: Ship Iris public beta
horizon: 2026-Q3
why: Validate the chief-of-staff thesis with real users before scaling.
status: active
tags: [mission, iris]
timestamp: 2026-07-07T08:00:00+05:30
---
Get 50 real users running the morning briefing daily.
```

```md
---
type: Side Quest
title: Learn Rust
category: side
tags: [learning]
timestamp: 2026-07-07T08:00:00+05:30
---
Fun, not on the critical path. Allowed only after the day's main quest is done.
```

### 5.3 Side quests: categorize, don't eliminate

A chief of staff shouldn't delete the fun. It should **fence** it. Iris tags each
project as Main Quest or Side Quest. When the day's important work is done, the
evening run can offer:

> "You've finished today's main quest. Want to spend an hour on a side quest —
> Rust or the podcast?"

That preserves enjoyment without letting side projects eat the mission.

### 5.4 Memory capability upgrades needed

- **Update/supersede** (today memory is append-only): mark a project "dropped"
  or a priority "changed." Implement as append-with-status + read-latest, so the
  append-only store still works (no destructive edits).
- **Ranked recall / `memory_search`** (already planned in `agentic-loop.md`): let
  the Planner pull *relevant* goals/projects instead of dumping the store.
- **A structured "state" read** the Planner can load in one cheap call (the
  `memory_summary` bootstrap, extended to include goals + top priorities).

## 6. Sheets as the source of truth + external-agent integration

The user's strongest strategic idea: **make the Google Sheet the shared source
of truth, then let other agents (ChatGPT, Gemini, Claude) read and write it too.**

Why this is the right bet:

- Iris already writes structured memory to the Sheet via the Apps Script proxy,
  running *as the user*, token-free on device (`docs/apps-script-proxy.md`).
- Most users won't *only* use Siri — they live in ChatGPT / Claude / Gemini. If
  those tools read and update the **same** Sheet, the user's life-context becomes
  portable, and Iris (the voice/proactive layer) gets dramatically more useful
  because the graph is always current.
- The Sheet is the neutral hub: no lock-in, user-owned, inspectable, immune to
  iCloud quota.

### 6.1 Architecture

```text
                     ┌───────────────────────────┐
                     │   Google Sheet (OKF store) │  ← single source of truth
                     │  goals·projects·priorities │
                     │  reflections·log·people    │
                     └────────────▲──────────────┘
                                  │  Apps Script proxy (runs as user,
                                  │  shared-secret, GET/JSON envelope)
        ┌──────────────┬──────────┴───────────┬───────────────┐
        │              │                       │               │
   Iris (Siri)   ChatGPT (GPT/       Claude (MCP        Gemini (Gems /
   voice +       Actions plugin)     server)            extension)
   automations
```

The proxy already exposes `memory_*`, `tasks_*`, `gmail_*`, `calendar_*` ops.
The same proxy can back:

- **A ChatGPT GPT / Action** — an OpenAPI schema pointing at the proxy's `/exec`
  URL; the user tells ChatGPT "remember I dropped the podcast project" and it
  `POST`s an update to the Sheet.
- **A Claude MCP server** — a thin MCP wrapper (Node/Python) around the same
  proxy ops, so Claude Desktop / Claude Code can read goals and append
  reflections.
- **A Gemini Gem / extension** — same proxy, same ops.

The proxy is the compatibility layer. Build it once, reuse it for every agent.

### 6.2 Why this multiplies Iris's value

The voice/proactive layer (Iris) and the deep-reasoning layer (Claude/ChatGPT)
stop being competitors and become **two front-ends over one brain**. You reflect
with Claude at your desk; Iris reminds and executes on your phone; both read the
same priorities. That coherence is the actual product.

### 6.3 Security note (must not skip)

The proxy is `ANYONE_ANONYMOUS` guarded only by a shared secret. Exposing it to
more clients raises the blast radius:

- Keep the secret out of any client the user doesn't control; each external
  integration should use its own secret if the proxy is extended to support
  multiple.
- Consider **read/write scoping** per client (e.g. a plugin that can append
  reflections but not `send_email`).
- The `send_email` confirm-gate must stay. No external agent gets un-gated send.
- Document clearly that anything written to the Sheet is user data the user is
  choosing to share across providers.

## 7. The viewing layer — plugin first, app later

The user floated two ways to *see* the data: an iOS app, or a plugin. Sequence
them by effort/value:

1. **Plugin/connector first (low effort, high value).** The ChatGPT Action /
   Claude MCP / Gemini extension in §6 *is* the viewing-and-editing layer for
   free — the user just asks their existing chat app. Ship this before writing
   any app.
2. **A lightweight dashboard next (medium effort).** The Sheet itself is already
   a dashboard. A read-only web view (even a simple Apps Script `doGet` HTML page
   or a small static site hitting the proxy) can render goals/projects/priorities
   nicely with near-zero infra.
3. **A native iOS app last (high effort).** Only if there's real demand for
   push, widgets, and a polished UI. It would read the same proxy/Sheet. Note
   this contradicts the current PLAN.md non-goal ("Building a native iOS app"),
   so it's explicitly a *future* option, not near-term.

## 8. Phased roadmap

Ordered by dependency and value. Each phase is shippable on its own.

### Phase 1 — Proactivity foundation (highest leverage)
- Add `daily_briefing` protocol + wire the planned `daily_briefing`/`meeting_prep`
  tools (`agentic-loop.md` lists them as planned).
- Ship an "Iris Morning" shortcut + a documented Time-of-Day automation setup.
- Add locked-device fallback (notification/note when voice input isn't available).
- Deliverable: a real morning briefing that reads Tasks + Calendar and names the
  top 1–3 priorities.

### Phase 2 — Knowledge graph
- Extend the OKF topic allowlist: `goals`, `projects`, `priorities`, `quests`,
  `reflections`, `people` (client + proxy).
- Add append-with-status (supersede) and extend `memory_summary` to load goals +
  top priorities cheaply.
- Deliverable: Iris can state your current goals and active projects on demand.

### Phase 3 — Reflection (the differentiator)
- Ship an "Iris Evening" shortcut: day-diff + one coaching question + store
  under `reflections`.
- Deliverable: the "you didn't do X — why?" loop, working end to end.

### Phase 4 — Strategy
- Ship an "Iris Weekly" shortcut: review projects vs goals vs reflections and
  recommend what to stop.
- Deliverable: a weekly "drop this project" recommendation.

### Phase 5 — Cross-agent source of truth
- Publish an OpenAPI schema for a ChatGPT Action over the proxy.
- Build a minimal Claude MCP wrapper around the proxy ops.
- Harden proxy auth/scoping for multi-client use.
- Deliverable: reflect with Claude/ChatGPT; Iris executes and reminds; one Sheet.

### Phase 6 — Viewing layer (optional)
- Read-only dashboard over the Sheet; native app only if demanded.

## 9. Honest limits (what this does NOT become)

Keeping the README's "honest limits" spirit:

- Iris still **cannot** read Mail/Messages/screen the way Apple Intelligence can.
  Gmail reach is only through the user's own proxy.
- Time-of-Day automations are **best-effort** and may need an unlock for voice.
- The six "agents" are **scheduled protocol roles sharing state**, not a live
  multi-agent swarm — that's a feature, not a shortcut: it's the only design that
  survives the ~25s budget and stays auditable.
- Proactivity does **not** expand what Iris is trusted to do autonomously. It
  still drafts, never auto-sends; every route stays predeclared and tested.
- A native iOS app remains a *non-goal* until Phases 1–5 prove demand.

## 10. Positioning (one line)

Not a chatbot. Not a task manager. Not another Notion AI.

> **Iris is a personal operating system: an AI layer that continuously
> understands your goals, notices what changed, challenges your assumptions when
> it should, and quietly handles the routine — so you can spend attention on the
> few things that actually matter.**

The wedge that makes it real on hardware Apple abandoned: **proactive daily
briefings + honest reflection, backed by a user-owned knowledge graph that every
AI you already use can share.**
