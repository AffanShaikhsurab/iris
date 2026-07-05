# OKF Knowledge Base

Iris uses a phone-local OKF-style knowledge base to give the agent
durable context without pretending it has Apple Intelligence-level private app
access.

OKF is a directory of Markdown concept files with YAML frontmatter. Iris
uses the same simple shape: each concept is readable by humans, retrievable by
the Shortcut, and safe to pass to the model only when relevant.

## Storage Location

The intended phone folder is:

```text
Shortcuts/IrisOKF/
  index.md
  log.md
  profile.md
  preferences.md
  people/
  projects/
  routines/
  facts/
```

The current Shortcut source uses fixed paths:

```text
/Shortcuts/IrisOKF/index.md
/Shortcuts/IrisOKF/profile.md
/Shortcuts/IrisOKF/preferences.md
/Shortcuts/IrisOKF/log.md
```

These paths must be validated on device because iOS Files permissions and path
resolution can vary.

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
> `memory_status` are now implemented in `shortcuts/iris.cherri` (append-only,
> hands-free, fixed-path Files actions; see `docs/memory-system-design.md`).
> Still pending: `memory_save`/overwrite, `memory_search`, confirmation-gated
> writes, and the startup bootstrap. All paths/WFKeys need on-device validation.

The memory layer is a fixed set of predeclared tools, not arbitrary file access.
The agent cannot browse files; every memory operation is routed through a tool.

- `memory_status` (implemented): checks whether the OKF folder can be read and
  returns its folder contents as an observation.
- `memory_lookup` (planned): retrieve relevant snippets from a startup bootstrap.
- `memory_list_topics` (planned): list available OKF folder contents.
- `memory_propose_write` (planned): propose a durable memory entry and ask the
  user before appending it.
- `memory_append_log` (planned): append a constrained entry to `log.md`.

The intended startup behavior is to read a small bootstrap from `index.md`,
`profile.md`, and `preferences.md` into `@loopContext` as `memory_summary`. That
bootstrap is not implemented yet, so today the agent has no memory context until
the remaining tools land.

## Result Envelopes

Memory tools return the same normalized observation shape as other Iris
tools:

```text
tool=memory_lookup
ok=true
count=1
result=memory lookup completed
records=<relevant snippets>
error=
```

If a memory write is declined:

```text
tool=memory_propose_write
ok=false
count=0
result=
records=<proposed memory>
error=user declined memory write
```

## Privacy Rules

- The OKF folder is local to the phone or the user's iCloud Drive, depending on
  the Files location used.
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
2. `memory_status` folder-readability check. (implemented)
3. Read-only bootstrap from `index.md`, `profile.md`, and `preferences.md`. (planned)
4. Read-only `memory_lookup` and `memory_list_topics`. (planned)
5. Confirmation-gated append to `log.md` via `memory_propose_write` /
   `memory_append_log`. (planned)
6. Later, structured concept-file creation and update/delete tools after device
   validation. (planned)

## Device Validation

Before marking memory stable, test:

- Missing OKF folder.
- Folder exists but files are missing.
- Files permission prompt appears and can be granted.
- Bootstrap files are read without stopping the Shortcut.
- `memory_lookup` returns relevant snippets and not the whole folder.
- `memory_propose_write` asks before appending.
- Declined writes produce `ok=false` and do not change files.
- Approved writes append to `log.md`.
