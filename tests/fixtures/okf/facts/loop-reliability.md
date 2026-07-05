---
type: Design Fact
title: Agent Loop Reliability
description: Reliability requirements for returning tool observations back to the agent.
tags:
  - loop
  - tools
  - reliability
timestamp: 2026-07-04T17:19:00+05:30
---

# Agent Loop Reliability

## Requirements

- Every data-producing tool should return a normalized observation to the next agent turn.
- Calendar results must not stop the run before ChatGPT receives and summarizes them.
- Debug logging must be explicit and should not overwrite Notes or Clipboard during normal runs.
- Unknown tools should produce a safe unsupported observation, not execute arbitrary actions.
