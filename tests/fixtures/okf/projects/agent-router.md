---
type: Project
title: Shortcut Agent Router
description: Siri replacement experiment using ChatGPT App Intent and predeclared Apple Shortcuts tools.
tags:
  - agent-router
  - siri
  - shortcuts
  - memory
timestamp: 2026-07-04T17:19:00+05:30
---

# Shortcut Agent Router

## Goal

Build a Siri-like assistant where ChatGPT plans, Apple Shortcuts executes, and
tool observations are returned to the agent until it can produce a final answer.

## Architecture Notes

- ChatGPT is the planner, not the executor.
- The Shortcut can only run predeclared tools.
- Tool output should return to the agent as a normalized observation.
- OKF memory should provide local context without claiming private app access.
