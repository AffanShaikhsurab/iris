---
type: Research / Integration Guide
title: Plugging Iris into ChatGPT, Claude, and Gemini (web apps)
summary: How to expose the Iris knowledge graph (the Apps Script proxy / Google Sheet source of truth) as a "plugin" inside the CONSUMER WEB APPS of ChatGPT, Claude, and Gemini — GPT Actions (OpenAPI, not MCP), Claude custom connectors (remote MCP), and the honest reality of Gemini's lack of an open consumer plugin platform. Excludes coding CLIs.
status: research
tags:
  - iris
  - integrations
  - chatgpt
  - claude
  - gemini
  - gpt-actions
  - mcp
  - apps-script-proxy
related:
  - docs/chief-of-staff-vision.md
  - docs/apps-script-proxy.md
  - docs/okf-knowledge-base.md
sources:
  - https://help.openai.com/en/articles/9442513-configuring-actions-in-gpts
  - https://help.openai.com/en/articles/8554407-create-a-custom-gpt
  - https://developers.openai.com/apps-sdk/
  - https://help.openai.com/en/articles/12584461-developer-mode-and-mcp-apps-in-chatgpt
  - https://claude.com/docs/connectors/custom/remote-mcp
  - https://claude.com/docs/connectors/building/authentication
  - https://ai.google.dev/gemini-api/docs/partner-integration
  - https://support.google.com/g/answer/16550932
  - https://developer.puter.com/tutorials/gemini-oauth/
---

# Plugging Iris into ChatGPT, Claude, and Gemini (web apps)

> Goal: let a user talk to ChatGPT / Claude / Gemini in their **normal web app**
> and have it read and write the **same Iris knowledge graph** — the Google Sheet
> source of truth reached through the Apps Script proxy (`docs/apps-script-proxy.md`).
> This is the cross-agent "one brain, many front-ends" plan from
> `docs/chief-of-staff-vision.md` §6.
>
> Scope: **consumer web apps only.** Coding CLIs (Gemini CLI, Claude Code) are
> explicitly out of scope. Content from vendor docs was rephrased/summarized for
> compliance; sources are linked inline and in the frontmatter.

## 0. The one-paragraph answer

There is no single "plugin" standard across the three. **ChatGPT web** takes a
plain **OpenAPI 3.1 schema** ("GPT Actions") — this is *not* MCP and is the
easiest, most direct fit for the Iris proxy. **Claude web** takes a **remote MCP
server** ("custom connector") — MCP is the *only* official path there, so
"not MCP" isn't an option for Claude. **Gemini's consumer web app has no open
self-serve third-party plugin platform at all** — the realistic routes are a
Workspace/Enterprise *connector* (business tier) or building your own thin chat
UI on the Gemini API with function calling. So the build is: **one OpenAPI spec
(ChatGPT) + one remote MCP server (Claude, and optionally ChatGPT Apps), both
thin facades over the existing proxy; Gemini gets a documented workaround.**

| Platform (web app) | Integration surface | Standard | MCP? | Self-serve for a solo dev? |
| --- | --- | --- | --- | --- |
| **ChatGPT** | GPT Actions in a Custom GPT | **OpenAPI 3.1** | No | Yes (needs a paid ChatGPT plan) |
| **ChatGPT** (newer) | Apps SDK / MCP connectors | **MCP (Streamable HTTP)** | Yes | Preview; full MCP/dev mode gated to Business/Enterprise today |
| **Claude** | Custom connector | **Remote MCP** | Yes (only option) | Yes (Pro/Team/Enterprise) |
| **Gemini** | Extensions/connectors | Google-controlled | No public spec | **No** open consumer platform |

## 1. ChatGPT — GPT Actions (OpenAPI, the easy win)

This is the classic "plugin in ChatGPT" and the **best first target** because it
speaks HTTP+JSON, exactly what the Apps Script proxy already serves.

### 1.1 What it is
A **Custom GPT** can be given **Actions**. Per OpenAI's help docs, an action has
two parts: (1) **authentication** and (2) a **schema** — an **OpenAPI
specification (JSON or YAML)** that tells ChatGPT which server to call, which
endpoints exist, what parameters they take, and an `operationId` per action.
([Configuring actions in GPTs](https://help.openai.com/en/articles/9442513-configuring-actions-in-gpts)).
Content rephrased for compliance.

### 1.2 Requirements / limits
- **Creating or editing a GPT requires a paid ChatGPT subscription**
  ([Create a custom GPT](https://help.openai.com/en/articles/8554407-create-a-custom-gpt)).
- Actions are **not available in "Pro mode"** models — the model picker limits to
  action-capable models when a GPT has custom actions
  ([Configuring actions in GPTs](https://help.openai.com/en/articles/9442513-configuring-actions-in-gpts)).
- Auth options: **None**, **API key** (Basic / Bearer / custom header), or
  **OAuth** (client id+secret, auth URL, token URL, scopes; the editor gives you
  a callback URL). Same source.
- In managed/enterprise workspaces, an **action-domain allowlist** can block
  actions entirely. Same source.

### 1.3 How Iris uses it
1. Stand up a small HTTPS endpoint that ChatGPT can call. Two options:
   - **Directly the Apps Script proxy** if you make its request/response shape
     OpenAPI-describable (it already returns JSON), **or**
   - a **thin FastAPI facade** in front of the proxy that gives clean REST paths
     and a stable OpenAPI doc (recommended — see §4).
2. Write an **OpenAPI 3.1 schema** describing the operations you want the GPT to
   use: `get_goals`, `get_priorities`, `append_reflection`, `add_task`,
   `search_memory`, etc. (map onto proxy `op=` values).
3. Auth: simplest is **API key → custom header** carrying the proxy shared
   secret. (Do NOT bake the secret into a *public* GPT; keep the GPT private, or
   move to OAuth / per-user secrets — see §5 security.)
4. In ChatGPT → **Create a GPT → Configure → Actions → Create new action**, paste
   the schema, set auth, and **test in Preview**.

### 1.4 Minimal OpenAPI sketch
```yaml
openapi: 3.1.0
info: { title: Iris Memory, version: "1.0.0" }
servers: [{ url: https://your-facade.example.com }]
paths:
  /priorities:
    get:
      operationId: getPriorities
      summary: Get the user's current ranked priorities.
      responses:
        "200": { description: OK }
  /reflections:
    post:
      operationId: appendReflection
      summary: Append a reflection entry to the knowledge graph.
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                topic: { type: string }
                body:  { type: string }
      responses:
        "200": { description: OK }
components:
  securitySchemes:
    ApiKeyAuth: { type: apiKey, in: header, name: X-Iris-Secret }
security: [{ ApiKeyAuth: [] }]
```

### 1.5 The newer path: Apps SDK / MCP in ChatGPT
OpenAI's **Apps SDK** (announced Oct 6, 2025, in preview) lets apps run *inside*
ChatGPT with custom UI, and it is **built on MCP** over Streamable HTTP
([Introducing apps in ChatGPT](https://openai.com/index/introducing-apps-in-chatgpt/);
[Apps SDK](https://developers.openai.com/apps-sdk/)). Custom MCP connectors /
developer mode are currently enabled for **Business/Enterprise/Edu** workspaces
by an admin ([Developer mode and MCP apps in ChatGPT](https://help.openai.com/en/articles/12584461-developer-mode-and-mcp-apps-in-chatgpt)).
Takeaway: for a solo dev **today**, **GPT Actions (OpenAPI) is the pragmatic
route**; the MCP server you build for Claude (§2) also becomes reusable here as
Apps SDK access broadens.

## 2. Claude — custom connectors (remote MCP)

For Claude's web app, the official integration is a **custom connector**, which
is a **remote MCP server**. There is no OpenAPI-action alternative — MCP is the
path. ([Third party connectors with remote MCP](https://claude.com/docs/connectors/custom/remote-mcp)).

### 2.1 What it is
A remote MCP server is an internet-hosted server exposing MCP **tools / prompts /
resources** that Claude can call on the user's behalf
([modelcontextprotocol.io](https://modelcontextprotocol.io/docs/tutorials/use-remote-mcp-server)).
The **same connector infrastructure backs Claude.ai (web), Desktop, mobile, and
Claude Code** ([Authentication for connectors](https://claude.com/docs/connectors/building/authentication)),
so one server covers the web app and more.

### 2.2 Requirements / limits
- A **Claude.ai account on Pro, Team, or Enterprise** to add custom connectors.
- The server should be **Streamable HTTP** transport and support an **auth**
  scheme (OAuth is the clean per-user option; a shared secret works for a
  single-user personal setup).
- User adds it in **Settings → Connectors → Add custom connector → server URL**,
  then completes any OAuth popup.

### 2.3 How Iris uses it
1. Build a **remote MCP server** (FastMCP / the official MCP SDK — you already
   know this is easy) that exposes tools like `get_priorities`,
   `append_reflection`, `add_task`, `search_memory`.
2. Each tool **calls the Apps Script proxy** (GET with `secret`, `op`, args) and
   returns the result.
3. Host it over HTTPS with Streamable HTTP; add OAuth (or a secret for personal
   use).
4. Add it in Claude.ai as a custom connector.

This is the same server you can later reuse for ChatGPT's Apps SDK (§1.5), so
build it once.

## 3. Gemini — the honest reality (no open consumer plugin platform)

There is **no public, self-serve "action/plugin" platform for the consumer
Gemini web app** comparable to GPT Actions. What exists:

- **Gemini Extensions / Apps / connectors** in the consumer app are largely
  **Google's own** (Workspace, Maps, YouTube, Flights) plus **select partners** —
  not an open developer submission flow for arbitrary third parties.
- **Gemini Enterprise / Workspace connectors** let a business **admin** connect
  third-party data sources (e.g. Asana, Mailchimp) so Gemini can *search* them
  from the side panel ([Connect your Google apps and third-party data](https://support.google.com/g/answer/16550932);
  [Use integrations with Gemini in Workspace](https://support.google.com/mail/answer/16796422)).
  This is business-tier and admin-configured, not a consumer plugin.
- **Gems** (custom Gemini personas) are **instructions only** — they **cannot
  call external APIs**. External data reaches Gemini through **function calling**,
  which is a **Gemini API** feature, not something available inside the consumer
  chat UI ([Function calling using the Gemini API](https://firebase.google.com/docs/ai-logic/function-calling)).
- There is also **no official OAuth flow letting a third-party app call Gemini on
  a user's behalf** ([Puter: How to do OAuth with Gemini](https://developer.puter.com/tutorials/gemini-oauth/)).
- **Gemini CLI extensions** exist but are the coding CLI — **out of scope**.

### 3.1 Realistic options for Gemini
1. **Build your own thin chat UI on the Gemini API** with **function calling**
   wired to the Iris proxy. This gives a "talk to Gemini, it updates my Sheet"
   experience, but it lives in *your* app, not in `gemini.google.com`.
2. **Enterprise/Workspace connector** — only if the user is on Gemini Enterprise
   / Workspace and an admin sets it up; and it's search-oriented, not full
   read/write tool calling.
3. **Wait / watch** — Google may open a broader agent/connector surface; treat
   consumer-Gemini plugins as *not available today* in the roadmap.

Bottom line: **do not promise a Gemini-web plugin.** Ship ChatGPT + Claude, and
offer Gemini users the "your own Gemini-API app" workaround.

## 4. The unifying build: two facades over one proxy

You do **not** rebuild the backend three times. Everything points at the existing
Apps Script proxy / Google Sheet. Build two thin facades:

```text
                         Apps Script proxy  (runs as user, /exec, shared secret)
                                   ▲   ▲
              ┌────────────────────┘   └───────────────────┐
   (A) OpenAPI 3.1 REST facade              (B) Remote MCP server (Streamable HTTP)
       └── ChatGPT GPT Actions                  ├── Claude custom connector
                                                └── ChatGPT Apps SDK (as it opens)

   Gemini: your own Gemini-API app w/ function calling → same proxy (no native plugin)
```

- **(A) OpenAPI facade** (e.g. FastAPI): clean REST paths + auto-generated
  OpenAPI 3.1 doc; each route calls a proxy `op`. Feeds ChatGPT GPT Actions and
  is trivially reusable by any function-calling client (including a Gemini-API
  app).
- **(B) MCP server** (FastMCP / MCP SDK): the same operations as MCP tools over
  Streamable HTTP; feeds Claude web now and ChatGPT Apps SDK later.

Both are small; both share request-building and auth to the proxy. Ship (A)
first (fastest to a working ChatGPT plugin), then (B).

## 5. Security (mandatory, do not skip)

Exposing the proxy to more clients widens the blast radius. Rules:

- **Never embed the proxy shared secret in a publicly shared GPT or client.**
  Keep the GPT private, or move to **OAuth / per-user secrets** for anything
  shared.
- **Prefer OAuth** for Claude connectors and OAuth-mode GPT Actions so each user
  authorizes their own data, rather than a single shared secret.
- **Scope per client**: a reflection/journal plugin should be able to append
  memory but **not** trigger `send_email`. Extend the proxy to accept a
  capability/scope tied to the secret or token.
- **Keep the `send_email` confirm-gate** (`notes.md`): no external agent gets
  un-gated send.
- Treat everything written to the Sheet as **user data the user chose to share
  across providers**; document that clearly.
- The facades should **validate inputs** and pass through only allowlisted `op`
  values — never let a caller pass an arbitrary proxy operation.

## 6. Recommended sequence

1. **ChatGPT GPT Action** over an OpenAPI facade (fastest, no MCP needed). Auth:
   private GPT + custom-header secret to start; OAuth before any sharing.
2. **Claude custom connector** via a remote MCP server (reuses the same proxy;
   reusable for ChatGPT Apps SDK later).
3. **Gemini**: document the "own Gemini-API app + function calling" workaround;
   revisit if Google opens a consumer connector platform.
4. **Harden**: OAuth + per-client scoping on the proxy before publishing anything
   beyond personal use.
