# Google Gmail + Google Tasks Integration Research

Research/design note for adding **Gmail** (read, draft, send) and **Google
Tasks** (read, create, update, complete) to Iris as predeclared tools called
from `shortcuts/iris.cherri`, using the same HTTP + `jsonRequest(...)` pattern
already used for NVIDIA NIM and Tavily.

> Scope note: this is a design document. It does **not** modify
> `shortcuts/iris.cherri`. It proposes tools, endpoints, an auth flow, and a
> phased plan so the change can be made deliberately later.

> Sources: Gmail API and Google Tasks API official documentation
> (developers.google.com/workspace), Google OAuth 2.0 token endpoint docs, and
> the Cherri language reference (cherrilang.org). Citations are inline.
> **Content from these sources was rephrased/summarized for licensing
> compliance; no large verbatim blocks are reproduced.**

---

## 1. Feasibility verdict per capability

| Capability | Verdict | Notes |
| --- | --- | --- |
| **Read mail** (search + read body/snippet) | Feasible | Gmail REST `messages.list` + `messages.get`. Body is base64url in `payload`, needs decoding on-device. Works with a read-only scope. |
| **Draft mail** | Feasible | `drafts.create` with a base64url raw MIME message. Safe (nothing sends). |
| **Send mail** | Feasible but gated | `messages.send` with base64url raw MIME. Must be **confirmation-gated**; never auto-send under hands-free Siri. |
| **Read tasks** | Feasible (easiest) | `tasklists.list` + `tasks.list`. Clean JSON, no encoding tricks. Best first integration. |
| **Write tasks** (add / complete / update) | Feasible | `tasks.insert` (POST) and `tasks.patch` (PATCH, `status:"completed"`). Low blast radius. |
| **Auth on a hands-free Shortcut** | Feasible with caveats | Requires a one-time out-of-band OAuth consent to mint a **refresh token**; per-run refresh -> access token. The token endpoint needs **form-encoding**, not JSON. See §4. |

The hard part is **not** the API calls; it is authentication without an
interactive browser step mid-run, plus one Shortcuts encoding limitation
(form-encoded token body) and one email limitation (base64url MIME). All are
solvable. Details and honest limits are in §4 and §8.

---

## 2. Gmail REST API

Base URL: `https://gmail.googleapis.com/gmail/v1`. The authenticated user is the
literal `me`. Every call carries `Authorization: Bearer <access_token>`.
(Source: Gmail API reference / guides, developers.google.com/workspace/gmail.)

### 2.1 Scopes (request the least you need)

| Scope | Grants |
| --- | --- |
| `https://www.googleapis.com/auth/gmail.readonly` | List/read messages and bodies. |
| `https://www.googleapis.com/auth/gmail.compose` | Create/manage drafts. |
| `https://www.googleapis.com/auth/gmail.send` | Send mail. |
| `https://www.googleapis.com/auth/gmail.modify` | Read + write short of full-delete (superset; avoid unless needed). |

For Iris, start with `gmail.readonly`. Add `gmail.compose` for drafts, and only
add `gmail.send` when send is actually shipped. (Source: Gmail API auth/scopes
docs.)

### 2.2 `messages.list` — search the inbox

```
GET https://gmail.googleapis.com/gmail/v1/users/me/messages?q=<query>&maxResults=10&labelIds=INBOX
Authorization: Bearer <access_token>
```

`q` uses the same search grammar as the Gmail search box (for example
`from:alice is:unread newer_than:2d`). The response is a thin list of ids only;
you then call `messages.get` per id. (Source: Gmail "List messages" guide.)

```json
{
  "messages": [
    {"id": "18f...", "threadId": "18f..."},
    {"id": "18e...", "threadId": "18e..."}
  ],
  "nextPageToken": "07894...",
  "resultSizeEstimate": 2
}
```

### 2.3 `messages.get` — read one message

```
GET https://gmail.googleapis.com/gmail/v1/users/me/messages/<id>?format=full
Authorization: Bearer <access_token>
```

`format` options:

- `metadata` — headers + `snippet`, no body. Cheapest; good for listing.
- `full` — parsed `payload` tree with headers and body parts.
- `minimal` — ids + labels + `snippet`.

Shape (trimmed):

```json
{
  "id": "18f...",
  "threadId": "18f...",
  "snippet": "Short preview of the message text ...",
  "payload": {
    "headers": [
      {"name": "From", "value": "Alice <alice@example.com>"},
      {"name": "Subject", "value": "Lunch?"}
    ],
    "body": {"data": "<base64url>"},
    "parts": [
      {"mimeType": "text/plain", "body": {"data": "<base64url>"}}
    ]
  }
}
```

Practical guidance for Iris:

- For a voice assistant, the **`snippet`** field (plain text, already decoded)
  is usually enough and avoids walking the MIME tree. Prefer
  `format=metadata` + `snippet` for the first version.
- The full body lives in `payload.body.data` or, for multipart mail, in
  `payload.parts[].body.data`, **base64url-encoded**. Decoding a nested MIME
  tree in Shortcuts is fiddly (see §8); do not attempt full-body extraction in
  v1. (Source: Gmail "List/get messages" guide.)

### 2.4 `drafts.create` — create a draft (nothing sends)

```
POST https://gmail.googleapis.com/gmail/v1/users/me/drafts
Authorization: Bearer <access_token>
Content-Type: application/json

{"message": {"raw": "<base64url RFC-2822 MIME message>"}}
```

Returns a draft `id` and the created `message`. (Source: Gmail "Drafts" guide;
the guide's cURL example uses exactly this body shape.)

### 2.5 `messages.send` — send mail (confirmation-gated)

```
POST https://gmail.googleapis.com/gmail/v1/users/me/messages/send
Authorization: Bearer <access_token>
Content-Type: application/json

{"raw": "<base64url RFC-2822 MIME message>"}
```

The `raw` value is a complete RFC-2822 message (the `To:`, `Subject:`, and body
lines) encoded as **base64url** (`+`->`-`, `/`->`_`, `=` padding stripped).
Google's own client libraries do exactly this before sending. (Source: Gmail
"Sending email" guide.)

Minimal RFC-2822 message to encode:

```
To: bob@example.com
Subject: Hello

This is the body.
```

---

## 3. Google Tasks REST API

Base URL: `https://tasks.googleapis.com/tasks/v1`. `Authorization: Bearer
<access_token>` on every call. (Source: Google Tasks API REST reference,
developers.google.com/workspace/tasks.)

### 3.1 Scopes

| Scope | Grants |
| --- | --- |
| `https://www.googleapis.com/auth/tasks.readonly` | List task lists and tasks. |
| `https://www.googleapis.com/auth/tasks` | Full read/write (insert, patch, complete, delete). |

### 3.2 `tasklists.list` — get the user's lists

```
GET https://tasks.googleapis.com/tasks/v1/users/@me/lists
Authorization: Bearer <access_token>
```

```json
{
  "kind": "tasks#taskLists",
  "items": [
    {"id": "MTIz...", "title": "My Tasks"},
    {"id": "NDU2...", "title": "Groceries"}
  ]
}
```

Most users only have the default `My Tasks` list. Iris can call this once and
use the first `items[].id` as the default `tasklist`. (Source: `tasklists.list`
reference.)

### 3.3 `tasks.list` — read tasks in a list

```
GET https://tasks.googleapis.com/tasks/v1/lists/<tasklist>/tasks?showCompleted=false&maxResults=100
Authorization: Bearer <access_token>
```

```json
{
  "kind": "tasks#tasks",
  "items": [
    {"id": "abc", "title": "Buy milk", "status": "needsAction", "due": "2026-01-10T00:00:00.000Z"},
    {"id": "def", "title": "Call dentist", "status": "needsAction"}
  ]
}
```

`status` is `needsAction` or `completed`. `due` is an RFC-3339 timestamp (date
portion is authoritative; Tasks ignores the time-of-day). (Source: `tasks.list`
reference.)

### 3.4 `tasks.insert` — add a task

```
POST https://tasks.googleapis.com/tasks/v1/lists/<tasklist>/tasks
Authorization: Bearer <access_token>
Content-Type: application/json

{"title": "Buy milk", "notes": "2%", "due": "2026-01-10T00:00:00.000Z"}
```

Returns the created `Task` (including its `id`). Requires the `tasks` scope.
(Source: `tasks.insert` reference.)

### 3.5 `tasks.patch` — update / complete a task

```
PATCH https://tasks.googleapis.com/tasks/v1/lists/<tasklist>/tasks/<task>
Authorization: Bearer <access_token>
Content-Type: application/json

{"status": "completed"}
```

`patch` semantics = send only changed fields. Set `status:"completed"` to
complete a task; set `status:"needsAction"` to reopen; send `title`/`notes`/`due`
to edit. Requires the `tasks` scope. (Source: `tasks.patch` reference.)

> Note: Cherri's `jsonRequest` HTTP method enum includes `PATCH`, so
> `tasks.patch` is expressible directly (see §5). (Source: Cherri Web actions
> reference, cherrilang.org.)

---

## 4. Authentication — the critical problem

A Shortcut running hands-free under Siri has **no way to run an interactive
OAuth consent browser mid-run**. So the interactive consent must happen **once,
out of band**, producing a long-lived **refresh token** that Iris can exchange
for a short-lived **access token** at the start of each run.

### Option 1 — OAuth2 refresh-token flow (recommended for a pure-Shortcut build)

One-time setup by the user (no code, ~10 minutes):

1. In Google Cloud Console: create a project, enable the **Gmail API** and
   **Tasks API**, configure the OAuth consent screen (External, add yourself as
   a **test user**), and create an **OAuth client ID** of type *Web application*
   (or *Desktop*). Note the **client id** and **client secret**.
2. Use the **OAuth 2.0 Playground** (or a tiny one-off local script) to run the
   consent flow with the exact scopes above and `access_type=offline` +
   `prompt=consent`, which returns a **refresh token**.
3. Paste **client id**, **client secret**, and **refresh token** into three new
   editable Text actions at the top of Iris — exactly like the existing
   `nvapi-` / `tvly-` key boxes.

At the start of each Iris run (or lazily before the first Google tool call),
Iris exchanges the refresh token for an access token:

```
POST https://oauth2.googleapis.com/token
Content-Type: application/x-www-form-urlencoded

grant_type=refresh_token
&client_id=<CLIENT_ID>
&client_secret=<CLIENT_SECRET>
&refresh_token=<REFRESH_TOKEN>
```

Response:

```json
{
  "access_token": "ya29....",
  "expires_in": 3599,
  "scope": "https://www.googleapis.com/auth/tasks.readonly ...",
  "token_type": "Bearer"
}
```

(Source: Google OAuth 2.0 token endpoint / "using OAuth 2.0 for web server
applications"; multiple implementations confirm the token endpoint requires the
`application/x-www-form-urlencoded` content type in the request body.)

**Shortcuts limitation and workaround.** Iris's `jsonRequest(...)` sends a body
with `Content-Type: application/json`. The Google token endpoint **rejects JSON**
and requires `application/x-www-form-urlencoded`; sending JSON yields
`invalid_request` / "Required parameter is missing: grant_type". Cherri's
`actions/web` include (already imported by Iris) provides a sibling builtin,
**`formRequest(url, method, body, headers)`**, which compiles to *Get Contents
of URL* with a **Form** request body — i.e. it form-encodes the dictionary. So
the token exchange must use `formRequest`, not `jsonRequest`. (Source: Cherri
Web actions reference: `jsonRequest`, `formRequest`, `fileRequest` all exist and
differ only in body encoding.)

Concrete token refresh, in the Iris style (literal dict, `formRequest`):

```cherri
@tokenResponse = formRequest("https://oauth2.googleapis.com/token", "POST", {
  "grant_type": "refresh_token",
  "client_id": "{@googleClientId}",
  "client_secret": "{@googleClientSecret}",
  "refresh_token": "{@googleRefreshToken}"
}, {"Accept": "application/json"})
@tokenDict = getDictionary(@tokenResponse)
@googleAccessToken = getValue(@tokenDict, "access_token")
```

A subsequent authorized API call reuses the existing `jsonRequest` pattern, with
the bearer token interpolated into the literal headers dict:

```cherri
@listsResponse = jsonRequest("https://tasks.googleapis.com/tasks/v1/users/@me/lists", "GET", {}, {
  "Authorization": "Bearer {@googleAccessToken}",
  "Accept": "application/json"
})
@listsDict = getDictionary(@listsResponse)
@listItems = getValue(@listsDict, "items")
@firstList = getFirstItem(@listItems)
@firstListDict = getDictionary(@firstList)
@defaultTaskList = getValue(@firstListDict, "id")
```

> If `formRequest` behaves unexpectedly on-device, a documented fallback is to
> POST the four parameters as a **URL query string** to
> `https://oauth2.googleapis.com/token?grant_type=refresh_token&client_id=...`
> with an empty body; Google's token endpoint accepts the parameters in the
> query string in practice. Treat this as a fallback, validated on a device,
> not the primary path.

Pros: pure Shortcut, no server to host, mirrors the existing key-paste UX.
Cons: three secrets live in the shortcut; the token endpoint needs the extra
`formRequest` path; the OAuth app stays in **Testing** unless verified (see §8),
which caps it at test users and can expire refresh tokens after 7 days.

### Option 2 — Google Apps Script web-app proxy (recommended for lowest on-device risk)

The user deploys a small **Apps Script web app** bound to their own Google
account. Because Apps Script runs *as the deploying user*, it can call
`GmailApp`/`Tasks` server-side with no token juggling on the phone. Iris calls a
single HTTPS endpoint with a **shared secret**, using the plain `jsonRequest`
pattern it already uses:

```cherri
@proxyResponse = jsonRequest("https://script.google.com/macros/s/<DEPLOY_ID>/exec", "POST", {
  "secret": "{@irisProxySecret}",
  "op": "tasks_list"
}, {"Content-Type": "application/json"})
```

The script authorizes Gmail/Tasks scopes once (at deploy/first-run consent) and
exposes small verbs (`gmail_search`, `gmail_read`, `draft_email`, `send_email`,
`tasks_list`, `tasks_add`, `tasks_complete`). The script returns already-shaped
JSON, so no base64url or form-encoding is needed on-device.

Pros: no OAuth token handling in the Shortcut, no `formRequest`/base64url work,
scopes and logic live server-side and are easy to revoke (delete the
deployment), and responses can be pre-shaped into Iris's envelope. Cons:
requires deploying and maintaining a script; the shared secret in the shortcut
is the only guard, so the endpoint must validate it and rate-limit; Apps Script
`doGet/doPost` has its own quotas.

**Recommendation.** For a self-hoster who wants the cleanest device code and the
least secret sprawl, **Option 2 (Apps Script proxy)** is the most practical and
the safest fit for a hands-free assistant. For a user who refuses to deploy any
server and wants everything inside the shortcut, **Option 1 (refresh token +
`formRequest`)** is fully workable. The two are not mutually exclusive: the tool
contract in §6 is identical regardless of which backend answers.

### Option 3 — Service account / domain-wide delegation (does NOT work here)

A service account can only access user data via **domain-wide delegation**,
which requires a **Google Workspace admin** to authorize the service account for
the domain. A **personal `@gmail.com` account has no Workspace admin console and
no domain to delegate**, so a service account cannot impersonate a consumer
Gmail user. Service accounts are therefore a dead end for personal Gmail/Tasks
and are only relevant to Workspace-org deployments. (Source: Google service
account / domain-wide delegation docs.)

---

## 5. Fit with Iris architecture and Cherri constraints

- **Predeclared tools only.** Each new capability is a fixed branch in the route
  dispatch (`if @tool.text == "..."`), added to the planner protocol tool list.
  The model can only select from the list; unknown names still fall through to
  the `ok=false` unknown-tool observation.
- **`jsonRequest` needs literal dicts.** Bodies and headers must be literal
  `{ ... }` with `{@var}` interpolation, never a variable. The token/access
  values are interpolated into literal header dicts (shown in §4).
- **`formRequest` for the token endpoint only.** Every Google *API* call uses
  `jsonRequest` (JSON body / GET). Only the OAuth token exchange uses
  `formRequest` (form body).
- **Parse with `getDictionary()` + `getValue()`, never bracket syntax.** Each
  nesting level needs its own `getDictionary()` first, matching the existing
  NIM/Tavily parse blocks. Arrays (`messages`, `items`) are iterated with
  `for x in @arr { @d = getDictionary(@x); ... }`, exactly like the Tavily
  `results` loop already in `iris.cherri`.
- **Never use `else if`.** Use flat guarded `if` blocks / default-then-override,
  per the compiler rule at the top of `iris.cherri`.
- **base64url for email.** Shortcuts' base64 encoding is standard base64; convert
  to url-safe (`+`->`-`, `/`->`_`, strip `=`) with `replaceText` before setting
  `raw` (see §8).
- **Envelope.** Every tool returns the standard
  `tool= / ok= / count= / result= / records= / error=` text envelope so the
  planner can chain tools, identical to existing routes (`docs/routes.md`,
  `agentic-loop.md`).

---

## 6. Proposed Iris predeclared tools

All follow the existing flat-JSON planner contract (top-level string args only,
`return_to_agent` to chain). Arguments reuse existing field names where possible
(`query`, `title`, `notes`, `recipient`, `body`) and add a few Google-specific
ones (`message_id`, `task_id`, `due`).

| Tool | Args | API mapping | Notes |
| --- | --- | --- | --- |
| `tasks_list` | (none) | `tasklists.list` then `tasks.list` on default list | Read-only. Best first tool. Returns titles + status in `records`. |
| `tasks_add` | `title`, `notes`, `due` | `tasks.insert` | Write. `due` optional RFC-3339 date; empty allowed. |
| `tasks_complete` | `title` **or** `task_id` | `tasks.list` to resolve title -> id, then `tasks.patch` `{status:"completed"}` | Write. If `title` matches >1 task, return `ok=false` asking the planner to disambiguate via `ask_user`. |
| `gmail_search` | `query` | `messages.list` (+ per-id `messages.get?format=metadata`) | Read-only. Returns sender/subject/snippet rows in `records`. Cap at ~5 ids for latency. |
| `gmail_read` | `message_id` **or** `query` | `messages.get?format=metadata` (or resolve top hit of a search) | Read-only. Returns headers + `snippet`. Full body deferred (see §8). |
| `draft_email` | `recipient`, `subject`(fold into body), `body` | `drafts.create` | Safe write (nothing sends). Build RFC-2822, base64url-encode, POST. |
| `send_email` | `recipient`, `body`, `confirm` | `messages.send` | **Confirmation-gated.** Only sends when `confirm=="yes"`; otherwise returns `ok=false` telling the planner to confirm with the user first. |

### How the planner should call them

- Add a one-line summary of each tool to the `@protocol` tool list (the current
  prompt already enumerates tools inline). Example additions:
  `tasks_list(no args) reads your Google Tasks, tasks_add(title,notes,due),
  tasks_complete(title) marks a task done, gmail_search(query) searches your
  Gmail, gmail_read(query) reads one email, draft_email(recipient,body) creates
  a Gmail draft and never sends, send_email(recipient,body) sends a Gmail
  message only after you confirm.`
- Remove/replace the current "No access to Mail, Gmail" line in the protocol
  once these ship, but **keep the "nothing is ever auto-sent without
  confirmation"** clause.
- `send_email` should be a two-step dance the model already supports: the model
  calls `draft_email` (or proposes text) -> speaks it -> `ask_user` "Send it?"
  -> only on "yes" calls `send_email` with `confirm="yes"`. The tool itself also
  hard-checks `confirm=="yes"` so a model slip cannot send silently.

### Example observation envelopes

```text
tool=tasks_list
ok=true
count=2
result=You have 2 open tasks.
records=- Buy milk (needsAction); - Call dentist (needsAction)
error=
```

```text
tool=send_email
ok=false
count=0
result=
records=
error=Not sent. Confirmation required: re-call send_email with confirm=yes only after the user agrees.
```

---

## 7. Security and privacy

- **Secrets stay local.** Client id/secret/refresh token (Option 1) or the proxy
  shared secret (Option 2) live only in editable Text actions in the user's
  local copy of the shortcut, exactly like the `nvapi-`/`tvly-` keys. They must
  never be committed or baked into a distributed artifact. Extend
  `scripts/validate-shortcut.py` to also reject real-looking Google client
  secrets / refresh tokens (e.g. `GOCSPX-`, `1//` refresh-token prefixes) in
  compiled output, mirroring the existing `nvapi-`/`tvly-` guards.
- **Scope minimization.** Ship read-only scopes first (`gmail.readonly`,
  `tasks.readonly`). Add `gmail.compose`, then `tasks`, then `gmail.send` only as
  each write tool lands. Never request `gmail.modify` or broader.
- **Never auto-send.** `send_email` is confirmation-gated in two places: the
  planner protocol and a hard `confirm=="yes"` check in the tool branch.
  Drafts (`draft_email`) are the default, safe path.
- **Revocation.** Option 1: the user revokes access at
  myaccount.google.com/permissions, or deletes the OAuth client. Option 2:
  delete/redeploy the Apps Script web app, or rotate the shared secret. Document
  both.
- **Data exposure.** Email snippets and task titles are sent to the NVIDIA NIM
  planner (and Tavily is unaffected). This is the same privacy posture already
  noted in `docs/configuration.md`; call it out explicitly for mail, since email
  content is more sensitive than a calendar title. Prefer `snippet`/metadata over
  full bodies to limit what leaves the device.
- **Transport.** All endpoints are HTTPS. The per-host network permission prompt
  must be primed once (add `oauth2.googleapis.com`, `gmail.googleapis.com`,
  `tasks.googleapis.com`, or `script.google.com` to the setup primer block).

---

## 8. Honest limitations and risks (iOS Shortcuts specifics)

- **OAuth app verification.** An unverified consumer OAuth app stays in
  **Testing** mode: only added **test users** can consent, the consent screen
  shows an "unverified app" warning, and **refresh tokens for a Testing app can
  expire after 7 days**, forcing the user to re-mint the token. Moving to
  Production requires Google's verification (especially for the "restricted"
  Gmail scopes), which is a real review process. This is the single biggest
  practical risk for Option 1, and a strong reason to prefer the Apps Script
  proxy (Option 2), which does not need app verification for personal use.
- **Access-token lifetime.** Access tokens last ~1 hour (`expires_in: 3599`).
  Refreshing at the top of each run is fine; a single Iris conversation is far
  shorter than an hour, so one refresh per run is enough.
- **Form-encoding.** The token endpoint requires `application/x-www-form-urlencoded`;
  `jsonRequest` cannot produce that, so `formRequest` (or the query-string
  fallback) is mandatory and must be device-validated. This is the one place
  Iris deviates from its all-`jsonRequest` pattern.
- **base64url email encoding is the fiddliest device step.** Shortcuts can
  base64-encode text, but Gmail's `raw` field needs **url-safe** base64 with
  padding stripped. Plan: build the RFC-2822 string, base64-encode it, then
  `replaceText('+','-')`, `replaceText('/','_')`, `replaceText('=','')`. This
  must be validated on a device; if it proves unreliable, the **Apps Script
  proxy builds the MIME server-side** and removes the problem entirely — another
  point for Option 2 for the send/draft tools specifically.
- **Reading full bodies is hard.** Multipart MIME means walking
  `payload.parts[]`, picking the `text/plain` part, and base64url-**decoding**
  it. For a voice assistant, `snippet` is a much better cost/quality trade. Do
  not ship full-body extraction in v1.
- **Latency budget.** iOS gives `Get Contents of URL` ~25s and Siri is
  impatient. Option 1 adds one token call before the API call (2 round-trips);
  `gmail_search` that then fans out to N `messages.get` calls can blow the
  budget. Mitigate: cap results (~5), prefer `format=metadata`, and cache the
  access token in a variable for the whole run so only one refresh happens.
- **Quotas.** Gmail and Tasks APIs have per-user rate limits; normal assistant
  use is nowhere near them, but a runaway loop could trip them. Iris's existing
  per-request tool-call budget (3) already bounds this.
- **No push / no background inbox.** These are request/response REST calls made
  during a run. Iris cannot watch the inbox or notify on new mail; it only reads
  when asked.

---

## 9. Phased implementation plan

**Phase 0 — Auth spike (validate the hard parts first).**
Pick Option 1 or 2. If Option 1, prove `formRequest` performs the refresh-token
exchange on a real device and returns an `access_token`. If Option 2, deploy a
minimal Apps Script that echoes an authorized `tasks_list`. Prime the new
network hosts in the setup block. No planner changes yet.

**Phase 1 — Read-only Tasks (`tasks_list`).**
Lowest risk, cleanest JSON, no encoding. Wire one tool, add it to the protocol,
return the standard envelope. This validates the whole auth-plus-parse pipeline
end to end.

**Phase 2 — Read-only Gmail (`gmail_search`, `gmail_read`).**
Use `gmail.readonly`, `format=metadata` + `snippet`. Cap result counts for
latency. Still no writes.

**Phase 3 — Safe writes (`tasks_add`, `tasks_complete`, `draft_email`).**
Add `tasks` and `gmail.compose` scopes. `draft_email` exercises base64url
encoding without the risk of sending. `tasks_complete` resolves title->id and
patches `status:"completed"`.

**Phase 4 — Confirmation-gated send (`send_email`).**
Add `gmail.send`. Enforce the two-step confirm (planner + hard `confirm=="yes"`
check). Update `docs/routes.md`, `agentic-loop.md`, and `docs/configuration.md`
(setup steps, scopes, revocation), and extend `scripts/validate-shortcut.py`
secret scanning.

Each phase is independently shippable and reversible, and each keeps Iris's
"predeclared tools only, draft-before-send" safety posture intact.

---

## 10. Source list

- Gmail API reference and guides (messages list/get/send, drafts):
  developers.google.com/workspace/gmail/api
- Google Tasks API REST reference (tasklists.list, tasks.list/insert/patch):
  developers.google.com/workspace/tasks
- Google OAuth 2.0 token endpoint / web-server flow (refresh token exchange,
  form-encoded body): developers.google.com/identity/protocols/oauth2
- Service accounts & domain-wide delegation (why it excludes personal Gmail):
  Google Cloud IAM / Workspace admin docs
- Cherri language — Web actions (`jsonRequest`, `formRequest`, `fileRequest`,
  `downloadURL`; HTTP method enum incl. PATCH): cherrilang.org/language/standard/web
- Iris internal docs: `agentic-loop.md`, `docs/routes.md`, `docs/architecture.md`,
  `docs/configuration.md`, `shortcuts/iris.cherri`

*Content from external sources above was rephrased and summarized for licensing
compliance; no large verbatim excerpts are reproduced.*
