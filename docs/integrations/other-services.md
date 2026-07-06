# Other personal-productivity services via the Apps Script proxy hub

Design/research note. A **menu of options**, not a build. It sketches how a few
non-Google personal-productivity services would plug into Iris through the
**Google Apps Script web-app proxy** described in
[`docs/apps-script-proxy.md`](../apps-script-proxy.md), so Iris gains new tool
names without any new on-device secrets or OAuth juggling.

> Scope note: this document does **not** modify `shortcuts/iris.cherri`. It
> specifies the proxy op-branches, the proposed Iris tool names, and setup
> steps so the change can be made deliberately later. Every service here is
> reached the same way: one Apps Script web app, called by Iris with
> `{secret, op, ...}`, that fans out to the third party with `UrlFetchApp`
> and a token held server-side.

> Licensing note: **Content from these docs was rephrased/summarized for
> compliance; no large verbatim excerpts.** Citations are inline and collected
> in [§7 Sources](#7-sources).

---

## 1. How non-Google services reuse the hub

The proxy already runs as the deploying owner and guards every call with a
shared secret (see `docs/apps-script-proxy.md` §2, §7). Google services use the
built-in `GmailApp` / advanced `Tasks` service. **Non-Google** services need
two extra pieces:

1. **`UrlFetchApp` for outbound HTTPS.** The manifest must keep the
   `https://www.googleapis.com/auth/script.external_request` scope so the script
   can make outbound calls. `UrlFetchApp.fetch(url, params)` sends the request;
   set `muteHttpExceptions: true` so a 4xx/5xx returns a response object you can
   inspect instead of throwing. (Source: Apps Script `UrlFetchApp` reference.)
2. **A per-service token in Script Properties, not in the request.** Store each
   third-party token once with
   `PropertiesService.getScriptProperties().setProperty('TODOIST_TOKEN', '...')`
   and read it at call time with `.getProperty(...)`. The token never leaves
   Google's servers and never touches the device or the shortcut. (Source: Apps
   Script `PropertiesService` / Script Properties reference.)

Everything else is unchanged: each op returns the standard Iris envelope
`{tool, ok, count, result, records, error}` as JSON via `ContentService`, and
Iris parses the same six keys it already reads. Iris simply gets **new tool
names**; the planner protocol and loop contract from `agentic-loop.md` are
untouched.

### Shared helpers (add once to `Code.gs`)

```javascript
// Read a per-service token stored once via Script Properties.
function _prop(key) {
  var v = PropertiesService.getScriptProperties().getProperty(key);
  return v ? String(v) : '';
}

// Thin UrlFetchApp wrapper: returns { code, json, text }.
function _http(url, method, headers, bodyObj) {
  var params = {
    method: method || 'get',
    headers: headers || {},
    muteHttpExceptions: true,       // inspect 4xx/5xx instead of throwing
    followRedirects: true
  };
  if (bodyObj !== undefined && bodyObj !== null) {
    params.contentType = 'application/json';
    params.payload = JSON.stringify(bodyObj);
  }
  var resp = UrlFetchApp.fetch(url, params);
  var code = resp.getResponseCode();
  var text = resp.getContentText();
  var json = null;
  try { json = text ? JSON.parse(text) : null; } catch (e) { json = null; }
  return { code: code, json: json, text: text };
}
```

`_env(tool, ok, count, result, records, error)` and `_json(obj)` are reused
verbatim from `docs/apps-script-proxy.md` §4.

---

## 2. Todoist

Tasks: **list**, **create**, **close**. Cleanest possible token auth.

> **Honest version note.** The user brief says "REST API v2," but Todoist has
> **deprecated** both REST API v2 and Sync v9 in favor of a **unified Todoist
> API v1**, with the old endpoints scheduled to **shut down on 2026-02-10**. A
> direct call to `api.todoist.com/rest/v2/...` already returns a
> "this endpoint is deprecated" error for some accounts. This sketch therefore
> targets the **current v1** endpoints (which cover the exact same list/create/
> close operations); porting from the v2 shapes is mechanical. (Source: Todoist
> developer guides + "Final Shutdown of Sync API v9 and REST API v2"
> announcement.)

- **Auth type:** personal API token (a static per-account bearer token, not a
  full OAuth dance). Get it from Todoist → **Settings → Integrations →
  Developer → API token**. Copy it once.
- **Base URL:** `https://api.todoist.com/api/v1`
- **Header:** `Authorization: Bearer <TODOIST_TOKEN>` (plus
  `Content-Type: application/json` on writes).

### Key endpoints

| Op | Method + path | Request | Response (shape) |
| --- | --- | --- | --- |
| List active tasks | `GET /api/v1/tasks` | optional `project_id`, `label`, `limit`, `cursor` query params | `{ "results": [ {id, content, due, priority, ...} ], "next_cursor": "..." }` (paginated) |
| Create task | `POST /api/v1/tasks` | JSON `{ "content": "...", "due_string": "tomorrow 9am", "priority": 2 }` | the created task object `{ id, content, url, due, ... }` |
| Close task | `POST /api/v1/tasks/{task_id}/close` | empty body | `204 No Content` on success |

(Source: Todoist API v1 reference — "Get Tasks", "Create a Task", "Close Task";
base URL + Bearer token shown in the v1 auth example.)

### Apps Script op-branch sketch

```javascript
// Add to the doPost dispatcher:
//   if (op === 'todoist_list')     { return _json(todoistList(req)); }
//   if (op === 'todoist_add')      { return _json(todoistAdd(req)); }
//   if (op === 'todoist_complete') { return _json(todoistComplete(req)); }

var TODOIST_BASE = 'https://api.todoist.com/api/v1';

function _todoistHeaders() {
  return { 'Authorization': 'Bearer ' + _prop('TODOIST_TOKEN') };
}

function todoistList(req) {
  var url = TODOIST_BASE + '/tasks?limit=' + (Number(req.limit) || 5);
  var r = _http(url, 'get', _todoistHeaders());
  if (r.code !== 200 || !r.json) {
    return _env('todoist_list', false, 0, '', '', 'Todoist list failed (HTTP ' + r.code + ').');
  }
  var items = r.json.results || [];
  var records = items.slice(0, 5).map(function (t) {
    return '- ' + t.content + (t.due && t.due.string ? ' (due ' + t.due.string + ')' : '');
  }).join('; ');
  return _env('todoist_list', true, items.length,
    'You have ' + items.length + ' active Todoist task' + (items.length === 1 ? '' : 's') + '.',
    records, '');
}

function todoistAdd(req) {
  var content = String(req.title || req.content || '').trim();
  if (!content) { return _env('todoist_add', false, 0, '', '', 'A task title is required.'); }
  var body = { content: content };
  if (req.due)      { body.due_string = String(req.due); }   // natural language, e.g. "tomorrow 9am"
  if (req.priority) { body.priority = Number(req.priority); } // 1..4
  var r = _http(TODOIST_BASE + '/tasks', 'post', _todoistHeaders(), body);
  if (r.code !== 200 && r.code !== 204) {
    return _env('todoist_add', false, 0, '', '', 'Todoist add failed (HTTP ' + r.code + ').');
  }
  return _env('todoist_add', true, 1, 'Added Todoist task: ' + content + '.', '- ' + content, '');
}

function todoistComplete(req) {
  var id = String(req.task_id || '').trim();
  if (!id) { return _env('todoist_complete', false, 0, '', '', 'A task_id is required (get it from todoist_list).'); }
  var r = _http(TODOIST_BASE + '/tasks/' + encodeURIComponent(id) + '/close', 'post', _todoistHeaders());
  if (r.code !== 204 && r.code !== 200) {
    return _env('todoist_complete', false, 0, '', '', 'Todoist close failed (HTTP ' + r.code + ').');
  }
  return _env('todoist_complete', true, 1, 'Closed Todoist task.', '', '');
}
```

### Proposed Iris tools

- `todoist_list` — "List my open Todoist tasks (returns titles + due dates)."
- `todoist_add` — "Create a Todoist task from a title, optional natural-language due date, and priority."
- `todoist_complete` — "Close a Todoist task by its `task_id` (obtained from `todoist_list`)."

### Setup

1. Todoist → Settings → Integrations → Developer → copy the **API token**.
2. In the Apps Script editor: **Project Settings → Script Properties → Add**,
   key `TODOIST_TOKEN`, value = the token. Save.
3. Add the three op-branches + helpers, redeploy the existing deployment as a
   **new version** (keeps the same `/exec` URL).
4. Smoke test: POST `{secret, op:"todoist_list"}` to `/exec`.

---

## 3. GitHub

List issues / notifications, and create an issue. Personal access token.

- **Auth type:** **personal access token (PAT)** — prefer a **fine-grained** PAT
  scoped to specific repos with `Issues: Read and write`; a classic PAT with the
  `repo` and `notifications` scopes also works. Create at GitHub → **Settings →
  Developer settings → Personal access tokens**. (Source: GitHub
  "Authenticating to the REST API" / "Managing your personal access tokens" —
  fine-grained tokens are recommended.)
- **Base URL:** `https://api.github.com`
- **Headers:**
  - `Authorization: Bearer <GITHUB_TOKEN>`
  - `Accept: application/vnd.github+json`
  - `X-GitHub-Api-Version: 2022-11-28`
  - `User-Agent: <something>` (GitHub **requires** a User-Agent header or it
    rejects the request).

### Key endpoints

| Op | Method + path | Request | Response (shape) |
| --- | --- | --- | --- |
| Issues assigned to you | `GET /issues` | query `filter=assigned&state=open&per_page=5` | array of issue objects `[{number, title, html_url, repository, ...}]` |
| Repo issues | `GET /repos/{owner}/{repo}/issues` | query `state`, `labels`, `per_page` | array of issues (note: PRs also appear; filter on absence of `pull_request` key) |
| Notifications | `GET /notifications` | query `all=false` (unread only) | array `[{id, subject:{title,type,url}, repository, unread}]` |
| Create issue | `POST /repos/{owner}/{repo}/issues` | JSON `{ "title": "...", "body": "...", "labels": [...] }` | the created issue `{ number, html_url, state, ... }` |

(Source: GitHub REST reference — "Issues", "Notifications", "Getting started
with the REST API" for path params like `{owner}`/`{repo}`. Note that GitHub
treats every PR as an issue, so issue lists can include PRs, identifiable by the
`pull_request` key.)

### Apps Script op-branch sketch

```javascript
// Dispatcher:
//   if (op === 'github_issues')        { return _json(githubIssues(req)); }
//   if (op === 'github_notifications') { return _json(githubNotifications(req)); }
//   if (op === 'github_create_issue')  { return _json(githubCreateIssue(req)); }

var GITHUB_BASE = 'https://api.github.com';

function _githubHeaders() {
  return {
    'Authorization': 'Bearer ' + _prop('GITHUB_TOKEN'),
    'Accept': 'application/vnd.github+json',
    'X-GitHub-Api-Version': '2022-11-28',
    'User-Agent': 'iris-apps-script-proxy'   // required by GitHub
  };
}

function githubIssues(req) {
  var url = GITHUB_BASE + '/issues?filter=assigned&state=open&per_page=5';
  var r = _http(url, 'get', _githubHeaders());
  if (r.code !== 200 || !r.json) {
    return _env('github_issues', false, 0, '', '', 'GitHub issues failed (HTTP ' + r.code + ').');
  }
  var rows = r.json.slice(0, 5).map(function (i) {
    return '#' + i.number + ' ' + i.title + ' (' + (i.repository ? i.repository.full_name : '') + ')';
  }).join('; ');
  return _env('github_issues', true, r.json.length,
    'You have ' + r.json.length + ' open assigned issue' + (r.json.length === 1 ? '' : 's') + '.',
    rows, '');
}

function githubNotifications(req) {
  var r = _http(GITHUB_BASE + '/notifications?all=false', 'get', _githubHeaders());
  if (r.code !== 200 || !r.json) {
    return _env('github_notifications', false, 0, '', '', 'GitHub notifications failed (HTTP ' + r.code + ').');
  }
  var rows = r.json.slice(0, 5).map(function (n) {
    return (n.subject ? n.subject.title : '(no title)') + ' [' + (n.subject ? n.subject.type : '') + ']';
  }).join('; ');
  return _env('github_notifications', true, r.json.length,
    'You have ' + r.json.length + ' unread notification' + (r.json.length === 1 ? '' : 's') + '.',
    rows, '');
}

function githubCreateIssue(req) {
  var owner = String(req.owner || '').trim();
  var repo  = String(req.repo || '').trim();
  var title = String(req.title || '').trim();
  if (!owner || !repo) { return _env('github_create_issue', false, 0, '', '', 'owner and repo are required.'); }
  if (!title)          { return _env('github_create_issue', false, 0, '', '', 'A title is required.'); }
  var body = { title: title };
  if (req.body) { body.body = String(req.body); }
  var url = GITHUB_BASE + '/repos/' + encodeURIComponent(owner) + '/' + encodeURIComponent(repo) + '/issues';
  var r = _http(url, 'post', _githubHeaders(), body);
  if (r.code !== 201 || !r.json) {
    return _env('github_create_issue', false, 0, '', '', 'GitHub create issue failed (HTTP ' + r.code + ').');
  }
  return _env('github_create_issue', true, 1,
    'Created issue #' + r.json.number + ' in ' + owner + '/' + repo + '.',
    r.json.html_url || '', '');
}
```

### Proposed Iris tools

- `github_issues` — "List open GitHub issues assigned to me across repos."
- `github_notifications` — "List my unread GitHub notifications (titles + type)."
- `github_create_issue` — "Open a GitHub issue in `owner/repo` with a title and optional body."

### Setup

1. GitHub → Settings → Developer settings → **Fine-grained PAT** → select the
   repos → grant `Issues: Read and write` (+ read for notifications). Copy it.
2. Apps Script **Script Properties**: key `GITHUB_TOKEN`, value = the PAT.
3. Add op-branches + helpers; redeploy as a new version.
4. Smoke test: `{secret, op:"github_issues"}`.

---

## 4. Slack (post a message)

Chosen as the third service — cleanest single-token write. (OpenWeather is an
even simpler read-only API-key example; see the feasibility table.)

- **Auth type:** **Bot token** (`xoxb-...`), a static token minted per Slack app.
  Create a Slack app at **api.slack.com/apps → Create New App**, add the
  **`chat:write`** OAuth scope under **OAuth & Permissions**, install the app to
  the workspace, and copy the **Bot User OAuth Token**. Invite the bot to the
  target channel. (Source: Slack `chat.postMessage` / OAuth & Permissions docs.)
- **Base URL:** `https://slack.com/api` (Web API — each method is a path, e.g.
  `/chat.postMessage`).
- **Header:** `Authorization: Bearer <SLACK_BOT_TOKEN>` +
  `Content-Type: application/json`.

### Key endpoints

| Op | Method + path | Request | Response (shape) |
| --- | --- | --- | --- |
| Post a message | `POST /api/chat.postMessage` | JSON `{ "channel": "C0123 or #name", "text": "..." }` | `{ "ok": true, "ts": "1700...", "channel": "C0123" }` or `{ "ok": false, "error": "channel_not_found" }` |
| List channels (optional) | `GET /api/conversations.list` | query `types=public_channel&limit=100` | `{ "ok": true, "channels": [{id, name}] }` |

Slack always returns HTTP 200; **success is signaled by the JSON `ok` field**,
not the status code. Rate limit is roughly one message per second per channel
(tighter for non-Marketplace apps in 2025+). (Source: Slack Web API — POST to
`https://slack.com/api/<method>` with a bot token; `ok`-flag convention;
`chat.postMessage` rate guidance.)

### Apps Script op-branch sketch

```javascript
// Dispatcher:
//   if (op === 'slack_post') { return _json(slackPost(req)); }

function slackPost(req) {
  // Confirmation gate, mirroring send_email in apps-script-proxy.md §4.
  if (String(req.confirm || '').toLowerCase() !== 'yes') {
    return _env('slack_post', false, 0, '', '',
      'Not sent. Re-call slack_post with confirm=yes only after the user agrees.');
  }
  var channel = String(req.channel || '').trim();
  var text = String(req.text || req.body || '').trim();
  if (!channel) { return _env('slack_post', false, 0, '', '', 'A channel is required.'); }
  if (!text)    { return _env('slack_post', false, 0, '', '', 'Message text is required.'); }

  var headers = { 'Authorization': 'Bearer ' + _prop('SLACK_BOT_TOKEN') };
  var r = _http('https://slack.com/api/chat.postMessage', 'post', headers,
    { channel: channel, text: text });

  // Slack returns HTTP 200 even on logical failure; check the ok flag.
  if (!r.json || r.json.ok !== true) {
    var why = (r.json && r.json.error) ? r.json.error : ('HTTP ' + r.code);
    return _env('slack_post', false, 0, '', '', 'Slack post failed: ' + why + '.');
  }
  return _env('slack_post', true, 1, 'Posted to Slack channel ' + channel + '.', '', '');
}
```

### Proposed Iris tool

- `slack_post` — "Post a message to a Slack channel (hard-gated on `confirm=yes`,
  like `send_email`)."

### Setup

1. api.slack.com/apps → create app → **OAuth & Permissions** → add bot scope
   `chat:write` → **Install to Workspace** → copy the **Bot User OAuth Token**.
2. Invite the bot to the destination channel (`/invite @yourbot`).
3. Apps Script **Script Properties**: key `SLACK_BOT_TOKEN`, value = `xoxb-...`.
4. Add the op-branch + helpers; redeploy as a new version.
5. Smoke test: `{secret, op:"slack_post", channel:"#test", text:"hi", confirm:"yes"}`.

---

## 5. Feasibility verdict table

Services users frequently ask about, and how cleanly each fits the
token-in-Script-Properties + `UrlFetchApp` proxy pattern.

| Service | Auth model | Proxy fit | Verdict |
| --- | --- | --- | --- |
| **Todoist** | Personal API token (static bearer) | Trivial — one token, clean REST | ✅ **Easy.** Use **API v1** (v2 deprecated, shuts down 2026-02-10). |
| **GitHub** | Personal access token (fine-grained preferred) | Trivial — static token, requires `User-Agent` header | ✅ **Easy.** Issues + notifications + create. |
| **Slack** | Bot token (`xoxb-`, static after install) | Easy — one-time app install, then static token; check `ok` flag | ✅ **Easy.** Post messages; gate writes on confirm. |
| **OpenWeather** | API key (query param `appid=`) | Trivial — read-only, no user data | ✅ **Easy.** Simplest possible; already partly covered by native `weather_summary`. |
| **Notion** | Internal integration token (static bearer) + `Notion-Version` header | Easy — share pages/DB with the integration first | ✅ **Easy** once pages are shared with the integration. |
| **Linear** | Personal API key (static) over GraphQL | Easy — single endpoint, GraphQL body | ✅ **Easy** (GraphQL, not REST). |
| **Trello** | API key + token (static pair, query params) | Easy — no OAuth server needed | ✅ **Easy.** |
| **Spotify** | 3-legged OAuth, short-lived access token | Needs a refresh token stored server-side | ⚠️ **Feasible with work** — see §5.1. |
| **Google Calendar/Tasks/Gmail** | Owner-authorized via Apps Script itself | Native — no `UrlFetchApp` token needed | ✅ **Easy** (built-in/advanced services; the original proxy doc). |
| **Fitbit / Withings** | 3-legged OAuth, refreshable token | Refresh token server-side | ⚠️ **Feasible with work** — see §5.1. |
| **Apple Reminders/Notes/Calendar** | On-device only (no public cloud API) | Not reachable from a server | ❌ **Not via proxy** — use Iris's native Shortcut actions instead. |
| **Kindle / Amazon (library + highlights)** | No clean public consumer API | — | ❌ **Not feasible cleanly** — see §5.2. |

### 5.1 Services requiring full 3-legged OAuth

Services like **Spotify, Fitbit, Withings, Strava**, and most consumer OAuth
apps don't hand out a static personal token. They issue a **short-lived access
token** plus a **refresh token** after an interactive authorization-code flow.
That is still **feasible** through the proxy, with one extra one-time step:

1. Register an OAuth app with the provider (client id + secret, redirect URI).
2. Run the consent flow **once** (in a browser, or a small helper) to obtain a
   **refresh token**.
3. Store `CLIENT_ID`, `CLIENT_SECRET`, and `REFRESH_TOKEN` in **Script
   Properties**.
4. In the op-branch, exchange the refresh token for a fresh access token via a
   `UrlFetchApp` POST to the provider's token endpoint (form-encoded), cache it
   in Script Properties with its expiry, and reuse until it expires — then
   refresh again. This mirrors the token-refresh logic from the current Path A
   Google flow, except it now lives **server-side** in the script instead of in
   the shortcut.

The device still holds nothing but the proxy URL + shared secret. The cost is
the one-time consent flow and refresh-token maintenance on the server, plus
handling provider-specific expiry and rotation. Note some providers rotate the
refresh token on each use — persist the new one when that happens.

### 5.2 Honest negative: Kindle / Amazon

There is **no clean, public, consumer-facing Amazon API** for a user's Kindle
library or reading highlights/notes. Amazon's official developer programs
(Product Advertising API, Login with Amazon, Selling Partner API, Alexa) are
aimed at commerce, sign-in, and sellers — **none** exposes "my books" or "my
highlights."

The only known route to Kindle highlights is **unofficial scraping of
`read.amazon.com/notebook`** (the Kindle Notes & Highlights web page) by
replaying a logged-in session cookie and parsing HTML. This is:

- **Fragile** — it breaks whenever Amazon changes the page markup, and it needs
  a valid, refreshed session cookie (Amazon sessions expire and are guarded by
  bot detection/CAPTCHA).
- **Against Amazon's Conditions of Use** — automated scraping of an
  authenticated Amazon session violates their terms; doing it from a shared
  server endpoint compounds the risk.
- **A poor fit for the proxy** — it can't rely on a stable token, would need
  cookie harvesting the user must repeat, and could get the account flagged.

**Verdict: do not build a Kindle/Amazon proxy tool.** If a user wants highlights
in Iris, the realistic path is manual **export** (e.g. copy/share highlight text
into Iris, then use the existing `summarize_provided_text` tool), not a live
integration. State this limitation plainly rather than shipping a scraper.

---

## 6. What stays the same across all of these

- **Envelope + loop contract unchanged.** Every op returns
  `{tool, ok, count, result, records, error}`; the planner protocol and bounded
  loop from `agentic-loop.md` don't change. New services = new **tool names**
  only, added to the registry and `docs/routes.md`.
- **No new on-device secrets.** All third-party tokens live in **Script
  Properties**, server-side. The shortcut still holds only the proxy `/exec` URL
  + shared secret.
- **One network host to prime.** Iris still talks only to `script.google.com`
  (302 redirect to `script.googleusercontent.com` auto-followed, per
  `apps-script-proxy.md` §2). The proxy reaches out to Todoist/GitHub/Slack via
  `UrlFetchApp`; those hosts never appear on the device.
- **Writes stay gated.** Anything that sends or mutates externally
  (`slack_post`, and any future send/delete) reuses the two-place `confirm=yes`
  gate from `send_email`.
- **Latency budget.** Each proxy op is one phone-side round-trip; the script
  fans out server-side. Cap results (≈5) and keep records short to stay inside
  Iris's ~25s Siri budget (`apps-script-proxy.md` §8).

---

## 7. Sources

- **Todoist API v1** (base URL `api.todoist.com/api/v1`, Bearer token, Get
  Tasks, Create a Task, Close Task): developer.todoist.com/api/v1
- **Todoist deprecation** (REST v2 + Sync v9 deprecated; shutdown 2026-02-10):
  developer.todoist.com/guides, developer.todoist.com/rest/v2 (deprecation
  banner), and the Doist "Final Shutdown of Sync API v9 and REST API v2"
  announcement.
- **GitHub REST API** (base `api.github.com`, `Authorization: Bearer`,
  `Accept: application/vnd.github+json`, `X-GitHub-Api-Version: 2022-11-28`,
  required `User-Agent`; Issues, Notifications, Create issue; PAT guidance):
  docs.github.com/rest/issues/issues, docs.github.com/rest/activity/notifications,
  docs.github.com/rest/overview/authenticating-to-the-rest-api,
  docs.github.com/rest/using-the-rest-api/getting-started-with-the-rest-api
- **Slack Web API** (POST to `https://slack.com/api/<method>` with a bot token,
  `chat.postMessage`, `ok`-flag convention, `chat:write` scope, rate limits):
  docs.slack.dev / api.slack.com/methods/chat.postMessage and OAuth &
  Permissions docs.
- **OpenWeather** (API key via `appid` query param): openweathermap.org/api
- **Apps Script `UrlFetchApp`** (`fetch`, `muteHttpExceptions`, `followRedirects`)
  and **`PropertiesService` / Script Properties**:
  developers.google.com/apps-script/reference/url-fetch and
  /apps-script/reference/properties
- **Iris internal docs:** `docs/apps-script-proxy.md`, `agentic-loop.md`,
  `docs/routes.md`, `docs/google-integrations-research.md`

*Content from these providers' documentation was rephrased/summarized for
compliance with licensing restrictions; no large verbatim excerpts are
reproduced.*
