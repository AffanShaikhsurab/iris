# Apps Script Web-App Proxy — a token-free Google backend for Iris

Design/research note for an **alternative** Google (Gmail + Google Tasks)
backend for Iris. Instead of the current Path A OAuth refresh-token flow (three
secrets on-device, a `formRequest` token exchange, base64url MIME work, and the
7-day Testing-mode refresh-token expiry), the user deploys a tiny **Google Apps
Script web app** in their own account. The script runs *as them*, holds the
Google authorization server-side, and exposes a handful of small JSON endpoints
that Iris calls with a **shared secret** using the same `jsonRequest(...)`
pattern it already uses for NVIDIA NIM and Tavily.

> Scope note: this is a design document. It does **not** modify
> `shortcuts/iris.cherri`. It specifies the script, the manifest, the deploy
> steps, and an integration plan so the change can be made deliberately later.
> It complements `docs/google-integrations-research.md` (which called this
> "Option 2") by fully specifying it.

> Sources: Google Apps Script documentation (Web Apps, Content service,
> deployment, authorization, advanced services, manifest, GmailApp, quotas) on
> developers.google.com/apps-script, plus community reports of the POST/302
> redirect behavior and the Cherri language reference. Citations are inline.
> **Content from Google's documentation was rephrased/summarized for compliance;
> no large verbatim excerpts.**

---

## 1. Overview — what this is and why it sidesteps OAuth verification

An Apps Script **web app** is a script that contains a `doGet(e)` or `doPost(e)`
entry point and is deployed to a stable HTTPS URL; that function must return
either an `HtmlOutput` or a `TextOutput` (from the Content service). (Source:
Apps Script "Web Apps" requirements; "Content service" guide.) For a JSON API we
return a `TextOutput` whose MIME type is set to JSON.

The pivotal property is the deployment's **execution identity**. A web app can
be deployed to **"Execute as me"** (`executeAs: USER_DEPLOYING`) — meaning the
script always runs under the *deploying owner's* authority regardless of who
calls it — or **"Execute as user accessing"** (`USER_ACCESSING`). (Source: Apps
Script web-app manifest `Webapp.executeAs`; "Permissions" section of the Web
Apps guide.) `Session.getEffectiveUser()` confirms this: for a web app set to
"execute as me" it returns the *developer's* account, not the caller's. (Source:
`Session.getEffectiveUser()` reference.)

Because the script runs as the owner, **the owner authorizes the Gmail/Tasks
scopes once** — at first run or first deploy — and every subsequent call reuses
that stored authorization. Iris (the caller) never sees or handles a Google
token; it only presents a shared secret. (Source: "Authorization for Google
Services" — Apps Script requires user authorization to reach private Google data
from built-in or advanced services, granted through a consent prompt.)

### Why this avoids the test-user / verification friction

The current Path A flow builds a **standalone Google Cloud OAuth client** that
requests sensitive/restricted Gmail scopes. Such a client shows an "app isn't
verified" warning and stays in **Testing** mode (test users only; refresh tokens
can expire after ~7 days) until it passes Google's OAuth verification review.
(Source: Google Cloud "unverified app" help; Apps Script "Troubleshoot
authentication and authorization" — sensitive scopes trigger the unverified
warning and verification requirement.)

With an Apps Script web app used **for your own account**, the authorization is
just *you granting your own script access to your own data*. You still see the
same unverified-app screen the first time (because the script uses sensitive
scopes), but as the developer/owner you can proceed past it and grant consent —
there is no separate "publish to production + submit for review" gate for
personal use, and no test-user allow-list to maintain. (Source: Apps Script
authorization docs; the unverified-app warning applies but the owner can
authorize their own script. Note: Apps Script projects **within the same
Workspace domain** are explicitly exempt from client verification — Source:
"OAuth Client Verification.") The practical result: **no token on the device, no
7-day refresh-token expiry to babysit, no test-user dance.**

Trade-off in one line: you swap "manage OAuth tokens in the shortcut" for
"deploy and maintain a small script." The Iris-side tool contract (§6 of the
research doc, reproduced in §6 here) is identical either way.

---

## 2. How a request flows end to end (and the critical 302 gotcha)

```text
Iris (Shortcut)                     Google
  |  POST /exec  {secret, op, ...}      |
  |------------------------------------>|  doPost(e) runs AS the owner
  |                                     |    - check e.postData.contents.secret
  |                                     |    - switch on op -> GmailApp / Tasks
  |  302 Found  Location: script.google |    - return ContentService JSON
  |     usercontent.com/...             |
  |<------------------------------------|
  |  GET that redirect URL  (auto)      |
  |------------------------------------>|
  |  200 OK  {ok:true, ...JSON...}      |
  |<------------------------------------|
```

**Device gotcha (confirmed 2026-07-06): the 302 lands on a DIFFERENT domain.**
The POST hits `script.google.com` but the `Location` redirect (which iOS follows
automatically) is on `script.googleusercontent.com`. iOS Shortcuts grants
network access **per-domain**, so warming only `script.google.com` (e.g. a GET
health check) leaves the redirect domain un-granted; the first real POST then
fails silently under hands-free Siri and the model reports a bogus "I need
permission to view Google Tasks" answer. Fix (implemented in `iris.cherri`): the
setup primer does a real **POST** (`tasks_list`) so BOTH `script.google.com` and
`script.googleusercontent.com` are granted during the one manual setup run. Run
Iris manually, say "setup", and tap Allow on every network prompt (including the
`googleusercontent.com` one).

- **Entry point + reading the request.** `doPost(e)` receives the POST body as a
  string at `e.postData.contents` (parse it with `JSON.parse`), and any URL
  query parameters at `e.parameter`. `doGet(e)` similarly exposes `e.parameter`.
  (Source: Apps Script Web Apps / Content service; the standard community
  pattern is `var data = JSON.parse(e.postData.contents)`.)
- **Returning JSON.** Build the response with
  `ContentService.createTextOutput(JSON.stringify(obj)).setMimeType(ContentService.MimeType.JSON)`.
  The default TextOutput MIME type is plain text, so setting `MimeType.JSON` is
  required for a JSON response. (Source: `ContentService.createTextOutput`,
  `TextOutput.setMimeType` reference.)
- **THE 302 REDIRECT — the single most important gotcha.** A request to the
  `/exec` URL does **not** return your body directly. Google answers with an
  **HTTP 302 redirect** to a one-time `https://script.googleusercontent.com/...`
  URL that actually carries the response body; the HTTP client must **follow the
  redirect** to get the JSON. Google's own Content-service guidance says: if you
  use the Content service to return data to another application, ensure the
  client follows redirects (e.g. `curl -L`). (Source: Apps Script Content
  service guidance; corroborated by multiple community reports of POSTs to
  `/exec` returning a 302 to `script.googleusercontent.com` that must be
  followed with a GET.)
  - **Good news for Iris:** iOS's networking (URLSession, which backs Shortcuts'
    *Get Contents of URL*) **follows 3xx redirects automatically by default**
    unless the app explicitly intercepts them. (Source: Apple Developer Forums —
    a 302 is followed automatically when the app does not implement the
    redirect-handling delegate.) Cherri's `jsonRequest` / `downloadURL` compile
    to *Get Contents of URL*, so the redirect is followed transparently and Iris
    receives the final JSON. This is **not** something Iris has to code around —
    but it **must be verified on-device**, because a client that keeps the
    original POST method on the redirect (instead of switching to GET) can hit a
    `405 Method Not Allowed` from `script.googleusercontent.com`. (Source:
    community reports of the POST-preserved-on-redirect 405.) Standard-compliant
    clients switch to GET on a 302; browsers, curl `-L`, and URLSession do.
- **Content-type handling.** Send the body as JSON. Apps Script exposes the raw
  string at `e.postData.contents` regardless of the declared content type, and
  `JSON.parse` handles it. Some clients send `text/plain;charset=utf-8` to dodge
  a CORS preflight, but Iris is a native HTTP client (no CORS), so
  `Content-Type: application/json` is fine. (Source: community `doPost` JSON
  patterns; `e.postData.contents` is the raw request body.)

---

## 3. `appsscript.json` manifest

The manifest declares the runtime, the **advanced Tasks service** (so
`Tasks.Tasklists`/`Tasks.Tasks` are available and auto-authorized), the exact
**OAuth scopes** to request, and the **web-app deployment** identity/access.
(Sources: Apps Script "Advanced services" — advanced services are enabled via
`dependencies.enabledAdvancedServices`; "Manifest — dependencies"; "Setting
OAuth scopes in appsscript.json"; web-app manifest `access`/`executeAs`.)

```json
{
  "timeZone": "America/New_York",
  "runtimeVersion": "V8",
  "exceptionLogging": "STACKDRIVER",
  "dependencies": {
    "enabledAdvancedServices": [
      {
        "userSymbol": "Tasks",
        "serviceId": "tasks",
        "version": "v1"
      }
    ]
  },
  "oauthScopes": [
    "https://www.googleapis.com/auth/tasks",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/script.external_request"
  ],
  "webapp": {
    "executeAs": "USER_DEPLOYING",
    "access": "ANYONE_ANONYMOUS"
  }
}
```

Notes:

- `Tasks` advanced service: enabling it in the editor (Services → Google Tasks
  API) writes this `enabledAdvancedServices` entry for you. It provides
  `Tasks.Tasklists.list`, `Tasks.Tasks.list/insert/patch`. (Source: Apps Script
  "Advanced Google services" — Tasks service; the docs' own examples call
  `Tasks.Tasklists.list()` and `Tasks.Tasks.insert(task, taskListId)`.)
- **Scope minimization:** ship the narrowest set that matches the tools you wire
  in. Start with just `tasks` + `gmail.readonly`; add `gmail.compose` when
  drafts land and `gmail.send` only when send lands. Setting scopes explicitly in
  the manifest stops Apps Script from auto-adding a broader auto-detected set.
  (Source: "Scopes" — for published apps set the narrowest scopes in the
  manifest.) `GmailApp` methods like `search`/`sendEmail` may otherwise pull in
  `https://mail.google.com/`; pin the narrow scopes to avoid that.
- `script.external_request` is only needed if the script itself makes outbound
  `UrlFetchApp` calls; the version below does not, but it is harmless to include
  and useful if you extend it.
- `webapp.access`: `ANYONE_ANONYMOUS` = "Anyone" (callable without a Google
  login) — required so Iris can call it with no Google auth header. The **shared
  secret in the body is the only guard** (see §7). Use `ANYONE` (any logged-in
  Google user) only if every caller can attach a Google identity, which
  Shortcuts cannot do here. (Source: web-app manifest `access` enum:
  `MYSELF`/`DOMAIN`/`ANYONE`/`ANYONE_ANONYMOUS`.)

---

## 4. Full `Code.gs` (copy-pasteable)

Single `doPost` dispatcher. It checks the shared secret first, switches on `op`,
and always returns Iris's `tool/ok/count/result/records/error` envelope as JSON
so the Shortcut can parse it with `getDictionary()`/`getValue()` and hand it
straight to the planner. Gmail send/draft use `GmailApp`, which builds the MIME
for you — **no base64url work** on either side. (Source: `GmailApp.sendEmail`,
`GmailApp.createDraft`, `GmailApp.search` reference; Tasks advanced-service
examples.)

```javascript
/**
 * Iris Google proxy — Apps Script web app.
 * Deploy as: Execute as = Me, Who has access = Anyone.
 * All auth to Gmail/Tasks is the OWNER's, granted once at authorize time.
 * The only caller guard is SHARED_SECRET, checked on every request.
 */

// Paste a long random string here AND into Iris's @irisProxySecret box.
// Generate one with: openssl rand -hex 24
var SHARED_SECRET = 'REPLACE_WITH_A_LONG_RANDOM_SECRET';

// Cap fan-out so a single call stays inside Iris's ~25s Siri budget.
var MAX_RESULTS = 5;

// Shared dispatcher used by BOTH doGet (query params) and doPost (JSON body).
function handle(req) {
  // Constant-work secret check. Reject before doing anything else.
  if (!req.secret || req.secret !== SHARED_SECRET) {
    return _env('auth', false, 0, '', '', 'Invalid or missing secret.');
  }
  var op = String(req.op || '');
  if (op === 'tasks_list')     { return tasksList(req); }
  if (op === 'tasks_add')      { return tasksAdd(req); }
  if (op === 'tasks_complete') { return tasksComplete(req); }
  if (op === 'gmail_search')   { return gmailSearch(req); }
  if (op === 'gmail_read')     { return gmailRead(req); }
  if (op === 'draft_email')    { return draftEmail(req); }
  if (op === 'send_email')     { return sendEmail(req); }
  if (op === 'calendar_list')  { return calendarList(req); }
  if (op === 'calendar_add')   { return calendarAdd(req); }
  if (op === 'health')         { return _env('health', true, 0, 'Iris proxy is deployed.', '', ''); }
  return _env(op || 'unknown', false, 0, '', '', 'Unknown or unsupported op.');
}

// Iris calls the proxy with a GET (query params), NOT a POST. An Apps Script
// /exec POST 302-redirects to script.googleusercontent.com and iOS Shortcuts
// follows that redirect as a bodyless GET, dropping the JSON body — so the
// proxy would see no body and error. A GET keeps secret/op/args in the URL,
// which survives the redirect. e.parameter holds the query params as strings.
function doGet(e) {
  try {
    var req = (e && e.parameter) ? e.parameter : {};
    // Bare URL with no op = browser health check.
    if (!req.op && !req.secret) {
      return _json(_env('health', true, 0, 'Iris proxy is deployed.', '', ''));
    }
    return _json(handle(req));
  } catch (err) {
    return _json(_env('error', false, 0, '', '', 'Proxy error: ' + err.message));
  }
}

// doPost kept for backward compatibility / non-Shortcuts callers.
function doPost(e) {
  try {
    if (!e || !e.postData || !e.postData.contents) {
      return _json(_env('unknown', false, 0, '', '', 'Empty request body.'));
    }
    var req = JSON.parse(e.postData.contents);
    return _json(handle(req));
  } catch (err) {
    return _json(_env('error', false, 0, '', '', 'Proxy error: ' + err.message));
  }
}

/* ---------- Tasks ---------- */

function _defaultTaskListId() {
  var lists = Tasks.Tasklists.list();
  if (!lists.items || !lists.items.length) { return null; }
  return lists.items[0].id; // usually "My Tasks"
}

function tasksList(req) {
  var listId = _defaultTaskListId();
  if (!listId) { return _env('tasks_list', true, 0, 'No task lists found.', '', ''); }
  var res = Tasks.Tasks.list(listId, { showCompleted: false, maxResults: 100 });
  var items = res.items || [];
  var open = items.filter(function (t) { return t.status !== 'completed'; });
  var records = open.map(function (t) { return t.title; }).join('; ');
  var msg = 'You have ' + open.length + ' open task' + (open.length === 1 ? '' : 's') + '.';
  return _env('tasks_list', true, open.length, msg, records, '');
}

function tasksAdd(req) {
  var title = String(req.title || '').trim();
  if (!title) { return _env('tasks_add', false, 0, '', '', 'A task title is required.'); }
  var listId = _defaultTaskListId();
  if (!listId) { return _env('tasks_add', false, 0, '', '', 'No task list to add to.'); }
  var task = { title: title };
  if (req.notes) { task.notes = String(req.notes); }
  if (req.due)   { task.due = String(req.due); } // RFC-3339, e.g. 2026-01-10T00:00:00.000Z
  var created = Tasks.Tasks.insert(task, listId);
  return _env('tasks_add', true, 1, 'Added task: ' + created.title + '.',
    '- ' + created.title, '');
}

function tasksComplete(req) {
  var listId = _defaultTaskListId();
  if (!listId) { return _env('tasks_complete', false, 0, '', '', 'No task list found.'); }

  var taskId = req.task_id ? String(req.task_id) : '';
  if (!taskId && req.title) {
    var wanted = String(req.title).trim().toLowerCase();
    var all = (Tasks.Tasks.list(listId, { showCompleted: false, maxResults: 100 }).items) || [];
    var hits = all.filter(function (t) {
      return (t.title || '').trim().toLowerCase() === wanted;
    });
    if (hits.length === 0) {
      return _env('tasks_complete', false, 0, '', '', 'No open task titled "' + req.title + '".');
    }
    if (hits.length > 1) {
      return _env('tasks_complete', false, hits.length, '', '',
        'Multiple tasks match that title; ask the user which one.');
    }
    taskId = hits[0].id;
  }
  if (!taskId) {
    return _env('tasks_complete', false, 0, '', '', 'A task title or task_id is required.');
  }
  var patched = Tasks.Tasks.patch({ status: 'completed' }, listId, taskId);
  return _env('tasks_complete', true, 1, 'Completed: ' + patched.title + '.', '', '');
}

/* ---------- Calendar ---------- */

function calendarList(req) {
  var now = new Date();
  var end = new Date(now.getTime() + 7 * 24 * 60 * 60 * 1000); // next 7 days
  var events = CalendarApp.getDefaultCalendar().getEvents(now, end);
  var capped = events.slice(0, MAX_RESULTS * 2);
  var records = capped.map(function (ev) {
    return '- ' + ev.getTitle() + ' (' + ev.getStartTime() + ')';
  }).join('; ');
  var msg = 'You have ' + events.length + ' event' +
    (events.length === 1 ? '' : 's') + ' in the next 7 days.';
  return _env('calendar_list', true, events.length, msg, records, '');
}

function calendarAdd(req) {
  var title = String(req.title || '').trim();
  if (!title) { return _env('calendar_add', false, 0, '', '', 'An event title is required.'); }
  var startStr = String(req.start || req.date || '').trim();
  if (!startStr) { return _env('calendar_add', false, 0, '', '', 'A start date and time is required.'); }
  var start = new Date(startStr);
  if (isNaN(start.getTime())) {
    return _env('calendar_add', false, 0, '', '',
      'Could not understand the start time; use an absolute date and time.');
  }
  var end = null;
  var endStr = String(req.end || '').trim();
  if (endStr) { end = new Date(endStr); }
  if (!end || isNaN(end.getTime())) { end = new Date(start.getTime() + 60 * 60 * 1000); }
  var ev = CalendarApp.getDefaultCalendar().createEvent(title, start, end);
  return _env('calendar_add', true, 1, 'Added event: ' + title + '.',
    '- ' + title + ' (' + start + ')', '');
}

/* ---------- Gmail ---------- */

function gmailSearch(req) {
  var q = String(req.query || '').trim();
  if (!q) { return _env('gmail_search', false, 0, '', '', 'A search query is required.'); }
  var threads = GmailApp.search(q, 0, MAX_RESULTS);
  var rows = [];
  for (var i = 0; i < threads.length; i++) {
    var m = threads[i].getMessages()[0];
    // Voice-friendly: sender name + subject only (Iris reads records aloud).
    // Use gmail_read for the full body of a specific message.
    var from = m.getFrom().replace(/<[^>]*>/g, '').replace(/"/g, '').trim();
    rows.push('From ' + from + ': ' + m.getSubject());
  }
  var msg = 'You have ' + threads.length + ' matching email' + (threads.length === 1 ? '' : 's') + '.';
  return _env('gmail_search', true, threads.length, msg, rows.join('; '), '');
}

function gmailRead(req) {
  var q = String(req.query || '').trim();
  if (!q) { return _env('gmail_read', false, 0, '', '', 'A search query is required.'); }
  var threads = GmailApp.search(q, 0, 1);
  if (!threads.length) { return _env('gmail_read', true, 0, 'No matching email found.', '', ''); }
  var m = threads[0].getMessages()[0];
  var records = 'from=' + m.getFrom() + '; subject=' + m.getSubject() +
    '; date=' + m.getDate() + '; body=' + _clip(m.getPlainBody(), 1200);
  return _env('gmail_read', true, 1, 'Read the top matching email.', records, '');
}

function draftEmail(req) {
  var to = String(req.recipient || '').trim();
  var subject = String(req.title || req.subject || '').trim();
  var body = String(req.body || '');
  if (!to)   { return _env('draft_email', false, 0, '', '', 'A recipient is required.'); }
  if (!body) { return _env('draft_email', false, 0, '', '', 'A body is required.'); }
  GmailApp.createDraft(to, subject, body); // GmailApp builds the MIME
  return _env('draft_email', true, 1, 'Draft created for ' + to + '.', '', '');
}

function sendEmail(req) {
  // Hard confirmation gate: never send unless confirm === "yes".
  if (String(req.confirm || '').toLowerCase() !== 'yes') {
    return _env('send_email', false, 0, '', '',
      'Not sent. Confirmation required: re-call send_email with confirm=yes only after the user agrees.');
  }
  var to = String(req.recipient || '').trim();
  var subject = String(req.title || req.subject || '').trim();
  var body = String(req.body || '');
  if (!to)   { return _env('send_email', false, 0, '', '', 'A recipient is required.'); }
  if (!body) { return _env('send_email', false, 0, '', '', 'A body is required.'); }
  GmailApp.sendEmail(to, subject, body);
  return _env('send_email', true, 1, 'Sent email to ' + to + '.', '', '');
}

/* ---------- helpers ---------- */

function _env(tool, ok, count, result, records, error) {
  return { tool: tool, ok: ok, count: count, result: result, records: records, error: error };
}

function _json(obj) {
  return ContentService
    .createTextOutput(JSON.stringify(obj))
    .setMimeType(ContentService.MimeType.JSON);
}

function _clip(s, n) {
  s = (s || '').replace(/\s+/g, ' ').trim();
  return s.length > n ? s.slice(0, n) + '…' : s;
}
```

Design choices worth calling out:

- **The envelope is built server-side.** Iris receives `{tool, ok, count,
  result, records, error}` already shaped, so the Shortcut just reads six keys
  and forwards them to the planner — no MIME parsing, no base64url, no
  `formRequest`.
- **`send_email` is gated in two places** (planner protocol + a hard
  `confirm === 'yes'` check here), mirroring the safety posture in
  `docs/routes.md` and `agentic-loop.md`.
- **`gmail_read` returns a trimmed plain-text body** because `getPlainBody()`
  gives already-decoded text — the multipart/base64url decode problem from the
  REST path disappears when the work happens server-side. (Source:
  `GmailMessage.getPlainBody`.)
- **Fan-out is capped at `MAX_RESULTS` (5)** to protect the ~25s Siri budget.

---

## 5. Deploy + authorize — step by step

One-time, ~10 minutes, no Google Cloud Console project needed (Apps Script
manages its own project):

1. **Create the project.** Go to <https://script.google.com> → **New project**.
   Rename it (e.g. "Iris Google Proxy").
2. **Add the manifest.** Project Settings (gear) → check **"Show `appsscript.json`
   manifest file in editor."** Open `appsscript.json` and paste the manifest from
   §3.
3. **Enable the advanced Tasks service.** In the editor, click **Services (+)** in
   the left rail → find **Google Tasks API** → **Add**. This wires
   `Tasks.Tasklists`/`Tasks.Tasks` and writes the `enabledAdvancedServices`
   entry. (Source: Apps Script "Advanced Google services" — add the service in
   the editor to enable it.)
4. **Paste the code.** Replace the default `Code.gs` with §4. Set `SHARED_SECRET`
   to a long random string (`openssl rand -hex 24`). Save.
5. **Authorize once.** Select the `tasksList` function in the toolbar and click
   **Run**. Google prompts you to authorize the scopes the script uses
   (Tasks + Gmail). You will see the **"app isn't verified"** screen because the
   script uses sensitive Gmail scopes — as the owner, click **Advanced → Go to
   (project) (unsafe)** → **Allow**. This is you granting your own script access
   to your own account. (Source: "Authorization for Google Services";
   "Troubleshoot authentication and authorization" — sensitive scopes show the
   unverified warning, which the owner can proceed through.) After this, the
   authorization is stored and reused on every web-app call.
6. **Deploy as a web app.** **Deploy → New deployment → Type: Web app.** Set:
   - **Execute as: Me** (`USER_DEPLOYING`) — so the script runs with *your*
     authorization for every caller.
   - **Who has access: Anyone** (`ANYONE_ANONYMOUS`) — so Iris can call it with
     no Google login; the shared secret is the guard.
   (Source: Apps Script web-app deploy settings and manifest
   `executeAs`/`access`.)
7. **Copy the `/exec` URL.** The deployment shows a stable URL ending in
   **`/exec`** (form:
   `https://script.google.com/macros/s/<DEPLOYMENT_ID>/exec`). Paste it into
   Iris's `@irisProxyUrl` box. This is the production URL; the separate `/dev`
   URL is edit-access-only and always runs the latest unsaved code — do **not**
   use `/dev` for Iris. (Source: Apps Script "Test a web app deployment" — the
   `/dev` test URL is only reachable by editors and runs the most recent code.)
8. **Smoke test.** Open the `/exec` URL in a browser — `doGet` should return the
   health JSON. Then POST a real op (see §6) and confirm you get JSON back.

### Re-deploying / versioning (important)

Apps Script deployments are **versioned**. Editing the code does **not**
automatically change what the `/exec` URL serves: you must either create a **New
deployment** (which mints a *new* `/exec` URL) or, to keep the **same** `/exec`
URL, use **Manage deployments → edit the existing deployment → Version: New
version**. Choosing "New version" on the existing deployment updates the code
behind the **stable** `/exec` URL, so Iris's `@irisProxyUrl` never has to change.
(Source: Apps Script deployment model; `projects.deployments.update` takes a new
`versionNumber` for an existing deployment.) Rule of thumb: **edit the existing
deployment and bump the version** to avoid re-pasting a URL into the shortcut.

---

## 6. Iris integration plan

The Google backend collapses to **two config boxes** and **one `jsonRequest`
per op**. This replaces the three Google secret boxes and the `formRequest`
token exchange in the current build. (This section describes proposed edits; it
does not change `shortcuts/iris.cherri` now.)

### Config (two editable Text actions, S-GPT style)

```cherri
/* OPTIONAL Google integration via an Apps Script proxy. Deploy the web app in
   docs/apps-script-proxy.md, then paste its /exec URL and your shared secret
   below. Leave the placeholders to disable Google tools. */
@irisProxyUrlRaw = text("https://script.google.com/macros/s/REPLACE-ME/exec")
@irisProxySecretRaw = text("iris-proxy-secret-REPLACE-ME")
@irisProxyUrl = replaceText('\s+', "", "{@irisProxyUrlRaw}", false, true)
@irisProxySecret = replaceText('\s+', "", "{@irisProxySecretRaw}", false, true)
/* A configured proxy URL ends in /exec on script.google.com. */
@proxyMatches = matchText('^https://script\.google\.com/macros/s/[A-Za-z0-9_-]+/exec$', "{@irisProxyUrl}", false)
@proxyOk = count(@proxyMatches)
```

### One call per op (literal dict + `{@var}` interpolation)

`jsonRequest` requires **literal** body/header dicts (per the compiler rule at
the top of `iris.cherri`); interpolate the secret and args into the literal.
Because the proxy returns the envelope already shaped, parsing is a flat read of
six keys — no nested walk.

```cherri
/* tasks_list — no args */
@proxyResp = jsonRequest("{@irisProxyUrl}", "POST", {
  "secret": "{@irisProxySecret}",
  "op": "tasks_list"
}, {"Content-Type": "application/json", "Accept": "application/json"})
@proxyDict = getDictionary(@proxyResp)
@obsOk = getValue(@proxyDict, "ok")
@obsCount = getValue(@proxyDict, "count")
@obsResult = getValue(@proxyDict, "result")
@obsRecords = getValue(@proxyDict, "records")
@obsError = getValue(@proxyDict, "error")
@toolResult = "tool=tasks_list\nok={@obsOk}\ncount={@obsCount}\nresult={@obsResult}\nrecords={@obsRecords}\nerror={@obsError}"
```

```cherri
/* tasks_add — title, notes, due */
@proxyResp = jsonRequest("{@irisProxyUrl}", "POST", {
  "secret": "{@irisProxySecret}",
  "op": "tasks_add",
  "title": "{@title}",
  "notes": "{@notes}",
  "due": "{@date}"
}, {"Content-Type": "application/json", "Accept": "application/json"})
```

```cherri
/* send_email — recipient, title, body, confirm (hard-gated server-side too) */
@proxyResp = jsonRequest("{@irisProxyUrl}", "POST", {
  "secret": "{@irisProxySecret}",
  "op": "send_email",
  "recipient": "{@recipient}",
  "title": "{@title}",
  "body": "{@body}",
  "confirm": "{@confirm}"
}, {"Content-Type": "application/json", "Accept": "application/json"})
```

`gmail_search`, `gmail_read`, `draft_email`, and `tasks_complete` follow the
same shape — only the `op` and the interpolated args change. Every branch ends
by forwarding the six envelope keys back into `@loopContext`, identical to how
the existing NIM/Tavily routes normalize their observations.

### What stays the same

- **Tool registry + envelope unchanged.** The predeclared tool names, the planner
  protocol lines, and the `tool/ok/count/result/records/error` envelope are all
  reused verbatim (`agentic-loop.md`, `docs/routes.md`). The proxy is a backend
  swap, invisible to the planner.
- **One network host to prime.** The setup primer only needs to reach
  `script.google.com` (plus its redirect target `script.googleusercontent.com`,
  which is followed automatically). That is one host instead of
  `oauth2.googleapis.com` + `gmail.googleapis.com` + `tasks.googleapis.com`.
- **The 302 follow is automatic** (§2). Prime the host once, then confirm on a
  real device that a POST to `/exec` returns the final JSON (not a bare 302
  document). If a device ever surfaces the redirect as a `405`/HTML page,
  fall back to reading the `Location` header and issuing a follow-up
  `downloadURL(...)` GET — but URLSession's default follow behavior should make
  this unnecessary.

### What goes away

`formRequest` (the token exchange), base64url MIME building for
draft/send, per-request access-token refresh, and two of the three Google secret
boxes. The `@googleAccessToken` caching logic and the `formRequest` token block
in `iris.cherri` are all removed in the proxy variant.

---

## 7. Security

- **Public endpoint, secret-guarded.** "Who has access: Anyone" means the
  `/exec` URL is reachable by anyone who has it. The **shared secret in the
  request body is the only authentication**, so it must be long and random
  (`openssl rand -hex 24`), checked on *every* request before any Gmail/Tasks
  call (as in §4), and never logged. Treat the `/exec` URL itself as
  semi-secret. (Source: web-app `access` = `ANYONE_ANONYMOUS` makes the app
  callable without a Google login.)
- **The secret lives only in the shortcut**, in an editable Text action, exactly
  like the `nvapi-`/`tvly-` keys — never committed, never in a distributed
  artifact. Extend `scripts/validate-shortcut.py` to reject a real-looking proxy
  secret / `/exec` URL in compiled output, mirroring the existing key guards.
- **Blast radius is the owner's whole account for the granted scopes.** Because
  the script runs "as me," anyone with the URL + secret can act as you within
  the granted Gmail/Tasks scopes. Mitigate by (a) shipping the **narrowest
  scopes** first (`tasks` + `gmail.readonly`), (b) keeping `send_email` hard-gated
  on `confirm=yes`, and (c) rotating the secret if you suspect leakage.
- **Revocation is easy and total.** Delete the deployment (Deploy → Manage
  deployments → Archive/Delete) or rotate `SHARED_SECRET` (and Iris's box) to cut
  Iris off instantly; revoke the script's account access at
  myaccount.google.com/permissions to pull the Gmail/Tasks grant entirely. This
  is cleaner than the OAuth-client path, where you revoke a client and re-mint
  tokens.
- **Data exposure is unchanged from the REST path.** Email snippets/bodies and
  task titles still flow to the NIM planner; prefer snippets/metadata and cap
  fan-out. Transport is HTTPS end to end.
- **Never auto-send.** Same two-place gate as the current build.

---

## 8. Limits & latency (honest)

- **Script runtime:** ~6 minutes per execution for both consumer and Workspace
  accounts — far beyond anything a single Iris op needs. (Source: Apps Script
  "Quotas / Current limitations.")
- **Gmail daily send limit:** Apps Script send quotas are **per recipient per
  day** and differ by account type — roughly **100/day for consumer (free
  @gmail.com)** accounts and higher (on the order of 1,500/day) for Google
  Workspace accounts. Iris can read the live remaining allowance with
  `MailApp.getRemainingDailyQuota()`. Normal assistant use is nowhere near the
  cap. (Source: Apps Script `MailApp.getRemainingDailyQuota`; release note that
  the consumer daily recipient quota was reduced to 100; "Quotas for Google
  Services.")
- **Simultaneous executions & UrlFetch:** 30 simultaneous executions per user;
  UrlFetch has response/size/POST limits — irrelevant here since the script
  makes no outbound fetches. (Source: "Current limitations.")
- **Cold-start latency — the real budget risk.** A web-app call that hasn't run
  recently pays a **cold-start** penalty (the script container spins up),
  typically adding a couple of seconds; a warm call is faster. Combined with the
  **302 → follow-up GET** round-trip, a first-of-session `gmail_search` can be
  the slowest call Iris makes. Relate this to Iris's **~25s** *Get Contents of
  URL* / Siri budget: it fits comfortably for a single op, but leaves little
  headroom to fan out. Mitigations already baked into §4: **cap `MAX_RESULTS`
  at 5**, return **snippets not full multipart bodies**, do the Gmail/Tasks work
  server-side (one HTTP round-trip from the phone instead of token-refresh +
  list + per-id get), and keep responses small. The 6-minute *script* limit is
  not the constraint; the *phone-side* 25s timeout is. Validate real latency on
  a device.
- **Quotas reset ~24h after first use and are per user; they can change without
  notice.** (Source: "Quotas for Google Services.")

---

## 9. Comparison vs the current Path A refresh-token flow

| Dimension | Path A: OAuth refresh token (current) | Path B: Apps Script proxy (this doc) |
| --- | --- | --- |
| Secrets on device | 3 (client id, client secret, refresh token) | 2 (proxy `/exec` URL + shared secret) |
| Token handling in shortcut | Per-run refresh via `formRequest`; cache access token | None — Google auth is server-side |
| Non-JSON request quirk | `formRequest` (form-encoded) for token endpoint | None — plain `jsonRequest` everywhere |
| base64url MIME (draft/send) | Built on-device with `replaceText` | Built server-side by `GmailApp` |
| Full email body | Hard (walk multipart, base64url-decode on-device) | Easy (`getPlainBody()` server-side) |
| OAuth verification / test users | Testing mode: test-user list, unverified warning, **refresh token can expire in ~7 days** | Owner authorizes own script once; no test-user list, no 7-day token expiry |
| Network hosts to prime | 3 (`oauth2`, `gmail`, `tasks` googleapis) | 1 (`script.google.com`; redirect target auto-followed) |
| Round-trips per op | 2+ (token refresh, then API, then per-id gets for mail) | 1 from the phone (script fans out server-side) + 302 follow |
| Setup work | Cloud project, OAuth client, consent screen, mint refresh token | Create script, enable Tasks service, paste code, deploy |
| Ongoing maintenance | Re-mint token when it expires; keep client | Re-deploy (bump version) when editing the script |
| Revocation | Revoke client / re-mint tokens | Delete deployment or rotate secret; instant |
| New failure modes | Form-encoding, token expiry, base64url bugs | 302-follow behavior, cold-start latency, public-endpoint secret hygiene |
| Server to host | None (pure shortcut) | A script you deploy and maintain |

**When to prefer which.** Path B (this doc) is the cleaner, safer fit for a
hands-free assistant: least on-device secret sprawl, no token expiry to babysit,
and the fiddly MIME/base64url/form-encoding work moves server-side. Path A stays
the choice for a user who refuses to deploy any server and wants everything
inside the shortcut. The planner-facing tool contract is identical, so the two
are interchangeable backends.

---

## 10. Sources

- Apps Script Web Apps (requirements, `doGet`/`doPost`, permissions, deploy,
  `/dev` vs `/exec`): developers.google.com/apps-script/guides/web and
  /apps-script/execution_gadgets
- Content service (`createTextOutput`, `setMimeType`, follow-redirects guidance):
  developers.google.com/apps-script/guides/content and
  /apps-script/reference/content
- Web-app manifest (`executeAs`, `access` enum): developers.google.com/apps-script/manifest/web-app-api-executable
- Authorization / effective user / verification: developers.google.com/apps-script/scripts_google_accounts,
  /apps-script/reference/base/session (`getEffectiveUser`),
  /apps-script/guides/client-verification,
  /apps-script/api/troubleshoot-authentication-authorization
- Advanced services + Tasks (`Tasks.Tasklists.list`, `Tasks.Tasks.list/insert/patch`):
  developers.google.com/apps-script/guides/services/advanced,
  /apps-script/advanced/tasks, /apps-script/manifest/dependencies
- GmailApp (`search`, `getMessages`, `getInboxThreads`, `sendEmail`,
  `createDraft`, `getPlainBody`): developers.google.com/apps-script/reference/gmail
- Quotas & limits (runtime, email daily quota, `MailApp.getRemainingDailyQuota`):
  developers.google.com/apps-script/guides/services/quotas,
  /apps-script/reference/mail/mail-app
- POST → 302 → `script.googleusercontent.com` redirect behavior: Apps Script
  Content-service redirect guidance + community reports (Stack Overflow /
  Google Apps Script Community threads)
- iOS URLSession follows 302 automatically by default: Apple Developer Forums
- Cherri Web actions (`jsonRequest`, `downloadURL`): cherrilang.org
- Iris internal docs: `docs/google-integrations-research.md`, `docs/routes.md`,
  `agentic-loop.md`, `shortcuts/iris.cherri`

*Content from Google's documentation was rephrased/summarized for compliance
with licensing restrictions; no large verbatim excerpts are reproduced.*
