# Notion via the Apps Script proxy hub

Design/research note for adding **Notion** to Iris as a **non-Google** service
routed through the same **Google Apps Script web-app proxy** described in
[`docs/apps-script-proxy.md`](../apps-script-proxy.md). Iris keeps calling one
web app (`/exec`) with `{secret, op, ...}`; the script adds a handful of
`notion_*` ops that reach the Notion REST API with `UrlFetchApp` and a token
held in **Script Properties**. The planner just gains new tool names; the
`{tool, ok, count, result, records, error}` envelope is unchanged.

> Scope note: this is a design document. It does **not** modify
> `shortcuts/iris.cherri`. It specifies the Notion auth model, the endpoints,
> copy-pasteable Apps Script `op` branches, the Iris-side tool contract, setup
> steps, and limits, so the change can be made deliberately later.

> Sources: the official Notion API documentation on developers.notion.com
> (authorization, authentication, versioning, search, create-a-page,
> append-block-children, query-a-database/data-source, retrieve-a-page,
> request limits) plus the Notion Help Center connection guide, retrieved via
> Context7 and web search. Citations are inline.
> **Content from Notion's docs was rephrased/summarized for compliance; no
> large verbatim excerpts.**

---

## 0. Why route Notion through the existing proxy

The proxy hub was built for Google (Gmail + Tasks) using "execute as me" auth,
but its dispatcher is service-agnostic: `doPost(e)` checks a shared secret,
switches on `op`, and returns the Iris envelope
(`docs/apps-script-proxy.md` §2, §4). For a **non-Google** service like Notion
there is no Google OAuth involved at all — the script simply makes an outbound
HTTPS call with `UrlFetchApp.fetch(...)`, carrying a **Notion integration
token** that lives in `PropertiesService.getScriptProperties()`, never in the
shortcut and never in the source. This mirrors how the proxy already reaches
NVIDIA NIM/Tavily-style services, and reuses:

- the same `/exec` URL and shared-secret guard (one host to prime),
- the same 302-follow behavior (handled automatically by URLSession),
- the same six-key envelope the planner already parses.

The only new manifest requirement is the **`script.external_request`** OAuth
scope, because the script now makes outbound fetches (it is already listed as
harmless-to-include in `docs/apps-script-proxy.md` §3).

---

## 1. Auth for personal use (internal integration)

### 1.1 Create an internal integration/connection

For a single user driving their own workspace, create an **internal
integration** (Notion now calls these **internal connections**). An internal
connection is scoped to **one workspace** and usable only by members of that
workspace — ideal for personal automations. You create it from
**notion.so/my-integrations** (or Settings → Connections → Develop or manage
integrations), give it a name, pick the workspace, and choose its capabilities
(read content, insert content, update content, and optionally read user
info). (Source: Notion "Internal connections" guide; "Authorization" guide;
Notion Help Center "Create integrations with the Notion API".)

Internal connections are the recommended starting point because you skip the
whole OAuth dance — you just need a token to start calling the API. (Source:
Notion "Developer quickstart".)

### 1.2 The integration token

Creating the internal connection yields an **integration token** (a bot bearer
token) used in the `Authorization: Bearer <token>` header. (Source: Notion
"Authentication" — the API accepts bearer tokens provided when you create an
internal connection, create a personal access token, or complete OAuth for a
public connection.)

**Token prefix — note the change.** Historically these tokens began with
**`secret_`**. Notion later changed the prefix to **`ntn_`** (rolled out around
late 2024), so newly minted internal tokens now look like
`ntn_XXXXXXXXXXXX...`. Treat **both** prefixes as valid when validating input —
older tokens starting with `secret_` still work, new ones start with `ntn_`.
(Source: Notion authentication docs; corroborated by the `notion-sdk-py`
`secret_` → `ntn_` change report.) The token is a **long-lived secret**: anyone
holding it can act as the bot within the granted capabilities and shared pages.

### 1.3 The required `Notion-Version` header

Every request **must** include a `Notion-Version` header naming a dated API
version; sending it is required, and unversioned requests are rejected.
(Source: Notion "Versioning" reference; changelog "Notion-Version header will
be required".)

- The long-standing **stable** value is **`2022-06-28`**. It is the safest pin
  for the classic endpoints used here (including
  `POST /v1/databases/{database_id}/query`). (Source: Notion versioning
  reference; widely used across SDKs and community integrations.)
- Newer dated versions exist. The **`2025-09-03`** upgrade introduced
  **multi-source databases ("data sources")**, which splits the old
  database-query endpoint into a data-source model (see §2.4). Docs also surface
  even newer dates (e.g. `2026-03-11`). Pinning `2022-06-28` keeps the simple
  single-endpoint database query working; adopt a newer version deliberately
  only when you need data sources. (Source: Notion versioning reference;
  Notion data-APIs guides referencing the 2025-09-03 data-sources model.)

**Recommendation for Iris:** pin `Notion-Version: 2022-06-28` in the proxy, as a
single `NOTION_VERSION` constant, so behavior is stable and testable.

### 1.4 The crucial sharing step (integrations see nothing by default)

A fresh integration can see **nothing**. Each page or database must be
explicitly **shared/connected with the integration** before the API can read or
write it. In the Notion UI you open the target page/database → **•••** menu →
**Connections** (formerly "Add connections") → pick your integration. Sharing a
parent page cascades access to its child pages. (Source: Notion "Authorization"
guide — an integration must be added to a page/database before it can access it;
Help Center connection guide.)

Practical consequence: if `POST /v1/search` returns an empty `results` array or
a `GET /v1/pages/{id}` returns a 404, the usual cause is **"not shared with the
integration,"** not a bad token. Document this front-and-center in setup (§5).

---

## 2. Key endpoints (request/response shape)

Base URL: `https://api.notion.com`. All requests send
`Authorization: Bearer <token>`, `Notion-Version: <version>`, and
`Content-Type: application/json`. (Source: Notion authentication/versioning
references; endpoint references below.)

### 2.1 `POST /v1/search` — find pages & databases by title

Body (all optional): `query` (title substring), `filter` (`{property:"object",
value:"page"|"database"}` — under 2025-09-03+, `value` may be `data_source`),
`sort` (`{timestamp:"last_edited_time", direction:"ascending"|"descending"}`),
plus `page_size` and `start_cursor` for pagination. The response is a list
object: `{object:"list", results:[...], next_cursor, has_more}`, where each
result is a page or database/data-source object. Search only returns items the
integration has been shared into. (Source: Notion "Search" reference /
`post-search`.)

```json
{ "query": "meeting notes",
  "filter": { "property": "object", "value": "page" },
  "sort": { "direction": "descending", "timestamp": "last_edited_time" } }
```

### 2.2 `POST /v1/pages` — create a page

Required: a **`parent`**, which is one of `{"page_id": "..."}` (child of a
page), `{"database_id": "..."}` (a row/entry in a database; under 2025-09-03 you
may target a `data_source_id`), or a workspace parent. Plus **`properties`**: for
a page under another page, the only property is `title`; for a database entry,
`properties` must match the database schema (the title property plus any others).
Optional **`children`**: an array of block objects to fill the page body in the
same call. The response is the created **page object** (with its `id` and `url`).
(Source: Notion "Create a page" reference / `post-page`.)

```json
{ "parent": { "page_id": "494c87d0-72c4-4cf6-960f-55f8427f7692" },
  "properties": {
    "title": { "title": [ { "type": "text", "text": { "content": "A note from Iris" } } ] }
  },
  "children": [
    { "object": "block", "type": "paragraph",
      "paragraph": { "rich_text": [ { "type": "text", "text": { "content": "Body text." } } ] } }
  ] }
```

### 2.3 `PATCH /v1/blocks/{block_id}/children` — append content

Appends new child blocks to a parent block/page (the page id doubles as its root
block id). Body: **`children`** (array of block objects), optional **`position`**
(`{"type":"end"}` default, `"start"`, or `after_block`). Limits: up to **100**
block children per request and up to **two** levels of nesting per call; existing
blocks cannot be moved. The response is a list of the newly created first-level
blocks. (Source: Notion "Append block children" reference / `patch-block-children`.)

```json
{ "children": [
    { "type": "paragraph",
      "paragraph": { "rich_text": [ { "text": { "content": "Appended line." } } ] } }
  ] }
```

### 2.4 `POST /v1/databases/{database_id}/query` — query a database

Returns pages (rows) in a database, filtered/sorted. Body (all optional):
**`filter`** (property-typed conditions, e.g. `{"property":"Status",
"select":{"equals":"In Progress"}}`), **`sorts`** (array of
`{property, direction}` or timestamp sorts), plus `page_size`/`start_cursor`.
The response is a list of page objects with their `properties`. (Source: Notion
"Query a database" reference.)

```json
{ "filter": { "property": "Status", "select": { "equals": "In Progress" } },
  "sorts": [ { "property": "Due", "direction": "ascending" } ],
  "page_size": 25 }
```

> **Data-sources note (version 2025-09-03+).** When databases gained multiple
> data sources, the query moved to
> `POST /v1/data_sources/{data_source_id}/query` and `POST /v1/pages` gained a
> `data_source_id` parent option. On the pinned `2022-06-28` version the classic
> `/v1/databases/{database_id}/query` endpoint above still works, so Iris starts
> there. If you later pin `2025-09-03+`, resolve the database's `data_source_id`
> first and switch the op to the data-source endpoint. (Source: Notion
> "Query a data source" reference; data-APIs guides on the 2025-09-03 upgrade.)

### 2.5 `GET /v1/pages/{id}` — retrieve a page

Returns the page object: `id`, `url`, `parent`, `properties`, `icon`, `cover`,
`created_time`/`last_edited_time`, and archive/trash flags. Note this returns the
page's **property values, not its body blocks** — to read body content you list
the page's block children (`GET /v1/blocks/{id}/children`), which this design
leaves out of the initial op set to keep responses small. (Source: Notion
"Retrieve a page" reference.)

### 2.6 Block & rich-text structure (high level)

- A **block object** has `object:"block"`, a `type` string (e.g. `paragraph`,
  `heading_1`, `bulleted_list_item`, `to_do`), and a **type-named key** holding
  that block's data (e.g. a `paragraph` block has a `paragraph` object).
- Text-bearing blocks carry a **`rich_text`** array. Each rich-text item is
  usually `{"type":"text","text":{"content":"...", "link":null},
  "annotations":{bold,italic,strikethrough,underline,code,color},
  "plain_text":"...","href":null}`. For **writes**, the minimum is
  `{"text":{"content":"..."}}` — Notion fills in the rest. (Source: Notion
  "Rich text"/page-property-values and block references.)

Keep Iris's writes to **plain paragraph blocks** built from a single
`rich_text` text item; that is enough for notes/append and avoids dumping large
schemas into the planner.

---
## 3. Apps Script sketch (`op` branches)

These branches slot into the existing `doPost` dispatcher in
`docs/apps-script-proxy.md` §4. The Notion token is read from **Script
Properties**, never hard-coded, and the `Notion-Version` header is a single
constant. Each branch returns the standard Iris envelope via the existing
`_env(...)`/`_json(...)` helpers.

### 3.1 Manifest prerequisite

Add the outbound-fetch scope (already suggested in the proxy manifest §3):

```json
{ "oauthScopes": [ "https://www.googleapis.com/auth/script.external_request" ] }
```

### 3.2 Setting the token in Script Properties

The token lives in `PropertiesService.getScriptProperties()` — not in the code,
not in the shortcut. Two ways to set it:

**A. Editor UI:** Apps Script editor → **Project Settings** (gear) → **Script
Properties** → **Add script property** → key `NOTION_TOKEN`, value your
`ntn_...` token → **Save**. Optionally add `NOTION_VERSION` (`2022-06-28`) and a
convenience `NOTION_DEFAULT_PARENT` (a page id for new notes).

**B. One-off code:** paste, run once, then **delete the token literal**:

```javascript
function setNotionToken() {
  PropertiesService.getScriptProperties().setProperties({
    NOTION_TOKEN: 'ntn_REPLACE_WITH_YOUR_TOKEN',
    NOTION_VERSION: '2022-06-28'
  });
}
```

### 3.3 Dispatcher additions

Add these lines beside the existing `op ===` checks in `doPost`:

```javascript
if (op === 'notion_search')      { return _json(notionSearch(req)); }
if (op === 'notion_create_page') { return _json(notionCreatePage(req)); }
if (op === 'notion_append')      { return _json(notionAppend(req)); }
if (op === 'notion_query_db')    { return _json(notionQueryDb(req)); }
```

### 3.4 Shared Notion helpers

```javascript
var NOTION_API = 'https://api.notion.com/v1';

function _notionProps() {
  var p = PropertiesService.getScriptProperties();
  return {
    token: p.getProperty('NOTION_TOKEN') || '',
    version: p.getProperty('NOTION_VERSION') || '2022-06-28',
    defaultParent: p.getProperty('NOTION_DEFAULT_PARENT') || ''
  };
}

/**
 * Call the Notion REST API. Returns { ok, status, body } where body is parsed
 * JSON. muteHttpExceptions lets us shape 4xx/5xx into the Iris envelope
 * instead of throwing.
 */
function _notionFetch(method, path, payload) {
  var cfg = _notionProps();
  if (!cfg.token) { return { ok: false, status: 0, body: { message: 'NOTION_TOKEN not set in Script Properties.' } }; }
  var options = {
    method: method,
    contentType: 'application/json',
    muteHttpExceptions: true,
    headers: {
      'Authorization': 'Bearer ' + cfg.token,
      'Notion-Version': cfg.version
    }
  };
  if (payload) { options.payload = JSON.stringify(payload); }
  var resp = UrlFetchApp.fetch(NOTION_API + path, options);
  var status = resp.getResponseCode();
  var body = {};
  try { body = JSON.parse(resp.getContentText() || '{}'); } catch (e) { body = {}; }
  return { ok: status >= 200 && status < 300, status: status, body: body };
}

// Pull a plain-text title out of a Notion page object (schema-agnostic).
function _notionPageTitle(page) {
  var props = (page && page.properties) || {};
  for (var k in props) {
    var pr = props[k];
    if (pr && pr.type === 'title' && pr.title && pr.title.length) {
      return pr.title.map(function (t) { return t.plain_text || (t.text && t.text.content) || ''; }).join('');
    }
  }
  return '(untitled)';
}

function _notionErr(tool, r) {
  var msg = (r.body && (r.body.message || r.body.code)) || ('HTTP ' + r.status);
  return _env(tool, false, 0, '', '', 'Notion error: ' + msg);
}

// Minimal paragraph block from a plain string.
function _notionParagraph(text) {
  return { object: 'block', type: 'paragraph',
    paragraph: { rich_text: [ { type: 'text', text: { content: String(text || '') } } ] } };
}
```

### 3.5 `notion_search`

```javascript
function notionSearch(req) {
  var q = String(req.query || '').trim();
  var payload = {
    query: q,
    sort: { direction: 'descending', timestamp: 'last_edited_time' },
    page_size: 5
  };
  var r = _notionFetch('post', '/search', payload);
  if (!r.ok) { return _notionErr('notion_search', r); }
  var results = (r.body.results) || [];
  var rows = results.slice(0, 5).map(function (item) {
    if (item.object === 'page') {
      return 'page: ' + _notionPageTitle(item) + ' | id=' + item.id;
    }
    var t = (item.title && item.title.length) ? item.title.map(function (x) { return x.plain_text || ''; }).join('') : '(db)';
    return 'database: ' + t + ' | id=' + item.id;
  });
  var msg = 'Found ' + results.length + ' Notion match' + (results.length === 1 ? '' : 'es') + '.';
  return _env('notion_search', true, results.length, msg, rows.join(' || '), '');
}
```

### 3.6 `notion_create_page`

```javascript
function notionCreatePage(req) {
  var title = String(req.title || '').trim();
  if (!title) { return _env('notion_create_page', false, 0, '', '', 'A page title is required.'); }

  var cfg = _notionProps();
  // Prefer an explicit parent arg; else fall back to NOTION_DEFAULT_PARENT.
  var parentPageId = String(req.target || req.parent_id || cfg.defaultParent || '').trim();
  var parentDbId = String(req.database_id || '').trim();

  var payload = { properties: {} };
  if (parentDbId) {
    payload.parent = { database_id: parentDbId };
    // Database entries need the DB's title property; "Name" is the common default.
    payload.properties[String(req.title_property || 'Name')] = {
      title: [ { type: 'text', text: { content: title } } ]
    };
  } else if (parentPageId) {
    payload.parent = { page_id: parentPageId };
    payload.properties.title = { title: [ { type: 'text', text: { content: title } } ] };
  } else {
    return _env('notion_create_page', false, 0, '', '',
      'No parent. Provide target (page id) or database_id, or set NOTION_DEFAULT_PARENT.');
  }

  if (req.body) { payload.children = [ _notionParagraph(req.body) ]; }

  var r = _notionFetch('post', '/pages', payload);
  if (!r.ok) { return _notionErr('notion_create_page', r); }
  var url = r.body.url || '';
  return _env('notion_create_page', true, 1, 'Created Notion page: ' + title + '.',
    'id=' + r.body.id + '; url=' + url, '');
}
```

### 3.7 `notion_append`

```javascript
function notionAppend(req) {
  var blockId = String(req.target || req.page_id || req.block_id || '').trim();
  var body = String(req.body || '').trim();
  if (!blockId) { return _env('notion_append', false, 0, '', '', 'A target page/block id is required.'); }
  if (!body)    { return _env('notion_append', false, 0, '', '', 'Body text to append is required.'); }

  var payload = { children: [ _notionParagraph(body) ] };
  var r = _notionFetch('patch', '/blocks/' + encodeURIComponent(blockId) + '/children', payload);
  if (!r.ok) { return _notionErr('notion_append', r); }
  var added = (r.body.results && r.body.results.length) || 0;
  return _env('notion_append', true, added, 'Appended content to the Notion page.', '', '');
}
```

### 3.8 `notion_query_db`

```javascript
function notionQueryDb(req) {
  var dbId = String(req.database_id || req.target || '').trim();
  if (!dbId) { return _env('notion_query_db', false, 0, '', '', 'A database_id is required.'); }

  var payload = { page_size: 5 };
  // Optional simple select filter passed as flat strings from the planner.
  if (req.filter_property && req.filter_value) {
    payload.filter = { property: String(req.filter_property),
      select: { equals: String(req.filter_value) } };
  }
  var r = _notionFetch('post', '/databases/' + encodeURIComponent(dbId) + '/query', payload);
  if (!r.ok) { return _notionErr('notion_query_db', r); }
  var results = (r.body.results) || [];
  var rows = results.slice(0, 5).map(function (pg) {
    return _notionPageTitle(pg) + ' | id=' + pg.id;
  });
  var msg = 'Found ' + results.length + ' row' + (results.length === 1 ? '' : 's') + ' in the database.';
  return _env('notion_query_db', true, results.length, msg, rows.join(' || '), '');
}
```

Design choices:

- **Token from Script Properties**, one `Notion-Version` constant, and
  `muteHttpExceptions:true` so 4xx/5xx become clean `ok=false` envelopes instead
  of thrown errors.
- **Fan-out capped at 5** and only titles/ids returned (no full block bodies) to
  protect Iris's ~25s phone-side budget (`docs/apps-script-proxy.md` §8).
- **Schema-agnostic title extraction** so `notion_search`/`notion_query_db`
  don't need to know each database's property names.
- **Flat args only** (`title`, `body`, `target`, `database_id`,
  `filter_property`, `filter_value`) so the planner can supply them as top-level
  strings.

---

## 4. Iris side (predeclared tools)

The proxy returns the envelope already shaped, so Iris parsing is a flat
six-key read — identical to the Google ops. These four tools are **predeclared**
in the registry (`agentic-loop.md`, `docs/routes.md`) and gated behind a
configured proxy URL + secret, exactly like the Google proxy tools.

### 4.1 Proposed tool names + one-line protocol descriptions

| Tool | One-line protocol description | Args (flat strings) |
| --- | --- | --- |
| `notion_search` | Search the user's Notion for pages/databases by title and return the top matches with ids. | `query` |
| `notion_create_page` | Create a Notion page under a parent page (`target`) or as a database entry (`database_id`); `body` fills the first paragraph. | `title`, `body`, `target`, `database_id` |
| `notion_append` | Append a paragraph of text to an existing Notion page/block (`target` = page id, from `notion_search`). | `target`, `body` |
| `notion_query_db` | Query a Notion database by id and return matching rows; optional simple select filter. | `database_id`, `filter_property`, `filter_value` |

Protocol notes to add to the planner prompt: ids come from a prior
`notion_search`; if the user names a page but no id is known, call
`notion_search` first, then use the returned id as `target`. Creating/appending
are content writes — keep them behind the normal review posture, and never
invent ids.

### 4.2 Example `jsonRequest` calls to the proxy

Same literal-dict + `{@var}` interpolation rule as the existing proxy ops
(`docs/apps-script-proxy.md` §6).

```cherri
/* notion_search — query only */
@proxyResp = jsonRequest("{@irisProxyUrl}", "POST", {
  "secret": "{@irisProxySecret}",
  "op": "notion_search",
  "query": "{@query}"
}, {"Content-Type": "application/json", "Accept": "application/json"})
@proxyDict = getDictionary(@proxyResp)
@obsOk = getValue(@proxyDict, "ok")
@obsCount = getValue(@proxyDict, "count")
@obsResult = getValue(@proxyDict, "result")
@obsRecords = getValue(@proxyDict, "records")
@obsError = getValue(@proxyDict, "error")
@toolResult = "tool=notion_search\nok={@obsOk}\ncount={@obsCount}\nresult={@obsResult}\nrecords={@obsRecords}\nerror={@obsError}"
```

```cherri
/* notion_create_page — title, body, parent page id in target */
@proxyResp = jsonRequest("{@irisProxyUrl}", "POST", {
  "secret": "{@irisProxySecret}",
  "op": "notion_create_page",
  "title": "{@title}",
  "body": "{@body}",
  "target": "{@target}"
}, {"Content-Type": "application/json", "Accept": "application/json"})
```

```cherri
/* notion_append — target page id + body */
@proxyResp = jsonRequest("{@irisProxyUrl}", "POST", {
  "secret": "{@irisProxySecret}",
  "op": "notion_append",
  "target": "{@target}",
  "body": "{@body}"
}, {"Content-Type": "application/json", "Accept": "application/json"})
```

```cherri
/* notion_query_db — database id + optional flat select filter */
@proxyResp = jsonRequest("{@irisProxyUrl}", "POST", {
  "secret": "{@irisProxySecret}",
  "op": "notion_query_db",
  "database_id": "{@database_id}",
  "filter_property": "{@filter_property}",
  "filter_value": "{@filter_value}"
}, {"Content-Type": "application/json", "Accept": "application/json"})
```

`notion_search`/`notion_query_db` reuse the read-observation pattern;
`notion_create_page`/`notion_append` forward the same six envelope keys back into
`@loopContext`, identical to the NIM/Tavily/Google routes. No new parse logic,
no new host — the proxy `/exec` URL is the only endpoint Iris touches.

---

## 5. Setup steps (for the user)

1. **Create the internal connection.** Go to
   <https://www.notion.so/my-integrations> → **New integration** →
   name it (e.g. "Iris"), pick your workspace, set type **Internal**, and enable
   **Read**, **Insert**, and **Update** content capabilities. (Source: Notion
   "Internal connections"/"Authorization"; Help Center connection guide.)
2. **Copy the token.** Open the integration → copy its **Internal Integration
   Secret** (starts with `ntn_`, or `secret_` on older integrations). Keep it
   private. (Source: Notion authentication docs; Help Center.)
3. **Share pages/databases with the integration.** In Notion, open each page or
   database Iris should touch → **•••** → **Connections** → add your "Iris"
   integration. Sharing a parent cascades to its children. **Nothing is visible
   until you do this.** (Source: Notion "Authorization" — the integration must be
   added to a page/database first.)
4. **Paste the token into Script Properties.** In the Apps Script project →
   Project Settings → Script Properties → add `NOTION_TOKEN` = your token (and
   optionally `NOTION_VERSION` = `2022-06-28`, `NOTION_DEFAULT_PARENT` = a page
   id for new notes). Save. (§3.2.)
5. **Add the code + redeploy.** Paste the §3 branches/helpers into `Code.gs`,
   ensure `script.external_request` is in the manifest, then **Manage
   deployments → edit → New version** to keep the same `/exec` URL
   (`docs/apps-script-proxy.md` §5).
6. **Smoke test.** POST `{secret, op:"notion_search", query:"<a shared page
   title>"}` to `/exec` and confirm the envelope lists the page. An empty result
   almost always means step 3 was skipped.

---

## 6. Limits & security

- **Rate limit ~3 requests/second (average), per integration token.** Bursts are
  tolerated but sustained overage returns HTTP **429** with a `Retry-After`
  header; the proxy's `muteHttpExceptions` path surfaces this as an `ok=false`
  envelope. Iris's low, interactive call volume stays well under the cap.
  (Source: Notion "Request limits" reference.) Other size caps apply — e.g. up
  to **100 block children per append** and payload/property-size limits.
  (Source: Notion "Append block children" and request-limits references.)
- **Token in Script Properties only.** The `ntn_` token never appears in
  `shortcuts/iris.cherri`, in compiled artifacts, or in the request body — only
  the shared proxy secret travels from the phone. This is stronger than the
  Google "execute as me" model because the Notion token is fully server-side.
  Consider extending `scripts/validate-shortcut.py` to reject a real-looking
  `ntn_`/`secret_` token in compiled output, mirroring the existing key guards.
- **Least privilege via capabilities + sharing.** The blast radius is only the
  capabilities you enabled (read/insert/update) **and** only the pages/databases
  you explicitly shared. Share the minimum; do not connect the integration at
  the workspace root unless you mean to.
- **Writes are content-affecting.** `notion_create_page`/`notion_append` mutate
  the workspace. Keep them behind Iris's normal review posture and never let the
  planner invent target ids.
- **Revocation is immediate and total.** Delete or refresh the integration token
  at notion.so/my-integrations (old token dies instantly), and/or remove the
  integration's **Connection** from a page/database to cut off access to just
  that surface. Rotating the proxy shared secret additionally cuts Iris off from
  the whole hub. (Source: Notion authentication/connection docs.)
- **Transport & data exposure.** HTTPS end to end; the proxy returns only titles
  and ids by default (no full block bodies), limiting how much workspace content
  flows to the NIM planner.

---

## 7. Sources

- Authorization / internal connections / sharing requirement:
  developers.notion.com/guides/get-started/authorization,
  /guides/get-started/internal-connections,
  notion.com/help/create-integrations-with-the-notion-api
- Authentication (bearer token) & token prefix (`secret_` → `ntn_`):
  developers.notion.com/reference/authentication; notion-sdk-py prefix-change report
- Versioning (`Notion-Version` required; `2022-06-28` stable; 2025-09-03 data
  sources): developers.notion.com/reference/versioning,
  /changelog/unversioned-requests-no-longer-accepted, data-APIs guides
- Endpoints: /reference/post-search, /reference/post-page,
  /reference/patch-block-children, /reference/post-database-query
  (and /reference/query-a-data-source for 2025-09-03+), /reference/retrieve-a-page
- Rich text / block structure: /reference/rich-text, /reference/block,
  /reference/page-property-values
- Request limits (~3 req/s, 429/Retry-After, size caps):
  developers.notion.com/reference/request-limits
- Apps Script proxy hub (dispatcher, envelope, deploy, security, limits):
  `docs/apps-script-proxy.md`; Iris tool contract: `agentic-loop.md`,
  `docs/routes.md`

*Content from Notion's docs was rephrased/summarized for compliance with
licensing restrictions; no large verbatim excerpts are reproduced.*
