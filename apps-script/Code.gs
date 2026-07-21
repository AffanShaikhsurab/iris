/**
 * Iris Google proxy — Apps Script web app (COMPLETE: Tasks + Gmail + Calendar + Memory).
 *
 * Deploy as: Execute as = Me, Who has access = Anyone.
 * All auth to Gmail/Tasks/Calendar/Sheets/Drive is the OWNER's, granted once at
 * authorize time. The only caller guard is SHARED_SECRET, checked on every request.
 *
 * SETUP / REDEPLOY:
 *   1. Paste this whole file over your existing Code.gs.
 *   2. Enable the Google Tasks advanced service (Services + > Google Tasks API)
 *      if it is not already on (needed for the Tasks.* calls).
 *   3. Run any function once (e.g. doGet) so Apps Script prompts to AUTHORIZE
 *      the NEW permissions — the memory ops use SpreadsheetApp + DriveApp, which
 *      your previous deployment never granted. Approve them.
 *   4. Deploy > Manage deployments > (edit) > New version > Deploy.
 *      Saving alone is NOT enough; /exec serves the last *deployed* version.
 *
 * The memory ops store an append-only, OKF-shaped Google Sheet named "Iris
 * Memory" (found-or-created by name in your Drive), so there is no spreadsheet
 * id to paste anywhere. topic is constrained to an allowlist so the model can
 * never address arbitrary storage.
 */

// Paste a long random string here AND into Iris's proxy-secret Text box
// (keep them identical). Generate one with: openssl rand -hex 24
// NOTE: this committed copy uses a PLACEHOLDER on purpose - never commit your
// real secret. Replace it in your deployed Apps Script project only.
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
  if (op === 'memory_append')  { return memoryAppend(req); }
  if (op === 'memory_read')    { return memoryRead(req); }
  if (op === 'memory_list')    { return memoryList(req); }
  if (op === 'memory_status')  { return memoryStatus(req); }
  if (op === 'memory_summary') { return memorySummary(req); }
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

/* ---------- Memory (append-only Google Sheet, OKF-shaped rows) ----------
 * One append-only sheet, resolved by NAME so no spreadsheet id is pasted
 * anywhere. Rows mirror the OKF frontmatter
 * [timestamp, topic, type, title, tags, body] so one row reconstructs to one
 * OKF concept. topic is constrained to an allowlist. The model-facing protocol
 * stays memory_append(topic, body) only — type/title/tags/timestamp are derived
 * server-side, never sent by the planner. Uses SpreadsheetApp + DriveApp, which
 * require re-authorization the first time after adding this section. */
var MEMORY_SHEET_NAME = 'Iris Memory';
var MEMORY_TAB = 'memory';
var MEMORY_HEADER = ['timestamp', 'topic', 'type', 'title', 'tags', 'body'];
var MEMORY_TOPICS = ['index', 'profile', 'preferences', 'log', 'notes', 'journal'];
var MAX_MEMORY_ROWS = 20;

// Derive the OKF `type` from the topic (server-side; the model never sends it).
function _okfType(topic) {
  var map = {
    log: 'Memory Entry', preferences: 'User Preference', profile: 'Profile',
    notes: 'Note', journal: 'Journal', index: 'Index'
  };
  return map[topic] || 'Memory Entry';
}

// Reconstruct an OKF concept block from a row [ts, topic, type, title, tags, body].
function _rowToOkf(r) {
  return '---\ntype: ' + (r[2] || 'Memory Entry') +
         '\ntitle: ' + (r[3] || '') +
         '\ntags: ' + (r[4] || '') +
         '\ntimestamp: ' + (r[0] || '') +
         '\n---\n' + (r[5] || '');
}

// Find-or-create the spreadsheet and tab. Self-healing: the first write creates
// everything, so no manual setup run is needed. Reruns are cheap.
function _memorySheet() {
  var files = DriveApp.getFilesByName(MEMORY_SHEET_NAME);
  var ss = files.hasNext() ? SpreadsheetApp.open(files.next())
                           : SpreadsheetApp.create(MEMORY_SHEET_NAME);
  var sh = ss.getSheetByName(MEMORY_TAB);
  if (!sh) {
    sh = ss.insertSheet(MEMORY_TAB);
    sh.appendRow(MEMORY_HEADER);
  }
  return sh;
}

// Constrain topic to the allowlist; anything else falls back to 'log'.
function _normTopic(t) {
  t = String(t || 'log').trim().toLowerCase();
  return MEMORY_TOPICS.indexOf(t) === -1 ? 'log' : t;
}

function memoryAppend(req) {
  var topic = _normTopic(req.topic);
  var body = String(req.body || '').trim();
  if (!body) { return _env('memory_append', false, 0, '', '', 'A body to remember is required.'); }
  var type = _okfType(topic);
  // title = optional first line of body (<=80 chars); tags optional (empty).
  // Timestamp is stamped server-side, never by the model.
  var title = body.split('\n')[0].slice(0, 80);
  var tags = String(req.tags || '');
  _memorySheet().appendRow([new Date().toISOString(), topic, type, title, tags, body]);
  return _env('memory_append', true, 1, 'Saved to ' + topic + ' memory.', body, '');
}

// Return the most recent MAX_MEMORY_ROWS rows for a topic (case-insensitive),
// as full row arrays, oldest-first within the window.
function _readTopicRows(topic) {
  var sh = _memorySheet();
  var values = sh.getDataRange().getValues(); // includes the header row
  var rows = [];
  for (var i = 1; i < values.length; i++) {
    if (String(values[i][1]).toLowerCase() === topic) { rows.push(values[i]); }
  }
  return rows.slice(-MAX_MEMORY_ROWS);
}

// Bodies only (column index 5), oldest-first within the window (so a join reads
// newest-last for voice).
function _readTopic(topic) {
  return _readTopicRows(topic).map(function (r) { return r[5]; });
}

function memoryRead(req) {
  var topic = _normTopic(req.topic);
  var rows = _readTopicRows(topic);
  if (!rows.length) {
    return _env('memory_read', false, 0, '', '', 'No memory found for that topic yet.');
  }
  // Default: bodies joined newest-last for voice (records stays plain text so
  // the Shortcut never getDictionary()s it). Pass format=okf to return the
  // reconstructed OKF concept blocks instead (e.g. for a non-voice client).
  var records;
  if (String(req.format || '') === 'okf') {
    records = rows.map(_rowToOkf).join('\n\n');
  } else {
    records = rows.map(function (r) { return r[5]; }).join('; ');
  }
  return _env('memory_read', true, rows.length, 'Read ' + topic + ' memory.', records, '');
}

function memoryList(req) {
  var sh = _memorySheet();
  var values = sh.getDataRange().getValues();
  var rows = [];
  for (var i = Math.max(1, values.length - MAX_MEMORY_ROWS); i < values.length; i++) {
    rows.push(values[i][1] + ': ' + values[i][5]); // topic: body
  }
  if (!rows.length) { return _env('memory_list', true, 0, 'Memory is empty.', '', ''); }
  return _env('memory_list', true, rows.length, 'Read memory.', rows.join('; '), '');
}

function memoryStatus(req) {
  var sh = _memorySheet();
  var n = Math.max(0, sh.getLastRow() - 1); // minus the header row
  if (n === 0) { return _env('memory_status', true, 0, 'Memory is set up but empty.', '', ''); }
  return _env('memory_status', true, n, 'Memory is available.', '', '');
}

// Standing profile summary injected on EVERY planner turn: profile + preferences
// + the last few index lines. Capped at 10000 chars (~2.1-2.4k tokens) - the
// profile is the most important standing context for a personal assistant, and
// this is a small slice of Iris's 60k working budget (bench-verified ~2s even at
// 60k). The client applies the same 10000-char cap.
function memorySummary(req) {
  var parts = []
    .concat(_readTopic('profile'))
    .concat(_readTopic('preferences'))
    .concat(_readTopic('index').slice(-3));
  var text = parts.join('; ');
  if (text.length > 10000) { text = text.slice(0, 10000); }
  return _env('memory_summary', true, parts.length, 'Memory summary.', text, '');
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
