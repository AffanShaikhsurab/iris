---
type: Guide
title: Iris Setup Guide (start to finish)
summary: The complete, ordered walkthrough — import the shortcut, paste your key, choose where memory is stored (phone vs Google Sheet), deploy the optional Apps Script backend, run the one-time "Allow All" permission primer, and set up the proactive morning/evening/weekly automations.
status: active
tags:
  - setup
  - onboarding
  - automations
  - memory
  - permissions
related:
  - ../README.md
  - configuration.md
  - apps-script-proxy.md
  - google-setup.md
  - proactive-agents-plan.md
  - shortcut-runtime-flow.md
---

# Iris Setup Guide

This is the one page that takes you from "I have nothing" to "Iris answers me by
voice and briefs me every morning." Follow it top to bottom. Every step says
whether it is **required** or **optional**, and roughly how long it takes.

If you only want the fastest possible start, do **Steps 1, 2, and 5** — that's a
working voice assistant. Steps 3, 4, 6, and 7 add cloud memory, Google, web
search, and the proactive daily briefings.

| Step | What | Required? | Time |
|---|---|---|---|
| 1 | Get the shortcut onto your iPhone | ✅ Required | 2 min |
| 2 | Paste your NVIDIA API key | ✅ Required | 3 min |
| 3 | Choose where memory is stored (phone vs cloud) | Optional | 1 min |
| 4 | Deploy the Apps Script backend (cloud memory + Google) | Optional | 10 min |
| 5 | Run the "Allow All" permission primer | ✅ Required | 2 min |
| 6 | Add a web-search key (Tavily) | Optional | 2 min |
| 7 | Set up the proactive automations (morning/evening/weekly) | Optional | 5 min |

> Throughout, "a Text action at the top of the shortcut" means: open Iris in the
> Shortcuts editor and scroll to the very top. The first several actions are
> plain **Text** boxes holding your settings (the S-GPT pattern). You identify
> each box by the placeholder text it contains, listed below.

---

## Step 1 — Get the shortcut onto your iPhone (required, ~2 min)

Pick one:

- **Download a signed build (easiest).** Open the latest
  [GitHub Actions build](https://github.com/AffanShaikhsurab/iris/actions/workflows/build-shortcut.yml)
  on your iPhone, open **Artifacts**, and download `Iris.signed.shortcut` (or
  `Iris.shortcut`). Tap it to add it to the Shortcuts app.
- **Build from source (developers).** See the
  [signing guide](signing.md); in short:
  ```bash
  brew tap electrikmilk/cherri && brew install electrikmilk/cherri/cherri
  mkdir -p dist
  cherri shortcuts/iris.cherri --debug --output "dist/Iris.shortcut"
  ```

After importing, open Iris once in the Shortcuts editor so you can see the Text
boxes at the top. Do **not** run it yet — first paste your key (Step 2).

---

## Step 2 — Paste your NVIDIA API key (required, ~3 min)

Iris thinks with a hosted NVIDIA NIM model. The free key is what makes it work.

1. Sign in at **[build.nvidia.com](https://build.nvidia.com)** (a free NVIDIA
   Developer account; phone verification is required).
2. Create a personal key at
   **[build.nvidia.com/settings/api-keys](https://build.nvidia.com/settings/api-keys)**.
   It starts with `nvapi-` and is shown **only once** — copy it now.
3. In the Shortcuts editor, find the **first Text action**, which contains
   `nvapi-REPLACE-ME`. Replace that placeholder with your real key.
4. Leave the **second Text action** (the model id, pre-filled with a fast
   default) alone unless you want to experiment — model-choice rules are in
   [configuration.md](configuration.md#default-backend-nvidia-nim-buildnvidiacom).

That's it — Iris now works as a voice assistant once you finish Step 5. If you
run it before pasting a valid key, Iris just speaks setup instructions and stops
(it won't fail cryptically).

> **Tip for your own device:** to avoid re-pasting keys every time you rebuild,
> put them in a gitignored `.env.local` and the build injects them for you. See
> [configuration.md → Local personal builds](configuration.md#local-personal-builds-bake-in-your-keys-with-envlocal).
> Never distribute a build made this way.

---

## Step 3 — Choose where your memory is stored (optional, ~1 min)

This is the toggle for **local vs cloud**. Iris can remember things you tell it
("remember I prefer morning workouts"), and you decide where those notes live.
The choice is a single Text action at the top of the shortcut.

Find the Text action whose value is exactly **`hybrid`** (it sits just below the
proxy URL/secret boxes). Set it to one of three values:

| Value | Where memory lives | Choose this if… |
|---|---|---|
| `hybrid` *(default)* | **Both** — writes to your phone first, then mirrors to a Google Sheet | You want the safest option: fast local reads, plus a cloud copy that survives even if iCloud Drive fills up. |
| `local` | **Phone only** — files under `Shortcuts/IrisOKF/` in iCloud Drive | You want everything on-device, no Google, fully offline. |
| `sheets` | **Cloud only** — a Google Sheet via the Apps Script backend | You want memory readable/editable from a computer or another agent, immune to iCloud quota. |

Notes:

- Any unrecognized value falls back to `hybrid` automatically.
- `sheets` and the Sheet half of `hybrid` only do anything once you complete
  **Step 4** (the Apps Script backend). Until then, `hybrid` behaves like
  `local`, and `sheets` simply has nowhere to write — Iris still runs and
  answers normally (memory is always fail-open, never a hard error).
- The choice is set **once in the editor**, on purpose. Iris deliberately does
  **not** ask you every run, because a spoken "local or cloud?" question would
  break the hands-free path (see
  [shortcut-runtime-flow.md](shortcut-runtime-flow.md)).

If you only want a phone-local assistant, set this to `local` and skip Step 4.

---

## Step 4 — Deploy the Apps Script backend (optional, ~10 min)

This one script unlocks **two** things at once: cloud memory (the `sheets`/
`hybrid` backends from Step 3) **and** Google Tasks, Gmail, and Calendar — all
without putting any Google password or token on your phone. The script runs in
**your own** Google account and holds the authorization server-side; Iris just
calls it with a shared secret.

### 4a. Create and paste the script

1. Go to **[script.google.com](https://script.google.com)** and click
   **New project**.
2. Delete the starter code, then open this repo's `apps-script/Code.gs`, copy its
   full contents, and paste them in. (The same code, with a line-by-line
   explanation, is in [apps-script-proxy.md](apps-script-proxy.md).)
3. Near the top of the script, replace `REPLACE_WITH_A_LONG_RANDOM_SECRET` with a
   long random string. Generate one however you like, e.g. on a Mac/Linux
   terminal:
   ```bash
   openssl rand -hex 24
   ```
   Keep this string — you'll paste it into Iris in a moment.
4. Open the manifest (`appsscript.json`) and confirm the OAuth scopes and the
   web-app settings match [apps-script-proxy.md §3](apps-script-proxy.md). The
   key settings are **Execute as: Me** and **Who has access: Anyone**.

### 4b. Deploy it as a web app

1. Click **Deploy → New deployment → Web app**.
2. Set **Execute as: Me** and **Who has access: Anyone**, then **Deploy**.
3. The first time, Google asks you to **authorize** the script's access to your
   own Tasks/Gmail/Calendar/Sheets. You'll see an **"unverified app"** warning —
   this is expected because it's your own personal script. Click
   **Advanced → Go to \<project name\> (unsafe)** and grant access. (Full
   explanation and screenshots-in-words: [google-setup.md](google-setup.md).)
4. Copy the **Web app URL**. It looks like
   `https://script.google.com/macros/s/AKfy.../exec`.

### 4c. Paste the two values into Iris

Back in the Shortcuts editor, at the top of Iris:

- Find the Text action containing `https://script.google.com/macros/s/REPLACE-ME/exec`
  and replace it with your real **Web app URL**.
- Find the Text action containing `iris-proxy-secret-REPLACE-ME` and replace it
  with the **exact same secret** you put in the script.

Leaving either placeholder disables all cloud features — Iris falls back to
phone-local memory and skips Google tools, with no error.

> Why a Google Sheet? It becomes a durable, user-owned record of your memory that
> you can open on any device, and it's immune to iCloud running out of space.
> Under `hybrid`, Iris writes to the phone first and mirrors to the Sheet, so you
> get both speed and durability. See
> [apps-script-proxy.md](apps-script-proxy.md) for the full design.

---

## Step 5 — Run the "Allow All" permission primer (required, ~2 min)

**This is the step people skip and then wonder why Siri says "something went
wrong."** Do not skip it.

iOS requires **per-action, first-use consent** for Calendar, Reminders, Notes,
Location, Weather, Files, and **each network host** Iris talks to. There is **no
bulk "allow everything" switch** anywhere in iOS — Apple designed consent to be
per-action and per-shortcut for privacy. A hands-free Siri run **cannot tap
"Allow"** on these popups, so if a permission hasn't been granted yet, the run
just aborts.

The fix is to trigger every popup **once**, in a single manual run, while your
phone is unlocked. Iris has a built-in primer for exactly this:

1. Open the **Shortcuts app** and run **Iris manually** (tap it) on an
   **unlocked** phone. Do **not** use Siri for this run.
2. When it asks what to do, say or type **`setup`** (or "grant permissions").
3. Tap **Always Allow** — not "Allow Once" — on **every** dialog that appears:
   network hosts (NVIDIA, and if configured Tavily / `script.google.com` /
   `script.googleusercontent.com`), plus Calendar, Reminders, Location, Weather,
   Files, and Notes.
4. For background/locked use, also set **Settings → Privacy & Security →
   Location Services → Shortcuts → Always**.

After this one pass, normal voice requests run without any popups.

**Important gotchas:**

- Grants are **per-shortcut** and **reset every time you re-import or rebuild**
  Iris. Re-run `setup` after any re-import — otherwise the first Siri run fails.
- The Google backend redirects across **two** domains
  (`script.google.com` → `script.googleusercontent.com`). The primer makes a real
  call so **both** get granted; tap Allow on both network prompts.
- Full details and troubleshooting:
  [configuration.md → Permissions primer](configuration.md#permissions-one-time-primer-for-popup-free-siri-runs).

Now try it hands-free: **"Hey Siri, Iris"** → ask anything.

---

## Step 5b — Import your memory from another assistant (optional, ~3 min)

If you already use ChatGPT, Claude, or Gemini, you can bring what it knows about
you into Iris in one shot. Do this **after** Step 5 (the key and network must be
granted), on an unlocked phone.

1. **Export from your current assistant.** Open ChatGPT / Claude / Gemini and
   paste this prompt:

   ```text
   I'm moving to another assistant and need to export what you know about me.
   List every memory and durable fact you have about me from our past
   conversations. Output everything in ONE code block so I can copy it. Format
   each line as:
   - [date if known] category: fact (verbatim where possible)
   Cover: how I like you to respond (tone, format, always/never rules); personal
   details (name, location, job, family, interests); projects, goals, and
   recurring topics; tools, languages, and frameworks I use; preferences and
   corrections I've made. Do not summarize or omit entries. After the code block,
   tell me if that is everything or if more remains.
   ```

   Copy the code block it produces.

2. **Import into Iris.** Run Iris (manually, unlocked) and, instead of a
   question, say or type **`import`**. When it asks, **paste** the exported text
   and send.

3. Iris compresses it into your **profile** and pulls out your **behavior
   preferences**, storing both in memory. From then on, that profile rides along
   on every conversation (the standing `user_profile` context), so Iris knows
   your name, location, goals, and how you like to be answered — without being
   told again.

Notes: the paste is treated as your data, capped at ~30,000 characters, and
processed by two quick model calls while you wait. If you use the `sheets` or
`hybrid` memory backend, redeploy `Code.gs` first (the summary cap was raised).
Nothing is imported until you run `import`; you can re-run it anytime to refresh.

## Step 6 — Add live web search (optional, ~2 min)

By default Iris answers from the model's own knowledge, so it can't know current
things (scores, news, prices, "latest…"). A free Tavily key adds real internet
search:

1. Sign up at **[app.tavily.com](https://app.tavily.com)** (free tier: 1,000
   searches/month, no credit card) and copy the key (starts with `tvly-`).
2. In the editor, find the Text action containing `tvly-REPLACE-ME` (the third
   Text box) and paste your key over it.
3. Re-run the `setup` primer once so the `api.tavily.com` network prompt is
   granted.

Leaving the placeholder simply keeps web search off; everything else still works.

---

## Step 7 — Set up the proactive automations (optional, ~5 min)

This is what turns Iris from "answers when asked" into a proactive assistant that
briefs you every morning, reflects with you at night, and reviews your week on
Sundays.

### Why you have to create these by hand (once)

**iOS does not let any app or shortcut create an automation for you.** There is
no "create automation" action, and a shared shortcut cannot install one on import
— Apple requires you to create personal automations yourself in the Shortcuts
app. So Iris cannot set this up automatically. The good news: you only do it
**once**, and after that Iris fully controls what each run does. See
[proactive-agents-plan.md](proactive-agents-plan.md) for the architecture and the
hard iOS constraints behind this.

### How the automations work

Iris reads a **mode** from the input the automation passes it:

- empty input → normal interactive `chat` (the voice loop) — this is the default.
- `morning` → an unattended **compute → save → notify** run: it gathers your
  memory, calendar, reminders, and tasks, makes one model call for a short
  briefing, saves it, and delivers it as a **notification** (a scheduled run
  usually fires while the phone is locked, where spoken prompts can't complete —
  so it notifies instead of talking).
- `evening`, `weekly` → planned; currently they fall back to `chat`.

When the phone is locked, the run notifies you; **tap the notification** to unlock
and let Iris speak the already-prepared briefing.

### Create the morning automation

1. **Prime permissions first** (Step 5) — automations only work reliably after
   the one-time manual `setup` run.
2. Open **Shortcuts → Automation tab → `+` (New Automation) → Create Personal
   Automation**.
3. Choose **Time of Day**, set it to e.g. **7:30 AM**, **Daily**, then **Next**.
4. Choose **Run Shortcut**, select **Iris**, and pass **`morning`** as its input
   (tap the shortcut's input field / "Shortcut Input" and type `morning`; if
   passing input proves fiddly on your iOS version, make a tiny wrapper shortcut
   that just runs Iris with the text `morning`).
5. Turn **Ask Before Running OFF** (so it runs immediately, unattended).
   Optionally turn **Notify When Run** off so only Iris's own notification shows.
6. Repeat later for **9:30 PM `evening`** and **Sunday 6:00 PM `weekly`** once
   those modes ship.

### How to run / test them

- **Right now, manually:** run Iris from the Shortcuts app and, instead of a
  question, pass the input `morning` (easiest via a wrapper shortcut, or by
  temporarily editing the automation and tapping ▶). You should get a
  notification with today's briefing and a saved `log` entry — with **zero**
  spoken prompts.
- **On schedule:** just wait for the time. Timing is **best-effort** — iOS may
  fire it a few minutes late, and it may skip if the phone is off/in Low Power
  Mode. Treat the briefing as "around 7:30," not to-the-second.

### Shaping the routine by voice

You don't edit the automation to change what the morning briefing contains — you
just tell Iris. Ask it to "remember my morning routine should include the
weather and my first meeting," and it stores that in memory; the `morning` run
reads it. Change it anytime by talking to Iris.

---

## Quick troubleshooting

| Symptom | Most likely cause | Fix |
|---|---|---|
| Siri says "something went wrong" but manual runs work | A permission popup a locked Siri run can't tap | Re-run `setup` (Step 5), tap **Always Allow** on everything |
| Every request fails right after a rebuild | Grants + pasted key reset on re-import | Re-paste the key and re-run `setup` |
| "The request timed out" | Model too slow for the ~25s iOS budget | Use a small, fast model id (see [configuration.md](configuration.md)) |
| Cloud memory / Google tools do nothing | Proxy URL or secret still a placeholder, or backend set to `local` | Finish Step 4; check the memory toggle (Step 3) |
| Google tools fail only on the second domain | Only `script.google.com` was granted | Re-run `setup`, allow the `script.googleusercontent.com` prompt too |
| Morning automation never fires | "Ask Before Running" still on, or permissions not primed | Turn it off; run `setup` manually first |

---

## Where to go deeper

- Model choice, keys, invocation options: [configuration.md](configuration.md)
- The Google/cloud backend in full: [apps-script-proxy.md](apps-script-proxy.md)
  · on-device OAuth alternative: [google-setup.md](google-setup.md)
- Why proactive runs notify instead of speak, and the six-role plan:
  [proactive-agents-plan.md](proactive-agents-plan.md)
- The runtime rules behind all of this: [shortcut-runtime-flow.md](shortcut-runtime-flow.md)
