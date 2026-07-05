# Configuration

Iris intentionally does not commit secrets.

## Default Backend: NVIDIA NIM (build.nvidia.com)

The planner is a hosted NVIDIA NIM model called through the OpenAI-compatible
endpoint:

```text
POST https://integrate.api.nvidia.com/v1/chat/completions
Authorization: Bearer nvapi-...
```

Setup:

1. Sign in at `build.nvidia.com` (NVIDIA Developer account; phone verification
   required) and generate a personal key at `build.nvidia.com/settings/api-keys`.
   The key starts with `nvapi-` and is shown only once.
2. Import the shortcut, then open it in the Shortcuts editor and paste the key
   into the **first Text action** at the top, replacing `nvapi-REPLACE-ME`
   (the S-GPT pattern). The key lives only inside the local copy of the
   shortcut. If the shortcut is run before this is done, it speaks these setup
   instructions and stops instead of failing.
3. The second Text action holds the model id. The default is
   `meta/llama-3.1-8b-instruct` — chosen for latency, not capability. iOS
   gives `Get Contents of URL` a fixed, non-configurable timeout around 25
   seconds (and the Siri voice path is less patient), while NIM's free tier
   frequently serves popular big models slowly under load (30+ second
   responses, occasional 504s). A slow call surfaces as "the request timed
   out" in the app and "Something went wrong" from Siri.

Model selection rules (any `provider/model-name` id from the catalog works):

- Prefer small dense instruct models (`meta/llama-3.1-8b-instruct`,
  `mistralai/mistral-7b-instruct-v0.3`).
- Big models (`meta/llama-3.3-70b-instruct`, `openai/gpt-oss-120b`) give
  better answers but risk timeouts at peak hours — try them, and switch back
  if runs start dying.
- NEVER use reasoning/thinking models (`nvidia/nemotron-3-nano*`,
  `deepseek-ai/*`, anything `*-thinking`): NIM hangs on some of them unless a
  `chat_template_kwargs` field is sent, and thinking output breaks the
  flat-JSON route contract and the timing budget.
- Avoid just-launched models; free-tier capacity for them is overloaded for
  weeks after launch.

Do NOT move the key back into import questions (Cherri `#question`): since
iOS 18.5 the Shortcuts Setup step silently fails to save the entered value
into the bound action, leaving the key empty. This surfaced as "the key is
not taking" at import and Siri's generic "Something went wrong" at runtime.
Whitespace is stripped from both Text values at runtime, so a stray newline
from pasting cannot corrupt the Authorization header.

The free tier is rate-limited (about 40 requests per minute per key, shared
across models, subject to change) rather than credit-based. One Iris
conversation uses roughly 2-10 requests.

Do not commit a real API key to this repository. Do not upload generated or
signed artifacts after answering the setup questions with a real key; only
distribute artifacts built from source, which contain the `nvapi-REPLACE-ME`
placeholder. `scripts/validate-shortcut.py` fails the build if a real-looking
`nvapi-` key appears in the compiled shortcut.

## Local personal builds: bake in your keys with `.env.local`

Re-importing a rebuilt shortcut normally means re-pasting every key in the
editor. To skip that on your own device, put your credentials in a gitignored
`.env.local` file and the build injects them automatically:

1. `cp .env.local.example .env.local`
2. Fill in your real values: `NIM_API_KEY` (required); `TAVILY_KEY`,
   `NIM_MODEL_ID`, and the three `GOOGLE_*` values (optional).
3. Build normally (`bash scripts/build-shortcuts.sh`). When `.env.local` exists,
   the build replaces the `...-REPLACE-ME` placeholders with your values in a
   **throwaway copy** of the source (the committed source is never touched), so
   `dist/Iris.shortcut` works the moment you import it — no manual pasting.

Safety model:

- `.env.local` is gitignored and is **never committed or pushed**; the committed
  source always keeps the placeholders.
- CI and any build **without** `.env.local` produce the safe placeholder
  version. **Never distribute a build made with a filled `.env.local`**, and do
  not run `scripts/validate-shortcut.py` on one — it fails by design when real
  keys are present.
- You still run the one-time **"setup"** primer after each import to grant iOS
  permissions; only the credentials are auto-filled, not the permission grants.

## Optional: Web Search (Tavily)

By default the agent answers from the model's own knowledge, so it cannot know
anything current ("when is the next FIFA match", live news, prices). Add a free
Tavily key to give it real internet search:

1. Sign up at `app.tavily.com` (free tier: 1,000 searches/month, **no credit
   card**) and copy the key (starts with `tvly-`).
2. Open the shortcut in the editor and paste it into the **third Text action**
   at the top (`tvly-REPLACE-ME`). Leaving the placeholder simply disables web
   search — everything else still works.
3. Re-run the "setup" primer once so the `api.tavily.com` network prompt is
   granted (see below).

How it works: the `web_search` tool sends the query to
`POST https://api.tavily.com/search` with `include_answer` and reads Tavily's
synthesized `answer` plus source snippets, feeds them back to the agent, and the
agent speaks a summarized answer. The model is instructed to use `web_search`
for real-time/recent/changing facts and `answer_search` (its own knowledge) for
timeless questions. The Tavily key, like the NVIDIA key, lives only in your
local copy; `scripts/validate-shortcut.py` fails the build if a real `tvly-` key
is ever baked into a compiled artifact.

## Optional: Google Tasks (Path A, refresh-token flow)

Iris can read and add Google Tasks. Auth uses a one-time OAuth consent that
yields a **refresh token**, which Iris exchanges for a short-lived access token
at the start of each Google request (no interactive browser step at run time).

One-time setup:

1. In **Google Cloud Console**: create a project, enable the **Google Tasks
   API**, and configure the **OAuth consent screen** (User type *External*; add
   your own Google account under **Test users**).
2. Create an **OAuth client ID** (type *Web application* or *Desktop app*) and
   note the **client ID** and **client secret** (the secret starts with
   `GOCSPX-`).
3. Get a **refresh token**: open the **OAuth 2.0 Playground**
   (developers.google.com/oauthplayground), click the gear → **Use your own
   OAuth credentials** and paste your client ID/secret, authorize the scope
   `https://www.googleapis.com/auth/tasks`, then exchange the code for tokens.
   Copy the **refresh token** (starts with `1//`).
4. Open Iris in the Shortcuts editor and paste the **client ID**, **client
   secret**, and **refresh token** into the three Google Text actions near the
   top, replacing the `google-...-REPLACE-ME` placeholders. Leaving the
   placeholders keeps the Google tools disabled.
5. Re-run the **"setup"** primer once so the `oauth2.googleapis.com` and
   `tasks.googleapis.com` network prompts are granted (the primer also does a
   token refresh, so a successful setup confirms your credentials work).

Notes and honest limits:

- The token exchange uses Cherri's `formRequest` (Google's token endpoint
  requires a form-encoded body, not JSON); every Tasks call uses the normal
  request path with a `Bearer` access token.
- While your OAuth app is unverified (**Testing** mode), Google may **expire the
  refresh token after 7 days**, after which you re-mint it via the Playground.
  Publishing/verifying the app removes this.
- The keys live only in your local copy of the shortcut;
  `scripts/validate-shortcut.py` fails the build if a real Google client secret
  (`GOCSPX-`) or refresh token (`1//...`) is ever baked into a compiled artifact.
- Gmail (read/draft/send) is not implemented yet; see
  `docs/google-integrations-research.md`.

## Permissions: one-time primer for popup-free Siri runs

iOS requires per-action, first-use consent for Calendar, Reminders, Notes,
Location, Weather, Files, and each network host. There is **no way to bulk-grant
or pre-authorize** these — not in code, not in Settings, not via MDM (verified
2026-07-05). Consent is stored per-shortcut and cannot be scripted away; Apple
designed it that way for privacy. A mid-run popup also drops Siri out of voice
mode and shows the result on screen instead of speaking it.

The workaround is to trigger every prompt once, in a single manual run, so
hands-free Siri never prompts again. The shortcut has a built-in **permission
primer** for this:

1. After importing the final build and pasting your key, run Iris
   **manually from the Shortcuts app, on an unlocked phone** (not via Siri).
2. Answer the first prompt with **"setup"** (or "grant permissions").
3. Tap **Always Allow** — not "Allow Once" — on every dialog that appears
   (network, Calendar, Reminders, Location, Weather, Files, Notes). The primer
   runs one read action per permission (plus a tiny network ping and one
   labeled setup note you can delete).
4. For background/locked Siri use, also set Settings → Privacy & Security →
   Location Services → Shortcuts to **Always**.

After this, normal voice requests run without popups. Notes:

- Grants are **per-shortcut** and reset when the shortcut is re-imported or
  recompiled — re-run "setup" after every re-import. A separate helper shortcut
  cannot grant permissions to this one.
- Calendar/Reminders read actions request **Full Access**, which also covers the
  create/add actions, so one setup pass is enough.
- The primer only runs when the request begins with a setup phrase; normal
  requests like "set a reminder" or "access my calendar" are unaffected.

## Invocation: launching without saying "Iris"

By default you launch the shortcut by its name: "Hey Siri, Iris". The
shortcut's name IS its Siri trigger phrase (set by `#define name Iris`
in `shortcuts/iris.cherri`).

iOS does not let a third-party shortcut replace Siri's built-in handling of
general "Hey Siri ..." questions, and the "Hey Siri" wake word itself cannot be
changed (verified 2026-07). So there is no way to make a bare "Hey Siri, what's
the weather" route into Iris. What you CAN do:

1. **Vocal Shortcuts (best "just say a word" option).** Settings →
   Accessibility → Vocal Shortcuts → Set Up Vocal Shortcuts, pick Iris,
   and record a short custom phrase (for example "assistant" or "computer").
   After that the phrase runs Iris on-device **without saying "Siri" at
   all**. It is the closest thing to a custom wake word.
2. **Rename to a shorter phrase.** Change `#define name Iris` to a short,
   natural word (for example `#define name Assistant`), rebuild, and re-import.
   Then "Hey Siri, Assistant" is all you say. Keep it distinct from Apple's own
   command words so Siri does not intercept it. Re-run the "setup" primer after
   re-import, because permission grants reset on re-import.
3. **No-voice launch.** Bind the shortcut to the **Action Button** (iPhone 15
   Pro and later), a **Back Tap** (Settings → Accessibility → Touch → Back Tap),
   or add it to the **Home Screen / Lock Screen / Today View / Control Center**.
   Any of these opens Iris directly, and it immediately asks "What
   should Iris do?" so you just talk.

Whichever entry point you use, the conversation model is unchanged: Iris
asks, listens, answers, and offers "Anything else?" until you say a stop word.

## Previous Backend: ChatGPT app App Intent

Earlier versions used the ChatGPT iOS app's own `com.openai.chat.AskIntent`
action (no API key, but requires the app installed and signed in, and each
call risks the app's helper/session failures). The device-validated action
shape is preserved in `shortcut-syntax-reference.md` and `notes.md` if a
no-key backend is wanted again.

## Speech Output

All spoken output goes through Siri itself: `Ask for Input` prompts are read
aloud by Siri and answered by dictation, and answers are embedded in the next
prompt. There is no custom text-to-speech in the shortcut. See
`docs/shortcut-runtime-flow.md` for the conversation model.

## OKF Memory Folder

Iris expects an optional OKF-style memory folder at:

```text
Shortcuts/IrisOKF/
```

The first files to create on the phone are:

```text
index.md
profile.md
preferences.md
log.md
```

The Shortcut currently reads fixed paths such as
`/Shortcuts/IrisOKF/index.md` and appends confirmed memory writes to
`/Shortcuts/IrisOKF/log.md`. These paths are experimental until validated
on a target iPhone because iOS Files/iCloud Drive path behavior can vary.

See `docs/okf-knowledge-base.md` for the concept format and privacy rules.

## Privacy Warning

The default ChatGPT path sends the user's typed, dictated, shared, or
clipboard-provided request to ChatGPT. API mode would send the same kind of data
to the configured model provider. Users should not route sensitive personal data
unless they understand the provider's privacy terms.

If OKF memory is enabled, any retrieved memory snippet included in the prompt is
also sent to ChatGPT. Keep memory entries concise and avoid sensitive data.
