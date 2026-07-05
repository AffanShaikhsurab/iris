# Google Setup Guide (Tasks + Gmail)

This is a complete, beginner-friendly walkthrough for connecting **Google
Tasks** and **Gmail** to Iris. You will create a Google Cloud project, enable
the two APIs, set up OAuth consent, create a client ID, mint a **refresh
token**, and paste three values into the shortcut. No coding is required, and
the whole thing takes about 10–15 minutes.

Iris uses a **Path A refresh-token flow**: you do the interactive Google
consent **once, out of band**, which produces a long-lived refresh token. Iris
stores that token in an editable Text action inside your local copy of the
shortcut and quietly exchanges it for a short-lived access token at the start of
each Google request. There is no browser step at run time, which is what makes
hands-free Siri use possible.

> **Licensing note:** Content from Google's official documentation was
> rephrased and summarized here for licensing compliance; no large verbatim
> excerpts are reproduced. Follow the inline links for the authoritative
> wording.

Official references used throughout:

- Google Tasks API — https://developers.google.com/workspace/tasks
- Gmail API — https://developers.google.com/workspace/gmail
- Gmail API scopes — https://developers.google.com/workspace/gmail/api/auth/scopes
- Using OAuth 2.0 to access Google APIs — https://developers.google.com/identity/protocols/oauth2
- OAuth 2.0 Playground — https://developers.google.com/oauthplayground

---

## What you will end up with

Three values pasted into the three `google-...-REPLACE-ME` Text actions at the
top of Iris:

1. **Client ID** — looks like `1234567890-abc123.apps.googleusercontent.com`
2. **Client secret** — starts with `GOCSPX-`
3. **Refresh token** — starts with `1//`

Keep these private. They live only inside your local copy of the shortcut (see
[Security notes](#security-notes)).

---

## Step 1 — Create a Google Cloud project and enable the APIs

1. Go to the **Google Cloud Console** at https://console.cloud.google.com and
   sign in with the Google account whose Tasks/Gmail you want Iris to use.
2. In the top project picker, click **New Project**, give it a name (for
   example `Iris`), and create it. Make sure that new project is selected before
   continuing.
3. Enable the **Tasks API**: open **APIs & Services → Library**, search for
   *Google Tasks API*, open it, and click **Enable**.
   (Reference: Google Tasks API — https://developers.google.com/workspace/tasks.)
4. Enable the **Gmail API**: back in **Library**, search for *Gmail API*, open
   it, and click **Enable**.
   (Reference: Gmail API — https://developers.google.com/workspace/gmail.)

You only need to do this once per project; both APIs can live in the same
project.

---

## Step 2 — Configure the OAuth consent screen

Before Google will issue tokens, your project needs an OAuth consent screen.

1. Open **APIs & Services → OAuth consent screen**.
2. Choose **User type: External** (this is the correct choice for a personal
   `@gmail.com` account; *Internal* is only available to Google Workspace
   organizations).
3. Fill in the required fields: an app name (for example `Iris`), your email as
   the **user support email**, and a **developer contact email**. The other
   fields can be left at their defaults.
4. On the **Test users** step, click **Add users** and add **your own Google
   account** (the same account you enabled the APIs on). This is essential — an
   app in Testing will only issue tokens to accounts listed here.

### Testing vs Published, and the 7-day refresh-token expiry

- A new OAuth app starts in **Testing** mode. In Testing:
  - Only the accounts you added under **Test users** can authorize it.
  - The consent screen shows an **"unverified app"** warning (this is expected —
    see [Troubleshooting](#troubleshooting)).
  - **Refresh tokens can expire after 7 days.** When that happens, Iris's Google
    tools stop working until you re-mint the refresh token (repeat
    [Step 4](#step-4--get-a-refresh-token-via-the-oauth-20-playground)).
- **Publishing** the app (Publishing status → **Publish app**) removes the
  7-day expiry for the standard flow. However, the **restricted** Gmail scopes
  (like `gmail.readonly`) trigger Google's verification review before an app can
  be fully published to the public. For personal use, the simplest path is to
  **stay in Testing** and just re-mint the token when it expires. It is a
  one-minute repeat of Step 4.

  (Reference: Using OAuth 2.0 —
  https://developers.google.com/identity/protocols/oauth2.)

---

## Step 3 — Create an OAuth client ID

1. Open **APIs & Services → Credentials**.
2. Click **Create Credentials → OAuth client ID**.
3. For **Application type**, choose **Web application** (recommended, because it
   works cleanly with the OAuth Playground) or **Desktop app**. Either works for
   this flow.
4. If you chose **Web application**, add the OAuth Playground redirect URI so
   the Playground can complete the flow: under **Authorized redirect URIs**, add

   ```
   https://developers.google.com/oauthplayground
   ```

5. Click **Create**. A dialog shows your **Client ID** and **Client secret**.
   - The **Client ID** looks like
     `1234567890-abc123.apps.googleusercontent.com`.
   - The **Client secret** starts with `GOCSPX-`.
6. Copy both somewhere safe for the next step (you can always reopen the client
   under **Credentials** to see them again).

---

## Step 4 — Get a refresh token via the OAuth 2.0 Playground

The OAuth 2.0 Playground runs the interactive consent for you and hands back a
refresh token, using **your own** client ID and secret.

1. Open the **OAuth 2.0 Playground** at
   https://developers.google.com/oauthplayground.
2. Click the **gear icon** (⚙️, top right) to open **OAuth 2.0 configuration**.
   - Tick **Use your own OAuth credentials**.
   - Paste your **Client ID** and **Client secret** from Step 3.
   - Close the settings panel.
3. In the left panel (**Step 1 — Select & authorize APIs**), find the field where
   you can **input your own scopes** and enter the following four scopes,
   separated by spaces:

   ```
   https://www.googleapis.com/auth/tasks
   https://www.googleapis.com/auth/gmail.readonly
   https://www.googleapis.com/auth/gmail.compose
   https://www.googleapis.com/auth/gmail.send
   ```

   These cover everything Iris might do: full Tasks read/write, reading mail,
   creating drafts, and sending mail. If you only want a subset, see
   [Which scopes for what](#which-scopes-for-what).
4. Click **Authorize APIs**. Sign in with your test-user Google account and
   grant access. If you see an **"unverified app"** screen, click **Advanced →
   Go to Iris (unsafe)** — this warning is expected for an app in Testing that
   you built yourself (see [Troubleshooting](#troubleshooting)).
5. You are returned to the Playground on **Step 2 — Exchange authorization code
   for tokens**. Click **Exchange authorization code for tokens**.
6. In the response, copy the **Refresh token** — it starts with `1//`. This is
   the value Iris needs.

   (Reference: OAuth 2.0 Playground —
   https://developers.google.com/oauthplayground; OAuth 2.0 overview —
   https://developers.google.com/identity/protocols/oauth2.)

> If the response has no refresh token, remove the app's prior access at
> https://myaccount.google.com/permissions and repeat this step so Google
> re-prompts for consent and issues a fresh one.

---

## Step 5 — Paste the three values into Iris

1. Open the **Shortcuts app** and tap **Iris** to open it in the editor.
2. Near the top you will find three editable Text actions with placeholders:

   - `google-client-id-REPLACE-ME`
   - `google-client-secret-REPLACE-ME`
   - `google-refresh-token-REPLACE-ME`

3. Replace each placeholder with the matching value:
   - Client ID → the `...apps.googleusercontent.com` value
   - Client secret → the `GOCSPX-...` value
   - Refresh token → the `1//...` value

   Iris strips whitespace at runtime, so a stray newline from pasting will not
   break anything. Leaving any placeholder in place keeps the Google tools
   disabled — Iris checks that the refresh token starts with `1/` before it
   enables Tasks/Gmail.

---

## Step 6 — Run the "setup" primer once

iOS grants network and data permissions **per host** and **per shortcut**, on
first use, and they cannot be pre-authorized. The primer fires all of these
prompts in one manual run so hands-free Siri never gets interrupted later.

1. Run **Iris manually from the Shortcuts app on an unlocked phone** (not via
   Siri).
2. When it asks what to do, answer **"setup"** (or "grant permissions").
3. Tap **Always Allow** on every prompt, including the network prompts for:
   - `oauth2.googleapis.com` (token refresh)
   - `tasks.googleapis.com` (Google Tasks)
   - `gmail.googleapis.com` (Gmail)

The primer also performs a real token refresh and a Tasks call, so a successful
run confirms your client ID, secret, and refresh token actually work. If setup
fails here, jump to [Troubleshooting](#troubleshooting).

> Re-run "setup" any time you re-import or recompile the shortcut — permission
> grants reset on re-import.

---

## Which scopes for what

Request the least you need. You can re-run the Playground (Step 4) with a
narrower or wider scope set at any time.

| You want to… | Scope to authorize |
| --- | --- |
| Read + add + complete Google Tasks | `https://www.googleapis.com/auth/tasks` |
| Read Gmail (search, snippets, headers) | `https://www.googleapis.com/auth/gmail.readonly` |
| Create Gmail drafts (nothing sends) | `https://www.googleapis.com/auth/gmail.compose` |
| Send Gmail | `https://www.googleapis.com/auth/gmail.send` |

Notes on Google's scope tiers (relevant to publishing/verification):

- `gmail.readonly` is a **restricted** scope; `gmail.compose` and `gmail.send`
  are **sensitive** scopes. An app's overall classification is set by its most
  restrictive scope, so requesting `gmail.readonly` classifies the whole app as
  restricted for verification purposes. For personal Testing-mode use this only
  matters if you later try to publish.
  (Reference: Gmail API scopes —
  https://developers.google.com/workspace/gmail/api/auth/scopes.)

---

## Revocation

You are always in control of this access:

- **Revoke Iris's access to your Google account:** go to
  https://myaccount.google.com/permissions, find the app (the name you gave the
  OAuth consent screen), and remove it. The refresh token stops working
  immediately.
- **Delete the credentials entirely:** in the Cloud Console under **APIs &
  Services → Credentials**, delete the OAuth client ID. You can also disable
  either API under **APIs & Services → Library**.
- After revoking, re-mint a fresh refresh token via [Step 4](#step-4--get-a-refresh-token-via-the-oauth-20-playground)
  if you want to reconnect.

---

## Security notes

- **Keys live only in the local shortcut.** Your client ID, client secret, and
  refresh token are stored in editable Text actions inside your copy of Iris on
  your device — exactly like the NVIDIA and Tavily keys. They are never sent
  anywhere except Google's own OAuth and API endpoints.
- **Never commit them.** Do not paste real credentials into the repository, and
  do not upload a shortcut artifact you built after entering real values. Only
  distribute artifacts built from source, which contain the
  `google-...-REPLACE-ME` placeholders.
- **The build blocks baked-in secrets.** `scripts/validate-shortcut.py` fails
  the build if a real-looking Google client secret (`GOCSPX-`) or refresh token
  (`1//...`) appears in a compiled shortcut, mirroring the existing `nvapi-` and
  `tvly-` guards.
- **Least privilege.** Authorize only the scopes you actually use, and prefer
  read-only scopes if you never intend to draft or send mail.

---

## Troubleshooting

**`invalid_grant` when Iris refreshes the token.**
The refresh token is no longer valid. The usual causes: it expired (Testing-mode
apps expire refresh tokens after ~7 days), access was revoked at
myaccount.google.com/permissions, or the client ID/secret pasted into Iris does
not match the client used to mint the token. Fix: confirm the three values
match, then re-mint the refresh token via [Step 4](#step-4--get-a-refresh-token-via-the-oauth-20-playground)
and paste the new `1//...` value into Iris.

**"Token expired" / Google tools suddenly stopped working after about a week.**
This is the 7-day Testing-mode refresh-token expiry. Re-run Step 4 to get a new
refresh token, or publish the app to remove the expiry (subject to Google's
verification requirements for restricted Gmail scopes). See
[Testing vs Published](#testing-vs-published-and-the-7-day-refresh-token-expiry).

**"Google hasn't verified this app" / unverified-app warning during Step 4.**
Expected for an app in Testing that you created yourself. Click **Advanced**,
then **Go to \<app name\> (unsafe)**, and continue. Make sure the account you are
signing in with is listed under **Test users** (Step 2); if it is not, Google
blocks the consent entirely.

**Missing scope / "insufficient permission" or `ACCESS_TOKEN_SCOPE_INSUFFICIENT`
when Iris calls Gmail or Tasks.**
The refresh token was minted without the scope that tool needs. For example,
sending mail needs `gmail.send`, drafts need `gmail.compose`, reading needs
`gmail.readonly`, and Tasks needs `tasks`. Re-run [Step 4](#step-4--get-a-refresh-token-via-the-oauth-20-playground)
and authorize the full scope list, then paste the new refresh token into Iris.

**Setup primer fails at the Google step.**
Check that all three values are pasted correctly (no placeholder left behind,
correct value in each box) and that you tapped **Always Allow** on the
`oauth2.googleapis.com` and `tasks.googleapis.com` network prompts. A wrong
client secret typically surfaces as `invalid_client`; a bad/expired refresh
token as `invalid_grant`.

---

*Content from Google's documentation was rephrased/summarized for licensing
compliance; no large verbatim excerpts are reproduced. See the inline links for
authoritative details.*
