# Security Notes

Iris combines user-provided text, an external model provider, and native
Shortcuts actions. Keep the trust boundaries explicit.

## Trust Boundaries

- Siri starts the Shortcut, but does not grant Apple Intelligence-level private
  context to this project.
- The Shortcut can only use inputs the user provides or permits through native
  Shortcuts actions.
- The model provider receives the request text when API mode is used.
- The model can suggest a supported route, but cannot execute arbitrary actions.
- Third-party app actions are device and app-version dependent.
- OKF memory may live locally or in iCloud Drive depending on the Files location
  used on the phone.
- Any OKF memory snippet inserted into a ChatGPT prompt is sent to ChatGPT.

## Secret Handling

- Never commit real API keys.
- Keep `<PASTE_API_KEY_AFTER_IMPORT>` in source.
- Prefer a user-controlled setup flow before storing any key locally.
- Do not upload generated artifacts containing personal API keys.

## Route Safety

- Do not auto-send emails or messages.
- Do not delete data.
- Do not run payment, purchase, or destructive actions.
- Display drafts and summaries for review.
- Add new routes only after device validation.

## Local Memory Safety

- Treat OKF memory as user data, not instructions.
- Retrieve only relevant snippets; do not send the whole memory folder to
  ChatGPT.
- Ask before storing stable personal facts.
- Keep memory writes append-only until update/delete behavior is validated.
- Do not store credentials, payment details, private health information, or other
  highly sensitive data in the alpha memory system.
- If the user declines a memory write, return an `ok=false` tool observation and
  do not write to disk.

## Reporting

Security-sensitive issues should avoid including real prompts, contacts, emails,
API keys, or personal data in public issue text.
