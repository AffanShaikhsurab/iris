# Release Checklist

Use this checklist before publishing a signed Shortcut artifact.

## Source Checks

- [ ] `shortcuts/agent_router.cherri` compiles in CI.
- [ ] `docs/routes.md` matches the supported route list.
- [ ] `README.md` describes the current signing/import status.
- [ ] No real API keys or private automation data are committed.
- [ ] The Cherri compiler version is recorded from the build logs.

## Artifact Checks

- [ ] Build comes from a tagged commit.
- [ ] Unsigned `.shortcut` artifact is uploaded.
- [ ] Signed `.shortcut` artifact is uploaded, or signing failure is documented.
- [ ] `signing-result.txt` and `signing-output.log` are retained.

## Device Checks

- [ ] Signed shortcut imports on iPhone.
- [ ] “Hey Siri, Agent Router” starts the shortcut.
- [ ] Summarize route works.
- [ ] Draft reply route works.
- [ ] Note/reminder route status is documented.
- [ ] ChatGPT fallback status is documented.

## Release Notes

Release notes should include:

- Supported routes.
- Known limitations.
- Signing status.
- Tested iOS version and device.
- Privacy warning for API-backed routing.
