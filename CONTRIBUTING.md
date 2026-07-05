# Contributing

Thanks for helping build Iris. The project is intentionally
small and explicit so contributors can understand what each generated Shortcut
action does.

## Local Setup

Install Cherri:

```bash
brew tap electrikmilk/cherri
brew install electrikmilk/cherri/cherri
```

Or:

```bash
go install github.com/electrikmilk/cherri@latest
```

Compile the Shortcut:

```bash
mkdir -p dist
cherri shortcuts/iris.cherri --debug --output "dist/Iris.shortcut"
```

For the known-good Windows path, use WSL2 with the Linux Cherri release and
HubSign as documented in `docs/signing.md`. The signed import artifacts should
start with the `AEA1` header.

## Development Guidelines

- Keep Cherri source inspectable. Prefer standard Cherri actions over raw
  actions when they are stable and documented.
- Add compile validation and docs for every new route.
- Do not commit secrets, API keys, personal prompts, webhook URLs, contacts, or
  device-specific automation data.
- Keep examples generic and safe for a public repository.
- Document device-specific or app-version-specific behavior.

## Adding A Route

1. Add the route to `shortcuts/iris.cherri`.
2. Prefer Cherri standard library actions; use `rawAction` only when needed.
3. Update `README.md` and `docs/routes.md`.
4. Compile the shortcut with `--debug`.
5. Test importing a signed version on device.
6. Record device behavior in `docs/device-testing.md` or release notes.

## Pull Request Checklist

- `cherri shortcuts/iris.cherri --debug --output "dist/Iris.shortcut"` succeeds.
- Documentation reflects user-facing behavior.
- No secrets or private data are included.
- New routes follow `docs/security.md`.
