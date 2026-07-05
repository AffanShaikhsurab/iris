# Roadmap

## Near Term

- Test the `build-shortcut.yml` workflow in a public GitHub repository.
- Record whether `shortcuts sign --mode anyone` works on GitHub-hosted macOS.
- Import the signed artifact on iPhone and verify each route manually.
- Replace placeholder API key setup with a safer user configuration flow.
- Add route-specific examples and screenshots after device validation.

## Route Expansion

- Calendar availability summary.
- Mail or Gmail draft preparation where app actions are available.
- Message reply drafting for shared text.
- Notes append/update routes.
- Reminder due-date parsing.
- Curated route packs from `docs/shortcut-catalog.md`, starting with generic
  search/open/capture routes instead of app-specific personal dashboards.

## Generator Improvements

- Track Cherri compiler changes and pin a known-good version in CI after the
  first successful build.
- Add a decompiler/dump workflow to compare generated shortcuts against manually
  built reference shortcuts.
- Add stricter validation for Shortcuts control-flow parameters.
- Generate route docs from the route registry.
