# Iris dataset for Fireworks SFT

Generated OpenAI-compatible function-calling views of the canonical
`../iris.jsonl` dataset. The source dataset remains unchanged.

## Files

- `train.jsonl` — 9,449 supervised fine-tuning examples.
- `eval.jsonl` — 1,045 held-out evaluation examples.

## Conversion rules

- Keeps every source record and both tool catalogs (7-tool mobile-actions and
  32-tool Iris profiles).
- Removes the source-only `metadata` field after splitting.
- Converts `developer` messages to `system`.
- Adds `type: "function"` to tool definitions and calls.
- Lowercases JSON Schema types (`OBJECT` → `object`, `STRING` → `string`).
- Serializes `function.arguments` objects as compact JSON strings.
- Adds `content: ""` to assistant tool-call messages because the Fireworks
  Qwen3 renderer requires every message to define `content`.
- Preserves multi-turn tool observations as labeled `user` messages because the
  documented Fireworks SFT message roles are `system`, `user`, and `assistant`.

Regenerate and validate both outputs from the repository root:

```powershell
python scripts/prepare-fireworks-dataset.py
```

Upload these derived files to Fireworks; do not upload the native files from
`../upload/` directly. The retained `google/mobile-actions` data is CC-BY-4.0;
keep its attribution when redistributing data or a resulting model.