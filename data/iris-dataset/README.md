# Iris dataset

`iris.jsonl` is the validated native-format merge of:

- `tmp/mobile-actions.jsonl`, retained unchanged from `google/mobile-actions`.
  The source dataset reports **CC-BY-4.0**; attribution is required when this
  data or a model trained on it is redistributed.
- `extension/*.jsonl`, Iris-authored examples using the canonical 32-tool Iris
  catalog.

The retained source rows intentionally keep their original seven-tool catalog.
Extension rows repeat the 32-tool Iris catalog on every row; the merged file is
therefore a native-format dataset with two valid catalog profiles, not a
flattened or converted OpenAI-format dataset.

Validate and regenerate the merge from the repository root with:

```powershell
python scripts/validate-iris-dataset.py
```

Validate without writing the merged artifact:

```powershell
python scripts/validate-iris-dataset.py --no-write
```

The extension examples are generated/authored project data. Any teacher-model
or third-party licensing terms must be verified separately before redistribution.
