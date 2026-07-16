# Iris dataset (upload bundle)

Native mobile-actions-format function-calling data for Iris.

- `iris.jsonl` - full merged dataset (all rows).
- `train.jsonl` - rows with metadata=train.
- `eval.jsonl` - rows with metadata=eval.

Rows: 10494 total, 9449 train, 1045 eval.
Composition: 9654 retained google/mobile-actions rows + 840 Iris-authored extension rows.

## Attribution and license

The retained rows come from `google/mobile-actions`, reported as CC-BY-4.0. Attribution to google/mobile-actions is required when redistributing this data or a model trained on it. Extension rows are Iris-authored; verify any teacher-model terms before redistribution.
