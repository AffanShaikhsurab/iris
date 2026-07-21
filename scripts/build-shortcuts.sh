#!/usr/bin/env bash
set -euo pipefail

mkdir -p dist
tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

# Cherri's parser is strict about CRLF. Normalize the source so Windows editing
# does not break Linux/macOS CI compilation. Also, for a LOCAL personal build,
# inject real credentials from a gitignored .env.local into this THROWAWAY temp
# copy (never the committed source), so re-imported builds work without
# re-pasting keys. CI has no .env.local and therefore builds the safe
# placeholder version; distributed artifacts must always be placeholder builds.
python3 - <<'PY' "shortcuts/iris.cherri" "$tmpdir/Iris.cherri"
from pathlib import Path
import sys

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
text = src.read_text().replace("\r\n", "\n")

env_path = Path(".env.local")
if env_path.exists():
    env = {}
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")

    def real(v):
        return bool(v) and "REPLACE-ME" not in v

    mapping = {
        "nvapi-REPLACE-ME": env.get("NIM_API_KEY", ""),
        "tvly-REPLACE-ME": env.get("TAVILY_KEY", ""),
        "https://script.google.com/macros/s/REPLACE-ME/exec": env.get("IRIS_PROXY_URL", ""),
        "iris-proxy-secret-REPLACE-ME": env.get("IRIS_PROXY_SECRET", ""),
    }
    injected = 0
    for placeholder, value in mapping.items():
        if real(value):
            text = text.replace(placeholder, value)
            injected += 1
    model = env.get("NIM_MODEL_ID", "")
    if real(model):
        text = text.replace('text("mistralai/mistral-small-4-119b-2603")', f'text("{model}")')
    if injected or real(model):
        sys.stderr.write(
            f"[build] .env.local found: injected {injected} credential(s) into "
            "the LOCAL build (not committed, not for distribution)\n")

dst.write_text(text, newline="\n")
PY

# Cherri (v2.3.0) writes its output next to the input file and only honors the
# basename of --output, so compile in the temp dir and collect artifacts after.
"${CHERRI_BIN:-cherri}" "$tmpdir/Iris.cherri" --output="$tmpdir/Iris.shortcut" "$@"

signed="$tmpdir/Iris.shortcut"
unsigned="$tmpdir/Iris_unsigned.shortcut"

if [[ -f "$signed" ]]; then
  cp "$signed" "dist/Iris.shortcut"
  cp "$signed" "dist/iris.shortcut"
fi

if [[ -f "$unsigned" ]]; then
  cp "$unsigned" "dist/Iris_unsigned.shortcut"
fi
