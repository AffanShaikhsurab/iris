#!/usr/bin/env bash
set -euo pipefail

mkdir -p dist
tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

# Cherri's parser is strict about CRLF. Normalize the source so Windows editing
# does not break Linux/macOS CI compilation.
python3 - <<'PY' "shortcuts/agent_router.cherri" "$tmpdir/Agent Router.cherri"
from pathlib import Path
import sys

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
dst.write_text(src.read_text().replace("\r\n", "\n"), newline="\n")
PY

# Cherri (v2.3.0) writes its output next to the input file and only honors the
# basename of --output, so compile in the temp dir and collect artifacts after.
"${CHERRI_BIN:-cherri}" "$tmpdir/Agent Router.cherri" --output="$tmpdir/Agent Router.shortcut" "$@"

signed="$tmpdir/Agent Router.shortcut"
unsigned="$tmpdir/Agent Router_unsigned.shortcut"

if [[ -f "$signed" ]]; then
  cp "$signed" "dist/Agent Router.shortcut"
  cp "$signed" "dist/agent-router.shortcut"
fi

if [[ -f "$unsigned" ]]; then
  cp "$unsigned" "dist/Agent Router_unsigned.shortcut"
fi
