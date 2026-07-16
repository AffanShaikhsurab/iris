"""Create a deterministic, allowlisted Iris training-container build context."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any

ROOT_FILES = ("Dockerfile.train", "requirements.lock")
ROOT_DIRS = ("configs", "src")
FIXED_TIME = (1980, 1, 1, 0, 0, 0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _files(root: Path) -> list[Path]:
    selected = [root / name for name in ROOT_FILES]
    for name in ROOT_DIRS:
        selected.extend(path for path in (root / name).rglob("*") if path.is_file())
    selected = [
        path for path in selected
        if "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"}
    ]
    missing = [path for path in selected if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing build inputs: {missing}")
    return sorted(selected, key=lambda path: path.relative_to(root).as_posix())


def _entry(path: Path, root: Path) -> tuple[zipfile.ZipInfo, bytes, dict[str, Any]]:
    relative = path.relative_to(root).as_posix()
    data = path.read_bytes()
    info = zipfile.ZipInfo(relative, FIXED_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    record = {"path": relative, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    return info, data, record

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    inventory: list[dict[str, Any]] = []
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in _files(root):
                info, data, record = _entry(path, root)
                archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
                inventory.append(record)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    manifest = {
        "schema_version": 1,
        "archive": {"path": output.as_posix(), "bytes": output.stat().st_size, "sha256": _sha256(output)},
        "files": inventory,
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"archive": manifest["archive"], "manifest": manifest_path.as_posix()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
