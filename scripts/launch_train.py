#!/usr/bin/env python3
"""torchrun-compatible launcher for multi-GPU DDP training.

Relative imports in iris_training require the package on sys.path; torchrun runs a
script path, so this shim adds src/ and delegates to iris_training.train.main().

Usage:
  torchrun --standalone --nproc_per_node=<N> scripts/launch_train.py --config <path>
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iris_training.train import main

if __name__ == "__main__":
    raise SystemExit(main())
