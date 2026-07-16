#!/usr/bin/env python3
"""Merge a trained LoRA adapter into a fresh immutable BF16 base and save the result.

Reloads the sealed base (never the quantized/training copy), attaches the adapter,
merges with safe_merge (fails on NaN/Inf), and writes a standalone safetensors model
plus tokenizer/template. This merged BF16 model is the truth artifact for downstream
GGUF/MLX quantization.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="Sealed BF16 base snapshot")
    parser.add_argument("--adapter-dir", type=Path, required=True, help="Trained LoRA adapter (work/out-full)")
    parser.add_argument("--output", type=Path, required=True, help="Destination for the merged model")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base, args.adapter_dir, local_files_only=True)
    merged = model.merge_and_unload(safe_merge=True)
    args.output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.output, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=False, use_fast=True
    )
    tokenizer.save_pretrained(args.output)
    print(f"merged BF16 model written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
