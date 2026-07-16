#!/usr/bin/env python3
"""Generate model outputs for Iris eval rows and score them with the deterministic evaluator.

Builds one case per eval row from the first assistant turn (the core routing decision):
prompt = messages before the first assistant; gold = that assistant's tool_calls.
Renders with MiniCPM5's native inference template (non-thinking), greedy-decodes, then
scores with iris_training.evaluate. No output repair.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

CONTROL_TOKEN = re.compile(r"<\|[^|]*?\|>")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iris_training.data import normalize_row, read_jsonl, to_model_boundary
from iris_training.evaluate import aggregate, evaluate_case


def _first_assistant_index(messages: list[dict[str, Any]]) -> int:
    for index, message in enumerate(messages):
        if message["role"] == "assistant":
            return index
    raise ValueError("row has no assistant message")


def build_case(raw: Any, location: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    row = normalize_row(raw, location)
    boundary = to_model_boundary(row)
    index = _first_assistant_index(boundary["messages"])
    prompt_messages = boundary["messages"][:index]
    gold_calls = boundary["messages"][index].get("tool_calls", [])
    case = {
        "case_id": location,
        "tools": row["tools"],
        "gold_tool_calls": [{"function": c["function"]} for c in gold_calls],
        "parallel_order_insensitive": len(gold_calls) > 1,
    }
    return case, prompt_messages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="Base MiniCPM5 snapshot dir")
    parser.add_argument("--adapter-dir", type=Path, default=None, help="LoRA adapter dir (work/out)")
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 means all rows")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--gguf", type=Path, default=None, help="Score a GGUF model via llama-cpp instead of Transformers")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=False, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def render(prompt_messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
        return tokenizer.apply_chat_template(
            prompt_messages, tools=tools, enable_thinking=False,
            add_generation_prompt=True, tokenize=False,
        )

    if args.gguf is not None:
        from llama_cpp import Llama

        llm = Llama(
            model_path=str(args.gguf), n_ctx=4096, verbose=False,
            n_gpu_layers=int(os.environ.get("IRIS_GGUF_GPU_LAYERS", "0")),
        )

        def generate(prompt_messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
            tokens = llm.tokenize(render(prompt_messages, tools).encode("utf-8"), add_bos=False, special=True)
            produced: list[int] = []
            for token in llm.generate(tokens, temp=0.0):
                if token == llm.token_eos() or len(produced) >= args.max_new_tokens:
                    break
                produced.append(token)
            return llm.detokenize(produced, special=True).decode("utf-8", errors="ignore")
    else:
        import torch
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir, local_files_only=True, trust_remote_code=False,
            attn_implementation="sdpa", torch_dtype=torch.bfloat16,
        )
        if args.adapter_dir is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, args.adapter_dir, local_files_only=True)
            model = model.merge_and_unload()
        model.to("cuda").eval()

        def generate(prompt_messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
            inputs = tokenizer(render(prompt_messages, tools), return_tensors="pt", return_token_type_ids=False).to("cuda")
            with torch.no_grad():
                generated = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens,
                    do_sample=False, pad_token_id=tokenizer.pad_token_id,
                )
            return tokenizer.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)

    rows = list(read_jsonl(args.eval_file))
    if args.limit:
        rows = rows[: args.limit]

    results: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    total = len(rows)
    for done, (line_number, raw) in enumerate(rows, 1):
        if done == 1 or done % 25 == 0 or done == total:
            print(f"[{done}/{total}] scoring...", file=sys.stderr, flush=True)
        location = f"{args.eval_file.name}:{line_number}"
        case, prompt_messages = build_case(raw, location)
        raw_completion = generate(prompt_messages, case["tools"])
        completion = CONTROL_TOKEN.sub("", raw_completion)
        if tokenizer.eos_token:
            completion = completion.replace(tokenizer.eos_token, "")
        completion = completion.strip()
        outputs.append({"case_id": case["case_id"], "output": completion})
        results.append(evaluate_case(case, completion))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "outputs.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for item in outputs:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    with (args.out_dir / "per-case.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for item in results:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    metrics = aggregate(results)
    (args.out_dir / "aggregate.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
