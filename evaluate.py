#!/usr/bin/env python3
"""Evaluate a causal LM on Maritime Benchmark v1.0.

Primary protocol: option likelihood (no instruction-following required).
Secondary protocol: choices-in-prompt generation.

The script writes one JSONL record per item and protocol, so every table cell can be
reproduced and audited. It is designed to run on a 12 GB RTX 5070 in WSL2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# torch/transformers are imported lazily so the pure functions can be unit-tested
# on a machine without a GPU stack.


def strip_label(text: str) -> str:
    """Remove a leading 'A. ', 'B) ', etc. from an option."""
    return re.sub(r"^\s*[A-D][\.\)\:]\s*", "", text).strip()


_LEADING_LABEL_RE = re.compile(r"^\s*[*_`]*\(?([A-Da-d])\)?[*_`]*\s*[\.\)\:\-–—]\s*\S")
_LONE_LETTER_RE = re.compile(
    r"^\s*[*_`\s]*\(?([A-Da-d])\)?[*_`\s]*[\.\)\:\-–—]?\s*$"
)
_ANSWER_CUE_RE = re.compile(
    r"(?:final\s+answer|answer|correct\s+(?:answer|option|choice)|choice|option|"
    r"select(?:ed)?|choose|pick)"
    r"\s*(?:is|was|would\s+be|:|=)?\s*[*_`]*\(?([A-Da-d])\)?[*_`]*"
    r"(?=\s*(?:[\.\)\:\,;]|\b(?:is|was|seems|appears|because|since)\b|$))",
    re.IGNORECASE,
)


def _normalise_for_match(text: str) -> str:
    """Lower-case, strip an option label, and collapse punctuation for comparison."""
    text = text.strip().lower()
    text = re.sub(r"^\s*[*_`]*\(?[a-d]\)?[*_`]*\s*[\.\)\:\-–—]\s*", "", text)
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def parse_letter(text: str, choices: list[str] | None = None) -> str:
    """Parse a generated answer into A-D.

    Free-form generations frequently begin with an article ("A vessel must ...") or
    mention letters in prose. We therefore accept only (a) a lone letter, (b) a
    letter followed by option punctuation, (c) an exact option-text match, or (d) a
    letter immediately after an explicit answer cue such as "Answer:" or "option is".
    Anything else is returned as '?' and counted as a parse failure.
    """
    if not text or not text.strip():
        return "?"
    stripped = text.strip()

    m = _LONE_LETTER_RE.match(stripped)
    if m:
        return m.group(1).upper()

    m = _LEADING_LABEL_RE.match(stripped)
    if m:
        return m.group(1).upper()

    if choices:
        target = _normalise_for_match(stripped)
        target = re.sub(
            r"^(?:final\s+answer|answer|correct\s+(?:answer|option|choice)|choice|option)"
            r"\s*(?:is|was|:|=)?\s*",
            "",
            target,
        ).strip()
        matches = [
            letter
            for letter, choice in zip("ABCD", choices)
            if target and target == _normalise_for_match(strip_label(choice))
        ]
        if len(matches) == 1:
            return matches[0]

    m = _ANSWER_CUE_RE.search(stripped)
    if m:
        return m.group(1).upper()
    return "?"


def build_generation_prompt(
    item: dict[str, Any],
    fewshot: list[dict[str, Any]] | None = None,
    include_instructions: bool = True,
) -> str:
    """Build a plain-text MCQ prompt with the options visible."""
    parts: list[str] = []
    for ex in fewshot or []:
        parts.append(
            f"Question: {ex['question']}\n"
            + "\n".join(ex["choices"])
            + f"\nAnswer: {ex['answer']}\n"
        )
    parts.append(
        f"Question: {item['question']}\n"
        + "\n".join(item["choices"])
        + "\nAnswer:"
    )
    if include_instructions and not fewshot:
        parts.insert(
            0,
            "Answer the multiple-choice question by selecting the single best option. "
            "Reply with only the letter A, B, C, or D.\n",
        )
    return "\n".join(parts)


def build_likelihood_context(item: dict[str, Any]) -> str:
    """Context for option-likelihood scoring.

    We deliberately do not include the option labels here: the choice text itself is
    the continuation. This avoids learning a letter prior from the prompt format.
    """
    return f"Question: {item['question']}\nAnswer:"


def choice_texts(item: dict[str, Any]) -> list[str]:
    return [strip_label(c) for c in item["choices"]]


def load_benchmark(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["items"]


def canonical_item_sha256(item: dict[str, Any]) -> str:
    payload = json.dumps(item, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def filter_items(
    items: list[dict[str, Any]],
    categories: set[str] | None,
    splits: set[str] | None,
    eval_splits: set[str] | None,
    ids: set[str] | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    out = []
    for item in items:
        if categories and item["category"] not in categories:
            continue
        if splits and item["split"] not in splits:
            continue
        if eval_splits and item["eval_split"] not in eval_splits:
            continue
        if ids and item["id"] not in ids:
            continue
        out.append(item)
    if limit:
        out = out[:limit]
    return out


def load_done(path: Path, benchmark_sha256: str | None = None) -> set[tuple[str, str, str]]:
    done: set[tuple[str, str, str]] = set()
    if not path.exists():
        return done
    ignored = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if benchmark_sha256 and r.get("benchmark_sha256") != benchmark_sha256:
                ignored += 1
                continue
            done.add((r.get("model", ""), r.get("protocol", ""), r.get("item_id", "")))
    if ignored:
        print(
            f"WARNING: ignoring {ignored} record(s) from a different benchmark revision",
            file=sys.stderr,
        )
    return done


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def get_device(requested: str):
    import torch

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def get_dtype(name: str):
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def load_model(path: str, device, dtype, attn_implementation: str, trust_remote_code: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code)
    kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    try:
        model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    except Exception as exc:
        # Some Qwen3.5 multimodal checkpoints expose only the image-text auto class.
        print(f"AutoModelForCausalLM failed ({exc}); trying AutoModelForImageTextToText", file=sys.stderr)
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(path, **kwargs)
    model.to(device)
    model.eval()
    return model, tok


def score_likelihood(model, tok, item: dict[str, Any], device) -> dict[str, Any]:
    """Return per-option log-probabilities and the argmax prediction."""
    import torch

    context = build_likelihood_context(item)
    ctx_ids = tok(context, return_tensors="pt", add_special_tokens=True).input_ids
    ctx_len = ctx_ids.shape[1]
    scores = []
    token_counts = []
    byte_counts = []
    texts = choice_texts(item)

    for text in texts:
        choice = " " + text.strip()
        choice_ids = tok(choice, return_tensors="pt", add_special_tokens=False).input_ids
        if choice_ids.shape[1] == 0:
            scores.append(float("-inf"))
            token_counts.append(1)
            byte_counts.append(max(1, len(choice.encode("utf-8"))))
            continue
        input_ids = torch.cat([ctx_ids, choice_ids], dim=1).to(device)
        attention_mask = torch.ones_like(input_ids)
        labels = torch.full_like(input_ids, -100)
        labels[:, ctx_len:] = choice_ids.to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        n = int(choice_ids.shape[1])
        sum_logprob = float(-out.loss.item() * n)
        scores.append(sum_logprob)
        token_counts.append(n)
        byte_counts.append(max(1, len(choice.encode("utf-8"))))

    token_norm = [s / max(1, n) for s, n in zip(scores, token_counts)]
    byte_norm = [s / max(1, n) for s, n in zip(scores, byte_counts)]
    pred = "ABCD"[max(range(4), key=lambda i: token_norm[i])]
    pred_byte = "ABCD"[max(range(4), key=lambda i: byte_norm[i])]
    pred_raw = "ABCD"[max(range(4), key=lambda i: scores[i])]
    return {
        "context": context,
        "choice_texts": texts,
        "sum_logprob": scores,
        "token_counts": token_counts,
        "byte_counts": byte_counts,
        "token_norm": token_norm,
        "byte_norm": byte_norm,
        "pred": pred,
        "pred_byte": pred_byte,
        "pred_raw": pred_raw,
    }


def generate_choice(
    model,
    tok,
    item: dict[str, Any],
    device,
    prompt_style: str,
    fewshot: list[dict[str, Any]] | None,
    max_new_tokens: int,
) -> dict[str, Any]:
    import torch

    prompt = build_generation_prompt(item, fewshot=fewshot)
    if prompt_style == "chat" and hasattr(tok, "apply_chat_template"):
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = prompt
    inputs = tok(text, return_tensors="pt").to(device)
    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id if tok.eos_token_id is not None else 0,
        )
    out_text = tok.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return {
        "prompt": text,
        "raw_output": out_text,
        "pred": parse_letter(out_text, item["choices"]),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate a model on Maritime Benchmark v1.0")
    ap.add_argument("--model", required=True, help="Path or HF id of the model")
    ap.add_argument("--model-name", default=None, help="Short name used in output records")
    ap.add_argument("--benchmark", default=str(Path(__file__).resolve().parent / "benchmark" / "maritime_bench_v1.json"))
    ap.add_argument("--output", required=True, help="JSONL output path")
    ap.add_argument("--scoring", choices=["likelihood", "generation", "both"], default="both")
    ap.add_argument("--prompt-style", choices=["plain", "chat"], default="plain")
    ap.add_argument("--fewshot", type=int, default=0, choices=[0, 5])
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    ap.add_argument("--attn-implementation", default="sdpa")
    ap.add_argument("--trust-remote-code", action="store_true", default=True)
    ap.add_argument("--categories", default=None, help="Comma-separated category filter")
    ap.add_argument("--split", default=None, help="Comma-separated public_core/port_local filter")
    ap.add_argument("--eval-split", default="test", help="Comma-separated dev/test filter")
    ap.add_argument("--ids", default=None, help="Comma-separated item ids")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--list", action="store_true", help="List filtered items and exit")
    args = ap.parse_args(argv)

    benchmark_path = Path(args.benchmark)
    benchmark_bytes = benchmark_path.read_bytes()
    benchmark_sha256 = hashlib.sha256(benchmark_bytes).hexdigest()
    benchmark_doc = json.loads(benchmark_bytes)
    items = benchmark_doc["items"]
    benchmark_version = benchmark_doc.get("version", "unknown")
    item_hashes = {item["id"]: canonical_item_sha256(item) for item in items}
    categories = set(args.categories.split(",")) if args.categories else None
    splits = set(args.split.split(",")) if args.split else None
    eval_splits = set(args.eval_split.split(",")) if args.eval_split else None
    ids = set(args.ids.split(",")) if args.ids else None
    items = filter_items(items, categories, splits, eval_splits, ids, args.limit)

    if args.list:
        from collections import Counter
        print(json.dumps({
            "n": len(items),
            "categories": dict(Counter(i["category"] for i in items)),
            "splits": dict(Counter(i["split"] for i in items)),
            "eval_splits": dict(Counter(i["eval_split"] for i in items)),
        }, indent=2))
        return 0

    model_name = args.model_name or Path(args.model).name
    output = Path(args.output)
    if not args.resume and output.exists():
        output.unlink()
    done = load_done(output, benchmark_sha256) if args.resume else set()

    device = get_device(args.device)
    dtype = get_dtype(args.dtype)
    print(f"Loading {args.model} on {device} ({args.dtype}) ...", flush=True)
    model, tok = load_model(args.model, device, dtype, args.attn_implementation, args.trust_remote_code)

    # Few-shot examples come only from the dev split.
    dev_items = [i for i in benchmark_doc["items"] if i["eval_split"] == "dev"]
    fewshot = dev_items[: args.fewshot] if args.fewshot else []

    protocols = ["likelihood", "generation"] if args.scoring == "both" else [args.scoring]
    for protocol in protocols:
        for n, item in enumerate(items, 1):
            key = (model_name, protocol, item["id"])
            if key in done:
                continue
            if protocol == "likelihood":
                rec = score_likelihood(model, tok, item, device)
                record = {
                    "model": model_name,
                    "model_path": args.model,
                    "protocol": "likelihood",
                    "benchmark_version": benchmark_version,
                    "benchmark_sha256": benchmark_sha256,
                    "item_sha256": item_hashes[item["id"]],
                    "item_id": item["id"],
                    "category": item["category"],
                    "split": item["split"],
                    "eval_split": item["eval_split"],
                    "answer": item["answer"],
                    "pred": rec["pred"],
                    "pred_byte": rec["pred_byte"],
                    "pred_raw": rec["pred_raw"],
                    "scores": {
                        "sum_logprob": rec["sum_logprob"],
                        "token_norm": rec["token_norm"],
                        "byte_norm": rec["byte_norm"],
                        "token_counts": rec["token_counts"],
                        "byte_counts": rec["byte_counts"],
                    },
                    "choice_texts": rec["choice_texts"],
                    "context": rec["context"],
                }
            else:
                rec = generate_choice(
                    model, tok, item, device, args.prompt_style, fewshot, args.max_new_tokens
                )
                record = {
                    "model": model_name,
                    "model_path": args.model,
                    "protocol": "generation",
                    "prompt_style": args.prompt_style,
                    "fewshot": args.fewshot,
                    "benchmark_version": benchmark_version,
                    "benchmark_sha256": benchmark_sha256,
                    "item_sha256": item_hashes[item["id"]],
                    "item_id": item["id"],
                    "category": item["category"],
                    "split": item["split"],
                    "eval_split": item["eval_split"],
                    "answer": item["answer"],
                    "pred": rec["pred"],
                    "raw_output": rec["raw_output"],
                    "prompt": rec["prompt"],
                }
            append_jsonl(output, record)
            if n % 10 == 0 or n == len(items):
                print(f"  {protocol} {n}/{len(items)}", flush=True)

    print(f"Done. Output: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
