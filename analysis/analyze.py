#!/usr/bin/env python3
"""Analyze per-item JSONL outputs from eval_mcq.py.

Produces:
  - overall and per-category accuracy with Wilson 95% CIs
  - macro-average and balanced accuracy over answer letters
  - constant-letter / longest / shortest / first baselines
  - paired McNemar exact tests against base and against the best baseline
  - Holm-Bonferroni correction across categories
  - a Markdown summary and a machine-readable JSON

Every table cell is derived from the per-item JSONL records. The script does not
read aggregate numbers from the manuscript or from any previous run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

try:
    from scipy.stats import binom, binomtest
except Exception:  # pragma: no cover
    binom = None
    binomtest = None


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar_exact(a_correct: list[bool], b_correct: list[bool]) -> dict[str, Any]:
    """Exact two-sided McNemar test on paired binary outcomes."""
    if len(a_correct) != len(b_correct):
        raise ValueError("paired outcomes must have the same length")
    b = sum(1 for a, c in zip(a_correct, b_correct) if a and not c)
    c = sum(1 for a, d in zip(a_correct, b_correct) if not a and d)
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p": 1.0}
    if binomtest is not None:
        p = float(binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)
    else:
        from math import comb

        k = min(b, c)
        p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))
    return {"b": b, "c": c, "p": p}


def random_baseline(n: int, p: float = 0.25) -> dict[str, Any]:
    """Uniform-random baseline and the central 95% binomial sampling band."""
    if n == 0:
        return {"n": 0, "expected_accuracy": p, "ci95": [0.0, 0.0], "note": "uniform over four options"}
    if binom is not None:
        lo = float(binom.ppf(0.025, n, p)) / n
        hi = float(binom.ppf(0.975, n, p)) / n
    else:  # pragma: no cover
        z = 1.96
        se = math.sqrt(p * (1 - p) / n)
        lo, hi = max(0.0, p - z * se), min(1.0, p + z * se)
    return {
        "n": n,
        "expected_accuracy": p,
        "ci95": [lo, hi],
        "note": "uniform over four options; CI is the central 95% binomial sampling band",
    }


def holm(pvals: list[tuple[str, float]]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values, preserving input names."""
    m = len(pvals)
    ordered = sorted(pvals, key=lambda x: x[1])
    out: dict[str, float] = {}
    prev = 0.0
    for rank, (name, p) in enumerate(ordered):
        adj = min(1.0, max(prev, p * (m - rank)))
        out[name] = adj
        prev = adj
    return out


def load_records(paths: list[Path]) -> list[dict[str, Any]]:
    recs: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
    return recs


WORD_RE = re.compile(r"(?u)\b\w+\b")


def _words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def _contains_sequence(haystack: list[str], needle: list[str]) -> bool:
    n = len(needle)
    if n == 0 or n > len(haystack):
        return False
    return any(haystack[i : i + n] == needle for i in range(len(haystack) - n + 1))


def load_contamination_subsets(
    items: list[dict[str, Any]],
    path: str | Path,
    benchmark_sha256: str | None = None,
) -> tuple[dict[str, set[str]], dict[str, Any]]:
    """Split item IDs by where a verified 9-gram match falls.

    ``none``: no verified n-gram match.
    ``question_only``: every verified example n-gram is inside the question.
    ``answer_or_boundary``: at least one verified example is in the correct answer
    or spans the question/answer boundary.

    The split is descriptive. It is not a causal test of contamination.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    audit_sha = data.get("benchmark_sha256")
    if benchmark_sha256 and audit_sha and audit_sha != benchmark_sha256:
        raise SystemExit(
            f"contamination audit was run on benchmark {audit_sha}, "
            f"but the evaluation benchmark is {benchmark_sha256}"
        )
    records = {rec["id"]: rec for rec in data.get("items", [])}
    subsets: dict[str, set[str]] = {
        "none": set(),
        "question_only": set(),
        "answer_or_boundary": set(),
    }
    for item in items:
        rec = records.get(item["id"])
        if rec is None:
            raise SystemExit(f"contamination audit has no record for item {item['id']}")
        if not rec.get("verified_match"):
            subsets["none"].add(item["id"])
            continue
        question = _words(item["question"])
        answer = _words(item["correct_text"])
        answer_or_boundary = False
        for doc in rec.get("verified_top_docs", []):
            example = _words(doc.get("example_ngram", ""))
            if _contains_sequence(answer, example) or not _contains_sequence(question, example):
                answer_or_boundary = True
                break
        key = "answer_or_boundary" if answer_or_boundary else "question_only"
        subsets[key].add(item["id"])
    return subsets, data


def option_texts(item: dict[str, Any]) -> list[str]:
    """Return the four option texts without the 'A. ' labels."""
    out = []
    for choice in item["choices"]:
        out.append(choice[3:] if choice[1:3] == ". " else choice)
    return out


def baseline_predictions(items: list[dict[str, Any]], kind: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        texts = option_texts(item)
        if kind in {"A", "B", "C", "D"}:
            out[item["id"]] = kind
        elif kind == "first":
            out[item["id"]] = "A"
        elif kind == "longest":
            lens = [len(t) for t in texts]
            out[item["id"]] = "ABCD"[max(range(4), key=lambda i: lens[i])]
        elif kind == "shortest":
            lens = [len(t) for t in texts]
            out[item["id"]] = "ABCD"[min(range(4), key=lambda i: lens[i])]
        else:
            raise ValueError(f"unknown baseline: {kind}")
    return out


def _dedupe(
    recs: list[dict[str, Any]],
    protocol: str,
    eval_split: str,
    benchmark_sha256: str | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Keep one record per (model, item), preferring the last complete record."""
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    for rec in recs:
        if rec.get("protocol") != protocol:
            continue
        if eval_split and rec.get("eval_split") != eval_split:
            continue
        if benchmark_sha256 and rec.get("benchmark_sha256") != benchmark_sha256:
            continue
        key = (rec.get("model", ""), rec.get("item_id", ""))
        seen[key] = rec
    return seen


def summarize(
    recs: list[dict[str, Any]],
    items: list[dict[str, Any]],
    protocol: str,
    pred_key: str,
    eval_split: str,
    benchmark_sha256: str | None = None,
    contamination_subsets: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
    cat_by_id = {i["id"]: i["category"] for i in items}
    seen = _dedupe(recs, protocol, eval_split, benchmark_sha256)
    model_names = sorted({model for model, _ in seen})

    results: dict[str, Any] = {}
    for model in model_names:
        rows = [r for (m, _), r in seen.items() if m == model]
        rows.sort(key=lambda r: r["item_id"])
        correct = [
            r.get(pred_key, r.get("pred")) == r["answer"]
            for r in rows
        ]
        n = len(rows)
        k = sum(correct)
        lo, hi = wilson(k, n)

        per_letter = []
        for letter in "ABCD":
            idx = [i for i, r in enumerate(rows) if r["answer"] == letter]
            if idx:
                per_letter.append(sum(correct[i] for i in idx) / len(idx))
        balanced = sum(per_letter) / len(per_letter) if per_letter else 0.0

        cats: dict[str, Any] = {}
        for cat in sorted({r["category"] for r in rows}):
            crows = [r for r in rows if r["category"] == cat]
            ck = sum(r.get(pred_key, r.get("pred")) == r["answer"] for r in crows)
            clo, chi = wilson(ck, len(crows))
            cats[cat] = {
                "n": len(crows),
                "correct": ck,
                "accuracy": ck / len(crows),
                "ci95": [clo, chi],
            }
        macro = sum(c["accuracy"] for c in cats.values()) / len(cats) if cats else 0.0

        per_split: dict[str, Any] = {}
        for split in sorted({r.get("split", "") for r in rows}):
            srows = [r for r in rows if r.get("split", "") == split]
            sk = sum(r.get(pred_key, r.get("pred")) == r["answer"] for r in srows)
            slo, shi = wilson(sk, len(srows))
            per_split[split] = {
                "n": len(srows),
                "correct": sk,
                "accuracy": sk / len(srows) if srows else 0.0,
                "ci95": [slo, shi],
            }

        per_contamination: dict[str, Any] = {}
        if contamination_subsets:
            for name in ("none", "question_only", "answer_or_boundary"):
                ids = contamination_subsets.get(name, set())
                crows = [r for r in rows if r["item_id"] in ids]
                if not crows:
                    continue
                ck = sum(r.get(pred_key, r.get("pred")) == r["answer"] for r in crows)
                clo, chi = wilson(ck, len(crows))
                per_contamination[name] = {
                    "n": len(crows),
                    "correct": ck,
                    "accuracy": ck / len(crows) if crows else 0.0,
                    "ci95": [clo, chi],
                }

        preds = [r.get(pred_key, r.get("pred")) for r in rows]
        pred_distribution = {letter: preds.count(letter) for letter in ["A", "B", "C", "D", "?"]}
        parse_failures = pred_distribution["?"]
        answer_distribution = {
            letter: sum(1 for r in rows if r["answer"] == letter) for letter in "ABCD"
        }

        results[model] = {
            "n": n,
            "correct": k,
            "accuracy": k / n if n else 0.0,
            "ci95": [lo, hi],
            "macro_accuracy": macro,
            "balanced_accuracy": balanced,
            "per_category": cats,
            "per_split": per_split,
            "per_contamination": per_contamination,
            "pred_distribution": pred_distribution,
            "answer_distribution": answer_distribution,
            "parse_failures": parse_failures,
            "parse_failure_rate": parse_failures / n if n else 0.0,
            "correct_by_item": {r["item_id"]: (r.get(pred_key, r.get("pred")) == r["answer"]) for r in rows},
            "pred_by_item": {r["item_id"]: r.get(pred_key, r.get("pred")) for r in rows},
        }

    # Baselines on the same item set as the first model. All models in a complete
    # run share the same test split, so this is also the common comparison set.
    first_model = model_names[0] if model_names else None
    if first_model:
        item_ids = set(results[first_model]["correct_by_item"])
        base_items = [i for i in items if i["id"] in item_ids]
        baselines: dict[str, Any] = {}
        for kind in ["A", "B", "C", "D", "first", "longest", "shortest"]:
            preds = baseline_predictions(base_items, kind)
            ck = sum(preds[i["id"]] == i["answer"] for i in base_items)
            lo, hi = wilson(ck, len(base_items))
            baselines[kind] = {
                "n": len(base_items),
                "correct": ck,
                "accuracy": ck / len(base_items) if base_items else 0.0,
                "ci95": [lo, hi],
                "correct_by_item": {
                    i["id"]: (preds[i["id"]] == i["answer"]) for i in base_items
                },
            }
        baselines["random_uniform"] = random_baseline(len(base_items))
        results["_baselines"] = baselines

    # McNemar against the base model and against the strongest baseline.
    if "base" in results:
        base_correct = results["base"]["correct_by_item"]
        for model, res in results.items():
            if model.startswith("_") or model == "base":
                continue
            common = sorted(set(base_correct) & set(res["correct_by_item"]))
            res["mcnemar_vs_base"] = mcnemar_exact(
                [res["correct_by_item"][i] for i in common],
                [base_correct[i] for i in common],
            )

    if "_baselines" in results:
        baselines = results["_baselines"]
        for model, res in results.items():
            if model.startswith("_"):
                continue
            model_ids = set(res["correct_by_item"])
            best_kind = None
            best_acc = -1.0
            for kind, b in baselines.items():
                if "correct_by_item" not in b:
                    continue
                common = model_ids & set(b["correct_by_item"])
                if not common:
                    continue
                acc = sum(b["correct_by_item"][i] for i in common) / len(common)
                if acc > best_acc:
                    best_acc = acc
                    best_kind = kind
            if best_kind is None:
                continue
            b = baselines[best_kind]
            common = sorted(model_ids & set(b["correct_by_item"]))
            res["mcnemar_vs_best_baseline"] = {
                "baseline": best_kind,
                **mcnemar_exact(
                    [res["correct_by_item"][i] for i in common],
                    [b["correct_by_item"][i] for i in common],
                ),
            }

    # Holm correction across categories for the best non-base model.
    candidates = [m for m in results if not m.startswith("_") and m != "base"]
    if candidates:
        best = max(candidates, key=lambda m: results[m]["accuracy"])
        pvals: list[tuple[str, float]] = []
        if "base" in results:
            base_correct = results["base"]["correct_by_item"]
            for cat in results[best]["per_category"]:
                common = sorted(
                    i
                    for i in results[best]["correct_by_item"]
                    if i in base_correct and cat_by_id.get(i) == cat
                )
                if common:
                    p = mcnemar_exact(
                        [results[best]["correct_by_item"][i] for i in common],
                        [base_correct[i] for i in common],
                    )["p"]
                    pvals.append((cat, p))
        results[best]["holm_vs_base_by_category"] = holm(pvals)

    return {
        "protocol": protocol,
        "pred_key": pred_key,
        "eval_split": eval_split,
        "benchmark_sha256": benchmark_sha256,
        "record_benchmark_hashes": sorted(
            {r.get("benchmark_sha256") for r in recs if r.get("benchmark_sha256")}
        ),
        "results": results,
    }


def fmt_pct(x: float) -> str:
    return f"{100*x:.1f}%"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument(
        "--benchmark",
        default=str(Path(__file__).resolve().parents[1] / "benchmark" / "maritime_bench_v1.json"),
    )
    ap.add_argument("--output", default=None)
    ap.add_argument("--eval-split", default="test")
    ap.add_argument("--protocols", default="likelihood,generation")
    ap.add_argument("--pred-key", default="pred")
    ap.add_argument(
        "--contamination",
        default=None,
        help="optional n-gram audit JSON; adds a descriptive overlap/non-overlap split",
    )
    args = ap.parse_args(argv)

    benchmark_bytes = Path(args.benchmark).read_bytes()
    benchmark_sha256 = hashlib.sha256(benchmark_bytes).hexdigest()
    items = json.loads(benchmark_bytes)["items"]
    contamination_subsets = None
    contamination_meta = None
    if args.contamination:
        contamination_subsets, audit = load_contamination_subsets(
            items, args.contamination, benchmark_sha256
        )
        contamination_meta = {
            "path": args.contamination,
            "ngram": audit.get("ngram"),
            "text_mode": audit.get("text_mode"),
            "subset_sizes": {k: len(v) for k, v in contamination_subsets.items()},
        }
    paths = sorted(Path(args.input_dir).glob("*.jsonl"))
    recs = load_records(paths)
    reports = [
        summarize(
            recs,
            items,
            protocol,
            args.pred_key,
            args.eval_split,
            benchmark_sha256,
            contamination_subsets,
        )
        for protocol in args.protocols.split(",")
    ]

    payload = {
        "input_dir": args.input_dir,
        "benchmark": args.benchmark,
        "benchmark_sha256": benchmark_sha256,
        "contamination": contamination_meta,
        "reports": reports,
    }
    out = Path(args.output) if args.output else Path(args.input_dir) / "analysis.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = ["# Evaluation summary", "", f"Input: `{args.input_dir}`", ""]
    for rep in reports:
        lines += [f"## Protocol: {rep['protocol']}", ""]
        res = rep["results"]
        models = [m for m in res if not m.startswith("_")]
        lines += ["| Model | n | Acc | 95% CI | Macro | Balanced |", "|---|---:|---:|---|---:|---:|"]
        for model in sorted(models, key=lambda m: -res[m]["accuracy"]):
            r = res[model]
            lo, hi = r["ci95"]
            lines.append(
                f"| {model} | {r['n']} | {fmt_pct(r['accuracy'])} | "
                f"{fmt_pct(lo)}–{fmt_pct(hi)} | {fmt_pct(r['macro_accuracy'])} | "
                f"{fmt_pct(r['balanced_accuracy'])} |"
            )
        if "_baselines" in res:
            lines += ["", "### Baselines", "", "| Baseline | Acc | 95% CI |", "|---|---:|---|"]
            for kind, b in res["_baselines"].items():
                acc = b.get("accuracy", b.get("expected_accuracy"))
                lo, hi = b["ci95"]
                lines.append(f"| {kind} | {fmt_pct(acc)} | {fmt_pct(lo)}–{fmt_pct(hi)} |")
        lines += [
            "",
            "### Prediction distribution and parse failures",
            "",
            "| Model | A | B | C | D | ? | Parse failure rate |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for model in sorted(models):
            r = res[model]
            d = r["pred_distribution"]
            lines.append(
                f"| {model} | {d['A']} | {d['B']} | {d['C']} | {d['D']} | {d['?']} | "
                f"{fmt_pct(r['parse_failure_rate'])} |"
            )
        lines += [
            "",
            "### Split-level accuracy",
            "",
            "| Model | Split | n | Acc | 95% CI |",
            "|---|---|---:|---:|---|",
        ]
        for model in sorted(models):
            for split, s in sorted(res[model]["per_split"].items()):
                lo, hi = s["ci95"]
                lines.append(
                    f"| {model} | {split} | {s['n']} | {fmt_pct(s['accuracy'])} | "
                    f"{fmt_pct(lo)}–{fmt_pct(hi)} |"
                )

        if any(res[m].get("per_contamination") for m in models):
            labels = {
                "none": "no verified match",
                "question_only": "question-only match",
                "answer_or_boundary": "answer/boundary match",
            }
            lines += [
                "",
                "### Contamination sensitivity",
                "",
                "| Model | Subset | n | Acc | 95% CI |",
                "|---|---|---:|---:|---|",
            ]
            for model in sorted(models):
                pc = res[model].get("per_contamination", {})
                for name in ("none", "question_only", "answer_or_boundary"):
                    if name not in pc:
                        continue
                    s = pc[name]
                    lo, hi = s["ci95"]
                    lines.append(
                        f"| {model} | {labels[name]} | {s['n']} | "
                        f"{fmt_pct(s['accuracy'])} | {fmt_pct(lo)}–{fmt_pct(hi)} |"
                    )
            if contamination_meta:
                lines += [
                    "",
                    f"Subsets are defined by the {contamination_meta['ngram']}-gram audit "
                    f"(`{contamination_meta['path']}`). The split is descriptive, not a "
                    "causal test of contamination.",
                ]

        lines += [
            "",
            "### Paired McNemar vs base",
            "",
            "| Model | b | c | p |",
            "|---|---:|---:|---:|",
        ]
        for model in sorted(models):
            m = res[model].get("mcnemar_vs_base")
            if m:
                lines.append(f"| {model} | {m['b']} | {m['c']} | {m['p']:.4g} |")

        lines += [
            "",
            "### Paired McNemar vs best baseline",
            "",
            "| Model | Baseline | b | c | p |",
            "|---|---|---:|---:|---:|",
        ]
        for model in sorted(models):
            m = res[model].get("mcnemar_vs_best_baseline")
            if m:
                lines.append(
                    f"| {model} | {m['baseline']} | {m['b']} | {m['c']} | {m['p']:.4g} |"
                )

        best = max(
            (m for m in models if m != "base"),
            key=lambda m: res[m]["accuracy"],
            default=None,
        )
        if best and "holm_vs_base_by_category" in res[best]:
            lines += [
                "",
                f"### Holm-corrected category tests: {best} vs base (exploratory)",
                "",
                "| Category | Holm p |",
                "|---|---:|",
            ]
            for cat, p in sorted(res[best]["holm_vs_base_by_category"].items()):
                lines.append(f"| {cat} | {p:.4g} |")
            lines += [
                "",
                f"`{best}` is the best non-base model by overall accuracy, selected post hoc; "
                "the category tests are exploratory and are not a licence to claim a "
                "category-specific gain.",
            ]
        lines.append("")
    md = "\n".join(lines)
    (Path(args.input_dir) / "analysis.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
