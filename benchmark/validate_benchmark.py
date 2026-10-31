#!/usr/bin/env python3
"""Validate Maritime Benchmark v1.0 and print an audit report."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT / "maritime_bench_v1.json"

BANNED_PREFIXES = (
    "all of the above",
    "none of the above",
    "both a and b",
    "both b and c",
    "a and b",
    "b and c",
    "c and d",
)


def fail(errors: list[str], msg: str) -> None:
    errors.append(msg)


def main(path: Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data["items"]
    errors: list[str] = []
    warnings: list[str] = []

    if data.get("language") != "en":
        fail(errors, "language must be 'en'")
    if len(items) != 200:
        fail(errors, f"expected 200 items, found {len(items)}")

    ids = set()
    questions = set()
    for item in items:
        iid = item.get("id", "?")
        for field in ("id", "category", "question", "choices", "answer",
                      "correct_text", "source", "source_url", "evidence",
                      "split", "difficulty", "eval_split", "meta"):
            if field not in item:
                fail(errors, f"{iid}: missing {field}")
        if iid in ids:
            fail(errors, f"duplicate id {iid}")
        ids.add(iid)

        q = re.sub(r"\s+", " ", item["question"].strip().lower())
        if q in questions:
            fail(errors, f"duplicate question {iid}")
        questions.add(q)

        choices = item["choices"]
        if len(choices) != 4:
            fail(errors, f"{iid}: {len(choices)} choices")
            continue
        if len({c.strip().lower() for c in choices}) != 4:
            fail(errors, f"{iid}: duplicate choices")
        for i, c in enumerate(choices):
            if not c.startswith(f"{'ABCD'[i]}. "):
                fail(errors, f"{iid}: choice {i} missing label")
            low = c.lower()
            if any(low.startswith(p) for p in BANNED_PREFIXES):
                fail(errors, f"{iid}: order-dependent choice: {c!r}")
            n_words = len(c[3:].split())
            if n_words < 2:
                fail(errors, f"{iid}: choice {i} too short")
            if n_words > 25:
                fail(errors, f"{iid}: choice {i} too long ({n_words} words)")
        if item["answer"] not in "ABCD":
            fail(errors, f"{iid}: bad answer {item['answer']}")
        else:
            idx = "ABCD".index(item["answer"])
            option_text = choices[idx][3:].strip()
            if option_text != item["correct_text"].strip():
                fail(errors, f"{iid}: correct_text not exactly at answer position")
        if item["split"] not in {"public_core", "port_local"}:
            fail(errors, f"{iid}: bad split")
        if item["eval_split"] not in {"dev", "test"}:
            fail(errors, f"{iid}: bad eval_split")
        if item["difficulty"] not in {"easy", "medium", "hard"}:
            fail(errors, f"{iid}: bad difficulty")
        if not item["source_url"].startswith(("http://", "https://")):
            fail(errors, f"{iid}: bad source_url")
        if len(item["evidence"].split()) < 4:
            warnings.append(f"{iid}: short evidence")

    key_counts = Counter(i["answer"] for i in items)
    key_by_cat = {
        c: dict(Counter(i["answer"] for i in items if i["category"] == c))
        for c in sorted({i["category"] for i in items})
    }
    longest_unique = sum(1 for i in items if i["meta"].get("correct_is_unique_longest"))
    longest_any = sum(1 for i in items if i["meta"].get("correct_is_longest"))
    split_counts = Counter(i["split"] for i in items)
    eval_counts = Counter(i["eval_split"] for i in items)
    cat_counts = Counter(i["category"] for i in items)

    if abs(key_counts["A"] - key_counts["C"]) > 2:
        fail(errors, f"key balance A/C too wide: {key_counts}")
    if abs(key_counts["B"] - key_counts["D"]) > 2:
        fail(errors, f"key balance B/D too wide: {key_counts}")
    if longest_unique / len(items) > 0.35:
        fail(errors, f"correct-unique-longest rate {longest_unique/len(items):.1%} > 35%")
    if eval_counts["dev"] != 40:
        fail(errors, f"dev split must be 40, found {eval_counts['dev']}")
    if split_counts["port_local"] < 10:
        fail(errors, f"port_local must have at least 10 items, found {split_counts['port_local']}")

    report = {
        "file": str(path),
        "version": data.get("version"),
        "total": len(items),
        "counts": dict(cat_counts),
        "splits": dict(split_counts),
        "eval_splits": dict(eval_counts),
        "key_counts": dict(key_counts),
        "key_by_category": key_by_cat,
        "correct_unique_longest": longest_unique,
        "correct_unique_longest_pct": round(100 * longest_unique / len(items), 1),
        "correct_among_longest": longest_any,
        "correct_among_longest_pct": round(100 * longest_any / len(items), 1),
        "errors": errors,
        "warnings": warnings,
    }
    print(json.dumps(report, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    raise SystemExit(main(p))
