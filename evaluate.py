#!/usr/bin/env python3
"""
Maritime SLM Benchmark Evaluation Script
Evaluates language models on the 163-question bilingual maritime benchmark:
- 148 Multiple-Choice Questions (MCQ) across 6 categories
- 15 Open-Ended Questions scored via keyword set recovery
"""

import os
import sys
import json
import math
import re
import argparse
import torch
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Language Models on the Maritime SLM Benchmark")
    parser.add_argument("--model", type=str, required=True, help="Path or HuggingFace ID of the model")
    parser.add_argument("--benchmark", type=str, default="maritime_eval_163.json", help="Path to benchmark JSON")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    parser.add_argument("--output", type=str, default="results.json", help="Path to save evaluation results")
    return parser.parse_args()

def evaluate():
    args = parse_args()
    print(f"Loading benchmark from: {args.benchmark}")
    with open(args.benchmark, "r", encoding="utf-8") as f:
        bench = json.load(f)
    
    all_questions = bench.get("ALL_QUESTIONS", {})
    open_ended = bench.get("OPEN_ENDED", [])
    
    total_mc = sum(len(v) for v in all_questions.values())
    print(f"Benchmark contains: {total_mc} MCQ across {len(all_questions)} categories, and {len(open_ended)} open-ended items.")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading model: {args.model} on {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    dtype = torch.bfloat16 if args.device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16 if args.device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True
    ).to(args.device)
    model.eval()

    results = {
        "model": args.model,
        "mcq_categories": {},
        "mcq_overall": {},
        "open_ended": {}
    }

    # Evaluate MCQ
    print("\nEvaluating Multiple-Choice Questions...")
    total_mc_correct = 0
    total_mc_count = 0

    for cat_name, questions in all_questions.items():
        cat_correct = 0
        cat_count = len(questions)
        for q in questions:
            prompt = q["q"] + "\n\nAnswer with just the letter (A, B, C, or D)."
            messages = [{"role": "user", "content": prompt}]
            if hasattr(tokenizer, "apply_chat_template"):
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                text = prompt
            
            inputs = tokenizer(text, return_tensors="pt").to(args.device)
            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else 0
                )
            out_text = tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            m = re.search(r"\b([A-D])\b", out_text)
            pred = m.group(1) if m else "?"
            if pred == q.get("answer"):
                cat_correct += 1
        
        acc = (cat_correct / cat_count) * 100 if cat_count > 0 else 0.0
        results["mcq_categories"][cat_name] = {
            "correct": cat_correct,
            "total": cat_count,
            "accuracy": round(acc, 2)
        }
        total_mc_correct += cat_correct
        total_mc_count += cat_count
        print(f"  {cat_name:<25}: {cat_correct}/{cat_count} ({acc:.1f}%)")

    overall_acc = (total_mc_correct / total_mc_count) * 100 if total_mc_count > 0 else 0.0
    p = overall_acc / 100.0
    stderr = math.sqrt(p * (1 - p) / total_mc_count) * 100 if total_mc_count > 0 else 0.0
    results["mcq_overall"] = {
        "correct": total_mc_correct,
        "total": total_mc_count,
        "accuracy": round(overall_acc, 2),
        "stderr": round(stderr, 2)
    }
    print(f"\nOverall MCQ Accuracy: {overall_acc:.1f}% (+/- {stderr:.1f}%)")

    # Evaluate Open-Ended
    if open_ended:
        print("\nEvaluating Open-Ended Questions...")
        total_open_score = 0.0
        for item in open_ended:
            prompt = item.get("prompt", item.get("q", ""))
            target_kw = [k.lower().strip() for k in item.get("target_keywords", item.get("keywords", []))]
            messages = [{"role": "user", "content": prompt}]
            if hasattr(tokenizer, "apply_chat_template"):
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                text = prompt
            
            inputs = tokenizer(text, return_tensors="pt").to(args.device)
            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else 0
                )
            out_text = tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).lower()
            recovered = sum(1 for kw in target_kw if kw in out_text)
            score = (recovered / len(target_kw)) * 10.0 if target_kw else 0.0
            total_open_score += score

        mean_open = total_open_score / len(open_ended)
        results["open_ended"] = {
            "mean_score_out_of_10": round(mean_open, 2),
            "num_items": len(open_ended)
        }
        print(f"Mean Open-Ended Quality: {mean_open:.2f} / 10.0")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults successfully written to: {args.output}")

if __name__ == "__main__":
    evaluate()
