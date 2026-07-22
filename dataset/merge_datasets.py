#!/usr/bin/env python3
"""Merge per-model feature CSVs in cleaned/ into unified long/wide comparison tables.

Reads cleaned/dataset1_features.csv..dataset6_features.csv (one model each,
output of clean_datasets.py + feature_engineering.py: the original schema
Model Name, Prompt, Input Tokens, Output Tokens, Quality, Feedback plus the
16 engineered prompt features), joins them on normalized prompt text, and
writes:

  merged/merged_long.csv  - one row per (prompt, model), with per-row features
  merged/merged_wide.csv  - one row per prompt, columns per model, plus one
                            set of prompt-level feature columns (features
                            depend only on prompt text, not on the model)

Does not modify or delete the source cleaned/*_features.csv files.
"""

import csv
import re
from pathlib import Path

HERE = Path(__file__).parent
IN_DIR = HERE / "cleaned"
OUT_DIR = HERE / "merged"

SOURCE_FILES = [
    "dataset1_features.csv",
    "dataset2_features.csv",
    "dataset3_features.csv",
    "dataset4_features.csv",
    "dataset5_features.csv",
    "dataset6_features.csv",
]

FEATURE_FIELDS = [
    "char_count", "word_count", "line_count", "sentence_count",
    "unique_words", "avg_word_length", "prompt_depth",
    "has_code", "has_json", "has_markdown", "has_math", "has_xml",
    "reasoning_prompt", "creative_prompt", "tool_usage_prompt", "rag_prompt",
]

# Fill in per-model $/1M-token pricing here when known. Left blank
# deliberately for models where I don't have a verified rate card.
PRICING_PER_MILLION = {
    # "claude-opus-4-8": {"input": None, "output": None},
}


def normalize_prompt(p: str) -> str:
    """Join key: lowercase, collapse whitespace, strip leading list markers."""
    p = p.strip().lower()
    p = re.sub(r"^\d+\.\s*", "", p)  # strip leading "1. " style numbering
    p = re.sub(r"\s+", " ", p)
    return p


def load_dataset(filename: str) -> list[dict]:
    path = IN_DIR / filename
    rows = []
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            input_tokens = r["Input Tokens"].strip()
            output_tokens = r["Output Tokens"].strip()
            ok = input_tokens.isdigit() and output_tokens.isdigit()
            row = {
                "source_file": filename,
                "model": r["Model Name"].strip(),
                "prompt_raw": r["Prompt"].strip(),
                "prompt_key": normalize_prompt(r["Prompt"]),
                "input_tokens": int(input_tokens) if ok else None,
                "output_tokens": int(output_tokens) if ok else None,
                "quality": r["Quality"].strip(),
                "feedback": r["Feedback"].strip(),
                "row_ok": ok,
            }
            for feat in FEATURE_FIELDS:
                row[feat] = r.get(feat, "")
            rows.append(row)
    return rows


def build_long(all_rows: list[dict]) -> list[dict]:
    return all_rows


def build_wide(all_rows: list[dict]) -> tuple[list[str], list[dict]]:
    models = sorted(set(r["model"] for r in all_rows))

    by_prompt: dict[str, dict] = {}
    for r in all_rows:
        key = r["prompt_key"]
        is_new = key not in by_prompt
        entry = by_prompt.setdefault(key, {"prompt_key": key, "prompt_raw": r["prompt_raw"]})
        # keep the longest raw prompt text as the canonical display version,
        # and its features (features depend on exact prompt text, e.g. a
        # leading "1. " numbering prefix changes char/word counts slightly)
        if is_new or len(r["prompt_raw"]) > len(entry["prompt_raw"]):
            entry["prompt_raw"] = r["prompt_raw"]
            for feat in FEATURE_FIELDS:
                entry[feat] = r[feat]

        model = r["model"]
        # if a model has multiple rows for the same prompt (dupes), keep the first
        if f"{model}::input_tokens" in entry:
            continue
        entry[f"{model}::input_tokens"] = r["input_tokens"]
        entry[f"{model}::output_tokens"] = r["output_tokens"]
        entry[f"{model}::quality"] = r["quality"]
        entry[f"{model}::row_ok"] = r["row_ok"]

    fieldnames = ["prompt_key", "prompt_raw", "num_models_covered"] + FEATURE_FIELDS
    for m in models:
        fieldnames += [f"{m}::input_tokens", f"{m}::output_tokens", f"{m}::quality", f"{m}::row_ok"]

    wide_rows = []
    for key, entry in by_prompt.items():
        covered = sum(1 for m in models if f"{m}::input_tokens" in entry)
        entry["num_models_covered"] = covered
        for m in models:
            for suffix in ("input_tokens", "output_tokens", "quality", "row_ok"):
                entry.setdefault(f"{m}::{suffix}", "")
        wide_rows.append(entry)

    # most-covered prompts first (best rows for cross-model comparison)
    wide_rows.sort(key=lambda e: (-e["num_models_covered"], e["prompt_key"]))
    return fieldnames, wide_rows


def main():
    OUT_DIR.mkdir(exist_ok=True)

    all_rows = []
    for f in SOURCE_FILES:
        rows = load_dataset(f)
        all_rows.extend(rows)
        bad = sum(1 for r in rows if not r["row_ok"])
        print(f"{f}: {len(rows)} rows loaded ({bad} with missing token counts)")

    long_path = OUT_DIR / "merged_long.csv"
    with open(long_path, "w", newline="", encoding="utf-8") as fh:
        fieldnames = (["source_file", "model", "prompt_raw", "prompt_key",
                       "input_tokens", "output_tokens", "quality", "feedback", "row_ok"]
                      + FEATURE_FIELDS)
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote {long_path} ({len(all_rows)} rows)")

    fieldnames, wide_rows = build_wide(all_rows)
    wide_path = OUT_DIR / "merged_wide.csv"
    with open(wide_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(wide_rows)
    print(f"Wrote {wide_path} ({len(wide_rows)} unique prompts)")

    full_coverage = sum(1 for r in wide_rows if r["num_models_covered"] == 6)
    print(f"\nPrompts covered by all 6 models: {full_coverage}")
    print("Coverage distribution:")
    from collections import Counter
    dist = Counter(r["num_models_covered"] for r in wide_rows)
    for n in sorted(dist, reverse=True):
        print(f"  {n} models: {dist[n]} prompts")


if __name__ == "__main__":
    main()
