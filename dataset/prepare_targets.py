#!/usr/bin/env python3
"""Phase 4 - Define Targets: split merged_long.csv into the two prediction tasks.

Reads merged/merged_long.csv (one row per prompt+model, with engineered
prompt features) and writes:

  merged/model_a_token_predictor.csv
      X: prompt features + model name
      y: input_tokens, output_tokens          (regression)

  merged/model_b_quality_predictor.csv
      X: prompt features + model name
      y: quality  (Bad / Average / Good / Excellent)   (classification)

Rows with row_ok == False (missing/invalid token counts, already filtered
mostly by clean_datasets.py, kept here as a belt-and-suspenders check) are
dropped from both. Model name is left as a raw categorical column ("model")
-- one-hot/ordinal encoding is a modeling-step concern, not a target-definition
concern, so it's deliberately not encoded here.

EXCLUDED_MODELS are dropped entirely: llama3 and opencode/big-pickle logged
input_tokens values wildly inconsistent with prompt length (e.g. char_count
156 -> input_tokens 6266 for big-pickle; llama3 averaged ~8x the token/char
ratio of every other model). That points to a collection bug in those two
sources rather than real signal, so they're excluded until re-collected.

Does not modify merged_long.csv.
"""

import csv
from pathlib import Path

HERE = Path(__file__).parent
IN_PATH = HERE / "merged" / "merged_long.csv"
OUT_DIR = HERE / "merged"

EXCLUDED_MODELS = {"llama3", "opencode/big-pickle"}

FEATURE_FIELDS = [
    "char_count", "word_count", "line_count", "sentence_count",
    "unique_words", "avg_word_length", "prompt_depth",
    "has_code", "has_json", "has_markdown", "has_math", "has_xml",
    "reasoning_prompt", "creative_prompt", "tool_usage_prompt", "rag_prompt",
]

X_FIELDS = ["model"] + FEATURE_FIELDS

MODEL_A_FIELDS = ["prompt_raw"] + X_FIELDS + ["input_tokens", "output_tokens"]
MODEL_B_FIELDS = ["prompt_raw"] + X_FIELDS + ["quality"]

QUALITY_ORDER = {"Bad": 0, "Average": 1, "Good": 2, "Excellent": 3}


def main():
    with open(IN_PATH, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = [r for r in reader if r["row_ok"] == "True"]

    excluded = sum(1 for r in rows if r["model"] in EXCLUDED_MODELS)
    rows = [r for r in rows if r["model"] not in EXCLUDED_MODELS]

    dropped = 0
    model_a_rows, model_b_rows = [], []
    for r in rows:
        if r["quality"] not in QUALITY_ORDER:
            dropped += 1
            continue

        base = {"prompt_raw": r["prompt_raw"], "model": r["model"]}
        for f in FEATURE_FIELDS:
            base[f] = r[f]

        model_a_rows.append({**base, "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"]})
        model_b_rows.append({**base, "quality": r["quality"]})

    a_path = OUT_DIR / "model_a_token_predictor.csv"
    with open(a_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MODEL_A_FIELDS)
        writer.writeheader()
        writer.writerows(model_a_rows)

    b_path = OUT_DIR / "model_b_quality_predictor.csv"
    with open(b_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MODEL_B_FIELDS)
        writer.writeheader()
        writer.writerows(model_b_rows)

    print(f"Model A (token predictor):   {len(model_a_rows)} rows -> {a_path}")
    print(f"Model B (quality predictor): {len(model_b_rows)} rows -> {b_path}")
    print(f"Excluded {excluded} rows from {sorted(EXCLUDED_MODELS)}")
    if dropped:
        print(f"Dropped {dropped} rows with unrecognized/missing quality label")

    from collections import Counter
    print("\nQuality label distribution (Model B):")
    dist = Counter(r["quality"] for r in model_b_rows)
    for label in sorted(dist, key=lambda l: QUALITY_ORDER[l]):
        print(f"  {label}: {dist[label]}")

    print("\nModel name distribution (both tasks):")
    dist = Counter(r["model"] for r in model_a_rows)
    for m, n in dist.most_common():
        print(f"  {m}: {n}")


if __name__ == "__main__":
    main()
