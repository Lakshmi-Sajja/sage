#!/usr/bin/env python3
"""Clean per-model CSVs in dataset/ (dataset1.csv..dataset6.csv).

For each source file, writes a cleaned copy to cleaned/<name>.csv after:
  - Removing exact duplicate rows
  - Dropping rows with missing/invalid token counts (can't be imputed)
  - Standardizing quality labels (title case, trimmed)
  - Standardizing model names (trimmed, consistent form)
  - Ensuring token counts are valid positive integers

Does not modify the source dataset*.csv files. Prints a per-file summary
of what was removed/changed.
"""

import csv
from pathlib import Path

HERE = Path(__file__).parent
OUT_DIR = HERE / "cleaned"

SOURCE_FILES = [
    "dataset1.csv",
    "dataset2.csv",
    "dataset3.csv",
    "dataset4.csv",
    "dataset5.csv",
    "dataset6.csv",
]

FIELDNAMES = ["Model Name", "Prompt", "Input Tokens", "Output Tokens", "Quality", "Feedback"]

VALID_QUALITY = {"bad", "average", "good", "excellent"}

# Canonical spelling for model names seen across files (trim whitespace only;
# no case variants existed in the source data, but this map lets future
# variants be normalized in one place).
MODEL_NAME_MAP = {
    "groq-llama-3.3-70b-versatile": "Groq-llama-3.3-70b-versatile",
    "deepseek-reasoner": "deepseek-reasoner",
    "llama3": "llama3",
    "deepseek v4 flash": "DeepSeek V4 Flash",
    "opencode/big-pickle": "opencode/big-pickle",
    "claude-opus-4-8": "claude-opus-4-8",
}


def standardize_quality(raw: str) -> str | None:
    q = raw.strip().lower()
    if q not in VALID_QUALITY:
        return None
    return q.capitalize()


def standardize_model(raw: str) -> str:
    m = raw.strip()
    return MODEL_NAME_MAP.get(m.lower(), m)


def clean_file(filename: str) -> dict:
    path = HERE / filename
    stats = {
        "total": 0,
        "dup_removed": 0,
        "bad_tokens_removed": 0,
        "bad_quality_removed": 0,
        "feedback_filled": 0,
        "kept": 0,
    }

    seen_rows = set()
    cleaned_rows = []

    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            stats["total"] += 1

            model = standardize_model(row["Model Name"])
            prompt = row["Prompt"].strip()
            input_tokens = row["Input Tokens"].strip()
            output_tokens = row["Output Tokens"].strip()
            quality_raw = row["Quality"]
            feedback = row["Feedback"].strip()

            # Exact duplicate row check (on standardized values)
            dedupe_key = (model, prompt, input_tokens, output_tokens, quality_raw.strip(), feedback)
            if dedupe_key in seen_rows:
                stats["dup_removed"] += 1
                continue
            seen_rows.add(dedupe_key)

            # Valid, positive integer token counts required
            if not (input_tokens.isdigit() and output_tokens.isdigit()):
                stats["bad_tokens_removed"] += 1
                continue
            if int(input_tokens) <= 0 or int(output_tokens) <= 0:
                stats["bad_tokens_removed"] += 1
                continue

            quality = standardize_quality(quality_raw)
            if quality is None:
                stats["bad_quality_removed"] += 1
                continue

            if not feedback:
                feedback = "N/A"
                stats["feedback_filled"] += 1

            cleaned_rows.append({
                "Model Name": model,
                "Prompt": prompt,
                "Input Tokens": input_tokens,
                "Output Tokens": output_tokens,
                "Quality": quality,
                "Feedback": feedback,
            })

    stats["kept"] = len(cleaned_rows)

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / filename
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(cleaned_rows)

    return stats


def main():
    grand_total = 0
    grand_kept = 0
    for f in SOURCE_FILES:
        stats = clean_file(f)
        grand_total += stats["total"]
        grand_kept += stats["kept"]
        print(
            f"{f}: {stats['total']} rows -> {stats['kept']} kept "
            f"(dupes removed: {stats['dup_removed']}, "
            f"bad token rows removed: {stats['bad_tokens_removed']}, "
            f"bad quality removed: {stats['bad_quality_removed']}, "
            f"feedback filled: {stats['feedback_filled']})"
        )
    print(f"\nTotal: {grand_total} rows -> {grand_kept} kept, "
          f"{grand_total - grand_kept} removed. Cleaned files written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
