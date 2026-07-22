#!/usr/bin/env python3
"""Phase 5 (Model B, LightGBM variant) - Train the quality predictor (classification).

Same X/y as dataset/train_quality_predictor.py:
  X: prompt features + model name (raw categorical, not one-hot encoded --
     LightGBM handles categorical columns natively via pandas 'category'
     dtype, same idea as CatBoost's cat_features)
  y: quality (Bad / Average / Good / Excellent)

Reads merged/model_b_quality_predictor.csv, does the same stratified (by
quality) train/test split as the RandomForest/CatBoost variants so
accuracy/macro-F1 are directly comparable, and saves the fitted model.
Classes are imbalanced (Excellent ~6% vs Average ~42%), handled via
LightGBM's class_weight="balanced".
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "dataset" / "merged" / "model_b_quality_predictor.csv"
MODEL_DIR = REPO_ROOT / "models" / "lightgbm"
RESULTS_PATH = REPO_ROOT / "results" / "lightgbm" / "quality_predictor.json"

NUMERIC_FEATURES = [
    "char_count", "word_count", "line_count", "sentence_count",
    "unique_words", "avg_word_length", "prompt_depth",
]
BOOL_FEATURES = [
    "has_code", "has_json", "has_markdown", "has_math", "has_xml",
    "reasoning_prompt", "creative_prompt", "tool_usage_prompt", "rag_prompt",
]
CATEGORICAL_FEATURES = ["model"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES + BOOL_FEATURES
TARGET = "quality"
CLASS_ORDER = ["Bad", "Average", "Good", "Excellent"]


def build_model() -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )


def main():
    df = pd.read_csv(DATA_PATH)
    for c in BOOL_FEATURES:
        df[c] = df[c].astype(bool).astype(int)
    for c in CATEGORICAL_FEATURES:
        df[c] = df[c].astype("category")

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_model()
    model.fit(X_train, y_train, categorical_feature=CATEGORICAL_FEATURES)

    preds = model.predict(X_test)

    print(f"Train rows: {len(X_train)}  Test rows: {len(X_test)}\n")
    acc = (preds == y_test.values).mean()
    macro_f1 = f1_score(y_test, preds, average="macro")
    print(f"Accuracy: {acc:.3f}   Macro F1: {macro_f1:.3f}\n")
    report = classification_report(y_test, preds, labels=CLASS_ORDER, zero_division=0, output_dict=True)
    print(classification_report(y_test, preds, labels=CLASS_ORDER, zero_division=0))

    print("Confusion matrix (rows=actual, cols=predicted), order:", CLASS_ORDER)
    cm = confusion_matrix(y_test, preds, labels=CLASS_ORDER)
    for label, row in zip(CLASS_ORDER, cm):
        print(f"  {label:<10}{row}")

    print("\nTop features:")
    importances = model.feature_importances_
    top = sorted(zip(FEATURES, importances), key=lambda t: -t[1])[:8]
    for name, imp in top:
        print(f"  {name:<35}{imp:.2f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODEL_DIR / "quality_predictor.txt"
    model.booster_.save_model(str(out_path))

    # LGBMClassifier label-encodes y internally (alphabetical order), and
    # that order is what the raw Booster's predict() returns columns in --
    # save it alongside the model so predict.py can decode class indices
    # back to labels without needing the sklearn wrapper.
    classes_path = MODEL_DIR / "quality_predictor_classes.json"
    classes_path.write_text(json.dumps(list(model.classes_)))
    print(f"\nSaved model to {out_path}")
    print(f"Saved class order to {classes_path}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({
        "model_type": "lightgbm",
        "task": "quality_predictor",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "classification_report": report,
        "confusion_matrix": {"labels": CLASS_ORDER, "matrix": cm.tolist()},
        "top_features": [{"name": n, "importance": float(i)} for n, i in top],
    }, indent=2))
    print(f"Saved metrics to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
