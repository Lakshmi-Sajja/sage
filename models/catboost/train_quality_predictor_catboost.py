#!/usr/bin/env python3
"""Phase 5 (Model B, CatBoost variant) - Train the quality predictor (classification).

Same X/y as dataset/train_quality_predictor.py:
  X: prompt features + model name (raw categorical, not one-hot encoded --
     CatBoost handles categorical columns natively via cat_features)
  y: quality (Bad / Average / Good / Excellent)

Reads merged/model_b_quality_predictor.csv, does the same stratified (by
quality) train/test split as the RandomForest baseline so accuracy/macro-F1
are directly comparable, and saves the fitted model. Classes are imbalanced
(Excellent ~6% vs Average ~42%), handled via CatBoost's auto_class_weights.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "dataset" / "merged" / "model_b_quality_predictor.csv"
MODEL_DIR = REPO_ROOT / "models" / "catboost"
RESULTS_PATH = REPO_ROOT / "results" / "quality_predictor_catboost.json"

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


def build_model() -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="MultiClass",
        auto_class_weights="Balanced",
        random_seed=42,
        cat_features=CATEGORICAL_FEATURES,
        verbose=False,
    )


def main():
    df = pd.read_csv(DATA_PATH)
    for c in BOOL_FEATURES:
        df[c] = df[c].astype(bool).astype(int)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_model()
    model.fit(X_train, y_train)

    preds = model.predict(X_test).ravel()

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
    importances = model.get_feature_importance()
    top = sorted(zip(FEATURES, importances), key=lambda t: -t[1])[:8]
    for name, imp in top:
        print(f"  {name:<35}{imp:.2f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODEL_DIR / "quality_predictor.cbm"
    model.save_model(str(out_path))
    print(f"\nSaved model to {out_path}")

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({
        "model_type": "catboost",
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
