#!/usr/bin/env python3
"""Phase 5 (Model B) - Train the quality predictor (classification).

X: prompt features + model name (one-hot encoded)
y: quality (Bad / Average / Good / Excellent)

Reads merged/model_b_quality_predictor.csv, does a stratified (by quality)
train/test split, fits a class-weighted RandomForestClassifier (the classes
are imbalanced -- Excellent is ~6% of rows vs. ~42% for Average), reports
accuracy/macro-F1/per-class metrics, and saves the fitted pipeline.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

HERE = Path(__file__).parent
DATA_PATH = HERE / "merged" / "model_b_quality_predictor.csv"
MODEL_DIR = HERE.parent / "models"

NUMERIC_FEATURES = [
    "char_count", "word_count", "line_count", "sentence_count",
    "unique_words", "avg_word_length", "prompt_depth",
]
BOOL_FEATURES = [
    "has_code", "has_json", "has_markdown", "has_math", "has_xml",
    "reasoning_prompt", "creative_prompt", "tool_usage_prompt", "rag_prompt",
]
CATEGORICAL_FEATURES = ["model"]
TARGET = "quality"
CLASS_ORDER = ["Bad", "Average", "Good", "Excellent"]


def build_pipeline() -> Pipeline:
    preprocess = ColumnTransformer([
        ("model_ohe", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("numeric", "passthrough", NUMERIC_FEATURES + BOOL_FEATURES),
    ])
    classifier = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", preprocess), ("model", classifier)])


def main():
    df = pd.read_csv(DATA_PATH)
    for c in BOOL_FEATURES:
        df[c] = df[c].astype(bool).astype(int)

    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES + BOOL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)

    print(f"Train rows: {len(X_train)}  Test rows: {len(X_test)}\n")
    acc = (preds == y_test).mean()
    macro_f1 = f1_score(y_test, preds, average="macro")
    print(f"Accuracy: {acc:.3f}   Macro F1: {macro_f1:.3f}\n")
    print(classification_report(y_test, preds, labels=CLASS_ORDER, zero_division=0))

    print("Confusion matrix (rows=actual, cols=predicted), order:", CLASS_ORDER)
    cm = confusion_matrix(y_test, preds, labels=CLASS_ORDER)
    for label, row in zip(CLASS_ORDER, cm):
        print(f"  {label:<10}{row}")

    feature_names = (
        list(pipeline.named_steps["preprocess"].named_transformers_["model_ohe"].get_feature_names_out(CATEGORICAL_FEATURES))
        + NUMERIC_FEATURES + BOOL_FEATURES
    )
    importances = pipeline.named_steps["model"].feature_importances_
    top = sorted(zip(feature_names, importances), key=lambda t: -t[1])[:8]
    print("\nTop features:")
    for name, imp in top:
        print(f"  {name:<35}{imp:.3f}")

    MODEL_DIR.mkdir(exist_ok=True)
    out_path = MODEL_DIR / "quality_predictor.joblib"
    joblib.dump(pipeline, out_path)
    print(f"\nSaved pipeline to {out_path}")


if __name__ == "__main__":
    main()
