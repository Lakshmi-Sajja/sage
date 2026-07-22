#!/usr/bin/env python3
"""Phase 5 (Model A) - Train the token predictor (regression).

X: prompt features + model name (one-hot encoded)
y: input_tokens, output_tokens (multi-output regression)

Reads merged/model_a_token_predictor.csv, does a stratified (by model)
train/test split, fits a RandomForestRegressor, reports MAE/RMSE/R^2 per
target on the held-out test set, and saves the fitted pipeline.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

HERE = Path(__file__).parent
DATA_PATH = HERE / "merged" / "model_a_token_predictor.csv"
MODEL_DIR = HERE.parent / "models"
RESULTS_PATH = HERE.parent / "results" / "token_predictor_rf.json"

NUMERIC_FEATURES = [
    "char_count", "word_count", "line_count", "sentence_count",
    "unique_words", "avg_word_length", "prompt_depth",
]
BOOL_FEATURES = [
    "has_code", "has_json", "has_markdown", "has_math", "has_xml",
    "reasoning_prompt", "creative_prompt", "tool_usage_prompt", "rag_prompt",
]
CATEGORICAL_FEATURES = ["model"]
TARGETS = ["input_tokens", "output_tokens"]


def build_pipeline() -> Pipeline:
    preprocess = ColumnTransformer([
        ("model_ohe", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("numeric", "passthrough", NUMERIC_FEATURES + BOOL_FEATURES),
    ])
    regressor = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", preprocess), ("model", regressor)])


def main():
    df = pd.read_csv(DATA_PATH)
    for c in BOOL_FEATURES:
        df[c] = df[c].astype(bool).astype(int)

    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES + BOOL_FEATURES]
    y = df[TARGETS]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=df["model"]
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    preds = np.clip(preds, a_min=0, a_max=None)  # token counts can't be negative

    print(f"Train rows: {len(X_train)}  Test rows: {len(X_test)}\n")
    print(f"{'Target':<15}{'MAE':>10}{'RMSE':>10}{'R2':>10}")
    metrics = {}
    for i, target in enumerate(TARGETS):
        mae = mean_absolute_error(y_test[target], preds[:, i])
        rmse = root_mean_squared_error(y_test[target], preds[:, i])
        r2 = r2_score(y_test[target], preds[:, i])
        print(f"{target:<15}{mae:>10.2f}{rmse:>10.2f}{r2:>10.3f}")
        metrics[target] = {"mae": mae, "rmse": rmse, "r2": r2}

    # Feature importances (averaged across the two targets since
    # RandomForestRegressor's multi-output impurity importances are shared)
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
    out_path = MODEL_DIR / "token_predictor.joblib"
    joblib.dump(pipeline, out_path)
    print(f"\nSaved pipeline to {out_path}")

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({
        "model_type": "random_forest",
        "task": "token_predictor",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "metrics": metrics,
        "top_features": [{"name": n, "importance": float(i)} for n, i in top],
    }, indent=2))
    print(f"Saved metrics to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
