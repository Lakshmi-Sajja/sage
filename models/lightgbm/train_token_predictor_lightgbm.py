#!/usr/bin/env python3
"""Phase 5 (Model A, LightGBM variant) - Train the token predictor (regression).

Same X/y as dataset/train_token_predictor.py:
  X: prompt features + model name (raw categorical, not one-hot encoded --
     LightGBM handles categorical columns natively via pandas 'category'
     dtype, same idea as CatBoost's cat_features)
  y: input_tokens, output_tokens (two independent LGBMRegressor models,
     since LightGBM doesn't support multi-output regression directly)

Reads merged/model_a_token_predictor.csv, does the same stratified (by model)
train/test split as the RandomForest/CatBoost variants so MAE/RMSE/R^2 are
directly comparable, and saves the fitted models.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "dataset" / "merged" / "model_a_token_predictor.csv"
MODEL_DIR = REPO_ROOT / "models" / "lightgbm"
RESULTS_PATH = REPO_ROOT / "results" / "lightgbm" / "token_predictor.json"

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
TARGETS = ["input_tokens", "output_tokens"]


def build_model() -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
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
    y = df[TARGETS]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=df["model"]
    )

    print(f"Train rows: {len(X_train)}  Test rows: {len(X_test)}\n")
    print(f"{'Target':<15}{'MAE':>10}{'RMSE':>10}{'R2':>10}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    models = {}
    metrics = {}
    for target in TARGETS:
        model = build_model()
        model.fit(X_train, y_train[target], categorical_feature=CATEGORICAL_FEATURES)
        preds = np.clip(model.predict(X_test), a_min=0, a_max=None)

        mae = mean_absolute_error(y_test[target], preds)
        rmse = root_mean_squared_error(y_test[target], preds)
        r2 = r2_score(y_test[target], preds)
        print(f"{target:<15}{mae:>10.2f}{rmse:>10.2f}{r2:>10.3f}")
        metrics[target] = {"mae": mae, "rmse": rmse, "r2": r2}

        out_path = MODEL_DIR / f"token_predictor_{target}.txt"
        model.booster_.save_model(str(out_path))
        models[target] = model

    print("\nTop features (input_tokens model):")
    importances = models["input_tokens"].feature_importances_
    top = sorted(zip(FEATURES, importances), key=lambda t: -t[1])[:8]
    for name, imp in top:
        print(f"  {name:<35}{imp:.2f}")

    print(f"\nSaved models to {MODEL_DIR}/token_predictor_<target>.txt")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({
        "model_type": "lightgbm",
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
