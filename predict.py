#!/usr/bin/env python3
"""Predict tokens, cost, and quality for a prompt across all trained models.

Runs the same feature engineering as the training pipeline
(dataset/feature_engineering.py) on the input prompt, feeds it through the
fitted token predictor and quality predictor, and prints one row per model
the pipeline was trained on -- the table described in the README.

Usage:
    python predict.py "Write a poem about the ocean"
    python predict.py --backend rf "Explain how TCP handshakes work"
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "dataset"))
from feature_engineering import extract_features  # noqa: E402

MODEL_DIR = HERE / "models"
CATBOOST_DIR = MODEL_DIR / "catboost"

NUMERIC_FEATURES = [
    "char_count", "word_count", "line_count", "sentence_count",
    "unique_words", "avg_word_length", "prompt_depth",
]
BOOL_FEATURES = [
    "has_code", "has_json", "has_markdown", "has_math", "has_xml",
    "reasoning_prompt", "creative_prompt", "tool_usage_prompt", "rag_prompt",
]
FEATURE_ORDER = NUMERIC_FEATURES + BOOL_FEATURES

# $ per 1M tokens. Placeholder rates for the models present in the training
# data -- edit to match actual provider pricing at the time of use; the
# dataset itself carries no pricing column.
PRICING = {
    "claude-opus-4-8": {"input": 15.00, "output": 75.00},
    "Groq-llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "DeepSeek V4 Flash": {"input": 0.27, "output": 1.10},
    "deepseek-reasoner": {"input": 0.55, "output": 2.19},
}


def build_feature_row(prompt: str) -> dict:
    features = extract_features(prompt)
    row = {f: features[f] for f in NUMERIC_FEATURES}
    for f in BOOL_FEATURES:
        row[f] = int(features[f] == "True")
    return row


def known_models() -> list:
    pipeline = joblib.load(MODEL_DIR / "token_predictor.joblib")
    encoder = pipeline.named_steps["preprocess"].named_transformers_["model_ohe"]
    return list(encoder.categories_[0])


def predict_rf(prompt_row: dict, models: list) -> tuple:
    token_pipeline = joblib.load(MODEL_DIR / "token_predictor.joblib")
    quality_pipeline = joblib.load(MODEL_DIR / "quality_predictor.joblib")

    X = pd.DataFrame([{**prompt_row, "model": m} for m in models])[["model"] + FEATURE_ORDER]

    token_preds = np.clip(token_pipeline.predict(X), a_min=0, a_max=None)
    quality_preds = quality_pipeline.predict(X)

    tokens = {m: (int(round(token_preds[i, 0])), int(round(token_preds[i, 1]))) for i, m in enumerate(models)}
    quality = {m: quality_preds[i] for i, m in enumerate(models)}
    return tokens, quality


def predict_catboost(prompt_row: dict, models: list) -> tuple:
    from catboost import CatBoostClassifier, CatBoostRegressor

    input_model = CatBoostRegressor()
    input_model.load_model(str(CATBOOST_DIR / "token_predictor_input_tokens.cbm"))
    output_model = CatBoostRegressor()
    output_model.load_model(str(CATBOOST_DIR / "token_predictor_output_tokens.cbm"))
    quality_model = CatBoostClassifier()
    quality_model.load_model(str(CATBOOST_DIR / "quality_predictor.cbm"))

    X = pd.DataFrame([{**prompt_row, "model": m} for m in models])[["model"] + FEATURE_ORDER]

    input_preds = np.clip(input_model.predict(X), a_min=0, a_max=None)
    output_preds = np.clip(output_model.predict(X), a_min=0, a_max=None)
    quality_preds = quality_model.predict(X).ravel()

    tokens = {m: (int(round(input_preds[i])), int(round(output_preds[i]))) for i, m in enumerate(models)}
    quality = {m: quality_preds[i] for i, m in enumerate(models)}
    return tokens, quality


def estimate_cost(model: str, input_tokens: int, output_tokens: int):
    rates = PRICING.get(model)
    if rates is None:
        return None
    return (input_tokens / 1_000_000) * rates["input"] + (output_tokens / 1_000_000) * rates["output"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prompt", help="Prompt text to estimate tokens/cost/quality for")
    parser.add_argument(
        "--backend", choices=["catboost", "rf"], default="catboost",
        help="Which trained model family to use (default: catboost)",
    )
    args = parser.parse_args()

    models = known_models()
    prompt_row = build_feature_row(args.prompt)

    predict_fn = predict_rf if args.backend == "rf" else predict_catboost
    tokens, quality = predict_fn(prompt_row, models)

    rows = []
    for m in models:
        in_tok, out_tok = tokens[m]
        cost = estimate_cost(m, in_tok, out_tok)
        rows.append((m, in_tok, out_tok, cost, quality[m]))
    rows.sort(key=lambda r: r[3] if r[3] is not None else float("inf"))

    name_w = max(len(r[0]) for r in rows) + 2
    print(f"{'Model':<{name_w}}{'Input Tok':>10}{'Output Tok':>12}{'Cost':>10}{'Quality':>12}")
    for m, in_tok, out_tok, cost, q in rows:
        cost_str = f"${cost:.4f}" if cost is not None else "n/a"
        print(f"{m:<{name_w}}{in_tok:>10}{out_tok:>12}{cost_str:>10}{q:>12}")


if __name__ == "__main__":
    main()
