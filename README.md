# SAGE — Smart Advisor for Generative modEls

Predicting LLM token usage and response quality from prompt features, and
estimating the token/cost/quality tradeoff across models for a given prompt.

Goal, for any input prompt:

| Model  | Input Tok | Output Tok | Cost  | Quality   |
| ------ | --------- | ---------- | ----- | --------- |
| Claude | 1800      | 300        | $0.60 | Average   |
| llama3 | 1900      | 350        | $2.50 | Good      |
| Groq   | 2000      | 420        | $5.60 | Excellent |

## Pipeline

Each phase reads the previous phase's output and writes its own; earlier
files are never modified in place.

1. **Raw data** — `dataset/raw_datasets/dataset{1..6}.csv`, one file per
   model: `Model Name, Prompt, Input Tokens, Output Tokens, Quality, Feedback`.
2. **Clean** — `dataset/clean_datasets.py` dedupes, drops rows with
   missing/invalid token counts, standardizes labels -> `dataset/cleaned/*.csv`.
3. **Feature engineering** — `dataset/feature_engineering.py` derives
   structural/semantic features from prompt text (length, code/JSON/markdown
   detection, reasoning/creative/tool-use/RAG heuristics, etc.) ->
   `dataset/cleaned/*_features.csv`.
4. **Merge** — `dataset/merge_datasets.py` joins all per-model feature
   files on prompt text -> `dataset/merged/merged_long.csv` (one row per
   prompt+model) and `merged_wide.csv` (one row per prompt, columns per model).
5. **Targets** — `dataset/prepare_targets.py` splits `merged_long.csv` into
   the two prediction tasks -> `dataset/merged/model_a_token_predictor.csv`
   (X: prompt features + model name, y: input/output tokens) and
   `model_b_quality_predictor.csv` (X: same, y: quality label). Rows for
   `llama3` and `opencode/big-pickle` are excluded here — their logged
   token counts don't track prompt length like every other model, pointing
   to a collection bug rather than real signal.
6. **Train** — three model families trained on the same X/y for comparison:
   - `dataset/train_token_predictor.py` / `train_quality_predictor.py` —
     RandomForest baselines (one-hot encoded categoricals), saved to
     `models/*.joblib`.
   - `models/catboost/train_token_predictor_catboost.py` /
     `train_quality_predictor_catboost.py` — CatBoost variants (native
     categorical handling), saved to `models/catboost/*.cbm`.
   - `models/lightgbm/train_token_predictor_lightgbm.py` /
     `train_quality_predictor_lightgbm.py` — LightGBM variants (native
     categorical handling via pandas `category` dtype), saved to
     `models/lightgbm/*.txt`.

7. **Predict** — `predict.py` (repo root) takes a raw prompt, runs it through
   the same feature engineering, and prints the tokens/cost/quality table
   above for every model seen during training. `--backend rf|catboost|lightgbm`
   picks which trained family to use (default `catboost`). Cost is computed
   from a hardcoded `PRICING` table in `predict.py` (not learned from data,
   since the dataset has no pricing column) -- edit it to match real
   provider rates.

Each training script writes its held-out metrics to
`results/<family>/<task>.json` (MAE/RMSE/R2 for the token predictor,
accuracy/macro-F1/classification report/confusion matrix for the quality
predictor) so the model families can be compared without re-running
training.

Dependencies are managed with [uv](https://github.com/astral-sh/uv) via the
root `pyproject.toml`/`uv.lock`; `uv run python ...` picks up the project's
venv automatically (creating/syncing it on first use).

Run the whole pipeline in order with `script.sh` (repo root):

```bash
./script.sh                        # clean -> features -> merge -> targets -> train all 3 families
./script.sh "your prompt here"      # same, then predict.py --backend {rf,catboost,lightgbm} on the prompt
```

Or run phases individually:

```bash
uv run python dataset/clean_datasets.py
uv run python dataset/feature_engineering.py
uv run python dataset/merge_datasets.py
uv run python dataset/prepare_targets.py
uv run python dataset/train_token_predictor.py
uv run python dataset/train_quality_predictor.py
uv run python models/catboost/train_token_predictor_catboost.py
uv run python models/catboost/train_quality_predictor_catboost.py
uv run python models/lightgbm/train_token_predictor_lightgbm.py
uv run python models/lightgbm/train_quality_predictor_lightgbm.py
uv run python predict.py "your prompt here"
```

## Layout

```
dataset/            pipeline scripts + data at each phase (raw_datasets -> cleaned -> merged)
models/             trained model artifacts (RandomForest .joblib, CatBoost .cbm, LightGBM .txt) + per-family training scripts
results/            held-out eval metrics (MAE/RMSE/R2, accuracy/F1), one subdir per model family (rf/, catboost/, lightgbm/), JSON per task
predict.py           CLI: prompt in -> tokens/cost/quality table out, per trained model
script.sh           runs the full pipeline end-to-end (see above)
pyproject.toml      project deps for uv (catboost, lightgbm, numpy, pandas, scikit-learn, joblib)
prompts/            held-out prompt lists used to generate/test data
experiments/trial1/ early static token-counting + pricing prototype (superseded, own pyproject.toml/requirements.txt)
experiments/trial2/ agent-loop token tracer + response/chat analysis prototype (superseded, own requirements.txt)
```

`experiments/` holds earlier prototypes kept for reference; the active
pipeline is `dataset/` + `models/` + `predict.py`.
