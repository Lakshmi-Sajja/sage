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
6. **Train** — two model families trained on the same X/y for comparison:
   - `dataset/train_token_predictor.py` / `train_quality_predictor.py` —
     RandomForest baselines (one-hot encoded categoricals), saved to
     `models/*.joblib`.
   - `models/catboost/train_token_predictor_catboost.py` /
     `train_quality_predictor_catboost.py` — CatBoost variants (native
     categorical handling), saved to `models/catboost/*.cbm`.

7. **Predict** — `predict.py` (repo root) takes a raw prompt, runs it through
   the same feature engineering, and prints the tokens/cost/quality table
   above for every model seen during training. `--backend rf|catboost`
   picks which trained family to use (default `catboost`). Cost is computed
   from a hardcoded `PRICING` table in `predict.py` (not learned from data,
   since the dataset has no pricing column) -- edit it to match real
   provider rates.

Each training script (both RandomForest and CatBoost variants) also writes
its held-out metrics to `results/<task>_<family>.json` (MAE/RMSE/R2 for the
token predictor, accuracy/macro-F1/classification report/confusion matrix
for the quality predictor) so the two model families can be compared without
re-running training.

Run phases in order from `dataset/`:

```bash
python clean_datasets.py
python feature_engineering.py
python merge_datasets.py
python prepare_targets.py
python train_token_predictor.py
python train_quality_predictor.py
python ../models/catboost/train_token_predictor_catboost.py
python ../models/catboost/train_quality_predictor_catboost.py
python ../predict.py "your prompt here"
```

## Layout

```
dataset/            pipeline scripts + data at each phase (raw_datasets -> cleaned -> merged)
models/             trained model artifacts (RandomForest .joblib, CatBoost .cbm) + CatBoost training scripts
results/            held-out eval metrics (MAE/RMSE/R2, accuracy/F1) per task and model family, as JSON
predict.py           CLI: prompt in -> tokens/cost/quality table out, per trained model
prompts/            held-out prompt lists used to generate/test data
experiments/trial1/ early static token-counting + pricing prototype (superseded)
experiments/trial2/ agent-loop token tracer + response/chat analysis prototype (superseded)
```

`experiments/` holds earlier prototypes kept for reference; the active
pipeline is `dataset/` + `models/` + `predict.py`.
