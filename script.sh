#!/usr/bin/env bash
# Runs the full SAGE pipeline in order: clean -> feature engineering -> merge
# -> prepare targets -> train (RandomForest, CatBoost, LightGBM).
#
# Usage:
#   ./script.sh                          # run the full pipeline
#   ./script.sh "your prompt here"       # pipeline, then predict.py on the prompt (all backends)

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

run() {
    echo
    echo "==> $*"
    uv run python "$@"
}

run dataset/clean_datasets.py
run dataset/feature_engineering.py
run dataset/merge_datasets.py
run dataset/prepare_targets.py

run dataset/train_token_predictor.py
run dataset/train_quality_predictor.py
run models/catboost/train_token_predictor_catboost.py
run models/catboost/train_quality_predictor_catboost.py
run models/lightgbm/train_token_predictor_lightgbm.py
run models/lightgbm/train_quality_predictor_lightgbm.py

if [[ $# -gt 0 ]]; then
    prompt="$1"
    for backend in rf catboost lightgbm; do
        echo
        echo "==> predict.py --backend $backend \"$prompt\""
        uv run python predict.py --backend "$backend" "$prompt"
    done
fi
