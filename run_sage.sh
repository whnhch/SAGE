#!/bin/bash
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"

FILENAME="house"
FILEPATH="./dataset/"
GAMMA=500000000.0

DATA_FRAC=1.0
MAX_COLUMNS=90

TRANSFORMER_PATH="/hub/cache"
HF_TOKEN_PATH="./hf_token.txt"
CLF_TRD_PATH="./precompute/"$FILENAME"_10000_decision_tree.pkl"
CLF_OUT_PATH="./precompute/"$FILENAME"_10000_outlier_decision_tree.pkl"
ATTR_PATH="./precompute/"$FILENAME"_attribute_aggfunc_ranks.json"
ATTR_RANK_PATH="./precompute/"$FILENAME"_interesting_attrs.json"

SEED=42
DF_SAMPLE_NUM=5
TAU_COR=0.8
TAU_RAT=5.0
DO_PRUNE=true
DO_CACHE=true
DO_PARALLEL=false
DO_TIMEOUT=false

THRESHOLD=0.3
K=5
INT_THRESHOLD=0.5

echo "Running SAGE with threshold=$THRESHOLD and k=$K"
python ./src/models/greedy.py \
    --filename "$FILENAME" \
    --filepath "$FILEPATH" \
    --transformer_path "$TRANSFORMER_PATH"\
    --hf_token_path "$HF_TOKEN_PATH"\
    --gamma "$GAMMA" \
    --int_threshold "$INT_THRESHOLD" \
    --threshold "$THRESHOLD" \
    --max_columns "$MAX_COLUMNS"\
    --seed "$SEED" \
    --k "$K" \
    --df_sample_num "$DF_SAMPLE_NUM" \
    --clf_trd_path "$CLF_TRD_PATH" \
    --clf_out_path "$CLF_OUT_PATH" \
    --tau_cor "$TAU_COR" \
    --tau_rat "$TAU_RAT" \
    --do_prune "$DO_PRUNE" \
    --do_timeout "$DO_TIMEOUT" \
    --output_dir "result/" \
    --data_num "$DATA_FRAC"\
    --max_columns "$MAX_COLUMNS"\
    --do_parallel "$DO_PARALLEL" \
    --do_cache "$DO_CACHE" \
    --attribute_aggfunc_ranks $ATTR_PATH \
    --interesting_attributes_path $ATTR_RANK_PATH\ 