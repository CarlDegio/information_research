#!/bin/bash

model=$1
num_layers=${2:-2}
n_seeds=${3:-10}

for ratio in 0.00 0.125 0.25 0.375 0.5; do
    echo "Ratio $ratio"
    python -u main.py \
    --dataset="mnist" \
    --mode="colored" \
    --model_type=$model \
    --hidden_dim=256 \
    --batch_size=512 \
    --l2_regularizer_weight=0.002 \
    --lr=0.001 \
    --penalty_anneal_iters=0 \
    --penalty_weight=0.000 \
    --epochs=300 \
    --early_stopping \
    --ratio=$ratio \
    --save_models \
    --seed=$n_seeds \
    --check_performance \
    --num_layers $num_layers \
    --get_entropy \
    # --get_mi
    # --eval_only # \
    
done;

# bash scripts/train_colored.sh "linear" 3 &