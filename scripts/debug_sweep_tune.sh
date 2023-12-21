#!/bin/bash  

export CUDA_VISIBLE_DEVICES=1

python -u substrate_metric_learning/sweep_tune.py \
    --seed 0 \
    --debug \
    --dataset_path data/train_dataset.ptg \
    --config_path configs/hparams_default.yaml \
    --tune_config_path configs/debug_hparams_tune.yaml
