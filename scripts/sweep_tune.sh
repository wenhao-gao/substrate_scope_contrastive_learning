#!/bin/bash  

export CUDA_VISIBLE_DEVICES=2
nohup python -u substrate_metric_learning/sweep_tune.py \
    --seed 0 \
    --epochs 100 \
    --count 16 \
    --dataset_path data/train_dataset.ptg \
    --config_path configs/hparams_default.yaml \
    --tune_config_path configs/hparams_tune.yaml &> substrate_metric_learning_tune.out&
