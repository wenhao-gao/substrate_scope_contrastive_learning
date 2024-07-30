#!/bin/bash  

export CUDA_VISIBLE_DEVICES=0

python -u substrate_metric_learning/pre_train.py \
    --seed 42 \
    --debug \
    --dataset_path data/arylhalide_database_w_idx_min5.csv \
    --config_path configs/hparams_default.yaml \
    --alpha 0.5 \
    --lr 0.0001 
