#!/bin/bash  

export CUDA_VISIBLE_DEVICES=0
nohup python -u substrate_metric_learning/pre_train.py \
    --seed 0 \
    --epochs 100 \
    --wandb online \
    --dataset_path data/arylhalide_database_w_idx_min5.csv \
    --config_path configs/hparams_default.yaml &> substrate_metric_learning_0.out&

# export CUDA_VISIBLE_DEVICES=1
# nohup python -u substrate_metric_learning/pre_train.py \
#     --seed 1 \
#     --epochs 200 \
#     --wandb online \
#     --dataset_path data/train_dataset_min5.ptg \
#     --config_path configs/max_r2.yaml &> substrate_metric_learning_1.out&

# export CUDA_VISIBLE_DEVICES=2
# nohup python -u substrate_metric_learning/pre_train.py \
#     --seed 2 \
#     --epochs 200 \
#     --wandb online \
#     --dataset_path data/train_dataset_min5.ptg \
#     --config_path configs/max_r2.yaml &> substrate_metric_learning_2.out&

# export CUDA_VISIBLE_DEVICES=3
# nohup python -u substrate_metric_learning/pre_train.py \
#     --seed 3 \
#     --epochs 200 \
#     --wandb online \
#     --dataset_path data/train_dataset_min5.ptg \
#     --config_path configs/max_r2.yaml &> substrate_metric_learning_3.out&