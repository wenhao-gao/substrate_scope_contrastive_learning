#!/bin/bash  

export CUDA_VISIBLE_DEVICES=0,1,2,3

nohup python -u substrate_metric_learning/ray_tune.py \
    --seed 0 \
    --epochs 200 \
    --count 50 \
    --repeat 1 \
    --max_concurrent 2 \
    --search_alg random \
    --dataset_path /home/whgao/substrate_metric_learning/data/train_dataset_min5.ptg \
    --config_path /home/whgao/substrate_metric_learning/configs/hparams_default.yaml &> substrate_metric_learning_tune.out&
