#!/bin/bash  

export CUDA_VISIBLE_DEVICES=0,1,2,3

python -u substrate_metric_learning/ray_tune.py \
    --seed 0 \
    --debug \
    --count 4 \
    --repeat 1 \
    --search_alg random \
    --dataset_path /home/whgao/substrate_metric_learning/data/train_dataset_min5.ptg \
    --config_path /home/whgao/substrate_metric_learning/configs/hparams_default.yaml
