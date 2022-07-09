#!/bin/bash

# This script reproduces all models from the paper

# python3 loss_landscapes/train_all.py \
#     --output-dir hpc-trains/loss_landscapes/small_10_with_sample \
#     --model smallCNN \
#     --independent-runs 10 \
#     --force-gpu;

python3 loss_landscapes/train_all.py \
    --output-dir hpc-trains/loss_landscapes/medium_10_sample \
    --model mediumCNN \
    --independent-runs 10 \
    --force-gpu;