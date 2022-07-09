#!/bin/bash

# This script analyzes the mean and standard deviation on
# validation and test accuracies for indepent runs 

python3 loss_landscapes/train_accuracies_analysis.py \
    --model mediumCNN \
    --model-folder hpc-trains/loss_landscapes/medium_5_sample \
    --independent-runs 5;

# python3 loss_landscapes/train_accuracies_analysis.py \
#     --model smallCNN \
#     --model-folder hpc-trains/loss_landscapes/small_10_with_sample \
#     --independent-runs 10;