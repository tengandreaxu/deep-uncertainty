#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --time=1:0:0
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --output=small_train.log 

module load gcc cuda/11.0.2 cudnn mvapich2
source ~/deep-uncertainty/venv/bin/activate
srun python3 loss_landscapes/train_all.py \
    --output-dir hpc-trains \
    --model smallCNN \
    --independent-runs 10 \
    --force-gpu;
