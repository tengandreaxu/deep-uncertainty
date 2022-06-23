#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --time=1:0:0
#SBATCH --qos=gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --output=bert_train.log 

module load gcc cuda/11.0.2 cudnn mvapich2
source ~/opinion-dynamics/izar/bert_training/bert_venv/bin/activate
srun python bert.py
