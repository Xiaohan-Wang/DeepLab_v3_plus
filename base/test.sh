#!/bin/bash
#SBATCH --gres=gpu:1 #number of GPU per node
#SBATCH --nodes=1
#SBATCH --partition=compsci-gpu

export MLFLOW_EXPERIMENT_NAME=base_trainer
python -u test.py
