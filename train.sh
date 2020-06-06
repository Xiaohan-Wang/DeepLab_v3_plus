#!/bin/bash
#SBATCH --gres=gpu:4 # number of GPU per node
#SBATCH --nodes=1
#SBATCH --partition=compsci-gpu
python -u train.py