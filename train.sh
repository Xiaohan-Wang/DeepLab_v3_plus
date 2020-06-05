#!/bin/bash
#SABTCH --gres:gpu=1 # number of GPU per node
#SBATCH --nodes=1
#SBATCH --partition=compsci-gpu
python -u train.py