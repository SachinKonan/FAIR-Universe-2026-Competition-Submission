#!/bin/bash
#SBATCH -A zhuangl
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -o fair-data/components/outlier/slurm-%x-%j.out
# usage: sbatch -J NAME --qos=Q -t T oe.sbatch OUT HOLDOUT STEPS BUDGET_MIN
cd research/components/outlier
nvidia-smi --query-gpu=name,memory.used --format=csv
python train_oe.py --stage both --out "$1" --holdout "$2" --steps "$3" --budget-min "$4" --fam-per-step 3 --log-every 100 --ckpt-every 250
