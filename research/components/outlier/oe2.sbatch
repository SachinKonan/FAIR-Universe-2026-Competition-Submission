#!/bin/bash
#SBATCH -A zhuangl
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -o fair-data/components/outlier/slurm-%x-%j.out
# usage: sbatch -J NAME --qos=Q -t T oe2.sbatch OUT HOLDOUT STEPS BUDGET_MIN [extra train_oe args]
cd research/components/outlier
nvidia-smi --query-gpu=name,memory.used --format=csv
python train_oe.py --stage both --out "$1" --holdout "$2" --steps "$3" --budget-min "$4" --log-every 100 --ckpt-every 250 "${@:5}"
