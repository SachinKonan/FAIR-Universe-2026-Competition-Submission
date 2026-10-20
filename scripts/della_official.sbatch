#!/bin/bash
# Princeton della: run scripts/run_official.sh on one 40 GB A100 with the Codabench worker's CPU/RAM
# (4 vCPUs, 32 GiB). Compute nodes have no internet, so run scripts/run_official.sh once on a login node
# first (or pre-pull the image into third_party/).
#   sbatch scripts/della_official.sbatch <submission.zip> <test data dir>
#SBATCH -J fair-official
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu-test
#SBATCH --constraint=gpu40
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o runs/della-official-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv
OUT=runs/official_$SLURM_JOB_ID
mkdir -p "$OUT"
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 2 > "$OUT/gpu.csv" &
SMI=$!
scripts/run_official.sh "$1" "$2" "$OUT"
kill $SMI
echo "peak GPU MiB: $(cut -d, -f2 "$OUT/gpu.csv" | tr -dc '0-9\n' | sort -n | tail -1)"
