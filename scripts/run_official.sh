#!/bin/bash
# Run a submission ZIP through the organizers' own ingestion program inside their runtime image
# (ayaga3540/fair_universe:v1.0) with Apptainer, mounted at the same paths Codabench uses.
#
#   scripts/run_official.sh <submission.zip> <dir holding WIDE12H_bin2_2arcmin_kappa_test_phase2_new_v2.npy> [out dir]
#
# Needs internet the first time (it clones the organizers' repo for ingestion_program/ and pulls the image into
# third_party/); later runs work offline, e.g. on a cluster compute node. Override the image with SIF=<path>.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
ZIP=$(realpath "${1:?submission zip}")
DATA=$(realpath "${2:?test data dir}")
OUT=$(realpath -m "${3:-$ROOT/runs/official_$(date +%Y%m%d_%H%M%S)}")
TP=$ROOT/third_party
SIF=${SIF:-$TP/fair_universe_v1.0.sif}
TEST=WIDE12H_bin2_2arcmin_kappa_test_phase2_new_v2.npy

mkdir -p "$TP"
if [ ! -d "$TP/Cosmology_Challenge/ingestion_program" ]; then
    git clone -q --depth 1 https://github.com/FAIR-Universe/Cosmology_Challenge.git "$TP/Cosmology_Challenge"
fi
if [ ! -f "$SIF" ]; then
    APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-$TP/apptainer_cache} \
        apptainer pull "$SIF" docker://ayaga3540/fair_universe:v1.0
fi

mkdir -p "$OUT/output" "$OUT/sub"
unzip -q -o "$ZIP" -d "$OUT/sub"
NV=""; command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1 && NV="--nv"
T0=$(date +%s)
apptainer exec $NV --containall \
    --bind "$OUT/sub:/app/ingested_program:ro" \
    --bind "$TP/Cosmology_Challenge/ingestion_program:/app/program:ro" \
    --bind "$OUT/output:/app/output" \
    --bind "$DATA/$TEST:/app/input_data/$TEST:ro" \
    --pwd /app/program \
    "$SIF" python3 run_ingestion.py --codabench 2>&1 | tee "$OUT/ingestion.log"
echo "wall $(( $(date +%s) - T0 )) s; scores in $OUT/output/result.json"
