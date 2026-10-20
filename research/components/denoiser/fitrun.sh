#!/bin/bash
# Usage: fitrun.sh <root> <out> <models> [combos] [extra args...]  (runs fit_v3.py on della-gh)
ROOT=$1; OUT=$2; MODELS=$3; COMBOS=${4:-}; shift 4; EXTRA="$*"
mkdir -p $OUT
timeout 660 ssh -o BatchMode=yes -n della-gh "cd research && timeout 600 python fit_v3.py --root $ROOT --out $OUT --models '$MODELS' --combos '$COMBOS' $EXTRA" > $OUT/fit.log 2>&1
grep -v 'nll step\|Warn' $OUT/fit.log
