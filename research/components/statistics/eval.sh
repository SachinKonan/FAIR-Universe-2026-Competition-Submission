#!/bin/bash
# usage: eval.sh <outname> <models> <combos> [extra args]; runs fit_eval.py on della-gh (blocking, <10 min)
OUT=fair-data/components/statistics/runs/$1
mkdir -p $OUT
timeout 900 ssh -o BatchMode=yes -n della-gh "cd research/components/statistics && python fit_eval.py --out $OUT --models '$2' --combos '$3' --reuse $4" 2>&1 | grep -v "nll step\|Warning\|warn" | tee -a $OUT/eval.log
