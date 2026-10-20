#!/bin/bash
# Chunked (<=8 min) resumable training of the all-families OE model on della-gh.
O=fair-data/components/outlier
STEPS=${1:-6000}
for i in $(seq 1 14); do
  if grep -q "saved step $STEPS" $O/run_all.log 2>/dev/null; then echo finished; break; fi
  echo "=== chunk $i $(date -u)" >> $O/run_all.log
  timeout 660 ssh -o BatchMode=yes -n della-gh "cd research/components/outlier && timeout 620 python train_oe.py --stage train --out $O/run_all --holdout= --steps $STEPS --budget-min 8 --fam-per-step 3 --log-every 100 --ckpt-every 250 2>&1 | grep -v Warn" >> $O/run_all.log 2>&1
done
