#!/bin/bash
D=research/components/consistency
O=fair-data/components/consistency/holdout
for mode in lco lbo; do
  for i in 1 2 3 4 5 6; do
    [ -f $O/$mode/cross.npz ] && break
    timeout 1500 ssh -o BatchMode=yes -n della-gh "cd $D && python holdout.py --mode $mode --budget 400" 2>&1 | grep --line-buffered -v Warn
    echo "$mode run $i rc=${PIPESTATUS[0]}"
  done
done
