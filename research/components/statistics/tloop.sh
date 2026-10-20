#!/bin/bash
LOG=fair-data/components/statistics/tshift.log
for i in $(seq 1 12); do
  timeout 900 ssh -o BatchMode=yes -n della-gh "cd research/components/statistics && python trainshift.py 300" >> $LOG 2>&1
  if grep -q "all done" $LOG; then echo finished; exit 0; fi
done
