#!/bin/bash
# usage: gloop.sh <tag> <spec> [n]; reruns gdirect.py (cached, resumable) until it completes once
TAG=$1; SPEC=$2; N=${3:-6}
LOG=fair-data/components/statistics/gdirect_$TAG.log
for i in $(seq 1 $N); do
  timeout 900 ssh -o BatchMode=yes -n della-gh "cd research/components/statistics && timeout 590 python gdirect.py --tag $TAG --spec '$SPEC' --test" >> $LOG 2>&1
  if grep -q "mean over .* infam" $LOG; then echo finished; exit 0; fi
done
