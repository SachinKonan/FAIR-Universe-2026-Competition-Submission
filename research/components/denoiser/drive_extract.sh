#!/bin/bash
# Usage: drive_extract.sh <ckpt> <root> <name> ; loops short della-gh runs until all sets are done.
CK=$1; ROOT=$2; NAME=${3:-den}
for i in $(seq 1 12); do
  timeout 660 ssh -o BatchMode=yes -n della-gh "cd research/components/denoiser && timeout 590 python extract_den.py --ckpt $CK --root $ROOT --name $NAME --budget 520 2>&1 | grep -v Warn" 
  if [ -f $ROOT/$NAME/test.npz ]; then echo "DRIVER DONE"; exit 0; fi
  sleep 3
done
echo "DRIVER GAVE UP"
