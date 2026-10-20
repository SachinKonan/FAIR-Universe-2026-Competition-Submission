#!/bin/bash
# Extract den_e (11 standard TPU denoisers) and den_w (4 wide), then full-data family fits and LCO refits.
R=fair-data/tpu/runs
V3=fair-data/v3
T=research/tpu
CDIR=research/components/consistency
PY=python
GH="timeout 1500 ssh -o BatchMode=yes -n della-gh"
E="$(ls $R/d1*/net_torch.pt | tr '\n' ' ') fair-data/components/denoiser/unet_s1/net_ep29.pt fair-data/components/denoiser/unet_s2/net_ep25.pt"
Wd=$(ls $R/w20*/net_torch.pt | tr '\n' ' ')
for spec in "den_e|$E"; do
  name=${spec%%|*}; ck=${spec#*|}
  for i in $(seq 1 12); do
    [ -f $V3/$name/test.npz ] && break
    $GH "cd $T && $PY extract_ens.py --ckpts $ck --name $name --budget 420" 2>&1 | grep -v Warn | grep -v "full\[" 
  done
  echo "$name extracted: $(ls $V3/$name | wc -l) files"
done
until [ -f $V3/den_w/test.npz ]; do sleep 30; done
for i in $(seq 1 8); do
  [ -f fair-data/components/consistency/fam/den_e.npz ] && [ -f fair-data/components/consistency/fam/den_w.npz ] && break
  $GH "cd $CDIR && $PY fit_families.py --mode families --only den_e,den_w --budget 420" 2>&1 | grep -v Warn | tail -3
done
for K in 0 1 2; do
  for i in 1 2 3 4; do
    [ -f fair-data/components/consistency/lco3/set${K}_std/den_w.npz ] && break
    $GH "cd $CDIR && $PY holdout3.py --set $K --variant std --fams den_e,den_w --budget 420" 2>&1 | grep -v Warn | tail -2
  done
done
echo pipeline-done
