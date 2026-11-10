#!/bin/bash
# widen_pkg.sh R PKG OUTDIR EXTRA_KEYS (go2-depth 2026-11-06): widen a blind package's model for the critic inputs the
# package's pilot + EXTRA_KEYS append (e.g. spinhold's spin_hold_offsets), into OUTDIR/model_N.pt, for blind_chain.sh's
# BL_M_PICK / BL_M_RUN. Paths are taken before fe_robot.sh (it sets PKG / N of its own).
set -u
R=$1 BPKG=$2 OUTD=$3 EXTRA=$4
cd /root/gpufree-data/go2-depth/depth_fe && source env.sh > /dev/null 2>&1
CK=$(ls $BPKG/model_*.pt | head -n 1); K=$(basename $CK .pt | cut -d_ -f2)
PILOT=$(cat $BPKG/blind_pilot.txt),$EXTRA
source jobs/fe_robot.sh $R > /dev/null
export BLIND_PILOT=$PILOT BLIND_ANCHOR_COEFF=0 BLIND_ANCHOR_CHECKPOINT=$PWD/$CK
mkdir -p $OUTD
$PY scripts/blind/widen_obs.py --task Unitree-$R-Blind-Polish $CK --out $OUTD/model_$K.pt 2>&1 | grep -v -i warn | tail -3
