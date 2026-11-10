#!/bin/bash
# probe_export.sh R CKPT OUT PILOT_EXTRA (go2-depth 2026-11-04): export a student checkpoint the chain did not pick, with
# the chain's environment (env.sh, fe_robot.sh, pilot + extra, anchor), for the tinynav slide gate as information.
# 2026-11-09: the blind package comes from v2_robot.sh as in v2_chain.sh (<R>_PKG / <R>_N before fe_robot.sh). Until
# then a fixed list chose it: Go2W go2w_polish_11300 (the v2 students descend from go2w_bl2_g27b_19800) and B2 none
# (an empty base pilot: the ONNX metadata said pyramidal cone, impratio 1, hip default 0.03 and had no gait clock).
set -u
cd /root/gpufree-data/go2-depth/depth_fe && source env.sh
R=$1 CK=$2 OUT=$3 EXTRA=$4 r=${1,,}
source jobs/v2_robot.sh $R > /dev/null || exit 2
export $V2_PKG_VAR=$V2_PKG
source jobs/fe_robot.sh $R || exit 2
[ -s $PKG/blind_pilot.txt ] || { echo "probe_export.sh: no $PKG/blind_pilot.txt" >&2; exit 2; }
export BLIND_PILOT=$(cat $PKG/blind_pilot.txt)$EXTRA BLIND_ANCHOR_COEFF=0 BLIND_ANCHOR_CHECKPOINT=$PWD/logs/rsl_rl/${r}_depth_teacher/blind_init_fe/model_$N.pt
mkdir -p $(dirname $OUT)
$PY scripts/go2_depth/export_student.py --task Unitree-$R-Depth-Student-Edge --checkpoint $CK --out $OUT 2>&1 | tail -n 2
md5sum $OUT
