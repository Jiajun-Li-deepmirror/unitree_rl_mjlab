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
[ -f $PKG/blind_params.txt ] && export BLIND_PARAMS=$(cat $PKG/blind_params.txt)
export BLIND_PILOT=$(cat $PKG/blind_pilot.txt)$EXTRA BLIND_ANCHOR_COEFF=0 BLIND_ANCHOR_CHECKPOINT=$PWD/logs/rsl_rl/${r}_depth_teacher/${FE_INIT_RUN:-blind_init_fe}/model_$N.pt
mkdir -p $(dirname $OUT)
# the one export (2026-11-11, unification item 3): the checkpoint's own run_env.sh when it has one, else the
# environment set up above; the ONNX copied to OUT
bash jobs/export_policy.sh depth $R $CK ${OUT%.onnx}_pkg 2>&1 | tail -n 2 && cp ${OUT%.onnx}_pkg/policy.onnx $OUT
md5sum $OUT
