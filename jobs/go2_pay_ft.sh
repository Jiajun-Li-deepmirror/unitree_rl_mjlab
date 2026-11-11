#!/bin/bash
# go2_pay_ft.sh [TAG] (go2-depth 2026-11-11, the user: "我现在着急用go2的负载模型，把一个gpu空出来，用来训练go2带负载少踢立面的训练，
# 就像之前一样"): Go2's blind P fine-tune from the deployed blind (logs/pkg/go2_fe_polish_10000 = go2_blind_v4, md5
# f65007b1), the way B2's payload / fewer-kicks blind was made (its l-series P):
#   - the package's 24 keys, plus the payload (12 kg at x 0.15, jobs/fe_robot.sh; ramped in over the first 1000 it,
#     BL_PAYLOAD_RAMP_MP=24000), paycorner (30% of the heavy robots at the corner), standpose (the loaded standing pose);
#   - B2's legged stairs keys against riser kicks: stumble4x (foot_stumble x4), feetmid, footslide, stairstab;
#   - T and M are the package's model (BL_T_PICK / BL_M_PICK: nothing retrained), its constraint multipliers
#     (v2_robot.sh V2_CONS); P 4000 it, the whole acceptance from +1500 every 500 on 3 seeds (two tiers), then tinynav;
#   - the inputs stay 480 / 274 (checked: no widening).
# Delivery: logs/pkg/go2_bl2_<TAG>_<K>, then its videos (jobs/blind_videos.sh). The depth round (jobs/v2_chain.sh from
# that package, as B2's n-series) follows by hand. Log logs/jobs/blind_go2_<TAG>.txt.
set -u
TAG=${1:-gp1}
cd /root/gpufree-data/go2-depth/depth_fe || exit 2
PKG=logs/pkg/go2_fe_polish_10000
RUN=logs/rsl_rl/go2_blind/pkg_go2_fe_polish_10000
mkdir -p $RUN
[ -f $RUN/model_10000.pt ] || cp $PKG/model_10000.pt $RUN/
[ -d $RUN/params ] || cp -r $PKG/params $RUN/
B=$(cat $PKG/blind_pilot.txt)
export BL_LOG=logs/jobs/blind_go2_$TAG.txt
export BL_T_PICK=$RUN/model_10000.pt BL_T_RUN=$RUN BL_T_PICK_WHY="the deployed go2_blind_v4 (go2_fe_polish_10000)"
export BL_M_PICK=$RUN/model_10000.pt BL_M_RUN=$RUN BL_M_PICK_WHY="the deployed go2_blind_v4 (go2_fe_polish_10000)"
export BL_PILOT_T=$B BL_PILOT_M=$B BL_PILOT_P="$B,payload,paycorner,standpose,stumble4x,feetmid,footslide,stairstab"
export BL_P_ITERS=${BL_P_ITERS:-4000} BL_FIRST_P=${BL_FIRST_P:-1500} BL_PAYLOAD_RAMP_MP=${BL_PAYLOAD_RAMP_MP:-24000}
exec bash jobs/blind_chain.sh Go2 $TAG
