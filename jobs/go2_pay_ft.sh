#!/bin/bash
# go2_pay_ft.sh [TAG] (go2-depth 2026-11-11, the user: "我现在着急用go2的负载模型，把一个gpu空出来，用来训练go2带负载少踢立面的训练，
# 就像之前一样"): Go2's blind P fine-tune, the way B2's payload / fewer-kicks blind was made (its l-series P).
#   gp1 (default source): from the deployed blind (logs/pkg/go2_fe_polish_10000 = go2_blind_v4, md5 f65007b1) with
#     its 24 keys, plus the payload (12 kg at x 0.15, jobs/fe_robot.sh; ramped in over the first 1000 it,
#     BL_PAYLOAD_RAMP_MP=24000), paycorner, standpose, and B2's legged stairs keys against riser kicks (stumble4x,
#     feetmid, footslide, stairstab); P 4000 it, checks from +1500. Inputs 480 / 274 (no widening).
#   A fine-tune of an earlier one (the user, 11-11 18:4x: "t2的负载精修可以不按照之前的重跑规则，可以直接做微调"):
#     FROM_RUN (a run dir) + FROM_K (its model_K) + FROM_PILOT (the pilot it trained with) + EXTRA_KEYS (added to
#     it for this P); the payload is already learned, so no ramp unless BL_PAYLOAD_RAMP_MP is given; P 3000 it,
#     checks from +500. P's inputs may be wider (blind_chain widens the checkpoint).
#   gp2 (11-11): from gp1 model_11700 + spinflip,spinhold (gp1 avg11500 failed the alternating spins, 0.26 / 0.40 m
#     at +-0.5 / 1 rad/s against 0.25; the deployed blind 0.11 / 0.25: the spin keys were never in its recipe).
# T and M are the source model (BL_T_PICK / BL_M_PICK: nothing retrained), its constraint multipliers
# (v2_robot.sh V2_CONS); the whole acceptance every 500 on 3 seeds (two tiers), then tinynav.
# Delivery: logs/pkg/go2_bl2_<TAG>_<K>, then its videos (jobs/blind_videos.sh). The depth round (jobs/v2_chain.sh from
# that package, as B2's n-series) follows by hand. Log logs/jobs/blind_go2_<TAG>.txt.
set -u
TAG=${1:-gp1}
cd /root/gpufree-data/go2-depth/depth_fe || exit 2
PKG=logs/pkg/go2_fe_polish_10000
if [ -n "${FROM_RUN:-}" ]; then
  RUN=$FROM_RUN K=$FROM_K B=$FROM_PILOT WHY="$(basename $FROM_RUN) model_$FROM_K"
  [ -f $RUN/model_$K.pt ] || { echo "go2_pay_ft.sh: no $RUN/model_$K.pt" >&2; exit 2; }
  PP="$B${EXTRA_KEYS:+,$EXTRA_KEYS}"
  export BL_P_ITERS=${BL_P_ITERS:-3000} BL_FIRST_P=${BL_FIRST_P:-500}
  [ -n "${BL_PAYLOAD_RAMP_MP:-}" ] && export BL_PAYLOAD_RAMP_MP
else
  RUN=logs/rsl_rl/go2_blind/pkg_go2_fe_polish_10000 K=10000 WHY="the deployed go2_blind_v4 (go2_fe_polish_10000)"
  mkdir -p $RUN
  [ -f $RUN/model_10000.pt ] || cp $PKG/model_10000.pt $RUN/
  [ -d $RUN/params ] || cp -r $PKG/params $RUN/
  B=$(cat $PKG/blind_pilot.txt)
  PP="$B,payload,paycorner,standpose,stumble4x,feetmid,footslide,stairstab${EXTRA_KEYS:+,$EXTRA_KEYS}"
  export BL_P_ITERS=${BL_P_ITERS:-4000} BL_FIRST_P=${BL_FIRST_P:-1500} BL_PAYLOAD_RAMP_MP=${BL_PAYLOAD_RAMP_MP:-24000}
fi
export BL_LOG=logs/jobs/blind_go2_$TAG.txt
export BL_T_PICK=$RUN/model_$K.pt BL_T_RUN=$RUN BL_T_PICK_WHY="$WHY"
export BL_M_PICK=$RUN/model_$K.pt BL_M_RUN=$RUN BL_M_PICK_WHY="$WHY"
export BL_PILOT_T=$B BL_PILOT_M=$B BL_PILOT_P="$PP"
exec bash jobs/blind_chain.sh Go2 $TAG
