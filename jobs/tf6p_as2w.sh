#!/bin/bash
# tf6p_as2w.sh (go2-depth 2026-11-05 08:50, t1): As2W two-stage student. The distilled tf6d avg1500 / 1999 slid back
# 1.0 / 0.0% in the tinynav slide gate but stepped on the start landing (1.0 per climb) and the top landing (3.1 / 2.8;
# 4.2 with map3's end wall out of view, so not the wall as on the B2W). PPO+BC (tf6's recipe) from tf6d avg1999
# (copied from t2) under tf_v6 avg16200, pilot tf6 + endwall, so wheelstep / topairup see the student's own steps.
# Waits for the Go2W tf5p training (pid 253184) to end: wait_free alone let tf5p OOM beside tf4p.
while kill -0 253184 2> /dev/null; do sleep 30; done
source /root/gpufree-data/go2-depth/depth_fe/jobs/ns_common.sh
exec env FE_PICKS=logs/picks_ta6 AS2W_PKG=$PWD/logs/pkg/as2w_mixed_8000 FE_CONS_INIT= \
  FE_PILOT_EXTRA=,wheelstep,wheelstep4,stairspeed2,noslide,topairup,longexit,endwall FE_TEACHER_RUN=tf_v6 \
  FE_STUDENT_RUN=fe_edge_tf6p FE_STUDENT_TASK=Unitree-As2W-Depth-Student-Edge-RL \
  FE_STUDENT_FROM=$PWD/logs/rsl_rl/as2w_depth_student/fe_edge_tf6d_avg1999/model_1999.pt \
  FE_STUDENT_ITERS=1500 FE_STUDENT_LR=1e-4 FE_STUDENT_LR_FINAL=5e-5 FE_STUDENT_FREE=14000 \
  FE_STUDENT_ARGS="--agent.algorithm.bc-coef-final=1.0 --agent.algorithm.critic-warmup=250 --agent.lr-ramp-iters=300" \
  DEPTH_STUDENT_HARD=1 DEPTH_STUDENT_DR_EDGE=0.4 DEPTH_STUDENT_FRICTION=0.3,1.6 DEPTH_STUDENT_STAND_LONG=1 \
  bash jobs/fe_chain.sh As2W
