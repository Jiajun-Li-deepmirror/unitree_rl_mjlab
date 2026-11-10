#!/bin/bash
# tf5p_go2w.sh (go2-depth 2026-11-05 07:20, t1): tf4p_go2w.sh's two-stage student under the tf_v5 (noslide2x) pick avg26200
# (up/down 0.869/0.897; top_flat: slide back 2.2% vs tf_v4 3.3%, top landing 11.7% vs 17.5%, clean 71.9 vs 64.6%,
# start landing 34.6% vs 21.5%). PPO+BC from tf4d avg1999, pilot as tf5. Waits for tf4p's training to free the GPU.
# 07:29 the first start OOM'd: wait_free saw 14.1 GB free beside tf4p's training (10.4 GB) before tf4p's checks
# started. Now it waits for tf4p's training (pid 189862) to end first.
while kill -0 189862 2> /dev/null; do sleep 30; done
source /root/gpufree-data/go2-depth/depth_fe/jobs/ns_common.sh
mkdir -p logs/picks_tg5p && cp logs/picks_tg5/go2w_teacher logs/picks_tg5p/go2w_teacher
exec env FE_PICKS=logs/picks_tg5p GO2W_PKG=$PWD/logs/pkg/go2w_polish_11300 FE_CONS_INIT=side_drift:2.0,clock_side:2.0 \
  FE_PILOT_EXTRA=,wheelstep,wheelstep4,stairspeed2,noslide,noslide2x,sidetgt75,topair,longexit,liftm8,stairspeed3 FE_TEACHER_RUN=tf_v5 \
  FE_STUDENT_RUN=fe_edge_tf5p FE_STUDENT_TASK=Unitree-Go2W-Depth-Student-Edge-RL \
  FE_STUDENT_FROM=$PWD/logs/rsl_rl/go2w_depth_student/fe_edge_tf4d_avg1999/model_1999.pt \
  FE_STUDENT_ITERS=1500 FE_STUDENT_LR=1e-4 FE_STUDENT_LR_FINAL=5e-5 FE_STUDENT_FREE=14000 \
  FE_STUDENT_ARGS="--agent.algorithm.bc-coef-final=1.0 --agent.algorithm.critic-warmup=250 --agent.lr-ramp-iters=300" \
  DEPTH_STUDENT_HARD=1 DEPTH_STUDENT_DR_EDGE=0.4 DEPTH_STUDENT_FRICTION=0.3,1.6 DEPTH_STUDENT_STAND_LONG=1 \
  bash jobs/fe_chain.sh Go2W
