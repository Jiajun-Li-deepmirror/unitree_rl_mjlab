#!/bin/bash
# tf5s_go2w.sh (go2-depth 2026-11-05 11:35, t2): Go2W climb speed. fe_edge_tf5p avg1499 (PPO+BC under tf_v5 from tf4d)
# passed the tinynav stairs gate (slid back 4.0%, start 0.0, top 0.4 / climb) but its depth_fe s0 failed: up 0.776
# (bar 0.80; teacher 0.869) and one of 24 robots fell on the 15-18 cm up band. tf5p's recipe again from its avg1499,
# plus stairspeed4 (the stairs speed deficit 2x more). Student-only (picks_tg5s with tf_v5 avg26200, copied from t1).
source /root/gpufree-data/go2-depth/depth_fe/jobs/ns_common.sh
mkdir -p logs/picks_tg5s && echo logs/rsl_rl/go2w_depth_teacher/tf_v5_avg26200/model_26200.pt > logs/picks_tg5s/go2w_teacher
exec env FE_PICKS=logs/picks_tg5s GO2W_PKG=$PWD/logs/pkg/go2w_polish_11300 FE_CONS_INIT=side_drift:2.0,clock_side:2.0 \
  FE_PILOT_EXTRA=,wheelstep,wheelstep4,stairspeed2,noslide,noslide2x,sidetgt75,topair,longexit,liftm8,stairspeed3,stairspeed4 FE_TEACHER_RUN=tf_v5 \
  FE_STUDENT_RUN=fe_edge_tf5s FE_STUDENT_TASK=Unitree-Go2W-Depth-Student-Edge-RL \
  FE_STUDENT_FROM=$PWD/logs/rsl_rl/go2w_depth_student/fe_edge_tf5p_avg1499/model_1499.pt \
  FE_STUDENT_ITERS=1500 FE_STUDENT_LR=1e-4 FE_STUDENT_LR_FINAL=5e-5 FE_STUDENT_FREE=14000 \
  FE_STUDENT_ARGS="--agent.algorithm.bc-coef-final=1.0 --agent.algorithm.critic-warmup=250 --agent.lr-ramp-iters=300" \
  DEPTH_STUDENT_HARD=1 DEPTH_STUDENT_DR_EDGE=0.4 DEPTH_STUDENT_FRICTION=0.3,1.6 DEPTH_STUDENT_STAND_LONG=1 \
  bash jobs/fe_chain.sh Go2W
