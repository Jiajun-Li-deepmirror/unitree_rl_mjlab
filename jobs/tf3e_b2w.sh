#!/bin/bash
# tf3e_b2w.sh (go2-depth 2026-11-05 08:45, t2): B2W student in flights closed by an end wall (pilot endwall). The tf3d
# (distilled) students passed the tinynav slide gate's slide (3.8-3.9%) and start landing (0.0) but lifted their
# front wheels on the top landing 2.0-2.1 times per climb; with map3's stairwell wall (1.325 m beyond that landing)
# moved out of view, 0.3. PPO+BC (tf3's recipe) from tf3d avg1500 under tf_v3 avg15900, pilot tf3 + endwall, so the
# RL reward (topair) sees the student's lifts in front of a wall. Student-only (picks_tb3e).
source /root/gpufree-data/go2-depth/depth_fe/jobs/ns_common.sh
mkdir -p logs/picks_tb3e && cp logs/picks_tb3/b2w_teacher logs/picks_tb3e/b2w_teacher
exec env FE_PICKS=logs/picks_tb3e B2W_PKG=$PWD/logs/pkg/b2w_polish_9000_sidecmd_desc \
  FE_CONS_INIT=back_drift:1.7,descent:2.3 \
  FE_PILOT_EXTRA=,wheelstep,wheelstep4,stairspeed2,noslide,noslide2x,wheelahead,bdhold,topair,longexit,endwall FE_TEACHER_RUN=tf_v3 \
  FE_STUDENT_RUN=fe_edge_tf3e FE_STUDENT_TASK=Unitree-B2W-Depth-Student-Edge-RL \
  FE_STUDENT_FROM=$PWD/logs/rsl_rl/b2w_depth_student/fe_edge_tf3d_avg1500/model_1500.pt \
  FE_STUDENT_ITERS=1500 FE_STUDENT_LR=1e-4 FE_STUDENT_LR_FINAL=5e-5 FE_STUDENT_FREE=15000 \
  FE_STUDENT_ARGS="--agent.algorithm.bc-coef-final=1.0 --agent.algorithm.critic-warmup=250 --agent.lr-ramp-iters=300" \
  DEPTH_STUDENT_HARD=1 bash jobs/fe_chain.sh B2W
