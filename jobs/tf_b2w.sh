#!/bin/bash
# tf_b2w.sh (go2-depth 2026-11-04 ~19:10): B2W topflat round with topflat30 + longexit (see tf3_as2w.sh). Teacher from nsa_v2's 15100 (raw; its
# average passed with margin 0.094, the 15600 one ran over speed) + wheelahead, bdhold, topflat, multipliers where
# nsa_v2 left them at 15100 (back_drift 1.7, descent 2.3), checks +800 / +1200. Student PPO + BC 1.0 from v6_ns (ns1
# avg1000), hard commands only, as nsa2_b2w.sh; B2W's student holds 14.4 GB, so it waits for 15 GB free.
source /root/gpufree-data/go2-depth/depth_fe/jobs/ns_common.sh
exec env FE_PICKS=logs/picks_tf B2W_PKG=$PWD/logs/pkg/b2w_polish_9000_sidecmd_desc \
  FE_CONS_INIT=back_drift:1.7,descent:2.3 FE_MARGIN=0.07 \
  FE_PILOT_EXTRA=,wheelstep,wheelstep4,stairspeed2,noslide,wheelahead,bdhold,topflat,topflat30,longexit FE_TEACHER_RUN=tf_v1 FE_TEACHER_KS="800 1200" \
  FE_TEACHER_FROM=$PWD/logs/rsl_rl/b2w_depth_teacher/2026-11-04_13-36-59_nsa_v2/model_15100.pt \
  FE_STUDENT_RUN=fe_edge_tf1 FE_STUDENT_TASK=Unitree-B2W-Depth-Student-Edge-RL \
  FE_STUDENT_FROM=$PWD/logs/rsl_rl/b2w_depth_student/fe_edge_ns1_avg1000/model_1000.pt \
  FE_STUDENT_ITERS=1500 FE_STUDENT_LR=1e-4 FE_STUDENT_LR_FINAL=5e-5 FE_STUDENT_FREE=15000 \
  FE_STUDENT_ARGS="--agent.algorithm.bc-coef-final=1.0 --agent.algorithm.critic-warmup=250 --agent.lr-ramp-iters=300" \
  DEPTH_STUDENT_HARD=1 FE_CHECK_AFTER_TRAIN=1 bash jobs/fe_chain.sh B2W
