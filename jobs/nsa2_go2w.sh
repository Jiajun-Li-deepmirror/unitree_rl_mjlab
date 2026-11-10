#!/bin/bash
# nsa2_go2w.sh (go2-depth 2026-11-04 ~15:55): Go2W wheelahead round, second try. nsa_go2w.sh's teacher (wheelahead
# from ns_v2 22600) failed drift at both checks: sideways (0, +-0.3, 0) crept forward p90 0.17-0.20 m over 3.1-3.2 m
# (bar 5%), ns_v2 0.10-0.12 m. It started with every constraint multiplier at 1x while ns_v2 ended at side_drift 2.0,
# clock_side 2.0 (side_drift fell to 1.0-1.6 during nsa_v1). Same round with those carried over.
source /root/gpufree-data/go2-depth/depth_fe/jobs/ns_common.sh
LAST=$(ls logs/rsl_rl/go2w_depth_teacher/2026-11-04_03-07-36_ns_v2/model_*.pt | sort -V | tail -n 1)
exec env FE_PICKS=logs/picks_nsa2 GO2W_PKG=$PWD/logs/pkg/go2w_polish_11300 FE_CONS_INIT=side_drift:2.0,clock_side:2.0 \
  FE_PILOT_EXTRA=,wheelstep,wheelstep4,stairspeed2,noslide,sidetgt75,wheelahead FE_TEACHER_RUN=nsa_v2 FE_TEACHER_KS="1000 1500" \
  FE_TEACHER_FROM=$PWD/$LAST \
  FE_STUDENT_RUN=fe_edge_nsa2 FE_STUDENT_TASK=Unitree-Go2W-Depth-Student-Edge-RL \
  FE_STUDENT_FROM=$PWD/logs/rsl_rl/go2w_depth_student/fe_edge_nsd_avg1000/model_1000.pt \
  FE_STUDENT_ITERS=1500 FE_STUDENT_LR=1e-4 FE_STUDENT_LR_FINAL=5e-5 \
  FE_STUDENT_ARGS="--agent.algorithm.bc-coef-final=1.0 --agent.algorithm.critic-warmup=250 --agent.lr-ramp-iters=300" \
  DEPTH_STUDENT_HARD=1 DEPTH_STUDENT_DR_EDGE=0.4 DEPTH_STUDENT_FRICTION=0.3,1.6 DEPTH_STUDENT_STAND_LONG=1 \
  FE_CHECK_AFTER_TRAIN=1 bash jobs/fe_chain.sh Go2W
