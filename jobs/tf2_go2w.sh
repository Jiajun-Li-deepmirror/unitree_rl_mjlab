#!/bin/bash
# tf2_go2w.sh (go2-depth 2026-11-04 ~19:30): Go2W topflat round 2 on t1, without wheelahead. Every Go2W teacher
# fine-tuned from ns_v2 22600 with wheelahead failed sideways drift (nsa_v1, nsa_v2, tf_v1: (0, +-0.3, 0) crept forward
# p50 0.10-0.14 / p90 0.15-0.20 m over ~3.2 m; bar 5%), whatever side_drift's multiplier (1x, 2x, 3x); ns_v2 itself
# (ns_v1 + sidetgt75, no wheelahead) read p50 0.05-0.07 / p90 0.10-0.13 at 22100 and 22600. wheelahead did nothing for
# the top-landing steps. So: ns_v2 22600 + topflat30 + longexit (tf3_as2w.sh), multipliers as ns_v2 left them
# (side_drift 2, clock_side 2), checks +800 / +1200; student PPO + BC 1.0 from nsd avg1000 as tf_go2w.sh.
source /root/gpufree-data/go2-depth/depth_fe/jobs/ns_common.sh
LAST=$(ls logs/rsl_rl/go2w_depth_teacher/2026-11-04_03-07-36_ns_v2/model_*.pt | sort -V | tail -n 1)
exec env FE_PICKS=logs/picks_tf2 GO2W_PKG=$PWD/logs/pkg/go2w_polish_11300 FE_CONS_INIT=side_drift:2.0,clock_side:2.0 \
  FE_PILOT_EXTRA=,wheelstep,wheelstep4,stairspeed2,noslide,sidetgt75,topflat,topflat30,longexit FE_TEACHER_RUN=tf_v2 FE_TEACHER_KS="800 1200" \
  FE_TEACHER_FROM=$PWD/$LAST \
  FE_STUDENT_RUN=fe_edge_tf2 FE_STUDENT_TASK=Unitree-Go2W-Depth-Student-Edge-RL \
  FE_STUDENT_FROM=$PWD/logs/rsl_rl/go2w_depth_student/fe_edge_nsd_avg1000/model_1000.pt \
  FE_STUDENT_ITERS=1500 FE_STUDENT_LR=1e-4 FE_STUDENT_LR_FINAL=5e-5 FE_STUDENT_FREE=14000 \
  FE_STUDENT_ARGS="--agent.algorithm.bc-coef-final=1.0 --agent.algorithm.critic-warmup=250 --agent.lr-ramp-iters=300" \
  DEPTH_STUDENT_HARD=1 DEPTH_STUDENT_DR_EDGE=0.4 DEPTH_STUDENT_FRICTION=0.3,1.6 DEPTH_STUDENT_STAND_LONG=1 \
  FE_CHECK_AFTER_TRAIN=1 bash jobs/fe_chain.sh Go2W
