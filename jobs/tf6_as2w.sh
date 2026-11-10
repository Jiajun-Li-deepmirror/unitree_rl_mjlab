#!/bin/bash
# tf6_as2w.sh (go2-depth 2026-11-05 ~02:50): As2W topairup round. tf5_as2w.sh (topair + longexit from ns_v1 15000) failed
# all three checks (+800 / +1200 / +1600): map3 descents 0.61-0.69 -> 0.54 of the command (bar 0.80; ns_v1 0.86) and
# slow flat commands drifting ((0.15, 0, 0.3) 77-78%, (0.2, 0, 0) / (0.3, 0, 0) up to 136 / 128%), training share of
# zone wheels lifted ~0.2 all along. topairup keeps only the wheels > 5 cm above the ground under the base (the top
# landing of a climb); the descent's bottom landing (front wheels below the base) no longer pays. Otherwise tf5.
source /root/gpufree-data/go2-depth/depth_fe/jobs/ns_common.sh
exec env FE_PICKS=logs/picks_ta6 AS2W_PKG=$PWD/logs/pkg/as2w_mixed_8000 FE_CONS_INIT= \
  FE_PILOT_EXTRA=,wheelstep,wheelstep4,stairspeed2,noslide,topairup,longexit FE_TEACHER_RUN=tf_v6 FE_TEACHER_KS="800 1200 1600" \
  FE_TEACHER_FROM=$PWD/logs/rsl_rl/as2w_depth_teacher/2026-11-04_05-39-29_ns_v1/model_15000.pt \
  FE_STUDENT_RUN=fe_edge_tf6 FE_STUDENT_TASK=Unitree-As2W-Depth-Student-Edge-RL \
  FE_STUDENT_FROM=$PWD/logs/rsl_rl/as2w_depth_student/fe_edge_nsrl_avg1499/model_1499.pt \
  FE_STUDENT_ITERS=1500 FE_STUDENT_LR=1e-4 FE_STUDENT_LR_FINAL=5e-5 FE_STUDENT_FREE=14000 \
  FE_STUDENT_ARGS="--agent.algorithm.bc-coef-final=1.0 --agent.algorithm.critic-warmup=250 --agent.lr-ramp-iters=300" \
  DEPTH_STUDENT_HARD=1 DEPTH_STUDENT_DR_EDGE=0.4 DEPTH_STUDENT_FRICTION=0.3,1.6 DEPTH_STUDENT_STAND_LONG=1 \
  FE_CHECK_AFTER_TRAIN=1 bash jobs/fe_chain.sh As2W
