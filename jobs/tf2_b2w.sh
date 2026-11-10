#!/bin/bash
# tf2_b2w.sh (go2-depth 2026-11-04 ~21:15): B2W topoff round. tf_v1 (topflat30 + longexit from nsa_v2 15100) failed
# both checks on (0.2, 0, 0) x127-129% only (bar 120%), while its descent multiplier climbed 2.3 -> 2.9: topflat paid the
# last swing of every descent. Now topoff15 (take-offs from a landing with flat ahead, -15 each) + longexit, and
# noslide2x (the B2W students still slide back 10-12% of the tinynav step-ups, bar 5%; its teachers 7-9% in mjlab).
# Otherwise as tf_b2w.sh: from nsa_v2's raw 15100 + wheelahead, bdhold, multipliers back_drift 1.7 / descent 2.3,
# checks +800 / +1200 / +1600, student PPO + BC 1.0 from v6_ns (ns1 avg1000), hard commands, waits for 15 GB free.
source /root/gpufree-data/go2-depth/depth_fe/jobs/ns_common.sh
exec env FE_PICKS=logs/picks_tf2b B2W_PKG=$PWD/logs/pkg/b2w_polish_9000_sidecmd_desc \
  FE_CONS_INIT=back_drift:1.7,descent:2.3 FE_MARGIN=0.07 \
  FE_PILOT_EXTRA=,wheelstep,wheelstep4,stairspeed2,noslide,noslide2x,wheelahead,bdhold,topoff,topoff15,longexit FE_TEACHER_RUN=tf_v2 FE_TEACHER_KS="800 1200 1600" \
  FE_TEACHER_FROM=$PWD/logs/rsl_rl/b2w_depth_teacher/2026-11-04_13-36-59_nsa_v2/model_15100.pt \
  FE_STUDENT_RUN=fe_edge_tf2 FE_STUDENT_TASK=Unitree-B2W-Depth-Student-Edge-RL \
  FE_STUDENT_FROM=$PWD/logs/rsl_rl/b2w_depth_student/fe_edge_ns1_avg1000/model_1000.pt \
  FE_STUDENT_ITERS=1500 FE_STUDENT_LR=1e-4 FE_STUDENT_LR_FINAL=5e-5 FE_STUDENT_FREE=15000 \
  FE_STUDENT_ARGS="--agent.algorithm.bc-coef-final=1.0 --agent.algorithm.critic-warmup=250 --agent.lr-ramp-iters=300" \
  DEPTH_STUDENT_HARD=1 FE_CHECK_AFTER_TRAIN=1 bash jobs/fe_chain.sh B2W
