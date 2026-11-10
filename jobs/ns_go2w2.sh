#!/bin/bash
# ns_go2w2.sh: Go2W noslide round, teacher part 2 (go2-depth 2026-11-04): ns_v1's averages all failed only the
# acceptance's drift on (0, +-0.3) (forward creep 5.1-6.3%, bar 5%) while sliding back less (avg20600: 3.2% of tries,
# riser contact 10.2, clean 71%; up 0.828). From ns_v1's last checkpoint with + sidetgt75 (wheelcons' side drift guard
# 0.095 -> 0.075, at most 2x); checks at +500 / +1000 / +1500; then the student as ns_go2w.sh.
source /root/gpufree-data/go2-depth/depth_fe/jobs/ns_common.sh
LAST=$(ls logs/rsl_rl/go2w_depth_teacher/2026-11-04_01-07-54_ns_v1/model_*.pt | sort -V | tail -n 1)
exec env GO2W_PKG=$PWD/logs/pkg/go2w_polish_11300 FE_CONS_INIT= \
  FE_PILOT_EXTRA=,wheelstep,wheelstep4,stairspeed2,noslide,sidetgt75 FE_TEACHER_RUN=ns_v2 FE_TEACHER_KS="500 1000 1500" \
  FE_TEACHER_FROM=$PWD/$LAST \
  FE_STUDENT_TASK=Unitree-Go2W-Depth-Student-Edge-RL \
  FE_STUDENT_FROM=$PWD/logs/rsl_rl/go2w_depth_student/fe_edge_ws1_avg1999/model_1999.pt \
  FE_STUDENT_ITERS=1500 FE_STUDENT_LR=1e-4 FE_STUDENT_LR_FINAL=5e-5 \
  FE_STUDENT_ARGS="--agent.algorithm.bc-coef-final=0.5 --agent.algorithm.critic-warmup=250 --agent.lr-ramp-iters=300" \
  bash jobs/fe_chain.sh Go2W
