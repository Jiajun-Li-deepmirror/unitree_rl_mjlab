#!/bin/bash
# ns_go2w_s2.sh (go2-depth 2026-11-04 ~06:20): Go2W noslide student round 2. Teacher = the ns pick (ns_v2 avg22600,
# margin 0.051). Student round 1 (fe_edge_ns1, PPO + BC -> 0.5 from v5_ws) failed drift on both averages (avg1499 up
# 0.804) and slid back as much as v5_ws (tinynav 12 vs 11%; mjlab stairs terrain 24.6 vs 24.2%, the teacher 15.3%).
# Round 2 from ns1 avg1499 with the slide terms at 2x (noslide2x) and the side drift guard (sidetgt75) in the student's
# own PPO, BC held higher (-> 0.7), 1500 it. Picks logs/picks_ns (go2w_teacher kept, so no teacher stage).
source /root/gpufree-data/go2-depth/depth_fe/jobs/ns_common.sh
exec env GO2W_PKG=$PWD/logs/pkg/go2w_polish_11300 FE_CONS_INIT= \
  FE_PILOT_EXTRA=,wheelstep,wheelstep4,stairspeed2,noslide,sidetgt75,noslide2x FE_TEACHER_RUN=ns_v2 \
  FE_STUDENT_TASK=Unitree-Go2W-Depth-Student-Edge-RL FE_STUDENT_RUN=fe_edge_ns2 \
  FE_STUDENT_FROM=$PWD/logs/rsl_rl/go2w_depth_student/fe_edge_ns1_avg1499/model_1499.pt \
  FE_STUDENT_ITERS=1500 FE_STUDENT_LR=1e-4 FE_STUDENT_LR_FINAL=5e-5 \
  FE_STUDENT_ARGS="--agent.algorithm.bc-coef-final=0.7 --agent.algorithm.critic-warmup=250 --agent.lr-ramp-iters=300" \
  bash jobs/fe_chain.sh Go2W
