#!/bin/bash
# nsa2_b2w.sh (go2-depth 2026-11-04 ~13:45): B2W wheelahead round, second try. nsa_b2w.sh's teacher (wheelahead from
# ns_v1 14100) failed both checks on small sideways speeds ((0, 0.15, 0) 128 / 133%, (0, +-0.3, 0) 118 / 125%; bar
# 120%) as its back_drift multiplier rose 1.68 -> 1.97; top-landing lifts 18.5 -> 13.8% of the near time. Again from
# ns_v1 14100 with wheelahead + bdhold (back_drift at most 1.7). Student: PPO + BC 1.0 against the new teacher from
# b2w_depth_v6_ns (ns1 avg1000), hard commands, no DR edges / friction (B2W students climbed box faces with them),
# critic warmup 250, lr ramp 300, 1500 it, checks after training. Runs next to the Go2W chain on t2.
source /root/gpufree-data/go2-depth/depth_fe/jobs/ns_common.sh
exec env FE_PICKS=logs/picks_nsa2 B2W_PKG=$PWD/logs/pkg/b2w_polish_9000_sidecmd_desc \
  FE_CONS_INIT=back_drift:1.68,descent:1.90 FE_MARGIN=0.07 \
  FE_PILOT_EXTRA=,wheelstep,wheelstep4,stairspeed2,noslide,wheelahead,bdhold FE_TEACHER_RUN=nsa_v2 FE_TEACHER_KS="1000 1500" \
  FE_TEACHER_FROM=$PWD/logs/rsl_rl/b2w_depth_teacher/2026-11-04_01-39-31_ns_v1/model_14100.pt \
  FE_STUDENT_RUN=fe_edge_nsa2 FE_STUDENT_TASK=Unitree-B2W-Depth-Student-Edge-RL \
  FE_STUDENT_FROM=$PWD/logs/rsl_rl/b2w_depth_student/fe_edge_ns1_avg1000/model_1000.pt \
  FE_STUDENT_ITERS=1500 FE_STUDENT_LR=1e-4 FE_STUDENT_LR_FINAL=5e-5 \
  FE_STUDENT_ARGS="--agent.algorithm.bc-coef-final=1.0 --agent.algorithm.critic-warmup=250 --agent.lr-ramp-iters=300" \
  DEPTH_STUDENT_HARD=1 FE_CHECK_AFTER_TRAIN=1 bash jobs/fe_chain.sh B2W
