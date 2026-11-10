#!/bin/bash
# ns_b2w.sh: B2W noslide round (ns_common.sh): teacher from ws_v2 model_13100 (multipliers ~1.68 / 1.90, margin 0.07);
# student PPO + BC 1.0 from fe_edge_ws1bc avg1000 (creep fixed; hard commands, DR edges 0.4), 1500 it. Not from ws1dr
# and without its foot friction 0.3-1.6: ws1dr climbs box faces in tinynav-sim (walk_box_go roll / pitch 15 / 21 deg,
# ws1bc 2 / 3; the stairs task draws 0.4-1.0 because high wheel friction lets a wheel climb a vertical face).
source /root/gpufree-data/go2-depth/depth_fe/jobs/ns_common.sh
exec env B2W_PKG=$PWD/logs/pkg/b2w_polish_9000_sidecmd_desc FE_CONS_INIT=back_drift:1.68,descent:1.90 FE_MARGIN=0.07 \
  FE_TEACHER_FROM=$PWD/logs/rsl_rl/b2w_depth_teacher/2026-11-03_16-13-37_ws_v2/model_13100.pt \
  FE_STUDENT_TASK=Unitree-B2W-Depth-Student-Edge-RL \
  FE_STUDENT_FROM=$PWD/logs/rsl_rl/b2w_depth_student/fe_edge_ws1bc_avg1000/model_1000.pt \
  FE_STUDENT_ITERS=1500 FE_STUDENT_LR=1e-4 FE_STUDENT_LR_FINAL=5e-5 \
  FE_STUDENT_ARGS="--agent.algorithm.bc-coef-final=1.0 --agent.algorithm.critic-warmup=250 --agent.lr-ramp-iters=300" \
  DEPTH_STUDENT_HARD=1 DEPTH_STUDENT_DR_EDGE=0.4 \
  bash jobs/fe_chain.sh B2W
