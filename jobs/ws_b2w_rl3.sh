#!/bin/bash
# ws_b2w_rl3.sh (go2-depth 2026-11-03 ~22:55): B2W wheelstep student, round 3. The PPO+BC 1.0 student (fe_edge_ws1bc)
# fixed the stop creep (back 1.0: 0.42 cm) but every average still falls 1/256 on the side 0.6 stop. fall_diag: env 905
# payload 8.5 kg (p89), CoM y +4.7 cm (p98), friction 1.59 (p98), kd low (p7); the delivered student 0/256. The As2W
# cure (v3dr / v4fric): more robots at the range ends (DEPTH_STUDENT_DR_EDGE 0.4) and the foot friction over 0.3-1.6 in
# the student's training (its stairs task draws 0.4-1.0). From ws1bc avg1499, 1000 it, BC 1.0, hard commands.
cd /root/gpufree-data/go2-depth/depth_fe
mkdir -p logs/picks_ws_rl3
[ -f logs/picks_ws_rl3/b2w_teacher ] || echo logs/rsl_rl/b2w_depth_teacher/ws_v2_avg13100/model_13100.pt > logs/picks_ws_rl3/b2w_teacher
exec env FE_PICKS=logs/picks_ws_rl3 FE_PILOT_EXTRA=,wheelstep,wheelstep4 FE_TEACHER_RUN=ws_v2 \
  FE_LOCK=${FE_LOCK:-/root/gpufree-data/go2-depth/ws_check.lock} \
  B2W_PKG=$PWD/logs/pkg/b2w_polish_9000_sidecmd_desc FE_CONS_INIT=back_drift:1.68,descent:1.90 FE_MARGIN=0.07 \
  FE_STUDENT_TASK=Unitree-B2W-Depth-Student-Edge-RL \
  FE_STUDENT_FROM=$PWD/logs/rsl_rl/b2w_depth_student/fe_edge_ws1bc_avg1499/model_1499.pt \
  FE_STUDENT_ITERS=1000 FE_STUDENT_LR=1e-4 FE_STUDENT_LR_FINAL=5e-5 FE_STUDENT_RUN=fe_edge_ws1dr \
  FE_STUDENT_ARGS="--agent.algorithm.bc-coef-final=1.0 --agent.algorithm.critic-warmup=250 --agent.lr-ramp-iters=300" \
  DEPTH_STUDENT_HARD=1 DEPTH_STUDENT_DR_EDGE=0.4 DEPTH_STUDENT_FRICTION=0.3,1.6 \
  bash jobs/fe_chain.sh B2W
