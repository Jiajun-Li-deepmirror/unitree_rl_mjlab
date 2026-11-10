#!/bin/bash
# ws_b2w_rl2.sh (go2-depth 2026-11-03 ~21:15): the B2W wheelstep student, round 2. The distilled fe_edge_ws1 (teacher
# ws_v2 avg13100) carries the fix into tinynav-sim (deployed metadata: landing steps 3.0 -> 0.7-0.8 per climb, clean
# 46 -> 50-52%, slid back 17 -> 16-17%) but every average fails its stops on creep (back 1.0: 1.04-1.34 cm, bar 1;
# the teacher 0.30, the delivered student 0.33). The As2W cure for student stop errors (v7bc): the student's PPO with
# the BC held at 1.0, more stops after fast commands (DEPTH_STUDENT_HARD), from the distilled avg1999. The first
# RL try from the delivered student (fe_edge_ws1rl, default BC decay) still stepped 3.0 per climb at 1000: stopped.
cd /root/gpufree-data/go2-depth/depth_fe
mkdir -p logs/picks_ws_rl2
[ -f logs/picks_ws_rl2/b2w_teacher ] || echo logs/rsl_rl/b2w_depth_teacher/ws_v2_avg13100/model_13100.pt > logs/picks_ws_rl2/b2w_teacher
exec env FE_PICKS=logs/picks_ws_rl2 FE_PILOT_EXTRA=,wheelstep,wheelstep4 FE_TEACHER_RUN=ws_v2 \
  FE_LOCK=${FE_LOCK:-/root/gpufree-data/go2-depth/ws_check.lock} \
  B2W_PKG=$PWD/logs/pkg/b2w_polish_9000_sidecmd_desc FE_CONS_INIT=back_drift:1.68,descent:1.90 FE_MARGIN=0.07 \
  FE_STUDENT_TASK=Unitree-B2W-Depth-Student-Edge-RL \
  FE_STUDENT_FROM=$PWD/logs/rsl_rl/b2w_depth_student/fe_edge_ws1_avg1999/model_1999.pt \
  FE_STUDENT_ITERS=1500 FE_STUDENT_LR=1e-4 FE_STUDENT_LR_FINAL=5e-5 FE_STUDENT_RUN=fe_edge_ws1bc \
  FE_STUDENT_ARGS="--agent.algorithm.bc-coef-final=1.0 --agent.algorithm.critic-warmup=250 --agent.lr-ramp-iters=300" \
  DEPTH_STUDENT_HARD=1 \
  bash jobs/fe_chain.sh B2W
