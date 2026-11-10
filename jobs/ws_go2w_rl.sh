#!/bin/bash
# ws_go2w_rl.sh (go2-depth 2026-11-04 ~00:10): the Go2W wheelstep student by PPO fine-tune, for its climb speed. The
# distilled fe_edge_ws1 (teacher ws_v3 avg19100: up 0.819, margin 0.019) fixed the landing steps (tinynav 2.7 -> 0.0
# per climb) but climbs under the bar (avg1000 / 1500 s0 up 0.790 / 0.796; bar 0.8). The student's own PPO on the
# round's rewards (wheelstep, wheelstep4, stairspeed2: the stairs speed deficit at 2x) with the BC decaying to 0.5
# (looser than As2W's 1.0, tighter than the default 0.3) from the distilled avg1999, 1000 it, critic warmup 250,
# lr ramp 300. Picks logs/picks_ws_go2w_rl.
cd /root/gpufree-data/go2-depth/depth_fe
mkdir -p logs/picks_ws_go2w_rl
[ -f logs/picks_ws_go2w_rl/go2w_teacher ] || echo logs/rsl_rl/go2w_depth_teacher/ws_v3_avg19100/model_19100.pt > logs/picks_ws_go2w_rl/go2w_teacher
exec env FE_PICKS=logs/picks_ws_go2w_rl FE_PILOT_EXTRA=,wheelstep,wheelstep4,stairspeed2 FE_TEACHER_RUN=ws_v3 \
  FE_LOCK=${FE_LOCK:-/root/gpufree-data/go2-depth/ws_check.lock} GO2W_PKG=$PWD/logs/pkg/go2w_polish_11300 FE_CONS_INIT= \
  FE_STUDENT_TASK=Unitree-Go2W-Depth-Student-Edge-RL \
  FE_STUDENT_FROM=$PWD/logs/rsl_rl/go2w_depth_student/fe_edge_ws1_avg1999/model_1999.pt \
  FE_STUDENT_ITERS=1000 FE_STUDENT_LR=1e-4 FE_STUDENT_LR_FINAL=5e-5 FE_STUDENT_RUN=fe_edge_ws1rl \
  FE_STUDENT_ARGS="--agent.algorithm.bc-coef-final=0.5 --agent.algorithm.critic-warmup=250 --agent.lr-ramp-iters=300" \
  bash jobs/fe_chain.sh Go2W
