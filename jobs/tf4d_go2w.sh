#!/bin/bash
# tf4d_go2w.sh (go2-depth 2026-11-05 ~03:20): Go2W student by distillation under the tf_v4 teacher pick (avg24600: up /
# down 0.834 / 0.920, top_flat lifted 22.8% / 54 per 100 wheel-s, ws_stair top landing 17.5%, slide back 3.3%). The
# PPO + BC 1.0 students climb ~0.04-0.07 slower than their teachers and fail map3 上楼慢 (tf2: 0.775 / 0.764 under 0.807;
# tf4 avg1000: 0.766 under 0.834); the one delivered Go2W student (v5_ws, 0.81) was a distillation (2000 it, lr 2e-4 ->
# 1e-4). Student-Edge (distillation) from fe_edge_tf4 avg1499, same pilot as tf4, picks_tg4d (teacher pick copied).
source /root/gpufree-data/go2-depth/depth_fe/jobs/ns_common.sh
mkdir -p logs/picks_tg4d && cp logs/picks_tg4/go2w_teacher logs/picks_tg4d/go2w_teacher
exec env FE_PICKS=logs/picks_tg4d GO2W_PKG=$PWD/logs/pkg/go2w_polish_11300 FE_CONS_INIT=side_drift:2.0,clock_side:2.0 \
  FE_PILOT_EXTRA=,wheelstep,wheelstep4,stairspeed2,noslide,sidetgt75,topair,longexit,liftm8,stairspeed3 FE_TEACHER_RUN=tf_v4 \
  FE_STUDENT_RUN=fe_edge_tf4d \
  FE_STUDENT_FROM=$PWD/logs/rsl_rl/go2w_depth_student/fe_edge_tf4_avg1499/model_1499.pt \
  FE_STUDENT_ITERS=2000 FE_STUDENT_LR=2e-4 FE_STUDENT_LR_FINAL=1e-4 \
  bash jobs/fe_chain.sh Go2W
