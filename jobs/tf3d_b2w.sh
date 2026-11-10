#!/bin/bash
# tf3d_b2w.sh (go2-depth 2026-11-05 ~04:50): B2W student by distillation under the tf_v3 teacher pick (avg15900: top_flat
# lifted 17.3% / 39 per 100 wheel-s, ws_stair top landing 8.4% (nsa_v2 17.8%), slide back 7.4%). The PPO + BC 1.0
# student fe_edge_tf3 (from v6_ns) passed mjlab on 3 seeds but took nothing of the new teacher's top landing: tinynav
# 24 climbs top 5.4 / climb (v6_ns 5.0), slid back 6.2%. Distillation carried its teacher for As2W (nsd: slid 6% vs
# PPO 17%). Student-Edge from fe_edge_tf3 avg1000, 2000 it, lr 2e-4 -> 1e-4 (ws_b2w.sh), hard commands, no DR edges
# (DR_EDGE 0.4 made the B2W students climb boxes), picks_tb3d (teacher pick copied).
source /root/gpufree-data/go2-depth/depth_fe/jobs/ns_common.sh
mkdir -p logs/picks_tb3d && cp logs/picks_tb3/b2w_teacher logs/picks_tb3d/b2w_teacher
exec env FE_PICKS=logs/picks_tb3d B2W_PKG=$PWD/logs/pkg/b2w_polish_9000_sidecmd_desc \
  FE_CONS_INIT=back_drift:1.7,descent:2.3 \
  FE_PILOT_EXTRA=,wheelstep,wheelstep4,stairspeed2,noslide,noslide2x,wheelahead,bdhold,topair,longexit FE_TEACHER_RUN=tf_v3 \
  FE_STUDENT_RUN=fe_edge_tf3d \
  FE_STUDENT_FROM=$PWD/logs/rsl_rl/b2w_depth_student/fe_edge_tf3_avg1000/model_1000.pt \
  FE_STUDENT_ITERS=2000 FE_STUDENT_LR=2e-4 FE_STUDENT_LR_FINAL=1e-4 FE_STUDENT_FREE=15000 \
  DEPTH_STUDENT_HARD=1 bash jobs/fe_chain.sh B2W
