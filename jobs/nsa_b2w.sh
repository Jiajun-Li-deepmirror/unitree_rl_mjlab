#!/bin/bash
# nsa_b2w.sh (go2-depth 2026-11-04 ~11:40, the user: B2W's front wheels still step on the top landing; "使用t2 gpu"):
# every wheeled student old and new steps its front wheels on map3's top landing (tinynav: B2W 5-6 per climb, 13-20
# cm, every ~0.6 s up to 70 cm past the last riser), and so does the teacher (ns_v1 avg14100 on stairs terrain: front
# wheels on ground above the base lifted 35.6% of the drivable time). Pilot + wheelahead: the per-wheel even ground
# only from the terrain ahead in the commanded direction, so a riser just climbed no longer frees the lifts.
# Teacher from ns_v1 model_14100 (1500 it, checks +1000 / +1500, the last passing with the margin); student distilled
# from b2w_depth_v6_ns (ns1 avg1000), 1500 it, hard commands (no DR edges / friction: suspected box climbing).
source /root/gpufree-data/go2-depth/depth_fe/jobs/ns_common.sh
exec env FE_PICKS=logs/picks_nsa B2W_PKG=$PWD/logs/pkg/b2w_polish_9000_sidecmd_desc \
  FE_CONS_INIT=back_drift:1.68,descent:1.90 FE_MARGIN=0.07 \
  FE_PILOT_EXTRA=,wheelstep,wheelstep4,stairspeed2,noslide,wheelahead FE_TEACHER_RUN=nsa_v1 FE_TEACHER_KS="1000 1500" \
  FE_TEACHER_FROM=$PWD/logs/rsl_rl/b2w_depth_teacher/2026-11-04_01-39-31_ns_v1/model_14100.pt \
  FE_STUDENT_RUN=fe_edge_nsa1 \
  FE_STUDENT_FROM=$PWD/logs/rsl_rl/b2w_depth_student/fe_edge_ns1_avg1000/model_1000.pt \
  FE_STUDENT_ITERS=1500 FE_STUDENT_LR=2e-4 FE_STUDENT_LR_FINAL=1e-4 DEPTH_STUDENT_HARD=1 \
  bash jobs/fe_chain.sh B2W
