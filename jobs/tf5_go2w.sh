#!/bin/bash
# tf5_go2w.sh (go2-depth 2026-11-05 ~05:00): Go2W slide-back round. The tf_v4 students keep the top landing clean
# (tinynav top 0.0-0.1 / climb) but slide back in 7.4% of the step-ups (bar 5%), PPO + BC and distilled alike. The
# slides (scratchpad check/stair_slide_diag.py, fe_edge_tf4d avg1000, 597 step-ups): 27% of the swings peak under the
# tread top and slide 21-25%; at +0..+3 cm 2.9%, above 0.5%; treads 3 (rise 0.186) and 12-13 (the last, deep) 16-18%.
# Now noslide2x on top of tf4's pilot (wheel_nosing -3 -> -6 per riser-contact second, wheel_slide_back -2 -> -4 per
# slide; B2W has it since tf_v2), from tf_v4's raw 24600 (the pick's last save), multipliers as tf_v4 left them
# (side_drift 2, clock_side 2), checks +800 / +1200 / +1600, margin 0.08 wanted; student by distillation (tf4d recipe)
# from fe_edge_tf4d avg1000.
source /root/gpufree-data/go2-depth/depth_fe/jobs/ns_common.sh
exec env FE_PICKS=logs/picks_tg5 GO2W_PKG=$PWD/logs/pkg/go2w_polish_11300 FE_CONS_INIT=side_drift:2.0,clock_side:2.0 FE_MARGIN=0.08 \
  FE_PILOT_EXTRA=,wheelstep,wheelstep4,stairspeed2,noslide,noslide2x,sidetgt75,topair,longexit,liftm8,stairspeed3 FE_TEACHER_RUN=tf_v5 FE_TEACHER_KS="800 1200 1600" \
  FE_TEACHER_FROM=$PWD/logs/rsl_rl/go2w_depth_teacher/2026-11-04_23-58-37_tf_v4/model_24600.pt \
  FE_STUDENT_RUN=fe_edge_tf5 \
  FE_STUDENT_FROM=$PWD/logs/rsl_rl/go2w_depth_student/fe_edge_tf4d_avg1000/model_1000.pt \
  FE_STUDENT_ITERS=2000 FE_STUDENT_LR=2e-4 FE_STUDENT_LR_FINAL=1e-4 \
  bash jobs/fe_chain.sh Go2W
