#!/bin/bash
# tf6d_as2w.sh (go2-depth 2026-11-05 06:15, t2): As2W topairup student by distillation. tf6_as2w.sh's PPO+BC student
# (from nsrl avg1499) OOM'd on t1 at ~it 200 beside the Go2W tf_v5 teacher and checks; PPO+BC students kept their old
# top-landing steps on B2W (tf3: 5.4 / climb), so this one distills (Student-Edge, as B2W tf3d) from nsrl avg1499
# toward the tf_v6 pick avg16200 (copied from t1, md5 71f8fa9e). Student-only: the teacher pick is copied in.
source /root/gpufree-data/go2-depth/depth_fe/jobs/ns_common.sh
mkdir -p logs/picks_ta6d && echo logs/rsl_rl/as2w_depth_teacher/tf_v6_avg16200/model_16200.pt > logs/picks_ta6d/as2w_teacher
exec env FE_PICKS=logs/picks_ta6d AS2W_PKG=$PWD/logs/pkg/as2w_mixed_8000 FE_CONS_INIT= \
  FE_PILOT_EXTRA=,wheelstep,wheelstep4,stairspeed2,noslide,topairup,longexit FE_TEACHER_RUN=tf_v6 \
  FE_STUDENT_RUN=fe_edge_tf6d \
  FE_STUDENT_FROM=$PWD/logs/rsl_rl/as2w_depth_student/fe_edge_nsrl_avg1499/model_1499.pt \
  FE_STUDENT_ITERS=2000 FE_STUDENT_LR=2e-4 FE_STUDENT_LR_FINAL=1e-4 FE_STUDENT_FREE=14000 \
  DEPTH_STUDENT_HARD=1 bash jobs/fe_chain.sh As2W
