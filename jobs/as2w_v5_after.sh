#!/bin/bash
# as2w_v5_after.sh (go2-depth 2026-11-02, user "先把感知模型全部训练好"): once the B2 student training on this box
# (t1) has ended, the As2W student round 5 here: from v4fric model_1999 (whose seed-1 stop falls came 5-6 s into
# the stand: DEPTH_STUDENT_STAND_LONG), friction 0.3-1.6, hard commands, DR edges (now with the PD gains).
cd /root/gpufree-data/go2-depth/depth_fe
while ps -eo args | grep -qE '^[^ ]*python[^ ]* scripts/train[.]py Unitree-B2-Depth-'; do sleep 60; done
echo "$(date '+%m-%d %H:%M:%S') As2W v5: the B2 training on t1 has ended; starting the As2W chain on t1" >> logs/jobs/fe_as2w.txt
exec env FE_LOCK=/root/gpufree-data/go2-blind/check.lock AS2W_PKG=$PWD/logs/pkg/as2w_mixed_8000 \
  FE_STUDENT_FROM=$PWD/logs/rsl_rl/as2w_depth_student/2026-11-02_09-14-31_fe_edge_v4fric/model_1999.pt \
  FE_STUDENT_ITERS=2000 FE_STUDENT_LR=1e-4 FE_STUDENT_LR_FINAL=1e-4 FE_STUDENT_RUN=fe_edge_v5stand \
  DEPTH_STUDENT_HARD=1 DEPTH_STUDENT_DR_EDGE=0.4 DEPTH_STUDENT_FRICTION=0.3,1.6 DEPTH_STUDENT_STAND_LONG=1 \
  FE_CHECK_AFTER_TRAIN=1 FE_SERIAL=1 bash jobs/fe_chain.sh As2W
