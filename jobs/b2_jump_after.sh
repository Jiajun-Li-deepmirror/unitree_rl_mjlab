#!/bin/bash
# b2_jump_after.sh (go2-depth 2026-11-02, user: the command-jump BC round as a backup, on a GPU that frees up): once
# the As2W student training on this box (t1) has ended, the B2 student PPO round 3 here: from v4rl model_1500 (the
# best so far), BC x8 on the steps after a command jump (go2-home: every B2 student spun up harder than its teacher
# in the first 0.12 s), hard commands, DR edges (with the PD gains), spins left / right even again; checks after
# training, with the deployed yaw ramp (fe_check.sh, the user's B2 acceptance from 11-02).
cd /root/gpufree-data/go2-depth/depth_fe
while ps -eo args | grep -qE '^[^ ]*python[^ ]* scripts/train[.]py Unitree-As2W-Depth-'; do sleep 60; done
echo "$(date '+%m-%d %H:%M:%S') B2 jump: the As2W training on t1 has ended; starting the B2 chain" >> logs/jobs/fe_b2.txt
# (15:31 11-02: B2_PKG first pointed at go2-blind/export/..., absent on t1: no pilot, and the teacher's critic did
# not load, 495 vs 483)
[ -s logs/pkg/b2_fe_mixed_5000/blind_pilot.txt ] || { echo "no B2 pilot"; exit 1; }
exec env FE_LOCK=/root/gpufree-data/go2-blind/check.lock FE_SERIAL=1 \
  B2_N=5000 B2_PKG=$PWD/logs/pkg/b2_fe_mixed_5000 B2_PH=Mixed B2_PSTART=3000 \
  FE_STUDENT_TASK=Unitree-B2-Depth-Student-Edge-RL \
  FE_STUDENT_FROM=$PWD/logs/rsl_rl/b2_depth_student/2026-11-02_10-00-35_fe_edge_v4rl/model_1500.pt \
  FE_STUDENT_ITERS=1500 FE_STUDENT_LR=1e-4 FE_STUDENT_LR_FINAL=5e-5 FE_STUDENT_RUN=fe_edge_v6jump \
  FE_STUDENT_ARGS=--agent.algorithm.bc-jump-weight=8 \
  DEPTH_STUDENT_HARD=1 DEPTH_STUDENT_DR_EDGE=0.4 FE_CHECK_AFTER_TRAIN=1 bash jobs/fe_chain.sh B2
