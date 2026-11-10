#!/bin/bash
# b2_rl_after.sh PID (go2-depth 2026-11-02): once process PID (the As2W fe_edge_v4fric chain) has exited, the B2 chain
# with the student's PPO fine-tune (Unitree-B2-Depth-Student-Edge-RL, rl/student_ppo.py) from the v3dr student 1999,
# teacher pick avg6000, 2000 it, lr ceiling 1e-4 -> 5e-5, hard commands and DR edges as v3dr; checks after training.
cd /root/gpufree-data/go2-depth/depth_fe
while kill -0 $1 2> /dev/null; do sleep 60; done
echo "$(date '+%m-%d %H:%M:%S') B2 RL: process $1 (As2W chain) has ended; starting the B2 chain" >> logs/jobs/fe_b2.txt
exec env B2_N=5000 B2_PKG=/root/gpufree-data/go2-blind/export/legged_feetedge/b2_fe_mixed_5000 B2_PH=Mixed B2_PSTART=3000 \
  FE_STUDENT_TASK=Unitree-B2-Depth-Student-Edge-RL \
  FE_STUDENT_FROM=$PWD/logs/rsl_rl/b2_depth_student/2026-11-02_01-53-26_fe_edge_v3dr/model_1999.pt \
  FE_STUDENT_ITERS=2000 FE_STUDENT_LR=1e-4 FE_STUDENT_LR_FINAL=5e-5 FE_STUDENT_RUN=fe_edge_v4rl \
  DEPTH_STUDENT_HARD=1 DEPTH_STUDENT_DR_EDGE=0.4 FE_CHECK_AFTER_TRAIN=1 bash jobs/fe_chain.sh B2
