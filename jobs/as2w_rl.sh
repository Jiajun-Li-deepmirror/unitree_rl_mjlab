#!/bin/bash
# as2w_rl.sh (go2-depth 2026-11-02, my call under the user's "按你的推荐"): the As2-W student's PPO fine-tune against its
# teacher (Unitree-As2W-Depth-Student-Edge-RL, as the B2's v4rl that passed), from v5stand model_1000 (seeds 0 / 1 pass
# with the deployed ramps; s2 fell standing after a fwd 1.5 stop, as avg1500 s1: the CoM-edge robots), on t1 beside
# the v5stand chain's last checks; checks after training.
cd /root/gpufree-data/go2-depth/depth_fe
[ -s logs/pkg/as2w_mixed_8000/blind_pilot.txt ] || { echo "no As2W pilot"; exit 1; }
echo "$(date '+%m-%d %H:%M:%S') As2W RL: starting the As2W chain with the student PPO fine-tune (run fe_edge_v6rl)" >> logs/jobs/fe_as2w.txt
exec env FE_LOCK=/root/gpufree-data/go2-blind/check.lock FE_SERIAL=1 AS2W_PKG=$PWD/logs/pkg/as2w_mixed_8000 \
  FE_STUDENT_TASK=Unitree-As2W-Depth-Student-Edge-RL \
  FE_STUDENT_FROM=$PWD/logs/rsl_rl/as2w_depth_student/2026-11-02_14-03-46_fe_edge_v5stand/model_1000.pt \
  FE_STUDENT_ITERS=1500 FE_STUDENT_LR=1e-4 FE_STUDENT_LR_FINAL=5e-5 FE_STUDENT_RUN=fe_edge_v6rl \
  DEPTH_STUDENT_HARD=1 DEPTH_STUDENT_DR_EDGE=0.4 DEPTH_STUDENT_FRICTION=0.3,1.6 DEPTH_STUDENT_STAND_LONG=1 \
  FE_CHECK_AFTER_TRAIN=1 bash jobs/fe_chain.sh As2W
