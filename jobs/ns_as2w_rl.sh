#!/bin/bash
# ns_as2w_rl.sh (go2-depth 2026-11-04 ~11:05): As2W noslide student, PPO stage. The distilled fe_edge_nsd carries the
# teacher's stairs (tinynav avg1000: slid back 6%, v4_fe 14%; landing steps 0.8 vs 3.0) but every average fails
# speed / drift-test falls / wheels-down (avg1999 s0 up 0.788). As B2W (ws1 -> ws1bc) and the As2W delivery (v5stand
# -> v7bc): PPO with BC held at 1.0 against the ns teacher (avg15000) on the round's rewards (stairspeed2 included),
# from nsd avg1999, hard commands, DR edges 0.4, friction 0.3-1.6, long stands, 1500 it, checks after training.
source /root/gpufree-data/go2-depth/depth_fe/jobs/ns_common.sh
exec env AS2W_PKG=$PWD/logs/pkg/as2w_mixed_8000 FE_CONS_INIT= FE_STUDENT_RUN=fe_edge_nsrl \
  FE_STUDENT_TASK=Unitree-As2W-Depth-Student-Edge-RL \
  FE_STUDENT_FROM=$PWD/logs/rsl_rl/as2w_depth_student/fe_edge_nsd_avg1999/model_1999.pt \
  FE_STUDENT_ITERS=1500 FE_STUDENT_LR=1e-4 FE_STUDENT_LR_FINAL=5e-5 \
  FE_STUDENT_ARGS="--agent.algorithm.bc-coef-final=1.0 --agent.algorithm.critic-warmup=250 --agent.lr-ramp-iters=300" \
  DEPTH_STUDENT_HARD=1 DEPTH_STUDENT_DR_EDGE=0.4 DEPTH_STUDENT_FRICTION=0.3,1.6 DEPTH_STUDENT_STAND_LONG=1 \
  FE_CHECK_AFTER_TRAIN=1 bash jobs/fe_chain.sh As2W
