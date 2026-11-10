#!/bin/bash
# nsa_go2w.sh (go2-depth 2026-11-04 ~13:50): Go2W wheelahead round. Deployed v5_ws fails the tinynav slide gate
# (slid back 11.4%, top-landing steps 2.6 / climb); its PPO / distilled ns students did not slide less (12-13%). As
# for As2W (distilled nsd 6.4% -> PPO + BC 1.0 nsrl 2.4%): teacher with pilot + wheelahead from ns_v2's last (22600),
# 1500 it, checks +1000 / +1500; student PPO + BC 1.0 (hard commands, DR edges 0.4, friction 0.3-1.6, long stands)
# from the distilled nsd avg1000.
source /root/gpufree-data/go2-depth/depth_fe/jobs/ns_common.sh
LAST=$(ls logs/rsl_rl/go2w_depth_teacher/2026-11-04_03-07-36_ns_v2/model_*.pt | sort -V | tail -n 1)
exec env FE_PICKS=logs/picks_nsa GO2W_PKG=$PWD/logs/pkg/go2w_polish_11300 FE_CONS_INIT= \
  FE_PILOT_EXTRA=,wheelstep,wheelstep4,stairspeed2,noslide,sidetgt75,wheelahead FE_TEACHER_RUN=nsa_v1 FE_TEACHER_KS="1000 1500" \
  FE_TEACHER_FROM=$PWD/$LAST \
  FE_STUDENT_RUN=fe_edge_nsa1 FE_STUDENT_TASK=Unitree-Go2W-Depth-Student-Edge-RL \
  FE_STUDENT_FROM=$PWD/logs/rsl_rl/go2w_depth_student/fe_edge_nsd_avg1000/model_1000.pt \
  FE_STUDENT_ITERS=1500 FE_STUDENT_LR=1e-4 FE_STUDENT_LR_FINAL=5e-5 \
  FE_STUDENT_ARGS="--agent.algorithm.bc-coef-final=1.0 --agent.algorithm.critic-warmup=250 --agent.lr-ramp-iters=300" \
  DEPTH_STUDENT_HARD=1 DEPTH_STUDENT_DR_EDGE=0.4 DEPTH_STUDENT_FRICTION=0.3,1.6 DEPTH_STUDENT_STAND_LONG=1 \
  FE_CHECK_AFTER_TRAIN=1 bash jobs/fe_chain.sh Go2W
