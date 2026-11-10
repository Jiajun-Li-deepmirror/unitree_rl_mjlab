#!/bin/bash
# nsa_as2w.sh (go2-depth 2026-11-04 ~13:50): As2W wheelahead round. The PPO student fe_edge_nsrl avg1499 passed the
# mjlab checks (3 seeds) and the tinynav slide gate (slid back 2.4%, start-landing steps 0.1 / climb) but steps its
# front wheels on the top landing 3.2 times a climb (6-13 cm, 0-40 cm past the last riser, while the rears climb: the
# trot partner of a climbing rear wheel, free in wheel_air_drivable because the riser just climbed sat in the front
# wheel's circle). Pilot + wheelahead; teacher from ns_v1's last (15000), 1500 it, checks +1000 / +1500; student:
# nsrl's recipe (PPO + BC 1.0, hard commands, DR edges 0.4, friction 0.3-1.6, long stands) from nsrl avg1499.
source /root/gpufree-data/go2-depth/depth_fe/jobs/ns_common.sh
LAST=$(ls logs/rsl_rl/as2w_depth_teacher/2026-11-04_05-39-29_ns_v1/model_*.pt | sort -V | tail -n 1)
exec env FE_PICKS=logs/picks_nsa AS2W_PKG=$PWD/logs/pkg/as2w_mixed_8000 FE_CONS_INIT= \
  FE_PILOT_EXTRA=,wheelstep,wheelstep4,stairspeed2,noslide,wheelahead FE_TEACHER_RUN=nsa_v1 FE_TEACHER_KS="1000 1500" \
  FE_TEACHER_FROM=$PWD/$LAST \
  FE_STUDENT_RUN=fe_edge_nsa1 FE_STUDENT_TASK=Unitree-As2W-Depth-Student-Edge-RL \
  FE_STUDENT_FROM=$PWD/logs/rsl_rl/as2w_depth_student/fe_edge_nsrl_avg1499/model_1499.pt \
  FE_STUDENT_ITERS=1500 FE_STUDENT_LR=1e-4 FE_STUDENT_LR_FINAL=5e-5 \
  FE_STUDENT_ARGS="--agent.algorithm.bc-coef-final=1.0 --agent.algorithm.critic-warmup=250 --agent.lr-ramp-iters=300" \
  DEPTH_STUDENT_HARD=1 DEPTH_STUDENT_DR_EDGE=0.4 DEPTH_STUDENT_FRICTION=0.3,1.6 DEPTH_STUDENT_STAND_LONG=1 \
  FE_CHECK_AFTER_TRAIN=1 bash jobs/fe_chain.sh As2W
