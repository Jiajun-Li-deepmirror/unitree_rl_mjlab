#!/bin/bash
# tf3_as2w.sh (go2-depth 2026-11-04 ~19:10): As2W topflat30 + longexit (walled up-flights end in 1.4 m of flat, map3 1.3 m)
# + gaitw1 (foot_gait 2x): -8 / -30 on the standard terrain moved the teachers' top-flat lifts only 35 -> 30-31%
# (top_flat_eval, 1024 x 60 s; that state ~100 wheel-s per 61k robot-s), and tf_v1 (-8) failed the clock at (0, 0.6)
# 77% (nsa_v1 81%, its students 78-80% at (-1, 0, +-0.2), bar 80%). Otherwise as tf_as2w.sh. Its header:
# As2W topflat round. wheelahead did not touch the top-landing steps (tinynav
# As2W students nsrl 3.2 -> nsa1 4.0 per climb; slid back 2.4% both); the teachers lift 28-32% of their time on flat
# ground ahead next to stairs (jobs/top_flat_eval.py). Teacher from nsa_v1's last (16500) + topflat (mdp.
# wheel_lift_flat_ahead -8 per wheel-second), 1000 it, checks +500 / +1000, pick last; student PPO + BC 1.0 (the nsa
# recipe) from nsrl avg1499 (as2w_depth_v5_ns, deployed). On t2 next to Go2W; the student waits for 14 GB free.
source /root/gpufree-data/go2-depth/depth_fe/jobs/ns_common.sh
exec env FE_PICKS=logs/picks_tf3 AS2W_PKG=$PWD/logs/pkg/as2w_mixed_8000 FE_CONS_INIT= \
  FE_PILOT_EXTRA=,wheelstep,wheelstep4,stairspeed2,noslide,wheelahead,topflat,topflat30,longexit,gaitw1 FE_TEACHER_RUN=tf_v3 FE_TEACHER_KS="800 1200" \
  FE_TEACHER_FROM=$PWD/logs/rsl_rl/as2w_depth_teacher/2026-11-04_13-34-45_nsa_v1/model_16500.pt \
  FE_STUDENT_RUN=fe_edge_tf3 FE_STUDENT_TASK=Unitree-As2W-Depth-Student-Edge-RL \
  FE_STUDENT_FROM=$PWD/logs/rsl_rl/as2w_depth_student/fe_edge_nsrl_avg1499/model_1499.pt \
  FE_STUDENT_ITERS=1500 FE_STUDENT_LR=1e-4 FE_STUDENT_LR_FINAL=5e-5 FE_STUDENT_FREE=14000 \
  FE_STUDENT_ARGS="--agent.algorithm.bc-coef-final=1.0 --agent.algorithm.critic-warmup=250 --agent.lr-ramp-iters=300" \
  DEPTH_STUDENT_HARD=1 DEPTH_STUDENT_DR_EDGE=0.4 DEPTH_STUDENT_FRICTION=0.3,1.6 DEPTH_STUDENT_STAND_LONG=1 \
  FE_CHECK_AFTER_TRAIN=1 bash jobs/fe_chain.sh As2W
