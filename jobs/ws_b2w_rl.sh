#!/bin/bash
# ws_b2w_rl.sh (go2-depth 2026-11-03 ~20:00): the B2W wheelstep student by PPO fine-tune instead of distillation.
# The distilled student (fe_edge_ws1, from the delivered v4desc avg2000 on the ws_v2 teacher avg13100) slid back off
# the nosings in 43-44% of its tinynav-sim step-ups at 1000 / 1500 iterations (delivered student 17%) although the
# teacher beat the old one on every stairs metric (riser contact 22.8 -> 18.6, clean 44 -> 52%, apex over riser +2.0
# -> +4.4 cm): the guess is that it cannot copy the teacher's per-wheel timing from its camera, so here the student's
# own PPO on the wheelstep rewards (Student-Edge-RL, the default BC decay so the rewards count; the As2W v6rl/v7bc
# runner with critic warmup 250 and a 300-iteration lr ramp) starts from the delivered student.
# Teacher pick in logs/picks_ws_rl/b2w_teacher (the ws_v2 avg13100); the pilot as that teacher trained (wheelstep,
# wheelstep4; no stairspeed2), its multipliers 1.68 / 1.90.
cd /root/gpufree-data/go2-depth/depth_fe
mkdir -p logs/picks_ws_rl
[ -f logs/picks_ws_rl/b2w_teacher ] || echo logs/rsl_rl/b2w_depth_teacher/ws_v2_avg13100/model_13100.pt > logs/picks_ws_rl/b2w_teacher
exec env FE_PICKS=logs/picks_ws_rl FE_PILOT_EXTRA=,wheelstep,wheelstep4 FE_TEACHER_RUN=ws_v2 \
  FE_LOCK=${FE_LOCK:-/root/gpufree-data/go2-depth/ws_check.lock} \
  B2W_PKG=$PWD/logs/pkg/b2w_polish_9000_sidecmd_desc FE_CONS_INIT=back_drift:1.68,descent:1.90 FE_MARGIN=0.07 \
  FE_STUDENT_TASK=Unitree-B2W-Depth-Student-Edge-RL \
  FE_STUDENT_FROM=$PWD/logs/rsl_rl/b2w_depth_student/fe_edge_v4desc_avg2000/model_2000.pt \
  FE_STUDENT_ITERS=1500 FE_STUDENT_LR=1e-4 FE_STUDENT_LR_FINAL=5e-5 FE_STUDENT_RUN=fe_edge_ws1rl \
  FE_STUDENT_ARGS="--agent.algorithm.critic-warmup=250 --agent.lr-ramp-iters=300" \
  bash jobs/fe_chain.sh B2W
