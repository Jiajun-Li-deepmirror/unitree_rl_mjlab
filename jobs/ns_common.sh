# ns_common.sh (sourced by ns_<r>.sh; go2-depth 2026-11-04 ~01:20, the user: "继续训练当前问题，尽量做到不会出现滑回问题"):
# the noslide round. Teacher fine-tuned from the wheelstep round's teacher with BLIND_PILOT + wheelstep, wheelstep4,
# stairspeed2, noslide (mdp.wheel_slide_back 2 per slide, wheel_nosing over every riser contact at 3 per second, the
# riser lift paid to 5 cm over the top), 2000 it, checks at +1000/1500/2000, the last passing one with the margin; then
# the student by PPO with BC against it on the same rewards (Student-Edge-RL). Picks logs/picks_ns.
cd /root/gpufree-data/go2-depth/depth_fe
export FE_PICKS=logs/picks_ns FE_PILOT_EXTRA=${NS_PILOT:-,wheelstep,wheelstep4,stairspeed2,noslide} FE_TEACHER_RUN=ns_v1 \
  FE_TEACHER_KS="1000 1500 2000" FE_TEACHER_PICK=last FE_STUDENT_RUN=fe_edge_ns1 \
  FE_LOCK=${FE_LOCK:-/root/gpufree-data/go2-depth/ws_check.lock}
