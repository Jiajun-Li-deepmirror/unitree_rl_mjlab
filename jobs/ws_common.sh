# ws_common.sh (sourced by ws_<r>.sh; go2-depth 2026-11-03, the user's wheeled-stairs faults: rear wheels stepping on the
# landing, wheels rolling back off a nosing): the shared settings of the "wheelstep" round. The picked teacher is
# fine-tuned with BLIND_PILOT + wheelstep (mdp.wheel_nosing, wheel_riser_lift land_share, per-wheel even ground for
# wheel_air_drivable / foot_swing_clearance) for 2000 iterations, the averages at +1000 / +1500 / +2000 are checked
# (fe_check.sh teacher, seed 0) and the last passing one with the speed margin is picked; then the delivered student
# is re-trained on it with its own round's recipe (seeds 0 1 2). Picks in logs/picks_ws (the delivered logs/picks stay).
cd /root/gpufree-data/go2-depth/depth_fe
# Round 2 (15:55): + wheelstep4 (the near-stairs lifts at 4x); WS_FROM_<R> (a ws_v1 checkpoint of that robot) restarts the teacher there.
# Round 3 default (17:35): + stairspeed2 (the stairs speed deficit at 2x): wheelstep4 slowed the Go2W climb to 0.78-0.80.
export FE_PICKS=logs/picks_ws FE_PILOT_EXTRA=${WS_PILOT:-,wheelstep,wheelstep4,stairspeed2} FE_TEACHER_RUN=${WS_RUN:-ws_v3} FE_TEACHER_KS="1000 1500 2000" \
  FE_TEACHER_PICK=last FE_STUDENT_RUN=fe_edge_ws1 FE_LOCK=${FE_LOCK:-/root/gpufree-data/go2-depth/ws_check.lock}
