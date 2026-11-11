# fe_robot.sh R (sourced; depth_fe, 2026-10-31): per robot, go2-home's blind delivery (PKG, its model_N, its training
# phase PH: the teacher trains Unitree-<R>-Depth-<PH>), PSTART (BLIND_PHASE_START_IT of a mixed-phase delivery: where
# its stairs ramp started), and the student the -Edge student warm-starts from (OCC):
#   legged: the deployed occlusion student (tinynav-sim <r>_depth_v3_occ = occ_v1 model_1999);
#   wheeled: the deployed wheeled student (tinynav-sim <r>_depth_v4, 10-29 depth_wheeled picks; no occlusion round).
# FE_LOCK: the check lock of this GPU (t2: go2-home's, so our checks and theirs take turns).
E=/root/gpufree-data/go2-blind/export/legged_feetedge
EW=/root/gpufree-data/go2-blind/export/wheeled_blind
PH=Polish PSTART="" WHEELED=0
unset BLIND_CONS_INIT
case $1 in
  Go2) PKG=$E/go2_fe_polish_10000 N=10000
       # Its polish (legcons) ended with these multipliers at their caps from 9500 on (go2-home 10-31); leg_contact
       # starts at 1: the old code counted any touch, the delivered one > 5 N.
       export BLIND_CONS_INIT=clock_side:3,side_drift:3,feet_edge:2 ;;
  As2) PKG=$E/as2_fe_polish_12000 N=12000 ;;
  # B2: set at delivery (B2_N; B2_PKG / B2_PH / B2_PSTART when it is not a polish package b2_fe_polish_<N>).
  B2)  N=${B2_N:-0} PKG=${B2_PKG:-$E/b2_fe_polish_${B2_N:-0}} PH=${B2_PH:-Polish} PSTART=${B2_PSTART:-} ;;
  # Wheeled deliveries of 10-31 (go2-home): Go2W the polish average 10600..11300 (wheelcons at ~1x over that span, so
  # every multiplier starts at 1); As2W the stairs-phase (mixed) average 7300..8000, its ramp from the teacher's 4000.
  Go2W) PKG=${GO2W_PKG:-$EW/go2w_polish_11300} N=${GO2W_N:-11300} WHEELED=1 ;;   # GO2W_PKG: a copy (t1 has no export)
  # As2W (11-09): AS2W_N / AS2W_PH / AS2W_PSTART for a delivery of the new pipeline (v2_robot.sh), as GO2W_N
  As2W) PKG=${AS2W_PKG:-$EW/as2w_mixed_8000} N=${AS2W_N:-8000} PH=${AS2W_PH:-Mixed} PSTART=${AS2W_PSTART-4000} WHEELED=1 ;;   # AS2W_PKG: a copy (t1 has no export)
  # B2W (go2-home 11-01): the polish average 8300..9000 (stairs35 + b2wcons, leghold20), copied into depth_fe (t1 has
  # no wheeled_blind export). Its back_drift multiplier sat at a median 1.35 over 8300..9000, descent at 1.0.
  B2W) PKG=${B2W_PKG:-/root/gpufree-data/go2-depth/depth_fe/logs/pkg/b2w_polish_9000} N=9000 WHEELED=1
       export BLIND_CONS_INIT=back_drift:1.35 ;;
  *) echo "fe_robot.sh: unknown robot $1" >&2; return 2 ;;
esac
# FE_PKG / FE_N / FE_PH / FE_PSTART (2026-11-10, jobs/full_chain.sh): the blind delivery the depth chain starts from,
# for any robot (the reproducible run's own package instead of the robot's earlier one)
[ -n "${FE_PKG:-}" ] && PKG=$FE_PKG
[ -n "${FE_N:-}" ] && N=$FE_N
[ -n "${FE_PH:-}" ] && PH=$FE_PH
[ -n "${FE_PSTART+x}" ] && PSTART=$FE_PSTART
# Payload (pilot payload, 2026-11-05; the user: up to the Unitree documented limits, a wide payload CoM): max kg, the
# payload point's range in the base frame (+-x, +-y, 0..z m). unitree.com: Go2 / Go2-W MAX ~12 kg, B2 / B2-W walking load
# > 40 kg (standing 120 kg, not used for walking). As2 / As2W: no official figure found (our models 20.7 / 26.0 kg, not
# the A2's 37 kg): about half their weight, the user's choice (2026-11-11: "按约一半体重定（10 kg / 13 kg）"). BLIND_WALL_STOP (pilot endwall): base centre to
# the wall where the forward command reaches zero (half-length + ~0.5 m).
# the payload the training randomizes, the end-wall stop, the foot radius: the one robot table (src/robot_table.py,
# 2026-11-11; the values that stood here before, unchanged)
eval "$(python3 "$(dirname "${BASH_SOURCE[0]}")/../src/robot_table.py" $1 --fe-sh)"
if [ $WHEELED = 1 ]; then
  OCC=/root/gpufree-data/go2-depth/depth_fe/logs/wh_v4/${1,,}_v4_student.pt
else
  OCC=/root/gpufree-data/go2-depth/depth_fe/logs/occ/${1,,}_occ_v1_model_1999.pt
fi
if [ -n "$PSTART" ]; then export BLIND_PHASE_START_IT=$PSTART; else unset BLIND_PHASE_START_IT; fi
export FE_LOCK  # set in env.sh on t2 (go2-home's /root/gpufree-data/go2-blind/check.lock)
