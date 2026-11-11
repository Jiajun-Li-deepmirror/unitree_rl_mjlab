#!/bin/bash
# accept_payload.sh R (go2-depth 2026-11-11, the user's requirement 7 "一项一种测法"): the acceptance payload of robot R
# as kg,x,y,z (a point mass on the base; m from the base origin). Every test and video reads it from here: fe_check's
# loaded stand, accept_videos.sh, tn_gate_watch's tinynav runs. The value is the robot table's (src/robot_table.py,
# the user's unification item 2): Go2 / Go2-W 12 kg, B2 / B2-W 40 kg (Unitree's figures); As2 10 kg / As2-W 13 kg
# (about half the body weight, the user 2026-11-11); at x +0.15, z +0.20 (B2 / B2-W z +0.30). Exit 2 on an unknown robot.
exec python3 "$(dirname "$0")/../src/robot_table.py" "${1:-}" --accept-payload
