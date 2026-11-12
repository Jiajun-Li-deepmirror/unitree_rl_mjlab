#!/bin/bash
# accept_payload.sh R (go2-depth 2026-11-11, the user's requirement 7 "一项一种测法"): the acceptance payload of robot R
# as kg,x,y,z (a point mass on the base; m from the base origin). Every test and video reads it from here: fe_check's
# loaded stand, accept_videos.sh, tn_gate_watch's tinynav runs. The value is the robot table's (src/robot_table.py,
# the user's unification item 2): Unitree's walking figures -- Go2 / Go2-W 8 kg, As2 15 kg, As2-W 16 kg, B2 / B2-W
# 40 kg; at x +0.15, z +0.20 (B2 / B2-W z +0.30). Exit 2 on an unknown robot.
# accept_payload.sh R typical (2026-11-12, the user: "加一档「常用负载」"; doc/redesign.md 2.1; the unified acceptance's
# third condition): half the walking payload as kg,x,y,z,dx,dy,dz -- each evaluation robot's mount point is drawn in
# x +- dx, y +- dy, z +- dz (U_PAYLOAD takes the seven values, src/tasks/unified/teacher_env.py). Exit 2 for a robot
# that has none yet.
case "${2:-}" in
  "") exec python3 "$(dirname "$0")/../src/robot_table.py" "${1:-}" --accept-payload ;;
  typical) exec python3 "$(dirname "$0")/../src/robot_table.py" "${1:-}" --typical-payload ;;
  *) echo "accept_payload.sh: unknown condition ${2}" >&2; exit 2 ;;
esac
