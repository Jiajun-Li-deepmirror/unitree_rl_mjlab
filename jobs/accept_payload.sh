#!/bin/bash
# accept_payload.sh R (go2-depth 2026-11-11, the user's requirement 7 "一项一种测法"): the acceptance payload of robot R
# as kg,x,y,z (a point mass on the base; m from the base origin). Every test and video reads it from here: fe_check's
# loaded stand, record_blind / record_student_v / blind_videos, marchpay, tn_gate_watch's tinynav runs.
# doc/acceptance.md §3: the payload limit (Go2 / Go2-W 12 kg, B2 / B2-W 40 kg: Unitree's figures; As2 10 kg / As2-W
# 13 kg: about half the body weight, the user 2026-11-11) at x +0.15, z +0.20 (B2 / B2-W z +0.30).
# Exit 2 on an unknown robot.
case ${1:-} in
  Go2|Go2W) echo 12,0.15,0,0.20 ;;
  As2) echo 10,0.15,0,0.20 ;;
  As2W) echo 13,0.15,0,0.20 ;;
  B2|B2W) echo 40,0.15,0,0.30 ;;
  *) echo "accept_payload.sh: no payload for '${1:-}'" >&2; exit 2 ;;
esac
