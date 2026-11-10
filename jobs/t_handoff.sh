#!/bin/bash
# t_handoff.sh R TAG (go2-depth 2026-11-10): the T-limit rule (the user: the blind T phase at its limit without a pass
# hands its last check to M, as M does to P; blind_chain.sh has it from 103338e+1) for a chain that ended under the code
# before it. Does nothing unless that robot's last TAG chain logged "blind T: no checkpoint passed" and nothing is
# delivered; then runs the blind round on from the last T check (BL_T_PICK / BL_T_RUN), no T trained again.
set -u
R=$1 TAG=$2 r=${1,,}
cd /root/gpufree-data/go2-depth/depth_fe
L=logs/jobs/blind_$r.txt
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> logs/jobs/repro_$r.txt; }
if grep -q "$R BLIND DELIVERY logs/pkg/${r}_bl2_${TAG}_" $L 2> /dev/null; then log "$R t_handoff $TAG: delivered already, nothing to do"; exit 0; fi
start=$(grep -n "$R blind chain $TAG:" $L | head -n 1 | cut -d: -f1)
[ -n "$start" ] || { log "$R t_handoff $TAG: no $TAG chain, nothing to do"; exit 0; }
if ! tail -n +$start $L | grep -q "$R blind T: no checkpoint passed the flat goals"; then
  log "$R t_handoff $TAG: its T did not end at the limit without a pass, nothing to do"; exit 0
fi
T=$(ls logs/rsl_rl/${r}_blind/bl2T_${TAG}_avg*/model_*.pt 2> /dev/null | sort -V | tail -n 1)
TR=$(ls -d logs/rsl_rl/${r}_blind/*_bl2T_$TAG 2> /dev/null | sort | tail -n 1)
[ -n "$T" ] && [ -n "$TR" ] || { log "$R t_handoff $TAG: no T check or run found"; exit 1; }
log "$R t_handoff $TAG: T ended at its limit without a pass -> M from the last check $T (run $TR)"
BL_T_PICK=$T BL_T_RUN=$TR BL_T_PICK_WHY="T at its limit without a pass, its last check (rule 2026-11-10)" \
  bash jobs/full_chain.sh $R $TAG blind
