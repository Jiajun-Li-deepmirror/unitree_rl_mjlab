#!/bin/bash
# spinalt_cal.sh (go2-depth 2026-11-06): spin_alt.py on the delivered and the deployed blind packages, to set its bar.
cd /root/gpufree-data/go2-depth/depth_fe && source env.sh > /dev/null 2>&1
E=/root/gpufree-data/go2-blind/export
SET=("Go2W logs/pkg/go2w_bl2_g14_13600" "Go2W $E/wheeled_blind/go2w_polish_11300" "B2 logs/pkg/b2_bl2_l4_9000" "B2 $E/legged_feetedge/b2_fe_mixed_5000")
[ "${1:-}" = deployed2 ] && SET=("B2W $E/wheeled_blind/b2w_polish_9000" "As2W $E/wheeled_blind/as2w_mixed_8000" "As2 $E/legged_feetedge/as2_fe_polish_12000")
for x in "${SET[@]}"; do
  set -- $x; R=$1 BPKG=$2
  CK=$(ls $BPKG/model_*.pt | head -n 1); PILOT=$(cat $BPKG/blind_pilot.txt)
  ( source jobs/fe_robot.sh $R > /dev/null
    export BLIND_PILOT=$PILOT BLIND_ANCHOR_COEFF=0 BLIND_ANCHOR_CHECKPOINT=$PWD/$CK
    A=""; [[ $R == B2 || $R == B2W || $R == As2W ]] && A="--yaw-accel 3.0"
    $PY scripts/blind/eval/spin_alt.py --task Unitree-$R-Blind-Flat-Scratch $CK $A 2>&1 | grep -v -i -E "warn|^\|" | tail -6 \
      > logs/checks/spinalt_cal_$(basename $BPKG).txt )
  echo "== $(basename $BPKG)"; cat logs/checks/spinalt_cal_$(basename $BPKG).txt
done
