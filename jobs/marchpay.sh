#!/bin/bash
# marchpay.sh R PKG EXTRA: long_stand.py --march 3 at 0 kg and the payload limit (11-07 user: payload + march in place)
R=$1 BPKG=$2 EXTRA=${3:-}
cd /root/gpufree-data/go2-depth/depth_fe && source env.sh > /dev/null 2>&1
source jobs/fe_robot.sh $R > /dev/null
CK=$(ls $BPKG/model_*.pt | head -n 1)
export BLIND_PILOT=$(cat $BPKG/blind_pilot.txt) BLIND_ANCHOR_COEFF=0 BLIND_ANCHOR_CHECKPOINT=$PWD/$CK
case $R in Go2|Go2W) PAY=12,0.15,0,0.20 ;; As2) PAY=10,0.15,0,0.20 ;; As2W) PAY=13,0.15,0,0.20 ;; *) PAY=40,0.15,0,0.30 ;; esac
O=logs/probe/marchpay_${R,,}_$(basename $BPKG).txt; : > $O
for pl in 0,0,0,0 $PAY; do
  $PY jobs/seed_run.py 0 scripts/blind/eval/long_stand.py --task Unitree-$R-Blind-Flat-Scratch $CK --seconds 60 --march 3 --payload $pl $EXTRA 2>&1 | grep -A1 "zero command" >> $O
done
echo done >> $O
