#!/bin/bash
# smoke_spinhold.sh R PKG (go2-depth 2026-11-06): widen PKG's model for the "spinhold" critic input, then 3
# iterations of the Polish task at 64 robots with the package's pilot + marchpose,spinhold.
set -u
R=$1 BPKG=$2
cd /root/gpufree-data/go2-depth/depth_fe && source env.sh > /dev/null 2>&1
CK=$(ls $BPKG/model_*.pt | head -n 1); N=$(basename $CK .pt | cut -d_ -f2); r=${R,,}
PILOT=$(cat $BPKG/blind_pilot.txt)
source jobs/fe_robot.sh $R > /dev/null
export BLIND_PILOT=$PILOT,marchpose,spinhold BLIND_ANCHOR_COEFF=0 BLIND_ANCHOR_CHECKPOINT=$PWD/$CK BLIND_PHASE_START_IT=2000
I=logs/rsl_rl/${r}_blind/smoke_spinhold_init
mkdir -p $I
$PY scripts/blind/widen_obs.py --task Unitree-$R-Blind-Polish $CK --out $I/model_$N.pt 2>&1 | grep -v -i warn | tail -4
$PY scripts/train.py Unitree-$R-Blind-Polish --env.scene.num-envs=64 --agent.run-name=smoke_spinhold --agent.max-iterations=3 \
  --agent.resume=True --agent.load-run="^smoke_spinhold_init\$" --agent.load-checkpoint="^model_$N\.pt\$" 2>&1 \
  | grep -a -E "Traceback|Error|spin_position_hold|spin_hold|Learning iteration|critic|hold_offsets" | head -30
