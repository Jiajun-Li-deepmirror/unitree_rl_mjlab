#!/bin/bash
# smoke.sh R CKPT PILOT_EXTRA [ITERS] [TAG] (go2-depth 2026-11-05, the new pipeline's smoke test): the robot's teacher
# task (fe_robot.sh's delivery pilot + PILOT_EXTRA) for ITERS (default 100) iterations at 512 envs from CKPT (a raw
# teacher checkpoint, copied to <r>_depth_teacher/smoke_<TAG>_init), then one summary line in logs/checks/smoke.txt:
# start / end mean reward and value loss, the last iterations' fell_over / illegal_contact terminations, and any
# nconmax / njmax overflow in the log. Every new reward / terrain / event change runs this before a real round.
set -u
cd /root/gpufree-data/go2-depth/depth_fe && source env.sh
R=$1 CK=$2 EXTRA=$3 IT=${4:-100} TAG=${5:-$(date +%H%M%S)}
r=${R,,}
# the delivery packages the depth chains use (fe_robot.sh's defaults point at go2-blind exports t1 lacks)
: ${GO2W_PKG:=$PWD/logs/pkg/go2w_polish_11300} ${AS2W_PKG:=$PWD/logs/pkg/as2w_mixed_8000}
: ${B2W_PKG:=$PWD/logs/pkg/b2w_polish_9000_sidecmd_desc}
export GO2W_PKG AS2W_PKG B2W_PKG
source jobs/fe_robot.sh $R || exit 2
[ -f $PKG/blind_pilot.txt ] || { echo "smoke: no $PKG/blind_pilot.txt"; exit 2; }
export BLIND_PILOT=$(cat $PKG/blind_pilot.txt)$EXTRA BLIND_ANCHOR_COEFF=0
export BLIND_ANCHOR_CHECKPOINT=$PWD/logs/rsl_rl/${r}_depth_teacher/blind_init_fe/model_$N.pt
T0=$(basename $CK .pt | cut -d_ -f2)
LRUN=smoke_${TAG}_init
mkdir -p logs/rsl_rl/${r}_depth_teacher/$LRUN && cp $CK logs/rsl_rl/${r}_depth_teacher/$LRUN/model_$T0.pt
LOG=logs/jobs/smoke_${r}_$TAG.log
$PY scripts/train.py Unitree-$R-Depth-$PH --env.scene.num-envs=512 --agent.run-name=smoke_$TAG --agent.resume=True \
  --agent.load-run="^$LRUN\$" --agent.load-checkpoint="^model_$T0\.pt\$" --agent.max-iterations=$IT > $LOG 2>&1
rc=$?
first=$(grep -a "^ *Mean value loss:" $LOG | head -n 3 | awk '{s+=$NF} END {printf "%.2f", s/NR}')
last=$(grep -a "^ *Mean value loss:" $LOG | tail -n 10 | awk '{s+=$NF} END {printf "%.2f", s/NR}')
rew0=$(grep -a "^ *Mean reward:" $LOG | head -n 3 | awk '{s+=$NF} END {printf "%.1f", s/NR}')
rew1=$(grep -a "^ *Mean reward:" $LOG | tail -n 10 | awk '{s+=$NF} END {printf "%.1f", s/NR}')
fell=$(grep -a "Episode_Termination/fell_over" $LOG | tail -n 20 | awk '{s+=$NF} END {printf "%.3f", s/NR}')
ill=$(grep -a "Episode_Termination/illegal_contact" $LOG | tail -n 20 | awk '{s+=$NF} END {printf "%.3f", s/NR}')
ovf=$(grep -a -c -i "overflow" $LOG)
line="$(date '+%m-%d %H:%M') smoke $R $TAG (rc $rc, $IT it, extra $EXTRA): reward $rew0 -> $rew1, value loss $first -> $last, fell_over $fell, illegal $ill, overflow lines $ovf"
echo "$line" | tee -a logs/checks/smoke.txt
[ $rc = 0 ] || tail -n 5 $LOG
