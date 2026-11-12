#!/bin/bash
# flat_gate.sh R CKPT (unitree-rl-mjlab-62 2026-11-13; go2-depth's decision 11-13 03:36): the flat part of the teacher's
# gate on one checkpoint -- tier-2's own flat tests (jobs/unified/accept.sh's unloaded flat set, on
# Unitree-<R>-U-Teacher-Flat) in both scan modes, judged by scripts/unified/flat_gate.py at the students' flat bars,
# no flat fall. Called by jobs/unified/teacher.sh after a seed's stairs gate passed (F_FLAT_GATE).
# Per mode, one test at a time: accept_matrix (96 / command), drift_all (F_FLAT_GATE_DRIFT_ENVS / command), spin_alt
# (768), eval_standing (F_FLAT_GATE_STAND_ENVS / group), march_eval (96), gait_clock (48 / command), long_stand 60 s and
# stop_then_move (F_FLAT_GATE_STANCE_ENVS), wheeled also gait_fb (96 / command); ~15 min a mode. Fewer robots than
# accept.sh's 768 (the user's floor: 256): the gate runs beside the formal run's two seeds (14.1 GB). Peaks measured
# 11-13 beside two teachers: drift_all at 384 +4.2-5.2 GB, accept_matrix +3.6, eval_standing at 384 +2.3, spin_alt
# +2.0, gait_clock +1.9, gait_fb +0.9, march +0.7.
# A test must never take a training down (11-13: a flat test beside three teachers OOM-killed one): each starts only
# with F_FLAT_GATE_FREE_MB free on the card; no room within F_FLAT_GATE_WAIT_S -> a tool fault at once (exit 4,
# teacher.sh judges again at its next gate). Out of GPU memory anyway: 2 min and again, up to 6 times.
# Files logs/checks/u_<r>_tflat_<run>_<model>_<on|off>_<test>.txt and ..._verdict.txt. Prints the verdict (first line
# PASS / FAIL (gap G): ... / FAULT: ...). Exit 0 PASS, 1 FAIL, 4 a tool fault.
set -u
R=$1 CK=$2 r=${1,,}
cd ${UNIFIED_ROOT:-/root/gpufree-data/go2-depth/depth_fe} || exit 2
source env.sh > /dev/null 2>&1
source jobs/unified/formal.env
export UNIFIED_TASKS=1 U_PAYLOAD=0,0,0,0
P=logs/checks/u_${r}_tflat_$(basename $(dirname $CK))_$(basename $CK .pt)
mkdir -p logs/checks
W="$PY jobs/unified/u_eval.py" S=scripts/blind/eval TF=Unitree-$R-U-Teacher-Flat
LIN="--lin-accel 2.0" YAW="--yaw-accel 3.0"
CLOCK=""; [[ $R == *W ]] && CLOCK="--clock-task Unitree-$R-U-Teacher"
free_mb() { nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | head -n 1 | awk -F', *' '{print $1 - $2}'; }
run() {   # OUT CMD...
  local out=$1 t0 i f; shift
  t0=$(date +%s)
  while f=$(free_mb) && [ "${f:-0}" -lt $F_FLAT_GATE_FREE_MB ]; do
    if [ $(( $(date +%s) - t0 )) -ge $F_FLAT_GATE_WAIT_S ]; then
      echo "FAULT: no room on the GPU for $(basename $out): $F_FLAT_GATE_FREE_MB MiB free needed, ${f:-?} MiB free for $F_FLAT_GATE_WAIT_S s" | tee ${P}_verdict.txt
      exit 4
    fi
    sleep 20
  done
  for i in 1 2 3 4 5 6 7; do
    "$@" > $out 2>&1 && return 0
    grep -qiE "out of memory|CUDA error 2" $out && [ $i -lt 7 ] || break
    sleep 120
  done
  return 1   # what the test could not give, the judge reports as not measured (a tool fault)
}
for m in on off; do
  export U_SCAN=$m
  Q=${P}_$m
  run ${Q}_am.txt $W $S/accept_matrix.py --task $TF --envs-per-cmd 96 $CK $LIN $YAW
  run ${Q}_drift3.txt $W $S/drift_all.py --task $TF $CK --num-envs $F_FLAT_GATE_DRIFT_ENVS $YAW $LIN
  run ${Q}_spinalt.txt $W $S/spin_alt.py --task $TF $CK --num-envs 768 $YAW
  run ${Q}_stand.txt $W $S/eval_standing.py --task $TF $CK --envs-per-group $F_FLAT_GATE_STAND_ENVS $LIN $YAW
  run ${Q}_march.txt $W $S/march_eval.py --task $TF $CK --envs 96
  run ${Q}_clock.txt $W $S/gait_clock.py --task $TF $CLOCK $CK --envs-per-cmd 48
  run ${Q}_loadstand.txt $W $S/long_stand.py --task $TF $CK --seconds 60 --num-envs $F_FLAT_GATE_STANCE_ENVS
  run ${Q}_stopmove.txt $W $S/stop_then_move.py --task $TF $CK --num-envs $F_FLAT_GATE_STANCE_ENVS
  if [[ $R == *W ]]; then run ${Q}_gait.txt $W $S/gait_fb.py --task $TF $CK --envs-per-cmd 96; fi
done
$PY scripts/unified/flat_gate.py $R $P > ${P}_verdict.txt 2>&1
rc=$?
cat ${P}_verdict.txt
exit $rc
