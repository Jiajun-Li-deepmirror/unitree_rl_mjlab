#!/bin/bash
# stairs_gate.sh R CKPT (unitree-rl-mjlab-62 2026-11-13; go2-depth's gate design): the stairs part of the teacher's gate on
# one checkpoint -- tier-2's stairs test (scripts/go2_blind/eval_stairs.py --metrics, Unitree-<R>-U-Teacher-Stairs,
# unloaded, 18 robots per tile) in both scan modes, one after the other (out of GPU memory: 2 min and again, 6 times),
# judged by scripts/unified/stairs_gate.py. Prints its verdict (first line PASS / FAIL (gap G): ... / FAULT: ...);
# files logs/checks/u_<r>_tgate_<run>_<model>_{on,off}.txt and ..._verdict.txt. Exit 0 PASS, 1 FAIL, 4 a tool fault.
# Called by jobs/unified/teacher.sh on a seed's "PRE-PASS K" (scripts/unified/teacher_gate.py --pre).
set -u
R=$1 CK=$2 r=${1,,}
cd ${UNIFIED_ROOT:-/root/gpufree-data/go2-depth/depth_fe} || exit 2
source env.sh > /dev/null 2>&1
export UNIFIED_TASKS=1 U_PAYLOAD=0,0,0,0
P=logs/checks/u_${r}_tgate_$(basename $(dirname $CK))_$(basename $CK .pt)
mkdir -p logs/checks
for m in on off; do
  for i in 1 2 3 4 5 6 7; do
    U_SCAN=$m $PY jobs/unified/u_eval.py scripts/go2_blind/eval_stairs.py --task Unitree-$R-U-Teacher-Stairs $CK \
      --envs-per-tile 18 --metrics > ${P}_$m.txt 2>&1 && break
    grep -qiE "out of memory|CUDA error 2" ${P}_$m.txt && [ $i -lt 7 ] || break
    sleep 120
  done
done
$PY scripts/unified/stairs_gate.py $R $P > ${P}_verdict.txt 2>&1
rc=$?
cat ${P}_verdict.txt
exit $rc
