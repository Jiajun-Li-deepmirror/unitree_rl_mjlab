#!/bin/bash
# fe_avg_seeds.sh R CK TAG (depth_fe, 2026-10-31): a student candidate (a checkpoint average) through fe_check.sh on
# seeds 0, 1, 2, stopping at the first failure; one line per seed in logs/jobs/fe_<r>.txt, "<R> AVG <TAG> PASS 3 seeds"
# when all pass. BLIND_PILOT etc. from the delivery as fe_chain.sh.
set -u
cd /root/gpufree-data/go2-depth/depth_fe && source env.sh
R=$1 r=${1,,} CK=$2 TAG=$3
source jobs/fe_robot.sh $R || exit 2  # also BLIND_PHASE_START_IT / BLIND_CONS_INIT
export BLIND_PILOT=$(cat $PKG/blind_pilot.txt) BLIND_ANCHOR_COEFF=0 BLIND_ANCHOR_CHECKPOINT=$PWD/logs/rsl_rl/${r}_depth_teacher/blind_init_fe/model_$N.pt
reached() { python3 -c "import re,sys; v=[float(x) for x in re.findall(r\"reached=([0-9.]+)\", open(sys.argv[1]).read())]; print(f\"{100*sum(v)/len(v):.1f}\" if v else \"nan\")" $1; }
for SEED in 0 1 2; do
  P=logs/checks/fe_${r}_student_${TAG}_s$SEED
  bash jobs/fe_check.sh student $R $CK $SEED $P
  v=$(cat ${P}_verdict.txt)
  echo "$(date "+%m-%d %H:%M:%S") $R student $TAG s$SEED: noop $(reached ${P}_noop.txt) op $(reached ${P}_op.txt) edge noop $(reached ${P}_noopedge.txt) op $(reached ${P}_opedge.txt) | $v" >> logs/jobs/fe_$r.txt
  [[ "$v" == PASS ]] || exit 1
done
echo "$(date "+%m-%d %H:%M:%S") $R AVG $TAG PASS 3 seeds ($CK)" >> logs/jobs/fe_$r.txt
