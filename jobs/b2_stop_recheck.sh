#!/bin/bash
# b2_stop_recheck.sh RUN K [SEEDS] (go2-depth 2026-11-02, user: B2's stop test with the deployed ramped start, the release a
# step): after b2_ramp_recheck.sh (the ramped drift spins), eval_standing again with --lin-accel 2.0 --yaw-accel 3.0
# on seeds 0 1 2 (the step-start results kept as _stand_step.txt) and the judgment again; logs to fe_b2.txt.
cd /root/gpufree-data/go2-depth/depth_fe && source env.sh
export PATH=$(dirname $PY):$PATH
export BLIND_PILOT=$(cat logs/pkg/b2_fe_mixed_5000/blind_pilot.txt) BLIND_ANCHOR_COEFF=0
RUN=$1 K=$2 L=logs/jobs/fe_b2.txt C=logs/checks
A=logs/rsl_rl/b2_depth_student/${RUN}_avg$K/model_$K.pt
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $L; }
while pgrep -f "jobs/b2_(seeds|ramp_recheck)[.]sh $RUN $K" > /dev/null; do sleep 60; done
mean_reached() { python3 -c "import re,sys; v=[float(x) for x in re.findall(r'reached=([0-9.]+)', open(sys.argv[1]).read())]; print(f'{100*sum(v)/len(v):.1f}' if v else 'nan')" $1; }
# FE_LOCK: another lock beside the box's check lock when the GPU has room (t1 18:50 11-02: no training, 4 of 24 GB)
exec 9>> ${FE_LOCK:-/root/gpufree-data/go2-blind/check.lock}
for SEED in ${3:-0 1 2}; do
  Q=$C/fe_b2_student_${RUN#fe_edge_}_avg${K}_s$SEED
  [ -f ${Q}_verdict.txt ] || { log "B2 stop recheck ${RUN}_avg$K s$SEED: no full check"; continue; }
  [ -f ${Q}_stand_step.txt ] || cp ${Q}_stand.txt ${Q}_stand_step.txt
  [ -f ${Q}_verdict_step.txt ] || cp ${Q}_verdict.txt ${Q}_verdict_step.txt
  flock 9
  $PY jobs/student_eval.py jobs/seed_run.py $SEED scripts/blind/eval/eval_standing.py --task Unitree-B2-Depth-Student-Flat $A --lin-accel 2.0 --yaw-accel 3.0 > ${Q}_stand.txt 2>&1
  flock -u 9
  v=$($PY jobs/fe_pass.py B2 $Q 2>&1 | tail -n 1)
  [[ "$v" == PASS || "$v" == FAIL* ]] || v="FAIL: judge error: $v"
  en=$(mean_reached ${Q}_noopedge.txt) eo=$(mean_reached ${Q}_opedge.txt)
  if ! python3 -c "import sys; n,o=map(float,sys.argv[1:]); sys.exit(0 if n >= 90 and o >= 95 else 1)" $en $eo; then
    if [[ "$v" == PASS ]]; then v="FAIL: 棱抹平楼梯：不通过"; else v="$v；棱抹平楼梯：不通过"; fi
  fi
  echo "$v" > ${Q}_verdict.txt
  log "B2 stop recheck ${RUN}_avg$K s$SEED: step verdict $(cat ${Q}_verdict_step.txt) -> with the ramped start (stops a step) and ramped spins $v; stand falls $(grep -E '\| *[0-9]+/256' ${Q}_stand.txt | awk -F'|' '{gsub(/ /, "", $NF); printf "%s ", $NF}')"
done
