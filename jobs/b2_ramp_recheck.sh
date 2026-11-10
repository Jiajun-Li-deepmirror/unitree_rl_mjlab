#!/bin/bash
# b2_ramp_recheck.sh RUN K (go2-depth 2026-11-02, user: B2's drift spins with the deployed 3 rad/s² yaw ramp): a B2
# student average whose full checks ran with step spins gets drift_all again with the ramp on seeds 0 1 2 (the step
# results kept as _drift3_step.txt) and is judged again (fe_pass + the -RealEdge bar, as fe_check.sh); logs to fe_b2.txt.
cd /root/gpufree-data/go2-depth/depth_fe && source env.sh
export PATH=$(dirname $PY):$PATH
export BLIND_PILOT=$(cat logs/pkg/b2_fe_mixed_5000/blind_pilot.txt) BLIND_ANCHOR_COEFF=0
RUN=$1 K=$2 L=logs/jobs/fe_b2.txt C=logs/checks
A=logs/rsl_rl/b2_depth_student/${RUN}_avg$K/model_$K.pt
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $L; }
while pgrep -f "jobs/b2_seeds[.]sh $RUN $K" > /dev/null; do sleep 60; done
mean_reached() { python3 -c "import re,sys; v=[float(x) for x in re.findall(r'reached=([0-9.]+)', open(sys.argv[1]).read())]; print(f'{100*sum(v)/len(v):.1f}' if v else 'nan')" $1; }
exec 9>> /root/gpufree-data/go2-blind/check.lock
for SEED in 0 1 2; do
  Q=$C/fe_b2_student_${RUN#fe_edge_}_avg${K}_s$SEED
  [ -f ${Q}_verdict.txt ] || { log "B2 ramp recheck ${RUN}_avg$K s$SEED: no full check"; continue; }
  [ -f ${Q}_drift3_step.txt ] || cp ${Q}_drift3.txt ${Q}_drift3_step.txt
  flock 9
  $PY jobs/student_eval.py jobs/seed_run.py $SEED scripts/blind/eval/drift_all.py --task Unitree-B2-Depth-Student-Flat $A --yaw-accel 3.0 > ${Q}_drift3.txt 2>&1
  flock -u 9
  v=$($PY jobs/fe_pass.py B2 $Q 2>&1 | tail -n 1)
  [[ "$v" == PASS || "$v" == FAIL* ]] || v="FAIL: judge error: $v"
  en=$(mean_reached ${Q}_noopedge.txt) eo=$(mean_reached ${Q}_opedge.txt)
  if ! python3 -c "import sys; n,o=map(float,sys.argv[1:]); sys.exit(0 if n >= 90 and o >= 95 else 1)" $en $eo; then
    if [[ "$v" == PASS ]]; then v="FAIL: 棱抹平楼梯：不通过"; else v="$v；棱抹平楼梯：不通过"; fi
  fi
  cp ${Q}_verdict.txt ${Q}_verdict_step.txt; echo "$v" > ${Q}_verdict.txt
  log "B2 ramp recheck ${RUN}_avg$K s$SEED: step verdict $(cat ${Q}_verdict_step.txt) -> with the 3 rad/s² yaw ramp $v; spins $(grep -E '^ *\(0\.0, 0\.0, -?1\.[05]\)' ${Q}_drift3.txt | awk -F'|' '{gsub(/ /, "", $NF); printf "%s ", $NF}')"
done
