#!/bin/bash
# ramp_recheck.sh R RUN K (go2-depth 2026-11-02): the student average RUN_avgK of robot R (B2 / B2W / As2W) under the
# deployed command ramps (fe_check.sh DRIFTA / STANDA: the user's 11-02 acceptance for a robot whose controller
# ramps the stick, the release a step). Per seed 0 1 2: a seed checked before the ramps gets drift_all, eval_standing
# and accept_matrix again (the step results kept as _drift3_step / _stand_step / _am_step / _verdict_step) and is
# judged again; an
# unchecked seed gets the full fe_check.sh. Stops at the first failing seed; logs to logs/jobs/fe_<r>.txt.
# The robot's chain variables (B2_N / B2_PKG / B2_PH / B2_PSTART, B2W_PKG, AS2W_PKG) as for its fe_chain.sh.
cd /root/gpufree-data/go2-depth/depth_fe && source env.sh
export PATH=$(dirname $PY):$PATH
R=$1 RUN=$2 K=$3
case $R in B2|B2W|As2W) DRIFTA="--yaw-accel 3.0 --lin-accel 2.0" STANDA="--lin-accel 2.0 --yaw-accel 3.0" ;; *) echo "$R: no ramp"; exit 2 ;; esac
source jobs/fe_robot.sh $R || exit 2
r=$(echo $R | tr '[:upper:]' '[:lower:]')
export BLIND_PILOT=$(cat $PKG/blind_pilot.txt) BLIND_ANCHOR_COEFF=0
export BLIND_ANCHOR_CHECKPOINT=$PWD/logs/rsl_rl/${r}_depth_teacher/blind_init_fe/model_$N.pt
[ -n "$BLIND_PILOT" ] || { echo "no pilot in $PKG"; exit 2; }
L=logs/jobs/fe_$r.txt C=logs/checks
A=logs/rsl_rl/${r}_depth_student/${RUN}_avg$K/model_$K.pt
[ -f $A ] || { echo "no $A"; exit 2; }
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $L; }
mean_reached() { python3 -c "import re,sys; v=[float(x) for x in re.findall(r'reached=([0-9.]+)', open(sys.argv[1]).read())]; print(f'{100*sum(v)/len(v):.1f}' if v else 'nan')" $1; }
falls() { grep -E "\| *[0-9]+/256" $1 | awk -F'|' '{gsub(/ /, "", $NF); printf "%s ", $NF}'; }
exec 9>> ${FE_LOCK:-$C/check.lock}
for SEED in 0 1 2; do
  Q=$C/fe_${r}_student_${RUN#fe_edge_}_avg${K}_s$SEED
  if [ -f ${Q}_verdict.txt ]; then
    [ -f ${Q}_verdict_step.txt ] || cp ${Q}_verdict.txt ${Q}_verdict_step.txt
    [ -f ${Q}_drift3_step.txt ] || cp ${Q}_drift3.txt ${Q}_drift3_step.txt
    [ -f ${Q}_stand_step.txt ] || cp ${Q}_stand.txt ${Q}_stand_step.txt
    [ -f ${Q}_am_step.txt ] || cp ${Q}_am.txt ${Q}_am_step.txt
    flock 9
    $PY jobs/student_eval.py jobs/seed_run.py $SEED scripts/blind/eval/drift_all.py --task Unitree-$R-Depth-Student-Flat $A $DRIFTA > ${Q}_drift3.txt 2>&1
    $PY jobs/student_eval.py jobs/seed_run.py $SEED scripts/blind/eval/eval_standing.py --task Unitree-$R-Depth-Student-Flat $A $STANDA > ${Q}_stand.txt 2>&1
    $PY jobs/student_eval.py jobs/seed_run.py $SEED scripts/blind/eval/accept_matrix.py --task Unitree-$R-Depth-Student-Flat --envs-per-cmd 32 $A $STANDA > ${Q}_am.txt 2>&1
    flock -u 9
    v=$($PY jobs/fe_pass.py $R $Q 2>&1 | tail -n 1)
    [[ "$v" == PASS || "$v" == FAIL* ]] || v="FAIL: judge error: $v"
    en=$(mean_reached ${Q}_noopedge.txt) eo=$(mean_reached ${Q}_opedge.txt)
    if ! python3 -c "import sys; n,o=map(float,sys.argv[1:]); sys.exit(0 if n >= 90 and o >= 95 else 1)" $en $eo; then
      if [[ "$v" == PASS ]]; then v="FAIL: 棱抹平楼梯：不通过"; else v="$v；棱抹平楼梯：不通过"; fi
    fi
    echo "$v" > ${Q}_verdict.txt
    log "$R ramp recheck ${RUN}_avg$K s$SEED: step verdict $(cat ${Q}_verdict_step.txt) -> with the deployed ramps $v; drift falls $(falls ${Q}_drift3.txt)| stop falls $(falls ${Q}_stand.txt)"
  else
    bash jobs/fe_check.sh student $R $A $SEED $Q
    v=$(cat ${Q}_verdict.txt)
    log "$R ramp check ${RUN}_avg$K s$SEED: noop $(mean_reached ${Q}_noop.txt) op $(mean_reached ${Q}_op.txt) edge noop $(mean_reached ${Q}_noopedge.txt) op $(mean_reached ${Q}_opedge.txt) speed up/down/margin $($PY jobs/fe_margin.py $R $Q) | $v"
  fi
  [[ "$v" == PASS ]] || { log "$R ramp recheck ${RUN}_avg$K: stops at s$SEED"; exit 1; }
done
log "$R ramp recheck ${RUN}_avg$K: PASS on seeds 0 1 2 with the deployed ramps"
