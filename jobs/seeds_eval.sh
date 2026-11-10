#!/bin/bash
# seeds_eval.sh R CK NAME (go2-depth 2026-11-02): the full student checks of checkpoint CK with the robot's package
# (<R>_PKG as for its fe_chain.sh), seeds 0 1 2 under its own lock (FE_LOCK_SEEDS; for an idle GPU), stopping at the
# first failing seed; logs to logs/jobs/fe_<r>.txt. As2-W v6rl avg300 (11-02): go2-home, its stops were fixed by
# model_200 while the sideways gait clock was still even (it drifts from model_400 as the BC weight falls).
cd /root/gpufree-data/go2-depth/depth_fe && source env.sh
export PATH=$(dirname $PY):$PATH
R=$1 CK=$2 NAME=$3
source jobs/fe_robot.sh $R || exit 2
r=$(echo $R | tr '[:upper:]' '[:lower:]')
export BLIND_PILOT=$(cat $PKG/blind_pilot.txt) BLIND_ANCHOR_COEFF=0
export BLIND_ANCHOR_CHECKPOINT=$PWD/logs/rsl_rl/${r}_depth_teacher/blind_init_fe/model_$N.pt
export FE_LOCK=${FE_LOCK_SEEDS:-logs/checks/seeds_eval.lock}
L=logs/jobs/fe_$r.txt C=logs/checks
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $L; }
mean_reached() { python3 -c "import re,sys; v=[float(x) for x in re.findall(r'reached=([0-9.]+)', open(sys.argv[1]).read())]; print(f'{100*sum(v)/len(v):.1f}' if v else 'nan')" $1; }
for SEED in 0 1 2; do
  Q=$C/fe_${r}_student_${NAME}_s$SEED
  bash jobs/fe_check.sh student $R $CK $SEED $Q
  v=$(cat ${Q}_verdict.txt)
  log "$R seeds eval $NAME s$SEED: noop $(mean_reached ${Q}_noop.txt) op $(mean_reached ${Q}_op.txt) edge noop $(mean_reached ${Q}_noopedge.txt) op $(mean_reached ${Q}_opedge.txt) speed up/down/margin $($PY jobs/fe_margin.py $R $Q) | $v"
  [[ "$v" == PASS ]] || { log "$R seeds eval $NAME: stops at s$SEED"; exit 1; }
done
log "$R seeds eval $NAME: PASS on seeds 0 1 2 with the deployed ramps"
