#!/bin/bash
# b2_seeds.sh RUN K (go2-depth 2026-11-02): the full student checks of a B2 student average on seeds 1 and 2 (its
# seed 0 is known), under the check lock, logged to fe_b2.txt; for the user's call on a near-miss candidate.
cd /root/gpufree-data/go2-depth/depth_fe && source env.sh
export PATH=$(dirname $PY):$PATH
export BLIND_PILOT=$(cat logs/pkg/b2_fe_mixed_5000/blind_pilot.txt) BLIND_ANCHOR_COEFF=0
export FE_LOCK=/root/gpufree-data/go2-blind/check.lock FE_SERIAL=1
RUN=$1 K=$2 L=logs/jobs/fe_b2.txt C=logs/checks
A=logs/rsl_rl/b2_depth_student/${RUN}_avg$K/model_$K.pt
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $L; }
mean_reached() { python3 -c "import re,sys; v=[float(x) for x in re.findall(r'reached=([0-9.]+)', open(sys.argv[1]).read())]; print(f'{100*sum(v)/len(v):.1f}' if v else 'nan')" $1; }
for SEED in 1 2; do
  Q=$C/fe_b2_student_${RUN#fe_edge_}_avg${K}_s$SEED
  bash jobs/fe_check.sh student B2 $A $SEED $Q
  log "B2 seeds ${RUN}_avg$K s$SEED: noop $(mean_reached ${Q}_noop.txt) op $(mean_reached ${Q}_op.txt) edge noop $(mean_reached ${Q}_noopedge.txt) op $(mean_reached ${Q}_opedge.txt) speed up/down/margin $($PY jobs/fe_margin.py B2 $Q) | $(cat ${Q}_verdict.txt)"
done
