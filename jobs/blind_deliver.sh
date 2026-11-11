#!/bin/bash
# blind_deliver.sh R TAG CK WHY (go2-depth 2026-11-11): deliver a blind P checkpoint by hand when its chain has ended
# without a pick although the checkpoint passes -- e.g. on the luck rule beyond jobs/luck_pass.py (the user 11-04:
# "luck-only fails (rare DR robot, one seed) count as pass, report it"). The same steps as jobs/blind_chain.sh's end:
#   1. the tinynav acceptance request logs/accept/bl_<r>_bl2P_<TAG>_<K>.req (tools/go2_depth/acceptance/tn_gate_watch.sh)
#   2. the package logs/pkg/<r>_bl2_<TAG>_<K>: model, the P run's params/ and run_env.sh, pilot, jobs/export_policy.sh
#   3. jobs/accept_videos.sh in the background
#   4. waits for the tinynav verdict (12 h at most): PASS -> "BLIND DELIVERY" in logs/jobs/blind_<r>_<TAG>.txt, else
#      the verdict is logged and the package stays a candidate (exit 1)
# CK is the averaged checkpoint (logs/rsl_rl/<r>_blind/bl2P_<TAG>_avg<K>/model_<K>.pt); WHY goes into the log.
set -u
R=$1 TAG=$2 CK=$3 WHY=${4:-}
r=${R,,}
cd /root/gpufree-data/go2-depth/depth_fe || exit 2
source env.sh > /dev/null 2>&1
L=logs/jobs/blind_${r}_$TAG.txt
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $L; }
[ -f "$CK" ] || { log "$R blind_deliver: no checkpoint $CK"; exit 2; }
K=$(basename $CK .pt | cut -d_ -f2)
RUN=$(ls -d logs/rsl_rl/${r}_blind/*_bl2P_$TAG 2> /dev/null | sort | tail -n 1)
[ -n "$RUN" ] && [ -f $RUN/run_env.sh ] || { log "$R blind_deliver: no P run *_bl2P_$TAG with run_env.sh"; exit 2; }
PILOT=$(bash -c "source $RUN/run_env.sh > /dev/null 2>&1; echo \$BLIND_PILOT")
log "$R blind_deliver $CK${WHY:+ ($WHY)}"
mkdir -p logs/accept
Q=logs/accept/bl_${r}_bl2P_${TAG}_$K
rm -f $Q.verdict
echo "$R $CK BLIND $PILOT" > $Q.req
log "$R blind P avg$K: tinynav requested ($Q.req)"
OUT=logs/pkg/${r}_bl2_${TAG}_$K
mkdir -p $OUT
cp $CK $OUT/model_$K.pt && cp -r $RUN/params $OUT/ && cp $RUN/run_env.sh $OUT/ || { log "$R blind_deliver: package copy failed"; exit 4; }
echo "$PILOT" | sed 's/^,//' > $OUT/blind_pilot.txt
echo "$(bash -c "source $RUN/run_env.sh > /dev/null 2>&1; echo \${BLIND_PARAMS:-}")" > $OUT/blind_params.txt
md5sum $OUT/model_$K.pt | sed "s|$OUT/||" > $OUT/checkpoint.md5
bash jobs/export_policy.sh blind $R $OUT/model_$K.pt $OUT > logs/jobs/blind_${r}_export.log 2>&1 \
  || { log "$R blind_deliver: export failed: $(tail -n 3 logs/jobs/blind_${r}_export.log | tr '\n' ' ')"; exit 4; }
log "$R blind package $OUT (candidate until tinynav; policy.onnx $(md5sum < $OUT/policy.onnx | cut -c1-8))"
LOG=$L setsid nohup bash jobs/accept_videos.sh blind $R $OUT > /dev/null 2>&1 < /dev/null &
log "$R blind videos recording (jobs/accept_videos.sh, pid $!)"
w=0
until [ -f $Q.verdict ] || [ $w -ge 43200 ]; do sleep 60; w=$((w + 60)); done
tv=$(cat $Q.verdict 2> /dev/null || echo "no verdict in 12 h")
log "$R P avg$K tinynav: $tv"
if [[ "$tv" == PASS* ]]; then
  log "$R BLIND DELIVERY $OUT (model md5 $(md5sum < $OUT/model_$K.pt | cut -c1-8), policy.onnx $(md5sum < $OUT/policy.onnx | cut -c1-8))${WHY:+; $WHY}"
  exit 0
fi
exit 1
