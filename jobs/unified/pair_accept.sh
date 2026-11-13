#!/bin/bash
# pair_accept.sh R DEPTH_CKPT BLIND_CKPT (unitree-rl-mjlab-62 2026-11-12; doc/redesign.md 2.2.1, the user: "加「相机失效、
# 切到盲走」测试"): the camera-failure test of the delivered pair -- the depth student and the blind student, each
# accepted on its own -- in mjlab: scripts/unified/camera_fail.py's flat and stairs parts, unloaded and with the
# walking payload (jobs/accept_payload.sh), ACCEPT_JOBS (2) at a time. Files logs/checks/u_<r>_pair_<tag>_<part>[_pay].txt
# (tag: the depth checkpoint's run and candidate); the verdict in logs/jobs/unified_<r>_pair[_smoke].verdict and the
# log logs/jobs/unified_<r>_accept.txt. A part is run again once when it gives no verdict line (a crash) or UNTESTED
# (too few robots standing at their failure in some cell), and up to 6 times 5 min apart when out of GPU memory; still
# without a verdict, it is a tool fault. Exit 0 PASS, 1 FAIL (the depth student is not delivered; the blind one is not
# affected), 4 a tool fault (not a result). PAIR_SMOKE=1: scale 0.05, unloaded only, judged as usual, its own verdict
# file (jobs/unified/smoke_students.sh: the tool chain, not the students).
# tinynav (tier 3, doc/redesign.md 2.3: map3 blinded mid-climb / mid-descent) is added once tinynav-sim has its switch.
set -u
R=$1 DCK=$2 BCK=$3 r=${1,,}
cd ${UNIFIED_ROOT:-/root/gpufree-data/go2-depth/depth_fe} || exit 2
source env.sh > /dev/null 2>&1
export UNIFIED_TASKS=1
SMOKE=${PAIR_SMOKE:-0} JOBS=${ACCEPT_JOBS:-2}
SFX=$([ $SMOKE = 1 ] && echo _smoke)
L=logs/jobs/unified_${r}_accept.txt V=logs/jobs/unified_${r}_pair$SFX.verdict
log() { echo "$(date '+%m-%d %H:%M:%S') $R pair: $([ $SMOKE = 1 ] && echo 'SMOKE ')$*" >> $L; }
PAY=$(bash jobs/accept_payload.sh $R) || exit 2
tag=$(basename $(dirname $DCK))_$(basename $DCK .pt)
P=logs/checks/u_${r}_pair${SFX}_$tag
SCALE=1.0 conds="unloaded loaded"
[ $SMOKE = 1 ] && SCALE=0.05 conds="unloaded"
# (a flow test's ACCEPT_TEST_DIV=<k>, jobs/unified/accept.sh: the scale 1/k)
[ "${ACCEPT_TEST_DIV:-1}" -gt 1 ] 2> /dev/null && SCALE=$(awk -v d=$ACCEPT_TEST_DIV 'BEGIN { print 1.0 / d }')
mkdir -p logs/checks
t0=$(date +%s)
log "camera-failure test: depth $DCK + blind $BCK ($conds)"
verdict_of() { grep -E "^camera_fail verdict:" $1 2> /dev/null | tail -n 1 | sed 's/^camera_fail verdict: //'; }
# Beside a training (2026-11-14, go2-depth: jobs/unified/students.sh's final acceptance, the other kind's student may
# still be training): a part starts only with its need free on the card -- PAIR_NEED_MB "flat:MiB stairs:MiB" (unset:
# no wait, as before) -- waiting up to ACCEPT_FREE_WAIT_S (0: no limit); no room in time leaves the part without a
# verdict, a tool fault (the same rule as accept.sh's memwait)
pair_need() { local e es; read -ra es <<< "${PAIR_NEED_MB:-}"; for e in "${es[@]}"; do [ "${e%%:*}" = $1 ] && { echo ${e#*:}; return; }; done; echo 0; }
roomwait() {   # PART OUT -> 0 room, 1 none in time (OUT says so)
  local need=$(pair_need $1) t0=$(date +%s) f
  [ "$need" -gt 0 ] 2> /dev/null || return 0
  while f=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | head -n 1 | awk -F', *' '{print $1 - $2}') \
        && [ "${f:-0}" -lt $need ]; do
    if [ "${ACCEPT_FREE_WAIT_S:-0}" -gt 0 ] && [ $(( $(date +%s) - t0 )) -ge $ACCEPT_FREE_WAIT_S ]; then
      echo "TOOL FAULT: no room on the card: needs $need MiB free, $f free for ${ACCEPT_FREE_WAIT_S} s" > $2
      log "TOOL FAULT: $(basename $2): no room on the card ($need MiB needed, $f free for ${ACCEPT_FREE_WAIT_S} s)"
      return 1
    fi
    sleep 20
  done
}
one_part() {   # PAYLOAD PART OUT
  local pay=$1 part=$2 out=$3 i retried=0 v
  export U_PAYLOAD=$pay
  for i in 1 2 3 4 5 6 7 8; do
    roomwait $part $out || return 0
    $PY scripts/unified/camera_fail.py $R $DCK $BCK --part $part --scale $SCALE --seed $((i - 1)) > $out 2>&1
    v=$(verdict_of $out)
    if [ -n "$v" ] && { [[ "$v" != UNTESTED* ]] || [ $SMOKE = 1 ]; }; then return 0; fi
    if grep -qiE "out of memory|CUDA error 2" $out; then
      [ $i -lt 7 ] && { sleep 300; continue; }
      return 0
    fi
    # a crash or too few robots tested: once more (another seed), then it stands as a tool fault
    [ $retried = 0 ] || return 0
    retried=1
  done
}
outs=()
for cond in $conds; do
  if [ $cond = unloaded ]; then pay=0,0,0,0 X=""; else pay=$PAY X=_pay; fi
  for part in flat stairs; do
    while [ $(jobs -rp | wc -l) -ge $JOBS ]; do wait -n; done
    out=${P}_$part$X.txt
    outs+=($out)
    one_part $pay $part $out &
  done
done
wait
fails=() fault=""
for out in "${outs[@]}"; do
  v=$(verdict_of $out)
  name=$(basename $out .txt | sed "s/^u_${r}_pair${SFX}_${tag}_//")
  if [ -z "$v" ]; then fault="$fault $name"; log "TOOL FAULT: $name: $(grep -E 'Error' $out | tail -n 1 | cut -c1-160)"
  elif [[ "$v" == UNTESTED* ]] && [ $SMOKE = 1 ]; then fails+=("$name: (smoke) $v")   # the tool ran: what smoke checks
  elif [[ "$v" == UNTESTED* ]]; then fault="$fault $name"; log "TOOL FAULT: $name: ${v#UNTESTED: }"
  elif [[ "$v" != PASS ]]; then fails+=("$name: ${v#FAIL: }"); fi
done
mins=$(( ($(date +%s) - t0) / 60 ))
if [ -n "$fault" ]; then echo "TOOL FAULT:$fault" > $V; log "($mins min) tool fault:$fault (not a result)"; exit 4; fi
if [ ${#fails[@]} = 0 ]; then echo "PASS $DCK $BCK" > $V; log "($mins min) PASS"; exit 0; fi
echo "FAIL: ${fails[*]}" > $V
log "($mins min) FAIL: ${fails[*]}"
exit 1
