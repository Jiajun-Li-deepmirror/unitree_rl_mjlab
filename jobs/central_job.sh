#!/bin/bash
# central_job.sh R TAG STAGE ID [VAR=value ...] (go2-depth 2026-11-11, the user: "中心队列：任何一块 GPU 空出来就自动
# 接手"): one job of the workstation's central queue (tools/go2_depth/queue/central_queue.py), started by it on a free
# box. First the smoke run (jobs/smoke_chain.sh R smoke<ID> SCOPE; the user: "每次正式开跑前 ... 把整条链走一遍") unless one
# already passed on this box with this code and this robot's recipe (logs/jobs/smoke/pass_<r>_<SCOPE> = REPO_COMMIT +
# the recipe's md5; CENTRAL_SMOKE=0 skips it). SCOPE: blind for a blind-only job (the user 11-11: "可以先不跑感知的冒烟"),
# else all (a depth round needs the whole chain walked); a passed all-smoke serves a blind job too. Then jobs/full_chain.sh R TAG STAGE with the VARs (e.g.
# REPRO_BLIND_FROM=M@rp6 for a rerun). logs/jobs/central/ID.pid while it runs, ID.rc its exit at the end:
#   full_chain's (0 pass, 1 fail, 3 not ready; a tool fault pauses inside full_chain, the job stays running),
#   5 the smoke run did not walk through (the pipeline is broken: the queue holds everything).
# Events in logs/jobs/central/central.txt.
set -u
R=$1 TAG=$2 ST=$3 ID=$4; shift 4
r=${R,,}
cd /root/gpufree-data/go2-depth/depth_fe || exit 2
C=logs/jobs/central
mkdir -p $C logs/jobs/smoke
echo $$ > $C/$ID.pid
log() { echo "$(date '+%m-%d %H:%M:%S') central $ID $R $TAG $ST: $*" >> $C/central.txt; }
fin() { echo $1 > $C/$ID.rc; exit $1; }
key="$(cat REPO_COMMIT) $( { python3 jobs/recipes/unified.py $r --sh; cat jobs/recipes/$r.sh; } 2> /dev/null | md5sum | cut -c1-8)"
SC=$([ $ST = blind ] && echo blind || echo all)
passed() { [ "$(cat logs/jobs/smoke/pass_${r}_$1 2> /dev/null)" = "$key" ]; }
if [ "${CENTRAL_SMOKE:-1}" = 1 ] && ! passed $SC && ! passed all; then
  log "smoke run first ($SC; code and recipe $key; logs/jobs/smoke/run_${r}_$ID.out)"
  t0=$(date +%s)
  bash jobs/smoke_chain.sh $R smoke$ID $SC > logs/jobs/smoke/run_${r}_$ID.out 2>&1
  src=$?
  if [ $src != 0 ]; then
    log "SMOKE FAILED (exit $src after $(( ($(date +%s) - t0) / 60 )) min): $(grep -E 'MISSING|TOOL FAULT' logs/jobs/smoke/run_${r}_$ID.out | head -n 3 | tr -s ' ' | tr '\n' ';' | cut -c1-300)"
    fin 5
  fi
  echo "$key" > logs/jobs/smoke/pass_${r}_$SC
  log "smoke passed ($(( ($(date +%s) - t0) / 60 )) min)"
fi
log "starts ($(cat REPO_COMMIT)${*:+; $*})"
t0=$(date +%s)
env "$@" bash jobs/full_chain.sh $R $TAG $ST
rc=$?
case $rc in 0) w=PASSED ;; 1) w="FAILED (a result: the rerun rule applies)" ;; 3) w="not ready (its previous round has not passed)" ;; *) w="ended with exit $rc" ;; esac
log "$w after $(( ($(date +%s) - t0) / 60 )) min"
fin $rc
