#!/bin/bash
# full_chain.sh R TAG [blind|teacher|s1|s2|all] (go2-depth 2026-11-10; the user: "做一次干净复现 ... 每个机型用仓库里的最终配方，
# 从零跑完整条链 ... 中间不做任何人工修正，再走同一套验收"). One robot's round of the reproducible run, with
# jobs/recipes/<r>.sh and nothing else. Four rounds (the user, 11-10), each passed before the robot's next:
#   blind:   jobs/blind_chain.sh T -> M -> P from zero with the recipe's pilots and constraint start; passes on the
#            unified acceptance (doc/acceptance.md: 3 seeds + tinynav) -> package logs/pkg/<r>_bl2_<TAG>_<K>
#   teacher: the blind package widened with the height scan, the depth teacher trained on (jobs/v2_chain.sh full,
#            V2_STAGE=teacher); passes on the teacher gate (mjlab check with the speed margin, 0.05 legged / 0.08 wheeled;
#            wheeled also the top-landing lift time <= RECIPE_TOPFLAT %) -> logs/picks_v2_<TAG>/<r>_teacher
#   s1:      stage-1 student distilled from scratch under that teacher (RECIPE_S1_ITERS, RECIPE_S1_LR); not checked,
#            its behavior loss logged for a look -> logs/picks_v2_<TAG>/<r>_student_stage1
#   s2:      stage-2 student (PPO + BC) from the stage-1 average; passes on the unified acceptance, exported
#   all:     the four in a row
# A round that fails is rerun after a recipe change (no resumed patching), under a new TAG, from the earliest stage the
# change touches (2026-11-11, the user: "从配方改动所在的最早阶段开始重跑，它之前没改的阶段沿用已通过的结果；它之后的阶段
# 全部重跑。只改步数上限时，从这个阶段开始，不重训前面的阶段。沿用的结果要在日志里写明来自哪个 TAG"; jobs/recipe_stage.sh
# names that stage from logs/jobs/recipe_<r>_<TAG>.sh, the recipe each TAG ran with, written here):
#   REPRO_BLIND_FROM=M@<tag> / P@<tag>: the blind round starts at M (T reused from <tag>) or P (T and M reused);
#     auto@<tag>: the stage recipe_stage.sh finds (T: from zero; a depth stage: the blind round is <tag>'s, not rerun)
#   REPRO_BLIND_TAG names the blind delivery the depth rounds start from (default TAG; REPRO_PKG: a package path, e.g.
#     copied from the other box)
#   REPRO_FROM_TAG the tag whose picks of the stages before this round are reused (copied into logs/picks_v2_<TAG>):
#     s1 reuses its teacher, s2 its teacher and stage-1 student
# Every reused result is logged with the TAG it comes from. Exit 0 pass, 1 fail, 3 not ready (the previous round has
# not passed).
# A tool fault (2026-11-11, the user: "崩溃、加载失败、找不到训练目录这类问题，属于工具链故障。它们应自动重试，或者暂停等人
# 处理，不应判为这一轮失败，也不应把队列推到下一个机型"; the chains exit 4) is never a result: the round is retried once
# after 2 min, then this job pauses until logs/jobs/repro_hold_<r> is removed, and goes on from the stage it reached
# (the blind round from this TAG's own picks).
# Log: logs/jobs/repro_<r>.txt. The tinynav gate runs on the workstation (tools/go2_depth/acceptance/tn_gate_watch.sh).
set -u
R=$1 TAG=$2 STAGE=${3:-all}
r=${R,,}
cd /root/gpufree-data/go2-depth/depth_fe && source env.sh
# The recipe. The blind round's comes from code (2026-11-11, the user: "配方由代码定义 ... 一份共用基础配方，加上腿式和轮式各自的
# 部分，再加少量每个机型自己的差异"): jobs/recipes/unified.py (T, M and P keys apart, the step limits, constraints from 1x).
# jobs/recipes/<r>.sh still gives the depth rounds' values. RECIPE_SOURCE=legacy: <r>.sh's blind values (T = M = its
# RECIPE_BLIND_TM), as every run before d13194e.
source jobs/recipes/$r.sh || exit 2
RECIPE_SOURCE=${RECIPE_SOURCE:-unified}
if [ $RECIPE_SOURCE = unified ]; then
  RVARS=$(python3 jobs/recipes/unified.py $r --sh) || { echo "full_chain.sh: jobs/recipes/unified.py $r failed" >&2; exit 2; }
  eval "$RVARS"
  unset RECIPE_BLIND_TM
fi
RECIPE_BLIND_T=${RECIPE_BLIND_T:-${RECIPE_BLIND_TM:-}} RECIPE_BLIND_M=${RECIPE_BLIND_M:-${RECIPE_BLIND_TM:-}}
# the values this run trains with, one VAR=value per line (the recipe snapshot; md5 in the log)
recipe_vals() { local v; for v in $(compgen -v RECIPE_ | sort); do [ $v = RECIPE_SOURCE ] || printf '%s=%q\n' $v "${!v}"; done; }
L=${REPRO_LOG:-logs/jobs/repro_$r.txt}
BLOG=${BL_LOG:-logs/jobs/blind_$r.txt} FLOG=${FE_LOG:-logs/jobs/fe_$r.txt}   # the chains' logs (a smoke run: logs/jobs/smoke/)
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $L; }
FAULTS=0
on_fault() {   # REASON
  FAULTS=$((FAULTS + 1))
  log "$R $STAGE $TAG: TOOL FAULT $FAULTS ($1) -- not a failed round"
  echo "$(date '+%m-%d %H:%M:%S') repro: $R $STAGE $TAG TOOL FAULT $FAULTS: $1" >> logs/jobs/repro_queue.txt
  [ "${CHAIN_SMOKE:-0}" = 1 ] && exit 4
  if [ $FAULTS -le 1 ]; then log "$R $STAGE $TAG: retried in 2 min"; sleep 120; return; fi
  local H=logs/jobs/repro_hold_$r
  echo "$R $STAGE $TAG: $1" > $H
  log "$R $STAGE $TAG: PAUSED after $FAULTS tool faults; remove $H to go on"
  echo "$(date '+%m-%d %H:%M:%S') repro: $R $STAGE $TAG PAUSED (remove $H to go on)" >> logs/jobs/repro_queue.txt
  while [ -f $H ]; do sleep 60; done
  log "$R $STAGE $TAG: $H removed, going on"
}
case $STAGE in all|blind|teacher|s1|s2) ;; *) echo "full_chain.sh: stage blind, teacher, s1, s2 or all" >&2; exit 2 ;; esac
# A robot rerun under its own tag (2026-11-11): logs/jobs/repro_tag_<r> names the tag its rounds run under from then on,
# so a queue started before the rerun (which still says the old tag) runs the robot's later rounds on the rerun's results
# CHAIN_SMOKE=1 (2026-11-11, jobs/smoke_chain.sh): the recipe with every stage a few iterations long, 64 robots,
# checks forced to pass; a smoke run never follows logs/jobs/repro_tag_<r>, never takes the queue's insert file and never
# pauses on a tool fault (it ends with exit 4)
if [ "${CHAIN_SMOKE:-0}" = 1 ]; then
  [[ $TAG == smoke* ]] || { echo "full_chain.sh: a smoke TAG starts with smoke" >&2; exit 2; }
  RECIPE_T_MAX=60 RECIPE_M_ITERS=60 RECIPE_P_ITERS=60 RECIPE_S1_ITERS=40
  export REPRO_INSERTED=1 BL_NUM_ENVS=64 BL_SAVE=10 BL_STEP=20 BL_FIRST_T=20 BL_FIRST_M=20 BL_FIRST_P=20 BL_P_SEEDS=0 \
    V2_TKS="20 40" V2_S2_ITERS=40 V2_S2_KS="20 39" FE_TEACHER_ITERS=41 FE_TEACHER_ENVS=64 FE_STUDENT_ENVS=32 FE_SAVE=10 \
    FE_LOCK=/tmp/chain_smoke_check.lock
fi
if [ "${CHAIN_SMOKE:-0}" != 1 ] && [ -s logs/jobs/repro_tag_$r ] && [ "$(cat logs/jobs/repro_tag_$r)" != "$TAG" ]; then
  NT=$(cat logs/jobs/repro_tag_$r); log "$R $STAGE $TAG: runs as $NT (logs/jobs/repro_tag_$r)"; TAG=$NT
fi
# Inserted jobs (2026-11-10): logs/jobs/repro_insert, one shell command per line, run before this job (a queue that is
# already running reads this file at its next job; e.g. jobs/t_handoff.sh for a chain that ended under older code)
INS=logs/jobs/repro_insert
if [ -z "${REPRO_INSERTED:-}" ] && [ -s $INS ]; then
  mv $INS $INS.$$
  while IFS= read -r job; do
    [ -n "$job" ] || continue
    echo "$(date '+%m-%d %H:%M:%S') repro: inserted before $R $STAGE $TAG: $job" >> logs/jobs/repro_queue.txt
    REPRO_INSERTED=1 bash -c "$job" < /dev/null
    echo "$(date '+%m-%d %H:%M:%S') repro: inserted job ended (exit $?): $job" >> logs/jobs/repro_queue.txt
  done < $INS.$$
  rm -f $INS.$$
fi
# Hand-off (2026-11-10, the user: go2-home's own GPU helps with the run): logs/jobs/repro_away_<r> names where this
# robot's rounds run instead; here they are skipped (exit 3)
if [ -f logs/jobs/repro_away_$r ]; then log "$R $STAGE $TAG: runs elsewhere ($(cat logs/jobs/repro_away_$r)), skipped here"; exit 3; fi
log "$R $STAGE $TAG: code $(cat REPO_COMMIT 2> /dev/null || echo '?'), recipe $RECIPE_SOURCE (values md5 $(recipe_vals | md5sum | cut -c1-8); T $(echo $RECIPE_BLIND_T | tr , '\n' | grep -c .) / M $(echo $RECIPE_BLIND_M | tr , '\n' | grep -c .) / P $(echo $RECIPE_BLIND_P | tr , '\n' | grep -c .) keys)"
# The recipe this TAG runs with, kept for a later rerun's stage (jobs/recipe_stage.sh); the first round of the TAG writes it
SNAP=logs/jobs/recipe_${r}_$TAG.sh
[ -f $SNAP ] || { recipe_vals > $SNAP; log "$R $STAGE $TAG: recipe kept as $SNAP"; }
# pick_of TAG PHASE -> "PICK RUN": a blind phase's pick of an earlier TAG (logs/jobs/blind_<r>_<TAG>.picks; before
# 2026-11-11 from the chain log and the phase's newest run dir)
pick_of() {
  local f=logs/jobs/blind_${r}_$1.picks p
  [ -f $f ] && grep "^$2 " $f | tail -n 1 | cut -d' ' -f2- && return
  p=$(grep "$R blind $2 PICK" $BLOG 2> /dev/null | grep -o "logs/rsl_rl/[^ ]*bl2$2_$1_avg[0-9]*/model_[0-9]*\.pt" | tail -n 1)
  [ -n "$p" ] && echo "$p $(ls -d logs/rsl_rl/${r}_blind/*_bl2$2_$1 2> /dev/null | sort | tail -n 1)"
}
BF=${REPRO_BLIND_FROM:-}
if [ -n "$BF" ] && { [ $STAGE = all ] || [ $STAGE = blind ]; }; then
  BFS=${BF%@*} BFT=${BF#*@}
  if [ "$BFS" = auto ]; then
    BFS=$(bash jobs/recipe_stage.sh logs/jobs/recipe_${r}_$BFT.sh <(recipe_vals) 2> /dev/null) || {
      log "$R $STAGE $TAG: REPRO_BLIND_FROM=$BF: no recipe of $BFT kept (logs/jobs/recipe_${r}_$BFT.sh); name the stage"; exit 1; }
    log "$R $STAGE $TAG: recipe since $BFT: $BFS"
    BFS=${BFS%% *}
  fi
  case $BFS in
    T) ;;
    M|P)
      TP=$(pick_of $BFT T)
      [ -n "$TP" ] || { log "$R $STAGE $TAG: no T pick of $BFT to reuse"; exit 1; }
      export BL_T_PICK=${TP% *} BL_T_RUN=${TP#* } BL_T_PICK_WHY="reused from TAG $BFT (recipe unchanged before $BFS)"
      log "$R blind $TAG: T reused from TAG $BFT: $BL_T_PICK (run $BL_T_RUN); runs from $BFS"
      if [ $BFS = P ]; then
        MP=$(pick_of $BFT M)
        [ -n "$MP" ] || { log "$R $STAGE $TAG: no M pick of $BFT to reuse"; exit 1; }
        export BL_M_PICK=${MP% *} BL_M_RUN=${MP#* } BL_M_PICK_WHY="reused from TAG $BFT (recipe unchanged before P)"
        log "$R blind $TAG: M reused from TAG $BFT: $BL_M_PICK (run $BL_M_RUN)"
      fi ;;
    teacher|s1|none)
      export REPRO_BLIND_TAG=${REPRO_BLIND_TAG:-$BFT}
      log "$R blind $TAG: the recipe's blind stages are unchanged since $BFT ($BFS): the blind round reused from TAG $BFT"
      [ $STAGE = blind ] && STAGE=reuse ;;
    *) log "$R $STAGE $TAG: REPRO_BLIND_FROM=$BF: stage T, M, P or auto"; exit 2 ;;
  esac
fi
if [ -n "${REPRO_BLIND_TAG:-}" ] && [ "$REPRO_BLIND_TAG" != $TAG ] && { [ $STAGE = all ] || [ $STAGE = reuse ]; }; then
  log "$R $STAGE $TAG: blind round not run (reused from TAG $REPRO_BLIND_TAG)"
  [ $STAGE = reuse ] && STAGE=blind
elif { [ $STAGE = all ] || [ $STAGE = blind ]; } && grep -q "$R BLIND DELIVERY logs/pkg/${r}_bl2_${TAG}_" $BLOG 2> /dev/null; then
  log "$R $STAGE $TAG: the blind round already delivered, not run again"
elif [ $STAGE = all ] || [ $STAGE = blind ]; then
  while :; do
    if [ $FAULTS -gt 0 ]; then   # after a tool fault: on from the stage this TAG reached (its own picks)
      TP=$(pick_of $TAG T) MP=$(pick_of $TAG M)
      if [ -n "$TP" ]; then
        export BL_T_PICK=${TP% *} BL_T_RUN=${TP#* } BL_T_PICK_WHY="this TAG's, kept after the tool fault"
        log "$R blind $TAG: after the tool fault, T reused from TAG $TAG: $BL_T_PICK"
      fi
      if [ -n "$TP" ] && [ -n "$MP" ]; then
        export BL_M_PICK=${MP% *} BL_M_RUN=${MP#* } BL_M_PICK_WHY="this TAG's, kept after the tool fault"
        log "$R blind $TAG: after the tool fault, M reused from TAG $TAG: $BL_M_PICK"
      fi
    fi
    BLIND_PARAMS=${RECIPE_BLIND_PARAMS:-} BL_PILOT_T=$RECIPE_BLIND_T BL_PILOT_M=$RECIPE_BLIND_M BL_PILOT_P=$RECIPE_BLIND_P BL_CONS=$RECIPE_BLIND_CONS BL_T_MAX=${RECIPE_T_MAX:-5000} \
    BL_M_ITERS=${RECIPE_M_ITERS:-5000} BL_P_ITERS=${RECIPE_P_ITERS:-2000} \
    BL_EXTRA= BL_P_EXTRA= BL_DROP= \
      bash jobs/blind_chain.sh $R $TAG
    rc=$?
    [ $rc = 4 ] || break
    on_fault "$(cat logs/jobs/blind_${r}_$TAG.fault 2> /dev/null || echo 'see logs/jobs/blind_'$r'.txt')"
  done
  [ $rc = 0 ] || { log "$R $STAGE $TAG: blind chain failed ($BLOG)"; exit 1; }
fi
BT=${REPRO_BLIND_TAG:-$TAG}
OUT=${REPRO_PKG:-$(grep "$R BLIND DELIVERY logs/pkg/${r}_bl2_${BT}_" $BLOG 2> /dev/null | tail -n 1 | sed 's/.*DELIVERY \([^ ]*\).*/\1/')}
if [ -z "$OUT" ] || [ ! -f $OUT/policy.onnx ]; then
  log "$R $STAGE $TAG: no blind delivery of $BT${OUT:+ ($OUT has no policy.onnx)}"
  if [ $STAGE = blind ] || [ $STAGE = all ]; then exit 1; else exit 3; fi
fi
OUT=$(cd $OUT && pwd)
PK=${OUT##*_}
if [ $STAGE = blind ]; then log "$R blind $TAG: BLIND PASS $OUT (model_$PK${REPRO_BLIND_TAG:+, the blind round of TAG $BT})"; exit 0; fi
# picks of an earlier tag (a later round rerun under a new TAG starts from them)
PD=logs/picks_v2_$TAG FT=${REPRO_FROM_TAG:-$TAG}
mkdir -p $PD
# only the stages before this round (2026-11-11): s1 reuses the teacher, s2 the teacher and the stage-1 student
case $STAGE in s1) REUSE="${r}_teacher" ;; s2) REUSE="${r}_teacher ${r}_student_stage1" ;; *) REUSE="" ;; esac
if [ $FT != $TAG ]; then
  for f in $REUSE; do
    [ -f $PD/$f ] || [ ! -f logs/picks_v2_$FT/$f ] || { cp logs/picks_v2_$FT/$f $PD/$f; log "$R $STAGE $TAG: $f reused from TAG $FT ($(cat $PD/$f))"; }
  done
fi
case $STAGE in
  s1) [ -f $PD/${r}_teacher ] || { log "$R s1 $TAG: no teacher pick (round 2 has not passed)"; exit 3; } ;;
  s2) [ -f $PD/${r}_student_stage1 ] || { log "$R s2 $TAG: no stage-1 student (round 3 has not run)"; exit 3; } ;;
esac
log "$R $STAGE $TAG: from the blind package $OUT (the blind round of TAG $BT)"
t0=$(date +%s)
while :; do
  FE_PKG=$OUT FE_N=$PK FE_PH=Polish FE_PSTART= V2_PKG_FROM=$OUT V2_CONS_FORCE=$RECIPE_DEPTH_CONS \
    V2_EXTRA_FORCE=$RECIPE_DEPTH_EXTRA V2_SF=none V2_S1_ITERS=$RECIPE_S1_ITERS V2_S1_LR=$RECIPE_S1_LR \
    V2_STAGE=$STAGE FE_TEACHER_TOPFLAT=${RECIPE_TOPFLAT:-} FE_TEACHER_STRICT=1 FE_INIT_RUN=blind_init_$(basename $OUT) \
    bash jobs/v2_chain.sh $R full $TAG
  rc=$?
  [ $rc = 4 ] || break
  on_fault "$(cat ${FLOG%.txt}.fault 2> /dev/null || echo "see $FLOG")"
done
case $STAGE in
  teacher)
    [ $rc = 0 ] && [ -f $PD/${r}_teacher ] && { log "$R teacher $TAG: TEACHER PASS $(cat $PD/${r}_teacher)"; exit 0; }
    log "$R teacher $TAG: no teacher passed ($FLOG)"; exit 1 ;;
  s1)
    [ $rc = 0 ] && [ -f $PD/${r}_student_stage1 ] || { log "$R s1 $TAG: stage 1 failed ($FLOG)"; exit 1; }
    # not checked (the user); the behavior loss along the run, for a look at distillation from scratch
    f=logs/jobs/fe_${r}_student_v2s1_$TAG.log
    bl=$(grep 'Mean behavior loss' $f | awk '{print $NF}' | awk -v n=$RECIPE_S1_ITERS 'NR==1||NR==int(n/10)||NR==int(n/4)||NR==int(n/2)||NR==int(3*n/4)||NR==n-1{printf "%s@%d ", $1, NR}')
    log "$R s1 $TAG: STAGE-1 DONE $(cat $PD/${r}_student_stage1) ($(( ($(date +%s) - t0) / 60 )) min; behavior loss ${bl:-?})"; exit 0 ;;
  s2|all)
    p=$(grep "$R STUDENT PICK .*v2s2_$TAG" $FLOG | tail -n 1)
    [ $rc = 0 ] && [ -n "$p" ] && { log "$R $STAGE $TAG: STUDENT PASS -- $(echo "$p" | cut -c16-)"; exit 0; }
    log "$R $STAGE $TAG: no student passed ($FLOG)"; exit 1 ;;
esac
