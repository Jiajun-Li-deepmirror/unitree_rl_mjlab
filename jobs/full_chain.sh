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
# Log: logs/jobs/repro_<r>.txt. The tinynav gate runs on the workstation (tools/go2_depth/acceptance/tn_gate_watch.sh).
set -u
R=$1 TAG=$2 STAGE=${3:-all}
r=${R,,}
cd /root/gpufree-data/go2-depth/depth_fe && source env.sh
source jobs/recipes/$r.sh || exit 2
L=logs/jobs/repro_$r.txt
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $L; }
case $STAGE in all|blind|teacher|s1|s2) ;; *) echo "full_chain.sh: stage blind, teacher, s1, s2 or all" >&2; exit 2 ;; esac
# A robot rerun under its own tag (2026-11-11): logs/jobs/repro_tag_<r> names the tag its rounds run under from then on,
# so a queue started before the rerun (which still says the old tag) runs the robot's later rounds on the rerun's results
if [ -s logs/jobs/repro_tag_$r ] && [ "$(cat logs/jobs/repro_tag_$r)" != "$TAG" ]; then
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
log "$R $STAGE $TAG: code $(cat REPO_COMMIT 2> /dev/null || echo '?'), recipe jobs/recipes/$r.sh (md5 $(md5sum < jobs/recipes/$r.sh | cut -c1-8))"
# The recipe this TAG runs with, kept for a later rerun's stage (jobs/recipe_stage.sh); the first round of the TAG writes it
SNAP=logs/jobs/recipe_${r}_$TAG.sh
[ -f $SNAP ] || { cp jobs/recipes/$r.sh $SNAP; log "$R $STAGE $TAG: recipe kept as $SNAP"; }
# pick_of TAG PHASE -> "PICK RUN": a blind phase's pick of an earlier TAG (logs/jobs/blind_<r>_<TAG>.picks; before
# 2026-11-11 from the chain log and the phase's newest run dir)
pick_of() {
  local f=logs/jobs/blind_${r}_$1.picks p
  [ -f $f ] && grep "^$2 " $f | tail -n 1 | cut -d' ' -f2- && return
  p=$(grep "$R blind $2 PICK" logs/jobs/blind_$r.txt 2> /dev/null | grep -o "logs/rsl_rl/[^ ]*bl2$2_$1_avg[0-9]*/model_[0-9]*\.pt" | tail -n 1)
  [ -n "$p" ] && echo "$p $(ls -d logs/rsl_rl/${r}_blind/*_bl2$2_$1 2> /dev/null | sort | tail -n 1)"
}
BF=${REPRO_BLIND_FROM:-}
if [ -n "$BF" ] && { [ $STAGE = all ] || [ $STAGE = blind ]; }; then
  BFS=${BF%@*} BFT=${BF#*@}
  if [ "$BFS" = auto ]; then
    BFS=$(bash jobs/recipe_stage.sh logs/jobs/recipe_${r}_$BFT.sh jobs/recipes/$r.sh 2> /dev/null) || {
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
elif { [ $STAGE = all ] || [ $STAGE = blind ]; } && grep -q "$R BLIND DELIVERY logs/pkg/${r}_bl2_${TAG}_" logs/jobs/blind_$r.txt 2> /dev/null; then
  log "$R $STAGE $TAG: the blind round already delivered, not run again"
elif [ $STAGE = all ] || [ $STAGE = blind ]; then
  BL_PILOT_TM=$RECIPE_BLIND_TM BL_PILOT_P=$RECIPE_BLIND_P BL_CONS=$RECIPE_BLIND_CONS BL_T_MAX=${RECIPE_T_MAX:-5000} \
  BL_M_ITERS=${RECIPE_M_ITERS:-5000} BL_P_ITERS=${RECIPE_P_ITERS:-2000} \
  BL_EXTRA= BL_P_EXTRA= BL_DROP= \
    bash jobs/blind_chain.sh $R $TAG || { log "$R $STAGE $TAG: blind chain failed (logs/jobs/blind_$r.txt)"; exit 1; }
fi
BT=${REPRO_BLIND_TAG:-$TAG}
OUT=${REPRO_PKG:-$(grep "$R BLIND DELIVERY logs/pkg/${r}_bl2_${BT}_" logs/jobs/blind_$r.txt 2> /dev/null | tail -n 1 | sed 's/.*DELIVERY \([^ ]*\).*/\1/')}
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
FE_PKG=$OUT FE_N=$PK FE_PH=Polish FE_PSTART= V2_PKG_FROM=$OUT V2_CONS_FORCE=$RECIPE_DEPTH_CONS \
  V2_EXTRA_FORCE=$RECIPE_DEPTH_EXTRA V2_SF=none V2_S1_ITERS=$RECIPE_S1_ITERS V2_S1_LR=$RECIPE_S1_LR \
  V2_STAGE=$STAGE ${RECIPE_TOPFLAT:+FE_TEACHER_TOPFLAT=$RECIPE_TOPFLAT} FE_TEACHER_STRICT=1 FE_INIT_RUN=blind_init_$(basename $OUT) \
  bash jobs/v2_chain.sh $R full $TAG
rc=$?
case $STAGE in
  teacher)
    [ $rc = 0 ] && [ -f $PD/${r}_teacher ] && { log "$R teacher $TAG: TEACHER PASS $(cat $PD/${r}_teacher)"; exit 0; }
    log "$R teacher $TAG: no teacher passed (logs/jobs/fe_$r.txt)"; exit 1 ;;
  s1)
    [ $rc = 0 ] && [ -f $PD/${r}_student_stage1 ] || { log "$R s1 $TAG: stage 1 failed (logs/jobs/fe_$r.txt)"; exit 1; }
    # not checked (the user); the behavior loss along the run, for a look at distillation from scratch
    f=logs/jobs/fe_${r}_student_v2s1_$TAG.log
    bl=$(grep 'Mean behavior loss' $f | awk '{print $NF}' | awk -v n=$RECIPE_S1_ITERS 'NR==1||NR==int(n/10)||NR==int(n/4)||NR==int(n/2)||NR==int(3*n/4)||NR==n-1{printf "%s@%d ", $1, NR}')
    log "$R s1 $TAG: STAGE-1 DONE $(cat $PD/${r}_student_stage1) ($(( ($(date +%s) - t0) / 60 )) min; behavior loss ${bl:-?})"; exit 0 ;;
  s2|all)
    p=$(grep "$R STUDENT PICK .*v2s2_$TAG" logs/jobs/fe_$r.txt | tail -n 1)
    [ $rc = 0 ] && [ -n "$p" ] && { log "$R $STAGE $TAG: STUDENT PASS -- $(echo "$p" | cut -c16-)"; exit 0; }
    log "$R $STAGE $TAG: no student passed (logs/jobs/fe_$r.txt)"; exit 1 ;;
esac
