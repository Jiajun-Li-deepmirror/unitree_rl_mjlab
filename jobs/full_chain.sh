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
# A round that fails is rerun from its own start after a recipe change (no resumed patching), under a new TAG:
# REPRO_BLIND_TAG names the blind delivery the depth rounds start from (default TAG; REPRO_PKG: a package path, e.g.
# copied from the other box), REPRO_FROM_TAG the tag whose teacher / stage-1 picks a later round starts from (copied
# into logs/picks_v2_<TAG>). Exit 0 pass, 1 fail, 3 not ready (the previous round has not passed).
# Log: logs/jobs/repro_<r>.txt. The tinynav gate runs on the workstation (tools/go2_depth/acceptance/tn_gate_watch.sh).
set -u
R=$1 TAG=$2 STAGE=${3:-all}
r=${R,,}
cd /root/gpufree-data/go2-depth/depth_fe && source env.sh
source jobs/recipes/$r.sh || exit 2
L=logs/jobs/repro_$r.txt
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $L; }
case $STAGE in all|blind|teacher|s1|s2) ;; *) echo "full_chain.sh: stage blind, teacher, s1, s2 or all" >&2; exit 2 ;; esac
# Hand-off (2026-11-10, the user: go2-home's own GPU helps with the run): logs/jobs/repro_away_<r> names where this
# robot's rounds run instead; here they are skipped (exit 3)
if [ -f logs/jobs/repro_away_$r ]; then log "$R $STAGE $TAG: runs elsewhere ($(cat logs/jobs/repro_away_$r)), skipped here"; exit 3; fi
log "$R $STAGE $TAG: code $(cat REPO_COMMIT 2> /dev/null || echo '?'), recipe jobs/recipes/$r.sh (md5 $(md5sum < jobs/recipes/$r.sh | cut -c1-8))"
if [ $STAGE = all ] || [ $STAGE = blind ]; then
  BL_PILOT_TM=$RECIPE_BLIND_TM BL_PILOT_P=$RECIPE_BLIND_P BL_CONS=$RECIPE_BLIND_CONS BL_T_MAX=${RECIPE_T_MAX:-5000} \
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
if [ $STAGE = blind ]; then log "$R blind $TAG: BLIND PASS $OUT (model_$PK)"; exit 0; fi
# picks of an earlier tag (a later round rerun under a new TAG starts from them)
PD=logs/picks_v2_$TAG FT=${REPRO_FROM_TAG:-$TAG}
mkdir -p $PD
if [ $FT != $TAG ]; then
  for f in ${r}_teacher ${r}_student_stage1; do
    [ -f $PD/$f ] || [ ! -f logs/picks_v2_$FT/$f ] || { cp logs/picks_v2_$FT/$f $PD/$f; log "$R $STAGE $TAG: $f from picks_v2_$FT ($(cat $PD/$f))"; }
  done
fi
case $STAGE in
  s1) [ -f $PD/${r}_teacher ] || { log "$R s1 $TAG: no teacher pick (round 2 has not passed)"; exit 3; } ;;
  s2) [ -f $PD/${r}_student_stage1 ] || { log "$R s2 $TAG: no stage-1 student (round 3 has not run)"; exit 3; } ;;
esac
log "$R $STAGE $TAG: from the blind package $OUT"
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
