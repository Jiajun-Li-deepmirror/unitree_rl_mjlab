#!/bin/bash
# full_chain.sh R TAG (go2-depth 2026-11-10; the user: "做一次干净复现 ... 每个机型用仓库里的最终配方，从零跑完整条链（盲走
# T → M → P → 打包 → 深度老师 → 学生 → 导出），中间不做任何人工修正，再走同一套验收"). One robot's whole chain from zero with
# jobs/recipes/<r>.sh and nothing else:
#   1. blind: jobs/blind_chain.sh with the recipe's pilots and constraint start (T from zero, M, P; the acceptance of
#      doc/acceptance.md: P on 3 seeds + the tinynav gate) -> logs/pkg/<r>_bl2_<TAG>_<K>
#   2. depth: jobs/v2_chain.sh full from that package (teacher from the widened blind, stage-1 student distilled from
#      scratch, stage-2 PPO + BC, 3 seeds + the tinynav gate, export)
# Log: logs/jobs/repro_<r>.txt. The tinynav gate runs on the workstation (tools/go2_depth/acceptance/tn_gate_watch.sh).
set -u
R=$1 TAG=$2
r=${R,,}
cd /root/gpufree-data/go2-depth/depth_fe && source env.sh
source jobs/recipes/$r.sh || exit 2
L=logs/jobs/repro_$r.txt
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $L; }
log "$R full chain $TAG: code $(cat REPO_COMMIT 2> /dev/null || echo '?'), recipe jobs/recipes/$r.sh (md5 $(md5sum < jobs/recipes/$r.sh | cut -c1-8))"
BL_PILOT_TM=$RECIPE_BLIND_TM BL_PILOT_P=$RECIPE_BLIND_P BL_CONS=$RECIPE_BLIND_CONS BL_EXTRA= BL_P_EXTRA= BL_DROP= \
  bash jobs/blind_chain.sh $R $TAG || { log "$R full chain $TAG: blind chain failed (logs/jobs/blind_$r.txt)"; exit 1; }
OUT=$(grep "$R BLIND DELIVERY logs/pkg/${r}_bl2_${TAG}_" logs/jobs/blind_$r.txt | tail -n 1 | sed 's/.*DELIVERY \([^ ]*\).*/\1/')
[ -n "$OUT" ] || { log "$R full chain $TAG: no blind delivery"; exit 1; }
PK=${OUT##*_}
log "$R full chain $TAG: blind delivery $OUT, depth chain next"
FE_PKG=$PWD/$OUT FE_N=$PK FE_PH=Polish FE_PSTART= V2_PKG_FROM=$PWD/$OUT V2_CONS_FORCE=$RECIPE_DEPTH_CONS \
  V2_EXTRA_FORCE=$RECIPE_DEPTH_EXTRA V2_SF=none V2_S1_ITERS=$RECIPE_S1_ITERS V2_S1_LR=$RECIPE_S1_LR \
  ${RECIPE_TOPFLAT:+FE_TEACHER_TOPFLAT=$RECIPE_TOPFLAT} \
  bash jobs/v2_chain.sh $R full $TAG || { log "$R full chain $TAG: depth chain failed (logs/jobs/fe_$r.txt)"; exit 1; }
log "$R full chain $TAG: done -- $(grep "$R STUDENT PICK" logs/jobs/fe_$r.txt | tail -n 1 | cut -c16-)"
