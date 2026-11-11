#!/bin/bash
# recipe_stage.sh OLD [NEW] (go2-depth 2026-11-11, the user's rerun rule: "一轮不过时，从配方改动所在的最早阶段开始重跑，
# 它之前没改的阶段沿用已通过的结果；它之后的阶段全部重跑。只改步数上限时，从这个阶段开始，不重训前面的阶段").
# OLD: the recipe a TAG ran with (logs/jobs/recipe_<r>_<TAG>.sh, written by jobs/full_chain.sh); NEW: the recipe now
# (default jobs/recipes/<r>.sh of OLD's robot). Prints the earliest stage the change touches and what changed:
#   T        RECIPE_BLIND_T (T's pilot), RECIPE_BLIND_CONS (the constraint start), RECIPE_T_MAX
#   M        RECIPE_BLIND_M (M's pilot), RECIPE_M_ITERS (M's length: M again from T's pick)
#   (a recipe from before the T / M split has one RECIPE_BLIND_TM, read as both T's and M's)
#   P        RECIPE_BLIND_P, RECIPE_P_ITERS
#   teacher  RECIPE_DEPTH_EXTRA, RECIPE_DEPTH_CONS, RECIPE_TOPFLAT (the blind package stays)
#   s1       RECIPE_S1_ITERS, RECIPE_S1_LR
#   none     nothing changed
# An unset knob reads as its default (RECIPE_T_MAX 5000, RECIPE_M_ITERS 5000, RECIPE_P_ITERS 2000), so writing a
# default out is no change. Exit 0; 2 on a missing file.
set -u
OLD=$1
r=$(basename "$OLD" .sh); r=${r#recipe_}; r=${r%%_*}
NEW=${2:-$(dirname "$0")/recipes/$r.sh}
[ -f "$OLD" ] && [ -f "$NEW" ] || { echo "recipe_stage.sh: no $OLD or $NEW" >&2; exit 2; }
VARS="RECIPE_BLIND_T:T RECIPE_BLIND_CONS:T RECIPE_T_MAX:T RECIPE_BLIND_M:M RECIPE_M_ITERS:M RECIPE_BLIND_P:P RECIPE_P_ITERS:P
RECIPE_DEPTH_EXTRA:teacher RECIPE_DEPTH_CONS:teacher RECIPE_TOPFLAT:teacher RECIPE_S1_ITERS:s1 RECIPE_S1_LR:s1"
vals() {   # the recipe's knobs, one VAR=value per line, defaults filled in
  ( set +u; source "$1" > /dev/null 2>&1
    : "${RECIPE_T_MAX:=5000}" "${RECIPE_M_ITERS:=5000}" "${RECIPE_P_ITERS:=2000}"
    : "${RECIPE_BLIND_T:=${RECIPE_BLIND_TM:-}}" "${RECIPE_BLIND_M:=${RECIPE_BLIND_TM:-}}"
    for vs in $VARS; do v=${vs%%:*}; printf '%s=%s\n' "$v" "${!v-}"; done )
}
A=$(vals "$OLD") B=$(vals "$NEW")
stage="" changed=""
for st in T M P teacher s1; do
  for vs in $VARS; do
    v=${vs%%:*}; [ "${vs##*:}" = $st ] || continue
    if [ "$(grep "^$v=" <<< "$A")" != "$(grep "^$v=" <<< "$B")" ]; then
      changed="$changed $v"; [ -n "$stage" ] || stage=$st
    fi
  done
done
echo "${stage:-none}${changed:+ (changed:$changed)}"
