#!/bin/bash
# baseline.sh R [SRC] [NAME] (unitree-rl-mjlab-62 2026-11-12, the review's item 12; doc/redesign.md step 5): the
# reading of the deployed blind policy under the unified acceptance's tier 2 -- the same tests, robots, payload and
# metrics as a unified student -- so a unified student can be put beside it. SRC (default for Go2: the deployed
# go2_blind_v4, logs/rsl_rl/go2_blind/pkg_go2_fe_polish_10000/model_10000.pt) becomes a stage-2 checkpoint in
# logs/rsl_rl/<r>_u_baseline/<NAME> (scripts/unified/baseline_ckpt.py), then jobs/unified/accept.sh BASELINE=1 runs
# tier 2 on it: no pick, no tinynav, no delivery. Readings: logs/checks/u_<r>_blind_<NAME>_<K>_*.txt and
# ..._verdict.txt (the judge's items against the bars); log logs/jobs/unified_<r>_accept.txt.
# Takes a GPU for ~45 min (the tier-2 tests); start it only when the user has said when.
set -u
R=$1 r=${1,,}
case $R in Go2) DEF=logs/rsl_rl/go2_blind/pkg_go2_fe_polish_10000/model_10000.pt DEFN=go2_blind_v4 ;; *) DEF="" DEFN="" ;; esac
SRC=${2:-$DEF} NAME=${3:-${DEFN:-baseline}}
[ -n "$SRC" ] || { echo "baseline.sh: no default source for $R; give SRC" >&2; exit 2; }
cd ${UNIFIED_ROOT:-/root/gpufree-data/go2-depth/depth_fe} || exit 2
source env.sh > /dev/null 2>&1
[ -f "$SRC" ] || { echo "baseline.sh: no $SRC" >&2; exit 2; }
OUT=logs/rsl_rl/${r}_u_baseline/$NAME
UNIFIED_TASKS=1 $PY scripts/unified/baseline_ckpt.py $R $SRC $OUT || exit 2
K=$(ls $OUT/model_*.pt | grep -oE 'model_[0-9]+' | cut -d_ -f2 | sort -n | tail -n 1)
BASELINE=1 CANDIDATES=first TN_GATE=skip bash jobs/unified/accept.sh $R blind $OUT $K
echo "baseline readings: logs/checks/u_${r}_blind_${NAME}_${K}_*.txt (verdict: logs/checks/u_${r}_blind_${NAME}_${K}_verdict.txt)"
