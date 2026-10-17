#!/bin/bash
# Reproduce the released blind Go2 policy (deploy/robots/go2/.../blind_v0): four
# stages, each resuming the previous one, with the same tasks, iteration counts,
# seed (42) and 4096 envs as the original runs. About 20 h on one RTX 5060 Ti.
# See doc/go2_blind.md.
#
#   bash scripts/go2_blind/train_release.sh       # stages A, B, C, D
#   bash scripts/go2_blind/train_release.sh C     # start at C (A and B already done)
#
# A  Unitree-Go2-Blind-StageA  from scratch           iterations     0 ->  6000
# B  Unitree-Go2-Blind-StageB  + terrain clearance    iterations  6000 ->  9000
# C  Unitree-Go2-Blind         + foothold gain,       iterations  9000 -> 18000
#                              action std x1.5 at the start
# D  Unitree-Go2-Blind-Map3    map3 fine-tune         iterations 18000 -> 20000
#
# A run of N iterations from iteration i ends with model_{i+N-1}.pt, hence the
# +1 in --agent.max-iterations. Check each stage before the next (doc/go2_blind.md):
# the original stage C run diverged after ~19100 iterations.
set -euo pipefail
cd "$(dirname "$0")/../.."

FIRST=${1:-A}
ROOT=logs/rsl_rl/go2_blind

latest_run() {  # Newest run directory name with the given run-name suffix.
  local dir
  dir=$(ls -d "$ROOT"/*_"$1" 2>/dev/null | sort | tail -n 1)
  [ -n "$dir" ] || { echo "no run *_$1 in $ROOT" >&2; exit 1; }
  basename "$dir"
}

train() {
  python scripts/train.py "$@" --env.scene.num-envs=4096
}

resume() {  # resume <task> <previous run name> <checkpoint> <iterations> <run name>
  train "$1" --agent.resume=True --agent.load-run="$(latest_run "$2")" \
    --agent.load-checkpoint="$3" --agent.max-iterations="$4" --agent.run-name="$5"
}

case $FIRST in
  A) train Unitree-Go2-Blind-StageA --agent.max-iterations=6001 --agent.run-name=release_a ;&
  B) resume Unitree-Go2-Blind-StageB release_a model_6000.pt 3001 release_b ;&
  C)
    B_RUN=$ROOT/$(latest_run release_b)
    python scripts/go2_blind/scale_action_std.py "$B_RUN/model_9000.pt" "$B_RUN/model_9000_explore.pt" --scale 1.5
    resume Unitree-Go2-Blind release_b model_9000_explore.pt 9001 release_c
    ;&
  D) resume Unitree-Go2-Blind-Map3 release_c model_18000.pt 2001 release_d ;;
  *) echo "usage: $0 [A|B|C|D]" >&2; exit 1 ;;
esac

echo "Done: $ROOT/$(latest_run release_d)/model_20000.pt"
