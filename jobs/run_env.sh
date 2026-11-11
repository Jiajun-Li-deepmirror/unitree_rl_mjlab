#!/bin/bash
# run_env.sh CK_OR_DIR (go2-depth 2026-11-11, the user's unification item 1): the run_env.sh that belongs to a checkpoint
# (the one in its directory: a training run, an average made by jobs/avg_ckpt.py, a package) or to a directory; prints
# its path, exit 1 when there is none (runs from before 11-11: rebuild the environment the old way). scripts/train.py
# writes it at every training's start: every BLIND_* / DEPTH_STUDENT_* / UNIFIED_TASKS / U_SMOKE / TRAIN_LOG_LAYOUT
# variable, RUN_TASK and RUN_ARGV. Use: E=$(bash jobs/run_env.sh $CK) && source $E
p=${1:?run_env.sh: CK_OR_DIR}
[ -d "$p" ] && d=$p || d=$(dirname "$p")
[ -f "$d/run_env.sh" ] && { echo "$d/run_env.sh"; exit 0; }
exit 1
