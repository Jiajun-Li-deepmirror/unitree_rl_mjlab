#!/bin/bash
# launch_run.sh NAME SEED ITERS ENVS [FT=SRC_RUN@K] [ARG=--flag=value ...] [VAR=VALUE ...] (go2-depth 2026-11-16, go2-home's run queue; from
# launch_arm.sh): robot Go2W's unified teacher, run name NAME, in the tree UNIFIED_ROOT (default the main tree: a
# commit's own tree from tools/go2_depth/tree_commit.sh), every other U_ variable unset, plus the run's switches.
# FT=SRC_RUN@K: a fine-tune -- SRC_RUN's model_K resumed (rsl_rl --agent.resume), ITERS MORE iterations (the run's
# checkpoints are numbered on from K: model_<K+100> ... model_<K+ITERS>); the run directory gets FINETUNE.txt.
# U_SMOKE=1 in the switches: a smoke (no gate). ARG=<flag>: passed to scripts/train.py as it is (2026-11-17: a diagnostic's
# --agent.save-interval=1000 -- six checkpoints when the disk is short). Prints "launched NAME ... pid P";
# logs/jobs/unified_go2w_NAME.log.
set -u
NAME=$1 SEED=$2 ITERS=$3 ENVS=$4; shift 4
ROOT=${UNIFIED_ROOT:-/root/gpufree-data/go2-depth/depth_fe}
cd $ROOT || exit 2
source env.sh > /dev/null 2>&1
for v in $(env | grep -oE '^U_[A-Z0-9_]+'); do unset $v; done
export UNIFIED_TASKS=1 TRAIN_LOG_LAYOUT=rsl_rl
FT="" SW=() XA=()
for kv in "$@"; do case $kv in FT=*) FT=${kv#FT=} ;; ARG=*) XA+=("${kv#ARG=}"); SW+=("$kv") ;; *) export "$kv"; SW+=("$kv") ;; esac; done
# (the date-prefixed name exactly: "*_$NAME" also matched this run's own smoke, usmoke_q_$NAME -- 2026-11-16, the
# base_s1 chain's launch was refused after its smoke usmoke_go2w_base_s1)
DT='[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]_[0-9][0-9]-[0-9][0-9]-[0-9][0-9]'
ls -d logs/rsl_rl/go2w_u_teacher/${DT}_$NAME > /dev/null 2>&1 && { echo "a run $NAME is there already"; exit 2; }
RES=()
if [ -n "$FT" ]; then
  SRC=${FT%@*} K=${FT#*@}
  [ -f logs/rsl_rl/go2w_u_teacher/$SRC/model_$K.pt ] || { echo "no $SRC/model_$K.pt"; exit 2; }
  RES=(--agent.resume=True "--agent.load-run=^$SRC\$" "--agent.load-checkpoint=^model_$K\\.pt\$")
fi
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
echo "$(date '+%m-%d %H:%M:%S') Go2W queued run $NAME (go2-depth for go2-home): seed $SEED, $ENVS robots, $ITERS it on $(cat REPO_COMMIT) ($ROOT)${FT:+, fine-tune from $FT}, switches: ${SW[*]:-none}; card at $used MiB" >> logs/jobs/unified_go2w_arms.txt
setsid nohup $PY scripts/train.py Unitree-Go2W-U-Teacher --env.scene.num-envs=$ENVS --agent.run-name=$NAME --agent.seed=$SEED \
  --agent.max-iterations=$ITERS "${RES[@]}" "${XA[@]}" > logs/jobs/unified_go2w_$NAME.log 2>&1 < /dev/null &
pid=$!
if [ -n "$FT" ]; then   # the fine-tune's mark in its run directory, once train.py made it
  ( for i in $(seq 120); do d=$(ls -d logs/rsl_rl/go2w_u_teacher/${DT}_$NAME 2>/dev/null | tail -n 1); [ -n "$d" ] && break; sleep 5; done
    [ -n "$d" ] && echo "fine-tune: resumed from logs/rsl_rl/go2w_u_teacher/$SRC/model_$K.pt; $ITERS more iterations (to $(( K + ITERS ))); commit $(cat REPO_COMMIT); switches ${SW[*]:-none}" > $d/FINETUNE.txt ) &
fi
echo "$(date '+%H:%M:%S') launched $NAME (Go2W, $ENVS robots, $ITERS it, seed $SEED${FT:+, fine-tune from $FT}, switches: ${SW[*]:-none}), pid $pid, log logs/jobs/unified_go2w_$NAME.log"
