#!/bin/bash
# tb_box.sh PORT [SINCE|-] [stop] -- the box's side of tb.sh (go2-depth 2026-11-12). tb.sh feeds it to the box's bash
# over ssh; nothing is installed on the box. See tb.sh.
set -u
export LC_ALL=C
PORT=$1 SINCE=${2:--} STOP=${3:-}
cd /root/gpufree-data/go2-depth/depth_fe || exit 2
source env.sh > /dev/null 2>&1
mkdir -p logs/tb_unified && cd logs/tb_unified || exit 2
alive() { [ -f PID ] && kill -0 "$(cat PID)" 2> /dev/null && grep -q tensorboard "/proc/$(cat PID)/cmdline" 2> /dev/null; }
if [ "$STOP" = stop ]; then alive && kill "$(cat PID)"; rm -f PID; echo stopped; exit 0; fi
[ "$SINCE" != - ] && echo "$SINCE" > SINCE
s=$(cat SINCE 2> /dev/null)
shown() { [[ $1 =~ _s[0-9]+$ ]] && [[ -z $s || ! $1 < $s ]]; }   # a run under its own name (<date>_<time>_<name>_s<seed>), not older than SINCE
n_runs=0
for e in ../rsl_rl/*_u_*; do
  [ -d "$e" ] || continue
  n=$(basename "$e"); mkdir -p "$n"
  for l in "$n"/*; do   # links that no longer belong: older than SINCE, or the run was renamed (_discard ...) or is gone
    [ -L "$l" ] || continue
    m=$(basename "$l")
    { [ -d "$e/$m" ] && shown "$m"; } || rm -f "$l"
  done
  for r in "$e"/*; do
    m=$(basename "$r")
    shown "$m" || continue
    [ -L "$n/$m" ] || ln -s "../$r" "$n/$m"
    n_runs=$((n_runs + 1))
  done
done
if alive; then echo "running (pid $(cat PID)), $n_runs run(s) shown${s:+ since $s}"; exit 0; fi
if ss -ltn 2> /dev/null | grep -q ":$PORT "; then echo "port $PORT is taken on the box and not by our TensorBoard: not started"; exit 1; fi
setsid nohup $PY -m tensorboard.main --logdir "$PWD" --host 127.0.0.1 --port "$PORT" --reload_interval 30 \
  --samples_per_plugin scalars=3000 > tensorboard.log 2>&1 < /dev/null &
echo $! > PID
echo "started (pid $!), $n_runs run(s) shown${s:+ since $s}"
