#!/bin/bash
# tb.sh [--since DATE_TIME] [--stop] (go2-depth 2026-11-12, the user: "现在有训练监控吗？类似tensorboard，可以实时知道训练曲线，
# 机器是否真的有在学习"): TensorBoard over the unified trainings' event files, live. One TensorBoard per box (it reads
# the files where the trainings write them, reloading every 30 s), bound to the box's 127.0.0.1 and forwarded to the
# workstation through the ssh control socket:
#   http://localhost:16061   t1 (Go2)          http://localhost:16062   t2 (Go2W)
# Shown: the runs of logs/rsl_rl/*_u_* on the box that still carry their own name (<date>_<time>_<name>_s<seed>; a
# renamed run -- _discard, _pre_relax ... -- is left out) and are not older than --since (e.g. 2026-11-12_13; the box
# remembers the last one given in logs/tb_unified/SINCE; none given ever: every run). The box side is tb_box.sh.
# Idempotent: it starts what is missing (the TensorBoard, the forward) and brings the shown runs up to date; the
# 30-minute loop calls it. --stop ends the TensorBoards (the pid in logs/tb_unified/PID) and cancels the forwards.
# Curves worth a look are listed in doc/unified_teacher.md ("Watching a training").
set -u
SINCE=- STOP=""
while [ $# -gt 0 ]; do case $1 in --since) SINCE=$2; shift ;; --stop) STOP=stop ;; *) echo "tb.sh: unknown argument $1" >&2; exit 2 ;; esac; shift; done
cd "$(dirname "$0")" || exit 2
d=/home/dm/.cache/go2-depth-work/ssh
declare -A PORT=([t1]=30714 [t2]=31365) TB=([t1]=16061 [t2]=16062) WHAT=([t1]=Go2 [t2]=Go2W)
for b in t1 t2; do
  P=${PORT[$b]} T=${TB[$b]}
  out=$(timeout 60 $d/r $P "bash -s -- $T '$SINCE' $STOP" < tb_box.sh) || { echo "[$b] ${out:-the box does not answer}"; continue; }
  o=(-o ControlPath=$d/%p -p $P)
  if [ -n "$STOP" ]; then
    ssh "${o[@]}" -O cancel -L 127.0.0.1:$T:127.0.0.1:$T root@183.147.142.40 2> /dev/null
    echo "[$b] $out"; continue
  fi
  ssh "${o[@]}" -O forward -L 127.0.0.1:$T:127.0.0.1:$T root@183.147.142.40 2> /dev/null
  ok="does not answer yet"
  for _ in 1 2 3 4 5 6; do curl -s -m 5 -o /dev/null http://127.0.0.1:$T/data/environment && { ok=answers; break; }; sleep 5; done
  echo "[$b] ${WHAT[$b]}: TensorBoard $out -> http://localhost:$T ($ok)"
done
