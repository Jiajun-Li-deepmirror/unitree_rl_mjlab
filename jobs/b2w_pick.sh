#!/bin/bash
# b2w_pick.sh A WHY (go2-depth 2026-11-02, early stage advance): A passed every teacher check (WHY says where), so the
# B2W teacher stage ends now: stop the chain, its checks and its teacher training, pick A, and start the chain again,
# which goes straight to the student (logs/picks/b2w_teacher set). Same chain settings as b2w_after.sh.
cd /root/gpufree-data/go2-depth/depth_fe
A=$1 WHY=$2 L=logs/jobs/fe_b2w.txt
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $L; }
[ -f "$A" ] || { echo "no $A"; exit 1; }
[ -f logs/picks/b2w_teacher ] && { echo "already picked: $(cat logs/picks/b2w_teacher)"; exit 1; }
chain=$(ps -eo pid,args | awk '$2 == "bash" && $3 == "jobs/fe_chain.sh" && $4 == "B2W" {print $1}')
train=$(ps -eo pid,args | awk '$0 ~ /scripts\/train[.]py Unitree-B2W-Depth-Polish/ && $0 !~ /awk/ {print $1}')
[ -n "$chain" ] && kill $chain
checks=$(ps -eo pid,args | awk '($0 ~ /jobs\/fe_check[.]sh teacher B2W/ || $0 ~ /fe_b2w_teacher_avg/) && $0 !~ /awk/ {print $1}')
[ -n "$checks" ] && kill $checks
for p in $(ps -eo pid,args | awk '$0 ~ /--task Unitree-B2W-/ && $0 ~ /fe_v1_avg/ && $0 !~ /awk/ {print $1}'); do kill $p; done
for p in $train; do kill -INT $p; for _ in $(seq 60); do kill -0 $p 2> /dev/null || break; sleep 2; done; kill $p 2> /dev/null; done
tmux kill-session -t fe_b2w 2> /dev/null
sleep 5
left=$(ps -eo args | grep -cE "Unitree-B2W-|fe_chain.sh B2W" | awk '{print $1 - 1}')
echo "$A" > logs/picks/b2w_teacher
log "B2W TEACHER PICK early $A (md5 $(md5sum < $A | cut -c1-8); $WHY); chain $chain, teacher $train, checks stopped ($left left)"
tmux new-session -d -s fe_b2w "env FE_MARGIN=0.07 B2W_PKG=$PWD/logs/pkg/b2w_polish_9000_sidecmd_desc FE_STUDENT_RUN=fe_edge_v4desc bash jobs/fe_chain.sh B2W"
