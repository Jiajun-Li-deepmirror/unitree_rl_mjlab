#!/bin/bash
# tf5_stop_student.sh (go2-depth 2026-11-05 06:40, t1): ends the tf5 chain (bash pid 96109, tmux tfg5) once its teacher
# stage is over, before its distilled student (slow climbs, see tf4p_go2w.sh) takes the GPU beside fe_edge_tf4p.
cd /root/gpufree-data/go2-depth/depth_fe
L=logs/jobs/fe_go2w.txt
done_line() { grep -a "Go2W TEACHER PICK\|Go2W NO TEACHER PASSED" $L | awk '$1" "$2 >= "11-05 06:00:00"' | tail -n 1; }
until [ -n "$(done_line)" ] || ! kill -0 96109 2> /dev/null; do sleep 20; done
echo "$(date '+%m-%d %H:%M:%S') $(done_line | cut -c1-200)"
kill 96109 2> /dev/null && echo "$(date '+%m-%d %H:%M:%S') killed chain 96109"
sleep 5
SP=$(grep -a "Go2W student started (pid" $L | awk '$1" "$2 >= "11-05 06:00:00"' | grep -a fe_edge_tf5 | grep -o "pid [0-9]*" | cut -d" " -f2)
[ -n "$SP" ] && kill $SP && echo "$(date '+%m-%d %H:%M:%S') killed student $SP"
echo "$(date '+%m-%d %H:%M:%S') Go2W tf5 chain ended after its teacher stage (tf5_stop_student.sh)" | tee -a $L
