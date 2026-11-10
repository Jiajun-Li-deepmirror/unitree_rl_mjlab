#!/bin/bash
# b2w_after.sh PID (go2-depth 2026-11-02): once process PID (the As2W fe_edge_v4fric chain) has exited, the B2W
# option-B chain (pilot …,sidecmd,b2wdesc; paused on t1 at 09:52 so B2 could go first) from the widened polish 9000,
# teacher speed margin 0.07.
cd /root/gpufree-data/go2-depth/depth_fe
# alive and not a zombie (11-02 13:00: a killed chain stayed <defunct> and kill -0 kept succeeding for 30 min)
while kill -0 $1 2> /dev/null && [ "$(ps -o stat= -p $1 | cut -c1)" != Z ]; do sleep 60; done
echo "$(date '+%m-%d %H:%M:%S') B2W: process $1 (As2W chain) has ended; starting the B2W chain on t2" >> logs/jobs/fe_b2w.txt
exec env FE_MARGIN=0.07 B2W_PKG=$PWD/logs/pkg/b2w_polish_9000_sidecmd_desc bash jobs/fe_chain.sh B2W
