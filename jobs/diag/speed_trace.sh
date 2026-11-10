#!/bin/bash
# speed_payload.sh TRAIN_PID CKPT OUT: speed_payload.py in the running training's environment, under the check lock.
cd /root/gpufree-data/go2-depth/depth_fe
while IFS= read -r -d '' kv; do export "$kv"; done < /proc/$1/environ
PY=/root/gpufree-data/envs/g1-dance/bin/python
exec 9>> ${FE_LOCK:-logs/checks/check.lock}
flock 9
$PY jobs/diag/speed_trace.py --task Unitree-Go2W-Blind-Flat-Scratch $2 > $3 2>&1
echo "done $(date '+%T')" >> $3
