#!/bin/bash
# am_nopay.sh TRAIN_PID CKPT OUT (11-06): the flat speed matrix of one checkpoint without payload (BLIND_PAYLOAD max=0),
# in the running training's environment otherwise (pilot incl. grip), under the check lock. go2-home: bin overspeed by payload.
cd /root/gpufree-data/go2-depth/depth_fe
while IFS= read -r -d '' kv; do export "$kv"; done < /proc/$1/environ
export BLIND_PAYLOAD=max=0,x=0.15,y=0.08,z=0.20
PY=/root/gpufree-data/envs/g1-dance/bin/python
exec 9>> ${FE_LOCK:-logs/checks/check.lock}
flock 9
$PY jobs/seed_run.py 0 scripts/blind/eval/accept_matrix.py --task Unitree-Go2W-Blind-Flat-Scratch --envs-per-cmd 32 $2 > $3 2>&1
echo "done $(date '+%T')" >> $3
