#!/bin/bash
# deliver.sh R KIND CKPT Q PKG (2026-11-12, split out of jobs/unified/accept.sh so the depth student's delivery can
# wait for the camera-failure pair test, doc/redesign.md 2.2.1): the accepted student's package -- the tinynav gate's
# export (logs/probe/<basename Q>: policy.onnx, deploy.yaml, ...; else exported here, scripts/unified/export_student.py)
# and the checkpoint -- in PKG, then the two acceptance videos (jobs/accept_videos.sh). Log $LOG (default
# logs/jobs/unified_<r>_accept.txt). jobs/unified/students.sh calls it with the line accept.sh left in <pick>.pending.
set -u
R=$1 KIND=$2 CK=$3 Q=$4 PKG=$5
r=${R,,}
cd ${UNIFIED_ROOT:-/root/gpufree-data/go2-depth/depth_fe} || exit 2
source env.sh > /dev/null 2>&1
export UNIFIED_TASKS=1
L=${LOG:-logs/jobs/unified_${r}_accept.txt}
log() { echo "$(date '+%m-%d %H:%M:%S') $R $KIND: $*" >> $L; }
mkdir -p $PKG
cp logs/probe/$(basename $Q)/* $PKG/ 2> /dev/null
cp $CK $PKG/
if [ -s $PKG/policy.onnx ]; then
  log "UNIFIED DELIVERY $PKG (policy.onnx $(md5sum < $PKG/policy.onnx | cut -c1-8))"
else
  $PY scripts/unified/export_student.py $R $KIND $CK $PKG >> $L 2>&1 && log "UNIFIED DELIVERY $PKG (exported here)" \
    || { log "TOOL FAULT: export for $PKG failed"; exit 4; }
fi
# (the videos run in this tree: accept_videos.sh's own default root is depth_fe, a tree elsewhere -- UNIFIED_ROOT --
# would read its relative checkpoint there and write the videos into depth_fe's logs/pkg)
ACCEPT_ROOT=$PWD LOG=$L bash jobs/accept_videos.sh u$KIND $R $CK $PKG/videos
