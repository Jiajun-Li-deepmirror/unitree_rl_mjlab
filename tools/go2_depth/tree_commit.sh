#!/bin/bash
# tree_commit.sh BOX COMMIT (go2-depth 2026-11-16, go2-home's queued runs): a tree of COMMIT's code on the box, for a
# training that must start on its own commit while the main tree moves on (the gates read the newest tools):
# /root/gpufree-data/go2-depth/trees/<COMMIT>/ = a copy of the main tree's code (src scripts jobs tools env.sh) brought
# to COMMIT by deploy.sh (DEPLOY_TREE; md5 checked), its logs/ a link to the main tree's logs (the runs, checks and
# records stay where every tool looks). Prints the tree's path. A tree already at COMMIT is kept.
set -u
B=$1 C=$(cd "$(dirname "$0")/../.." && git rev-parse --short "$2") || exit 2
declare -A PORT=([t1]=30714 [t2]=31365)
P=${PORT[$B]:-}; [ -n "$P" ] || { echo "tree_commit: no box $B" >&2; exit 2; }
SSH=/home/dm/.cache/go2-depth-work/ssh/r M=/root/gpufree-data/go2-depth/depth_fe T=/root/gpufree-data/go2-depth/trees/$C
have=$($SSH $P "cat $T/REPO_COMMIT 2>/dev/null" < /dev/null | tr -d '[:space:]')
if [ "$have" != "$C" ]; then
  # (env.sh sets PYTHONPATH to the main tree: the copy's points at the copy, or its trainings would import the main
  # tree's code; src/assets is a relative link in the main tree: the copy gets the files as hard links -- no space,
  # and deploy.sh's atomic replacements give a changed file its own inode in the copy alone)
  $SSH $P "set -e; rm -rf $T.part; mkdir -p $T.part; cd $M; cp -a src scripts jobs env.sh REPO_COMMIT $T.part/; [ -d tools ] && cp -a tools $T.part/ || true; ln -s $M/logs $T.part/logs; if [ -L $T.part/src/assets ]; then A=\$(readlink -f $M/src/assets); rm $T.part/src/assets; cp -al \$A $T.part/src/assets; fi; sed -i 's#^export PYTHONPATH=$M\$#export PYTHONPATH=$T#' $T.part/env.sh; grep -q '^export PYTHONPATH=$T\$' $T.part/env.sh; rm -rf $T; mv $T.part $T" < /dev/null || exit 2
  DEPLOY_TREE=$T bash "$(dirname "$0")/deploy.sh" --boxes $B --force $C | tail -n 1
fi
r=$($SSH $P "cat $T/REPO_COMMIT" < /dev/null | tr -d '[:space:]')
[ "$r" = "$C" ] || { echo "tree_commit: $T is at '$r', not $C" >&2; exit 2; }
echo "$T"
