#!/bin/bash
# deploy.sh [--check] [--force] [--boxes t1,t2] [COMMIT] (go2-depth 2026-11-11, the user's requirement 6: "机器上的代码只能通过一个
# 部署脚本更新，带 md5 校验；每次运行记录 commit"). Brings the boxes' tree (/root/gpufree-data/go2-depth/depth_fe) from
# its REPO_COMMIT to COMMIT (default HEAD; it must be pushed or at least committed here):
#   1. the files git changed between the two (added or modified; a file git deleted is reported, not removed);
#   2. before: each file's md5 on the box must be its md5 at the box's REPO_COMMIT (or absent for a new file) --
#      a box edit that is in no commit stops the deploy (--force deploys over it, listing it);
#   3. each file goes in atomically (a new inode: running scripts keep the old text), mode kept;
#   4. after: every file's md5 must be COMMIT's; then REPO_COMMIT = COMMIT.
# --check: only steps 1-2 and the drift report. go2-home deploys by itself (no route there): tell it the commit.
set -u
CHECK=0 FORCE=0 BOXES=t1,t2
while [ $# -gt 0 ]; do case $1 in --check) CHECK=1 ;; --force) FORCE=1 ;; --boxes) BOXES=$2; shift ;; *) break ;; esac; shift; done
cd "$(dirname "$0")/../.." || exit 2
C=$(git rev-parse --short "${1:-HEAD}") || exit 2
SSH=/home/dm/.cache/go2-depth-work/ssh/r D=/root/gpufree-data/go2-depth/depth_fe
declare -A PORT=([t1]=30714 [t2]=31365)
rc=0
for b in ${BOXES//,/ }; do
  P=${PORT[$b]:-}; [ -n "$P" ] || { echo "deploy: no box $b" >&2; rc=2; continue; }
  OLD=$($SSH $P "cat $D/REPO_COMMIT" < /dev/null | tr -d '[:space:]')
  git cat-file -e "$OLD^{commit}" 2> /dev/null || { echo "[$b] REPO_COMMIT '$OLD' is no commit here: deploy by hand"; rc=2; continue; }
  if [ "$(git rev-parse $OLD)" = "$(git rev-parse $C)" ]; then echo "[$b] already at $C"; continue; fi
  mapfile -t F < <(git diff --name-only --diff-filter=AMR $OLD $C)
  mapfile -t DEL < <(git diff --name-only --diff-filter=D $OLD $C)
  echo "[$b] $OLD -> $C: ${#F[@]} file(s)${DEL:+, ${#DEL[@]} deleted in git (left on the box: ${DEL[*]})}"
  [ ${#F[@]} -gt 0 ] || { [ $CHECK = 1 ] || $SSH $P "echo $C > $D/REPO_COMMIT" < /dev/null; continue; }
  # md5 on the box now vs at OLD
  want=$(for f in "${F[@]}"; do if git cat-file -e "$OLD:$f" 2> /dev/null; then echo "$(git show "$OLD:$f" | md5sum | cut -c1-32) $f"; else echo "absent $f"; fi; done)
  have=$($SSH $P "cd $D && for f in ${F[*]}; do if [ -f \$f ]; then echo \"\$(md5sum < \$f | cut -c1-32) \$f\"; else echo \"absent \$f\"; fi; done" < /dev/null)
  drift=$(diff <(echo "$want" | sort -k2) <(echo "$have" | sort -k2) | grep '^>' | cut -c3-)
  if [ -n "$drift" ]; then
    echo "[$b] on the box and in no commit (md5 differs from $OLD):"; echo "$drift" | sed 's/^/    /'
    if [ $FORCE != 1 ]; then echo "[$b] not deployed (--force deploys over them)"; rc=1; continue; fi
  fi
  [ $CHECK = 1 ] && { echo "[$b] check only"; continue; }
  git archive $C "${F[@]}" | $SSH $P "cd $D && T=\$(mktemp -d $D/../dep.XXXX) && tar -x -C \$T && for f in ${F[*]}; do mkdir -p \$(dirname \$f); [ -f \$f ] && chmod --reference=\$f \$T/\$f; mv -f \$T/\$f \$f.new.\$\$ && mv -f \$f.new.\$\$ \$f; done; rm -rf \$T"
  want=$(for f in "${F[@]}"; do echo "$(git show "$C:$f" | md5sum | cut -c1-32) $f"; done)
  have=$($SSH $P "cd $D && for f in ${F[*]}; do echo \"\$(md5sum < \$f | cut -c1-32) \$f\"; done" < /dev/null)
  if [ "$(echo "$want" | sort)" = "$(echo "$have" | sort)" ]; then
    $SSH $P "cd $D && echo $C > REPO_COMMIT.tmp && mv REPO_COMMIT.tmp REPO_COMMIT" < /dev/null
    echo "[$b] deployed $C, md5 checked (${#F[@]} file(s))"
  else
    echo "[$b] md5 after the copy differs -- REPO_COMMIT left at $OLD:"; diff <(echo "$want" | sort) <(echo "$have" | sort) | sed 's/^/    /'; rc=1
  fi
done
exit $rc
