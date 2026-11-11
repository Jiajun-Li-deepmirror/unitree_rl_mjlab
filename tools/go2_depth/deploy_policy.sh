#!/bin/bash
# deploy_policy.sh R KIND PKG NAME [--apply] [--sim-only | --jetson-only] (go2-depth 2026-11-11, the user's unification
# item 3: deploying goes from one package format with one script; the per-robot deploy scripts of earlier rounds are
# replaced). Installs package PKG (jobs/export_policy.sh's layout: policy.onnx, deploy.yaml, blind_pilot.txt,
# manifest.json; a local directory, pulled from a box beforehand) of robot R as KIND (blind | depth) under NAME:
#   tinynav-sim (each tree of TN_TREES, default /home/dm/tinynav-sim/mujoco and the gate's tnsim-wheeled/mujoco):
#     assets/policy/<r>_<KIND>_<NAME>.onnx   blind: scripts/import_blind_policy.py (deploy.yaml's layout, brake,
#                                           gait clock, solver from the pilot); depth: the export as it is
#     assets/policy/policy.sha256           its line
#     sim/robots.py                         the robot's perception= (depth) or blind= (blind)
#   Jetson staging (JETSON_STAGE, default ~/.cache/go2-depth-work/jetson_ctrl/robot_ctrl):
#     deploy/robots/<r>/config/policy/velocity/<KIND>_<NAME>/{exported/policy.onnx, params/deploy.yaml, SOURCE.txt}
#     deploy/robots/<r>/config/config.yaml  policy_dir (depth) or blind_dir (blind; policy_dir when there is none),
#                                           the old one kept in config.yaml.before_<KIND>_<NAME>
# Without --apply it only prints what it would do. It never commits (tinynav-sim) nor syncs (the Jetson: sync.sh after
# an idle check) -- those stay the user's calls -- and it leaves the README tables to be written by hand.
set -u
R=$1 KIND=$2 PKG=$3 NAME=$4; shift 4
APPLY=0 SIM=1 JET=1
for a in "$@"; do case $a in --apply) APPLY=1 ;; --sim-only) JET=0 ;; --jetson-only) SIM=0 ;; *) echo "deploy_policy.sh: $a?" >&2; exit 2 ;; esac; done
r=${R,,}
case $KIND in blind|depth) ;; *) echo "deploy_policy.sh: KIND blind or depth" >&2; exit 2 ;; esac
for f in policy.onnx deploy.yaml manifest.json; do [ -s "$PKG/$f" ] || { echo "deploy_policy.sh: $PKG/$f missing" >&2; exit 2; }; done
PY=${PY:-/home/dm/miniconda3/envs/unitree_rl_mjlab/bin/python}
FILE=${r}_${KIND}_${NAME}.onnx DIR=${KIND}_${NAME}
TN_TREES=${TN_TREES:-/home/dm/tinynav-sim/mujoco /home/dm/.cache/go2-depth-work/tnsim-wheeled/mujoco}
JS=${JETSON_STAGE:-/home/dm/.cache/go2-depth-work/jetson_ctrl/robot_ctrl}
do_() { if [ $APPLY = 1 ]; then "$@"; else echo "  would: $*"; fi; }
echo "deploy $R $KIND from $PKG as $FILE / $DIR ($([ $APPLY = 1 ] && echo applying || echo dry run))"
python3 -c "import json,sys; m=json.load(open(sys.argv[1])); print('  package:', m['kind'], m['robot'], m['checkpoint']['path'], 'iter', m['checkpoint']['iteration'], 'onnx', m['policy_onnx_md5'][:8], 'code', m['code'])" "$PKG/manifest.json"
if [ $SIM = 1 ]; then
  for T in $TN_TREES; do
    [ -d "$T/assets/policy" ] || { echo "  [sim] no tree $T, skipped"; continue; }
    out=$T/assets/policy/$FILE
    echo "  [sim] $T"
    if [ $KIND = blind ]; then do_ bash -c "cd $T && $PY scripts/import_blind_policy.py $PKG $out"
    else do_ cp "$PKG/policy.onnx" "$out"; fi
    if [ $APPLY = 1 ]; then
      [ -s "$out" ] || { echo "  [sim] $out not written" >&2; exit 1; }
      sum=$(sha256sum "$out" | cut -d' ' -f1)
      { grep -v "  $FILE\$" $T/assets/policy/policy.sha256; echo "$sum  $FILE"; } > $T/assets/policy/policy.sha256.new && mv $T/assets/policy/policy.sha256.new $T/assets/policy/policy.sha256
      python3 - "$T/sim/robots.py" "$r" "$([ $KIND = depth ] && echo perception || echo blind)" "$FILE" <<'EOF'
import re, sys
path, robot, field, name = sys.argv[1:5]
s = open(path).read()
m = re.search(r'\n    "%s": Robot\(' % re.escape(robot), s)
assert m, f"no entry for {robot} in {path}"
end = s.index("\n    \"", m.end()) if "\n    \"" in s[m.end():] else len(s)
block = s[m.start():end]
new, n = re.subn(r'\b%s="[^"]*"' % field, f'{field}="{name}"', block, count=1)
assert n == 1, f"no {field}= in {robot}'s entry"
open(path, "w").write(s[:m.start()] + new + s[end:])
print(f"  [sim] {path}: {robot} {field}={name}")
EOF
    else
      echo "  would: policy.sha256 += $FILE; sim/robots.py $r $([ $KIND = depth ] && echo perception || echo blind)=\"$FILE\""
    fi
  done
fi
if [ $JET = 1 ]; then
  RD=$JS/deploy/robots/$r/config
  [ -f $RD/config.yaml ] || { echo "  [jetson] no $RD/config.yaml" >&2; exit 2; }
  key=$([ $KIND = depth ] && echo policy_dir || { grep -q '^\s*blind_dir:' $RD/config.yaml && echo blind_dir || echo policy_dir; })
  echo "  [jetson] $RD/policy/velocity/$DIR, $key -> config/policy/velocity/$DIR (now: $(grep -E "^\s*$key:" $RD/config.yaml | head -n 1 | sed 's/^ *//'))"
  do_ mkdir -p $RD/policy/velocity/$DIR/exported $RD/policy/velocity/$DIR/params
  do_ cp "$PKG/policy.onnx" $RD/policy/velocity/$DIR/exported/policy.onnx
  do_ cp "$PKG/deploy.yaml" $RD/policy/velocity/$DIR/params/deploy.yaml
  if [ $APPLY = 1 ]; then
    { echo "$R $KIND $DIR: from $PKG ($(date '+%Y-%m-%d %H:%M'), tools/go2_depth/deploy_policy.sh)"; cat "$PKG/manifest.json"; } > $RD/policy/velocity/$DIR/SOURCE.txt
    cp $RD/config.yaml $RD/config.yaml.before_$DIR
    sed -i -E "s|^(\s*$key:\s*).*|\1config/policy/velocity/$DIR|" $RD/config.yaml
    echo "  [jetson] $key now: $(grep -E "^\s*$key:" $RD/config.yaml | head -n 1 | sed 's/^ *//') (backup config.yaml.before_$DIR); not synced"
  fi
fi
if [ $APPLY = 1 ]; then echo "done: review, run tinynav-sim's tests, commit there, write the README rows, sync the Jetson after an idle check"; fi
