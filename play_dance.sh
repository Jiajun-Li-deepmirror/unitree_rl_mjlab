#!/usr/bin/env bash
# Usage (from any directory): play_dance.sh <input-video-stem> [extra play.py args...]
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
if [ $# -lt 1 ]; then
  echo "Usage: $0 <input-video-stem> [extra play.py args...]" >&2
  exit 2
fi
DANCE=$1; shift
case "$DANCE" in
  */*|.|..) echo "Expected a video name without path or extension" >&2; exit 2 ;;
esac
MOTION_FILE="$PROJECT_ROOT/output/$DANCE/$DANCE.npz"
if [ ! -f "$MOTION_FILE" ]; then
  echo "Accepted motion not found: $MOTION_FILE" >&2
  exit 1
fi
# play.py selects the latest checkpoint under output/<video>/training/.
exec "${PYTHON:-python3}" "$PROJECT_ROOT/src/unitree_rl_mjlab/scripts/play.py" \
  Unitree-G1-Tracking-No-State-Estimation \
  --motion-file "$MOTION_FILE" --num-envs 1 --video --video-length 500 "$@"
