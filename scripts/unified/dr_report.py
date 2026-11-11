#!/usr/bin/env python3
"""dr_report.py TASK: the randomization a check task's play config applies (go2-depth 2026-11-12, doc/redesign.md 2.1:
every report states it), one line per event: name, mode, function, its ranges; plus the observation noise. Run with
UNIFIED_TASKS=1 (and U_PAYLOAD as the check runs)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("UNIFIED_TASKS", "1")

from mjlab.tasks.registry import load_env_cfg  # noqa: E402

import src.tasks  # noqa: E402,F401

SKIP = {"asset_cfg", "sensor_name", "stair_terrain_names", "pose_range", "velocity_range"}


def main(task: str) -> int:
  cfg = load_env_cfg(task, play=True)
  for name, e in cfg.events.items():
    if name in ("randomize_terrain", "reset_base"):
      continue
    params = {k: v for k, v in e.params.items() if k not in SKIP and not callable(v)}
    fn = getattr(e.func, "__name__", type(e.func).__name__)
    print(f"{name} ({e.mode}, {fn}): {params}")
  noisy = [n for n, t in cfg.observations["actor"].terms.items() if t.noise is not None] if "actor" in cfg.observations else []
  print(f"observation noise on: {', '.join(noisy) or '-'}")
  print(f"payload (U_PAYLOAD): {os.environ.get('U_PAYLOAD', '0,0,0,0')}")
  return 0


if __name__ == "__main__":
  sys.exit(main(sys.argv[1]))
