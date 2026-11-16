#!/usr/bin/env python3
"""scenario_switch_test.py -- go2-home's round-1 scenario switches on the Go2W's B1 build (go2-depth 2026-11-17).
No GPU. Each build in its own process (the unified tasks resolve the robot at registration).

  (1) U_ENDWALL_GUARD=1: the key endwall in the training build (added when the recipe lacks it; once beside
      U_KEYS_ADD=endwall), the forward command's wall stop at the camera + 0.45 m (0.757 m base-to-wall) capped at
      1.5 x the distance left (robots._endwall_guard); the play build has no wall stop; a bad value stops the build;
      the command's cap itself (TerrainAwareVelocityCommand._wall_limit) on numbers, and its ramp without the gain;
  (2) U_SLOPE_SHARE=<x>: each sloped tile x of the other ground's episodes, the rest scaled to keep the sum;
  (3) U_HEAVY_SLOPE=<share>: the base_payload event's heavy share on the sloped tiles; a share not above the
      event's own stops the build;
  every switch leaves the rest of the build as B1's (rewards, events, commands, curricula; the switch's own fields
  aside).

Local: PYTHONPATH=$PWD python scripts/unified/scenario_switch_test.py"""
from __future__ import annotations

import json
import os
import subprocess
import sys

PROBE = r'''
import dataclasses, json, os, sys
sys.path.insert(0, os.getcwd())
play = sys.argv[1] == "1"
import mjlab.tasks, src.tasks
from mjlab.tasks.registry import load_env_cfg
cfg = load_env_cfg("Unitree-Go2W-U-Teacher-Flat" if play else "Unitree-Go2W-U-Teacher", play=play)
t = cfg.commands["twist"]
import re
strip = lambda s: re.sub(r" at 0x[0-9a-f]+", "", s)
rep = lambda x: strip(repr(sorted((x or {}).items(), key=repr)))
out = {
  "wall": {"terrains": list(t.end_wall_terrain_names), "sensor": t.wall_sensor_name, "stop": t.end_wall_stop_m,
           "slow": t.end_wall_slow_m, "gain": t.wall_guard_gain},
  "sensors": sorted(s.name for s in cfg.scene.sensors),
  "pilot": os.environ.get("BLIND_PILOT", ""),
  "rewards": {k: [v.weight, rep(v.params)] for k, v in cfg.rewards.items()},
  "events": {k: rep(v.params) for k, v in cfg.events.items()},
  "twist": {f.name: strip(repr(getattr(t, f.name))) for f in dataclasses.fields(t)},
  "curriculum": {k: rep(v.params) for k, v in cfg.curriculum.items()},
}
sp = cfg.curriculum.get("u_stairs_share")
out["shares"] = dict(sp.params.get("other_shares") or {}) if sp else None
bp = cfg.events["base_payload"].params
out["payload"] = {"heavy_share": bp["heavy_share"], "on": bp.get("heavy_share_on"), "names": list(bp.get("heavy_terrain_names", ()))}
print("JSON" + json.dumps(out))
'''

ok = True


def check(good: bool, text: str) -> None:
  global ok
  ok &= bool(good)
  print(("  ok    " if good else "  FAIL  ") + text)


def probe(play=False, **env):
  e = {k: v for k, v in os.environ.items() if not k.startswith("U_")}
  e.update(UNIFIED_TASKS="1", U_ROBOTS="Go2W", U_CMD_WIDE="1.25", U_DR_WIDE="1.25", PYTHONPATH=os.getcwd(), **env)
  r = subprocess.run([sys.executable, "-c", PROBE, "1" if play else "0"], env=e, capture_output=True, text=True)
  for line in r.stdout.splitlines():
    if line.startswith("JSON"):
      return json.loads(line[4:])
  return {"error": (r.stderr or r.stdout)[-500:]}


def same_but(a, b, *, events=(), twist=(), curriculum=()):
  """Whether build b equals build a outside the named events / twist fields / curricula (the differing parts printed)."""
  for part, skip in (("rewards", ()), ("events", events), ("twist", twist), ("curriculum", curriculum)):
    diff = sorted(k for k in set(a[part]) | set(b[part]) if k not in skip and a[part].get(k) != b[part].get(k))
    if diff:
      print(f"        differs in {part}: {diff[:8]}")
  return (a["rewards"] == b["rewards"]
          and {k: v for k, v in a["events"].items() if k not in events} == {k: v for k, v in b["events"].items() if k not in events}
          and {k: v for k, v in a["twist"].items() if k not in twist} == {k: v for k, v in b["twist"].items() if k not in twist}
          and {k: v for k, v in a["curriculum"].items() if k not in curriculum} == {k: v for k, v in b["curriculum"].items() if k not in curriculum})


b1 = probe()
if "error" in b1:
  print("B1 build failed:", b1["error"])
  sys.exit(1)
print("(1) U_ENDWALL_GUARD:")
check(b1["wall"]["gain"] is None, f"B1: no guard (wall stop {b1['wall']['stop']}, ramp {b1['wall']['slow']}, terrains {b1['wall']['terrains']})")
g = probe(U_ENDWALL_GUARD="1")
check("error" not in g and (g["wall"]["terrains"] or g["wall"]["sensor"]) and abs(g["wall"]["stop"] - 0.757) < 1e-9
      and g["wall"]["gain"] == 1.5, f"the guard's stop 0.757 m and gain 1.5 with the endwall key's wall stop ({g.get('wall') or g.get('error')})")
gk = probe(U_ENDWALL_GUARD="1", U_KEYS_ADD="endwall")
check("error" not in gk and gk["wall"] == g["wall"] and gk["rewards"] == g["rewards"], "beside U_KEYS_ADD=endwall: the same build (the key once)")
ek = probe(U_KEYS_ADD="endwall")
check("error" not in ek and same_but(ek, g, twist=("end_wall_stop_m", "wall_guard_gain")),
      f"against the key endwall alone: only the stop ({ek.get('wall', {}).get('stop')} -> 0.757) and the gain differ")
gp = probe(play=True, U_ENDWALL_GUARD="1")
check("error" not in gp and gp["wall"]["sensor"] is None and "wall_fan" not in gp["sensors"], "the play build: no wall fan, no wall stop")
x = probe(U_ENDWALL_GUARD="2")
check("error" in x and "U_ENDWALL_GUARD" in x["error"], "U_ENDWALL_GUARD=2 stops the build")
import torch  # noqa: E402
from types import SimpleNamespace  # noqa: E402
sys.path.insert(0, os.getcwd())
from src.tasks.velocity.mdp.velocity_command import TerrainAwareVelocityCommand as TC  # noqa: E402
raw, d = torch.tensor([1.0, 1.0, 1.0, 0.2]), torch.tensor([1.757, 1.0, 0.7, 1.757])
cap = TC._wall_limit(SimpleNamespace(cfg=SimpleNamespace(wall_guard_gain=1.5, end_wall_stop_m=0.757, end_wall_slow_m=0.5)), raw, d)
check(torch.allclose(cap, torch.tensor([1.0, 0.3645, 0.0, 0.2])), f"the cap: min(raw, 1.5 x (d - 0.757)) {cap.tolist()}")
ramp = TC._wall_limit(SimpleNamespace(cfg=SimpleNamespace(wall_guard_gain=None, end_wall_stop_m=0.9, end_wall_slow_m=0.5)),
                      torch.tensor([1.0, 1.0, 1.0]), torch.tensor([1.4, 1.15, 0.8]))
check(torch.allclose(ramp, torch.tensor([1.0, 0.5, 0.0])), f"without the gain: the ramp as before {ramp.tolist()}")
print("(2) U_SLOPE_SHARE:")
sh = b1["shares"] or {}
check(sh and abs(sh.get("slope_up", 0) - 0.05) < 1e-9 and abs(sh.get("slope_down", 0) - 0.05) < 1e-9 and abs(sum(sh.values()) - 1) < 1e-9,
      f"B1: 0.05 each, the shares sum to 1 ({sh})")
s = probe(U_SLOPE_SHARE="0.1")
ss = s.get("shares") or {}
check("error" not in s and abs(ss["slope_up"] - 0.1) < 1e-9 and abs(ss["slope_down"] - 0.1) < 1e-9 and abs(sum(ss.values()) - 1) < 1e-9
      and all(abs(ss[k] / sh[k] - 0.8 / 0.9) < 1e-9 for k in sh if not k.startswith("slope")),
      f"U_SLOPE_SHARE=0.1: 0.1 each, the rest x 0.8 / 0.9 ({ss})")
check(same_but(b1, s, curriculum=("u_stairs_share",)), "the rest of the build as B1's")
x = probe(U_SLOPE_SHARE="0.4")
check("error" in x and "U_SLOPE_SHARE" in x["error"], "U_SLOPE_SHARE=0.4 stops the build")
print("(3) U_HEAVY_SLOPE:")
check(b1["payload"]["on"] is None, f"B1: one heavy share ({b1['payload']['heavy_share']})")
h = probe(U_HEAVY_SLOPE="0.8")
check("error" not in h and h["payload"]["on"] == 0.8 and sorted(h["payload"]["names"]) == ["slope_down", "slope_up"]
      and h["payload"]["heavy_share"] == b1["payload"]["heavy_share"], f"U_HEAVY_SLOPE=0.8: 0.8 on the sloped tiles ({h.get('payload') or h.get('error')})")
check(same_but(b1, h, events=("base_payload",)), "the rest of the build as B1's")
x = probe(U_HEAVY_SLOPE="0.3")
check("error" in x and "U_HEAVY_SLOPE" in x["error"], "U_HEAVY_SLOPE=0.3 (not above the event's 0.5) stops the build")
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
