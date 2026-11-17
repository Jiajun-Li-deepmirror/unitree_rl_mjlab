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
task = sys.argv[2] if len(sys.argv) > 2 else None
import mjlab.tasks, src.tasks
from mjlab.tasks.registry import load_env_cfg
cfg = load_env_cfg(task or ("Unitree-Go2W-U-Teacher-Flat" if play else "Unitree-Go2W-U-Teacher"), play=play)
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
out["gate"] = {k: {"max_riser": cfg.rewards[k].params.get("max_riser"), "min_gain": cfg.rewards[k].params.get("min_gain"),
                   "cmd": cfg.rewards[k].params.get("min_lin_cmd"), "ahead": cfg.rewards[k].params.get("riser_ahead")}
               for k in ("wheel_riser_lift", "foothold_height_gain") if k in cfg.rewards}
out["topflat"] = cfg.rewards["wheel_lift_flat_ahead"].weight if "wheel_lift_flat_ahead" in cfg.rewards else None
gen = cfg.scene.terrain.terrain_generator if cfg.scene.terrain is not None else None
out["last_riser"] = {n: getattr(s, "last_riser_scale", None) for n, s in (gen.sub_terrains.items() if gen else ()) if n.startswith("walled")}
pp = cfg.rewards["pose"].params if "pose" in cfg.rewards else {}
out["pose"] = {"radius": pp.get("uneven_radius"), "feet": None if pp.get("uneven_feet_cfg") is None else list(pp["uneven_feet_cfg"].site_names)}
wd = cfg.rewards.get("wheel_air_drivable")
out["wad_near"] = None if wd is None else wd.params.get("near_scale")
cons = cfg.curriculum.get("u_constraints")
out["caps"] = {i["name"]: i["max_scale"] for i in cons.params["items"]} if cons is not None else {}
bp = cfg.events["base_payload"].params
out["payload"] = {"heavy_share": bp["heavy_share"], "on": bp.get("heavy_share_on"), "names": list(bp.get("heavy_terrain_names", ())),
                  "groups": [[sorted(g), s] for g, s in bp.get("heavy_share_groups", ())]}
print("JSON" + json.dumps(out))
'''

ok = True


def check(good: bool, text: str) -> None:
  global ok
  ok &= bool(good)
  print(("  ok    " if good else "  FAIL  ") + text)


def probe(play=False, task=None, **env):
  e = {k: v for k, v in os.environ.items() if not k.startswith("U_")}
  e.update(UNIFIED_TASKS="1", U_ROBOTS="Go2W", U_CMD_WIDE="1.25", U_DR_WIDE="1.25", PYTHONPATH=os.getcwd(), **env)
  r = subprocess.run([sys.executable, "-c", PROBE, "1" if play else "0"] + ([task] if task else []), env=e, capture_output=True, text=True)
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
print("(4) U_RISER_GATE:")
check(b1["gate"].get("wheel_riser_lift", {}).get("max_riser") is None, f"B1: no gate ({b1['gate']})")
rg = probe(U_RISER_GATE="1")
check("error" not in rg and rg["gate"]["wheel_riser_lift"]["max_riser"] == 0.30
      and rg["gate"]["foothold_height_gain"]["min_gain"] == 0.05 and rg["gate"]["wheel_riser_lift"]["cmd"] == 0.1
      and rg["gate"]["wheel_riser_lift"]["ahead"] is True and rg["gate"]["foothold_height_gain"]["cmd"] == 0.1,
      f"the training build: riser <= 0.30 m and ahead, record gain >= 0.05 m, both with a planar command >= 0.1 ({rg.get('gate') or rg.get('error')})")
check(same_but({**b1, "rewards": {k: v for k, v in b1["rewards"].items() if k not in ("wheel_riser_lift", "foothold_height_gain")}},
               {**rg, "rewards": {k: v for k, v in rg["rewards"].items() if k not in ("wheel_riser_lift", "foothold_height_gain")}}),
      "the rest of the build as B1's (the two terms' weights unchanged too)")
check(all(b1["rewards"][k][0] == rg["rewards"][k][0] for k in ("wheel_riser_lift", "foothold_height_gain")), "their weights unchanged")
rp = probe(play=True, task="Unitree-Go2W-U-Teacher-Stairs", U_RISER_GATE="1")
check("error" not in rp and rp["gate"].get("wheel_riser_lift", {}).get("max_riser") == 0.30
      and rp["gate"].get("foothold_height_gain", {}).get("min_gain") == 0.05, f"the -Stairs check build too ({rp.get('gate') or rp.get('error')})")
x = probe(U_RISER_GATE="2")
check("error" in x and "U_RISER_GATE" in x["error"], "U_RISER_GATE=2 stops the build")
print("(5) U_KEYS_ADD=topflat / topflat30 on the 68-term recipe (go2-home's L1 / L2):")
check(b1["topflat"] is None, "B1: no wheel_lift_flat_ahead")
k1 = probe(U_KEYS_ADD="topflat")
check("error" not in k1 and k1["topflat"] == -8.0
      and same_but(b1, {**k1, "rewards": {k: v for k, v in k1["rewards"].items() if k != "wheel_lift_flat_ahead"}}, curriculum=("u_quality",)),
      f"topflat: wheel_lift_flat_ahead at {k1.get('topflat')} (-8), the rest as B1's but u_quality ({k1.get('error', '')[:120]})")
check("wheel_lift_flat_ahead" in k1["curriculum"]["u_quality"] and "wheel_lift_flat_ahead" not in b1["curriculum"]["u_quality"],
      "it comes in with the stairs quality (u_quality: the M keys have not got it, so it rises from ~0 to its P weight along q)")
k2 = probe(U_KEYS_ADD="topflat,topflat30")
check("error" not in k2 and k2["topflat"] == -30.0, f"topflat30: at {k2.get('topflat')} (-30) ({k2.get('error', '')[:120]})")
print("(6) U_POSE_UNEVEN (go2-home's L5):")
check(b1["pose"]["feet"] is None, f"B1: pose's uneven mask by the circle ({b1['pose']})")
pf = probe(U_POSE_UNEVEN="feet")
check("error" not in pf and pf["pose"]["feet"] == ["FR", "FL", "RR", "RL"]
      and same_but({**b1, "rewards": {k: v for k, v in b1["rewards"].items() if k != "pose"}},
                   {**pf, "rewards": {k: v for k, v in pf["rewards"].items() if k != "pose"}})
      and pf["rewards"]["pose"][0] == b1["rewards"]["pose"][0], f"feet: the mask from the four feet, the rest as B1's ({pf.get('pose') or pf.get('error')})")
pr = probe(U_POSE_UNEVEN="r:0.25")
check("error" not in pr and pr["pose"]["radius"] == 0.25 and pr["pose"]["feet"] is None, f"r:0.25: the circle at 0.25 m ({pr.get('pose')})")
pc = probe(play=True, task="Unitree-Go2W-U-Teacher-Stairs", U_POSE_UNEVEN="feet")
check("error" not in pc and pc["pose"]["feet"] == ["FR", "FL", "RR", "RL"], "the -Stairs check build too")
for bad in ("r:0.6", "edge"):
  x = probe(U_POSE_UNEVEN=bad)
  check("error" in x and "U_POSE_UNEVEN" in x["error"], f"U_POSE_UNEVEN={bad} stops the build")
print("(7) U_LAST_RISER (go2-home's L6):")
check(b1["last_riser"] and all(v is None for v in b1["last_riser"].values()), f"B1: every riser as drawn ({b1['last_riser']})")
lr = probe(U_LAST_RISER="0.4,1.0")
check("error" not in lr and lr["last_riser"] == {"walled_down": [0.4, 1.0], "walled_up": [0.4, 1.0]} and same_but(b1, lr),
      f"the training build: walled_up / walled_down's last riser 0.4-1.0 x the mean, the rest as B1's ({lr.get('last_riser') or lr.get('error')})")
lp = probe(play=True, task="Unitree-Go2W-U-Teacher", U_LAST_RISER="0.4,1.0")
check("error" not in lp and all(v is None for v in lp["last_riser"].values()), f"the play build unchanged ({lp.get('last_riser')})")
for bad in ("0.1,1.0", "1.0,0.5"):
  x = probe(U_LAST_RISER=bad)
  check("error" in x and "U_LAST_RISER" in x["error"], f"U_LAST_RISER={bad} stops the build")
print("(8) U_PAYLOAD_FULL0=1 alone (go2-home's P1):")
pf0 = probe(U_PAYLOAD_FULL0="1")
check("error" not in pf0 and '"fixed", True' in pf0["curriculum"]["u_payload"].replace("'", '"') and "fixed" not in b1["curriculum"]["u_payload"]
      and same_but(b1, pf0, curriculum=("u_payload",)), f"the payload's full limit from the start (u_payload fixed), the rest as B1's ({pf0.get('error', '')[:100]})")
print("(9) U_HEAVY_STAIRS (go2-home's P2):")
check(b1["payload"]["groups"] == [], "B1: no tile groups")
hs = probe(U_HEAVY_STAIRS="0.8")
want = sorted(("stairs_down_narrow", "stairs_down_wide", "stairs_up_narrow", "stairs_up_wide", "walled_down", "walled_up"))
check("error" not in hs and hs["payload"]["groups"] == [[want, 0.8]] and same_but(b1, hs, events=("base_payload",)),
      f"0.8 on the six stairs tiles, the rest as B1's ({hs.get('payload') or hs.get('error')})")
hb = probe(U_HEAVY_STAIRS="0.8", U_HEAVY_SLOPE="0.8")
check("error" not in hb and hb["payload"]["on"] == 0.8 and hb["payload"]["groups"] == [[want, 0.8]], "beside U_HEAVY_SLOPE: both")
x = probe(U_HEAVY_STAIRS="0.4")
check("error" in x and "U_HEAVY_STAIRS" in x["error"], "U_HEAVY_STAIRS=0.4 stops the build")
print("(10) U_WAD_NEAR (go2-home's W3):")
check(b1["wad_near"] == 4.0, f"B1: near_scale {b1['wad_near']}")
wn = probe(U_WAD_NEAR="0")
check("error" not in wn and wn["wad_near"] == 0.0 and same_but(b1, {**wn, "rewards": {**wn["rewards"], "wheel_air_drivable": b1["rewards"]["wheel_air_drivable"]}}),
      f"U_WAD_NEAR=0: near_scale 0, the rest as B1's ({wn.get('wad_near', wn.get('error', '')[:100])})")
x = probe(U_WAD_NEAR="5")
check("error" in x and "U_WAD_NEAR" in x["error"], "U_WAD_NEAR=5 stops the build")
print("(11) U_ITEM_CAP=wheels_down:1.34 (go2-home's W1):")
check(b1["caps"].get("wheels_down") == 3.0, f"B1: wheels_down cap {b1['caps'].get('wheels_down')}")
wc = probe(U_ITEM_CAP="wheels_down:1.34")
check("error" not in wc and wc["caps"]["wheels_down"] == 1.34 and {k: v for k, v in wc["caps"].items() if k != "wheels_down"}
      == {k: v for k, v in b1["caps"].items() if k != "wheels_down"},
      f"wheels_down's cap 1.34, every other item's as B1's ({wc.get('caps', {}).get('wheels_down', wc.get('error', '')[:100])})")
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
