#!/usr/bin/env python3
"""topple_probe.py TASK CHECKPOINT [--num-envs N] [--steps S] [--window 2.0] [--spread-rows] (go2-depth 2026-11-13):
what a robot's last seconds before a fall on the stairs look like, per payload class, beside the same seconds of the
robots that did not fall.

Why: the As2 teacher (go2-home's mAMc) falls with the acceptance payload (15 kg at x +0.15, z +0.20: 85% of its own
weight) on 25 / 52 of 576 traverses / stops of the steep flights, every one a forward topple on a descent, the stops'
mostly within 2 s of the zero command -- and neither more training nor a lower commanded speed helps (it runs at
1.2-1.4 times the command there). The records say where and how often, not what gives way first. This probe says it:
the training task under its own draws (the checkpoint restored as a resume does), a ring buffer of `window` seconds
per robot of
  the commanded and the actual speed, the pitch (nose down > 0) and its rate, the feet on the ground (all / front),
  the largest torque of the front and of the rear legs as a share of the actuators' limit (and how often at it),
  how far the whole robot's centre of mass stands behind its front feet (the topple margin; < 0: beyond them),
  the leg joints' smallest distance from their limits, the largest action.
At every fall (fell_over) and illegal contact on a flight the buffer is kept; for the robots still up it is kept every
`window` seconds (the windows that end on a flight). Printed per class -- direction (down / up) x payload (light /
mid / heavy, and for the heavy the load's place along the body: front / centre / rear of x +-5 cm) -- the medians of the failed episodes at
-2.0 ... -0.02 s before the end, the share of them that ended at a stop (and how long after the zero command), and
the same quantities' medians over the windows that did not end in a fall.

Run with UNIFIED_TASKS=1 from the repo root, the run's switches in the environment (its timeline's env_vars; U_RECIPE=0
for a run from before recipe.py)."""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("UNIFIED_TASKS", "1")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["U_EVAL_ENVS"] = "0"   # every robot acts alike here and all of them are measured

import numpy as np  # noqa: E402
import torch  # noqa: E402

SIGNALS = ("cmd m/s", "speed m/s", "pitch deg", "pitch rate deg/s", "feet down", "front feet down", "front torque",
           "front calf torque", "rear torque", "front margin cm", "joint margin rad", "action max", "zero cmd",
           "on stairs")
I = {s: i for i, s in enumerate(SIGNALS)}
SHOWN = len(SIGNALS) - 2   # the last two are masks
AT = (-2.0, -1.5, -1.0, -0.6, -0.4, -0.2, -0.1, -0.02)


def column_names(env) -> list[str]:
  """Each curriculum column's sub-terrain name (as fail_probe.py)."""
  g = env.scene.terrain.cfg.terrain_generator
  names = list(g.sub_terrains.keys())
  prop = np.array([c.proportion for c in g.sub_terrains.values()], dtype=float)
  cum = np.cumsum(prop / prop.sum())
  return [names[int(np.min(np.where(i / g.num_cols + 0.001 < cum)[0]))] for i in range(g.num_cols)]


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("task")
  ap.add_argument("checkpoint")
  ap.add_argument("--num-envs", type=int, default=768)
  ap.add_argument("--steps", type=int, default=3000)
  ap.add_argument("--window", type=float, default=2.0)
  ap.add_argument("--spread-rows", action="store_true", help="start on rows spread evenly over all of them")
  ap.add_argument("--no-load", action="store_true", help="no checkpoint: large random actions (a dry run of the probe itself: the robots fall)")
  ap.add_argument("--any-ground", action="store_true", help="keep the falls and windows off the flights too")
  a = ap.parse_args()

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  import src.tasks  # noqa: F401
  from src.tasks.velocity.blind_env_cfg import MAP3_STAIR_TERRAIN_NAMES
  from src.tasks.velocity.mdp.rewards import uneven_terrain_mask

  cfg = load_env_cfg(a.task)
  cfg.scene.num_envs = a.num_envs
  hits: dict[str, torch.Tensor] = {}

  def wrap(name):
    term = cfg.terminations.get(name)
    if term is None:
      return
    orig = term.func

    def f(env, **kw):
      out = orig(env, **kw)
      hits[name] = out.clone()
      return out

    term.func = f

  wrap("fell_over")
  wrap("illegal_contact")
  agent = load_rl_cfg(a.task)
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  venv = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  runner = load_runner_cls(a.task)(venv, asdict(agent), log_dir=None, device="cuda:0")
  if not a.no_load:
    runner.load(a.checkpoint, map_location="cuda:0")
  policy = runner.get_inference_policy(device="cuda:0")
  dev, dt, n = env.device, env.step_dt, a.num_envs
  if a.no_load:
    n_act = env.action_manager.action.shape[1]
    policy = lambda o: 3.0 * torch.randn(n, n_act, device=dev)   # noqa: E731
  robot = env.scene["robot"]
  terrain = env.scene.terrain
  cols = column_names(env)
  stairs_col = torch.tensor([c in MAP3_STAIR_TERRAIN_NAMES for c in cols], device=dev)
  up_col = torch.tensor([c in MAP3_STAIR_TERRAIN_NAMES and "up" in c for c in cols], device=dev)
  if a.spread_rows:
    n_rows = int(terrain.max_terrain_level)
    ldt = terrain.terrain_levels.dtype
    spread = {k: (torch.randperm(n, device=dev) % n_rows).to(ldt) for k in ("up", "down", "other")}
    if getattr(env, "u_rows", None) is not None:
      env.u_rows = spread
    on, up = stairs_col[terrain.terrain_types], up_col[terrain.terrain_types]
    terrain.terrain_levels[:] = torch.where(up, spread["up"], torch.where(on, spread["down"], spread["other"]))
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    cm = env.curriculum_manager
    compute = cm.compute
    cm.compute = lambda *a_, **k_: None
    try:
      env.reset()
    finally:
      cm.compute = compute

  # the legs' joints, front and rear, and their actuators' limits
  jn = list(robot.joint_names)
  front = torch.tensor([i for i, x in enumerate(jn) if x.startswith(("FL", "FR")) and "wheel" not in x], device=dev)
  rear = torch.tensor([i for i, x in enumerate(jn) if x.startswith(("RL", "RR")) and "wheel" not in x], device=dev)
  legs = torch.cat([front, rear])
  assert len(front) == 6 and len(rear) == 6, jn
  # the torques come per actuator (robot.actuator_names), the limits from the actuators' configuration (as
  # diagnostics.physics reads them)
  an = list(robot.actuator_names)
  effort = torch.full((len(an),), float("nan"), device=dev)
  for act in env.cfg.scene.entities["robot"].articulation.actuators:
    expr = tuple(getattr(act, "target_names_expr", ()) or getattr(getattr(act, "base_cfg", None), "target_names_expr", ()))
    c = act
    while not hasattr(c, "effort_limit") and hasattr(c, "base_cfg"):
      c = c.base_cfg
    lim = getattr(c, "effort_limit", None)
    if not expr or lim is None:
      continue
    for i, x in enumerate(an):
      if any(re.fullmatch(e, x) for e in expr):
        effort[i] = float(lim)
  a_front = torch.tensor([i for i, x in enumerate(an) if x.startswith(("FL", "FR")) and "wheel" not in x], device=dev)
  a_rear = torch.tensor([i for i, x in enumerate(an) if x.startswith(("RL", "RR")) and "wheel" not in x], device=dev)
  a_front_calf = torch.tensor([i for i, x in enumerate(an) if x.startswith(("FL", "FR")) and "calf" in x], device=dev)
  assert len(a_front) == 6 and len(a_rear) == 6 and len(a_front_calf) == 2, an
  assert bool(torch.isfinite(effort[torch.cat([a_front, a_rear])]).all()), (an, effort.tolist())
  # the feet: the contact sensor's order, and the sites the centre-of-mass cost reads (or the feet's own bodies)
  feet_sensor = env.scene["feet_ground_contact"]
  slot_names = list(dict.fromkeys(sl.primary_name for sl in feet_sensor._slots))
  front_slots = torch.tensor([i for i, x in enumerate(slot_names) if x.startswith(("FL", "FR"))], device=dev)
  body_ids = robot.indexing.body_ids
  mass_all = env.sim.model.body_mass
  root = int(robot.indexing.root_body_id)                   # the body the payload event loads (base_payload_point)
  m0 = float(env.sim.get_default_field("body_mass")[root])
  x0 = float(env.sim.get_default_field("body_ipos")[root][0])
  limit = float(getattr(env, "u_payload_full", 0.0) or 0.0)
  kg_now = lambda: getattr(env, "payload_kg", torch.zeros(n, device=dev)).reshape(-1).float()   # noqa: E731

  def state() -> torch.Tensor:
    d = robot.data
    cmd = env.command_manager.get_command("twist")
    g = d.projected_gravity_b
    pitch = torch.rad2deg(torch.atan2(g[:, 0], -g[:, 2]))
    found = (feet_sensor.data.found > 0).float()
    tau = d.actuator_force.abs() / effort
    mass = mass_all[:, body_ids]
    com = (d.body_com_pos_w * mass.unsqueeze(-1)).sum(dim=1) / mass.sum(dim=1, keepdim=True)
    q = d.root_link_quat_w
    yaw = torch.atan2(2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2))
    head = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=1)
    feet = feet_pos()
    ahead = ((feet[..., :2] - com[:, None, :2]) * head[:, None, :]).sum(dim=-1)   # each foot ahead of the CoM, m
    margin = ahead.topk(2, dim=1).values.mean(dim=1) * 100.0
    lim = d.soft_joint_pos_limits[:, legs]
    pos = d.joint_pos[:, legs]
    jm = torch.minimum(pos - lim[..., 0], lim[..., 1] - pos).min(dim=1).values
    act = env.action_manager.action.abs().max(dim=1).values
    return torch.stack([
      torch.norm(cmd[:, :2], dim=1), d.root_link_lin_vel_b[:, 0], pitch, torch.rad2deg(d.root_link_ang_vel_b[:, 1]),
      found.sum(dim=1), found[:, front_slots].sum(dim=1), tau[:, a_front].max(dim=1).values,
      tau[:, a_front_calf].max(dim=1).values, tau[:, a_rear].max(dim=1).values, margin, jm, act,
      (torch.norm(cmd[:, :3], dim=1) < 0.1).float(), uneven_terrain_mask(env, "terrain_scan").float(),
    ], dim=1)

  foot_bodies = [i for i, x in enumerate(robot.body_names) if "foot" in x.lower()]
  if len(foot_bodies) != 4:   # no foot bodies by that name: the calves' ends are not known here, take the sensor's
    foot_bodies = [i for i, x in enumerate(robot.body_names) if x in slot_names]
  assert len(foot_bodies) == 4, (robot.body_names, slot_names)
  foot_bodies_t = torch.tensor(foot_bodies, device=dev)

  def feet_pos() -> torch.Tensor:
    return robot.data.body_link_pos_w[:, foot_bodies_t]

  w = max(2, round(a.window / dt))
  k_sig = len(SIGNALS)
  buf = torch.full((w, n, k_sig), float("nan"), device=dev)
  slot = 0
  age = torch.zeros(n, dtype=torch.long, device=dev)        # steps of this episode in the buffer
  stand = torch.zeros(n, device=dev)                         # seconds the command has been zero
  falls: list[dict] = []
  calm: list[dict] = []
  obs = venv.get_observations()
  seen = {"fell over": 0, "illegal contact": 0, "of them on a stairs column": 0, "on a flight": 0,
          "on a flight, at least a quarter window old": 0}

  def ordered(ids: torch.Tensor) -> torch.Tensor:
    """The robots' windows, oldest step first: [len(ids), w, k]."""
    order = (torch.arange(w, device=dev) + slot) % w
    return buf[order][:, ids].permute(1, 0, 2)

  def klass(ids: torch.Tensor, types: torch.Tensor, kg: torch.Tensor, bx: torch.Tensor) -> list[tuple[str, str, str]]:
    """Direction by the column's name, payload class by its share of the limit (constraints' LIGHT_FRAC / HEAVY_FRAC),
    and for the heavy the load's place along the body: its x from the base's centre of mass, payload included
    (the base's own noise, +-base_com_range, blurs it by a few cm)."""
    out = []
    for i in ids.tolist():
      c = cols[int(types[i])]
      direction = "up" if "up" in c else "down"
      m = float(kg[i])
      frac = m / limit if limit > 0 else 0.0
      pay = "light" if frac < 0.25 else ("heavy" if frac >= 2.0 / 3.0 else "mid")
      place = "-"
      if pay == "heavy":
        px = ((m0 + m) * float(bx[i]) - m0 * x0) / m
        place = "front" if px > 0.05 else ("rear" if px < -0.05 else "centre")
      out.append((direction, pay, place))
    return out

  for k in range(a.steps):
    types = terrain.terrain_types.clone()
    kg = kg_now().clone()
    bx = env.sim.model.body_ipos[:, root, 0].clone()
    stand_before = stand.clone()
    hits.clear()
    with torch.no_grad():
      obs, _, dones, _ = venv.step(policy(obs))
    fell = hits.get("fell_over", torch.zeros(n, dtype=torch.bool, device=dev)).bool()
    touch = hits.get("illegal_contact", torch.zeros(n, dtype=torch.bool, device=dev)).bool() & ~fell
    done = dones.bool().reshape(-1)
    last = buf[(slot - 1) % w]                              # the step before this one, every robot
    on_flight = stairs_col[types] & (last[:, I["on stairs"]] > 0.5)
    if a.any_ground:
      on_flight = torch.ones_like(on_flight)
    bad = (fell | touch) & done & on_flight & (age >= w // 4)
    seen["fell over"] += int((fell & done).sum())
    seen["illegal contact"] += int((touch & done).sum())
    seen["of them on a stairs column"] += int(((fell | touch) & done & stairs_col[types]).sum())
    seen["on a flight"] += int(((fell | touch) & done & on_flight).sum())
    seen["on a flight, at least a quarter window old"] += int(bad.sum())
    if bad.any() and k > 50:
      ids = bad.nonzero().flatten()
      win = ordered(ids).cpu().numpy()
      for j, (i, kl) in enumerate(zip(ids.tolist(), klass(ids, types, kg, bx))):
        short = w - int(min(age[i], w))
        if short:
          win[j, :short] = np.nan
        falls.append({"class": kl, "win": win[j], "how": "fell" if bool(fell[i]) else "touch",
                      "stand_s": float(stand_before[i]), "col": cols[int(types[i])]})
    # the robots still up: a window every `window` seconds, where it ends on a flight under the same episode
    if k > 50 and k % w == 0:
      keep = ~done & (age >= w) & on_flight
      if keep.any():
        ids = keep.nonzero().flatten()
        if len(ids) > 64:
          ids = ids[torch.randperm(len(ids), device=dev)[:64]]
        win = ordered(ids).cpu().numpy()
        for j, kl in enumerate(klass(ids, types, kg, bx)):
          calm.append({"class": kl, "win": win[j]})
    age = torch.where(done, torch.zeros_like(age), age)
    s = state()
    buf[slot] = s
    slot = (slot + 1) % w
    age += 1
    stand = torch.where(s[:, I["zero cmd"]] > 0.5, stand + dt, torch.zeros_like(stand))
    stand = torch.where(done, torch.zeros_like(stand), stand)

  print(f"{a.checkpoint}\n{a.task}: {n} robots, {a.steps} steps ({a.steps * dt:.0f} s each); payload limit {limit:g} kg; "
        f"window {a.window:g} s; {len(falls)} failed episodes on the flights kept, {len(calm)} windows without one")
  print("ended episodes: " + "; ".join(f"{k_} {v}" for k_, v in seen.items()))
  print("actuator limits, Nm: " + ", ".join(f"{x} {effort[i]:g}" for i, x in enumerate(an) if x.startswith("FL")))
  print("signals: torque = the largest |torque| / limit over the six front (rear) leg joints; front margin = the "
        "whole robot's centre of mass behind its two front feet, cm over the ground (< 0: beyond them); joint margin "
        "= the leg joints' smallest distance from a limit")
  idx = [max(0, min(w - 1, w + round(t / dt))) for t in AT]

  def med(x: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
      return np.nanmedian(x, axis=0)

  def share(x: np.ndarray, sig: str, test) -> float:
    with np.errstate(all="ignore"):
      v = x[..., I[sig]]
      return float(np.nanmean(np.where(np.isnan(v), np.nan, test(v).astype(float))))

  def shares(x: np.ndarray) -> str:
    return (f"torque at the limit: front {share(x, 'front torque', lambda v: v >= 0.98):.1%} of the steps (front calf "
            f"{share(x, 'front calf torque', lambda v: v >= 0.98):.1%}), rear {share(x, 'rear torque', lambda v: v >= 0.98):.1%}; "
            f"at most one front foot down {share(x, 'front feet down', lambda v: v <= 1.0):.1%}, none "
            f"{share(x, 'front feet down', lambda v: v < 0.5):.1%}; centre of mass beyond the front feet "
            f"{share(x, 'front margin cm', lambda v: v < 0.0):.1%}; a leg joint within 0.02 rad of a limit "
            f"{share(x, 'joint margin rad', lambda v: v < 0.02):.1%}")

  classes = sorted({f["class"] for f in falls} | {c["class"] for c in calm})
  for kl in classes:
    fs = [f for f in falls if f["class"] == kl]
    cs = [c for c in calm if c["class"] == kl]
    name = f"{kl[0]} / {kl[1]}" + (f" / load {kl[2]}" if kl[2] != "-" else "")
    print(f"\n== {name}: {len(fs)} failed, {len(cs)} windows without a fall")
    if cs:
      allc = np.stack([c["win"] for c in cs])               # [m, w, k]
      mc = med(allc.reshape(-1, k_sig))
      print("   no fall, medians: " + "; ".join(f"{s_} {mc[i]:.2f}" for i, s_ in enumerate(SIGNALS[:SHOWN])))
      print(f"   no fall: {shares(allc)}; zero command {share(allc, 'zero cmd', lambda v: v > 0.5):.1%}")
    if len(fs) < 3:
      continue
    allf = np.stack([f["win"] for f in fs])                 # [m, w, k]
    how = sum(f["how"] == "fell" for f in fs)
    at_stop = [f["stand_s"] for f in fs if f["stand_s"] > 0.0]
    early = sum(1 for s_ in at_stop if s_ <= 2.0)
    by_col: dict[str, int] = {}
    for f in fs:
      by_col[f["col"]] = by_col.get(f["col"], 0) + 1
    print(f"   failed: {how} fell over, {len(fs) - how} illegal contact; {len(at_stop)} at a stop ({early} within 2 s "
          f"of the zero command, median {np.median(at_stop) if at_stop else float('nan'):.1f} s after it); by "
          f"sub-terrain: " + ", ".join(f"{c} {v}" for c, v in sorted(by_col.items(), key=lambda kv: -kv[1])))
    last1 = allf[:, w - round(1.0 / dt):w - round(0.2 / dt)]   # -1.0 .. -0.2 s
    print(f"   failed, -1.0..-0.2 s: {shares(last1)}")
    print("   failed, medians at   " + "".join(f"{t:>8.2f}" for t in AT) + "  s before the end")
    for i, s_ in enumerate(SIGNALS[:SHOWN + 1]):
      row = med(allf[:, idx, i])
      print(f"   {s_:>20s} " + "".join(f"{v:>8.2f}" for v in row))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
