#!/usr/bin/env python3
"""descent_probe.py TASK CHECKPOINT [--envs-per-tile 18] [--vx 0.5] [--seed 0] (unitree-rl-mjlab-62 2026-11-15): the
legs' contacts with the steps and the body's rocking on a flight, per band and direction. The user, 11-15 after the
Go2 teacher's videos: "go2空载下楼的时候为什么也遥遥晃晃的？你有记录下楼时后小腿与台阶的碰撞吗" and "下楼时，主要关注后腿
是否会有碰撞，还需要看下下楼时机身晃动幅度". The stairs gate reads foot kicks on risers going up and leg_contact (a
thigh or calf pressing, pooled over the stairs columns); neither says which leg, where on the step, nor how the body
moves meanwhile.

The run is the stairs gate's traverse (scripts/go2_blind/eval_stairs.py: the Stairs check task of TASK's robot, play
build, the stair tiles 4 bands 5-18 cm x up / down x narrow / wide, --envs-per-tile robots each, a forward command
--vx from the tile centre outwards; U_SCAN / U_PAYLOAD from the environment as jobs/unified/stairs_gate.sh sets them).
One contact sensor per leg and segment against the terrain, from the robot's leg_ground_touch patterns (thigh, calf;
the [FR][LR] of a pattern taken per leg), a contact = > 5 N in the sensor's history (as the leg_contact item). On
the flight (Chebyshev distance 1.0-3.6 m from the tile centre, about 8-10 steps), per robot:

  contacts   rear (RL + RR) and front (FL + FR) apart, calf and thigh apart: the contact onsets per robot per flight,
             the share of robots with any, the share of the contact steps against a riser or a step's edge (the
             contact normal's |n_z| < 0.7; else resting on a tread), the peak force (p50 over the robots that touch)
  sway       the std of the base's roll and pitch (deg, from the projected gravity, around the robot's own mean on
             the flight: the slope's lean is out) and the RMS of the roll / pitch rates (deg/s); the central platform
             before the flight (0.2-0.9 m, level) as each robot's own reference
  jolt       the mean |roll rate| / |pitch rate| over the 0.1 s after a rear calf's contact onset against the
             flight's other steps (a ratio near 1: the contacts do not rock the body)
  path       (the user 11-15: "不是旋转晃动，是左右晃动，像是喝醉了走不了直线") the lateral offset from the line along the
             spawn heading, fitted against the forward distance: the drift angle, and the residual split by a 0.5 s
             average into the slow weaving over several steps ("meander", std and peak-to-peak) and the rest ("gait
             sway"); the heading off the spawn heading (mean |.|, std, max |.|); the body-frame lateral speed and the
             yaw rate (RMS); the lateral speed / yaw rate in the 0.1 s after a calf's onset, front and rear, on the
             12-18 cm descents. --straight spawns square to the steps (the check's stair spawn is k x 90 deg
             +- 15 deg: squaring up would read as drift)

A robot that falls counts until its fall. One "SUM" line per band and direction for comparing arms. Judges nothing.
Run from the repo root with UNIFIED_TASKS=1 (set here); a student checkpoint (the depth student's memory) through
jobs/unified/u_eval.py:

  U_SCAN=on U_PAYLOAD=0,0,0,0 python scripts/unified/descent_probe.py Unitree-Go2-U-Teacher-Stairs CK
  U_SCAN=on U_PAYLOAD=0,0,0,0 python jobs/unified/u_eval.py scripts/unified/descent_probe.py TASK CK

The Go2 formal teacher seed 0 model_5200 (11-15, unloaded, seen; 72 robots per band and direction; ~1 min on a
laptop RTX 3060), 15-18 cm down: rear calf 5.94 onsets per flight (97 % of the robots, 86 % on a step's edge, peak
258 N p50), front calf 9.50 (636 N), thighs ~0, a leg on the steps 15 % of the flight's steps (12-15 cm 8.9 %, up
<= 0.6 %); sway roll / pitch std 2.65 / 5.20 deg, rates 43 / 43 deg/s against 0.8 / 1.2 deg and 18 / 19 deg/s on the
platform, 15-18 cm up 2.99 / 4.93 deg, 42 / 46 deg/s; jolt ratio 0.97 / 0.89 (8-15 cm down 1.1-1.4). The path,
--straight: 15-18 cm down meander 8.8 cm peak-to-peak (p90 11.4; 5-8 cm down 2.7, 15-18 cm up 4.2), gait sway 0.7 cm,
drift 2.5 deg (p90 6.1), heading max 7.0 deg, lateral speed 0.109 m/s (5-8 cm down 0.049, 15-18 cm up 0.091), yaw
rate 16.3 deg/s; after a front calf's onset the yaw rate 21.7 deg/s against 10.8 (rear 16.2), the lateral speed 0.098
against 0.077 m/s.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("UNIFIED_TASKS", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import torch  # noqa: E402

NUM_ROWS = 4
STEP_RANGE = (0.05, 0.18)
LEGS = ("FL", "FR", "RL", "RR")
SIDES = {"rear": ("RL", "RR"), "front": ("FL", "FR")}
FORCE_N = 5.0      # a contact, as the leg_contact item (src/tasks/velocity/mdp/constraints.py _leg_contact)
EDGE_NZ = 0.7      # |n_z| below: a riser or a step's edge
FLIGHT = (1.0, 3.6)
PLATFORM = (0.2, 0.9)
JOLT_S = 0.1


def band(lvl: int) -> str:
  lo, hi = STEP_RANGE
  return f"{100 * (lo + (hi - lo) * lvl / NUM_ROWS):.0f}-{100 * (lo + (hi - lo) * (lvl + 1) / NUM_ROWS):.0f}cm"


def leg_segments(cfg) -> dict[tuple[str, str], tuple[str, ...]]:
  """(leg, "thigh" | "calf") -> the geom patterns of that leg, from the robot's leg_ground_touch sensor."""
  sensor = next(s for s in cfg.scene.sensors if s.name == "leg_ground_touch")
  pats = sensor.primary.pattern
  pats = (pats,) if isinstance(pats, str) else tuple(pats)
  out: dict[tuple[str, str], tuple[str, ...]] = {}
  for leg in LEGS:
    for seg in ("thigh", "calf"):
      mine = tuple(p.replace("[FR][LR]", leg) for p in pats if seg in p)
      if mine:
        out[(leg, seg)] = mine
  return out


def spearman(x: np.ndarray, y: np.ndarray) -> float:
  if len(x) < 5 or np.std(x) == 0 or np.std(y) == 0:
    return float("nan")
  rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
  return float(np.corrcoef(rx, ry)[0, 1])


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("task")
  ap.add_argument("checkpoint")
  ap.add_argument("--vx", type=float, default=0.5)
  ap.add_argument("--envs-per-tile", type=int, default=18)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--payload", default=None,
                  help="kg,x,y,z: a fixed point mass for an OLD (non-unified) task, scripts/blind/eval/payload_opt.py's "
                       "way (a unified task takes U_PAYLOAD; an old one keeps its training draw otherwise)")
  ap.add_argument("--straight", action="store_true",
                  help="spawn square to the steps: the stair spawn's yaw noise (k x 90 deg +- 15 deg by default, "
                       "mdp.reset_root_state_terrain_aware) set to 0, so turning square to the steps cannot read as "
                       "drift")
  a = ap.parse_args()
  if a.payload and "-U-" in a.task:
    ap.error("--payload is for an old task; a unified one (-U-) takes U_PAYLOAD (both would stack two payload events)")

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.sensor import ContactMatch, ContactSensorCfg
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from src.tasks.velocity.blind_env_cfg import STAIR_TERRAIN_NAMES

  cfg = load_env_cfg(a.task, play=True)
  cfg.seed = a.seed
  gen = cfg.scene.terrain.terrain_generator
  gen.sub_terrains = {k: v for k, v in gen.sub_terrains.items() if k in STAIR_TERRAIN_NAMES}
  for v in gen.sub_terrains.values():
    v.proportion = 1.0
  gen.num_rows = NUM_ROWS
  gen.num_cols = len(gen.sub_terrains) * 2
  cfg.events.pop("randomize_terrain")
  if a.payload:
    sys.path.insert(0, str(ROOT / "scripts" / "blind" / "eval"))
    import payload_opt
    payload_opt.apply(cfg, a.payload)
  if a.straight:
    cfg.events["reset_base"].params["stair_yaw_noise"] = 0.0
    cfg.events["reset_base"].params["narrow_yaw_noise"] = 0.0
  n = gen.num_rows * gen.num_cols * a.envs_per_tile
  cfg.scene.num_envs = n
  segs = leg_segments(cfg)
  cfg.scene.sensors = tuple(cfg.scene.sensors or ()) + tuple(
    ContactSensorCfg(name=f"descent_{leg}_{seg}", primary=ContactMatch(mode="geom", entity="robot", pattern=pats),
                     secondary=ContactMatch(mode="body", pattern="terrain"), fields=("found", "force", "normal"),
                     reduce="none", num_slots=1, history_length=4)
    for (leg, seg), pats in segs.items())

  agent = load_rl_cfg(a.task)
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  venv = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  runner = load_runner_cls(a.task)(venv, asdict(agent), device="cuda:0")
  runner.load(a.checkpoint, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  policy = runner.get_inference_policy(device="cuda:0")
  dev, dt = env.device, env.step_dt
  terrain, robot, cmd = env.scene.terrain, env.scene["robot"], env.command_manager.get_term("twist")
  sens = {k: env.scene[f"descent_{k[0]}_{k[1]}"] for k in segs}

  names = list(gen.sub_terrains.keys())
  cum = np.cumsum(np.full(len(names), 1.0 / len(names)))
  col = [names[np.min(np.where(c / gen.num_cols + 0.001 < cum)[0])] for c in range(gen.num_cols)]
  tile = torch.arange(n, device=dev) // a.envs_per_tile
  levels, types = tile % gen.num_rows, tile // gen.num_rows
  lv = levels.cpu().numpy()
  dr = np.array(["down" if "down" in col[t] else "up" for t in types.cpu().numpy()])

  terrain.terrain_levels[:] = levels
  terrain.terrain_types[:] = types
  terrain.env_origins[:] = terrain.terrain_origins[levels, types]
  env.reset()
  obs = venv.get_observations()
  cmd.vel_command_b[:] = torch.tensor((a.vx, 0.0, 0.0), device=dev)
  cmd.time_left[:] = 1e9

  def cheb():
    return (robot.data.root_link_pos_w[:, :2] - terrain.env_origins[:, :2]).abs().max(dim=1).values

  def zeros():
    return torch.zeros(n, device=dev)

  alive = torch.ones(n, dtype=torch.bool, device=dev)
  done = torch.zeros_like(alive)
  fell = torch.zeros_like(alive)
  zones = {"platform": PLATFORM, "flight": FLIGHT}
  acc = {z: {q: zeros() for q in ("n", "r", "r2", "p", "p2", "wr2", "wp2")} for z in zones}
  c = {k: {q: zeros() for q in ("steps", "onsets", "edge", "fmax")} for k in segs}
  prev = {k: torch.zeros(n, dtype=torch.bool, device=dev) for k in segs}
  any_leg = zeros()
  post_steps = max(1, int(round(JOLT_S / dt)))
  since = torch.full((n,), 10 ** 6, device=dev)
  jl = {q: zeros() for q in ("post_n", "post_r", "post_p", "oth_n", "oth_r", "oth_p")}
  # the path (lateral section): each step's planar position, heading, body-frame lateral velocity, yaw rate, and the
  # calf contact onsets, front and rear -- a few MB for the whole run
  yaw0 = robot.data.heading_w.clone()
  p0 = robot.data.root_link_pos_w[:, :2].clone()
  tr: dict[str, list] = {q: [] for q in ("x", "y", "yaw", "vy", "wz", "d", "act", "fc", "rc")}
  for _ in range(int(round(FLIGHT[1] / max(abs(a.vx), 0.1) * 2.0 / dt))):
    with torch.no_grad():
      obs, _, _, _ = venv.step(policy(obs))
    term = env.termination_manager.terminated
    act = alive & ~done
    fell |= term & act
    alive &= ~(term & act)
    act = alive & ~done
    d = cheb()
    g = robot.data.projected_gravity_b
    roll = torch.rad2deg(torch.atan2(g[:, 1], -g[:, 2]))
    pitch = torch.rad2deg(torch.atan2(g[:, 0], -g[:, 2]))
    w = torch.rad2deg(robot.data.root_link_ang_vel_b)
    for z, (lo, hi) in zones.items():
      f, A = (act & (d >= lo) & (d <= hi)).float(), acc[z]
      A["n"] += f
      A["r"] += roll * f
      A["r2"] += roll * roll * f
      A["p"] += pitch * f
      A["p2"] += pitch * pitch * f
      A["wr2"] += w[:, 0] ** 2 * f
      A["wp2"] += w[:, 1] ** 2 * f
    on_fl = act & (d >= FLIGHT[0]) & (d <= FLIGHT[1])
    touching = torch.zeros(n, dtype=torch.bool, device=dev)
    fc_on = torch.zeros(n, dtype=torch.bool, device=dev)
    rc_on = torch.zeros(n, dtype=torch.bool, device=dev)
    for k in segs:
      sd = sens[k].data
      fh = torch.norm(sd.force_history.reshape(n, -1, 3), dim=-1).max(dim=1).values
      hit = fh > FORCE_N
      nz = sd.normal.reshape(n, -1)[:, 2].abs()
      h = hit & on_fl
      C = c[k]
      C["steps"] += h.float()
      C["onsets"] += (hit & ~prev[k] & on_fl).float()
      C["edge"] += (h & (nz < EDGE_NZ)).float()
      C["fmax"] = torch.where(h, torch.maximum(C["fmax"], fh), C["fmax"])
      if k[0] in SIDES["rear"] and k[1] == "calf":
        since = torch.where(hit & ~prev[k] & on_fl, torch.zeros_like(since), since)
        rc_on |= hit & ~prev[k] & on_fl
      elif k[1] == "calf":
        fc_on |= hit & ~prev[k] & on_fl
      prev[k] = hit
      touching |= h
    any_leg += touching.float()
    for tag, m in (("post", on_fl & (since < post_steps)), ("oth", on_fl & ~(since < post_steps))):
      jl[f"{tag}_n"] += m.float()
      jl[f"{tag}_r"] += w[:, 0].abs() * m.float()
      jl[f"{tag}_p"] += w[:, 1].abs() * m.float()
    since += 1
    pw = robot.data.root_link_pos_w[:, :2]
    for q, v in (("x", pw[:, 0]), ("y", pw[:, 1]), ("yaw", robot.data.heading_w),
                 ("vy", robot.data.root_link_lin_vel_b[:, 1]), ("wz", w[:, 2]), ("d", d), ("act", act),
                 ("fc", fc_on), ("rc", rc_on)):
      tr[q].append(v.clone())
    done |= alive & (d >= FLIGHT[1])

  def np_(t):
    return t.cpu().numpy()

  C = {k: {q: np_(v) for q, v in d_.items()} for k, d_ in c.items()}
  S = {z: {q: np_(v) for q, v in A.items()} for z, A in acc.items()}
  J = {q: np_(v) for q, v in jl.items()}
  fl_n = S["flight"]["n"]
  anyl = np_(any_leg)

  def sway(z):
    A = S[z]
    k_ = np.maximum(A["n"], 1)
    rstd = np.sqrt(np.maximum(A["r2"] / k_ - (A["r"] / k_) ** 2, 0))
    pstd = np.sqrt(np.maximum(A["p2"] / k_ - (A["p"] / k_) ** 2, 0))
    return rstd, pstd, np.sqrt(A["wr2"] / k_), np.sqrt(A["wp2"] / k_)

  sw = {z: sway(z) for z in zones}
  print(f"checkpoint {a.checkpoint}\ntask {a.task}, scan {os.environ.get('U_SCAN', '(task default)')}, payload "
        f"{a.payload + ' (--payload)' if a.payload else os.environ.get('U_PAYLOAD', '(task default)')}, vx {a.vx}, {n} robots ({n // (NUM_ROWS * 2)} per band and "
        f"direction), spawn yaw {'square to the steps' if a.straight else 'k x 90 deg +- 15 deg'}, falls {int(np_(fell).sum())}, got past {int(np_(done).sum())}; segments "
        + ", ".join(f"{leg} {seg} {'|'.join(p)}" for (leg, seg), p in segs.items() if leg == "RL"))

  print("\n== contacts on the flight (> 5 N): onsets per robot per flight / robots with any / on an edge or riser "
        "(share of the contact steps) / peak force p50 N ==")
  for direc in ("down", "up"):
    for lvl in range(NUM_ROWS):
      m = (lv == lvl) & (dr == direc) & (fl_n > 0)
      cells = []
      for side, legs in SIDES.items():
        for seg in ("calf", "thigh"):
          ks = [(leg, seg) for leg in legs if (leg, seg) in C]
          if not ks:
            continue
          on = sum(C[k]["onsets"][m] for k in ks)
          st = sum(C[k]["steps"][m] for k in ks)
          ed = sum(C[k]["edge"][m] for k in ks)
          fm = np.maximum.reduce([C[k]["fmax"][m] for k in ks])
          fm = fm[fm > 0]
          cells.append(f"{side} {seg} {on.mean():4.2f} {100 * (on > 0).mean():3.0f}% edge "
                       f"{100 * ed.sum() / max(st.sum(), 1):3.0f}% F {np.median(fm) if fm.size else 0:4.0f}")
      share = anyl[m].sum() / max(fl_n[m].sum(), 1)
      print(f"{band(lvl):8s} {direc:4s}: " + " | ".join(cells) + f" | any leg {100 * share:4.1f}% of the steps")
  print("  per leg (calf onsets per robot per flight): " + "; ".join(
    f"{band(lvl)} {direc} " + " ".join(f"{leg} {C[(leg, 'calf')]['onsets'][(lv == lvl) & (dr == direc)].mean():.2f}"
                                       for leg in LEGS if (leg, "calf") in C)
    for direc in ("down", "up") for lvl in (2, 3)))

  print("\n== sway per robot: p50 / p90 over the robots of a band and direction ==")
  for z in zones:
    rstd, pstd, wr, wp = sw[z]
    for direc in ("down", "up"):
      for lvl in range(NUM_ROWS):
        m = (lv == lvl) & (dr == direc) & (S[z]["n"] >= 10)
        if not m.any():
          continue

        def pp(x):
          return f"{np.median(x[m]):5.2f} / {np.percentile(x[m], 90):5.2f}"
        print(f"{z:8s} {band(lvl):8s} {direc:4s}: roll std {pp(rstd)} deg | pitch std {pp(pstd)} deg | roll rate rms "
              f"{pp(wr)} deg/s | pitch rate rms {pp(wp)} deg/s | n {int(m.sum())}")

  print(f"\n== jolt: mean |roll rate| / |pitch rate| (deg/s) over the {post_steps} steps after a rear calf's contact "
        "onset against the flight's other steps; per robot Spearman of the rear calf onsets with the rate rms ==")
  rev = sum(C[(leg, "calf")]["onsets"] for leg in SIDES["rear"] if (leg, "calf") in C)
  _, _, wr_f, wp_f = sw["flight"]
  for direc in ("down", "up"):
    for lvl in range(NUM_ROWS):
      m = (lv == lvl) & (dr == direc) & (fl_n >= 10)
      pn, on_ = J["post_n"][m].sum(), J["oth_n"][m].sum()
      if pn == 0:
        print(f"{band(lvl):8s} {direc:4s}: no rear calf contact")
        continue
      pr, pq = J["post_r"][m].sum() / pn, J["post_p"][m].sum() / pn
      orr, oq = J["oth_r"][m].sum() / max(on_, 1), J["oth_p"][m].sum() / max(on_, 1)
      print(f"{band(lvl):8s} {direc:4s}: after roll {pr:5.1f} pitch {pq:5.1f} (steps {int(pn)}) | other roll {orr:5.1f} "
            f"pitch {oq:5.1f} | ratio {pr / max(orr, 1e-6):4.2f} / {pq / max(oq, 1e-6):4.2f} | rho roll "
            f"{spearman(rev[m], wr_f[m]):+.2f} pitch {spearman(rev[m], wp_f[m]):+.2f}")

  print()
  for direc in ("down", "up"):
    for lvl in range(NUM_ROWS):
      m = (lv == lvl) & (dr == direc) & (fl_n >= 10)
      vals = {}
      for side, legs in SIDES.items():
        for seg in ("calf", "thigh"):
          ks = [(leg, seg) for leg in legs if (leg, seg) in C]
          vals[f"{side}_{seg}"] = sum(C[k]["onsets"][m] for k in ks).mean() if ks else float("nan")
      rstd, pstd, wr, wp = sw["flight"]
      print(f"SUM {band(lvl)} {direc} " + " ".join(f"{k} {v:.2f}" for k, v in vals.items())
            + f" roll_std {np.median(rstd[m]):.2f} pitch_std {np.median(pstd[m]):.2f} roll_rate {np.median(wr[m]):.1f}"
            f" pitch_rate {np.median(wp[m]):.1f} falls {int(np_(fell)[(lv == lvl) & (dr == direc)].sum())}")
  lateral(tr, np_(yaw0), np_(p0), lv, dr, dt)
  env.close()
  return 0


def lateral(tr, yaw0, p0, lv, dr, dt) -> None:
  """The path's straightness on the flight (the user 11-15: "不是旋转晃动，是左右晃动，像是喝醉了走不了直线"). Per robot,
  over its flight steps: the lateral offset from the line through its spawn point along its spawn heading (the
  command is straight ahead, wz 0), a straight-line fit of it against the forward distance (its slope = the drift
  angle), and the residual split by a 0.5 s moving average into the slow part ("meander": weaving over several steps)
  and the rest ("gait sway": the side-to-side motion within a gait cycle); the heading off the spawn heading; the
  body-frame lateral speed and the yaw rate; and the lateral speed / yaw rate in the 0.1 s after a calf's contact
  onset (front, rear) against the flight's other steps."""
  T = {q: torch.stack(v).cpu().numpy() for q, v in tr.items()}   # [steps, robots]
  ex, ey = np.cos(yaw0), np.sin(yaw0)
  dx, dy = T["x"] - p0[:, 0], T["y"] - p0[:, 1]
  s, lat = dx * ex + dy * ey, -dx * ey + dy * ex
  dyaw = np.degrees(np.arctan2(np.sin(T["yaw"] - yaw0), np.cos(T["yaw"] - yaw0)))
  fl = T["act"] & (T["d"] >= FLIGHT[0]) & (T["d"] <= FLIGHT[1])
  win = max(3, int(round(0.5 / dt)))
  post = max(1, int(round(JOLT_S / dt)))
  n = lat.shape[1]
  R = {q: np.full(n, np.nan) for q in ("meander", "meander_pp", "sway", "drift", "hd_mean", "hd_std", "hd_max",
                                       "vy", "wz")}
  jolt = {q: [0.0, 0] for q in ("fc_vy", "fc_wz", "rc_vy", "rc_wz", "oth_vy", "oth_wz")}
  for i in range(n):
    idx = np.where(fl[:, i])[0]
    if len(idx) < 3 * win:
      continue
    si, li = s[idx, i], lat[idx, i]
    b, a0 = np.polyfit(si, li, 1)
    r = li - (a0 + b * si)
    m = np.convolve(np.pad(r, win // 2, mode="edge"), np.ones(win) / win, mode="valid")[:len(r)]
    R["meander"][i] = 100 * np.std(m)
    R["meander_pp"][i] = 100 * (m.max() - m.min())
    R["sway"][i] = 100 * np.std(r - m)
    R["drift"][i] = abs(np.degrees(np.arctan(b)))
    hd = dyaw[idx, i]
    R["hd_mean"][i], R["hd_std"][i], R["hd_max"][i] = np.mean(np.abs(hd)), np.std(hd), np.max(np.abs(hd))
    R["vy"][i] = np.sqrt(np.mean(T["vy"][idx, i] ** 2))
    R["wz"][i] = np.sqrt(np.mean(T["wz"][idx, i] ** 2))
    R.setdefault("_i", []).append(i)
    if dr[i] != "down" or lv[i] < 2:
      continue
    after = {"fc": np.zeros(len(T["vy"]), bool), "rc": np.zeros(len(T["vy"]), bool)}
    for kind in after:
      for t0 in np.where(T[kind][:, i])[0]:
        after[kind][t0:t0 + post] = True
    for kind in ("fc", "rc"):
      sel = idx[after[kind][idx]]
      jolt[f"{kind}_vy"][0] += np.abs(T["vy"][sel, i]).sum()
      jolt[f"{kind}_vy"][1] += len(sel)
      jolt[f"{kind}_wz"][0] += np.abs(T["wz"][sel, i]).sum()
      jolt[f"{kind}_wz"][1] += len(sel)
    sel = idx[~(after["fc"] | after["rc"])[idx]]
    jolt["oth_vy"][0] += np.abs(T["vy"][sel, i]).sum()
    jolt["oth_vy"][1] += len(sel)
    jolt["oth_wz"][0] += np.abs(T["wz"][sel, i]).sum()
    jolt["oth_wz"][1] += len(sel)

  print("\n== the path on the flight: meander (the slow weaving, 0.5 s average, cm: std and peak-to-peak), gait sway "
        "(the rest, cm std), drift angle (|deg|), heading off the spawn heading (mean |.| / std / max |.| deg), "
        "lateral speed rms (m/s), yaw rate rms (deg/s); p50 / p90 over the robots ==")
  for direc in ("down", "up"):
    for lvl in range(NUM_ROWS):
      m = (lv == lvl) & (dr == direc) & ~np.isnan(R["meander"])
      if not m.any():
        continue

      def pp(x, f="5.1f"):
        return f"{np.median(x[m]):{f}} / {np.percentile(x[m], 90):{f}}"
      print(f"{band(lvl):8s} {direc:4s}: meander {pp(R['meander'])} cm, p-p {pp(R['meander_pp'])} cm | sway "
            f"{pp(R['sway'])} cm | drift {pp(R['drift'])} deg | heading {pp(R['hd_mean'])} / std {pp(R['hd_std'])} / "
            f"max {pp(R['hd_max'])} deg | vy {pp(R['vy'], '5.3f')} m/s | wz {pp(R['wz'])} deg/s | n {int(m.sum())}")
  print("  after a calf onset (0.1 s), 12-18 cm down pooled: |vy| front "
        f"{jolt['fc_vy'][0] / max(jolt['fc_vy'][1], 1):.3f} rear {jolt['rc_vy'][0] / max(jolt['rc_vy'][1], 1):.3f} "
        f"other {jolt['oth_vy'][0] / max(jolt['oth_vy'][1], 1):.3f} m/s; |wz| front "
        f"{jolt['fc_wz'][0] / max(jolt['fc_wz'][1], 1):.1f} rear {jolt['rc_wz'][0] / max(jolt['rc_wz'][1], 1):.1f} "
        f"other {jolt['oth_wz'][0] / max(jolt['oth_wz'][1], 1):.1f} deg/s")
  for direc in ("down", "up"):
    for lvl in range(NUM_ROWS):
      m = (lv == lvl) & (dr == direc) & ~np.isnan(R["meander"])
      if m.any():
        print(f"SUMLAT {band(lvl)} {direc} " + " ".join(f"{q} {np.median(R[q][m]):.3f}" for q in
                                                        ("meander", "meander_pp", "sway", "drift", "hd_mean", "hd_max",
                                                         "vy", "wz"))
              + f" hd_max_p90 {np.percentile(R['hd_max'][m], 90):.3f} n {int(m.sum())}")


if __name__ == "__main__":
  sys.exit(main())
