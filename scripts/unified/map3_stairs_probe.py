#!/usr/bin/env python3
"""map3_stairs_probe.py TASK CHECKPOINT [--num-envs 64] [--up-vx 0.5] [--flights F ...] [--teacher-scan | --no-teacher-scan]
(unitree-rl-mjlab-62 2026-11-15, go2-depth's request): two readings on the map3 stairwell that no gate has, both on
every up flight. The user 11-15 on the Go2W teacher's videos: "上楼梯在中间停止做不到" and "在平台的时候会出现前面是墙然后
做抬腿翻越动作".

Each up flight of map3 (scripts/go2_blind/eval_map3.py's --set flights: spawned on the bottom landing, the operator
holding the flight heading and its centerline as eval_map3 does) runs one script per robot, its own clock:

  climb      --up-vx up the flight until the base has gained half the flight's rise
  mid stop   zero command 3 s (the operator off)                                        -> reading (c)
  climb      on until the base has gained the whole rise (less 4 cm), then on along the landing until the first rail
             ahead is within 1.0 m (at most 3 s)
  stand 2 s, turn in place wz +1.0 for 3 s, stand 2 s on the top landing               -> reading (c')

(c)  the mid-flight stop: per robot the braking distance (the displacement along the flight over the first 1 s after
     the zero command, + = on up the flight) and the drift after it (1-3 s); a robot "held" when the drift is within
     +-5 cm (the stairs gate's stop: < 5 cm over the hold's last 2 s), "went on up" above +5 cm, "slid back" below
     -5 cm; the largest backward excursion from the stop point; the mean horizontal speed over 1.5-2.0 s; falls.
(c') the top landing facing the end wall: over the stand / turn / stand script, the distance from the base along the
     flight's direction to the first rail ahead that spans the base's lateral position; the share of steps with a
     front foot / wheel off the ground (no contact pressing > 5 N) and the front lift-offs per robot, split by that
     distance (its p50 printed), the stand (4 s) and the turn (3 s) apart; the front feet's highest point above their
     lowest in the first stand (the wheel's centre / the calf's tip) in the turn and the last stand; the robots with a
     front foot pressing on a vertical face (contact normal |n_z| < 0.5: on a landing, the wall); falls.
U_SCAN / U_PAYLOAD from the environment as the gates set them. A privileged-scan teacher (a *-U-Teacher-* task,
default --teacher-scan) gets eval_map3.teacher_boxes (the storeys above cut, the rails cut, the well filled), as
record_map3_tour.py --teacher-scan. Judges nothing. Run from the repo root (UNIFIED_TASKS=1 set here); a student
checkpoint through jobs/unified/u_eval.py.
"""
from __future__ import annotations

import argparse
import gc
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "go2_blind"))
os.environ.setdefault("UNIFIED_TASKS", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import torch  # noqa: E402

MID_SHARE = 0.5
GOAL_SHARE = 0.85
STOP_S = 3.0
ONTO_MAX_S = 3.0
ONTO_WALL_M = 1.0
LANDING = ((2.0, 0.0), (3.0, 1.0), (2.0, 0.0))   # (seconds, wz) on the top landing
HELD_M = 0.05
FORCE_N = 5.0


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("task")
  ap.add_argument("checkpoint")
  ap.add_argument("--num-envs", type=int, default=64)
  ap.add_argument("--up-vx", type=float, default=0.5)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--flights", nargs="*", default=None, help="flight names (default: every up flight)")
  ap.add_argument("--teacher-scan", action=argparse.BooleanOptionalAction, default=None)
  a = ap.parse_args()

  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.sensor import ContactMatch, ContactSensorCfg

  import eval_map3 as em

  teacher_scan = ("-U-Teacher-" in a.task) if a.teacher_scan is None else a.teacher_scan
  boxes = em.load_map3_boxes()
  ups = [s for s in em.flight_scenarios(boxes, a.up_vx, 0.4) if s.name.endswith("_up")]
  if a.flights:
    ups = [s for s in ups if s.name[:-3] in a.flights]
  print(f"checkpoint {a.checkpoint}\ntask {a.task}, scan {os.environ.get('U_SCAN', '(task default)')}, payload "
        f"{os.environ.get('U_PAYLOAD', '(task default)')}, up {a.up_vx} m/s, {a.num_envs} robots per flight, "
        f"teacher scan boxes {teacher_scan}, {len(ups)} up flights")
  rows = []
  for sc in ups:
    rise = float(sc.success.split(":")[1]) / GOAL_SHARE
    climb_s = sum(d for d, vx, *_ in sc.segments if vx)
    sc.segments = [(1.0, 0, 0, 0), (climb_s + STOP_S + ONTO_MAX_S + sum(d for d, _ in LANDING) + 4.0, a.up_vx, 0, 0)]
    bx = em.teacher_boxes(boxes, sc) if teacher_scan else boxes
    cfg = em.make_env_cfg(sc, bx, a.num_envs, a.seed, a.task)
    cfg.scene.sensors = tuple(cfg.scene.sensors or ()) + tuple(
      ContactSensorCfg(name=f"m3_{leg}_foot", primary=ContactMatch(mode="geom", entity="robot",
                                                                  pattern=(rf"^{leg}_foot_collision$",)),
                       secondary=ContactMatch(mode="body", pattern="terrain"), fields=("found", "force", "normal"),
                       reduce="none", num_slots=1, history_length=4)
      for leg in ("FL", "FR"))
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    venv, policy = em.load_policy(env, a.checkpoint, a.task)
    rows.append(run_flight(env, venv, policy, sc, bx, rise, a.up_vx))
    env.close()
    del env, venv, policy
    gc.collect()
    torch.cuda.empty_cache()
  report(rows)
  return 0


def np_cpu(t):
  return t.detach().cpu().numpy()


def run_flight(env, venv, policy, sc, bx, rise: float, up_vx: float) -> dict:
  import eval_map3 as em

  dev, dt, n = env.device, env.step_dt, env.num_envs
  robot, term = env.scene["robot"], env.command_manager.get_term("twist")
  feet = [env.scene[f"m3_{leg}_foot"] for leg in ("FL", "FR")]
  # the front feet's height: the wheel's centre (a wheeled robot), else the calf's end (the foot sphere at its tip)
  wheel_ids, _ = robot.find_bodies([r"^FL_wheel_link$", r"^FR_wheel_link$"]) if any(
    "wheel_link" in b for b in robot.body_names) else ([], [])
  calf_ids, _ = robot.find_bodies([r"^FL_calf$", r"^FR_calf$"])

  def front_feet_z():
    if wheel_ids:
      return robot.data.body_link_pos_w[:, wheel_ids, 2]
    q = robot.data.body_link_quat_w[:, calf_ids]          # w x y z
    w, x, y = q[..., 0], q[..., 1], q[..., 2]
    zz = 1 - 2 * (x * x + y * y)                          # the calf frame's z axis, its world z component
    return robot.data.body_link_pos_w[:, calf_ids, 2] - 0.213 * zz
  env.reset()
  obs = venv.get_observations()
  yaw = math.radians(sc.yaw_deg)
  axis = torch.tensor((math.cos(yaw), math.sin(yaw)), device=dev)
  # the rails ahead along the flight's direction, in world coordinates
  shift = np_cpu(env.scene.env_origins[0]) - (np.asarray(sc.spawn) + em.MAP3_OFFSET)   # the generator's placement
  rails = [(p + em.MAP3_OFFSET + shift, s) for name, s, p in bx if name.startswith("rail")]
  rc = torch.tensor(np.array([p for p, _ in rails]), dtype=torch.float32, device=dev)
  rh = torch.tensor(np.array([s for _, s in rails]), dtype=torch.float32, device=dev)
  ax = int(abs(axis[1]) > abs(axis[0]))
  sgn = float(torch.sign(axis[ax]))

  def wall_ahead(pos):
    """distance along the flight's direction to the first rail ahead spanning the base's lateral position and
    height (inf when none)."""
    lat_ok = (pos[:, None, 1 - ax] - rc[None, :, 1 - ax]).abs() <= rh[None, :, 1 - ax]
    z_ok = (rc[None, :, 2] + rh[None, :, 2] > pos[:, None, 2] - 0.1) & (rc[None, :, 2] - rh[None, :, 2] < pos[:, None, 2] + 0.5)
    near_face = (rc[None, :, ax] - sgn * rh[None, :, ax] - pos[:, None, ax]) * sgn
    d = torch.where(lat_ok & z_ok & (near_face > -0.05), near_face, torch.full_like(near_face, float("inf")))
    return d.min(dim=1).values

  em.set_command(env, up_vx, 0.0, 0.0)
  with torch.no_grad():
    obs, _, _, _ = venv.step(policy(obs))   # the command resamples to (up_vx, 0, 0) once, then is written per robot
  z0 = robot.data.root_link_pos_w[:, 2].clone()
  phase = torch.zeros(n, dtype=torch.long, device=dev)   # 0 climb, 1 stop, 2 climb, 3 onto, 4..6 landing, 7 done
  t_ph = torch.zeros(n, device=dev)
  fell = torch.zeros(n, dtype=torch.bool, device=dev)
  fell_stop = torch.zeros(n, dtype=torch.bool, device=dev)
  fell_land = torch.zeros(n, dtype=torch.bool, device=dev)
  fell_phase = torch.full((n,), -1, dtype=torch.long, device=dev)
  s_stop = torch.full((n,), float("nan"), device=dev)
  disp1, disp3, back_min = s_stop.clone(), s_stop.clone(), torch.zeros(n, device=dev)
  v_acc, v_n = torch.zeros(n, device=dev), torch.zeros(n, device=dev)
  stopped = torch.zeros(n, dtype=torch.bool, device=dev)
  reached = torch.zeros(n, dtype=torch.bool, device=dev)
  land = {q: torch.zeros(n, device=dev) for q in ("stand_n", "stand_off", "stand_lift", "turn_n", "turn_off",
                                                  "turn_lift", "wall_sum", "wall_touch")}
  g0 = torch.full((n,), float("inf"), device=dev)          # the front feet's lowest height in the first stand
  lift_turn, lift_stand = torch.zeros(n, device=dev), torch.zeros(n, device=dev)
  prev_c = torch.ones(n, 2, dtype=torch.bool, device=dev)
  land_dur = [d for d, _ in LANDING]
  max_steps = int(round(sum(d for d, *_ in sc.segments) / dt))
  for _ in range(max_steps):
    pos = robot.data.root_link_pos_w
    s_now = (pos[:, :2] - env.scene.env_origins[:, :2]) @ axis
    gain = pos[:, 2] - z0
    # phase transitions
    go_stop = (phase == 0) & (gain >= MID_SHARE * rise) & ~fell
    s_stop = torch.where(go_stop, s_now, s_stop)
    stopped |= go_stop
    phase = torch.where(go_stop, torch.ones_like(phase), phase)
    t_ph = torch.where(go_stop, torch.zeros_like(t_ph), t_ph)
    end_stop = (phase == 1) & (t_ph >= STOP_S)
    phase = torch.where(end_stop, torch.full_like(phase, 2), phase)
    top = (phase == 2) & (gain >= rise - 0.04) & ~fell
    reached |= top
    phase = torch.where(top, torch.full_like(phase, 3), phase)
    t_ph = torch.where(top | end_stop, torch.zeros_like(t_ph), t_ph)
    dw = wall_ahead(pos)
    for k, dur in ((3, ONTO_MAX_S), (4, land_dur[0]), (5, land_dur[1]), (6, land_dur[2])):
      nxt = (phase == k) & ((t_ph >= dur) | ((dw <= ONTO_WALL_M) if k == 3 else False))
      phase = torch.where(nxt, torch.full_like(phase, k + 1), phase)
      t_ph = torch.where(nxt, torch.zeros_like(t_ph), t_ph)
    if bool(((phase == 7) | fell).all()):
      break
    # commands per robot: walking phases with the operator (eval_map3's), the rest still / turning
    walk = (phase == 0) | (phase == 2) | (phase == 3)
    heading_err = torch.atan2(torch.sin(yaw - robot.data.heading_w), torch.cos(yaw - robot.data.heading_w))
    rel = pos[:, :2] - env.scene.env_origins[:, :2]
    lateral = -axis[1] * rel[:, 0] + axis[0] * rel[:, 1]
    zero = torch.zeros(n, device=dev)
    term.vel_command_b[:, 0] = torch.where(walk, torch.full_like(zero, up_vx), zero)
    term.vel_command_b[:, 1] = torch.where(walk, (-lateral).clamp(-0.2, 0.2), zero)
    term.vel_command_b[:, 2] = torch.where(walk, heading_err.clamp(-0.5, 0.5),
                                           torch.where(phase == 5, torch.full_like(zero, LANDING[1][1]), zero))
    with torch.no_grad():
      obs, _, dones, extras = venv.step(policy(obs))
    terminated = dones.bool() & ~extras.get("time_outs", torch.zeros_like(dones)).bool()
    new_fall = terminated & ~fell
    fell_phase = torch.where(new_fall, phase, fell_phase)
    fell_stop |= new_fall & (phase == 1)
    fell_land |= new_fall & (phase >= 4) & (phase <= 6)
    fell |= terminated
    t_ph += dt
    alive = ~fell
    pos = robot.data.root_link_pos_w
    s_now = (pos[:, :2] - env.scene.env_origins[:, :2]) @ axis
    in_stop = (phase == 1) & alive
    d_rel = s_now - s_stop
    back_min = torch.where(in_stop, torch.minimum(back_min, d_rel), back_min)
    at1 = in_stop & (t_ph >= 1.0) & torch.isnan(disp1)
    disp1 = torch.where(at1, d_rel, disp1)
    at3 = in_stop & (t_ph >= STOP_S - 0.5 * dt) & torch.isnan(disp3)
    disp3 = torch.where(at3, d_rel, disp3)
    win = in_stop & (t_ph >= 1.5) & (t_ph <= 2.0)
    v_acc += torch.where(win, torch.norm(robot.data.root_link_lin_vel_w[:, :2], dim=1), zero)
    v_n += win.float()
    on_land = (phase >= 4) & (phase <= 6) & alive
    c = torch.stack([(torch.norm(f.data.force_history.reshape(n, -1, 3), dim=-1).max(dim=1).values > FORCE_N)
                     for f in feet], dim=1)
    wall_c = torch.stack([f.data.normal.reshape(n, -1)[:, 2].abs() < 0.5 for f in feet], dim=1) & c
    land["wall_touch"] += (on_land & wall_c.any(dim=1)).float()
    fz = front_feet_z().min(dim=1).values, front_feet_z().max(dim=1).values
    g0 = torch.where(on_land & (phase == 4), torch.minimum(g0, fz[0]), g0)
    up = torch.where(torch.isfinite(g0), fz[1] - g0, zero)
    lift_turn = torch.where(on_land & (phase == 5), torch.maximum(lift_turn, up), lift_turn)
    lift_stand = torch.where(on_land & (phase == 6), torch.maximum(lift_stand, up), lift_stand)
    off = (~c).any(dim=1)
    lifts = (prev_c & ~c).sum(dim=1).float()
    prev_c = c
    dw = wall_ahead(pos)
    land["wall_sum"] += torch.where(on_land, dw.clamp(max=5.0), zero)
    for tag, m in (("stand", on_land & (phase != 5)), ("turn", on_land & (phase == 5))):
      land[f"{tag}_n"] += m.float()
      land[f"{tag}_off"] += (m & off).float()
      land[f"{tag}_lift"] += torch.where(m, lifts, zero)
  np_ = lambda t: t.cpu().numpy()
  return {"name": sc.name[:-3], "rise": rise, "stopped": np_(stopped), "brake": np_(disp1),
          "disp2": np_(disp3 - disp1), "back_min": np_(back_min),
          "v": np_(v_acc / v_n.clamp(min=1)), "v_n": np_(v_n), "fell_stop": np_(fell_stop), "reached": np_(reached),
          "fell_land": np_(fell_land), "fell": np_(fell), **{k: np_(v) for k, v in land.items()}, "dt": dt,
          "lift_turn": np_(lift_turn), "lift_stand": np_(lift_stand), "phase": np_(phase), "fell_phase": np_(fell_phase)}


PHASES = ("climb to mid", "mid stop", "climb on", "onto landing", "stand", "turn", "stand 2", "done")


def report(rows) -> None:
  print("== where each robot ended (alive) and where the fallen fell, per flight ==")
  for r in rows:
    alive = r["fell_phase"] < 0
    end = {PHASES[k]: int(((r["phase"] == k) & alive).sum()) for k in range(len(PHASES))}
    falls = {PHASES[k]: int((r["fell_phase"] == k).sum()) for k in range(len(PHASES))}
    print(f"{r['name']:10s} ended: " + ", ".join(f"{k} {v}" for k, v in end.items() if v)
          + " | fell in: " + (", ".join(f"{k} {v}" for k, v in falls.items() if v) or "none"))

  print("\n== (c) the mid-flight stop on the climb: braking (0-1 s after the zero command) and drift (1-3 s); held = "
        "drift within +-5 cm ==")
  allv = {q: [] for q in ("disp2", "back_min", "v", "brake")}
  tot = {"n": 0, "held": 0, "up": 0, "back": 0, "fell": 0}
  for r in rows:
    m = r["stopped"] & ~np.isnan(r["disp2"])
    d = r["disp2"][m]
    held, up, back = (np.abs(d) <= HELD_M), (d > HELD_M), (d < -HELD_M)
    fs = int(r["fell_stop"].sum())
    tot["n"] += int(m.sum()) + fs
    tot["held"] += int(held.sum())
    tot["up"] += int(up.sum())
    tot["back"] += int(back.sum())
    tot["fell"] += fs
    for q in allv:
      allv[q].append(r[q][m])
    q = lambda x, p: float(np.percentile(x, p)) if x.size else float("nan")
    print(f"{r['name']:10s} rise {r['rise']:.2f} m: stopped {int(r['stopped'].sum())}, held {100 * held.mean() if d.size else 0:5.1f}%, "
          f"went on up {100 * up.mean() if d.size else 0:5.1f}%, slid back {100 * back.mean() if d.size else 0:5.1f}%, "
          f"fell {fs} | braking p50 / p90 {q(r['brake'][m], 50):+.3f} / {q(r['brake'][m], 90):+.3f} m | drift 1-3 s "
          f"p10 / p50 / p90 {q(d, 10):+.3f} / {q(d, 50):+.3f} / {q(d, 90):+.3f} m, "
          f"backward max p90 {q(-r['back_min'][m], 90):.3f} m | speed 1.5-2 s p50 / p90 {q(r['v'][m], 50):.3f} / "
          f"{q(r['v'][m], 90):.3f} m/s")
  d = np.concatenate(allv["disp2"]) if allv["disp2"] else np.array([])
  n = max(tot["n"], 1)
  print(f"ALL mid stop: robots {tot['n']}, held {100 * tot['held'] / n:.1f}%, went on up {100 * tot['up'] / n:.1f}%, "
        f"slid back {100 * tot['back'] / n:.1f}%, fell {tot['fell']}; braking p50 "
        f"{np.median(np.concatenate(allv['brake'])) if d.size else float('nan'):+.3f} m; drift p50 "
        f"{np.median(d) if d.size else float('nan'):+.3f} m, |drift| p90 "
        f"{np.percentile(np.abs(d), 90) if d.size else float('nan'):.3f} m, speed p90 "
        f"{np.percentile(np.concatenate(allv['v']), 90) if d.size else float('nan'):.3f} m/s")

  print("\n== (c') the top landing, facing the end wall (walked on until the first rail ahead is within 1.0 m): stand 2 s, "
        "turn in place wz +1.0 3 s, stand 2 s; front foot / wheel off the ground (no contact > 5 N) ==")
  agg = {k: 0.0 for k in ("stand_n", "stand_off", "stand_lift", "turn_n", "turn_off", "turn_lift")}
  nrob, fl_tot, reach_tot, rob_tot, walls, lifts_all, touch_all = 0, 0, 0, 0, [], [], []
  for r in rows:
    dt = r["dt"]
    has = (r["stand_n"] + r["turn_n"]) > 0
    nrob += int(has.sum())
    for k in agg:
      agg[k] += float(r[k][has].sum())
    w = r["wall_sum"][has] / np.maximum((r["stand_n"] + r["turn_n"])[has], 1)
    walls.append(w)
    line = []
    for tag in ("stand", "turn"):
      share = r[f"{tag}_off"][has].sum() / max(r[f"{tag}_n"][has].sum(), 1)
      line.append(f"{tag}: front off {100 * share:5.1f}% of the time, {r[f'{tag}_lift'][has].sum() / max(int(has.sum()), 1):.2f} "
                  f"lifts / robot")
    fl_tot += int(r["fell_land"].sum())
    reach_tot += int(r["reached"].sum())
    rob_tot += len(r["reached"])
    lt, ls = r["lift_turn"][has], r["lift_stand"][has]
    q = lambda x, p: float(np.percentile(x, p)) if x.size else float("nan")
    lifts_all.append(lt)
    touch_all.append(r["wall_touch"][has])
    print(f"{r['name']:10s} reached the top {100 * r['reached'].mean():5.1f}%, on the landing {int(has.sum())}, wall ahead p50 "
          f"{np.median(w) if w.size else float('nan'):.2f} m, fell on the landing {int(r['fell_land'].sum())} | " + " | ".join(line)
          + f" | front foot highest above the stand: turn p50 / p90 {100 * q(lt, 50):.1f} / {100 * q(lt, 90):.1f} cm, last stand "
          f"{100 * q(ls, 50):.1f} / {100 * q(ls, 90):.1f} cm; > 10 cm {100 * (lt > 0.10).mean() if lt.size else 0:.0f}% | front foot "
          f"on the wall {100 * (r['wall_touch'][has] > 0).mean() if has.any() else 0:.0f}% of the robots, "
          f"{r['wall_touch'][has].sum() * dt / max(int(has.sum()), 1):.2f} s / robot")
  w = np.concatenate(walls) if walls else np.array([])
  print(f"ALL top landing: reached {reach_tot} / {rob_tot}, on the landing {nrob}, wall ahead p50 "
        f"{np.median(w) if w.size else float('nan'):.2f} m, fell on the landing {fl_tot}; stand (4 s): front off "
        f"{100 * agg['stand_off'] / max(agg['stand_n'], 1):.1f}%, {agg['stand_lift'] / max(nrob, 1):.2f} lifts / robot; turn "
        f"(3 s): front off {100 * agg['turn_off'] / max(agg['turn_n'], 1):.1f}%, {agg['turn_lift'] / max(nrob, 1):.2f} lifts / robot")
  lt = np.concatenate(lifts_all) if lifts_all else np.array([])
  tc = np.concatenate(touch_all) if touch_all else np.array([])
  if lt.size:
    print(f"ALL front foot highest above the stand in the turn: p50 / p90 {100 * np.median(lt):.1f} / "
          f"{100 * np.percentile(lt, 90):.1f} cm, > 10 cm {100 * (lt > 0.10).mean():.0f}% of the robots; a front foot on the "
          f"wall (contact normal |n_z| < 0.5): {100 * (tc > 0).mean():.0f}% of the robots")


if __name__ == "__main__":
  sys.exit(main())
