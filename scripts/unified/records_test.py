#!/usr/bin/env python3
"""records_test.py [OUT_DIR] (go2-depth 2026-11-12): the training records (src/tasks/unified/diagnostics.py) and
their reader (run_report.py) on made-up data. No simulator and no training: a stand-in environment and runner on the
CPU feed the real `episodes` term and the real Recorder a scripted "training" in which known things go wrong --

  run A (iterations 0-1199, then an exception): the walled_up ground fails 70% of its episodes on the left front
        calf from iteration 300 on; on the flat ground the robots with a payload fail 45% of theirs (the light ones
        5%), those with the payload at the front 75%; the "other" grounds' fail rate stays over the limit, so the
        reference schedule's first milestone ("walk") is overdue at its iteration; the action noise climbs from
        iteration 700; one constraint's multiplier reaches its limit while its reading stays short; the "up" family
        relaxes (the evaluation robots fail half of their climbs on their fixed rows, the light ones 0.37 against
        a reference of 0.15) with no stairs constraint on;
        progress passes 0.5 with the quality stage waiting; the "training" ends in a RuntimeError at iteration 1199.
        The stairs quality stays 0, so the walled_up ground and the relaxed family raise no alert: before the
        quality stage a stairs ground's fail rate is the curriculum's own doing (~0.6 at the rows it promotes to)
  run B (resumed from A's checkpoint 1100, iterations 1100-1299): finishes normally; the milestones A reached
        before 1100 are carried over (reached once per training), "walk" is still overdue
  run C (iterations 0-559): the stairs quality stage from the start (q 0.6), the same walled_up ground and relaxed
        "up" family: now both are alerts and problems, judged on the evaluation robots' episodes on their fixed rows

  no run: the real curriculum._eval_fail, constraints._families and constraints.measure on a few hundred made-up
        evaluation episodes (the stairs families' relaxing against the reference, the survival items' reading),
        constraints.stop_creep on scripted stands (the acceptance's time base: a push, a command and a new episode
        end the window), constraints.load on a checkpoint whose creep item was measured another way,
        and teacher_gate.py on a made-up TensorBoard file (a gate-only item judged at the bar of the code on disk)

-- and checks that the records hold exactly what was fed in (every episode counted once, per ground / cause / body),
that the expected events and alerts are there and no part of the records faulted, and that run_report.py names the
planted problems. Prints the report of every run. Exit code 0: all checks passed.

Run from the repo root with the training environment's python (needs torch and mjlab for the imports only)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from collections import deque
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("UNIFIED_TASKS", "1")

import torch  # noqa: E402

from src.tasks.unified import constraints as ucons  # noqa: E402
from src.tasks.unified import curriculum as ucur  # noqa: E402
from src.tasks.unified import diagnostics  # noqa: E402
from src.tasks.unified.teacher_env import constraint_items  # noqa: E402
from src.tasks.velocity.blind_env_cfg import MAP3_STAIR_TERRAIN_NAMES  # noqa: E402

N, K_EVAL, STEPS = 256, 32, 24
GROUNDS = {"stairs_down_narrow": 0.05, "stairs_down_wide": 0.05, "stairs_up_narrow": 0.10, "stairs_up_wide": 0.05,
           "random_rough": 0.10, "flat": 0.15, "walled_down": 0.20, "walled_up": 0.30}
ROW_STEPS = [0.05 + 0.015 * r for r in range(9)]   # the fixed rows' lowest steps: bands 0 0 1 1 1 2 2 3 3
BODIES = ["base", "FL_calf", "FR_calf", "RL_calf", "RR_calf"]
JOINTS = [f"{leg}_{part}_joint" for leg in ("FL", "FR", "RL", "RR") for part in ("hip", "thigh", "calf")]


class Terminations:
  active_terms = ["time_out", "fell_over", "illegal_contact", "nan"]

  def __init__(self):
    z = lambda: torch.zeros(N, dtype=torch.bool)   # noqa: E731
    self.terms = {k: z() for k in self.active_terms}

  time_outs = property(lambda s: s.terms["time_out"])
  terminated = property(lambda s: s.terms["fell_over"] | s.terms["illegal_contact"] | s.terms["nan"])
  dones = property(lambda s: s.terminated | s.time_outs)

  def get_term(self, name):
    return self.terms[name]

  def get_term_cfg(self, name):
    return NS(time_out=name == "time_out", params={"sensor_name": "body_ground_touch", "force_threshold": 10.0})


def fake_env(items: list[dict]):
  gen = NS(sub_terrains={k: NS(proportion=v) for k, v in GROUNDS.items()}, num_cols=20)
  cols = diagnostics.column_grounds(NS(cfg=NS(terrain_generator=gen), terrain_types=torch.zeros(1)))
  terrain = NS(cfg=NS(terrain_generator=gen), terrain_types=torch.randint(0, 20, (N,)),
               terrain_levels=torch.randint(0, 10, (N,)), env_origins=torch.zeros(N, 3), max_terrain_level=10)
  data = NS(root_link_pos_w=torch.zeros(N, 3), root_link_quat_w=torch.tensor([[1.0, 0, 0, 0]]).repeat(N, 1),
            actuator_force=torch.zeros(N, 12), joint_pos=torch.zeros(N, 12), joint_vel=torch.zeros(N, 12),
            soft_joint_pos_limits=torch.tensor([-1.0, 1.0]).repeat(N, 12, 1),
            projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]).repeat(N, 1))
  robot = NS(data=data, actuator_names=tuple(JOINTS), joint_names=tuple(JOINTS), indexing=NS(root_body_id=0))
  i = torch.arange(N)
  loaded = (i // 2) % 2 == 1                      # (independent of the scan mode, i % 2)
  place = (i // 4) % 4                            # 0 front, 1 back, 2 side, 3 centre
  ipos = torch.zeros(N, 2, 3)
  ipos[:, 0, 0] = torch.where(loaded, torch.tensor([0.06, -0.06, 0.0, 0.01])[place], torch.tensor(0.0))
  ipos[:, 0, 1] = torch.where(loaded & (place == 2), 0.05, 0.0)
  sim = NS(model=NS(body_ipos=ipos), get_default_field=lambda name: torch.zeros(2, 3))
  sensor = NS(_slots=[NS(primary_name=b) for b in BODIES], data=NS(force_history=torch.zeros(N, len(BODIES), 4, 3)))

  class Scene(dict):
    pass

  scene = Scene(robot=robot, body_ground_touch=sensor)
  scene.terrain = terrain
  acts = [NS(target_names_expr=(".*hip_.*",), effort_limit=23.5), NS(target_names_expr=(".*thigh_.*",), effort_limit=23.5),
          NS(base_cfg=NS(target_names_expr=(".*calf_.*",), effort_limit=45.0))]
  twist = torch.zeros(N, 3)
  cons = NS(on=ucons.constraints.on, scale={c["name"]: 1.0 for c in items}, value={}, values={}, count={},
            relax={"up": False, "down": False, "other": False}, ref={},
            base={t: 1.0 for c in items for t in c["terms"]})
  cons.target = lambda c, p, q=1.0: ucons.constraints.target(cons, c, p, q)
  env = NS(
    num_envs=N, device="cpu", common_step_counter=0, step_dt=0.02, episode_length_buf=torch.zeros(N, dtype=torch.long),
    scene=scene, termination_manager=Terminations(), u_scan_on=(torch.arange(N) % 2).float(),
    payload_kg=torch.where(loaded, 4.0, 0.0).reshape(N, 1), u_payload_full=8.0, u_eval_mask=torch.arange(N) < K_EVAL,
    sim=sim,
    command_manager=NS(get_command=lambda name: twist,
                       get_term=lambda name: NS(cfg=NS(ranges=NS(lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.5, 0.5),
                                                                ang_vel_z=(-1.0, 1.0))))),
    curriculum_manager=NS(get_term_cfg=lambda name: NS(params={"items": items})),
    reward_manager=NS(active_terms=sorted(cons.base), get_term_cfg=lambda name: NS(weight=-1.0)),
    action_manager=NS(active_terms=["joint_pos"], action_term_dim=[12]),
    event_manager=NS(get_term_cfg=lambda name: NS(params={"heavy_max": 4.0})),
    cfg=NS(scene=NS(entities={"robot": NS(articulation=NS(actuators=acts))})),
    u_progress_term=NS(p=0.0, q=0.0, level=0.0, up=0.0, fail=0.0, fail_eval=0.0, fail_mode={"seen": 0.0, "blind": 0.0},
                       clock=0.0, q_gate="waiting_row", clock_slow=False, clock_fail=0.0),
    u_constraints=cons, u_family_fail={"up": 0.0, "down": 0.0, "other": 0.0},
    # (the evaluation robots' fixed stairs rows, curriculum.eval_rows: nine rows within the acceptance's steps)
    u_eval_fail={"up": 0.0, "down": 0.0}, u_eval_stat={k: {"rate": [0.0] * 9, "n": [0] * 9} for k in ("up", "down")},
    # (the same by payload class and scan mode, curriculum._eval_fail; the rows' lowest steps 5 .. 17 cm)
    u_eval_split={}, u_eval_row_count=9, u_eval_row_step=ROW_STEPS,
    u_rows={k: torch.zeros(N) for k in ("up", "down", "other")},
  )
  env.twist, env.cols, env.loaded, env.front = twist, cols, loaded, loaded & (place == 0)
  return env


class Writer:
  def __init__(self):
    self.tags = set()

  def add_scalar(self, tag, value, it):
    self.tags.add(tag)


class Runner:
  """What the Recorder touches of rsl_rl's OnPolicyRunner."""

  def __init__(self, env, log_dir: str, script):
    self.env = NS(unwrapped=env, clip_actions=10.0)
    self.cfg = {"num_steps_per_env": STEPS, "save_interval": 100, "seed": 0, "algorithm": {"learning_rate": 1e-3}}
    actor, critic = torch.nn.Linear(4, 12), torch.nn.Linear(4, 1)
    self.alg = NS(actor=actor, critic=critic, max_grad_norm=1.0, learning_rate=1e-3,
                  optimizer=torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-3),
                  storage=NS(rewards=torch.zeros(STEPS, N, 1), values=torch.zeros(STEPS, N, 1),
                             returns=torch.zeros(STEPS, N, 1), dones=torch.zeros(STEPS, N, 1),
                             actions=torch.zeros(STEPS, N, 12)))
    self.logger = NS(log_dir=log_dir, writer=Writer(), ep_extras=[], cfg=self.cfg, num_envs=N,
                     rewbuffer=deque(maxlen=100), lenbuffer=deque(maxlen=100), log=self._log)
    self.current_learning_iteration = 0
    self._commit, self._segments, self._segment = "test000", [], {"from": 0, "to": 0, "commit": "test000"}
    self.script = script
    self.fed = {"n": 0, "fail": 0, "by_ground": {}, "calf": 0, "flat_loaded": [0, 0], "flat_front": [0, 0]}
    self.rec = diagnostics.attach(self)

  def _log(self, it, start_it, total_it, collect_time, learn_time, loss_dict, learning_rate, action_std, rnd_weight,
           print_minimal=False, width=80, pad=40):
    self.logger.ep_extras.clear()

  def save(self, path, infos=None):
    Path(path).write_text("checkpoint")

  def load(self, path, load_cfg=None, strict=True, map_location=None):
    self.current_learning_iteration = int(Path(path).stem.split("_")[1])
    return {}

  def learn(self, num_learning_iterations, init_at_random_ep_len=False):
    env, ep = self.env.unwrapped, self.env.unwrapped.u_episodes
    start = self.current_learning_iteration
    total = start + num_learning_iterations
    g = torch.Generator().manual_seed(start)
    for it in range(start, total):
      s = self.script(it, env, self)
      for _ in range(4):   # four resets' worth of finished episodes per iteration
        env.common_step_counter += STEPS // 4
        ids = torch.randperm(N, generator=g)[:24]
        tm = env.termination_manager
        for t in tm.terms.values():
          t[:] = False
        ground = [env.cols[int(c)] for c in env.scene.terrain.terrain_types[ids]]
        rate = [s["fail"].get(x, 0.05) for x in ground]
        for j, x in enumerate(ground):   # the planted payload falls, on the flat ground
          if x == "flat" and env.loaded[ids[j]]:
            rate[j] = s["flat_front"] if env.front[ids[j]] else s["flat_loaded"]
        fail = torch.tensor([torch.rand((), generator=g).item() < x for x in rate])
        tm.terms["illegal_contact"][ids[fail]] = True
        tm.terms["time_out"][ids[~fail]] = True
        force = env.scene["body_ground_touch"].data.force_history
        force[:] = 0.0
        force[ids[fail], 1, 0, 2] = 50.0   # the left front calf
        env.episode_length_buf[ids] = torch.where(fail, 150, 1000)
        env.scene["robot"].data.root_link_pos_w[ids, 0] = torch.where(fail, 1.2, 6.5)
        env.twist[ids, 0] = 0.6
        ep(env, ids, stair_terrain_names=MAP3_STAIR_TERRAIN_NAMES)
        self.fed["n"] += len(ids)
        self.fed["fail"] += int(fail.sum())
        self.fed["calf"] += int(fail.sum())
        for j, (x, bad) in enumerate(zip(ground, fail.tolist())):
          d = self.fed["by_ground"].setdefault(x, [0, 0])
          d[0] += 1
          d[1] += int(bad)
          for key, mask in (("flat_loaded", env.loaded), ("flat_front", env.front)):
            if x == "flat" and mask[ids[j]]:
              self.fed[key][0] += 1
              self.fed[key][1] += int(bad)
        env.scene.terrain.terrain_types[ids] = torch.randint(0, 20, (len(ids),), generator=g)
      self.alg.optimizer.zero_grad()
      x = torch.randn(64, 4, generator=g)
      (self.alg.actor(x).pow(2).mean() + self.alg.critic(x).pow(2).mean()).backward()
      torch.nn.utils.clip_grad_norm_(self.alg.actor.parameters(), 1.0)
      torch.nn.utils.clip_grad_norm_(self.alg.critic.parameters(), 1.0)
      self.alg.optimizer.step()
      st = self.alg.storage
      st.returns[:] = torch.randn(STEPS, N, 1, generator=g)
      st.values[:] = st.returns + 0.3 * torch.randn(STEPS, N, 1, generator=g)
      st.actions[:] = s["std"] * torch.randn(STEPS, N, 12, generator=g)
      self.logger.rewbuffer.append(10.0)
      self.logger.lenbuffer.append(800.0)
      self.logger.ep_extras.append({"Episode_Reward/track": torch.tensor(1.0), "Curriculum/u_progress/p": 0.5})
      self.current_learning_iteration = it
      self.logger.log(it=it, start_it=start, total_it=total, collect_time=1.0, learn_time=0.2,
                      loss_dict={"value": 0.1, "surrogate": -0.01, "entropy": 5.0}, learning_rate=1e-3,
                      action_std=torch.full((64, 12), s["std"]), rnd_weight=None)
      if it % 100 == 0:
        self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))
      if s.get("raise"):
        raise RuntimeError("planted: CUDA error: device-side assert triggered")
    self.save(os.path.join(self.logger.log_dir, f"model_{total - 1}.pt"))


def script(items: list[dict]):
  """The planted story (module docstring)."""
  capped = next(c for c in items if c.get("enforce", True) and c["target"] is not None and c.get("by") == "p")

  def at(it, env, runner):
    t, cons = env.u_progress_term, env.u_constraints
    quality = runner.logger.log_dir.endswith("C")   # run C: in the stairs quality stage from the start
    t.p = 1.0 if quality else min(1.0, it / 1500.0)   # (a made-up pace)
    t.q, t.q_gate = (0.6, "running") if quality else (0.0, "waiting_row")
    t.level, t.up, t.clock = 2.0 + it / 600.0, 3.5 if quality else 1.5, it * STEPS * 1.0
    bad = it >= 300 or quality
    t.fail = 0.35 if bad else 0.05
    t.fail_eval, t.clock_slow, t.clock_fail = t.fail - 0.02, bad, 0.4 if bad else 0.05
    # (the training robots' climbing fail rate, at the rows the curriculum promotes them to: ~0.6 whatever they climb)
    env.u_family_fail.update(up=0.6, other=0.4 if bad else 0.05)
    # the evaluation robots on their fixed rows: the low rows climbed, the higher ones not from iteration 300 on
    rows = [0.1, 0.2, 0.3] + [0.65 if bad else 0.2] * 6
    env.u_eval_stat["up"] = {"rate": rows, "n": [20] * 9}
    env.u_eval_fail["up"] = sum(rows) / len(rows)   # (0.5 from iteration 300: over the relaxing limit)
    # of those the light robots' (what relaxes the stairs constraints, against the reference) and the loaded ones'
    light = [0.04, 0.1, 0.2] + [0.5 if bad else 0.1] * 6
    cell = lambda rate, n: {m: {"rate": list(rate), "n": [n] * 9} for m in ucur.SPLIT_MODES}   # noqa: E731
    env.u_eval_split["up"] = {"light": cell(light, 20), "mid": cell(rows, 20), "heavy": cell([0.0] * 9, 0)}
    cons.ref["up"] = 0.15
    cons.relax["up"] = bad
    for c in items:
      for m in c.get("modes", ucons.MODES):
        key = f"{c['name']}@{m}"
        if c["target"] is not None:   # every reading at its bar's good side ...
          cons.values[key] = c["target"] * (1.1 if c["sense"] == "min" else 0.9)
        else:
          cons.values[key] = 0.5
        cons.count[key] = it
    # ... but one: short of its bar with the multiplier at the limit from iteration 500
    for m in capped.get("modes", ucons.MODES):
      cons.values[f"{capped['name']}@{m}"] = capped["target"] * (0.5 if capped["sense"] == "min" else 2.0)
    for c in items:
      cons.value[c["name"]] = cons.values[f"{c['name']}@{c.get('modes', ucons.MODES)[0]}"]
    if it >= 500:
      cons.scale[capped["name"]] = capped["max_scale"]
    return {"fail": {"walled_up": 0.7} if bad else {}, "std": 0.4 if it < 700 else 0.4 + (it - 700) / 400.0,
            "flat_loaded": 0.35 if bad else 0.05, "flat_front": 0.75 if bad else 0.05,
            "raise": it == 1199 and runner.logger.log_dir.endswith("A")}

  return at, capped["name"]


def read(path: Path) -> list[dict]:
  return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def units(ok, out: Path) -> None:
  """The stairs families' relaxing and the survival items on the real functions, and the gate on a made-up
  TensorBoard file (module docstring)."""
  env = NS(num_envs=6, u_eval_row_count=9, u_eval_row_step=ROW_STEPS, common_step_counter=0, u_payload_full=8.0,
           payload_kg=torch.tensor([0.0, 2.0, 2.1, 5.3, 5.4, 8.0]).reshape(6, 1), u_family_fail={"other": 0.1},
           u_progress_term=NS(fail=0.1))
  ok("unit pay_class: light <= 2 kg, heavy >= 5.33 kg of 8", ucur.pay_class(env, torch.arange(6)) == [0, 0, 1, 1, 2, 2])
  me = NS(relax={f: False for f in ucons.FAMILIES}, ref={}, light={}, ref_step=None)
  step = [0]

  def feed(rate, per_row: int, cls: int = 0, its: int = 1) -> float | None:
    """per_row episodes on every fixed row failing `rate` of them (per row: a list), evenly spread, half in each
    scan mode; then the families judged `its` iterations later. The light robots' reading."""
    rate = rate if isinstance(rate, list) else [rate] * 9
    rows = [r for j in range(per_row) for r in range(9)]
    failed = [int((j + 1) * rate[r] + 1e-9) > int(j * rate[r] + 1e-9) for j in range(per_row) for r in range(9)]
    seen = [(j + r) % 2 == 0 for j in range(per_row) for r in range(9)]
    if per_row:
      ucur._eval_fail(env, "up", rows, failed, ucur.EVAL_FAIL_ALPHA, [cls] * len(rows), seen)
    step[0] += its * ucur.STEPS_PER_IT
    ucons.constraints._families(me, env, step[0])
    return ucur.eval_light(env, "up")[0]

  ucons.constraints._families(me, env, 0)
  ok("unit relax: no evaluation episodes, not judged", me.relax == {f: False for f in ucons.FAMILIES} and not me.ref
     and me.light["up"] == {"fail": None, "n": 0} and ucons.measure(env, "evalfail:up", scan="seen") is None)
  v = feed(0.2, 20)
  ok(f"unit relax: 180 light episodes, not judged yet ({v})", abs(v - 0.2) < 1e-6 and "up" not in me.ref)
  v = feed(0.2, 20)
  ref0 = me.ref.get("up")
  ok(f"unit relax: judged from 200 episodes, the reference starts at the reading ({v}, {ref0})",
     ref0 is not None and abs(ref0 - v) < 1e-9 and abs(v - 0.2) < 0.05 and not me.relax["up"])
  v = feed(0.6, 20)
  ok(f"unit relax: the fail rate rises -> up and stairs relax, down does not ({v} vs {me.ref['up']})",
     v > ref0 + 0.2 and abs(me.ref["up"] - ref0) < 0.001 and me.relax["up"] and me.relax["stairs"]
     and not me.relax["down"] and not me.relax["other"])
  feed(1.0, 40, cls=1)
  feed(1.0, 40, cls=2)
  ok("unit relax: the loaded robots' falls do not move the light reading",
     abs(ucur.eval_light(env, "up")[0] - v) < 1e-9 and ucur.eval_light(env, "up", "all", "mid")[0] == 1.0
     and ucur.eval_light(env, "up", "all", "heavy") == (1.0, 360) and env.u_eval_fail["up"] > v + 0.1)
  for _ in range(4):
    v = feed(0.2, 20)
  ok(f"unit relax: back near the reference -> restored ({v} vs {me.ref['up']})",
     v < me.ref["up"] + ucons.RELAX_GAP_OFF and not me.relax["up"] and not me.relax["stairs"])
  for _ in range(5):
    v = feed(0.0, 20)
  ref1 = me.ref["up"]
  for _ in range(50):   # 500 iterations at this reading: the reference follows it down by 1 - 1/e
    feed(0.0, 0, its=10)
  ref2 = me.ref["up"]
  ok(f"unit relax: the reference follows a lower reading in ~500 iterations ({ref1} -> {ref2}, reading {v})",
     0.55 < (ref1 - ref2) / (ref1 - v) < 0.70)
  for _ in range(5):
    v = feed(0.5, 20)
  for _ in range(50):   # 500 iterations: up by 1 - exp(-1/4) only, and relaxed all along
    feed(0.5, 0, its=10)
  ok(f"unit relax: and a higher one in ~2000 ({ref2} -> {me.ref['up']}, reading {v})",
     0.15 < (me.ref["up"] - ref2) / (v - ref2) < 0.30 and me.relax["up"])
  feed(0.0, 0, its=100000)
  ok("unit relax: a jump of the step counter counts as ten iterations", me.ref["up"] - ref2 < 0.3 * (v - ref2))
  # the survival items' reading: the light robots' worst step-height band, per scan mode
  env.u_eval_split = {}
  feed([0.0] * 5 + [0.5] * 2 + [1.0] * 2, 20)
  feed([1.0] * 9, 20, cls=1)
  ok(f"unit survive: per band {ucur.eval_bands(env, 'up', 'seen')}", ucur.eval_bands(env, "up", "seen") == [0.0, 0.0, 0.5, 1.0]
     and ucons.measure(env, "evalfail:up", scan="blind") == 1.0 and ucons.measure(env, "evalfail:down") is None)
  env.u_eval_split = {}
  feed([0.0] * 5 + [0.5] * 2 + [None] * 2, 0)
  ok("unit survive: a split of other rows is counted afresh", not ucur._split_fits({"light": {}}, 9)
     and ucur._split_fits(ucur._split_new(9), 9) and not ucur._split_fits(ucur._split_new(8), 9))

  # constraints.stop_creep: the creep on the acceptance's time base, on scripted stands. Every robot slides 3 cm in
  # the 2 s after its stand begins (step 50) and drifts 2 mm/s from then on:
  #   0  stands on                          -> counted 4-6 s into the stand, reading 2 mm/s x (2..4 s) = 4..8 mm
  #   1  pushed 3 s into the stand          -> the reference is retaken 2 s after the push, counted 4-6 s after it
  #   2  gets a command 3.5 s into it       -> never counted
  #   3  a new episode 5 s into it          -> the window ends there; counted again 4-6 s into the new stand
  #   4  as robot 0 with the full payload   -> counted by u_stop_creep, not by u_stop_creep_light
  dt, n, start = 0.02, 5, 50
  hold = NS(was_standing=torch.zeros(n, dtype=torch.bool), excused_since=torch.zeros(n, dtype=torch.long))
  pos = torch.zeros(n, 3)
  cenv = NS(num_envs=n, device="cpu", common_step_counter=0, step_dt=dt, u_payload_full=8.0, u_slope_cols=None,
            payload_kg=torch.tensor([0.0, 0.0, 0.0, 0.0, 8.0]), scene={"robot": NS(data=NS(root_link_pos_w=pos))},
            reward_manager=NS(get_term_cfg=lambda name: NS(params={"push_window": 2.0}, func=hold)))
  sc = ucons.stop_creep(None, cenv)
  since = torch.full((n,), start, dtype=torch.long)   # when each robot's stand began / it was last pushed
  seen: dict[int, list[tuple[float, float]]] = {i: [] for i in range(n)}   # robot -> (s into the stand, reading)
  light_n = all_n = 0
  for step in range(1, start + 600):
    tau = (step - start) * dt
    cenv.common_step_counter = step
    standing = torch.full((n,), step >= start)
    standing[2] = start <= step < start + int(3.5 / dt)
    if step == start + int(3.0 / dt):
      since[1] = step          # the push (the hold term stamps its clock)
    if step == start + int(5.0 / dt):
      sc.reset(torch.tensor([3]))
      since[3] = step          # the hold term after its own reset: a new stand
    hold.was_standing, hold.excused_since = standing, since.clone()
    pos[:, 0] = 0.03 * min(max(tau, 0.0), 2.0) / 2.0 + 0.002 * max(tau - 2.0, 0.0)
    sc(cenv)
    value, mask = ucons.CUSTOM["u_stop_creep"](cenv)
    for i in torch.nonzero(mask).flatten().tolist():
      seen[i].append((tau, float(value[i])))
    all_n += int(mask.sum())
    light_n += int(ucons.CUSTOM["u_stop_creep_light"](cenv)[1].sum())
    if step == start + int(5.0 / dt) - 1:
      mid = ucons.measure(cenv, "u_stop_creep_light", "mean")   # 5 s in: robots 0 and 3, 3 s past their reference
  span = lambda i: (round(seen[i][0][0], 2), round(seen[i][-1][0], 2)) if seen[i] else None   # noqa: E731
  peak = lambda i: max(v for _, v in seen[i])   # noqa: E731
  ok(f"unit creep: counted 4-6 s into the stand, the slide before the 2 s mark left out ({span(0)}, peak "
     f"{1000 * peak(0):.1f} mm of a 3.8 cm path)", span(0) == (4.0, 6.0) and abs(peak(0) - 0.008) < 2e-4
     and abs(seen[0][0][1] - 0.004) < 2e-4)
  ok(f"unit creep: a push restarts the 2 s ({span(1)})", span(1) == (7.0, 9.0) and abs(peak(1) - 0.008) < 2e-4)
  ok("unit creep: a stand that ends before the window is not counted", not seen[2])
  ok(f"unit creep: a new episode ends the window and starts another ({span(3)})",
     [round(t, 2) for t, _ in seen[3]][:2] == [4.0, 4.02] and round(max(t for t, _ in seen[3] if t < 6.0), 2) == 4.98
     and round(min(t for t, _ in seen[3] if t > 6.0), 2) == 9.0 and span(3) == (4.0, 11.0)
     and abs(max(v for t, v in seen[3] if t < 6.0) - 0.006) < 2e-4 and abs(peak(3) - 0.008) < 2e-4)
  ok(f"unit creep: the light measure leaves the loaded robot out ({light_n} of {all_n} readings; mean at 5 s "
     f"{1000 * mid:.1f} mm)", seen[4] == seen[0] and all_n - light_n == len(seen[4]) and abs(mid - 0.006) < 2e-4)
  # the march's drift: the same tracker under its own key, chained 5 s windows. Every robot drifts 6 mm/s from the
  # start of its march (step 50):
  #   0  marches on                         -> counted 6-7, 11-12, 16-17 s into the march (the last second of each
  #                                            5 s window after the 2 s), reading 6 mm/s x (4..5 s) = 24..30 mm
  #   1  pushed 6.5 s into the march        -> the first window ends there; counted again 12.5-13.5 s into it
  #   2  gets a command 5 s into it         -> never counted
  #   3  as robot 0 with the full payload   -> counted by u_march_drift, not by u_march_drift_light
  n = 4
  mhold = NS(was_standing=torch.zeros(n, dtype=torch.bool), excused_since=torch.zeros(n, dtype=torch.long))
  mpos = torch.zeros(n, 3)
  menv = NS(num_envs=n, device="cpu", common_step_counter=0, step_dt=dt, u_payload_full=8.0, u_slope_cols=None,
            payload_kg=torch.tensor([0.0, 0.0, 0.0, 8.0]), scene={"robot": NS(data=NS(root_link_pos_w=mpos))},
            reward_manager=NS(get_term_cfg=lambda name: NS(params={"push_window": 2.0}, func=mhold)))
  md = ucons.stop_creep(NS(params={"key": "u_march_drift"}), menv)
  since = torch.full((n,), start, dtype=torch.long)
  seen = {i: [] for i in range(n)}
  light_n = all_n = 0
  for step in range(1, start + 1000):
    tau = (step - start) * dt
    menv.common_step_counter = step
    marching = torch.full((n,), step >= start)
    marching[2] = start <= step < start + int(5.0 / dt)
    if step == start + int(6.5 / dt):
      since[1] = step
    mhold.was_standing, mhold.excused_since = marching, since.clone()
    mpos[:, 0] = 0.006 * max(tau, 0.0)
    md(menv, term="march_position_hold", settle_s=2.0, window_s=5.0, min_window_s=4.0, repeat=True, key="u_march_drift")
    value, mask = ucons.CUSTOM["u_march_drift"](menv)
    for i in torch.nonzero(mask).flatten().tolist():
      seen[i].append((round(tau, 2), float(value[i])))
    all_n += int(mask.sum())
    light_n += int(ucons.CUSTOM["u_march_drift_light"](menv)[1].sum())
  times = lambda i: [t for t, _ in seen[i]]   # noqa: E731
  edges = lambda i: [t for j, t in enumerate(times(i)) if j == 0 or round(t - times(i)[j - 1], 2) > 0.02]   # noqa: E731
  ok(f"unit march: its own key; chained 5 s windows, each counted over its last second (from {edges(0)}, "
     f"{len(seen[0])} readings of {1000 * min(v for _, v in seen[0]):.0f}-{1000 * max(v for _, v in seen[0]):.0f} mm)",
     menv.u_march_drift is md and not hasattr(menv, "u_stop_creep") and edges(0) == [6.0, 11.0, 16.0]
     and len(seen[0]) == 150 and times(0)[-1] == 16.98 and abs(min(v for _, v in seen[0]) - 0.024) < 2e-4
     and abs(max(v for _, v in seen[0]) - 0.02988) < 2e-4)
  ok(f"unit march: a push ends the window and restarts the 2 s (from {edges(1)})",
     edges(1) == [6.0, 12.5, 17.5] and 6.48 in times(1) and 6.5 not in times(1) and 13.48 in times(1)
     and abs(max(v for _, v in seen[1]) - 0.02988) < 2e-4)
  ok("unit march: a march that ends before a window's last second is not counted", not seen[2])
  ok(f"unit march: the light measure leaves the loaded robot out ({light_n} of {all_n} readings)",
     seen[3] == seen[0] and all_n - light_n == len(seen[3]))
  # constraints.march_reset: the hips' reset over a march. Hips 0 / 3 / 6 / 9 of 12 joints; the worst one starts
  # `h0` off its default and reaches `h1` 3 s into the march (step 50):
  #   0  0.14 -> 0.02 rad                   -> counted 4-5 s into the march, reading 0.02
  #   1  0.01 -> 0.01 (no splay)            -> never counted (nothing to set back)
  #   2  as 0, a command 3 s into it        -> never counted
  #   3  as 0, a new episode 4.5 s into it  -> counted 4-4.5 s; the next march begins at 0.02: not counted
  #   4  as 0 with the full payload         -> counted by u_march_hip, not by u_march_hip_light
  n, hips = 5, [0, 3, 6, 9]
  h0 = torch.tensor([0.14, 0.01, 0.14, 0.14, 0.14])
  h1 = torch.tensor([0.02, 0.01, 0.02, 0.02, 0.02])
  jpos, jdef = torch.zeros(n, 12), torch.full((n, 12), 0.3)
  twist = NS(march=torch.zeros(n, dtype=torch.bool))
  henv = NS(num_envs=n, device="cpu", common_step_counter=0, step_dt=dt, u_payload_full=8.0, u_slope_cols=None,
            payload_kg=torch.tensor([0.0, 0.0, 0.0, 0.0, 8.0]),
            scene={"robot": NS(data=NS(joint_pos=jpos, default_joint_pos=jdef))},
            command_manager=NS(get_term=lambda name: twist),
            reward_manager=NS(get_term_cfg=lambda name: NS(params={"asset_cfg": NS(name="robot", joint_ids=hips)})))
  mr = ucons.march_reset(None, henv)
  seen = {i: [] for i in range(n)}
  light_n = all_n = 0
  for step in range(1, start + 500):
    tau = (step - start) * dt
    henv.common_step_counter = step
    marching = torch.full((n,), step >= start)
    marching[2] = start <= step < start + int(3.0 / dt)
    if step == start + int(4.5 / dt):
      mr.reset(torch.tensor([3]))
    twist.march = marching
    off = h0 + (h1 - h0) * min(max(tau, 0.0), 3.0) / 3.0
    jpos[:] = jdef
    jpos[:, hips] = jdef[:, hips] + 0.5 * off.unsqueeze(1)
    jpos[:, 6] = jdef[:, 6] - off          # the worst hip (to the other side)
    jpos[:, 1] = jdef[:, 1] + 1.0          # not a hip: left out
    mr(henv)
    value, mask = ucons.CUSTOM["u_march_hip"](henv)
    for i in torch.nonzero(mask).flatten().tolist():
      seen[i].append((round(tau, 2), float(value[i])))
    all_n += int(mask.sum())
    light_n += int(ucons.CUSTOM["u_march_hip_light"](henv)[1].sum())
  span = lambda i: (seen[i][0][0], seen[i][-1][0]) if seen[i] else None   # noqa: E731
  ok(f"unit hips: a march from a splayed stance is read 4-5 s in, the worst hip's distance ({span(0)}, "
     f"{seen[0][0][1]:.3f} rad)", span(0) == (4.0, 5.0) and all(abs(v - 0.02) < 1e-6 for _, v in seen[0]))
  ok("unit hips: a march from the default stance and one that ends early are not counted", not seen[1] and not seen[2])
  ok(f"unit hips: a new episode ends the reading; its march from 0.02 rad is not counted ({span(3)})",
     span(3) == (4.0, 4.48))
  ok(f"unit hips: the light measure leaves the loaded robot out ({light_n} of {all_n} readings)",
     seen[4] == seen[0] and all_n - light_n == len(seen[4]))
  # constraints.load: an item measured another way than in the checkpoint starts afresh, the others are kept; an
  # item split off another one starts at that one's multiplier
  items = constraint_items("legged")
  old_state = {"scale": {c["name"]: 2.0 for c in items if c["name"] != "march_hip"} | {"creep": 3.0, "march": 2.5},
               "value": {c["name"]: 0.5 for c in items},
               "values": {f"{c['name']}@{m}": 0.5 for c in items for m in ucons.MODES},
               "count": {f"{c['name']}@{m}": 900 for c in items for m in ucons.MODES}, "relax": {}, "ref": {}}
  cons = ucons.constraints(NS(params={"items": items}), NS())
  cons.load(dict(old_state))
  ok(f"unit load: a checkpoint from before the creep and march measures -- they start afresh, the rest is kept "
     f"({cons.redefined})", cons.redefined == ["creep", "march"] and not {"creep", "march"} & {*cons.scale, *cons.value}
     and not [k for k in {*cons.values, *cons.count} if k.startswith(("creep@", "march@"))] and cons.scale["spin"] == 2.0
     and cons.count["spin@seen"] == 900 and cons.values["creep_total@blind"] == 0.5)
  ok(f"unit load: march_hip, split off march, starts at march's multiplier then ({cons.scale.get('march_hip')})",
     cons.scale.get("march_hip") == 2.5)
  # (2026-11-13) the items the flat protocols measure anew keep their multipliers (constraints.CARRY_SCALE)
  ok(f"unit load: the items now read from the flat scripts keep their multipliers, their readings start afresh "
     f"({cons.remeasured})", sorted(cons.remeasured) == ["clock", "side_drift", "straight_drift"]
     and all(cons.scale[k] == 2.0 and k not in cons.value and f"{k}@seen" not in cons.count for k in cons.remeasured))
  cons2 = ucons.constraints(NS(params={"items": items}), NS())
  cons2.load(dict(old_state) | {"sig": cons.state()["sig"]})
  ok("unit load: a checkpoint of the same measures is taken whole", not cons2.redefined and cons2.scale["creep"] == 3.0
     and cons2.count["creep@seen"] == 900 and cons2.scale["march"] == 2.5 and "march_hip" not in cons2.scale)
  cons3 = ucons.constraints(NS(params={"items": items}), NS())
  cons3.load(dict(old_state) | {"sig": cons.state()["sig"] | {"march": "u_stand_hold_light|mean"}})
  ok(f"unit load: any item whose signature differs ({cons3.redefined})", cons3.redefined == ["march"]
     and "march" not in cons3.scale and cons3.scale["creep"] == 3.0 and "march_hip" not in cons3.scale)

  # teacher_gate.py: a gate-only item at the bar of the code on disk, the others at the bar the training logged
  from torch.utils.tensorboard import SummaryWriter
  from src.tasks.unified.teacher_env import SURVIVE_BAR

  def gate(name: str, survive: float, logged_bar: float, absent=(), bar_only=(), args=(), flat_until=None) -> str:
    run = out / "gate" / "go2_u_teacher" / name
    w = SummaryWriter(str(run))
    for it in range(0, 601):
      w.add_scalar("Curriculum/u_progress/p", 1.0, it)
      w.add_scalar("Curriculum/u_progress/q", 1.0, it)
      for c in constraint_items("legged"):
        if not c["enforce"] or c["name"] in absent:
          continue
        tag = f"Curriculum/u_constraints/{c['name']}/"
        good = c["target"] * (1.1 if c["sense"] == "min" else 0.9)
        bad = c["target"] * (0.5 if c["sense"] == "min" else 2.0)
        for m in () if c["name"] in bar_only else c["modes"]:
          w.add_scalar(tag + f"value_{m}", survive if c["name"] == "survive_up" and m == "blind" else
                       min(good, survive) if c["name"].startswith("survive") else
                       bad if flat_until is not None and it > flat_until and c["name"] == "clock" else good, it)
        w.add_scalar(tag + "bar", logged_bar if c["name"].startswith("survive") else c["target"], it)
    w.close()
    r = subprocess.run([sys.executable, str(ROOT / "scripts/unified/teacher_gate.py"), str(run), *args],
                       capture_output=True, text=True, cwd=str(ROOT))
    return r.stdout.strip() + r.stderr.strip()[-300:]

  # (the logged bar is ignored for a gate-only item: 0.99 would pass anything, 0.0 nothing)
  g = gate("pass", SURVIVE_BAR - 0.02, 0.0)
  # (the first checkpoint with a full window behind it: 500)
  ok(f"unit gate: under the bar on disk passes ({g[:40]})", g.startswith("PASS 500 ") and "survive_up_blind" in g)
  g = gate("miss", SURVIVE_BAR + 0.05, 0.99)
  ok(f"unit gate: over it does not ({g})", g.startswith("NOT YET 600 ") and "survive_up_blind" in g
     and "survive_up_seen" not in g and "survive_down" not in g and g.count(" vs ") == 1)
  # (2026-11-13) a run whose code had no such item (from before the flat protocols, or U_FLAT_SCRIPTS=0)
  from src.tasks.unified.teacher_env import FLAT_NEW
  g = gate("before_flat", SURVIVE_BAR - 0.02, 0.0, absent=FLAT_NEW)
  ok(f"unit gate: a run that logged nothing of an item is judged on the items it trained with ({g[-110:]})",
     g.startswith("PASS 500 ") and "head_tail" not in g.split("teacher_gate:")[0]
     and "not judged: " + " ".join(FLAT_NEW) in g)
  g = gate("no_reading", SURVIVE_BAR - 0.02, 0.0, bar_only=("stance",))
  ok(f"unit gate: an item the run has (its bar is logged) but never read stays unmeasured ({g})",
     g.startswith("NOT YET 600 ") and "stance unmeasured" in g and "not judged" not in g)
  # (2026-11-13) --pre: the stairs-quality items are read, not judged; the newest checkpoint that passes
  g = gate("pre", SURVIVE_BAR + 0.05, 0.99, args=("--pre",))
  ok(f"unit gate --pre: a stairs item over its bar does not hold the pre-pass, the newest checkpoint ({g[:60]})",
     g.startswith("PRE-PASS 600 ") and "survive" not in g and "kicks" not in g and "clock_seen" in g)
  g = gate("pre_flat", SURVIVE_BAR - 0.02, 0.0, bar_only=("stance",), args=("--pre",))
  ok(f"unit gate --pre: a flat item never read still holds it ({g})",
     g.startswith("NOT YET 600 ") and "stance unmeasured" in g)
  g = gate("pre_newest", SURVIVE_BAR - 0.02, 0.0, args=("--pre", "--hold", "200"), flat_until=450)
  ok(f"unit gate --pre: the newest checkpoint whose window passes, not the last one ({g[:40]})",
     g.startswith("PRE-PASS 400 "))
  wheel_counter(ok)
  meter_eval_mask(ok)
  sticky_payload_units(ok)
  loaded_kick_units(ok)
  flat_units(ok, out)


def flat_units(ok, out: Path) -> None:
  """flat.py, the flat protocols (2026-11-13). The command bins and the statistics; the items with the scripts on
  and off (U_FLAT_SCRIPTS) and a checkpoint from before them; then the tracker on a kinematic stand-in: every robot
  follows its command exactly but for planted errors, so each reading has a known value -- forward speed 90% of the
  command, a 0.01 rad/s turn under a command without one (a 10 s line: 5.73 deg, a drift of tan 0.05), 5 mm/s of
  creep under a zero command (ten times that on the heavy robots), the leg joints 0.12 rad off at the most and 0.07
  on average, the splayed robots' hips 0.14 rad out until a march brings them to 0.02, 85% of the feet in step.
  Then the scripts given up, the checkpoint state, the fault path and the records' tags."""
  import collections
  import math

  from src.tasks.unified import flat as uflat
  from src.tasks.unified.teacher_env import FLAT_NEW

  # -- the bins and the statistics
  bins = dict(collections.Counter(uflat.flat_bin(c) for c in uflat.FLAT_TRACK_CMDS))
  ok(f"unit flat: the acceptance's 44 tracking commands by bin ({bins})", bins == {
    "fwd": 4, "fast": 1, "back": 2, "side": 5, "diag": 2, "turn": 8, "arc": 8, "arc_tight": 2, "fast_arc": 6,
    "back_arc": 2, "back_arc_fast": 2, "mix": 2})
  ok("unit flat: the stops' bins (one per pre-stop command) and the lines'",
     [uflat.flat_bin(c) for c in uflat.FLAT_STOP_CMDS] == ["fwd", "back", "side", "turn", "arc", "fast"]
     and sorted({uflat.flat_line_bin(c) for c in uflat.FLAT_LINE_CMDS}) == ["back", "fwd", "fwd_slow", "side"])
  vals = [0.3, -1.2, 0.7, 2.5, 0.1, -0.4, 1.9, 0.0, 0.6, -2.2, 1.1]
  t = torch.tensor(vals, dtype=torch.float64)
  q = lambda p: float(torch.quantile(t, p))   # noqa: E731
  near = lambda a, b, tol=1e-9: a is not None and abs(a - b) < tol   # noqa: E731
  ok("unit flat: the statistics (mean, |mean - 1|, the quantiles as torch.quantile, both tails, |median|)",
     near(uflat.flat_stat(vals, "mean"), float(t.mean())) and near(uflat.flat_stat(vals, "dev1"), abs(float(t.mean()) - 1))
     and all(near(uflat.flat_stat(vals, f"p{p}"), q(p / 100)) for p in (10, 50, 90))
     and near(uflat.flat_stat(vals, "tail"), max(abs(q(0.1)), abs(q(0.9))))
     and near(uflat.flat_stat(vals, "absmed"), abs(q(0.5))))
  pairs = [(100 * j, float(j)) for j in range(10)]   # (step, value): steps 0 .. 900
  ok("unit flat: a bin's samples -- the fresh ones; else the newest `need`, none older than max_age; else none",
     uflat.flat_recent(pairs, 1000, 3, 400, 5000) == [6.0, 7.0, 8.0, 9.0]
     and uflat.flat_recent(pairs, 1000, 6, 400, 5000) == [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
     and uflat.flat_recent(pairs, 1000, 6, 400, 500) is None and uflat.flat_recent(pairs, 1000, 11, 400, 5000) is None)
  fits = lambda yaw, cmd, steps=500: uflat.flat_path_fits(0.0, 0.0, yaw, cmd, steps, 0.02, 0.0, 2.0, -50.0, 50.0)  # noqa: E731
  ok("unit flat: a 10 s line fits the strip along it and not across it, a sideways one the other way round",
     fits(0.0, (1.0, 0.0, 0.0)) and not fits(0.5 * math.pi, (1.0, 0.0, 0.0)) and fits(0.5 * math.pi, (0.0, 0.6, 0.0))
     and not fits(0.0, (0.0, 0.6, 0.0)) and fits(1.0, (0.0, 0.0, 1.5)) and not fits(0.0, (1.0, 0.0, 0.2), 1000))

  # -- the items, scripts on and off
  def items_for(value: str | None) -> dict:
    old = os.environ.pop("U_FLAT_SCRIPTS", None)
    if value is not None:
      os.environ["U_FLAT_SCRIPTS"] = value
    try:
      return {kind: {c["name"]: c for c in constraint_items(kind)} for kind in ("legged", "wheeled")}
    finally:
      os.environ.pop("U_FLAT_SCRIPTS", None)
      if old is not None:
        os.environ["U_FLAT_SCRIPTS"] = old

  on, off = items_for(None), items_for("0")
  moved = ("clock", "side_drift", "straight_drift", "creep", "march_hip")
  copies = {"clock_side": "clock", "side_drift_p90": "side_drift", "straight_drift_p90": "straight_drift",
            "creep_pooled": "creep", "march_hip_in": "march_hip"}
  for kind in ("legged", "wheeled"):
    a, b = on[kind], off[kind]
    ok(f"unit flat items ({kind}): U_FLAT_SCRIPTS=0 gives the items as they were -- no flat: metric, none of the new "
       f"names, the five measured as constraints.CARRY_SCALE lists",
       not any(c["metric"].startswith("flat:") for c in b.values()) and not {*FLAT_NEW, *copies} & set(b)
       and all(ucons.item_sig(b[k]) in ucons.CARRY_SCALE[k] for k in moved))
    ok(f"unit flat items ({kind}): on -- the five read the scripts, their old readings stay as logged-only copies, "
       f"the six new items are enforced (cap 2), every other item is the same",
       all(a[k]["metric"].startswith("flat:") and a[k]["enforce"] for k in moved)
       and all(a[c]["metric"] == b[k]["metric"] and a[c].get("stat") == b[k].get("stat") and not a[c]["enforce"]
               and not a[c]["terms"] for c, k in copies.items())
       and all(a[k]["enforce"] and a[k]["metric"].startswith("flat:") and a[k]["max_scale"] == 2.0 for k in FLAT_NEW)
       and {k: c for k, c in a.items() if k not in (*moved, *copies, *FLAT_NEW)}
       == {k: c for k, c in b.items() if k not in moved})

  # -- a checkpoint from before the scripts (its signatures: the old measures), and one of them read without them
  items = constraint_items("legged")
  before = [c["name"] for c in items if c["name"] not in (*FLAT_NEW, *copies)]
  sig = {c["name"]: ucons.item_sig(c) for c in items} | {k: next(iter(ucons.CARRY_SCALE[k])) for k in moved}
  state = {"scale": {k: 2.0 for k in before} | {"creep": 3.0}, "value": {k: 0.5 for k in before},
           "values": {f"{k}@{m}": 0.5 for k in before for m in ucons.MODES},
           "count": {f"{k}@{m}": 900 for k in before for m in ucons.MODES}, "relax": {}, "ref": {},
           "sig": {k: sig[k] for k in before}}
  cons = ucons.constraints(NS(params={"items": items}), NS())
  cons.load(dict(state))
  ok(f"unit flat load: a checkpoint from before the scripts -- the five re-measured items keep their multipliers and "
     f"read afresh ({cons.remeasured}), nothing else starts over ({cons.redefined}), the new items have none yet",
     sorted(cons.remeasured) == sorted(moved) and not cons.redefined and cons.scale["creep"] == 3.0
     and cons.scale["clock"] == 2.0 and not set(moved) & set(cons.value)
     and not [k for k in {*cons.values, *cons.count} if k.split("@")[0] in moved] and cons.count["spin@seen"] == 900
     and not set(FLAT_NEW) & set(cons.scale))
  items_off = list(off["legged"].values())
  cons_off = ucons.constraints(NS(params={"items": items_off}), NS())
  cons_off.load({**cons.state(), "flat": {"step": 5, "buf": {}, "tally": {}, "drops": {}}})
  ok(f"unit flat load: the other way (a scripts-on checkpoint under U_FLAT_SCRIPTS=0) the five start afresh "
     f"({cons_off.redefined})", sorted(cons_off.redefined) == sorted(moved) and not cons_off.remeasured
     and not set(moved) & set(cons_off.scale) and cons_off.scale["spin"] == 2.0)

  # -- the descent switches (2026-11-13): stop_feet by the direction faced, the stairs speed over the ground
  def descent_unit() -> None:
    def items_env(kind: str = "legged", **env: str) -> dict:
      keys = ("U_STOP_FEET_DIR", "U_STAIR_SPEED_H", "U_Q_SCALE_MAX")
      old = {k: os.environ.pop(k, None) for k in keys}
      os.environ.update(env)
      try:
        found = constraint_items(kind)
        if "U_Q_SCALE_MAX" in env:
          from src.tasks.unified import teacher_env as te
          te._probe_switches(NS(rewards={}, events={}), found)
        return {c["name"]: c for c in found}
      finally:
        for k in keys:
          os.environ.pop(k, None)
          if old[k] is not None:
            os.environ[k] = old[k]

    for kind in ("legged", "wheeled"):
      base, split = items_env(kind), items_env(kind, U_STOP_FEET_DIR="3", U_Q_SCALE_MAX="1")
      two = [split.get(f"stop_feet_{d}", {}) for d in ("up", "down")]
      ok(f"unit descent ({kind}): U_STOP_FEET_DIR=3 -- stop_feet becomes stop_feet_up / stop_feet_down in its place "
         f"(the same measure by `facing`, the same cost term, each on its direction's fail rate), they start at 3 with "
         f"their own cap under U_Q_SCALE_MAX=1; every other item as without the switch but for that cap",
         [n for n in split if not n.startswith("stop_feet")] == [n for n in base if n != "stop_feet"]
         and list(split).index("stop_feet_up") == list(base).index("stop_feet") and "stop_feet" not in split
         and all(c.get("metric") == "u_stairs_stop_feet" and c["params"] == {"facing": d} and c["family"] == d
                 and c["terms"] == ["feet_stand_uneven"] and c["init_scale"] == 3.0 and c["max_scale"] == 3.0
                 and c["target"] == 0.9 and c["by"] == "q" for c, d in zip(two, ("up", "down")))
         and all(split[n]["max_scale"] == 1.0 for n in split if split[n]["by"] == "q" and not n.startswith("stop_feet"))
         and {n: {k: v for k, v in c.items() if k not in ("max_scale", "start_scale")} for n, c in split.items()
              if not n.startswith("stop_feet")}
         == {n: {k: v for k, v in c.items() if k not in ("max_scale", "start_scale")} for n, c in base.items()
             if n != "stop_feet"})
      speed = items_env(kind, U_STAIR_SPEED_H="1")
      ok(f"unit descent ({kind}): U_STAIR_SPEED_H=1 -- descent and climb read u_stair_speed_h, their old measure is "
         f"the one constraints.CARRY_SCALE lists, every other item is the same",
         all(speed[n]["metric"] == "u_stair_speed_h" and ucons.item_sig(base[n]) in ucons.CARRY_SCALE[n]
             and {**speed[n], "metric": base[n]["metric"]} == base[n] for n in ("descent", "climb"))
         and {n: c for n, c in speed.items() if n not in ("descent", "climb")}
         == {n: c for n, c in base.items() if n not in ("descent", "climb")})
    base = items_env()
    state = {"scale": {n: 2.0 for n in base} | {"stop_feet": 0.5}, "value": {n: 0.5 for n in base},
             "values": {f"{n}@{m}": 0.5 for n in base for m in ucons.MODES},
             "count": {f"{n}@{m}": 900 for n in base for m in ucons.MODES}, "relax": {}, "ref": {},
             "sig": {n: ucons.item_sig(c) for n, c in base.items()}}
    cons = ucons.constraints(NS(params={"items": list(items_env(U_STOP_FEET_DIR="3").values())}), NS())
    cons.load(dict(state))
    ok(f"unit descent load: a checkpoint from before the split -- nothing starts over ({cons.redefined}, "
       f"{cons.remeasured}), the two new items have no multiplier yet (the first update gives them init_scale)",
       not cons.redefined and not cons.remeasured and not {"stop_feet_up", "stop_feet_down"} & set(cons.scale)
       and cons.scale["kicks"] == 2.0 and cons.count["descent@seen"] == 900)
    cons = ucons.constraints(NS(params={"items": list(items_env(U_STAIR_SPEED_H="1").values())}), NS())
    cons.load(dict(state))
    ok(f"unit descent load: a checkpoint measured in the body frame under U_STAIR_SPEED_H=1 -- descent and climb keep "
       f"their multipliers and read afresh ({cons.remeasured}), nothing else starts over ({cons.redefined})",
       sorted(cons.remeasured) == ["climb", "descent"] and not cons.redefined and cons.scale["descent"] == 2.0
       and cons.scale["climb"] == 2.0 and "descent" not in cons.value and "descent@seen" not in cons.count
       and cons.count["kicks@seen"] == 900)
    # the measures on a stand-in: 8 robots, each stopped on a stairs column -- nose up / level / nose down / moving
    gx = torch.tensor([-0.4, -0.4, 0.0, 0.0, 0.4, 0.4, 0.4, 0.4])
    found = torch.ones(8, 4)
    found[1, 0] = found[5, 2] = found[6, 3] = 0.0
    vel = torch.zeros(8, 3)
    vel[7, 0] = 0.5
    robot = NS(data=NS(projected_gravity_b=torch.stack([gx, torch.zeros(8), -torch.ones(8)], dim=1),
                       root_link_lin_vel_b=vel))
    env = NS(scene={"feet_ground_contact": NS(data=NS(found=found)), "robot": robot},
             command_manager=NS(get_command=lambda name: torch.zeros(8, 3),
                                get_term=lambda name: NS(is_stair_env=torch.ones(8, dtype=torch.bool))))
    read = {f: ucons._stairs_stop_feet(env, facing=f) for f in (None, "up", "down")}
    share = {f: float(v[m].mean()) for f, (v, m) in read.items()}
    ok(f"unit descent measure: stop_feet over the stopped robots -- pooled {share[None]:.3f} (7 robots), nose up "
       f"{share['up']:.2f} (2), nose down {share['down']:.3f} (3; the moving one is not counted)",
       [int(m.sum()) for _, m in read.values()] == [7, 2, 3] and abs(share[None] - 4 / 7) < 1e-6
       and share["up"] == 0.5 and abs(share["down"] - 1 / 3) < 1e-6)
    half = math.radians(30.0) / 2   # pitched 30 deg nose down and turned 90 deg, 1 m/s along the body's x
    qp = torch.tensor([math.cos(half), 0.0, math.sin(half), 0.0])
    qy = torch.tensor([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)])
    from mjlab.utils.lab_api.math import quat_apply, quat_mul
    q = quat_mul(qy, qp).reshape(1, 4)
    v_w = quat_apply(q, torch.tensor([[1.0, 0.0, 0.0]]))
    env = NS(scene={"robot": NS(data=NS(root_link_quat_w=q, root_link_lin_vel_w=v_w))})
    v_h = ucons._yaw_frame_velocity(env)[0]
    ok(f"unit descent measure: 1 m/s along a body pitched 30 deg nose down is {float(v_h[0]):.3f} m/s over the ground "
       f"ahead and {float(v_h[2]):.3f} m/s up, whatever the heading",
       abs(float(v_h[0]) - math.cos(math.radians(30.0))) < 1e-5 and abs(float(v_h[1])) < 1e-5
       and abs(float(v_h[2]) + 0.5) < 1e-5)

  descent_unit()

  # -- the tracker on the kinematic stand-in
  n, k_eval, dt = 256, 192, 0.02
  i = torch.arange(n)
  on_flat = (i % 8) != 7                              # one robot in eight on another ground
  kg = torch.tensor([0.0, 1.6, 4.0, 6.0])[i % 4]      # light, light, in between, heavy (the limit: 8 kg)
  heavy, mid = (i % 4) == 3, (i % 4) == 2
  splay = (i % 16) == 1                               # the hips out at the episode's start
  gen = torch.Generator().manual_seed(0)
  term = NS(vel_command_b=torch.zeros(n, 3), time_left=torch.zeros(n), march=torch.zeros(n, dtype=torch.bool),
            cfg=NS(ranges=NS(lin_vel_x=(-1.0, 1.5), lin_vel_y=(-0.6, 0.6), ang_vel_z=(-1.5, 1.5))))
  env = NS(num_envs=n, device="cpu", common_step_counter=0, step_dt=dt, max_episode_length=1000,
           episode_length_buf=torch.zeros(n, dtype=torch.long), u_eval_mask=i < k_eval, payload_kg=kg.reshape(n, 1),
           u_payload_full=8.0, u_scan_on=(i % 2).float(), command_manager=NS(get_term=lambda name: term))
  st = NS(pos=torch.zeros(n, 2), yaw=torch.zeros(n), vel=torch.zeros(n, 3), hip=torch.zeros(n), broken=False)

  class Tracker(uflat.flat_measure):
    def _kin(self, env):
      if st.broken:
        raise ValueError("planted")
      return {"pos": st.pos, "yaw": st.yaw, "vel": st.vel, "hip": st.hip, "leg_max": torch.full((n,), 0.12),
              "leg_mean": torch.full((n,), 0.07)}

    def _flat_mask(self, env):
      return on_flat

    def _bounds(self, env):
      return (4.0, -60.0, 60.0)

    def _centre(self, env):
      return torch.zeros(n)

    def _gait_value(self, env):
      return torch.full((n,), 0.85)

    def _gait_counts(self, env, cmd):
      return abs(cmd[0]) <= 1.0

  params = {"min_n": 4, "min_q": 6, "seed": 0}
  tr = Tracker(NS(params=params), env)
  bad = {"training robot": 0, "other ground": 0, "payload in between": 0, "command": 0}

  def reset(ids) -> None:
    u = torch.rand(len(ids), generator=gen)
    anywhere = 6.28 * torch.rand(len(ids), generator=gen) - 3.14
    st.pos[ids] = 0.0
    # (flat_reset_yaw's part in the environment: most start along the strip or across it)
    st.yaw[ids] = torch.where(u < 0.3, 0.0, torch.where(u < 0.6, math.pi, torch.where(
      u < 0.7, 0.5 * math.pi, torch.where(u < 0.8, -0.5 * math.pi, anywhere))))
    st.hip[ids] = torch.where(splay[ids], 0.14, 0.05)
    env.episode_length_buf[ids] = 0
    term.time_left[ids] = 0.0
    term.march[ids] = False
    tr.reset(ids)

  def step() -> None:
    cmd = term.vel_command_b
    zero = cmd.abs().amax(dim=1) < 1e-6
    vel = cmd * torch.tensor([0.9, 1.0, 1.0])
    vel[:, 2] += torch.where(~zero & (cmd[:, 2].abs() < 1e-6), 0.01, 0.0)
    c, s = torch.cos(st.yaw), torch.sin(st.yaw)
    creep = torch.where(zero & ~term.march, torch.where(heavy, 0.05, 0.005), 0.0)
    st.pos = st.pos + dt * torch.stack([c * vel[:, 0] - s * vel[:, 1] + creep, s * vel[:, 0] + c * vel[:, 1]], dim=1)
    st.yaw = st.yaw + dt * vel[:, 2]
    st.vel = vel
    st.hip = torch.where(term.march, (st.hip - 0.002).clamp(min=0.02), st.hip)
    env.episode_length_buf += 1
    env.common_step_counter += 1
    tr(env)
    bad["training robot"] += int((tr.on & ~env.u_eval_mask).sum())
    bad["other ground"] += int((tr.on & ~on_flat).sum())
    bad["payload in between"] += int((tr.on & mid).sum())
    bad["command"] += int(((term.vel_command_b - tr.cmd).abs().amax(dim=1)[tr.on] > 1e-6).sum())
    done = (env.episode_length_buf >= 1000).nonzero().flatten()
    if len(done):
      reset(done)
    # the command term: a drawn command for 3 s when the time is up (a script holds the time at 1 s)
    term.time_left -= dt
    new = (term.time_left <= 0.0).nonzero().flatten()
    if len(new):
      term.vel_command_b[new] = (torch.rand(len(new), 3, generator=gen) - 0.5) * torch.tensor([1.0, 0.6, 1.0])
      term.time_left[new] = 3.0
      term.march[new] = False

  reset(i)
  step()
  light = env.u_eval_mask & on_flat & ~heavy & ~mid
  n_light, n_heavy = int((tr.on & light).sum()), int((tr.on & heavy).sum())
  ok(f"unit flat scripts: at the first step {n_light} of the {int(light.sum())} light evaluation robots on the flat "
     f"column are under a script (share 0.7), {n_heavy} of the {int((env.u_eval_mask & on_flat & heavy).sum())} heavy "
     f"ones (0.4)", 0.55 * 96 <= n_light <= 0.85 * 96 and 0.15 * 24 <= n_heavy <= 0.65 * 24)
  for _ in range(4000):
    step()
  ok(f"unit flat scripts: only light or heavy evaluation robots on the flat column are ever under a script, and "
     f"their command is the script's ({bad})", not any(bad.values()))
  ok(f"unit flat scripts: over four episodes no script runs into an episode's end or leaves the strip ({tr.drops}); "
     f"every script kind finished ({sorted({g for g, _, _ in tr.tally})})", not tr.drops
     and {g for g, _, _ in tr.tally} == {"line", "march", "stop", "track"})
  rd = tr.read
  speed_lin, speed_yaw = rd("speed_lin", "dev1"), rd("speed_yaw", "dev1")
  ok(f"unit flat readings: the forward speed at 90% of the command reads 0.1 off on the worst linear bin "
     f"({speed_lin}), the turn rate none ({speed_yaw})", near(speed_lin, 0.1, 1e-3) and near(speed_yaw, 0.0, 1e-3))
  head_t, head_m, drift = rd("head", "tail"), rd("head", "absmed"), rd("drift", "tail")
  ok(f"unit flat readings: a 10 s line turning 0.01 rad/s -- heading {head_t} / {head_m} deg (5.73), drift {drift} "
     f"(tan 0.05), on the sideways and the straight lines alike", near(head_t, math.degrees(0.1), 0.05)
     and near(head_m, math.degrees(0.1), 0.05) and near(drift, math.tan(0.05), 2e-3)
     and near(rd("drift", "tail", bins=["side"]), math.tan(0.05), 2e-3)
     and near(rd("drift", "tail", bins=["fwd", "fwd_slow", "back"]), math.tan(0.05), 2e-3))
  creep, creep_seen = rd("creep", "mean"), rd("creep", "mean", scan="seen")
  heavy_creep = [v for (r, m, b), buf in tr.buf.items() if r == "creep" and m == "heavy" for v in buf.v]
  ok(f"unit flat readings: 5 mm/s of creep over the 199 judged steps reads {creep} m (0.0199) on the light robots' "
     f"worst bin; the heavy robots' ten times that ({len(heavy_creep)} samples) reaches no constraint",
     near(creep, 0.0199, 5e-4) and near(creep_seen, 0.0199, 5e-4) and near(rd("creep", "mean", bins=["rest"]), 0.0199, 5e-4)
     and len(heavy_creep) > 0 and all(near(v, 0.199, 5e-3) for v in heavy_creep))
  stance, stance_stop = rd("stance", "p90"), rd("stance_stop", "p90", bins=["fast", "fwd"])
  ok(f"unit flat readings: the stance -- the leg joints' largest offset over the 9 s rest ({stance}), their mean 3 s "
     f"after a stop from a forward command ({stance_stop})", near(stance, 0.12, 1e-6) and near(stance_stop, 0.07, 1e-6))
  hip = rd("march_hip", "mean")
  hip_before = [v for (r, m, b), buf in tr.buf.items() if r == "march_before" for v in buf.v]
  ok(f"unit flat readings: the march script runs on the splayed robots only -- hips 0.14 rad before it "
     f"({len(hip_before)} samples), {hip} after the stand that follows", near(hip, 0.02, 1e-6) and len(hip_before) > 0
     and all(near(v, 0.14, 1e-6) for v in hip_before))
  gait = rd("gait", "mean", worst="min")
  ok(f"unit flat readings: the gait's worst bin ({gait}); none above 1 m/s forward (the acceptance judges none there)",
     near(gait, 0.85, 1e-6) and not [b for r, _, b in tr.buf if r == "gait" and b in ("fast", "fast_arc")])

  # -- the scripts given up: a fall, a robot carried off the strip (a line that ran 3 s still gives its drift), a
  #    command rewritten by something else
  def given_up(reason: str, pick, act) -> tuple:
    for _ in range(800):
      j = pick().nonzero().flatten()
      if len(j):
        break
      step()
    j = int(j[0])
    was, lines = tr.drops.get(reason, 0), sum(v for (g, _, _), v in tr.tally.items() if g == "line")
    act(j)
    step()
    return tr.drops.get(reason, 0) - was, sum(v for (g, _, _), v in tr.tally.items() if g == "line") - lines, j

  fell = given_up("fell", lambda: tr.on & (tr.k > 20), lambda j: reset(torch.tensor([j])))
  strip = given_up("strip", lambda: tr.on & (tr.kind == uflat.F_MOVE) & (tr.k > 20),
                   lambda j: st.pos.__setitem__((j, 1), 3.9))
  line = given_up("strip", lambda: tr.on & (tr.kind == uflat.F_LINE) & (tr.k >= 150) & (tr.k < 400),
                  lambda j: st.pos.__setitem__((j, 1), 3.9))
  cmd = given_up("cmd", lambda: tr.on & (tr.k > 20), lambda j: term.vel_command_b.__setitem__((j, 0), 0.123))
  ok(f"unit flat scripts: given up and counted once each -- a fall {fell[:2]}, off the strip while tracking "
     f"{strip[:2]}, off it 3 s or more into a line (its drift still taken) {line[:2]}, the command rewritten "
     f"{cmd[:2]}; the robot is handed back", fell[:2] == (1, 0) and strip[:2] == (1, 0) and line[:2] == (1, 1)
     and cmd[:2] == (1, 0) and not bool(tr.on[[strip[2], line[2], cmd[2]]].any()))

  # -- the checkpoint: the tracker's own state, and inside the constraints'
  asks = [(("creep", "mean"), {}), (("drift", "tail"), {"bins": ["side"]}), (("head", "absmed"), {}),
          (("speed_lin", "dev1"), {"scan": "blind"}), (("gait", "mean"), {"worst": "min"}), (("stance", "p90"), {})]
  tr2 = Tracker(NS(params=params), NS(num_envs=n, device="cpu"))
  tr2.load(json.loads(json.dumps(tr.state())))
  ok("unit flat state: the samples, tallies and drops come back from the (JSON-plain) state, the readings with them",
     all(tr.read(*a, **kw) is not None and tr2.read(*a, **kw) == tr.read(*a, **kw) for a, kw in asks)
     and tr2.tally == tr.tally and tr2.drops == tr.drops and tr2.table()["bins"] == tr.table()["bins"])
  cons_a = ucons.constraints(NS(params={"items": items}), env)
  env.u_constraints = cons_a
  saved = cons_a.state()
  env3 = NS(num_envs=n, device="cpu")
  tr3 = Tracker(NS(params=params), env3)
  cons_b = ucons.constraints(NS(params={"items": items}), env3)
  cons_b.load(dict(saved))
  got = tr3.read("creep", "mean")
  cons_b.load({k: v for k, v in saved.items() if k != "flat"})
  ok("unit flat state: the samples travel in the constraints' checkpoint state; one without them loads and leaves "
     "the tracker as it is", "flat" in saved and got is not None and got == tr.read("creep", "mean")
     and tr3.read("creep", "mean") == got and not cons_b.redefined and not cons_b.remeasured)

  # -- a fault inside the tracker: it stops, the robots get drawn commands again, the flat: items' readings are
  #    dropped (their multipliers hold), the training goes on
  cons_a.values.update({"creep@seen": 0.5, "spin@seen": 0.4})
  cons_a.count.update({"creep@seen": 900, "spin@seen": 900})
  cons_a.value.update({"creep": 0.5, "spin": 0.4})
  cons_a.scale.update({"creep": 1.7})
  was_on, drops = tr.on.clone(), dict(tr.drops)
  st.broken = True
  step()
  st.broken = False
  for _ in range(5):
    step()
  ok(f"unit flat fault: the tracker stops for good ({(tr.fault or '').strip().splitlines()[-1:]}), its "
     f"{int(was_on.sum())} robots are handed back, read() gives nothing, the constraints forget the flat: readings "
     f"and keep the multiplier", bool(was_on.any()) and tr.fault is not None and "planted" in tr.fault
     and not bool(tr.on.any()) and not tr.info and bool((term.time_left[was_on] > 1.0).all())
     and tr.read("creep", "mean") is None and "planted" in tr.table()["fault"] and tr.drops == drops
     and "creep" not in cons_a.value and "creep@seen" not in cons_a.values and "creep@seen" not in cons_a.count
     and cons_a.scale["creep"] == 1.7 and cons_a.values.get("spin@seen") == 0.4 and cons_a.count.get("spin@seen") == 900)

  # -- the records: the table in the snapshots, a tag per bin, the fault as an alert
  table = tr2.table() | {"fault": "Traceback (most recent call last):\n  ...\nValueError: planted"}
  envd = fake_env(items)
  envd.u_flat = NS(table=lambda: table)
  diagnostics.episodes(None, envd)
  run = Runner(envd, str(out / "FLAT"), script(items)[0])
  os.makedirs(out / "FLAT", exist_ok=True)
  run.learn(45)
  run.rec.ended or run.rec._atexit()
  ev, snaps = read(out / "FLAT/records/timeline.jsonl"), read(out / "FLAT/records/record.jsonl")
  tags = run.logger.writer.tags
  want = ({f"Diag/flat/{shown}/{b}" for shown, d in table["bins"].items() for b in d}
          | {f"Diag/flat_n/creep/{b}" for b in table["bins"]["creep"]} | {f"Diag/flat_drop/{k}" for k in table["drops"]}
          | {"Diag/flat/scripted"})
  ok(f"unit flat records: a tag per reading, bin and mode ({len(want)}), the sample counts of the main readings "
     f"only, the scripts given up by reason", len(want) > 40 and want <= tags
     and {"Diag/flat_drop/fell", "Diag/flat_drop/strip", "Diag/flat_drop/cmd"} <= tags
     and not [x for x in tags if x.startswith(("Diag/flat_n/stop_dist/", "Diag/flat_n/head/"))])
  ok("unit flat records: the snapshots carry the table; the tracker's fault is the alert flat_fault",
     bool(snaps) and snaps[-1].get("flat", {}).get("drops") == table["drops"] and "fault" not in [e["ev"] for e in ev]
     and any(e["ev"] == "alert" and e.get("name") == "flat_fault" and "planted" in e["msg"] for e in ev))


def sticky_payload_units(ok) -> None:
  """teacher_env.sticky_payload (U_PAYLOAD_STICKY): the payload class stays with the robot, the mass is drawn inside
  the class at every reset, the evaluation robots draw their class anew, and the payloads over all the robots are
  distributed as the inner event's. The inner event here: mdp.base_payload_point's mass draw alone."""
  from src.tasks.unified import teacher_env
  from src.tasks.unified.constraints import HEAVY_FRAC, LIGHT_FRAC

  n, hm = 60000, 8.0

  def inner(env, env_ids, light_range, heavy_max, heavy_share, corner_share=0.0, **kw):
    k = len(env_ids)
    heavy = torch.rand(k) < heavy_share
    m = torch.where(heavy, torch.rand(k) * heavy_max,
                    light_range[0] + torch.rand(k) * (light_range[1] - light_range[0]))
    corner = heavy & (torch.rand(k) < corner_share)
    m = torch.where(corner, heavy_max * (0.75 + 0.25 * torch.rand(k)), m)
    env.payload_kg[env_ids] = m

  band = lambda m: (m > LIGHT_FRAC * hm).long() + (m >= HEAVY_FRAC * hm).long()   # noqa: E731
  p = dict(inner=inner, light_range=(-1.0, 2.0), heavy_max=hm, heavy_share=0.5, corner_share=0.3,
           pos_ranges=((-0.25, 0.25), (-0.12, 0.12), (-0.05, 0.3)), base_com_range=0.05, ramp_steps=0)
  env = NS(num_envs=n, device="cpu", payload_kg=torch.zeros(n), u_eval_mask=torch.arange(n) < n // 10)
  ids = torch.arange(n)
  teacher_env.sticky_payload(env, ids, redraw=0.0, **p)
  c0, m0 = env.u_pay_class.clone(), env.payload_kg.clone()
  ok("sticky payload: every robot has a class after its first reset, and its mass is in the class's band",
     bool((c0 >= 0).all()) and bool((band(m0) == c0).all()))
  hs, cs = 0.5, 0.3
  exact = [1 - hs + hs * (1 - cs) * LIGHT_FRAC, hs * (1 - cs) * (HEAVY_FRAC - LIGHT_FRAC),
           hs * (cs + (1 - cs) * (1 - HEAVY_FRAC))]
  got = [float((c0 == k).float().mean()) for k in range(3)]
  ok(f"sticky payload: the classes' shares are the inner event's ({[round(v, 3) for v in got]})",
     all(abs(g - e) < 0.01 for g, e in zip(got, exact)))
  plain = NS(payload_kg=torch.zeros(n))
  inner(plain, ids, **{k: v for k, v in p.items() if k != "inner"})
  q = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
  ok("sticky payload: the masses over all the robots are distributed as the inner event's",
     float((torch.quantile(m0, q) - torch.quantile(plain.payload_kg, q)).abs().max()) < 0.25
     and abs(float((m0 >= 0.75 * hm).float().mean()) - hs * (cs + (1 - cs) * 0.25)) < 0.01
     and abs(float((m0 < 0).float().mean()) - (1 - hs) / 3) < 0.01)
  teacher_env.sticky_payload(env, ids, redraw=0.0, **p)
  tr = ~env.u_eval_mask
  ok("sticky payload: at the next reset the training robots keep their class and draw a new mass inside it",
     bool((env.u_pay_class[tr] == c0[tr]).all()) and bool((band(env.payload_kg) == env.u_pay_class).all())
     and float((env.payload_kg[tr] != m0[tr]).float().mean()) > 0.99)
  ok("sticky payload: the evaluation robots draw their class anew at every reset",
     0.4 < float((env.u_pay_class[~tr] != c0[~tr]).float().mean()) < 0.7)
  c1 = env.u_pay_class.clone()
  teacher_env.sticky_payload(env, ids, redraw=0.2, **p)
  moved = float((env.u_pay_class[tr] != c1[tr]).float().mean())
  ok(f"sticky payload: redraw 0.2 moves the class of 0.2 x (1 - sum p^2) of the robots ({moved:.3f})",
     abs(moved - 0.2 * (1 - sum(e * e for e in exact))) < 0.01)
  c2, m2 = env.u_pay_class.clone(), env.payload_kg.clone()
  sub = ids[::3]
  teacher_env.sticky_payload(env, sub, redraw=1.0, **p)
  rest = torch.ones(n, dtype=torch.bool)
  rest[sub] = False
  ok("sticky payload: a reset of some robots leaves the others' class and payload",
     bool((env.u_pay_class[rest] == c2[rest]).all()) and bool((env.payload_kg[rest] == m2[rest]).all()))
  half = dict(p, heavy_max=4.0)   # (the limit follows the progress: the classes are bands of the limit now)
  teacher_env.sticky_payload(env, ids, redraw=0.0, **half)
  hv = env.u_pay_class == 2
  ok("sticky payload: at half the limit the heavy class carries the top third of half the limit",
     float(env.payload_kg[hv].min()) >= HEAVY_FRAC * 4.0 - 1e-6 and float(env.payload_kg[hv].max()) <= 4.0 + 1e-6)
  from src.tasks.velocity import mdp
  ok("sticky payload: the event carries the inner event's model fields and recompute level",
     teacher_env.sticky_payload.model_fields == mdp.base_payload_point.model_fields
     and teacher_env.sticky_payload.recompute == mdp.base_payload_point.recompute)


def loaded_kick_units(ok) -> None:
  """The probe switches of 2026-11-13 (teacher_env._probe_switches): U_STAIR_SPEED_LOADED (the stairs speed cost by
  payload, the descent / climb items over the robots up to the common payload), U_KICK_KEEPOUT (the keep-out zone of
  the step faces as one of the kicks item's costs), U_M_WEIGHT (a term's weight before the stairs quality)."""
  import numpy as np

  from src.tasks.unified import teacher_env as te

  keys = ("U_STAIR_SPEED_LOADED", "U_STAIR_SPEED_H", "U_KICK_KEEPOUT", "U_M_WEIGHT", "U_STOP_FEET_DIR")

  def under(fn, **env: str):
    old = {k: os.environ.pop(k, None) for k in keys}
    os.environ.update(env)
    try:
      return fn()
    finally:
      for k in keys:
        os.environ.pop(k, None)
        if old[k] is not None:
          os.environ[k] = old[k]

  def items_env(kind: str = "legged", **env: str) -> dict:
    return under(lambda: {c["name"]: c for c in constraint_items(kind)}, **env)

  speed = ("descent", "climb")
  for kind in ("legged", "wheeled"):
    base = items_env(kind)
    for env, metric in (({"U_STAIR_SPEED_LOADED": "0"}, "u_stair_speed_c"),
                        ({"U_STAIR_SPEED_LOADED": "0.3", "U_STAIR_SPEED_H": "1"}, "u_stair_speed_hc")):
      got = items_env(kind, **env)
      ok(f"unit loaded ({kind}): {env} -- descent and climb read {metric} (a measure of constraints.CUSTOM whose "
         f"multipliers carry over from the plain ones), every other item is the same",
         all(got[n]["metric"] == metric and {**got[n], "metric": base[n]["metric"]} == base[n] for n in speed)
         and metric in ucons.CUSTOM and all({"stair_speed|mean", "u_stair_speed_h|mean", "u_stair_speed_c|mean",
                                            "u_stair_speed_hc|mean"} <= ucons.CARRY_SCALE[n] for n in speed)
         and {n: c for n, c in got.items() if n not in speed} == {n: c for n, c in base.items() if n not in speed})
    kick = items_env(kind, U_KICK_KEEPOUT="2")
    if kind == "legged":
      ok("unit keep-out (legged): U_KICK_KEEPOUT -- riser_keepout joins the kicks item's costs, nothing else changes",
         kick["kicks"]["terms"] == base["kicks"]["terms"] + [te.KEEPOUT_TERM] and te.KEEPOUT_TERM == "riser_keepout"
         and {**kick["kicks"], "terms": base["kicks"]["terms"]} == base["kicks"]
         and {n: c for n, c in kick.items() if n != "kicks"} == {n: c for n, c in base.items() if n != "kicks"})
    else:
      ok("unit keep-out (wheeled): U_KICK_KEEPOUT changes nothing (every robot's tasks are built at import)",
         kick == base)
  base = items_env()
  state = {"scale": {n: 2.0 for n in base}, "value": {n: 0.5 for n in base},
           "values": {f"{n}@{m}": 0.5 for n in base for m in ucons.MODES},
           "count": {f"{n}@{m}": 900 for n in base for m in ucons.MODES}, "relax": {}, "ref": {},
           "sig": {n: ucons.item_sig(c) for n, c in base.items()}}
  cons = ucons.constraints(NS(params={"items": list(items_env(U_STAIR_SPEED_LOADED="0").values())}), NS())
  cons.load(dict(state))
  ok(f"unit loaded load: a checkpoint measured over every payload under U_STAIR_SPEED_LOADED -- descent and climb "
     f"keep their multipliers and read afresh ({cons.remeasured}), nothing else starts over ({cons.redefined})",
     sorted(cons.remeasured) == ["climb", "descent"] and not cons.redefined and cons.scale["descent"] == 2.0
     and cons.scale["climb"] == 2.0 and "climb" not in cons.value and cons.count["kicks@seen"] == 900)

  # -- the discount: 5 robots at 0 / 0.5 / 0.5833 / 0.6667 / 1 of an 8 kg limit, each short of the stairs speed
  limit, w = 8.0, -12.0
  kg = torch.tensor([0.0, 0.5, (0.5 + ucons.HEAVY_FRAC) / 2, ucons.HEAVY_FRAC, 1.0]) * limit
  raw = torch.tensor([0.6, 0.6, 0.6, 0.6, 0.3])
  rm = NS(_term_names=["stair_speed_deficit"], get_term_cfg=lambda t: NS(weight=w), _step_reward=(raw * w)[:, None])
  env = NS(num_envs=5, device="cpu", payload_kg=kg, u_payload_full=limit, reward_manager=rm)
  for floor, want in ((0.0, [1.0, 1.0, 0.5, 0.0, 0.0]), (0.25, [1.0, 1.0, 0.625, 0.25, 0.25])):
    total = raw * w + ucons.stair_speed_loaded(env, floor=floor)
    ok(f"unit loaded cost: floor {floor} -- the speed cost x {want} at 0 / 0.5 / 0.583 / 0.667 / 1 of the limit "
       f"({[round(v, 3) for v in (total / (raw * w)).tolist()]})",
       bool(torch.allclose(total, raw * w * torch.tensor(want), atol=1e-5)) and ucons.COMMON_FRAC == 0.5)
  ok("unit loaded cost: nothing before the first payload draw, nothing without the cost term",
     float(ucons.stair_speed_loaded(NS(num_envs=5, device="cpu", reward_manager=rm)).abs().sum()) == 0.0
     and float(ucons.stair_speed_loaded(env, term="nosuch").abs().sum()) == 0.0)
  value, mask = ucons.upto_common(lambda e, **kw: (torch.arange(5.0), torch.ones(5, dtype=torch.bool)))(env)
  _, every = ucons.upto_common(lambda e, **kw: (torch.arange(5.0), torch.ones(5, dtype=torch.bool)))(NS())
  ok(f"unit loaded measure: the robots up to the common payload ({mask.tolist()}); every robot before the first draw",
     mask.tolist() == [True, True, False, False, False] and value.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
     and bool(every.all()))

  # -- the keep-out zone on a raster of 2 cm cells: a 15 cm step up at x = 1 m, a 1 m wall from y = 1.8 m
  H = np.zeros((100, 100))
  H[50:, :] = 0.15
  H[:, 90:] = 1.0
  term = ucons.riser_keepout(NS(params={}), NS())
  term.set_grid(H, 0.0, 0.0, 0.02, "cpu")
  r = 0.022
  feet = {"far from the step": ((0.50, 0.5, 0.0), 0.0), "3 cm before it, on the ground": ((0.97, 0.5, 0.0), 1.0),
          "3 cm before it, 1 cm under its top": ((0.97, 0.5, 0.14), 0.5),
          "3 cm before it, 3 cm over its top": ((0.97, 0.5, 0.18), 0.0),
          "7 cm before it, on the ground (outer ring)": ((0.93, 0.5, 0.0), 0.5),
          "10 cm before it, on the ground": ((0.90, 0.5, 0.0), 0.0),
          "on the step, 5 cm past its edge": ((1.05, 0.5, 0.15), 0.0),
          "3 cm from the wall (higher than a step)": ((0.50, 1.77, 0.0), 0.0),
          "outside the raster": ((-5.0, 9.0, 0.0), 0.0)}
  pos = torch.tensor([[x, y, sole + r] for (x, y, sole), _ in feet.values()])
  got = term.zone(pos, foot_radius=r)
  ok("unit keep-out zone: " + "; ".join(f"{k} {float(v):.2f}" for k, v in zip(feet, got)),
     bool(torch.allclose(got, torch.tensor([v for _, v in feet.values()]), atol=1e-4)))
  four = pos[[1, 2, 4, 0]]
  found = torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 2.0, 0.0]])
  env = NS(scene={"robot": NS(data=NS(site_pos_w=torch.stack([four, four]))),
                  "feet_ground_contact": NS(data=NS(found=found))},
           device="cpu", u_lib_meter=NS(grid=NS(H=H, x0=0.0, y0=0.0, res=0.02)))
  lazy = ucons.riser_keepout(NS(params={}), NS())
  out = lazy(env, sensor_name="feet_ground_contact", asset_cfg=NS(name="robot", site_ids=[0, 1, 2, 3]), foot_radius=r)
  ok(f"unit keep-out term: the sum over the swinging feet, on the stairs meter's raster ({out.tolist()}: four feet "
     f"in the air 2.0, the two on the ground left out 0.5)",
     bool(torch.allclose(out, torch.tensor([2.0, 0.5]), atol=1e-4)) and lazy.H.dtype == torch.float32)

  # -- U_M_WEIGHT
  factors = under(te.m_weight_factors, U_M_WEIGHT="a:3, b:0.3,c:0.5,zz:2")
  items = [{"name": "x", "by": "q", "terms": ["a"], "start": {"a": -1.0}}, {"name": "y", "by": "p", "terms": ["p1"]}]
  weights = {"b": (-1.0, -10.0)}
  rewards = {"a": NS(weight=-2.0), "b": NS(weight=-10.0), "c": NS(weight=-4.0), "p1": NS(weight=-1.0)}
  missing = te.m_weights(factors, items, weights, rewards, {"p1"})
  ok(f"unit U_M_WEIGHT: a constraint's term -- its start weight x 3 ({items[0]['start']}); a difference of the M and "
     f"P keys -- the M side x 0.3; a term equal in both -- (x 0.5, as it is) ({weights}); a term this robot has not "
     f"is reported ({missing}); none set: nothing",
     factors == {"a": 3.0, "b": 0.3, "c": 0.5, "zz": 2.0} and items[0]["start"] == {"a": -3.0}
     and weights == {"b": (-0.3, -10.0), "c": (-2.0, -4.0)} and missing == ["zz"] and under(te.m_weight_factors) == {})
  try:
    te.m_weights({"p1": 2.0}, items, {}, rewards, {"p1"})
    refused = False
  except AssertionError:
    refused = True
  ok("unit U_M_WEIGHT: a flat constraint's term is refused", refused)


def meter_eval_mask(ok) -> None:
  """The shared stairs meter (src/metrics/mj.py) and the evaluation robots: the env's own training meters (marked by
  the teacher's runner) count only them; a meter an evaluation script builds counts every robot it was given, also on
  an env whose runner set the mask (eval_stairs.py on a teacher: 72 of 576 robots counted until 2026-11-12)."""
  from types import SimpleNamespace

  from src.metrics.mj import StairsMeter

  ids = torch.tensor([0, 1, 5, 6])
  masked = SimpleNamespace(u_eval_mask=torch.arange(8) < 2)
  m = object.__new__(StairsMeter)
  m.ids, m.eval = ids, False
  ok("stairs meter: an evaluation script's meter counts every robot on an env with evaluation robots",
     m._eval_only(masked) is None)
  m.eval = True
  ok("stairs meter: the training's meter counts the evaluation robots only",
     m._eval_only(masked).tolist() == [True, True, False, False])
  ok("stairs meter: no evaluation robots, every robot counts", m._eval_only(SimpleNamespace()) is None)


def wheel_counter(ok) -> None:
  """The shared wheeled stairs counter (src/metrics/stairs.py) on a grid of several flights with different steps: one
  wheel climbs the 15 cm flight -- riser 1 clean, 2 after one slide-back, 3 after touching the nose, the others clean --
  and every step-up counts. (Until 2026-11-12 treads were neighbours by their index among the whole grid's heights:
  on a training terrain the counter saw a few percent of the step-ups.)"""
  import numpy as np
  from src.metrics.stairs import WheeledStairs
  from src.metrics.terrain import JUMP, Grid

  res, n, steps = 0.02, 9, (0.15, 0.052, 0.078, 0.093, 0.118, 0.135, 0.177, 0.187)
  H = np.zeros((int(5.0 / res), int(2.0 * len(steps) / res)))
  for f, sh in enumerate(steps):   # flight f: y in [2 f, 2 f + 2), climbing along +x from x = 1, treads 0.3 m
    for i in range(H.shape[0]):
      k = 0 if (i + 0.5) * res < 1.0 else min(int(((i + 0.5) * res - 1.0) / 0.3) + 1, n)
      H[i, int(2.0 * f / res):int(2.0 * (f + 1) / res)] = round(k * sh, 3)
  g = object.__new__(Grid)
  g.H, g.x0, g.y0, g.res, g._steps = H, 0.0, 0.0, res, {}
  lv = np.unique(np.round(H, 3))
  g.levels = lv[np.concatenate([[True], np.diff(lv) > JUMP])]
  R = 0.09   # the wheel's radius: its axle's height over what it stands on
  c, sh, y, t = WheeledStairs(g, 1, 1), steps[0], 1.0, [0.0]

  def put(z, *contacts):
    c.step(t[0], 0.02, [[list(contacts)]], np.array([[[1.0, y, z]]]), np.array([[1.0, 0.0]]), np.array([False]),
           np.array([True]))
    t[0] += 0.02

  flat = lambda k, dx=0.15: ((1.0 + 0.3 * (k - 1) + dx if k else 0.8, y, k * sh), 1.0)   # noqa: E731
  nose = lambda k: ((1.0 + 0.3 * (k - 1) + 0.001, y, k * sh), 0.6)                         # noqa: E731
  put(R, flat(0))
  for k in range(1, n + 1):
    low = (k - 1) * sh + R
    if k in (2, 3):   # off its stand, on the nose, the axle 6 cm up
      put(low, flat(k - 1, 0.28))
      put(low + 0.06, nose(k))
    if k == 2:   # back on its stand, the riser's face at axle height; then over the nose
      put(low, flat(1, 0.2), ((1.0 + 0.3 - 0.001, y, sh + R), 0.0))
      put(low + 0.06, nose(2))
    put(k * sh + R, flat(k))
    put(k * sh + R, flat(k))
  got = {k: int(v[0]) for k, v in c.counts(np.array([True])).items()}
  ok(f"unit wheeled counter among {len(steps)} flights ({len(g.levels)} grid levels): every step-up counts ({got})",
     got["ups"] == n + 1 and got["slips"] == 1 and got["edged"] == 2 and got["clean"] == n - 2)

  # a step lower than the wheel's radius (7.8 cm): standing at the riser the wheel touches its nose without leaving
  # the ground. The nose contact coming and going, a contact flicker and a 1 cm hop are no slide-backs (until
  # 2026-11-12 each of them was one); off the stand on the nose with the axle 4 cm up and back down is one.
  c, sh, y, t = WheeledStairs(g, 1, 1), steps[2], 5.0, [0.0]
  put(R, flat(0))
  for _ in range(5):
    put(R, flat(0), nose(1))
    put(R, flat(0))
  put(R + 0.002, nose(1))   # the ground contact drops out for a step
  put(R, flat(0), nose(1))
  put(R + 0.01, nose(1))
  put(R, flat(0))
  touch = {k: int(v[0]) for k, v in c.counts(np.array([True])).items()}
  put(R + 0.02, nose(1))
  put(R + 0.04, nose(1))
  put(R, flat(0), nose(1))
  slid = {k: int(v[0]) for k, v in c.counts(np.array([True])).items()}
  put(R + 0.04, nose(1))
  put(sh + R, flat(1))
  got = {k: int(v[0]) for k, v in c.counts(np.array([True])).items()}
  ok(f"unit wheeled counter on a 7.8 cm step: touching the riser from the ground is no slide-back ({touch}), leaving "
     f"the stand and coming back is one ({slid}), then over the nose ({got})",
     touch["ups"] == 0 and touch["slips"] == 0 and slid["ups"] == 1 and slid["slips"] == 1
     and got["ups"] == 2 and got["slips"] == 1 and got["edged"] == 1)

  # a hop on flat-topped ground next to the riser, never touching it, is no slide-back either
  c, t = WheeledStairs(g, 1, 1), [0.0]
  put(R, flat(0))
  put(R, flat(0), nose(1))
  put(R + 0.05)
  put(R, flat(0))
  got = {k: int(v[0]) for k, v in c.counts(np.array([True])).items()}
  ok(f"unit wheeled counter: a hop beside the riser is no slide-back ({got})", got["ups"] == 0 and got["slips"] == 0)


def main() -> int:
  out = Path(sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="records_test_"))
  items = constraint_items("legged")
  at, capped = script(items)
  checks: list[tuple[str, bool]] = []
  ok = lambda name, cond: checks.append((name, bool(cond)))   # noqa: E731

  # run A: ends in the planted exception
  env = fake_env(items)
  diagnostics.episodes(None, env)
  a = Runner(env, str(out / "A"), at)
  os.makedirs(out / "A", exist_ok=True)
  try:
    a.learn(1300)
    ok("A raised", False)
  except RuntimeError:
    ok("A raised", True)
  a.rec.ended or a.rec._atexit()
  ev, snaps = read(out / "A/records/timeline.jsonl"), read(out / "A/records/record.jsonl")
  kinds = [e["ev"] for e in ev]
  names = {(e["ev"], e.get("name")) for e in ev}
  ok("A start first", kinds[0] == "start" and ev[0]["code"] == "test000" and len(ev[0]["items"]) == len(items))
  ok("A no fault", "fault" not in kinds)
  now = __import__("time").time()
  status_a = json.loads((out / "A/records/status.json").read_text())
  ok("A times: the events', the records' and the status' clock is the wall clock (to the second)",
     all(abs(now - x["t"]) < 600 for x in ev + snaps) and abs(now - status_a["t"]) < 600 and ev[-1]["t"] >= ev[0]["t"])
  ok("A exception + end", "exception" in kinds and ev[-1]["ev"] == "end" and ev[-1]["reason"] == "exception")
  ok("A checkpoints", kinds.count("checkpoint") == 12)
  ok("A milestone p 0.5", any(e["ev"] == "milestone" and e["name"] == "p" and e["mark"] == 0.5 for e in ev))
  for want in (("alert", "std_rising"), ("alert", f"cap:{capped}"),
               ("alert", "q_waiting"), ("alert", "clock_slow"), ("relax", "up"),
               ("cap", capped), ("clock_slow", "p"), ("q_gate", "q"), ("constraint_on", capped),
               ("alert", "payload_fail:flat"), ("alert", "late:walk"), ("milestone", "expect:commands"),
               ("milestone", "expect:climb_first")):
    ok(f"A {want[0]} {want[1]}", want in names)
  ok("A no alert on a healthy ground", ("alert", "fail_high:flat") not in names
     and ("alert", "payload_fail:random_rough") not in names)
  # (the stairs before the quality stage: the curriculum's rows fail by design, no stairs constraint is held back)
  ok("A no stairs alert before the quality stage", ("alert", "fail_high:walled_up") not in names
     and ("alert", "relax:up") not in names and ("alert", "eval_gap") not in names)
  relaxed = next((e for e in ev if e["ev"] == "relax" and e["name"] == "up"), {})
  ok("A relax up: the evaluation robots' fixed-row fail rate", abs((relaxed.get("eval_fail") or {}).get("up", 0) - 0.5)
     < 1e-6)
  lit = (relaxed.get("eval_light") or {}).get("up") or {}
  ok(f"A relax up: the light robots' rate, its reference and episodes ({lit})",
     abs(lit.get("fail", 0) - 3.34 / 9) < 1e-4 and lit.get("ref") == 0.15 and lit.get("n") == 180)
  slow = next((e for e in ev if e["ev"] == "alert" and e["name"] == "clock_slow"), {})
  ok("A clock_slow: the other ground's fail rate", slow.get("fail") == 0.4 and "other ground" in slow.get("msg", ""))
  first = next((e for e in ev if e["ev"] == "milestone" and e["name"] == "expect:climb_first"), {})
  ok("A climb_first: on the low fixed rows", first.get("it") == 300 and "low" in first.get("what", ""))
  # the reference schedule
  exp = {e["name"]: e["by"] for e in ev[0].get("expect", [])}
  ok("A start: the schedule and the clock", exp.get("walk") == diagnostics.TIME_START_IT + 500 and len(exp) == 7
     and ev[0].get("clock", {}).get("span") == diagnostics.TIME_SPAN_IT and ev[0].get("reached") == {})
  late = [e for e in ev if e["ev"] == "alert" and e["name"] == "late:walk"]
  ok("A late:walk raised at its iteration", late and late[0]["it"] == exp["walk"] and late[0]["now"]["other_fail"] == 0.4)
  ok("A no late alert before a deadline", not any(e["ev"] == "alert" and e["name"].startswith("late:") and
                                                   e["it"] < exp[e["name"][5:]] for e in ev))
  ok("A no late alert for a milestone reached", ("alert", "late:commands") not in names)
  ok("A the walk milestone is not reached", ("milestone", "expect:walk") not in names)
  ok("A alert texts: floats shortened", not any(
    len(x.split(".")[-1]) > 4 for e in ev if e["ev"] == "alert"
    for x in __import__("re").findall(r"\d+\.\d+", e["msg"])))
  # every episode fed in is counted once
  n = sum(d["n"] for s in snaps for d in s.get("ep", {}).values())
  fail = sum(d["fail"] for s in snaps for d in s.get("ep", {}).values())
  calf = sum(v for s in snaps for d in s.get("ep", {}).values() for k, v in d.get("bodies", {}).items() if k == "FL_calf")
  cause = sum(d.get("cause", {}).get("illegal_contact", 0) for s in snaps for d in s.get("ep", {}).values())
  # (the run's last record, written on the exception, carries the episodes of the iterations after the last snapshot)
  ok(f"A episodes counted ({n} of {a.fed['n']})", n == a.fed["n"] and fail == a.fed["fail"])
  ok("A final record", snaps[-1].get("final") == "exception" and snaps[-1]["it_end"] == 1199 and snaps[-1]["ep"])
  ok("A fails = causes = calf contacts", fail == cause == calf)
  for g, (gn, gf) in a.fed["by_ground"].items():
    got = [sum(s["ep"].get(g, {}).get(k, 0) for s in snaps if "ep" in s) for k in ("n", "fail")]
    ok(f"A ground {g}", got == [gn, gf])
  # the payload: the mid class is the planted 4 kg robots, the "front" place those with the payload at the front
  cls = lambda key, name: [sum(s["ep"].get("flat", {}).get(key, {}).get(name, [0, 0])[i] for s in snaps if "ep" in s)   # noqa: E731
                           for i in (0, 1)]
  ok(f"A flat: the loaded robots' episodes ({cls('pay', 'mid')} of {a.fed['flat_loaded']})",
     cls("pay", "mid") == a.fed["flat_loaded"] and cls("pay", "heavy") == [0, 0])
  ok(f"A flat: the payload at the front ({cls('load', 'front')} of {a.fed['flat_front']})",
     cls("load", "front") == a.fed["flat_front"])
  ok("A flat: every loaded episode has a place", [sum(cls("load", k)[i] for k in diagnostics.LOAD) for i in (0, 1)]
     == a.fed["flat_loaded"] and all(cls("load", k)[0] > 0 for k in diagnostics.LOAD))
  ok("A where: failed episodes at 1.2 m / 3 s", all(
    d["where"]["dist"][3] == d["fail"] and d["where"]["sec"][3] == d["fail"]
    for s in snaps for d in s.get("ep", {}).values() if d.get("fail")))
  ok("A snapshot blocks", all(k in snaps[-2] for k in ("perf", "loss", "grad", "policy", "rollout", "cur", "cons",
                                                         "relax", "weights", "gate", "phys", "ep", "alerts")))
  ok("A explained variance over the training robots", 0.8 < snaps[-2]["rollout"]["ev"] < 1.0)
  ok("A gradient norms", snaps[-2]["grad"].get("actor") is not None and snaps[-2]["grad"]["actor"] <= 1.0 + 1e-4)
  ok("A console tags", any("tags" in s and "Episode_Reward/track" in s["tags"] for s in snaps))
  ok("A gate view misses the planted item", any(x.startswith(capped) for x in snaps[-2]["gate"]["miss"]))
  status = json.loads((out / "A/records/status.json").read_text())
  ok("A status", status.get("ended") == "exception" and status["it"] >= 1190 and "std_rising" in status["alerts"])
  ok("A status: the schedule", status["expect"]["walk"] == [exp["walk"], None] and "late:walk" in status["alerts"]
     and status["expect"]["commands"][1] is not None and status["expect"]["commands"][1] <= 400)
  ok("A snapshots: the schedule", snaps[-2].get("expect", {}).get("walk", 0) is None
     and snaps[-2]["expect"]["climb_first"] is not None)
  ok("A tensorboard tags", {"Diag/ev", "Diag/fail/walled_up", "Diag/phys/sat_calf", "Diag/alerts",
                             "Diag/eval_fail/up_row0", "Diag/eval_fail/up_row8", "Diag/eval_band/up_seen_b0",
                             "Diag/eval_band/up_blind_b3", "Diag/eval_class/up_light", "Diag/eval_class/up_mid"}
     <= a.logger.writer.tags and "Diag/eval_class/up_heavy" not in a.logger.writer.tags)
  cur = snaps[-2]["cur"]
  ok("A cur: the fixed rows", abs(cur["eval_fail"]["up"] - 0.5) < 1e-6 and cur["eval_fail_rows"]["up"][:3] == [0.1, 0.2, 0.3]
     and cur["eval_fail_rows"]["down"] == [None] * 9 and cur["clock_fail"] == 0.4 and cur["family_fail"]["up"] == 0.6)
  lit = cur.get("eval_light", {}).get("up", {})
  near = lambda a, b: len(a) == len(b) and all(abs(x - y) < 1e-6 for x, y in zip(a, b))   # noqa: E731
  ok(f"A cur: the light robots by step-height band ({lit.get('bands')})", cur.get("eval_row_bands") == [0, 0, 1, 1, 1, 2, 2, 3, 3]
     and near(lit["bands"]["seen"], [0.07, 0.4, 0.5, 0.5]) and near(lit["bands"]["all"], lit["bands"]["blind"])
     and lit["n_mode"] == {"all": 180, "seen": 180, "blind": 180} and "down" not in cur["eval_light"])
  ok("A cur: the payload classes", abs(cur["eval_class"]["up"]["mid"]["fail"] - 0.5) < 1e-6
     and cur["eval_class"]["up"]["heavy"] == {"fail": None, "n": 0})
  ok("A start: the survival items are gate-only", all(
    any(i["name"] == n and i["enforce"] and i["terms"] == [] and i["metric"] == f"evalfail:{n[8:]}" for i in ev[0]["items"])
    for n in ("survive_up", "survive_down")))
  ok("A episodes: the ground's family", snaps[-2]["ep"]["walled_up"]["family"] == "up"
     and snaps[-2]["ep"]["flat"]["family"] == "other" and snaps[-2]["ep"]["walled_down"]["family"] == "down")

  # run B: resumed from A's checkpoint, ends normally
  env = fake_env(items)
  diagnostics.episodes(None, env)
  os.makedirs(out / "B", exist_ok=True)
  b = Runner(env, str(out / "B"), at)
  b.load(str(out / "A/model_1100.pt"))
  b.learn(200)
  evb = read(out / "B/records/timeline.jsonl")
  ok("B resumed", evb[0]["ev"] == "start" and evb[0]["resumed"]["it"] == 1100 and evb[0]["it"] == 1100)
  ok("B end done", evb[-1]["ev"] == "end" and evb[-1]["reason"] == "done" and not any(e["ev"] == "fault" for e in evb))
  reached_a = {e["name"][7:]: e["it"] for e in ev if e["ev"] == "milestone" and e["name"].startswith("expect:")}
  ok(f"B carries A's milestones {reached_a}", evb[0].get("reached") == reached_a and len(reached_a) >= 2
     and not any(e["ev"] == "milestone" and e["name"].startswith("expect:") for e in evb))
  ok("B late:walk again", any(e["ev"] == "alert" and e["name"] == "late:walk" for e in evb))

  # run C: the stairs quality stage, the evaluation robots failing on their fixed rows
  env = fake_env(items)
  diagnostics.episodes(None, env)
  os.makedirs(out / "C", exist_ok=True)
  c = Runner(env, str(out / "C"), at)
  c.learn(560)
  evc = read(out / "C/records/timeline.jsonl")
  ok("C end done", evc[-1]["ev"] == "end" and evc[-1]["reason"] == "done" and not any(e["ev"] == "fault" for e in evc))
  high = next((e for e in evc if e["ev"] == "alert" and e["name"] == "fail_high:walled_up"), {})
  ok(f"C fail_high:walled_up on the evaluation robots ({high.get('rate')}, {high.get('n')} episodes)",
     high.get("role") == "eval" and 0.55 < high.get("rate", 0) < 0.85 and 50 <= high.get("n", 0) < high.get("all_n", 0))
  ok("C no alert on a healthy stairs ground", not any(e["ev"] == "alert" and e["name"] in (
    "fail_high:stairs_up_wide", "fail_high:walled_down", "eval_gap", "q_waiting") for e in evc))
  hold = next((e for e in evc if e["ev"] == "alert" and e["name"] == "relax:up"), {})
  ok("C relax:up with its constraints on", 480 <= hold.get("it", 0) <= 520
     and abs((hold.get("eval_fail") or {}).get("up", 0) - 0.5) < 1e-6)

  # the reader
  rep = {}
  for run in ("A", "B", "C"):
    r = subprocess.run([sys.executable, str(ROOT / "scripts/unified/run_report.py"), str(out / run)],
                       capture_output=True, text=True)
    rep[run] = r.stdout
    print(f"\n######## run_report.py {run} (exit {r.returncode}) ########\n{r.stdout}{r.stderr}")
    ok(f"report {run} runs", r.returncode == 0 and not r.stderr.strip())
  for text in ("训练没有跑完", "planted: CUDA error", "FL_calf", "动作噪声在涨",
               f"约束 {capped} 的乘子已到上限", "质量阶段还没开始", "评估机器人在固定楼梯行",
               "阶段预期落后：平地会走", "带负载的机器人摔得比轻载的多", "偏前的负载最容易摔", "== 进度预估", "== 负载"):
    ok(f"report A names: {text}", text in rep["A"])
  ok("report A: no stairs problem before the quality stage", "walled_up 地面失败率高" not in rep["A"]
     and "up 地面的约束一直在放松" not in rep["A"])
  for text in ("walled_up 地面失败率高（评估机器人固定行上", "up 地面的约束一直在放松",
               "轻载评估机器人固定行失败率 上楼 0.371（参考值 0.15，180 回合）", "比参考值高 0.08 以上放松",
               "可见 7% / 40% / 50% / 50%  遮挡 7% / 40% / 50% / 50%", "轻 0.371（180 回合），中 0.5（180 回合），重 -（0 回合）",
               "只进门槛", "评估机器人各固定行的失败率：0:10% 1:20% 2:30% 3:65%", "FL_calf"):
    ok(f"report C names: {text}", text in rep["C"])
  ok("report C: the alerts in Chinese", "episodes fail" not in rep["C"] and "constraints relaxed" not in rep["C"])
  ok("report A: the alerts in Chinese", "of the action" not in rep["A"] and "episodes fail" not in rep["A"]
     and "not reached by" not in rep["A"])
  ok("report B reads A too", "续训自" in rep["B"] and "正常结束" in rep["B"])
  ok("report B: A after the checkpoint is left out", "只算到续训点迭代 1100" in rep["B"])
  ok("report B: A's milestones", "阶段预期落后：平地会走" in rep["B"] and "达到于迭代 300" in rep["B"])
  r = subprocess.run([sys.executable, str(ROOT / "scripts/unified/run_report.py"), str(out / "A"), "--json"],
                     capture_output=True, text=True)
  js = json.loads(r.stdout) if r.returncode == 0 else {}
  ok("report --json", js.get("problems") and js.get("eta", {}).get("gate") and js.get("payload")
     and any(x["late"] for x in js.get("expect", [])))

  units(ok, out)
  bad = [name for name, good in checks if not good]
  print(f"\n{len(checks) - len(bad)} / {len(checks)} checks passed" + ("" if not bad else "; FAILED: " + "; ".join(bad)))
  print(f"records in {out}")
  return 1 if bad else 0


if __name__ == "__main__":
  sys.exit(main())
