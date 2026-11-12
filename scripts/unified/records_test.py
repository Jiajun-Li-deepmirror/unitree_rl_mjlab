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

  def gate(name: str, survive: float, logged_bar: float) -> str:
    run = out / "gate" / "go2_u_teacher" / name
    w = SummaryWriter(str(run))
    for it in range(0, 601):
      w.add_scalar("Curriculum/u_progress/p", 1.0, it)
      w.add_scalar("Curriculum/u_progress/q", 1.0, it)
      for c in constraint_items("legged"):
        if not c["enforce"]:
          continue
        tag = f"Curriculum/u_constraints/{c['name']}/"
        good = c["target"] * (1.1 if c["sense"] == "min" else 0.9)
        for m in c["modes"]:
          w.add_scalar(tag + f"value_{m}", survive if c["name"] == "survive_up" and m == "blind" else
                       min(good, survive) if c["name"].startswith("survive") else good, it)
        w.add_scalar(tag + "bar", logged_bar if c["name"].startswith("survive") else c["target"], it)
    w.close()
    r = subprocess.run([sys.executable, str(ROOT / "scripts/unified/teacher_gate.py"), str(run)], capture_output=True,
                       text=True, cwd=str(ROOT))
    return r.stdout.strip() + r.stderr.strip()[-300:]

  # (the logged bar is ignored for a gate-only item: 0.99 would pass anything, 0.0 nothing)
  g = gate("pass", SURVIVE_BAR - 0.02, 0.0)
  # (the first checkpoint with a full window behind it: 500)
  ok(f"unit gate: under the bar on disk passes ({g[:40]})", g.startswith("PASS 500 ") and "survive_up_blind" in g)
  g = gate("miss", SURVIVE_BAR + 0.05, 0.99)
  ok(f"unit gate: over it does not ({g})", g.startswith("NOT YET 600 ") and "survive_up_blind" in g
     and "survive_up_seen" not in g and "survive_down" not in g and g.count(" vs ") == 1)
  wheel_counter(ok)


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
  c, sh, y, t = WheeledStairs(g, 1, 1), steps[0], 1.0, [0.0]

  def put(*contacts):
    c.step(t[0], 0.02, [[list(contacts)]], np.array([[[1.0, y, 0.09]]]), np.array([[1.0, 0.0]]), np.array([False]),
           np.array([True]))
    t[0] += 0.02

  flat = lambda k, dx=0.15: ((1.0 + 0.3 * (k - 1) + dx if k else 0.8, y, k * sh), 1.0)   # noqa: E731
  nose = lambda k: ((1.0 + 0.3 * (k - 1) + 0.001, y, k * sh), 0.6)                         # noqa: E731
  put(flat(0))
  for k in range(1, n + 1):
    if k in (2, 3):
      put(flat(k - 1, 0.28), nose(k))
      put(nose(k))
    if k == 2:   # back on its stand, the riser's face at axle height; then over the nose
      put(flat(1, 0.2), ((1.0 + 0.3 - 0.001, y, sh + 0.09), 0.0))
      put(flat(1, 0.28), nose(2))
    put(flat(k))
    put(flat(k))
  got = {k: int(v[0]) for k, v in c.counts(np.array([True])).items()}
  ok(f"unit wheeled counter among {len(steps)} flights ({len(g.levels)} grid levels): every step-up counts ({got})",
     got["ups"] == n + 1 and got["slips"] == 1 and got["edged"] == 2 and got["clean"] == n - 2)


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
