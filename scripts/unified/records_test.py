#!/usr/bin/env python3
"""records_test.py [OUT_DIR] (go2-depth 2026-11-12): the training records (src/tasks/unified/diagnostics.py) and
their reader (run_report.py) on made-up data. No simulator and no training: a stand-in environment and runner on the
CPU feed the real `episodes` term and the real Recorder a scripted "training" in which known things go wrong --

  run A (iterations 0-1199, then an exception): the walled_up ground fails 70% of its episodes on the left front
        calf from iteration 300 on; on the flat ground the robots with a payload fail 45% of theirs (the light ones
        5%), those with the payload at the front 75%; the "other" grounds' fail rate stays over the limit, so the
        reference schedule's first milestone ("walk") is overdue at its iteration; the action noise climbs from
        iteration 700; one constraint's multiplier reaches its limit while its reading stays short; the "up" family
        relaxes (the evaluation robots fail half of their climbs on their fixed rows) with no stairs constraint on;
        progress passes 0.5 with the quality stage waiting; the "training" ends in a RuntimeError at iteration 1199.
        The stairs quality stays 0, so the walled_up ground and the relaxed family raise no alert: before the
        quality stage a stairs ground's fail rate is the curriculum's own doing (~0.6 at the rows it promotes to)
  run B (resumed from A's checkpoint 1100, iterations 1100-1299): finishes normally; the milestones A reached
        before 1100 are carried over (reached once per training), "walk" is still overdue
  run C (iterations 0-559): the stairs quality stage from the start (q 0.6), the same walled_up ground and relaxed
        "up" family: now both are alerts and problems, judged on the evaluation robots' episodes on their fixed rows

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
from src.tasks.unified import diagnostics  # noqa: E402
from src.tasks.unified.teacher_env import constraint_items  # noqa: E402
from src.tasks.velocity.blind_env_cfg import MAP3_STAIR_TERRAIN_NAMES  # noqa: E402

N, K_EVAL, STEPS = 256, 32, 24
GROUNDS = {"stairs_down_narrow": 0.05, "stairs_down_wide": 0.05, "stairs_up_narrow": 0.10, "stairs_up_wide": 0.05,
           "random_rough": 0.10, "flat": 0.15, "walled_down": 0.20, "walled_up": 0.30}
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
            relax={"up": False, "down": False, "other": False},
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
                             "Diag/eval_fail/up_row0", "Diag/eval_fail/up_row8"} <= a.logger.writer.tags)
  cur = snaps[-2]["cur"]
  ok("A cur: the fixed rows", abs(cur["eval_fail"]["up"] - 0.5) < 1e-6 and cur["eval_fail_rows"]["up"][:3] == [0.1, 0.2, 0.3]
     and cur["eval_fail_rows"]["down"] == [None] * 9 and cur["clock_fail"] == 0.4 and cur["family_fail"]["up"] == 0.6)
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
  for text in ("walled_up 地面失败率高（评估机器人固定行上", "up 地面的约束一直在放松", "评估机器人固定行失败率 0.5",
               "评估机器人各固定行的失败率：0:10% 1:20% 2:30% 3:65%", "FL_calf"):
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

  bad = [name for name, good in checks if not good]
  print(f"\n{len(checks) - len(bad)} / {len(checks)} checks passed" + ("" if not bad else "; FAILED: " + "; ".join(bad)))
  print(f"records in {out}")
  return 1 if bad else 0


if __name__ == "__main__":
  sys.exit(main())
