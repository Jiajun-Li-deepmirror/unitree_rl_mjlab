"""The students' step-clocked training schedules, at their end (go2-depth 2026-11-13).

The students train in the teacher's environment "at its end" (students.py's docstring): the unified curriculum is
fixed at progress 1 and the constraint scales are the teacher's final ones
(teacher_env.unified_teacher_env_cfg(student=...), students.restore_teacher_curriculum). That covers the schedules
the unified code runs on its progress. The blind builder's own schedules run on the environment's step clock
(mdp/curriculums.py: reward_params' stages by env.common_step_counter, ramp_terms' ramps by the staged clock) -- and a
student's environment is a new one, its clock starts at 0, in each of its stages (a stage is a new process). So the
students trained under the values of the teacher's FIRST iterations, not its last:

  twist.march_prob (the share of the zero commands that march in place)   0.05   the teacher ends at 0.5   (1000 -> 2500 it)
  track_linear_velocity / track_angular_velocity std                      0.5 / 0.707    ...  0.35 / 0.45  (from 2000 it)
  foot_swing_clearance weight (legged)                                    -0.5           ...  -3.0         (500 -> 3000 it)

Seen in the student probes' records of 11-13 (#3's two stages, SA, S2H: Curriculum/ramps/twist.march_prob 0.050 and
foot_swing_clearance.weight -0.5 at every iteration, the tracking stds 0.5 / 0.707): stage 1 imitated the teacher on a
tenth of its marching share, stage 2's PPO pulled toward the loose first-iterations rewards. The blind students' flat
readings (line drift, spin, the march's drift) stayed outside the bars the teacher is inside of.

at_end(cfg) puts every such schedule of a training configuration at its end from step 0: the stages apply from the
first step, the ramps are at their end value, and the terms they move carry the end value in the configuration too
(so nothing depends on the first curriculum call). The curriculum terms stay: Curriculum/... in a student's log says
the value in effect. A step-clocked schedule this does not know stops the build (a new one must not start over
unseen). Called by students.student_env_cfg for the training builds only (the checks' terrains carry no training
schedule; the teacher's builds are not touched: scripts/unified/student_sched_test.py, recipe_test.py).
"""

from __future__ import annotations

from copy import deepcopy

# the step-clocked schedules' own keys (mdp/curriculums.py): reward_params / reward_weight / commands_vel stages
# ("step"), ramp_terms' and stairs_share's ramps ("start_step", "end_step")
STEP_KEYS = ("step", "start_step", "end_step")
STAGES = ("param_stages", "weight_stages", "velocity_stages")


def _left(x, path: str, out: list[str]) -> list[str]:
  """The step keys with a step after 0 anywhere in x (what a schedule still waits for)."""
  if isinstance(x, dict):
    for k, v in x.items():
      if k in STEP_KEYS and isinstance(v, (int, float)) and v > 0:
        out.append(f"{path}.{k}={v}")
      _left(v, f"{path}.{k}", out)
  elif isinstance(x, (list, tuple)):
    for i, v in enumerate(x):
      _left(v, f"{path}[{i}]", out)
  return out


def at_end(cfg) -> list[str]:
  """Every step-clocked schedule of the training configuration at its end from step 0 (module docstring). Returns
  what it set, one line each. Works on copies of the terms it changes: the builders may share them between builds."""
  done: list[str] = []
  if not cfg.curriculum:
    return done

  def reward(name: str):
    cfg.rewards[name] = deepcopy(cfg.rewards[name])
    return cfg.rewards[name]

  for name in list(cfg.curriculum):
    term = cfg.curriculum[name]
    params = getattr(term, "params", None)
    if not isinstance(params, dict) or not any(k in params for k in STAGES + ("ramps",)):
      continue
    term = deepcopy(term)
    cfg.curriculum[name] = term
    p = term.params
    if "param_stages" in p:    # mdp.reward_params: each stage holds once the step counter is past its step
      p["param_stages"] = [{**s, "step": -1} for s in p["param_stages"]]
      r = reward(p["reward_name"])
      for s in p["param_stages"]:   # in the list's order, as the term applies them: later stages win
        r.params.update(s["params"])
        done.append(f"{p['reward_name']}: {s['params']}")
    if "weight_stages" in p:   # mdp.reward_weight: the same, on the weight
      p["weight_stages"] = [{**s, "step": -1} for s in p["weight_stages"]]
      r = reward(p["reward_name"])
      for s in p["weight_stages"]:
        r.weight = s["weight"]
        done.append(f"{p['reward_name']}.weight: {s['weight']}")
    if "velocity_stages" in p:   # mdp.commands_vel: the ranges of the last stage from the first step
      p["velocity_stages"] = [{**s, "step": -1} for s in p["velocity_stages"]]
      done.append(f"{name}: {len(p['velocity_stages'])} velocity stage(s) from step 0")
    if "ramps" in p:           # mdp.ramp_terms: start_step = end_step = 0 is the end value at every step
      ramps = []
      for r in p["ramps"]:
        end = r["end"]
        ramps.append({**r, "start": end, "start_step": 0, "end_step": 0})
        if r["kind"] == "reward_weight":
          reward(r["name"]).weight = end
          done.append(f"{r['name']}.weight: {end}")
        elif r["kind"] == "reward_param":
          reward(r["name"]).params[r["key"]] = end
          done.append(f"{r['name']}.{r['key']}: {end}")
        elif r["kind"] == "command":
          cfg.commands[r["name"]] = deepcopy(cfg.commands[r["name"]])
          setattr(cfg.commands[r["name"]], r["key"], end)
          done.append(f"{r['name']}.{r['key']}: {end}")
        else:
          raise ValueError(f"ramp kind {r['kind']!r}")
      p["ramps"] = ramps
  left = []
  for name, term in cfg.curriculum.items():
    _left(getattr(term, "params", None), f"curriculum/{name}", left)
  for name, term in (cfg.events or {}).items():   # events.base_payload_point's mass ramp by the step counter
    steps = (getattr(term, "params", None) or {}).get("ramp_steps", 0)
    if steps and steps > 0:
      left.append(f"events/{name}.ramp_steps={steps}")
  assert not left, f"a step-clocked schedule the students' build does not put at its end: {left}"
  return done
