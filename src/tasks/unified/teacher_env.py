"""The unified privileged teacher's environment (go2-depth 2026-11-11, the user's redesign; see robots.py for the
base it starts from). On top of the base:

1. Privileged actor. The depth teacher's actor (10 frames of proprioception + the elevated height scan) also sees
   the base velocity, the feet's heights, air times and contacts, and the robot's randomized
   physics (base CoM offset, payload, foot friction: mdp.dr_params, also added to the critic). Things a blind
   student can estimate from its history; not the cost terms' anchors (hold / drift offsets stay critic-only, so
   the teacher's holding is a behaviour the students can copy).
2. Rewards in three groups:
   CORE        fixed weights: tracking, gait clock, posture, termination, smoothness, swing clearance, climbing
               progress, flat speed;
   CONSTRAINED the cost terms serving an acceptance item; their weights are set by the constraints (constraints.py);
   OTHER       the remaining shaping terms of the recipe, kept at their weights ("full" variant) or removed ("lean":
               the user's "奖励只留十来项核心的").
3. One run through the curriculum (curriculum.py): terrain levels -> progress p -> stairs share, command ranges,
   payload limit, constraint switch-on and tightening. The recipe's iteration-based stairs share, command stages and
   old constraint set are replaced; the early cost ramps keep only the terms the constraints do not drive.
4. A zero-reward meter (constraints.meter) last in the rewards: the stairs event counts per step-up.
"""

from __future__ import annotations

import os
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
import torch
from mjlab.envs.mdp import dr
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

import src.tasks.velocity.mdp as mdp
from src.tasks.velocity.blind_env_cfg import MAP3_STAIR_TERRAIN_NAMES, TEACHER_STAIRS_SHARE

from src.metrics import mj as metrics_mj

from . import constraints as ucons
from . import curriculum as ucur
from .models import PRIV_GROUP
from .robots import URobot, base_env_cfg, unified_recipe

# (foot_contact_forces stays critic-only: world-frame vectors, see symmetry.py)
PRIVILEGED_ACTOR_TERMS = ("base_lin_vel", "foot_height", "foot_air_time", "foot_contact")

CORE = {
  "legged": (
    "track_linear_velocity", "track_angular_velocity", "foot_gait", "pose", "is_terminated", "joint_acc_l2",
    "joint_pos_limits", "action_rate_l2", "body_ang_vel", "lin_vel_z_even", "body_orientation",
    "foot_swing_clearance", "foot_terrain_clearance", "foothold_height_gain", "foothold_height_drop", "foot_slip",
    "flat_speed_deficit", "stance_leg_tilt",
  ),
  "wheeled": (
    "track_linear_velocity", "track_angular_velocity", "foot_gait", "pose", "is_terminated", "is_alive",
    "joint_acc_l2", "joint_pos_limits", "action_rate_l2", "body_ang_vel", "lin_vel_z_even", "body_orientation",
    "foot_swing_clearance", "foothold_height_gain", "wheel_riser_lift", "wheel_slip", "flat_speed_deficit",
    "stance_leg_tilt",
  ),
}


def constraint_items(kind: str) -> list[dict]:
  """The acceptance items as constraints (doc/acceptance.md; targets: the earlier constraint sets' calibrated
  targets for the acceptance-measured items, the delivered lineage's training level for the stairs rates)."""
  legged = kind == "legged"

  def c(name, metric, sense, target, loose, activate, terms, params=None, stat="mean", lr=0.03, min_scale=0.5,
        max_scale=3.0, start_scale=1.0, enforce=True):
    return {"name": name, "metric": metric, "params": params or {}, "stat": stat, "sense": sense, "target": target,
            "loose": loose, "activate": activate, "terms": terms, "lr": lr, "min_scale": min_scale,
            "max_scale": max_scale, "start_scale": start_scale, "enforce": enforce}

  drift = 0.06 if legged else 0.075   # p90 in training; the acceptance's 5% (calibrated by the blind chains)
  items = [
    # flat ground, from progress 0.15 (once the robots walk)
    c("clock", "gait_match", "min", 0.85 if legged else 0.82, 0.6, 0.15, ["foot_gait"], {"min_abs_vy": 0.5},
      min_scale=1.0),
    # the flat items on the light robots (<= 3 kg, the acceptance's unloaded tests); the heavy ones have their own
    # reward group (constraints.heavy_group, the user 2026-11-11)
    c("side_drift", "u_drift_light", "max", drift, 0.2, 0.15, ["side_drift"], {"term": "side_drift"}, stat="p90"),
    c("straight_drift", "u_drift_light", "max", drift, 0.2, 0.15, ["straight_drift"], {"term": "straight_drift"},
      stat="p90"),
    c("back_drift", "u_back_drift_light", "max", 0.03, 0.1, 0.15, ["side_drift"],
      {"term": "side_drift", "settled": True, "min_abs_vy": 0.45}),
    # creep and march over the light robots only (<= 25% of the payload limit, as the acceptance's unloaded tests)
    c("creep", "u_stand_hold_light", "max", 0.008, 0.03, 0.15, ["standing_position_hold", "standing_speed"],
      {"term": "standing_position_hold"}, stat="p90"),
    c("march", "u_stand_hold_light", "max", 0.04, 0.15, 0.15, ["march_position_hold", "march_hip_default"],
      {"term": "march_position_hold"}, stat="p90"),
    c("spin", "u_spin_light", "max", 0.05, 0.15, 0.15, ["spin_position_hold"]),
    # stairs, from progress 0.5 (stairs share rising), at 0.25 of their weight before
    c("descent", "stair_speed", "min", 0.9, 0.6, 0.5, ["stair_speed_deficit"], {"descend": True}, start_scale=0.25),
    c("climb", "stair_speed", "min", 0.85, 0.5, 0.5, ["stair_speed_deficit"], {"descend": False}, start_scale=0.25),
    c("leg_contact", "leg_contact", "max", 0.015, 0.1, 0.5, ["leg_ground_contact"], start_scale=0.25),
    c("stop_feet", "u_stairs_stop_feet", "min", 0.9, 0.6, 0.5, ["feet_stand_uneven"], start_scale=0.25),
    # logged only: the roll bar is not calibrated (the user, 11-11: calibrate first)
    c("roll", "u_stairs_roll", "max", 0.105, 0.3, 0.5, [], stat="p90", enforce=False),
  ]
  if legged:
    # the shared metrics (src/metrics, tinynav's counters; 2026-11-12, doc/redesign.md): per step walked, the tinynav
    # bars over its 8.15 steps per run (gp3 avg15700's routes): kicks 2 / run -> 0.245, slips 0.2 / run -> 0.0245;
    # placements mid >= 80%, edge <= 10% as in tinynav
    items += [
      c("edge", "lib:edge", "max", 0.10, 0.25, 0.5, ["feet_edge"], start_scale=0.25),
      c("mid", "lib:mid", "min", 0.80, 0.6, 0.5, ["feet_edge"], start_scale=0.25),
      c("slides", "lib:slips", "max", 0.0245, 0.1, 0.5, ["foot_slide_back"], start_scale=0.25),
      c("kicks", "lib:kicks", "max", 0.245, 0.6, 0.5, ["foot_stumble", "foot_step_touch"], start_scale=0.25),
    ]
  else:
    items += [
      c("wheels_down", "wheels_down", "min", 0.75, 0.5, 0.15, ["wheel_air_drivable"]),
      c("slow_arc", "arc_speed", "min", 0.9, 0.7, 0.15, ["track_linear_velocity"], min_scale=1.0),
      # slide-back from the shared metrics (tinynav's counter, 2026-11-12): the acceptance's 5% of step-ups. The idle
      # and top-landing rates stay on the meter's event terms until tinynav's per-climb bars are converted (wheeled
      # steps per run not measured yet).
      c("slides", "lib:slide_back", "max", 0.05, 0.15, 0.5, ["wheel_slide_back", "wheel_nosing"], start_scale=0.25),
      c("top_landing", "rate:top", "max", 0.033, 0.1, 0.5, ["wheel_flat_step_air"], start_scale=0.25),
      # As2W a1f (passed; tinynav idle 0.0-0.3 / climb): 0.6 idle lifts per step-up in training
      c("idle", "rate:idle", "max", 0.6, 1.5, 0.5, ["wheel_idle_step"], start_scale=0.25),
    ]
  return items


METER_EVENTS = {
  "legged": {"slide": "foot_slide_back", "kick": "foot_stumble"},
  "wheeled": {"slide": "wheel_slide_back", "top": "wheel_flat_step_air", "idle": "wheel_idle_step"},
}


def _add_privileged(cfg: ManagerBasedRlEnvCfg) -> None:
  """The privileged terms as their own observation group (models.UPrivLatentModel encodes it into a compact vector;
  2026-11-12, doc/redesign.md 3.5)."""
  critic = cfg.observations["critic"].terms
  if "dr_params" not in critic:
    critic["dr_params"] = ObservationTermCfg(
      func=mdp.dr_params, params={"friction_cfg": cfg.events["foot_friction"].params["asset_cfg"]}
    )
  priv = {}
  for name in PRIVILEGED_ACTOR_TERMS + ("dr_params",):
    term = deepcopy(critic[name])
    term.noise = None
    term.history_length = 0
    priv[f"priv_{name}"] = term
  cfg.observations[PRIV_GROUP] = ObservationGroupCfg(terms=priv, concatenate_terms=True, enable_corruption=False)


# ---- scan masking: the blind mode (2026-11-12, doc/redesign.md 3.5) ----
SCAN_MASK_SHARE = 0.4   # of the episodes, the actor's scan is masked: the teacher also learns to walk without it


def u_masked(env, inner, **params) -> torch.Tensor:
  """An observation term's value times the episode's scan flag (0: masked)."""
  x = inner(env, **params)
  on = getattr(env, "u_scan_on", None)
  return x if on is None else x * on[:, None]


def scan_flag(env) -> torch.Tensor:
  on = getattr(env, "u_scan_on", None)
  if on is None:
    return torch.ones(env.num_envs, 1, device=env.device)
  return on[:, None]


def u_scan_mask(env, env_ids, share: float = SCAN_MASK_SHARE, mode: str | None = None) -> None:
  """Reset event: each episode is seen (1) or blind (0). mode "on" / "off" (the students' tasks) or U_SCAN=on / off
  (evaluations) fixes it; else a share of the episodes is blind."""
  if getattr(env, "u_scan_on", None) is None:
    env.u_scan_on = torch.ones(env.num_envs, device=env.device)
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  mode = mode or os.environ.get("U_SCAN", "mix")
  if mode == "on":
    v = torch.ones(len(env_ids), device=env.device)
  elif mode == "off":
    v = torch.zeros(len(env_ids), device=env.device)
  else:
    v = (torch.rand(len(env_ids), device=env.device) >= share).float()
  env.u_scan_on[env_ids] = v


def _add_scan_mask(cfg: ManagerBasedRlEnvCfg) -> None:
  from src.tasks.velocity.depth_env_cfg import TEACHER_SCAN_NAME

  actor = cfg.observations["actor"].terms
  term = actor[TEACHER_SCAN_NAME]
  assert not isinstance(term.func, type), "the actor scan term is a class; u_masked wraps functions"
  term.params = {**term.params, "inner": term.func}
  term.func = u_masked
  actor["scan_flag"] = ObservationTermCfg(func=scan_flag)
  cfg.observations["critic"].terms["scan_flag"] = ObservationTermCfg(func=scan_flag)
  cfg.events["u_scan_mask"] = EventTermCfg(mode="reset", func=u_scan_mask, params={})


# ---- joint friction and damping, randomized (2026-11-12, doc/redesign.md 3.3) ----
# Not published for Go2 / Go2W / B2 / B2W; Unitree's as2.xml gives friction loss 1 / 1 / 2 N m and damping 0.2 / 0.2 /
# 0.5 N m s/rad on its 60 / 60 / 90 N m joints. Uniform from 0 to these shares of each joint group's torque limit
# (friction 2.5%: As2's 1.7-2.2% inside; damping 1.5x As2's ratios), until real-robot measurements narrow them.
JOINT_FRICTION_SHARE = 0.025
JOINT_DAMPING_SHARE = {"hip": 1.5 * 0.2 / 60, "thigh": 1.5 * 0.2 / 60, "calf": 1.5 * 0.5 / 90}


def _effort(actuator) -> float | None:
  a = actuator
  while not hasattr(a, "effort_limit") and hasattr(a, "base_cfg"):
    a = a.base_cfg
  return getattr(a, "effort_limit", None)


def _add_joint_dr(cfg: ManagerBasedRlEnvCfg) -> None:
  for a in cfg.scene.entities["robot"].articulation.actuators:
    expr = tuple(getattr(a, "target_names_expr", ()) or getattr(getattr(a, "base_cfg", None), "target_names_expr", ()))
    group = next((g for g in ("hip", "thigh", "calf") if any(g in e for e in expr)), None)
    effort = _effort(a)
    if group is None or effort is None:
      continue   # wheels: their brake / drive model stays as it is
    # one SceneEntityCfg per event: resolving writes the joint ids into it, and a shared one fails the second check
    cfg.events[f"u_joint_friction_{group}"] = EventTermCfg(
      mode="startup", func=dr.joint_friction,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=expr), "ranges": (0.0, JOINT_FRICTION_SHARE * effort),
              "operation": "abs"})
    cfg.events[f"u_joint_damping_{group}"] = EventTermCfg(
      mode="startup", func=dr.joint_damping,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=expr),
              "ranges": (0.0, JOINT_DAMPING_SHARE[group] * effort), "operation": "abs"})


def _rewards(cfg: ManagerBasedRlEnvCfg, kind: str, items: list[dict], lean: bool) -> tuple[list[str], list[str]]:
  rewards = cfg.rewards
  constrained = sorted({t for c in items for t in c["terms"]})
  missing = [t for t in constrained + list(CORE[kind]) if t not in rewards]
  assert not missing, f"terms not in the recipe's environment: {missing}"
  other = [t for t in rewards if t not in constrained and t not in CORE[kind]]
  if lean:
    for t in other:
      del rewards[t]
  # the heavy robots' reward group (after the terms it reads): each flat item's costs at their configured weight on
  # the robots with >= 8 kg, whatever the light robots' multipliers (constraints.heavy_group)
  for item, (_, _, _, terms) in ucons.HEAVY_ITEMS.items():
    terms = [t for t in terms if t in rewards]
    if terms:
      rewards[f"u_heavy_{item}"] = RewardTermCfg(func=ucons.heavy_group, weight=1.0, params={"terms": terms})
  rewards["u_meter"] = RewardTermCfg(func=ucons.meter, weight=1e-9, params={"events": METER_EVENTS[kind]})
  # the shared stairs metrics (tinynav's counters) over 512 robots, for the "lib:" constraints
  rewards["u_lib_meter"] = RewardTermCfg(func=ucons.lib_meter, weight=1e-9,
                                         params={"wheeled": kind == "wheeled", "sample": 512})
  return constrained, other


def _curriculum(cfg: ManagerBasedRlEnvCfg, robot: URobot, items: list[dict], constrained: list[str],
                dropped: list[str]) -> None:
  old = cfg.curriculum
  stages = old["command_vel"].params["velocity_stages"]
  keys = set(unified_recipe(robot)["P"])
  final_share = 0.5 if "stairs50" in keys else 0.65
  new: dict[str, CurriculumTermCfg] = {"u_progress": CurriculumTermCfg(func=ucur.progress, params={})}
  for name, term in old.items():
    if name == "command_vel":
      first, last = stages[0], stages[-1]
      new["u_commands"] = CurriculumTermCfg(func=ucur.progress_commands, params={
        "command_name": "twist",
        "initial": {k: first[k] for k in ("lin_vel_x", "lin_vel_y", "ang_vel_z") if first.get(k) is not None},
        "final": {k: last[k] for k in ("lin_vel_x", "lin_vel_y", "ang_vel_z") if last.get(k) is not None},
      })
    elif name == "stairs_share":
      new["u_stairs_share"] = CurriculumTermCfg(func=ucur.progress_stairs_share, params={
        "stair_terrain_names": MAP3_STAIR_TERRAIN_NAMES, "start": TEACHER_STAIRS_SHARE, "end": final_share,
      })
    elif name == "constraints":
      continue   # the recipe's constraint set; the teacher's own comes last
    elif name == "ramps":
      skip = set(constrained) | set(dropped)
      ramps = [r for r in term.params["ramps"] if not (r["kind"].startswith("reward") and r["name"] in skip)]
      new[name] = CurriculumTermCfg(func=term.func, params={**term.params, "ramps": ramps})
    else:
      new[name] = term
  payload = cfg.events["base_payload"].params
  payload["ramp_steps"] = 0   # the event's own iteration ramp off: the progress sets the limit
  new["u_payload"] = CurriculumTermCfg(func=ucur.progress_payload, params={"event_name": "base_payload",
                                                                         "full_max": payload["heavy_max"]})
  new["u_constraints"] = CurriculumTermCfg(func=ucons.constraints, params={"items": items})
  new["u_heavy"] = CurriculumTermCfg(func=ucons.heavy_log, params={})   # light vs heavy readings, logged only
  if os.environ.get("U_SMOKE") == "1":
    # A few-iteration smoke run (jobs/unified/teacher.sh --smoke): measure from the start, multipliers move after a
    # few samples, the meter reports after 50 steps; the progress is forced along so every switch-on runs.
    new["u_progress"].params = {"skip_steps": 0, "ema_steps": 50.0, "force_rate": 0.002}
    new["u_constraints"].params.update({"skip_steps": 0, "warmup": 5, "ema_steps": 50.0})
    new["u_heavy"].params.update({"skip_steps": 0, "ema_steps": 50.0})
    cfg.rewards["u_meter"].params["min_n"] = 50
    cfg.rewards["u_lib_meter"].params.update({"min_steps": 5.0, "ema_steps": 200.0})
  cfg.curriculum = new


def unified_teacher_env_cfg(robot: URobot, play: bool = False, lean: bool = False,
                            student: str | None = None) -> ManagerBasedRlEnvCfg:
  """student "blind" / "depth" (students.py): the teacher's environment with the scan fixed off / on, the curriculum
  at its end (progress 1) and the constraint scales frozen at the teacher's (restored from its checkpoint)."""
  cfg = base_env_cfg(robot, play=play)
  metrics_mj.add_sensor(cfg)   # the shared metrics' contact sensor (points and normals)
  _add_privileged(cfg)
  _add_scan_mask(cfg)
  _add_joint_dr(cfg)
  items = constraint_items(robot.kind)
  constrained, other = _rewards(cfg, robot.kind, items, lean)
  if not play:
    _curriculum(cfg, robot, items, constrained, other if lean else [])
  if student is not None:
    cfg.events["u_scan_mask"].params["mode"] = "off" if student == "blind" else "on"
    if "u_progress" in cfg.curriculum:
      cfg.curriculum["u_progress"].params["fixed_p"] = 1.0
      cfg.curriculum["u_constraints"].params["frozen"] = True
  return cfg
