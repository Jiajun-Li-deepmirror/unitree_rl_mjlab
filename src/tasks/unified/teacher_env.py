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
   The stairs quality comes in its own stage (2026-11-12; curriculum.py's q, which starts once the robots climb):
   until then the rewards are the recipe's M-key ones -- quality_diff() lists the weights and parameters its P keys
   change -- and the stairs constraints are off; q moves the rewards to the P values and the constraints to their bars.
4. A zero-reward meter (constraints.meter) last in the rewards: the stairs event counts per step-up.
5. Two scan modes (u_scan_mask): a share of the episodes walks blind, the scan masked. The depth student copies the
   seen mode and the blind student the blind one, so every constraint is measured in both (the evaluation robots: a
   fixed half in each) and the gate asks both to pass (constraints.py).
6. Joint friction, damping and armature randomized per robot (_add_joint_dr).
7. The payload's mount point from a wide box (_payload_point; the robot table's payload_point): the mount on the
   final machine is not known.
8. Two sloped tiles among the other ground (_add_slopes): readings only, no acceptance item.
9. Static friction above kinetic friction on the feet and wheels (stick_slip.py), in the checks' environments too.
"""

from __future__ import annotations

import inspect
import os
from copy import deepcopy
from dataclasses import replace

import mjlab.terrains as terrain_gen
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
from . import diagnostics
from . import stick_slip
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

  def c(name, metric, sense, target, loose, activate, terms, params=None, stat="mean", lr=ucons.LR, min_scale=0.5,
        max_scale=3.0, start_scale=1.0, enforce=True, family="other", modes=ucons.MODES):
    # activate "q": a stairs-quality item, on the quality stage instead of the progress (constraints.py)
    # target: the students' bar (the gate judges there); tight: the teacher's own target, a margin inside it
    # family: the ground whose fail rate relaxes the item; modes: the scan modes it applies to (constraints.py)
    by = "q" if activate == "q" else "p"
    assert family in ucons.FAMILIES and set(modes) <= set(ucons.MODES), name
    item = {"name": name, "metric": metric, "params": params or {}, "stat": stat, "sense": sense, "target": target,
            "loose": loose, "activate": 0.0 if by == "q" else activate, "by": by, "terms": terms, "lr": lr,
            "min_scale": min_scale, "max_scale": max_scale, "start_scale": start_scale, "enforce": enforce,
            "family": family, "modes": tuple(modes)}
    if enforce:
      item["tight"] = ucons.with_margin(sense, target)
    return item

  drift = 0.06 if legged else 0.075   # p90 in training; the acceptance's 5% (calibrated by the blind chains)
  items = [
    # flat ground, from progress 0.15 (once the robots walk)
    c("clock", "gait_match", "min", 0.85 if legged else 0.82, 0.6, 0.15, ["foot_gait"], {"min_abs_vy": 0.5},
      min_scale=1.0),
    # the flat items on the light robots (<= 25% of the payload limit: the acceptance's unloaded tests); the heavy
    # ones (>= 2/3 of it) have their own
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
    # stairs: the quality stage (q; off and at the recipe's M weights until the robots climb)
    # (family: descent relaxes on the descents' fail rate, climb on the climbing columns', the rest on the worse)
    c("descent", "stair_speed", "min", 0.9, 0.6, "q", ["stair_speed_deficit"], {"descend": True}, family="down"),
    c("climb", "stair_speed", "min", 0.85, 0.5, "q", ["stair_speed_deficit"], {"descend": False}, family="up"),
    c("leg_contact", "leg_contact", "max", 0.015, 0.1, "q", ["leg_ground_contact"], family="stairs"),
    c("stop_feet", "u_stairs_stop_feet", "min", 0.9, 0.6, "q", ["feet_stand_uneven"], family="stairs"),
    # logged only: the roll bar is not calibrated (the user, 11-11: calibrate first)
    c("roll", "u_stairs_roll", "max", 0.105, 0.3, "q", [], stat="p90", enforce=False, family="stairs"),
  ]
  if legged:
    # the shared metrics (src/metrics, tinynav's counters; 2026-11-12, doc/redesign.md): per step walked, the tinynav
    # bars over its 8.15 steps per run (gp3 avg15700's routes): kicks 2 / run -> 0.245, slips 0.2 / run -> 0.0245;
    # placements mid >= 80%, edge <= 10% as in tinynav. The placement items apply to the seen mode only: where a
    # foot lands on the tread is the scan's work (the blind mode's reading is logged, value_blind, and the blind
    # student's own acceptance judges it); kicks and slips apply to both, at the same bars (doc/acceptance.md: one
    # standard for both students)
    items += [
      c("edge", "lib:edge", "max", 0.10, 0.25, "q", ["feet_edge"], family="stairs", modes=("seen",)),
      c("mid", "lib:mid", "min", 0.80, 0.6, "q", ["feet_edge"], family="stairs", modes=("seen",)),
      c("slides", "lib:slips", "max", 0.0245, 0.1, "q", ["foot_slide_back"], family="stairs"),
      c("kicks", "lib:kicks", "max", 0.245, 0.6, "q", ["foot_stumble", "foot_step_touch"], family="stairs"),
    ]
  else:
    items += [
      c("wheels_down", "wheels_down", "min", 0.75, 0.5, 0.15, ["wheel_air_drivable"]),
      c("slow_arc", "arc_speed", "min", 0.9, 0.7, 0.15, ["track_linear_velocity"], min_scale=1.0),
      # slide-back from the shared metrics (tinynav's counter, 2026-11-12): the acceptance's 5% of step-ups. The idle
      # and top-landing rates stay on the meter's event terms until tinynav's per-climb bars are converted (wheeled
      # steps per run not measured yet).
      c("slides", "lib:slide_back", "max", 0.05, 0.15, "q", ["wheel_slide_back", "wheel_nosing"], family="up"),
      c("top_landing", "rate:top", "max", 0.033, 0.1, "q", ["wheel_flat_step_air"], family="up"),
      # As2W a1f (passed; tinynav idle 0.0-0.3 / climb): 0.6 idle lifts per step-up in training
      c("idle", "rate:idle", "max", 0.6, 1.5, "q", ["wheel_idle_step"], family="stairs"),
      # the shared metrics' idle steps (tinynav's counter: before the first riser / after the last, per step-up),
      # logged only, no bar, and not to be read yet: in training the counter has no start level (metrics/mj.py builds
      # it without one), so idle_start reads 0 and every idle lift off the highest level reached so far counts as
      # idle_end; the readings first have to be set against tinynav's per-climb counts on a delivered policy
      c("idle_start", "lib:idle_start", "max", None, None, "q", [], enforce=False, family="stairs"),
      c("idle_end", "lib:idle_end", "max", None, None, "q", [], enforce=False, family="stairs"),
    ]
  return items


# A term the recipe's M keys do not have starts the quality stage at this share of its P weight: next to nothing in
# the reward, but not 0 -- the reward manager skips zero-weight terms, and the event counters would stop.
Q_EPS = 1e-3


def _m_default(func, key: str):
  """The value a reward term uses for a parameter its cfg does not set: its function's (a class's __call__) default."""
  sig = inspect.signature(func.__call__ if isinstance(func, type) else func)
  par = sig.parameters.get(key)
  assert par is not None and par.default is not inspect.Parameter.empty, f"{func.__name__}: no default for {key!r}"
  return par.default


def _same(a, b) -> bool:
  try:
    return bool(a == b)
  except Exception:
    return repr(a) == repr(b)


def quality_diff(robot: URobot, rewards: dict, constrained_p: set[str], constrained_q: set[str],
                 dropped: tuple[str, ...] = (), play: bool = False, terrain: str = "map3") -> tuple[dict, dict, dict]:
  """What the recipe's P keys change in the rewards against its M keys (`rewards`: the P environment's; `dropped`:
  the terms the lean variant took out of it). Returns
    weights  {term: (M weight, P weight)}   terms no constraint scales; a P-only term starts at Q_EPS x its weight
    params   {term: {key: (M value, "lerp" | "switch")}}   numbers are interpolated, anything else switches
    start    {term: M weight}               for the terms the stairs-quality constraints scale (Q_EPS x for P-only)
  constraints.quality_terms and the "by q" constraints apply them along the stairs quality q."""
  m = base_env_cfg(robot, play=play, terrain=terrain, keys="M").rewards
  only_m = [t for t in m if t not in rewards and t not in dropped and not t.startswith("u_")]
  weights: dict = {}
  params: dict = {}
  start: dict = {}
  for t, tp in rewards.items():
    if t.startswith("u_"):
      continue
    tm = m.get(t)
    w_m = tm.weight if tm is not None else Q_EPS * tp.weight
    if t in constrained_q:
      start[t] = w_m
    elif w_m != tp.weight:
      # a flat item's term keeps its weight x multiplier from the first iteration: it must not differ
      assert t not in constrained_p, f"{robot.name}: {t} is scaled by a flat constraint and differs between M and P"
      weights[t] = (w_m, tp.weight)
    if tm is None:
      continue
    gone = [k for k in tm.params if k not in tp.params]
    assert not gone, f"{robot.name}: {t} has parameters under the M keys only: {gone}"
    for key, b in tp.params.items():
      a = tm.params[key] if key in tm.params else _m_default(tp.func, key)
      if _same(a, b):
        continue
      num = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (a, b))
      assert not isinstance(a, SceneEntityCfg), f"{robot.name}: {t}.{key}: an M scene entity is not supported"
      params.setdefault(t, {})[key] = (a, "lerp" if num else "switch")
  # (terms only the M keys have would stay out of the teacher; none in the present recipes)
  assert not only_m, f"{robot.name}: reward terms under the M keys only: {only_m}"
  return weights, params, start


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
  (evaluations) fixes it; else a share of the episodes is blind -- and the evaluation robots (env.u_eval_mask, the
  teacher's runner) keep one mode each, even index seen and odd blind, so each mode's constraints read half of them
  whatever the share (their columns and rows are drawn anew every episode, the same for both halves)."""
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
    ev = getattr(env, "u_eval_mask", None)
    if ev is not None:
      v = torch.where(ev[env_ids], (env_ids % 2 == 0).float(), v)
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


# ---- joint friction, damping and armature, randomized (2026-11-12, doc/redesign.md 3.3) ----
# Not published for Go2 / Go2W / B2 / B2W; Unitree's as2.xml gives friction loss 1 / 1 / 2 N m and damping 0.2 / 0.2 /
# 0.5 N m s/rad on its 60 / 60 / 90 N m joints. Uniform from 0 to these shares of each joint group's torque limit
# (friction 2.5%: As2's 1.7-2.2% inside; damping 1.5x As2's ratios), until real-robot measurements narrow them.
JOINT_FRICTION_SHARE = 0.025
JOINT_DAMPING_SHARE = {"hip": 1.5 * 0.2 / 60, "thigh": 1.5 * 0.2 / 60, "calf": 1.5 * 0.5 / 90}
# The legs' rotor inertia (armature): each joint's model value x a uniform factor per robot. The models' values are
# estimates (Go2 / Go2W 0.01 / 0.01 / 0.02 kg m^2 for hip / thigh / calf, B2 / B2W 0.1, As2 0.026 / 0.026 / 0.038; none
# measured on our robots), +-30% until they are.
JOINT_ARMATURE_SCALE = (0.7, 1.3)


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
    cfg.events[f"u_joint_armature_{group}"] = EventTermCfg(
      mode="startup", func=dr.joint_armature,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=expr), "ranges": JOINT_ARMATURE_SCALE,
              "operation": "scale"})


def _payload_point(cfg: ManagerBasedRlEnvCfg, robot: URobot) -> None:
  """The payload's mount point drawn from the robot table's wide box (robot.payload_point; 2026-11-12, the user:
  "负载的重心偏移可以尽可能的大一些，因为我也不知道之后整机上会装在哪里"), in place of the blind chains' range. The
  event's corner share (heavy robots at the box's front or rear end, at its top) follows the box. Where the robots
  fall with it: the records' breakdown by payload position (diagnostics.py)."""
  if robot.payload_point is None:
    return
  ev = cfg.events["base_payload"]
  assert ev.func is mdp.base_payload_point, "the training terrain's payload event is not the payload point"
  box = tuple((float(lo), float(hi)) for lo, hi in robot.payload_point)
  assert len(box) == 3 and all(lo <= hi for lo, hi in box), f"{robot.name}: payload_point {robot.payload_point}"
  ev.params["pos_ranges"] = box


# ---- slopes (2026-11-12; the user, through the students' session: "斜坡可以稍微加一点，你定") ----
SLOPE_SHARE = 0.10   # of the episodes on the other ground (not stairs), half on each sloped tile
SLOPE_MAX = 0.25     # the top row's gradient along the tile's axes (14 deg; 16 deg at most, between them)


def _add_slopes(cfg: ManagerBasedRlEnvCfg) -> dict[str, float]:
  """Two sloped tiles among the other ground: a pyramid hill (slope_down: the robot starts on its flat top, 3.5 m
  across along the tile's axes and 2 m square between them) and a pit (slope_up: at its bottom), one column each out
  of the flat ground's three, the gradient rising with the other ground's row from 0 to SLOPE_MAX (the slope runs
  2.25 m on from the flat part, 0.56 m of height at the top row; a promotion takes 3 m walked, so 1.25 m of it or
  more). No acceptance item is measured on them (constraints.off_slopes
  leaves their robots out; the cost terms still act there): the records show them as grounds of their own.
  Returns the other ground's episode shares per tile name for the column draw (curriculum.progress_stairs_share):
  the sloped tiles SLOPE_SHARE together, the rest split as the columns were before."""
  gen = cfg.scene.terrain.terrain_generator
  subs = gen.sub_terrains
  others = {n: s.proportion for n, s in subs.items() if n not in MAP3_STAIR_TERRAIN_NAMES}
  assert set(others) == {"random_rough", "flat"}, f"the other ground: {sorted(others)}"
  step = sum(s.proportion for s in subs.values()) / gen.num_cols   # one column
  assert subs["flat"].proportion > 2.5 * step, "the flat ground keeps a column"
  hill = terrain_gen.HfPyramidSlopedTerrainCfg(proportion=step, slope_range=(0.0, SLOPE_MAX), platform_width=2.0)
  new = {}
  for name, sub in subs.items():
    if name == "flat":   # before the flat tile, which stays next to the walled ones (blind_env_cfg's column order)
      new[ucur.SLOPE_NAMES[0]] = hill
      new[ucur.SLOPE_NAMES[1]] = replace(hill, inverted=True)
      sub = replace(sub, proportion=sub.proportion - 2 * step)
    new[name] = sub
  gen.sub_terrains = new
  total = sum(others.values())
  shares = {n: (1.0 - SLOPE_SHARE) * p / total for n, p in others.items()}
  shares |= {n: SLOPE_SHARE / 2 for n in ucur.SLOPE_NAMES}
  return shares


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
  # the robots with >= 2/3 of the payload limit, whatever the light robots' multipliers (constraints.heavy_group)
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
                dropped: list[str], quality: tuple[dict, dict] = ({}, {}),
                other_shares: dict[str, float] | None = None) -> None:
  old = cfg.curriculum
  stages = old["command_vel"].params["velocity_stages"]
  keys = set(unified_recipe(robot)["P"])
  final_share = 0.5 if "stairs50" in keys else 0.65
  new: dict[str, CurriculumTermCfg] = {"u_progress": CurriculumTermCfg(func=ucur.progress, params={})}
  # the training records' count of every finished episode (diagnostics.py): before the terms that redraw the robot's
  # column and row, so it still sees where the episode ran
  new["u_episodes"] = CurriculumTermCfg(func=diagnostics.episodes,
                                        params={"stair_terrain_names": MAP3_STAIR_TERRAIN_NAMES})
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
        "other_shares": other_shares,
      })
    elif name == "constraints":
      continue   # the recipe's constraint set; the teacher's own comes last
    elif name == "ramps":
      skip = set(constrained) | set(dropped)
      ramps = [r for r in term.params["ramps"] if not (r["kind"].startswith("reward") and r["name"] in skip)]
      new[name] = CurriculumTermCfg(func=term.func, params={**term.params, "ramps": ramps})
      # the quality stage and the recipe's iteration ramps must not drive the same weight or parameter
      q_weights, q_params = quality
      clash = [r for r in ramps if (r["kind"] == "reward_weight" and r["name"] in q_weights)
               or (r["kind"] == "reward_param" and r["key"] in q_params.get(r["name"], {}))]
      assert not clash, f"{robot.name}: ramped by the recipe and by the quality stage: {clash}"
    else:
      new[name] = term
  payload = cfg.events["base_payload"].params
  payload["ramp_steps"] = 0   # the event's own iteration ramp off: the progress sets the limit
  new["u_payload"] = CurriculumTermCfg(func=ucur.progress_payload, params={"event_name": "base_payload",
                                                                         "full_max": payload["heavy_max"]})
  if quality[0] or quality[1]:   # the M -> P reward differences, along the stairs quality q
    new["u_quality"] = CurriculumTermCfg(func=ucons.quality_terms, params={"weights": quality[0], "params": quality[1]})
  new["u_constraints"] = CurriculumTermCfg(func=ucons.constraints, params={"items": items})
  new["u_heavy"] = CurriculumTermCfg(func=ucons.heavy_log, params={})   # light vs heavy readings, logged only
  if os.environ.get("U_SMOKE") == "1":
    # A few-iteration smoke run (jobs/unified/teacher.sh --smoke): measure from the start, multipliers move after a
    # few samples, the meter reports after 50 steps; the progress is forced along so every switch-on runs.
    new["u_progress"].params = {"skip_steps": 0, "ema_steps": 50.0, "force_rate": 0.002, "force_q": True}
    new["u_constraints"].params.update({"skip_steps": 0, "warmup": 5, "ema_steps": 50.0})
    new["u_heavy"].params.update({"skip_steps": 0, "ema_steps": 50.0})
    cfg.rewards["u_meter"].params["min_n"] = 50
    cfg.rewards["u_lib_meter"].params.update({"min_steps": 5.0, "ema_steps": 200.0})
  cfg.curriculum = new


def _fixed_payload(cfg: ManagerBasedRlEnvCfg) -> None:
  """The checks' payload (2026-11-12, doc/redesign.md 2.1): a fixed point mass from U_PAYLOAD ("kg,x,y,z", default
  0 kg -- the unloaded condition; the acceptance sets the walking payload for the loaded one) instead of the training
  draw. The other randomization (CoM offset, friction, gains, ...) stays, the same in every condition.

  "kg,x,y,z,dx,dy,dz" (the typical payload, the user 2026-11-12 "加一档「常用负载」"; jobs/accept_payload.sh R typical):
  the mass is fixed, the mount point is drawn for each robot at its reset, uniform in x +- dx, y +- dy, z +- dz --
  where the payload will sit on the finished robot is not known."""
  v = [float(s) for s in os.environ.get("U_PAYLOAD", "0,0,0,0").split(",")]
  if len(v) not in (4, 7):
    raise ValueError(f"U_PAYLOAD: 4 values (kg,x,y,z) or 7 (kg,x,y,z,dx,dy,dz), got {len(v)}")
  m, x, y, z = v[:4]
  dx, dy, dz = v[4:] if len(v) == 7 else (0.0, 0.0, 0.0)
  # the whole event replaced: on the check terrains the blind builder keeps mjlab's base-mass draw (-1..+2 kg), not
  # the training's payload point (2026-11-12)
  cfg.events["base_payload"] = EventTermCfg(
    mode="reset", func=mdp.base_payload_point,
    params={"asset_cfg": SceneEntityCfg("robot"), "light_range": (m, m), "heavy_max": 0.0, "heavy_share": 0.0,
            "pos_ranges": ((x - dx, x + dx), (y - dy, y + dy), (z - dz, z + dz)), "base_com_range": 0.0,
            "ramp_steps": 0},
  )


def unified_teacher_env_cfg(robot: URobot, play: bool = False, lean: bool = False,
                            student: str | None = None, terrain: str = "map3") -> ManagerBasedRlEnvCfg:
  """student "blind" / "depth" (students.py): the teacher's environment with the scan fixed off / on, the curriculum
  at its end (progress 1) and the constraint scales frozen at the teacher's (restored from its checkpoint)."""
  cfg = base_env_cfg(robot, play=play, terrain=terrain)
  metrics_mj.add_sensor(cfg)   # the shared metrics' contact sensor (points and normals)
  _add_privileged(cfg)
  _add_scan_mask(cfg)
  _add_joint_dr(cfg)
  stick_slip.add(cfg)   # after the sensor and the privileged group
  other_shares = None
  if terrain == "map3":
    _payload_point(cfg, robot)
    other_shares = _add_slopes(cfg)
    items = constraint_items(robot.kind)
    constrained, other = _rewards(cfg, robot.kind, items, lean)
    if not play:
      by_q = {t for c in items if c["by"] == "q" for t in c["terms"]}
      by_p = {t for c in items if c["by"] == "p" for t in c["terms"]}
      assert not by_q & by_p, f"terms scaled by a flat and a stairs-quality constraint: {sorted(by_q & by_p)}"
      weights, params, start = quality_diff(robot, cfg.rewards, by_p, by_q, tuple(other) if lean else (),
                                            play=play, terrain=terrain)
      for c in items:
        if c["by"] == "q":
          c["start"] = {t: start[t] for t in c["terms"]}
          c["base"] = {t: cfg.rewards[t].weight for t in c["terms"]}
      _curriculum(cfg, robot, items, constrained, other if lean else [], (weights, params), other_shares)
  # else: a check's terrain ("flat", "stairs"; the blind builder leaves out the training schedule there): the
  # observations, actions and physics are the training ones, the rewards and curriculum are not used
  if terrain != "map3":
    _fixed_payload(cfg)
  if student is not None:
    cfg.events["u_scan_mask"].params["mode"] = "off" if student == "blind" else "on"
    if "u_progress" in cfg.curriculum:
      cfg.curriculum["u_progress"].params["fixed_p"] = 1.0
      cfg.curriculum["u_constraints"].params["frozen"] = True
  return cfg
