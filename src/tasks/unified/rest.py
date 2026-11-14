"""A wheeled robot's stand by the ground it stands on (go2-depth 2026-11-14; the probe switch U_REST_OLD=even,
robots.base_env_cfg).

The unified recipe's stand has two keys the delivered Go2W's did not have: "restlock" (once the standing hold has its
anchor, the pose term's target is the pose at the lock and stand_still_even is off) and "standtgt" (the leg targets'
hold, standing_target_hold). Two probes from the formal Go2W's model_3600 (170dd4c), 600 iterations each:

  - without both (U_REST_OLD=1, uprobe_s114): the flat gate's stop creep and line drift pass for the first time
    (worst p90 creep 1.11 / 1.32 cm, drift 4.1 / 3.7 %; with the keys 1.2-1.8 cm medians and the creep multiplier at
    its cap from 2000 on) -- and the stairs gate loses the stop on a steep descent: blind, 15-18 cm going down, 71 %
    of the stopped robots stay within the 0.5 m (bar 90 %), with the typical payload 46 % (a median of 1.32 m rolled
    on down the flight; four wheels down 0.84 / 0.71);
  - with both (the control of the same length, uprobe_s111): 100 % stay on every band in both scan modes (typical
    payload, blind, 15-18 cm down 90 %), and the flat creep fails as before.

On a flight the default pose is not a pose the robot can rest in (front and rear legs differ by the steps under
them): a pose term that pulls to it through the whole stop, with nothing holding the leg targets, walks the robot
on down. On even ground the pose the braking left is not the rest pose either, and locking it keeps the legs coming
back for seconds -- the creep. So each form where it works:

  even ground     the delivered form: pose and stand_still_even pull to the default pose through the whole stand,
                  no leg-target hold;
  uneven ground   the lock form: the pose at the lock is the pose term's target, the leg targets are held
                  (stand_still_even has its own even-ground mask: nothing there either way).

"Uneven" is mdp.uneven_terrain_mask on the pose term's scan sensor, radius and threshold -- the mask the pose term
itself picks its tolerance by. The shared kernels (src/tasks/velocity/mdp/rewards.py) are not changed: the two
classes below wrap them.

U_REST_OLD=tilt[,<rad>] (2026-11-14 20:30): the same two forms, picked by the base's tilt instead of the scan -- the
lock form where the gravity direction is `lock_tilt` rad or more off the base's vertical (0.10: a robot standing on
a flight), the delivered form below it. Why: uprobe_s131 (`even`, 2048 robots, 600 iterations from the formal
model_3600) keeps the stop on every flight (100 % stay) and passes the flat creep WITH the scan (medians 0.30-1.00 cm
in every group, bar 1.0) but not with the scan masked (fwd 1.0 / arc / fwd 1.5: 1.10 / 1.16 / 1.66 cm; the P5 arm
s114 0.88 / 0.73 / 0.87). 40 % of the episodes walk with the scan masked (teacher_env.SCAN_MASK_SHARE): such a
robot is paid by a ground it cannot see, so its stand is one behaviour for both grounds. The tilt is in every
actor's observation (the projected gravity) and on a flight it is the thing that makes the default pose no rest
pose: 5-18 cm risers under a 0.39 m wheelbase give 7-27 deg, a stand on the flat 0.5-1.5 deg."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from src.tasks.velocity.mdp import rewards as old_rew

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg


TILT = 0.10   # rad: U_REST_OLD=tilt's default


def lock_ground(env: ManagerBasedRlEnv, sensor_name: str, radius: float, threshold: float,
                lock_tilt: float | None) -> torch.Tensor:
  """Where the lock form counts. `lock_tilt` None: uneven ground under the robot (the scan; U_REST_OLD=even). Else:
  the base tilted by `lock_tilt` rad or more -- the gravity direction in the base frame, which a robot walking with
  its scan masked observes too (U_REST_OLD=tilt)."""
  if lock_tilt is None:
    return old_rew.uneven_terrain_mask(env, sensor_name, radius, threshold)
  g = env.scene["robot"].data.projected_gravity_b[:, :2]
  return torch.norm(g, dim=1) >= math.sin(lock_tilt)


class pose_lock_uneven(old_rew.variable_posture_terrain_aware):
  """variable_posture_terrain_aware whose `hold_term` lock counts on uneven ground only: there the value with the
  lock (the pose at the lock as the target), on even ground the value without it (the default pose). The lock pose
  itself follows on every ground, as in the kernel -- a robot that rolls from a landing onto a flight while locked
  keeps the pose it locked in."""

  def __call__(self, env: ManagerBasedRlEnv, *, sensor_name: str, hold_term: str | None = None,
               uneven_radius: float = 0.45, uneven_threshold: float = 0.05, lock_tilt: float | None = None,
               **params) -> torch.Tensor:
    kw = dict(params, sensor_name=sensor_name, uneven_radius=uneven_radius, uneven_threshold=uneven_threshold)
    free = super().__call__(env, hold_term=None, **kw)
    if hold_term is None:
      return free
    locked = super().__call__(env, hold_term=hold_term, **kw)
    return torch.where(lock_ground(env, sensor_name, uneven_radius, uneven_threshold, lock_tilt), locked, free)


class target_hold_uneven(old_rew.standing_target_hold):
  """standing_target_hold on uneven ground only (the anchor and the low-pass follow on every ground, as in the
  kernel; on even ground the term is 0)."""

  def __call__(self, env: ManagerBasedRlEnv, *, sensor_name: str, uneven_radius: float = 0.45,
               uneven_threshold: float = 0.05, lock_tilt: float | None = None, **params) -> torch.Tensor:
    cost = super().__call__(env, **params)
    return cost * lock_ground(env, sensor_name, uneven_radius, uneven_threshold, lock_tilt).float()


def apply_even(cfg: ManagerBasedRlEnvCfg, lock_tilt: float | None = None) -> None:
  """U_REST_OLD=even (`lock_tilt` None) or =tilt (the tilt in rad) on a built base configuration (module
  docstring). Stops the build when the stand's terms are not the ones it was written for."""
  r = cfg.rewards
  pose = r["pose"]
  assert pose.func is old_rew.variable_posture_terrain_aware and pose.params.get("hold_term"), (
    "U_REST_OLD=even: the pose term is not variable_posture_terrain_aware with a hold_term (the key restlock)")
  pose.func = pose_lock_uneven
  still = r["stand_still_even"]
  assert still.params.get("hold_term") == pose.params["hold_term"], "U_REST_OLD=even: stand_still_even without the lock"
  still.params = {k: v for k, v in still.params.items() if k != "hold_term"}
  hold = r["standing_target_hold"]
  assert hold.func is old_rew.standing_target_hold, "U_REST_OLD=even: standing_target_hold is another term"
  hold.func = target_hold_uneven
  mask = {k: pose.params[k] for k in ("uneven_radius", "uneven_threshold") if k in pose.params}
  hold.params = {**hold.params, "sensor_name": pose.params["sensor_name"], **mask}
  if lock_tilt is not None:
    pose.params = {**pose.params, "lock_tilt": lock_tilt}
    hold.params = {**hold.params, "lock_tilt": lock_tilt}
