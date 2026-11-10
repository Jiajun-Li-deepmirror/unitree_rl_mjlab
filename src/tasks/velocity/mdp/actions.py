"""Action terms for the blind velocity tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions.actions import (
  JointPositionAction,
  JointPositionActionCfg,
  JointVelocityAction,
  JointVelocityActionCfg,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = [
  "BrakedJointVelocityAction",
  "BrakedJointVelocityActionCfg",
  "HeldJointPositionAction",
  "HeldJointPositionActionCfg",
]


class BrakedJointVelocityAction(JointVelocityAction):
  """Joint velocity targets, forced to zero while the velocity command is zero and
  every joint already turns slower than `brake_speed`.

  A parking brake for wheels: velocity-controlled wheels hold no position, and a
  policy's small leftover wheel targets at the zero command crept the Go2W 2 cm
  per 4 s; with the targets zeroed the same policy stood within 0.01 cm. Braking
  at once on letting go of the stick tipped it over from 1.5 m/s (11 of 64), so
  the policy slows down first and the brake only holds it once nearly stopped
  (and lets go if a push spins the wheels up). The deploy side does the same
  (JointVelocityAction `zero_when_command_zero`, `brake_below_speed`) on the
  command after its deadzone, which is exactly zero there, as in training.

  With `hold_stiffness` > 0 a braked joint also holds the angle it had when the
  brake engaged (a position target with that stiffness, on top of the damping).
  Damping alone resists no steady force: stopped mid-flight on stairs, where the
  slanted legs push the wheels along the treads, the Go2W and B2W final policies
  (2026-10-21) rolled 6-7 cm in 2 s and held still in 49% / 41% of the
  eval_stairs stop runs; with a hold of 20 / 40 Nm/rad and no retraining, 100% /
  91% (drift 0.7 / 2.1 cm). The deploy side holds the same way
  (JointVelocityAction `brake_hold_kp`).

  A policy can keep its wheels turning just above `brake_speed` at the zero
  command: stopped on B2W's highest risers it held them at 1.0-1.6 rad/s (above
  0.885), so the hold engaged only in short bursts and the robot slid back 0.27 m
  (go2-depth in tinynav-sim, 2026-10-22; in mjlab 75% held on 15-18 cm). With
  `release_speed_ratio` the speed limit is that times `brake_speed` once braked (a
  braked joint lets go only when pushed), and with `brake_after_s` also once the
  command has been zero that long (so a policy turning its wheels slowly is braked
  after it, while a robot still coasting faster than the wider limit is not): in
  that test a 1 s timer or a 3x release left 2-6 cm instead of 11-13 cm, with the
  same coasting from 1.5 m/s and no falls.

  With `brake_max_forward` the brake also works while the forward command is at
  most that (m/s): stepping sideways or turning in place, where a wheeled robot
  moves by stepping and its wheel targets are charged to zero anyway
  (stepping_wheel_target). By damping alone the grounded wheels roll under the
  legs' push and the steps make up for it, by a share that changes from checkpoint
  to checkpoint: stepping at (0, 0.3, 0) the As2W wheels rolled back 0-8% of the
  sideways travel (p50; v13wh 5000 ~0, the drcritic pilot 4000 ~7.5%, its steps
  +4.4%), and the drift left over failed the gate at one checkpoint and passed at
  the next (scratchpad w18/kit/go2-blind/tools/flat_items_dr.py, 2026-10-30). The
  deploy side brakes the same way (JointVelocityAction `brake_max_forward`), on
  the command after its deadzone.
  """

  cfg: BrakedJointVelocityActionCfg

  def __init__(self, cfg: BrakedJointVelocityActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    assert cfg.brake_after_s is None or cfg.release_speed_ratio is not None, (
      "brake_after_s widens the release_speed_ratio limit; set both"
    )
    self._braked = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self._zero_time = torch.zeros(self.num_envs, device=self.device)
    self._hold_pos = torch.zeros(self.num_envs, self._num_targets, device=self.device)
    # (actuator with the PD gains, its columns of this term's joints).
    self._holds: list[tuple[object, list[int]]] = []
    if cfg.hold_stiffness > 0.0:
      for actuator in self._entity._actuators:
        if not set(actuator.target_names) & set(self._target_names):
          continue
        while not hasattr(actuator, "stiffness"):  # e.g. DelayedActuator
          actuator = actuator._base_actuator
        shared = [n for n in actuator.target_names if n in self._target_names]
        if shared:
          self._holds.append((actuator, [actuator.target_names.index(n) for n in shared]))
      held = sum(len(cols) for _, cols in self._holds)
      assert held == self._num_targets, f"hold_stiffness: found PD gains for {held} of {self._num_targets} joints"

  def process_actions(self, actions: torch.Tensor):
    super().process_actions(actions)
    command = self._env.command_manager.get_command(self.cfg.command_name)
    assert command is not None
    speed = torch.abs(self._entity.data.joint_vel[:, self._target_ids])
    zero = torch.all(command == 0.0, dim=1)
    if self.cfg.brake_max_forward is not None:
      zero = torch.abs(command[:, 0]) <= self.cfg.brake_max_forward
    limit = torch.full_like(zero, self.cfg.brake_speed, dtype=speed.dtype)
    if self.cfg.release_speed_ratio is not None:
      wide = self._braked
      if self.cfg.brake_after_s is not None:
        self._zero_time = torch.where(zero, self._zero_time + self._env.step_dt, 0.0)
        # Half a step of slack: summed float steps fall just short of whole seconds.
        wide = wide | (self._zero_time >= self.cfg.brake_after_s - 0.5 * self._env.step_dt)
      limit = torch.where(wide, self.cfg.release_speed_ratio * self.cfg.brake_speed, limit)
    brake = zero & torch.all(speed < limit.unsqueeze(1), dim=1)
    self._processed_actions = torch.where(brake.unsqueeze(1), 0.0, self._processed_actions)
    if self._holds:
      pos = self._entity.data.joint_pos[:, self._target_ids]
      self._hold_pos = torch.where((brake & ~self._braked).unsqueeze(1), pos, self._hold_pos)
    self._braked = brake

  def apply_actions(self) -> None:
    super().apply_actions()
    if not self._holds:
      return
    braked = self._braked.unsqueeze(1)
    # Released joints get their own angle as the target (with zero stiffness), so
    # a delayed position target is never stale when the brake engages.
    pos = self._entity.data.joint_pos[:, self._target_ids]
    self._entity.set_joint_position_target(
      torch.where(braked, self._hold_pos, pos), joint_ids=self._target_ids
    )
    stiffness = torch.where(braked, self.cfg.hold_stiffness, 0.0)  # (envs, 1)
    for actuator, cols in self._holds:
      actuator.stiffness[:, cols] = stiffness

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    ids = slice(None) if env_ids is None else env_ids
    self._braked[ids] = False
    self._zero_time[ids] = 0.0


@dataclass(kw_only=True)
class BrakedJointVelocityActionCfg(JointVelocityActionCfg):
  """JointVelocityActionCfg with the zero-command brake (see BrakedJointVelocityAction)."""

  command_name: str = "twist"
  brake_speed: float = float("inf")
  """Joint speed (rad/s) below which every joint must turn for the brake to hold."""
  hold_stiffness: float = 0.0
  """Stiffness (Nm/rad) holding a braked joint at its angle when the brake
  engaged; 0 brakes by damping alone."""
  release_speed_ratio: float | None = None
  """Speed limit of a braked joint, times `brake_speed`: it stays braked until a
  joint turns faster than that."""
  brake_after_s: float | None = None
  """With release_speed_ratio: the wider limit also applies once the command has
  been zero this long (s), braking joints that keep turning slowly."""
  brake_max_forward: float | None = None
  """Brake whenever |forward command| is at most this (m/s; sideways, turning in
  place), not only at the zero command."""

  def build(self, env: ManagerBasedRlEnv) -> BrakedJointVelocityAction:
    return BrakedJointVelocityAction(self, env)


class HeldJointPositionAction(JointPositionAction):
  """Joint position targets held where they were once the robot stands still: a standstill hold for a wheeled
  robot's legs, next to the wheels' parking brake (BrakedJointVelocityAction).

  Once the velocity command has been zero (not marching in place) for `hold_after_s` and every joint of the term
  turns slower than `hold_speed`, the targets stay at the policy's targets of that step; they follow the policy
  again when the command leaves zero, the robot marches, or a joint turns faster than `release_speed` (pushed),
  and hold again once calm. On braked wheels every leg move moves the base: the B2W teachers kept bringing their
  body back over the wheels for seconds after a stop and crept 1.3-6.5 cm in the stop acceptance's last 4 s (bar
  1), where their own targets frozen 2 s after the release left 0.12-0.30 cm, and neither reward terms (restlock,
  the leg-target hold up to x10) nor a stop clock in the observation taught a policy to hold still (2026-10-29 to
  11-01, scratchpad w17/creep_split.py, w20). The deploy side holds the same way (JointPositionAction
  `hold_after_s`, `hold_speed`, `hold_release_speed`) on the command after its deadzone, which is exactly zero
  there, as in training.
  """

  cfg: HeldJointPositionActionCfg

  def __init__(self, cfg: HeldJointPositionActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._held = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self._zero_time = torch.zeros(self.num_envs, device=self.device)
    self._hold_target = torch.zeros(self.num_envs, self._num_targets, device=self.device)

  @property
  def held(self) -> torch.Tensor:
    return self._held

  def process_actions(self, actions: torch.Tensor):
    super().process_actions(actions)
    if self.cfg.hold_after_s is None:
      return
    term = self._env.command_manager.get_term(self.cfg.command_name)
    command = self._env.command_manager.get_command(self.cfg.command_name)
    assert command is not None
    zero = torch.all(command == 0.0, dim=1)
    march = getattr(term, "march", None)
    if march is not None:
      zero = zero & ~march
    self._zero_time = torch.where(zero, self._zero_time + self._env.step_dt, 0.0)
    speed = torch.abs(self._entity.data.joint_vel[:, self._target_ids]).amax(dim=1)
    # Half a step of slack: summed float steps fall just short of whole seconds.
    due = self._zero_time >= self.cfg.hold_after_s - 0.5 * self._env.step_dt
    engage = zero & ~self._held & due & (speed < self.cfg.hold_speed)
    keep = speed <= self.cfg.release_speed
    if self.cfg.hold_base_rate is not None or self.cfg.release_base_rate is not None:
      # The base's angular rate too (go2-depth 2026-11-07, "holdcalm"; the deploy side reads the IMU gyro): the Go2W
      # depth teacher's legs calmed below hold_speed for a moment after a 1.5 m/s stop while the base still rocked,
      # froze there and 26 of 256 fell (0 without the hold).
      rate = torch.norm(self._entity.data.root_link_ang_vel_b, dim=1)
      if self.cfg.hold_base_rate is not None:
        engage = engage & (rate < self.cfg.hold_base_rate)
      if self.cfg.release_base_rate is not None:
        keep = keep & (rate <= self.cfg.release_base_rate)
    self._held = zero & ((self._held & keep) | engage)
    self._hold_target = torch.where(engage.unsqueeze(1), self._processed_actions, self._hold_target)
    self._processed_actions = torch.where(self._held.unsqueeze(1), self._hold_target, self._processed_actions)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    ids = slice(None) if env_ids is None else env_ids
    self._held[ids] = False
    self._zero_time[ids] = 0.0


@dataclass(kw_only=True)
class HeldJointPositionActionCfg(JointPositionActionCfg):
  """JointPositionActionCfg with the standstill hold (see HeldJointPositionAction)."""

  command_name: str = "twist"
  hold_after_s: float | None = None
  """Hold once the command has been zero this long (s); None never holds."""
  hold_speed: float = 0.1
  """Joint speed (rad/s) below which every joint must turn for the hold to engage."""
  release_speed: float = 1.0
  """A held term lets go once a joint turns faster than this (rad/s)."""
  hold_base_rate: float | None = None
  """With it the hold engages only while the base turns slower than this (|angular velocity|, rad/s)."""
  release_base_rate: float | None = None
  """With it a held term also lets go once the base turns faster than this (rad/s)."""

  def build(self, env: ManagerBasedRlEnv) -> HeldJointPositionAction:
    return HeldJointPositionAction(self, env)
