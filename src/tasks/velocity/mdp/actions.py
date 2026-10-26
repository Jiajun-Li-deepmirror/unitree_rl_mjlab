"""Action terms for the blind velocity tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions.actions import JointVelocityAction, JointVelocityActionCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = ["BrakedJointVelocityAction", "BrakedJointVelocityActionCfg"]


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

  def build(self, env: ManagerBasedRlEnv) -> BrakedJointVelocityAction:
    return BrakedJointVelocityAction(self, env)
