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
  """

  cfg: BrakedJointVelocityActionCfg

  def process_actions(self, actions: torch.Tensor):
    super().process_actions(actions)
    command = self._env.command_manager.get_command(self.cfg.command_name)
    assert command is not None
    speed = torch.abs(self._entity.data.joint_vel[:, self._target_ids])
    brake = torch.all(command == 0.0, dim=1) & torch.all(speed < self.cfg.brake_speed, dim=1)
    self._processed_actions = torch.where(brake.unsqueeze(1), 0.0, self._processed_actions)


@dataclass(kw_only=True)
class BrakedJointVelocityActionCfg(JointVelocityActionCfg):
  """JointVelocityActionCfg with the zero-command brake (see BrakedJointVelocityAction)."""

  command_name: str = "twist"
  brake_speed: float = float("inf")
  """Joint speed (rad/s) below which every joint must turn for the brake to hold."""

  def build(self, env: ManagerBasedRlEnv) -> BrakedJointVelocityAction:
    return BrakedJointVelocityAction(self, env)
