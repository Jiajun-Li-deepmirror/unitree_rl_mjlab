"""Wheel velocity action with a low-speed brake (go2-blind).

A velocity-controlled wheel only damps toward its target, and at a zero
command the policies left opposing targets on the wheels (+0.66 / -0.56
rad/s on one side), so the wheeled robots crept 2-6 cm per 4 s standing;
standing rewards alone only halved that. With a zero command and every wheel
slower than `brake_below_speed` at the rim, the wheel targets are zeroed.
Deploy does the same (tinynav-sim reads the ONNX `wheel_brake_rad_s`
metadata). Braking at any speed threw 11 of 64 robots when the stick was
released at 1.5 m/s, hence the speed condition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions import JointVelocityAction, JointVelocityActionCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class BrakedWheelVelocityActionCfg(JointVelocityActionCfg):
  wheel_radius: float
  brake_below_speed: float = 0.1
  """Rim speed (m/s) every wheel must be under for the brake to engage."""
  hold_gain: float = 0.0
  """While braking, drive each wheel back to its angle at brake onset with
  this velocity target per radian of drift (1/s, kp / kd of the wheel servo).
  0 = zero targets, which only damp: go2-blind's wheeled robots slowly rolled
  down stairs, and a position hold lifted its stair stops from 41-49% to
  91-100% without retraining."""
  command_name: str = "twist"

  @property
  def brake_rad_s(self) -> float:
    return self.brake_below_speed / self.wheel_radius

  def build(self, env: ManagerBasedRlEnv) -> BrakedWheelVelocityAction:
    return BrakedWheelVelocityAction(self, env)


class BrakedWheelVelocityAction(JointVelocityAction):
  cfg: BrakedWheelVelocityActionCfg

  def __init__(self, cfg: BrakedWheelVelocityActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.braking = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.policy_targets = torch.zeros_like(self._processed_actions)
    self.hold_pos = torch.zeros_like(self._processed_actions)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    self.braking[env_ids] = False

  def process_actions(self, actions: torch.Tensor) -> None:
    super().process_actions(actions)
    # Commands under the deadzone are exactly zero (the command term zeroes
    # them, as the deploy-side stick deadzone does).
    command = self._env.command_manager.get_command(self.cfg.command_name)
    still = torch.all(command == 0.0, dim=1)
    wheel_vel = self._entity.data.joint_vel[:, self._target_ids]
    slow = torch.all(wheel_vel.abs() < self.cfg.brake_rad_s, dim=1)
    braking = still & slow
    brake_target = torch.zeros_like(self._processed_actions)
    if self.cfg.hold_gain > 0.0:
      wheel_pos = self._entity.data.joint_pos[:, self._target_ids]
      onset = (braking & ~self.braking).unsqueeze(1)
      self.hold_pos = torch.where(onset, wheel_pos, self.hold_pos)
      brake_target = self.cfg.hold_gain * (self.hold_pos - wheel_pos)
    self.braking = braking
    self.policy_targets = self._processed_actions.clone()
    self._processed_actions = torch.where(
      self.braking.unsqueeze(1), brake_target, self._processed_actions
    )
