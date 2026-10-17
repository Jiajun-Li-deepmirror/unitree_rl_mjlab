"""Velocity actuator for continuous (unlimited) wheel joints.

mjlab's BuiltinVelocityActuator inherits its control range from the joint
range, which a free-spinning wheel joint doesn't have (MuJoCo then rejects
the empty range). This one sets the speed range explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
from mjlab.actuator import BuiltinVelocityActuatorCfg
from mjlab.actuator.builtin_actuator import BuiltinVelocityActuator


@dataclass(kw_only=True)
class WheelVelocityActuatorCfg(BuiltinVelocityActuatorCfg):
  velocity_limit: float
  """rad/s; commanded wheel speeds are clipped to +-this."""

  def build(self, entity, target_ids: list[int], target_names: list[str]):
    return WheelVelocityActuator(self, entity, target_ids, target_names)


class WheelVelocityActuator(BuiltinVelocityActuator):
  cfg: WheelVelocityActuatorCfg

  def edit_spec(self, spec: mujoco.MjSpec, target_names: list[str]) -> None:
    super().edit_spec(spec, target_names)
    for actuator in self._mjs_actuators:
      actuator.inheritrange = 0.0
      actuator.ctrlrange[:] = (-self.cfg.velocity_limit, self.cfg.velocity_limit)
