"""Per-episode control latency for training.

mjlab's DelayedActuator resamples its lag every physics step (or only every
`delay_update_period` steps, starting from zero after a reset) and replays the
zero joint target written while resetting. A real control loop has one
roughly fixed delay and keeps holding the robot's pose. Here every episode
draws one lag in [delay_min_lag, delay_max_lag] physics steps, and after a
reset the delay history starts from the robot's own joint positions.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch
from mjlab.actuator import DelayedActuator, DelayedActuatorCfg
from mjlab.actuator.actuator import ActuatorCmd

NEVER = 1 << 30  # physics steps between lag updates inside an episode


@dataclass(kw_only=True)
class EpisodeLatencyActuatorCfg(DelayedActuatorCfg):
  delay_update_period: int = NEVER

  def build(self, entity, target_ids, target_names) -> EpisodeLatencyActuator:
    return EpisodeLatencyActuator(self, self.base_cfg.build(entity, target_ids, target_names))


class EpisodeLatencyActuator(DelayedActuator):
  def initialize(self, mj_model, model, data, device) -> None:
    super().initialize(mj_model, model, data, device)
    self._fresh = torch.ones(data.nworld, dtype=torch.bool, device=device)
    # The robot's actuator groups (one per motor type) are always reset together,
    # in order, with the same environments. Identically seeded generators then
    # draw the same lag for every group: one control loop, one delay per robot.
    self._generator = torch.Generator(device=device)
    self._generator.manual_seed(torch.initial_seed())

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    idx = slice(None) if env_ids is None else env_ids
    self._fresh[idx] = True
    count = self._fresh[idx].numel()
    lags = torch.randint(self.cfg.delay_min_lag, self.cfg.delay_max_lag + 1, (count,),
                         device=self._fresh.device, generator=self._generator)
    self.set_lags(lags, env_ids)

  @property
  def lags(self) -> torch.Tensor:
    """Current lag per environment, in physics steps."""
    return next(iter(self._delay_buffers.values())).current_lags

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    if self._fresh.any():
      # The first target after a reset is the zero written while resetting:
      # start the delay history from the current joint positions instead.
      fresh = self._fresh[:, None]
      cmd = ActuatorCmd(position_target=torch.where(fresh, cmd.pos, cmd.position_target),
                        velocity_target=cmd.velocity_target, effort_target=cmd.effort_target,
                        pos=cmd.pos, vel=cmd.vel)
      self._fresh[:] = False
    return super().compute(cmd)


def with_episode_latency(cfg, max_latency_ms: float, min_latency_ms: float = 0.0):
  """Wrap every robot actuator so each episode runs with one latency in [min, max] ms."""
  dt = cfg.sim.mujoco.timestep
  low, high = (int(round(ms / 1000 / dt)) for ms in (min_latency_ms, max_latency_ms))
  robot = cfg.scene.entities["robot"]
  # The articulation config is shared by every task built from the same robot
  # constants; replace it rather than editing it in place.
  robot.articulation = dataclasses.replace(robot.articulation, actuators=tuple(
    EpisodeLatencyActuatorCfg(base_cfg=a, delay_target="position", delay_min_lag=low, delay_max_lag=high)
    for a in robot.articulation.actuators))
  return cfg
