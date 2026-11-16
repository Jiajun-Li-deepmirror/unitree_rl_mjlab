"""The hard actions shaped like the tests that judge them (go2-depth 2026-11-16, go2-home's U_HARD_FIXED pieces after
the shape survey; teacher_env._round_switches puts this command in place of the twist command when a piece needs it).

TerrainAwareVelocityCommand (src/tasks/velocity/mdp/velocity_command.py, the shared command, unchanged) plus:
  stair_stand_time_range  a stand drawn on a stairs tile holds this long (s) instead of stair_resampling_time_range --
                          eval_stairs zeroes the command mid-flight, holds 4 s and judges the 2-4 s window; with
                          (1.5, 4.0) no training stand on the stairs outlasted 4 s (U_STAIR_STOP_HOLD);
  march_from_splay_prob   the first command of an episode on flat ground, for a robot the reset left with its hips
                          splayed (events' reset_hip_splay; the worst hip more than splay_threshold from its default),
                          is a march in place with this probability -- march_eval's reset test marches 5 s from
                          hips splayed 8 deg, which training met only when the first flat draw was a zero march
                          (~10 % of the splayed resets) (U_MARCH_FROM_SPLAY).
The flat stands' hold (U_FLAT_STAND_HOLD) is the shared command's own stand_resampling_time_range.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.tasks.velocity.mdp.velocity_command import TerrainAwareVelocityCommand, TerrainAwareVelocityCommandCfg


class HardShapeCommand(TerrainAwareVelocityCommand):
  cfg: "HardShapeCommandCfg"

  def __init__(self, cfg: "HardShapeCommandCfg", env) -> None:
    super().__init__(cfg, env)
    self._fresh = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    robot = env.scene["robot"]
    self._hip_ids = [i for i, n in enumerate(robot.joint_names) if "hip" in n]

  def reset(self, env_ids):
    # the first resample of an episode (CommandTerm.reset resamples; the reset events -- the hip splay among them --
    # ran before it; episode_length_buf is zeroed only after the managers' resets)
    self._fresh[env_ids] = True
    try:
      return super().reset(env_ids)
    finally:
      self._fresh[env_ids] = False

  def _resample(self, env_ids: torch.Tensor) -> None:
    super()._resample(env_ids)
    if len(env_ids) == 0:
      return
    stair = self.is_stair_env[env_ids]
    rng = self.cfg.stair_stand_time_range
    if rng is not None:
      stand = (self.vel_command_b[env_ids] == 0.0).all(dim=1) & stair
      ids = env_ids[stand]
      if len(ids) > 0:
        self.time_left[ids] = self.time_left[ids].uniform_(*rng)
    p = self.cfg.march_from_splay_prob
    if p > 0.0 and self._hip_ids:
      fresh = self._fresh[env_ids] & ~stair
      if fresh.any():
        data = self._env.scene["robot"].data
        off = (data.joint_pos[env_ids][:, self._hip_ids] - data.default_joint_pos[env_ids][:, self._hip_ids]).abs()
        splayed = off.amax(dim=1) > self.cfg.splay_threshold
        pick = fresh & splayed & (torch.rand(len(env_ids), device=self.device) < p)
        ids = env_ids[pick]
        if len(ids) > 0:
          self.vel_command_b[ids] = 0.0
          self.march[ids] = True
          if self.cfg.march_resampling_time_range is not None:
            self.time_left[ids] = self.time_left[ids].uniform_(*self.cfg.march_resampling_time_range)


@dataclass(kw_only=True)
class HardShapeCommandCfg(TerrainAwareVelocityCommandCfg):
  stair_stand_time_range: tuple[float, float] | None = None
  march_from_splay_prob: float = 0.0
  splay_threshold: float = 0.05   # rad: the worst hip's distance from its default that counts as splayed

  def build(self, env) -> HardShapeCommand:
    return HardShapeCommand(self, env)


def from_cfg(old: TerrainAwareVelocityCommandCfg, **new) -> HardShapeCommandCfg:
  """The same command with the new fields (every field of `old` copied)."""
  import dataclasses

  if isinstance(old, HardShapeCommandCfg):
    return dataclasses.replace(old, **new)
  fields = {f.name: getattr(old, f.name) for f in dataclasses.fields(old) if f.init}
  return HardShapeCommandCfg(**fields, **new)
