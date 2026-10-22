from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .velocity_command import UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


class VelocityStage(TypedDict):
  step: int
  lin_vel_x: tuple[float, float] | None
  lin_vel_y: tuple[float, float] | None
  ang_vel_z: tuple[float, float] | None


class RewardWeightStage(TypedDict):
  step: int
  weight: float


def terrain_levels_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  command = env.command_manager.get_command(command_name)
  assert command is not None

  # Compute the distance the robot walked.
  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
  )

  # Robots that walked far enough progress to harder terrains.
  move_up = distance > terrain_generator.size[0] / 2

  # Robots that walked less than half of their required distance go to simpler
  # terrains.
  move_down = (
    distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
  )
  move_down *= ~move_up

  # Update terrain levels.
  terrain.update_env_origins(env_ids, move_up, move_down)

  return torch.mean(terrain.terrain_levels.float())


def commands_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
  del env_ids  # Unused.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
  for stage in velocity_stages:
    if env.common_step_counter > stage["step"]:
      if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
        cfg.ranges.lin_vel_x = stage["lin_vel_x"]
      if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
        cfg.ranges.lin_vel_y = stage["lin_vel_y"]
      if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
        cfg.ranges.ang_vel_z = stage["ang_vel_z"]
  return {
    # "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
    # "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
    # "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),
    # "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
    # "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),
    # "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
  }


def reward_weight(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str,
  weight_stages: list[RewardWeightStage],
) -> torch.Tensor:
  """Update a reward term's weight based on training step stages."""
  del env_ids  # Unused.
  reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)
  for stage in weight_stages:
    if env.common_step_counter > stage["step"]:
      reward_term_cfg.weight = stage["weight"]
  return torch.tensor([reward_term_cfg.weight])


def terrain_levels_cmd_distance(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  move_up_distance: float = 3.0,
  move_down_ratio: float = 0.5,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> torch.Tensor:
  """Terrain curriculum based on distance commanded over the whole episode.

  Promote when the robot got `move_up_distance` away from its spawn point.
  Demote when it covered less than `move_down_ratio` of the distance it was
  commanded to walk, or when the episode ended in a fall. Requires a command
  term exposing `cmd_distance` (TerrainAwareVelocityCommand).
  """
  asset: Entity = env.scene[asset_cfg.name]
  terrain = env.scene.terrain
  assert terrain is not None

  command_term = env.command_manager.get_term(command_name)
  cmd_distance = command_term.cmd_distance[env_ids]  # type: ignore[attr-defined]

  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
  )
  move_up = distance > move_up_distance
  fell = env.termination_manager.terminated[env_ids]
  move_down = ((distance < move_down_ratio * cmd_distance) | fell) & ~move_up

  terrain.update_env_origins(env_ids, move_up, move_down)
  return torch.mean(terrain.terrain_levels.float())


def reward_params(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str,
  param_stages: list[dict],
) -> dict[str, torch.Tensor]:
  """Update a reward term's params based on training step stages.

  Each stage is {"step": int, "params": {name: value}}; later stages win.
  """
  del env_ids  # Unused.
  reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)
  for stage in param_stages:
    if env.common_step_counter > stage["step"]:
      reward_term_cfg.params.update(stage["params"])
  staged_keys = {key for stage in param_stages for key in stage["params"]}
  return {key: torch.tensor(reward_term_cfg.params[key]) for key in staged_keys}


def _ramp(step: int, start: float, end: float, start_step: int, end_step: int) -> float:
  """`start` until `start_step`, then linearly to `end` at `end_step`."""
  if end_step <= start_step:
    return end if step >= end_step else start
  t = min(max((step - start_step) / (end_step - start_step), 0.0), 1.0)
  return start + t * (end - start)


def ramp_terms(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  ramps: list[dict],
) -> dict[str, torch.Tensor]:
  """Set reward weights, reward params and command settings by training step.

  Every term is in the task from the first step; this only moves its strength
  (user 2026-10-22: stages by weights, not recipes that appear at some
  iteration). Each ramp is {"kind": "reward_weight" | "reward_param" | "command",
  "name": reward term or command term, "key": param or attribute (not for
  reward_weight), "start", "end", "start_step", "end_step"}; values go linearly
  from start to end between the two steps (env steps: iterations x
  steps per env).
  """
  del env_ids  # Unused.
  step = env.common_step_counter
  out = {}
  for r in ramps:
    value = _ramp(step, r["start"], r["end"], r["start_step"], r["end_step"])
    if r["kind"] == "reward_weight":
      env.reward_manager.get_term_cfg(r["name"]).weight = value
      label = f"{r['name']}.weight"
    elif r["kind"] == "reward_param":
      env.reward_manager.get_term_cfg(r["name"]).params[r["key"]] = value
      label = f"{r['name']}.{r['key']}"
    elif r["kind"] == "command":
      setattr(env.command_manager.get_term(r["name"]).cfg, r["key"], value)
      label = f"{r['name']}.{r['key']}"
    else:
      raise ValueError(f"ramp kind {r['kind']!r}")
    out[label] = torch.tensor(value)
  return out


def stairs_share(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  stair_terrain_names: tuple[str, ...],
  start: float,
  end: float,
  start_step: int,
  end_step: int,
) -> dict[str, torch.Tensor]:
  """Move resetting robots between stair and other terrain columns so that a
  share of them, ramped by training step, trains on stairs.

  The stairs are there from the first step, at `start` of the robots (user
  2026-10-22: never without stairs, fewer at first), rising to `end`. Each
  resetting robot draws stairs with that probability and a random column of its
  kind; its difficulty row stays (terrain_levels runs first and has set it).
  """
  from src.tasks.velocity.mdp.terrain_utils import sub_terrain_column_mask

  terrain = env.scene.terrain
  assert terrain is not None and terrain.terrain_origins is not None
  mask = sub_terrain_column_mask(env, stair_terrain_names)
  assert mask is not None and mask.any() and (~mask).any(), "need stair and other columns"
  p = _ramp(env.common_step_counter, start, end, start_step, end_step)
  if len(env_ids) > 0:
    stair_cols = mask.nonzero().flatten()
    other_cols = (~mask).nonzero().flatten()
    n = len(env_ids)
    on_stairs = torch.rand(n, device=env.device) < p
    pick_s = stair_cols[torch.randint(len(stair_cols), (n,), device=env.device)]
    pick_o = other_cols[torch.randint(len(other_cols), (n,), device=env.device)]
    terrain.terrain_types[env_ids] = torch.where(on_stairs, pick_s, pick_o)
    terrain.env_origins[env_ids] = terrain.terrain_origins[
      terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]
    ]
  share = mask[terrain.terrain_types].float().mean()
  return {"target": torch.tensor(p), "share": share}
