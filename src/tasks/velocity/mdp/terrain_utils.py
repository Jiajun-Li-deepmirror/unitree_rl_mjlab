"""Helpers for querying which sub-terrain each environment was spawned on."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def sub_terrain_column_mask(
  env: ManagerBasedRlEnv, sub_terrain_names: tuple[str, ...]
) -> torch.Tensor | None:
  """Per-column mask of curriculum terrain columns whose type is in `sub_terrain_names`.

  Mirrors the column allocation in
  mjlab.terrains.terrain_generator.TerrainGenerator._generate_curriculum_terrains.
  Returns None when the terrain is not a curriculum generator (column -> type is
  then undefined).
  """
  terrain = env.scene.terrain
  if terrain is None or terrain.terrain_origins is None:
    return None
  generator_cfg = terrain.cfg.terrain_generator
  if generator_cfg is None or not generator_cfg.curriculum:
    return None

  names = list(generator_cfg.sub_terrains.keys())
  proportions = np.array(
    [sub_cfg.proportion for sub_cfg in generator_cfg.sub_terrains.values()]
  )
  proportions /= np.sum(proportions)
  cum_proportions = np.cumsum(proportions)

  mask = []
  for index in range(generator_cfg.num_cols):
    sub_index = np.min(np.where(index / generator_cfg.num_cols + 0.001 < cum_proportions)[0])
    mask.append(names[sub_index] in sub_terrain_names)
  return torch.tensor(mask, dtype=torch.bool, device=env.device)


def spawned_on_sub_terrain(
  env: ManagerBasedRlEnv,
  sub_terrain_names: tuple[str, ...],
  column_mask: torch.Tensor | None = None,
) -> torch.Tensor:
  """Per-env bool: whether the env's current terrain column is one of `sub_terrain_names`."""
  if column_mask is None:
    column_mask = sub_terrain_column_mask(env, sub_terrain_names)
  if column_mask is None:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  terrain = env.scene.terrain
  assert terrain is not None
  return column_mask[terrain.terrain_types]
