"""One training run through the whole curriculum, driven by how the robots do (the unified teacher, go2-depth
2026-11-11; the user: "一个训练跑完全部课程，不再分 T/M/P ... 地形难度、指令范围、负载、随机化强度，都按表现自动升级，楼梯份额也由
课程控制").

`progress` measures one number p in [0, 1], never going back, the larger of
  - performance: the terrain curriculum's mean level (mjlab's terrain levels, promoted by distance walked, demoted by
    falls and short walks), its rise from where the run started toward GOAL_LEVEL_FRAC of the levels, smoothed;
  - a gated clock (2026-11-11, the user: "直接修改然后重新训练"): from TIME_START_IT it covers 0 -> 1 in TIME_SPAN_IT
    iterations, at a quarter of the pace while more than MAX_FAIL_RATE of the episodes end in a fall or an illegal
    contact (the blind recipe's staged clock, mdp.staged_clock). The terrain levels alone stalled the first Go2 teacher
    at p 0.19 (mean level ~1 at 3000-4000 it: with 8% stairs and many stand / march / spin commands few robots walk the
    3 m a promotion needs; the blind T sits there too), and the stairs share only rises from p 0.3.
Everything else reads env.u_progress:

  stairs share     TEACHER_SHARE until p 0.3, then up to the robot's final share by p 0.8 (mdp.stairs_share's
                   column draw, with p in place of the training step)
  commands         the initial ranges until p 0.25, the final ones after (as mdp.commands_vel's two stages)
  payload          the heavy robots' payload limit rises with p, full at p 0.5 (mdp.base_payload_point's heavy_max)
  constraints      each one switches on at its own p and tightens from a loose target to its bar (constraints.py)

Each term logs what it set (Curriculum/u_*), and p goes into the checkpoint (runner.py), so a resumed run carries on
where it was.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

GOAL_LEVEL_FRAC = 0.5   # p = 1 when the mean level is here (robots past the top level are sent back to random rows)
TIME_START_IT, TIME_SPAN_IT, MAX_FAIL_RATE = 1500, 10000, 0.3
STEPS_PER_IT = 24   # the runners' num_steps_per_env


def ramp(p: float, a: float, b: float) -> float:
  """0 below a, 1 above b, linear between."""
  if b <= a:
    return float(p >= a)
  return min(1.0, max(0.0, (p - a) / (b - a)))


class progress:
  """Curriculum term: env.u_progress from the terrain levels (see the module docstring). Runs before the terms that
  read it (first in the curriculum)."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self.p = 0.0
    self.level = None        # smoothed mean level
    self.level0 = None       # where the run started
    self.last_step = None
    self.clock = None        # the gated clock (env steps)
    self.fail = None         # smoothed share of resets that ended in a fall / illegal contact
    env.u_progress = 0.0
    env.u_progress_term = self   # the runner saves and restores it

  def reset(self, env_ids=None) -> None:
    pass

  def state(self) -> dict:
    return {"p": self.p, "level": self.level, "level0": self.level0, "clock": self.clock, "fail": self.fail}

  def load(self, state: dict, env: ManagerBasedRlEnv) -> None:
    self.p, self.level, self.level0 = state["p"], state["level"], state["level0"]
    # a checkpoint from before the clock: it starts at the restored step count
    self.clock = state.get("clock") if state.get("clock") is not None else float(env.common_step_counter)
    self.fail = state.get("fail")
    self.last_step = env.common_step_counter
    env.u_progress = self.p

  def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor, ema_steps: float = 2000.0,
               skip_steps: int = 2400, force_rate: float = 0.0, time_start_it: int = TIME_START_IT,
               time_span_it: int = TIME_SPAN_IT, max_fail_rate: float = MAX_FAIL_RATE) -> dict[str, torch.Tensor]:
    terrain = env.scene.terrain
    assert terrain is not None
    mean = float(terrain.terrain_levels.float().mean())
    step = env.common_step_counter
    # the gated clock: the resets now, how many ended in a fall or illegal contact (not a time-out)
    if env_ids is not None and len(env_ids) > 0:
      tm = env.termination_manager
      failed = float((tm.terminated[env_ids] & ~tm.time_outs[env_ids]).float().mean())
      self.fail = failed if self.fail is None else self.fail + 0.01 * (failed - self.fail)
    if self.clock is None or self.last_step is None:
      self.clock = float(step)
    else:
      d = max(0, step - self.last_step)
      self.clock += d * (0.25 if (self.fail or 0.0) > max_fail_rate else 1.0)
    self.last_step = step
    p_time = ramp(self.clock / STEPS_PER_IT, time_start_it, time_start_it + time_span_it)
    if step >= skip_steps:   # every robot resets at the start; the first levels say nothing yet
      if self.level is None:
        self.level = self.level0 = mean
      else:
        a = 1.0 - math.exp(-1.0 / ema_steps)
        self.level += a * (mean - self.level)
      goal = GOAL_LEVEL_FRAC * float(terrain.max_terrain_level)
      if goal > self.level0:
        self.p = max(self.p, min(1.0, (self.level - self.level0) / (goal - self.level0)))
      self.p = max(self.p, p_time)
    if force_rate > 0.0:   # smoke runs only: progress per env step regardless of the levels
      self.p = min(1.0, max(self.p, step * force_rate))
    env.u_progress = self.p
    return {"p": torch.tensor(self.p), "level": torch.tensor(mean), "p_time": torch.tensor(p_time),
            "fail_rate": torch.tensor(self.fail or 0.0)}


def progress_stairs_share(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  stair_terrain_names: tuple[str, ...],
  start: float,
  end: float,
  p_start: float = 0.3,
  p_end: float = 0.8,
) -> dict[str, torch.Tensor]:
  """mdp.stairs_share's draw (each resetting robot to a stairs column with probability s, its difficulty row kept),
  with s = start -> end as env.u_progress goes p_start -> p_end."""
  from src.tasks.velocity.mdp.terrain_utils import sub_terrain_column_mask

  terrain = env.scene.terrain
  assert terrain is not None and terrain.terrain_origins is not None
  mask = sub_terrain_column_mask(env, stair_terrain_names)
  assert mask is not None and mask.any() and (~mask).any(), "need stair and other columns"
  s = start + (end - start) * ramp(getattr(env, "u_progress", 0.0), p_start, p_end)
  if len(env_ids) > 0:
    stair_cols = mask.nonzero().flatten()
    other_cols = (~mask).nonzero().flatten()
    n = len(env_ids)
    on_stairs = torch.rand(n, device=env.device) < s
    pick_s = stair_cols[torch.randint(len(stair_cols), (n,), device=env.device)]
    pick_o = other_cols[torch.randint(len(other_cols), (n,), device=env.device)]
    terrain.terrain_types[env_ids] = torch.where(on_stairs, pick_s, pick_o)
    terrain.env_origins[env_ids] = terrain.terrain_origins[terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]]
  share = mask[terrain.terrain_types].float().mean()
  return {"target": torch.tensor(s), "share": share}


def progress_commands(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  initial: dict,
  final: dict,
  p_switch: float = 0.25,
) -> dict[str, torch.Tensor]:
  """The command ranges: `initial` until env.u_progress reaches p_switch, `final` from then on (both
  {"lin_vel_x": (lo, hi), "lin_vel_y": ..., "ang_vel_z": ...}, as mdp.commands_vel's stages)."""
  del env_ids
  ranges = env.command_manager.get_term(command_name).cfg.ranges
  stage = final if getattr(env, "u_progress", 0.0) >= p_switch else initial
  for key, value in stage.items():
    setattr(ranges, key, tuple(value))
  return {"final": torch.tensor(float(stage is final)), "vx_max": torch.tensor(float(ranges.lin_vel_x[1]))}


def progress_payload(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  event_name: str,
  full_max: float,
  p_full: float = 0.5,
) -> dict[str, torch.Tensor]:
  """The payload event's heavy_max (mdp.base_payload_point: the heavy robots' payload is uniform on [0, heavy_max])
  = full_max x env.u_progress / p_full, at most full_max. Read at each robot's reset."""
  del env_ids
  limit = full_max * ramp(getattr(env, "u_progress", 0.0), 0.0, p_full)
  env.event_manager.get_term_cfg(event_name).params["heavy_max"] = limit
  return {"heavy_max": torch.tensor(limit)}
