"""Zero non-finite observations instead of aborting training.

The Go2-W runs died twice on a NaN observation (once in `actor`, once in a
critic-only term) with a finite physics state, ~1 env in 30M steps and not
reproducible with random actions: most likely a degenerate wheel-cylinder
contact. rsl_rl raises on any NaN, so sanitize the env outputs and report
which terms were hit.
"""

from __future__ import annotations

import itertools

import torch


def guard_nan_observations(env) -> None:
  """Wrap `env.step` (an RslRlVecEnvWrapper) to zero non-finite observations."""
  step = env.step
  manager = env.unwrapped.observation_manager

  def term_names(group: str, cols: torch.Tensor) -> list[str]:
    names = manager.active_terms.get(group)
    dims = manager.group_obs_term_dim.get(group)
    if not names or not dims:
      return []
    ends = list(itertools.accumulate(int(torch.tensor(d).prod()) for d in dims))
    return sorted({names[next(i for i, e in enumerate(ends) if c < e)] for c in cols.tolist()})

  def guarded_step(actions):
    obs, rewards, dones, extras = step(actions)
    for group, value in obs.items():
      bad = ~torch.isfinite(value)
      if bad.any():
        flat = bad.reshape(bad.shape[0], -1)
        cols = torch.nonzero(flat.any(dim=0)).flatten()
        print(
          f"[obs_guard] non-finite '{group}' obs in {int(flat.any(dim=1).sum())} envs "
          f"(terms {term_names(group, cols)}); zeroed",
          flush=True,
        )
        obs[group] = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    return obs, rewards, dones, extras

  env.step = guarded_step
