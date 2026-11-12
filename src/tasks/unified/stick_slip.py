"""Static friction above kinetic friction on the feet and wheels (2026-11-12; the user, through the students'
session: "模拟静摩擦大于动摩擦").

MuJoCo's contact has one friction coefficient: what holds a standing foot is also what brakes a sliding one. On the
robot a foot that starts to slide loses grip, so a slide, once begun, goes further than the simulation's. Here each
robot draws a ratio r = mu_s / mu_k from RATIO every episode, and every control step each foot's (wheel's)
coefficient is set from what its contact point does:

  holding   mu x sqrt(r)   until the contact point's speed along the surface passes V_SLIDE (plus SLIDE_REL x the
                           foot geom's own speed: a rolling wheel's contact point is only as still as the contact
                           position is exact)
  sliding   mu / sqrt(r)   until that speed is below V_STICK again; a foot in the air counts as sliding, so it
                           touches down on the kinetic value and grips once it rests

mu is the robot's randomized coefficient (the foot_friction event's draw), kept as the geometric mean of the two: the
split makes neither the average grip nor the average slide easier than before. The feet's geoms have contact priority
1, so their coefficient is the contact's. The model's field is written in place, as mjlab's dr.geom_friction writes
it. The critic's and the privileged group's dr_params read mu, not the value that switches (u_dr_params).

In every environment of the unified pipeline (the teacher, the students, the checks' terrains): add(cfg).
U_STICK_RATIO="lo,hi" overrides RATIO ("1,1": no split, the physics as before). tinynav-sim does not model the split.
The share of the feet in contact that slide goes to the training records (env.u_foot_sliding, diagnostics.physics).
"""

from __future__ import annotations

import os

import torch
from mjlab.managers.event_manager import EventTermCfg

import src.tasks.velocity.mdp as mdp
from src.metrics import mj as metrics_mj

RATIO = (1.0, 1.3)   # mu_s / mu_k: uniform per robot and episode
V_SLIDE = 0.05       # m/s: a holding contact point faster than this slides ...
SLIDE_REL = 0.05     # ... plus this share of the foot geom's own speed
V_STICK = 0.02       # m/s (plus the same share): a sliding contact point slower than this holds again
EVENT = "u_stick_slip"


def ratio_range() -> tuple[float, float]:
  v = os.environ.get("U_STICK_RATIO")
  if not v:
    return RATIO
  lo, hi = (float(x) for x in v.split(","))
  assert 1.0 <= lo <= hi, f"U_STICK_RATIO={v}"
  return lo, hi


class stick_slip:
  """Event term, mode "step": runs every control step on all robots, after the resets and the forward pass and
  before the observations (module docstring). Set up at its first step: the model's friction rows are per robot only
  once the environment has expanded the randomized fields, after the managers are built."""

  model_fields = ("geom_friction",)   # per-robot rows (the foot_friction event asks for them too)

  def __init__(self, cfg, env):
    del cfg, env
    self.ready = False

  def _setup(self, env, friction_geoms: tuple[str, ...]) -> None:
    robot = env.scene["robot"]
    sensor = env.scene[metrics_mj.SENSOR]
    prim = sensor.cfg.primary
    assert prim.mode == "geom" and prim.entity == "robot", "the feet metric sensor's primaries: the robot's foot geoms"
    pats = (prim.pattern,) if isinstance(prim.pattern, str) else tuple(prim.pattern)
    local, names = robot.find_geoms(pats)   # as the sensor resolves them: its feet, in its order
    want, _ = robot.find_geoms(tuple(friction_geoms))
    assert sorted(local) == sorted(want), f"the feet sensor's geoms {names} are not the foot_friction event's"
    n, f, dev = env.num_envs, len(local), env.device
    assert sensor.data.found.shape[1] == f * metrics_mj.SLOTS, "the feet metric sensor's slots"
    self.local = torch.as_tensor(local, device=dev, dtype=torch.long)
    gids = robot.indexing.geom_ids[self.local].long()
    self.env_grid, self.geom_grid = torch.meshgrid(torch.arange(n, device=dev), gids, indexing="ij")
    fr = env.sim.model.geom_friction
    assert fr.shape[0] == n, "geom_friction has no per-robot rows (no friction randomization?)"
    self.mu = fr[self.env_grid, self.geom_grid][..., 0].clone()   # (N, F): the randomized coefficient
    self.sliding = torch.ones(n, f, dtype=torch.bool, device=dev)
    self.lo, self.hi = ratio_range()
    self.root = torch.sqrt(self.lo + (self.hi - self.lo) * torch.rand(n, device=dev))   # sqrt(r) per robot
    env.u_mu_nominal = self.mu.mean(dim=1)
    self.ready = True

  def reset(self, env_ids=None) -> None:
    if not self.ready:
      return
    ids = slice(None) if env_ids is None else env_ids
    n = self.root[ids].shape[0]
    self.root[ids] = torch.sqrt(self.lo + (self.hi - self.lo) * torch.rand(n, device=self.root.device))
    self.sliding[ids] = True

  def __call__(self, env, env_ids, friction_geoms: tuple[str, ...]) -> None:
    del env_ids
    if not self.ready:
      self._setup(env, friction_geoms)
    n, f = self.sliding.shape
    k = metrics_mj.SLOTS
    d = env.scene[metrics_mj.SENSOR].data   # slot 0: each foot's nearest contact with the terrain
    found = d.found.view(n, f, k)[:, :, 0] > 0
    pos = d.pos.view(n, f, k, 3)[:, :, 0]
    normal = d.normal.view(n, f, k, 3)[:, :, 0]
    rd = env.scene["robot"].data
    centre_v = rd.geom_lin_vel_w[:, self.local]
    arm = pos - rd.geom_pos_w[:, self.local]
    v = centre_v + torch.linalg.cross(rd.geom_ang_vel_w[:, self.local], arm, dim=-1)   # the contact point's velocity
    v = v - (v * normal).sum(dim=-1, keepdim=True) * normal                              # along the surface
    speed = torch.linalg.vector_norm(v, dim=-1)
    slack = SLIDE_REL * torch.linalg.vector_norm(centre_v, dim=-1)
    starts = speed > V_SLIDE + slack
    stops = speed < V_STICK + slack
    self.sliding = ~found | torch.where(self.sliding, ~stops, starts)
    root = self.root[:, None]
    fr = env.sim.model.geom_friction
    rows = fr[self.env_grid, self.geom_grid]
    rows[..., 0] = torch.where(self.sliding, self.mu / root, self.mu * root)
    fr[self.env_grid, self.geom_grid] = rows
    env.u_foot_sliding = (found & self.sliding).float().sum() / found.float().sum().clamp(min=1.0)


def u_dr_params(env, friction_cfg, **params) -> torch.Tensor:
  """mdp.dr_params with the robot's randomized friction coefficient (stick_slip's mu) in place of the model's present
  value, which switches with each foot's hold."""
  x = mdp.dr_params(env, friction_cfg=friction_cfg, **params)
  mu = getattr(env, "u_mu_nominal", None)
  if mu is None:   # before the event's first step: the model still holds the draw
    return x
  return torch.cat([x[:, :-1], ((mu - 1.0) / 0.5).unsqueeze(1)], dim=1)


def add(cfg) -> None:
  """The stick-slip event, and dr_params reading the randomized coefficient (after the metrics' sensor and the
  privileged group are in the cfg)."""
  ev = cfg.events["foot_friction"]
  assert ev.mode == "startup", "stick_slip reads the friction draw once: foot_friction has to be a startup event"
  cfg.events[EVENT] = EventTermCfg(mode="step", func=stick_slip,
                                   params={"friction_geoms": tuple(ev.params["asset_cfg"].geom_names)})
  swapped = 0
  for group in cfg.observations.values():
    for term in group.terms.values():
      if term.func is mdp.dr_params:
        term.func = u_dr_params
        swapped += 1
  assert swapped >= 1, "no dr_params observation term to point at the randomized coefficient"
