"""Flat-ground anchor for stairs fine-tunes: keep the actor close to a frozen flat
policy on flat ground while it learns stairs.

The wheeled stairs fine-tunes learned to climb but lost their flat skills: Go2W
v5 map3 model_8000 (2026-10-20) missed 26 of the 44 flat acceptance commands (its
flat start: 3) and crept 33 cm per 4 s at the zero command (0.01 cm). Before each
PPO update, AnchoredPPO makes one pass over the rollout's mini-batches that pulls
the actor's mean action toward the frozen teacher's on the samples whose critic
height scan is flat, the only ground the teacher was trained on; stairs samples
are left to PPO alone. Samples commanded to move without turning are left out
too (anchor_skip_straight): there the straight_drift reward should improve on the
teacher, which drifted 3 deg (median) to 10-15 deg (p90) per 10 s at 0.5-1 m/s.
With anchor_skip_sideways=False only those without a sideways part are: the Go2W
run that also left out sideways commands lost 15-25% of its sideways speed
between model_7500 and model_8000 (2026-10-20). With anchor_straight_coeff > 0
the left-out samples are anchored too, at that weight. Go2W and As2W stepped (~1
touchdown per second per foot) when driving straight on flat ground, which their
flat models did on four wheels (2026-10-21); the stepping comes from the stairs
training (lifting ahead of risers a blind policy cannot see), and anchoring the
straight samples at 0.3 only delayed it (Go2W: still on four wheels at model_8100
with it, stepping at model_8000 without it, stepping again from model_8400). The
heavier lift penalty of the Go2W / As2W stairs configs stopped it.

With a gait clock in the critic observation (anchor_phase_slice; wheeled robots
from the seventh round) samples marching in place (zero command, clock running)
are left out: the flat teacher stands still there, which marching must not.
"""

from __future__ import annotations

import copy
import math

import torch
from torch import nn
from rsl_rl.algorithms import PPO

from .runner import MirrorVelocityOnPolicyRunner


class AnchoredPPO(PPO):
  """PPO plus a flat-ground behavior-cloning pass toward a frozen teacher actor."""

  def __init__(
    self,
    actor,
    critic,
    storage,
    anchor_checkpoint: str,
    anchor_coeff: float = 1.0,
    anchor_scan_slice: tuple[int, int] = (0, 0),
    anchor_flat_std: float = 0.005,
    anchor_command_slice: tuple[int, int] = (0, 0),
    anchor_skip_straight: bool = True,
    anchor_skip_sideways: bool = True,
    anchor_straight_coeff: float = 0.0,
    anchor_phase_slice: tuple[int, int] = (0, 0),
    **kwargs,
  ) -> None:
    super().__init__(actor, critic, storage, **kwargs)
    self.teacher = copy.deepcopy(actor)
    checkpoint = torch.load(anchor_checkpoint, map_location=self.device, weights_only=False)
    self.teacher.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    self.teacher.eval()
    for p in self.teacher.parameters():
      p.requires_grad_(False)
    self.anchor_coeff = anchor_coeff
    self.anchor_scan = slice(*anchor_scan_slice)
    self.anchor_flat_std = anchor_flat_std
    self.anchor_command = slice(*anchor_command_slice)
    self.anchor_skip_straight = anchor_skip_straight
    self.anchor_skip_sideways = anchor_skip_sideways
    self.anchor_straight_coeff = anchor_straight_coeff
    self.anchor_phase = slice(*anchor_phase_slice) if anchor_phase_slice[1] > anchor_phase_slice[0] else None

  def update(self) -> dict[str, float]:
    anchor_loss, flat_share = self._anchor_pass()
    loss_dict = super().update()
    loss_dict["anchor"] = anchor_loss
    loss_dict["anchor_flat_share"] = flat_share
    return loss_dict

  def _anchor_pass(self) -> tuple[float, float]:
    total, updates, flat_count, count = 0.0, 0, 0, 0
    for batch in self.storage.mini_batch_generator(self.num_mini_batches, 1):
      obs = batch.observations
      # Flat ground: every ray of the critic's height scan hits the same height.
      flat = obs["critic"][:, self.anchor_scan].std(dim=1) < self.anchor_flat_std
      weight = flat.float()
      if self.anchor_skip_straight:
        command = obs["critic"][:, self.anchor_command]
        straight = (command[:, 2].abs() <= 1e-6) & (command[:, :2].norm(dim=1) >= 0.1)
        if not self.anchor_skip_sideways:
          straight &= command[:, 1].abs() <= 1e-6
        weight = torch.where(straight, weight * self.anchor_straight_coeff, weight)
      if self.anchor_phase is not None:
        command = obs["critic"][:, self.anchor_command]
        marching = (command.norm(dim=1) <= 1e-6) & (obs["critic"][:, self.anchor_phase].norm(dim=1) > 1e-6)
        weight = torch.where(marching, 0.0, weight)
      selected = weight > 0
      flat_count += int(selected.sum())
      count += selected.numel()
      if not selected.any():
        continue
      sub = obs[selected]
      w = weight[selected]
      with torch.no_grad():
        target = self.teacher(sub)
      err = (self.actor(sub) - target).pow(2).mean(dim=1)
      loss = self.anchor_coeff * (w * err).sum() / w.sum()
      self.optimizer.zero_grad()
      loss.backward()
      nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
      self.optimizer.step()
      total += loss.item()
      updates += 1
    return total / max(updates, 1), flat_count / max(count, 1)


class AnchoredMirrorVelocityOnPolicyRunner(MirrorVelocityOnPolicyRunner):
  """MirrorVelocityOnPolicyRunner whose PPO is anchored on flat ground to the
  actor of ANCHOR_CHECKPOINT (a path from the repo root; set in a subclass)."""

  ANCHOR_CHECKPOINT = ""
  ANCHOR_COEFF = 1.0
  ANCHOR_SKIP_SIDEWAYS = True
  ANCHOR_STRAIGHT_COEFF = 0.0

  def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu"):
    if not self.ANCHOR_CHECKPOINT:
      raise ValueError(f"{type(self).__name__}: no ANCHOR_CHECKPOINT.")
    manager = env.unwrapped.observation_manager
    start, slices = 0, {}
    for name, dims in zip(
      manager.active_terms["critic"], manager.group_obs_term_dim["critic"], strict=True
    ):
      size = int(math.prod(dims))
      slices[name] = (start, start + size)
      start += size
    if "height_scan" not in slices or slices.get("command", (0, 0))[1] - slices.get("command", (0, 0))[0] != 3:
      raise ValueError("The anchor needs height_scan and a 3-d command in the critic observation.")
    algorithm = {
      **train_cfg["algorithm"],
      "class_name": "src.tasks.velocity.rl.anchor:AnchoredPPO",
      "anchor_checkpoint": self.ANCHOR_CHECKPOINT,
      "anchor_coeff": self.ANCHOR_COEFF,
      "anchor_skip_sideways": self.ANCHOR_SKIP_SIDEWAYS,
      "anchor_straight_coeff": self.ANCHOR_STRAIGHT_COEFF,
      "anchor_scan_slice": slices["height_scan"],
      "anchor_command_slice": slices["command"],
      "anchor_phase_slice": slices.get("phase", (0, 0)),
    }
    super().__init__(env, {**train_cfg, "algorithm": algorithm}, log_dir, device)
