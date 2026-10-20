"""Flat-ground anchor for stairs fine-tunes (go2-blind d9ed8ba): keep the actor
close to a frozen flat policy on flat ground while it learns stairs.

go2-blind's wheeled stairs fine-tunes learned to climb but lost their flat
skills (its Go2W v5 map3 model_8000 missed 26 of the 44 flat acceptance
commands, 3 at its flat start, and crept 33 cm per 4 s at the zero command).
Before each PPO update, AnchoredPPO makes one pass over the rollout's
mini-batches that pulls the actor's mean action toward the frozen teacher's on
the samples whose critic height scan is flat, the only ground the teacher was
trained on; stairs samples are left to PPO alone. Samples commanded to move
straight without a sideways part are left out too: there straight_drift
should improve on the teacher (go2-blind's Go2W run that also left out the
sideways commands lost 15-25% of its sideways speed).

The anchor checkpoint is the flat teacher widened for the heading input
(scripts/go2_depth/widen_obs.py), given by the DEPTH_ANCHOR_CHECKPOINT
environment variable (a path from the repo root), so the job script names it.
"""

from __future__ import annotations

import copy
import math
import os

import torch
from rsl_rl.algorithms import PPO
from torch import nn

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
      if self.anchor_skip_straight:
        command = obs["critic"][:, self.anchor_command]
        straight = (command[:, 2].abs() <= 1e-6) & (command[:, :2].norm(dim=1) >= 0.1)
        if not self.anchor_skip_sideways:
          straight &= command[:, 1].abs() <= 1e-6
        flat &= ~straight
      flat_count += int(flat.sum())
      count += flat.numel()
      if not flat.any():
        continue
      sub = obs[flat]
      with torch.no_grad():
        target = self.teacher(sub)
      loss = self.anchor_coeff * (self.actor(sub) - target).pow(2).mean()
      self.optimizer.zero_grad()
      loss.backward()
      nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
      self.optimizer.step()
      total += loss.item()
      updates += 1
    return total / max(updates, 1), flat_count / max(count, 1)


class AnchoredMirrorVelocityOnPolicyRunner(MirrorVelocityOnPolicyRunner):
  """MirrorVelocityOnPolicyRunner whose PPO is anchored on flat ground to the
  actor of $DEPTH_ANCHOR_CHECKPOINT, sideways commands included (go2-blind's
  heading runs)."""

  ANCHOR_COEFF = 1.0
  ANCHOR_SKIP_SIDEWAYS = False

  def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu"):
    checkpoint = os.environ.get("DEPTH_ANCHOR_CHECKPOINT", "")
    if not checkpoint:
      raise ValueError(f"{type(self).__name__}: set DEPTH_ANCHOR_CHECKPOINT.")
    manager = env.unwrapped.observation_manager
    start, slices = 0, {}
    for name, dims in zip(
      manager.active_terms["critic"], manager.group_obs_term_dim["critic"], strict=True
    ):
      size = int(math.prod(dims))
      slices[name] = (start, start + size)
      start += size
    if "height_scan" not in slices or "command" not in slices:
      raise ValueError("The anchor needs height_scan and command in the critic observation.")
    # The command term may carry a history (oldest first): its last 3 values
    # are the current command.
    command_end = slices["command"][1]
    algorithm = {
      **train_cfg["algorithm"],
      "class_name": "src.tasks.velocity.rl.anchor:AnchoredPPO",
      "anchor_checkpoint": checkpoint,
      "anchor_coeff": self.ANCHOR_COEFF,
      "anchor_skip_sideways": self.ANCHOR_SKIP_SIDEWAYS,
      "anchor_scan_slice": slices["height_scan"],
      "anchor_command_slice": (command_end - 3, command_end),
    }
    super().__init__(env, {**train_cfg, "algorithm": algorithm}, log_dir, device)
