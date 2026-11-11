"""The unified teacher's actor (go2-depth 2026-11-12, doc/redesign.md 3.5): the privileged terms go through a small
encoder into a compact vector before they reach the policy MLP, so a student can learn to estimate that vector from its
own history instead of the raw physics.

UPrivLatentModel(obs, obs_groups, obs_set, ...): MLPModel whose obs_set lists a group named PRIV_GROUP ("actor_priv");
that group is normalized on its own, encoded by an MLP (PRIV_HIDDEN -> PRIV_LATENT), and concatenated after the other
groups' (normalized) observations. Any other set (the critic) is a plain MLPModel.
"""

from __future__ import annotations

import torch
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import MLP, EmpiricalNormalization
from tensordict import TensorDict

PRIV_GROUP = "actor_priv"
PRIV_LATENT = 8
PRIV_HIDDEN = (64,)


class UPrivLatentModel(MLPModel):
  def __init__(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str, output_dim: int,
               hidden_dims=(256, 256, 256), activation: str = "elu", obs_normalization: bool = False,
               distribution_cfg: dict | None = None) -> None:
    groups = list(obs_groups[obs_set])
    self._has_priv = PRIV_GROUP in groups
    self._priv_dim = int(obs[PRIV_GROUP].shape[-1]) if self._has_priv else 0
    base = {**obs_groups, obs_set: [g for g in groups if g != PRIV_GROUP]}
    super().__init__(obs, base, obs_set, output_dim, hidden_dims=hidden_dims, activation=activation,
                     obs_normalization=obs_normalization, distribution_cfg=distribution_cfg)
    if self._has_priv:
      self.priv_normalizer = EmpiricalNormalization(self._priv_dim) if obs_normalization else torch.nn.Identity()
      self.priv_encoder = MLP(self._priv_dim, PRIV_LATENT, PRIV_HIDDEN, activation)

  def _get_latent_dim(self) -> int:
    return self.obs_dim + (PRIV_LATENT if getattr(self, "_has_priv", False) else 0)

  def priv_latent(self, obs: TensorDict) -> torch.Tensor:
    """The compact privileged vector (the students' estimation target)."""
    return self.priv_encoder(self.priv_normalizer(obs[PRIV_GROUP]))

  def get_latent(self, obs: TensorDict, masks=None, hidden_state=None) -> torch.Tensor:
    latent = super().get_latent(obs, masks, hidden_state)
    if not self._has_priv:
      return latent
    return torch.cat([latent, self.priv_latent(obs)], dim=-1)

  def update_normalization(self, obs: TensorDict) -> None:
    super().update_normalization(obs)
    if self._has_priv and self.obs_normalization:
      self.priv_normalizer.update(obs[PRIV_GROUP])

  def as_jit(self) -> torch.nn.Module:
    return _ExportPrivLatent(self) if self._has_priv else super().as_jit()

  def as_onnx(self, verbose: bool) -> torch.nn.Module:
    return _ExportPrivLatent(self, verbose) if self._has_priv else super().as_onnx(verbose)


class _ExportPrivLatent(torch.nn.Module):
  """Deterministic export of a UPrivLatentModel: input = the base groups' observations followed by the privileged
  group's ("obs", as rsl_rl's exports concatenate the actor's groups in obs_groups order: actor, actor_priv)."""

  is_recurrent: bool = False

  def __init__(self, model: UPrivLatentModel, verbose: bool = False) -> None:
    super().__init__()
    import copy

    self.verbose = verbose
    self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
    self.priv_normalizer = copy.deepcopy(model.priv_normalizer)
    self.priv_encoder = copy.deepcopy(model.priv_encoder)
    self.mlp = copy.deepcopy(model.mlp)
    self.deterministic_output = (model.distribution.as_deterministic_output_module()
                                 if model.distribution is not None else torch.nn.Identity())
    self.base_dim = model.obs_dim
    self.input_size = model.obs_dim + model._priv_dim

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    base = self.obs_normalizer(x[..., :self.base_dim])
    z = self.priv_encoder(self.priv_normalizer(x[..., self.base_dim:]))
    return self.deterministic_output(self.mlp(torch.cat([base, z], dim=-1)))

  @torch.jit.export
  def reset(self) -> None:
    pass

  def get_dummy_inputs(self) -> tuple[torch.Tensor]:
    return (torch.zeros(1, self.input_size),)

  @property
  def input_names(self) -> list[str]:
    return ["obs"]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]
