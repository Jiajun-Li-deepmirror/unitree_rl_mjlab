"""Depth student model, distillation runner and tinynav-sim ONNX export.

The exported ONNX follows the tinynav-sim ``go2_depth_v2`` contract:

  inputs   proprio (1, 47), proprio_history (1, 470),
           depth_history (1, 2, 60, 86), memory_h_in (1, 1, 128)
  outputs  actions (1, 12), memory_h_out (1, 1, 128)
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from rsl_rl.models import MLPModel
from rsl_rl.modules import RNN, HiddenState
from rsl_rl.runners import DistillationRunner
from tensordict import TensorDict

from mjlab.rl import RslRlBaseRunnerCfg, RslRlModelCfg
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata

ONNX_CONTRACT = "go2_depth_v2"
PROPRIO_GROUP = "proprio"
PROPRIO_HISTORY_GROUP = "proprio_history"
DEPTH_GROUP = "depth"


def _depth_encoder(channels: int, height: int, width: int, latent_dim: int) -> nn.Module:
  convs = nn.Sequential(
    nn.Conv2d(channels, 16, kernel_size=5, stride=2),
    nn.ELU(),
    nn.Conv2d(16, 32, kernel_size=3, stride=2),
    nn.ELU(),
    nn.Conv2d(32, 32, kernel_size=3, stride=2),
    nn.ELU(),
    nn.Flatten(),
  )
  with torch.no_grad():
    flat_dim = convs(torch.zeros(1, channels, height, width)).shape[-1]
  return nn.Sequential(convs, nn.Linear(flat_dim, latent_dim), nn.ELU())


class DepthStudentModel(MLPModel):
  """1D obs (normalized) + CNN depth latent -> GRU -> MLP.

  1D observation groups are concatenated in the order listed in the
  observation set; the single 4D group (B, C, H, W) is the depth history.
  """

  is_recurrent: bool = True

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
    activation: str = "elu",
    obs_normalization: bool = True,
    distribution_cfg: dict | None = None,
    depth_latent_dim: int = 64,
    rnn_hidden_dim: int = 128,
  ) -> None:
    self.rnn_hidden_dim = rnn_hidden_dim
    super().__init__(
      obs,
      obs_groups,
      obs_set,
      output_dim,
      hidden_dims,
      activation,
      obs_normalization,
      distribution_cfg,
    )
    channels, height, width = obs[self.depth_group].shape[1:]
    self.depth_shape = (channels, height, width)
    self.depth_encoder = _depth_encoder(channels, height, width, depth_latent_dim)
    self.rnn = RNN(self.obs_dim + depth_latent_dim, rnn_hidden_dim, 1, "gru")

  def _get_obs_dim(
    self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str
  ) -> tuple[list[str], int]:
    groups_1d = []
    obs_dim = 0
    depth_groups = []
    self.group_dims_1d = []
    for group in obs_groups[obs_set]:
      if obs[group].dim() == 4:
        depth_groups.append(group)
      elif obs[group].dim() == 2:
        groups_1d.append(group)
        obs_dim += obs[group].shape[-1]
        self.group_dims_1d.append(obs[group].shape[-1])
      else:
        raise ValueError(f"Invalid observation shape for {group}: {obs[group].shape}")
    if len(depth_groups) != 1:
      raise ValueError(f"Expected exactly one 2D observation group, got {depth_groups}.")
    self.depth_group = depth_groups[0]
    return groups_1d, obs_dim

  def _get_latent_dim(self) -> int:
    return self.rnn_hidden_dim

  def get_latent(
    self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
  ) -> torch.Tensor:
    latent_1d = super().get_latent(obs)
    depth = obs[self.depth_group]
    lead = depth.shape[:-3]
    depth_latent = self.depth_encoder(depth.reshape(-1, *self.depth_shape))
    depth_latent = depth_latent.reshape(*lead, -1)
    latent = torch.cat([latent_1d, depth_latent], dim=-1)
    return self.rnn(latent, masks, hidden_state).squeeze(0)

  def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
    self.rnn.reset(dones, hidden_state)

  def get_hidden_state(self) -> HiddenState:
    return self.rnn.hidden_state  # type: ignore[return-value]

  def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
    self.rnn.detach_hidden_state(dones)

  @property
  def output_std(self) -> torch.Tensor:
    # Deterministic student; the runner logs this as the action std.
    if self.distribution is None:
      return torch.zeros(1)
    return self.distribution.std

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    return _OnnxDepthStudent(self)


class _OnnxDepthStudent(nn.Module):
  """Deterministic single-step export with explicit GRU memory in/out."""

  input_names = ["proprio", "proprio_history", "depth_history", "memory_h_in"]
  output_names = ["actions", "memory_h_out"]

  def __init__(self, model: DepthStudentModel) -> None:
    super().__init__()
    if list(model.obs_groups) != [PROPRIO_GROUP, PROPRIO_HISTORY_GROUP]:
      raise ValueError(
        f"Student 1D groups must be {[PROPRIO_GROUP, PROPRIO_HISTORY_GROUP]}, "
        f"got {model.obs_groups}."
      )
    self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
    self.depth_encoder = copy.deepcopy(model.depth_encoder)
    self.gru = copy.deepcopy(model.rnn.rnn)
    self.mlp = copy.deepcopy(model.mlp)
    if model.distribution is not None:
      self.deterministic_output = model.distribution.as_deterministic_output_module()
    else:
      self.deterministic_output = nn.Identity()
    self.group_dims_1d = list(model.group_dims_1d)
    self.depth_shape = model.depth_shape
    self.hidden_size = model.rnn_hidden_dim

  def forward(
    self,
    proprio: torch.Tensor,
    proprio_history: torch.Tensor,
    depth_history: torch.Tensor,
    memory_h_in: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    latent_1d = self.obs_normalizer(torch.cat([proprio, proprio_history], dim=-1))
    latent = torch.cat([latent_1d, self.depth_encoder(depth_history)], dim=-1)
    out, h = self.gru(latent.unsqueeze(0), memory_h_in)
    actions = self.deterministic_output(self.mlp(out.squeeze(0)))
    return actions, h

  def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
    proprio_dim, history_dim = self.group_dims_1d
    return (
      torch.zeros(1, proprio_dim),
      torch.zeros(1, history_dim),
      torch.zeros(1, *self.depth_shape),
      torch.zeros(1, 1, self.hidden_size),
    )


@dataclass
class DistillationAlgorithmCfg:
  num_learning_epochs: int = 1
  gradient_length: int = 15
  learning_rate: float = 1e-3
  max_grad_norm: float | None = 1.0
  loss_type: str = "mse"
  optimizer: str = "adam"
  class_name: str = "Distillation"


@dataclass
class DepthDistillationRunnerCfg(RslRlBaseRunnerCfg):
  class_name: str = "DistillationRunner"
  teacher_checkpoint: str = ""
  """Path to the teacher PPO checkpoint (model_N.pt). Required unless resuming."""
  student: RslRlModelCfg = field(
    default_factory=lambda: RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      obs_normalization=True,
      class_name="src.tasks.velocity.rl.distillation:DepthStudentModel",
    )
  )
  teacher: RslRlModelCfg = field(
    default_factory=lambda: RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    )
  )
  algorithm: DistillationAlgorithmCfg = field(default_factory=DistillationAlgorithmCfg)


class DepthDistillationRunner(DistillationRunner):
  """DAgger-style distillation that loads the teacher and exports tinynav ONNX."""

  def __init__(
    self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu"
  ) -> None:
    teacher_checkpoint = train_cfg.pop("teacher_checkpoint", "")
    for key in ("student", "teacher"):
      for opt in ("cnn_cfg", "distribution_cfg"):
        if train_cfg[key].get(opt) is None:
          train_cfg[key].pop(opt, None)
    super().__init__(env, train_cfg, log_dir, device)
    if teacher_checkpoint:
      loaded = torch.load(teacher_checkpoint, map_location=device, weights_only=False)
      self.alg.load(loaded, {"teacher": True}, strict=True)
      print(f"[INFO] Loaded teacher from {teacher_checkpoint}")

  def save(self, path: str, infos=None) -> None:
    env_state = {"common_step_counter": self.env.unwrapped.common_step_counter}
    saved_dict = self.alg.save()
    saved_dict["iter"] = self.current_learning_iteration
    saved_dict["infos"] = {**(infos or {}), "env_state": env_state}
    torch.save(saved_dict, path)
    self.export_policy_to_onnx(os.path.dirname(path), "policy.onnx")

  def export_policy_to_onnx(
    self, path: str, filename: str = "policy.onnx", verbose: bool = False
  ) -> None:
    onnx_model = self.alg.get_policy().as_onnx(verbose=verbose)
    onnx_model.to("cpu")
    onnx_model.eval()
    os.makedirs(path, exist_ok=True)
    onnx_path = os.path.join(path, filename)
    torch.onnx.export(
      onnx_model,
      onnx_model.get_dummy_inputs(),
      onnx_path,
      export_params=True,
      opset_version=18,
      verbose=verbose,
      input_names=onnx_model.input_names,
      output_names=onnx_model.output_names,
      dynamic_axes={},
      dynamo=False,
    )
    env = self.env.unwrapped
    metadata = get_base_metadata(env, "local")
    metadata["observation_names"] = env.observation_manager.active_terms[PROPRIO_GROUP]
    metadata["contract"] = ONNX_CONTRACT
    attach_metadata_to_onnx(onnx_path, metadata)
