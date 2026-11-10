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
from rsl_rl.algorithms import Distillation
from rsl_rl.models import MLPModel
from rsl_rl.modules import RNN, HiddenState
from rsl_rl.runners import DistillationRunner
from tensordict import TensorDict

from mjlab.rl import RslRlBaseRunnerCfg, RslRlModelCfg
from src.tasks.velocity.mdp.depth import DEPTH_CAMERAS, DEPTH_MAX_M, FRONT_DEPTH_CAMERA
from src.tasks.velocity.rl.obs_guard import guard_nan_observations
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata

ONNX_CONTRACT = "go2_depth_v2"
"""Default `contract` metadata (tinynav-sim's Go2 student interface)."""
PROPRIO_GROUP = "proprio"
PROPRIO_HISTORY_GROUP = "proprio_history"
DEPTH_GROUP = "depth"
TERRAIN_TARGET_GROUP = "terrain_target"


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

  When the observations carry a `terrain_target` group (training only), an
  auxiliary head reconstructs it from the GRU state. The camera cannot see
  under the body, so this makes the memory keep the steps the feet are on.
  The head is not part of the exported policy.
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
    self.aux_head = None
    if TERRAIN_TARGET_GROUP in obs.keys():
      self.aux_head = nn.Sequential(
        nn.Linear(rnn_hidden_dim, 256), nn.ELU(), nn.Linear(256, obs[TERRAIN_TARGET_GROUP].shape[-1])
      )
    self.last_latent: torch.Tensor | None = None

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
    self.last_latent = self.rnn(latent, masks, hidden_state).squeeze(0)
    return self.last_latent

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


class DepthDistillation(Distillation):
  """rsl_rl Distillation plus the student's terrain reconstruction loss."""

  def __init__(self, *args, aux_loss_coef: float = 0.0, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.aux_loss_coef = aux_loss_coef

  def update(self) -> dict[str, float]:
    if not self.aux_loss_coef or getattr(self.student, "aux_head", None) is None:
      return super().update()
    # Same as Distillation.update, plus the auxiliary loss on the GRU state.
    self.num_updates += 1
    mean_behavior_loss = mean_aux_loss = 0.0
    loss = 0
    cnt = 0
    for _ in range(self.num_learning_epochs):
      self.student.reset(hidden_state=self.last_hidden_states[0])
      self.teacher.reset(hidden_state=self.last_hidden_states[1])
      self.student.detach_hidden_state()
      for batch in self.storage.generator():
        actions = self.student(batch.observations)
        behavior_loss = self.loss_fn(actions, batch.privileged_actions)
        aux_pred = self.student.aux_head(self.student.last_latent)
        aux_loss = nn.functional.mse_loss(aux_pred, batch.observations[TERRAIN_TARGET_GROUP])
        loss = loss + behavior_loss + self.aux_loss_coef * aux_loss
        mean_behavior_loss += behavior_loss.item()
        mean_aux_loss += aux_loss.item()
        cnt += 1
        if cnt % self.gradient_length == 0:
          self.optimizer.zero_grad()
          loss.backward()
          if self.is_multi_gpu:
            self.reduce_parameters()
          if self.max_grad_norm:
            nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
          self.optimizer.step()
          self.student.detach_hidden_state()
          loss = 0
        self.student.reset(batch.dones.view(-1))
        self.teacher.reset(batch.dones.view(-1))
        self.student.detach_hidden_state(batch.dones.view(-1))
    self.storage.clear()
    self.last_hidden_states = (self.student.get_hidden_state(), self.teacher.get_hidden_state())
    self.student.detach_hidden_state()
    return {"behavior": mean_behavior_loss / cnt, "terrain_aux": mean_aux_loss / cnt}


@dataclass
class DistillationAlgorithmCfg:
  num_learning_epochs: int = 1
  gradient_length: int = 15
  learning_rate: float = 1e-3
  max_grad_norm: float | None = 1.0
  loss_type: str = "mse"
  optimizer: str = "adam"
  aux_loss_coef: float = 0.0
  """Weight of the student's terrain reconstruction loss (0 disables it)."""
  class_name: str = "src.tasks.velocity.rl.distillation:DepthDistillation"


@dataclass
class DepthDistillationRunnerCfg(RslRlBaseRunnerCfg):
  class_name: str = "DistillationRunner"
  teacher_checkpoint: str = ""
  """Path to the teacher PPO checkpoint (model_N.pt). Required unless resuming."""
  student_checkpoint: str = ""
  """Optional distillation checkpoint to start the student (and its optimizer)
  from, keeping the teacher from `teacher_checkpoint`. Unlike resume, which
  would also restore the old teacher."""
  learning_rate_final: float | None = None
  """If set, decay the learning rate log-linearly from algorithm.learning_rate
  to this over the run."""
  onnx_contract: str = ONNX_CONTRACT
  """`contract` metadata of the exported ONNX."""
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
    student_checkpoint = train_cfg.pop("student_checkpoint", "")
    self.onnx_contract = train_cfg.pop("onnx_contract", ONNX_CONTRACT)
    self.learning_rate_final = train_cfg.pop("learning_rate_final", None)
    for key in ("student", "teacher"):
      for opt in ("cnn_cfg", "distribution_cfg"):
        if train_cfg[key].get(opt) is None:
          train_cfg[key].pop(opt, None)
    super().__init__(env, train_cfg, log_dir, device)
    guard_nan_observations(self.env)
    if teacher_checkpoint:
      loaded = torch.load(teacher_checkpoint, map_location=device, weights_only=False)
      self.alg.load(loaded, {"teacher": True}, strict=True)
      print(f"[INFO] Loaded teacher from {teacher_checkpoint}")
    if student_checkpoint:
      loaded = torch.load(student_checkpoint, map_location=device, weights_only=False)
      missing, unexpected = self.alg.student.load_state_dict(
        loaded["student_state_dict"], strict=False
      )
      if unexpected or any(not k.startswith("aux_head.") for k in missing):
        raise RuntimeError(f"Student checkpoint mismatch: {missing=} {unexpected=}")
      # Same parameters: keep the optimizer state too (not from a PPO fine-tune's, rl/student_ppo.py).
      if not missing and "actor_state_dict" not in loaded:
        self.alg.optimizer.load_state_dict(loaded["optimizer_state_dict"])
      print(f"[INFO] Loaded student from {student_checkpoint} (new: {sorted({k.split('.')[0] for k in missing})})")

  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
    if self.learning_rate_final is not None:
      start, lr0, update = self.current_learning_iteration, self.alg.learning_rate, self.alg.update

      def update_with_schedule() -> dict[str, float]:
        frac = min(1.0, (self.current_learning_iteration - start) / max(1, num_learning_iterations - 1))
        self.alg.learning_rate = lr0 * (self.learning_rate_final / lr0) ** frac
        for group in self.alg.optimizer.param_groups:
          group["lr"] = self.alg.learning_rate
        return update()

      self.alg.update = update_with_schedule
    super().learn(num_learning_iterations, init_at_random_ep_len)

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
    metadata["contract"] = self.onnx_contract
    # The joints in the order joint_vel reads them, which tinynav-sim takes from
    # joint_names: go2-blind's wheeled tasks list the legs, then the wheels
    # (preserve_order), not the model's per-leg order.
    proprio_terms = list(env.observation_manager.active_terms[PROPRIO_GROUP])
    if "joint_vel" in proprio_terms:
      term_cfg = env.observation_manager._group_obs_term_cfgs[PROPRIO_GROUP][proprio_terms.index("joint_vel")]  # noqa: SLF001
      asset_cfg = term_cfg.params.get("asset_cfg")
      ids = None if asset_cfg is None else asset_cfg.joint_ids
      if ids is not None and not isinstance(ids, slice):
        order = [int(i) for i in ids]
        assert sorted(order) == list(range(len(metadata["joint_names"]))), "joint_vel must read every joint"
        for key in ("joint_names", "joint_stiffness", "joint_damping", "default_joint_pos"):
          metadata[key] = [metadata[key][i] for i in order]
    # The `actions` output is the action terms concatenated in this order,
    # e.g. "joint_pos[position]*0.25:FL_hip_joint,...;wheel_vel[velocity]*10:..."
    # (wheel speed targets have no offset; joint targets add the default pose).
    layout = []
    for name in env.action_manager.active_terms:
      term = env.action_manager.get_term(name)
      kind = "velocity" if "vel" in type(term).__name__.lower() else "position"
      scale = term.cfg.scale
      layout.append(f"{name}[{kind}]*{scale:g}:" + ",".join(term.target_names))
      if hasattr(term.cfg, "brake_rad_s"):
        # Deploy zeroes these velocity targets at a zero command once all of
        # them are slower than this (mdp/wheel_action.py).
        metadata["wheel_brake_rad_s"] = f"{term.cfg.brake_rad_s:.6g}"
      elif getattr(term.cfg, "brake_speed", float("inf")) < float("inf"):
        # go2-blind's brake (mdp/actions.py BrakedJointVelocityAction), the same rule.
        metadata["wheel_brake_rad_s"] = f"{term.cfg.brake_speed:.6g}"
        # Its hold, as deploy.yaml's brake_hold_kp / brake_after_s / brake_release_ratio.
        if term.cfg.hold_stiffness > 0.0:
          metadata["wheel_brake_hold_kp"] = f"{term.cfg.hold_stiffness:.6g}"
        if term.cfg.brake_after_s is not None:
          metadata["wheel_brake_after_s"] = f"{term.cfg.brake_after_s:.6g}"
        if term.cfg.release_speed_ratio is not None:
          metadata["wheel_brake_release_ratio"] = f"{term.cfg.release_speed_ratio:.6g}"
        # brake_max_forward (2026-11-02, As2W / B2W from 10-31): it also brakes up to that forward command.
        if getattr(term.cfg, "brake_max_forward", None) is not None:
          metadata["wheel_brake_max_forward"] = f"{term.cfg.brake_max_forward:.6g}"
      if getattr(term.cfg, "hold_after_s", None) is not None:
        # go2-blind's standstill leg hold (mdp/actions.py HeldJointPositionAction; B2W from 11-01), as
        # deploy.yaml's hold_after_s / hold_speed / hold_release_speed; tinynav-sim contract leg_hold_*.
        metadata["leg_hold_after_s"] = f"{term.cfg.hold_after_s:.6g}"
        metadata["leg_hold_speed"] = f"{term.cfg.hold_speed:.6g}"
        metadata["leg_hold_release_speed"] = f"{term.cfg.release_speed:.6g}"
    metadata["action_layout"] = ";".join(layout)
    # go2-blind's speed clock (its deploy.yaml gait_phase and gait_march, as
    # scripts/blind/export_deploy_yaml.py writes them; tinynav-sim GaitClock).
    group = env.observation_manager._group_obs_term_cfgs[PROPRIO_GROUP]  # noqa: SLF001
    names = list(env.observation_manager.active_terms[PROPRIO_GROUP])
    if "phase" in names and group[names.index("phase")].params.get("ref_speed") is not None:
      clock = group[names.index("phase")].params
      metadata["gait_period"] = repr(float(clock["period"]))
      metadata["gait_ref_speed"] = repr(float(clock["ref_speed"]))
      if clock.get("foot_xy") is not None:
        metadata["gait_foot_xy"] = ";".join(",".join(repr(float(v)) for v in f) for f in clock["foot_xy"])
      twist = env.command_manager.get_term(clock["command_name"]).cfg
      metadata["gait_march"] = "1" if getattr(twist, "march_prob", 0.0) > 0.0 else "0"
      # Wheeled: the phase reads zero under a command the wheels drive.
      for key in ("drivable_max_lin_vel_y", "drivable_max_ang_vel_z"):
        if clock.get(key) is not None:
          metadata[f"gait_{key}"] = repr(float(clock[key]))
    # The contact solver it trained on (go2-blind's "grip": elliptic, impratio 10);
    # tinynav-sim sets its model to it.
    metadata["sim_cone"] = str(env.cfg.sim.mujoco.cone)
    metadata["sim_impratio"] = f"{float(env.cfg.sim.mujoco.impratio):.6g}"
    # The depth camera's nominal mount (base frame; training randomizes it around
    # this): the runtime places its camera there.
    camera = next(c for c in env.cfg.scene.entities["robot"].cameras if c.name == FRONT_DEPTH_CAMERA)
    metadata["depth_camera_body"] = camera.body
    metadata["depth_camera_pos"] = ",".join(repr(float(v)) for v in camera.pos)
    metadata["depth_camera_quat"] = ",".join(repr(float(v)) for v in camera.quat)
    # What a depth_history frame holds: width x height pixels of a pinhole with
    # its principal point at the centre, focal length in pixels, optical Z or ray
    # distance, normalized to [0.05, max] / max m.
    spec = DEPTH_CAMERAS[env.cfg.observations[DEPTH_GROUP].terms["depth"].params.get("camera", "pie")]
    metadata["depth_camera"] = spec.name
    metadata["depth_size"] = f"{spec.width},{spec.height}"
    metadata["depth_focal_px"] = repr(spec.focal_px)
    metadata["depth_kind"] = "optical_z" if spec.optical_z else "ray"
    metadata["depth_max_m"] = repr(DEPTH_MAX_M)
    attach_metadata_to_onnx(onnx_path, metadata)
