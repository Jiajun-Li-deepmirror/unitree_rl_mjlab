"""PPO fine-tune of a distilled depth student (go2-depth, 2026-11-02, the user's choice for B2).

The B2 depth_fe students fell 1 of 256 robots in two flat checks in every checkpoint and round (walking sideways at
-0.6 m/s with a +8.6 kg payload, spinning up to 1.5 rad/s on foot friction 0.31) where their teacher fell 0; more of
those commands (DEPTH_STUDENT_HARD) and robots (DEPTH_STUDENT_DR_EDGE) in the distillation did not move them. On flat
ground the student sees what the teacher's actor sees (proprioception and its 10-step history; the teacher's extra
input is the terrain scan), so these are imitation errors where a small one is fatal, which the distillation's
action error does not weigh. Here the distilled student is the PPO actor on the training reward:

- actor: the student (DepthStudentModel) with a Gaussian at teacher_std_scale x the teacher's final std, its
  weights from a distillation checkpoint (student_state_dict);
- critic: the depth teacher's (the same "critic" group), weights and normalizer from its PPO checkpoint;
- a behavior-cloning term toward the frozen teacher's mean action on every sample (bc_coef, log-linearly to
  bc_coef_final over the run), so it improves on the teacher's behavior without leaving it;
- the first critic_warmup updates train the critic alone (the teacher's critic, for a slightly different actor);
- the learning rate adapts to desired_kl as in PPO, under a ceiling that goes log-linearly from learning_rate to
  learning_rate_final;
- with bc_jump_weight > 1 the behavior cloning weighs the steps within the student's proprio history (10 steps,
  0.2 s) of a command jump (|dvx| or |dvy| >= bc_jump_v, |dwz| >= bc_jump_wz) that much more (2026-11-02, B2:
  go2-home's spin_compare.py: every B2 student turned harder than its teacher in the first 0.12 s of a spin-up
  from a stand, where whether a low-friction robot fell was decided, and a spin-up is ~5 of an episode's steps).

Checkpoints also hold student_state_dict / teacher_state_dict as a distillation checkpoint does (the distribution
dropped), so the student averaging, the checks and the export read them unchanged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO
from rsl_rl.utils import resolve_callable, unpad_trajectories

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.rl.runner import MjlabOnPolicyRunner
from src.tasks.velocity.rl.distillation import ONNX_CONTRACT, DepthDistillationRunner
from src.tasks.velocity.rl.obs_guard import guard_nan_observations

TEACHER_SET = "teacher"


class StudentPPO(PPO):
  """PPO for a recurrent student actor plus behavior cloning toward a frozen teacher (set by the runner)."""

  def __init__(
    self,
    actor,
    critic,
    storage,
    bc_coef: float = 1.0,
    bc_coef_final: float | None = None,
    critic_warmup: int = 0,
    bc_jump_weight: float = 1.0,
    bc_jump_v: float = 0.5,
    bc_jump_wz: float = 1.0,
    **kwargs,
  ) -> None:
    super().__init__(actor, critic, storage, **kwargs)
    if not self.actor.is_recurrent:
      raise ValueError("StudentPPO expects the recurrent depth student as its actor.")
    self.teacher: nn.Module | None = None
    self.bc_coef = bc_coef
    self.bc_coef_final = bc_coef_final
    self.critic_warmup = critic_warmup
    self.lr_ceiling = self.learning_rate
    self.num_updates = 0
    self.bc_jump_weight, self.bc_jump_v, self.bc_jump_wz = bc_jump_weight, bc_jump_v, bc_jump_wz
    # (offset, frames, scale) of the command term in proprio_history (set by the runner)
    self.jump_layout: tuple[int, int, torch.Tensor] | None = None

  def update(self) -> dict[str, float]:
    # PPO.update for a recurrent actor (no symmetry, no RND), plus the behavior-cloning term and the critic warm-up.
    assert self.teacher is not None, "the runner sets the teacher"
    warmup = self.num_updates < self.critic_warmup
    self.num_updates += 1
    self.learning_rate = min(self.learning_rate, self.lr_ceiling)
    for param_group in self.optimizer.param_groups:
      param_group["lr"] = self.learning_rate
    sums = {"value": 0.0, "surrogate": 0.0, "entropy": 0.0, "bc": 0.0, "bc_jump_share": 0.0}
    generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
    for batch in generator:
      if self.normalize_advantage_per_mini_batch:
        with torch.no_grad():
          batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)  # type: ignore
      self.actor(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[0], stochastic_output=True)
      actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
      values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
      distribution_params = self.actor.output_distribution_params
      entropy = self.actor.output_entropy
      with torch.no_grad():
        teacher_mean = self.teacher(batch.observations, masks=batch.masks)
      sq = (self.actor.output_mean - teacher_mean).pow(2).mean(dim=-1)
      if self.bc_jump_weight > 1.0 and self.jump_layout is not None:
        with torch.no_grad():
          off, frames, scale = self.jump_layout
          hist = unpad_trajectories(batch.observations["proprio_history"], batch.masks)
          cmd = hist[..., off:off + 3 * frames].reshape(*hist.shape[:-1], frames, 3) / scale
          d = (cmd[..., -1:, :] - cmd).abs().amax(dim=-2)
          jump = (d[..., :2].amax(dim=-1) >= self.bc_jump_v) | (d[..., 2] >= self.bc_jump_wz)
          w = 1.0 + (self.bc_jump_weight - 1.0) * jump.float()
        bc_loss = (sq * w).sum() / w.sum()
        sums["bc_jump_share"] += jump.float().mean().item()
      else:
        bc_loss = sq.mean()

      if self.desired_kl is not None and self.schedule == "adaptive" and not warmup:
        with torch.inference_mode():
          kl_mean = torch.mean(self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params))  # type: ignore
          if kl_mean > self.desired_kl * 2.0:
            self.learning_rate = max(1e-6, self.learning_rate / 1.5)
          elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
            self.learning_rate = min(self.lr_ceiling, self.learning_rate * 1.5)
          for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

      ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore
      surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore
      surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(  # type: ignore
        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
      )
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
      if self.use_clipped_value_loss:
        value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
        value_loss = torch.max((values - batch.returns).pow(2), (value_clipped - batch.returns).pow(2)).mean()
      else:
        value_loss = (batch.returns - values).pow(2).mean()

      extra = self.extra_loss(batch)
      if warmup:
        loss = self.value_loss_coef * value_loss
      else:
        loss = (
          surrogate_loss
          + self.value_loss_coef * value_loss
          - self.entropy_coef * entropy.mean()
          + self.bc_coef * bc_loss
        )
        if extra is not None:
          loss = loss + extra
      self.optimizer.zero_grad()
      loss.backward()
      nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
      nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
      self.optimizer.step()

      sums["value"] += value_loss.item()
      sums["surrogate"] += surrogate_loss.item()
      sums["entropy"] += entropy.mean().item()
      sums["bc"] += bc_loss.item()

    self.storage.clear()
    num_updates = self.num_learning_epochs * self.num_mini_batches
    out = {k: v / num_updates for k, v in sums.items()}
    out["bc_coef"] = self.bc_coef
    out["critic_warmup"] = float(warmup)
    return out

  def extra_loss(self, batch) -> torch.Tensor | None:
    """A subclass's extra term for the actor, after the actor's forward pass on the mini-batch (None: none; added
    outside the critic warm-up). src/tasks/unified/students.py's UStudentPPO: the privileged-latent estimate."""
    del batch
    return None

  def save(self) -> dict:
    saved = super().save()
    # As a distillation checkpoint: the deterministic student and the teacher.
    saved["student_state_dict"] = {
      k: v for k, v in saved["actor_state_dict"].items() if not k.startswith("distribution.")
    }
    saved["teacher_state_dict"] = self.teacher.state_dict()
    return saved

  def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    if self.teacher is not None and "teacher_state_dict" in loaded_dict:
      self.teacher.load_state_dict(loaded_dict["teacher_state_dict"], strict=strict)
    return super().load(loaded_dict, load_cfg, strict)


@dataclass
class StudentPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  learning_rate: float = 1e-4
  desired_kl: float = 0.005
  entropy_coef: float = 0.0
  bc_coef: float = 1.0
  """Weight of the mean squared error between the student's and the teacher's mean actions."""
  bc_coef_final: float | None = None
  """If set, bc_coef goes log-linearly to this over the run."""
  critic_warmup: int = 50
  """Updates that train the critic alone at the start."""
  bc_jump_weight: float = 1.0
  """Weight of the steps after a command jump in the behavior cloning (1: off; see StudentPPO)."""
  bc_jump_v: float = 0.5
  bc_jump_wz: float = 1.0
  class_name: str = "src.tasks.velocity.rl.student_ppo:StudentPPO"


@dataclass
class DepthStudentPpoRunnerCfg(RslRlOnPolicyRunnerCfg):
  teacher_checkpoint: str = ""
  """The depth teacher's PPO checkpoint (model_N.pt): the behavior-cloning target and the critic's start."""
  student_checkpoint: str = ""
  """The distilled student (a distillation checkpoint's student_state_dict): the actor's start."""
  learning_rate_final: float | None = None
  """If set, the learning rate's ceiling goes log-linearly from algorithm.learning_rate to this over the run."""
  lr_ramp_iters: int = 0
  """If > 0, the ceiling rises linearly from lr_ramp_start over this many iterations first (2026-11-02, As2-W:
  go2-home, the first actor updates after the critic warm-up gave a fast right spin a falling tail, m100 5/256,
  which the later updates fixed again)."""
  lr_ramp_start: float = 1e-5
  teacher_std_scale: float = 1.0
  """The actor's action std starts at this x the teacher's final std."""
  onnx_contract: str = ONNX_CONTRACT
  """`contract` metadata of the exported ONNX."""
  actor: RslRlModelCfg = field(
    default_factory=lambda: RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      obs_normalization=True,
      distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.4, "std_type": "scalar"},
      class_name="src.tasks.velocity.rl.distillation:DepthStudentModel",
    )
  )
  critic: RslRlModelCfg = field(
    default_factory=lambda: RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True)
  )
  teacher: RslRlModelCfg = field(
    default_factory=lambda: RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      obs_normalization=True,
      distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    )
  )
  algorithm: StudentPpoAlgorithmCfg = field(default_factory=StudentPpoAlgorithmCfg)


class DepthStudentPpoRunner(MjlabOnPolicyRunner):
  """Runs StudentPPO: loads the student, the teacher and its critic, and saves and exports as the distillation."""

  def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
    teacher_checkpoint = train_cfg.pop("teacher_checkpoint", "")
    student_checkpoint = train_cfg.pop("student_checkpoint", "")
    self.learning_rate_final = train_cfg.pop("learning_rate_final", None)
    self.lr_ramp_iters = int(train_cfg.pop("lr_ramp_iters", 0))
    self.lr_ramp_start = float(train_cfg.pop("lr_ramp_start", 1e-5))
    teacher_std_scale = train_cfg.pop("teacher_std_scale", 1.0)
    self.onnx_contract = train_cfg.pop("onnx_contract", ONNX_CONTRACT)
    teacher_cfg = dict(train_cfg.pop("teacher"))
    for opt in ("cnn_cfg", "distribution_cfg"):
      if teacher_cfg.get(opt) is None:
        teacher_cfg.pop(opt, None)
    if teacher_cfg.get("distribution_cfg") is not None:
      teacher_cfg["distribution_cfg"] = dict(teacher_cfg["distribution_cfg"])
    super().__init__(env, train_cfg, log_dir, device)
    guard_nan_observations(self.env)

    obs = self.env.get_observations()
    teacher_class = resolve_callable(teacher_cfg.pop("class_name"))
    teacher = teacher_class(obs, self.cfg["obs_groups"], TEACHER_SET, self.env.num_actions, **teacher_cfg).to(device)
    teacher.eval()
    for p in teacher.parameters():
      p.requires_grad_(False)
    self.alg.teacher = teacher
    if teacher_checkpoint:
      loaded = torch.load(teacher_checkpoint, map_location=device, weights_only=False)
      teacher.load_state_dict(loaded["actor_state_dict"], strict=True)
      self.alg.critic.load_state_dict(loaded["critic_state_dict"], strict=True)
      print(f"[INFO] Loaded teacher and its critic from {teacher_checkpoint}")
    if student_checkpoint:
      loaded = torch.load(student_checkpoint, map_location=device, weights_only=False)
      missing, unexpected = self.alg.actor.load_state_dict(loaded["student_state_dict"], strict=False)
      if unexpected or any(not k.startswith(("distribution.", "aux_head.", "priv_head.")) for k in missing):
        raise RuntimeError(f"Student checkpoint mismatch: {missing=} {unexpected=}")
      print(f"[INFO] Loaded student from {student_checkpoint} (new: {sorted({k.split('.')[0] for k in missing})})")
    # where the command sits in the student's proprio history, for bc_jump_weight
    om = self.env.unwrapped.observation_manager
    names = list(om.active_terms["proprio"])
    dims = [int(torch.tensor(d).prod()) for d in om.group_obs_term_dim["proprio"]]
    if "command" in names and dims[names.index("command")] == 3:
      i = names.index("command")
      frames = int(om.group_obs_dim["proprio_history"][0]) // sum(dims)
      scale = om._group_obs_term_cfgs["proprio"][i].scale  # noqa: SLF001
      scale = torch.as_tensor(scale if scale is not None else 1.0, dtype=torch.float, device=device).expand(3)
      self.alg.jump_layout = (sum(dims[:i]) * frames, frames, scale)
      if self.alg.bc_jump_weight > 1.0:
        print(f"[INFO] BC x{self.alg.bc_jump_weight:g} within {frames} steps of a command jump "
              f"(|dv| >= {self.alg.bc_jump_v:g}, |dwz| >= {self.alg.bc_jump_wz:g}; command scale {scale.tolist()})")
    if teacher_checkpoint and student_checkpoint:
      std = self.alg.actor.distribution
      with torch.no_grad():
        std.std_param.copy_(teacher.distribution.std_param * teacher_std_scale)
      print(f"[INFO] Actor std from the teacher's x {teacher_std_scale}: {std.std_param.detach().cpu().numpy().round(3)}")

  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
    start, lr0, bc0, update = self.current_learning_iteration, self.alg.learning_rate, self.alg.bc_coef, self.alg.update

    def update_with_schedule() -> dict[str, float]:
      frac = min(1.0, (self.current_learning_iteration - start) / max(1, num_learning_iterations - 1))
      if self.learning_rate_final is not None:
        self.alg.lr_ceiling = lr0 * (self.learning_rate_final / lr0) ** frac
      done = self.current_learning_iteration - start
      if 0 < self.lr_ramp_iters and done < self.lr_ramp_iters:
        ceiling = self.alg.lr_ceiling
        self.alg.lr_ceiling = self.lr_ramp_start + (ceiling - self.lr_ramp_start) * done / self.lr_ramp_iters
      if self.alg.bc_coef_final is not None and bc0 > 0:
        self.alg.bc_coef = bc0 * (self.alg.bc_coef_final / bc0) ** frac
      return update()

    self.alg.update = update_with_schedule
    super().learn(num_learning_iterations, init_at_random_ep_len)

  def save(self, path: str, infos=None) -> None:
    super().save(path, infos)
    self.export_policy_to_onnx(os.path.dirname(path), "policy.onnx")

  export_policy_to_onnx = DepthDistillationRunner.export_policy_to_onnx
