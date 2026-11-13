"""The unified pipeline's students (go2-depth 2026-11-12, doc/redesign.md 3.6): a blind and a depth student distilled
from one unified teacher v2, then fine-tuned with PPO plus behavior cloning toward it.

Environment: the teacher's own (teacher_env.unified_teacher_env_cfg with student=...): the same terrain, commands,
payload, randomization and constraint costs, the curriculum at its end (progress 1; the step-clocked schedules of the
blind builder too: student_sched.at_end) and the constraint scales frozen at the teacher's final values (restored
from its checkpoint by the runners). The teacher's scan mode is fixed:
  blind  scan off -- the teacher walks in its blind mode; the student sees the proprio terms only, as the group
         "actor" with the teacher's history: the deployed blind policies' input, so the blind deploy contract
         (jobs/export_policy.sh layout, robot_ctrl's blind controller, tinynav-sim's import) is unchanged
  depth  scan on -- the teacher sees the terrain; the student gets the front depth camera as the deployed depth
         students (apply_depth_student with the robot's navcore mount and the real camera's errors): groups proprio,
         proprio_history, depth; the tinynav-sim depth contract
The teacher reads its actor group (proprio + scan + scan flag; renamed "teacher" for the blind student) and
"actor_priv".

Stages (rsl_rl): 1 = DAgger distillation (the student acts, the teacher labels: rl/distillation.py's runner);
2 = PPO + behavior cloning toward the teacher's mean action, the teacher's critic as the start (rl/student_ppo.py's).

The privileged latent (2026-11-12, the review's item 8; doc/redesign.md 3.5 "the students estimate it from their
history"): both students carry a head (priv_head, not exported) that estimates the teacher's compact privileged vector
(models.UPrivLatentModel.priv_latent: CoM, payload, friction, true velocity, foot states -> 8 numbers) from the
student's own features (the blind MLP's last hidden layer, the depth student's GRU state). With priv_coef 0 (the
default) the head is a probe: it trains on the detached features with its own optimizer, so the student is exactly as
before and the log says how much of the vector its features hold; with priv_coef > 0 the estimation error is part of
the student's loss (RMA-style). Logged as Loss/priv_mse and Loss/priv_fvu (the error over the target's variance:
1 = nothing estimated, 0 = all of it). Set with --agent.algorithm.priv-coef.
"""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass

import torch
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.rl import RslRlModelCfg
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP
from rsl_rl.utils import unpad_trajectories

from src.tasks.velocity.depth_env_cfg import NAVCORE_CAMERA_POS, NAVCORE_CAMERA_QUAT, TEACHER_SCAN_NAME, apply_depth_student
from src.tasks.velocity.mdp.depth import NAVCORE_DEPTH_CAMERA
from src.tasks.velocity.rl.distillation import (
  TERRAIN_TARGET_GROUP,
  DepthDistillation,
  DepthDistillationRunner,
  DepthDistillationRunnerCfg,
  DepthStudentModel,
  DistillationAlgorithmCfg,
)
from rsl_rl.algorithms import PPO
from torch import nn

from src.tasks.velocity.rl.student_ppo import (
  DepthStudentPpoRunner,
  DepthStudentPpoRunnerCfg,
  StudentPPO,
  StudentPpoAlgorithmCfg,
)

from . import diagnostics, student_sched
from .models import PRIV_GROUP, PRIV_LATENT
from .robots import URobot
from .teacher_env import unified_teacher_env_cfg

STUDENT_GROUPS = {"blind": ("actor",), "depth": ("proprio", "proprio_history", "depth")}
# the teacher's actor group: renamed "teacher" for the blind student (whose input is "actor", the blind deploy
# contract's group); kept as "actor" for the depth student (its export's metadata reads "actor", as before)
TEACHER_GROUPS = {"blind": ("teacher", PRIV_GROUP), "depth": ("actor", PRIV_GROUP)}
TEACHER_SET = "teacher"
TEACHER_MODEL = "src.tasks.unified.models:UPrivLatentModel"
S1_ITERATIONS = 4000
S2_ITERATIONS = 3000


# The checks' foot friction where the blind builder's check terrain differs from the training's (2026-11-12, the user:
# "统一成 0.4–1.0"): the wheeled robots train at 0.4-1.0 (blind_env_cfg.py sets it on any terrain with stairs: a
# high-friction wheel climbs a riser face, a simulator exploit), and their stairs check had it too, but the flat check
# kept the default 0.3-1.6 -- outside the training's range. Both checks now draw from the training's range.
CHECK_FRICTION = {"wheeled": (0.4, 1.0)}


def student_env_cfg(robot: URobot, kind: str, play: bool = False, terrain: str = "map3"):
  assert kind in STUDENT_GROUPS, kind
  cfg = unified_teacher_env_cfg(robot, play=play, student=kind, terrain=terrain)
  if terrain == "map3" and not play:
    # the training build: the schedules on the environment's step clock at their end too (2026-11-13,
    # student_sched.py: a student's clock starts at 0, so it trained under the teacher's first iterations' values)
    student_sched.at_end(cfg)
  if terrain != "map3" and robot.kind in CHECK_FRICTION and "foot_friction" in cfg.events:
    cfg.events["foot_friction"].params["ranges"] = CHECK_FRICTION[robot.kind]
  if kind == "depth":
    kw = {} if play else {"real_camera": True, "occlusion": True, "edge_blur": True}
    cfg = apply_depth_student(cfg, play=play, camera_pos=NAVCORE_CAMERA_POS[robot.key],
                              camera_quat=NAVCORE_CAMERA_QUAT, camera=NAVCORE_DEPTH_CAMERA, **kw)
  if kind == "blind":
    teacher = cfg.observations.pop("actor")
    cfg.observations["teacher"] = teacher
    student = deepcopy(teacher)
    student.terms = {k: deepcopy(v) for k, v in teacher.terms.items() if k not in (TEACHER_SCAN_NAME, "scan_flag")}
    cfg.observations["actor"] = student
    # rl/student_ppo.py's runner reads a "proprio" group (the command's place in the depth student's history, for its
    # command-jump BC weight): one term without the command keeps that off for the blind student
    cfg.observations["proprio"] = ObservationGroupCfg(terms={"base_ang_vel": deepcopy(teacher.terms["base_ang_vel"])},
                                                      concatenate_terms=True, enable_corruption=False)
    cfg.observations["proprio"].terms["base_ang_vel"].history_length = 0
  return cfg


# ---- the privileged-latent estimate (module docstring) ----
PRIV_HEAD_HIDDEN = (64,)
PRIV_PROBE_LR = 1e-3
BLIND_STUDENT_MODEL = "src.tasks.unified.students:UBlindStudentModel"
DEPTH_STUDENT_MODEL = "src.tasks.unified.students:UDepthStudentModel"


def _priv_head(obs, in_dim: int) -> nn.Module | None:
  """The estimate's head, when the observations carry the teacher's privileged group (the students' environments
  do; any other environment builds the plain model)."""
  return MLP(in_dim, PRIV_LATENT, PRIV_HEAD_HIDDEN, "elu") if PRIV_GROUP in obs.keys() else None


class UBlindStudentModel(MLPModel):
  """The blind student: rsl_rl's MLPModel (the deployed blind actor; its export, _OnnxMLPModel, copies the normalizer
  and the MLP only, so the deploy contract is unchanged) that keeps its last hidden layer's output for priv_head."""

  def __init__(self, obs, obs_groups, obs_set, output_dim, hidden_dims=(512, 256, 128), activation: str = "elu",
               obs_normalization: bool = False, distribution_cfg: dict | None = None) -> None:
    super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims=hidden_dims, activation=activation,
                     obs_normalization=obs_normalization, distribution_cfg=distribution_cfg)
    self.priv_head = _priv_head(obs, int(list(hidden_dims)[-1]))
    self.priv_features: torch.Tensor | None = None

  def forward(self, obs, masks=None, hidden_state=None, stochastic_output: bool = False) -> torch.Tensor:
    # MLPModel.forward, with the MLP run layer by layer to keep the last hidden layer's output
    obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
    x = self.get_latent(obs, masks, hidden_state)
    layers = list(self.mlp)
    for layer in layers[:-1]:
      x = layer(x)
    self.priv_features = x
    out = layers[-1](x)
    if self.distribution is not None:
      if stochastic_output:
        self.distribution.update(out)
        return self.distribution.sample()
      return self.distribution.deterministic_output(out)
    return out


class UDepthStudentModel(DepthStudentModel):
  """The depth student (rl/distillation.py's DepthStudentModel, its export unchanged) with priv_head on its GRU state."""

  def __init__(self, obs, obs_groups, obs_set, output_dim, **kwargs) -> None:
    super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
    self.priv_head = _priv_head(obs, self.rnn_hidden_dim)

  @property
  def priv_features(self) -> torch.Tensor | None:
    return self.last_latent


class PrivEstimate:
  """The students' privileged-latent loss (module docstring). loss() after the student's forward pass on a batch:
  returns coef x MSE to add to the student's loss (coef > 0), or None after a probe step on the detached features
  (coef 0); pop_log() the means since the last call."""

  def __init__(self, coef: float = 0.0) -> None:
    self.coef = float(coef)
    self.opt: torch.optim.Optimizer | None = None
    self.sums = {"priv_mse": 0.0, "priv_fvu": 0.0}
    self.n = 0

  def loss(self, student, teacher, obs, masks=None) -> torch.Tensor | None:
    head = getattr(student, "priv_head", None)
    feat = getattr(student, "priv_features", None)
    if head is None or feat is None or not hasattr(teacher, "priv_encoder") or PRIV_GROUP not in obs.keys():
      return None
    with torch.no_grad():
      p = obs[PRIV_GROUP]
      if masks is not None:   # a recurrent mini-batch: padded trajectories, as the GRU's unpadded output
        p = unpad_trajectories(p, masks)
      target = teacher.priv_encoder(teacher.priv_normalizer(p))
    if self.coef > 0.0:
      mse = nn.functional.mse_loss(head(feat), target)
      out = self.coef * mse
    else:
      if self.opt is None:
        self.opt = torch.optim.Adam(head.parameters(), lr=PRIV_PROBE_LR)
      mse = nn.functional.mse_loss(head(feat.detach()), target)
      self.opt.zero_grad(set_to_none=True)
      mse.backward()
      self.opt.step()
      # the student's own optimizer holds these parameters too: leave it no gradient to step them with
      self.opt.zero_grad(set_to_none=True)
      out = None
    var = target.reshape(-1, target.shape[-1]).var(dim=0).mean().clamp_min(1e-8)
    self.sums["priv_mse"] += float(mse.detach())
    self.sums["priv_fvu"] += float(mse.detach() / var)
    self.n += 1
    return out

  def pop_log(self) -> dict[str, float]:
    if not self.n:
      return {}
    out = {k: v / self.n for k, v in self.sums.items()}
    self.sums = dict.fromkeys(self.sums, 0.0)
    self.n = 0
    return out


class UDistillation(DepthDistillation):
  """Stage 1 (rl/distillation.py's DepthDistillation: rsl_rl's DAgger update plus the depth student's optional
  terrain reconstruction) plus the privileged-latent estimate (PrivEstimate)."""

  def __init__(self, *args, priv_coef: float = 0.0, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.priv = PrivEstimate(priv_coef)

  def update(self) -> dict[str, float]:
    terrain = bool(self.aux_loss_coef) and getattr(self.student, "aux_head", None) is not None
    self.num_updates += 1
    behavior_sum = terrain_sum = 0.0
    loss = 0
    cnt = 0
    for _ in range(self.num_learning_epochs):
      self.student.reset(hidden_state=self.last_hidden_states[0])
      self.teacher.reset(hidden_state=self.last_hidden_states[1])
      self.student.detach_hidden_state()
      for batch in self.storage.generator():
        actions = self.student(batch.observations)
        behavior_loss = self.loss_fn(actions, batch.privileged_actions)
        loss = loss + behavior_loss
        behavior_sum += behavior_loss.item()
        if terrain:
          aux = nn.functional.mse_loss(self.student.aux_head(self.student.last_latent),
                                       batch.observations[TERRAIN_TARGET_GROUP])
          loss = loss + self.aux_loss_coef * aux
          terrain_sum += aux.item()
        extra = self.priv.loss(self.student, self.teacher, batch.observations)
        if extra is not None:
          loss = loss + extra
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
    out = {"behavior": behavior_sum / max(cnt, 1)}
    if terrain:
      out["terrain_aux"] = terrain_sum / max(cnt, 1)
    out.update(self.priv.pop_log())
    return out


@dataclass
class UDistillationAlgorithmCfg(DistillationAlgorithmCfg):
  priv_coef: float = 0.0
  """Weight of the privileged-latent estimate in the student's loss (0: a probe only; module docstring)."""
  class_name: str = "src.tasks.unified.students:UDistillation"


@dataclass
class UStudentPpoAlgorithmCfg(StudentPpoAlgorithmCfg):
  priv_coef: float = 0.0
  """As UDistillationAlgorithmCfg.priv_coef, in stage 2."""
  class_name: str = "src.tasks.unified.students:UStudentPPO"


def _teacher_model_cfg() -> RslRlModelCfg:
  return RslRlModelCfg(hidden_dims=(512, 256, 128), obs_normalization=True, class_name=TEACHER_MODEL,
                       distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"})


def _student_model_cfg(kind: str, distribution: bool) -> RslRlModelCfg:
  dist = {"class_name": "GaussianDistribution", "init_std": 0.4, "std_type": "scalar"} if distribution else None
  if kind == "blind":   # the deployed blind policies' actor: MLP on the proprio history
    # a Gaussian head in stage 1 too: rsl_rl's runner logs the policy's output_std every iteration (distillation
    # itself uses the mean)
    dist = {"class_name": "GaussianDistribution", "init_std": 0.4, "std_type": "scalar"}
    return RslRlModelCfg(hidden_dims=(512, 256, 128), obs_normalization=True, class_name=BLIND_STUDENT_MODEL,
                         distribution_cfg=dist)
  return RslRlModelCfg(hidden_dims=(512, 256, 128), obs_normalization=True, class_name=DEPTH_STUDENT_MODEL,
                       distribution_cfg=dist)


def distill_runner_cfg(robot: URobot, kind: str) -> DepthDistillationRunnerCfg:
  return DepthDistillationRunnerCfg(
    experiment_name=f"{robot.key}_u_student_{kind}", logger="tensorboard", num_steps_per_env=24,
    max_iterations=S1_ITERATIONS, save_interval=100,
    obs_groups={"student": STUDENT_GROUPS[kind], "teacher": TEACHER_GROUPS[kind]},
    student=_student_model_cfg(kind, distribution=False), teacher=_teacher_model_cfg(),
    algorithm=UDistillationAlgorithmCfg(),
  )


def ppo_runner_cfg(robot: URobot, kind: str, clip_actions: float | None = None) -> DepthStudentPpoRunnerCfg:
  return DepthStudentPpoRunnerCfg(
    experiment_name=f"{robot.key}_u_student_{kind}_rl", logger="tensorboard", num_steps_per_env=24,
    max_iterations=S2_ITERATIONS, save_interval=100, clip_actions=clip_actions,
    obs_groups={"actor": STUDENT_GROUPS[kind], "critic": ("critic",), TEACHER_SET: TEACHER_GROUPS[kind]},
    actor=_student_model_cfg(kind, distribution=True), teacher=_teacher_model_cfg(),
    algorithm=UStudentPpoAlgorithmCfg(bc_coef=1.0, bc_coef_final=0.3),
  )


# ---- runners ----
def restore_teacher_curriculum(env, teacher_checkpoint: str, device: str) -> None:
  """The teacher's final constraint scales (and progress, fixed at 1 anyway) into the student's environment."""
  if not teacher_checkpoint:
    return
  loaded = torch.load(teacher_checkpoint, map_location=device, weights_only=False)
  state = (loaded.get("infos") or {}).get("unified") or {}
  u = env.unwrapped
  cons = getattr(u, "u_constraints", None)
  if cons is not None and "constraints" in state:
    cons.load(state["constraints"])
    print(f"[unified] student: the teacher's {len(state['constraints'].get('scale', {}))} constraint scales")
    line_floor(cons)


# U_STUDENT_LINE_FLOOR=<x> (2026-11-13, go2-depth's design; unset: the teacher's final multipliers as they are): the
# students' straight-line flat items at least x times their configured weight. The teacher relaxes these items to its
# floor (0.5) once it meets them, and the students' constraints are frozen at the teacher's final values -- while the
# blind students drift 6-7% sideways at 0.3 m/s (the bar 5%) with no pressure left on it. Applied once after the
# teacher's scales are loaded; frozen afterwards like the others. Under U_FLAT_ONLY=1 the stairs see min(x, 1).
LINE_ITEMS = ("side_drift", "straight_drift", "back_drift", "head_tail", "head_med", "spin", "march")


def line_floor(cons) -> list[str]:
  v = os.environ.get("U_STUDENT_LINE_FLOOR")
  if not v:
    return []
  x = float(v)
  changed = []
  for name in LINE_ITEMS:
    if name in cons.scale:
      old = float(cons.scale[name])
      cons.scale[name] = max(old, x)
      changed.append(f"{name} {old:.2f} -> {cons.scale[name]:.2f}")
  print(f"[unified] student: U_STUDENT_LINE_FLOOR={x:g}: {', '.join(changed) or 'none of ' + ', '.join(LINE_ITEMS)}")
  return changed


def _is_blind(env) -> bool:
  return "depth" not in env.unwrapped.observation_manager.active_terms


def _export_blind(runner, path: str, filename: str) -> None:
  """The blind deploy contract's export (scripts/go2_blind/export_onnx.py): the actor MLP with its normalizer on
  the "actor" group, mjlab's base metadata."""
  from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata

  policy = runner.alg.student if hasattr(runner.alg, "student") else runner.alg.actor
  onnx_model = policy.as_onnx(verbose=False)
  onnx_model.to("cpu")
  onnx_model.eval()
  os.makedirs(path, exist_ok=True)
  out = os.path.join(path, filename)
  torch.onnx.export(onnx_model, onnx_model.get_dummy_inputs(), out, export_params=True, opset_version=18,
                    input_names=onnx_model.input_names, output_names=onnx_model.output_names, dynamic_axes={},
                    dynamo=False)
  attach_metadata_to_onnx(out, get_base_metadata(runner.env.unwrapped, "unified blind student"))


class UStudentDistillationRunner(DepthDistillationRunner):
  def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
    teacher_checkpoint = train_cfg.get("teacher_checkpoint", "")
    super().__init__(env, train_cfg, log_dir, device)
    restore_teacher_curriculum(self.env, teacher_checkpoint, device)
    diagnostics.attach(self, kind="student")   # the training records (<run>/records/, diagnostics.py)

  def export_policy_to_onnx(self, path: str, filename: str = "policy.onnx", verbose: bool = False) -> None:
    if _is_blind(self.env):
      _export_blind(self, path, filename)
    else:
      super().export_policy_to_onnx(path, filename, verbose)


class UStudentPpoRunner(DepthStudentPpoRunner):
  def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
    teacher_checkpoint = train_cfg.get("teacher_checkpoint", "")
    super().__init__(env, train_cfg, log_dir, device)
    restore_teacher_curriculum(self.env, teacher_checkpoint, device)
    diagnostics.attach(self, kind="student")   # the training records (<run>/records/, diagnostics.py)

  def export_policy_to_onnx(self, path: str, filename: str = "policy.onnx", verbose: bool = False) -> None:
    if _is_blind(self.env):
      _export_blind(self, path, filename)
    else:
      DepthDistillationRunner.export_policy_to_onnx(self, path, filename, verbose)


class UStudentPPO(StudentPPO):
  """rl/student_ppo.py's StudentPPO (PPO + behavior cloning toward the frozen teacher, critic warm-up) for either a
  recurrent actor (the depth student: StudentPPO's own update) or a feed-forward one (the blind student's MLP: the
  same losses on the feed-forward mini-batches; no command-jump weighting, which reads the depth student's history)."""

  def __init__(self, actor, critic, storage, bc_coef: float = 1.0, bc_coef_final: float | None = None,
               critic_warmup: int = 0, bc_jump_weight: float = 1.0, bc_jump_v: float = 0.5, bc_jump_wz: float = 1.0,
               priv_coef: float = 0.0, **kwargs) -> None:
    PPO.__init__(self, actor, critic, storage, **kwargs)
    self.priv = PrivEstimate(priv_coef)
    self.teacher = None
    self.bc_coef = bc_coef
    self.bc_coef_final = bc_coef_final
    self.critic_warmup = critic_warmup
    self.lr_ceiling = self.learning_rate
    self.num_updates = 0
    self.bc_jump_weight, self.bc_jump_v, self.bc_jump_wz = bc_jump_weight, bc_jump_v, bc_jump_wz
    self.jump_layout = None

  def extra_loss(self, batch) -> torch.Tensor | None:
    """StudentPPO's hook (the recurrent update): the privileged-latent estimate on the mini-batch."""
    return self.priv.loss(self.actor, self.teacher, batch.observations, masks=batch.masks)

  def update(self) -> dict[str, float]:
    if self.actor.is_recurrent:
      out = super().update()
      out.update(self.priv.pop_log())
      return out
    assert self.teacher is not None, "the runner sets the teacher"
    warmup = self.num_updates < self.critic_warmup
    self.num_updates += 1
    self.learning_rate = min(self.learning_rate, self.lr_ceiling)
    for g in self.optimizer.param_groups:
      g["lr"] = self.learning_rate
    sums = {"value": 0.0, "surrogate": 0.0, "entropy": 0.0, "bc": 0.0}
    for batch in self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs):
      if self.normalize_advantage_per_mini_batch:
        with torch.no_grad():
          batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)
      self.actor(batch.observations, stochastic_output=True)
      log_prob = self.actor.get_output_log_prob(batch.actions)
      values = self.critic(batch.observations)
      params = self.actor.output_distribution_params
      entropy = self.actor.output_entropy
      with torch.no_grad():
        teacher_mean = self.teacher(batch.observations)
      bc_loss = (self.actor.output_mean - teacher_mean).pow(2).mean()
      extra = self.priv.loss(self.actor, self.teacher, batch.observations)
      if self.desired_kl is not None and self.schedule == "adaptive" and not warmup:
        with torch.inference_mode():
          kl = torch.mean(self.actor.get_kl_divergence(batch.old_distribution_params, params))
          if kl > self.desired_kl * 2.0:
            self.learning_rate = max(1e-6, self.learning_rate / 1.5)
          elif 0.0 < kl < self.desired_kl / 2.0:
            self.learning_rate = min(self.lr_ceiling, self.learning_rate * 1.5)
          for g in self.optimizer.param_groups:
            g["lr"] = self.learning_rate
      ratio = torch.exp(log_prob - torch.squeeze(batch.old_actions_log_prob))
      adv = torch.squeeze(batch.advantages)
      surrogate = torch.max(-adv * ratio, -adv * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)).mean()
      if self.use_clipped_value_loss:
        vc = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
        value_loss = torch.max((values - batch.returns).pow(2), (vc - batch.returns).pow(2)).mean()
      else:
        value_loss = (batch.returns - values).pow(2).mean()
      if warmup:
        loss = self.value_loss_coef * value_loss
      else:
        loss = surrogate + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean() + self.bc_coef * bc_loss
        if extra is not None:
          loss = loss + extra
      self.optimizer.zero_grad()
      loss.backward()
      nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
      nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
      self.optimizer.step()
      sums["value"] += value_loss.item()
      sums["surrogate"] += surrogate.item()
      sums["entropy"] += entropy.mean().item()
      sums["bc"] += bc_loss.item()
    self.storage.clear()
    n = self.num_learning_epochs * self.num_mini_batches
    out = {k: v / n for k, v in sums.items()}
    out["bc_coef"] = self.bc_coef
    out["critic_warmup"] = float(warmup)
    out.update(self.priv.pop_log())
    return out
