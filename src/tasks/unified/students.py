"""The unified pipeline's students (go2-depth 2026-11-12, doc/redesign.md 3.6): a blind and a depth student distilled
from one unified teacher v2, then fine-tuned with PPO plus behavior cloning toward it.

Environment: the teacher's own (teacher_env.unified_teacher_env_cfg with student=...): the same terrain, commands,
payload, randomization and constraint costs, the curriculum at its end (progress 1) and the constraint scales frozen
at the teacher's final values (restored from its checkpoint by the runners). The teacher's scan mode is fixed:
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
"""

from __future__ import annotations

import os
from copy import deepcopy

import torch
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.rl import RslRlModelCfg

from src.tasks.velocity.depth_env_cfg import NAVCORE_CAMERA_POS, NAVCORE_CAMERA_QUAT, TEACHER_SCAN_NAME, apply_depth_student
from src.tasks.velocity.mdp.depth import NAVCORE_DEPTH_CAMERA
from src.tasks.velocity.rl.distillation import (
  DepthDistillationRunner,
  DepthDistillationRunnerCfg,
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

from .models import PRIV_GROUP
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


def student_env_cfg(robot: URobot, kind: str, play: bool = False):
  assert kind in STUDENT_GROUPS, kind
  cfg = unified_teacher_env_cfg(robot, play=play, student=kind)
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


def _teacher_model_cfg() -> RslRlModelCfg:
  return RslRlModelCfg(hidden_dims=(512, 256, 128), obs_normalization=True, class_name=TEACHER_MODEL,
                       distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"})


def _student_model_cfg(kind: str, distribution: bool) -> RslRlModelCfg:
  dist = {"class_name": "GaussianDistribution", "init_std": 0.4, "std_type": "scalar"} if distribution else None
  if kind == "blind":   # the deployed blind policies' actor: MLP on the proprio history
    # a Gaussian head in stage 1 too: rsl_rl's runner logs the policy's output_std every iteration (distillation
    # itself uses the mean)
    dist = {"class_name": "GaussianDistribution", "init_std": 0.4, "std_type": "scalar"}
    return RslRlModelCfg(hidden_dims=(512, 256, 128), obs_normalization=True, class_name="MLPModel",
                         distribution_cfg=dist)
  return RslRlModelCfg(hidden_dims=(512, 256, 128), obs_normalization=True,
                       class_name="src.tasks.velocity.rl.distillation:DepthStudentModel", distribution_cfg=dist)


def distill_runner_cfg(robot: URobot, kind: str) -> DepthDistillationRunnerCfg:
  return DepthDistillationRunnerCfg(
    experiment_name=f"{robot.key}_u_student_{kind}", logger="tensorboard", num_steps_per_env=24,
    max_iterations=S1_ITERATIONS, save_interval=100,
    obs_groups={"student": STUDENT_GROUPS[kind], "teacher": TEACHER_GROUPS[kind]},
    student=_student_model_cfg(kind, distribution=False), teacher=_teacher_model_cfg(),
    algorithm=DistillationAlgorithmCfg(),
  )


def ppo_runner_cfg(robot: URobot, kind: str, clip_actions: float | None = None) -> DepthStudentPpoRunnerCfg:
  return DepthStudentPpoRunnerCfg(
    experiment_name=f"{robot.key}_u_student_{kind}_rl", logger="tensorboard", num_steps_per_env=24,
    max_iterations=S2_ITERATIONS, save_interval=100, clip_actions=clip_actions,
    obs_groups={"actor": STUDENT_GROUPS[kind], "critic": ("critic",), TEACHER_SET: TEACHER_GROUPS[kind]},
    actor=_student_model_cfg(kind, distribution=True), teacher=_teacher_model_cfg(),
    algorithm=StudentPpoAlgorithmCfg(bc_coef=1.0, bc_coef_final=0.3, class_name="src.tasks.unified.students:UStudentPPO"),
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
               **kwargs) -> None:
    PPO.__init__(self, actor, critic, storage, **kwargs)
    self.teacher = None
    self.bc_coef = bc_coef
    self.bc_coef_final = bc_coef_final
    self.critic_warmup = critic_warmup
    self.lr_ceiling = self.learning_rate
    self.num_updates = 0
    self.bc_jump_weight, self.bc_jump_v, self.bc_jump_wz = bc_jump_weight, bc_jump_v, bc_jump_wz
    self.jump_layout = None

  def update(self) -> dict[str, float]:
    if self.actor.is_recurrent:
      return super().update()
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
    return out
