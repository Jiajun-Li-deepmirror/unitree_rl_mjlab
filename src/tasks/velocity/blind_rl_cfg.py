"""RL runner configuration for the blind velocity tasks (same PPO setup as Go2)."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def make_blind_ppo_runner_cfg(
  experiment_name: str, max_iterations: int, entropy_coef: float = 0.01
) -> RslRlOnPolicyRunnerCfg:
  """PPO runner for a blind velocity task, logging to tensorboard."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=entropy_coef,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name=experiment_name,
    logger="tensorboard",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=max_iterations,
  )


def make_depth_student_runner_cfg(experiment_name: str, onnx_contract: str):
  """DAgger distillation of a depth student from its depth teacher's actor, as the
  Go2's (config/go2/rl_cfg.py unitree_go2_depth_student_runner_cfg)."""
  from src.tasks.velocity.rl.distillation import DepthDistillationRunnerCfg

  return DepthDistillationRunnerCfg(
    experiment_name=experiment_name,
    logger="tensorboard",
    num_steps_per_env=24,
    max_iterations=8000,
    save_interval=100,
    obs_groups={
      "student": ("proprio", "proprio_history", "depth"),
      "teacher": ("actor",),
    },
    onnx_contract=onnx_contract,
  )


def make_depth_student_ppo_runner_cfg(experiment_name: str, onnx_contract: str, clip_actions: float | None = None):
  """PPO fine-tune of a distilled depth student with its teacher's critic and a pull toward the teacher's actions
  (rl/student_ppo.py; go2-depth 2026-11-02, B2)."""
  from src.tasks.velocity.rl.student_ppo import DepthStudentPpoRunnerCfg, StudentPpoAlgorithmCfg

  return DepthStudentPpoRunnerCfg(
    experiment_name=experiment_name,
    logger="tensorboard",
    num_steps_per_env=24,
    max_iterations=2000,
    save_interval=100,
    clip_actions=clip_actions,
    obs_groups={
      "actor": ("proprio", "proprio_history", "depth"),
      "critic": ("critic",),
      "teacher": ("actor",),
    },
    onnx_contract=onnx_contract,
    # Close to the teacher first, more room for PPO later.
    algorithm=StudentPpoAlgorithmCfg(bc_coef=1.0, bc_coef_final=0.3),
  )
