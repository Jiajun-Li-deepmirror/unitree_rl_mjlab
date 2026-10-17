"""RL configuration for Unitree Go2 velocity task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def unitree_go2_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree Go2 velocity task."""
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
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="go2_velocity",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10001,
  )


def unitree_go2_blind_ppo_runner_cfg(
  max_iterations: int, entropy_coef: float = 0.01
) -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for the Unitree Go2 blind velocity tasks.

  The map3 fine-tune halves `entropy_coef`: the end of the stage 2 run (from
  ~19100 iterations) diverged with the action std growing 0.50 -> 0.62.
  """
  cfg = unitree_go2_ppo_runner_cfg()
  cfg.experiment_name = "go2_blind"
  cfg.logger = "tensorboard"
  cfg.max_iterations = max_iterations
  cfg.algorithm.entropy_coef = entropy_coef
  return cfg
