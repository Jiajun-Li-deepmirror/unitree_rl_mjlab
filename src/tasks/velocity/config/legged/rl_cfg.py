"""RL configuration shared by the robots' depth tasks."""


def depth_student_runner_cfg(robot: str, onnx_contract: str):
  """DAgger distillation of a depth student from its teacher."""
  from src.tasks.velocity.rl.distillation import DepthDistillationRunnerCfg

  return DepthDistillationRunnerCfg(
    experiment_name=f"{robot}_depth_student",
    logger="tensorboard",
    num_steps_per_env=24,
    max_iterations=5000,
    save_interval=100,
    obs_groups={
      "student": ("proprio", "proprio_history", "depth"),
      "teacher": ("actor",),
    },
    onnx_contract=onnx_contract,
  )
