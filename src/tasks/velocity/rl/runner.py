import os
from copy import deepcopy

import wandb

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner

from src.tasks.velocity.rl.obs_guard import guard_nan_observations
from src.tasks.velocity.rl.symmetry import mirror_left_right


class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def __init__(self, env, *args, **kwargs) -> None:
    super().__init__(env, *args, **kwargs)
    guard_nan_observations(self.env)

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_path = path.split("model")[0]
    filename = "policy.onnx"
    self.export_policy_to_onnx(policy_path, filename)
    run_name: str = (
      wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
    )  # type: ignore[assignment]
    onnx_path = os.path.join(policy_path, filename)
    metadata = get_base_metadata(self.env.unwrapped, run_name)
    attach_metadata_to_onnx(onnx_path, metadata)
    if self.logger.logger_type in ["wandb"]:
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class MirrorVelocityOnPolicyRunner(VelocityOnPolicyRunner):
  """With go2-blind's left-right mirror loss (actor only, coefficient 1.0).

  Its wheeled v5 robots trained with it; the loss settled near 0.014 and a
  right-turn deficit of 10-13% remained on its Go2W, so it does not
  guarantee symmetry.
  """

  def __init__(self, env, train_cfg: dict, *args, **kwargs) -> None:
    # A copy: rsl_rl adds the env to symmetry_cfg, and train.py writes the
    # caller's dict to params/agent.yaml afterwards.
    train_cfg = deepcopy(train_cfg)
    train_cfg["algorithm"]["symmetry_cfg"] = {
      "use_data_augmentation": False,
      "use_mirror_loss": True,
      "data_augmentation_func": mirror_left_right,
      "mirror_loss_coeff": 1.0,
    }
    super().__init__(env, train_cfg, *args, **kwargs)

