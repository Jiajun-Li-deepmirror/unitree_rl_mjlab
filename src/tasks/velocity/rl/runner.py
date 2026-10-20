import os

import wandb

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner


class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

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
  """VelocityOnPolicyRunner with rsl_rl's left-right mirror loss on the actor.

  The actor's action for the mirrored observation is pulled towards the mirror
  of its action (mdp.symmetry.mirror_left_right), so behaviors learned on one
  side carry over to the other. Wheeled policies trained without it turned
  left and right at speed very differently (e.g. As2W 2026-10-19: 91% vs 57%).
  """

  MIRROR_LOSS_COEFF = 1.0

  def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu"):
    # A copy: rsl_rl stores the env in symmetry_cfg, and train.py dumps the
    # dict it passed in to YAML afterwards.
    symmetry_cfg = {
      "use_data_augmentation": False,
      "use_mirror_loss": True,
      "data_augmentation_func": "src.tasks.velocity.mdp.symmetry:mirror_left_right",
      "mirror_loss_coeff": self.MIRROR_LOSS_COEFF,
    }
    train_cfg = {**train_cfg, "algorithm": {**train_cfg["algorithm"], "symmetry_cfg": symmetry_cfg}}
    super().__init__(env, train_cfg, log_dir, device)
