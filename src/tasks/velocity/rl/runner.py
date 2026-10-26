import math
import os

import torch
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

  With "stdcap" in BLIND_PILOT the action std is capped after every update from
  iteration STD_CAP_ITS[0], at STD_CAP[0] there falling to STD_CAP[1] at
  STD_CAP_ITS[1] (legged v8 candidate, 2026-10-24 ~19:00). The v7 teachers' std rose
  17-19% over the cost ramps (1000-2000), and marching in place without
  randomisation their mean action drifted about twice as far as their sampled one
  (2000: Go2 12.0 / 7.1 cm, As2 23.9 / 12.5 cm per 20 s): the march hold only
  charges the sampled drift, so the policies offset drift their own noise causes,
  which the acceptance and the robot, acting on the mean, do not have.
  """

  MIRROR_LOSS_COEFF = 1.0
  STD_CAP = (0.35, 0.25)
  STD_CAP_ITS = (1000, 2500)

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
    if "stdcap" in os.environ.get("BLIND_PILOT", "").split(","):
      update = self.alg.update

      def update_capped() -> dict[str, float]:
        loss = update()
        # learn() sets current_learning_iteration after the update.
        self.cap_action_std(self.current_learning_iteration + 1)
        return loss

      self.alg.update = update_capped

  def action_std_cap(self, it: int) -> float | None:
    """The cap at iteration `it` (None before STD_CAP_ITS[0])."""
    (a, b), (start, end) = self.STD_CAP_ITS, self.STD_CAP
    if it < a:
      return None
    return start + (end - start) * min(1.0, (it - a) / (b - a))

  def cap_action_std(self, it: int) -> None:
    cap = self.action_std_cap(it)
    dist = self.alg.actor.distribution
    if cap is None or dist is None:
      return
    with torch.no_grad():
      if dist.std_type == "scalar":
        dist.std_param.clamp_(max=cap)
      else:
        dist.log_std_param.clamp_(max=math.log(cap))
