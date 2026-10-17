"""Export a blind Go2 checkpoint to ONNX for the C++ deploy and check it against torch.

Same export as VelocityOnPolicyRunner.save (actor with its observation normalizer,
input `obs` [1, 470], plus metadata), then 20 steps x 64 robots comparing the ONNX
reference evaluator with the torch policy.

  python scripts/go2_blind/export_onnx.py logs/rsl_rl/go2_blind/<run>/model_N.pt \
    --out deploy/robots/go2/config/policy/velocity/blind_v0/exported
"""

import argparse
import os
from dataclasses import asdict

import numpy as np
import onnx
import torch
from onnx.reference import ReferenceEvaluator

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK = "Unitree-Go2-Blind"


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("checkpoint")
  parser.add_argument("--out", required=True, help="Output directory; the file is policy.onnx.")
  args = parser.parse_args()

  env_cfg = load_env_cfg(TASK, play=True)
  env_cfg.scene.num_envs = 64
  agent_cfg = load_rl_cfg(TASK)
  env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
  venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = load_runner_cls(TASK)(venv, asdict(agent_cfg), device="cuda:0")
  runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  os.makedirs(args.out, exist_ok=True)
  runner.export_policy_to_onnx(args.out, "policy.onnx")
  path = os.path.join(args.out, "policy.onnx")
  attach_metadata_to_onnx(path, get_base_metadata(env, f"local:{os.path.basename(args.checkpoint)}"))

  policy = runner.get_inference_policy(device="cuda:0")
  env.reset()
  obs = venv.get_observations()
  model = onnx.load(path)
  sess = ReferenceEvaluator(model)
  name = model.graph.input[0].name
  shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
  err = 0.0
  for _ in range(20):
    with torch.no_grad():
      a = policy(obs)
    x = obs["actor"].cpu().numpy().astype(np.float32)
    y = np.concatenate([sess.run(None, {name: x[i : i + 1]})[0] for i in range(x.shape[0])])
    err = max(err, float(np.abs(y - a.cpu().numpy()).max()))
    obs, _, _, _ = venv.step(a)
  print(f"exported {path} from {args.checkpoint}: input '{name}' {shape}; max |onnx - torch| over 20 steps {err:.1e}")
  env.close()


if __name__ == "__main__":
  main()
