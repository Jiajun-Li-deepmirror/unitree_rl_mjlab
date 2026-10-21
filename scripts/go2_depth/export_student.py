"""Export a depth-student checkpoint to tinynav-sim's ONNX, contract metadata
included, as the student runner does at every save for its latest checkpoint
(the map3 pick is often an earlier one).

  PYTHONPATH=. python scripts/go2_depth/export_student.py \\
    --task Unitree-Go2-Depth-Student-Heading \\
    --checkpoint logs/rsl_rl/go2_depth_student/<run>/model_6500.pt --out go2_depth_v2_heading.onnx
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--task", required=True, help="the student's task, e.g. Unitree-Go2-Depth-Student-Heading")
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--out", required=True, help="ONNX file to write")
  args = parser.parse_args()

  env_cfg = load_env_cfg(args.task, play=True)
  env_cfg.scene.num_envs = 1
  agent_cfg = load_rl_cfg(args.task)
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0"), clip_actions=agent_cfg.clip_actions)
  train_cfg = asdict(agent_cfg)
  # The teacher is not part of the export.
  train_cfg["teacher_checkpoint"] = ""
  train_cfg["student_checkpoint"] = args.checkpoint
  runner = load_runner_cls(args.task)(env, train_cfg, None, "cuda:0")
  out = Path(args.out).resolve()
  runner.export_policy_to_onnx(str(out.parent), out.name)
  print(f"wrote {out} (contract {runner.onnx_contract})")
  env.close()


if __name__ == "__main__":
  main()
