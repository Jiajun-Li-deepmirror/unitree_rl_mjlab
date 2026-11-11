#!/usr/bin/env python3
"""export_student.py R KIND CHECKPOINT OUT (go2-depth 2026-11-12): a unified student's stage-2 checkpoint as a package
for tinynav-sim and the robot, in jobs/export_policy.sh's layout:
  policy.onnx      blind: the actor MLP on "actor" (the blind deploy contract, mjlab's base metadata; identical input
                   to the deployed blind policies); depth: the depth contract (rl/distillation.py's export)
  deploy.yaml      scripts/blind/export_deploy_yaml.py on the blind check task, for both kinds (robot_ctrl's config)
  blind_pilot.txt  the unified recipe's keys the environment was built with (tinynav-sim's import_blind_policy.py takes
                   the contact solver from "grip")
  manifest.json    kind, robot, checkpoint (path, md5, iteration), md5s, code, time
Run with UNIFIED_TASKS=1 from the repo root."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("UNIFIED_TASKS", "1")

import torch  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402

import src.tasks  # noqa: E402,F401
from src.tasks.unified.robots import DROPPED_KEYS, ROBOTS, unified_recipe  # noqa: E402


def main(argv: list[str]) -> int:
  robot, kind, ck, out = argv[0], argv[1], argv[2], Path(argv[3])
  task = f"Unitree-{robot}-U-Student-{kind.capitalize()}-Flat"
  out.mkdir(parents=True, exist_ok=True)
  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = 4
  agent = load_rl_cfg(task)
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  venv = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  runner = load_runner_cls(task)(venv, asdict(agent), device="cuda:0")
  runner.load(ck, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  runner.export_policy_to_onnx(str(out), "policy.onnx")
  env.close()
  r = robot.lower()
  # deploy.yaml from the blind check task for both kinds (the same robot, actions and gains; the depth student's
  # "actor" group is its teacher's, with the scan -- as the older depth packages took theirs from Blind-Polish)
  yaml_task = f"Unitree-{robot}-U-Student-Blind-Flat"
  rc = subprocess.run([sys.executable, str(ROOT / "scripts" / "blind" / "export_deploy_yaml.py"), yaml_task, "--robot", r,
                       "--out", str(out / "deploy.yaml")], cwd=ROOT, capture_output=True, text=True)
  if rc.returncode != 0 or not (out / "deploy.yaml").is_file():
    print(f"export_student: deploy.yaml failed: {rc.stderr[-400:]}", file=sys.stderr)
    return 1
  urobot = next(u for u in ROBOTS.values() if u.name == robot)
  keys = [k for k in unified_recipe(urobot)["P"] if k not in DROPPED_KEYS]
  (out / "blind_pilot.txt").write_text(",".join(keys) + "\n")
  md5 = lambda p: hashlib.md5(Path(p).read_bytes()).hexdigest()
  it = None
  try:
    it = int(torch.load(ck, map_location="cpu", weights_only=False).get("iter"))
  except Exception:
    pass
  commit = (ROOT / "REPO_COMMIT").read_text().strip() if (ROOT / "REPO_COMMIT").is_file() else ""
  m = {"kind": f"unified_{kind}", "robot": robot, "checkpoint": {"path": ck, "md5": md5(ck), "iteration": it},
       "policy_onnx_md5": md5(out / "policy.onnx"), "deploy_yaml_md5": md5(out / "deploy.yaml"), "code": commit,
       "model": "uros", "exported": time.strftime("%Y-%m-%d %H:%M:%S")}
  (out / "manifest.json").write_text(json.dumps(m, indent=1))
  print(f"exported unified {kind} {robot} {ck} -> {out} (policy.onnx {m['policy_onnx_md5'][:8]})")
  return 0


if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))
