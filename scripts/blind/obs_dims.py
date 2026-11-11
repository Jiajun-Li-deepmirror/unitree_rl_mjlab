"""obs_dims.py TASK... -- each task's policy (actor) and critic input widths under the current environment
(BLIND_PILOT, BLIND_PAYLOAD, ...), one line per task: `TASK actor N critic M`.

jobs/blind_chain.sh calls it at the chain start (2026-11-11, the user: "链开始时可以先算出每个阶段配方对应的观测维度，提前检查
前后阶段能否对上"), so that a stage whose inputs cannot follow the stage before is found before any training, not when
P loads M (Go2W / B2 rp4: P's spinhold appended 2 critic inputs). Builds each task's environment with one robot.
"""

import sys

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

for task in sys.argv[1:]:
  cfg = load_env_cfg(task)
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  dims = env.observation_manager.group_obs_dim
  print(f"{task} actor {int(dims['actor'][0])} critic {int(dims['critic'][0])}", flush=True)
  env.close()
