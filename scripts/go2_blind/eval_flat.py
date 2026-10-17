"""Flat-ground command tracking of a blind Go2 policy (joystick feel).

Each command in CMDS runs on --envs-per-cmd robots on flat ground (play config: no
pushes, no observation noise) for --duration seconds. Over the last --window seconds
it reports per command:
  ratio : mean body-frame velocity / command, on the commanded axes,
  error : mean absolute tracking error per axis (vx, vy in m/s, wz in rad/s),
          uncommanded axes included,
  falls, zero-command drift (cm over the window), and foot contact changes per
  second per foot (in-place turns must step, not twist).

  python scripts/go2_blind/eval_flat.py logs/rsl_rl/go2_blind/<run>/model_N.pt [more.pt]
"""

import argparse
from dataclasses import asdict

import torch

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK = "Unitree-Go2-Blind-Flat"
CMDS = [
  (0, 0, 0), (0.3, 0, 0), (0.5, 0, 0), (1.0, 0, 0), (1.5, 0, 0), (-0.5, 0, 0), (-1.0, 0, 0),
  (0, 0.3, 0), (0, 0.6, 0), (0, -0.6, 0), (0, 0, 0.5), (0, 0, 1.5), (0, 0, -1.5),
  (0.8, 0, 0.8), (0.6, 0.4, 0), (0.15, 0, 0.3),
]  # fmt: skip


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("checkpoints", nargs="+")
  parser.add_argument("--envs-per-cmd", type=int, default=32)
  parser.add_argument("--duration", type=float, default=8.0)
  parser.add_argument("--window", type=float, default=4.0)
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args()

  n = args.envs_per_cmd
  env_cfg = load_env_cfg(TASK, play=True)
  env_cfg.scene.num_envs = len(CMDS) * n
  env_cfg.seed = args.seed
  agent_cfg = load_rl_cfg(TASK)
  env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
  venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = load_runner_cls(TASK)(venv, asdict(agent_cfg), device="cuda:0")
  dev, dt = env.device, env.step_dt
  robot = env.scene["robot"]
  feet = env.scene["feet_ground_contact"]
  cmd = env.command_manager.get_term("twist")
  C = torch.tensor(CMDS, device=dev, dtype=torch.float32).repeat_interleave(n, 0)
  steps, start = int(round(args.duration / dt)), int(round((args.duration - args.window) / dt))

  for ckpt in args.checkpoints:
    runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
    policy = runner.get_inference_policy(device="cuda:0")
    env.reset()
    obs = venv.get_observations()
    fell = torch.zeros(len(C), dtype=torch.bool, device=dev)
    vel_sum = torch.zeros(len(C), 3, device=dev)
    err_sum = torch.zeros(len(C), 3, device=dev)
    flips = torch.zeros(len(C), device=dev)
    prev_contact, pos0 = None, None
    for s in range(steps):
      cmd.vel_command_b[:] = C
      cmd.time_left[:] = 1e9
      with torch.no_grad():
        obs, _, dones, extras = venv.step(policy(obs))
      fell |= dones.bool() & ~extras.get("time_outs", torch.zeros_like(dones)).bool()
      if s < start:
        continue
      v = torch.cat([robot.data.root_link_lin_vel_b[:, :2], robot.data.root_link_ang_vel_b[:, 2:3]], 1)
      vel_sum += v
      err_sum += (v - C).abs()
      contact = feet.data.found > 0
      if prev_contact is None:
        pos0 = robot.data.root_link_pos_w[:, :2].clone()
      else:
        flips += (contact != prev_contact).float().mean(-1)
      prev_contact = contact
    k = steps - start
    drift = (robot.data.root_link_pos_w[:, :2] - pos0).norm(dim=-1)
    ok = ~fell

    def per_cmd(x: torch.Tensor) -> torch.Tensor:
      """Mean over the non-fallen robots of each command."""
      x = x.view(len(CMDS), n, *x.shape[1:])
      m = ok.view(len(CMDS), n).float().view(len(CMDS), n, *([1] * (x.dim() - 2)))
      return (x * m).sum(1) / m.sum(1).clamp(min=1)

    vel, err = per_cmd(vel_sum / k), per_cmd(err_sum / k)
    drift_c, flips_c = per_cmd(drift), per_cmd(flips / (k - 1) / dt)
    falls = fell.view(len(CMDS), n).sum(1)

    print(f"\n{ckpt}: {n} robots per command, mean over the last {args.window:.0f} s of {args.duration:.0f} s")
    print(f"{'command (vx, vy, wz)':>22} | {'actual / command':>16} | {'|err| vx':>8} {'vy':>5} {'wz':>5} | falls | other")
    for i, c in enumerate(CMDS):
      ratio = " ".join(f"{a}{float(vel[i, j]) / c[j]:.0%}" for j, a in enumerate("xyw") if c[j] != 0) or "-"
      other = ""
      if c == (0, 0, 0):
        other = f"drift {100 * float(drift_c[i]):.1f} cm, contact changes {float(flips_c[i]):.1f}/s"
      elif c[0] == 0 and c[1] == 0:
        other = f"contact changes {float(flips_c[i]):.1f}/s/foot"
      e = err[i]
      print(f"{str(c):>22} | {ratio:>16} | {float(e[0]):8.3f} {float(e[1]):5.3f} {float(e[2]):5.3f} | {int(falls[i]):2d}/{n} | {other}")
    moving = [i for i, c in enumerate(CMDS) if c != (0, 0, 0)]
    e = err[moving].mean(0)
    print(f"{'mean over moving cmds':>22} | {'':>16} | {float(e[0]):8.3f} {float(e[1]):5.3f} {float(e[2]):5.3f} | "
          f"{int(falls.sum()):2d}/{n * len(CMDS)}")
  env.close()


if __name__ == "__main__":
  main()
