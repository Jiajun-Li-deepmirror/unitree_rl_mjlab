"""fall_diag.py SCRIPT ARGS... (depth_fe, 2026-11-02): run a flat check script (drift_all.py, eval_standing.py)
unchanged, and when it closes its env print which robots fell and their randomized parameters next to their group's
spread: base payload (kg), base CoM offset (cm), foot friction, PD gain scales (kp, kd: mean / min over the joints).
Through seed_run.py / student_eval.py like the checks: student_eval.py seed_run.py SEED fall_diag.py SCRIPT ARGS.
The 11-01 students fell 1-2 of 256 robots in the same items of the same env seed in every checkpoint and round."""
import inspect
import os
import runpy
import sys

import torch

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402

_close = ManagerBasedRlEnv.close


def _rows(env, name, idx):
  from src.tasks.velocity.mdp.observations import _model_rows

  return _model_rows(env, name, idx)


def _params(env):
  out = {}
  robot = env.scene["robot"]
  root = robot.indexing.root_body_id
  try:
    out["payload_kg"] = (_rows(env, "body_mass", root) - env.sim.get_default_field("body_mass")[root]).reshape(-1)
  except Exception as e:  # noqa: BLE001
    print(f"[fall_diag] payload: {e!r}")
  try:
    com = (_rows(env, "body_ipos", root) - env.sim.get_default_field("body_ipos")[root]) * 100
    com = com.reshape(env.num_envs, 3)
    for i, a in enumerate("xyz"):
      out[f"com_{a}_cm"] = com[:, i]
  except Exception as e:  # noqa: BLE001
    print(f"[fall_diag] com: {e!r}")
  try:
    fc = env.cfg.events["foot_friction"].params["asset_cfg"]
    try:
      fc.resolve(env.scene)
    except Exception:  # noqa: BLE001
      pass
    geoms = robot.indexing.geom_ids[fc.geom_ids].long()
    out["friction"] = _rows(env, "geom_friction", geoms)[..., 0].mean(dim=1).reshape(-1)
  except Exception as e:  # noqa: BLE001
    print(f"[fall_diag] friction: {e!r}")
  for field, label in (("actuator_gainprm", "kp"), ("actuator_biasprm", "kd")):
    try:
      ids = robot.indexing.ctrl_ids.long()
      col = 0 if label == "kp" else 2
      cur = _rows(env, field, ids)[..., col]
      dflt = env.sim.get_default_field(field)[ids][..., col]
      ratio = cur / torch.where(dflt.abs() > 1e-9, dflt, torch.ones_like(dflt))
      ratio = ratio.reshape(env.num_envs, -1)
      out[f"{label}_mean"] = ratio.mean(dim=1)
      out[f"{label}_min"] = ratio.min(dim=1).values
    except Exception as e:  # noqa: BLE001
      print(f"[fall_diag] {label}: {e!r}")
  return out


def close(self):
  g = inspect.currentframe().f_back.f_globals
  try:
    fell, n = g["fell"], g.get("n")
    names = [str(c) for c in g["CMDS"]] if "CMDS" in g else [name for name, _ in g["GROUPS"]]
    n = n or fell.numel() // len(names)
    par = _params(self)
    q = torch.tensor((0.0, 0.1, 0.5, 0.9, 1.0), device=fell.device)
    print(f"\n[fall_diag] {int(fell.sum())} robots fell of {fell.numel()} ({len(names)} groups x {n})")
    for k, v in par.items():
      print(f"[fall_diag] all robots {k:>12}: min p10 p50 p90 max " + " ".join(f"{x:7.3f}" for x in v.float().quantile(q).tolist()))
    for e in fell.nonzero().flatten().tolist():
      gi = e // n
      cells = []
      for k, v in par.items():
        grp = v[gi * n : (gi + 1) * n].float()
        rank = float((grp < v[e]).float().mean()) * 100
        cells.append(f"{k} {float(v[e]):.3f} (p{rank:.0f})")
      print(f"[fall_diag] FELL env {e} group {names[gi]} #{e - gi * n}: " + ", ".join(cells))
  except Exception as e:  # noqa: BLE001
    print(f"[fall_diag] dump failed: {e!r}")
  return _close(self)


if __name__ == "__main__":
  script = sys.argv[1]
  sys.argv = sys.argv[1:]
  sys.path[:1] = [os.path.dirname(os.path.abspath(script)), os.getcwd()]
  ManagerBasedRlEnv.close = close
  runpy.run_path(script, run_name="__main__")
