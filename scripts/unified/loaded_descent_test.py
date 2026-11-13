"""The cost constraints.loaded_descent on a stub environment (CPU) + the As2 teacher's build with the switch."""
import math
import os
import sys
from types import SimpleNamespace as NS

import torch

sys.path.insert(0, os.getcwd())
os.environ["U_ROBOTS"] = "Go2,Go2W,As2"
os.environ.pop("UNIFIED_TASKS", None)
from src.tasks.unified import constraints as uc   # noqa: E402


def env_of(pitch_deg, vx, cmd, kg, stairs, limit=30.0):
  n = len(pitch_deg)
  p = torch.deg2rad(torch.tensor(pitch_deg))
  g = torch.stack([torch.sin(p), torch.zeros(n), -torch.cos(p)], dim=1)
  quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1)   # no yaw: the world's x is the heading's
  vel = torch.stack([torch.tensor(vx), torch.zeros(n), torch.zeros(n)], dim=1)
  data = NS(projected_gravity_b=g, root_link_quat_w=quat, root_link_lin_vel_w=vel)
  command = torch.stack([torch.tensor(cmd), torch.zeros(n), torch.zeros(n)], dim=1)
  cm = NS(get_command=lambda name: command, get_term=lambda name: NS(is_stair_env=torch.tensor(stairs)))
  return NS(num_envs=n, device="cpu", scene={"robot": NS(data=data)}, command_manager=cm,
            payload_kg=torch.tensor(kg), u_payload_full=limit)


ok = True


def check(good, text):
  global ok
  ok &= bool(good)
  print(f"  {'ok  ' if good else 'FAIL'}  {text}")


#            pitch  vx    cmd   kg     stairs
rows = [(5.0, 0.6, 0.6, 30.0, True),     # at the free pitch, at the command: 0
        (12.5, 0.6, 0.6, 30.0, True),    # half way 5 -> 20 deg: 0.5
        (25.0, 0.6, 0.6, 30.0, True),    # above the full pitch: 1
        (12.5, 0.9, 0.6, 30.0, True),    # + 0.3 m/s over: 0.5 + (0.3 - 0.1) / 0.4 = 1.0
        (12.5, 1.5, 0.6, 30.0, True),    # + far over: 0.5 + 1
        (12.5, 0.9, 0.0, 30.0, True),    # the zero command: the pitch part alone, 0.5
        (12.5, 0.9, 0.6, 15.0, True),    # the common payload: 0
        (12.5, 0.9, 0.6, 17.5, True),    # half way to the heavy class: 0.5 x 1.0
        (12.5, 0.9, 0.6, 0.0, True),     # unloaded: 0
        (12.5, 0.9, 0.6, 30.0, False),   # not on a stairs column: 0
        (-20.0, 0.9, 0.6, 30.0, True),   # nose up (a climb): 0
        (1.0, 0.9, 0.6, 30.0, True)]     # level, fast: 0 (the tracking reward's matter)
want = [0.0, 0.5, 1.0, 1.0, 1.5, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0]
e = env_of(*[list(c) for c in zip(*rows)])
got = uc.loaded_descent(e).tolist()
for r, w, x in zip(rows, want, got):
  check(abs(w - x) < 1e-4, f"pitch {r[0]:5.1f} deg, vx {r[1]} at {r[2]} commanded, {r[3]:4.1f} kg of 30, stairs {r[4]}: {x:.3f} (want {w})")
got = uc.loaded_descent(e, over=0.0, pitch_free=math.radians(8), pitch_full=math.radians(18)).tolist()
check(abs(got[3] - 0.45) < 1e-4 and abs(got[4] - 0.45) < 1e-4, f"over 0, 8-18 deg: the pitch part alone ({got[3]:.3f}, {got[4]:.3f}; want 0.45)")
e.payload_kg = None
check(float(uc.loaded_descent(e).abs().sum()) == 0.0, "before the first payload draw: 0")

from src.tasks.unified.robots import ROBOTS   # noqa: E402
from src.tasks.unified.teacher_env import unified_teacher_env_cfg as teacher   # noqa: E402

for name in ("As2", "Go2"):
  base = teacher(ROBOTS[name])
  check("u_loaded_descent" not in base.rewards, f"{name}: no term without the switch")
  for value, weight, params in (("10", -10.0, {}), ("8,0.5", -8.0, {"over": 0.5}),
                                ("8,0,8,18", -8.0, {"over": 0.0, "pitch_free": math.radians(8), "pitch_full": math.radians(18)})):
    os.environ["U_LOADED_DESCENT"] = value
    cfg = teacher(ROBOTS[name])
    t = cfg.rewards.get("u_loaded_descent")
    check(t is not None and t.weight == weight and t.params == params and t.func is uc.loaded_descent,
          f"{name}: U_LOADED_DESCENT={value} -> weight {t.weight if t else None}, params {t.params if t else None}")
    same = [k for k in base.rewards if k not in cfg.rewards or cfg.rewards[k].weight != base.rewards[k].weight]
    check(not same and set(cfg.rewards) - set(base.rewards) == {"u_loaded_descent"}, f"{name}: every other term as without it")
    for st in ("blind",):
      scfg = teacher(ROBOTS[name], student=st)
      check("u_loaded_descent" in scfg.rewards, f"{name}: the {st} student's build carries it too")
  for bad in ("0", "25", "8,5", "8,1,20,10"):
    os.environ["U_LOADED_DESCENT"] = bad
    try:
      teacher(ROBOTS[name])
      check(False, f"{name}: U_LOADED_DESCENT={bad} must stop the build")
    except AssertionError:
      check(True, f"{name}: U_LOADED_DESCENT={bad} stops the build")
  os.environ.pop("U_LOADED_DESCENT")
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
