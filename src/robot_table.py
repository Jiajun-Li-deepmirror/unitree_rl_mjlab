#!/usr/bin/env python3
"""The robots' physical parameters: the one table (go2-depth 2026-11-11, the user's unification item 2: "机型参数只留一张表").

Everything that is a fact about a robot, not about a training: kind, the payload the training randomizes (limit and the
point's range) and the one the acceptance tests at, the end-wall stop distance, the foot radius, wheel radius / scale,
and the robot parameters the blind environment reads (BLIND_PARAMS). Read by
  jobs/fe_robot.sh           BLIND_PAYLOAD / BLIND_WALL_STOP / BLIND_FOOT_R          (--fe-sh)
  jobs/accept_payload.sh     the acceptance payload kg,x,y,z                          (--accept-payload)
  jobs/recipes/unified.py    the robot parameters (BLIND_PARAMS) and the documented values
  src/tasks/unified/robots.py the unified teacher's robots
(Which delivered package a robot's next round starts from is not here: that is lineage, jobs/v2_robot.sh.)
Plain Python, no imports from the project: the boxes' shell scripts run it with any python3.

  python3 src/robot_table.py Go2W --fe-sh          export lines (BLIND_PAYLOAD keeps ${PAYLOAD_RAMP:+,ramp=...} for eval)
  python3 src/robot_table.py Go2W --accept-payload 12,0.15,0,0.20
  python3 src/robot_table.py Go2W --params         BLIND_PARAMS (k=v,...)
  python3 src/robot_table.py --list                every robot, one line each
"""

from __future__ import annotations

import sys

# name: kind, training payload (limit, point range), acceptance payload (kg, x, y, z), wall stop (m), foot radius (m),
# wheel radius (m) / action scale, robot parameters (BLIND_PARAMS), notes. The values are those the chains used up to
# 2026-11-11 (jobs/fe_robot.sh, jobs/accept_payload.sh, jobs/recipes/unified.py), unchanged.
ROBOTS: dict[str, dict] = {
  "Go2": dict(kind="legged", train_payload="max=12,x=0.15,y=0.08,z=0.20", accept_payload="12,0.15,0,0.20",
              wall_stop=0.85, foot_r=0.022,
              notes="payload: Unitree's figure (max ~12 kg by config); foot sphere 2.2 cm"),
  "Go2W": dict(kind="wheeled", train_payload="max=12,x=0.15,y=0.08,z=0.20", accept_payload="12,0.15,0,0.20",
               wall_stop=0.85, foot_r=0.022, wheel_r=0.086, wheel_scale=10,
               notes="payload: Unitree's figure; leg kp 20/20/40 (model)"),
  "As2": dict(kind="legged", train_payload="max=10,x=0.15,y=0.08,z=0.20", accept_payload="10,0.15,0,0.20",
              wall_stop=0.9, foot_r=0.028,
              notes="payload: no official figure, about half the body weight (the user, 2026-11-11)"),
  "As2W": dict(kind="wheeled", train_payload="max=13,x=0.15,y=0.08,z=0.20", accept_payload="13,0.15,0,0.20",
               wall_stop=0.9, foot_r=None, wheel_r=0.086, wheel_scale=8,
               notes="payload: no official figure, about half the body weight (the user, 2026-11-11); leg kp 40/40/60"),
  "B2": dict(kind="legged", train_payload="max=40,x=0.30,y=0.15,z=0.30", accept_payload="40,0.15,0,0.30",
             wall_stop=1.05, foot_r=0.032,
             notes="payload: Unitree's figure; trained to x 0.30, accepted at x 0.15 (the corner 0.30 is reported only)"),
  "B2W": dict(kind="wheeled", train_payload="max=40,x=0.30,y=0.15,z=0.30", accept_payload="40,0.15,0,0.30",
              wall_stop=1.05, foot_r=0.032, wheel_r=0.113, wheel_scale=6,
              params={"leg_kp_scale": 1.5, "leg_kd_scale": 1.2, "leg_hold_s": 2.0, "stairs_final": 0.35},
              notes="leg kp 240 / kd 6 (82 kg on kp 160 crept after stops); legs hold 2 s after a stop; stairs share "
                    "capped at 0.35 (its flat items slid past it)"),
}


def get(name: str) -> dict:
  if name not in ROBOTS:
    raise KeyError(f"robot_table: no robot {name!r} (have {', '.join(ROBOTS)})")
  return ROBOTS[name]


def params_str(name: str) -> str:
  return ",".join(f"{k}={v}" for k, v in get(name).get("params", {}).items())


def fe_sh(name: str) -> str:
  """The export line jobs/fe_robot.sh evaluates (the PAYLOAD_RAMP part expands in the caller's shell)."""
  r = get(name)
  out = f"export BLIND_PAYLOAD={r['train_payload']}${{PAYLOAD_RAMP:+,ramp=$PAYLOAD_RAMP}} BLIND_WALL_STOP={r['wall_stop']}"
  if r.get("foot_r") is not None:
    out += f" BLIND_FOOT_R={r['foot_r']}"
  return out


def main(argv: list[str]) -> int:
  if not argv or argv[0] in ("-h", "--help"):
    print(__doc__)
    return 0
  if argv[0] == "--list":
    for n, r in ROBOTS.items():
      print(f"{n:5s} {r['kind']:8s} train {r['train_payload']}  accept {r['accept_payload']}  wall {r['wall_stop']}  "
            f"foot {r.get('foot_r')}  params {params_str(n) or '-'}")
    return 0
  name, what = argv[0], (argv[1] if len(argv) > 1 else "--fe-sh")
  try:
    r = get(name)
  except KeyError as e:
    print(e, file=sys.stderr)
    return 2
  if what == "--fe-sh":
    print(fe_sh(name))
  elif what == "--accept-payload":
    print(r["accept_payload"])
  elif what == "--params":
    print(params_str(name))
  elif what == "--kind":
    print(r["kind"])
  else:
    print(f"robot_table: unknown option {what}", file=sys.stderr)
    return 2
  return 0


if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))
