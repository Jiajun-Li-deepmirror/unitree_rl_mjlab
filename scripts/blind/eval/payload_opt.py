"""--payload kg,x,y,z for the flat evaluations (go2-depth 2026-11-11, the user: the loaded runs of the same tests --
creep, march drift, drift, alternating turns at the payload limit -- measured and reported beside the unloaded ones,
not judged). A fixed point mass on the base at (x, y, z) m in the base frame, the payload acceptance's (as
long_stand.py / stop_then_move.py; jobs/accept_payload.sh gives the robot's). Without --payload a test runs exactly as
before (the play config's own base-mass randomization)."""

from __future__ import annotations


def add(parser) -> None:
  parser.add_argument("--payload", default=None,
                      help="kg,x,y,z: a fixed point mass on the base (the loaded run of this test; default: as before)")


def apply(cfg, spec: str | None) -> None:
  if not spec:
    return
  from mjlab.managers.event_manager import EventTermCfg
  from mjlab.managers.scene_entity_config import SceneEntityCfg

  import src.tasks.velocity.mdp as mdp

  m, x, y, z = (float(v) for v in spec.split(","))
  cfg.events["base_payload"] = EventTermCfg(
    mode="reset", func=mdp.base_payload_point,
    params={"asset_cfg": SceneEntityCfg("robot"), "light_range": (m, m), "heavy_max": 0.0, "heavy_share": 0.0,
            "pos_ranges": ((x, x), (y, y), (z, z)), "base_com_range": 0.0, "ramp_steps": 0},
  )
  cfg.events.pop("base_com", None)
