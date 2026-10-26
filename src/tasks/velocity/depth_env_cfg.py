"""Depth-camera tasks on go2-blind's phased recipe (port of go2-depth's
config/legged/depth.py, 2026-10-27; user 2026-10-26 ~15:40: "尽量参考当前配方").

The teacher is the phased blind task (flat teacher, mixed, polish; see
apply_blind_phased and BLIND_PILOT) with a privileged height scan in the actor
and the critic, cast from above the walls (mdp.ElevatedGridPatternCfg). It is a
sensor of its own: terrain_scan and every reward that reads it stay the blind
recipe's (cast from the base, walls read as floor). The actor's proprio terms
keep their 10-step history per term, so its first dims match the student's
`proprio_history` layout (go2-depth's layout); the scan has no history.

The student is go2-depth's: proprioception (now and 10-step history) and two
10 Hz frames of the front depth camera, distilled from the teacher's `actor`
group (rl/distillation.py).
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import GridPatternCfg, RayCastSensorCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from src.tasks.velocity.mdp import depth as depth_mdp
from src.tasks.velocity.mdp.terrains import ElevatedGridPatternCfg

# Proprioceptive actor terms, in the order the deployed student receives them
# (tinynav-sim's StudentContract order; heading_error last).
PROPRIO_TERM_NAMES = (
  "base_ang_vel",
  "projected_gravity",
  "command",
  "phase",
  "joint_pos",
  "joint_vel",
  "actions",
  "heading_error",
)
ACTOR_HISTORY_LENGTH = 10

TEACHER_SCAN_NAME = "teacher_scan"
# go2-depth's teacher scan: rays start 8 m up so wall tops are seen (walls ~6 m on
# its hardest stairwells), 1.2 m wide (blind: 1.0) so both walls of a ~1 m flight
# are in view; walls clipped to -1 m, the floor scaled as the blind height_scan.
SCAN_Z_OFFSET = 8.0
SCAN_WIDTH = 1.2
SCAN_CLIP = (-1.0, 2.0)
SCAN_SCALE = 1.0 / 5.0

DEPTH_SENSOR_NAME = "front_depth"
TERRAIN_TARGET_GROUP = "terrain_target"
TERRAIN_TARGET_CLIP = (-0.5, 1.5)


def apply_depth_teacher(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  """A blind task (any terrain, any phase) plus the teacher's privileged scan."""
  assert cfg.scene.sensors is not None
  scan = next(s for s in cfg.scene.sensors if s.name == "terrain_scan")
  assert isinstance(scan, RayCastSensorCfg) and isinstance(scan.pattern, GridPatternCfg)
  teacher_scan = deepcopy(scan)
  teacher_scan.name = TEACHER_SCAN_NAME
  teacher_scan.pattern = ElevatedGridPatternCfg(
    size=(scan.pattern.size[0], SCAN_WIDTH),
    resolution=scan.pattern.resolution,
    direction=scan.pattern.direction,
    z_offset=SCAN_Z_OFFSET,
  )
  teacher_scan.max_distance = SCAN_Z_OFFSET + scan.max_distance
  teacher_scan.debug_vis = False
  cfg.scene.sensors = cfg.scene.sensors + (teacher_scan,)

  critic_scan = deepcopy(cfg.observations["critic"].terms["height_scan"])
  critic_scan.params = {**critic_scan.params, "sensor_name": TEACHER_SCAN_NAME}
  critic_scan.clip = SCAN_CLIP
  critic_scan.scale = SCAN_SCALE
  critic_scan.noise = None
  cfg.observations["critic"].terms[TEACHER_SCAN_NAME] = critic_scan

  actor = cfg.observations["actor"]
  history = actor.history_length or ACTOR_HISTORY_LENGTH
  actor.history_length = None
  # Copies: the blind config shares these term objects with the critic.
  actor.terms = {name: deepcopy(term) for name, term in actor.terms.items()}
  for name, term in actor.terms.items():
    assert name in PROPRIO_TERM_NAMES, f"actor term {name!r} is not a student proprio term"
    term.history_length = history
  actor_scan = deepcopy(critic_scan)
  actor_scan.noise = Unoise(n_min=-0.05, n_max=0.05)
  actor_scan.history_length = 0
  actor.terms[TEACHER_SCAN_NAME] = actor_scan
  return cfg


def apply_depth_student(
  cfg: ManagerBasedRlEnvCfg,
  play: bool = False,
  camera_pos: tuple[float, float, float] = depth_mdp.DEPTH_CAMERA_POS,
  camera_quat: tuple[float, float, float, float] = depth_mdp.DEPTH_CAMERA_QUAT,
  camera_body: str = "base_link",
) -> ManagerBasedRlEnvCfg:
  """A depth teacher task plus the front depth camera and the student groups.

  Groups (go2-depth's make_depth_student_env_cfg): `actor` (the teacher's input,
  unchanged), `critic` (unused), `proprio` (the actor's proprio terms now),
  `proprio_history` (x 10, term-major, oldest first) and `depth` (2 x 60 x 86);
  the last three are the deployed ONNX inputs of the same name. `terrain_target`
  is the training target of the student's optional terrain head.
  """
  robot = cfg.scene.entities["robot"]
  camera = depth_mdp.front_depth_camera_spec_cfg(pos=camera_pos, quat=camera_quat, body=camera_body)
  cfg.scene.entities["robot"] = replace(robot, cameras=robot.cameras + (camera,))
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (depth_mdp.front_depth_camera_cfg(DEPTH_SENSOR_NAME),)

  if not play:
    # Mounting tolerance; also covers tinynav's native GL depth reading ~0.5 px
    # (~0.5 deg pitch) lower than the ray-cast training render (go2-depth).
    cam = SceneEntityCfg("robot", camera_names=(depth_mdp.FRONT_DEPTH_CAMERA,))
    cfg.events["depth_camera_pose"] = EventTermCfg(
      mode="reset",
      func=dr.cam_quat,
      params={
        "roll_range": (-math.radians(1.0), math.radians(1.0)),
        "pitch_range": (-math.radians(2.0), math.radians(2.0)),
        "yaw_range": (-math.radians(1.0), math.radians(1.0)),
        "asset_cfg": cam,
      },
    )
    cfg.events["depth_camera_pos"] = EventTermCfg(
      mode="reset",
      func=dr.cam_pos,
      params={"ranges": (-0.01, 0.01), "operation": "add", "asset_cfg": deepcopy(cam)},
    )

  actor = cfg.observations["actor"]
  proprio_terms = {}
  for name in PROPRIO_TERM_NAMES:
    if name not in actor.terms:
      continue
    term = deepcopy(actor.terms[name])
    term.history_length = 0
    proprio_terms[name] = term
  cfg.observations["proprio"] = ObservationGroupCfg(
    terms=proprio_terms, concatenate_terms=True, enable_corruption=not play
  )
  cfg.observations["proprio_history"] = ObservationGroupCfg(
    terms=deepcopy(proprio_terms),
    concatenate_terms=True,
    enable_corruption=not play,
    history_length=ACTOR_HISTORY_LENGTH,
  )
  depth_params: dict = {"sensor_name": DEPTH_SENSOR_NAME}
  if not play:
    depth_params |= {"stale_prob": 0.2, "noise_std": 0.01, "dropout_prob": 0.01}
  cfg.observations["depth"] = ObservationGroupCfg(
    terms={"depth": ObservationTermCfg(func=depth_mdp.depth_history, params=depth_params)},
    concatenate_terms=True,
    enable_corruption=False,
  )
  # Not a policy input: the scan without noise, in meters, walls clipped closer.
  target = deepcopy(cfg.observations["critic"].terms[TEACHER_SCAN_NAME])
  target.noise = None
  target.history_length = 0
  target.clip = TERRAIN_TARGET_CLIP
  target.scale = None
  cfg.observations[TERRAIN_TARGET_GROUP] = ObservationGroupCfg(
    terms={"height_scan": target}, concatenate_terms=True, enable_corruption=False
  )
  return cfg

