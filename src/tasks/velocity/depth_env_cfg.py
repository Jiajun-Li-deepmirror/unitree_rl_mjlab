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
import os
from copy import deepcopy
from dataclasses import replace

import torch

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

# The navcore camera unit on the head (CAD of 2026-10-23, base frame, m): the front
# of its left lens, the depth's (its bag: depth = the rectified left camera of the
# 100 mm stereo pair, RGB in the middle); the lenses sit at y +0.05 (left) / 0 /
# -0.05 and look level along +x (MuJoCo camera: -z forward, +y up). The camera is
# mdp.NAVCORE_DEPTH_CAMERA. B2: its assembly b2_asm_xianggang_asm.stp (Creo, 2026-10-27;
# the unit sits where the LiDAR was, on NAVCORE_BASE_B2), lens barrels 403-411 mm
# ahead, base frame from the hip / thigh joint axes (front-rear 655.5 mm, sim 657).
NAVCORE_CAMERA_POS = {
  "go2": (0.307, 0.050, 0.1273),
  "go2w": (0.307, 0.050, 0.1273),  # the Go2's trunk (go2w.xml: same hips, trunk geoms, IMU)
  "as2": (0.320, 0.050, 0.178),
  "b2": (0.411, 0.050, 0.158),
  "b2w": (0.411, 0.050, 0.158),   # the same trunk and mount (user 2026-10-27)
  "as2w": (0.320, 0.050, 0.178),  # As2's mount (user 2026-10-29)
}
NAVCORE_CAMERA_QUAT = (0.5, 0.5, -0.5, -0.5)

# A real stereo camera for the students that go on the robot (user 2026-10-27:
# depth goes to the real robot; see mdp.depth_history), from navcore's bag of
# 2026-10-26 (static office scene): depth 158 ms after capture (p5 138, p95 190),
# taken by the policy at its next control step once it arrives (the deploy should
# do that; sampling at a fixed 10 Hz tick would add up to 100 ms); every pixel
# valid (the learned stereo fills holes); frame-to-frame error 1-5% of the depth
# moving whole surfaces; valid at 0.18 m. Not yet measured, so guessed: drops and
# freezes, calibration scale, wrong patches, edges, mounting (+-1.5 deg roll / yaw,
# +-3 deg pitch, +-1 / +-1.5 cm x / z, +-1 cm y) and the field of view (+-3%).
REAL_CAMERA_DEPTH = {
  "latency_range": (7.0, 9.5),
  "latency_jitter": 1.0,
  "drop_prob": 0.02,
  "freeze_prob": 0.003,
  "freeze_captures": (3, 10),
  "scale_range": (0.98, 1.02),
  "frame_scale_std": 0.02,
  "field_std": 0.02,
  "z_noise": 0.003,
  "min_range_m": (0.05, 0.15),
  "patch_prob": 0.2,
  "patch_size": ((10, 40), (10, 40)),
  "patch_scale": (0.7, 1.3),
  "hole_prob": 0.05,
  "hole_size": ((3, 12), (3, 16)),
  "edge_prob": 0.3,
  "edge_jump_m": 0.25,
  "crop_jitter": 1,
}
# Robustness fine-tune (user 2026-10-29, "做1,3,2" step 2a): something in front of the
# lens or a covered lens, on top of REAL_CAMERA_DEPTH (mdp.depth_history occlusions).
# One event per ~10 s of captures, 0.2-2 s long: about a tenth of the time occluded.
REAL_CAMERA_OCCLUSION = {
  "occlusion_prob": 0.01,
  "occlusion_captures": (2, 20),
  "occlusion_kinds": (0.5, 0.25, 0.25),
  "occlusion_depth_m": (0.15, 0.4),
  "occlusion_patch": ((0.3, 1.0), (0.3, 1.0)),
}
# Stair-edge fine-tune (user 2026-10-30): the real camera smooths depth edges (risers
# become ~14 cm ramps), fitted on the B2's real stair frames: rows sigma ~1.75 student
# px, columns about half; drawn per episode around that, a fifth of episodes sharp
# (mdp.depth_history edge smoothing).
REAL_CAMERA_EDGE_BLUR = {
  "edge_blur_prob": 0.8,
  "edge_blur_sigma": (1.0, 2.5),
  "edge_blur_aspect": (0.3, 0.7),
}
REAL_CAMERA_ROLL_YAW_DEG = 1.5
REAL_CAMERA_PITCH_DEG = 3.0
REAL_CAMERA_X_Z_M = (0.01, 0.015)
REAL_CAMERA_FOVY_SCALE = (0.97, 1.03)


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
  if "teachblur" in set(filter(None, os.environ.get("BLIND_PILOT", "").split(","))):
    # The teacher's actor scan as the student's camera sees it (2026-11-05, the new pipeline: a teacher the student
    # can follow): risers blurred along the heading, a per-episode height bias and slope, stale cells
    # (mdp.teacher_scan_camlike); the critic keeps the exact scan.
    from src.tasks.velocity.mdp.observations import teacher_scan_camlike

    size_x, res = teacher_scan.pattern.size[0], teacher_scan.pattern.resolution
    actor_scan.func = teacher_scan_camlike
    actor_scan.params = {
      **actor_scan.params,
      "nx": int(round(size_x / res)) + 1,
      "ny": int(round(SCAN_WIDTH / res)) + 1,
      "resolution": res,
      "sigma_cells": (0.0, 0.8),
      "bias_m": 0.02,
      "slope": 0.05,
      "dropout": 0.05,
    }
  actor.terms[TEACHER_SCAN_NAME] = actor_scan
  return cfg


# Share of the student's robots drawn from the outer DR_EDGE_BAND of each end of the payload, base CoM and foot
# friction ranges (DEPTH_STUDENT_DR_EDGE, training only); the rest are drawn as before.
DR_EDGE_BAND = 0.1


def _edge_sample(lo, hi, shape, device, log: bool = False) -> torch.Tensor:
  share = float(os.environ.get("DEPTH_STUDENT_DR_EDGE", "0"))
  lo = torch.as_tensor(lo, dtype=torch.float, device=device)
  hi = torch.as_tensor(hi, dtype=torch.float, device=device)
  if log:
    lo, hi = lo.log(), hi.log()
  width = hi - lo
  base = lo + torch.rand(shape, device=device) * width
  band = torch.rand(shape, device=device) * DR_EDGE_BAND * width
  edge = torch.where(torch.rand(shape, device=device) < 0.5, lo + band, hi - band)
  out = torch.where(torch.rand(shape, device=device) < share, edge, base)
  return out.exp() if log else out


def _edge_uniform(lo, hi, shape, device):
  return _edge_sample(lo, hi, shape, device)


def _edge_log_uniform(lo, hi, shape, device):
  return _edge_sample(lo, hi, shape, device, log=True)


EDGE_DISTRIBUTIONS = {
  "uniform": dr.Distribution(name="edge_uniform", sample=_edge_uniform),
  "log_uniform": dr.Distribution(name="edge_log_uniform", sample=_edge_log_uniform),
}


def apply_depth_student(
  cfg: ManagerBasedRlEnvCfg,
  play: bool = False,
  camera_pos: tuple[float, float, float] = depth_mdp.DEPTH_CAMERA_POS,
  camera_quat: tuple[float, float, float, float] = depth_mdp.DEPTH_CAMERA_QUAT,
  camera_body: str = "base_link",
  camera: depth_mdp.DepthCameraSpec = depth_mdp.PIE_DEPTH_CAMERA,
  real_camera: bool = False,
  occlusion: bool = False,
  edge_blur: bool = False,
) -> ManagerBasedRlEnvCfg:
  """A depth teacher task plus the front depth camera and the student groups.

  `camera`: what the policy sees (mdp.DEPTH_CAMERAS: tinynav-sim's PIE camera by
  default, NAVCORE_DEPTH_CAMERA the real one). `real_camera`: train on a real
  stereo camera's latency, frame timing, errors and mounting (REAL_CAMERA_*)
  instead of tinynav-sim's. `occlusion` (with `real_camera`): also things in front
  of the lens and a covered lens (REAL_CAMERA_OCCLUSION). `edge_blur` (with
  `real_camera`): the stereo camera's smoothing of depth edges (REAL_CAMERA_EDGE_BLUR).

  Groups (go2-depth's make_depth_student_env_cfg): `actor` (the teacher's input,
  unchanged), `critic` (unused), `proprio` (the actor's proprio terms now),
  `proprio_history` (x 10, term-major, oldest first) and `depth` (2 x height x
  width of the camera); the last three are the deployed ONNX inputs of the same
  name. `terrain_target` is the training target of the student's optional
  terrain head.
  """
  robot = cfg.scene.entities["robot"]
  mount = depth_mdp.front_depth_camera_spec_cfg(pos=camera_pos, quat=camera_quat, body=camera_body, camera=camera)
  cfg.scene.entities["robot"] = replace(robot, cameras=robot.cameras + (mount,))
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (depth_mdp.front_depth_camera_cfg(DEPTH_SENSOR_NAME, camera),)

  if not play:
    # Mounting tolerance; also covers tinynav's native GL depth reading ~0.5 px
    # (~0.5 deg pitch) lower than the ray-cast training render (go2-depth).
    cam = SceneEntityCfg("robot", camera_names=(depth_mdp.FRONT_DEPTH_CAMERA,))
    roll_yaw, pitch = (REAL_CAMERA_ROLL_YAW_DEG, REAL_CAMERA_PITCH_DEG) if real_camera else (1.0, 2.0)
    cfg.events["depth_camera_pose"] = EventTermCfg(
      mode="reset",
      func=dr.cam_quat,
      params={
        "roll_range": (-math.radians(roll_yaw), math.radians(roll_yaw)),
        "pitch_range": (-math.radians(pitch), math.radians(pitch)),
        "yaw_range": (-math.radians(roll_yaw), math.radians(roll_yaw)),
        "asset_cfg": cam,
      },
    )
    x, z = REAL_CAMERA_X_Z_M
    ranges = {0: (-x, x), 1: (-0.01, 0.01), 2: (-z, z)} if real_camera else (-0.01, 0.01)
    cfg.events["depth_camera_pos"] = EventTermCfg(
      mode="reset",
      func=dr.cam_pos,
      params={"ranges": ranges, "operation": "add", "asset_cfg": deepcopy(cam)},
    )
    if real_camera:
      cfg.events["depth_camera_fovy"] = EventTermCfg(
        mode="reset",
        func=dr.cam_fovy,
        params={"ranges": REAL_CAMERA_FOVY_SCALE, "operation": "scale", "asset_cfg": deepcopy(cam)},
      )

  if not play and os.environ.get("DEPTH_STUDENT_HARD"):
    # More of the commands the students fell on (go2-depth, 2026-11-01): the As2W and B2 depth_fe students fell
    # 1-2 of 256 walking sideways at 0.6 m/s, spinning up to 1.5 rad/s from a stand (B2) and stopping from fast
    # commands (As2W), where their teachers fell 0 on seeds 0 and 1. Only the student's training commands change
    # (DEPTH_STUDENT_HARD set): fast sideways walking and stands after fast commands at least 0.3 / 0.75, and the
    # legged spin-ups from a stand doubled (at least 0.4). The teacher, the evaluations and the deployment do not.
    twist = cfg.commands["twist"]
    twist.fast_side_prob = max(2.0 * twist.fast_side_prob, 0.3)
    twist.stop_after_fast_prob = max(twist.stop_after_fast_prob, 0.75)
    if twist.fast_spin_prob > 0.0:
      twist.fast_spin_prob = max(2.0 * twist.fast_spin_prob, 0.4)

  if not play and os.environ.get("DEPTH_STUDENT_SPIN_LEFT"):
    # Spin-ups from a stand turn left (+wz) this share of the time (go2-depth, 2026-11-02, B2): its students tilted
    # more spinning up to the left (max tilt p99 25 deg at +1.5 rad/s vs 9.4 at -1.5; the teacher, trained with the
    # mirror loss, 13.9 / 9.8; go2-home's drift_diag.py) and fell there 1 of 256. The distillation and the student
    # PPO have no mirror loss, and the camera sits 5 cm left. Training commands only.
    cfg.commands["twist"].fast_spin_pos_prob = float(os.environ["DEPTH_STUDENT_SPIN_LEFT"])

  if not play and os.environ.get("DEPTH_STUDENT_STAND_LONG"):
    # Zero commands off the stairs hold 4-8 s, not 2-5 (go2-depth, 2026-11-02, As2W; the pilots' "standlong" /
    # "standlongw"): the stop acceptance stands 6 s, and 2 of 3 of the As2W students' stop falls came 5-6 s after the
    # stop (go2-home's stopfall diagnosis), a stand they never trained. Robots whose pilot already has it keep theirs.
    twist = cfg.commands["twist"]
    if twist.stand_resampling_time_range is None:
      twist.stand_resampling_time_range = (4.0, 8.0)

  if not play and os.environ.get("DEPTH_STUDENT_FRICTION") and "foot_friction" in cfg.events:
    # The student's foot friction drawn as the flat acceptance draws it (go2-depth, 2026-11-02, the user's choice for
    # As2W): wheeled stairs tasks train on 0.4-1.0 (blind_env_cfg: high wheel friction lets a spinning wheel climb a
    # riser face) and "lowfric" draws it log-uniform, while the flat checks draw 0.3-1.6 uniform; 3 of the As2W
    # students' fallers (also the deployed v4's) stood on 1.26-1.42. With DEPTH_STUDENT_FRICTION=<lo>,<hi> the
    # student's robots draw it uniform over that range. A student only imitates, so no riser exploit can be learned;
    # the teacher, the evaluations and the deployment do not change. Before DEPTH_STUDENT_DR_EDGE, which then draws the
    # edges of this range.
    lo, hi = (float(x) for x in os.environ["DEPTH_STUDENT_FRICTION"].split(","))
    ev = cfg.events["foot_friction"]
    ev.params = {**ev.params, "ranges": (lo, hi), "distribution": "uniform"}

  if not play and os.environ.get("DEPTH_STUDENT_DR_EDGE"):
    # More robots at the edges of the randomization (go2-depth, 2026-11-02): the As2W and B2 depth_fe students' rare
    # falls (1-2 of 256) were the same robots on every rerun, checkpoint and round of an env seed, robots near the
    # ends of the ranges (B2: payload +8.6 kg p92, foot friction 0.31 p2; As2W: base CoM offsets p93-p98, payload
    # p94), where their teachers fell 0 (scratchpad wh2_port/jobs/fall_diag.py). With DEPTH_STUDENT_DR_EDGE=<share>
    # that share of the student's robots draws its payload, each base CoM axis and its foot friction from the outer
    # DR_EDGE_BAND of either end of the range (the ranges are unchanged). The teacher, the evaluations and the
    # deployment do not change.
    # pd_gains (2026-11-02): the B2 spin faller had a calf kp at p1 and a thigh kp at p99, an As2W stop faller a calf
    # kp at p99.6; each joint's kp and kd are drawn on its own.
    for name in ("base_payload", "base_com", "foot_friction", "pd_gains"):
      if name in cfg.events:
        ev = cfg.events[name]
        dist = ev.params.get("distribution", "uniform")
        if isinstance(dist, str) and dist in EDGE_DISTRIBUTIONS:
          ev.params = {**ev.params, "distribution": EDGE_DISTRIBUTIONS[dist]}

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
  depth_params: dict = {"sensor_name": DEPTH_SENSOR_NAME, "camera": camera.name}
  if not play:
    depth_params |= {"noise_std": 0.01, "dropout_prob": 0.01}
    depth_params |= REAL_CAMERA_DEPTH if real_camera else {"stale_prob": 0.2}
    if occlusion:
      assert real_camera, "occlusions ride on the real camera's randomization"
      depth_params |= REAL_CAMERA_OCCLUSION
    if edge_blur:
      assert real_camera, "edge smoothing rides on the real camera's randomization"
      depth_params |= REAL_CAMERA_EDGE_BLUR
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

