"""Write the C++ deploy parameters (deploy.yaml) of a blind velocity task.

Reads the task's play config: policy joint order (the order of the actor's
joint observations, which is also the action order), default pose, actuator
gains, action terms and actor observations, and maps the joints to the robot's
SDK motor order. The joint-target clip is default +- 2.0 rad (raw action +-8)
for position terms and +-8 raw actions for velocity terms; it only catches
anomalous outputs.

  python scripts/blind/export_deploy_yaml.py Unitree-As2-Blind-Flat --robot as2 \
    --out deploy/robots/as2/config/policy/velocity/blind_v0/params/deploy.yaml
"""

import argparse
import re
from pathlib import Path

import yaml

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.actuator import (
  BuiltinPositionActuatorCfg,
  BuiltinVelocityActuatorCfg,
  DelayedActuatorCfg,
  IdealPdActuatorCfg,
)
from mjlab.entity import Entity
from mjlab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.string import resolve_matching_names, resolve_matching_names_values

import src.tasks.velocity.mdp as mdp

LEGS_SDK = [f"{leg}_{j}_joint" for leg in ("FR", "FL", "RR", "RL") for j in ("hip", "thigh", "calf")]
WHEELS_SDK = [f"{leg}_wheel_joint" for leg in ("FR", "FL", "RR", "RL")]
# Motor index order of the Unitree SDK low-level command, per robot.
SDK_JOINT_NAMES = {
  "go2": LEGS_SDK,
  "b2": LEGS_SDK,
  "as2": LEGS_SDK,
  "go2w": LEGS_SDK + WHEELS_SDK,
  "b2w": LEGS_SDK + WHEELS_SDK,
  "as2w": LEGS_SDK + WHEELS_SDK,
}
CLIP_RAW = 8.0
# mjlab observation term -> deploy observation (deploy/include/.../observations.h).
OBS_NAMES = {
  "base_ang_vel": "base_ang_vel",
  "projected_gravity": "projected_gravity",
  "command": "velocity_commands",
  "phase": "gait_phase",
  "joint_pos": "joint_pos_rel",
  "joint_vel": "joint_vel_rel",
  "actions": "last_action",
}


class FlowList(list):
  """Dumped inline, like the hand-written deploy files."""


yaml.add_representer(FlowList, lambda d, v: d.represent_sequence("tag:yaml.org,2002:seq", v, flow_style=True))


def rnd(x: float) -> float:
  return round(float(x), 6)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("task")
  parser.add_argument("--robot", required=True, choices=sorted(SDK_JOINT_NAMES))
  parser.add_argument("--out", required=True)
  args = parser.parse_args()

  cfg = load_env_cfg(args.task, play=True)
  robot_cfg = cfg.scene.entities["robot"]
  model_joints = list(Entity(robot_cfg).joint_names)
  sdk = SDK_JOINT_NAMES[args.robot]

  def joints_of(names: tuple[str, ...], preserve_order: bool = False) -> list[str]:
    return resolve_matching_names(names, model_joints, preserve_order)[1]

  actor = cfg.observations["actor"]
  vel_cfg = actor.terms["joint_vel"].params.get("asset_cfg")
  if vel_cfg is not None and vel_cfg.joint_names is not None:
    order = joints_of(tuple(vel_cfg.joint_names), vel_cfg.preserve_order)
  else:
    order = model_joints
  assert sorted(order) == sorted(model_joints) and sorted(order) == sorted(sdk), (order, sdk)

  default = dict.fromkeys(order, 0.0)
  for name in order:
    for expr, value in robot_cfg.init_state.joint_pos.items():
      if re.fullmatch(expr, name):
        default[name] = value

  kp = dict.fromkeys(order, 0.0)
  kd = dict.fromkeys(order, 0.0)
  for act in robot_cfg.articulation.actuators:
    base = act.base_cfg if isinstance(act, DelayedActuatorCfg) else act
    for name in joints_of(tuple(base.target_names_expr)):
      if isinstance(base, (BuiltinPositionActuatorCfg, IdealPdActuatorCfg)):
        kp[name], kd[name] = base.stiffness, base.damping
      elif isinstance(base, BuiltinVelocityActuatorCfg):
        kp[name], kd[name] = 0.0, base.damping
      else:
        raise TypeError(f"unsupported actuator {type(base).__name__}")

  actions, action_dim = {}, 0
  for term in cfg.actions.values():
    names = joints_of(tuple(term.actuator_names), term.preserve_order)
    ids = [order.index(n) for n in names]
    assert ids == list(range(action_dim, action_dim + len(ids))), "action terms must follow the policy joint order"
    action_dim += len(ids)
    if isinstance(term.scale, dict):
      # As in mjlab's JointAction: joints no pattern matches keep scale 1.
      scale = [1.0] * len(ids)
      index_list, _, value_list = resolve_matching_names_values(term.scale, names)
      for i, value in zip(index_list, value_list):
        scale[i] = rnd(value)
    else:
      scale = [rnd(term.scale)] * len(ids)
    if isinstance(term, JointPositionActionCfg):
      key = "JointPositionAction"
      offset = [rnd(default[n]) for n in names]
      clip = [FlowList([rnd(o - s * CLIP_RAW), rnd(o + s * CLIP_RAW)]) for o, s in zip(offset, scale)]
    elif isinstance(term, JointVelocityActionCfg):
      key = "JointVelocityAction"
      offset = [0.0] * len(ids)
      clip = [FlowList([rnd(-s * CLIP_RAW), rnd(s * CLIP_RAW)]) for s in scale]
    else:
      raise TypeError(f"unsupported action {type(term).__name__}")
    actions[key] = {
      "clip": clip,
      "joint_names": FlowList([".*"]) if len(ids) == len(order) else FlowList(names),
      "scale": FlowList(scale),
      "offset": FlowList(offset),
      "joint_ids": None if len(ids) == len(order) else FlowList(ids),
    }
    if isinstance(term, mdp.BrakedJointVelocityActionCfg):
      # Wheels brake (targets 0) while the joystick command is zero and every wheel
      # turns slower than brake_below_speed (rad/s), as in training.
      actions[key]["zero_when_command_zero"] = True
      actions[key]["brake_below_speed"] = rnd(term.brake_speed)

  observations = {}
  for name, term in actor.terms.items():
    params, dim = {}, None
    if name == "command":
      params, dim = {"command_name": "base_velocity"}, 3
    elif name == "phase":
      params, dim = {"period": term.params["period"]}, 2
    elif name in ("joint_pos", "joint_vel"):
      a = term.params.get("asset_cfg")
      names = joints_of(tuple(a.joint_names), a.preserve_order) if a is not None and a.joint_names else order
      dim = len(names)
      if names != order:
        params = {"asset_cfg": {"joint_ids": FlowList([order.index(n) for n in names])}}
    elif name == "actions":
      dim = action_dim
    else:
      dim = 3
    assert term.scale is None, f"obs scale not handled: {name}"
    observations[OBS_NAMES[name]] = {
      "params": params,
      "clip": None,
      "scale": FlowList([1.0] * dim),
      "history_length": actor.history_length,
    }

  twist = cfg.commands["twist"]
  out = {
    "joint_ids_map": FlowList([sdk.index(n) for n in order]),
    "step_dt": rnd(cfg.sim.mujoco.timestep * cfg.decimation),
    "stiffness": FlowList([rnd(kp[n]) for n in sdk]),
    "damping": FlowList([rnd(kd[n]) for n in sdk]),
    "default_joint_pos": FlowList([rnd(default[n]) for n in order]),
    "commands": {
      "base_velocity": {
        "scale_to_range": True,
        "zero_threshold": twist.zero_threshold,
        "ranges": {
          "lin_vel_x": FlowList(twist.ranges.lin_vel_x),
          "lin_vel_y": FlowList(twist.ranges.lin_vel_y),
          "ang_vel_z": FlowList(twist.ranges.ang_vel_z),
          "heading": None,
        },
      }
    },
    "actions": actions,
    "observations": observations,
  }
  path = Path(args.out)
  path.parent.mkdir(parents=True, exist_ok=True)
  header = (
    f"# Generated by scripts/blind/export_deploy_yaml.py from {args.task} ({args.robot}).\n"
    "# Policy joint order: " + ", ".join(order) + "\n"
  )
  path.write_text(header + yaml.dump(out, sort_keys=False, width=200))
  print(f"wrote {path}: {len(order)} joints, actions {action_dim}, obs {list(observations)}")


if __name__ == "__main__":
  main()
