"""Export a robot's training MJCF for tinynav-sim's multi-robot plant.

Writes the robot exactly as the depth tasks simulate it -- the robot's
EntityCfg with the blind/depth collision config (`blind_collision`) and its
actuators applied (position PD for the legs, velocity for the wheels, with
effort limits and armature) -- as one MJCF, plus a copy of its meshes:

  <out>/mjcf/<robot>.xml        meshdir="assets", mesh files renamed to .msh
                                 (tinynav-sim ships decimated MSH meshes in its
                                 model/ bundle; see its mujoco/scripts/
                                 decimate_mesh.py)
  <out>/meshes/<robot>/<file>   the original mesh files, input to decimation

Training-time actuator delays are domain randomization and are not exported.
The keyframe is dropped (tinynav-sim spawns the robot itself); the depth
camera mount is printed for tinynav-sim's robot registry.

  PYTHONPATH=. python scripts/go2_depth/export_sim_robot.py as2 b2 --out /tmp/robots
"""

from __future__ import annotations

import argparse
import importlib
import re
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

from mjlab.entity import Entity

from src.tasks.velocity.config.legged.blind import blind_collision

SPECS = {
  "go2": ("src.tasks.velocity.config.go2.blind_env_cfgs", "GO2_SPEC"),
  "go2w": ("src.tasks.velocity.config.go2w.depth_env_cfgs", "GO2W_SPEC"),
  "as2": ("src.tasks.velocity.config.as2.depth_env_cfgs", "AS2_SPEC"),
  "as2w": ("src.tasks.velocity.config.as2w.depth_env_cfgs", "AS2W_SPEC"),
  "b2": ("src.tasks.velocity.config.b2.depth_env_cfgs", "B2_SPEC"),
  "b2w": ("src.tasks.velocity.config.b2w.depth_env_cfgs", "B2W_SPEC"),
}


def _wheel_velocity_servos(cfg, xml: str) -> str:
  """IdealPdActuatorCfg computes its torque in mjlab and leaves a bare torque
  motor in the MJCF; with zero stiffness (the wheels) it is a velocity servo,
  torque = damping * (target - velocity), written here as MuJoCo's affine
  servo so tinynav-sim's control is the velocity target, as for the robots
  whose wheels were BuiltinVelocityActuatorCfg."""
  for actuator in cfg.articulation.actuators:
    base = getattr(actuator, "base_cfg", actuator)
    if type(base).__name__ != "IdealPdActuatorCfg" or base.stiffness != 0.0:
      continue
    kd, effort = float(base.damping), float(base.effort_limit)
    for pattern in base.target_names_expr:
      def servo(m: re.Match) -> str:
        if not re.fullmatch(pattern, m.group(1)):
          return m.group(0)
        return (f'<general name="{m.group(1)}" joint="{m.group(1)}" ctrllimited="false" '
                f'forcelimited="true" forcerange="{-effort:g} {effort:g}" biastype="affine" '
                f'gainprm="{kd:g}" biasprm="0 0 {-kd:g}"/>')
      xml = re.sub(r'<general name="([^"]+)" joint="[^"]+"[^>]*/>', servo, xml)
  return xml


def export(robot: str, out: Path) -> None:
  module, attr = SPECS[robot]
  spec = getattr(importlib.import_module(module), attr)
  cfg = replace(spec.robot_cfg(), collisions=(blind_collision(spec),))
  xml = Entity(cfg).spec.to_xml()

  xml = re.sub(r"\s*<size nkey=\"\d+\"/>", "", xml)
  xml = re.sub(r"\s*<keyframe>.*?</keyframe>", "", xml, flags=re.S)
  xml = xml.replace('meshdir="assets/"', 'meshdir="assets"')
  xml = re.sub(r' content_type="[^"]*"', "", xml)
  meshes = re.findall(r'<mesh [^>]*file="([^"]+)"', xml)
  xml = re.sub(r'file="([^"]+)\.(obj|stl|STL)"', r'file="\1.msh"', xml)
  xml = _wheel_velocity_servos(cfg, xml)

  commit = subprocess.run(
    ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
  ).stdout.strip()
  header = (
    f"<!--\n  {robot}: unitree_rl_mjlab training MJCF (go2-depth @ {commit}),\n"
    f"  scripts/go2_depth/export_sim_robot.py: {attr}.robot_cfg() with\n"
    "  blind_collision() and its actuators applied. Meshes are the decimated MSH\n"
    f"  set in the model/ bundle (model/{robot}/assets); meshdir is rewritten by\n"
    "  sim/plant.py. Regenerate instead of editing.\n-->\n"
  )
  (out / "mjcf").mkdir(parents=True, exist_ok=True)
  (out / "mjcf" / f"{robot}.xml").write_text(header + xml)

  mesh_out = out / "meshes" / robot
  mesh_out.mkdir(parents=True, exist_ok=True)
  asset_dir = Path(__file__).resolve().parents[2] / "src/assets/robots" / f"unitree_{robot}/xmls/assets"
  for name in meshes:
    shutil.copy2(asset_dir / name, mesh_out / name)
  print(
    f"{robot}: {len(meshes)} meshes, camera pos {spec.camera_pos} quat {spec.camera_quat}, "
    f"contract {spec.onnx_contract}, init z {cfg.init_state.pos[2]}"
  )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("robots", nargs="+", choices=sorted(SPECS))
  parser.add_argument("--out", type=Path, required=True)
  args = parser.parse_args()
  for robot in args.robots:
    export(robot, args.out)


if __name__ == "__main__":
  main()
