#!/usr/bin/env python3
"""align_uros.py [ROBOT ...] [--check]: build src/assets/robots/unitree_<r>/xmls/<r>_uros.xml, our model with every
body's inertial (mass, centre of mass, inertia) taken from Unitree's own description in unitree_ros (go2-depth
2026-11-11, the user: "机器参数从 https://github.com/unitreerobotics/unitree_ros 里面获取 ... 先和unitree ros对齐，没有的
参数再和官网对齐").

Source: src/assets/robots/unitree_<r>/uros/ (the URDF -- and for As2 Unitree's as2.xml -- at unitree_ros 725b59b).
Each URDF link is assigned to one body of ours: itself when we have a body of that name, else the nearest ancestor
we have (the motor rotors, wheel motors, head / tail / lidar parts and the like sit on fixed joints). The links of a
body are combined (mass, centre of mass, inertia by the parallel-axis theorem) in that body's frame. Our model keeps
its kinematics (checked: every body's pose at the zero posture equals the URDF's), collision geometry (foot and wheel
radii checked equal) and names; only the <inertial> of each body is replaced. Joint passive damping from Unitree's
as2.xml (As2 legs: 0.2 / 0.2 / 0.5 N m s/rad) goes onto the As2 and As2W leg joints (the same legs; the As2W URDF has
no <dynamics>); the friction loss and the URDF torque limits are actuator settings, in src/assets/robots/uros.py.

--check: print each body's mass / centre-of-mass / inertia change and the totals, write nothing.
"""

from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ROBOTS = {"go2": "go2_description.urdf", "go2w": "go2w_description.urdf", "as2": "as2.urdf", "as2w": "as2w.urdf",
          "b2": "b2_description.urdf", "b2w": "b2w_description.urdf"}
# Unitree's as2.xml joint defaults (classes hip_joint / thigh_joint / calf_joint): passive damping on the As2 legs
LEG_DAMPING = {"as2": {"hip": 0.2, "thigh": 0.2, "calf": 0.5}, "as2w": {"hip": 0.2, "thigh": 0.2, "calf": 0.5}}


def candidates(n: str) -> list[str]:
  """The URDF link names our body n may have."""
  return [n, {"base_link": "base", "base": "base_link"}.get(n, n), n.replace("_wheel_link", "_foot"),
          n.replace("_wheel", "_foot")]


def compile_urdf(path: Path) -> mujoco.MjModel:
  """The URDF as MuJoCo bodies, one per link (no fusing), visual and collision geometry dropped, root at the origin."""
  r = ET.parse(path).getroot()
  for link in r.findall("link"):
    for tag in ("visual", "collision"):
      for e in link.findall(tag):
        link.remove(e)
    inert = link.find("inertial")
    if inert is not None and float(inert.find("mass").get("value")) < 1e-6:
      link.remove(inert)   # a massless marker (B2's lidar_link: 4e-15 kg with a degenerate inertia)
  # the root link stays welded at the origin: only the zero posture is needed
  for e in r.findall("mujoco"):
    r.remove(e)
  mj = ET.SubElement(r, "mujoco")
  ET.SubElement(mj, "compiler", {"fusestatic": "false", "discardvisual": "true", "balanceinertia": "true"})
  return mujoco.MjModel.from_xml_string(ET.tostring(r, encoding="unicode"))


class posed:
  """Body frames and inertia frames at the zero posture, relative to body `root`'s frame (our base starts above
  the ground; the URDF root sits at the origin)."""

  def __init__(self, m: mujoco.MjModel, root: int = 1):
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    R0, p0 = d.xmat[root].reshape(3, 3), d.xpos[root].copy()
    self.xpos = (d.xpos - p0) @ R0
    self.xipos = (d.xipos - p0) @ R0
    self.xmat = np.stack([(R0.T @ x.reshape(3, 3)).flatten() for x in d.xmat])
    self.ximat = np.stack([(R0.T @ x.reshape(3, 3)).flatten() for x in d.ximat])


def combined(u: mujoco.MjModel, ours: mujoco.MjModel) -> dict[str, dict]:
  """our body name -> {mass, ipos, iquat, diag} from the URDF links assigned to it, in our body's frame."""
  names = {u.body(i).name: i for i in range(1, u.nbody)}
  ours_of: dict[int, str] = {}
  for j in range(1, ours.nbody):
    n = ours.body(j).name
    hit = next((names[c] for c in candidates(n) if c in names), None)
    if hit is None:
      raise SystemExit(f"align_uros: our body {n} has no link in the URDF")
    ours_of[hit] = n
  root = next(i for i, nn in ours_of.items() if nn == ours.body(1).name)
  du, do = posed(u, root), posed(ours, 1)
  groups: dict[str, list[int]] = {n: [] for n in ours_of.values()}
  for i in range(1, u.nbody):
    k = i
    while k not in ours_of and k > 0:
      k = u.body_parentid[k]
    if k == 0:
      if u.body_mass[i] > 1e-9:
        raise SystemExit(f"align_uros: URDF link {u.body(i).name} ({u.body_mass[i]} kg) has no body of ours above it")
      continue
    groups[ours_of[k]].append(i)
  out = {}
  for n, links in groups.items():
    j = ours.body(n).id
    # our body's pose must be the URDF link's (inertials are given in the body frame)
    k = next(i for i, nn in ours_of.items() if nn == n)
    if np.abs(du.xpos[k] - do.xpos[j]).max() > 2e-3 or np.abs(du.xmat[k] - do.xmat[j]).max() > 2e-3:
      raise SystemExit(f"align_uros: {n}: pose differs from the URDF's ({du.xpos[k]} vs {do.xpos[j]})")
    M = sum(u.body_mass[i] for i in links)
    c = sum(u.body_mass[i] * du.xipos[i] for i in links) / M
    I = np.zeros((3, 3))
    for i in links:
      R = du.ximat[i].reshape(3, 3)
      r = du.xipos[i] - c
      I += R @ np.diag(u.body_inertia[i]) @ R.T + u.body_mass[i] * (r @ r * np.eye(3) - np.outer(r, r))
    Rb = do.xmat[j].reshape(3, 3)
    ipos = Rb.T @ (c - do.xpos[j])
    Ib = Rb.T @ I @ Rb
    w, V = np.linalg.eigh(Ib)
    if np.linalg.det(V) < 0:
      V[:, 0] = -V[:, 0]
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, V.flatten())
    out[n] = {"mass": M, "ipos": ipos, "iquat": q, "diag": np.maximum(w, 1e-9), "links": [u.body(i).name for i in links]}
  return out


def fmt(v) -> str:
  return " ".join(f"{float(x):.8g}" for x in v)


def build(r: str, check: bool) -> None:
  d = ROOT / "src" / "assets" / "robots" / f"unitree_{r}"
  src, dst = d / "xmls" / f"{r}.xml", d / "xmls" / f"{r}_uros.xml"
  u = compile_urdf(d / "uros" / ROBOTS[r])
  ours = mujoco.MjModel.from_xml_path(str(src))
  new = combined(u, ours)
  if check:
    print(f"== {r}: total URDF {u.body_mass.sum():.3f} kg, ours {ours.body_mass.sum():.3f} kg")
    for n, v in new.items():
      j = ours.body(n).id
      dm, dc = v["mass"] - ours.body_mass[j], np.abs(v["ipos"] - ours.body_ipos[j]).max()
      di = np.abs(np.sort(v["diag"]) - np.sort(ours.body_inertia[j])).max() / max(ours.body_inertia[j].max(), 1e-9)
      if abs(dm) > 1e-3 or dc > 1e-3 or di > 0.02:
        print(f"   {n:16s} mass {ours.body_mass[j]:.3f} -> {v['mass']:.3f}  com moves {dc * 1000:.1f} mm  "
              f"inertia {di * 100:.0f}%  (links {', '.join(v['links'])})")
    return
  tree = ET.parse(src)
  for body in tree.getroot().iter("body"):
    n = body.get("name")
    if n not in new:
      continue
    v = new[n]
    old = body.find("inertial")
    idx = list(body).index(old) if old is not None else 0
    if old is not None:
      body.remove(old)
    body.insert(idx, ET.Element("inertial", {"pos": fmt(v["ipos"]), "quat": fmt(v["iquat"]), "mass": f"{v['mass']:.8g}",
                                             "diaginertia": fmt(v["diag"])}))
  if r in LEG_DAMPING:
    for joint in tree.getroot().iter("joint"):
      jm = re.fullmatch(r"[FR][LR]_(hip|thigh|calf)_joint", joint.get("name", ""))
      if jm:
        joint.set("damping", str(LEG_DAMPING[r][jm.group(1)]))
  tree.getroot().insert(0, ET.Comment(f" Generated by tools/robot_params/align_uros.py from {r}.xml: inertials from "
                                      f"unitree_ros (uros/{ROBOTS[r]}, commit 725b59b). Do not edit by hand. "))
  ET.indent(tree, space="  ")
  tree.write(dst, encoding="unicode")
  got = mujoco.MjModel.from_xml_path(str(dst))
  print(f"{r}: wrote {dst.relative_to(ROOT)}: {got.body_mass.sum():.3f} kg (URDF {u.body_mass.sum():.3f}, was {ours.body_mass.sum():.3f})")


def main(argv: list[str]) -> int:
  check = "--check" in argv
  for r in [a for a in argv if not a.startswith("--")] or list(ROBOTS):
    build(r, check)
  return 0


if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))
