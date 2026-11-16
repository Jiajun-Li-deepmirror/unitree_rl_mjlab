"""Acceptance item 31, energy (2026-11-16, go2-home's spec; report only until w27d is measured and the bars are set).

  (a) the flat cost of transport: the positive mechanical power of the actuators (legs and wheels), integrated, over
      the total mass (the payload in) x g x the distance -- scripts/blind/eval/drift_all.py, "COT" rows
  (b) the standing heat proxy at the zero command: sum of tau^2 over the leg actuators (N^2 m^2; no winding
      resistance known, the bar is set from w27d's), each robot's time mean, p50 / p90 -- long_stand.py, "TAU2" line
  (c) the climb energy: the positive mechanical work per metre of rise on the up flights -- scripts/unified/
      descent_probe.py, "ENERGY" rows

mjlab's actuator_force is in actuator order; the joint velocities in joint order: act_joint_ids maps them by name.
"""
import torch


def act_joint_ids(robot) -> torch.Tensor:
  names = list(robot.joint_names)
  ids = []
  for a in robot.actuator_names:
    n = a if a in names else next((j for j in names if a.endswith(j) or j.endswith(a)), None)
    assert n is not None, f"actuator {a}: no joint of that name ({names})"
    ids.append(names.index(n))
  return torch.tensor(ids, device=robot.data.joint_vel.device, dtype=torch.long)


def pos_power(robot, ids: torch.Tensor) -> torch.Tensor:
  """The positive mechanical power per robot (W): sum over the actuators of max(force x joint speed, 0)."""
  return (robot.data.actuator_force * robot.data.joint_vel[:, ids]).clamp(min=0.0).sum(dim=1)


def leg_tau2(robot, ids: torch.Tensor) -> torch.Tensor:
  """Sum of tau^2 over the leg actuators per robot (N^2 m^2)."""
  legs = torch.tensor([i for i, j in enumerate(ids.tolist()) if "wheel" not in robot.joint_names[j]],
                      device=ids.device, dtype=torch.long)
  return (robot.data.actuator_force[:, legs] ** 2).sum(dim=1)


def total_mass(env, robot) -> torch.Tensor:
  """Each robot's total mass (kg), the payload in (read after the payload event)."""
  bm = env.sim.model.body_mass
  ids = robot.indexing.body_ids
  return bm[:, ids].sum(dim=1) if bm.dim() == 2 else bm[ids].sum().expand(env.num_envs)
