"""Measured ground contact and rigid-body velocity at each foot's strongest contact."""
import torch

FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
FOOT_SENSORS = ("left_foot_ground", "right_foot_ground")


def contact_point_velocity(linear, angular, body_position, contact_position):
  return linear + torch.cross(angular, contact_position - body_position, dim=-1)


def foot_contact_state(env, command, force_threshold=1.0):
  ids = [command.cfg.body_names.index(name) for name in FOOT_BODIES]
  data = [env.scene[name].data for name in FOOT_SENSORS]
  forces = torch.stack([torch.linalg.vector_norm(d.force[:, 0], dim=-1) for d in data], dim=-1)
  found = torch.stack([d.found[:, 0] > 0 for d in data], dim=-1)
  position = torch.stack([d.pos[:, 0] for d in data], dim=1)
  contact = found & (forces > force_threshold)
  velocity = contact_point_velocity(command.robot_body_lin_vel_w[:, ids],
      command.robot_body_ang_vel_w[:, ids], command.robot_body_pos_w[:, ids], position)
  # Sensors report zero positions without contact; do not expose fictitious velocities.
  velocity = torch.where(contact[..., None], velocity, torch.zeros_like(velocity))
  reference_height = command.body_pos_w[:, ids, 2] - env.scene.env_origins[:, None, 2]
  reference_speed = torch.linalg.vector_norm(command.body_lin_vel_w[:, ids, :2], dim=-1)
  return {"foot_contact": contact, "foot_contact_force_n": forces,
          "foot_contact_point_velocity_world": velocity,
          "foot_ankle_velocity_world": command.robot_body_lin_vel_w[:, ids],
          "reference_foot_planted": (reference_height < .05) & (reference_speed < .3),
          "reference_foot_height_m": reference_height}
