"""Initialize reference targets without advancing simulation or motion time."""

def initialize_reference_targets(cmd):
  """Fill mjlab 1.2's relative targets at t=0 without advancing physics/time.

  MotionCommand.reset() leaves these buffers uninitialized until its first
  update. Match MotionCommand._update_command's alignment, but do not increment
  time_steps, resample the motion, or discard a policy step.
  """
  from mjlab.utils.lab_api.math import quat_apply, quat_inv, quat_mul, yaw_quat

  anchor = cmd.anchor_pos_w[:, None, :]
  delta_pos = cmd.robot_anchor_pos_w[:, None, :].clone()
  delta_pos[..., 2] = anchor[..., 2]
  delta_quat = yaw_quat(quat_mul(cmd.robot_anchor_quat_w, quat_inv(cmd.anchor_quat_w)))
  delta_quat = delta_quat[:, None, :].expand(-1, len(cmd.cfg.body_names), -1)
  cmd.body_pos_relative_w = delta_pos + quat_apply(delta_quat, cmd.body_pos_w - anchor)
  cmd.body_quat_relative_w = quat_mul(delta_quat, cmd.body_quat_w)
  cmd._update_metrics()

