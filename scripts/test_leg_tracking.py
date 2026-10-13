"""Reward and diagnostic regressions for large leg excursions."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import project_paths  # noqa: F401
from tracking_diagnostics import summarize
from src.tasks.tracking.mdp.rewards import motion_joint_position_error_exp, motion_swing_height_error_exp


class LegTests(unittest.TestCase):
  def test_attenuation_is_visible_even_when_average_error_is_small(self):
    x = np.sin(np.linspace(0, 2*np.pi, 100))[:, None]
    data = {"reference_joint_pos": x, "policy_joint_pos": .5*x,
            "env_id": np.zeros(100), "joint_names": np.array(["left_knee_joint"]), "body_names": []}
    result = summarize(data)
    self.assertAlmostEqual(result["joints"]["left_knee_joint"]["centered_tracking_gain"], .5)
    legs = result["joint_groups"]["legs"]
    self.assertGreater(legs["high_excursion_rmse_rad"], legs["rmse_rad"])

  def test_reward_uses_joint_names_and_does_not_average_in_arms(self):
    cmd = SimpleNamespace(robot=SimpleNamespace(joint_names=["arm", "knee"]),
                          joint_pos=torch.tensor([[9., 1.], [0., 1.]]),
                          robot_joint_pos=torch.tensor([[0., 1.], [0., .5]]))
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: cmd))
    reward = motion_joint_position_error_exp(env, "motion", .25, ("knee",))
    torch.testing.assert_close(reward, torch.tensor([1., np.exp(-4)], dtype=torch.float32))

  def test_swing_reward_does_not_force_grounded_foot_to_lift(self):
    cmd = SimpleNamespace(cfg=SimpleNamespace(body_names=["left", "right"]),
        body_pos_w=torch.tensor([[[0., 0., .03], [0., 0., .20]]]),
        robot_body_pos_w=torch.tensor([[[0., 0., .04], [0., 0., .20]]]))
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: cmd),
                          scene=SimpleNamespace(env_origins=torch.zeros((1, 3))))
    reward = motion_swing_height_error_exp(env, "motion", .05, ("left", "right"))
    self.assertEqual(reward.item(), 1.)
    cmd.robot_body_pos_w[0, 1, 2] = .03
    self.assertLess(motion_swing_height_error_exp(env, "motion", .05, ("left", "right")).item(), .51)

  def test_refinement_keeps_baseline_control_observations_and_safety(self):
    from src.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg, unitree_g1_leg_tracking_env_cfg
    baseline = unitree_g1_flat_tracking_env_cfg(has_state_estimation=False)
    candidate = unitree_g1_leg_tracking_env_cfg()
    self.assertEqual(baseline.observations, candidate.observations)
    self.assertEqual(baseline.actions, candidate.actions)
    self.assertEqual(baseline.terminations, candidate.terminations)
    self.assertEqual(baseline.events, candidate.events)
    self.assertEqual(candidate.scene.entities["robot"].articulation, baseline.scene.entities["robot"].articulation)
    for key, value in baseline.rewards.items():
      self.assertEqual(candidate.rewards[key], value)
    self.assertNotIn("motion_leg_joint_pos", baseline.rewards)


if __name__ == "__main__":
  unittest.main()
