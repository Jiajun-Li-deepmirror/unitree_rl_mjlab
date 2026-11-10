"""Reward and diagnostic regressions for large leg excursions."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import project_paths  # noqa: F401
from tracking_diagnostics import summarize, summarize_contact
from src.tasks.tracking.mdp.rewards import motion_joint_position_error_exp, motion_swing_height_error_exp
from src.tasks.tracking.mdp.foot_contact import foot_contact_state, contact_point_velocity


class LegTests(unittest.TestCase):
  def test_rolling_contact_is_stationary_even_when_ankle_moves(self):
    actual = contact_point_velocity(torch.tensor([[[1., 0., 0.]]]),
        torch.tensor([[[0., 10., 0.]]]), torch.tensor([[[0., 0., .1]]]), torch.zeros(1, 1, 3))
    torch.testing.assert_close(actual, torch.zeros_like(actual))

  def test_airborne_foot_is_not_penalized_and_sensors_are_in_named_order(self):
    from src.tasks.tracking.mdp.rewards import motion_measured_contact_slip
    def sensor(force):
      return SimpleNamespace(data=SimpleNamespace(found=torch.ones(1, 1),
          force=torch.tensor([[[force, 0., 0.]]]), pos=torch.zeros(1, 1, 3)))
    class Scene(dict):
      env_origins = torch.zeros(1, 3)
    scene = Scene(left_foot_ground=sensor(.5), right_foot_ground=sensor(10.))
    # Robot command order is deliberately reversed.
    cmd = SimpleNamespace(cfg=SimpleNamespace(body_names=['right_ankle_roll_link','left_ankle_roll_link']),
        body_pos_w=torch.zeros(1,2,3), body_lin_vel_w=torch.zeros(1,2,3),
        robot_body_pos_w=torch.zeros(1,2,3), robot_body_ang_vel_w=torch.zeros(1,2,3),
        robot_body_lin_vel_w=torch.tensor([[[2.,0.,0.],[100.,0.,0.]]]))
    env = SimpleNamespace(scene=scene, command_manager=SimpleNamespace(get_term=lambda _:cmd))
    state = foot_contact_state(env,cmd)
    self.assertEqual(state['foot_contact'].tolist(), [[False,True]])
    self.assertEqual(motion_measured_contact_slip(env,'motion').item(),4.)

  def test_contact_diagnostics_separate_slip_from_reference_phase_mismatch(self):
    d={'foot_contact':np.array([[1,0],[1,1]]),'reference_foot_planted':np.array([[1,1],[0,1]]),
       'foot_contact_point_velocity_world':np.array([[[.1,0,0],[0,0,0]],[[.3,0,0],[.2,0,0]]]),
       'foot_ankle_velocity_world':np.ones((2,2,3))}
    result=summarize_contact(d)
    self.assertAlmostEqual(result['contact_point_speed']['mean_m_s'],.2)
    self.assertEqual(result['reference_planted_without_contact_samples'],1)
    self.assertEqual(result['contact_point_speed_reference_unplanted']['samples'],1)

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

  def test_balanced_profile_adds_arms_and_contact_without_changing_control(self):
    from src.tasks.tracking.config.g1.env_cfgs import unitree_g1_leg_tracking_env_cfg, unitree_g1_balanced_leg_tracking_env_cfg
    original, balanced = unitree_g1_leg_tracking_env_cfg(), unitree_g1_balanced_leg_tracking_env_cfg()
    for attr in ('observations','actions','terminations','events','sim','decimation'):
      self.assertEqual(getattr(original,attr),getattr(balanced,attr))
    for name,value in original.rewards.items():
      if name != 'motion_swing_height':
        self.assertEqual(balanced.rewards[name],value)
    self.assertEqual(balanced.rewards['motion_swing_height'].params['std'],.10)
    self.assertEqual(len(balanced.rewards['motion_arm_joint_pos'].params['joint_names']),14)
    self.assertLess(balanced.rewards['measured_foot_contact_slip'].weight,0)


if __name__ == "__main__":
  unittest.main()
