"""Numerical regressions for control timing, retiming and free-flight diagnosis."""
import unittest

import numpy as np

from compare_control import predicted_pd_force, first_upcross
from diagnose_control import retime_motion
from reference_flight_diagnostics import flight_intervals


class ControlDiagnosticsTests(unittest.TestCase):
  def motion(self):
    return dict(fps=np.array([50.]), joint_pos=np.arange(6, dtype=float).reshape(3, 2),
        joint_vel=np.full((3, 2), 100.), body_pos_w=np.zeros((3, 1, 3)),
        body_quat_w=np.array([[[1.,0,0,0]], [[-1.,0,0,0]], [[1.,0,0,0]]]),
        body_lin_vel_w=np.ones((3, 1, 3))*2, body_ang_vel_w=np.ones((3, 1, 3))*4)

  def test_retiming_scales_all_velocities_but_preserves_control_fps(self):
    source = self.motion()
    result = retime_motion(source, .5, .02)
    self.assertEqual(result['joint_pos'].shape, (5, 2))
    np.testing.assert_array_equal(result['joint_pos'][1], [1.,2.])
    np.testing.assert_array_equal(result['fps'], [50.])
    np.testing.assert_allclose(result['joint_vel'], 50.)
    np.testing.assert_allclose(result['body_lin_vel_w'], 1.)
    np.testing.assert_allclose(result['body_ang_vel_w'], 2.)
    np.testing.assert_allclose(np.linalg.norm(result['body_quat_w'], axis=-1), 1.)
    np.testing.assert_allclose(abs(result['body_quat_w'][...,0]),1.)
    np.testing.assert_array_equal(source['joint_vel'],100.)

  def test_normal_speed_does_not_resample_or_alias_original(self):
    source=self.motion(); result=retime_motion(source,1.,.02)
    for name in source:
      np.testing.assert_array_equal(result[name],source[name])
      self.assertFalse(np.shares_memory(result[name],source[name]))

  def test_pd_torque_uses_actuator_joint_mapping_and_clips_force(self):
    d=dict(joint_names=np.array(['hip','knee']), actuator_joint_names=np.array(['knee','hip']),
        joint_pos=np.array([[2.,3.]]),joint_vel=np.array([[4.,5.]]),actuator_ctrl=np.array([[4.,0.]]),
        actuator_kp=np.array([100.,10.]),actuator_kd=np.array([2.,1.]),actuator_force_limits=np.array([80.,50.]))
    np.testing.assert_allclose(predicted_pd_force(d),[[80.,-24.]])

  def test_crossing_uses_interpolated_time_and_reports_missing_crossing(self):
    self.assertAlmostEqual(first_upcross(np.array([0.,.02,.04]),np.array([20.,40.,80.]),60.),.03)
    self.assertAlmostEqual(first_upcross(np.array([0.,.02,.04]),np.array([20.,40.,80.]),60.,held=True),.04)
    self.assertIsNone(first_upcross(np.array([0.,.02]),np.array([10.,15.]),20.))

  def test_ballistic_com_passes_free_flight_necessary_condition(self):
    t=np.arange(21)*.02
    com=np.zeros((21,3));com[:,2]=1+2*t-.5*9.81*t*t
    r=flight_intervals(t,com,np.full(21,.03),np.array([0.,0.,-9.81]),[0.,.4])[0]
    self.assertAlmostEqual(r['fitted_com_vertical_acceleration_m_s2'],-9.81)
    self.assertAlmostEqual(r['equivalent_upward_external_force_in_bodyweights'],0.)
    self.assertLess(r['best_ballistic_fit_max_residual_cm'],1e-10)

  def test_hovering_without_ground_contact_requires_external_support(self):
    t=np.arange(21)*.02;com=np.ones((21,3))
    r=flight_intervals(t,com,np.full(21,.03),np.array([0.,0.,-9.81]),[0.,.4])[0]
    self.assertAlmostEqual(r['equivalent_upward_external_force_in_bodyweights'],1.)
    self.assertEqual(flight_intervals(t,com,np.zeros(21),np.array([0.,0.,-9.81]),[0.,.4]),[])


if __name__ == '__main__':
  unittest.main()
