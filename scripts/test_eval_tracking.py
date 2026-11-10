"""Evaluation regressions. Run in the unitree_rl_mjlab environment on CPU."""
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_tracking as ev
import torch


class RolloutTests(unittest.TestCase):
  def wrapper(self, fail_first=False):
    cmd = SimpleNamespace(motion=SimpleNamespace(joint_pos=torch.zeros(4, 29)),
                          metrics={"error_body_pos": torch.zeros(1)}, _update_metrics=lambda: None)
    env = SimpleNamespace(num_envs=1, step_dt=0.02, scene={"robot": SimpleNamespace(actuators=[])},
                          command_manager=SimpleNamespace(get_term=lambda _: cmd),
                          termination_manager=SimpleNamespace(
                            active_terms=["ee_body_pos"],
                            get_term=lambda _: torch.tensor([fail_first])))
    wrapper = SimpleNamespace(unwrapped=env, steps=0, reset=lambda: None,
                              get_observations=lambda: None)

    def step(action):
      wrapper.steps += 1
      return None, None, torch.tensor([int(fail_first)]), None
    wrapper.step = step
    return wrapper

  def test_first_step_failure_is_reported(self):
    wrapper = self.wrapper(fail_first=True)
    with patch.object(ev, "initialize_reference_targets"):
      duration, complete, terms, samples = ev.rollout(wrapper, lambda _: None)
    self.assertEqual(wrapper.steps, 1)
    self.assertEqual(complete, 0)
    self.assertEqual(terms, [{"env": 0, "t": 0.02, "terms": ["ee_body_pos"]}])
    self.assertEqual(samples, [])

  def test_full_motion_has_no_discarded_step_or_wrap(self):
    wrapper = self.wrapper()
    with patch.object(ev, "initialize_reference_targets"):
      duration, complete, terms, samples = ev.rollout(wrapper, lambda _: None)
    self.assertEqual(wrapper.steps, 3)
    self.assertEqual(duration, 0.06)
    self.assertEqual(complete, 1)
    self.assertEqual(terms, [])
    self.assertEqual([t for t, _, _ in samples], [0.02, 0.04, 0.06])

  def test_video_callback_excludes_automatically_reset_terminal_state(self):
    wrapper = self.wrapper(fail_first=True)
    recorded = []
    def record(env, cmd, t, valid):
      recorded.append((t, valid.tolist()))
    with patch.object(ev, "initialize_reference_targets"):
      ev.rollout(wrapper, lambda _: None, record)
    self.assertEqual(recorded, [(0.0, [True]), (0.02, [False])])


class ConfigTests(unittest.TestCase):
  def test_video_overlay_can_be_disabled_without_changing_training_config(self):
    import src.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg
    for enabled in (True, False):
      cfg = ev.configure_video(load_env_cfg(ev.TASK, play=True), enabled)
      self.assertEqual(cfg.commands["motion"].debug_vis, enabled)
      self.assertEqual(cfg.commands["motion"].viz.ghost_color, ev.REFERENCE_RGBA)
      self.assertEqual(cfg.viewer.max_extra_envs, 0)

  def test_play_modes_remove_joint_noise_without_changing_training(self):
    import src.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg
    for task in (ev.TASK, "Unitree-G1-23Dof-Tracking-No-State-Estimation"):
      with self.subTest(task=task):
        train = load_env_cfg(task)
        play = load_env_cfg(task, play=True)
        self.assertEqual(train.commands["motion"].joint_position_range, (-0.1, 0.1))
        self.assertEqual(play.commands["motion"].joint_position_range, (0.0, 0.0))

  def test_randomize_only_keeps_startup_events(self):
    import src.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg
    for randomize in (False, True):
      cfg = ev.prepare_eval_cfg(load_env_cfg(ev.TASK, play=True), "motion.npz", 2, randomize, 17)
      self.assertEqual(cfg.seed, 17)
      self.assertEqual(cfg.commands["motion"].joint_position_range, (0.0, 0.0))
      self.assertNotIn("push_robot", cfg.events)
      self.assertEqual(bool(cfg.events), randomize)
      if randomize:
        self.assertTrue(all(event.mode == "startup" for event in cfg.events.values()))

  def test_robustness_options_keep_pushes_and_noise_and_delay_every_actuator(self):
    import src.tasks  # noqa: F401
    from mjlab.actuator import DelayedActuatorCfg
    from mjlab.tasks.registry import load_env_cfg
    plain = ev.prepare_eval_cfg(load_env_cfg(ev.TASK, play=True), "motion.npz", 2, True, 17)
    self.assertFalse(plain.observations["actor"].enable_corruption)
    self.assertFalse(any(isinstance(a, DelayedActuatorCfg)
                         for a in plain.scene.entities["robot"].articulation.actuators))
    push = load_env_cfg(ev.TASK).events["push_robot"]
    cfg = ev.prepare_eval_cfg(load_env_cfg(ev.TASK, play=True), "motion.npz", 2, False, 17,
                              push=push, obs_noise=True, latency_steps=4)
    self.assertEqual(list(cfg.events), ["push_robot"])
    self.assertEqual(cfg.events["push_robot"].mode, "interval")
    self.assertTrue(cfg.observations["actor"].enable_corruption)
    actuators = cfg.scene.entities["robot"].articulation.actuators
    self.assertTrue(actuators)
    for a in actuators:
      self.assertIsInstance(a, DelayedActuatorCfg)
      self.assertEqual((a.delay_min_lag, a.delay_max_lag), (4, 4))


def standing_motion(directory):
  """A standing fixture from forward kinematics; no historical dance data or policy needed."""
  import mujoco
  import numpy as np
  sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "motion_extraction"))
  from g1_defaults import DEFAULT_DOF, DEFAULT_ROOT_Z
  from validate_motion import DEFAULT_XML
  model = mujoco.MjModel.from_xml_path(str(DEFAULT_XML))
  data = mujoco.MjData(model)
  data.qpos[:7] = [0, 0, DEFAULT_ROOT_Z, 1, 0, 0, 0]
  data.qpos[7:] = DEFAULT_DOF
  mujoco.mj_forward(model, data)
  motion = Path(directory) / "standing.npz"
  n, b = 50, model.nbody - 1
  np.savez(motion, fps=np.array([50.0]),
           joint_pos=np.tile(data.qpos[7:], (n, 1)), joint_vel=np.zeros((n, model.nv - 6)),
           body_pos_w=np.tile(data.xpos[1:], (n, 1, 1)),
           body_quat_w=np.tile(data.xquat[1:], (n, 1, 1)),
           body_lin_vel_w=np.zeros((n, b, 3)), body_ang_vel_w=np.zeros((n, b, 3)))
  return str(motion)


class CpuLatencyTests(unittest.TestCase):
  def test_delayed_targets_reach_the_actuators_one_control_step_late(self):
    import src.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg
    tmp = tempfile.TemporaryDirectory()
    self.addCleanup(tmp.cleanup)
    motion = standing_motion(tmp.name)
    ctrl = {}
    for steps in (0, 4):  # 4 physics steps = one 20 ms control step
      env = ManagerBasedRlEnv(cfg=ev.prepare_eval_cfg(load_env_cfg(ev.TASK, play=True), motion, 1,
                                                      False, 42, latency_steps=steps), device="cpu")
      try:
        venv = RslRlVecEnvWrapper(env)
        venv.reset()
        ev.restart_delays(env)
        ev.initialize_reference_targets(env.command_manager.get_term("motion"))
        venv.step(torch.zeros((1, 29)))
        before = env.sim.data.ctrl.clone()
        venv.step(torch.full((1, 29), 0.5))
        ctrl[steps] = (before, env.sim.data.ctrl.clone())
      finally:
        env.close()
    undelayed_before, undelayed_after = ctrl[0]
    delayed_before, delayed_after = ctrl[4]
    self.assertGreater((undelayed_after - undelayed_before).abs().max().item(), 1e-3)
    # The first target is held from the start (not the zero target written while resetting).
    torch.testing.assert_close(delayed_before, undelayed_before)
    # The new target was sent, but the delayed actuators still apply the previous one.
    torch.testing.assert_close(delayed_after, delayed_before)


class CpuResetTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    import src.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg
    tmp = tempfile.TemporaryDirectory()
    cls.addClassCleanup(tmp.cleanup)
    motion = standing_motion(tmp.name)
    cfg = ev.prepare_eval_cfg(load_env_cfg(ev.TASK, play=True), motion, 2, False, 42)
    cls.env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    cls.venv = RslRlVecEnvWrapper(cls.env)
    cls.cmd = cls.env.command_manager.get_term("motion")

  @classmethod
  def tearDownClass(cls):
    cls.env.close()

  def test_reset_is_repeatable_and_relative_targets_are_ready_at_zero(self):
    self.venv.reset()
    ev.initialize_reference_targets(self.cmd)
    first = self.cmd.robot_joint_pos.clone()
    torch.testing.assert_close(first, self.cmd.joint_pos, atol=1e-6, rtol=0)
    torch.testing.assert_close(self.cmd.body_pos_relative_w, self.cmd.robot_body_pos_w,
                               atol=1e-5, rtol=0)
    self.assertEqual(self.cmd.time_steps.tolist(), [0, 0])
    self.venv.reset()
    ev.initialize_reference_targets(self.cmd)
    torch.testing.assert_close(first, self.cmd.robot_joint_pos, atol=0, rtol=0)
    self.assertFalse(self.env.termination_manager.compute().any().item())

  def test_first_real_step_has_no_uninitialized_target_termination(self):
    self.venv.reset()
    ev.initialize_reference_targets(self.cmd)
    _, _, dones, _ = self.venv.step(torch.zeros((2, 29)))
    self.assertEqual(dones.tolist(), [0, 0])
    self.assertEqual(self.cmd.time_steps.tolist(), [1, 1])

  def test_ghost_uses_current_reference_world_pose_and_does_not_mutate_physics(self):
    import mujoco
    import numpy as np
    from mjlab.viewer.native.visualizer import MujocoNativeDebugVisualizer
    self.venv.reset()
    model = self.env.sim.mj_model
    original_color, original_material = model.geom_rgba.copy(), model.geom_matid.copy()
    before = self.env.sim.data.qpos.clone()
    ev.prepare_reference_overlay(self.env)
    positions = self.cmd.motion.body_pos_w.clone()
    joints = self.cmd.motion.joint_pos.clone()
    positions[5, 0, :2] += torch.tensor([0.3, -0.2])
    joints[5, 0] += 0.25
    captured = []
    visualizer = SimpleNamespace(get_env_indices=lambda _: [0],
      add_ghost_mesh=lambda qpos, **kwargs: captured.append((qpos, kwargs["model"])))
    with patch.object(self.cmd, "time_steps", torch.tensor([5, 6])), \
         patch.object(self.cmd.motion, "body_pos_w", positions), \
         patch.object(self.cmd.motion, "joint_pos", joints):
      self.cmd._debug_vis_impl(visualizer)
      qpos, ghost = captured[0]
      root = self.cmd.robot.indexing.free_joint_q_adr.cpu().numpy()
      np.testing.assert_allclose(qpos[root[:3]], self.cmd.body_pos_w[0, 0].numpy())
      np.testing.assert_allclose(qpos[root[3:7]], self.cmd.body_quat_w[0, 0].numpy())
      np.testing.assert_allclose(qpos[self.cmd.robot.indexing.joint_q_adr.cpu().numpy()], joints[5])
    np.testing.assert_array_equal(model.geom_rgba, original_color)
    np.testing.assert_array_equal(model.geom_matid, original_material)
    torch.testing.assert_close(self.env.sim.data.qpos.clone(), before)
    # Inspect the actual MuJoCo scene without an OpenGL context: appearance
    # must survive material processing, not only a Python color assignment.
    scene = mujoco.MjvScene(model, maxgeom=10000)
    native = MujocoNativeDebugVisualizer(scene, model, env_idx=0)
    native.add_ghost_mesh(qpos, model=ghost)
    self.assertGreater(scene.ngeom, 0)
    for i in range(scene.ngeom):
      np.testing.assert_allclose(scene.geoms[i].rgba, ev.REFERENCE_RGBA, atol=1e-6)


if __name__ == "__main__":
  unittest.main()
