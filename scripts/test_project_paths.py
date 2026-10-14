"""Training output routing regressions without performing any training."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import project_paths
import train


class OutputPathsTests(unittest.TestCase):
  def test_motion_name_overrides_experiment_and_current_directory(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      motion = root / "output" / "街舞.v1" / "街舞.v1.npz"
      motion.parent.mkdir(parents=True)
      motion.touch()
      agent = SimpleNamespace(experiment_name="old_experiment", run_name="")
      cfg = SimpleNamespace(motion_file=str(motion), agent=agent, gpu_ids=[])
      with patch.object(project_paths, "PROJECT_ROOT", root), \
           patch.object(train, "select_gpus", return_value=(None, 0)), \
           patch.object(train, "run_train") as run, patch.dict(train.os.environ):
        train.launch_training("test", cfg)
      log_dir = run.call_args.args[2]
      self.assertEqual(log_dir.parent, motion.parent / "training")
      self.assertTrue(log_dir.name.startswith("街舞.v1_"))
      self.assertEqual(agent.experiment_name, "街舞.v1")

  def test_nontracking_tasks_still_write_inside_output(self):
    path = project_paths.training_dir(None, "walking")
    self.assertEqual(path, project_paths.PROJECT_ROOT / "output/walking/training")

  def test_candidate_stays_under_its_source_video_output(self):
    with tempfile.TemporaryDirectory() as tmp:
      root=Path(tmp)
      motion=root/'output/dance/contact_v1/dance_contact_v1.npz'
      with patch.object(project_paths,'PROJECT_ROOT',root):
        self.assertEqual(project_paths.training_dir(motion),motion.parent/'training')
        self.assertEqual(project_paths.motion_output_dir(motion),motion.parent)

  def test_experiment_path_cannot_escape_output(self):
    with self.assertRaises(ValueError):
      project_paths.training_dir(None, "../../escape")

  def test_video_name_containing_model_does_not_truncate_export_path(self):
    from src.tasks.tracking.rl import runner as tracking
    runner = tracking.MotionTrackingOnPolicyRunner.__new__(tracking.MotionTrackingOnPolicyRunner)
    motion = SimpleNamespace(cfg=SimpleNamespace(anchor_body_name="torso", body_names=["torso"]))
    runner.env = SimpleNamespace(unwrapped=SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: motion)))
    runner.logger = SimpleNamespace(logger_type="tensorboard")
    runner.export_motion_policy_to_onnx = Mock()
    runner.export_policy_to_onnx = Mock()
    run = "output/model_dance/training/model_dance_run"
    with patch.object(tracking.MjlabOnPolicyRunner, "save"), \
         patch.object(tracking, "get_base_metadata", return_value={}), \
         patch.object(tracking, "attach_metadata_to_onnx"):
      runner.save(run + "/model_1.pt")
    runner.export_motion_policy_to_onnx.assert_called_once_with(run, "model_dance_run.onnx")
    runner.export_policy_to_onnx.assert_called_once_with(run, "policy.onnx")


if __name__ == "__main__":
  unittest.main()
