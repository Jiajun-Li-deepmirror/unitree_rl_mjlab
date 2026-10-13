"""Shared output layout and source import path for the enclosing dance pipeline."""
import os
from pathlib import Path
import sys

RL_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = RL_ROOT.parent.parent
# The enclosing project's src/ directory must not shadow this package's src/.
sys.path.insert(0, str(RL_ROOT))


def motion_output_dir(motion_file):
    return PROJECT_ROOT / "output" / Path(motion_file).stem


def training_dir(motion_file=None, experiment_name="training"):
    name = Path(motion_file).stem if motion_file else experiment_name
    if not name or name in {".", ".."} or Path(name).name != name:
        raise ValueError("experiment name must be a single folder name")
    return PROJECT_ROOT / "output" / name / "training"


def project_relative(path):
    return os.path.relpath(path, PROJECT_ROOT)
