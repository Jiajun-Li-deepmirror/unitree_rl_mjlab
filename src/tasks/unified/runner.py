"""The unified teacher's runner (go2-depth 2026-11-11): the blind tasks' PPO with the left-right mirror loss (as
MirrorVelocityOnPolicyRunner, unchanged, with symmetry.py's mirror for the privileged terms), and the curriculum's
state in every checkpoint. The constraint
multipliers, their measurements, the progress and the meter's sums are saved under infos["unified"] and restored on
load, so a resumed run (a crash, a box change) carries on with them instead of starting every multiplier at 1 (the
blind chains' BLIND_CONS_INIT workaround). A checkpoint without them (any other run) loads as before."""

from __future__ import annotations

from src.tasks.velocity.rl import MirrorVelocityOnPolicyRunner, VelocityOnPolicyRunner


class UnifiedTeacherRunner(VelocityOnPolicyRunner):
  # MirrorVelocityOnPolicyRunner's mirror loss with the teacher's mirror (symmetry.py: its privileged terms too)
  MIRROR_LOSS_COEFF = MirrorVelocityOnPolicyRunner.MIRROR_LOSS_COEFF

  def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu"):
    symmetry_cfg = {
      "use_data_augmentation": False,
      "use_mirror_loss": True,
      "data_augmentation_func": "src.tasks.unified.symmetry:mirror_left_right",
      "mirror_loss_coeff": self.MIRROR_LOSS_COEFF,
    }
    train_cfg = {**train_cfg, "algorithm": {**train_cfg["algorithm"], "symmetry_cfg": symmetry_cfg}}
    super().__init__(env, train_cfg, log_dir, device)

  def _parts(self) -> dict:
    env = self.env.unwrapped
    return {name: getattr(env, attr, None) for name, attr in
            (("progress", "u_progress_term"), ("constraints", "u_constraints"), ("meter", "u_meter"))}

  def save(self, path: str, infos=None):
    env = self.env.unwrapped
    state = {"progress_p": getattr(env, "u_progress", 0.0)}
    for name, part in self._parts().items():
      if part is not None:
        state[name] = part.state()
    super().save(path, {**(infos or {}), "unified": state})

  def load(self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None):
    infos = super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)
    state = (infos or {}).get("unified")
    if state:
      env = self.env.unwrapped
      parts = self._parts()
      if parts["progress"] is not None and "progress" in state:
        parts["progress"].load(state["progress"], env)
      if parts["constraints"] is not None and "constraints" in state:
        parts["constraints"].load(state["constraints"])
      if parts["meter"] is not None and "meter" in state:
        parts["meter"].load(state["meter"])
      print(f"[unified] restored progress {state.get('progress_p', 0.0):.3f} and "
            f"{len(state.get('constraints', {}).get('scale', {}))} constraint multipliers from {path}")
    return infos
