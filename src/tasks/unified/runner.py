"""The unified teacher's runner (go2-depth 2026-11-11): the blind tasks' PPO with the left-right mirror loss (as
MirrorVelocityOnPolicyRunner, unchanged, with symmetry.py's mirror for the privileged terms), and the curriculum's
state in every checkpoint. The constraint
multipliers, their measurements, the progress and the meter's sums are saved under infos["unified"] and restored on
load, so a resumed run (a crash, a box change) carries on with them instead of starting every multiplier at 1 (the
blind chains' BLIND_CONS_INIT workaround). A checkpoint without them (any other run) loads as before.

Evaluation robots (2026-11-12, constraints.py's docstring): the first constraints.eval_count(num_envs) robots act with
the policy's mean action, and the constraints are measured on them alone -- what the acceptance judges, not what the
exploration noise does to it. They take no part in the update: their advantages are zero (no policy gradient; the
other robots' advantages are normalized among themselves) and their value targets are the values the critic gave
(the mean action's returns are not the sampled policy's)."""

from __future__ import annotations

import torch

from src.tasks.velocity.rl import MirrorVelocityOnPolicyRunner, VelocityOnPolicyRunner

from .constraints import eval_count


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
    self._split_eval(eval_count(env.num_envs))

  def _split_eval(self, k: int) -> None:
    """The first k robots evaluate the mean action (module docstring)."""
    if k <= 0:
      return
    alg, e = self.alg, self.env.unwrapped
    if getattr(alg, "normalize_advantage_per_mini_batch", False):
      raise RuntimeError("evaluation robots: advantages normalized per mini-batch are not supported")
    e.u_eval_mask = torch.zeros(e.num_envs, dtype=torch.bool, device=e.device)
    e.u_eval_mask[:k] = True
    for m in (getattr(e, "u_meter", None), getattr(e, "u_lib_meter", None)):
      if m is not None:
        m.eval = True   # the stairs counts restart on a checkpoint taken over every robot
    act, compute_returns = alg.act, alg.compute_returns

    def act_eval(obs):
      actions = act(obs)
      actions[:k] = alg.transition.distribution_params[0][:k]   # the Gaussian's mean
      alg.transition.actions_log_prob = alg.actor.get_output_log_prob(actions).detach()
      return actions

    def returns_eval(obs):
      compute_returns(obs)
      st = alg.storage
      adv = st.returns - st.values
      train = adv[:, k:]
      st.advantages = (adv - train.mean()) / (train.std() + 1e-8)
      st.advantages[:, :k] = 0.0
      st.returns[:, :k] = st.values[:, :k]

    alg.act, alg.compute_returns = act_eval, returns_eval
    print(f"[unified] evaluation robots: the first {k} of {e.num_envs} act with the mean action; the constraints are "
          f"measured on them")

  def _parts(self) -> dict:
    env = self.env.unwrapped
    return {name: getattr(env, attr, None) for name, attr in
            (("progress", "u_progress_term"), ("constraints", "u_constraints"), ("meter", "u_meter"),
             ("lib_meter", "u_lib_meter_term"))}

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
      if parts["lib_meter"] is not None and "lib_meter" in state:
        parts["lib_meter"].load(state["lib_meter"])
      print(f"[unified] restored progress {state.get('progress_p', 0.0):.3f} and "
            f"{len(state.get('constraints', {}).get('scale', {}))} constraint multipliers from {path}")
    return infos
