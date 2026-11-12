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
(the mean action's returns are not the sampled policy's). In the teacher's mixed scan modes they are a fixed half
seen and half blind (teacher_env.u_scan_mask).

Which code trained which iterations (2026-11-12, the from-zero retrain: the runs before it changed code under way,
and what a checkpoint had been trained on was only in the notes): every checkpoint carries
infos["unified"]["segments"], a list of {"from", "to", "commit"} -- the iterations and the code version
(code_version()) of each stretch of the run; a resume carries the list on and opens a new stretch when the code
differs.

Training records (2026-11-12, diagnostics.py): the runner attaches the recorder that writes <run>/records/ -- the
timeline of what happened, periodic snapshots of the curriculum, constraints, optimization and physics, every finished
episode counted per ground, and the alerts; scripts/unified/run_report.py reads them.

The recipe's gated clock (2026-11-12, mdp.staged_clock: Go2's early ramps wait until the robots walk) is saved too:
it was not in the checkpoint, so a resume before iteration 1000 kept a clock at 0 (the ramps started over) and a
later one started it at the step counter (the ramps jumped ahead by the time the clock had waited)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import torch

from src.tasks.velocity.rl import MirrorVelocityOnPolicyRunner, VelocityOnPolicyRunner

from . import diagnostics
from .constraints import eval_count


def code_version() -> str:
  """The code this process runs: the tree's REPO_COMMIT file (the boxes' trees, written by the deploy), else git's
  HEAD with "-dirty" when tracked files differ from it, else "unknown"."""
  root = Path(__file__).resolve().parents[3]
  f = root / "REPO_COMMIT"
  try:
    if f.is_file() and f.read_text().split():
      return f.read_text().split()[0]
    git = lambda *a: subprocess.run(("git", "-C", str(root)) + a, capture_output=True, text=True,   # noqa: E731
                                    timeout=20).stdout.strip()
    head = git("rev-parse", "--short", "HEAD")
    if head:
      return head + ("-dirty" if git("status", "--porcelain", "--untracked-files=no") else "")
  except Exception:
    pass
  return "unknown"


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
    self._commit = code_version()
    self._segments: list[dict] = []   # the stretches before this process's (from the checkpoint it resumed)
    self._segment = {"from": 0, "to": 0, "commit": self._commit}
    print(f"[unified] code {self._commit}")
    self._split_eval(eval_count(env.num_envs))
    diagnostics.attach(self)   # the training records (after the evaluation split: its hooks wrap what is there)

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
    # their scan modes, fixed by index from here on (the reset event), for the episodes already running too
    try:
      params = e.event_manager.get_term_cfg("u_scan_mask").params
    except ValueError:
      params = None
    if params is not None:
      from .teacher_env import u_scan_mask

      u_scan_mask(e, torch.arange(k, device=e.device), **params)
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
    clock = getattr(env, "_staged_clock", None)
    if clock is not None:
      state["staged_clock"] = {"value": float(clock.value), "walking": bool(clock.walking),
                               "fall_rate": float(clock.fall_rate),
                               "track_ratio": None if clock.track_ratio is None else float(clock.track_ratio)}
    self._segment["to"] = self.current_learning_iteration
    state["segments"] = [dict(x) for x in self._segments + [self._segment]]
    super().save(path, {**(infos or {}), "unified": state})

  def load(self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None):
    infos = super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)
    state = (infos or {}).get("unified")
    it = self.current_learning_iteration
    past = [dict(x) for x in ((state or {}).get("segments") or [])]
    if not past and it > 0:   # a checkpoint from before the list
      past = [{"from": 0, "to": it, "commit": "unknown"}]
    if past and past[-1]["commit"] == self._commit and past[-1]["to"] == it:
      self._segment = past.pop()   # the same code carries on
    else:
      self._segment = {"from": it, "to": it, "commit": self._commit}
    self._segments = past
    print("[unified] trained so far on: " + (", ".join(f"{x['from']}-{x['to']} {x['commit']}" for x in past) or "-")
          + f"; from {self._segment['from']} on {self._commit}")
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
      sc = state.get("staged_clock")
      if sc is not None:   # the step counter is restored by now: the clock carries on from its saved value
        from src.tasks.velocity.mdp.curriculums import _GatedClock
        clock = _GatedClock(env.common_step_counter)
        clock.value, clock.walking = sc["value"], bool(sc["walking"])
        clock.fall_rate, clock.track_ratio = sc["fall_rate"], sc["track_ratio"]
        env._staged_clock = clock
        print(f"[unified] gated clock restored: iteration {clock.value / 24.0:.0f}, walking {clock.walking}")
      print(f"[unified] restored progress {state.get('progress_p', 0.0):.3f} and "
            f"{len(state.get('constraints', {}).get('scale', {}))} constraint multipliers from {path}")
    return infos
