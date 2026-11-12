"""The flat items the way the acceptance measures them (go2-depth 2026-11-12, the flat-measure alignment):
flat_measure, a reward term with zero output that runs the acceptance's own flat protocols on evaluation robots on
the flat column and keeps their readings per command bin; the constraints read them as "flat:<reading>"
(constraints.measure). teacher_env.py wires it in (_flat_tracker, the items in _flat_script_items);
U_FLAT_SCRIPTS=0 turns it off: the flat items as they were measured before."""

from __future__ import annotations

import collections
import math
import os
import random
import traceback
from typing import TYPE_CHECKING

import torch

from src.tasks.velocity.mdp.constraints import METRICS

from .constraints import HEAVY_FRAC, LIGHT_FRAC, payload_limit
from .curriculum import STEPS_PER_IT

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# A teacher that met its own flat constraints failed the acceptance's flat tables (Go2 ufull_s0 model_4000, scan on /
# off):
#   stop creep   training 0.56 / 0.47 cm (one mean over every stop)      acceptance: after 1.5 m/s 1.19 / 1.30 cm (bar 1)
#   gait         0.863 (the sideways commands only, one mean)            (-0.5, 0, +-0.2): 77-78% (bar 80%)
#   drift        0.051 (every robot >= 1 m into a 2-5 s command, p90)    10 s at 1.0 m/s: p10 13.4% of the distance,
#                                                                        heading p10 -11.9 deg (bars 5%, 8 deg)
#   speed        no item                                                 (0.6, +-0.4): y 67-73%; 1.5 m/s + turn: w 74-78%
#   stance       no item                                                 p90 0.158 / 0.149 rad (bar 0.15)
#   march hips   0.036-0.039 rad while marching (multiplier at its cap)  mean 1.5 deg after the stand that follows
# The acceptance runs fixed commands for fixed times, most of them from a standstill; the training's commands are
# random and last 2-5 s, so no formula over them is the acceptance's statistic. flat_measure therefore runs the
# acceptance's own protocols on evaluation robots on the flat column (they act on the mean action and are outside the
# update: the training data is as before) and keeps the newest samples of each reading per command bin and mode:
#
#   script   segments (env steps of 0.02 s)                    readings                         the acceptance's script
#   track    MOVE 325: one of accept_matrix.py's 44 commands,   speed_x / speed_y / speed_w     accept_matrix.py (8 s, the
#            the body velocity and the gait match averaged      (mean velocity / command, the   last 4 s), gait_clock.py
#            over the last 4 s (2.5 s to settle; the            axes commanded >= 0.05), gait   (8 s, the last 4 s)
#            acceptance gives 4 s)
#   stop     [REST 450: zero command from the reset] then       creep (m; REST: bin "rest"),    eval_standing.py (3 s command,
#            pairs GO 150 (one of its six commands) + STOP 300: stance (REST: the worst leg     the displacement 2 s -> 6 s
#            the position 101 steps after the stop is the       joint's largest offset 3-9 s),  after the stop; rest: 5 -> 9 s),
#            reference, the distance from it at 300 the creep   stance_stop (STOP: the leg      long_stand.py (60 s; here 9 s),
#                                                               joints' mean offset 3 s after   stop_then_move.py ("after stop")
#                                                               the stop), stop_dist (logged)
#   line     LINE 500 from the reset standstill: one of         drift (the signed offset across drift_all.py (10 s, p10 / p90 of
#            drift_all.py's straight / sideways commands        the line / the distance along   the offset against the median
#                                                               it), head (the heading change,  distance; heading median and
#                                                               deg); speed / gait over 4-8 s   p10 / p90)
#   march    STAND0 150, MARCH 250 (march flag, zero command),  march_hip (the worst hip's      march_eval.py's reset test (hips
#            STAND1 150                                         offset at the end of STAND1,    splayed 0.14 rad; mean over the
#                                                               rad), march_before, march_drift robots after the last stand)
# A script begins at a robot's reset (the first policy step of the episode still ran under the drawn command) and is
# given up -- no sample -- when the robot leaves the flat strip, falls, or something else rewrites its command; the
# counts are logged (Diag/flat_drop/<reason>). The robots under a script are not pushed (flat_no_push; the
# acceptance's flat tests have no pushes). The rest of the training randomization stays (friction, gains, the
# joints' reset noise, the payload up to LIGHT_FRAC of the limit): the readings are a little harder than the
# acceptance's play configuration, never easier by construction.
#
# The constraints read "flat:<reading>" (measure): the WORST bin's statistic over the bins with enough samples.

FLAT_SHARE = (0.7, 0.4)   # the shares of the light / heavy evaluation robots on the flat column under a script


def scripts_on() -> bool:
  """U_FLAT_SCRIPTS=0: no flat protocols -- the flat items measured on the drawn commands as before (an A / B probe)."""
  return os.environ.get("U_FLAT_SCRIPTS", "1") != "0"


FLAT_FAST = 1.2        # a forward command from here on is "fast" (the acceptance's 1.5 m/s)
FLAT_AXIS_MIN = 0.05   # an axis counts as commanded from here on (accept_matrix.py judges the axes with |c| >= 0.05)
F_MOVE, F_GO, F_STOP, F_REST, F_LINE, F_STAND0, F_MARCH, F_STAND1 = range(8)
# scripts/blind/eval/accept_matrix.py CMDS (the speed table; gait_clock.py's are the same 44)
FLAT_TRACK_CMDS = (
  (0.3, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0), (1.5, 0.0, 0.0), (-0.5, 0.0, 0.0), (-1.0, 0.0, 0.0),
  (0.0, 0.3, 0.0), (0.0, -0.3, 0.0), (0.0, 0.6, 0.0), (0.0, -0.6, 0.0), (0.6, 0.4, 0.0), (0.6, -0.4, 0.0),
  (0.0, 0.0, 0.2), (0.0, 0.0, -0.2), (0.0, 0.0, 0.5), (0.0, 0.0, -0.5), (0.0, 0.0, 1.0), (0.0, 0.0, -1.0),
  (0.0, 0.0, 1.5), (0.0, 0.0, -1.5),
  (0.5, 0.0, 0.15), (0.5, 0.0, -0.15), (1.0, 0.0, 0.15), (1.0, 0.0, -0.15), (1.0, 0.0, 0.2), (1.0, 0.0, -0.2),
  (1.5, 0.0, 0.15), (1.5, 0.0, -0.15), (1.5, 0.0, 0.2), (1.5, 0.0, -0.2), (1.5, 0.0, 0.3), (1.5, 0.0, -0.3),
  (-0.5, 0.0, 0.2), (-0.5, 0.0, -0.2), (-1.0, 0.0, 0.2), (-1.0, 0.0, -0.2), (0.8, 0.0, 0.8), (0.8, 0.0, -0.8),
  (0.3, 0.3, 0.2), (0.0, 0.4, -0.3), (0.15, 0.0, 0.3), (0.2, 0.0, 0.0), (0.0, 0.15, 0.0), (1.0, 0.0, 0.1),
)
# scripts/blind/eval/eval_standing.py GROUPS without "rest": one bin each (fwd, back, side, turn, arc, fast)
FLAT_STOP_CMDS = ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.6, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.3), (1.5, 0.0, 0.0))
# scripts/blind/eval/drift_all.py CMDS, the straight and the sideways ones
FLAT_LINE_CMDS = ((0.5, 0.0, 0.0), (1.0, 0.0, 0.0), (-0.5, 0.0, 0.0), (0.0, 0.3, 0.0), (0.0, -0.3, 0.0),
                  (0.0, 0.6, 0.0), (0.0, -0.6, 0.0))
FLAT_ZERO = (0.0, 0.0, 0.0)
# a constraint's reading over several stored ones (the linear axes share their cost terms)
FLAT_GROUPS = {"speed_lin": ("speed_x", "speed_y"), "speed_yaw": ("speed_w",)}
FLAT_LIGHT = ("seen", "blind", "all")   # the light robots' modes ("all": an environment without scan modes)
# the table and tags: (reading, statistic, the name it is shown under, whether its sample count is a tag too)
FLAT_SHOW = (("creep", "mean", "creep", True), ("stop_dist", "mean", "stop_dist", False),
             ("stance", "p90", "stance", True), ("stance_stop", "p90", "stance_stop", False),
             ("gait", "mean", "gait", True), ("speed_x", "mean", "speed_x", True), ("speed_y", "mean", "speed_y", True),
             ("speed_w", "mean", "speed_w", True), ("drift", "tail", "drift", True), ("head", "tail", "head", False),
             ("head", "absmed", "head_med", False), ("march_hip", "mean", "march_hip", True),
             ("march_before", "mean", "march_before", False), ("march_drift", "mean", "march_drift", False))
# the start headings the reset gives a share of the evaluation robots on the flat column (flat_reset_yaw): along the
# strip either way for the straight lines, across it for the sideways ones (the body's y axis along the strip)
FLAT_YAWS = ((0.0, 0.375), (math.pi, 0.375), (0.5 * math.pi, 0.125), (-0.5 * math.pi, 0.125))


def flat_bin(c) -> str:
  """The bin of a command (vx, vy, wz): the pre-stop command of a stop, the command of a speed / gait sample. The
  acceptance's failing commands have bins of their own (diag, fast_arc, back_arc)."""
  vx, vy, wz = (float(v) for v in c)
  x, y, w = abs(vx) >= FLAT_AXIS_MIN, abs(vy) >= FLAT_AXIS_MIN, abs(wz) >= FLAT_AXIS_MIN
  if not (x or y or w):
    return "rest"
  if x and not y and not w:
    return "fast" if vx >= FLAT_FAST else "fwd" if vx > 0.0 else "back"
  if y and not x and not w:
    return "side"
  if w and not x and not y:
    return "turn"
  if x and y and not w:
    return "diag"
  if x and w and not y:
    if vx >= FLAT_FAST:
      return "fast_arc"
    if vx > 0.0:
      return "arc_tight" if abs(wz) >= 0.5 else "arc"
    return "back_arc" if vx > -0.75 else "back_arc_fast"
  return "mix"


def flat_line_bin(c) -> str:
  """The bin of a 10 s line's command: forward at 1 m/s, forward slow, backward, sideways (both ways in one bin: the
  statistic takes both tails)."""
  vx, vy, _ = (float(v) for v in c)
  if abs(vy) >= FLAT_AXIS_MIN:
    return "side"
  return "fwd" if vx >= 0.75 else "fwd_slow" if vx > 0.0 else "back"


def flat_quantile_stat(stat: str) -> bool:
  """Whether a statistic is a tail quantile (it needs more samples than a mean or a median)."""
  return stat == "tail" or (stat.startswith("p") and stat[1:].replace(".", "", 1).isdigit())


def flat_stat(values, stat: str) -> float:
  """A statistic of a list of samples: "mean"; "dev1" = |mean - 1| (a speed ratio's distance from the command);
  "pNN" = that quantile (linear between the order statistics, as torch.quantile); "tail" = max(|p10|, |p90|) (the
  acceptance's drift / heading verdict takes both tails); "absmed" = |median|."""
  v = sorted(float(x) for x in values)
  n = len(v)

  def q(p: float) -> float:
    x = p * (n - 1)
    i = min(int(math.floor(x)), n - 1)
    j = min(i + 1, n - 1)
    return v[i] + (v[j] - v[i]) * (x - i)

  if stat == "mean":
    return sum(v) / n
  if stat == "dev1":
    return abs(sum(v) / n - 1.0)
  if stat == "tail":
    return max(abs(q(0.1)), abs(q(0.9)))
  if stat == "absmed":
    return abs(q(0.5))
  if flat_quantile_stat(stat):
    return q(float(stat[1:]) / 100.0)
  raise ValueError(f"flat_stat: unknown statistic {stat!r}")


def flat_path_fits(x: float, y: float, yaw: float, cmd, steps: int, dt: float, yc: float, room: float, x_lo: float,
                   x_hi: float, n: int = 12) -> bool:
  """Whether the path of command `cmd` (body frame) held for `steps` from pose (x, y, yaw) stays within `room` of the
  strip's centre line y = yc and inside x_lo .. x_hi (the command followed exactly: a constant twist's arc)."""
  vx, vy, wz = (float(v) for v in cmd)
  s0, c0 = math.sin(yaw), math.cos(yaw)
  for j in range(1, n + 1):
    t = steps * dt * j / n
    if abs(wz) < 1e-6:
      dx, dy = (vx * c0 - vy * s0) * t, (vx * s0 + vy * c0) * t
    else:
      a = yaw + wz * t
      s1, c1 = math.sin(a), math.cos(a)
      dx = (vx * (s1 - s0) + vy * (c1 - c0)) / wz
      dy = (vx * (c0 - c1) + vy * (s1 - s0)) / wz
    if abs(y + dy - yc) > room or not x_lo <= x + dx <= x_hi:
      return False
  return True


class flat_samples:
  """The newest `size` samples of one reading in one bin and mode, each with the env step it was taken at."""

  def __init__(self, size: int):
    self.v: collections.deque = collections.deque(maxlen=size)
    self.t: collections.deque = collections.deque(maxlen=size)
    self.n = 0   # samples ever

  def add(self, value: float, step: int) -> None:
    self.v.append(float(value))
    self.t.append(int(step))
    self.n += 1


def flat_recent(pairs: list, step: int, need: int, fresh: int, max_age: int) -> list | None:
  """The samples a statistic is taken over, from (step, value) pairs: those of the last `fresh` steps when there are
  `need` of them; else the newest `need`, as long as none of them is older than `max_age` (a rare bin still counts,
  on older samples); else None -- the bin does not count."""
  new = [v for t, v in pairs if step - t <= fresh]
  if len(new) >= need:
    return new
  if len(pairs) >= need:
    last = sorted(pairs, key=lambda p: p[0])[-need:]
    if step - last[0][0] <= max_age:
      return [v for _, v in last]
  return None


class flat_measure:
  """Reward term with zero output (a tiny nonzero weight so the manager calls it; last in the reward dict): the
  acceptance's flat protocols on evaluation robots on the flat column (the section's comment above). env.u_flat.

  Which robots: an evaluation robot (env.u_eval_mask; a run without them has no scripts and no readings) whose
  episode begins on a `flat_names` column, with a payload up to LIGHT_FRAC of the limit (`share[0]` of them; mode
  "seen" / "blind" by its scan mode -- the constraints' modes) or from HEAVY_FRAC (`share[1]`; mode "heavy": in the
  table and the tags only, never read by a constraint -- the heavy robots have no constraint). The others, and the
  payloads in between, keep the drawn commands: the items that still read those (back_drift, march, spin, the heavy
  log) need them, and leave out the robots under a script (constraints.off_scripts).

  A script's commands are the acceptance's, restricted to the command ranges of the moment (the ramps' 1.5 m/s and
  0.6 m/s sideways come late) and to the paths that stay on the strip from the robot's own pose (flat_path_fits; the
  column is 8 m wide, so a 10 s line needs a start heading along it: flat_reset_yaw). With probability `least_prob`
  the bin with the fewest finished samples for the robot's mode is taken, so every bin fills.

  Each sample goes to buf[(reading, mode, bin)] (flat_samples, the newest `size`). read() gives a constraint the
  worst bin's statistic over the bins that count: `min_n` samples for a mean or a median, `min_q` for a tail
  quantile, taken from the last `fresh_steps` env steps (200 iterations), or the newest min_n / min_q when the bin
  is slower than that (none older than `max_age`). table() is the per-bin view for the records (diagnostics:
  Diag/flat/...).

  state() / load(): the samples, tallies and drop counts travel in the constraints' checkpoint state
  (constraints.state()["flat"]; a checkpoint without it: no samples yet); the scripts in flight are not saved (a
  resume starts new episodes).

  A fault inside the term stops it for good (self.fault; the scripted robots go back to drawn commands, the flat:
  items go unmeasured and their multipliers hold; diagnostics raises flat_fault): the training runs on."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    p = dict(getattr(cfg, "params", None) or {})
    g = p.get
    self.command_name = g("command_name", "twist")
    self.hip_term = g("hip_term", "march_hip_default")     # its joints: the hips (march_hip, the plain / splayed start)
    self.pose_term = g("pose_term", "standing_pose_load")  # its joints: the leg joints of the stance readings
    self.gait_term = g("gait_term", "foot_gait")
    self.flat_names = tuple(g("flat_names", ("flat",)))
    self.share = tuple(float(x) for x in g("share", FLAT_SHARE))   # (light, heavy) robots under a script
    self.move_steps, self.move_from = int(g("move_steps", 325)), int(g("move_from", 125))
    self.go_steps, self.go_extra = int(g("go_steps", 150)), int(g("go_extra", 25))
    self.stop_steps, self.stop_ref = int(g("stop_steps", 300)), int(g("stop_ref", 101))
    self.stop_pose = int(g("stop_pose", 150))
    self.rest_steps, self.rest_ref = int(g("rest_steps", 450)), int(g("rest_ref", 251))
    self.rest_from = int(g("rest_from", 150))
    self.line_steps, self.line_min = int(g("line_steps", 500)), int(g("line_min", 150))
    self.line_track = tuple(int(x) for x in g("line_track", (200, 400)))
    self.stand_steps, self.march_steps = int(g("stand_steps", 150)), int(g("march_steps", 250))
    self.size, self.min_n, self.min_q = int(g("size", 48)), int(g("min_n", 8)), int(g("min_q", 12))
    self.fresh_steps, self.max_age = int(g("fresh_steps", 4800)), int(g("max_age", 24000))
    self.room, self.line_room, self.edge = float(g("room", 3.2)), float(g("line_room", 2.0)), float(g("edge", 0.5))
    self.border = float(g("border", 10.0))            # how far past the grid's ends a path may run (the flat border)
    self.line_prob = float(g("line_prob", 0.65))      # a plain start: the 10 s line, when one fits
    self.rest_prob = float(g("rest_prob", 0.3))       # a plain start's stop script begins with the 9 s rest
    # the other scripts' weights, (plain start, splayed start): the stops and the line begin at the default stance
    # as the acceptance's do; a splayed start is the march test's (its hips' reset)
    self.weights = {"stop": (3.0, 0.0), "track": (1.0, 1.0), "march": (0.5, 2.0)} | dict(g("weights", {}))
    self.least_prob = float(g("least_prob", 0.5))
    self.plain_max = float(g("plain_max", 0.105))     # the worst hip within the reset's joint noise: a plain start
    self.march_min = float(g("march_min", 0.08))      # the worst hip at least this far out: the march script may run
    self.show_n = int(g("show_n", 4))                 # table(): a bin is shown from this many samples
    self.rng = random.Random(int(g("seed", 0)))
    n, dev = env.num_envs, env.device
    zeros = lambda *shape, dtype=torch.float: torch.zeros(n, *shape, dtype=dtype, device=dev)   # noqa: E731
    self.fresh = zeros(dtype=torch.bool)    # the episode began since the last call
    self.on = zeros(dtype=torch.bool)       # under a script
    self.kind = zeros(dtype=torch.long)     # the segment (F_*)
    self.k = zeros(dtype=torch.long)        # policy steps the segment's command has been in force
    self.length = zeros(dtype=torch.long)
    self.cmd = zeros(3)
    self.ref0, self.ref_yaw = zeros(2), zeros()   # position and heading when the segment began
    self.ref = zeros(2)                     # the creep's reference position
    self.acc, self.acc_n = zeros(4), zeros()      # sums over the window: body vx, vy, wz, gait match; its steps
    self.peak, self.aux = zeros(), zeros()        # REST: the stance's running maximum; STOP: the stance 3 s in
    self.info: dict[int, dict] = {}         # robot -> {"script", "mode", "plain", "pre"} while under a script
    self.buf: dict[tuple, flat_samples] = {}   # (reading, mode, bin) -> samples
    self.tally: dict[tuple, int] = {}       # (script group, mode, bin) -> finished samples (the least-first choice)
    self.drops: dict[str, int] = {}         # reason -> scripts given up
    self.version = 0                        # samples ever (read()'s cache)
    self.step = 0
    self.fault: str | None = None
    self._rebase: int | None = None
    self._cache: dict = {}
    self._geom = None
    self._cols = None
    self._joints = None
    self._gait_rule = None
    env.u_flat = self

  # -- the environment's side (each one a method: the unit checks replace them with a kinematic stand-in) ----------

  def _kin(self, env: ManagerBasedRlEnv) -> dict:
    """Position (N, 2), heading, body velocity (vx, vy, wz) and the joints' offsets: the worst hip (`hip_term`'s
    joints), the leg joints' maximum and mean (`pose_term`'s); None for a term the environment does not have."""
    if self._joints is None:
      rm = env.reward_manager
      ids = lambda t: rm.get_term_cfg(t).params["asset_cfg"].joint_ids if t in rm._term_names else None   # noqa: E731
      self._joints = (ids(self.hip_term), ids(self.pose_term))
    data = env.scene["robot"].data
    off = (data.joint_pos - data.default_joint_pos).abs()
    hip, leg = self._joints
    return {"pos": data.root_link_pos_w[:, :2], "yaw": data.heading_w,
            "vel": torch.cat([data.root_link_lin_vel_b[:, :2], data.root_link_ang_vel_b[:, 2:3]], dim=1),
            "hip": None if hip is None else off[:, hip].amax(dim=1),
            "leg_max": None if leg is None else off[:, leg].amax(dim=1),
            "leg_mean": None if leg is None else off[:, leg].mean(dim=1)}

  def _flat_mask(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    """The robots whose episode is on a flat column."""
    from src.tasks.velocity.mdp.terrain_utils import spawned_on_sub_terrain, sub_terrain_column_mask
    if self._cols is None:
      self._cols = sub_terrain_column_mask(env, self.flat_names)
      if self._cols is None:   # (no terrain curriculum: no flat column to tell)
        self._cols = False
    if self._cols is False:
      return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return spawned_on_sub_terrain(env, self.flat_names, self._cols)

  def _bounds(self, env: ManagerBasedRlEnv) -> tuple[float, float, float]:
    """(a column's half width, x_lo, x_hi): the strip runs along world x through every row; `border` of the flat
    ground around the grid is usable past its ends."""
    if self._geom is None:
      terrain = env.scene.terrain
      size = terrain.cfg.terrain_generator.size
      x = terrain.terrain_origins[..., 0]
      self._geom = (0.5 * float(size[1]), float(x.min()) - 0.5 * float(size[0]) - self.border,
                    float(x.max()) + 0.5 * float(size[0]) + self.border)
    return self._geom

  def _centre(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    """Each robot's strip centre line (world y of its column)."""
    return env.scene.env_origins[:, 1]

  def _gait_value(self, env: ManagerBasedRlEnv) -> torch.Tensor | None:
    """The share of feet in step with the gait clock this step (mdp.constraints' gait_match, the acceptance's
    gait_clock.py); None without the gait term."""
    if self.gait_term not in env.reward_manager._term_names:
      return None
    return METRICS["gait_match"](env, gait_term=self.gait_term)[0]

  def _gait_counts(self, env: ManagerBasedRlEnv, cmd) -> bool:
    """Whether the acceptance judges the gait under this command (gait_clock.py:101-113): |vx| <= 1 m/s, and for a
    robot that steps only under sideways / turning commands (stepping_only: the wheeled ones) one of those."""
    if self._gait_rule is None:
      rm = env.reward_manager
      p = rm.get_term_cfg(self.gait_term).params if self.gait_term in rm._term_names else None
      self._gait_rule = (False,) if p is None else (
        True, bool(p.get("stepping_only", False)), float(p.get("max_lin_vel_y", 0.1)),
        float(p.get("max_ang_vel_z", 0.1)))
    if not self._gait_rule[0]:
      return False
    _, stepping_only, max_vy, max_wz = self._gait_rule
    return abs(cmd[0]) <= 1.0 and (not stepping_only or abs(cmd[1]) > max_vy or abs(cmd[2]) > max_wz)

  # -- the samples ----------------------------------------------------------------------------------------------

  def _add(self, reading: str, mode: str, bin_: str, value: float) -> None:
    if value != value or abs(value) == float("inf"):
      return
    key = (reading, mode, bin_)
    buf = self.buf.get(key)
    if buf is None:
      buf = self.buf[key] = flat_samples(self.size)
    buf.add(value, self.step)
    self.version += 1

  def _count(self, group: str, mode: str, bin_: str) -> None:
    self.tally[(group, mode, bin_)] = self.tally.get((group, mode, bin_), 0) + 1

  def _drop(self, reason: str) -> None:
    self.drops[reason] = self.drops.get(reason, 0) + 1

  def _samples(self, reading: str, bin_: str, modes, need: int) -> list | None:
    pairs = []
    for m in modes:
      buf = self.buf.get((reading, m, bin_))
      if buf is not None:
        pairs.extend(zip(buf.t, buf.v))
    return flat_recent(pairs, self.step, need, self.fresh_steps, self.max_age) if pairs else None

  def read(self, name: str, stat: str = "mean", scan: str | None = None, bins=None, worst: str = "max") -> float | None:
    """A constraint's reading: the worst (`worst`: "max", or "min" for a value that must stay high) statistic `stat`
    over the bins (all, or `bins`) of reading `name` (or FLAT_GROUPS[name]'s) that count, on the light robots in
    scan mode `scan` (None: all of them). None when no bin counts or after a fault."""
    if self.fault is not None:
      return None
    # (the constraints ask at every env step: one answer per iteration)
    tick = self.step // STEPS_PER_IT
    if self._cache.get("tick") != tick:
      self._cache = {"tick": tick}
    key = (name, stat, scan, None if bins is None else tuple(bins), worst)
    if key in self._cache:
      return self._cache[key]
    modes = FLAT_LIGHT if scan is None else (scan,)
    need = self.min_q if flat_quantile_stat(stat) else self.min_n
    readings = FLAT_GROUPS.get(name, (name,))
    out = None
    for reading, b in sorted({(r, b) for r, _, b in self.buf if r in readings and (bins is None or b in bins)}):
      v = self._samples(reading, b, modes, need)
      if v is None:
        continue
      s = flat_stat(v, stat)
      out = s if out is None else min(out, s) if worst == "min" else max(out, s)
    self._cache[key] = out
    return out

  def table(self) -> dict:
    """The per-bin view for the records: {"bins": {shown name: {"<bin>_<mode>": {"v", "n" (samples behind v), "total"
    (samples ever)}}}, "n_tags": the shown names whose counts are tags, "drops", "done", "scripted", "fault"}."""
    bins: dict[str, dict] = {}
    for reading, stat, shown, _ in FLAT_SHOW:
      for (r, mode, b), buf in sorted(self.buf.items()):
        if r != reading:
          continue
        v = flat_recent(list(zip(buf.t, buf.v)), self.step, min(self.show_n, self.min_n), self.fresh_steps,
                        self.max_age)
        if v is not None:
          bins.setdefault(shown, {})[f"{b}_{mode}"] = {"v": flat_stat(v, stat), "n": len(v), "total": buf.n}
    return {"bins": bins, "n_tags": [shown for _, _, shown, n in FLAT_SHOW if n], "drops": dict(self.drops),
            "done": {f"{g}/{b}_{m}": n for (g, m, b), n in sorted(self.tally.items())},
            "scripted": int(self.on.sum()), "fault": self.fault}

  def state(self) -> dict:
    return {"step": int(self.step),
            "buf": {"|".join(k): {"v": list(b.v), "t": list(b.t), "n": int(b.n)} for k, b in self.buf.items()},
            "tally": {"|".join(k): int(n) for k, n in self.tally.items()}, "drops": dict(self.drops)}

  def load(self, state: dict) -> None:
    """A checkpoint's samples (constraints.load). The stamps are env steps: the runner restores the step counter, so
    they stay as they are; were it not restored, the first call shifts them to the counter it finds."""
    self.buf, self.tally = {}, {}
    for key, d in (state.get("buf") or {}).items():
      buf = self.buf[tuple(key.split("|"))] = flat_samples(self.size)
      for v, t in zip(d["v"], d["t"]):
        buf.add(v, t)
      buf.n = int(d.get("n", buf.n))
    for key, n in (state.get("tally") or {}).items():
      self.tally[tuple(key.split("|"))] = int(n)
    self.drops = {k: int(v) for k, v in (state.get("drops") or {}).items()}
    self.step = int(state.get("step", 0))
    self._rebase = self.step
    self.version += 1
    self._cache.clear()

  # -- the scripts ----------------------------------------------------------------------------------------------

  def reset(self, env_ids=None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.fresh[env_ids] = True

  def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
    if self.fault is None:
      try:
        self._step(env)
      except Exception:
        self.fault = traceback.format_exc()
        print("[unified] flat_measure stopped on a fault (the flat: items go unmeasured, the training runs on):\n"
              + self.fault, flush=True)
        try:
          self._release(env.command_manager.get_term(self.command_name), self.on.clone())
        except Exception:
          pass
        self.on[:] = False
        self.info.clear()
        cons = getattr(env, "u_constraints", None)
        if cons is not None and hasattr(cons, "forget"):
          cons.forget("flat:")   # the multipliers hold where they are instead of following a reading that stopped
    self.fresh[:] = False
    return torch.zeros(env.num_envs, device=env.device)

  def _release(self, term, ids: torch.Tensor) -> None:
    """Hand the robots in `ids` (mask) back to the command term: it draws a command at its next update."""
    if bool(ids.any()):
      term.time_left[ids] = 0.0
      if getattr(term, "march", None) is not None:
        term.march[ids] = False

  def _hold(self, term) -> None:
    """Write the scripted robots' command and keep the term from changing it: no resample, no sign flip of a turn,
    no standing / heading / wall / corridor rewrite (velocity_command.py's _update_command)."""
    on = self.on
    if not bool(on.any()):
      return
    term.vel_command_b[on] = self.cmd[on]
    term.time_left[on] = 1.0
    if getattr(term, "march", None) is not None:
      term.march[on] = self.kind[on] == F_MARCH
    for name, value in (("_flip_left", float("inf")), ("is_standing_env", False), ("is_heading_env", False),
                        ("_wall_cut", False), ("corridor_hold", False)):
      t = getattr(term, name, None)
      if isinstance(t, torch.Tensor):
        t[on] = value
    vx = getattr(term, "_vx_raw", None)
    if isinstance(vx, torch.Tensor):
      vx[on] = self.cmd[on, 0]

  def _ranges(self, term) -> tuple:
    r = term.cfg.ranges
    return (tuple(r.lin_vel_x), tuple(r.lin_vel_y), tuple(r.ang_vel_z))

  def _pick(self, r: dict, mode: str, group: str, kind: int, cmds, binner, steps: int, room: float,
            need: int, extra: int = 0):
    """A segment (kind, command, steps) for the robot in row r: a command of `cmds` inside the command ranges whose
    path over steps + extra stays on the strip, the least-sampled bin first with probability least_prob; None when
    `need` steps do not fit in the episode or no command fits."""
    if need + 2 > r["rem"]:
      return None
    fits: dict[str, list] = {}
    for c in cmds:
      if any(c[a] != 0.0 and not r["ranges"][a][0] - 1e-6 <= c[a] <= r["ranges"][a][1] + 1e-6 for a in range(3)):
        continue
      if flat_path_fits(r["x"], r["y"], r["yaw"], c, steps + extra, r["dt"], r["yc"], room, r["x_lo"], r["x_hi"]):
        fits.setdefault(binner(c), []).append(c)
    if not fits:
      return None
    bins = sorted(fits)
    if self.rng.random() < self.least_prob:
      low = min(self.tally.get((group, mode, b), 0) for b in bins)
      bins = [b for b in bins if self.tally.get((group, mode, b), 0) == low]
    return (kind, self.rng.choice(fits[self.rng.choice(bins)]), steps)

  def _pick_move(self, r: dict, mode: str):
    return self._pick(r, mode, "track", F_MOVE, FLAT_TRACK_CMDS, flat_bin, self.move_steps, self.room, self.move_steps)

  def _pick_go(self, r: dict, mode: str):
    return self._pick(r, mode, "stop", F_GO, FLAT_STOP_CMDS, flat_bin, self.go_steps, self.room,
                      self.go_steps + self.stop_steps, extra=self.go_extra)

  def _begin(self, i: int, r: dict):
    """The script of a robot whose episode just began on the flat column: its first segment, or None (not chosen,
    a payload between light and heavy, nothing fits)."""
    load = r["load"]
    if load != load or load <= LIGHT_FRAC:
      light, share = True, self.share[0]
    elif load >= HEAVY_FRAC:
      light, share = False, self.share[1]
    else:
      return None
    if self.rng.random() >= share:
      return None
    scan = r["scan"]
    mode = "heavy" if not light else "all" if scan != scan else "seen" if scan > 0.5 else "blind"
    hip = r["hip"]
    plain = not hip > self.plain_max   # (no hip reading: plain)
    col = 0 if plain else 1
    info = {"script": "", "mode": mode, "plain": plain, "pre": None}
    seg = None
    if plain and self.rng.random() < self.line_prob:
      seg = self._pick(r, mode, "line", F_LINE, FLAT_LINE_CMDS, flat_line_bin, self.line_steps, self.line_room,
                       self.line_steps)
      info["script"] = "line"
    if seg is None:
      can_march = (r["can_march"] and hip >= self.march_min
                   and 2 * self.stand_steps + self.march_steps + 2 <= r["rem"])
      opts = [(s, w[col]) for s, w in sorted(self.weights.items()) if w[col] > 0.0 and (s != "march" or can_march)]
      if not opts:
        return None
      script = self.rng.choices([s for s, _ in opts], weights=[w for _, w in opts])[0]
      if script == "march":
        seg = (F_STAND0, FLAT_ZERO, self.stand_steps)
      elif script == "stop":
        if plain and self.rng.random() < self.rest_prob and self.rest_steps + 2 <= r["rem"]:
          seg = (F_REST, FLAT_ZERO, self.rest_steps)
        else:
          seg = self._pick_go(r, mode)
      if seg is None:
        script, seg = "track", self._pick_move(r, mode)
      info["script"] = script
    if seg is None:
      return None
    self.info[i] = info
    return seg

  def _after(self, i: int, r: dict):
    """The segment after the one robot i just finished, or None (the script is over: nothing more fits)."""
    info = self.info[i]
    kind = r["kind"]
    if kind == F_STAND0:
      return (F_MARCH, FLAT_ZERO, self.march_steps)
    if kind == F_MARCH:
      return (F_STAND1, FLAT_ZERO, self.stand_steps)
    if kind == F_GO:
      return (F_STOP, FLAT_ZERO, self.stop_steps)
    if info["script"] == "stop":
      seg = self._pick_go(r, info["mode"])
      if seg is not None:
        return seg
    return self._pick_move(r, info["mode"])

  def _emit_track(self, env, i: int, r: dict, window: int) -> None:
    """The speed ratios and the gait match of a full averaging window under r's command."""
    if r["acc_n"] < window - 0.5:
      return
    mode, cmd = self.info[i]["mode"], r["cmd"]
    b = flat_bin(cmd)
    for a, name in enumerate(("speed_x", "speed_y", "speed_w")):
      if abs(cmd[a]) >= FLAT_AXIS_MIN:
        self._add(name, mode, b, r["acc"][a] / r["acc_n"] / cmd[a])
    if r["gait"] and self._gait_counts(env, cmd):
      self._add("gait", mode, b, r["acc"][3] / r["acc_n"])

  def _emit_line(self, env, i: int, r: dict, whole: bool) -> None:
    """The line's drift and heading change in the frame of the start heading (drift_all.py:99-110, 320-331 of
    accept_tables.py): a straight command's offset is across the heading and its distance along it, a sideways
    command's the other way round."""
    mode, cmd = self.info[i]["mode"], r["cmd"]
    dx, dy = r["x"] - r["ref0"][0], r["y"] - r["ref0"][1]
    c0, s0 = math.cos(r["ref_yaw"]), math.sin(r["ref_yaw"])
    along, across = c0 * dx + s0 * dy, -s0 * dx + c0 * dy
    off, travel = (along, abs(across)) if abs(cmd[1]) >= FLAT_AXIS_MIN else (across, abs(along))
    b = flat_line_bin(cmd)
    self._add("drift", mode, b, off / max(travel, 0.1))
    d = r["yaw"] - r["ref_yaw"]
    self._add("head", mode, b, math.degrees(math.atan2(math.sin(d), math.cos(d))))
    self._count("line", mode, b)
    if whole:
      self._emit_track(env, i, r, self.line_track[1] - self.line_track[0])

  def _emit(self, env, i: int, r: dict) -> None:
    """The samples of the segment robot i just finished."""
    info, kind = self.info[i], r["kind"]
    mode = info["mode"]
    dist = lambda p: math.hypot(r["x"] - p[0], r["y"] - p[1])   # noqa: E731
    if kind == F_MOVE:
      self._emit_track(env, i, r, self.move_steps - self.move_from)
      self._count("track", mode, flat_bin(r["cmd"]))
    elif kind == F_STOP:
      b = flat_bin(info["pre"])
      self._add("creep", mode, b, dist(r["ref"]))
      self._add("stop_dist", mode, b, dist(r["ref0"]))
      if r["legs"] and info["plain"]:
        self._add("stance_stop", mode, b, r["aux"])
      self._count("stop", mode, b)
    elif kind == F_REST:
      self._add("creep", mode, "rest", dist(r["ref"]))
      if r["legs"]:
        self._add("stance", mode, "rest", r["peak"])
      self._count("stop", mode, "rest")
    elif kind == F_LINE:
      self._emit_line(env, i, r, True)
    elif kind == F_STAND0:
      self._add("march_before", mode, "all", r["hip"])
    elif kind == F_MARCH:
      self._add("march_drift", mode, "all", dist(r["ref0"]))
    elif kind == F_STAND1:
      self._add("march_hip", mode, "all", r["hip"])
      self._count("march", mode, "all")

  def _step(self, env: ManagerBasedRlEnv) -> None:
    ev = getattr(env, "u_eval_mask", None)
    if ev is None or max(self.share) <= 0.0:
      return
    term = env.command_manager.get_term(self.command_name)
    step = int(env.common_step_counter)
    if self._rebase is not None:   # the first call after a load: were the step counter not restored, shift the stamps
      shift = step - self._rebase
      if not 0 <= shift <= self.fresh_steps:
        for buf in self.buf.values():
          buf.t = collections.deque((t + shift for t in buf.t), maxlen=self.size)
      self._rebase = None
    self.step = step
    kin = self._kin(env)
    pos, yaw = kin["pos"], kin["yaw"]
    kind, half = self.kind, self._bounds(env)
    centre = self._centre(env)
    # a robot under a script whose episode ended: it fell (a script never runs into the episode's end)
    lost = self.on & self.fresh
    live = self.on & ~lost
    self.k += live.long()
    k = self.k
    inside = ((pos[:, 1] - centre).abs() <= half[0] - self.edge) & (pos[:, 0] >= half[1]) & (pos[:, 0] <= half[2])
    held = (term.vel_command_b[:, :3] - self.cmd).abs().amax(dim=1) < 1e-4
    if getattr(term, "march", None) is not None:
      held &= term.march == (kind == F_MARCH)
    out, changed = live & ~inside, live & inside & ~held
    ok = live & inside & held
    # the averaging windows: MOVE's last 4 s, the LINE's 4 s -> 8 s
    win = ok & (((kind == F_MOVE) & (k > self.move_from))
                | ((kind == F_LINE) & (k > self.line_track[0]) & (k <= self.line_track[1])))
    w = win.float()
    self.acc[:, :3] += w.unsqueeze(1) * kin["vel"]
    gait = self._gait_value(env)
    if gait is not None:
      self.acc[:, 3] += w * gait
    self.acc_n += w
    # the creep's reference: 101 steps after the stop / 251 into the rest (eval_standing.py:135-136)
    take = ok & (((kind == F_STOP) & (k == self.stop_ref)) | ((kind == F_REST) & (k == self.rest_ref)))
    self.ref = torch.where(take.unsqueeze(1), pos, self.ref)
    if kin["leg_max"] is not None:
      m = ok & (kind == F_REST) & (k >= self.rest_from)
      self.peak = torch.where(m, torch.maximum(self.peak, kin["leg_max"]), self.peak)
      self.aux = torch.where(ok & (kind == F_STOP) & (k == self.stop_pose), kin["leg_mean"], self.aux)
    done = ok & (k >= self.length)
    cand = self.fresh & ev & self._flat_mask(env)
    self.on &= ~lost
    work = done | out | changed | lost | cand
    idx = work.nonzero(as_tuple=False).flatten()
    if idx.numel() == 0:
      self._hold(term)
      return
    nan = torch.full((env.num_envs,), float("nan"), device=env.device)
    kg = getattr(env, "payload_kg", None)
    scan = getattr(env, "u_scan_on", None)
    hip = kin["hip"]
    cols = [pos[:, 0], pos[:, 1], yaw, k.float(), kind.float(), done.float(), out.float(), changed.float(),
            lost.float(), cand.float(), self.acc_n, self.peak, self.aux, self.ref_yaw,
            nan if hip is None else hip, nan if kg is None else kg.float() / payload_limit(env),
            nan if scan is None else scan.float(), (env.max_episode_length - env.episode_length_buf).float(), centre,
            self.acc, self.cmd, self.ref, self.ref0]
    rows = torch.cat([c.reshape(env.num_envs, -1) for c in cols], dim=1)[idx].tolist()
    shared = {"dt": float(env.step_dt), "x_lo": half[1], "x_hi": half[2], "ranges": self._ranges(term),
              "legs": kin["leg_max"] is not None, "gait": gait is not None,
              "can_march": hip is not None and getattr(term, "march", None) is not None}
    start, stop = [], []
    for i, v in zip(idx.tolist(), rows):
      r = {**shared, "x": v[0], "y": v[1], "yaw": v[2], "k": int(v[3]), "kind": int(v[4]), "acc_n": v[10],
           "peak": v[11], "aux": v[12], "ref_yaw": v[13], "hip": v[14], "load": v[15], "scan": v[16], "rem": int(v[17]),
           "yc": v[18], "acc": v[19:23], "cmd": tuple(v[23:26]), "ref": v[26:28], "ref0": v[28:30]}
      seg = None
      if v[8]:      # fell under a script
        self._drop("fell")
        self.info.pop(i, None)
      elif v[6]:    # left the strip: a line that had run long enough still gives its drift (a lower bound)
        if r["kind"] == F_LINE and r["k"] >= self.line_min:
          self._emit_line(env, i, r, False)
        self._drop("strip")
        self.info.pop(i, None)
        stop.append(i)
      elif v[7]:    # something else rewrote the command
        self._drop("cmd")
        self.info.pop(i, None)
        stop.append(i)
      elif v[5]:
        self._emit(env, i, r)
        seg = self._after(i, r)
        if seg is None:
          self.info.pop(i, None)
          stop.append(i)
      if v[9]:      # a new episode on the flat column
        seg = self._begin(i, r)
      if seg is not None:
        if seg[0] == F_GO:
          self.info[i]["pre"] = seg[1]
        start.append((i, seg))
    dev = env.device
    if stop:
      ids = torch.zeros_like(self.on)
      ids[torch.tensor(stop, device=dev, dtype=torch.long)] = True
      self.on &= ~ids
      self._release(term, ids)
    if start:
      ids = torch.tensor([i for i, _ in start], device=dev, dtype=torch.long)
      self.kind[ids] = torch.tensor([s[0] for _, s in start], device=dev, dtype=torch.long)
      self.cmd[ids] = torch.tensor([s[1] for _, s in start], device=dev, dtype=self.cmd.dtype)
      self.length[ids] = torch.tensor([s[2] for _, s in start], device=dev, dtype=torch.long)
      self.k[ids] = 0
      self.acc[ids] = 0.0
      self.acc_n[ids] = 0.0
      self.peak[ids] = 0.0
      self.aux[ids] = 0.0
      self.ref0[ids] = pos[ids]
      self.ref[ids] = pos[ids]
      self.ref_yaw[ids] = yaw[ids]
      self.on[ids] = True
    self._hold(term)


def flat_no_push(env: ManagerBasedRlEnv, env_ids, flat_inner, **params):
  """Event wrapper (the push event): the robots under a flat script are left out -- the acceptance's flat tests
  have no pushes. Without the tracker, or after its fault, the event is as it was."""
  t = getattr(env, "u_flat", None)
  if t is not None and t.fault is None and env_ids is not None and bool(t.on.any()):
    if not isinstance(env_ids, torch.Tensor):
      env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    env_ids = env_ids[~t.on[env_ids.long()]]
    if env_ids.numel() == 0:
      return None
  return flat_inner(env, env_ids, **params)


def flat_reset_yaw(env: ManagerBasedRlEnv, env_ids, flat_inner, flat_share: float = 0.75, flat_spread: float = 0.08,
                   **params):
  """Event wrapper (the base's reset): after the reset as it was, `flat_share` of the evaluation robots on the flat
  column are placed again with a start heading within `flat_spread` of one of FLAT_YAWS -- along the strip or
  across it. The column is 8 m wide: a 10 s line at 1 m/s (drift_all.py) or 6.5 s at 1.5 m/s only stays on it from
  such a heading; from a uniform one about one start in ten would. The other robots, and every robot without the
  tracker / evaluation robots / scripts, start as before."""
  flat_inner(env, env_ids, **params)
  t = getattr(env, "u_flat", None)
  ev = getattr(env, "u_eval_mask", None)
  if t is None or t.fault is not None or ev is None or flat_share <= 0.0 or max(t.share) <= 0.0:
    return
  ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
  if not isinstance(ids, torch.Tensor):
    ids = torch.arange(env.num_envs, device=env.device)[ids]
  at = ids.long()
  pick = ev[at] & t._flat_mask(env)[at] & (torch.rand(at.numel(), device=env.device) < flat_share)
  ids = ids[pick]
  if ids.numel() == 0:
    return
  group = torch.multinomial(torch.tensor([w for _, w in FLAT_YAWS], device=env.device), ids.numel(), replacement=True)
  for j, (yaw, _) in enumerate(FLAT_YAWS):
    sel = ids[group == j]
    if sel.numel():
      pose = {**params["pose_range"], "yaw": (yaw - flat_spread, yaw + flat_spread)}
      flat_inner(env, sel, **{**params, "pose_range": pose})
