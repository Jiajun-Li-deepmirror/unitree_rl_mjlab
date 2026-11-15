"""The unified teacher's recipe, in code (go2-depth 2026-11-13).

Until now the recipe was a line of U_ switches on every probe's command line (teacher_env._probe_switches: each one
"off unless set"), repeated by hand for the gates, which took it from the run's timeline. The formal scripts
(jobs/unified/teacher.sh, stairs_gate.sh, flat_gate.sh, students.sh, accept.sh) carry no switches: a formal run from
them would have trained the plain task, not the recipe the probes arrived at. The user's order (2026-11-11): "配方由
代码定义，不从日志里提取 ... 一份共用基础配方，加上腿式和轮式各自的部分，再加少量每个机型自己的差异，每项差异写明原因".

So the recipe is this table -- COMMON, the robot's KIND, the few switches of one ROBOT -- and it is the default:
`scope(robot)` puts the robot's switches into the environment for the duration of a build
(teacher_env.unified_teacher_env_cfg: every task of the package, the students' too, on every terrain), the runner
reads its two through `runner_switches` (the entropy bonus, the std cap), and the gate reads the constraint items
under the same scope. The switches' code is untouched: what a probe's command line built, the clean environment
builds now (scripts/unified/recipe_test.py compares the two on every task's configuration).

A probe still works as before:
  - a switch set in the environment wins over the recipe's value (U_STD_CAP=0.34 on a command line: the recipe plus
    that one difference);
  - U_RECIPE=0 switches the table off: the environment's switches alone decide, as before this file (a run from
    before it is evaluated that way: its timeline's env_vars are its whole recipe).
A run's timeline says which it was: the start event's "recipe" (record(): the switches in effect, the ones the
environment set itself, "off" for U_RECIPE=0) beside the "env_vars" it always had.

Tested on Go2 and Go2W (the probes on t1 / t2) and As2 (go2-home's); the other robots (U_ROBOTS) take their kind's
part untested."""

from __future__ import annotations

import os
from contextlib import contextmanager

# every robot (each switch's account: teacher_env._probe_switches' docstring)
COMMON: dict[str, str] = {
  "U_Q_SCALE_MAX": "1",         # the stairs-quality items' multipliers stay at or below 1: the P weights as configured
  "U_TERMINATED_Q": "10",       # a fall costs 1x -> 10x along the stairs quality ("负载一定不能摔倒")
  "U_ENTROPY_Q": "0.002",       # the entropy bonus 0.005 -> 0.002 along the stairs quality (runner._entropy_by_q)
  "U_METER_EMA": "2400",        # the stairs meters' memory, env steps (20000 lagged a change of behaviour)
  "U_FLAT_ONLY": "1",           # a flat item's multiplier presses on flat ground only
  "U_PAYLOAD_STICKY": "0.05",   # the payload class stays with the robot, drawn anew at a reset with this chance
  "U_STAIR_SPEED_H": "1",       # the stairs speed over the ground, as the acceptance measures it
  "U_HEADING_OBS": "1",         # the line costs hold the heading the actor is shown ...
  "U_HEADING_LAG": "2",         # ... and the heading's lag behind the commands costs (weight -2)
  "U_STAIRS_P": "0.0,0.15",     # the stairs share ramps in over progress 0-0.15 (the task's own: 0.3-0.8). The Go2W
                                # from zero: progress 0.49 at 900 iterations against 0.08. Every robot's probes
                                # started with it (Go2 uprobe_s4, go2-home's As2 uprobe_s1); the Go2's formal run
                                # of 84e7965 without it: a stairs share of 3-7 % until 2000 iterations
}

KIND: dict[str, dict[str, str]] = {
  "legged": {
    "U_STOP_FEET_DIR": "3",       # four feet at a stop on a flight: one item per direction faced, both multipliers from 3
    "U_FLAT_DIAG_AXIS": "0.2",    # a diagonal command's flat speed per axis, as the acceptance judges it (the wheeled have it)
    "U_FLAT_SPEED_W": "4",        # the flat speed deficit at the weight -4 (the wheeled have it)
    "U_STAIR_PARK": "10",         # standing on a descending column under a forward command costs
    "U_BODY_CLEAR": "50",         # the bodies whose touch ends the episode keep clear of the steps
    "U_COM_SUPPORT": "15,0.5",    # on uneven ground the centre of mass, payload included, over the feet: in full
                                  # at a stop (Go2's loaded falls on tier 2: 11 -> 5), at half under a command
                                  # (the 8 kg falls on the 15-18 cm flights, 1152 episodes per scan mode: 5.30 /
                                  # 3.30 % -> 2.69 / 2.26 %, uprobe_s140 / s141). Not the full share under a command
                                  # (s153: 6 unloaded and 11 typical-payload falls walking blind, nose-up body
                                  # strikes on the climb) and not the weight 30 (s152: four feet at a 15-18 cm
                                  # descent stop 0.58)
    "U_ALIVE": "4",               # a living step pays +4 per second (the wheeled have it from their builder). The
                                  # Go2 from zero without it (the formal run of 84e7965): the flat costs above put
                                  # the first iterations' return below an ended episode's -4, and the policy ended
                                  # its episodes by a body contact -- for 2000 iterations on the climbing columns
    # A robot told to stop on a flight stays there (tier 2 asks 90 % of every band; the Go2's formal teachers of
    # 84e7965 walked on down the 15-18 cm flights: stayed 30 % at model_5200, at 1.35x the commanded speed). Three
    # switches, one recipe part (2026-11-14):
    "U_STAND_POSE_EVEN": "1",     # the standing pose cost acts on even ground only: on a flight it was the largest
                                  # cost of a stop, and it pulled the robot off the steps
    "U_DESCENT_SPEED": "10,1.1",  # going down a flight faster than 1.1x the command costs (weight 10)
    "U_STOP_HOLD_Q": "1",         # both come in along the stairs quality, the P weights' own ramp: before the robots
                                  # climb, the pose removal keeps them from climbing (go2-home's ZI / ZJ from a
                                  # from-zero model_1500: the climbing row 0.10 against 1.17 at 3000)
    "U_ITEM_FLOOR": "leg_contact:3,descent:2",   # two stairs-quality items start at a multiplier above 1:
                                  # leg_contact 3 -- with the stop-hold part the robots rested a calf on a step at a
                                  # descent stop (four feet down 0.5-0.7 on 12-18 cm; uprobe_s133: back above the
                                  # bars, the blind falls with it; 2 does not do it, 6 costs the climb);
                                  # descent 2 -- walking with the scan masked the descents fell to 0.78x of the
                                  # command under floor 3 alone (bar 0.8x; go2-home's ZRd2 -> ZRe2 from zero:
                                  # 0.91-0.96x, uprobe_s140 / s141 the same)
    # Evidence, the Go2 from the formal checkpoints: go2-home's pair ZK / ZC (1024 robots, 5000): stayed 99 % against
    # 58 % without a load and 96 % against 39 % with the full one, the descents at 0.80-0.85x the command, the full
    # payload's falls on 5-15 cm 7 against 23. The stop-place cost of the wheeled does not do it (3 / 8 / 20: the Go2
    # is not held). As2 (go2-home's mAJ, the pose switch alone, on throughout): the loaded falls at the stops 179 ->
    # 88; its line had the pose switch as the robot's own until this entry. Not the heavy class's fade
    # (U_STAND_POSE_EVEN=1,<heavy>; G11: out).
    # The flights walked straight and with the legs off the steps (the user on the Go2 teacher's videos, 2026-11-15:
    # "像是喝醉了走不了直线", "下楼时，主要关注后腿是否会有碰撞", "朝向最大偏离也要训练，门槛你定" -- the bar: the heading's
    # p90 within 8 deg on every band, up and down, from this version on). Two switches, one recipe part (2026-11-16):
    "U_DESCENT_LEG": "6",         # a thigh or calf pressing on a step while the robot walks down a flight costs
    "U_FLIGHT_HEADING": "5,0.07", # turning off the commanded heading along a flight costs ((the error / 0.07 rad)^2)
    # Evidence, the Go2 from the formal seed 0's model_5200, 600 iterations on, the newest 4 checkpoints averaged
    # (uprobe_s178; in brackets the formal run's own at the same iteration, averaged the same way): the stairs gate
    # passes (gap 0.12); on the 15-18 cm flights, without a load / with the typical one: the heading's p90 4.6 / 5.4
    # deg down (9.6 / 9.1) and 4.5 / 5.6 deg up (10.4 / 10.5), the front calves on the steps 4.1 / 5.5 times a
    # descent (7.8 / 11.3), the rear ones 2.7 / 2.8 (2.9 / 2.6); the full payload's falls on 5-15 cm 1 (10; the leg
    # cost alone, uprobe_s165: 0). The price: the descent's weave 8.0 / 7.8 cm peak to peak (6.7 / 5.7). The weave
    # cost in its place (U_FLIGHT_WEAVE=5, uprobe_s166: 5.0 cm) leaves the heading at 8.3 deg and puts the rear
    # calves on the steps 4.2 times a descent; the weave cost beside the leg cost (uprobe_s177): 7.2 cm, the heading
    # 6.6 / 7.3 deg. The user's order of the three: the rear legs first, the heading judged, the weave trained and not
    # judged. The As2 takes the part untested (go2-home's probes). Not the wheeled kind: the Go2W's heading cost at 5
    # leaves 8.4 deg, at 15 the robot slides back at the stops (uprobe_s164 / s170), and with the leg cost its gate
    # fails (uprobe_s174).
  },
  "wheeled": {
    "U_STAIR_APPROACH": "0",          # the stairs speed cost holds on a climbing column's level ground too (the Go2W
                                      # stood before the flight); not for the heavy class (the 0)
    "U_BODY_CLEAR": "50,0.06,0.10",   # as the legged, with the wheeled margin and reach (Go2W's loaded falls on tier
                                      # 2: 16 at the plain 50 -> 5)
    "U_STOP_PLACE": "12",             # a robot told to stop on a flight stays where it was told (within 0.30 m free,
                                      # in full from 0.60 m): the Go2W of 84e7965 rolled back off a climbing flight
                                      # after the stop command (stayed 35 % / 17 % on 12-15 / 15-18 cm at 3000
                                      # iterations; tier 2 asks 90 % of every band). The weight 12 from the formal
                                      # model_1500: 100 % / 99 % at 2000, 100 % / 100 % at 2500 (s70: the stop ok 99 %, no
                                      # fall); 8 held 96 % / 51 % and came and went, 5 did not hold. Not for the
                                      # legged: no weight (3 / 8 / 20) holds the Go2 on a 15-18 cm descent
    # The wheeled freeze (2026-11-15; the Go2W's probes from its formal model_3600, 600 iterations each, the last
    # one -- uprobe_s157 -- with the three together):
    "U_COM_SUPPORT": "15",            # the centre-of-mass cost while standing (no share under a command: with one
                                      # the Go2W lost its steep ascent stop, s151): the full payload's falls on tier
                                      # 2 28 -> 4-7 (s134 / s146 / s157), and with the tilt form below the stop on a
                                      # 15-18 cm ascent stays (s150 without it: 64 % with the scan masked; s157 99 %)
    "U_REST_OLD": "tilt,0.20",        # the stand by the base's tilt (rest.py): the lock form from 0.20 rad on (a robot
                                      # standing on a flight -- the stop held on every band), the delivered form
                                      # below it (flat ground: the creep after a stop 1.0-1.45 cm against 1.2-1.8
                                      # with the lock form everywhere; without the lock form the stop on a steep
                                      # descent is lost, s114: 71 % / 46 % stay). Not `even` (the scan: creep fails
                                      # with the scan masked), not 0.10 (flat creep fails), not 0.05
    "U_CLOCK_SPLIT": "1",             # the gait clock's fast strafes and gentle arcs in bins of their own
                                      # (flat.flat_gait_bin): the clock item measures the commands the flat gate
                                      # judges (pooled, the bins read 0.79-0.92 "met" while the gate failed them)
    # The stop on a climbing flight, from zero (2026-11-15). The formal run of 7ffc056 (the three switches above,
    # never run from zero before) lost it once the stairs quality was in full: seed 0's tier 2 stayed 54 % / 18 %
    # (scan / masked) on 15-18 cm up at model_2500 and 21 % / 7 % at model_3100 -- the robots rolled back down the
    # flight to its foot (1.4 m), the full payload's falls 39. go2-home's WC0 (the same recipe without the clock
    # split, from zero): 4 % / 4 % at 2500, 3 % / 18 % at 3000, 62 % / 72 % at 3600 -- not the clock split, and the
    # stop comes late and in part by itself. From seed 0's model_3100, 600 iterations each, tier 2 at model_3699:
    #   the recipe unchanged (uprobe_s161): 32 % / 42 %, 15-18 cm down 51 % / 75 %, the stop ok 83 % / 87 %
    #   the two switches below (uprobe_s159): 100 % / 94 %, every other band 100 %, the stop ok 100 % / 99 %, the
    #     typical payload's 99 % / 98 %, the full one's 98 % (68 % unchanged), slide-back 0.025
    #   U_STOP_PLACE=20 (uprobe_s160): 99 % / 57 %, 15-18 cm down 64 % -- not the weight
    # The full payload's falls: 39 at the start, 8 with the two switches, 3 unchanged, 5 with the weight 20 (bar 6):
    # the open item of this recipe, watched at the formal run's gates.
    "U_STAND_POSE_EVEN": "1",         # the standing pose cost acts on even ground only (the legged kind's entry):
                                      # with it on a flight the robots left the steps for the level ground below
    "U_STOP_HOLD_Q": "1",             # and its removal comes in along the stairs quality (as the legged: from the
                                      # first iteration the removal keeps a robot from climbing)
    # The wheeled freeze e (2026-11-15). The user on the videos of the formal run of 2c808e7 (seed 1, model_3800):
    # "下楼梯的pitch角速度很大，一直会前后仰俯". Probes from that seed's model_5000, 600 iterations each, read at
    # model_5599 on 15-18 cm, unloaded, straight down (descent_probe.py --straight; the control = the formal's 5600):
    "U_DESCENT_PITCH": "4",           # the base rocking fore and aft on a descent costs: the pitch rate 72 -> 37.5
                                      # deg/s rms (uprobe_s169; the mean of its last 4 checkpoints 34), the stairs
                                      # gate passes (the mean: no fall in any mode, the full payload's 2). The weight
                                      # 1 does half of it (48 deg/s, uprobe_s163); with the descent's leg cost
                                      # (U_DESCENT_LEG=6, uprobe_s174) the gate fails. Its price, on the climb: the
                                      # roll rate 32 -> 43-50 deg/s, the heading's p90 2-3 deg wider (13.7 -> 15.2-
                                      # 16.5); the heading cost that would take it back does not hold the loaded
                                      # stop (U_FLIGHT_HEADING 5 / 15, uprobe_s164 / s170: out) -- the climb's
                                      # heading is reported, not judged, in this version
    # (U_FAST_SIDE, the fast strafes' share of the flat commands, is not in the recipe. Alone at 0.1 it lifts the
    # one item the mean of the last 8 checkpoints fails in the flat gate, the gait clock on (0, +-0.6, 0), from
    # 79 / 80 % to 86 / 85 % (uprobe_s180). With the pitch cost above it gains nothing: that cost and the mean of
    # the last 4 checkpoints alone read 85-88 % on the strafes and 83-90 % on the fast gentle arcs and pass both
    # tier-2 gates (uprobe_s169); with the share added the strafes stay at 85-87 % and the arcs (1.0, 0, +-0.1..0.2)
    # drop to 73-79 % with the scan on, 4 of the 31 judged commands under the bar 80 % (uprobe_s182). The share
    # 0.2 drops them too (77 % at (1.0, 0, 0.15), uprobe_s181).)
  },
}

# one robot's own, each with its reason
# (None now: the As2's U_STAND_POSE_EVEN=1 is the legged kind's since 2026-11-14. The Go2's loaded falls tripled
# with that switch alone and on throughout (62 against 20, s50 / s51): with the overspeed cost and along the stairs
# quality they do not. The Go2W gained nothing from it under the first wheeled recipe (s38); with the stand's
# centre-of-mass cost and the tilt form it holds the stop on a climbing flight: the wheeled kind's since 2026-11-15.)
ROBOT: dict[str, dict[str, str]] = {}

# read by the runner when it starts (runner._entropy_by_q, _std_cap_by_q), not by the builders
RUNNER = ("U_ENTROPY_Q", "U_STD_CAP")


def off() -> bool:
  """U_RECIPE=0: the environment's switches alone (module docstring)."""
  return os.environ.get("U_RECIPE") == "0"


def recipe(robot) -> dict[str, str]:
  """The robot's switches from the table ({} with U_RECIPE=0)."""
  if off():
    return {}
  return {**COMMON, **KIND.get(robot.kind, {}), **ROBOT.get(robot.name, {})}


def explicit(robot) -> dict[str, str]:
  """The recipe's switches the environment sets itself: a probe's differences (they win)."""
  return {k: os.environ[k] for k in recipe(robot) if k in os.environ}


def effective(robot) -> dict[str, str]:
  """The recipe's switches as they apply: the table's values, the environment's where it sets one."""
  return {k: os.environ.get(k, v) for k, v in recipe(robot).items()}


@contextmanager
def scope(robot):
  """The robot's recipe in the environment for the duration: the switches the environment does not set itself.
  Restored afterwards -- the package builds every robot's tasks in one process, and one robot's switches must not
  reach the next one's build. Nested scopes of one robot are harmless (the inner one finds everything set)."""
  add = {k: v for k, v in recipe(robot).items() if k not in os.environ}
  os.environ.update(add)
  try:
    yield
  finally:
    for k in add:
      os.environ.pop(k, None)


def runner_switches(robot) -> dict[str, str]:
  """The runner's switches (RUNNER) in effect for the robot (None: no robot known -- the environment alone)."""
  table = recipe(robot) if robot is not None else {}
  out = {k: os.environ.get(k, table.get(k)) for k in RUNNER}
  return {k: v for k, v in out.items() if v}


def of_experiment(name: str | None):
  """The robot of a run by its experiment directory (<robot>_u_teacher, <robot>_u_student_<kind>[_rl]); None when
  it is none of the registered robots'."""
  from .robots import ROBOTS

  key = (name or "").split("_u_")[0]
  return next((r for r in ROBOTS.values() if r.key == key), None)


def record(robot) -> dict:
  """For a run's timeline: the recipe in effect, what the environment set itself, and whether the table was off."""
  if robot is None:
    return {"robot": None}
  return {"robot": robot.name, "off": off(), "switches": effective(robot), "explicit": explicit(robot)}
