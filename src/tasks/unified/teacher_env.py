"""The unified privileged teacher's environment (go2-depth 2026-11-11, the user's redesign; see robots.py for the
base it starts from). On top of the base:

1. Privileged actor. The depth teacher's actor (10 frames of proprioception + the elevated height scan) also sees
   the base velocity, the feet's heights, air times and contacts, and the robot's randomized
   physics (base CoM offset, payload, foot friction: mdp.dr_params, also added to the critic). Things a blind
   student can estimate from its history; not the cost terms' anchors (hold / drift offsets stay critic-only, so
   the teacher's holding is a behaviour the students can copy).
2. Rewards in three groups:
   CORE        fixed weights: tracking, gait clock, posture, termination, smoothness, swing clearance, climbing
               progress, flat speed;
   CONSTRAINED the cost terms serving an acceptance item; their weights are set by the constraints (constraints.py);
   OTHER       the remaining shaping terms of the recipe, kept at their weights ("full" variant) or removed ("lean":
               the user's "奖励只留十来项核心的").
3. One run through the curriculum (curriculum.py): terrain levels -> progress p -> stairs share, command ranges,
   payload limit, constraint switch-on and tightening. The recipe's iteration-based stairs share, command stages and
   old constraint set are replaced; the early cost ramps keep only the terms the constraints do not drive.
   The stairs quality comes in its own stage (2026-11-12; curriculum.py's q, which starts once the robots climb):
   until then the rewards are the recipe's M-key ones -- quality_diff() lists the weights and parameters its P keys
   change -- and the stairs constraints are off; q moves the rewards to the P values and the constraints to their bars.
4. A zero-reward meter (constraints.meter) last in the rewards: the stairs event counts per step-up.
5. Two scan modes (u_scan_mask): a share of the episodes walks blind, the scan masked. The depth student copies the
   seen mode and the blind student the blind one, so every constraint is measured in both (the evaluation robots: a
   fixed half in each) and the gate asks both to pass (constraints.py).
6. Joint friction, damping and armature randomized per robot (_add_joint_dr).
7. The payload's mount point from a wide box (_payload_point; the robot table's payload_point): the mount on the
   final machine is not known.
8. Two sloped tiles among the other ground (_add_slopes): readings only, no acceptance item.
9. Static friction above kinetic friction on the feet and wheels (stick_slip.py), in the checks' environments too.
"""

from __future__ import annotations

import inspect
import math
import os
from copy import deepcopy
from dataclasses import replace

import mjlab.terrains as terrain_gen
from mjlab.envs import ManagerBasedRlEnvCfg
import torch
from mjlab.envs.mdp import dr
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

import src.tasks.velocity.mdp as mdp
from src.tasks.velocity.blind_env_cfg import MAP3_STAIR_TERRAIN_NAMES, TEACHER_STAIRS_SHARE

from src.metrics import mj as metrics_mj

from . import constraints as ucons
from . import curriculum as ucur
from . import diagnostics
from . import flat as uflat
from . import recipe as urecipe
from . import stick_slip
from .models import PRIV_GROUP
from .robots import URobot, base_env_cfg, unified_recipe

# (foot_contact_forces stays critic-only: world-frame vectors, see symmetry.py)
PRIVILEGED_ACTOR_TERMS = ("base_lin_vel", "foot_height", "foot_air_time", "foot_contact")

CORE = {
  "legged": (
    "track_linear_velocity", "track_angular_velocity", "foot_gait", "pose", "is_terminated", "joint_acc_l2",
    "joint_pos_limits", "action_rate_l2", "body_ang_vel", "lin_vel_z_even", "body_orientation",
    "foot_swing_clearance", "foot_terrain_clearance", "foothold_height_gain", "foothold_height_drop", "foot_slip",
    "flat_speed_deficit", "stance_leg_tilt",
  ),
  "wheeled": (
    "track_linear_velocity", "track_angular_velocity", "foot_gait", "pose", "is_terminated", "is_alive",
    "joint_acc_l2", "joint_pos_limits", "action_rate_l2", "body_ang_vel", "lin_vel_z_even", "body_orientation",
    "foot_swing_clearance", "foothold_height_gain", "wheel_riser_lift", "wheel_slip", "flat_speed_deficit",
    "stance_leg_tilt",
  ),
}

# The survival items' bar (constraint_items: survive_up / survive_down): the share of the light evaluation robots'
# stairs episodes that may end in a fall or an illegal contact on the worst step-height band. Provisional
# (2026-11-12): to be set from the from-zero teachers' readings in their quality stage. teacher_gate.py judges a
# gate-only item at the bar in this file, so a change takes effect at the gate's next run, without a restart.
SURVIVE_BAR = 0.20
# The sideways command range's start by robot kind (curriculum.progress_commands' lateral_ramp: start m/s, p_from,
# p_to). A wheeled robot has to step for |vy| above ~0.1 m/s; with the full +-0.4 m/s from the first iteration the
# from-zero Go2W teachers (2026-11-12) failed 47-56% of their flat episodes until iteration 1300-1550, 86% of those
# under a sideways command. The test arm from seed 0's model_1300 with this ramp: 52% -> 5-9% in 150 iterations
# (seed 1 got there unaided at 1500-1550, seed 0 had not by 1550). Not yet run from iteration 0.
LAT_RAMP = {"wheeled": (0.1, 0.10, 0.25)}


def constraint_items(kind: str) -> list[dict]:
  """The acceptance items as constraints (doc/acceptance.md; targets: the earlier constraint sets' calibrated
  targets for the acceptance-measured items, the delivered lineage's training level for the stairs rates)."""
  legged = kind == "legged"

  def c(name, metric, sense, target, loose, activate, terms, params=None, stat="mean", lr=ucons.LR, min_scale=0.5,
        max_scale=3.0, start_scale=1.0, enforce=True, family="other", modes=ucons.MODES):
    # activate "q": a stairs-quality item, on the quality stage instead of the progress (constraints.py)
    # target: the students' bar (the gate judges there); tight: the teacher's own target, a margin inside it
    # family: the ground whose fail rate relaxes the item; modes: the scan modes it applies to (constraints.py)
    by = "q" if activate == "q" else "p"
    assert family in ucons.FAMILIES and set(modes) <= set(ucons.MODES), name
    item = {"name": name, "metric": metric, "params": params or {}, "stat": stat, "sense": sense, "target": target,
            "loose": loose, "activate": 0.0 if by == "q" else activate, "by": by, "terms": terms, "lr": lr,
            "min_scale": min_scale, "max_scale": max_scale, "start_scale": start_scale, "enforce": enforce,
            "family": family, "modes": tuple(modes)}
    if enforce:
      item["tight"] = ucons.with_margin(sense, target)
    return item

  drift = 0.06 if legged else 0.075   # p90 in training; the acceptance's 5% (calibrated by the blind chains)
  items = [
    # flat ground, from progress 0.15 (once the robots walk)
    c("clock", "gait_match", "min", 0.85 if legged else 0.82, 0.6, 0.15, ["foot_gait"], {"min_abs_vy": 0.5},
      min_scale=1.0),
    # the flat items on the light robots (<= 25% of the payload limit: the acceptance's unloaded tests); the heavy
    # ones (>= 2/3 of it) have their own
    # reward group (constraints.heavy_group, the user 2026-11-11)
    c("side_drift", "u_drift_light", "max", drift, 0.2, 0.15, ["side_drift"], {"term": "side_drift"}, stat="p90"),
    c("straight_drift", "u_drift_light", "max", drift, 0.2, 0.15, ["straight_drift"], {"term": "straight_drift"},
      stat="p90"),
    c("back_drift", "u_back_drift_light", "max", 0.03, 0.1, 0.15, ["side_drift"],
      {"term": "side_drift", "settled": True, "min_abs_vy": 0.45}),
    # creep and march over the light robots only (<= 25% of the payload limit, as the acceptance's unloaded tests)
    # creep (2026-11-12): on the acceptance's time base and statistic -- the net displacement from 2 s after the
    # stand began or the last push, windows 2-4 s long, the mean; bar: the acceptance's 1 cm (the teacher's own target
    # 0.8 cm, under the training randomization: the acceptance's play configuration is milder). Before: the distance
    # from the hold anchor over the whole stand, p90 <= 0.8 cm (teacher 0.64) -- the Go2 teacher read 1.7-1.8 cm
    # at 3000 with the multiplier at its 3x cap, where the acceptance's measure on the same robots gave a p90 of
    # 0.96 cm (constraints.stop_creep). That reading stays in the records as creep_total, logged only.
    c("creep", "u_stop_creep_light", "max", 0.01, 0.03, 0.15, ["standing_position_hold", "standing_speed"]),
    c("creep_total", "u_stand_hold_light", "max", None, 0.03, 0.15, [], {"term": "standing_position_hold"},
      stat="p90", enforce=False),
    # march (2026-11-12): as the acceptance's two march readings, each its own item with its own cost term.
    #   march: the net displacement over 5 s of marching, the mean, bar 5 cm (the teacher's own target 4 cm) --
    #     constraints.stop_creep on march_position_hold, chained 5 s windows.
    #   march_hip: the worst hip's distance from its default 4-5 s into a march that began with the stance splayed,
    #     the mean, bar 2 deg (teacher 1.6) -- constraints.march_reset. New: the hips' cost hung on the drift reading.
    #   march_total: the measure until now, logged only -- the p90 of the distance from the march's anchor over the
    #     whole march (10-20 s) against 4 cm (teacher 3.2). The Go2W seeds sat at its 3x cap with 6-7 cm while seed
    #     1's model_2600 passed the acceptance's march (3.2 cm over 5 s, hips 1.2 deg); the Go2 teacher at 3300 read
    #     4.0-4.5 cm there and 2.5 cm / 2.4 deg on the acceptance: its one miss was the hips, which nothing measured.
    c("march", "u_march_drift_light", "max", 0.05, 0.15, 0.15, ["march_position_hold"]),
    c("march_hip", "u_march_hip_light", "max", math.radians(2.0), 0.10, 0.15, ["march_hip_default"]),
    c("march_total", "u_stand_hold_light", "max", None, 0.15, 0.15, [], {"term": "march_position_hold"},
      stat="p90", enforce=False),
    c("spin", "u_spin_light", "max", 0.05, 0.15, 0.15, ["spin_position_hold"]),
    # stairs: the quality stage (q; off and at the recipe's M weights until the robots climb)
    # (family: descent relaxes on the descents' fail rate, climb on the climbing columns', the rest on the worse)
    c("descent", "stair_speed", "min", 0.9, 0.6, "q", ["stair_speed_deficit"], {"descend": True}, family="down"),
    c("climb", "stair_speed", "min", 0.85, 0.5, "q", ["stair_speed_deficit"], {"descend": False}, family="up"),
    c("leg_contact", "leg_contact", "max", 0.015, 0.1, "q", ["leg_ground_contact"], family="stairs"),
    c("stop_feet", "u_stairs_stop_feet", "min", 0.9, 0.6, "q", ["feet_stand_uneven"], family="stairs"),
    # Staying up on the stairs (2026-11-12): gate-only items -- no reward term, no multiplier (constraints.py). The
    # share of the light evaluation robots' episodes on their fixed rows that end in a fall or an illegal contact,
    # on the worst of the acceptance's four step-height bands (curriculum.eval_worst_band), per scan mode. The other
    # stairs items say how the robot climbs, none whether it stays up: the 11-12 probe's old teacher failed 84-96%
    # of its climbs on rows 7-8 with p = q = 1. The mid and heavy robots' rates are recorded only.
    c("survive_up", "evalfail:up", "max", SURVIVE_BAR, 0.5, "q", [], family="up"),
    c("survive_down", "evalfail:down", "max", SURVIVE_BAR, 0.5, "q", [], family="down"),
    # logged only: the roll bar is not calibrated (the user, 11-11: calibrate first)
    c("roll", "u_stairs_roll", "max", 0.105, 0.3, "q", [], stat="p90", enforce=False, family="stairs"),
    # logged only (2026-11-13): the heading off the commanded one while walking straight on over the steps
    # (constraints._stairs_heading); the acceptance judges the wheeled robots' heading off the flight in tinynav
    # (mean <= 3 deg = 0.052 rad); the two readings are not calibrated against each other
    c("heading", "u_stairs_heading", "max", 0.052, 0.3, "q", [], enforce=False, family="stairs"),
    # logged only (2026-11-13): the robots that stand on the level ground of a climbing column under a forward
    # command -- before the flight -- up to the common payload and in the heavy class (constraints._stairs_park:
    # the Go2W package teacher learnt to stand before the tall flights, tier 2 read 19% stalled, and no training
    # reading showed it). The acceptance: at most 5% stalled per band, unloaded and at the common payload.
    c("park", "u_stairs_park", "max", 0.05, 0.3, "q", [], enforce=False, family="up"),
    c("park_heavy", "u_stairs_park_heavy", "max", 0.05, 0.3, "q", [], enforce=False, family="up"),
    # logged only (2026-11-13): the same on the descending columns, the whole column -- the top platform, the flight
    # and past it (constraints._stairs_park_down: from 6600 to 7600 iterations the Go2 candidates' seen mode came to
    # wait at the edge of the top platform, tier 2 read 68-100% stalled on the descents from 8 cm up, and in training
    # it showed only as a lower descent speed). With U_STAIR_PARK the first is enforced, on its own cost (below).
    c("park_down", "u_stairs_park_down", "max", 0.05, 0.3, "q", [], enforce=False, family="down"),
    c("park_down_heavy", "u_stairs_park_down_heavy", "max", 0.05, 0.3, "q", [], enforce=False, family="down"),
    # logged only (2026-11-13): the same on even ground under a straight command, p90 (constraints._flat_heading;
    # the acceptance's lines: heading p10 / p90 within 8 deg = 0.14 rad 10 s in, from each line's own start)
    c("heading_flat", "u_flat_heading", "max", 0.14, 0.3, 0.15, [], stat="p90", enforce=False),
  ]
  if legged:
    # the shared metrics (src/metrics, tinynav's counters; 2026-11-12, doc/redesign.md): per step walked, the tinynav
    # bars over its 8.15 steps per run (gp3 avg15700's routes): kicks 2 / run -> 0.245, slips 0.2 / run -> 0.0245;
    # placements mid >= 80%, edge <= 10% as in tinynav. The placement items apply to the seen mode only: where a
    # foot lands on the tread is the scan's work (the blind mode's reading is logged, value_blind, and the blind
    # student's own acceptance judges it); kicks and slips apply to both, at the same bars (doc/acceptance.md: one
    # standard for both students)
    items += [
      c("edge", "lib:edge", "max", 0.10, 0.25, "q", ["feet_edge"], family="stairs", modes=("seen",)),
      c("mid", "lib:mid", "min", 0.80, 0.6, "q", ["feet_edge"], family="stairs", modes=("seen",)),
      c("slides", "lib:slips", "max", 0.0245, 0.1, "q", ["foot_slide_back"], family="stairs"),
      c("kicks", "lib:kicks", "max", 0.245, 0.6, "q", ["foot_stumble", "foot_step_touch"], family="stairs"),
    ]
  else:
    items += [
      c("wheels_down", "wheels_down", "min", 0.75, 0.5, 0.15, ["wheel_air_drivable"]),
      c("slow_arc", "arc_speed", "min", 0.9, 0.7, 0.15, ["track_linear_velocity"], min_scale=1.0),
      # slide-back from the shared metrics (tinynav's counter, 2026-11-12): the acceptance's 5% of step-ups. The idle
      # and top-landing rates stay on the meter's event terms until tinynav's per-climb bars are converted (wheeled
      # steps per run not measured yet).
      c("slides", "lib:slide_back", "max", 0.05, 0.15, "q", ["wheel_slide_back", "wheel_nosing"], family="up"),
      c("top_landing", "rate:top", "max", 0.033, 0.1, "q", ["wheel_flat_step_air"], family="up"),
      # As2W a1f (passed; tinynav idle 0.0-0.3 / climb): 0.6 idle lifts per step-up in training
      c("idle", "rate:idle", "max", 0.6, 1.5, "q", ["wheel_idle_step"], family="stairs"),
      # the shared metrics' idle steps (tinynav's counter: before the first riser / after the last, per step-up),
      # logged only, no bar, and not to be read yet: in training the counter has no start level (metrics/mj.py builds
      # it without one), so idle_start reads 0 and every idle lift off the highest level reached so far counts as
      # idle_end; the readings first have to be set against tinynav's per-climb counts on a delivered policy
      c("idle_start", "lib:idle_start", "max", None, None, "q", [], enforce=False, family="stairs"),
      c("idle_end", "lib:idle_end", "max", None, None, "q", [], enforce=False, family="stairs"),
    ]
  # -- the descent switches (probes, _probe_switches; read here: the items' terms and signatures are set up from them)
  v = os.environ.get("U_STOP_FEET_DIR")
  if v:
    # stop_feet by the direction the robot faces on the flight, each on that direction's fail rate, both on the one
    # cost term (constraints._apply: the stronger weight). "init_scale": on a checkpoint in the quality stage the
    # new items start at <x> (constraints.__call__) -- at the step of 0.003 a multiplier takes ~1300 iterations from
    # 1 to 3 at a shortfall of 0.3, longer than a probe; "own_cap": their cap is theirs, not U_Q_SCALE_MAX's.
    x = float(v)
    assert 0.5 <= x <= 10.0, f"U_STOP_FEET_DIR {v}: the start multiplier, 0.5-10"
    at = next(i for i, item in enumerate(items) if item["name"] == "stop_feet")
    items[at:at + 1] = [
      {**c(f"stop_feet_{d}", "u_stairs_stop_feet", "min", 0.9, 0.6, "q", ["feet_stand_uneven"], {"facing": d},
           max_scale=max(3.0, x), family=d), "init_scale": x, "own_cap": True}
      for d in ("up", "down")]
  if os.environ.get("U_STAIR_SPEED_H") == "1":
    for item in items:
      if item["name"] in ("descent", "climb"):
        item["metric"] = "u_stair_speed_h"   # (a resume keeps the multipliers: constraints.CARRY_SCALE)
  v = os.environ.get("U_ITEM_CAP")   # a probe's switch (_probe_switches): "<item>:<cap>,..."
  if v:
    # the named stairs-quality items' multipliers go up to their own cap, whatever U_Q_SCALE_MAX says ("own_cap").
    # A name this kind of robot has no item of is skipped: every robot's tasks are registered at import.
    caps = {k.strip(): float(x) for k, x in (kv.split(":") for kv in v.split(","))}
    assert all(1.0 <= x <= 10.0 for x in caps.values()), f"U_ITEM_CAP {v}: caps 1-10"
    for item in items:
      if item["name"] in caps:
        assert item["by"] == "q" and item["terms"], f"U_ITEM_CAP: {item['name']} is not a stairs-quality item with costs"
        item["max_scale"], item["own_cap"] = caps[item["name"]], True
  v = os.environ.get("U_ITEM_FLOOR")   # a probe's switch (_probe_switches): "<item>:<floor>,..."
  if v:
    # the named stairs-quality items' multipliers start from a floor (the first update clamps a checkpoint's
    # multiplier up to it: constraints.__call__), their cap at least that and their own. Skips as U_ITEM_CAP does.
    floors = {k.strip(): float(x) for k, x in (kv.split(":") for kv in v.split(","))}
    assert all(1.0 <= x <= 6.0 for x in floors.values()), f"U_ITEM_FLOOR {v}: floors 1-6"
    for item in items:
      if item["name"] in floors:
        assert item["by"] == "q" and item["terms"], f"U_ITEM_FLOOR: {item['name']} is not a stairs-quality item with costs"
        x = floors[item["name"]]
        item["min_scale"], item["max_scale"], item["own_cap"] = x, max(item["max_scale"], x), True
  if os.environ.get("U_STAIR_PARK"):
    # park_down is enforced and scales the standstill cost (_add_stair_park); its cap is its own (PARK_CAP, not
    # U_Q_SCALE_MAX's: at the cost's weight alone the account is close on the steepest rows -- constraints.stairs_park)
    at = next(i for i, item in enumerate(items) if item["name"] == "park_down")
    items[at] = {**c("park_down", "u_stairs_park_down", "max", 0.05, 0.3, "q", [STAIR_PARK_TERM], max_scale=PARK_CAP,
                     family="down"), "own_cap": True}
  if os.environ.get("U_COM_SUPPORT"):
    # logged only, with the cost (_add_com_support): the centre of mass off the feet's centre (m) at the stops on the
    # flights, up to the common payload and in the heavy class (constraints._com_stop)
    items += [c("com_stop", "u_com_stop", "max", 0.05, 0.3, "q", [], enforce=False, family="stairs"),
              c("com_stop_heavy", "u_com_stop_heavy", "max", 0.05, 0.3, "q", [], enforce=False, family="stairs")]
  if os.environ.get("U_STOP_STILL") == "1":
    # logged only: the stop in the middle of a flight as tier 2 judges it -- the share of the robots within 5 cm
    # between 2 s and 4 s after the zero command (the gate: 95%), nose up / nose down, up to the common payload and in
    # the heavy class (constraints._stairs_stop_still)
    items += [c(f"stop_still_{d}{h}", f"u_stairs_stop_still{h}", "min", 0.95, 0.8, "q", [], {"facing": d},
                enforce=False, family="stairs") for d in ("up", "down") for h in ("", "_heavy")]
  # -- the loaded-falls and kicks switches (probes, _probe_switches)
  if os.environ.get("U_STAIR_SPEED_LOADED"):
    # the heavier robots' stairs speed cost is discounted (_rewards: constraints.stair_speed_loaded), so the items
    # that scale that cost read the robots it still holds for in full: up to the common payload
    upto = {"stair_speed": "u_stair_speed_c", "u_stair_speed_h": "u_stair_speed_hc"}
    for item in items:
      if item["name"] in ("descent", "climb"):
        item["metric"] = upto[item["metric"]]   # (a resume keeps the multipliers: constraints.CARRY_SCALE)
  if os.environ.get("U_KICK_KEEPOUT") and kind == "legged":
    # the keep-out cost (_add_riser_keepout) is one of the kicks item's costs: in with the stairs quality, scaled by
    # the item's multiplier like the stumble costs (legged only: every robot's tasks are registered at import, and a
    # wheeled robot rolls up the risers)
    next(item for item in items if item["name"] == "kicks")["terms"].append(KEEPOUT_TERM)
  if uflat.scripts_on():
    items = _flat_script_items(items, c)
  return items


# the items the flat protocols add (their multipliers' cap: the probe switch U_FLAT_NEW_MAX)
FLAT_NEW = ("head_tail", "head_med", "speed_lin", "speed_yaw", "stance", "stance_stop")


def _flat_script_items(items: list[dict], c) -> list[dict]:
  """The flat items on the acceptance's own protocols (flat.py, 2026-11-12; U_FLAT_SCRIPTS=0: `items` as they are).
  Each reads the WORST command bin of the scripted evaluation robots' samples, per scan mode, with the acceptance's
  statistic; bar = the acceptance's, the teacher's own target a margin inside it (constraints.with_margin) under
  the training randomization. Same bars for legged and wheeled (accept_tables.py: SIDE_DRIFT_BAR is empty).

    item            was (training)                           now                                     acceptance
    clock           gait_match, |vy| >= 0.5, one mean,       flat:gait, the worst bin's mean, 0.80    gait_clock.py: each command
                    0.85 / 0.82                                                                       below 1 m/s >= 80%
    side_drift      u_drift_light p90 of |offset| / path     flat:drift "tail" = max(|p10|, |p90|)   drift_all.py + accept_tables
    straight_drift  >= 1 m into a 2-5 s command, 0.06 /      of the signed offset / distance after   .py:320-331: both tails
                    0.075                                    10 s from a standstill, 0.05            against 5% of the distance
    head_tail       --                                       flat:head "tail", 8 deg                 the same lines: heading
    head_med        --                                       flat:head "absmed", 3 deg               p10 / p90 <= 8, median <= 3
    creep           u_stop_creep_light: every stop, one      flat:creep: the mean per pre-stop        eval_standing.py: the worst
                    mean, 0.01 m                             command, the worst bin, 0.01 m          group's mean <= 1 cm
    march_hip       u_march_hip_light: the worst hip 4-5 s   flat:march_hip: after the 3 s stand     march_eval.py: mean <= 2 deg
                    INTO the march, 2 deg                    that follows a 5 s march, 2 deg         after that stand
    speed_lin       --                                       flat:speed_lin "dev1" = |mean velocity  accept_matrix.py: 80-120% per
    speed_yaw       --                                       / command - 1| per axis, worst bin, 0.2 commanded axis
    stance          --                                       flat:stance p90 of the worst leg         long_stand.py / stop_then_move
    stance_stop     --                                       joint's offset over 3-9 s of rest /     .py: p90 <= 0.15 rad
                                                             the mean offset 3 s after a stop, 0.15

  The multipliers of the five re-measured items carry over a resume (constraints.CARRY_SCALE); the six new ones
  start at 1 and are capped at 2 (the tracking and posture terms act on every ground). The readings until now
  stay in the records, logged only: clock_side, side_drift_p90, straight_drift_p90, creep_pooled, march_hip_in.
  back_drift, march, spin and the *_total items are unchanged (they read the robots on drawn commands)."""
  old = {x["name"]: x for x in items}
  act = 0.15
  new = {x["name"]: x for x in (
    c("clock", "flat:gait", "min", 0.80, 0.6, act, ["foot_gait"], {"worst": "min"}, min_scale=1.0),
    c("side_drift", "flat:drift", "max", 0.05, 0.2, act, ["side_drift"], {"bins": ["side"]}, stat="tail"),
    c("straight_drift", "flat:drift", "max", 0.05, 0.2, act, ["straight_drift"],
      {"bins": ["fwd", "fwd_slow", "back"]}, stat="tail"),
    c("creep", "flat:creep", "max", 0.01, 0.03, act, ["standing_position_hold", "standing_speed"]),
    c("march_hip", "flat:march_hip", "max", math.radians(2.0), 0.10, act, ["march_hip_default"]),
  )}
  added = [
    c("head_tail", "flat:head", "max", 8.0, 20.0, act, ["straight_drift", "side_drift"], stat="tail", max_scale=2.0),
    c("head_med", "flat:head", "max", 3.0, 8.0, act, ["straight_drift", "side_drift"], stat="absmed", max_scale=2.0),
    c("speed_lin", "flat:speed_lin", "max", 0.2, 0.4, act, ["track_linear_velocity", "flat_speed_deficit"],
      stat="dev1", min_scale=1.0, max_scale=2.0),
    c("speed_yaw", "flat:speed_yaw", "max", 0.2, 0.4, act, ["track_angular_velocity"], stat="dev1", min_scale=1.0,
      max_scale=2.0),
    c("stance", "flat:stance", "max", 0.15, 0.3, act, ["standing_pose_load"], stat="p90", max_scale=2.0),
    c("stance_stop", "flat:stance_stop", "max", 0.15, 0.3, act, ["standing_pose_load"], {"bins": ["fast", "fwd"]},
      stat="p90", max_scale=2.0),
  ]
  assert tuple(x["name"] for x in added) == FLAT_NEW
  logged = [{**{k: v for k, v in old[src].items() if k != "tight"}, "name": name, "target": None, "terms": [],
             "enforce": False}
            for name, src in (("clock_side", "clock"), ("side_drift_p90", "side_drift"),
                              ("straight_drift_p90", "straight_drift"), ("creep_pooled", "creep"),
                              ("march_hip_in", "march_hip"))]
  out = []
  for x in items:
    out.append(new.get(x["name"], x))
    if x["name"] == "spin":
      out += added + logged
  return out


def _flat_tracker(cfg: ManagerBasedRlEnvCfg) -> None:
  """flat.flat_measure as a reward term with zero output (the last one), and its two event wrappers: no pushes on
  a robot under a script, and a start heading along / across the flat strip for a share of the evaluation robots
  on the flat column (a 10 s line only fits the 8 m wide column from such a heading)."""
  cfg.rewards["u_flat"] = RewardTermCfg(func=uflat.flat_measure, weight=1e-9, params={})
  for name, wrapper in (("push_robot", uflat.flat_no_push), ("reset_base", uflat.flat_reset_yaw)):
    ev = cfg.events.get(name)
    if ev is None:
      continue
    assert not isinstance(ev.func, type), f"{name}: a class event; the flat wrappers wrap functions"
    ev.params = {**ev.params, "flat_inner": ev.func}
    ev.func = wrapper


# A term the recipe's M keys do not have starts the quality stage at this share of its P weight: next to nothing in
# the reward, but not 0 -- the reward manager skips zero-weight terms, and the event counters would stop.
Q_EPS = 1e-3


def _m_default(func, key: str):
  """The value a reward term uses for a parameter its cfg does not set: its function's (a class's __call__) default."""
  sig = inspect.signature(func.__call__ if isinstance(func, type) else func)
  par = sig.parameters.get(key)
  assert par is not None and par.default is not inspect.Parameter.empty, f"{func.__name__}: no default for {key!r}"
  return par.default


def _same(a, b) -> bool:
  try:
    return bool(a == b)
  except Exception:
    return repr(a) == repr(b)


def quality_diff(robot: URobot, rewards: dict, constrained_p: set[str], constrained_q: set[str],
                 dropped: tuple[str, ...] = (), play: bool = False, terrain: str = "map3") -> tuple[dict, dict, dict]:
  """What the recipe's P keys change in the rewards against its M keys (`rewards`: the P environment's; `dropped`:
  the terms the lean variant took out of it). Returns
    weights  {term: (M weight, P weight)}   terms no constraint scales; a P-only term starts at Q_EPS x its weight
    params   {term: {key: (M value, "lerp" | "switch")}}   numbers are interpolated, anything else switches
    start    {term: M weight}               for the terms the stairs-quality constraints scale (Q_EPS x for P-only)
  constraints.quality_terms and the "by q" constraints apply them along the stairs quality q."""
  m = base_env_cfg(robot, play=play, terrain=terrain, keys="M").rewards
  only_m = [t for t in m if t not in rewards and t not in dropped and not t.startswith("u_")]
  weights: dict = {}
  params: dict = {}
  start: dict = {}
  for t, tp in rewards.items():
    if t.startswith("u_"):
      continue
    tm = m.get(t)
    w_m = tm.weight if tm is not None else Q_EPS * tp.weight
    if t in constrained_q:
      start[t] = w_m
    elif w_m != tp.weight:
      # a flat item's term keeps its weight x multiplier from the first iteration: it must not differ
      assert t not in constrained_p, f"{robot.name}: {t} is scaled by a flat constraint and differs between M and P"
      weights[t] = (w_m, tp.weight)
    if tm is None:
      continue
    gone = [k for k in tm.params if k not in tp.params]
    assert not gone, f"{robot.name}: {t} has parameters under the M keys only: {gone}"
    for key, b in tp.params.items():
      a = tm.params[key] if key in tm.params else _m_default(tp.func, key)
      if _same(a, b):
        continue
      num = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (a, b))
      assert not isinstance(a, SceneEntityCfg), f"{robot.name}: {t}.{key}: an M scene entity is not supported"
      params.setdefault(t, {})[key] = (a, "lerp" if num else "switch")
  # (terms only the M keys have would stay out of the teacher; none in the present recipes)
  assert not only_m, f"{robot.name}: reward terms under the M keys only: {only_m}"
  return weights, params, start


METER_EVENTS = {
  "legged": {"slide": "foot_slide_back", "kick": "foot_stumble"},
  "wheeled": {"slide": "wheel_slide_back", "top": "wheel_flat_step_air", "idle": "wheel_idle_step"},
}


def _add_privileged(cfg: ManagerBasedRlEnvCfg) -> None:
  """The privileged terms as their own observation group (models.UPrivLatentModel encodes it into a compact vector;
  2026-11-12, doc/redesign.md 3.5)."""
  critic = cfg.observations["critic"].terms
  if "dr_params" not in critic:
    critic["dr_params"] = ObservationTermCfg(
      func=mdp.dr_params, params={"friction_cfg": cfg.events["foot_friction"].params["asset_cfg"]}
    )
  priv = {}
  for name in PRIVILEGED_ACTOR_TERMS + ("dr_params",):
    term = deepcopy(critic[name])
    term.noise = None
    term.history_length = 0
    priv[f"priv_{name}"] = term
  cfg.observations[PRIV_GROUP] = ObservationGroupCfg(terms=priv, concatenate_terms=True, enable_corruption=False)


# ---- scan masking: the blind mode (2026-11-12, doc/redesign.md 3.5) ----
SCAN_MASK_SHARE = 0.4   # of the episodes, the actor's scan is masked: the teacher also learns to walk without it


def u_masked(env, inner, **params) -> torch.Tensor:
  """An observation term's value times the episode's scan flag (0: masked)."""
  x = inner(env, **params)
  on = getattr(env, "u_scan_on", None)
  return x if on is None else x * on[:, None]


def scan_flag(env) -> torch.Tensor:
  on = getattr(env, "u_scan_on", None)
  if on is None:
    return torch.ones(env.num_envs, 1, device=env.device)
  return on[:, None]


def u_scan_mask(env, env_ids, share: float = SCAN_MASK_SHARE, mode: str | None = None) -> None:
  """Reset event: each episode is seen (1) or blind (0). mode "on" / "off" (the students' tasks) or U_SCAN=on / off
  (evaluations) fixes it; else a share of the episodes is blind -- and the evaluation robots (env.u_eval_mask, the
  teacher's runner) keep one mode each, even index seen and odd blind, so each mode's constraints read half of them
  whatever the share (their columns and rows are drawn anew every episode, the same for both halves)."""
  if getattr(env, "u_scan_on", None) is None:
    env.u_scan_on = torch.ones(env.num_envs, device=env.device)
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  mode = mode or os.environ.get("U_SCAN", "mix")
  if mode == "on":
    v = torch.ones(len(env_ids), device=env.device)
  elif mode == "off":
    v = torch.zeros(len(env_ids), device=env.device)
  else:
    v = (torch.rand(len(env_ids), device=env.device) >= share).float()
    ev = getattr(env, "u_eval_mask", None)
    if ev is not None:
      v = torch.where(ev[env_ids], (env_ids % 2 == 0).float(), v)
  env.u_scan_on[env_ids] = v


def _add_scan_mask(cfg: ManagerBasedRlEnvCfg) -> None:
  from src.tasks.velocity.depth_env_cfg import TEACHER_SCAN_NAME

  actor = cfg.observations["actor"].terms
  term = actor[TEACHER_SCAN_NAME]
  assert not isinstance(term.func, type), "the actor scan term is a class; u_masked wraps functions"
  term.params = {**term.params, "inner": term.func}
  term.func = u_masked
  actor["scan_flag"] = ObservationTermCfg(func=scan_flag)
  cfg.observations["critic"].terms["scan_flag"] = ObservationTermCfg(func=scan_flag)
  cfg.events["u_scan_mask"] = EventTermCfg(mode="reset", func=u_scan_mask, params={})


# ---- joint friction, damping and armature, randomized (2026-11-12, doc/redesign.md 3.3) ----
# Not published for Go2 / Go2W / B2 / B2W; Unitree's as2.xml gives friction loss 1 / 1 / 2 N m and damping 0.2 / 0.2 /
# 0.5 N m s/rad on its 60 / 60 / 90 N m joints. Uniform from 0 to these shares of each joint group's torque limit
# (friction 2.5%: As2's 1.7-2.2% inside; damping 1.5x As2's ratios), until real-robot measurements narrow them.
JOINT_FRICTION_SHARE = 0.025
JOINT_DAMPING_SHARE = {"hip": 1.5 * 0.2 / 60, "thigh": 1.5 * 0.2 / 60, "calf": 1.5 * 0.5 / 90}
# The legs' rotor inertia (armature): each joint's model value x a uniform factor per robot. The models' values are
# estimates (Go2 / Go2W 0.01 / 0.01 / 0.02 kg m^2 for hip / thigh / calf, B2 / B2W 0.1, As2 0.026 / 0.026 / 0.038; none
# measured on our robots), +-30% until they are.
JOINT_ARMATURE_SCALE = (0.7, 1.3)


def _effort(actuator) -> float | None:
  a = actuator
  while not hasattr(a, "effort_limit") and hasattr(a, "base_cfg"):
    a = a.base_cfg
  return getattr(a, "effort_limit", None)


def _add_joint_dr(cfg: ManagerBasedRlEnvCfg) -> None:
  for a in cfg.scene.entities["robot"].articulation.actuators:
    expr = tuple(getattr(a, "target_names_expr", ()) or getattr(getattr(a, "base_cfg", None), "target_names_expr", ()))
    group = next((g for g in ("hip", "thigh", "calf") if any(g in e for e in expr)), None)
    effort = _effort(a)
    if group is None or effort is None:
      continue   # wheels: their brake / drive model stays as it is
    # one SceneEntityCfg per event: resolving writes the joint ids into it, and a shared one fails the second check
    cfg.events[f"u_joint_friction_{group}"] = EventTermCfg(
      mode="startup", func=dr.joint_friction,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=expr), "ranges": (0.0, JOINT_FRICTION_SHARE * effort),
              "operation": "abs"})
    cfg.events[f"u_joint_damping_{group}"] = EventTermCfg(
      mode="startup", func=dr.joint_damping,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=expr),
              "ranges": (0.0, JOINT_DAMPING_SHARE[group] * effort), "operation": "abs"})
    cfg.events[f"u_joint_armature_{group}"] = EventTermCfg(
      mode="startup", func=dr.joint_armature,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=expr), "ranges": JOINT_ARMATURE_SCALE,
              "operation": "scale"})


KEEPOUT_TERM = "riser_keepout"   # (no "u_" prefix: quality_diff takes it as a cost of the P keys the M keys lack)


def _add_riser_keepout(cfg: ManagerBasedRlEnvCfg, robot: URobot) -> None:
  """U_KICK_KEEPOUT=<w> (a probe's switch, _probe_switches): the cost constraints.riser_keepout at the weight -w, on
  the feet and the contact sensor of the recipe's swing clearance cost. Before constraint_items / _rewards: it is one
  of the kicks item's terms (constraint_items), so it starts at Q_EPS of its weight and comes in with the stairs
  quality (quality_diff: a cost the M keys do not have)."""
  v = os.environ.get("U_KICK_KEEPOUT")
  if not v or robot.kind != "legged":
    return
  w = float(v)
  assert 0.0 < w <= 50.0, f"U_KICK_KEEPOUT {v}: the cost's weight"
  src = cfg.rewards["foot_terrain_clearance"]
  params = {"sensor_name": src.params["sensor_name"], "asset_cfg": deepcopy(src.params["asset_cfg"])}
  if "foot_radius" in src.params:
    params["foot_radius"] = src.params["foot_radius"]
  if os.environ.get("U_KICK_KEEPOUT_DIR") == "1":   # only the faces a foot moves towards (riser_keepout's approach)
    params["approach"] = True
  cfg.rewards[KEEPOUT_TERM] = RewardTermCfg(func=ucons.riser_keepout, weight=-w, params=params)


STAIR_PARK_TERM = "stairs_park"   # (no "u_" prefix, as KEEPOUT_TERM: in with the stairs quality)
PARK_CAP = 4.0                    # the cap of the park_down item's multiplier under U_STAIR_PARK


def _add_stair_park(cfg: ManagerBasedRlEnvCfg) -> None:
  """U_STAIR_PARK=<w>[,<heavy>] (a probe's switch, _probe_switches): the cost constraints.stairs_park at the weight
  -w, for the heavy class at <heavy> times it (0 without it). Before constraint_items / _rewards: it is the park_down
  item's term (constraint_items), so it starts at Q_EPS of its weight and comes in with the stairs quality
  (quality_diff: a cost the M keys do not have). Every kind of robot: the reading and the cost are of the command
  and the speed over the ground."""
  v = os.environ.get("U_STAIR_PARK")
  if not v:
    return
  w, _, heavy = v.partition(",")
  w, heavy = float(w), float(heavy or 0.0)
  assert 0.0 < w <= 20.0 and 0.0 <= heavy <= 1.0, f"U_STAIR_PARK {v}: <weight 0-20>[,<the heavy class's share 0-1>]"
  command = cfg.rewards["stair_speed_deficit"].params["command_name"]
  cfg.rewards[STAIR_PARK_TERM] = RewardTermCfg(func=ucons.stairs_park, weight=-w,
                                               params={"command_name": command, "heavy": heavy})


BODY_CLEAR_TERM = "body_clear"   # (no "u_" prefix, as KEEPOUT_TERM: in with the stairs quality)


def _add_body_clear(cfg: ManagerBasedRlEnvCfg, robot: URobot) -> None:
  """U_BODY_CLEAR=<w>[,<margin>[,<reach>]] (a probe's switch, _probe_switches): the cost constraints.body_clear at
  the weight -w, on the geoms of the illegal-contact termination's sensor; its margin over a step's top (m, 0.03)
  and its reach towards a step face (m, 0.05) when given. Before _rewards / quality_diff: a cost the M keys do not
  have, so it starts at Q_EPS of its weight and comes in with the stairs quality (no constraint scales it)."""
  v = os.environ.get("U_BODY_CLEAR")
  if not v or "illegal_contact" not in cfg.terminations:
    return
  f = [float(x) for x in v.split(",")]
  assert 1 <= len(f) <= 3 and 0.0 < f[0] <= 50.0, f"U_BODY_CLEAR {v}: <the cost's weight 0-50>[,<margin>[,<reach>]]"
  params = {"sensor_name": cfg.terminations["illegal_contact"].params["sensor_name"]}
  if len(f) > 1:
    assert 0.03 <= f[1] <= 0.10, f"U_BODY_CLEAR {v}: the margin over a step's top, 0.03-0.10 m"
    params["margin"] = f[1]
  if len(f) > 2:
    assert 0.05 <= f[2] <= 0.15, f"U_BODY_CLEAR {v}: the reach towards a step face, 0.05-0.15 m"
    params["reach"] = f[2]
  cfg.rewards[BODY_CLEAR_TERM] = RewardTermCfg(func=ucons.body_clear, weight=-f[0], params=params)


COM_TERM = "com_support"   # (no "u_" prefix, as KEEPOUT_TERM: in with the stairs quality)


def _add_com_support(cfg: ManagerBasedRlEnvCfg) -> None:
  """U_COM_SUPPORT=<w>[,<moving>] (a probe's switch, _probe_switches): the cost constraints.com_support at the weight
  -w, under a command at <moving> times it (0 without it: the stops alone). Before _rewards / quality_diff: a cost
  the M keys do not have, so it starts at Q_EPS of its weight and comes in with the stairs quality (no constraint
  scales it). Every kind of robot: the feet's sites carry the same names on the wheeled (the wheels' centres)."""
  v = os.environ.get("U_COM_SUPPORT")
  if not v or "stair_speed_deficit" not in cfg.rewards:
    return
  from src.tasks.velocity.blind_env_cfg import FOOT_SITE_NAMES

  w, _, moving = v.partition(",")
  w, moving = float(w), float(moving or 0.0)
  assert 0.0 < w <= 30.0 and 0.0 <= moving <= 1.0, f"U_COM_SUPPORT {v}: <weight 0-30>[,<the share under a command 0-1>]"
  src = cfg.rewards["stair_speed_deficit"].params
  cfg.rewards[COM_TERM] = RewardTermCfg(
    func=ucons.com_support, weight=-w,
    params={"asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES), "sensor_name": src["sensor_name"],
            "command_name": src["command_name"], "moving": moving})


_M_WEIGHT_NOTED: set[tuple[str, str]] = set()   # (one line per robot and term, whatever the number of its tasks)


def m_weight_factors() -> dict[str, float]:
  """U_M_WEIGHT=<term>:<factor>,... (a probe's switch, _probe_switches) as {term: factor}."""
  out = {}
  for part in filter(None, (x.strip() for x in os.environ.get("U_M_WEIGHT", "").split(","))):
    t, _, f = part.partition(":")
    assert t and f and t not in out and 0.0 <= float(f) <= 100.0, f"U_M_WEIGHT {part}: <term>:<factor>, each term once"
    out[t] = float(f)
  return out


def m_weights(factors: dict[str, float], items: list[dict], weights: dict, rewards: dict, by_p: set) -> list[str]:
  """U_M_WEIGHT: each term's weight before the stairs quality (q = 0) times its factor; from q = Q_IN on it is the P
  weight as before (a constraint's term: the item's "start", constraints._apply; any other: `weights`,
  constraints.quality_terms -- a term equal in the M and P keys becomes a difference of the two). Returns the terms
  this environment does not have (every robot's tasks are registered at import: another robot's term is no error)."""
  missing = []
  for t, f in factors.items():
    assert t not in by_p and not t.startswith("u_"), f"U_M_WEIGHT {t}: not a cost the stairs quality brings in"
    owners = [c for c in items if c["by"] == "q" and t in c["terms"]]
    for c in owners:
      c["start"][t] *= f
    if owners:
      continue
    if t not in rewards:
      missing.append(t)
      continue
    w_m, w_p = weights.get(t, (rewards[t].weight, rewards[t].weight))
    weights[t] = (w_m * f, w_p)
  return missing


def _payload_point(cfg: ManagerBasedRlEnvCfg, robot: URobot) -> None:
  """The payload's mount point drawn from the robot table's wide box (robot.payload_point; 2026-11-12, the user:
  "负载的重心偏移可以尽可能的大一些，因为我也不知道之后整机上会装在哪里"), in place of the blind chains' range. The
  event's corner share (heavy robots at the box's front or rear end, at its top) follows the box. Where the robots
  fall with it: the records' breakdown by payload position (diagnostics.py)."""
  if robot.payload_point is None:
    return
  ev = cfg.events["base_payload"]
  assert ev.func is mdp.base_payload_point, "the training terrain's payload event is not the payload point"
  box = tuple((float(lo), float(hi)) for lo, hi in robot.payload_point)
  assert len(box) == 3 and all(lo <= hi for lo, hi in box), f"{robot.name}: payload_point {robot.payload_point}"
  ev.params["pos_ranges"] = box


def sticky_payload(env, env_ids, inner, redraw: float, light_range: tuple[float, float], heavy_max: float,
                   heavy_share: float, corner_share: float = 0.0, **params) -> None:
  """Reset event: `inner` (mdp.base_payload_point) with the payload class staying with the robot (2026-11-13, the
  probe switch U_PAYLOAD_STICKY). The payload was drawn anew at every reset while a robot's three difficulty rows
  stay with it (curriculum.progress_stairs_share): a heavy episode started on a row earned by light ones -- the Go2
  teacher at 5100 it: the heavy robots failed 70-76% of their episodes at the training rows, the light ones 23%
  (fixed rows: 0.37-0.42 against 0.11-0.13) -- and its fall sent the robot's next, light episode a row down.
  Now each robot keeps a class, light / mid / heavy (curriculum.PAY_CLASSES): its payload below LIGHT_FRAC, up to
  HEAVY_FRAC or above HEAVY_FRAC of the event's limit now (heavy_max, which follows the progress), and draws the
  class anew with probability `redraw` at a reset. The mass and the point are drawn at every reset as before,
  inside the class. The classes' shares and the draws within each are the inner event's own (the light range, or
  [0, heavy_max] for heavy_share of the robots, corner_share of those at the corner), so the payloads over all the
  robots are distributed as before; which robot carries which is what changes, and each class's rows find its own
  frontier. The evaluation robots (env.u_eval_mask) draw their class at every reset: their stairs rows are fixed,
  and every class then stands on every row and tile alike. env.u_pay_class: each robot's class (-1 before its
  first draw), saved with the rows (curriculum.progress)."""
  from .constraints import HEAVY_FRAC, LIGHT_FRAC

  assert not params.get("corner_terrain_names") and params.get("ramp_steps", 0) <= 0, \
    "sticky_payload: the corner on every tile and no ramp of the event's own"
  dev = env.device
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=dev)
  env_ids = env_ids.to(dev, dtype=torch.long)
  n = len(env_ids)
  if n == 0:
    return
  cls = getattr(env, "u_pay_class", None)
  if cls is None or len(cls) != env.num_envs:
    cls = env.u_pay_class = torch.full((env.num_envs,), -1, dtype=torch.long, device=dev)
  hs, cs = float(heavy_share), float(corner_share)
  # of all the robots: the light range, the corner, and the three bands of [0, heavy_max]
  w_range, w_corner = 1.0 - hs, hs * cs
  w_band = [hs * (1.0 - cs) * f for f in (LIGHT_FRAC, HEAVY_FRAC - LIGHT_FRAC, 1.0 - HEAVY_FRAC)]
  p_cls = [w_range + w_band[0], w_band[1], w_band[2] + w_corner]
  new = (cls[env_ids] < 0) | (torch.rand(n, device=dev) < redraw)
  ev = getattr(env, "u_eval_mask", None)
  if ev is not None:
    new = new | ev[env_ids]
  if bool(new.any()):
    cls[env_ids[new]] = torch.multinomial(torch.tensor(p_cls, device=dev), int(new.sum()), replacement=True)
  c = cls[env_ids]
  u = torch.rand(n, device=dev)
  in_range = (c == 0) & (u < w_range / max(p_cls[0], 1e-9))
  at_corner = (c == 2) & (u < w_corner / max(p_cls[2], 1e-9))
  inner(env, env_ids[in_range], light_range=light_range, heavy_max=heavy_max, heavy_share=0.0, corner_share=0.0,
        **params)
  inner(env, env_ids[at_corner], light_range=light_range, heavy_max=heavy_max, heavy_share=1.0, corner_share=1.0,
        **params)
  edges = (0.0, LIGHT_FRAC * heavy_max, HEAVY_FRAC * heavy_max, heavy_max)
  for k in range(3):   # (the inner event's light range is uniform: the band's)
    band = (c == k) & ~in_range & ~at_corner
    inner(env, env_ids[band], light_range=(edges[k], edges[k + 1]), heavy_max=heavy_max, heavy_share=0.0,
          corner_share=0.0, **params)


# the inner event's per-world model fields and recompute level (mjlab's requires_model_fields marks the function)
sticky_payload.model_fields = mdp.base_payload_point.model_fields
sticky_payload.recompute = mdp.base_payload_point.recompute


# ---- slopes (2026-11-12; the user, through the students' session: "斜坡可以稍微加一点，你定") ----
SLOPE_SHARE = 0.10   # of the episodes on the other ground (not stairs), half on each sloped tile
SLOPE_MAX = 0.25     # the top row's gradient along the tile's axes (14 deg; 16 deg at most, between them)


def _add_slopes(cfg: ManagerBasedRlEnvCfg) -> dict[str, float]:
  """Two sloped tiles among the other ground: a pyramid hill (slope_down: the robot starts on its flat top, 3.5 m
  across along the tile's axes and 2 m square between them) and a pit (slope_up: at its bottom), one column each out
  of the flat ground's three, the gradient rising with the other ground's row from 0 to SLOPE_MAX (the slope runs
  2.25 m on from the flat part, 0.56 m of height at the top row; a promotion takes 3 m walked, so 1.25 m of it or
  more). No acceptance item is measured on them (constraints.off_slopes
  leaves their robots out; the cost terms still act there): the records show them as grounds of their own.
  Returns the other ground's episode shares per tile name for the column draw (curriculum.progress_stairs_share):
  the sloped tiles SLOPE_SHARE together, the rest split as the columns were before."""
  gen = cfg.scene.terrain.terrain_generator
  subs = gen.sub_terrains
  others = {n: s.proportion for n, s in subs.items() if n not in MAP3_STAIR_TERRAIN_NAMES}
  assert set(others) == {"random_rough", "flat"}, f"the other ground: {sorted(others)}"
  step = sum(s.proportion for s in subs.values()) / gen.num_cols   # one column
  assert subs["flat"].proportion > 2.5 * step, "the flat ground keeps a column"
  hill = terrain_gen.HfPyramidSlopedTerrainCfg(proportion=step, slope_range=(0.0, SLOPE_MAX), platform_width=2.0)
  new = {}
  for name, sub in subs.items():
    if name == "flat":   # before the flat tile, which stays next to the walled ones (blind_env_cfg's column order)
      new[ucur.SLOPE_NAMES[0]] = hill
      new[ucur.SLOPE_NAMES[1]] = replace(hill, inverted=True)
      sub = replace(sub, proportion=sub.proportion - 2 * step)
    new[name] = sub
  gen.sub_terrains = new
  total = sum(others.values())
  shares = {n: (1.0 - SLOPE_SHARE) * p / total for n, p in others.items()}
  shares |= {n: SLOPE_SHARE / 2 for n in ucur.SLOPE_NAMES}
  return shares


def _rewards(cfg: ManagerBasedRlEnvCfg, kind: str, items: list[dict], lean: bool) -> tuple[list[str], list[str]]:
  rewards = cfg.rewards
  constrained = sorted({t for c in items for t in c["terms"]})
  missing = [t for t in constrained + list(CORE[kind]) if t not in rewards]
  assert not missing, f"terms not in the recipe's environment: {missing}"
  other = [t for t in rewards if t not in constrained and t not in CORE[kind]]
  if lean:
    for t in other:
      del rewards[t]
  # the heavy robots' reward group (after the terms it reads): each flat item's costs at their configured weight on
  # the robots with >= 2/3 of the payload limit, whatever the light robots' multipliers (constraints.heavy_group)
  for item, (_, _, _, terms) in ucons.HEAVY_ITEMS.items():
    terms = [t for t in terms if t in rewards]
    if terms:
      rewards[f"u_heavy_{item}"] = RewardTermCfg(func=ucons.heavy_group, weight=1.0, params={"terms": terms})
  if os.environ.get("U_FLAT_ONLY") == "1":   # a probe's switch (_probe_switches), after the heavy groups
    heavy_terms = sorted({t for _, _, _, ts in ucons.HEAVY_ITEMS.values() for t in ts if t in rewards})
    sensor = rewards["straight_drift"].params.get("uneven_sensor_name") if "straight_drift" in rewards else None
    params = {"sensor_name": sensor or "terrain_scan", "heavy_terms": heavy_terms}
    if os.environ.get("U_STAND_POSE_EVEN") == "1" and "standing_pose_load" in rewards:   # a probe's switch
      params["zero_terms"] = ("standing_pose_load",)
    if os.environ.get("U_FLAT_ONLY_KEEP"):   # a probe's switch (_probe_switches)
      keep = tuple(t for t in os.environ["U_FLAT_ONLY_KEEP"].split(",") if t)
      assert keep and all(t in rewards for t in keep) and not set(keep) & set(params.get("zero_terms", ())), \
        f"U_FLAT_ONLY_KEEP {keep}: reward terms of this build (and none of U_STAND_POSE_EVEN's)"
      params["keep_terms"] = keep
    rewards["u_flat_only"] = RewardTermCfg(func=ucons.flat_only, weight=1.0, params=params)
  assert os.environ.get("U_STAND_POSE_EVEN") != "1" or os.environ.get("U_FLAT_ONLY") == "1", \
    "U_STAND_POSE_EVEN goes with U_FLAT_ONLY=1 (its term takes the cost off the uneven ground)"
  if os.environ.get("U_LOADED_FALL"):   # a probe's switch (_probe_switches), after is_terminated
    rewards["u_loaded_fall"] = RewardTermCfg(func=ucons.loaded_fall, weight=1.0,
                                             params={"scale": float(os.environ["U_LOADED_FALL"])})
  if os.environ.get("U_STAIR_SPEED_LOADED"):   # a probe's switch (_probe_switches), after stair_speed_deficit
    floor = float(os.environ["U_STAIR_SPEED_LOADED"])
    assert 0.0 <= floor < 1.0 and "stair_speed_deficit" in rewards, f"U_STAIR_SPEED_LOADED {floor}: 0 <= x < 1"
    params = {"floor": floor}
    if os.environ.get("U_STAIR_SPEED_LOADED_UP") == "1":   # only where the robot is commanded up
      params |= {"up_only": True, "sensor_name": rewards["stair_speed_deficit"].params["sensor_name"]}
    rewards["u_stair_speed_loaded"] = RewardTermCfg(func=ucons.stair_speed_loaded, weight=1.0, params=params)
  # the heading switches (probes, _probe_switches): nothing for a robot without the heading observation
  shown = "heading_error" in cfg.observations["critic"].terms
  lines = [t for t, r in rewards.items() if r.func is mdp.straight_drift]
  if os.environ.get("U_HEADING_OBS") == "1" and shown and lines:   # after the line costs it reads
    rewards["u_heading_ref"] = RewardTermCfg(func=ucons.heading_ref, weight=1e-9, params={"terms": lines})
  if os.environ.get("U_HEADING_LAG") and shown:
    w = float(os.environ["U_HEADING_LAG"])
    assert 0.0 < w <= 20.0, f"U_HEADING_LAG {w}: the cost's weight, 0-20"
    assert os.environ.get("U_HEADING_OBS") == "1", "U_HEADING_LAG goes with U_HEADING_OBS=1"
    sensor = rewards["straight_drift"].params.get("uneven_sensor_name") if "straight_drift" in rewards else None
    rewards["u_heading_lag"] = RewardTermCfg(func=ucons.heading_lag, weight=-w,
                                             params={"sensor_name": sensor or "terrain_scan"})
  # the gait clock on even ground only (a probe, _probe_switches): the legged; nothing for a robot whose clock has
  # its own ground mask already (the wheeled group's stepping_only clock) or that has no clock
  gait = rewards.get("foot_gait")
  if os.environ.get("U_GAIT_EVEN") == "1" and gait is not None and gait.func is mdp.feet_gait \
      and not gait.params.get("stepping_only") and not gait.params.get("march_only"):
    sensor = rewards["straight_drift"].params.get("uneven_sensor_name") if "straight_drift" in rewards else None
    rewards["u_gait_free"] = RewardTermCfg(func=ucons.gait_free, weight=1.0, params={
      "sensor_name": sensor or "terrain_scan", "command_name": gait.params["command_name"],
      "command_threshold": gait.params["command_threshold"]})
  # the clock item's floor on the commands to move only (a probe, _probe_switches: U_CLOCK_FLOOR=<x>,move): a march
  # in place keeps the clock at the calm weight; after foot_gait, any robot with a clock
  if os.environ.get("U_CLOCK_FLOOR", "").endswith(",move") and gait is not None and gait.func is mdp.feet_gait:
    sensor = rewards["straight_drift"].params.get("uneven_sensor_name") if "straight_drift" in rewards else None
    rewards["u_gait_march"] = RewardTermCfg(func=ucons.gait_march_calm, weight=1.0, params={
      "sensor_name": sensor or "terrain_scan", "even_only": "u_flat_only" in rewards,
      "command_name": gait.params["command_name"]})
  rewards["u_meter"] = RewardTermCfg(func=ucons.meter, weight=1e-9, params={"events": METER_EVENTS[kind]})
  # the stop creep on the acceptance's time base (after standing_position_hold: it reads that term's clock)
  rewards["u_stop_creep"] = RewardTermCfg(func=ucons.stop_creep, weight=1e-9, params={})
  if os.environ.get("U_STOP_STILL") == "1":
    # the stop on a flight on tier 2's time base (the stop_still items): the reference 2 s after the stand began,
    # counted 3.5-4 s after it began
    rewards["u_stairs_stop"] = RewardTermCfg(func=ucons.stop_creep, weight=1e-9, params={
      "key": "u_stairs_stop", "settle_s": 2.0, "window_s": 2.0, "min_window_s": 1.5})
  # the march's drift the same way (after march_position_hold) and the hips' reset over a march
  rewards["u_march_drift"] = RewardTermCfg(func=ucons.stop_creep, weight=1e-9, params={
    "key": "u_march_drift", "term": "march_position_hold", "settle_s": 2.0, "window_s": 5.0, "min_window_s": 4.0,
    "repeat": True})
  rewards["u_march_reset"] = RewardTermCfg(func=ucons.march_reset, weight=1e-9, params={})
  # the shared stairs metrics (tinynav's counters) over 512 robots, for the "lib:" constraints
  rewards["u_lib_meter"] = RewardTermCfg(func=ucons.lib_meter, weight=1e-9,
                                         params={"wheeled": kind == "wheeled", "sample": 512})
  return constrained, other


def _curriculum(cfg: ManagerBasedRlEnvCfg, robot: URobot, items: list[dict], constrained: list[str],
                dropped: list[str], quality: tuple[dict, dict] = ({}, {}),
                other_shares: dict[str, float] | None = None) -> None:
  old = cfg.curriculum
  stages = old["command_vel"].params["velocity_stages"]
  keys = set(unified_recipe(robot)["P"])
  final_share = 0.5 if "stairs50" in keys else 0.65
  new: dict[str, CurriculumTermCfg] = {"u_progress": CurriculumTermCfg(func=ucur.progress, params={})}
  # the training records' count of every finished episode (diagnostics.py): before the terms that redraw the robot's
  # column and row, so it still sees where the episode ran
  new["u_episodes"] = CurriculumTermCfg(func=diagnostics.episodes,
                                        params={"stair_terrain_names": MAP3_STAIR_TERRAIN_NAMES})
  for name, term in old.items():
    if name == "command_vel":
      first, last = stages[0], stages[-1]
      new["u_commands"] = CurriculumTermCfg(func=ucur.progress_commands, params={
        "command_name": "twist",
        "initial": {k: first[k] for k in ("lin_vel_x", "lin_vel_y", "ang_vel_z") if first.get(k) is not None},
        "final": {k: last[k] for k in ("lin_vel_x", "lin_vel_y", "ang_vel_z") if last.get(k) is not None},
      })
      # the lateral range grows with the progress (curriculum.progress_commands): by robot kind (LAT_RAMP);
      # U_LAT_RAMP="start,p_from,p_to" or "off" overrides it (the run's records list the variable)
      lat = LAT_RAMP.get(robot.kind)
      if os.environ.get("U_LAT_RAMP"):
        v = os.environ["U_LAT_RAMP"]
        lat = None if v == "off" else tuple(float(s) for s in v.split(","))
        if lat is not None and len(lat) != 3:
          raise ValueError(f"U_LAT_RAMP: 3 values (start,p_from,p_to) or off, got {v!r}")
      if lat is not None:
        new["u_commands"].params["lateral_ramp"] = lat
    elif name == "stairs_share":
      new["u_stairs_share"] = CurriculumTermCfg(func=ucur.progress_stairs_share, params={
        "stair_terrain_names": MAP3_STAIR_TERRAIN_NAMES, "start": TEACHER_STAIRS_SHARE, "end": final_share,
        "other_shares": other_shares,
      })
    elif name == "constraints":
      continue   # the recipe's constraint set; the teacher's own comes last
    elif name == "ramps":
      skip = set(constrained) | set(dropped)
      ramps = [r for r in term.params["ramps"] if not (r["kind"].startswith("reward") and r["name"] in skip)]
      new[name] = CurriculumTermCfg(func=term.func, params={**term.params, "ramps": ramps})
      # the quality stage and the recipe's iteration ramps must not drive the same weight or parameter
      q_weights, q_params = quality
      clash = [r for r in ramps if (r["kind"] == "reward_weight" and r["name"] in q_weights)
               or (r["kind"] == "reward_param" and r["key"] in q_params.get(r["name"], {}))]
      assert not clash, f"{robot.name}: ramped by the recipe and by the quality stage: {clash}"
    else:
      new[name] = term
  payload = cfg.events["base_payload"].params
  payload["ramp_steps"] = 0   # the event's own iteration ramp off: the progress sets the limit
  new["u_payload"] = CurriculumTermCfg(func=ucur.progress_payload, params={"event_name": "base_payload",
                                                                         "full_max": payload["heavy_max"]})
  if quality[0] or quality[1]:   # the M -> P reward differences, along the stairs quality q
    new["u_quality"] = CurriculumTermCfg(func=ucons.quality_terms, params={"weights": quality[0], "params": quality[1]})
  new["u_constraints"] = CurriculumTermCfg(func=ucons.constraints, params={"items": items})
  new["u_heavy"] = CurriculumTermCfg(func=ucons.heavy_log, params={})   # light vs heavy readings, logged only
  if os.environ.get("U_SMOKE") == "1":
    # A few-iteration smoke run (jobs/unified/teacher.sh --smoke): measure from the start, multipliers move after a
    # few samples, the meter reports after 50 steps; the progress is forced along so every switch-on runs.
    new["u_progress"].params = {"skip_steps": 0, "ema_steps": 50.0, "force_rate": 0.002, "force_q": True}
    new["u_constraints"].params.update({"skip_steps": 0, "warmup": 5, "ema_steps": 50.0})
    new["u_heavy"].params.update({"skip_steps": 0, "ema_steps": 50.0})
    cfg.rewards["u_meter"].params["min_n"] = 50
    cfg.rewards["u_lib_meter"].params.update({"min_steps": 5.0, "ema_steps": 200.0})
    if "u_flat" in cfg.rewards:
      cfg.rewards["u_flat"].params.update({"min_n": 2, "min_q": 2})
  cfg.curriculum = new


def _probe_switches(cfg: ManagerBasedRlEnvCfg, items: list[dict]) -> None:
  """Switches for the probes (2026-11-12: the Go2W teachers' returns went from +28 to -29 within 400 iterations of the
  stairs quality stage -- the slide and idle multipliers at 2.7 and 1.6, the episodes ended by an illegal contact
  0.26 -> 0.93 -- and each probe asks about one cause). All off unless set; a run's records list every U_* variable
  (diagnostics.py):
    U_Q_SCALE_MAX=<x>    the stairs-quality constraints' multipliers stay at or below x (1: the recipe's P weights);
    U_TERMINATED_X=<k>   an episode ended by a fall or an illegal contact costs k times as much (is_terminated);
    U_PAYLOAD_Q=0        the payload limit does not follow the stairs quality (curriculum.progress_payload);
    U_Q_LR=<x>           the stairs-quality multipliers' step per update (constraints.LR 0.003: 1 -> 3 in ~370
                         iterations, and the costs grew fivefold in 200 iterations with the behaviour unchanged);
    U_Q_SPAN=<it>        the iterations the stairs quality takes for 0 -> 1 (curriculum.Q_SPAN_IT);
    U_TERMINATED_Q=<x>   the same cost, but along the stairs quality: 1x at q 0 -> <x> times at q Q_IN, with the P
                         weights (constraints.quality_terms) -- before the quality stage a fall costs what it did,
                         so the robots still try the rows they fail on;
    U_ENTROPY_Q=<x>      the entropy bonus along the stairs quality, the task's -> <x> (runner._entropy_by_q: with a
                         larger fall cost the action std rose 0.72 -> 1.00 in 450 iterations);
    U_STD_CAP=<x>        a cap on the action noise (the Gaussian's std, per action) along the stairs quality, the
                         task's initial std -> <x> (runner._std_cap_by_q: at the end of the probes the std sat at
                         0.47 (Go2) / 0.50 (Go2W) and rose with the bonus at 0.002; the delivered policies' 0.34 /
                         0.43);
    U_STAIRS_P=<a>,<b>   the progress over which the stairs share of the training robots goes from its start to its
                         end (curriculum.progress_stairs_share: 0.3,0.8 -- the from-zero Go2 teacher's climbing row
                         fell 2.0 -> 0.55 over iterations 400-2500 at the start share and rose only with the share);
    U_Q_ROW=<row>        the climbing row the stairs quality waits for (curriculum.Q_ROW 3: the old flow put the P
                         weights on a policy at rows 5-6);
    U_Q_HEALTH=<k>       the stairs quality by health (curriculum.progress, q_back): it waits while the light
                         fixed-row fail rate sits above its reference and goes back at k times its pace when the
                         constraints would relax;
    U_METER_EMA=<steps>  the stairs meters' memory in env steps (constraints.meter and lib_meter: 20000 = 833
                         iterations, so a reading had covered a third of a change of behaviour by the time its
                         multiplier was at the cap);
    U_LOADED_FALL=<k>    a fall or an illegal contact costs a loaded robot more: 1x at a quarter of the payload limit
                         -> k times at the limit (constraints.loaded_fall, set up in _rewards: the heavy evaluation
                         robots failed three times as many fixed-row stairs episodes as the light ones);
    U_PAYLOAD_STICKY=<x> the payload class (light / mid / heavy) stays with the robot and is drawn anew with
                         probability x at a reset (sticky_payload; unset: the payload anew at every reset).
    U_FLAT_ONLY=1        a flat item's multiplier presses on flat ground only: over uneven ground its terms stay at
                         the configured weight or below (constraints.flat_only, set up in _rewards: the flat line
                         item at its 3x cap cost the climbing Go2 1.5 a second, as much as its quality costs).
    U_FLAT_ONLY_KEEP=<term>[,<term>]   with U_FLAT_ONLY=1: these reward terms keep their flat items' multipliers
                         over uneven ground too (flat_only's keep_terms). 2026-11-13, the Go2W's formal teachers
                         from zero: the stop in the middle of a flight (tier 2: still moving 2-4 s after the zero
                         command going up, falls after a stop going down with the full payload) -- the probe
                         U_FLAT_ONLY_KEEP=standing_position_hold,standing_speed (the creep item's terms).
    U_STOP_STILL=1       logged only: the stop in the middle of a flight as tier 2's stop run judges it -- four
                         items stop_still_{up,down}[_heavy], the share of the robots that moved less than 5 cm
                         between 2 s and 4 s after the zero command (the gate asks 95%), nose up / nose down, up to
                         the common payload / in the heavy class (constraints._stairs_stop_still; a second
                         stop_creep term, u_stairs_stop). The training had no reading of it: the Go2W's formal
                         teachers failed tier 2 on it four times with every training reading in order.
  The descent (2026-11-13: tier 2 on the Go2 candidate, s29 model_5598 -- four feet down in 0.57-0.75 of the stops
  going down the 8-18 cm steps and the descents at 0.73-0.81x the command, where training read 0.93 and 0.9):
    U_STOP_FEET_DIR=<x>  stop_feet as two items, stop_feet_up / stop_feet_down: the robots stopped on the flight
                         nose up / nose down (constraints._stairs_stop_feet `facing`; the pooled item counts both
                         and the robots stopped on a stairs column's level ground). Both start at the multiplier
                         <x> with the cap max(3, x), whatever U_Q_SCALE_MAX says (read in constraint_items);
    U_STAIR_SPEED_H=1    the stairs speed over the ground instead of the body frame's -- along a 30 deg flight the
                         body's x is 15% more than the ground covered, which is what the acceptance measures: the
                         descent / climb items read constraints._stair_speed_h (read in constraint_items) and the
                         cost term stair_speed_deficit becomes constraints.stair_speed_deficit_h.
    U_ITEM_CAP=<item>:<cap>,...  the named stairs-quality items' multipliers may rise to their own cap, whatever
                         U_Q_SCALE_MAX says (read in constraint_items). 2026-11-13, the Go2 candidate with the
                         heading observation and the ground-speed climb (uprobe_s47): the climb item read 0.74 /
                         0.71 against its target 0.85 with its multiplier at the cap 1 for the whole probe, and
                         tier 2's blind climbs were 0.77x of the command (the bar: 0.8x);
    U_ITEM_FLOOR=<item>:<floor>,...  the named stairs-quality items' multipliers do not go below their floor (1-6),
                         their cap at least that (read in constraint_items) -- in effect from a checkpoint's first
                         update on, where U_ITEM_CAP's rise takes hundreds of iterations. 2026-11-13, tier 2 of the
                         Go2 uprobe_s62 9091 against the delivered go2_blind_v4 tile by tile: the riser kicks are
                         level on the climbs (0.353 / step against 0.358) and the whole gap is on the descents
                         (0.361 against 0.262), with the mid-tread share 0.02-0.04 lower on almost every tile
                         (0.754 against 0.778). v4 trained with its feet_edge cost at 2 times the weight; here the
                         edge / mid items hold it at U_Q_SCALE_MAX's cap of 1 with mid at 0.756 against its target
                         0.84. U_ITEM_FLOOR=mid:<x> puts feet_edge at <x> times its weight (several items on one
                         term: the strongest, constraints._apply);
    U_STAIR_APPROACH=<x> with U_STAIR_SPEED_H=1: the stairs speed cost also holds on the level ground of a climbing
                         column under a forward command (before the flight, on its landing) -- in full up to the
                         common payload, <x> times it from 2/3 of the limit up, linear in between (0.0: a heavy
                         robot may wait on the level; 1: every payload alike). Without it the cost starts within
                         0.45 m of a step, and standing before the flight costs the flat speed cost alone: the
                         Go2W package teacher learnt to stand there (constraints.stair_speed_deficit_h; the logged
                         items park / park_heavy read it);
    U_STAIR_SPEED_LOADED=<x>  going slowly on the stairs costs a heavily loaded robot less: the cost as it is up to
                         the common payload (half the limit), <x> times it from 2/3 of the limit up, linear in
                         between (constraints.stair_speed_loaded, read in _rewards); the descent / climb items read
                         the robots up to the common payload (read in constraint_items). 0: a heavy robot may stop
                         on the stairs for nothing -- the acceptance judges only its falls; with
                         U_STAIR_SPEED_LOADED_UP=1 only where it is commanded up (uprobe_s40, every direction:
                         the 15-18 cm climbs' falls at 8 kg 49% -> 6%, the descents' 6% -> 21%);
    U_BODY_CLEAR=<w>     the cost body_clear at the weight -<w>: a point of the bodies whose touch ends the episode
                         (trunk, hips, the As2's thigh roots) within 3 cm over the terrain or of a step's top next
                         to it (constraints.body_clear, read in _add_body_clear) -- the climbs' fails are the chin
                         / the thigh roots on a step, with a payload the more; U_BODY_CLEAR=<w>,<margin>[,<reach>]
                         with another margin (m) and reach towards a step face (m, 0.05): the Go2W at weight 50
                         (uprobe_s46) passes the stairs unloaded and with the acceptance's 8 kg still loses 28 of
                         576 on the climbs, the head (base2) on a step -- the loaded trunk rides lower than the
                         3 cm the unloaded one learned to keep;
    U_KICK_KEEPOUT=<w>   legged: the cost riser_keepout at the weight -<w> -- a swinging foot close to a step face
                         and below its top (constraints.riser_keepout) -- as one of the kicks item's costs (read in
                         _add_riser_keepout and constraint_items); with U_KICK_KEEPOUT_DIR=1 only the faces the
                         foot is moving towards (uprobe_s39 at weight 4 charged a descent's swing for the face
                         behind it: the seen mode stopped at the top of 24% of its descents);
    U_M_WEIGHT=<term>:<f>,...  the weight of <term> before the stairs quality (q = 0) times <f>; the P weight from
                         q = Q_IN on as before (m_weights, read in unified_teacher_env_cfg).
    U_STAND_POSE_EVEN=1  with U_FLAT_ONLY=1: standing_pose_load (the flat stance item's cost: a resting robot's leg
                         joints off the default pose, times 1 + payload / 8 kg) acts on even ground only
                         (constraints.flat_only's zero_terms, read in _rewards) -- on a flight it was the largest
                         cost of a loaded robot's stop (4.2-5.4 a second), and more than half of the full payload's
                         falls in tier 2 were stops on a descent.
  The heading on even ground (2026-11-13: tier 2 on the Go2 candidate, s34 model_6097 -- 12-17 deg off 10 s into a
  line to the right, line drift 15-19% against 5%, the compound commands' yaw at 46-86%; read in _rewards):
    U_HEADING_OBS=1      the line costs (straight_drift, side_drift) hold the heading the actor is shown -- the
                         heading observation's target, kept across the commands -- instead of the heading at each
                         command's start (constraints.heading_ref), on even ground and on the stairs;
    U_HEADING_LAG=<w>    the cost u_heading_lag at the weight -<w>: under a command to move, on even ground,
                         the base heading's distance from the heading the commands have turned to, over the
                         observation's whole range (constraints.heading_lag; with U_HEADING_OBS).
  The flat protocols (flat.py; U_FLAT_SCRIPTS=0 -- no scripts, the flat items as before -- is read where the items
  and the tracker are set up, not here):
    U_FLAT_SHARE=<l>[,<h>] the shares of the light / heavy evaluation robots on the flat column under a script
                         (flat.FLAT_SHARE; 0: none of that class);
    U_FLAT_AXIS=<share>  how many of the evaluation robots on the flat column start along / across the strip
                         (flat.flat_reset_yaw, 0.75; 0: every start heading as drawn -- few 10 s lines fit);
    U_FLAT_NEW_MAX=<x>   the cap of the new flat items' multipliers (FLAT_NEW; 2);
    U_DR_LIGHT=1         the training task without the joint friction / damping / armature draws and without the
                         stick-slip (unified_teacher_env_cfg; the checks' -Flat / -Stairs tasks and the students'
                         tasks keep them, whatever the environment says). 2026-11-13: the delivered go2_blind_v4
                         trained without them and passes the flat items under them with a margin the unified
                         teacher lacks (stop creep 0.3-0.5 cm against 0.7-1.5, the strafe's gait clock 95 % against
                         68-79 %, stance p90 0.125 rad against 0.13-0.17) on the same reward settings -- do these
                         draws in training cost the flat precision?
    U_CREEP_CAP=<x>      the cap of the creep item's multiplier (3; it scales standing_position_hold and
                         standing_speed). 2026-11-13, tier 2: the stop creep is the one flat item both robots
                         miss with the multiplier at its cap -- Go2 uprobe_s51 7095 a median of 1.07-1.51 cm after
                         1.0 / 1.5 m/s (bar 1), Go2W uprobe_s39 5995 1.34-1.79 seen and 2.86 blind. No stepping,
                         no wheel roll (stop_anatomy on the Go2W: the wheels' roll 0.00 cm, the base over the
                         wheels 1.2-2.0 cm with the thighs still turning 0.02 rad from 2 to 6 s after the stop):
                         the posture settles out of the braking lean for seconds, and at 3 times the weight the
                         hold charges 0.3 a second for a centimetre. A cap over 3 did not help (Go2W uprobe_s45,
                         cap 10). A cap of 1-3 (2026-11-13 13:35): the delivered Go2W blind model on the same flat
                         test (go2w_bl2_g14_13600: both costs at their weight, no multiplier) stops from
                         -1.0 m/s in 0.74 s and creeps 0.86 cm, uprobe_s51 7991 in 0.42 s and 2.12 cm (forwards
                         0.49 s / 0.43 cm against 0.35 s / 0.43-1.27 cm) -- does 3 times standing_speed buy the
                         hard brake whose lean then settles as the creep the item measures (a loop that pins
                         the multiplier at its cap)?
    U_CLOCK_FLOOR=<x>    the floor of the clock item's multiplier (1; it scales foot_gait on even ground), its cap
                         at least that. 2026-11-13, tier 2 of the Go2 uprobe_s59 8592: the gait clock is the one
                         flat item it misses -- the strafe (0, +-0.6, 0) 62-69 % in step and (0.6, +-0.4, 0)
                         73-79 % (bar 80 %; go2_blind_v4 95 % at the weight alone, without payload and the joint
                         draws) -- with the multiplier at 1.56 after 8600 iterations: the training's worst bin
                         reads 0.78-0.85 against its target 0.84, a shortfall that moves the multiplier by 0.04
                         in 200 iterations. `U_CLOCK_FLOOR=<x>,move` (read in _rewards too): the floor presses on
                         the commands to move only -- a robot marching in place earns the clock at the calm weight
                         (constraints.gait_march_calm). uprobe_s63 9091 (floor 3), tier 2: strafes 92-94 % and
                         diagonals 89-93 % in step, and the blind march's drift 26.9 cm against s59's 12.6;
    U_FLAT_DIAG_AXIS=<m> the flat speed deficit judges a diagonal command (both parts >= <m> m/s) per axis, as the
                         acceptance does (mdp.stair_speed_deficit axis_min; the wheeled have it at 0.2 from the old
                         flow's "diagaxis", nothing changes for them). Go2 uprobe_s42 / s43 6596: the diagonal's
                         sideways part at 72-77 % in training and (0.6, +-0.4) y 74-76 % in tier 2 (bar 80 %),
                         the speed along the command inside the 0.9-1.1 band, so nothing was charged;
    U_FLAT_SPEED_W=<w>   the flat speed deficit at the weight -<w> where it is lighter (the legged: -2; the wheeled
                         have -4 from the old flow's "speedw4", nothing changes for them). Go2 uprobe_s43 6596
                         (U_HEADING_OBS=1), tier 2: seen every forward command 90 % and more, blind (0.5, 0, 0) and
                         (1.0, 0, 0) and their arcs x 75-79 % (bar 80 %), the speed_lin multiplier at its cap.
  The gait clock (2026-11-13: the Go2 teacher's riser kicks stay at 0.41-0.49 a step in tier 2 from 5598 to 7095
  iterations, seen and blind alike, bar 0.31; read in _rewards):
    U_GAIT_EVEN=1        the legged gait clock is judged on even ground only: over uneven ground a robot under a
                         command to move earns foot_gait in full whatever its feet do (constraints.gait_free). The
                         clock is a 0.6 s trot below 0.8 m/s, so the stride is speed x 0.6 s (0.27 m at the climbing
                         speed) against treads of 0.25 / 0.32 m: the footholds drift across the edges. Nothing for
                         the wheeled (their clock steps on even ground only already).
  Waiting above a descent (2026-11-13: the Go2 candidates' seen mode from 6600 to 7600 iterations -- tier 2 68-100%
  stalled at the top of the descents from 8 cm up, unloaded and at the common payload; read in _add_stair_park and
  constraint_items):
    U_STAIR_PARK=<w>[,<heavy>]   the cost constraints.stairs_park at the weight -<w> (0-20): standing on a
                         descending column under a forward command, 1 at a standstill to 0 at half the commanded
                         speed; in full up to the common payload, <heavy> times it (0) for the heavy class. The
                         logged item park_down becomes enforced (bar 5% of those robots' time) and scales it, its
                         multiplier up to PARK_CAP whatever U_Q_SCALE_MAX says. The account it answers (standing
                         pays 3.6 a second over descending) is in the cost's docstring.
  The falls with payload (2026-11-13: forwards at and after the stops on the descents, on the Go2, the Go2W and the
  As2 alike -- tier 2 with the acceptance's payload; read in _add_com_support and constraint_items):
    U_COM_SUPPORT=<w>[,<moving>]   the cost constraints.com_support at the weight -<w> (0-30): over uneven ground,
                         the whole robot's centre of mass (the payload with it) off the centre of its four feet /
                         wheels over the ground -- nothing within 0.12 of the support's size (Go2: 3 cm), 1 from 0.92
                         of it on (23 cm); in full at the zero command, <moving> times it (0) under a command. On top of
                         standing_pose_load, not instead of it (the cost's docstring). Logs the items com_stop /
                         com_stop_heavy: that distance (m) at the stops on the flights, up to the common payload
                         and in the heavy class."""
  v = os.environ.get("U_Q_SCALE_MAX")
  if v:
    for c in items:
      if c["by"] == "q" and not c.get("own_cap"):
        c["max_scale"] = max(c["min_scale"], min(c["max_scale"], float(v)))
        c["start_scale"] = min(c["start_scale"], c["max_scale"])
  if os.environ.get("U_STAIR_SPEED_H") == "1":
    term = cfg.rewards["stair_speed_deficit"]
    assert getattr(term.func, "__name__", "") == "stair_speed_deficit" and not {*term.params} - {
      "command_name", "sensor_name", "ratio", "min_speed", "ratio_high", "axis_min", "max_payload"}, term
    term.func = ucons.stair_speed_deficit_h
  v = os.environ.get("U_STAIR_APPROACH")
  if v:
    assert os.environ.get("U_STAIR_SPEED_H") == "1" and 0.0 <= float(v) <= 1.0, \
      f"U_STAIR_APPROACH {v}: the heavy class's share 0-1, with U_STAIR_SPEED_H=1 (the cost that has it)"
    cfg.rewards["stair_speed_deficit"].params["approach_heavy"] = float(v)
  v = os.environ.get("U_TERMINATED_X")
  if v:
    cfg.rewards["is_terminated"].weight *= float(v)
  v = os.environ.get("U_Q_LR")
  if v:
    for c in items:
      if c["by"] == "q":
        c["lr"] = float(v)
  v = os.environ.get("U_METER_EMA")
  if v:
    for name in ("u_meter", "u_lib_meter"):
      cfg.rewards[name].params["ema_steps"] = float(v)
  v = os.environ.get("U_PAYLOAD_STICKY")
  if v:
    ev = cfg.events["base_payload"]
    assert ev.func is mdp.base_payload_point and 0.0 <= float(v) <= 1.0, f"U_PAYLOAD_STICKY {v}: a probability"
    ev.params = {**ev.params, "inner": ev.func, "redraw": float(v)}
    ev.func = sticky_payload
  # -- the flat protocols (flat.py)
  if "u_flat" in cfg.rewards:
    v = os.environ.get("U_FLAT_SHARE")
    if v:
      share = [float(x) for x in v.split(",")]
      assert 1 <= len(share) <= 2 and all(0.0 <= x <= 1.0 for x in share), f"U_FLAT_SHARE {v}: one or two shares"
      cfg.rewards["u_flat"].params["share"] = (share[0], share[1] if len(share) > 1 else uflat.FLAT_SHARE[1])
    v = os.environ.get("U_FLAT_AXIS")
    if v and cfg.events["reset_base"].func is uflat.flat_reset_yaw:
      cfg.events["reset_base"].params["flat_share"] = float(v)
  v = os.environ.get("U_FLAT_NEW_MAX")
  if v:
    for c in items:
      if c["name"] in FLAT_NEW:
        c["max_scale"] = max(c["min_scale"], float(v))
  v = os.environ.get("U_CREEP_CAP")
  if v:
    assert 1.0 <= float(v) <= 30.0, f"U_CREEP_CAP {v}: the creep item's cap, 1-30"
    for c in items:
      if c["name"] == "creep":
        c["max_scale"] = float(v)
  v = os.environ.get("U_CLOCK_FLOOR")
  if v:
    x, *rest = v.split(",")
    assert 1.0 <= float(x) <= 6.0 and rest in ([], ["move"]), f"U_CLOCK_FLOOR {v}: the clock item's floor, 1-6[,move]"
    for c in items:
      if c["name"] == "clock":
        c["min_scale"], c["max_scale"] = float(x), max(c["max_scale"], float(x))
  v = os.environ.get("U_FLAT_DIAG_AXIS")
  if v and "flat_speed_deficit" in cfg.rewards:
    assert 0.05 <= float(v) <= 0.5, f"U_FLAT_DIAG_AXIS {v}: the smallest part of a diagonal command, m/s"
    term = cfg.rewards["flat_speed_deficit"]
    if "axis_min" not in term.params:   # the wheeled group's own (0.2) stays
      term.params = {**term.params, "axis_min": float(v)}
  v = os.environ.get("U_FLAT_SPEED_W")
  if v and "flat_speed_deficit" in cfg.rewards:
    assert 2.0 <= float(v) <= 8.0, f"U_FLAT_SPEED_W {v}: the cost's weight, 2-8"
    term = cfg.rewards["flat_speed_deficit"]
    if -term.weight < float(v):   # a heavier one stays (the wheeled group's -4)
      term.weight = -float(v)


def _fixed_payload(cfg: ManagerBasedRlEnvCfg) -> None:
  """The checks' payload (2026-11-12, doc/redesign.md 2.1): a fixed point mass from U_PAYLOAD ("kg,x,y,z", default
  0 kg -- the unloaded condition; the acceptance sets the walking payload for the loaded one) instead of the training
  draw. The other randomization (CoM offset, friction, gains, ...) stays, the same in every condition.

  "kg,x,y,z,dx,dy,dz" (the typical payload, the user 2026-11-12 "加一档「常用负载」"; jobs/accept_payload.sh R typical):
  the mass is fixed, the mount point is drawn for each robot at its reset, uniform in x +- dx, y +- dy, z +- dz --
  where the payload will sit on the finished robot is not known."""
  v = [float(s) for s in os.environ.get("U_PAYLOAD", "0,0,0,0").split(",")]
  if len(v) not in (4, 7):
    raise ValueError(f"U_PAYLOAD: 4 values (kg,x,y,z) or 7 (kg,x,y,z,dx,dy,dz), got {len(v)}")
  m, x, y, z = v[:4]
  dx, dy, dz = v[4:] if len(v) == 7 else (0.0, 0.0, 0.0)
  # the whole event replaced: on the check terrains the blind builder keeps mjlab's base-mass draw (-1..+2 kg), not
  # the training's payload point (2026-11-12)
  cfg.events["base_payload"] = EventTermCfg(
    mode="reset", func=mdp.base_payload_point,
    params={"asset_cfg": SceneEntityCfg("robot"), "light_range": (m, m), "heavy_max": 0.0, "heavy_share": 0.0,
            "pos_ranges": ((x - dx, x + dx), (y - dy, y + dy), (z - dz, z + dz)), "base_com_range": 0.0,
            "ramp_steps": 0},
  )


def unified_teacher_env_cfg(robot: URobot, play: bool = False, lean: bool = False,
                            student: str | None = None, terrain: str = "map3") -> ManagerBasedRlEnvCfg:
  """The robot's task under its recipe (recipe.py, 2026-11-13: every U_ switch read below takes the robot's value
  from the table unless the environment sets it itself; U_RECIPE=0 -- the environment alone, as the probes ran).
  student "blind" / "depth" (students.py): the teacher's environment with the scan fixed off / on, the curriculum
  at its end (progress 1) and the constraint scales frozen at the teacher's (restored from its checkpoint)."""
  with urecipe.scope(robot):
    return _teacher_env_cfg(robot, play, lean, student, terrain)


def _teacher_env_cfg(robot: URobot, play: bool, lean: bool, student: str | None, terrain: str) -> ManagerBasedRlEnvCfg:
  cfg = base_env_cfg(robot, play=play, terrain=terrain)
  metrics_mj.add_sensor(cfg)   # the shared metrics' contact sensor (points and normals)
  _add_privileged(cfg)
  _add_scan_mask(cfg)
  # U_DR_LIGHT=1 (a probe's switch, _probe_switches): the training task alone goes without these two
  if not (os.environ.get("U_DR_LIGHT") == "1" and terrain == "map3" and not play and student is None):
    _add_joint_dr(cfg)
    stick_slip.add(cfg)   # after the sensor and the privileged group
  other_shares = None
  if terrain == "map3":
    _payload_point(cfg, robot)
    other_shares = _add_slopes(cfg)
    _add_riser_keepout(cfg, robot)
    _add_body_clear(cfg, robot)
    _add_stair_park(cfg)
    _add_com_support(cfg)
    items = constraint_items(robot.kind)
    constrained, other = _rewards(cfg, robot.kind, items, lean)
    if not play:
      by_q = {t for c in items if c["by"] == "q" for t in c["terms"]}
      by_p = {t for c in items if c["by"] == "p" for t in c["terms"]}
      assert not by_q & by_p, f"terms scaled by a flat and a stairs-quality constraint: {sorted(by_q & by_p)}"
      weights, params, start = quality_diff(robot, cfg.rewards, by_p, by_q, tuple(other) if lean else (),
                                            play=play, terrain=terrain)
      for c in items:
        if c["by"] == "q":
          c["start"] = {t: start[t] for t in c["terms"]}
          c["base"] = {t: cfg.rewards[t].weight for t in c["terms"]}
      for t in m_weights(m_weight_factors(), items, weights, cfg.rewards, by_p):   # (_probe_switches)
        if (robot.name, t) not in _M_WEIGHT_NOTED:
          _M_WEIGHT_NOTED.add((robot.name, t))
          print(f"[unified] U_M_WEIGHT: {robot.name} has no cost {t} -- nothing changed for it")
      if uflat.scripts_on():
        _flat_tracker(cfg)
      _probe_switches(cfg, items)
      if os.environ.get("U_TERMINATED_Q"):   # (_probe_switches)
        assert "is_terminated" not in weights, "is_terminated: already an M -> P difference of the recipe"
        w = cfg.rewards["is_terminated"].weight
        weights["is_terminated"] = (w, w * float(os.environ["U_TERMINATED_Q"]))
      _curriculum(cfg, robot, items, constrained, other if lean else [], (weights, params), other_shares)
      if os.environ.get("U_STAIRS_P"):
        a, b = (float(v) for v in os.environ["U_STAIRS_P"].split(","))
        assert 0.0 <= a < b <= 1.0, f"U_STAIRS_P: 0 <= a < b <= 1, got {a}, {b}"
        cfg.curriculum["u_stairs_share"].params.update(p_start=a, p_end=b)
      if os.environ.get("U_Q_SPAN"):   # (_probe_switches; the curriculum's terms exist from here on)
        cfg.curriculum["u_progress"].params["q_span_it"] = int(os.environ["U_Q_SPAN"])
      if os.environ.get("U_Q_ROW"):
        cfg.curriculum["u_progress"].params["q_row"] = float(os.environ["U_Q_ROW"])
      if os.environ.get("U_Q_HEALTH"):
        cfg.curriculum["u_progress"].params["q_back"] = float(os.environ["U_Q_HEALTH"])
  # else: a check's terrain ("flat", "stairs"; the blind builder leaves out the training schedule there): the
  # observations, actions and physics are the training ones, the rewards and curriculum are not used
  if terrain != "map3":
    _fixed_payload(cfg)
  if student is not None:
    cfg.events["u_scan_mask"].params["mode"] = "off" if student == "blind" else "on"
    if "u_progress" in cfg.curriculum:
      cfg.curriculum["u_progress"].params["fixed_p"] = 1.0
      cfg.curriculum["u_constraints"].params["frozen"] = True
  return cfg
