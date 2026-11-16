# Go2W experiment log

One row per training run: what changed, why, against which control, the gates' verdict at 6000, and the conclusion.
This is the record of every change's reason and result. Add a row when a run is launched and fill it when its gates
are in.

- Started 2026-11-16 by go2-depth on go2-home's request. go2-home leads the experiments; its rows are added as it
  sends them.
- Gates: the teacher stairs and flat gates on the average of the last 4 checkpoints at 6000 (avg4), judged under
  the acceptance specification of 2026-11-16 (doc/acceptance_go2_go2w.md; accept_judge.SPEC, main >= 3f83885).
- Conclusion: **IN** (the change goes into the recipe), **OUT** (it does not), **open** (not judged yet).

## Reference

The model to come close to is the deployed lineage g27b (blind) → w27d (depth). Re-judged 2026-11-16 with the spec
judge (go2-home), w27d passes the stairs and flat gates on its own legacy robot model and on the aligned (uros) model.
How it was trained, stage by stage: go2-depth's lineage table, built from each run's params/env.yaml.

## Reference runs (judged, not trained)

| date | model | where | physics | verdict under the spec | conclusion |
|---|---|---|---|---|---|
| 11-16 ~18:20 | w27d (deployed Go2W depth; go2-home) | go2-home's card, tree uprobe_f40fd87; verdict logs/checks/u_go2w_base_w27d_verdict.txt, seed repeats in _s1 / _s2 | the legacy model it trained on (old tree, built-in actuators) | everything passes except the full-load flat falls (bar 0), all of them backward (-1.0, 0, 0): per seed accept_matrix 2/48, gait_fb 1-2/96, gait clock 0-2/48; rear-wheel lift ~5 % there; wheels down 94.6 / 95.1 / 97.1 % passes once pooled (bar 95) | the user (11-16, "可以"): the falls bar stays 0 at full load; w27d stays deployed as the reference |
| 11-16 ~22:40 | w27d on the new items 21-30 (go2-home) | go2-home's card; verdict uprobe_c3075af/logs/checks/u_go2w_ref_w27d_verdict.txt (judge dffddbf) | legacy model, built-in actuators | unloaded / typical / full: 21 wall 15.1 / 7.3 / 5.8 % (the old 0.85 m drive-up; to rerun on the guard profile); 22 push 1.0 m/s falls 0 / 5 / 63 of 768; 23 slopes reach 96.9 % (1 fall) / 92.2 % (11) / 87.5 % (16); 24 camera errors pass but 1 full-load stairs fall; 26 descent pitch 15-18 cm 51.1 / 48.7 / 38.5 deg/s, climb roll <= 25; 27 shin strikes <= 1.2 % (pass); 28 top landing 3.21 / 2.82 / 2.95 per climb; 29 straight ~0.04 cm / 0.04 deg, sideways 0.6 m/s 0.5-0.9 cm / 3.5-4.1 deg; 30 speed CV <= 0.25 (pass) | the deployed reference fails 21, 22 (typical, full), 23, 24 (1 fall), 26 (15-18 cm), 28; the user fixed 29 (straight 1 cm / 2 deg, sideways 1 cm / 5 deg) and 30 (0.3) from these readings |
| 11-16 ~22:35 | the Go2W teacher pick (avg4 5000 of the formal v2 seed 0) on 21 / 23 / 26-30 (go2-depth) | t1, 16 per flight / cell, unloaded | gom8010-less (U_MOTOR=builtin) | 21 22.4 % (drive-up 22.4, turn 5.6; the guard profile); 23 turns 0 falls, 0 stuck, swing p50 / p90 4.3-5.0 / 6.6-7.4 cm; slopes 98.4 % / 1 fall (64 per cell); 26 pass (max 41.4 deg/s); 27 15-18 cm down rear 19.0 % (fail); 28 3.38 per climb; 29 sideways 0.6 pitch 2.8-3.0 deg; 30 CV 0.07-0.16 | small samples; the acceptance sizes at the arms' end gates |

## Runs

| date | run | box | commit | switches | the single change and why | control | gates at 6000 (spec) | conclusion |
|---|---|---|---|---|---|---|---|---|
| 11-16 00:28 | go2w_2026-11-16_00-28-21_ufull_s0 (formal Go2W teacher v2, seed 0) | t2 | 77a29ea | the Go2W recipe (19 recipe switches; 68 reward terms) | — the control | — | teacher.sh picked avg4 5000 at the cap, below the gate: stairs PASS (full-load falls 6 = the teacher cap), flat FAIL gap 0.02 (scan-off sideways drift 5.2 / 5.1 %, gait clock 66-74 % on the fast gentle arcs; that flat gate drew foot friction 0.3-1.6, the Go2's default, wider than the Go2W's training 0.4-1.0 -- see Notes); avg4 at 6000 not gated yet under the spec | control |
| 11-16 ~15:50 | go2w_payA12_s0 (arm A) | go2-home | d2bf1b2 | U_PAYLOAD_TRAIN_MAX=12 | the training payload limit 12 kg (the test stays 8): g27b trained with 12 kg, so 8 kg was 2/3 of its training limit and is 100 % of ours | control s0 | open (ends ~22:40) | open |
| 11-16 15:53 | go2w_budB4096_s0 (arm B) | t1 | d2bf1b2 | 4096 robots (no switch) | twice the samples per iteration at the same iterations: g27b trained 19800 iterations at 4096 robots, we train 6000 at 2048 | control s0 | open (ends ~00:45 on 11-17) | open |
| 11-16 ~19:00 (planned) | go2w_min_s0 (minimal recipe, on the motor model) | t2 | 36d0fe8 | U_RECIPE_MIN=1 (with it U_PAYLOAD_TRAIN_MAX=12, U_HARD_FIXED=1 = U_MARCH_FIXED, U_FAST_STOP, U_PAYLOAD_FULL0, U_STAIR_STOP_HOLD, U_MARCH_FROM_SPLAY, U_FLAT_STAND_HOLD); the robot model with the motor model gom8010 (the table's default since 36d0fe8; smoke usmoke_min_motor on t1 PASS 17:35). Note: the run's records list the 19 recipe switches, but with U_RECIPE_MIN=1 only their structural parts act (U_HEADING_OBS, U_PAYLOAD_STICKY, U_STAIRS_P, U_METER_EMA, U_ENTROPY_Q: the entropy still follows q, as in the control) | the user's "从选配方开始": 18 reward terms at today's weights instead of 68; no constraint multipliers; no quality stage; the training conditions of A and D, the hard actions shaped like their tests; and the motor model (not a single factor: go2-home 11-16, the base for the next runs, no min-without-motor pair) | control s0 (judged with U_MOTOR=builtin) | open | open |

## Switches written, not run

| commit | switches | what | status |
|---|---|---|---|
| 72592f5 | U_HARD_FIXED=1 (U_MARCH_FIXED=0.5, U_FAST_STOP=0.75,0.3, U_PAYLOAD_FULL0=1) | arm D: the hard actions at full share from iteration 0, as g27b / w27d trained | not run as a single factor on the 68-term recipe (go2-home 11-16, after the user's question); part of the minimal recipe |
| 72592f5 | U_CMD_WIDE=1.25 | arm C1: the final command ranges x 1.25 | not run (same reason) |
| 72592f5 | U_DR_WIDE=1.25 | arm C2: the randomization's spread x 1.25 | not run; the Go2W's training DR already equals the test's except the push |

## Open items

- Stair turns (go2-home 11-16, piece 5 of the shape survey): training has stair_turn_prob 0.0, while the map3 tour and
  the F1 landing test turn in place on the landings. Skipped for now: g27b passed F1 without it.
- The motor model gom8010 (the user: the motor and joint values from Unitree's own sources; then "可以不用那么准确，
  取一个中间值就行了或者平均", and it is the robot model, not an experiment: realistic motors may lower the sim scores
  while helping the real robot, so it is not judged by an A/B):
  - parameters only, the built-in position actuators unchanged (no torque-speed curve, no explicit PD);
  - torque limits: hip / thigh 21.8 N m = the mean of the GO-M8010-6's 20.2 (up to the corner) and 23.4 (stall); calf
    32.7 (x 1.5, the Go2W's calf linkage); wheel 21.8. Before: 23.7 / 23.7 / 35.55 / 23.7;
  - every leg joint: friction loss 0.2 N m, passive damping 0.1 N m s/rad, armature 0.01 kg m^2 (unitree_mujoco).
    Before: no friction or damping in the model, armature 0.01 / 0.01 / 0.02;
  - the teacher's joint randomization around them: friction and damping x 0.5-1.5, armature x 0.7-1.3. Before:
    friction and damping uniform from 0 to a share of the torque limit;
  - Go2W only (robot table: motor=gom8010 next to model=uros); the Go2 keeps its build.
  Step 1's measurement (the teacher pick, t1 logs/probe/motor/): the legs pass the 20.2 / 30.3 N m limits on under
  0.15 % of steps on flat and the unloaded stairs and ~1 % of the 8 kg climbs' hip / thigh steps; the wheels run past
  the motor's corner (13.5 rad/s) on 4-7 % of steps.
- **Rule: U_MOTOR=builtin for the checkpoints trained before the motor model.** U_MOTOR=builtin (an evaluation
  switch, like U_MODEL=legacy) builds the Go2W as before the motor model, field by field equal to the code at
  4d25c78 (scripts/unified/motor_switch_test.py). Every Go2W run before go2w_min_s0 (11-16 ~19:00, the first on
  36d0fe8) trained without the motor model: the formal teachers and students, the control, arms A and B. Their gates,
  acceptance runs, videos and exports in a tree at or after the motor model's commit must set U_MOTOR=builtin; a
  tree before it builds them that way anyway. A run whose records (run_env.sh / the timeline's env_vars) show no
  motor= in its BLIND_PARAMS trained without it.
- The training terrain terrain=wide (the user via go2-home, 2026-11-16: "可以"; the 1.25x rule, step 3), a robot
  parameter like the motor model (robot table: terrain=wide next to model=uros and motor=gom8010; the Go2W only):
  - stairs (every stairs and walled sub-terrain): step_height_range 0.05-0.23 m (was 0.05-0.20; the tests reach 18 cm,
    map3's risers 18.6); the terrain curriculum's levels spread over the wider range;
  - slopes: slope_range 0-0.36 rise / run (19.8 deg; was 0.25, 14 deg; item 23 tests 15 deg);
  - random_rough unchanged (0.02-0.06 m; the test's +-3 cm is inside);
  - tread depth (the user, 2026-11-16; map3 0.219-0.35 m): the walled stairs 0.18-0.40 m (was 0.20-0.30; the ±0.02 noise
    kept); the open pyramid stairs take one depth per tile type: narrow 0.18, wide 0.40 (were 0.25 / 0.32);
  - the depth student's training camera latency 6-12 control steps, 120-240 ms (was 7.0-9.5, the real bag's p5-p95
    138-190 ms); the jitter kept; play / evaluation cameras unchanged;
  - training builds only (the teacher's and the students' trainings); every play / evaluation terrain stays.
  **Rule: U_TERRAIN=base for the checkpoints trained before it** (every Go2W run launched on a commit before 06e2647:
  go2w_min_s0 and everything before it, and the runs launched on 36d0fe8): their training terrain as it was; evaluation terrains do not depend on
  it, so it matters only for a training or a resume. Tests: scripts/unified/terrain_wide_test.py.
- The training payload's mount box payload_box=tinynav (the user via go2-home, 2026-11-16: "可以"; a robot parameter
  like the motor model, the Go2W): 1.25x tinynav-sim's PAYLOAD_LIMITS (12 kg, |x| 0.15, |y| 0.08, z 0-0.20):
  x ±0.19, y ±0.10, z 0-0.25 (was the robot table's payload_point x ±0.25, y ±0.12, z -0.05-0.30); the corner share
  and the redraw as before; the limit is the recipe's (the minimal recipe's 12 kg). Training builds only.
  **Rule: U_PAYLOAD_BOX=base for the checkpoints trained before it** (a commit before this one's). The acceptance's
  typical box for the Go2W became x ±0.15 at the same time (was ±0.20); the full-load point is unchanged.
- The training randomization dr=wide (the user via go2-home, 2026-11-16: "好的"; go2-home's survey of every training
  range against the tests, the 1.25x rule): the push's x / y +-1.25 m/s (was +-0.5; item 22 judges 1.0), the reset's
  hip splay +-0.175 rad (was +-0.15; march_eval's reset splays 0.14). A robot parameter (robot table dr=wide, the
  Go2W), training builds only. **Rule: U_DR_MODEL=base for the checkpoints trained before it.**
- The checks' foot friction (2026-11-16): the teacher's -Flat check drew the Go2's default 0.3-1.6 for the Go2W; it now
  takes the training's 0.4-1.0 as the students' checks always did (teacher_env CHECK_FRICTION). Training with
  U_DR_WIDE=1.25 draws 0.32-1.25 = 1.25x of it. Teacher flat gates before this read on the wider 0.3-1.6.
  Re-reads: every Go2W teacher flat gate before f81238e (the formal teachers' picks, among them the pick's scan-off
  drift 5.19 % against the 5 % bar, and the arms' flat gates read before tonight) drew foot friction 0.3-1.6; a re-read
  at main draws 0.4-1.0. Compare such readings only within one friction range.
- Tomorrow's base (the user, 2026-11-16): the minimal recipe + gom8010 + terrain=wide + payload_box=tinynav + dr=wide +
  U_CMD_WIDE=1.25 + U_DR_WIDE=1.25, from zero; scripts/unified/base_1117_test.py checks that they compose (U_DR_WIDE
  widens the motor model's joint factors around 1: friction / damping x 0.375-1.625, armature x 0.625-1.375).
- Open: the training's end-wall stop differs from the deployed guard (go2-home 2026-11-16, measured): the key endwall
  stops the forward command 0.85 m base-to-wall with a 0.2 m ramp (endwallshort), the deployed guard (tinynav-sim
  sim/guard.py) at 0.45 m camera-to-wall (the Go2W's camera 0.307 m ahead: ~0.76 m base-to-wall) with a linear ramp
  of 1.5/s per metre. go2w_minwall_s0 keeps the training's values; if endwall enters the recipe, its stop and ramp are
  aligned to the guard first (0.76 m, the linear ramp). Item 21's drive-up already follows the guard. The Jetson-side
  guard (robot_ctrl jetson_ctrl): not checked yet (go2-home asks the user).
- Known conflict, the Go2W wheel torque: unitree_ros gives 23.7 N m / 30.1 rad/s, the same motor; unitree_mujoco gives
  ctrlrange 15. The motor's values are taken. The 15 may be a flat stand-in for the curve at working speed: the curve
  gives ~15.9 N m at 1.5 m/s (~17 rad/s at the wheel).

## Notes

- 2026-11-16: no friction mismatch on either robot. The Go2 trains and tests on foot friction 0.3-1.6, the Go2W on
  0.4-1.0. An earlier survey line said otherwise and was wrong.
- The minimal recipe keeps the recipe table's structural switches: U_HEADING_OBS, U_PAYLOAD_STICKY, U_STAIRS_P,
  U_METER_EMA and U_ENTROPY_Q (the entropy 0.005 → 0.002 along the stairs quality q). A run's records list all 19
  recipe switches next to U_RECIPE_MIN=1. The reward terms the other recipe switches add are deleted afterwards.
