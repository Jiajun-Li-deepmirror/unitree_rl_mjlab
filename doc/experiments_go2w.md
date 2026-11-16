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

## Runs

| date | run | box | commit | switches | the single change and why | control | gates at 6000 (spec) | conclusion |
|---|---|---|---|---|---|---|---|---|
| 11-16 00:28 | go2w_2026-11-16_00-28-21_ufull_s0 (formal Go2W teacher v2, seed 0) | t2 | 77a29ea | the Go2W recipe (19 recipe switches; 68 reward terms) | — the control | — | teacher.sh picked avg4 5000 at the cap, below the gate: stairs PASS (full-load falls 6 = the teacher cap), flat FAIL gap 0.02 (scan-off sideways drift 5.2 / 5.1 %, gait clock 66-74 % on the fast gentle arcs); avg4 at 6000 not gated yet under the spec | control |
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
- Known conflict, the Go2W wheel torque: unitree_ros gives 23.7 N m / 30.1 rad/s, the same motor; unitree_mujoco gives
  ctrlrange 15. The motor's values are taken. The 15 may be a flat stand-in for the curve at working speed: the curve
  gives ~15.9 N m at 1.5 m/s (~17 rad/s at the wheel).

## Notes

- 2026-11-16: no friction mismatch on either robot. The Go2 trains and tests on foot friction 0.3-1.6, the Go2W on
  0.4-1.0. An earlier survey line said otherwise and was wrong.
- The minimal recipe keeps the recipe table's structural switches: U_HEADING_OBS, U_PAYLOAD_STICKY, U_STAIRS_P,
  U_METER_EMA and U_ENTROPY_Q (the entropy 0.005 → 0.002 along the stairs quality q). A run's records list all 19
  recipe switches next to U_RECIPE_MIN=1. The reward terms the other recipe switches add are deleted afterwards.
