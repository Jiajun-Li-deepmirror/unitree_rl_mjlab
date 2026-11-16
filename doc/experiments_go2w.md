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
| 11-16 ~19:00 (planned) | go2w_min_s0 (minimal recipe) | t2 | f94c9fa | U_RECIPE_MIN=1 (with it U_PAYLOAD_TRAIN_MAX=12, U_HARD_FIXED=1) | the user's "从选配方开始": 18 reward terms at today's weights instead of 68; no constraint multipliers; no quality stage; the training conditions of A and D | control s0 | open | open |

## Switches written, not run

| commit | switches | what | status |
|---|---|---|---|
| 72592f5 | U_HARD_FIXED=1 (U_MARCH_FIXED=0.5, U_FAST_STOP=0.75,0.3, U_PAYLOAD_FULL0=1) | arm D: the hard actions at full share from iteration 0, as g27b / w27d trained | not run as a single factor on the 68-term recipe (go2-home 11-16, after the user's question); part of the minimal recipe |
| 72592f5 | U_CMD_WIDE=1.25 | arm C1: the final command ranges x 1.25 | not run (same reason) |
| 72592f5 | U_DR_WIDE=1.25 | arm C2: the randomization's spread x 1.25 | not run; the Go2W's training DR already equals the test's except the push |

## Notes

- 2026-11-16: no friction mismatch on either robot. The Go2 trains and tests on foot friction 0.3-1.6, the Go2W on
  0.4-1.0. An earlier survey line said otherwise and was wrong.
- The minimal recipe keeps the recipe table's structural switches: U_HEADING_OBS, U_PAYLOAD_STICKY, U_STAIRS_P,
  U_METER_EMA and U_ENTROPY_Q (the entropy 0.005 → 0.002 along the stairs quality q). A run's records list all 19
  recipe switches next to U_RECIPE_MIN=1. The reward terms the other recipe switches add are deleted afterwards.
