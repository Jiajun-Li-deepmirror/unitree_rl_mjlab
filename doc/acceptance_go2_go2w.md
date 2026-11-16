# Go2 / Go2W acceptance (frozen 2026-11-16)

The user signed this specification on 2026-11-16 at ~15:50 CST, answering go2-home ("好的"). go2-home decided the open
points (go2-depth's seven defaults). The bars live in code in `scripts/unified/accept_judge.py` (`SPEC`, `spec_judge`),
which the students' acceptance (`jobs/unified/accept.sh`) and the teacher gates (`scripts/unified/stairs_gate.py`,
`scripts/unified/flat_gate.py`) read.

## Principles (the user)

- The three conditions carry the **same items**; only the values may differ. The conditions are:
  - unloaded;
  - typical payload (4 kg, its mount drawn per robot from the robot table's typical box: the Go2W x ±0.15, y ±0.08,
    z 0–0.20 m in the base frame since 2026-11-16, tinynav-sim's PAYLOAD_LIMITS (the user: "可以"; it was x ±0.20);
    the Go2 x ±0.20, y ±0.08, z 0–0.20);
  - full payload (8 kg at the acceptance point 0.15, 0, 0.20).
- Concrete numbers only; there is no "not worse than the deployed model" bar.
- A foot that touches the tread without load does **not** count as down. The four-feet item is contact based.
- Full-load stairs, the Go2: judged on 5–15 cm only; 15–18 cm is measured and reported. This covers every eval_stairs
  item, descent_probe's heading and meander, and items 26 / 27.
- Full-load stairs, the Go2W (the user, 2026-11-16, "可以"): judged on every band, 5–18 cm, at the full-load column's
  values. Map3's risers are 15–18.6 cm almost everywhere (L0 / R2 up to 18.6, only F1_entry 14.9), and the full-load
  map3 reach is judged already, so the eval_stairs bands match it (accept_judge.FULL_ALL_BANDS).
- A reading near a bar gets 3 runs, judged on the pooled readings; the better run is never picked. This is a
  measurement rule (go2-home): the judge reads one file per test and condition.

## The verdict: three gates, the rest targets (the user, 2026-11-16, "可以")

PASS / FAIL comes from three hard gates alone (the Go2W; the Go2 at its round), per condition unloaded / typical / full:
1. flat following: item 1 (flat speed 80–120 % / >= 75 % / >= 65 % of the command) and item 2's drift share
   (<= 5 / 5 / 8 %; its heading part is a target);
2. stairs success: item 10, map3's reach with the operator >= 95 / 95 / 90 %, without >= 90 / 90 / 85 %;
3. no falls, as a rate: every test's falls over its robots, pooled per condition (the stairs over all bands; the depth
   student's -RealEdge runs too) <= 0.2 / 0.2 / 0.5 %, and no single test above 1 % in any condition. It replaces
   "0 falls" everywhere: an item failing on its falls alone is no target. A missing test leaves the rate unknown: FAULT.

Every other item (3–30 without 25, item 2's heading, item 24) is a target: judged met / not met at the numbers below,
printed after the gates as "目标未达" in the loop's order -- the stairs' stop held and the slide-back first (the user's
safety note), then by the distance to the target -- each with w27d's reading. The judge writes the target table
<prefix>_targets.md beside the verdict (item, condition, target, the new reading, w27d's, met / not met): a deployed
model is replaced only after the user's own look at it (not a gate). Item 22 (the push) was dropped (2026-11-16).

Sample sizes stay fixed per test so the rate means one thing; the slope cells run 256 robots (were 64).

The teacher gates (stairs_gate / flat_gate) take the same gates where they measure them, with the teacher's fall rate
<= 1 % per condition and per run: the flat gate's flat speed and drift; the stairs gate's eval_stairs traverse success
at map3's operator bar (the teacher gates run no map3: a stand-in for gate 2); every other item a target (listed). The
FAIL gap: the gates' gaps + 0.001 per target not met.

## Bars

The numbers of every item; the gates' rows decide, the rest are targets.

| # | item (test) | unloaded | typical | full |
|---|---|---|---|---|
| 1 | flat speed, 44 commands (accept_matrix) | 80–120 % of the command | >= 75 % | >= 65 % |
| 2 | drift (drift_all); heading p50 <= 3°, p10 / p90 <= 8° in all three | <= 5 % | 5 % | 8 % |
| 3 | spin in place, end p90 (drift_all's spins) | <= 0.2 m | 0.2 m | 0.3 m |
| 4 | alternating spin, end p50 (spin_alt) | <= 0.25 m | 0.25 m | 0.35 m |
| 5 | stop (eval_standing): lift; creep Go2 / Go2W; brake from 1.0 m/s | 1 %; 1 / 1.5 cm; 30 cm | as unloaded | 1 %; 2 / 2.5 cm; 40 cm |
| 6 | march (march_eval's reset): touchdowns; hip reset; drift | >= 0.5 /s; <= 2°; <= 5 cm | as unloaded | 0.5 /s; 3°; 8 cm |
| 7 | gait clock match, every judged command (gait_clock --min-match) | >= 80 % | 80 % | 70 % |
| 8 | stance p90: standing 60 s / after an emergency stop | 0.17 / 0.15 rad | as unloaded | 0.20 / 0.18 rad |
| 9 | four wheels down (Go2W; gait_fb) | >= 98 % | 98 % | 95 % |
| 10 | map3 reach with / without the operator | 95 / 90 % | as unloaded | 90 / 85 % |
| 11 | stairs speed, per band and direction, share of the command (eval_stairs) | 0.8–1.2 | >= 0.75 | >= 0.65 |
| 12 | stop held mid-flight, within 0.5 m, per band and direction (eval_stairs stop-stay) | >= 95 % | 95 % | 90 % |
| 13 | four feet at the stop, per band and direction | >= 0.90 | 0.90 | 0.80 |
| 14 | heading max p90 on a flight, per band and direction (descent_probe --straight) | <= 8° | 8° | 10° |
| 15 | meander p-p p50, per band and direction (descent_probe --straight) Go2 / Go2W | 8 / 5 cm | as unloaded | 10 / 7 cm |
| 16 | edge share (Go2) | <= 0.10 | 0.10 | 0.12 |
| 17 | kicks per step on the climbs (Go2) | <= 0.35 | 0.35 | 0.40 |
| 18 | slips per step (Go2) / slide-back (Go2W) | 0.06 / 0.039 | as unloaded | 0.07 / 0.05 |
| 19 | mid-tread share (Go2) | >= 0.76 | 0.76 | 0.72 |
| 20 | stalled on the stairs, per band and direction | <= 5 % | 5 % | 5 % |
| 21 | no leg lift with a wall ahead (面前是墙不抬腿, added 2026-11-16 by the user via go2-home): on map3's top landings, the drive-up along the landing under the deployed wall guard's profile (tinynav-sim: speed min(0.5, 1.5/s x (camera-to-wall − 0.45 m)), the camera 0.307 m ahead of the base: the Go2W stops ~0.76 m base-to-wall; it was a stop at 0.85 m), then stand 2 s / turn in place 3 s / stand 2 s (map3_stairs_probe.py, 64 robots per up flight); of the robots that took the top, the share whose front foot / wheel rises more than 10 cm above the first stand or that fall on the landing -- the larger of (A) drive-up + first stand and (B) turn + last stand | <= 5 % | 5 % | 5 % |
| 24 | depth noise (深度噪声; the depth student only, added 2026-11-16 by the user via go2-home): items 10-20 again on the -RealEdge task (Unitree-<R>-U-Student-Depth-Stairs-RealEdge: the real camera's latency, frame timing, depth errors and mounting spread, edge smoothing on every capture), next to the clean run; labelled 相机误差 | the clean bars | the clean bars | the clean bars |
| 22 | push (抗推): DROPPED by the user 2026-11-16 (real users rarely push the robot); not run, not in gate 3. Training keeps its pushes (dr=wide, ±1.25 m/s); push_test.py stays a tool | — | — | — |
| 23 | slopes and rough ground (斜坡和不平地面): a 15° ramp up and a 15° ramp down (box geoms: 2 m flat, 6 m run, 2 m flat) and rough ground ±3 cm (the training's random-rough generator), walked across at 0.5 and 1.0 m/s, 64 robots per cell (slope_rough_test.py); reached = 7.5 m along within 7.5 / v x 1.5 + 2 s; the worst of the 6 cells | >= 95 %, 0 falls | 95 %, 0 | 90 %, 0 |
| 26 | pitch / roll rate on the flights (下楼点头、上楼晃): RMS per robot, p50 per band; the descents' pitch rate, the climbs' roll rate (descent_probe --straight's SUM) | <= 45 °/s | 45 °/s | 45 °/s (5–15 cm) |
| 27 | shin / thigh strikes on the steps (小腿大腿磕台阶): % of the flight's steps with a front / rear calf or thigh on the steps, per band, direction and side (descent_probe's SUMQ) | <= 2 % | 2 % | 2 % (5–15 cm) |
| 28 | top-landing extra steps / wheel lifts (顶部平台多余踏步/抬轮): per climb on map3's top landings, a wheel / foot on the landing's level that leaves the ground > 0.06 s with its apex > 2 cm, in the climb's end, the drive-up and the first stand (map3_stairs_probe, with item 21) | <= 1.0 | 1.0 | 1.0 |
| 29 | flat body bob (平地颠、晃): the base height and pitch std per robot over the steady part (3 s on), p50 over the robots, every straight command of drift_all (BOB rows; the user 2026-11-16 after w27d's reading) | fwd / back: height <= 1 cm, pitch <= 2°; sideways: height <= 1 cm, pitch <= 5° | as unloaded | as unloaded |
| 30 | stair stop-and-go (楼梯上走走停停): the coefficient of variation of the speed along the flight per robot, p50 per band and direction (descent_probe's SUMQ; the user 2026-11-16) | <= 0.3 | 0.3 | 0.3 (the Go2 5–15 cm, the Go2W every band) |
| gate 3 | falls as a rate, every test pooled per condition (stairs over all bands); no single test above 1 % | <= 0.2 % | 0.2 % | 0.5 % |

Reported only (long-term targets): kicks 0.245, slips 0.0245, mid 0.80; map3's speed; the Go2's full-load 15–18 cm
readings. The full-load column's "(5–15 cm)" applies to the Go2; the Go2W is judged on every band.

The teacher gates use the same items and values where they already measure them, with no new tests:
- stairs gate: both scan modes, the typical and the full payload;
- flat gate: its own tests.

They keep the teacher's fall tolerance: at most 2 per scan mode unloaded and typical, at most 6 under the full
payload (the Go2 on 5–15 cm, the Go2W over every band). They run no descent_probe, so items 14 and 15 are judged in the students' acceptance only.

## Tests per condition (accept.sh)

Every test runs in all three conditions:
- am, drift3, spinalt, march, clock, stand, loadstand, stopmove;
- stairs and descent;
- gait (the Go2W only);
- the map3 flights with and without the operator (noop, op).
- the top landings facing the wall (wall; item 21), after the map3 flights.
- slopes / rough ground (item 23), with the flat tests (item 22's push: dropped).
- Items 26, 27 and 30 come from the descent run, 28 from the wall run, 29 from drift3: no runs of their own.
- the depth student only (item 24): stairs and the map3 flights again on the -RealEdge task (stairsedge, noopedge,
  opedge); accept_judge --edge makes them required.

The full payload's gait clock runs with `--min-match 0.7`; the others keep 0.8. These runs are new:
- am, clock and the map3 flights under the typical and the full payload;
- gait (wheels down) under the full payload.

They take the evaluation time up by about half.

Item 21's notes (2026-11-16):
- the shape: uprobe_s162's measure (the turn) plus the last stand, and the drive-up (go2-home: the user's complaint
  came from a robot on the landing with a wall ahead, the drive-up included). The drive-up runs at the climb's speed
  (0.5 m/s) from the top, slowed by the deployed wall guard's ramp (the guard in tinynav-sim's sim/guard.py) to its
  stop, ~0.76 m base-to-wall on the Go2W. In the drive-up and the first stand a lift counts from a front
  foot's take-off at the landing's level (the bottom floor + the flight's rise, 3 cm tolerance, as item 28): the last
  riser's step-up takes off from the tread below and is not a lift (measured 2026-11-16: with the first stand's lowest
  point as the reference instead, 92 % of the pick's robots read as lifting in the drive-up, against 15 % in the turn). The judged share is the larger of the two parts; a fall on the landing counts in both;
- front feet / wheels only. For the Go2 (legged) a turn in place steps, and the drive-up's first steps may still lift
  a foot over the last nosing, so its own step height counts. Before the bar is read as wall-induced on the Go2, its
  turn on flat ground should be measured as a control;
- a robot that never reaches the landing is not counted (the climbs' items judge it). No robot on any landing: a fail;
- the play / evaluation builds carry no command wall stop (robots._play_wall_stop): every model is judged on the
  same commands; U_WALL_STOP_PLAY=1 keeps the end-wall key's fan and stop for a diagnostic. Training keeps them.