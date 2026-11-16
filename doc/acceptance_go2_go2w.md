# Go2 / Go2W acceptance (frozen 2026-11-16)

The user signed this specification on 2026-11-16 at ~15:50 CST, answering go2-home ("好的"). go2-home decided the open
points (go2-depth's seven defaults). The bars live in code in `scripts/unified/accept_judge.py` (`SPEC`, `spec_judge`),
which the students' acceptance (`jobs/unified/accept.sh`) and the teacher gates (`scripts/unified/stairs_gate.py`,
`scripts/unified/flat_gate.py`) read.

## Principles (the user)

- The three conditions carry the **same items**; only the values may differ. The conditions are:
  - unloaded;
  - typical payload (4 kg at the robot table's typical point);
  - full payload (8 kg at the acceptance point 0.15, 0, 0.20).
- Concrete numbers only; there is no "not worse than the deployed model" bar.
- A foot that touches the tread without load does **not** count as down. The four-feet item is contact based.
- Full-load stairs are judged on 5–15 cm only; 15–18 cm is measured and reported. This covers every eval_stairs item
  and descent_probe's heading and meander.
- A reading near a bar gets 3 runs, judged on the pooled readings; the better run is never picked. This is a
  measurement rule (go2-home): the judge reads one file per test and condition.

## Bars

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
| — | falls in every test | 0 | 0 | 0 (stairs: 5–15 cm) |

Reported only (long-term targets): kicks 0.245, slips 0.0245, mid 0.80; map3's speed; the full load's 15–18 cm
readings.

The teacher gates use the same items and values where they already measure them, with no new tests:
- stairs gate: both scan modes, the typical and the full payload;
- flat gate: its own tests.

They keep the teacher's fall tolerance: at most 2 per scan mode unloaded and typical, at most 6 under the full
payload (5–15 cm). They run no descent_probe, so items 14 and 15 are judged in the students' acceptance only.

## Tests per condition (accept.sh)

Every test runs in all three conditions:
- am, drift3, spinalt, march, clock, stand, loadstand, stopmove;
- stairs and descent;
- gait (the Go2W only);
- the map3 flights with and without the operator (noop, op).

The full payload's gait clock runs with `--min-match 0.7`; the others keep 0.8. These runs are new:
- am, clock and the map3 flights under the typical and the full payload;
- gait (wheels down) under the full payload.

They take the evaluation time up by about half.
