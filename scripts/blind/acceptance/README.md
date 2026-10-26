# Blind locomotion acceptance

One acceptance for all six blind robots (Go2, As2, B2 legged; Go2W, As2W, B2W wheeled):
the same sections in a fixed order; wheeled robots add driving forward / backward with
all four wheels down. The first table checks every model against the training goals:

| Goal | Threshold |
|---|---|
| Flat speed, small commands included | actual / command 80-120% on the commanded axes |
| Straight and sideways drift (10 s) | heading median <= 3 deg, p90 <= 8 deg; offset <= 5% of the distance |
| Turning in place (10 s) | base within 0.2 m of its spot (p90) |
| Stopping | four feet / wheels down; creep <= 1 cm per 4 s; braking from 1 m/s <= 30 cm |
| Marching in place (when the thermal guard asks) | 20 s drift <= 10 cm; hottest hip below standing; no fall |
| Hip abduction torque | standing and at 1 m/s >= 30% below the deployed model |
| Stairs (map3 flights, never trained on) | with operator >= 95%, without >= 90%; speed 0.8-1.2x; stopping mid-flight >= 95% |
| Wheeled forward / backward | all four wheels down >= 98% at every speed |
| No falls; push recovery | every test; the push test is still to be written |

Evaluating a checkpoint (from the repo root): `scripts/blind/eval/` holds the tests
(map3_par.py runs scripts/go2_blind/eval_map3.py's flights in parallel processes;
accept_matrix, drift_all, eval_standing, climb_posture, march_eval, gait_fb, and
step1_diag.py hipload), plus scripts/go2_blind/eval_stairs.py. All at once take ~3 min
per checkpoint on an RTX 4090 (peak ~10 GB GPU, ~110 GB RAM).

Tables: copy each model's outputs to `results/<id>_<kind>.txt` (kinds noop, op, am,
stairs, drift3, stand, climb, hip, march, gait), list the models in `manifest.json`
(per robot, the deployed model first as the baseline), then

    python accept_tables.py                    # markdown
    python accept_tables.py --json dash        # one JSON dataset per section + sections.json
    python dash_page.py dash/sections.json dash/index.html

The JSON files are the datasets of a claude.ai Dashboard artifact (one attached file
each) and index.html its page. A goal cell says 通过, or 不通过： followed by short tags.
