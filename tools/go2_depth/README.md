# go2-depth workstation tools

What runs on the workstation (not on the GPU boxes) around the training chains in `jobs/`:
the tinynav-sim acceptance gate, deployment into tinynav-sim, comparison videos and the
chain monitor. The robots' controllers and their sim2sim tests are in the separate
`robot_ctrl` repository (Jetson deployment: its `tools/stage_model.sh` and `sync.sh`).

Paths: the scripts find their code next to themselves. State (pulled exports, logs, verdicts)
goes to `GO2_DEPTH_STATE` (default `~/.cache/go2-depth-work/check`). The tinynav-sim tree the
acceptance runs in is `TINYNAV_MUJOCO` (default `~/.cache/go2-depth-work/tnsim-wheeled/mujoco`),
with `ROBOT_ASSETS=~/.cache/go2-depth-work/model_stage` and `MUJOCO_GL=egl`. Boxes are reached
through `~/.cache/go2-depth-work/ssh/r PORT CMD` (t1 30714, t2 31365).

## acceptance/

| Script | What it does |
|---|---|
| `tn_gate_watch.sh PORT` | Serves the chains' tinynav gate: for each `logs/accept/<run>_<K>.req` on the box without a `.verdict`, exports the checkpoint there (`jobs/probe_export.sh`, or the blind export + `scripts/import_blind_policy.py`), pulls it, runs the acceptance below and writes the verdict back (PASS / FAIL: items). Log: `$GO2_DEPTH_STATE/accept_watch.txt`. Run one per box, in the background. |
| `slide_gate.sh ROBOT ONNX` | Wheeled slide-back gate on map3 L0 (12 climbs, 12 more when close; `stair_wheels.py`). |
| `accept_stairs.py ROBOT ONNX [RUNS]` | The user's stairs requirements on map3 and the pyramid scene: legged kicks / mid-tread / slips / roll, wheeled slide-back / idle steps / roll / heading, falls; `PAYLOAD=kg,x,y,z` for the payload runs, `NOSLIP=10` for MuJoCo's static friction. |
| `accept_map3.py ROBOT ONNX [RUNS]` | All 18 map3 flights plus the stairs posture (info). |
| `stair_wheels.py`, `edge_tn.py`, `obstacles.py` | Wheel stepping / slide-back on stairs, foot-on-edge stances, walking into boxes and walls. |
| `roll_probe.py`, `lat_tail.py` | Body roll on stairs; sideways stance-angle tails. |
| `posture_test.py`, `creep_test.py` | tinynav-sim checks of the posture reset (drive, hard stop, stand) and of standing contact creep (hip angles, foot spacing, slip, with / without noslip). Run in the viewer container's `mujoco/`. |

## deploy/

`deploy_tinynav.py ROBOT [--depth ONNX NAME] [--blind ONNX NAME]`: copies the models into both
tinynav-sim trees, points `sim/robots.py` at them, removes the replaced files (only the deployed
models stay), rewrites `policy.sha256`, runs the tests in the `tnsim-web` container and restarts
the viewer. `--dry-run` shows what it would change. The README's model row is edited by hand.

## video/

`stack_compare.py TOP.mp4 BOTTOM.mp4 OUT.mp4 [--title ...] [--top-label 无负载] [--bottom-label 负载]`:
the unloaded and the payload acceptance clips (recorded on a box with `jobs/record_blind.sh` or
`jobs/record_student_v.sh`) stacked into one video with Chinese label strips.

## monitor/

`wait_logs.sh`: exits when a chain log or a tinynav verdict on t1 / t2 changes (`MAXS` seconds at
most); the monitoring loop waits on it.
