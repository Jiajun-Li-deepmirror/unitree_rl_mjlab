"""fe_pass.py R PREFIX (depth_fe, 2026-10-31): go2-home's fix/mx_pass.py, the verdict of one check: PASS when every
judged goal passes (flat speed, drift, spin, stop, march, gait clock, stairs, falls; hip load not judged), else
FAIL: <goal: reason; ...>. Reads PREFIX_<kind>.txt as go2-home's acceptance (scripts/blind/acceptance)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts/blind/acceptance").resolve()))
import accept_tables as at  # noqa: E402

r, prefix = sys.argv[1:3]
p = {k: f"{prefix}_{k}.txt" for k in at.KINDS}
cells = at.goal_cells(r, 1, "", p, [])
goals = {"g_speed": "平地速度", "g_drift": "跑偏", "g_spin": "原地旋转", "g_spinalt": "交替原地转", "g_loadstand": "负重站立", "g_stop": "停车",
         "g_march": "原地踏步", "g_clock": "步态节奏", "g_stairs": "楼梯", "g_falls": "摔倒"}
if r in at.WHEELED:
  goals["g_wheels"] = "四轮着地"
# FE_PASS_SKIP (2026-11-05): goals not judged (comma-separated keys; the blind teacher phase's flat-only gate skips g_stairs).
import os
for k in filter(None, os.environ.get("FE_PASS_SKIP", "").split(",")):
  goals.pop(k, None)
# A check started before spin_alt.py joined fe_check.sh (11-06 ~23:50) has no PREFIX_spinalt.txt: not judged there.
if not os.path.exists(p["spinalt"]):
  goals.pop("g_spinalt", None)
if not os.path.exists(p["loadstand"]):
  goals.pop("g_loadstand", None)
bad = [f"{n}：{cells.get(k, '未测')}" for k, n in goals.items() if cells.get(k) != "通过"]
print("PASS" if not bad else "FAIL: " + "；".join(bad))
