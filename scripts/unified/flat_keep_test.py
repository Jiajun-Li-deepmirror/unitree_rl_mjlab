import os, sys
sys.path.insert(0, os.getcwd())
os.environ["U_ROBOTS"] = "Go2,Go2W"
os.environ.pop("UNIFIED_TASKS", None)
from src.tasks.unified.robots import ROBOTS
from src.tasks.unified.teacher_env import unified_teacher_env_cfg as teacher
ok = True
base = teacher(ROBOTS["Go2W"]).rewards["u_flat_only"].params
ok &= "keep_terms" not in base
os.environ["U_FLAT_ONLY_KEEP"] = "standing_position_hold,standing_speed"
for n in ("Go2W", "Go2"):
  p = teacher(ROBOTS[n]).rewards["u_flat_only"].params
  ok &= p.get("keep_terms") == ("standing_position_hold", "standing_speed")
  print(n, p)
os.environ["U_FLAT_ONLY_KEEP"] = "no_such_term"
try:
  teacher(ROBOTS["Go2W"]); ok = False
except AssertionError as e:
  print("rejected:", str(e)[:90])
del os.environ["U_FLAT_ONLY_KEEP"]
ok &= teacher(ROBOTS["Go2W"]).rewards["u_flat_only"].params == base
print("PASS" if ok else "FAIL")
