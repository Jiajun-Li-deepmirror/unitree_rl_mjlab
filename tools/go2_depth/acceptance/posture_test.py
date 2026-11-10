"""tinynav-sim posture reset check: ROBOT KG X Y Z -- drive 1.5 m/s 3 s, hard stop, stand 8 s; with and without the reset."""
import sys, importlib.util
import numpy as np
sys.argv, args = ["view.py"], sys.argv[1:]
robot = args[0]; kg, x, y, z = (float(v) for v in args[1:5])
spec = importlib.util.spec_from_file_location("viewmod", "view.py")
vm = importlib.util.module_from_spec(spec); spec.loader.exec_module(vm)
from sim.accel import AccelLimit
from sim.posture import PostureReset
from sim import contract as C

from sim.keyboard import ConstantCmd


class Script(ConstantCmd):
    t = 0.0
    def __init__(self):
        super().__init__(0.0)
    def vector(self):
        return np.array([1.5 if 1.0 <= self.t < 4.0 else 0.0, 0.0, 0.0], np.float32)

for on in (False, True):
    src = Script()
    cmd = AccelLimit(src)
    sc = vm.get_scene("flat"); pos, yaw = sc.spawns[sc.default_spawn]
    w = vm.World("flat", robot, "blind", cmd, pos, yaw)
    w.set_payload(kg, x, y, z)
    w.start()
    import os; w.model.opt.noslip_iterations = int(os.environ.get("NOSLIP", "0"))
    p = PostureReset(robot); p.enabled = on
    dev_at = {}
    for k in range(int(13.0 / C.CONTROL_DT)):
        src.t = k * C.CONTROL_DT
        w.rt.march = p.marching
        w.rt.tick()
        p.step(w.rt)
        for ts in (5.0, 6.0, 9.0, 12.9):
            if abs(src.t - ts) < C.CONTROL_DT / 2:
                dev_at[ts] = (round(p.deviation, 3), p.worst_joint)
    print(f"{robot} {kg} kg reset {'ON ' if on else 'OFF'}: deviation at 5/6/9/12.9 s {dev_at}  "
          f"up {w.rt.base_uprightness():.2f} tries {p.tries}", flush=True)
    w.close()
