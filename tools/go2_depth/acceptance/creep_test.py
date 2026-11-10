"""tinynav-sim standing creep: ROBOT KG SECONDS [noslip_iters] [friction] -- stand at zero command, every 5 s:
hip angle off default (deg, outward +), hip target off default (ctrl), foot spacing change (cm, base frame y,
front / rear), foot slide in contact (cm), max |Ft|/Fn over the feet."""
import sys, importlib.util
import numpy as np, mujoco
sys.argv, args = ["view.py"], sys.argv[1:]
robot, kg, secs = args[0], float(args[1]), float(args[2])
noslip = int(args[3]) if len(args) > 3 else 0
mu = float(args[4]) if len(args) > 4 and args[4] != "-" else None
imp = float(args[5]) if len(args) > 5 else None
spec = importlib.util.spec_from_file_location("viewmod", "view.py")
vm = importlib.util.module_from_spec(spec); spec.loader.exec_module(vm)
from sim.keyboard import ConstantCmd
from sim.posture import hip_out_deg
from sim import contract as C
sc = vm.get_scene("flat"); pos, yaw = sc.spawns[sc.default_spawn]
w = vm.World("flat", robot, "blind", ConstantCmd(0.0), pos, yaw)
w.set_payload(kg, 0.0, 0.0, 0.1)
if mu is not None:
    w.set_friction(mu)
w.start()
m, d = w.model, w.data
m.opt.noslip_iterations = noslip
import os
if os.environ.get("SOLIMP"):
    m.geom_solimp[list(w._feet), :3] = [float(v) for v in os.environ["SOLIMP"].split(",")]
if os.environ.get("ITERS"):
    m.opt.iterations = int(os.environ["ITERS"]); m.opt.ls_iterations = int(os.environ["ITERS"])
if imp is not None:
    m.opt.impratio = imp
feet = list(w._feet)
fnames = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(i))[:2] for i in feet]
hip_act = {}
for g, (ids, _p, _s, off, _dof, qadr) in zip(w.rt.contract.action_groups, w.rt._action_map):
    if off is None: continue
    for j, n in enumerate(g.joints):
        if n.endswith("_hip_joint"): hip_act[n[:2]] = (int(ids[j]), float(off[j]))
print(f"{robot} {kg} kg noslip {noslip} mu {mu or 0.6} dt {m.opt.timestep} cone {mujoco.mjtCone(m.opt.cone).name} impratio {m.opt.impratio} "
      f"integrator {mujoco.mjtIntegrator(m.opt.integrator).name} iters {m.opt.iterations}")
def foot_y():
    R = d.xmat[1].reshape(3, 3); p0 = d.xpos[1]
    return {n: float((R.T @ (d.geom_xpos[g] - p0))[1]) for n, g in zip(fnames, feet)}
y0 = None; last = {n: d.geom_xpos[g].copy() for n, g in zip(fnames, feet)}; slide = {n: 0.0 for n in fnames}
f6 = np.zeros(6); ratio_max = 0.0
for k in range(int(secs / C.CONTROL_DT) + 1):
    w.rt.tick()
    t = k * C.CONTROL_DT
    for n, g in zip(fnames, feet):
        incontact = any((d.contact[c].geom1 == g or d.contact[c].geom2 == g) for c in range(d.ncon))
        if incontact and t > 1.0:
            slide[n] += float(np.linalg.norm((d.geom_xpos[g] - last[n])[:2]))
        last[n] = d.geom_xpos[g].copy()
    if t > 1.0:
        for c in range(d.ncon):
            if d.contact[c].geom1 in feet or d.contact[c].geom2 in feet:
                mujoco.mj_contactForce(m, d, c, f6)
                if f6[0] > 1.0:
                    ratio_max = max(ratio_max, float(np.hypot(f6[1], f6[2]) / f6[0]))
    if abs(t - 1.0) < 1e-6:
        y0 = foot_y()
    if k % int(5.0 / C.CONTROL_DT) == 0 and t >= 5.0:
        labels, hip = hip_out_deg(w.rt)
        tgt = {n: np.degrees((d.ctrl[a] - off) * (1 if n[1] == "L" else -1)) for n, (a, off) in hip_act.items()}
        y = foot_y()
        fr = 100 * ((y["FL"] - y["FR"]) - (y0["FL"] - y0["FR"])); rr = 100 * ((y["RL"] - y["RR"]) - (y0["RL"] - y0["RR"]))
        print(f"t {t:5.1f}  hip out deg " + " ".join(f"{l} {v:+5.1f}" for l, v in zip(labels, hip))
              + "  | target " + " ".join(f"{n} {tgt[n]:+5.1f}" for n in labels)
              + f"  | spacing +cm front {fr:+5.2f} rear {rr:+5.2f}  | slide cm " + " ".join(f"{n} {100*slide[n]:.2f}" for n in fnames)
              + f"  | max Ft/Fn {ratio_max:.2f}  up {w.rt.base_uprightness():.2f}  phy_ms {w.rt.physics_ms:.2f}", flush=True)
        ratio_max = 0.0
