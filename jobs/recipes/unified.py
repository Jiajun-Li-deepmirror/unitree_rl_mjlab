#!/usr/bin/env python3
"""The unified blind recipe, defined in code (go2-depth 2026-11-11, the user: "训练配方尽量统一 ... 配方由代码定义，不从日志里提取 ...
一份共用基础配方，加上腿式和轮式各自的部分，再加少量每个机型自己的差异，每项差异写明原因 ... T 和 M 的开关分成两组").

DRAFT for review: the chain still reads jobs/recipes/<r>.sh. Layers, each with its T, M and P keys (P = M + P's own):

  COMMON          every robot
  WHEELED/LEGGED  the robot's kind
  ROBOTS[r]       physical parameters only (payload, gains, sizes ...) plus the few keys only this robot needs,
                  each with its reason

Every key is listed once with a status:
  sure      the same on every robot of its layer (it already was on most, or it serves a bar every robot has)
  verify    unified, but a from-zero run must confirm it (it was tuned on one robot, or on a lineage that could
            already climb: As2 rp3's M never climbed with keys of that kind in T/M)
  param     not a key any more: a physical quantity in ROBOTS (listed for the record)
  dropped   no longer needed under the unified acceptance

  python3 jobs/recipes/unified.py ROBOT          the T / M / P keys and parameters
  python3 jobs/recipes/unified.py ROBOT --sh     as shell variables (RECIPE_BLIND_T / _M / _P, budgets)
  python3 jobs/recipes/unified.py --diff         per robot: keys gained / lost against jobs/recipes/<r>.sh
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Key:
    name: str
    phase: str           # "T" (from T on, so in M and P too), "M" (from M on), "P" (P only)
    status: str          # sure | verify | param | dropped
    why: str


def k(name, phase, status, why):
    return Key(name, phase, status, why)


# ---------------------------------------------------------------------------------------------------------------
# COMMON: every robot. T = M for now (the structure keeps them apart so that M can change without retraining T).
COMMON = [
    # flat skills, gait clock, standing / marching -- on all six robots' T/M already
    *[k(n, "T", "sure", "on all six robots' T/M today") for n in (
        "speedclock", "arcclock", "critic", "grip", "hipzero", "nohipload", "lowent", "lowfric", "speedboth",
        "sidew3", "straightw3", "marchsettle", "marchsmooth", "marchw40", "marchquiet", "marchreset",
        "standsettle", "standsmooth", "standpose", "payload")],
    # stairs layout and stairs line, on all six T/M today
    *[k(n, "T", "sure", "on all six robots' T/M today") for n in (
        "stairboth", "stairline", "stairs50", "landings", "wumore")],
    k("gaitw1", "T", "sure", "gait clock pays 2x: on five robots, only Go2 lacked it; the clock bar (80%) is the same for all"),
    k("stairshare", "T", "sure", "stairs line counts the sideways share: on all six, but legged had it in P (B2 in T/M)"),
    k("spinflip", "T", "verify", "turn-in-place reversals: in T/M on As2W / Go2 / As2, P on Go2W / B2; the alternating-spin "
                                 "bar (0.25 m) is a flat skill"),
    k("spinhold", "T", "verify", "turn-in-place holds the spot (critic +2 inputs): from T on, so P never needs a widened "
                                 "checkpoint (Go2W / B2 rp4 crashed at P on it)"),
    # P: stairs stability, the user's stairs requirements (both kinds)
    k("stairroll", "P", "sure", "body roll on stairs x5: five robots' P, B2W lacked it"),
    k("stairstab", "P", "sure", "body rate on stairs x2: five robots' P, B2W lacked it"),
    k("tilt3", "P", "sure", "stance legs vertical on stairs x3: five robots' P, B2W lacked it"),
    k("paycorner", "P", "verify", "30% of heavy payloads at the acceptance's corner: Go2W / As2W only; the payload "
                                  "acceptance is the same for all"),
]

# ---------------------------------------------------------------------------------------------------------------
WHEELED = [
    *[k(n, "T", "sure", "on all three wheeled T/M today") for n in (
        "wheelclock", "arcstep", "diagaxis", "drivelp", "drivepose", "liftflat", "longexit", "marchreset2",
        "riserprox", "sidecmd", "sidew6", "slowarc", "stairhead", "steptarget")],
    # one constraint set for every wheeled robot (constraints = acceptance items, and the bars are the same)
    *[k(n, "T", "verify", "Go2W / As2W constraint set; B2W ran its own b2wcons -- one set for all three (the bars "
                          "are shared); B2W's back_drift / descent metrics to be merged in (code change)") for n in (
        "wheelcons", "sidetgt75", "consease")],
    *[k(n, "T", "verify", "Go2W / As2W flat-speed fixes (small / sideways commands over 120%); the flat speed bar "
                          "is the same for all") for n in ("speedw4", "straightcmd", "trackmin")],
    *[k(n, "T", "verify", "As2W / B2W only; serves shared bars (side brake, slow arcs, payload-aware critic)") for n in (
        "brakeside", "drcritic", "slowarc2")],
    k("marchclear", "T", "verify", "6 cm march lift (the user: march lift too low): Go2W P, As2W T/M -- the march is a "
                                   "flat skill"),
    k("marchpose", "T", "verify", "marching keeps the walking posture tolerance: Go2W P, As2W T/M"),
    k("restlock", "T", "verify", "at rest the posture terms keep the settled pose (braked wheels: re-posing moves the "
                                 "base): B2W only; the creep bar (1 cm) is the same for all"),
    k("standtgt", "T", "verify", "at rest the legs hold their targets: B2W only, same reason"),
    # P: the stairs-quality keys (on all three today) and the ones only Go2W / As2W had
    *[k(n, "P", "sure", "on all three wheeled P today") for n in (
        "liftm8", "noslide", "noslide2x", "stairspeed2", "stairspeed3", "topairup", "wheelahead", "wheelstep",
        "wheelstep4")],
    k("stairspeed4", "P", "verify", "stairs speed deficit x2 again: Go2W / As2W; the 0.8-1.2x bar is shared"),
    k("stairhead3", "P", "verify", "stairs heading x3: Go2W / As2W; tinynav judges the heading (<= 3 deg) for every wheeled"),
    k("stairhead6", "P", "verify", "x2 again, same reason"),
    k("stepclean", "P", "verify", "each step touched once: Go2W / As2W; the user's requirement for every wheeled robot"),
    k("idlestep", "P", "verify", "idle wheel steps before the first riser: As2W; the start-landing bar (<= 0.5 per climb) "
                                 "is shared"),
    k("stairstandw", "P", "verify", "all wheels down when stopped on stairs: As2W; the 'stopped, 4 feet' bar is shared"),
]

# ---------------------------------------------------------------------------------------------------------------
LEGGED = [
    *[k(n, "T", "verify", "on all three legged T/M today; As2 rp3's M never climbed with these from zero (suspects: "
                          "feetedge, stop*, yawslew) -- B2 rp4 did") for n in (
        "feetedge", "spinup", "standlong", "stopfast", "stopside", "yawslew")],
    k("stairpay16", "T", "sure", "height pay cut-off 1.6x: all three legged T/M"),
    k("basetouch", "T", "verify", "dense cost on base-ground touches: As2 only (a Go2 fix that never reached Go2's recipe)"),
    k("flatorient3", "T", "verify", "body orientation x3 on flat ground: As2 only, same story"),
    k("legcons", "T", "verify", "legged constraint set (clock, drift, feet at edges): Go2 only in T/M; edgecons (P) is its "
                                "stairs part alone -- one set from T on for all three"),
    k("sidew6l", "T", "verify", "sideways drift x6 for the unanchored legged M: B2 P only; the drift bar is shared"),
    *[k(n, "P", "sure", "on all three legged P today") for n in (
        "feetmid", "feetonce", "feetonce2", "footslide", "stairline2", "stairstand4", "stumble4x", "support2")],
    k("edgecons", "P", "dropped", "covered by legcons from T on"),
    k("spintight", "P", "dropped", "2 cm spin target for B2's old 0.17 m bar; the unified bar is 0.25 m"),
]


# ---------------------------------------------------------------------------------------------------------------
@dataclass
class Robot:
    kind: str
    params: dict
    keys: list = field(default_factory=list)       # this robot's own keys (Key), each with its reason


ROBOTS = {
    "go2w": Robot("wheeled", {"payload": "max=12,x=0.15,y=0.08,z=0.20 (Unitree spec)", "wall_stop_m": 0.85,
                              "wheel_scale": 10, "wheel_radius_m": 0.086, "leg_kp": "20/20/40 (model)"}),
    "as2w": Robot("wheeled", {"payload": "max=13,x=0.15,y=0.08,z=0.20 (half the body weight, no spec found)",
                              "wall_stop_m": 0.9, "wheel_scale": 8, "wheel_radius_m": 0.086, "leg_kp": "40/40/60 (model)"}),
    "b2w": Robot("wheeled", {"payload": "max=40,x=0.30,y=0.15,z=0.30 (Unitree spec)", "wall_stop_m": 1.05,
                             "wheel_scale": 6, "wheel_radius_m": 0.113,
                             "leg_kp": "240/6 -- was the key legkp15: 82 kg on kp 160 crept after stops",
                             "leg_hold_s": "2.0 -- was the key leghold20: legs hold their targets after a stop",
                             "stairs_final_share": "0.35 -- was stairs35: B2W's flat items slid as the share rose past ~0.35"},
                 [k("marchgait3", "T", "verify", "B2W never stepped marching (0 touchdowns/s at 2000-3000): the clock "
                                                 "pays 3x while marching -- try without it once the common march keys run"),
                  k("standw4w", "T", "verify", "standing hold x4: B2W crept 2.5-3.6 cm after stops (bar 1 cm) -- heavy "
                                               "body on braked wheels"),
                  k("standlongw", "T", "verify", "zero commands held 4-8 s: B2W creep, same reason"),
                  k("b2wdesc", "T", "verify", "descent target in the B2W constraint set: to be merged into the shared one"),
                  k("bdhold", "T", "verify", "back_drift multiplier held <= 1.7: B2W overshoot of small sideways speeds")]),
    "go2": Robot("legged", {"payload": "max=12,x=0.15,y=0.08,z=0.20 (Unitree spec)", "wall_stop_m": 0.85, "foot_r_m": 0.022}),
    "as2": Robot("legged", {"payload": "max=10,x=0.15,y=0.08,z=0.20 (half the body weight, no spec found)",
                            "wall_stop_m": 0.9, "foot_r_m": 0.028}),
    "b2": Robot("legged", {"payload": "max=40,x=0.30,y=0.15,z=0.30 (Unitree spec)", "wall_stop_m": 1.05, "foot_r_m": 0.032}),
}
BUDGETS = {"RECIPE_T_MAX": 5000, "RECIPE_M_ITERS": 10000, "RECIPE_P_ITERS": 6000}   # caps; each phase stops at its pass
CONS_INIT = ""   # every constraint starts at 1x from zero (the old per-robot values were resumed runs' multipliers)


def recipe(r: str) -> dict:
    rob = ROBOTS[r]
    layers = COMMON + (WHEELED if rob.kind == "wheeled" else LEGGED) + rob.keys
    live = [x for x in layers if x.status in ("sure", "verify")]
    T = [x.name for x in live if x.phase == "T"]
    M = T + [x.name for x in live if x.phase == "M"]
    P = M + [x.name for x in live if x.phase == "P"]
    return {"T": T, "M": M, "P": P, "params": rob.params, "budgets": BUDGETS, "keys": layers}


def current(r: str) -> tuple[set, set]:
    out = subprocess.run(["bash", "-c", f"source {HERE}/{r}.sh; echo \"$RECIPE_BLIND_TM\"; echo \"$RECIPE_BLIND_P\""],
                         capture_output=True, text=True).stdout.splitlines()
    return set(filter(None, out[0].split(","))), set(filter(None, out[1].split(",")))


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__); return 0
    if argv[0] == "--diff":
        for r in ROBOTS:
            u = recipe(r); tm, p = current(r)
            print(f"== {r}: unified T/M {len(u['M'])}, P {len(u['P'])} (now T/M {len(tm)}, P {len(p)})")
            print(f"   into T/M: {' '.join(sorted(set(u['M']) - tm)) or '-'}")
            print(f"   out of T/M (to P or dropped): {' '.join(sorted(tm - set(u['M']))) or '-'}")
            print(f"   P gains: {' '.join(sorted(set(u['P']) - p)) or '-'}")
            print(f"   P loses: {' '.join(sorted(p - set(u['P']))) or '-'}")
        return 0
    r = argv[0].lower()
    u = recipe(r)
    if "--sh" in argv:
        print(f"RECIPE_BLIND_T={','.join(u['T'])}")
        print(f"RECIPE_BLIND_M={','.join(u['M'])}")
        print(f"RECIPE_BLIND_P={','.join(u['P'])}")
        print(f"RECIPE_BLIND_CONS={CONS_INIT}")
        for kk, v in u["budgets"].items():
            print(f"{kk}={v}")
        return 0
    print(f"{r} ({ROBOTS[r].kind}): T {len(u['T'])}, M {len(u['M'])}, P {len(u['P'])}")
    for x in u["keys"]:
        print(f"  {x.phase} {x.status:8s} {x.name:12s} {x.why}")
    print("  params: " + "; ".join(f"{a} {b}" for a, b in u["params"].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
