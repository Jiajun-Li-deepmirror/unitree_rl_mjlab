#!/usr/bin/env python3
"""curve_watch.py RUN_DIR --phase T|M|P [--log L --label X --pid P] [--replay] (go2-depth 2026-11-11, the user's requirement 3:
"每次训练都要和已交付谱系在同一迭代步的关键曲线对比，明显偏离就报警 ... Go2W rp3 的 M 楼梯上抬轮一直是平的，本来几小时内就能看出来").

Watches one blind training run's TensorBoard curves and logs "<label> CURVE ALARM <rule>: ..." to L when a rule trips (once,
until the curve is back: "<label> curve back: <rule>"). Live: every --every s (default 600) until --pid ends. --replay: walks
a finished run every 100 iterations and prints the transitions (how the rules were set).

The recipes change between runs, so the rules compare a run with itself (its curriculum, its own earlier values), not
with another run's reward scale. Offsets are from the run's first iteration (its phase start). W = the mean of the last
200 iterations.
  share_lag   (M, P) stairs share behind its curriculum: W(share) / W(target) < 0.70 while W(target) >= 0.15.
              The curriculum only raises the share as robots finish stairs, so a lag means they do not climb.
  climb_flat  (M) the climb reward did not grow: W(wheel_riser_lift) (wheeled; foothold_height_gain legged) under
              1.2x its W at +500 from +2000, under 2x from +3000. (+2000 added 11-11 on go2-home's replay of As2 rp3 M,
              which only +3000 caught: Go2W rp3 0.95x / rp4 1.37x / B2 rp3 2.6x / rp4 3.3x at +2000. A terrain-level-drop
              rule does not separate them: Go2W rp4, which climbed, dropped 0.86 levels in M, rp3 0.73.)
  reward_drop (T after +1000; M, P after +500) W(mean_reward) < 0.5x its best W since +500 (best > 5), unless the
              terrain level rose by 0.5 or more since that best (a harder curriculum lowers the reward).
  short_eps   (T after +1500; M, P after +500) W(mean_episode_length) < 0.5x its best W, the same terrain proviso.
Set on the rp3 / rp4 runs (2026-11-11): Go2W rp3 M (never climbed) trips share_lag at +600 (share 0.07 of 0.15)
and climb_flat at +2000; Go2W rp4 M and B2 rp4 T / M (climbed, passed) trip nothing; B2 rp3 M (its P later failed map3)
trips reward_drop (17 -> 8).
"""
from __future__ import annotations

import argparse
import bisect
import glob
import os
import sys
import time

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

W = 200
TAGS = {"share": "Curriculum/stairs_share/share", "target": "Curriculum/stairs_share/target",
        "riser": "Episode_Reward/wheel_riser_lift", "foothold": "Episode_Reward/foothold_height_gain",
        "reward": "Train/mean_reward", "eplen": "Train/mean_episode_length", "terrain": "Curriculum/terrain_levels"}


def load(run: str) -> dict[str, "Curve"]:
    out: dict[str, list[tuple[int, float]]] = {}
    for f in sorted(glob.glob(os.path.join(run, "events.out.tfevents*"))):
        e = EventAccumulator(f, size_guidance={"scalars": 0})
        e.Reload()
        have = set(e.Tags()["scalars"])
        for k, t in TAGS.items():
            if t in have:
                out.setdefault(k, []).extend((s.step, s.value) for s in e.Scalars(t))
    return {k: Curve(sorted(v)) for k, v in out.items()}


class Curve:
    """steps and values with prefix sums: the mean over a window in O(log n)."""

    def __init__(self, pts: list[tuple[int, float]]):
        self.steps = [s for s, _ in pts]
        self.pre = [0.0]
        for _, v in pts:
            self.pre.append(self.pre[-1] + v)

    def __iter__(self):
        return iter((s, self.pre[i + 1] - self.pre[i]) for i, s in enumerate(self.steps))


def wmean(c: "Curve", at: int) -> float | None:
    i, j = bisect.bisect_right(c.steps, at - W), bisect.bisect_right(c.steps, at)
    return (c.pre[j] - c.pre[i]) / (j - i) if j > i else None


def best(c: "Curve", lo: int, at: int) -> tuple[float, int] | None:
    """the best window mean since LO and where it was."""
    vals = [(wmean(c, s), s) for s in range(lo + W, at + 1, 100)]
    vals = [v for v in vals if v[0] is not None]
    return max(vals) if vals else None


def rules(cv: dict, phase: str, s0: int, at: int) -> dict[str, str | None]:
    """rule -> alarm text (None: fine; rules that do not apply yet are left out)."""
    off = at - s0
    r: dict[str, str | None] = {}
    if phase in ("M", "P") and "share" in cv and "target" in cv:
        sh, tg = wmean(cv["share"], at), wmean(cv["target"], at)
        if sh is not None and tg is not None and tg >= 0.15:
            r["share_lag"] = (f"stairs share {sh:.3f} of the curriculum's {tg:.3f} ({sh / tg:.0%} < 70%): robots do not finish "
                              "the stairs") if sh / tg < 0.70 else None
    if phase == "M" and off >= 2000:
        k = "riser" if "riser" in cv else "foothold"
        need = 2.0 if off >= 3000 else 1.2
        if k in cv:
            a, b = wmean(cv[k], s0 + 500), wmean(cv[k], at)
            if a is not None and b is not None:
                r["climb_flat"] = (f"{TAGS[k].split('/')[-1]} {b:.4f} at +{off}, {b / a if a > 0 else 0:.2f}x its +500 value "
                                   f"{a:.4f} (< {need}x): the climb is not being learned") if b < need * max(a, 1e-6) else None
    for k, name, start in (("reward", "reward_drop", 1000 if phase == "T" else 500),
                           ("eplen", "short_eps", 1500 if phase == "T" else 500)):
        if k in cv and off >= start + W:
            now, b = wmean(cv[k], at), best(cv[k], s0 + 500, at)
            if now is None or b is None or (k == "reward" and b[0] <= 5):
                continue
            top, where = b
            # harder terrain explains a lower reward / shorter episodes: only when the terrain level did not rise by
            # 0.5 since the best (B2 rp4 T: reward 51 -> 28 while the level went 1.8 -> 4.4, and it passed)
            tl = (wmean(cv["terrain"], where), wmean(cv["terrain"], at)) if "terrain" in cv else (None, None)
            harder = tl[0] is not None and tl[1] is not None and tl[1] >= tl[0] + 0.5
            r[name] = (f"{TAGS[k].split('/')[-1]} {now:.1f}, under half its best {top:.1f} (it {where}) with the terrain level "
                       f"{'?' if tl[0] is None else f'{tl[0]:.1f} -> {tl[1]:.1f}'}"
                       if now < 0.5 * top and not harder else None)
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--phase", required=True, choices=("T", "M", "P"))
    ap.add_argument("--log", default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--pid", type=int, default=None)
    ap.add_argument("--every", type=int, default=600)
    ap.add_argument("--replay", action="store_true")
    a = ap.parse_args()
    label = a.label or f"{os.path.basename(a.run.rstrip('/'))} {a.phase}"
    state: dict[str, bool] = {}

    def say(at: int, s0: int, alarms: dict[str, str | None]) -> None:
        for k, txt in alarms.items():
            on = txt is not None
            if on == state.get(k, False):
                continue
            state[k] = on
            line = (f"{label} CURVE ALARM {k} (+{at - s0}, it {at}): {txt}" if on else
                    f"{label} curve back: {k} (+{at - s0}, it {at})")
            if a.log and not a.replay:
                with open(a.log, "a") as f:
                    f.write(f"{time.strftime('%m-%d %H:%M:%S')} {line}\n")
            print(line, flush=True)

    if a.replay:
        cv = load(a.run)
        steps = [s for c in cv.values() for s, _ in c]
        s0, s1 = min(steps), max(steps)
        for at in range(s0 + 100, s1 + 1, 100):
            say(at, s0, rules(cv, a.phase, s0, at))
        print(f"{label}: replayed +0..+{s1 - s0}; alarms on at the end: {[k for k, v in state.items() if v] or 'none'}")
        return 0
    while True:
        alive = a.pid is None or os.path.exists(f"/proc/{a.pid}")
        try:
            cv = load(a.run)
        except Exception as e:  # a half-written event file: next time
            cv = {}
            print(f"{label}: curves not read ({e})", file=sys.stderr)
        steps = [s for c in cv.values() for s, _ in c]
        if steps:
            say(max(steps), min(steps), rules(cv, a.phase, min(steps), max(steps)))
        if not alive:
            return 0
        time.sleep(a.every)


if __name__ == "__main__":
    sys.exit(main())
