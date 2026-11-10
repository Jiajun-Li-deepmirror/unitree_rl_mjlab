"""Deploy a robot's new depth student and / or blind policy into tinynav-sim (go2-depth 2026-11-10; replaces the
per-version deploy_<robot>_v<N>.sh scripts).

    python tools/go2_depth/deploy/deploy_tinynav.py ROBOT [--depth ONNX NAME] [--blind ONNX NAME]
        [--tree MUJOCO_DIR ...] [--container tnsim-web] [--no-restart] [--dry-run]

NAME is the file name without .onnx (e.g. go2w_depth_v6_w27d). A blind ONNX comes from
scripts/import_blind_policy.py (it carries the contract and the contact solver). For each tinynav-sim mujoco/
tree (default: the web viewer's /home/dm/tinynav-sim/mujoco and the acceptance tree tnsim-wheeled):

  - copies the ONNX to assets/policy/NAME.onnx,
  - points sim/robots.py's ROBOT entry at it (perception= / blind=),
  - removes the robot's replaced policy file (assets/policy keeps only the deployed models, the user 2026-11-10),
  - rewrites assets/policy/policy.sha256 for the files there.

Then the tests in the viewer container (sim/tests, robot_emu/tests) and a viewer restart. The README's model table
(mujoco/README.md) is edited by hand: the script prints the row to update. The Jetson side is robot_ctrl's
tools/stage_model.sh.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

TREES = ["/home/dm/tinynav-sim/mujoco", "/home/dm/.cache/go2-depth-work/tnsim-wheeled/mujoco"]


def robot_entry(src: str, robot: str) -> tuple[int, int]:
    """Span of `"robot": Robot(...)` in robots.py's ROBOTS dict."""
    m = re.search(r'^\s*"' + re.escape(robot) + r'":\s*Robot\(', src, re.M)
    if not m:
        raise SystemExit(f"no {robot!r} entry in robots.py")
    depth, i = 0, m.end() - 1
    while True:
        c = src[i]
        depth += c == "("
        depth -= c == ")"
        if depth == 0:
            return m.start(), i + 1
        i += 1


def deploy(tree: Path, robot: str, kind: str, onnx: Path, name: str, dry: bool) -> None:
    rp = tree / "sim/robots.py"
    src = rp.read_text()
    a, b = robot_entry(src, robot)
    entry = src[a:b]
    key = "perception" if kind == "depth" else "blind"
    m = re.search(key + r'="([^"]+)"', entry)
    if not m:
        raise SystemExit(f"{robot}: no {key}= in its robots.py entry")
    old = m.group(1)
    new = f"{name}.onnx"
    pol = tree / "assets/policy"
    print(f"{tree}: {robot} {key} {old} -> {new}")
    if dry:
        return
    shutil.copyfile(onnx, pol / new)
    rp.write_text(src[:a] + entry.replace(f'{key}="{old}"', f'{key}="{new}"') + src[b:])
    still_used = re.findall(r'(?:perception|blind)="([^"]+)"', rp.read_text())
    if old != new and old not in still_used and (pol / old).exists():
        (pol / old).unlink()
        print(f"  removed {old}")


def rewrite_sha(tree: Path) -> None:
    pol = tree / "assets/policy"
    lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}" for p in sorted(pol.glob("*.onnx"))]
    (pol / "policy.sha256").write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("robot")
    ap.add_argument("--depth", nargs=2, metavar=("ONNX", "NAME"))
    ap.add_argument("--blind", nargs=2, metavar=("ONNX", "NAME"))
    ap.add_argument("--tree", action="append", help="tinynav-sim mujoco/ dir (repeatable; default both trees)")
    ap.add_argument("--container", default="tnsim-web")
    ap.add_argument("--no-restart", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not (args.depth or args.blind):
        ap.error("give --depth and / or --blind")
    for kind, val in (("depth", args.depth), ("blind", args.blind)):
        if val and not Path(val[0]).is_file():
            ap.error(f"{kind}: no file {val[0]}")
    for t in map(Path, args.tree or TREES):
        for kind, val in (("depth", args.depth), ("blind", args.blind)):
            if val:
                deploy(t, args.robot, kind, Path(val[0]), val[1], args.dry_run)
        if not args.dry_run:
            rewrite_sha(t)
    if args.dry_run:
        return 0
    tests = ("cd /workspace/tinynav-sim/mujoco && python -m pytest -q sim/tests 2>&1 | tail -n 1; "
             "python -m pytest -q robot_emu/tests 2>&1 | tail -n 1")
    subprocess.run(["docker", "exec", args.container, "bash", "-c", tests], check=False)
    if not args.no_restart:
        subprocess.run(["docker", "restart", args.container], check=False, stdout=subprocess.DEVNULL)
        print(f"restarted {args.container}")
    print(f"update mujoco/README.md's {args.robot} row by hand (model, date, acceptance)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
