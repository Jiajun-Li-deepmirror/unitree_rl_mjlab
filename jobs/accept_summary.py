"""accept_summary.py PREFIX (go2-depth 2026-11-03): one line of the numbers behind a student check (fe_check.sh's
PREFIX_<kind>.txt files): map3 reached (no operator / operator / both with the real camera's edge smoothing), the flat
acceptance matrix, the stairs set, drift (straight / sideways as accept_tables: max(|p10|, |p90|) of the offset across
the travel over |p50| of the travel; spins: largest wander p90), the stop test, the march reset, the gait clock, the
wheeled four-wheel check, and the verdict. Read-only."""
import re
import sys

P = sys.argv[1]


def read(kind):
    try:
        return open(f"{P}_{kind}.txt", errors="replace").read()
    except FileNotFoundError:
        return ""


def reached(kind):
    v = [float(x) for x in re.findall(r"reached=([0-9.]+)", read(kind))]
    return f"{100 * sum(v) / len(v):.1f}" if v else "-"


def rows(text, ncols):
    out = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= ncols and re.match(r"^\(?-?[0-9.]", parts[0].lstrip("( ")) or (len(parts) >= ncols and parts[0][:1] == "("):
            out.append(parts)
    return out


def falls(cell):
    m = re.match(r"(\d+)/(\d+)", cell.strip())
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


res = {}
res["map3"] = f"{reached('noop')}/{reached('op')}"
res["edge"] = f"{reached('noopedge')}/{reached('opedge')}"
am = read("am")
m = re.search(r"failed (\d+) of (\d+)", am)
amf = sum(falls(r[-2])[0] for r in rows(am, 4) if re.match(r"\d+/\d+", r[-2]))
res["am"] = f"{m.group(1)}/{m.group(2)} fail, {amf} fell" if m else "-"
st = re.findall(r"overall: (.*)", read("stairs"))
res["stairs"] = st[-1] if st else "-"

drift = read("drift3")
straight, side, spin, dfell = [], [], [], 0
for r in rows(drift, 7):
    try:
        cmd = tuple(float(x) for x in r[0].strip("() ").split(","))
    except ValueError:
        continue
    num = lambda s: [float(x) for x in s.split()]  # noqa: E731
    head, along, across, end, far = num(r[1]), num(r[2]), num(r[3]), num(r[4]), num(r[5])
    dfell += falls(r[6])[0]
    vx, vy, wz = cmd
    if vx and not vy and not wz:
        straight.append(100 * max(abs(across[0]), abs(across[2])) / abs(along[1]))
    elif vy and not vx and not wz:
        side.append(100 * max(abs(along[0]), abs(along[2])) / abs(across[1]))
    elif wz and not vx and not vy:
        spin.append(far[1])
res["drift"] = (f"straight {max(straight):.1f}%, side {max(side):.1f}%, spin wander {max(spin):.2f} m, fell {dfell}"
                if straight and side else "-")

stand = read("stand")
stop, creep50, creep90, sfell = {}, [], [], 0
for line in stand.splitlines():
    parts = [p.strip() for p in line.split("|")]
    if len(parts) >= 6 and re.match(r"^(rest|fwd|back|side|turn|arc)", parts[0]):
        s = parts[1].split()
        c = parts[2].split()
        stop[parts[0]] = float(s[1])
        creep50.append(float(c[0]))
        creep90.append(float(c[1]))
        sfell += falls(parts[-1])[0]
res["stop"] = (f"stop fwd/back 1.0 {stop.get('fwd 1.0', 0):.1f}/{stop.get('back 1.0', 0):.1f} cm, "
               f"creep p50 max {max(creep50):.2f} cm (p90 {max(creep90):.2f}), fell {sfell}" if creep50 else "-")

mr = re.findall(r"after ([0-9.]+) deg, worst robot ([0-9.]+) deg \| march 5 s drift ([0-9.]+) cm, worst ([0-9.]+) cm.*fell (\d+/\d+)", read("march"))
res["march"] = f"reset {mr[-1][0]} deg (worst {mr[-1][1]}), drift {mr[-1][2]} cm (worst {mr[-1][3]}), fell {mr[-1][4]}" if mr else "-"

clock = read("clock")
m = re.search(r"failed (\d+) of (\d+)", clock)
judged, side06 = [], {}
for line in clock.splitlines():
    parts = [p.strip() for p in line.split("|")]
    if len(parts) >= 5 and parts[-1] in ("ok", "FAIL"):
        v = float(parts[1].rstrip("%"))
        judged.append(v)
        if parts[0] in ("(0, 0.6, 0)", "(0, -0.6, 0)"):
            side06[parts[0]] = v
res["clock"] = (f"{m.group(1)}/{m.group(2)} fail, min {min(judged):.0f}%, side ±0.6 "
                f"{side06.get('(0, 0.6, 0)', 0):.0f}/{side06.get('(0, -0.6, 0)', 0):.0f}%") if m and judged else "-"

gait = read("gait")
g = []
for line in gait.splitlines():
    parts = [p.strip() for p in line.split("|")]
    if len(parts) >= 6 and parts[0].startswith("("):
        g.append((float(parts[2].rstrip("%")), falls(parts[5])[0]))
res["gait"] = f"four wheels down min {min(x for x, _ in g):.0f}%, fell {sum(y for _, y in g)}" if g else "-"
res["verdict"] = read("verdict").strip() or "-"
print(" ;; ".join(f"{k}={v}" for k, v in res.items()))
