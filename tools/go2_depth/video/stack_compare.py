"""Stack two acceptance clips into one comparison video, each under a Chinese label strip (go2-depth 2026-11-09, the
user: "把负载和无负载放在一个视频中"; top = unloaded, bottom = payload).

    python tools/go2_depth/video/stack_compare.py TOP.mp4 BOTTOM.mp4 OUT.mp4 \
        [--top-label "无负载"] [--bottom-label "负载 12 kg"] [--title "Go2-W depth_v6_w27d 平地"]

The clips come from jobs/accept_videos.sh (the one acceptance-video script: both clips of a pair from the same seed,
start and camera rule, only the payload differs). Each clip gets a 44 px strip above it (Noto Sans CJK Bold); the shorter clip holds its last frame until
the longer one ends; the strips and clips are stacked with ffmpeg's vstack (the bundled imageio-ffmpeg build has no
drawtext, hence the PNG strips). Widths must match (both 1280 from the recorders); H.264, CRF 23, 25 fps.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
STRIP_H = 44


def ffmpeg() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def probe(ff: str, path: str) -> tuple[int, int, float]:
    """(width, height, seconds) of a clip, read from ffmpeg's banner."""
    out = subprocess.run([ff, "-i", path], capture_output=True, text=True).stderr
    import re

    w, h = map(int, re.search(r"Video: .*?(\d{3,5})x(\d{3,5})", out).groups())
    hh, mm, ss = re.search(r"Duration: (\d+):(\d+):([\d.]+)", out).groups()
    return w, h, int(hh) * 3600 + int(mm) * 60 + float(ss)


def strip(text: str, width: int, out: Path) -> None:
    im = Image.new("RGB", (width, STRIP_H), (24, 24, 24))
    d = ImageDraw.Draw(im)
    font = ImageFont.truetype(FONT, 28)
    tw = d.textlength(text, font=font)
    d.text(((width - tw) / 2, 5), text, font=font, fill=(255, 230, 120))
    im.save(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("top")
    ap.add_argument("bottom")
    ap.add_argument("out")
    ap.add_argument("--top-label", default="无负载")
    ap.add_argument("--bottom-label", default="负载")
    ap.add_argument("--title", default=None, help="prefix of both labels, e.g. the robot and model")
    args = ap.parse_args()
    ff = ffmpeg()
    (w0, _, d0), (w1, _, d1) = probe(ff, args.top), probe(ff, args.bottom)
    if w0 != w1:
        raise SystemExit(f"widths differ: {w0} vs {w1}")
    pre = f"{args.title} · " if args.title else ""
    with tempfile.TemporaryDirectory() as tmp:
        s0, s1 = Path(tmp, "top.png"), Path(tmp, "bottom.png")
        strip(pre + args.top_label, w0, s0)
        strip(pre + args.bottom_label, w1, s1)
        end = max(d0, d1)
        pad0, pad1 = end - d0, end - d1
        graph = (f"[0:v]tpad=stop_mode=clone:stop_duration={pad0:.3f},fps=25[a];"
                 f"[1:v]tpad=stop_mode=clone:stop_duration={pad1:.3f},fps=25[b];"
                 "[2:v][a][3:v][b]vstack=inputs=4[v]")
        cmd = [ff, "-y", "-loglevel", "error", "-i", args.top, "-i", args.bottom,
               "-loop", "1", "-t", f"{end:.3f}", "-i", str(s0), "-loop", "1", "-t", f"{end:.3f}", "-i", str(s1),
               "-filter_complex", graph, "-map", "[v]", "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p",
               "-r", "25", args.out]
        subprocess.run(cmd, check=True)
    w, h, d = probe(ff, args.out)
    print(json.dumps({"out": args.out, "size": f"{w}x{h}", "seconds": round(d, 1)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
