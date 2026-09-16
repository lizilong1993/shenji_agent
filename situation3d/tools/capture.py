#!/usr/bin/env python3
"""Deterministic screenshot capture for the situation3d web app.

The web app runs on the remote container; this driver runs locally and only
orchestrates. It exists because `--screenshot` alone gives no usable readiness
signal: a capture may come back with the Cesium globe not yet rasterised
(uniformly black), and the same `--virtual-time-budget` value produces both
outcomes across runs. So every capture is validated by inspecting pixels, and
invalid captures are retried.

Validation: two independent checks, because a screenshot can be wrong in two
different ways.

1. **Rendered at all.** The globe fills the centre of the viewport, so a patch
   there must be brighter than the empty starfield. An unrendered globe sits
   near luma 4.
2. **Rendered at the right detail level.** A globe whose imagery tiles have not
   finished refining is still bright -- it shows a coarse level stretched over
   the surface -- and reads as a smooth gradient. Mean luma cannot tell that
   apart from a finished frame, but edge energy can: a refined view measures
   about 1.9 mean absolute gradient across the patch, a coarse one about 0.13.
   This is what makes captures reproducible without guessing a
   `--virtual-time-budget`.

Usage:
    python capture.py --url "/?case=joint_full&tick=360" --out shots/tick360.png
    python capture.py --url "/?case=joint_full&follow=R01" --out shots/follow.png --tries 5
    python capture.py --url "/?components=0" --out shots/nocomp.png

Note on URLs: once the empty-state change lands, `/` no longer loads a situation, so
every capture that needs targets must carry `?case=<id>` explicitly.

Caveat on low-altitude views: imagery is pre-downloaded to z11 across the theatre
(z12-14 only over the urban AOIs), so a camera a few kilometres above open sea
renders heavily stretched imagery and legitimately fails the edge-energy gate.
Use `--keep-failed` to inspect the frame instead of chasing it.

Requires: Pillow locally, ssh/scp access to the remote host, and
bootstrap_remote_headless.sh already run on the remote.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REMOTE_SHOT = HERE / "remote_shot.sh"

DEFAULT_HOST = "server-10-254-1-155-user"
REMOTE_BASE = "/remote-home/lizilong/shenji_agent/situation3d/web"

# A patch this size at the image centre always falls inside the globe.
PROBE_SIZE = 240
MIN_MEAN_LUMA = 40.0
# Mean absolute gradient across the probe patch. Calibrated against captures of
# the same camera: fully refined ~1.9, coarse-level placeholder ~0.13.
MIN_SHARPNESS = 0.55


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def ssh(host: str, command: str, timeout: int = 360) -> subprocess.CompletedProcess:
    return run(["ssh", "-o", "ConnectTimeout=20", host, command], timeout=timeout)


def scp_from(host: str, remote: str, local: str, timeout: int = 180) -> bool:
    return run(["scp", "-q", f"{host}:{remote}", local], timeout=timeout).returncode == 0


def scp_to(host: str, local: Path, remote: str, timeout: int = 60) -> bool:
    return run(["scp", "-q", str(local), f"{host}:{remote}"], timeout=timeout).returncode == 0


def probe(png: Path) -> tuple[float, tuple[int, int, int], float]:
    """Luma and mean RGB of the centre patch, plus its edge energy.

    The patch is taken from the middle of the globe pane rather than the middle
    of the window, because the side panels occupy roughly a third of the width.
    """
    from PIL import Image

    with Image.open(png) as im:
        rgb = im.convert("RGB")
        w, h = rgb.size
        # Globe pane: side panels are ~268 px and ~306 px wide at 1600 px.
        left = 268
        right = w - 306
        top = 48
        bottom = h - 42
        cx = (left + right) // 2
        cy = (top + bottom) // 2
        half = PROBE_SIZE // 2
        patch = rgb.crop((cx - half, cy - half, cx + half, cy + half)).convert("L")
        px = list(patch.get_flattened_data())

    n = len(px)
    side = PROBE_SIZE
    total = 0
    samples = 0
    for y in range(side - 1):
        row = y * side
        for x in range(side - 1):
            v = px[row + x]
            total += abs(px[row + x + 1] - v) + abs(px[row + side + x] - v)
            samples += 2
    sharpness = total / max(1, samples)

    rgb_patch = Image.open(png).convert("RGB").crop(
        (cx - half, cy - half, cx + half, cy + half))
    rpx = list(rgb_patch.get_flattened_data())
    mean = tuple(sum(p[i] for p in rpx) // n for i in range(3))
    luma = 0.2126 * mean[0] + 0.7152 * mean[1] + 0.0722 * mean[2]
    return luma, mean, sharpness


def capture(
    host: str,
    url_path: str,
    out: Path,
    tries: int,
    vtb: int,
    width: int,
    height: int,
    keep_failed: bool = False,
) -> bool:
    # Ship the remote-side script each run so the two files cannot drift apart.
    remote_script = "/tmp/situation3d_remote_shot.sh"
    if not scp_to(host, REMOTE_SHOT, remote_script):
        print("ERROR: could not upload remote_shot.sh")
        return False

    url = f"http://127.0.0.1:8848{url_path}"
    out.parent.mkdir(parents=True, exist_ok=True)

    last_tmp = None
    for attempt in range(1, tries + 1):
        remote_png = f"/tmp/situation3d_capture_{os.getpid()}.png"
        cmd = f"sh {remote_script} '{url}' {remote_png} {vtb} {width} {height}"
        res = ssh(host, cmd)
        if res.returncode != 0:
            print(f"  attempt {attempt}/{tries}: remote render failed: {res.stderr.strip()[:160]}")
            continue

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            if not scp_from(host, remote_png, str(tmp_path)):
                print(f"  attempt {attempt}/{tries}: scp back failed")
                continue
            luma, mean, sharpness = probe(tmp_path)
            if keep_failed:
                if last_tmp is not None:
                    last_tmp.unlink(missing_ok=True)
                last_tmp = Path(str(tmp_path) + ".keep")
                last_tmp.write_bytes(tmp_path.read_bytes())
            if luma < MIN_MEAN_LUMA:
                print(
                    f"  attempt {attempt}/{tries}: globe not rendered "
                    f"(centre mean RGB={mean}, luma={luma:.1f} < {MIN_MEAN_LUMA})"
                )
                continue
            if sharpness < MIN_SHARPNESS:
                print(
                    f"  attempt {attempt}/{tries}: imagery still at a coarse level "
                    f"(edge energy={sharpness:.3f} < {MIN_SHARPNESS})"
                )
                continue
            out.write_bytes(tmp_path.read_bytes())
            print(f"  attempt {attempt}/{tries}: OK  mean RGB={mean}  "
                  f"luma={luma:.1f}  edge={sharpness:.3f}")
            print(f"captured {out} ({out.stat().st_size} bytes)  url={url_path}")
            return True
        finally:
            tmp_path.unlink(missing_ok=True)

    if keep_failed and last_tmp is not None and last_tmp.exists():
        out.write_bytes(last_tmp.read_bytes())
        last_tmp.unlink(missing_ok=True)
        print(f"kept last (failed) frame at {out}")
    print(f"FAILED after {tries} attempts: {url_path}")
    return False


def main() -> int:
    global MIN_SHARPNESS, MIN_MEAN_LUMA
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="/", help="path + query on the local web server, e.g. '/?tick=360'")
    ap.add_argument("--out", required=True, help="local output PNG path")
    ap.add_argument("--host", default=DEFAULT_HOST, help=f"ssh host alias (default {DEFAULT_HOST})")
    ap.add_argument("--tries", type=int, default=4)
    # 30000 fails often at 1600x900 (SwiftShader needs real time, not virtual
    # time); 45000 has been reliable. The retry loop covers the rest.
    ap.add_argument("--vtb", type=int, default=60000, help="virtual time budget in ms")
    ap.add_argument("--min-sharpness", type=float, default=MIN_SHARPNESS,
                    help="minimum edge energy of the globe patch")
    # Diagnostic escape hatch: keep a frame that would normally be rejected so
    # the failure can be looked at instead of only being reported.
    ap.add_argument("--min-luma", type=float, default=MIN_MEAN_LUMA)
    ap.add_argument("--keep-failed", action="store_true",
                    help="write the last attempt even if validation fails")
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=900)
    args = ap.parse_args()

    MIN_SHARPNESS = args.min_sharpness
    MIN_MEAN_LUMA = args.min_luma
    ok = capture(
        args.host,
        args.url,
        Path(args.out),
        args.tries,
        args.vtb,
        args.width,
        args.height,
        keep_failed=args.keep_failed,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
