#!/usr/bin/env python3
"""Deterministic screenshot capture for the situation3d web app.

The web app runs on the remote container; this driver runs locally and only
orchestrates. It exists because `--screenshot` alone gives no usable readiness
signal: a capture may come back with the Cesium globe not yet rasterised
(uniformly black), and the same `--virtual-time-budget` value produces both
outcomes across runs. So every capture is validated by inspecting pixels, and
invalid captures are retried.

Validation: the globe fills the centre of the viewport, so a patch at the image
centre must be brighter than the globe's default base colour / empty starfield.
A rendered Natural Earth II view has a mean luma around 100-140; an unrendered
one sits near 4.

Usage:
    python capture.py --url "/?tick=360" --out shots/tick360.png
    python capture.py --url "/" --out shots/overview.png --tries 5 --vtb 40000

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


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def ssh(host: str, command: str, timeout: int = 360) -> subprocess.CompletedProcess:
    return run(["ssh", "-o", "ConnectTimeout=20", host, command], timeout=timeout)


def scp_from(host: str, remote: str, local: str, timeout: int = 180) -> bool:
    return run(["scp", "-q", f"{host}:{remote}", local], timeout=timeout).returncode == 0


def scp_to(host: str, local: Path, remote: str, timeout: int = 60) -> bool:
    return run(["scp", "-q", str(local), f"{host}:{remote}"], timeout=timeout).returncode == 0


def probe_luma(png: Path) -> tuple[float, tuple[int, int, int]]:
    """Mean luma of the centre patch, plus the mean RGB for reporting."""
    from PIL import Image

    with Image.open(png) as im:
        rgb = im.convert("RGB")
        w, h = rgb.size
        half = PROBE_SIZE // 2
        box = (w // 2 - half, h // 2 - half, w // 2 + half, h // 2 + half)
        patch = rgb.crop(box)
        px = list(patch.get_flattened_data())
    n = len(px)
    mean = tuple(sum(p[i] for p in px) // n for i in range(3))
    luma = 0.2126 * mean[0] + 0.7152 * mean[1] + 0.0722 * mean[2]
    return luma, mean


def capture(
    host: str,
    url_path: str,
    out: Path,
    tries: int,
    vtb: int,
    width: int,
    height: int,
) -> bool:
    # Ship the remote-side script each run so the two files cannot drift apart.
    remote_script = "/tmp/situation3d_remote_shot.sh"
    if not scp_to(host, REMOTE_SHOT, remote_script):
        print("ERROR: could not upload remote_shot.sh")
        return False

    url = f"http://127.0.0.1:8848{url_path}"
    out.parent.mkdir(parents=True, exist_ok=True)

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
            luma, mean = probe_luma(tmp_path)
            if luma < MIN_MEAN_LUMA:
                print(
                    f"  attempt {attempt}/{tries}: globe not rendered "
                    f"(centre mean RGB={mean}, luma={luma:.1f} < {MIN_MEAN_LUMA})"
                )
                continue
            out.write_bytes(tmp_path.read_bytes())
            print(f"  attempt {attempt}/{tries}: OK  mean RGB={mean}  luma={luma:.1f}")
            print(f"captured {out} ({out.stat().st_size} bytes)  url={url_path}")
            return True
        finally:
            tmp_path.unlink(missing_ok=True)

    print(f"FAILED after {tries} attempts: {url_path}")
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="/", help="path + query on the local web server, e.g. '/?tick=360'")
    ap.add_argument("--out", required=True, help="local output PNG path")
    ap.add_argument("--host", default=DEFAULT_HOST, help=f"ssh host alias (default {DEFAULT_HOST})")
    ap.add_argument("--tries", type=int, default=4)
    # 30000 fails often at 1600x900 (SwiftShader needs real time, not virtual
    # time); 45000 has been reliable. The retry loop covers the rest.
    ap.add_argument("--vtb", type=int, default=45000, help="virtual time budget in ms")
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=900)
    args = ap.parse_args()

    ok = capture(
        args.host,
        args.url,
        Path(args.out),
        args.tries,
        args.vtb,
        args.width,
        args.height,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
