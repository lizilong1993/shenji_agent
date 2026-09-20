#!/usr/bin/env python3
"""Deterministic screenshot capture for the situation3d web app.

The web app runs on the remote container; this driver runs locally and only
orchestrates.

**How the remote frame is taken changed, and it matters.** It used to run
Chromium once with `--screenshot` + `--virtual-time-budget`, and validate the
resulting pixels. That approach produced black frames for every camera height
between roughly 12 km and 300 km -- three attempts at 20 km, 100 km and 300 km,
all luma 3.3, while 900 km and 6 km always worked. The conclusion recorded at the
time was "the app cannot render at those heights in a headless software
rasteriser". That was wrong, and the DevTools protocol is what showed it:

    at 20 km the page reports ~5 fps sustained, globe.tilesLoaded true,
    no context loss, no page errors

The page was rendering perfectly the whole time; the *capture* was broken.
`--virtual-time-budget` fast-forwards the page's timers while tile decode and
SwiftShader rasterisation consume real time, so with a few hundred more tiles in
view at low altitude the budget expires long before the frame is complete --
and `--screenshot` then composites a canvas that has not been drawn.

So the frame is now taken over DevTools: navigate, wait until the globe stops
fetching tiles (`settle`), settle a moment longer, then `Page.captureScreenshot`.
No virtual time is involved. Measured after the change, first attempt each:

    20 km   luma 73.3  edge 13.671      300 km  luma 83.0  edge 2.938
    12 km   luma 81.5  edge  4.498        6 km  luma 96.4  edge 13.238

Validation is still applied, because a screenshot can be wrong in two ways.
1. **Rendered at all.** The globe fills the centre of the viewport, so a patch
   there must be brighter than the empty starfield. An unrendered globe sits
   near luma 4.
2. **Rendered at the right detail level.** A globe whose imagery tiles have not
   finished refining is still bright -- it shows a coarse level stretched over
   the surface -- and reads as a smooth gradient. Mean luma cannot tell that
   apart from a finished frame, but edge energy can: a refined view measures
   about 1.9 mean absolute gradient across the patch, a coarse one about 0.13.

Usage:
    python capture.py --url "/?case=joint_full&tick=360" --out shots/tick360.png
    python capture.py --url "/?case=joint_full&follow=R01" --out shots/follow.png --tries 5
    python capture.py --url "/?components=0" --out shots/nocomp.png
    python capture.py --url "/?case=joint_full&overview=1" --out shots/overview.png

Note on URLs: `/` no longer loads a situation, so every capture that needs
targets must carry `?case=<id>` explicitly.

Caveat on low-altitude views over open water: imagery is pre-downloaded to z11
across the theatre (z12-17 only over the urban AOIs), so a camera a few
kilometres above the sea renders heavily stretched imagery and can legitimately
fail the edge-energy gate. Use `--keep-failed` to inspect the frame instead of
chasing it.

Requires: ssh/scp access to the remote host, and bootstrap_remote_headless.sh
already run on the remote. Pillow is used for the pixel probe when installed;
without it the same measurement is done by a standard-library PNG reader, so the
driver also works on a bare Python with no packages. The remote side reuses the
DevTools client from `tools/inspect_remote.py`.
"""

from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
# remote_shot.sh is kept for reference only: the plain `--screenshot` +
# `--virtual-time-budget` path it implements is what produced black frames at low
# altitude (see the module docstring). The live path goes through DevTools.
REMOTE_SHOT = HERE / "remote_shot.sh"

# The alias lives in ~/.ssh/config on the workstation. Override it with
# SITUATION3D_SSH_HOST when the checkout is driven from a machine that reaches
# the container by IP instead (for example a WSL shell whose known_hosts has no
# entry for it).
DEFAULT_HOST = os.environ.get("SITUATION3D_SSH_HOST", "server-10-254-1-155-user")
REMOTE_BASE = "/remote-home/lizilong/shenji_agent/situation3d/web"

# Extra ssh/scp options, space separated, also from the environment. Needed
# where the host is addressed by IP and the host key is not in known_hosts:
#   SITUATION3D_SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
SSH_OPTS = os.environ.get("SITUATION3D_SSH_OPTS", "").split()

# A patch this size at the image centre always falls inside the globe.
PROBE_SIZE = 240
MIN_MEAN_LUMA = 40.0
# Mean absolute gradient across the probe patch. Calibrated against captures of
# the same camera: fully refined ~1.9, coarse-level placeholder ~0.13.
MIN_SHARPNESS = 0.55


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def ssh(host: str, command: str, timeout: int = 360) -> subprocess.CompletedProcess:
    return run(["ssh", "-o", "ConnectTimeout=20", *SSH_OPTS, host, command], timeout=timeout)


def scp_from(host: str, remote: str, local: str, timeout: int = 180) -> bool:
    return run(["scp", "-q", *SSH_OPTS, f"{host}:{remote}", local],
               timeout=timeout).returncode == 0


def scp_to(host: str, local: Path, remote: str, timeout: int = 60) -> bool:
    return run(["scp", "-q", *SSH_OPTS, str(local), f"{host}:{remote}"],
               timeout=timeout).returncode == 0


# --------------------------------------------------------------------------
# PNG reading: Pillow when present, standard library otherwise
# --------------------------------------------------------------------------
#
# The measurement only ever needs the centre patch of an 8-bit RGB screenshot,
# so the standard-library path reads just the rows that overlap that patch --
# unfiltering a whole 1600x900 PNG in pure Python costs seconds, a 240-row band
# costs a fraction of that. Both paths must return identical numbers or the
# thresholds below would mean different things depending on the interpreter.


def _pixels_pil(png: Path) -> tuple[int, int, list[bytes]]:
    from PIL import Image

    with Image.open(png) as im:
        rgb = im.convert("RGB")
        w, h = rgb.size
        data = list(rgb.get_flattened_data())
    rows = [bytes(v for p in data[y * w:(y + 1) * w] for v in p) for y in range(h)]
    return w, h, rows


def _pixels_stdlib(png: Path) -> tuple[int, int, list[bytes]]:
    """Decode an 8-bit PNG to rows of RGB bytes. No interlacing support."""
    data = png.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a png: %s" % png)
    pos = 8
    idat = bytearray()
    width = height = depth = ctype = None
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        tag = data[pos + 4:pos + 8]
        chunk = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if tag == b"IHDR":
            width, height, depth, ctype, _comp, _filt, interlace = struct.unpack(
                ">IIBBBBB", chunk[:13])
            if interlace:
                raise ValueError("interlaced PNG not supported")
        elif tag == b"IDAT":
            idat += chunk
        elif tag == b"IEND":
            break
    if depth != 8 or ctype not in (2, 6):
        raise ValueError("expected an 8-bit RGB/RGBA PNG, got depth=%s ctype=%s"
                         % (depth, ctype))
    channels = 3 if ctype == 2 else 4
    raw = zlib.decompress(bytes(idat))
    stride = width * channels
    prev = bytearray(stride)
    rows: list[bytes] = []
    p = 0
    for _y in range(height):
        f = raw[p]
        p += 1
        line = bytearray(raw[p:p + stride])
        p += stride
        if f == 1:
            for i in range(channels, stride):
                line[i] = (line[i] + line[i - channels]) & 0xFF
        elif f == 2:
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 0xFF
        elif f == 3:
            for i in range(channels, stride):
                a = line[i - channels] if i >= channels else 0
                line[i] = (line[i] + ((a + prev[i]) >> 1)) & 0xFF
        elif f == 4:
            for i in range(stride):
                a = line[i - channels] if i >= channels else 0
                b = prev[i]
                c = prev[i - channels] if i >= channels else 0
                pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[i] = (line[i] + pr) & 0xFF
        elif f != 0:
            raise ValueError("unknown PNG filter %d" % f)
        prev = line
        if channels == 3:
            rows.append(bytes(line))
        else:
            rows.append(bytes(v for i in range(0, stride, 4) for v in line[i:i + 3]))
    return width, height, rows


def read_pixels(png: Path) -> tuple[int, int, list[bytes]]:
    try:
        return _pixels_pil(png)
    except ImportError:
        return _pixels_stdlib(png)


def probe(png: Path) -> tuple[float, tuple[int, int, int], float]:
    """Luma and mean RGB of the centre patch, plus its edge energy.

    The patch is taken from the middle of the globe pane rather than the middle
    of the window, because the side panels occupy roughly a third of the width.
    """
    w, h, rows = read_pixels(png)
    # Globe pane: side panels are ~268 px and ~306 px wide at 1600 px.
    left = 268
    right = w - 306
    top = 48
    bottom = h - 42
    cx = (left + right) // 2
    cy = (top + bottom) // 2
    half = PROBE_SIZE // 2

    # Centre patch as a flat RGB bytearray (patch-local coordinates).
    px = bytearray()
    for y in range(cy - half, cy + half):
        row = rows[y]
        px += row[(cx - half) * 3:(cx + half) * 3]

    n = PROBE_SIZE
    luma_sum = 0
    chan = [0, 0, 0]
    for i in range(0, len(px), 3):
        r, g, b = px[i], px[i + 1], px[i + 2]
        chan[0] += r
        chan[1] += g
        chan[2] += b
        luma_sum += 0.2126 * r + 0.7152 * g + 0.0722 * b
    count = n * n
    mean = (chan[0] // count, chan[1] // count, chan[2] // count)
    luma = luma_sum / count

    total = 0
    samples = 0
    for y in range(n - 1):
        row = y * n * 3
        nxt = (y + 1) * n * 3
        for x in range(n - 1):
            i = row + x * 3
            v = px[i]
            total += abs(px[i + 3] - v) + abs(px[nxt + x * 3] - v)
            samples += 2
    sharpness = total / max(1, samples)
    return luma, mean, sharpness


# Rendered remotely through the DevTools protocol. The remote side does the
# waiting, because only it can see whether the globe has actually finished
# fetching tiles; the local side only validates the pixels that come back.
REMOTE_CDP = r'''
import base64, json, os, random, shutil, signal, subprocess, sys, tempfile, time
# Reuse the DevTools client from inspect_remote.py rather than shipping a second
# copy: the waiting logic and the transport must not drift apart.
sys.path.insert(0, os.environ.get("SITUATION3D_TOOLS",
                                  "/remote-home/lizilong/shenji_agent/situation3d/tools"))
import inspect_remote as ir

url, out_path, timeout_ms, wait_ms, width, height = (
    sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]),
    int(sys.argv[5]), int(sys.argv[6]))

port = 9300 + random.randint(0, 300)
profile = tempfile.mkdtemp(prefix="situation3d-shot-")
env = dict(os.environ)
env["LD_LIBRARY_PATH"] = ":".join([
    os.path.join(ir.CHROMEDEPS, "usr/lib/x86_64-linux-gnu"),
    os.path.join(ir.CHROMEDEPS, "lib/x86_64-linux-gnu"),
    os.path.join(ir.CHROMEDEPS, "usr/lib"),
])
env["FONTCONFIG_FILE"] = os.path.join(ir.CHROMEDEPS, "fonts.conf")
env["FONTCONFIG_PATH"] = ir.CHROMEDEPS
env["XDG_CONFIG_HOME"] = os.path.join(profile, "cfg")
env["XDG_CACHE_HOME"] = os.path.join(profile, "cache")
env.setdefault("HOME", "/remote-home/lizilong")
os.makedirs(env["XDG_CONFIG_HOME"], exist_ok=True)
os.makedirs(env["XDG_CACHE_HOME"], exist_ok=True)

proc = subprocess.Popen([
    ir.DEFAULT_CHROME, "--headless=new", "--no-sandbox", "--disable-dev-shm-usage",
    "--disable-setuid-sandbox", "--disable-gpu-sandbox", "--enable-unsafe-swiftshader",
    "--use-gl=angle", "--use-angle=swiftshader", "--font-render-hinting=none",
    "--remote-debugging-port=%d" % port, "--user-data-dir=%s" % profile,
    "--window-size=%d,%d" % (width, height), "--hide-scrollbars", "about:blank",
], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

cdp = None
try:
    ir.wait_for_devtools(port, 40)
    target = ir.first_page_target(port)
    cdp = ir.DevTools(target["webSocketDebuggerUrl"])
    cdp.call("Page.enable")
    cdp.call("Runtime.enable")
    cdp.call("Page.navigate", url=url)

    deadline = time.time() + 60
    while time.time() < deadline:
        if cdp.evaluate("!!(window.Cesium && window.SituationHost && "
                        "window.SituationHost.viewer && window.SituationHost.viewer())"):
            break
        time.sleep(0.5)
    ir.settle(cdp, timeout_ms)
    time.sleep(max(0.0, wait_ms / 1000.0))
    shot = cdp.call("Page.captureScreenshot", format="png")
    with open(out_path, "wb") as fh:
        fh.write(base64.b64decode(shot["data"]))
    print("shot OK")
finally:
    if cdp is not None:
        try:
            cdp.close()
        except Exception:
            pass
    try:
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=10)
    except Exception:
        pass
    shutil.rmtree(profile, ignore_errors=True)
'''


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
    remote_script = f"/tmp/situation3d_cdp_shot_{os.getpid()}.py"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(REMOTE_CDP)
        local_script = Path(fh.name)
    try:
        if not scp_to(host, local_script, remote_script):
            print("ERROR: could not upload the remote capture script")
            return False
    finally:
        local_script.unlink(missing_ok=True)

    url = f"http://127.0.0.1:8848{url_path}"
    out.parent.mkdir(parents=True, exist_ok=True)

    last_tmp = None
    for attempt in range(1, tries + 1):
        remote_png = f"/tmp/situation3d_capture_{os.getpid()}.png"
        cmd = f"python3 {remote_script} '{url}' {remote_png} {vtb} 4000 {width} {height}"
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
    # Budget for waiting on the remote side until the globe stops fetching
    # tiles. No longer a Chrome virtual-time budget -- see the module docstring
    # for why that had to change.
    ap.add_argument("--vtb", type=int, default=90000,
                    help="ms to wait for the remote tile queue to drain")
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
