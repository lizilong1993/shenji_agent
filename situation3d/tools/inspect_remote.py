#!/usr/bin/env python3
"""Runtime inspection for situation3d, over the DevTools protocol.

Why this exists
---------------
`tools/capture.py` can only judge the rendered pixels. It cannot say *which*
imagery levels Cesium asked for, how many surface tiles are resident, or why a
view stayed coarse, because `--screenshot` gives no channel back from the page.

`--dump-dom` looks like it should, and does not: it serialises the DOM once the
load event and the virtual-time budget are done, which for a WebGL page is long
before the globe has rasterised. Measured against a URL that a screenshot of the
same page renders fine, the DOM dump reported `surface tiles 0`, `imagery
levels min=0 max=1`, `tiles rendered false` -- i.e. it describes the page before
the first frame, not after it. So DOM dumping is not a readiness signal either.

This tool launches headless Chromium with `--remote-debugging-port` and speaks
the DevTools protocol over a hand-rolled WebSocket client, then waits for the
tile queue to drain before reading. That makes the app's own state and the tile
requests it issued both readable at a moment we choose.

Everything is standard library: the container has no pip, no node and no npm.

Usage (on the container, from the repository root):

    python3 situation3d/tools/inspect_remote.py \
        --url "http://127.0.0.1:8848/?cam=121.57,25.03,6000,-45" \
        --out /tmp/inspect.json

Options:
    --url URL          page to load (required)
    --out PATH         write the report as JSON (default: stdout)
    --shot PATH        also save the final frame as PNG
    --wait-ms N        settle time after the tile queue drains (default 1500)
    --timeout-ms N     give up waiting for the queue (default 90000)
    --width/--height   viewport (default 1600x900)
    --chrome PATH      chromium binary
    --keep             leave the browser running and print its ws endpoint
    --debug            dump chromium stderr
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

CHROMEDEPS = os.environ.get("CHROMEDEPS", "/tmp/chromedeps")
DEFAULT_CHROME = (
    "/remote-home/lizilong/.cache/ms-playwright/chromium-1223/chrome-linux64/chrome"
)


# ---------------------------------------------------------------------------
# DevTools transport: HTTP discovery + a minimal WebSocket client
# ---------------------------------------------------------------------------


class CdpError(RuntimeError):
    pass


class DevTools:
    def __init__(self, ws_url: str, timeout: float = 60.0):
        from urllib.parse import urlparse

        u = urlparse(ws_url)
        # Chromium hands back `ws://localhost:PORT/...`; connecting by name can
        # pick the IPv6 loopback while it listens on IPv4.
        host = u.hostname or "127.0.0.1"
        if host in ("localhost", "::1"):
            host = "127.0.0.1"
        port = u.port or 80
        path = u.path + (("?" + u.query) if u.query else "")
        self.sock = socket.create_connection((host, port), timeout=20)
        self.sock.settimeout(timeout)
        self._buf = b""
        self._id = 0
        self._handshake(host, port, path)

    def _handshake(self, host: str, port: int, path: str) -> None:
        key = base64.b64encode(os.urandom(16)).decode()
        req = (
            "GET %s HTTP/1.1\r\n"
            "Host: %s:%d\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            "Sec-WebSocket-Key: %s\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n" % (path, host, port, key)
        )
        self.sock.sendall(req.encode())
        while b"\r\n\r\n" not in self._buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise CdpError("devtools closed during handshake")
            self._buf += chunk
        head, _, rest = self._buf.partition(b"\r\n\r\n")
        if b" 101" not in head.split(b"\r\n")[0]:
            raise CdpError("websocket upgrade failed: %s" % head.split(b"\r\n")[0])
        self._buf = rest

    # -- framing ----------------------------------------------------------

    def _recv_exact(self, n: int) -> bytes:
        while len(self._buf) < n:
            chunk = self.sock.recv(max(4096, n - len(self._buf)))
            if not chunk:
                raise CdpError("devtools connection closed")
            self._buf += chunk
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def _recv_frame(self) -> tuple[int, bytes]:
        b0, b1 = self._recv_exact(2)
        opcode = b0 & 0x0F
        masked = bool(b1 & 0x80)
        length = b1 & 0x7F
        if length == 126:
            (length,) = struct.unpack(">H", self._recv_exact(2))
        elif length == 127:
            (length,) = struct.unpack(">Q", self._recv_exact(8))
        mask = self._recv_exact(4) if masked else None
        payload = self._recv_exact(length) if length else b""
        if mask:
            payload = bytes(v ^ mask[i & 3] for i, v in enumerate(payload))
        return opcode, payload

    def _send_text(self, text: str) -> None:
        data = text.encode("utf-8")
        mask = os.urandom(4)
        n = len(data)
        header = bytearray([0x81])
        if n < 126:
            header.append(0x80 | n)
        elif n < 65536:
            header.append(0x80 | 126)
            header += struct.pack(">H", n)
        else:
            header.append(0x80 | 127)
            header += struct.pack(">Q", n)
        header += mask
        header += bytes(v ^ mask[i & 3] for i, v in enumerate(data))
        self.sock.sendall(bytes(header))

    # -- request / response ----------------------------------------------

    def call(self, method: str, **params):
        self._id += 1
        mid = self._id
        self._send_text(json.dumps({"id": mid, "method": method, "params": params}))
        deadline = time.time() + 120
        while time.time() < deadline:
            opcode, payload = self._recv_frame()
            if opcode == 0x8:
                raise CdpError("devtools sent close")
            if opcode != 0x1:
                continue  # ping / pong / binary
            try:
                msg = json.loads(payload.decode("utf-8"))
            except ValueError:
                continue
            if msg.get("id") == mid:
                if "error" in msg:
                    raise CdpError("%s: %s" % (method, msg["error"]))
                return msg.get("result", {})
        raise CdpError("timeout waiting for %s" % method)

    def evaluate(self, expression: str, await_promise: bool = False):
        res = self.call(
            "Runtime.evaluate",
            expression=expression,
            returnByValue=True,
            awaitPromise=await_promise,
        )
        if res.get("exceptionDetails"):
            det = res["exceptionDetails"]
            raise CdpError(
                "page exception: %s"
                % (
                    det.get("exception", {}).get("description")
                    or det.get("text")
                )
            )
        return res.get("result", {}).get("value")

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


def http_json(url: str, timeout: float = 5.0):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_for_devtools(port: int, timeout: float) -> dict:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            return http_json("http://127.0.0.1:%d/json/version" % port)
        except Exception as exc:  # noqa: BLE001 - retried
            last = exc
            time.sleep(0.25)
    raise CdpError("devtools endpoint never came up: %s" % last)


def first_page_target(port: int) -> dict:
    targets = http_json("http://127.0.0.1:%d/json/list" % port)
    for t in targets:
        if t.get("type") == "page":
            return t
    if not targets:
        raise CdpError("no devtools targets")
    return targets[0]


# ---------------------------------------------------------------------------
# page-side probe
# ---------------------------------------------------------------------------
#
# Installed before navigation, so it sees every request from the first frame.
# It only reads; nothing here changes how the app renders.

PROBE_SOURCE = r"""
(() => {
  if (window.__probeInstalled) return;
  window.__probeInstalled = true;
  const R = { tileRequests: [], errors: [], frames: 0, providers: {} };
  window.__probe = R;

  // Hook the shared imagery-provider prototype rather than a provider instance:
  // the app's own instrument() only wraps the basemap provider, so the urban
  // layers would otherwise be invisible.
  function hook(CtorTag) {
    const proto = CtorTag && CtorTag.prototype;
    if (!proto || proto.__probeHooked) return false;
    const original = proto.requestImage;
    if (typeof original !== 'function') return false;
    proto.requestImage = function (x, y, level, request) {
      try {
        const rect = this.rectangle;
        R.tileRequests.push({
          x: x, y: y, level: level,
          maxLevel: this.maximumLevel,
          rect: rect ? [rect.west, rect.south, rect.east, rect.north]
            .map(function (v) { return +(v * 180 / Math.PI).toFixed(4); }) : null,
        });
        if (R.tileRequests.length > 60000) R.tileRequests.splice(0, 30000);
      } catch (e) { /* never break rendering for bookkeeping */ }
      return original.call(this, x, y, level, request);
    };
    proto.__probeHooked = true;
    return true;
  }
  const timer = setInterval(function () {
    if (window.Cesium && window.Cesium.UrlTemplateImageryProvider) {
      if (hook(window.Cesium.UrlTemplateImageryProvider)) clearInterval(timer);
    }
  }, 20);

  window.addEventListener('error', function (e) {
    R.errors.push(String(e.message || e.error || 'error').slice(0, 240));
  });
  window.addEventListener('unhandledrejection', function (e) {
    R.errors.push('unhandled: ' +
      String((e.reason && e.reason.message) || e.reason).slice(0, 240));
  });

  function tick() { R.frames++; requestAnimationFrame(tick); }
  requestAnimationFrame(tick);
})();
"""


def settle(cdp: DevTools, timeout_ms: int) -> dict:
    """Wait until the globe stops fetching, then report what it settled into."""
    deadline = time.time() + timeout_ms / 1000.0
    sig = None
    stable = 0
    last = {}
    while time.time() < deadline:
        snap = cdp.evaluate(
            r"""(() => {
              const p = window.__probe || { tileRequests: [], frames: 0 };
              const v = (window.SituationHost && window.SituationHost.viewer)
                ? window.SituationHost.viewer() : null;
              const g = v && v.scene && v.scene.globe;
              return {
                frames: p.frames,
                requests: p.tileRequests.length,
                loaded: g ? g.tilesLoaded : null,
                queue: g && g._surface && g._surface._tileLoadQueue
                  ? g._surface._tileLoadQueue.length : -1,
              };
            })()"""
        )
        if not snap:
            break
        last = snap
        cur = (snap.get("loaded"), snap.get("queue"), snap.get("requests"))
        if cur == sig:
            stable += 1
        else:
            stable = 0
        sig = cur
        # `frames > 30` keeps a "nothing loaded yet" page from passing the test.
        if snap.get("frames", 0) > 30 and snap.get("loaded") is True and stable >= 4:
            break
        time.sleep(0.5)
    return last


REPORT_SOURCE = r"""(() => {
  const p = window.__probe || { tileRequests: [], errors: [], frames: 0 };
  // The app keeps its viewer inside a closure; window.SituationHost is the
  // interface it publishes for the situation module (src/situation.js).
  const v = (window.SituationHost && window.SituationHost.viewer)
    ? window.SituationHost.viewer() : null;
  const scene = v && v.scene;
  const globe = scene && scene.globe;
  const cam = v && v.camera;
  const layers = [];
  if (v && v.imageryLayers) {
    for (let i = 0; i < v.imageryLayers.length; i++) {
      const L = v.imageryLayers.get(i);
      const prov = L.imageryProvider;
      const rect = prov && prov.rectangle;
      layers.push({
        i: i,
        show: L.show,
        minLevel: prov && prov.minimumLevel,
        maxLevel: prov && prov.maximumLevel,
        ready: prov && prov.ready,
        rect: rect ? [rect.west, rect.south, rect.east, rect.north]
          .map(function (x) { return +(x * 180 / Math.PI).toFixed(4); }) : null,
        url: prov && prov.url ? String(prov.url).slice(0, 80) : null,
      });
    }
  }
  const hist = {};
  const byLayer = {};
  for (let i = 0; i < p.tileRequests.length; i++) {
    const t = p.tileRequests[i];
    hist[t.level] = (hist[t.level] || 0) + 1;
    const key = (t.rect ? t.rect.join(',') : 'no-rect') + ' maxLevel=' + t.maxLevel;
    if (!byLayer[key]) byLayer[key] = { requests: 0, maxLevel: -1, levels: {} };
    byLayer[key].requests++;
    byLayer[key].levels[t.level] = (byLayer[key].levels[t.level] || 0) + 1;
    if (t.level > byLayer[key].maxLevel) byLayer[key].maxLevel = t.level;
  }
  const pos = cam && cam.positionCartographic;
  const surface = globe && globe._surface;
  return {
    cesium: window.Cesium && window.Cesium.VERSION,
    camera: pos ? {
      lon: +(pos.longitude * 180 / Math.PI).toFixed(6),
      lat: +(pos.latitude * 180 / Math.PI).toFixed(6),
      heightKm: +(pos.height / 1000).toFixed(3),
    } : null,
    framesRendered: p.frames,
    globeTilesLoaded: globe ? globe.tilesLoaded : null,
    surfaceTilesLoaded: surface ? surface.tilesLoaded : null,
    tileLoadQueue: surface && surface._tileLoadQueue
      ? surface._tileLoadQueue.length : null,
    imageryRequests: p.tileRequests.length,
    imageryLevelHistogram: hist,
    tileLevelsByLayer: byLayer,
    imageryLayers: layers,
    pageErrors: p.errors.slice(0, 12),
    diagText: (document.getElementById('diagBox') || {}).textContent || null,
    dom: (() => {
      const list = document.getElementById('targetList');
      const rows = list ? list.querySelectorAll('tr').length : -1;
      return { targetRows: rows, title: document.title };
    })(),
  };
})()"""


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", required=True)
    ap.add_argument("--out")
    ap.add_argument("--shot")
    ap.add_argument("--wait-ms", type=int, default=1500)
    ap.add_argument("--timeout-ms", type=int, default=90000)
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--chrome", default=DEFAULT_CHROME)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.chrome):
        print("ERROR: chromium not found at %s" % args.chrome, file=sys.stderr)
        return 2
    if not os.path.isdir(os.path.join(CHROMEDEPS, "usr/lib/x86_64-linux-gnu")):
        print("ERROR: %s missing -- run tools/bootstrap_remote_headless.sh" % CHROMEDEPS,
              file=sys.stderr)
        return 3

    port = 9200 + random.randint(0, 500)
    profile = tempfile.mkdtemp(prefix="situation3d-cdp-")
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ":".join([
        os.path.join(CHROMEDEPS, "usr/lib/x86_64-linux-gnu"),
        os.path.join(CHROMEDEPS, "lib/x86_64-linux-gnu"),
        os.path.join(CHROMEDEPS, "usr/lib"),
    ])
    env["FONTCONFIG_FILE"] = os.path.join(CHROMEDEPS, "fonts.conf")
    env["FONTCONFIG_PATH"] = CHROMEDEPS
    env["XDG_CONFIG_HOME"] = os.path.join(profile, "cfg")
    env["XDG_CACHE_HOME"] = os.path.join(profile, "cache")
    env.setdefault("HOME", "/remote-home/lizilong")
    os.makedirs(env["XDG_CONFIG_HOME"], exist_ok=True)
    os.makedirs(env["XDG_CACHE_HOME"], exist_ok=True)

    proc = subprocess.Popen(
        [
            args.chrome,
            "--headless=new",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-setuid-sandbox",
            "--disable-gpu-sandbox",
            "--enable-unsafe-swiftshader",
            "--use-gl=angle",
            "--use-angle=swiftshader",
            "--font-render-hinting=none",
            "--remote-debugging-port=%d" % port,
            "--user-data-dir=%s" % profile,
            "--window-size=%d,%d" % (args.width, args.height),
            "--hide-scrollbars",
            "about:blank",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    cdp = None
    report: dict = {}
    try:
        wait_for_devtools(port, 40)
        target = first_page_target(port)
        cdp = DevTools(target["webSocketDebuggerUrl"])
        cdp.call("Page.enable")
        cdp.call("Runtime.enable")
        cdp.call("Page.addScriptToEvaluateOnNewDocument", source=PROBE_SOURCE)
        cdp.call("Page.navigate", url=args.url)

        # Give the page a moment to create the viewer before polling: the tile
        # queue test is meaningless while `window.viewer` does not exist yet.
        deadline = time.time() + 60
        while time.time() < deadline:
            if cdp.evaluate("!!(window.Cesium && window.SituationHost && "
                            "window.SituationHost.viewer && window.SituationHost.viewer())"):
                break
            time.sleep(0.5)
        report["ready"] = cdp.evaluate(
            "({ cesium: !!window.Cesium, host: !!window.SituationHost, "
            "title: document.title, "
            "status: (document.getElementById('statusBar')||document.getElementById('status')||{}).textContent || null })"
        )

        report["settle"] = settle(cdp, args.timeout_ms)
        time.sleep(max(0, args.wait_ms) / 1000.0)
        report.update(cdp.evaluate(REPORT_SOURCE))

        if args.shot:
            shot = cdp.call("Page.captureScreenshot", format="png")
            Path(args.shot).write_bytes(base64.b64decode(shot["data"]))
            report["shot"] = args.shot
    finally:
        if cdp is not None:
            cdp.close()
        if not args.keep:
            try:
                proc.send_signal(signal.SIGKILL)
            except OSError:
                pass
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            shutil.rmtree(profile, ignore_errors=True)
        else:
            print("browser kept on port %d (profile %s)" % (port, profile))

    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print("report: %s" % args.out)
    else:
        print(text)

    if args.debug:
        try:
            proc.stdout.close()
            err = proc.stderr.read().decode("utf-8", "replace")
            print(err[-4000:], file=sys.stderr)
        except (OSError, ValueError):
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
