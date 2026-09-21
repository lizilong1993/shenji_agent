#!/usr/bin/env python3
"""交互测试：逐个点击图层开关，记录运行时状态与报错。

在远程容器上跑：
    python3 tools/interactive_test.py --out /tmp/interactive.json

设计要点
--------
* 每次切换都**从 DOM 现取复选框**并派发 change 事件，走的是与用户点击完全相同的
  代码路径（`dom.*` 缓存引用会与文档脱钩，用缓存去测测不出真实行为）。
* 每个开关切换后读一次运行时状态，并把 console.error / 未捕获异常 / 404 一起收进来。
* 判定靠"生效状态是否变化"和"是否报错"，不依赖肉眼看截图。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import inspect_remote as ir  # noqa: E402

STATE_PROBE = r"""(() => {
  const H = window.SituationHost;
  const S = H && H.state;
  const v = H && H.viewer ? H.viewer() : null;
  if (!v || !S) return { error: 'no viewer/state' };
  const layers = [];
  for (let i = 0; i < v.imageryLayers.length; i++) {
    const L = v.imageryLayers.get(i);
    const p = L.imageryProvider;
    layers.push({ i: i, show: L.show,
                  url: p && p.url ? String(p.url).slice(0, 44) : null });
  }
  const prims = [];
  for (let i = 0; i < v.scene.primitives.length; i++) {
    const p = v.scene.primitives.get(i);
    if (p && p.appearance && p.appearance.constructor &&
        p.appearance.constructor.name === 'PerInstanceColorAppearance') {
      prims.push({ i: i, show: p.show, ready: p.ready });
    }
  }
  return {
    imagery: layers,
    shownImagery: layers.filter(function (l) { return l.show; }).length,
    buildings: { has: !!S.buildingsPrimitive,
                 show: S.buildingsPrimitive ? S.buildingsPrimitive.show : null,
                 busy: !!S.buildingBusy, key: S.buildingsKey,
                 pending: !!S.pendingPrimitive },
    buildingPrims: prims,
    terrainOn: !!S.terrainOn,
    terrainType: v.terrainProvider ? v.terrainProvider.constructor.name : null,
    places: window.Places && window.Places.debugPlaces ? window.Places.debugPlaces() : null,
    wsf3dOverlay: S.wsf3dOverlayLayer ? S.wsf3dOverlayLayer.show : null,
    urbanLayers: S.urbanLayers ? S.urbanLayers.length : 0,
    hotspotLayers: S.hotspotLayers ? S.hotspotLayers.length : 0,
    note: (document.getElementById('switchNote') || {}).textContent || '',
    status: (document.getElementById('statusText') || {}).textContent || '',
  };
})()"""


def click_js(cid, on=None):
    expr = ('!b.checked' if on is None else ('true' if on else 'false'))
    return ("(() => { const b = document.getElementById(%r); if (!b) return 'missing'; "
            "b.checked = %s; b.dispatchEvent(new Event('change')); return b.checked; })()"
            % (cid, expr))


class Session:
    def __init__(self, url, width=1600, height=900):
        self.port = 9400 + random.randint(0, 200)
        self.profile = tempfile.mkdtemp(prefix="itest-")
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = ":".join([
            os.path.join(ir.CHROMEDEPS, "usr/lib/x86_64-linux-gnu"),
            os.path.join(ir.CHROMEDEPS, "lib/x86_64-linux-gnu"),
            os.path.join(ir.CHROMEDEPS, "usr/lib"),
        ])
        env["FONTCONFIG_FILE"] = os.path.join(ir.CHROMEDEPS, "fonts.conf")
        env["FONTCONFIG_PATH"] = ir.CHROMEDEPS
        env["XDG_CONFIG_HOME"] = os.path.join(self.profile, "cfg")
        env["XDG_CACHE_HOME"] = os.path.join(self.profile, "cache")
        os.makedirs(env["XDG_CONFIG_HOME"], exist_ok=True)
        os.makedirs(env["XDG_CACHE_HOME"], exist_ok=True)
        self.proc = subprocess.Popen([
            ir.DEFAULT_CHROME, "--headless=new", "--no-sandbox", "--disable-dev-shm-usage",
            "--disable-setuid-sandbox", "--disable-gpu-sandbox",
            "--enable-unsafe-swiftshader", "--use-gl=angle", "--use-angle=swiftshader",
            "--font-render-hinting=none",
            "--remote-debugging-port=%d" % self.port,
            "--user-data-dir=%s" % self.profile,
            "--window-size=%d,%d" % (width, height), "--hide-scrollbars", "about:blank",
        ], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.events = []
        ir.wait_for_devtools(self.port, 40)
        target = ir.first_page_target(self.port)
        self.cdp = ir.DevTools(target["webSocketDebuggerUrl"])
        self.cdp.call("Page.enable")
        self.cdp.call("Runtime.enable")
        self.cdp.call("Log.enable")
        orig = self.cdp._recv_frame

        def tapped():
            op, payload = orig()
            if op == 0x1:
                try:
                    msg = json.loads(payload.decode("utf-8"))
                except ValueError:
                    return op, payload
                m = msg.get("method")
                if m == "Runtime.exceptionThrown":
                    d = msg.get("params", {}).get("exceptionDetails", {})
                    desc = (d.get("exception") or {}).get("description") or d.get("text")
                    self.events.append({"kind": "exception", "text": str(desc)[:200]})
                elif m == "Runtime.consoleAPICalled":
                    t = msg.get("params", {}).get("type")
                    if t in ("error", "warning"):
                        args = msg.get("params", {}).get("args", [])
                        txt = " ".join(str(a.get("value", a.get("description", ""))) for a in args)
                        self.events.append({"kind": "console." + t, "text": txt[:200]})
                elif m == "Log.entryAdded":
                    e = msg.get("params", {}).get("entry", {})
                    u = str(e.get("url", ""))
                    if e.get("level") == "error" and "favicon" not in u:
                        self.events.append({"kind": "network", "url": u[:120],
                                            "text": str(e.get("text"))[:100]})
            return op, payload

        self.cdp._recv_frame = tapped
        self.cdp.call("Page.navigate", url=url)

    def wait_ready(self, timeout=90):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.cdp.evaluate("!!(window.SituationHost && window.SituationHost.viewer() && window.Cesium)"):
                return True
            time.sleep(0.5)
        return False

    def ev(self, js):
        return self.cdp.evaluate(js)

    def state(self):
        return self.ev(STATE_PROBE)

    def close(self):
        try:
            self.cdp.close()
        except Exception:
            pass
        try:
            self.proc.send_signal(signal.SIGKILL)
            self.proc.wait(timeout=10)
        except Exception:
            pass
        shutil.rmtree(self.profile, ignore_errors=True)


# (id, 说明, 切换后需要等待的秒数)  —— 建筑体块要等几何准备，给得久一点
TOGGLES = [
    ("modelPlaces", "地名", 2),
    ("modelPlaces", "地名（再开）", 2),
    ("modelWsf3dOverlay", "建筑高度图层 WSF3D（开）", 3),
    ("modelWsf3dOverlay", "建筑高度图层 WSF3D（关）", 1),
    ("modelTianditu", "天地图（关）", 2),
    ("modelImagery", "离线影像底图（开）", 2),
    ("modelImagery", "离线影像底图（关）", 1),
    ("modelTianditu", "天地图（开）", 2),
    ("modelHotspotImagery", "热点区域高清影像（关）", 1),
    ("modelHotspotImagery", "热点区域高清影像（开）", 1),
    ("modelUrbanImagery", "城区高清影像（关）", 2),
    ("modelUrbanImagery", "城区高清影像（开）", 2),
    ("modelTerrain", "地形模型 DEM（开）", 12),
    ("modelTerrain", "地形模型 DEM（关）", 3),
    ("modelBuildings", "建筑体块（开）", 14),
    ("modelBuildings", "建筑体块（关）", 2),
    ("modelWeather", "气象模型（开）", 2),
    ("modelWeather", "气象模型（关）", 1),
    ("compCompass", "指南针（关）", 1),
    ("compCompass", "指南针（开）", 1),
    ("compScalebar", "比例尺（关）", 1),
    ("compScalebar", "比例尺（开）", 1),
    ("compViewpoint", "视点坐标（关）", 1),
    ("compViewpoint", "视点坐标（开）", 1),
    ("layerTargets", "可见目标（关）", 1),
    ("layerTargets", "可见目标（开）", 2),
    ("layerLabels", "目标标牌（关）", 1),
    ("layerLabels", "目标标牌（开）", 1),
    ("layerTracks", "历史轨迹（关）", 2),
    ("layerTracks", "历史轨迹（开）", 2),
    ("modelSensor", "传感器覆盖（开）", 3),
    ("modelSensor", "传感器覆盖（关）", 1),
    ("layerHotspots", "热点区域（关）", 2),
    ("layerHotspots", "热点区域（开）", 2),
    ("layerRoutes", "计划航路（关）", 1),
    ("layerRoutes", "计划航路（开）", 1),
    ("layerLevelBanner", "区域级态势横幅（关）", 1),
    ("layerLevelBanner", "区域级态势横幅（开）", 1),
    ("filterRed", "红方过滤（关）", 2),
    ("filterRed", "红方过滤（开）", 2),
    ("layerOverview", "鸟瞰图（关）", 2),
    ("layerOverview", "鸟瞰图（开）", 3),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8848/?case=joint_full&tick=800")
    ap.add_argument("--out", default="/tmp/interactive.json")
    args = ap.parse_args()

    s = Session(args.url)
    report = {"url": args.url, "steps": [], "summary": {}}
    try:
        if not s.wait_ready():
            print("ERROR: 页面未就绪")
            return 2
        ir.settle(s.cdp, 90000)
        time.sleep(6)
        report["initial"] = s.state()
        print("=== 初始状态")
        print("  影像层 show=%s 建筑 %s 地形 %s(%s) 地名 %s" % (
            report["initial"].get("shownImagery"),
            json.dumps(report["initial"].get("buildings"), ensure_ascii=False),
            report["initial"].get("terrainOn"), report["initial"].get("terrainType"),
            (report["initial"].get("places") or {}).get("shown")))

        for cid, label, wait in TOGGLES:
            before = len(s.events)
            res = s.ev(click_js(cid))
            time.sleep(wait)
            st = s.state()
            errs = [e for e in s.events[before:]]
            step = {"id": cid, "label": label, "clicked": res,
                    "note": st.get("note", "")[:110],
                    "shownImagery": st.get("shownImagery"),
                    "buildings": st.get("buildings"),
                    "terrainOn": st.get("terrainOn"),
                    "terrainType": st.get("terrainType"),
                    "placesShown": (st.get("places") or {}).get("shown"),
                    "placesStats": (st.get("places") or {}).get("lastStats"),
                    "urbanLayers": st.get("urbanLayers"),
                    "wsf3dOverlay": st.get("wsf3dOverlay"),
                    "errors": errs[:4]}
            report["steps"].append(step)
            flag = "ERR" if errs else "ok "
            print("[%s] %-28s imagery=%s buildings=%s terrain=%s/%s places=%s %s" % (
                flag, label, st.get("shownImagery"),
                (st.get("buildings") or {}).get("has"),
                st.get("terrainOn"), st.get("terrainType"),
                (st.get("places") or {}).get("shown"),
                ("" if not errs else " | " + json.dumps(errs[:2], ensure_ascii=False)[:220])))
            if errs:
                for e in errs[:3]:
                    print("      -> %s" % json.dumps(e, ensure_ascii=False)[:200])

        net = [e for e in s.events if e.get("kind") == "network"]
        nf = {}
        for e in net:
            u = e.get("url", "").split("?")[0]
            nf[u] = nf.get(u, 0) + 1
        report["summary"] = {
            "total_events": len(s.events),
            "exceptions": len([e for e in s.events if e.get("kind") == "exception"]),
            "console_errors": len([e for e in s.events if e.get("kind") == "console.error"]),
            "network_404": len(net),
            "network_top": dict(sorted(nf.items(), key=lambda kv: -kv[1])[:8]),
        }
        print("\n=== 汇总")
        print(json.dumps(report["summary"], ensure_ascii=False, indent=1))
    finally:
        s.close()

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    print("report -> %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
