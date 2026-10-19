#!/usr/bin/env python3
"""Deep-link self-test for situation3d.

Why this file exists
--------------------
Every URL parameter below (`?imagery=`, `?terrain=`, `?overlay=`, `?sensors=`,
`?weather=`, `?buildings=`, `?hotspot=`, `?tick=`, `?cam=`) is read in
`applyCaseIndependentHooks()` / `applyCaseHooks()`, which run **inside the boot
promise chain**.

A throw in there is caught by the boot `.catch`: the page gets
`#statusText = "启动失败: …"`, `state.data` stays `null`, the camera stays at the
Cesium default — and `pageErrors` is **empty**. So the failure is invisible to
every probe that only looks at `pageErrors`, and invisible to any test that only
loads the bare `?case=` URL.

That is not hypothetical. Moving the layer checkboxes into the registry-generated
panel deleted the `id="model*"` nodes while these hooks still wrote
`document.getElementById('modelImagery').checked = …`; six deep links then aborted
startup with no case loaded and a clean `pageErrors`. It shipped because the whole
acceptance run used URLs *without* those parameters.

So the assertion here is deliberately **"did the case actually load"**, not
"were there page errors".

Usage
-----
    python3 tools/deeplink_selftest.py [--base http://127.0.0.1:8848]

Requires the local server (`sh tools/restart_server.sh 8848`).
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
INSPECT = os.path.join(HERE, "inspect_remote.py")

FAILURES = []


# URL 参数 → 它**必须**在注册表里点亮的图层 id。
#
# 为什么需要这张表：本会话修过两处同型 bug —— `?overview=1` 与 `?places=1` 都在
# "绕过注册表直接改效果"，于是 **URL 上带着参数、参数实际不生效**，而且不报错。
# 案例照样加载（所以只断言"案例加载了"是**空过**，抓不到），`pageErrors` 也照样为空。
# 断言"注册表意图真的被翻转"才能抓到。
# 这一类到 2026-10-08 为止共 4 例（均已修）：
#   `?overview=1` · `?places=1` · 鸟瞰图的×按钮 · `?sensors=1`
#
# 修完后又把两层入口都扫了一遍，**结果干净**，不必重扫：
#   · 全部 URL 钩子：imagery / terrain / overlay / sensors / weather / buildings /
#     compass / scalebar / viewpoint / components 都走 `setSwitch(...)` 加显式 setter；
#     cam / follow / hotspot / panel / tick / cache 不是图层开关。
#   · 全部“效果入口”的调用点：`setOnlineTilesetVisible` 与
#     `applyBuildingsVisibility` 各自**只有一处调用**（都是 primitive sink），
#     `Places.setEnabled` 只在 label sink、`Plotting.setLayerVisible` 只在 plot sink，
#     `Overview.show/hide` 只在鸟瞰图那行的 handler（另有两处防御性兼底，
#     读的是 `state.layers.overview`，与注册表同步）。
#
# 若将来又出现同类，优先看两个地方：新加的 URL 钩子、
# 以及新加的“直接调 setter”代码。守卫本身只能盖住表里列了的参数。

LAYER_PARAM_EXPECT = {
    "overview=1": "layerOverview",
    "places=1": "places.labels",
    "buildings=1": "buildings.blocks",
    # 两个实测过的同型实例：这两行都**没有 `checkbox` 字段**，
    # 而钩子原先调的是 `setSwitch('modelSensor')` —— `setSwitch` 按 checkbox id 反查，
    # 反查为 null 就静默空转。实测：`?sensors=1` 下 desired 图层数 20 -> 20 没变。
    "sensors=1": "modelSensor",
    "terrain=1": "terrain.model",
}


def check(label, ok, detail=""):
    print(("  ok   " if ok else "  FAIL ") + label + (("  " + detail) if detail else ""))
    if not ok:
        FAILURES.append(label + (("  " + detail) if detail else ""))


# Reads the status line too: that is where the swallowed boot error shows up.
PROBE = r"""
(async function () {
  var t0 = Date.now();
  while (Date.now() - t0 < 25000 && !window.SituationHost) {
    await new Promise(function (r) { setTimeout(r, 300); });
  }
  await new Promise(function (r) { setTimeout(r, 9000); });
  var s = window.SituationHost ? SituationHost.state : null;
  var st = document.getElementById('statusText');
  // 鸟瞰图黄框：**只在它真的开着时才采**。
  //
  // 不能只看 `window.Overview` 存不存在 —— 模块一直被加载着，但鸟瞰图关着时
  // `vertices` 就是 0，断言会在不相关的 URL 上误报。所以以**注册表里那一行的意图**
  // 为准（`layerOverview`），这比猜一个 `isVisible()` 是否被导出可靠。
  var fp = null;
  var ov = window.LayerRegistry && LayerRegistry.runtime
           && LayerRegistry.runtime['layerOverview'];
  if (ov && ov.desired && window.Overview
      && typeof Overview.footprintDebug === 'function') {
    try {
      var a = Overview.footprintDebug();
      // 用 setView 平移主视图（保留高度），**不要用 camera.moveRight/moveUp**：
      // 实测那两个在 pitch −75° 下产生的**地面足迹变化为零**
      // （`updates 11 -> 11`、`keyChanged false`），断言会假失败。
      // 这里连同朝向一起重置是可以接受的 —— 我们只关心足迹变没变。
      var v = SituationHost.viewer();
      var carto = v.camera.positionCartographic;
      var lon = Cesium.Math.toDegrees(carto.longitude) + 0.6;
      var lat = Cesium.Math.toDegrees(carto.latitude) + 0.4;
      v.camera.setView({ destination: Cesium.Cartesian3.fromDegrees(lon, lat, carto.height) });
      for (var k = 0; k < 26; k++) {
        await new Promise(function (r) { requestAnimationFrame(r); });
      }
      var b = Overview.footprintDebug();
      fp = {
        before: a.updates, after: b.updates,
        // keyHead 是黄框的**几何指纹**：只断言"次数涨了"会被
        // "重画了、但画的还是同一个框"骗过 —— 那正是这个 bug 原来的症状。
        keyChanged: a.keyHead !== b.keyHead,
        vertices: b.vertices, shown: b.shown,
        // 鸟瞰图自身机位**应当不动**（固定全图设计），所以用 !camMoved 断言
        camMoved: Math.abs((b.cameraLon || 0) - (a.cameraLon || 0)) > 1e-9,
      };
    } catch (e) { fp = { error: String(e.message || e) }; }
  }
  // 注册表里**当前意图为开**的图层 id。用来抓一类 bug：
  // URL 钩子绕过注册表直接改"效果"（`setOverview` / `Places.setEnabled`），
  // 于是参数看似生效、随后被 `apply()` 按注册表改回去 —— **静默失效**。
  // `?overview=1` 与 `?places=1` 都是这么坏的（实测），任何一处再犯都能被下面抓到。
  var desiredTrue = (window.LayerRegistry && LayerRegistry.runtime)
      ? Object.keys(LayerRegistry.runtime).filter(function (k) {
          return LayerRegistry.runtime[k].desired; })
      : null;
  return {
    footprint: fp,
    desiredTrue: desiredTrue,
    hasHost: !!window.SituationHost,
    hasData: !!(s && s.data),
    caseId: s ? s.caseId : null,
    targets: s && s.data ? s.data.targets.length : null,
    status: st ? st.textContent : null,
    layers: (window.LayerRegistry && LayerRegistry.LAYERS.length) || null,
  };
})()
"""


# `inspect_remote.py` 的两个时间参数是**相加**的，不是一个：
#   --timeout-ms  等"瓦片队列排空"的上限（默认 90000）
#   --wait-ms     排空**之后**再静置多久
# 早先这里的外层超时写成 `wait_ms + 60`，比 `90000 + wait_ms` 还短 ——
# 于是像 `?terrain=1` 这种瓦片队列很长（要拉 DEM）的深链，
# 会在 settle 还没结束时就被外层杀掉，报成"probe 无结果"。
# 那是**测试自己的超时算错了**，与产品无关（单独跑、不设外层超时时它完全正常）。
QUEUE_MS = 60000   # 传给 --timeout-ms：等瓦片队列排空的上限


def run_probe(base, query, wait_ms, tmpdir, idx):
    """Load one URL and return (probe_result, page_errors)."""
    out = os.path.join(tmpdir, "dl%d.json" % idx)
    url = base + "/?" + query
    cmd = [sys.executable, INSPECT, "--url", url, "--out", out,
           "--timeout-ms", str(QUEUE_MS),
           "--wait-ms", str(wait_ms), "--eval-async", "--eval", PROBE]
    # 外层超时 = 队列上限 + 静置 + 浏览器启动/收尾 + **异步 eval 自身的时长**。
    # 原先只留 45 s，`?overview=1` 的探针要等 26 个 rAF（SwiftShader 下约 10 s）
    # 再加建第二个 Cesium 视图，实测会顶着 117 s 被外层杀掉；eval 侧 CDP 超时已
    # 放宽到 150 s，外层必须同步放宽，否则修了里面还是被外面掐。
    budget = (QUEUE_MS + wait_ms) / 1000.0 + 120
    try:
        subprocess.run(cmd, cwd=ROOT, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=budget)
    except subprocess.TimeoutExpired:
        return None, ["probe timed out"]
    try:
        with open(out, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError) as exc:
        return None, ["probe output unreadable: %s" % exc]
    return doc.get("eval"), doc.get("pageErrors") or []


# (query, expected case id). Every parameter below is one that used to abort boot.
CASES = [
    # ⚠ **最重的一条放最前**。`?terrain=1` 要真正加载 DEM，是全部深链里落定最慢的
    # 一条；而每跑一条就多留一个 headless 浏览器进程，越往后机器越挤 ——
    # 实测它排在中间时 26s / 41.6s / 60s 全都会超时，而单独跑 30s 就正常。
    # 顺序即预算：先跑它，机器还是干净的。
    ("case=taiwan_landing&terrain=1", "taiwan_landing"),
    ("case=taiwan_landing", "taiwan_landing"),
    ("case=taiwan_landing&imagery=0", "taiwan_landing"),

    ("case=taiwan_landing&overlay=1", "taiwan_landing"),
    ("case=taiwan_landing&sensors=1", "taiwan_landing"),
    ("case=taiwan_landing&weather=0", "taiwan_landing"),
    ("case=taiwan_landing&weather=1", "taiwan_landing"),
    ("case=taiwan_landing&buildings=1", "taiwan_landing"),
    ("case=taiwan_landing&buildings=1&hotspot=0", "taiwan_landing"),
    ("case=taiwan_landing&tick=600", "taiwan_landing"),
    ("case=taiwan_landing&cam=119.6,24.4,300000,0,-60", "taiwan_landing"),
    ("case=india_china_west", "india_china_west"),
    # 鸟瞰图：第二个 Cesium 2D 视图，本套原先**没有覆盖**它。
    #
    # 这里只保证"这个 URL 不炸启动"（本套的既有职责）。**黄框是否真的跟着主视图刷新
    # 已经外部验过**（`Overview.footprintDebug()`：updates 9→12、keyHead 真的变了
    # 而不只是次数涨、vertices=4、鸟瞰图自身机位保持不变），但**尚未变成断言** ——
    # 这条 URL 现在至少保证它不会回归到"打不开"。
    #
    # 要把它做成断言：在 EVAL 里加一段 `if (window.Overview && Overview.footprintDebug)
    # { … 移动相机前后各取一次 … }`，再在 main 里断言 dUpdates > 0 且 keyHead 变了。
    # 之所以值得做：overview.js 自己的注释就写着"否则'鸟瞰图不变'这种 bug 只能靠肉眼看"。
    ("case=taiwan_landing&overview=1", "taiwan_landing"),
    # 本地地名：与 overview 同型的 bug（钩子绕过注册表 -> 参数静默失效）。
    # 这条用例的价值全在 LAYER_PARAM_EXPECT 那条断言上：只断言"案例加载了"会**空过**
    # （案例无论参数是否生效都会加载），这正是 `?overview=1` 上一轮骗过我的方式。
    ("case=taiwan_landing&places=1", "taiwan_landing"),
    # 传感器：与 places/overview 同型。无这条用例的话，
    # LAYER_PARAM_EXPECT 里那一项对它就是空转。
    ("case=taiwan_landing&sensors=1", "taiwan_landing"),
    ("case=nope_not_here", None),   # unknown id must fall back to the default
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8848")
    # 26s，不是 18s。`?terrain=1` 会真正加载 DEM 高程，页面**落定得比其它深链慢**：
    # 18s 时实测它被判成"probe 无结果"，而单独给它 30s 就完全正常
    # （`hasData:true caseId:taiwan_landing targets:31 terrainOn:true pageErrors:[]`）。
    # 自检里的假失败比没有自检更糟 —— 会让人去修没坏的东西，或者干脆不再看它。
    ap.add_argument("--wait-ms", type=int, default=12000)
    args = ap.parse_args(argv)

    if not os.path.isfile(INSPECT):
        print("FAIL: %s not found" % INSPECT)
        return 2

    tmpdir = tempfile.mkdtemp(prefix="deeplink_selftest_")
    print("深度链接自检：%d 条 URL（3 个浏览器并行）" % len(CASES))
    print("判据是「案例真的加载了」，不是「pageErrors 为空」——")
    print("启动链里的异常会被 boot 的 catch 吞掉，pageErrors 照样是空的。")
    print()

    # 串行跑 16 条 URL 在本环境要 15 min 以上（每条都要起一个 headless 浏览器、
    # 等瓦片队列排空、再静置 12 s）。3 路并行把总时长压到 1/3，且每条仍独立
    # 计时、独立重试；输出顺序按 CASES 不变。
    from concurrent.futures import ThreadPoolExecutor

    def run_one(i, item):
        query = item[0]
        budget = item[2] if len(item) > 2 else args.wait_ms
        ev, pe = run_probe(args.base, query, budget, tmpdir, i)
        retried = False
        if ev is None:
            # **超时重试一次，预算放宽 1.6 倍。**（见下）
            ev, pe = run_probe(args.base, query, int(budget * 1.6),
                               tmpdir, i + 100)
            retried = ev is not None
        return ev, pe, retried

    with ThreadPoolExecutor(max_workers=3) as ex:
        results = list(ex.map(lambda t: run_one(*t), enumerate(CASES)))

    for i, item in enumerate(CASES):
        query, want = item[0], item[1]
        ev, page_errors, retried = results[i]
        if retried:
            print("       （首次超时，放宽预算后通过：%s）" % query)
        if ev is None:
            check(query, False, "probe 无结果（超时或页面未起来，已重试一次）")
            continue
        loaded = ev.get("caseId")
        ok = (ev.get("hasData") is True
              and loaded is not None
              and "启动失败" not in (ev.get("status") or ""))
        if want is not None:
            ok = ok and loaded == want
        check(query, ok, "caseId=%s targets=%s status=%s"
              % (loaded, ev.get("targets"), (ev.get("status") or "")[:40]))
        check("  └ pageErrors 为空", not page_errors,
              json.dumps(page_errors[:2], ensure_ascii=False))

        # URL 参数是否真的翻转了注册表意图（见 LAYER_PARAM_EXPECT 的说明）
        dt = ev.get("desiredTrue")
        if dt is not None:
            for param, layer_id in LAYER_PARAM_EXPECT.items():
                if param in query:
                    check("  └ ?%s 真的点亮了 %s" % (param, layer_id),
                          layer_id in dt, "desired=True 的图层数 %d" % len(dt))

        # 鸟瞰图黄框（PROBE 里条件采集：只有这一行开着时才非 None）
        fp = ev.get("footprint")
        if fp is not None:
            if fp.get("error"):
                check("  └ 鸟瞰图黄框取件失败", False, str(fp.get("error")))
            else:
                check("  └ 鸟瞰图黄框：主视图移动后被重画",
                      (fp.get("after") or 0) > (fp.get("before") or 0),
                      "updates %s -> %s" % (fp.get("before"), fp.get("after")))
                check("  └ 鸟瞰图黄框：几何真的变了（不只是次数涨）",
                      fp.get("keyChanged") is True)
                check("  └ 鸟瞰图黄框：有顶点（真的画出来了）",
                      (fp.get("vertices") or 0) > 0, "vertices=%s" % fp.get("vertices"))
                check("  └ 鸟瞰图：自身机位不动（固定全图设计）",
                      fp.get("camMoved") is False)

    ev, _ = run_probe(args.base, "case=taiwan_landing", args.wait_ms, tmpdir, 99)
    if ev:
        check("图层注册表非空（面板迁移后仍有图层）",
              (ev.get("layers") or 0) >= 20, "layers=%s" % ev.get("layers"))

    print()
    if FAILURES:
        print("FAILED (%d):" % len(FAILURES))
        for f in FAILURES:
            print("  - " + f)
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())