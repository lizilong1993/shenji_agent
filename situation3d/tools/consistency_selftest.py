#!/usr/bin/env python3
"""意图↔效果一致性自检。

为什么需要它：这个项目里最危险的缺陷形态是**图层行在注册表里存在、
`desired` 也能置上，但场景里什么都不发生**。它不报错、`pageErrors` 为空、
案例照样加载 —— 只能靠“翻转意图、再读效果”来抓。

已抓到的一例（2026-10-08）：`wsf3d.overlay` 的图层创建写在 `setWsf3dOverlay()` 里，
而那个函数只有 URL 钩子会调 —— 面板上“建筑高度图层（WSF3D）”那个复选框**一直是空转的**。
修了之后就需要这条自检把它钉住。

用法：python3 tools/consistency_selftest.py [--base http://127.0.0.1:8848] [--case taiwan_landing]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSPECT = os.path.join(ROOT, "tools", "inspect_remote.py")

# `inspect_remote.py` 的 --timeout-ms（等瓦片队列排空）与 --wait-ms（排空后静置）
# 是**相加**的；外层超时必须比两者之和还宽。当初就是这里算错了才把
# `?terrain=1` 误报成“probe 无结果”。
QUEUE_MS = 60000
WAIT_MS = 18000

FAILURES = []


def check(label, ok, detail=""):
    print(("  ok   " if ok else "  FAIL ") + label + (("  " + detail) if detail else ""))
    if not ok:
        FAILURES.append(label + (("  " + detail) if detail else ""))


# 逐对翻转 `desired`，再读**场景/模块自己的状态**。
# 只看 `desired` 是不够的 —— 缺陷恰恰就在“desired 对了、效果没发生”。
PROBE = r"""
(async function () {
  var t0 = Date.now();
  while (Date.now() - t0 < 20000 && !window.SituationHost) {
    await new Promise(function (r) { setTimeout(r, 300); });
  }
  await new Promise(function (r) { setTimeout(r, 7000); });
  function frames(n) {
    return new Promise(function (res) {
      var k = 0;
      function step() { if (++k >= n) return res(); requestAnimationFrame(step); }
      requestAnimationFrame(step);
    });
  }
  if (!window.SituationHost || !window.LayerRegistry) { return { fatal: "no host/registry" }; }
  var out = { rows: [], fatal: null, stateLayers: (SituationHost.state.layers || null) };

  // 每个图层的“效果读数”。统一返回**数量**（布尔就用 0/1），
  // 因为断言统一是“先关再开、数量必须上升”。
  //
  // ⚠ **不要写绝对断言**（例如“置开后全部 show === true”）：
  // `state.filters`（红/蓝/海/陆/空）也会独立隐藏实体，绝对断言会**假失败**。
  // 这个代码库自己的注释就写着：自检里的假失败比没有自检更糟。
  function countShown(arr) {
    if (!arr) return 0;
    var list = Array.isArray(arr) ? arr : Object.keys(arr).map(function (k) { return arr[k]; });
    var n = 0;
    list.forEach(function (e) { if (e && e.show) n++; });
    return n;
  }
  function imageryShown(id) {
    var L = LayerRegistry.layersOf(id) || [];
    var n = 0;
    L.forEach(function (x) { if (x && x.show) n++; });
    return n;
  }
  // 目标：`state.targets[id]` 是**包装对象**（`{ def, entity, sensorEntity, trail, ... }`），
  // **不是实体** —— 所以 `countShown(ST.targets)` 恒为 0（包装对象没有 `.show`）。
  // 真正的落点是它里面的 `entity.show`，而 `entity.show` 由 `isVisible(def)` 算出
  // （图层开关 + 阵营/域过滤 + 是否被移除），因此数它就是在数“图层开关的真实效果”。
  function countTargetsShown() {
    var m = ST.targets || {};
    var n = 0;
    Object.keys(m).forEach(function (k) {
      var w = m[k];
      if (w && w.entity && w.entity.show) n++;
    });
    return n;
  }

  function modPlaces() {
    var s = window.Places ? String(Places.debugInfo()) : "";
    var m = s.match(/enabled\s+(true|false)/);
    // `load()` 未完成时 `debugInfo()` 返回 "not loaded" —— 那时**该图层确实没显示**，
    // 所以返 0 而不是 -1（-1 会被当成“无法取证”直接报失败）；
    // 置开后的轮询会等它真的加载完。
    return m ? (m[1] === "true" ? 1 : 0) : 0;
  }

  var ST = SituationHost.state;
  var SPECS = [
    // —— 已有的 4 项（上一轮已验证）——
    { id: "places.labels",   what: "离线地名（模块自己缓存的 enabled）", read: modPlaces },
    { id: "wsf3d.overlay",   what: "建筑高度图层（场景 show 计数）", read: function () { return imageryShown("wsf3d.overlay"); } },
    { id: "tianditu.label",  what: "天地图注记（场景 show 计数）", read: function () { return imageryShown("tianditu.label"); } },
    { id: "imagery.offline", what: "离线底图（两级 show 计数）", read: function () { return imageryShown("imagery.offline"); } },
    // —— 本轮新增：取数口已核实——
    { id: "tianditu.base",   what: "天地图底图（场景 show 计数）", read: function () { return imageryShown("tianditu.base"); } },
    // ⚠ **`layerTargets` 暂不纳入**：实测 `state.targets` 里的值**不是带 `.show` 的实体**
    // （`countShown(ST.targets)` 恒为 0），于是“关掉后=0”是 **0==0 恒真**、
    // “开上后”必然失败 —— 那是**假失败**，不是产品缺陷。
    // 宁可不盖，也不能放一条会嗧的断言进来。
    // 要补它，先找到目标图层真正的场景落点（大概率是 `refilter()` 控制的
    // 某个集合，而不是 `state.targets`），再把它加回来。
    { id: "layerTargets",    what: "可见目标（包装对象里 entity.show 的计数）",
      read: countTargetsShown },
    { id: "layerHotspots",   what: "热点区域（实体 show 计数）", read: function () { return countShown(ST.hotspotEntities); } },
    { id: "layerRoutes",     what: "计划航路（实体 show 计数）", read: function () { return countShown(ST.routeEntities); } },
    { id: "layerOverview",   what: "鸟瞰图（第二个 view 的可见性）", read: function () {
        return (window.Overview && typeof Overview.isVisible === "function") ? (Overview.isVisible() ? 1 : 0) : -1; } },
    { id: "weather.model",   what: "气象模型（场景雾开关）", read: function () {
        var v = SituationHost.viewer(); return v ? (v.scene.fog.enabled ? 1 : 0) : -1; } },
  ];

  for (var i = 0; i < SPECS.length; i++) {
    var s = SPECS[i], row = { id: s.id, what: s.what };
    row.desired0 = LayerRegistry.runtime[s.id].desired;
    // 先**关**再**开**：这样不受初始值影响（有些图层默认就是开的）。
    LayerRegistry.setDesired(s.id, false);
    LayerRegistry.apply();
    await frames(4);
    row.nOff = s.read();
    LayerRegistry.setDesired(s.id, true);
    LayerRegistry.apply();
    // 开上后**轮询到出现为止**：有些效果是异步的
    // （例：`places.labels` 走 `Places.load()`，未完成时 `debugInfo()` 返回 "not loaded"）。
    // 固定帧数会把“还没加载完”误报成“效果没出现”。
    // 但**不能无限等** —— eval 有自己的超时，所以设上限。
    row.nOn = s.read();
    for (var _try = 0; _try < 40 && (row.nOn === 0 || row.nOn === -1); _try++) {
      await frames(3);
      row.nOn = s.read();
    }
    row.desired1 = LayerRegistry.runtime[s.id].desired;
    // 复原初始意图
    LayerRegistry.setDesired(s.id, row.desired0);
    LayerRegistry.apply();
    await frames(3);
    out.rows.push(row);
  }
  return out;
})()
"""


def run_probe(base, case, tmpdir):
    out = os.path.join(tmpdir, "consistency.json")
    url = "%s/?case=%s" % (base, case)
    cmd = [sys.executable, INSPECT, "--url", url, "--out", out,
           "--timeout-ms", str(QUEUE_MS), "--wait-ms", str(WAIT_MS),
           "--eval-async", "--eval", PROBE]
    # ⚠ 这里必须把 **eval 自己的耗时**也算进去。
    # `--timeout-ms`（等瓦片队列排空）与 `--wait-ms`（排空后静置）是**相加**的，
    # 而 eval 是在它们**之后**才跑。扩到 10 个图层后 eval 约 90 s
    # （10 层 × 26 帧，headless 下 ~5fps），而原来只留了 60 s 余量 ——
    # 结果是“两次都拿不到结果”。这与本仓库 `deeplink_selftest.py`
    # 里记载的那个坑是**同一个**（当初 `?terrain=1` 就是这么误报的）。
    budget = (QUEUE_MS + WAIT_MS) / 1000.0 + 180
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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8848")
    # 默认**两个案例都跑**。为什么：两个案例的图层集合与规模不同
    # （台海 31 目标/11 热点；中印 10 目标/7 热点、且**没有城区 AOI**），
    # 只跑一个案例会把“只在某个案例下才坏”的问题漏掉。
    # 而且两个案例的计数**应该不同**（实测：31 vs 10、11 vs 7）——
    # 这也是“读的是真实场景数据、不是常量”的侧面证据。
    ap.add_argument("--case", action="append", default=None,
                    help="可重复；省略时默认跑 taiwan_landing 与 india_china_west")
    args = ap.parse_args(argv)

    cases = args.case or ["taiwan_landing", "india_china_west"]
    tmpdir = tempfile.mkdtemp(prefix="consistency-")
    for ci, case in enumerate(cases):
        print("意图↔效果一致性自检  %s/?case=%s" % (args.base, case))
        print()
        _one(args.base, case, tmpdir, ci)
        print()
    if FAILURES:
        print("FAILED (%d):" % len(FAILURES))
        for f in FAILURES:
            print("  - " + f)
        return 1
    print("ALL CHECKS PASSED")
    return 0


def _one(base, case, tmpdir, ci):
    ev, page_errors = run_probe(base, case, tmpdir)
    if ev is None:
        ev, page_errors = run_probe(base, case, tmpdir)
    if ev is None:
        check("页面能跑起来", False, "两次都拿不到结果（服务未启动？）")
        print()
        print("FAILED (%d):" % len(FAILURES))
        for f in FAILURES:
            print("  - " + f)
        return 1

    if ev.get("fatal"):
        check("页面能跑起来", False, str(ev.get("fatal")))
        return 1
    check("案例正常加载（无启动异常）", not page_errors,
          json.dumps(page_errors[:2], ensure_ascii=False))

    rows = ev.get("rows") or []
    if not rows:
        check("取到至少一行取证", False, "rows 为空")
    for row in rows:
        rid = row.get("id")
        # 这条只证“置 true 确实写进了注册表”；真正的断言在下面的数量比较。
        check("%s  desired 真的被置开" % rid, row.get("desired1") is True,
              "初值=%s" % row.get("desired0"))
        n_off, n_on = row.get("nOff"), row.get("nOn")
        if n_off is None or n_on is None or n_off < 0 or n_on < 0:
            check("%s  能取到效果读数" % rid, False,
                  "读数为 None/不可用（图层未创建、模块未就绪）")
            continue
        check("%s  关掉后效果消失" % rid, n_off == 0, "可见数=%s" % n_off)
        check("%s  开上后效果出现" % rid, n_on > n_off,
              "可见数 %s -> %s" % (n_off, n_on))
if __name__ == "__main__":
    raise SystemExit(main())
