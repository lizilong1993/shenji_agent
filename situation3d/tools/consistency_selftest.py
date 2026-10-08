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
  await new Promise(function (r) { setTimeout(r, 10000); });
  function frames(n) {
    return new Promise(function (res) {
      var k = 0;
      function step() { if (++k >= n) return res(); requestAnimationFrame(step); }
      requestAnimationFrame(step);
    });
  }
  if (!window.SituationHost || !window.LayerRegistry) { return { fatal: "no host/registry" }; }
  var out = { rows: [], fatal: null };

  // 每个图层的“效果读数”。返回值必须能区分 true/false，
  // 带不出来就返 null（表示“这个图层没法用这种方式取证”，不当失败）。
  var SPECS = [
    { id: "places.labels",   what: "离线地名（模块自己缓存的 enabled）", read: function () {
        var s = window.Places ? String(Places.debugInfo()) : "";
        var m = s.match(/enabled\s+(true|false)/);
        return m ? (m[1] === "true") : null; } },
    { id: "wsf3d.overlay",   what: "建筑高度图层（场景里那一层的 show）", read: function () {
        var L = LayerRegistry.layersOf("wsf3d.overlay") || [];
        if (!L.length) return null;
        return L.every(function (x) { return !!x.show; }); } },
    { id: "tianditu.label",  what: "天地图注记（场景里那一层的 show）", read: function () {
        var L = LayerRegistry.layersOf("tianditu.label") || [];
        if (!L.length) return null;
        return L.every(function (x) { return !!x.show; }); } },
    { id: "imagery.offline", what: "离线底图（两级，全部 show 一致）", read: function () {
        var L = LayerRegistry.layersOf("imagery.offline") || [];
        if (!L.length) return null;
        return L.every(function (x) { return !!x.show; }); } },
  ];

  for (var i = 0; i < SPECS.length; i++) {
    var s = SPECS[i], row = { id: s.id, what: s.what };
    row.desired0 = LayerRegistry.runtime[s.id].desired;
    row.mod0 = s.read();
    LayerRegistry.setDesired(s.id, true);
    LayerRegistry.apply();
    await frames(8);
    row.desired1 = LayerRegistry.runtime[s.id].desired;
    row.mod1 = s.read();
    LayerRegistry.setDesired(s.id, false);
    LayerRegistry.apply();
    await frames(8);
    row.desired2 = LayerRegistry.runtime[s.id].desired;
    row.mod2 = s.read();
    // 复原成初始意图，避免影响后续条目
    LayerRegistry.setDesired(s.id, row.desired0);
    LayerRegistry.apply();
    await frames(4);
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
    budget = (QUEUE_MS + WAIT_MS) / 1000.0 + 60
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
    ap.add_argument("--case", default="taiwan_landing")
    args = ap.parse_args(argv)

    print("意图↔效果一致性自检  %s/?case=%s" % (args.base, args.case))
    print()
    tmpdir = tempfile.mkdtemp(prefix="consistency-")
    ev, page_errors = run_probe(args.base, args.case, tmpdir)
    if ev is None:
        ev, page_errors = run_probe(args.base, args.case, tmpdir)
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
        check("%s  desired 真的被置开" % rid, row.get("desired1") is True,
              "初值=%s" % row.get("desired0"))
        check("%s  desired 真的被置关" % rid, row.get("desired2") is False)
        m1, m2 = row.get("mod1"), row.get("mod2")
        if m1 is None or m2 is None:
            # 拿不到效果读数（例：图层未挂载）—— 这**正是**
            # 本自检要抓的东西，所以不能当成“无法取证”放过。
            check("%s  置开后能取到效果读数" % rid, False,
                  "读数为 None（图层未创建/模块未就绪）")
            continue
        check("%s  置开后效果真的发生" % rid, m1 is True,
              "模块状态=%s" % m1)
        check("%s  置关后效果真的消失" % rid, m2 is False,
              "模块状态=%s" % m2)

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
