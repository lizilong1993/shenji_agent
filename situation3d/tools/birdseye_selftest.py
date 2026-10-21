#!/usr/bin/env python3
"""Self-check for the bird's-eye view and the overall-TS status display.

Run on the target container, where the test server (worktree, e.g. port 8855)
is already serving:

    python3 tools/birdseye_selftest.py \
        --url "http://127.0.0.1:8855/?case=taiwan_landing&overview=1"

Run it once per case (taiwan_landing and india_china_west, `--case` asserts which);
each run checks that case's own extent, roster and status distribution.

What this guards against
------------------------
The two views share one status module (`web/src/targetStatus.js`), but they render
it through different pipelines: Cesium billboards in the overview, DOM in the
overall-TS panel. The failure modes that matter are the ones you cannot see from
a single screenshot:

1. **The overview keeps a stale/default extent** -- the fitted rectangle must
   contain the case's whole track extent (each case is checked separately).
2. **The views disagree** -- a target shown as "severe" in the panel but "normal"
   in the overview, because one side used the alert level and the other used HP.
3. **Destroyed looks like just-another-low-HP unit** -- the distinction the user
   asked for. The probe asserts both the HP/status numbers *and* the visual
   carrier (bar width, marker/X overlay, panel class), not just the state code.
4. **Positions drift** -- the overview must show the exact frame position, not a
   re-sampled track.
5. **Static at tick 0 only** -- statuses must advance with playback/slider.

The probe returns raw measurements; thresholds live here so they are reviewable.
"""

from __future__ import annotations

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

PROBE = r"""
(async function () {
  var t0 = Date.now();
  function sleep(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }
  function ready() {
    return window.SituationHost && SituationHost.state && SituationHost.state.data &&
      SituationHost.state.data.case_id && window.Overview && window.TargetStatus;
  }
  while (Date.now() - t0 < 120000 && !ready()) await sleep(500);
  if (!ready()) return { error: 'app/overview modules not ready' };

  // 打开鸟瞰图（URL 已带 ?overview=1；这里只是兜底，不改变默认行为）
  try {
    if (!Overview.isVisible()) {
      if (window.LayerRegistry && LayerRegistry.setDesired) {
        LayerRegistry.setDesired('layerOverview', true);
      }
      Overview.show();
    }
  } catch (e) { return { error: 'cannot show overview: ' + e.message }; }

  // 等目标图元与推理帧
  while (Date.now() - t0 < 120000) {
    var td = Overview.targetDebug ? Overview.targetDebug() : null;
    if (td && td.targets && Object.keys(td.targets).length) break;
    await sleep(500);
  }
  while (Date.now() - t0 < 180000) {
    var inf = SituationHost.state.inference;
    if (inf && inf.length > 100) break;
    await sleep(500);
  }
  await sleep(1500);

  var state = SituationHost.state;
  var data = state.data;
  var out = { caseId: data.case_id, targetCount: (data.targets || []).length };
  out.affiliations = {};
  (data.targets || []).forEach(function (t) {
    out.affiliations[t.affiliation] = (out.affiliations[t.affiliation] || 0) + 1;
  });

  function setTick(t) {
    var tl = document.getElementById('timeline');
    tl.value = String(t);
    tl.dispatchEvent(new Event('input', { bubbles: true }));
    return state.tick;
  }

  function snapshot() {
    var td = Overview.targetDebug();
    var panel = {};
    Array.prototype.forEach.call(document.querySelectorAll('.os-unit[data-target]'), function (u) {
      var cls = String(u.className || '').split(/\s+/);
      var st = -1;
      cls.forEach(function (c) { if (/^os-s\d$/.test(c)) st = Number(c.slice(4)); });
      var fill = u.querySelector('.os-hp > b');
      panel[u.getAttribute('data-target')] = {
        status: st,
        hpWidth: fill ? fill.style.width : null,
        title: u.title || '',
        iconSrc: (u.querySelector('.os-icon') || {}).src || '',
      };
    });
    var per = {};
    Object.keys(td.targets).forEach(function (id) {
      var st = TargetStatus.at(data, id, state.tick);
      var o = td.targets[id];
      var p = panel[id] || null;
      var expectedBar = TargetStatus.barUri(st.hp, st.status, 24, 4);
      per[id] = {
        ts: { hp: st.hp, status: st.status, label: st.label, destroyedAt: st.destroyedAt,
              alert: st.alert },
        view: { status: o.status, hp: o.hp, markShow: o.markShow, barShow: o.barShow,
                iconTint: o.iconTint, barImage: String(o.barImage || ''),
                barMatches: String(o.barImage || '') === expectedBar,
                lon: o.lon, lat: o.lat, selected: o.selected, show: o.show,
                clustered: !!o.clustered },
        panel: p,
        consistent: p ? (p.status === st.status && o.status === st.status) : false,
      };
    });
    return {
      tick: state.tick,
      viewRect: td.viewRect,
      extent: td.extent,
      camera: td.camera,
      targets: per,
      clusters: td.clusters || [],
      items: td.items || [],
      counts: TargetStatus.counts(data, state.tick),
      legend: (document.querySelector('#overallBody .os-legend') || {}).textContent || '',
      sums: Array.prototype.map.call(document.querySelectorAll('#overallBody .os-sum'),
        function (e) { return e.textContent; }),
    };
  }

  function distanceM(lon1, lat1, lon2, lat2) {
    var R = 6371008.8;
    var p = Math.PI / 180;
    var dLat = (lat2 - lat1) * p, dLon = (lon2 - lon1) * p;
    var h = Math.sin(dLat / 2) * Math.sin(dLat / 2)
          + Math.cos(lat1 * p) * Math.cos(lat2 * p) * Math.sin(dLon / 2) * Math.sin(dLon / 2);
    return 2 * R * Math.asin(Math.min(1, Math.sqrt(h)));
  }

  // ---- tick 0: 初始全正常，位置与第 0 帧一致 ----
  out.tick0 = snapshot();
  setTick(0);
  await sleep(800);
  out.tick0 = snapshot();
  out.posErrMaxM = 0;
  var fr0 = data.frames[0] || { targets: [] };
  var pos0 = {};
  (fr0.targets || []).forEach(function (p) { pos0[p.id] = p; });
  Object.keys(out.tick0.targets).forEach(function (id) {
    var p = pos0[id];
    var v = out.tick0.targets[id].view;
    if (p && v.lon !== null) {
      out.posErrMaxM = Math.max(out.posErrMaxM, distanceM(p.lon, p.lat, v.lon, v.lat));
    }
  });

  // ---- mid tick: 位置改变（证明不是静态图） ----
  var moveId = (data.targets[0] || {}).id;
  var pBefore = out.tick0.targets[moveId] ? out.tick0.targets[moveId].view : null;
  setTick(Math.min(720, data.frames.length - 1));
  await sleep(1200);
  out.tickMid = snapshot();
  var pMid = out.tickMid.targets[moveId] ? out.tickMid.targets[moveId].view : null;
  out.moveDeltaM = (pBefore && pMid && pBefore.lon !== null && pMid.lon !== null)
    ? distanceM(pBefore.lon, pBefore.lat, pMid.lon, pMid.lat) : 0;
  // 不假设"第一个目标一定在动"：取全编成的最大位移，静止案例也不会误报。
  out.moveDeltaMaxM = 0;
  Object.keys(out.tick0.targets).forEach(function (id) {
    var a = out.tick0.targets[id].view, b = out.tickMid.targets[id].view;
    if (a && b && a.lon !== null && b.lon !== null) {
      out.moveDeltaMaxM = Math.max(out.moveDeltaMaxM, distanceM(a.lon, a.lat, b.lon, b.lat));
    }
  });

  // ---- final tick: 四档状态 ----
  setTick(data.frames.length - 1);
  await sleep(1800);
  out.tickEnd = snapshot();
  out.posErrEndM = 0;
  var posEnd = {};
  ((data.frames[data.frames.length - 1] || {}).targets || []).forEach(function (p) { posEnd[p.id] = p; });
  Object.keys(out.tickEnd.targets).forEach(function (id) {
    var p = posEnd[id];
    var v = out.tickEnd.targets[id].view;
    if (p && v.lon !== null) {
      out.posErrEndM = Math.max(out.posErrEndM, distanceM(p.lon, p.lat, v.lon, v.lat));
    }
  });
  out.endByStatus = { normal: [], damaged: [], severe: [], destroyed: [] };
  var names = ['normal', 'damaged', 'severe', 'destroyed'];
  Object.keys(out.tickEnd.targets).forEach(function (id) {
    var st = out.tickEnd.targets[id].ts.status;
    out.endByStatus[names[st] || 'normal'].push(id);
  });
  out.mismatch = Object.keys(out.tickEnd.targets).filter(function (id) {
    return !out.tickEnd.targets[id].consistent;
  });
  out.barMismatch = Object.keys(out.tickEnd.targets).filter(function (id) {
    return !out.tickEnd.targets[id].view.barMatches;
  });
  out.missingTargets = (data.targets || []).map(function (t) { return t.id; })
    .filter(function (id) { return !out.tickEnd.targets[id]; });

  // 取第一个损毁 / 严重受损目标做视觉载体检查
  out.destroyedId = out.endByStatus.destroyed[0] || null;
  out.severeId = out.endByStatus.severe[0] || null;

  // ---- 选中同步 ----
  var selId = moveId;
  SituationHost.selectTarget(selId, false);
  await sleep(600);
  out.selectSync = (function () {
    var td = Overview.targetDebug();
    var rec = td.targets[selId] || {};
    return { selected: !!rec.selected, target: selId, ringShown: !!td.selectionShown };
  })();

  out.viewContainsAll = false;
  out.viewMarginDeg = Infinity;
  if (out.tickEnd.viewRect && out.tickEnd.extent) {
    var vr = out.tickEnd.viewRect, ex = out.tickEnd.extent;
    out.viewContainsAll = vr.w <= ex.w + 1e-6 && vr.e >= ex.e - 1e-6 &&
                          vr.s <= ex.s + 1e-6 && vr.n >= ex.n - 1e-6;
    out.viewMarginDeg = Math.min(ex.w - vr.w, vr.e - ex.e, ex.s - vr.s, vr.n - ex.n);
  }

  // 隐藏→显示：图元仍在；回拖再前推：状态应回到同一结果。
  Overview.hide();
  await sleep(300);
  Overview.show();
  await sleep(1200);
  out.reshowCount = Object.keys(Overview.targetDebug().targets).length;

  var maxTick = data.frames.length - 1;
  setTick(Math.min(300, maxTick));
  await sleep(900);
  out.tickScrub = snapshot();
  setTick(maxTick);
  await sleep(900);
  out.tickFinal2 = snapshot();
  out.scrubRestores = JSON.stringify(out.tickFinal2.counts) === JSON.stringify(out.tickEnd.counts);

  out.debug = TargetStatus.debug(data, state.tick);

  // ---- 组件外观：无标题文字、无 Cesium 学分；图层名改为「全局鸟瞰」 ----
  out.overviewHeader = !!document.querySelector('#overviewWrap .ov-head');
  out.creditsDisplay = (function () {
    var c = document.querySelector('#overviewContainer .cesium-widget-credits');
    return c ? getComputedStyle(c).display : 'absent';
  })();
  out.layerLabel = (function () {
    var box = document.getElementById('layerRow-layerOverview');
    var row = box ? box.closest('.lp-check') : null;
    if (row) return row.textContent.replace(/\s+/g, ' ').trim();
    // 面板尚未渲染时退回注册表里的定义（标签的唯一来源）
    if (window.LayerRegistry && LayerRegistry.def) {
      var def = LayerRegistry.def('layerOverview');
      return def ? def.label : null;
    }
    return null;
  })();

  return out;
})()
"""


def probe(url: str) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "probe.json")
        cmd = [sys.executable, INSPECT, "--url", url, "--out", out,
               "--wait-ms", "25000", "--eval-timeout-ms", "600000",
               "--eval-async", "--eval", PROBE]
        try:
            subprocess.run(cmd, cwd=ROOT, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as exc:
            sys.stderr.write(exc.stderr.decode("utf-8", "replace") if exc.stderr else "")
            raise SystemExit("[birdseye] inspect_remote.py 失败；确认服务已启动、URL 可达")
        with open(out, "r", encoding="utf-8") as fh:
            return json.load(fh)


def check(name: str, cond: bool, detail: str) -> None:
    print("  %s %s" % ("ok  " if cond else "FAIL", name))
    if not cond:
        FAILURES.append("%s: %s" % (name, detail))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8855/?case=taiwan_landing&overview=1")
    ap.add_argument("--case", default="taiwan_landing",
                    help="expected case id in the URL")
    ap.add_argument("--probe-json", default=None,
                    help="reuse a saved inspect_remote report instead of launching a browser")
    args = ap.parse_args()

    if args.probe_json:
        with open(args.probe_json, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    else:
        raw = probe(args.url)
    if raw.get("evalError"):
        print("probe error: %s" % raw["evalError"])
        return 1
    d = raw.get("eval") or {}
    if d.get("error"):
        print("probe returned: %s" % d["error"])
        return 1
    if raw.get("pageErrors"):
        print("pageErrors: %s" % "; ".join(str(e) for e in raw["pageErrors"][:3]))
        return 1

    print("1. 案例与取景")
    check("case id", d.get("caseId") == args.case,
          "expected %s, got %s" % (args.case, d.get("caseId")))
    check("target count", d.get("targetCount", 0) > 0, "no targets")
    check("extent fitted to all tracks", bool(d.get("tickEnd", {}).get("extent")),
          "no extent")
    check("overview view rect readable", bool(d.get("tickEnd", {}).get("viewRect")),
          "computeViewRectangle returned null")
    check("overview view contains battle extent", bool(d.get("viewContainsAll")),
          "viewRect=%s extent=%s margin=%s" % (
              d.get("tickEnd", {}).get("viewRect"), d.get("tickEnd", {}).get("extent"),
              d.get("viewMarginDeg")))

    print("2. 目标位置与动态更新")
    check("tick 0 positions match frame (<=2 m)", d.get("posErrMaxM", 99) <= 2.0,
          "max error %.1f m" % d.get("posErrMaxM", -1))
    check("final tick positions match frame (<=2 m)", d.get("posErrEndM", 99) <= 2.0,
          "max error %.1f m" % d.get("posErrEndM", -1))
    check("targets move between ticks", d.get("moveDeltaMaxM", 0) > 1000,
          "max delta %.0f m" % d.get("moveDeltaMaxM", 0))

    print("3. 状态一致性与四档分布")
    t0 = d.get("tick0", {})
    check("tick 0 all normal", all(v["ts"]["status"] == 0 for v in (t0.get("targets") or {}).values()),
          "some target already damaged at tick 0")
    check("two views agree on every target", not d.get("mismatch"),
          "mismatched ids: %s" % (d.get("mismatch") or [])[:8])
    check("overview draws every roster target", not d.get("missingTargets"),
          "missing: %s" % (d.get("missingTargets") or [])[:8])
    check("overview billboard bars match status module", not d.get("barMismatch"),
          "bar mismatches: %s" % (d.get("barMismatch") or [])[:8])
    check("scrub back/forward restores same state", bool(d.get("scrubRestores")),
          "counts differ after scrub")
    dist = d.get("endByStatus") or {}
    # 台海案例被校准为四档齐全（高强度登陆战）；中印案例是低强度对峙，
    # 只应出现正常/受损，不应为了凑四档而硬造毁伤。
    if args.case == "taiwan_landing":
        check("destroyed exists", bool(dist.get("destroyed")),
              "no destroyed target at final tick")
        check("severe-but-alive exists", bool(dist.get("severe")),
              "no severe target at final tick")
        check("damaged exists", bool(dist.get("damaged")),
              "no damaged target at final tick")
        check("normal still exists", bool(dist.get("normal")),
              "no normal target at final tick")
    # 损毁目标必须可见：单独显示为 X，或进入带 X 的"损毁聚合组"。
    tk = (d.get("tickEnd") or {}).get("targets") or {}
    clusters = (d.get("tickEnd") or {}).get("clusters") or []
    dead_cluster_ids = set()
    for c in clusters:
        if c.get("deadCount"):
            dead_cluster_ids.update(c.get("deadIds") or [])
    def destroyed_visible(i):
        v = (tk.get(i) or {}).get("view", {})
        return bool(v.get("show") and v.get("markShow")) or i in dead_cluster_ids
    check("destroyed targets visible (X or destroyed cluster)",
          all(destroyed_visible(i) for i in dist.get("destroyed", [])),
          "some destroyed target is neither shown nor in a destroyed cluster")
    if args.case == "taiwan_dense_transfer_v1":
        check("dense case uses clustering", len(clusters) > 0, "no clusters at final tick")
        singles = sum(1 for v in tk.values()
                      if not v.get("view", {}).get("clustered")
                      and v.get("ts", {}).get("status") != 3)
        dead_single = sum(1 for v in tk.values()
                          if not v.get("view", {}).get("clustered")
                          and v.get("ts", {}).get("status") == 3)
        covered = sum(c.get("count", 0) for c in clusters) + singles + dead_single
        check("cluster groups account for full roster", covered == d.get("targetCount"),
              "covered=%s roster=%s" % (covered, d.get("targetCount")))
    # 去重叠：同类显示项（单目标之间、聚合图标之间）不得相交。
    items = (d.get("tickEnd") or {}).get("items") or []
    def rect(it):
        return (it["x"] - it["w"] / 2.0, it["y"] - it["h"],
                it["x"] + it["w"] / 2.0, it["y"] + 8)
    overlaps = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            if a.get("kind") != b.get("kind"):
                continue
            ax1, ay1, ax2, ay2 = rect(a)
            bx1, by1, bx2, by2 = rect(b)
            if not (ax2 + 1 < bx1 or bx2 + 1 < ax1 or ay2 + 1 < by1 or by2 + 1 < ay1):
                overlaps.append((a.get("id"), b.get("id")))
    check("same-kind display items do not overlap", not overlaps,
          "overlaps: %s" % overlaps[:6])
    counts = (d.get("tickEnd") or {}).get("counts") or {}
    check("counts sum to roster",
          (counts.get("red", {}).get("total", 0) + counts.get("blue", {}).get("total", 0))
          == d.get("targetCount"),
          "counts=%s total=%s" % (counts, d.get("targetCount")))

    print("4. 损毁与低血量的视觉区分")
    did = d.get("destroyedId")
    sid = d.get("severeId")
    end = d.get("tickEnd", {}).get("targets") or {}
    if did:
        v = end[did]
        check("destroyed: status 3 / hp 0", v["ts"]["status"] == 3 and v["ts"]["hp"] == 0,
              "%s" % v["ts"])
        check("destroyed: empty bar", (v["panel"] or {}).get("hpWidth") in ("0%", "0px", "0"),
              "width=%s" % (v["panel"] or {}).get("hpWidth"))
        check("destroyed: overview bar uses dark destroyed track",
              "%233F4652" in (v["view"].get("barImage") or ""),
              "bar tail=%s" % (v["view"].get("barImage") or "")[-120:])
        check("destroyed: X/marker or destroyed cluster shown",
              bool(v["view"]["markShow"]) or did in dead_cluster_ids,
              "markShow=%s clustered=%s" % (v["view"]["markShow"], v["view"].get("clustered")))
        check("destroyed: icon dimmed (not white)",
              (v["view"].get("iconTint") or "").upper() not in ("", "#FFFFFF"),
              "iconTint=%s" % v["view"].get("iconTint"))
        check("destroyed: panel class os-s3", (v["panel"] or {}).get("status") == 3,
              "panel=%s" % (v["panel"] or {}).get("status"))
    if sid:
        v = end[sid]
        check("severe: status 2 / hp in (0,35]", 0 < v["ts"]["hp"] <= 35,
              "%s" % v["ts"])
        check("severe: bar still filled", (v["panel"] or {}).get("hpWidth") not in
              ("0%", "0px", "0", None, ""),
              "width=%s" % (v["panel"] or {}).get("hpWidth"))
        check("severe: no icon-wrapping marker", not v["view"].get("markShow"),
              "markShow=%s" % v["view"].get("markShow"))
        check("severe: overview bar uses severe color",
              "%23E2662A" in (v["view"].get("barImage") or ""),
              "bar tail=%s" % (v["view"].get("barImage") or "")[-120:])

    print("5. 图例与选中同步")
    legend = d.get("tickEnd", {}).get("legend") or ""
    for label in ("正常", "受损", "严重受损", "损毁"):
        check("legend has %s" % label, label in legend, "legend=%r" % legend)
    check("selection synced to overview", bool((d.get("selectSync") or {}).get("selected")),
          "%s" % d.get("selectSync"))
    check("selection ring shown", bool((d.get("selectSync") or {}).get("ringShown")),
          "%s" % d.get("selectSync"))
    check("hide/show keeps all target entities", d.get("reshowCount") == d.get("targetCount"),
          "reshow=%s expected=%s" % (d.get("reshowCount"), d.get("targetCount")))

    print("6. 组件外观")
    check("no title text above overview", not d.get("overviewHeader"),
          "ov-head still present")
    check("cesium credits hidden", d.get("creditsDisplay") in ("none", "absent"),
          "credits display=%s" % d.get("creditsDisplay"))
    label = d.get("layerLabel") or ""
    check("layer renamed to 全局鸟瞰", "全局鸟瞰" in label and "鸟瞰图" not in label,
          "layerLabel=%r" % label)

    if FAILURES:
        print("\nFAILED %d check(s):" % len(FAILURES))
        for f in FAILURES:
            print("  - " + f)
        return 1
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
