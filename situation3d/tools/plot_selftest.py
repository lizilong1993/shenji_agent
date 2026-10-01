#!/usr/bin/env python3
"""Self-check for the tactical-plotting geometry (`web/src/plotting.js`).

Run on the target container, where there is no test runner:

    python3 tools/plot_selftest.py
    python3 tools/plot_selftest.py --url "http://127.0.0.1:8848/?case=taiwan_landing"

What this guards against
------------------------
Plotting geometry is computed in the browser, so its assertions used to live in
throwaway `--eval` snippets: they were re-typed for every change, and nothing
failed if a later edit broke an earlier rule. Three specific classes of error
have already shipped once each and are what this file pins down:

1. **Empty assertions.** A check written as `[].every(...)` passes whatever the
   code does. A real one did exactly that here: "the arrow is hidden when its
   layer is off" was reported green while zero entities existed. Every check
   below asserts a *count* before it asserts a property.
2. **Silently dead configuration.** A `params` key declared in `CATALOGUE` that no
   renderer reads looks like a working feature. `tools/plot_selftest.py` does not
   test each key (that is the FIX tasks' job), but `ringsFor` is exercised over
   *every* catalogue entry so a type that cannot generate geometry at all fails.
3. **NaN from degenerate control points.** Collinear or coincident control points
   produced NaN coordinates with no exception -- the arrow simply did not draw.
   The regression sweep asserts every coordinate is finite.

Checks:
1. `centredRun` -- a whole number of items, centred, spanning the run exactly.
2. `splitAtCorners` -- a straight line, a 90 deg bend, a shallow bend below the
   threshold, and a custom threshold.
3. `decorationScale` -- the visibility floor returns 0, a normal case returns > 0.
4. `boundaryPieces` -- two equal segments, a measurable mid gap, and two side
   labels that are perpendicular to the line and equidistant from the break.
5. `arrayPieces` -- the three decoration kinds produce marks of the right arity.
6. Catalogue sweep -- every entry produces finite geometry via `ringsFor`.

Exits non-zero with a readable message on the first failing group.
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
DEFAULT_URL = "http://127.0.0.1:8848/?case=taiwan_landing"

FAILURES = []


def fail(group: str, msg: str) -> None:
    FAILURES.append("%s: %s" % (group, msg))


# The probe returns raw measurements only; every threshold lives here so the
# numbers are reviewable in Python rather than buried in a JS snippet.
PROBE = r"""
(async function () {
  var t0 = Date.now();
  while (Date.now() - t0 < 40000 &&
         (!window.SituationHost || !SituationHost.state.data ||
          !SituationHost.state.data.case_id)) {
    await new Promise(function (r) { setTimeout(r, 500); });
  }
  await new Promise(function (r) { setTimeout(r, 3000); });
  if (!window.Plotting) return { error: 'window.Plotting missing' };
  var G = Plotting._geometry, out = {};

  // 1. centredRun
  var r1 = G.centredRun(100, 10, 5);
  var r2 = G.centredRun(97, 10, 5);
  out.run = {
    exact_n: r1.length,
    exact_first: r1[0],
    exact_last_end: r1[r1.length - 1] + 10,
    uneven_n: r2.length,
    uneven_lead: r2[0],
    uneven_tail: 97 - (r2[r2.length - 1] + 10),
    too_small: G.centredRun(5, 10, 5).length
  };

  // 2. splitAtCorners
  function pl(c) { return c.map(function (q) { return { x: q[0], y: q[1] }; }); }
  out.corners = {
    straight: G.splitAtCorners(pl([[0, 0], [1, 0], [2, 0]])).length,
    bend90: G.splitAtCorners(pl([[0, 0], [1, 0], [1, 1]])).length,
    shallow: G.splitAtCorners(pl([[0, 0], [1, 0], [2, 0.1]])).length,
    custom100: G.splitAtCorners(pl([[0, 0], [1, 0], [1, 1]]), 100).length
  };

  // 3. decorationScale
  out.scale = {
    floored: G.decorationScale(14, 100000, 200000, 0.05),
    normal: G.decorationScale(14, 700, 200000, 0.05)
  };

  // 4. boundaryPieces
  //
  // ⚠ **对称性与垂直度必须在"平面米制"里量，不能在经纬度里量。**
  // 构造发生在以首控制点为中心的局部平面里（plotting.js 的 toPlane/fromPlane），
  // 那里两个番号严格镜像、严格垂直于主线。而经纬度空间不是等距的：同一纬度圈上
  // 1° 经度比 1° 纬度短约 9%，直接拿"度"算点积会得到约 0.08（≈4.6°）的假偏差。
  // 第一版探针就是这么写的，于是把正确的实现在这里判成了失败。
  var A = [119.4, 24.2], B = [120.6, 25.0];
  function metres(a, b) {
    var mLat = 110540, mLon = 111320 * Math.cos(a[1] * Math.PI / 180);
    return Math.hypot((b[0] - a[0]) * mLon, (b[1] - a[1]) * mLat);
  }
  function plen(pts) {
    var s = 0; for (var i = 1; i < pts.length; i++) s += metres(pts[i - 1], pts[i]);
    return s;
  }
  var bp = G.piecesFor('boundary', [A, B]);
  var e0 = bp.lines[0][bp.lines[0].length - 1], s1 = bp.lines[1][0];
  // 投到同一平面再量（toPlane 以 A 为原点，与构造一致）
  var pA = G.toPlane([A], A)[0];
  var pG = G.toPlane([bp.glyph.pt], A)[0];
  var pL = G.toPlane(bp.labels.map(function (l) { return l.pt; }), A);
  function pd(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }
  var vdx = pA.x - pG.x, vdy = pA.y - pG.y;         // 断口→A 的平面方向
  var vll = Math.hypot(vdx, vdy) || 1e-9;
  var lx = pL[0].x - pG.x, ly = pL[0].y - pG.y;
  var ll = Math.hypot(lx, ly) || 1e-9;
  out.boundary = {
    segments: bp.lines.length,
    len_ratio: plen(bp.lines[1]) / plen(bp.lines[0]),
    gap_m: metres(e0, s1),
    glyph_text: bp.glyph.text,
    label_count: bp.labels.length,
    label_off_left: pd(pL[0], pG),
    label_off_right: pd(pL[1], pG),
    perp_dot: Math.abs((lx * vdx + ly * vdy) / (ll * vll))
  };

  // 5. arrayPieces
  function arity(kind) {
    var a = G.arrayPieces(kind, [A, B]);
    var sizes = {};
    a.marks.forEach(function (m) { sizes[m.length] = (sizes[m.length] || 0) + 1; });
    return { marks: a.marks.length, runs: a.runs, sizes: sizes };
  }
  out.array = { wire: arity('wire_fence'), ditch: arity('anti_tank_ditch'),
                mine: arity('mineline') };

  // 6. catalogue sweep
  var P2 = [[119.6, 24.3], [120.2, 24.9]];
  var P3 = [[119.6, 24.3], [120.2, 24.9], [119.9, 24.1]];
  var P4 = [[119.3, 24.1], [119.55, 24.1], [119.55, 24.28], [119.3, 24.28]];
  var bad = [], entries = Plotting.catalogue();
  entries.forEach(function (c) {
    var pts = (c.family === 'zone') ? P4
            : (c.family === 'arrow2' || c.family === 'band' ||
               c.family === 'line' || c.family === 'array') ? P2 : P3;
    try {
      var rings = G.ringsFor(c.type, pts);
      var minPts = (c.family === 'line') ? 2 : 3;
      if (c.family === 'array') {
        if (rings.length !== 0) bad.push(c.type + ':array-should-have-no-ring');
        return;
      }
      if (!rings.length) { bad.push(c.type + ':no-ring'); return; }
      rings.forEach(function (r) {
        if (r.length < minPts) bad.push(c.type + ':pts');
        r.forEach(function (q) {
          if (!isFinite(q[0]) || !isFinite(q[1])) bad.push(c.type + ':nan');
        });
      });
    } catch (e) { bad.push(c.type + ':' + e.message); }
  });
  out.sweep = { total: entries.length, bad: bad };

  // 7. FIX 回归：修复项各自的可判定断言
  var cat = {};
  entries.forEach(function (c) { cat[c.type] = c; });
  out.fixes = {
    // FIX-1：元数据如实（不再声称未实现的算法族）+ custom 标记可读
    assembly_family: cat['assembly_area'] ? cat['assembly_area'].algorithmFamily : null,
    mined_family: cat['mined_area'] ? cat['mined_area'].algorithmFamily : null,
    custom_flag: cat['maritime_blockade'] ? cat['maritime_blockade'].custom : null,
    noncustom_flag: cat['assembly_area'] ? cat['assembly_area'].custom : null,
    // FIX-3：齿朝向两种取值下的首齿尖有向距离（应符号相反、绝对值相等）
    tooth: (function () {
      function apexSign(flag) {
        var orig = Plotting.CATALOGUE['anti_tank_ditch'].params.toothOutward;
        Plotting.CATALOGUE['anti_tank_ditch'].params.toothOutward = flag;
        var a = G.arrayPieces('anti_tank_ditch', [[119.4, 24.2], [120.6, 25.0]]);
        Plotting.CATALOGUE['anti_tank_ditch'].params.toothOutward = orig;
        if (!a.marks.length) return null;
        var tri = a.marks[0];
        // 有向面积符号：>0 与 <0 表示齿尖在主线两侧
        var cross = (tri[1][0] - tri[0][0]) * (tri[2][1] - tri[0][1])
                  - (tri[1][1] - tri[0][1]) * (tri[2][0] - tri[0][0]);
        return cross;
      }
      return { outward: apexSign(true), inward: apexSign(false) };
    })(),
    // FIX-4：不传 sideLabels 不产生空标签；传了就渲染两个
    boundary_plain: (function () {
      var r = Plotting.add({ type: 'boundary', layer: 'plot.defense',
                             points: [[119.4, 24.2], [120.6, 25.0]] });
      var n = r.entities.length; Plotting.remove(r.id); return n;
    })(),
    boundary_labelled: (function () {
      var r = Plotting.add({ type: 'boundary', layer: 'plot.defense',
                             points: [[119.4, 24.2], [120.6, 25.0]],
                             sideLabels: ['甲', '乙'] });
      var n = r.entities.length;
      var texts = r.entities.map(function (e) {
        return e.label && e.label.text ? e.label.text.getValue() : null;
      }).filter(Boolean);
      Plotting.remove(r.id);
      return { n: n, texts: texts };
    })(),
    // FIX-2：线型由 params.dash 决定
    line_materials: (function () {
      // 目录里**没有**一个应当画实线的 `family:'line'` 类型（coordination_line 与
      // phase_line 都是 dash:true），所以不能挑一个"现成的实线类型"来对比——
      // 第一版就是这么写的，结果拿 phase_line 当实线基准，必然失败。
      // 改为**直接翻转参数**来验证因果：同一个类型，dash 开关一开一关，材质应当跟着变。
      function matOf(type, dashOverride) {
        var def = Plotting.CATALOGUE[type];
        var orig = def.params.dash;
        if (dashOverride !== undefined) def.params.dash = dashOverride;
        var r = Plotting.add({ type: type, layer: 'plot.fire',
                               points: [[119.4, 24.2], [120.6, 25.0]] });
        // ⚠ 不能用 `constructor.name` 判材质类型：vendored Cesium 是**压缩过的**，
        // 类名被改成 `x1` 之类，按名字比对必然得到假失败（第一版就是这样）。
        // `instanceof` 不受压缩影响，是唯一可靠的判法。
        var dashed = null;
        r.entities.forEach(function (e) {
          if (e.polyline && e.polyline.material) {
            dashed = e.polyline.material instanceof Cesium.PolylineDashMaterialProperty;
          }
        });
        Plotting.remove(r.id);
        if (dashOverride !== undefined) def.params.dash = orig;
        return dashed;
      }
      return { dashed: matOf('phase_line', true), solid: matOf('phase_line', false) };
    })()
  };
  return out;
})()
"""


def probe(url: str) -> dict:
    """Drive the headless browser and return the measurement dict."""
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "probe.json")
        # `inspect_remote.py` takes the expression as a **string** on --eval; it has no
        # --eval-file. Passing argv as a list avoids the shell entirely, which is what
        # makes a multi-line JS payload safe here: there is no quoting layer to escape.
        cmd = [sys.executable, INSPECT, "--url", url, "--out", out,
               "--wait-ms", "20000", "--eval-async", "--eval", PROBE]
        try:
            subprocess.run(cmd, cwd=ROOT, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as exc:
            sys.stderr.write(exc.stderr.decode("utf-8", "replace") if exc.stderr else "")
            raise SystemExit("[plot] inspect_remote.py 失败；确认服务已启动、URL 可达")
        with open(out, "r", encoding="utf-8") as fh:
            return json.load(fh)


def check(name: str, cond: bool, detail: str) -> None:
    print("  %s %s" % ("ok  " if cond else "FAIL", name))
    if not cond:
        fail(name, detail)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default=DEFAULT_URL)
    args = ap.parse_args()

    raw = probe(args.url)
    if raw.get("evalError"):
        print("probe error: %s" % raw["evalError"])
        return 1
    d = raw.get("eval") or {}
    if d.get("error"):
        print("probe returned: %s" % d["error"])
        return 1
    if raw.get("pageErrors"):
        print("pageErrors: %s" % "; ".join(raw["pageErrors"][:3]))
        return 1

    print("1. centredRun -- whole number of items, centred")
    r = d["run"]
    check("exact fit: 7 items spanning 0..100",
          r["exact_n"] == 7 and r["exact_first"] == 0 and abs(r["exact_last_end"] - 100) < 1e-9,
          "got n=%s first=%s end=%s" % (r["exact_n"], r["exact_first"], r["exact_last_end"]))
    check("uneven run: lead equals tail",
          r["uneven_n"] == 6 and abs(r["uneven_lead"] - r["uneven_tail"]) < 1e-9,
          "lead=%s tail=%s" % (r["uneven_lead"], r["uneven_tail"]))
    check("does not fit -> empty", r["too_small"] == 0, "got %s" % r["too_small"])

    print("2. splitAtCorners -- 30 deg default threshold")
    c = d["corners"]
    check("straight -> 1 run", c["straight"] == 1, "got %s" % c["straight"])
    check("90 deg -> 2 runs", c["bend90"] == 2, "got %s" % c["bend90"])
    check("shallow bend -> 1 run", c["shallow"] == 1, "got %s" % c["shallow"])
    check("threshold 100 -> 1 run", c["custom100"] == 1, "got %s" % c["custom100"])

    print("3. decorationScale -- visibility floor")
    s = d["scale"]
    check("oversized item -> 0", s["floored"] == 0, "got %s" % s["floored"])
    check("normal item -> in (0,1]", 0 < s["normal"] <= 1, "got %s" % s["normal"])

    print("4. boundaryPieces -- two segments, mid gap, perpendicular labels")
    bd = d["boundary"]
    check("two segments", bd["segments"] == 2, "got %s" % bd["segments"])
    check("segments equal (break at midpoint)",
          abs(bd["len_ratio"] - 1.0) < 0.02, "ratio=%.4f" % bd["len_ratio"])
    check("gap is measurable", 1000 < bd["gap_m"] < 20000, "gap=%.0fm" % bd["gap_m"])
    check("glyph present", bool(bd["glyph_text"]), "text=%r" % bd["glyph_text"])
    check("two side labels", bd["label_count"] == 2, "got %s" % bd["label_count"])
    check("labels equidistant from break",
          abs(bd["label_off_left"] - bd["label_off_right"]) < 1.0,
          "%.1f vs %.1f" % (bd["label_off_left"], bd["label_off_right"]))
    check("labels perpendicular to line", bd["perp_dot"] < 1e-3,
          "dot=%.5f" % bd["perp_dot"])

    print("5. arrayPieces -- three decoration kinds")
    ar = d["array"]
    check("wire marks are 2-point strokes",
          ar["wire"]["marks"] > 0 and set(ar["wire"]["sizes"]) == {"2"},
          "marks=%s sizes=%s" % (ar["wire"]["marks"], ar["wire"]["sizes"]))
    check("ditch marks are 3-point teeth",
          ar["ditch"]["marks"] > 0 and set(ar["ditch"]["sizes"]) == {"3"},
          "marks=%s sizes=%s" % (ar["ditch"]["marks"], ar["ditch"]["sizes"]))
    check("mine marks are 4-point diamonds",
          ar["mine"]["marks"] > 0 and set(ar["mine"]["sizes"]) == {"4"},
          "marks=%s sizes=%s" % (ar["mine"]["marks"], ar["mine"]["sizes"]))

    print("6. catalogue sweep -- every type generates finite geometry")
    sw = d["sweep"]
    check("24 types present", sw["total"] == 24, "got %s" % sw["total"])
    check("no NaN / missing ring", not sw["bad"], "bad=%s" % ", ".join(sw["bad"][:6]))

    print("7. FIX regression -- metadata honesty, line style, tooth side, instance labels")
    fx = d.get("fixes") or {}
    check("FIX-1: assembly_area no longer claims closed-spline",
          fx.get("assembly_family") == "area", "got %r" % fx.get("assembly_family"))
    check("FIX-1: mined_area no longer claims repeated-glyph",
          fx.get("mined_family") == "area", "got %r" % fx.get("mined_family"))
    check("FIX-1: custom marker readable",
          fx.get("custom_flag") is True and fx.get("noncustom_flag") is False,
          "custom=%r noncustom=%r" % (fx.get("custom_flag"), fx.get("noncustom_flag")))
    tc = fx.get("tooth") or {}
    check("FIX-3: tooth side flips with toothOutward",
          tc.get("outward") and tc.get("inward") and
          (tc["outward"] > 0) != (tc["inward"] > 0),
          "outward=%r inward=%r" % (tc.get("outward"), tc.get("inward")))
    check("FIX-4: no side labels -> 3 entities (2 lines + glyph)",
          fx.get("boundary_plain") == 3, "got %s" % fx.get("boundary_plain"))
    bl = fx.get("boundary_labelled") or {}
    # 5 = 2 段线 + 1 等级代字 + 2 侧番号；等级代字也是 label 实体，所以要期望三个文本。
    check("FIX-4: sideLabels -> 5 entities, glyph + both side texts",
          bl.get("n") == 5 and sorted(bl.get("texts") or []) == ["T", "乙", "甲"],
          "n=%s texts=%s" % (bl.get("n"), bl.get("texts")))
    lm = fx.get("line_materials") or {}
    check("FIX-2: dash:true -> dashed material",
          lm.get("dashed") is True, "got %r" % lm.get("dashed"))
    check("FIX-2: no dash -> NOT dashed material",
          lm.get("solid") is False, "got %r" % lm.get("solid"))

    print()
    if FAILURES:
        print("FAILED (%d): %s" % (len(FAILURES), "; ".join(FAILURES)))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())