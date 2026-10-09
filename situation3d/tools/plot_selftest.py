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
7. Theatre facilities -- hotspot port/airport counts for the case under test
   (`EXPECT_FACILITIES`); `facilityDebug()` is preferred, `SituationHost.state`
   is the fallback.

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
  // 8. OPT-3 回归：屏幕空间装饰的缩放重建
  out.screen = (function () {
    var A = [119.4, 24.2], B = [120.6, 25.0];
    var origin = G.toPlane([A], A)[0];
    var run = G.toPlane([A, B], A);
    function markSpan(mpp) {
      var marks = G.decorAlongRun(run, 'wire', mpp, {});
      if (!marks.length) return null;
      var m = marks[0];
      return Math.hypot(m[1].x - m[0].x, m[1].y - m[0].y);
    }
    // 两个区间都要测，因为 `share` 上限在长段上会生效、在短段上不生效：
    //   mpp = 200/400 时段长约 750/376 px，5% 上限 > 图元像素尺寸 → 不触顶
    //   mpp = 1000/2000 时段长约 150/75 px → 触顶，地面尺寸被压住
    var sA = markSpan(200), sB = markSpan(400);
    var sC = markSpan(1000), sD = markSpan(2000);
    var runM = 0;
    for (var i = 1; i < run.length; i++) runM += Math.hypot(run[i].x - run[i - 1].x, run[i].y - run[i - 1].y);
    return {
      mpp_finite: (function () { var v = G.metersPerPixel(); return isFinite(v) && v > 0; })(),
      uncapped_ratio: (sA && sB) ? (sB / sA) : null,
      run_m: runM,
      // 触顶时：屏幕尺寸应 ≈ 段长屏幕像素的 5%
      capped_px_at_2000: sD ? (sD / 2000) : null,
      // ⚠ 期望值要乘 √2：`markSpan` 量的是铁丝网 **X 的一条对角线**，而
      // `itemPx * scale` 是那条边的分量（w = itemM/2, hh = itemM/2）——
      // 对角线长 = 2·√(w²+hh²) = itemM·√2。第一版漏了这个因子，把正确的实现判成失败。
      capped_expected_px_at_2000: (runM / 2000) * 0.05 * Math.SQRT2,
      // 同一档位内重复调用不应重建；强制重建应生效并计数
      no_rebuild_same_bucket: Plotting.refreshScreenSpaceDecorations(false) === false,
      forced_rebuild: Plotting.refreshScreenSpaceDecorations(true) === true,
      debug_has_bucket: (function () {
        var d = Plotting.debug();
        return typeof d.bucket === 'number' && typeof d.rebuilds === 'number' && d.rebuilds >= 1;
      })(),
      // 强制重建后，已存在实例的图元数不应变化（重建是幂等的）
      idempotent: (function () {
        var r = Plotting.add({ type: 'wire_fence', layer: 'plot.defense',
                               points: [[119.4, 24.2], [120.6, 25.0]] });
        var before = r.entities.length;
        Plotting.refreshScreenSpaceDecorations(true);
        var after = r.entities.length;
        Plotting.remove(r.id);
        return { before: before, after: after };
      })()
    };
  })();
  // 9. OPT-1 回归：多层火力扇面环带
  out.fan = (function () {
    var A = [119.6, 24.3];
    // 控制点 2 定半径（约 50 km），控制点 3 定张角
    var sector = [A, [120.05, 24.3], [119.9, 23.9]];
    var circ = [A, [120.05, 24.3], [120.05, 24.3]];
    var mLat = 110540, mLon = 111320 * Math.cos(A[1] * Math.PI / 180);
    function metres(a, b) { return Math.hypot((b[0] - a[0]) * mLon, (b[1] - a[1]) * mLat); }
    function profile(pts) {          // 每环到圆心距离的最小/最大值
      var c = pts[0], lo = Infinity, hi = -Infinity;
      // 圆心是控制点 A，不是 ring[0]（多带时 ring[0] 在外弧上）
      pts.forEach(function (q) {
        var d = metres(A, q);
        if (d < lo) lo = d;
        if (d > hi) hi = d;
      });
      return { lo: lo, hi: hi };
    }
    function run(type, bands, spread) {
      var def = Plotting.CATALOGUE[type];
      var ob = def.params.rangeBands, os = def.params.spreadDeg;
      def.params.rangeBands = bands;
      if (spread !== undefined) def.params.spreadDeg = spread;
      var pts = (spread !== undefined && spread >= 360) ? circ : sector;
      var rings = G.ringsFor(type, pts).map(profile);
      def.params.rangeBands = ob;
      if (spread !== undefined) def.params.spreadDeg = os;
      return rings;
    }
    var one = run('fire_range', [1.0], 70);
    var three = run('fire_range', [0.4, 0.7, 1.0], 70);
    var full = run('circular_range', [0.5, 1.0], 360);
    return {
      single_rings: one.length,
      single_hi: one[0] ? one[0].hi : null,
      single_lo: one[0] ? one[0].lo : null,
      three_rings: three.length,
      three_lo: three.map(function (r) { return r.lo; }),
      three_hi: three.map(function (r) { return r.hi; }),
      full_rings: full.length,
      full_lo: full.map(function (r) { return r.lo; }),
      full_hi: full.map(function (r) { return r.hi; }),
      // 非法输入不得产生退化环
      bad_input_rings: run('fire_range', [0, -1, NaN], 70).length,
      // 圆形单带：首尾弧点应重合（同一角度、同一半径）
      circ_closed_m: (function () {
        var def = Plotting.CATALOGUE['circular_range'];
        var ob = def.params.rangeBands;
        def.params.rangeBands = [1.0];
        var r = G.ringsFor('circular_range', circ)[0];
        def.params.rangeBands = ob;
        return metres(r[1], r[r.length - 1]);
      })()
    };
  })();
  // 10. OPT-2 回归：高度层空域
  out.alt = (function () {
    var P4 = [[119.3, 24.1], [119.55, 24.1], [119.55, 24.28], [119.3, 24.28]];
    function addOne(type) {
      var r = Plotting.add({ type: type, layer: 'plot.intel', points: P4 });
      var poly = null, wall = null, labelText = null;
      r.entities.forEach(function (e) {
        if (e.polygon) poly = e.polygon;
        if (e.wall) wall = e.wall;
        if (e.label && e.label.text) labelText = e.label.text.getValue();
      });
      var out = {
        n: r.entities.length,
        has_height: !!(poly && poly.height !== undefined),
        height: poly && poly.height !== undefined ? poly.height.getValue() : null,
        extruded: poly && poly.extrudedHeight !== undefined ? poly.extrudedHeight.getValue() : null,
        // ⚠ Cesium 把传入的数组包成 Property，`minimumHeights.length` 是 undefined
        // （JSON 里直接消失，读出来像"墙没建"）。必须 `.getValue()` 再取长度。
        wall_lows: wall && wall.minimumHeights ? wall.minimumHeights.getValue().length : null,
        wall_highs: wall && wall.maximumHeights ? wall.maximumHeights.getValue().length : null,
        ring_pts: G.ringsFor(type, P4)[0].length,
        label: labelText
      };
      Plotting.remove(r.id);
      return out;
    }
    var nfz = addOne('no_fly_zone');
    var plain = addOne('battle_position');       // 无 altLayer，应走贴地分支
    return {
      nofly: nfz, plain: plain,
      fmt: {
        msl: G.formatAltitude(1500, 'MSL'),
        fl: G.formatAltitude(150, 'FL'),
        bare: G.formatAltitude(3000, null),
        passthrough_num: G.formatAltitude('FL150', 'MSL'),
        passthrough_txt: G.formatAltitude('1500MSL', null),
        blank: G.formatAltitude('', 'AGL')
      },
      illegal: (function () {
        var def = Plotting.CATALOGUE['no_fly_zone'];
        var orig = def.params.altLayer;
        def.params.altLayer = [10000, 0];        // 上限 <= 下限
        var r = Plotting.add({ type: 'no_fly_zone', layer: 'plot.intel', points: P4 });
        var hasVol = r.entities.some(function (e) { return e.polygon && e.polygon.extrudedHeight; });
        Plotting.remove(r.id);
        def.params.altLayer = orig;
        return { has_volume: hasVol, parsed: G.parseAltLayer({ altLayer: [10000, 0] }) };
      })()
    };
  })();
  // 11. OPT-4 回归：顶点预算
  out.budget = (function () {
    var A = [119.6, 24.3];
    var sector = [A, [120.05, 24.3], [119.9, 23.9]];
    var circ = [A, [120.05, 24.3], [120.05, 24.3]];
    function bbox(pts) {
      var x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      pts.forEach(function (q) {
        if (q[0] < x0) x0 = q[0];
        if (q[0] > x1) x1 = q[0];
        if (q[1] < y0) y0 = q[1];
        if (q[1] > y1) y1 = q[1];
      });
      return [x0, x1, y0, y1];
    }
    function worst(type, bands, spread) {
      var def = Plotting.CATALOGUE[type];
      var ob = def.params.rangeBands, os = def.params.spreadDeg;
      def.params.rangeBands = bands;
      if (spread !== undefined) def.params.spreadDeg = spread;
      var pts = (spread >= 360) ? circ : sector;
      var rings = G.ringsFor(type, pts);
      // 直接调 decimateRing 前后对比外接盒（同一条环）
      var raw = G.ringsFor(type, pts).map(function (r) { return r; });
      var maxV = 0;
      rings.forEach(function (r) { if (r.length > maxV) maxV = r.length; });
      def.params.rangeBands = ob;
      if (spread !== undefined) def.params.spreadDeg = os;
      return { max_verts: maxV, bbox: rings.length ? bbox(rings[0]) : null, nrings: rings.length };
    }
    var many = worst('circular_range', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 360);
    // 外接盒保持：对同一条环，降采样后 bbox 应逐位相同
    var ring = [];
    for (var i = 0; i < 4000; i++) {
      var t = i / 4000 * Math.PI * 2;
      ring.push([119.6 + Math.cos(t) * 0.5 * (1 + 0.3 * Math.sin(7 * t)),
                 24.3 + Math.sin(t) * 0.5]);
    }
    var d = G.decimateRing(ring, 512);
    var b0 = bbox(ring), b1 = bbox(d);
    return {
      many_rings: many.nrings, many_max_verts: many.max_verts,
      cap: G.MAX_RING_VERTS,
      decimated_from: ring.length, decimated_to: d.length,
      bbox_same: b0.every(function (v, k) { return v === b1[k]; })
    };
  })();
  // 12. FIX-5 回归：阵列图元合并为批量图元（数量与图元数无关）
  out.batch = await (async function () {
    function frames(n) {
      return new Promise(function (res) {
        var k = 0;
        function step() { if (++k >= n) return res(); requestAnimationFrame(step); }
        requestAnimationFrame(step);
      });
    }
    var scene = SituationHost.viewer().scene;
    var A = [119.4, 24.2];
    // 同一条线上取长短两种跨度：marks 数差很多，批量图元数应完全相同
    // `tiny` 是**明确低于可见下限**的段：图元会被丢弃、只剩线（DECOR_MIN_PX=3）。
    // `long` 是明显高于下限的段：应有大量图元但仍只有 1 个批量图元。
    // 第一版拿 `tiny` 当"图元较少"的对照组，结果 marks=0，把正确的下限行为
    // 判成了失败——所以现在把这两种情形分开断言。
    var tiny = [A, [119.42, 24.215]];
    var long = [A, [120.6, 25.0]];
    function probe(pts, type) {
      var before = scene.primitives.length;
      var r = Plotting.add({ type: type, layer: 'plot.defense', points: pts });
      return frames(25).then(function () {
        var d = Plotting.debug().items.filter(function (i) { return i.id === r.id; })[0] || {};
        var ready = r.primitives.map(function (pr) { return pr.ready === true; });
        Plotting.remove(r.id);
        return frames(3).then(function () {
          return { marks: d.marks, entities: d.entities, primitives: d.primitives,
                   ready: ready, leaked: scene.primitives.length - before };
        });
      });
    }
    var s = await probe(tiny, 'wire_fence');
    var l = await probe(long, 'wire_fence');
    var f = await probe(long, 'anti_tank_ditch');
    return { wire_tiny: s, wire_long: l, fill_long: f, min_px: 3 };
  })();
  return out;
})()
"""


# Second payload, for the theatre-filtering checks: counts the port/airport
# hotspots the page built for the current case. `SituationHost.facilityDebug()`
# is the contract (`app.js`); when the hook is missing the counts are derived
# from `SituationHost.state` instead and `has_debug` says which path ran, so
# losing the hook cannot make this check pass on zero entities.
PROBE_FACILITY = r"""
(async function () {
  var t0 = Date.now();
  while (Date.now() - t0 < 40000 && !(window.SituationHost &&
         SituationHost.state.data && SituationHost.state.data.case_id)) {
    await new Promise(function (r) { setTimeout(r, 500); });
  }
  await new Promise(function (r) { setTimeout(r, 3000); });
  var H = window.SituationHost;
  if (!H) return { error: 'window.SituationHost missing' };
  var S = H.state || {}, hs = (S.data && S.data.hotspots) || [];
  var list = Array.isArray(hs) ? hs : Object.keys(hs).map(function (k) { return hs[k]; });
  list = list.reduce(function (a, x) { return a.concat(Array.isArray(x) ? x : [x]); }, []);
  var fd = (typeof H.facilityDebug === 'function') ? H.facilityDebug() : null;
  if (fd && typeof fd.then === 'function') fd = await fd;
  // `facilityDebug()` reports {data:{kind:count}, anchors:{...}, labels:{...}}.
  function kindN(kind) {
    var d = (fd && fd.data) || {};
    var v = d[kind];
    if (typeof v === 'number') return v;
    if (v && typeof v.length === 'number') return v.length;
    return null;
  }
  function sum(x) {
    if (typeof x === 'number') return x;
    if (!x) return null;
    var ks = Object.keys(x), t = 0, i;
    for (i = 0; i < ks.length; i++) t += (typeof x[ks[i]] === 'number' ? x[ks[i]] : 1);
    return t;
  }
  function kindOf(x) { return '' + ((x && (x.kind || x.type)) || '') + ' ' + ((x && (x.name || x.label)) || ''); }
  var scan = { port: 0, airport: 0 };
  list.forEach(function (x) {
    var s = kindOf(x).toLowerCase();
    if (s.indexOf('airport') >= 0 || s.indexOf('\u673a\u573a') >= 0) scan.airport += 1;
    else if (s.indexOf('port') >= 0 || s.indexOf('\u6e2f') >= 0) scan.port += 1;
  });
  var he = S.hotspotEntities;
  var he_n = he ? (he.length === undefined ? Object.keys(he).length : he.length) : null;
  function or(v, fb) { return typeof v === 'number' ? v : fb; }
  return {
    case_id: S.data && S.data.case_id,
    has_debug: !!fd,
    port: or(kindN('port'), scan.port),
    airport: or(kindN('airport'), scan.airport),
    port_fd: kindN('port'),
    airport_fd: kindN('airport'),
    port_scan: scan.port,
    airport_scan: scan.airport,
    data: or(sum(fd && fd.data), list.length),
    anchors: or(sum(fd && fd.anchors), he_n),
    labels: sum(fd && fd.labels),
    data_kinds: (fd && fd.data) ? JSON.stringify(fd.data) : null,
    anchor_kinds: (fd && fd.anchors) ? JSON.stringify(fd.anchors) : null,
    facility_bboxes: list.filter(function (x) { return x && (x.kind === 'port' || x.kind === 'airport'); })
        .map(function (x) { return { kind: x.kind, name: x.name, bbox: x.bbox }; }),
    label_kinds: (fd && fd.labels) ? JSON.stringify(fd.labels) : null
  };
})()
"""

HANEDA_NAME = "\u65e5\u672c \u7fbd\u7530\u673a\u573a"

# Expected hotspot counts per case: (port min, port max, airport min, theatre
# bbox west/south/east/north from `engine/scenario.py`); port max
# None means no upper bound. Mirrors the theatre bbox filter in
# `engine/scenario.py` (`THEATRE_BBOXES`): taiwan_strait keeps its ports,
# india_china drops every port but keeps its airports. The last tuple element
# lists names declared exempt from the box: only the Haneda airport, whose entry
# at `engine/scenario.py:102-103` says it sits outside the theatre bbox and gets
# its own imagery/elevation download. The list is self-checked in both
# directions, so it cannot silently become a no-op.
EXPECT_FACILITIES = {
    "taiwan_landing": (4, None, 3, (115.5, 20.0, 122.5, 26.5), (HANEDA_NAME,)),
    "india_china_west": (0, 0, 3, (77.0, 31.5, 81.5, 36.5), ()),
}



# Third payload: the trajectory window / predicted leg / focus mask contract that
# `app.js` exposes through `SituationHost.focusDebug()` and drives through
# `SituationHost.selectTarget()`. Every number below is read from the page, and
# the selection is toggled in-place, so "the thing really happened" is what the
# assertions in section 8 compare.
PROBE_FOCUS = r"""
(async function () {
  var t0 = Date.now();
  while (Date.now() - t0 < 40000 && !(window.SituationHost &&
         SituationHost.state.data && SituationHost.state.data.case_id)) {
    await new Promise(function (r) { setTimeout(r, 500); });
  }
  await new Promise(function (r) { setTimeout(r, 3000); });
  var H = window.SituationHost;
  if (!H) return { error: 'window.SituationHost missing' };
  if (typeof H.focusDebug !== 'function') return { error: 'focusDebug missing' };
  function frames(n) {
    return new Promise(function (res) {
      var i = 0;
      (function step() { if (++i >= n) return res(); requestAnimationFrame(step); })();
    });
  }
  function layer(id, on) {
    try { LayerRegistry.setDesired(id, on); LayerRegistry.apply(); return true; }
    catch (e) { return false; }
  }
  var S = H.state, o = { mask_layer_ok: false, tracks_on: false };
  function snap(tag, d) {
    o[tag] = {
      trailWindowTicks: d.trailWindowTicks, predictMode: d.predictMode, selected: d.selected,
      maxTrailVertices: d.maxTrailVertices, trailVertexSum: d.trailVertexSum,
      trailFull: d.trailFull, trailDrawn: d.trailDrawn, tick: d.tick,
      predictedPolylines: d.predictedPolylines, predictedPoints: d.predictedPoints,
      historySolid: d.historySolid, maskExists: d.maskExists, maskShow: d.maskShow
    };
  }
  // `selectTarget()` and the mask land on the next sync / RenderGovernor apply, so
  // a read taken right after the call still shows the previous frame. Poll until
  // the state is stable and record how long it took; a timeout keeps the last raw
  // frame so the failure can be diagnosed instead of guessed.
  async function poll(cond, budget_ms) {
    var t = Date.now(), n = 0, d = H.focusDebug();
    while (Date.now() - t < budget_ms && !cond(d)) {
      await new Promise(function (r) { setTimeout(r, 250); });
      d = H.focusDebug();
      n += 1;
    }
    return { d: d, n: n, ok: cond(d), ms: Date.now() - t };
  }
  var mask0 = (LayerRegistry.runtime['overlay.focusMask'] || {}).desired;
  o.mask_desired0 = typeof mask0 === 'undefined' ? null : mask0;
  o.tracks_on = layer('layerTracks', true);
  if (typeof H.selectTarget === 'function') H.selectTarget(null);
  // A windowed trail is only meaningful once the full trail is longer than the
  // window: run the sim past tick 240, then pause so later reads are stable.
  // A timeout is reported as advance_ok=false and fails the assertion - never
  // silently skipped.
  o.playing0 = !!S.playing;
  var t_adv = Date.now();
  S.playing = true;
  while (Date.now() - t_adv < 30000 && !(S.tick > 240)) {
    await new Promise(function (r) { setTimeout(r, 250); });
  }
  S.playing = false;
  o.tick_at_read = S.tick;
  o.advance_ok = S.tick > 240;
  await frames(10);
  snap('base', H.focusDebug());
  o.mask_layer_ok = layer('overlay.focusMask', true);
  await frames(6);
  snap('mask_unselected', H.focusDebug());
  var tarr = (S.data && S.data.targets) || S.targets || [];
  if (!Array.isArray(tarr)) tarr = Object.keys(tarr).map(function (k) { return tarr[k]; });
  var t0x = tarr.length ? tarr[0] : null;
  o.targets_n = tarr.length;
  o.target_id = t0x ? (t0x.id || t0x.target_id || null) : null;
  if (o.target_id && typeof H.selectTarget === 'function') H.selectTarget(o.target_id);
  var p1 = await poll(function (d) { return d.predictedPolylines === 1; }, 3000);
  o.select_ok = p1.ok; o.select_polls = p1.n; o.select_ms = p1.ms;
  var p2 = await poll(function (d) { return d.maskShow === true; }, 12000);
  o.mask_ok = p2.ok; o.mask_polls = p2.n; o.mask_ms = p2.ms;
  o.selected_last = { selected: p2.d.selected, predictedPolylines: p2.d.predictedPolylines,
                      predictedPoints: p2.d.predictedPoints, historySolid: p2.d.historySolid,
                      maskExists: p2.d.maskExists, maskShow: p2.d.maskShow,
                      maxTrailVertices: p2.d.maxTrailVertices, trailFull: p2.d.trailFull,
                      trailDrawn: p2.d.trailDrawn, trailVertexSum: p2.d.trailVertexSum,
                      tick: p2.d.tick };
  snap('selected', p2.d);
  if (typeof H.selectTarget === 'function') H.selectTarget(null);
  var p3 = await poll(function (d) { return d.maskShow === false; }, 12000);
  o.mask_hide_ok = p3.ok; o.mask_hide_polls = p3.n; o.mask_hide_ms = p3.ms;
  o.unselected_last = { selected: p3.d.selected, predictedPolylines: p3.d.predictedPolylines,
                        maskShow: p3.d.maskShow, tick: p3.d.tick };
  snap('unselected', p3.d);
  if (o.mask_desired0 !== null) layer('overlay.focusMask', o.mask_desired0);
  return o;
})()
"""



PROBE_RELATIONS = r"""
(async function () {
  var t0 = Date.now();
  while (Date.now() - t0 < 40000 && !(window.SituationHost &&
         SituationHost.state.data && SituationHost.state.data.case_id)) {
    await new Promise(function (r) { setTimeout(r, 500); });
  }
  await new Promise(function (r) { setTimeout(r, 3000); });
  var H = window.SituationHost;
  if (!H) return { error: 'window.SituationHost missing' };
  if (typeof H.relationsDebug !== 'function') return { error: 'relationsDebug missing' };
  var S = H.state, out = {};
  // Run the sim past the window so distances (and the hostile-line set) reflect a
  // realistic mid-scenario tick instead of the paused tick-0 frame.
  S.playing = true;
  var t_adv = Date.now();
  while (Date.now() - t_adv < 30000 && !(S.tick > 240)) {
    await new Promise(function (r) { setTimeout(r, 250); });
  }
  S.playing = false;
  out.tick = S.tick;
  out.advance_ok = S.tick > 240;
  var defs = (S.data && S.data.targets) || [];
  if (!Array.isArray(defs)) defs = Object.keys(defs).map(function (k) { return defs[k]; });
  function attrs(d) { return (d && (d.attributes || d.attrs)) || {}; }
  function idOf(d) { return d && (d.id || d.target_id); }
  var leads = defs.filter(function (d) { return attrs(d).pair_role === 'lead'; });
  if (!leads.length) return { error: 'no lead target in the served dataset' };
  var wrap = S.targets || {};
  var now = null;
  try { now = H.viewer().clock.currentTime; } catch (e) { now = null; }
  function pos(d) {
    var w = wrap[idOf(d)];
    var e = w && w.entity;
    if (!e || !e.position || !now) return null;
    try {
      var c = Cesium.Cartographic.fromCartesian(e.position.getValue(now));
      return { lon: Cesium.Math.toDegrees(c.longitude), lat: Cesium.Math.toDegrees(c.latitude) };
    } catch (err) { return null; }
  }
  function distKm(a, b) {
    var R = 6371.0, p = Math.PI / 180;
    var dLat = (b.lat - a.lat) * p, dLon = (b.lon - a.lon) * p;
    var s = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
            Math.cos(a.lat * p) * Math.cos(b.lat * p) * Math.sin(dLon / 2) * Math.sin(dLon / 2);
    return 2 * R * Math.asin(Math.min(1, Math.sqrt(s)));
  }
  // Case-level engage check: the criterion is engage unless the case has no
  // engage-coverage hostile pair at all (then the page falls back to influence).
  var caseEngage = false;
  for (var a = 0; a < defs.length && !caseEngage; a++) {
    var pa = pos(defs[a]);
    if (!pa) continue;
    for (var b = a + 1; b < defs.length; b++) {
      var db = defs[b];
      if (db.affiliation === defs[a].affiliation) continue;
      var pb = pos(db);
      if (!pb) continue;
      if (distKm(pa, pb) <= (Number(defs[a].engage_km) || 0) + (Number(db.engage_km) || 0)) { caseEngage = true; break; }
    }
  }
  out.case_engage_ok = caseEngage;
  function countsFor(ld) {
    var c0 = pos(ld), engN = 0, infN = 0;
    if (!c0) return { eng: 0, inf: 0 };
    defs.forEach(function (d) {
      if (d === ld || d.affiliation === ld.affiliation) return;
      var p = pos(d);
      if (!p) return;
      var dk = distKm(c0, p);
      if (dk <= (Number(ld.engage_km) || 0) + (Number(d.engage_km) || 0)) engN++;
      if (dk <= (Number(ld.influence_km) || 0) + (Number(d.influence_km) || 0)) infN++;
    });
    return { eng: engN, inf: infN };
  }
  // Pick the lead with the most hostile associations under the case criterion, so
  // the hostile-line path is actually exercised (a 0-line sample would make the
  // whole check vacuous).
  var crit0 = caseEngage ? 'engage' : 'influence';
  var best = leads[0], bestN = -1, perLead = [];
  leads.forEach(function (ld) {
    var c = countsFor(ld);
    perLead.push({ id: idOf(ld), formation: attrs(ld).formation_id, engage: c.eng, influence: c.inf });
    var n = crit0 === 'engage' ? c.eng : c.inf;
    if (n > bestN) { bestN = n; best = ld; }
  });
  var lead = best, leadId = idOf(lead), formation = attrs(lead).formation_id;
  out.lead_candidates = perLead;
  out.lead_id = leadId;
  out.formation = formation;
  out.defs_n = defs.length;
  out.members = defs.filter(function (d) { return attrs(d).formation_id === formation; }).length;
  out.peers_n = out.members - 1;
  var p0 = pos(lead);
  out.lead_pos = p0;
  var eng = [], inf = [];
  defs.forEach(function (d) {
    if (d === lead || d.affiliation === lead.affiliation) return;
    var p = pos(d);
    if (!p0 || !p) return;
    var dk = distKm(p0, p);
    if (dk <= (Number(lead.engage_km) || 0) + (Number(d.engage_km) || 0)) eng.push(idOf(d));
    if (dk <= (Number(lead.influence_km) || 0) + (Number(d.influence_km) || 0)) inf.push(idOf(d));
  });
  out.engage_n = eng.length;
  out.influence_n = inf.length;
  out.engage_ids = eng.slice(0, 8);
  out.influence_ids = inf.slice(0, 8);
  out.criterion = (caseEngage || inf.length === 0) ? 'engage' : 'influence';
  out.expected_hostile = out.criterion === 'engage' ? eng.length : inf.length;
  if (typeof H.selectTarget === 'function') H.selectTarget(null);
  await new Promise(function (r) { setTimeout(r, 800); });
  var u = H.relationsDebug();
  out.unselected = { selected: u.selected, ownLines: u.ownLines, hostileLines: u.hostileLines,
                     total: u.total, formationMembers: u.formationMembers };
  if (typeof H.selectTarget === 'function') H.selectTarget(leadId);
  for (var k = 0; k < 32; k++) {
    await new Promise(function (r) { setTimeout(r, 250); });
    var probeNow = H.relationsDebug();
    if (probeNow && probeNow.selected === leadId && probeNow.total === probeNow.ownLines + probeNow.hostileLines) break;
  }
  var s2 = H.relationsDebug();
  out.selected = { selected: s2.selected, ownLines: s2.ownLines, hostileLines: s2.hostileLines,
                   total: s2.total, formationMembers: s2.formationMembers, activityCodes: s2.activityCodes,
                   activityVocabulary: s2.activityVocabulary, lineLabelOk: s2.lineLabelOk,
                   lineLabelCount: s2.lineLabelCount, linesVisible: s2.linesVisible,
                   criterion: s2.criterion };
  if (typeof H.selectTarget === 'function') H.selectTarget(null);
  await new Promise(function (r) { setTimeout(r, 800); });
  var u2 = H.relationsDebug();
  out.after_deselect = { selected: u2.selected, total: u2.total };
  return out;
})()
"""

def ontology_codes(path):
    """Codes the ontology defines: leaf intents + parent categories + status layers."""
    with open(path, "r", encoding="utf-8") as fh:
        onto = json.load(fh)
    leaves, parents, status = set(), set(), set()
    for parent in onto.get("parent_categories") or []:
        if parent.get("code"):
            parents.add(parent["code"])
        for child in parent.get("children") or []:
            code = child.get("code") if isinstance(child, dict) else child
            if code:
                leaves.add(code)
    for st in onto.get("status_layers") or []:
        if st.get("code"):
            status.add(st["code"])
    return leaves | parents | status, (len(leaves), len(parents), len(status))



PROBE_OCCUPATION = r"""
(async function () {
  var t0 = Date.now();
  while (Date.now() - t0 < 40000 && !(window.SituationHost &&
         SituationHost.state.data && SituationHost.state.data.case_id)) {
    await new Promise(function (r) { setTimeout(r, 500); });
  }
  await new Promise(function (r) { setTimeout(r, 3000); });
  var H = window.SituationHost;
  if (!H) return { error: 'window.SituationHost missing' };
  if (typeof H.occupationDebug !== 'function') return { error: 'occupationDebug missing' };
  var S = H.state, out = {};
  function snap(tag) {
    var d = H.occupationDebug();
    out[tag] = { tick: d.tick, statuses: d.statuses, counts: d.counts,
                 maskVertices: d.maskVertices, maskTick: d.maskTick };
    return d;
  }
  // Clear every plotting record first: the occupation layer MUST keep moving
  // without any user plotting, which is the decoupling this check pins down.
  var cleared = null;
  try {
    if (window.Plotting && typeof Plotting.clearLayer === 'function') {
      Plotting.clearLayer();
      cleared = true;
    } else {
      cleared = 'no Plotting.clearLayer';
    }
  } catch (e) { cleared = String(e); }
  out.plotting_cleared = cleared;
  var a = snap('after_clear');
  var tick0 = S.tick;
  // Compare more than the two endpoints: on the Taiwan case the map changes and
  // comes back (endpoints equal) before the first sustained change near tick 92.
  // So sample while advancing 120 ticks and keep every change that is seen.
  var prev = a.statuses || {};
  var total_changes = 0, max_diff = 0, first_change = null;
  S.playing = true;
  var t_adv = Date.now();
  while (Date.now() - t_adv < 30000 && S.tick < tick0 + 120) {
    await new Promise(function (r) { setTimeout(r, 120); });
    var cur = H.occupationDebug();
    var cs = cur.statuses || {}, diff = 0;
    Object.keys(cs).forEach(function (k) { if (prev[k] !== cs[k]) diff += 1; });
    if (diff > 0) {
      total_changes += diff;
      if (first_change === null) first_change = cur.tick;
      if (diff > max_diff) max_diff = diff;
      prev = cs;
    }
  }
  S.playing = false;
  out.advanced = S.tick - tick0;
  var b = snap('after_advance');
  var names = Object.keys(b.statuses || {}), endpoint = 0;
  names.forEach(function (k) {
    if ((a.statuses || {})[k] !== b.statuses[k]) endpoint += 1;
  });
  out.status_changes = total_changes;
  out.endpoint_changes = endpoint;
  out.max_diff = max_diff;
  out.first_change_tick = first_change;
  out.names_n = names.length;
  var c = b.counts || {};
  out.counts_sum = (c.red || 0) + (c.blue || 0) + (c.contested || 0) + (c.neutral || 0);
  out.non_neutral = (c.red || 0) + (c.blue || 0) + (c.contested || 0);
  out.mask_moved = (a.maskTick !== b.maskTick) || (a.maskVertices !== b.maskVertices);
  return out;
})()
"""


def probe(url: str, js: str = PROBE) -> dict:
    """Drive the headless browser and return the measurement dict."""
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "probe.json")
        # `inspect_remote.py` takes the expression as a **string** on --eval; it has no
        # --eval-file. Passing argv as a list avoids the shell entirely, which is what
        # makes a multi-line JS payload safe here: there is no quoting layer to escape.
        cmd = [sys.executable, INSPECT, "--url", url, "--out", out,
               "--wait-ms", "20000", "--eval-async", "--eval", js]
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

    print("8. OPT-3 -- screen-space decorations rebuild on zoom")
    sc = d.get("screen") or {}
    check("OPT-3: metersPerPixel finite and positive",
          sc.get("mpp_finite") is True, "got %r" % sc.get("mpp_finite"))
    ur = sc.get("uncapped_ratio")
    check("OPT-3: below the cap, ground size scales with resolution (x2 -> ~2x)",
          ur is not None and 1.8 <= ur <= 2.2, "ratio %r" % ur)
    got_px = sc.get("capped_px_at_2000")
    want_px = sc.get("capped_expected_px_at_2000")
    check("OPT-3: above the cap, screen size is held to ~5% of the run",
          got_px is not None and want_px and abs(got_px - want_px) / want_px < 0.15,
          "got %.3f px, expected %.3f px" % (got_px or -1, want_px or -1))
    check("OPT-3: no rebuild while inside the same zoom bucket",
          sc.get("no_rebuild_same_bucket") is True, "got %r" % sc.get("no_rebuild_same_bucket"))
    check("OPT-3: forced rebuild reports true",
          sc.get("forced_rebuild") is True, "got %r" % sc.get("forced_rebuild"))
    check("OPT-3: debug exposes bucket and rebuild count",
          sc.get("debug_has_bucket") is True, "got %r" % sc.get("debug_has_bucket"))
    idem = sc.get("idempotent") or {}
    check("OPT-3: rebuild is idempotent (entity count unchanged)",
          idem.get("before") is not None and idem.get("before") == idem.get("after"),
          "before=%r after=%r" % (idem.get("before"), idem.get("after")))

    print("9. OPT-1 -- multi-band range fan")
    fn = d.get("fan") or {}
    check("OPT-1: single band -> 1 ring", fn.get("single_rings") == 1,
          "got %r" % fn.get("single_rings"))
    check("OPT-1: single band spans centre..radius", fn.get("single_lo") is not None
          and fn.get("single_lo") < 100, "lo=%r" % fn.get("single_lo"))
    check("OPT-1: 3 bands -> 3 rings", fn.get("three_rings") == 3,
          "got %r" % fn.get("three_rings"))
    lo3, hi3 = fn.get("three_lo") or [], fn.get("three_hi") or []
    check("OPT-1: bands nest without gaps (lo[k] == hi[k-1])",
          len(lo3) == 3 and abs(lo3[1] - hi3[0]) / max(hi3[0], 1) < 1e-6
          and abs(lo3[2] - hi3[1]) / max(hi3[1], 1) < 1e-6,
          "lo=%s hi=%s" % (lo3, hi3))
    check("OPT-1: band radii increase outward",
          len(hi3) == 3 and hi3[0] < hi3[1] < hi3[2], "hi=%s" % hi3)
    check("OPT-1: circular -> one ring per band", fn.get("full_rings") == 2,
          "got %r" % fn.get("full_rings"))
    # ⚠ 不能用 `x or default`：闭合距离的正确值是 **0**，而 `0 or 1e9` 会取 1e9，
    # 于是"完全闭合"被报成失败。用 `is None` 显式判缺，不要靠真值判断。
    closed = fn.get("circ_closed_m")
    check("OPT-1: circular single band closes (first arc point == last)",
          closed is not None and closed < 1.0,
          "%s m" % ("None" if closed is None else "%.3f" % closed))
    check("OPT-1: zero/negative/NaN bands are filtered out",
          fn.get("bad_input_rings") == 1, "got %r" % fn.get("bad_input_rings"))

    print("10. OPT-2 -- airspace with altitude layers")
    al = d.get("alt") or {}
    nz = al.get("nofly") or {}
    pl = al.get("plain") or {}
    check("OPT-2: no_fly_zone has height/extrudedHeight",
          nz.get("has_height") is True and nz.get("height") == 0 and nz.get("extruded") == 10000,
          "height=%r extruded=%r" % (nz.get("height"), nz.get("extruded")))
    check("OPT-2: wall arrays match ring vertex count",
          nz.get("wall_lows") == nz.get("ring_pts") and nz.get("wall_highs") == nz.get("ring_pts"),
          "lows=%r highs=%r ring=%r" % (nz.get("wall_lows"), nz.get("wall_highs"), nz.get("ring_pts")))
    check("OPT-2: altitude label formatted from datum",
          nz.get("label") == "0FT AGL ~ 10000FT AGL", "got %r" % nz.get("label"))
    check("OPT-2: type without altLayer stays a ground polygon",
          pl.get("has_height") is False and pl.get("wall_lows") is None,
          "has_height=%r wall=%r" % (pl.get("has_height"), pl.get("wall_lows")))
    fmt = al.get("fmt") or {}
    check("OPT-2: formatAltitude MSL / FL / bare",
          fmt.get("msl") == "1500FT MSL" and fmt.get("fl") == "FL150" and fmt.get("bare") == "3000FT",
          "msl=%r fl=%r bare=%r" % (fmt.get("msl"), fmt.get("fl"), fmt.get("bare")))
    check("OPT-2: non-numeric altitudes pass through untouched",
          fmt.get("passthrough_num") == "FL150" and fmt.get("passthrough_txt") == "1500MSL"
          and fmt.get("blank") == "",
          "num=%r txt=%r blank=%r" % (fmt.get("passthrough_num"), fmt.get("passthrough_txt"), fmt.get("blank")))
    ill = al.get("illegal") or {}
    check("OPT-2: altLayer[1] <= altLayer[0] is rejected (no negative volume)",
          ill.get("has_volume") is False and ill.get("parsed") is None,
          "has_volume=%r parsed=%r" % (ill.get("has_volume"), ill.get("parsed")))

    print("11. OPT-4 -- vertex budget")
    bg = d.get("budget") or {}
    check("OPT-4: every ring respects the vertex cap",
          bg.get("many_max_verts") is not None and bg.get("many_max_verts") <= bg.get("cap"),
          "max=%r cap=%r" % (bg.get("many_max_verts"), bg.get("cap")))
    check("OPT-4: decimation actually triggers",
          bg.get("decimated_to") is not None and bg.get("decimated_to") < bg.get("decimated_from"),
          "%r -> %r" % (bg.get("decimated_from"), bg.get("decimated_to")))
    check("OPT-4: bounding box preserved bit-for-bit",
          bg.get("bbox_same") is True, "got %r" % bg.get("bbox_same"))

    print("12. FIX-5 -- array marks batched into primitives")
    bt = d.get("batch") or {}
    wt, wl, fl = bt.get("wire_tiny") or {}, bt.get("wire_long") or {}, bt.get("fill_long") or {}
    # 不变式：有图元 → 恰好 1 个批量图元；无图元 → 0 个。与 marks 数量无关。
    check("FIX-5: primitive count is 1 iff there are marks (never grows with marks)",
          (wt.get("primitives") == (1 if (wt.get("marks") or 0) > 0 else 0)) and
          wl.get("primitives") == 1 and (wl.get("marks") or 0) > 0,
          "tiny marks=%r prim=%r | long marks=%r prim=%r" % (
              wt.get("marks"), wt.get("primitives"), wl.get("marks"), wl.get("primitives")))
    check("FIX-5: marks below the visibility floor still leave the line drawn",
          (wt.get("marks") or 0) == 0 and (wt.get("entities") or 0) >= 1,
          "marks=%r entities=%r" % (wt.get("marks"), wt.get("entities")))
    check("FIX-5: batched primitive actually builds (ready)",
          wl.get("ready") == [True], "long=%r" % wl.get("ready"))
    check("FIX-5: filled marks (ditch teeth) also batch to 1 primitive",
          fl.get("primitives") == 1, "got %r" % fl.get("primitives"))
    check("FIX-5: removing a record leaks no scene primitives",
          wt.get("leaked") == 0 and wl.get("leaked") == 0 and fl.get("leaked") == 0,
          "tiny=%r long=%r fill=%r" % (wt.get("leaked"), wl.get("leaked"), fl.get("leaked")))
    check("FIX-5: marks no longer created as one entity each",
          (wl.get("entities") or 0) <= 2, "entities=%r marks=%r" % (wl.get("entities"), wl.get("marks")))
    # 7. Theatre filtering: how many port/airport hotspots the case produced.
    print()
    print("7. theatre facilities -- hotspot port/airport counts")
    raw_fac = probe(args.url, PROBE_FACILITY)
    if raw_fac.get("evalError"):
        print("probe error: %s" % raw_fac["evalError"])
        return 1
    fac = raw_fac.get("eval") or {}
    if fac.get("error"):
        print("probe returned: %s" % fac["error"])
        return 1
    print("  facility raw: %s" % json.dumps(fac, sort_keys=True))
    expect = EXPECT_FACILITIES.get(fac.get("case_id"))
    if expect is None:
        print("  no expectation for case %r -- numbers above are informational"
              % (fac.get("case_id"),))
    else:
        pmin, pmax, amin, tbox, exempt = expect
        port, airport = fac.get("port"), fac.get("airport")
        span = "%d" % pmin if pmax is None else "%d..%d" % (pmin, pmax)
        check("port count in %s" % span,
              isinstance(port, int) and port >= pmin and (pmax is None or port <= pmax),
              "port=%r" % (port,))
        check("airport count >= %d" % amin,
              isinstance(airport, int) and airport >= amin,
              "airport=%r" % (airport,))
        ak = json.loads(fac.get("anchor_kinds") or "{}")
        dk = json.loads(fac.get("data_kinds") or "{}")
        for kind in ("port", "airport"):
            check("anchors[%s] == data[%s]" % (kind, kind),
                  ak.get(kind, 0) == dk.get(kind, 0),
                  "anchors=%r data=%r" % (ak.get(kind), dk.get(kind)))
        extra = sorted(k for k in ak if k not in ("port", "airport"))
        check("no anchors for non-facility kinds", not extra, "extra=%s" % (extra,))
        # Facilities sit inside the theatre box; the only allowed exception is
        # HANEDA_NAME (`engine/scenario.py:102-103`: outside the box, covered by
        # its own imagery/elevation download). Checked in both directions - an
        # exempt name that is inside the box, or never shows up, fails too, so the
        # exemption cannot quietly turn this check into a no-op.
        bad, seen = [], set()
        for it in fac.get("facility_bboxes") or []:
            b = it.get("bbox")
            name = it.get("name")
            if not isinstance(b, list) or len(b) != 4:
                bad.append("%s %s bbox=%r" % (it.get("kind"), name, b))
                continue
            if (tbox[0] <= b[0] and b[1] >= tbox[1]
                    and b[2] <= tbox[2] and b[3] <= tbox[3]):
                if name in exempt:
                    bad.append("exempt but inside: %s %r" % (name, b))
            elif name in exempt:
                seen.add(name)
            else:
                bad.append("outside theatre: %s %s %r" % (it.get("kind"), name, b))
        missing = sorted(set(exempt) - seen)
        if missing:
            bad.append("exempt never out of bounds: %s" % (missing,))
        print("  exempt=%s out_of_box_seen=%s missing=%s"
              % (sorted(exempt), sorted(seen), missing))
        check("facility bboxes inside theatre box (or exempt)", not bad, "; ".join(bad[:3]))

    # 8. trajectory window / predicted leg / focus mask.
    print()
    print("8. trajectory window, predicted leg, focus mask")
    foc_raw = probe(args.url, PROBE_FOCUS)
    if foc_raw.get("evalError"):
        print("probe error: %s" % foc_raw["evalError"])
        return 1
    foc = foc_raw.get("eval") or {}
    if foc.get("error"):
        print("probe returned: %s" % foc["error"])
        return 1
    print("  focus raw: %s" % json.dumps(foc, sort_keys=True))
    base = foc.get("base") or {}
    sel = foc.get("selected") or {}
    desel = foc.get("unselected") or {}
    moff = foc.get("mask_unselected") or {}
    win = base.get("trailWindowTicks")
    check("sim ran past the window before reading (not a tick-0 sample)",
          foc.get("advance_ok") is True, "tick=%r" % (foc.get("tick_at_read"),))
    check("TRAIL_WINDOW_TICKS is 120", win == 120, "got %r" % (win,))
    check("drawn trail is clamped to the window (max == window)",
          isinstance(base.get("maxTrailVertices"), int) and base.get("maxTrailVertices") == win,
          "max=%r window=%r full=%r drawn=%r sum=%r tick=%r" % (
              base.get("maxTrailVertices"), win, base.get("trailFull"),
              base.get("trailDrawn"), base.get("trailVertexSum"), base.get("tick")))
    check("full trail is longer than the window (truncation is real)",
          max(base.get("trailFull") or 0, sel.get("trailFull") or 0) > (win or 0),
          "base_full=%r sel_full=%r window=%r" % (
              base.get("trailFull"), sel.get("trailFull"), win))
    check("windowed vertex sum fits targets * window",
          isinstance(base.get("trailVertexSum"), int) and isinstance(foc.get("targets_n"), int)
          and base["trailVertexSum"] <= foc["targets_n"] * (win or 0),
          "sum=%r targets=%r window=%r" % (
              base.get("trailVertexSum"), foc.get("targets_n"), win))
    check("selection settles within the poll budget (predictedPolylines==1)",
          foc.get("select_ok") is True,
          "polls=%r ms=%r last=%s" % (foc.get("select_polls"), foc.get("select_ms"),
                                      json.dumps(foc.get("selected_last"), sort_keys=True)))
    check("selecting a target yields exactly one predicted polyline",
          sel.get("predictedPolylines") == 1,
          "selected=%r polylines=%r" % (sel.get("selected"), sel.get("predictedPolylines")))
    check("history line stays solid while the predicted leg is separate",
          sel.get("historySolid") is True, "historySolid=%r" % (sel.get("historySolid"),))
    check("selected target shows its full trail (max == trailFull > window)",
          isinstance(sel.get("trailFull"), int) and sel.get("trailFull") > (win or 0)
          and sel.get("maxTrailVertices") == sel.get("trailFull"),
          "max=%r full=%r drawn=%r window=%r" % (
              sel.get("maxTrailVertices"), sel.get("trailFull"),
              sel.get("trailDrawn"), win))
    check("focus mask becomes visible while selected (polled, desired=true)",
          foc.get("mask_ok") is True,
          "polls=%r ms=%r last=%s" % (foc.get("mask_polls"), foc.get("mask_ms"),
                                      json.dumps(foc.get("selected_last"), sort_keys=True)))
    check("focus mask is visible while a target is selected",
          sel.get("maskExists") is True and sel.get("maskShow") is True,
          "maskExists=%r maskShow=%r" % (sel.get("maskExists"), sel.get("maskShow")))
    check("deselecting removes the predicted polyline",
          desel.get("predictedPolylines") == 0,
          "selected=%r polylines=%r" % (desel.get("selected"), desel.get("predictedPolylines")))
    check("focus mask hides after deselect (polled)",
          foc.get("mask_hide_ok") is True,
          "polls=%r ms=%r last=%s" % (foc.get("mask_hide_polls"), foc.get("mask_hide_ms"),
                                      json.dumps(foc.get("unselected_last"), sort_keys=True)))
    check("focus mask is hidden while nothing is selected",
          moff.get("maskShow") is False and desel.get("maskShow") is False,
          "unselected=%r deselect=%r" % (moff.get("maskShow"), desel.get("maskShow")))
    # 9. formation pairs, recomputed from the served dataset (not writer-c's code).
    print()
    print("9. wingman/lead pairing in the dataset")
    from urllib.parse import parse_qs, urlparse
    case = (parse_qs(urlparse(args.url).query).get("case") or [None])[0] or "taiwan_landing"
    ds_path = os.path.join(ROOT, "web", "data", "situations", "%s.json" % case)
    with open(ds_path, "r", encoding="utf-8") as fh:
        ds = json.load(fh)
    targets = ds.get("targets") or []
    by_id = {}
    for t in targets:
        by_id[t.get("id") or t.get("target_id")] = t
    wingmen, leads, bad = [], [], []
    for t in targets:
        a = t.get("attributes") or {}
        tid = t.get("id") or t.get("target_id")
        if a.get("pair_role") == "lead":
            leads.append(tid)
        if a.get("pair_role") != "wingman":
            continue
        wingmen.append(tid)
        ref = a.get("paired_with")
        other = by_id.get(ref)
        if other is None:
            bad.append("%s->%s missing" % (tid, ref))
            continue
        oa = other.get("attributes") or {}
        if other.get("affiliation") != t.get("affiliation"):
            bad.append("%s(%s)->%s(%s) side" % (tid, t.get("affiliation"), ref, other.get("affiliation")))
        if oa.get("pair_role") != "lead":
            bad.append("%s->%s role=%r" % (tid, ref, oa.get("pair_role")))
    check("every wingman points at an existing same-side lead (n=%d)" % len(wingmen),
          len(wingmen) > 0 and len(leads) > 0 and not bad,
          "wingmen=%s leads=%s bad=%s" % (wingmen[:6], leads[:6], bad[:4]))
    print("  dataset %s: targets=%d wingmen=%d leads=%d" % (case, len(targets), len(wingmen), len(leads)))

    # 10. relation lines and activity codes (frontend relationsDebug contract).
    print()
    print("10. relation lines and activity codes")
    rel_raw = probe(args.url, PROBE_RELATIONS)
    if rel_raw.get("evalError"):
        print("probe error: %s" % rel_raw["evalError"])
        return 1
    rel = rel_raw.get("eval") or {}
    if rel.get("error"):
        print("probe returned: %s" % rel["error"])
        return 1
    print("  relations raw: %s" % json.dumps(rel, sort_keys=True))
    unsel = rel.get("unselected") or {}
    sel = rel.get("selected") or {}
    desel = rel.get("after_deselect") or {}
    check("nothing selected -> zero relation lines",
          unsel.get("selected") is None and unsel.get("total") == 0
          and unsel.get("ownLines") == 0 and unsel.get("hostileLines") == 0,
          "unselected=%s" % json.dumps(unsel, sort_keys=True))
    check("sim advanced past tick 240 before measuring",
          rel.get("advance_ok") is True, "tick=%r" % (rel.get("tick"),))
    check("the lead target is really selected",
          sel.get("selected") == rel.get("lead_id"),
          "selected=%r lead=%r" % (sel.get("selected"), rel.get("lead_id")))
    engage_n = rel.get("engage_n") or 0
    influence_n = rel.get("influence_n") or 0
    # §3.3 lists engage before influence, and "can actually shoot each other" is the
    # harder reading, so engage coverage is the criterion. Fallback (documented):
    # a case whose hostile pairs never intersect engage coverage falls back to
    # influence, or the feature would look unimplemented. The criterion is printed,
    # never implied, and cross-checked against the page's own `criterion` field.
    criterion = rel.get("criterion") or ("engage" if engage_n > 0 else "influence")
    expected_hostile = rel.get("expected_hostile")
    if not isinstance(expected_hostile, int):
        expected_hostile = engage_n if criterion == "engage" else influence_n
    print("  hostile criterion=%s expected=%d engage=%d influence=%d case_engage_ok=%r"
          % (criterion, expected_hostile, engage_n, influence_n, rel.get("case_engage_ok")))
    check("dataset has hostile coverage intersections (non-vacuous, criterion=%s)" % criterion,
          expected_hostile > 0,
          "engage_n=%r influence_n=%r case_engage_ok=%r" % (engage_n, influence_n, rel.get("case_engage_ok")))
    peers_n = rel.get("peers_n")
    fm = sel.get("formationMembers")
    fm_n = len(fm) if isinstance(fm, list) else fm
    check("formationMembers == dataset peers (formation members minus the lead itself)",
          fm_n == peers_n,
          "debug=%r dataset_peers=%r members_incl_self=%r formation=%r" % (
              fm_n, peers_n, rel.get("members"), rel.get("formation")))
    check("own lines == other formation members (one line per peer)",
          sel.get("ownLines") == peers_n,
          "ownLines=%r peers=%r members_incl_self=%r" % (sel.get("ownLines"), peers_n, rel.get("members")))
    check("hostile lines == %s coverage intersections" % criterion,
          sel.get("hostileLines") == expected_hostile,
          "hostileLines=%r criterion=%s expected=%r engage=%r influence=%r ids=%s" % (
              sel.get("hostileLines"), criterion, expected_hostile, engage_n, influence_n,
              json.dumps(rel.get("engage_ids") if criterion == "engage" else rel.get("influence_ids"))))
    check("page criterion matches the independently computed criterion",
          sel.get("criterion") == criterion,
          "page=%r computed=%r engage=%r influence=%r" % (sel.get("criterion"), criterion, engage_n, influence_n))
    check("line labels are drawn for every relation line",
          sel.get("lineLabelOk") is True and sel.get("lineLabelCount") == sel.get("total"),
          "lineLabelOk=%r lineLabelCount=%r total=%r" % (
              sel.get("lineLabelOk"), sel.get("lineLabelCount"), sel.get("total")))
    check("relation lines are really visible in the scene",
          isinstance(sel.get("linesVisible"), int) and sel.get("linesVisible") > 0
          and sel.get("linesVisible") == sel.get("total"),
          "linesVisible=%r total=%r lineLabelCount=%r" % (
              sel.get("linesVisible"), sel.get("total"), sel.get("lineLabelCount")))
    check("total == own + hostile",
          isinstance(sel.get("total"), int)
          and sel.get("total") == (sel.get("ownLines") or 0) + (sel.get("hostileLines") or 0),
          "total=%r own=%r hostile=%r" % (sel.get("total"), sel.get("ownLines"), sel.get("hostileLines")))
    check("deselect clears the relation lines",
          desel.get("total") == 0 and desel.get("selected") is None,
          "after=%s" % json.dumps(desel, sort_keys=True))
    codes = sel.get("activityCodes") or []
    flat = [c.get("code") if isinstance(c, dict) else c for c in codes]
    allowed, counts = ontology_codes(os.path.join(ROOT, "engine", "data", "intent_ontology_v2.json"))
    unknown = sorted(set(x for x in flat if x not in allowed))
    if not sel.get("activityVocabulary"):
        # writer-c is switching activityCodes from the static vocabulary to the
        # codes actually on screen; asserting against the static list would be a
        # fake pass (the plan's own warning), so it stays off until the new field
        # lands and this branch flips by itself.
        print("  activity codes: %s (assertion deferred until activityVocabulary lands)" % (flat,))
    else:
        check("displayed activity codes are all defined in the ontology",
              bool(flat) and not unknown,
              "n=%d unknown=%s codes=%s vocab=%s (leaves=%d parents=%d status=%d)" % (
                  len(flat), unknown, flat[:10], sel.get("activityVocabulary"),
                  counts[0], counts[1], counts[2]))
    # 11. occupation (direction 2): plotting-decoupled movement + minimap mask.
    print()
    print("11. occupation statuses and minimap mask")
    occ_raw = probe(args.url, PROBE_OCCUPATION)
    if occ_raw.get("evalError"):
        print("probe error: %s" % occ_raw["evalError"])
        return 1
    occ = occ_raw.get("eval") or {}
    if occ.get("error"):
        print("probe returned: %s" % occ["error"])
        return 1
    print("  occupation raw: %s" % json.dumps(occ, sort_keys=True))
    before = occ.get("after_clear") or {}
    after = occ.get("after_advance") or {}
    st = after.get("statuses") or {}
    counts = after.get("counts") or {}
    four = {"red", "blue", "contested", "neutral"}
    check("plotting was cleared before the occupation sample",
          occ.get("plotting_cleared") is True, "cleared=%r" % (occ.get("plotting_cleared"),))
    check("statuses is a non-empty map keyed by hotspot name",
          isinstance(st, dict) and len(st) > 0 and occ.get("names_n") == len(st),
          "keys=%r names_n=%r" % (len(st), occ.get("names_n")))
    check("statuses values within the four states",
          set(st.values()) <= four, "extra=%s" % sorted(set(st.values()) - four))
    hist = {}
    for v in st.values():
        hist[v] = hist.get(v, 0) + 1
    check("counts match the status histogram",
          all((counts.get(k) or 0) == hist.get(k, 0) for k in four)
          and sum(counts.get(k) or 0 for k in four) == len(st),
          "counts=%s histogram=%s" % (json.dumps(counts, sort_keys=True), json.dumps(hist, sort_keys=True)))
    ds_occ = None
    for f in (ds.get("frames") or []):
        if f.get("tick") == after.get("tick"):
            ds_occ = f.get("occupation")
            break
    if isinstance(ds_occ, dict):
        diff = sorted(k for k in set(ds_occ) | set(st) if ds_occ.get(k) != st.get(k))
        check("frontend statuses == dataset occupation at the same tick",
              not diff, "tick=%r differing=%s dataset=%s page=%s" % (
                  after.get("tick"), diff[:4],
                  json.dumps(dict((k, ds_occ.get(k)) for k in diff[:4]), sort_keys=True),
                  json.dumps(dict((k, st.get(k)) for k in diff[:4]), sort_keys=True)))
    else:
        check("dataset has an occupation map at the page tick",
              False, "tick=%r frames=%d" % (after.get("tick"), len(ds.get("frames") or [])))
    check("occupation keeps moving after every plotting record is cleared",
          (occ.get("status_changes") or 0) > 0 and (occ.get("advanced") or 0) >= 120,
          "status_changes=%r advanced=%r first_change_tick=%r max_diff=%r endpoint_changes=%r" % (
              occ.get("status_changes"), occ.get("advanced"), occ.get("first_change_tick"),
              occ.get("max_diff"), occ.get("endpoint_changes")))
    check("counts are not all zero after the tick advance",
          (occ.get("counts_sum") or 0) == len(st) and (occ.get("non_neutral") or 0) > 0,
          "counts_sum=%r names=%r non_neutral=%r" % (occ.get("counts_sum"), len(st), occ.get("non_neutral")))
    check("minimap mask has vertices",
          isinstance(before.get("maskVertices"), int) and before.get("maskVertices") > 0
          and isinstance(after.get("maskVertices"), int) and after.get("maskVertices") > 0,
          "before=%r after=%r" % (before.get("maskVertices"), after.get("maskVertices")))
    check("minimap mask moves with the tick",
          occ.get("mask_moved") is True,
          "maskTick=%r->%r maskVertices=%r->%r" % (
              before.get("maskTick"), after.get("maskTick"),
              before.get("maskVertices"), after.get("maskVertices")))
    print()
    if FAILURES:
        print("FAILED (%d): %s" % (len(FAILURES), "; ".join(FAILURES)))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())