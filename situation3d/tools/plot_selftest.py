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

    print()
    if FAILURES:
        print("FAILED (%d): %s" % (len(FAILURES), "; ".join(FAILURES)))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())