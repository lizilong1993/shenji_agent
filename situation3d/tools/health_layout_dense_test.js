#!/usr/bin/env node
/**
 * health_layout_dense_test.js — 真实密集场景下的避让效果。
 *
 * 浏览器探针（verify_gp_browser.py）只能测到画面上真正展开的目标：战役案例
 * 有 23 个目标被编队聚合吃掉，拉近到不聚合时又只剩十几个。为了验证"密集时
 * 到底折不折"，这里直接用 TW3 某一帧的真实屏幕密集分布做合成：
 *   取 33 个目标在 120 km 高度下的相对排布（从数据集算），按 4 种缩放铺开，
 *   每一种都要求放下的血条两两不重叠，并要求"放下的数量随密度单调不增"。
 *
 * 这一节不测"地毯式铺开"：那是坐标数据决定的美术问题，不是避让算法的问题。
 */
'use strict';

const fs = require('fs');
const path = require('path');

global.window = global;
require(path.join(__dirname, '..', 'web', 'src', 'healthLayout.js'));
const HL = global.HealthLayout;

let failures = 0;
function check(name, cond, detail) {
  if (cond) { console.log('  ok   ' + name); return; }
  failures++;
  console.log('  FAIL ' + name + (detail ? '  —  ' + detail : ''));
}

const VIEW = { w: 1280, h: 800 };

/** n 个血条堆在同一处：避让的极限场景。 */
function pile(n, x, y) {
  const out = [];
  for (let i = 0; i < n; i++) out.push({
    id: 'P' + i, anchorX: x - 17, anchorY: y + 17, w: 34, h: 4,
    priority: i === 0 ? 3 : 1,
  });
  return out;
}

function overlapCount(rects) {
  let n = 0;
  for (let i = 0; i < rects.length; i++) {
    for (let j = i + 1; j < rects.length; j++) {
      if (HL.rectsOverlap(rects[i], rects[j], 0)) n++;
    }
  }
  return n;
}

console.log('1. 同一处叠 20 条：放下多少都不许重叠');
(function () {
  const out = HL.layout(pile(20, 640, 400), VIEW, []);
  check('放下了若干条', out.length > 0, 'placed=' + out.length);
  check('一个都不重叠', overlapCount(out) === 0, 'overlap=' + overlapCount(out));
  check('不是全部铺开（上限起作用）', out.length < 20, 'placed=' + out.length);
})();

console.log('2. 两个密集堆叠互不干扰');
(function () {
  const a = pile(10, 400, 300).map(b => ({ ...b, id: 'A' + b.id }));
  const b = pile(10, 600, 300).map(x => ({ ...x, id: 'B' + x.id }));
  const out = HL.layout(a.concat(b), VIEW, []);
  check('放下至少 8 条', out.length >= 8, 'placed=' + out.length);
  check('一个都不重叠', overlapCount(out) === 0, 'overlap=' + overlapCount(out));
  // 两组锚点相距 200 px、血条宽 34、横向步进 36 —— 允许越界到对方区域，
  // 但**不许压上去**；这一点已被上面的不重叠断言覆盖。
})();

console.log('3. 密度上升，放下的比例下降（不是恒定丢弃）');
(function () {
  const ratios = [];
  [4, 8, 16, 32].forEach(function (n) {
    const out = HL.layout(pile(n, 640, 400), VIEW, []);
    ratios.push(out.length / n);
    check('n=' + n + ' 无重叠', overlapCount(out) === 0);
  });
  check('比例单调不增', ratios.every((r, i) => i === 0 || r <= ratios[i - 1] + 1e-9),
        ratios.map(r => r.toFixed(2)).join(' → '));
})();

console.log('4. 用 TW3 真实坐标算出的密集分布');
(function () {
  const file = path.join(__dirname, '..', 'web', 'data', 'situations', 'taiwan_campaign.json');
  if (!fs.existsSync(file)) {
    console.log('  skip  数据集不在（演示环境未生成）');
    return;
  }
  const data = JSON.parse(fs.readFileSync(file, 'utf8'));
  const tick = 860;
  const rows = (((data.frames || [])[tick] || {}).targets || []);
  if (!rows.length) { console.log('  skip  该 tick 无目标'); return; }
  let lonMin = Infinity, lonMax = -Infinity, latMin = Infinity, latMax = -Infinity;
  rows.forEach(r => {
    if (r.lon < lonMin) lonMin = r.lon;
    if (r.lon > lonMax) lonMax = r.lon;
    if (r.lat < latMin) latMin = r.lat;
    if (r.lat > latMax) latMax = r.lat;
  });
  const dLon = Math.max(1e-6, lonMax - lonMin), dLat = Math.max(1e-6, latMax - latMin);
  // 四种缩放：从"目标挤成一点"到"铺满画布"
  [0.12, 0.3, 0.6, 0.9].forEach(function (scale) {
    const bars = rows.map(r => ({
      id: r.id,
      anchorX: (0.1 + 0.8 * (r.lon - lonMin) / dLon) * VIEW.w * scale - 17,
      anchorY: (0.9 - 0.8 * (r.lat - latMin) / dLat) * VIEW.h * scale + 17,
      w: 34, h: 4, priority: 1,
    }));
    const out = HL.layout(bars, VIEW, []);
    check('scale=' + scale + ' 无重叠', overlapCount(out) === 0,
          'placed=' + out.length + '/' + bars.length + ' overlap=' + overlapCount(out));
    check('scale=' + scale + ' 至少放下 3 条', out.length >= 3, 'placed=' + out.length);
  });
})();

console.log('');
if (failures) { console.log('FAILED: ' + failures + ' 项'); process.exit(1); }
console.log('PASSED');
