#!/usr/bin/env node
/**
 * health_layout_test.js — 血条避让的纯函数行为。
 *
 * 锁住三件事：
 *   1. 不重叠：任何两个放下的血条矩形都必须不相交；
 *   2. 就近：能用默认位就用默认位，只有被占才让开，且代价越大越靠后；
 *   3. 宁缺毋滥：推到 maxOffsetPx 之外或出视口就不画，不能画到别人身上。
 */
'use strict';

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

function bars(n, x, y, stepX) {
  const out = [];
  for (let i = 0; i < n; i++) {
    out.push({ id: 'T' + i, anchorX: x + i * stepX - 17, anchorY: y + 17, w: 34, h: 4, priority: 1 });
  }
  return out;
}

console.log('1. 单个血条落在锚点正下方');
(function () {
  const out = HL.layout(bars(1, 400, 300, 0), VIEW, []);
  check('放下 1 个', out.length === 1);
  const r = out[0];
  check('水平居中', Math.abs((r.x + r.w / 2) - 400) <= 1, 'x=' + r.x);
  check('落在锚点下方 gapPx 处', r.y === 300 + HL.DEFAULT_OPTIONS.gapPx, 'y=' + r.y);
})();

console.log('2. 同位置三个血条不重叠');
(function () {
  const same = [
    { id: 'A', anchorX: 400 - 17, anchorY: 317, w: 34, h: 4, priority: 3 },
    { id: 'B', anchorX: 400 - 17, anchorY: 317, w: 34, h: 4, priority: 2 },
    { id: 'C', anchorX: 400 - 17, anchorY: 317, w: 34, h: 4, priority: 1 },
  ];
  const out = HL.layout(same, VIEW, []);
  check('三个都放得下', out.length === 3, 'got ' + out.length);
  let overlap = 0;
  for (let i = 0; i < out.length; i++) {
    for (let j = i + 1; j < out.length; j++) {
      if (HL.rectsOverlap(out[i], out[j], 0)) overlap++;
    }
  }
  check('两两不重叠', overlap === 0, 'overlap=' + overlap);
  check('高优先级的拿到默认位', out[0].id === 'A' && out[0].y === 317, out[0].id + '@' + out[0].y);
})();

console.log('3. 密集一排：按优先级依次让开，且不重叠');
(function () {
  const dense = bars(12, 300, 300, 6);     // 间隔 6 px，血条宽 34 —— 必然互相压
  const out = HL.layout(dense, VIEW, []);
  check('至少放下 6 个', out.length >= 6, 'got ' + out.length);
  let overlap = 0;
  for (let i = 0; i < out.length; i++) {
    for (let j = i + 1; j < out.length; j++) {
      if (HL.rectsOverlap(out[i], out[j], 0)) overlap++;
    }
  }
  check('两两不重叠', overlap === 0, 'overlap=' + overlap);
})();

console.log('4. 宁可没有也不画到别人身上');
(function () {
  // 所有候选位都被障碍物占满 -> 一个都不该放
  const blocked = [];
  const o = HL.DEFAULT_OPTIONS;
  for (let row = 0; row <= o.rows; row++) {
    for (let col = -o.cols; col <= o.cols; col++) {
      blocked.push({ x: 400 - 17 + col * o.stepX, y: 317 + row * o.stepY, w: 34, h: 4 });
    }
  }
  const out = HL.layout(bars(1, 400, 300, 0), VIEW, blocked);
  check('全被占时不放', out.length === 0, 'got ' + out.length);
})();

console.log('5. 出视口的丢弃');
(function () {
  const near = [{ id: 'edge', anchorX: 4 - 17, anchorY: 790 + 17, w: 34, h: 4, priority: 1 }];
  const out = HL.layout(near, { w: 1280, h: 800 }, []);
  check('贴底边的锚点不画出界', out.length === 0, 'got ' + out.length);
})();

console.log('6. 同一输入同一输出（可复现）');
(function () {
  const input = bars(8, 500, 320, 10);
  const a = JSON.stringify(HL.layout(input, VIEW, []));
  const b = JSON.stringify(HL.layout(input, VIEW, []));
  check('两次结果逐字节相同', a === b);
})();

console.log('7. 优先级高的先拿好位置');
(function () {
  const input = [
    { id: 'low', anchorX: 400 - 17, anchorY: 317, w: 34, h: 4, priority: 1 },
    { id: 'high', anchorX: 400 - 17, anchorY: 317, w: 34, h: 4, priority: 5 },
  ];
  const out = HL.layout(input, VIEW, []);
  check('high 排第一', out[0].id === 'high', out[0].id);
  check('high 拿到默认位', out[0].y === 317, 'y=' + out[0].y);
  check('low 被迫下移', out[1].y > out[0].y, out[1].y + ' vs ' + out[0].y);
})();

console.log('');
if (failures) { console.log('FAILED: ' + failures + ' 项'); process.exit(1); }
console.log('PASSED');
