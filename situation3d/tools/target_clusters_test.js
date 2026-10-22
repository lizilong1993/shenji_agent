#!/usr/bin/env node
/* Node 用例：编队聚合纯函数（web/src/targetClusters.js 的 computeClusters）。
 *
 * 不加载 Cesium：只验证"哪些目标应当聚、何时应当拆、id 是否稳定"这几条判据。
 * 运行：node tools/target_clusters_test.js
 */
'use strict';

const path = require('path');
const T = require(path.join(__dirname, '..', 'web', 'src', 'targetClusters.js'));

let failures = 0, checks = 0;
function ok(cond, msg) {
  checks++;
  if (!cond) { failures++; console.error('FAIL: ' + msg); }
}
function ids(clusters) {
  return clusters.map(function (c) {
    return c.affiliation + ':' + c.members.slice().sort().join(',');
  }).sort();
}

function item(id, aff, x, y) { return { id: id, affiliation: aff, x: x, y: y }; }

// 1. 同阵营、屏幕距离 10 px -> 聚合
{
  const out = T.computeClusters([
    item('a', 'red', 100, 100), item('b', 'red', 110, 100),
  ], []);
  ok(out.length === 1 && out[0].count === 2, '同阵营 10px 应聚成 1 簇');
}

// 2. 同阵营、40 px（> joinPx 38）且无历史 -> 不聚合
{
  const out = T.computeClusters([
    item('a', 'red', 100, 100), item('b', 'red', 140, 100),
  ], []);
  ok(out.length === 0, '同阵营 40px 不应主动聚合');
}

// 3. 不同阵营即便 5 px 也不混聚
{
  const out = T.computeClusters([
    item('a', 'red', 100, 100), item('b', 'blue', 105, 100),
  ], []);
  ok(out.length === 0, '红蓝目标不得混合聚合');
}

// 4. 滞回：上一轮已聚合，当前簇半径 30（join 38 以内、split 46 以内）-> 保持
{
  const prev = [{ id: 'k1', affiliation: 'red', members: ['a', 'b'] }];
  const out = T.computeClusters([
    item('a', 'red', 100, 100), item('b', 'red', 160, 100),
  ], prev);
  ok(out.length === 1 && out[0].id === 'k1', '滞回区间内应保持聚合且沿用 id');
}

// 5. 超过 splitPx（46）-> 拆散
{
  const prev = [{ id: 'k1', affiliation: 'red', members: ['a', 'b'] }];
  const out = T.computeClusters([
    item('a', 'red', 100, 100), item('b', 'red', 300, 100),
  ], prev);
  ok(out.length === 0, '簇半径 100 > splitPx 应拆散');
}

// 6. 不串联：5 个等距 24 px 的目标不能聚成一整串
{
  const items = [];
  for (let i = 0; i < 5; i++) items.push(item('t' + i, 'red', 100 + i * 24, 100));
  const out = T.computeClusters(items, []);
  ok(out.length >= 2, '等距长链应拆成多簇（不得整串聚合）');
  out.forEach(function (c) {
    ok(c.spread <= T.DEFAULTS.maxSpreadPx + 0.01, '簇内 spread 超限：' + c.spread);
  });
}

// 7. id 稳定：成员小幅移动后沿用上一轮 id
{
  const prev = [{ id: 'k9', affiliation: 'blue', members: ['x', 'y', 'z'] }];
  const out = T.computeClusters([
    item('x', 'blue', 200, 200), item('y', 'blue', 210, 205), item('z', 'blue', 205, 214),
  ], prev);
  ok(out.length === 1 && out[0].id === 'k9', '成员小幅移动应沿用簇 id');
}

// 8. 新簇 id 唯一且来自 leader
{
  const out = T.computeClusters([
    item('b', 'red', 300, 300), item('a', 'red', 306, 300),
    item('d', 'red', 600, 300), item('c', 'red', 607, 300),
  ], []);
  ok(out.length === 2, '两组远距离目标应形成 2 簇');
  const all = out.map(function (c) { return c.id; });
  ok(all.length === new Set(all).size, '簇 id 必须唯一');
}

// 9. 缩放：同一编队按 3.5 倍拉伸到屏幕后应拆散
{
  const prev = [{ id: 'k2', affiliation: 'red', members: ['a', 'b', 'c'] }];
  const base = [item('a', 'red', 0, 0), item('b', 'red', 20, 0), item('c', 'red', 0, 20)];
  const zoomed = base.map(function (it) {
    return item(it.id, it.affiliation, it.x * 3.5, it.y * 3.5);
  });
  const out = T.computeClusters(zoomed, prev);
  ok(out.length === 0, '放大 3.5 倍后应恢复独立图标');
}

// 10. 单目标永远不返回
{
  const out = T.computeClusters([item('solo', 'red', 50, 50)], []);
  ok(out.length === 0, '单目标不应生成编队簇');
}

// 11. maxSpread 阻止"两个簇合并后摊得过开"
{
  const prev = [
    { id: 'kA', affiliation: 'red', members: ['a', 'b'] },
    { id: 'kB', affiliation: 'red', members: ['c', 'd'] },
  ];
  const out = T.computeClusters([
    item('a', 'red', 100, 100), item('b', 'red', 120, 100),
    item('c', 'red', 200, 100), item('d', 'red', 220, 100),
  ], prev);
  ok(out.length === 2, '相距 100px 的两簇不得合并');
}

// 12. 红蓝编队图标重叠时按屏幕位移分开（不改变阵营、不合并）
{
  const list = [
    { id: 'a', affiliation: 'red', x: 100, y: 100 },
    { id: 'b', affiliation: 'blue', x: 108, y: 100 },
    { id: 'c', affiliation: 'blue', x: 300, y: 100 },
  ];
  T.separateClusters(list, { w: 46, h: 46, maxOffset: 30 });
  const dx = list[1].x - list[0].x, dy = list[1].y - list[0].y;
  ok(Math.abs(dx) >= 45.9 || Math.abs(dy) >= 45.9, '红蓝编队矩形应被推到不重叠');
  ok(list[2].x === 300 && list[2].y === 100, '互不重叠的编队不应移动');
  list.forEach(function (c) {
    ok(Math.abs(c.dx) <= 30 && Math.abs(c.dy) <= 30, '位移不得超过 maxOffset');
  });
}

console.log((failures ? 'FAIL ' : 'OK ') + (checks - failures) + '/' + checks + ' checks');
process.exit(failures ? 1 : 0);
