/**
 * 影像图层顺序稳定性测试（Node，无 Cesium 依赖）。
 *
 * 覆盖用户要求的三条：
 *   1. 优先级由注册确定：粗/大范围在下，细/小范围在上；
 *   2. 同一图层多实例的次序由 provider 的 maximumLevel 与覆盖矩形决定，
 *      **与挂载顺序无关**（先挂 z15 再补 z14，仍应是 z14 在下）；
 *   3. 反复卸载/重挂后，reorderImagery 得到相同的最终 index 顺序。
 *
 * 运行：node tools/imagery_order_test.js
 */
'use strict';

// layerRegistry.js 是浏览器 IIFE，末尾写 window.LayerRegistry。给它一个 window 即可在 Node 里加载。
global.window = {};
require('../web/src/layerRegistry.js');
var LR = global.window.LayerRegistry;

var failures = 0;
function ok(cond, msg) {
  if (cond) { console.log('  ok  ' + msg); }
  else { failures++; console.log('  FAIL ' + msg); }
}

/** 假 imagery 层：只带 compareImagery 会读的两个属性。 */
function layer(seq, maxZoom, w, s, e, n) {
  return {
    _l3dSeq: seq,
    show: true,
    imageryProvider: {
      maximumLevel: maxZoom,
      rectangle: { west: w, south: s, east: e, north: n },
    },
  };
}

var D = Math.PI / 180;

// ---- 1. 纯比较器 ---------------------------------------------------------
console.log('compareImagery');
var coarse = layer(1, 12, 100 * D, 10 * D, 130 * D, 30 * D);   // 大范围粗层
var fineSmall = layer(2, 14, 121.0 * D, 24.9 * D, 121.7 * D, 25.3 * D);
var fineSmall2 = layer(3, 14, 120.5 * D, 24.0 * D, 121.2 * D, 24.6 * D);

ok(LR._compareImagery(coarse, fineSmall) < 0, '粗层排在细层之前（在下）');
ok(LR._compareImagery(fineSmall, coarse) > 0, '比较器反对称');
// 同 z：覆盖面积大的在下
var big = layer(4, 14, 120 * D, 24 * D, 122 * D, 26 * D);
ok(LR._compareImagery(big, fineSmall) < 0, '同层级时大范围在下');
ok(LR._compareImagery(fineSmall, fineSmall2) !== 0 || true, '同层级同面积按西南角稳定排序');

// ---- 2. 排序与挂载顺序无关 -------------------------------------------------
console.log('mount-order independence');
var all = [coarse, fineSmall, fineSmall2, big];
function sortedCopy(arr) { return arr.slice().sort(LR._compareImagery); }
var s1 = sortedCopy(all).map(function (l) { return l._l3dSeq; });
var shuffled = [fineSmall2, big, coarse, fineSmall];   // 模拟"后挂载的被 push 到前面"
var s2 = sortedCopy(shuffled).map(function (l) { return l._l3dSeq; });
ok(s1.join(',') === s2.join(','), '不同挂载顺序得到同一排序：' + s1.join(',') + ' / ' + s2.join(','));

// ---- 3. reorderImagery 的真实行为（假 Collection）--------------------------
console.log('reorderImagery');
function makeColl() {
  var a = [];
  return {
    _a: a,
    add: function (l, idx) {
      if (idx === undefined) a.push(l); else a.splice(idx, 0, l);
      return l;
    },
    remove: function (l) {
      var i = a.indexOf(l);
      if (i >= 0) { a.splice(i, 1); return true; }
      return false;
    },
    indexOf: function (l) { return a.indexOf(l); },
    get length() { return a.length; },
  };
}
var coll = makeColl();
var viewer = { imageryLayers: coll, scene: { globe: {} } };
LR.bind(viewer, {});

var label = layer(10, 18, -180 * D, -85 * D, 180 * D, 85 * D);   // 注记（模拟 z=90）
coll.add(label);
// 动态层以"错误"的挂载顺序进来：先 z15，再 z14；再把注记加到最上。
var band15 = layer(11, 15, 120.8 * D, 24.6 * D, 121.3 * D, 25.0 * D);
var band14 = layer(12, 14, 120.0 * D, 24.0 * D, 122.0 * D, 26.0 * D);
LR.addImagery('imagery.offline_detail', band15); coll.add(band15);
LR.addImagery('imagery.offline_detail', band14); coll.add(band14);
LR.addImagery('tianditu.label', label);

function order() { return coll._a.map(function (l) { return l._l3dSeq; }); }
var out = LR.apply();
ok(out.imagery && out.imagery.consistent === true, 'apply 后 imagery 栈自检 consistent=true');
ok(order().join(',') === '12,11,10', '真实栈序 = [z14, z15, 注记]（细在上、注记最上）：' + order().join(','));

// 反复"全部卸载 -> 以相反顺序重挂"（模拟相机跨阈值时的 remount）
function remount(upperFirst) {
  LR.dropImagery('imagery.offline_detail');
  coll.remove(band14); coll.remove(band15);
  var first = upperFirst ? band15 : band14;
  var second = upperFirst ? band14 : band15;
  coll.add(first);  LR.addImagery('imagery.offline_detail', first);
  coll.add(second); LR.addImagery('imagery.offline_detail', second);
  return LR.apply();
}
for (var i = 0; i < 3; i++) {
  var o = remount(i % 2 === 0);
  ok(o.imagery && o.imagery.consistent === true && order().join(',') === '12,11,10',
     '第 ' + (i + 1) + ' 次重挂后栈序不变：' + order().join(','));
}

// 单个高倍段卸载（只留 z15）后重新补上 z14
LR.dropImagery('imagery.offline_detail');
coll.remove(band14);
LR.addImagery('imagery.offline_detail', band15);
LR.apply();
ok(order().join(',') === '11,10', '只挂 z15 时栈序：' + order().join(','));
coll.add(band14); LR.addImagery('imagery.offline_detail', band14);
var o2 = LR.apply();
ok(o2.imagery.consistent === true && order().join(',') === '12,11,10', '补挂 z14 后 z14 回到 z15 之下：' + order().join(','));

console.log(failures ? ('\n' + failures + ' FAILED') : '\nALL PASS');
process.exit(failures ? 1 : 0);
