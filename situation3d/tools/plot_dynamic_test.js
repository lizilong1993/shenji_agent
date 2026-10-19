#!/usr/bin/env node
/**
 * 无浏览器冒烟测试：标绘的动态图层 + 目标轨迹标绘。
 *
 * 为什么需要它
 * ------------
 * 这个仓库的前端验证通常要起 Cesium 页面（`plot_selftest.py`）。但本轮改动的
 * 核心是**数据/图层逻辑**：图层集合由实际标绘要素驱动、目标轨迹改走
 * `target_track` 标绘。这些逻辑不依赖 WebGL，可以在 Node 里用最小 Cesium 桩
 * 和真实的 `plotting.js` / `layerRegistry.js` 验证；浏览器不可用时也能给出
 * 可判定的结论。
 *
 * 覆盖：
 *   1. 类别元数据与 target_track 的 system 标记；
 *   2. onChange 触发动态图层注册/注销；
 *   3. 未声明类别的自动成层；
 *   4. 目标轨迹按逐点高度渲染、标签为「态势标绘」；
 *   5. 系统标绘不导出、不可导入；
 *   6. 图层显隐 sink 对目标轨迹生效。
 *
 * 用法：node tools/plot_dynamic_test.js
 */
'use strict';

const fs = require('fs');
const path = require('path');

const ROOT = path.dirname(__dirname);
const SRC = path.join(ROOT, 'web', 'src');

let failures = 0;
function check(name, cond, detail) {
  console.log((cond ? '  ok   ' : '  FAIL ') + name + (detail && !cond ? '  ' + detail : ''));
  if (!cond) failures++;
}

// ---------------------------------------------------------------- Cesium 桩
function color(hex) {
  return {
    hex: hex,
    withAlpha: function (a) { return color(hex + '@' + a); },
  };
}
const Cesium = {
  Color: {
    fromCssColorString: color,
    WHITE: color('#ffffff'),
    fromCssColorString_: color,
  },
  Cartesian3: {
    fromDegrees: function (lon, lat, h) { return { lon: lon, lat: lat, h: h }; },
    fromDegreesArrayHeights: function (flat) { return flat.slice(); },
    fromDegreesArray: function (flat) { return flat.slice(); },
  },
  PolylineDashMaterialProperty: function (opts) { this.opts = opts; this.dashed = true; },
  LabelStyle: { FILL_AND_OUTLINE: 1 },
  VerticalOrigin: { BOTTOM: 0 },
  ClassificationType: { BOTH: 1 },
  PolygonHierarchy: function (h) { this.hierarchy = h; },
  Cartesian2: function (x, y) { this.x = x; this.y = y; },
  Math: { toRadians: function (d) { return d * Math.PI / 180; } },
  NearFarScalar: function () {},
};

let entitySeq = 0;
const viewer = {
  entities: {
    values: [],
    add: function (e) {
      const rec = Object.assign({ id: 'entity-' + (++entitySeq), show: true }, e);
      this.values.push(rec);
      return rec;
    },
    remove: function (e) {
      const i = this.values.indexOf(e);
      if (i >= 0) this.values.splice(i, 1);
    },
  },
  scene: {
    requestRender: function () {},
    primitives: { add: function (p) { return p; }, remove: function () {} },
  },
  camera: {
    changed: { addEventListener: function () {} },
    positionCartographic: { height: 100000 },
    frustum: { fovy: Math.PI / 3 },
  },
  canvas: { clientHeight: 800, height: 800 },
};

global.window = global;
global.Cesium = Cesium;

// ------------------------------------------------------- 加载真实模块（IIFE）
function load(file) {
  const code = fs.readFileSync(path.join(SRC, file), 'utf8');
  // 模块都是 (function(){...})()，直接 eval 到当前全局；window 指回 global
  (0, eval)(code);
}
load('plotting.js');
load('layerRegistry.js');

const Plotting = global.Plotting;
const LayerRegistry = global.LayerRegistry;
check('Plotting / LayerRegistry 已加载', !!Plotting && !!LayerRegistry);

Plotting.bind(viewer);
// 注册表的 plot sink：与 app.js 的接线一致
LayerRegistry.bind(viewer, {
  plot: function (id, want) { Plotting.setLayerVisible(id, !!want); },
});

// ------------------------------------------- 从 app.js 抽出真实的动态同步代码
const appSrc = fs.readFileSync(path.join(SRC, 'app.js'), 'utf8');
const startAt = appSrc.indexOf('var _plotLayerDesired = {};');
const endAt = appSrc.indexOf('/**\n * 注册各图层的**要素提供者**', startAt);
if (startAt < 0 || endAt < 0) {
  console.error('无法从 app.js 定位动态图层代码块');
  process.exit(1);
}
let refreshCalls = 0;
global.refreshPlotLayerSelect = function () { refreshCalls++; };
global.applyLayers = function () { LayerRegistry.apply(); };
global.LayerPanel = { invalidate: function () {} };
const dynamic = new Function(
  appSrc.slice(startAt, endAt) +
  '\nreturn { syncPlotLayers: syncPlotLayers, desired: _plotLayerDesired,' +
  ' plotLayerOptions: plotLayerOptions };'
)();
const syncPlotLayers = dynamic.syncPlotLayers;
Plotting.onChange(function () { syncPlotLayers(); });

function plotLayerIds() {
  return LayerRegistry.list('plot').filter(function (l) { return l.group === '标绘'; })
    .map(function (l) { return l.id; }).sort();
}

// ---------------------------------------------------------------- 1. 元数据
const cats = Plotting.categories();
const catIds = cats.map(function (c) { return c.id; });
const missingCats = Plotting.catalogue().map(function (c) { return c.category; })
  .filter(function (c) { return catIds.indexOf(c) < 0; });
check('categories 覆盖目录里的全部类别（且包含 target_track）',
  missingCats.length === 0 && catIds.indexOf('track') >= 0,
  'cats=' + catIds.join(',') + ' missing=' + missingCats.join(','));
const trackCat = cats.filter(function (c) { return c.id === 'track'; })[0];
check('目标轨迹类别存在且名称为「目标轨迹」',
  !!trackCat && trackCat.layer === 'plot.track' && trackCat.zh === '目标轨迹',
  JSON.stringify(trackCat));

// 需求 2 的文案：四个源码文件里不得再出现“预测/预计”
const banned = ['app.js', 'plotting.js', 'layerRegistry.js', 'situation.js'].filter(function (f) {
  return /预测|预计/.test(fs.readFileSync(path.join(SRC, f), 'utf8'));
});
check('源码中无“预测/预计”残留', banned.length === 0, banned.join(','));

// 手动标绘的归属图层选项不应包含系统专用类别（plot.track）
const layerOpts = dynamic.plotLayerOptions();
check('手动标绘图层选项不含系统类别，且包含手动类别',
  layerOpts.length > 0 &&
  layerOpts.every(function (o) { return o[0] !== 'plot.track'; }) &&
  layerOpts.some(function (o) { return o[0] === 'plot.fire'; }),
  JSON.stringify(layerOpts));

const cat = Plotting.catalogue();
const tt = cat.filter(function (c) { return c.type === 'target_track'; })[0];
check('catalogue 含 target_track（system=true, line 族, plot.track）',
  !!tt && tt.system === true && tt.family === 'line' && tt.defaultLayer === 'plot.track',
  JSON.stringify(tt));

// ------------------------------------------- 2. 动态图层：增 → 注册，删 → 注销
const A = [119.4, 24.2], B = [120.6, 25.0];
check('初始没有空的标绘类别图层', plotLayerIds().length === 0,
  JSON.stringify(plotLayerIds()));

const p1 = Plotting.add({ type: 'phase_line', layer: 'plot.fire', points: [A, B] });
const defFire = LayerRegistry.def('plot.fire');
check('添加要素后类别图层自动注册', !!defFire && defFire.label === '标绘·火力',
  JSON.stringify(defFire && { id: defFire.id, label: defFire.label, z: defFire.z }));
check('图层要素提供者返回该要素',
  LayerRegistry.elements('plot.fire').length === 1);

const p2 = Plotting.add({ type: 'coordination_line', layer: 'plot.fire', points: [A, B] });
check('同一类别第二个要素不产生第二个图层',
  plotLayerIds().filter(function (x) { return x === 'plot.fire'; }).length === 1 &&
  LayerRegistry.elements('plot.fire').length === 2);

const p3 = Plotting.add({ type: 'phase_line', layer: 'plot.custom_ext', points: [A, B] });
const defCustom = LayerRegistry.def('plot.custom_ext');
check('未声明的类别也能自动成层（不依赖固定分类）',
  !!defCustom && defCustom.label === '标绘·custom_ext',
  JSON.stringify(defCustom && defCustom.label));

Plotting.remove(p3.id);
Plotting.remove(p1.id);
Plotting.remove(p2.id);
check('最后一个要素删除后图层注销（恢复到空集合）',
  plotLayerIds().length === 0, JSON.stringify(plotLayerIds()));

// 关掉的类别在图层重建后应保持关（显隐意图跨注销保留）
dynamic.desired['plot.fire'] = false;
const p4 = Plotting.add({ type: 'phase_line', layer: 'plot.fire', points: [A, B] });
check('图层重建时恢复用户关掉的显隐意图',
  !!LayerRegistry.def('plot.fire') &&
  LayerRegistry.runtime['plot.fire'].desired === false,
  JSON.stringify(LayerRegistry.runtime['plot.fire'] && LayerRegistry.runtime['plot.fire'].desired));
Plotting.remove(p4.id);
delete dynamic.desired['plot.fire'];
check('清理后复归空集合', plotLayerIds().length === 0, JSON.stringify(plotLayerIds()));

// 场景被 removeAll 清空后，记录仍在；redrawAll 应把几何画回来（teardownScene 用）
const p5 = Plotting.add({ type: 'phase_line', layer: 'plot.fire', points: [A, B] });
viewer.entities.values.length = 0;
const redrawn = Plotting.redrawAll();
check('redrawAll 在场景清空后恢复标绘几何',
  redrawn === 1 && p5.entities.length === 1 && viewer.entities.values.length === 1,
  JSON.stringify({ redrawn: redrawn, entities: p5.entities.length,
                   viewer: viewer.entities.values.length }));
Plotting.remove(p5.id);

// ------------------------------------------- 3. 目标轨迹：走标绘、带高度、虚线
const trackPts = [[119.6, 24.3], [119.7, 24.4], [119.8, 24.5]];
const trackH = [1500, 2000, 2500];
const track = Plotting.add({
  type: 'target_track', layer: 'plot.track', points: trackPts, heights: trackH,
  hostility: 'friendly', label: '态势标绘',
});
const trackEntities = viewer.entities.values.filter(function (e) {
  return track.entities.indexOf(e) >= 0;
});
const trackLine = trackEntities.filter(function (e) { return e.polyline; })[0];
const trackLabel = trackEntities.filter(function (e) { return e.label; })[0];
check('目标轨迹建成 plot.track 图层', !!LayerRegistry.def('plot.track') &&
  LayerRegistry.def('plot.track').label === '标绘·目标轨迹');
check('目标轨迹是高程折线（clampToGround=false）',
  !!trackLine && trackLine.polyline.clampToGround === false,
  JSON.stringify(trackLine && trackLine.polyline && trackLine.polyline.clampToGround));
check('目标轨迹逐点高度原样进入几何',
  !!trackLine && trackLine.polyline.positions.length === trackPts.length * 3 &&
  trackLine.polyline.positions[2] === 1500 && trackLine.polyline.positions[5] === 2000,
  JSON.stringify(trackLine && trackLine.polyline.positions));
check('目标轨迹为虚线（计划语义）',
  !!trackLine && trackLine.polyline.material instanceof Cesium.PolylineDashMaterialProperty);
check('目标轨迹标签为「态势标绘」',
  !!trackLabel && trackLabel.label.text === '态势标绘',
  JSON.stringify(trackLabel && trackLabel.label.text));
check('目标轨迹标签锚在轨迹高度（不落到地面 0 米）',
  !!trackLabel && trackLabel.position.h === 2000,
  JSON.stringify(trackLabel && trackLabel.position));
check('目标轨迹要素在图层提供者里只读（system）',
  LayerRegistry.elements('plot.track').some(function (e) {
    return e.sub && e.sub.indexOf('系统生成') >= 0 &&
      JSON.stringify(e.actions) === '["locate"]';
  }));

// 高度与点数不符时退回贴地（不会错位）
const badTrack = Plotting.add({
  type: 'target_track', layer: 'plot.track', points: trackPts,
  heights: [100], hostility: 'friendly', label: '态势标绘',
});
const badLine = badTrack.entities.filter(function (e) { return e.polyline; })[0];
check('高度数量不符时退回贴地渲染', !!badLine && badLine.polyline.clampToGround === true);
Plotting.remove(badTrack.id);

// ------------------------------------------- 4. 导出/导入：系统标绘不落盘
const exported = Plotting.exportAll();
check('exportAll 不包含系统生成的目标轨迹',
  exported.every(function (p) { return p.type !== 'target_track'; }) &&
  exported.length === 0);
const rejected = Plotting.importAll([
  { type: 'target_track', layer: 'plot.track', points: trackPts },
]);
check('importAll 拒绝系统生成的标绘类型',
  rejected.added === 0 && rejected.errors.length === 1 &&
  rejected.errors[0].indexOf('系统生成') >= 0,
  JSON.stringify(rejected));

// ------------------------------------------- 5. 图层显隐 sink 对轨迹生效
LayerRegistry.setDesired('plot.track', false);
global.applyLayers();
check('关闭目标轨迹图层后，它的实体会隐藏',
  track.entities.every(function (e) { return e.show === false; }),
  JSON.stringify(track.entities.map(function (e) { return e.show; })));
LayerRegistry.setDesired('plot.track', true);
global.applyLayers();
check('重新打开后实体恢复显示',
  track.entities.every(function (e) { return e.show === true; }));

Plotting.remove(track.id);
check('移除轨迹后类别图层注销',
  !LayerRegistry.def('plot.track') && plotLayerIds().length === 0,
  JSON.stringify(plotLayerIds()));

// ------------------------------------------- 6. app.js 的轨迹标绘接线（真实代码）
//
// 从 app.js 抽出 trackPlotPoints / kinematicTrackPoints / syncTrackPlot 三个函数，
// 用桩状态验证：计划剩余段 → 标绘控制点、选中/换帧 → 同一条记录更新、
// 取消选中 → 记录与图层一起消失。
const fnStart = appSrc.indexOf('/**\n * 计划航路剩余段');
const fnEnd = appSrc.indexOf('/**\n * 幂等开关：', fnStart);
check('能从 app.js 定位轨迹标绘函数', fnStart >= 0 && fnEnd > fnStart);
const st = {
  tick: 0, selected: 'T1', targets: {}, trackPlotId: null, trackPlotKey: null,
};
const targetDef = {
  id: 'T1', affiliation: 'blue', altitude_m: 900,
  waypoints: [[24.4, 119.7], [24.5, 119.8], [24.6, 119.9]],
};
let posNow = { lon: 119.65, lat: 24.35, alt: 1200 };
const appTrackFns = new Function(
  'state', 'positionFor', 'Plotting', 'TRACK_MODE', 'TRACK_HORIZON_TICKS', 'TICK_SECONDS', 'isVisible',
  appSrc.slice(fnStart, fnEnd) +
  '\nreturn { trackPlotPoints: trackPlotPoints, syncTrackPlot: syncTrackPlot };'
)(st, function () { return posNow; }, Plotting, 'planned', 60, 30, function () { return true; });
st.targets.T1 = { def: targetDef };

const planPts = appTrackFns.trackPlotPoints(targetDef);
check('计划剩余段 = 当前位置 + 当前航段终点及其后续点',
  planPts.length === 3 &&
  planPts[0][0] === 119.65 && planPts[0][1] === 24.35 && planPts[0][2] === 1200 &&
  planPts[1][0] === 119.8 && planPts[1][1] === 24.5 && planPts[1][2] === 900,
  JSON.stringify(planPts));
check('无航路点时返回空（不画假轨迹）',
  appTrackFns.trackPlotPoints({ id: 'T2', waypoints: [] }).length === 0);

appTrackFns.syncTrackPlot();
let trackItem = Plotting.debug().items.filter(function (it) { return it.id === st.trackPlotId; })[0];
check('选中目标 → 建立一条 target_track 标绘记录',
  !!trackItem && trackItem.type === 'target_track' && trackItem.layer === 'plot.track' &&
  trackItem.system === true && trackItem.points === 3 && trackItem.heights === 3,
  JSON.stringify(trackItem));
check('蓝方目标的轨迹按敌方色（hostility=enemy）',
  !!trackItem && trackItem.hostility === 'enemy',
  JSON.stringify(trackItem && trackItem.hostility));
const firstId = st.trackPlotId;

// 换帧/目标移动：同一条记录更新，不新增一条
posNow = { lon: 119.78, lat: 24.48, alt: 1000 };
st.tick = 1;
appTrackFns.syncTrackPlot();
const itemsNow = Plotting.debug().items.filter(function (it) { return it.layer === 'plot.track'; });
check('换帧只更新同一条标绘记录（不新增）',
  st.trackPlotId === firstId && itemsNow.length === 1,
  'id=' + st.trackPlotId + ' n=' + itemsNow.length);
check('换帧后控制点跟随目标位置重算',
  itemsNow[0].points === 3 && itemsNow[0].heights === 3,
  JSON.stringify({ points: itemsNow[0].points, heights: itemsNow[0].heights }));

// 地面目标：全 0 高度不传 heights，保持贴地（开地形后不被埋）
posNow = { lon: 119.8, lat: 24.5, alt: 0 };
targetDef.domain = 'land';
targetDef.altitude_m = 0;
st.tick = 2;
appTrackFns.syncTrackPlot();
const landItem = Plotting.debug().items.filter(function (it) { return it.id === st.trackPlotId; })[0];
check('地面目标不携带 heights（走贴地折线）',
  !!landItem && landItem.heights === 0,
  JSON.stringify(landItem && { points: landItem.points, heights: landItem.heights }));

// 取消选中：记录与动态图层一起消失
st.selected = null;
appTrackFns.syncTrackPlot();
check('取消选中后移除轨迹记录并注销 plot.track 图层',
  st.trackPlotId === null && !LayerRegistry.def('plot.track') &&
  Plotting.debug().items.every(function (it) { return it.layer !== 'plot.track'; }));


console.log();
if (failures) {
  console.log('FAILED (' + failures + ')');
  process.exit(1);
}
console.log('ALL DYNAMIC PLOTTING CHECKS PASSED');
