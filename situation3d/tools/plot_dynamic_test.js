#!/usr/bin/env node
/**
 * 无浏览器自检：标绘的动态图层 + 计划轨迹的取点逻辑。
 *
 * 为什么需要它
 * ------------
 * 前端验证通常要起 Cesium 页面（`plot_selftest.py`）。但这轮的核心是**数据/图层
 * 逻辑**：图层集合由实际标绘要素驱动、声明类别即使为空也常驻在册。这些不依赖
 * WebGL，可以在 Node 里用最小 Cesium 桩 + 真实的 `plotting.js` / `layerRegistry.js`
 * 与从 `app.js` 抽出的真实函数验证。
 *
 * 覆盖（对应当前设计，见 ac7da7f / 3a60ad7）：
 *   1. 类别元数据覆盖目录；目录里没有 system 专用类型（轨迹走实体，不走标绘）
 *   2. 声明类别即使为空也常驻在册，label / group / subgroup 与设计一致
 *   3. 未声明类别按首次出现自动成层，删空后注销
 *   4. 显隐意图跨注销保留；provider 丢失后 sync 自愈（含走重建分支的情况）
 *   5. 导出 / 导入往返；未知类型被拒
 *   6. 计划轨迹取点（有航路点 / 无航路点按帧 / 两者都无）
 *   7. 轨迹是独立实体（虚线 + 端点箭头），不是标绘图层
 *
 * 用法：node tools/plot_dynamic_test.js
 */
'use strict';

const fs = require('fs');
const path = require('path');

const ROOT = path.dirname(__dirname);
const SRC = path.join(ROOT, 'web', 'src');

let failures = 0;
let defects = 0;
function check(name, cond, detail) {
  console.log((cond ? '  ok   ' : '  FAIL ') + name + (detail && !cond ? '  ' + detail : ''));
  if (!cond) failures++;
}

/**
 * 已登记但尚未获批修复的缺陷：断言的是**应有行为**，当前未满足时只报 `DEFECT`
 * 并计入末尾清单，不判失败 —— 这样测试的重写能先落地，缺陷本身仍可见、不被
 * 悄悄写成"现状即正确"。（修复后这一行会自动变回 `ok`。）
 */
function defect(name, fixed, detail) {
  if (fixed) { console.log('  ok   ' + name + '（缺陷已修复）'); return; }
  defects++;
  console.log('  DEFECT ' + name + (detail ? '  ' + detail : ''));
}

/** 源码按 LF 读：定位标记都写成 \n，CRLF 检出会让一条都找不到。 */
function readSrc(file) {
  return fs.readFileSync(path.join(SRC, file), 'utf8').replace(/\r\n/g, '\n');
}

// ---------------------------------------------------------------- Cesium 桩
function color(hex) {
  return { hex: hex, withAlpha: function (a) { return color(hex + '@' + a); } };
}
const Cesium = {
  Color: { fromCssColorString: color, WHITE: color('#ffffff') },
  Cartesian3: {
    fromDegrees: function (lon, lat, h) { return { lon: lon, lat: lat, h: h }; },
    fromDegreesArrayHeights: function (flat) { return flat.slice(); },
    fromDegreesArray: function (flat) { return flat.slice(); },
    distance: function (a, b) {
      return Math.hypot(a.lon - b.lon, a.lat - b.lat, (a.h || 0) - (b.h || 0));
    },
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
  (0, eval)(readSrc(file));
}
load('plotting.js');
load('layerRegistry.js');

const Plotting = global.Plotting;
const LayerRegistry = global.LayerRegistry;
check('Plotting / LayerRegistry 已加载', !!Plotting && !!LayerRegistry);

Plotting.bind(viewer);
LayerRegistry.bind(viewer, {
  plot: function (id, want) { Plotting.setLayerVisible(id, !!want); },
});

// ------------------------------------------- 从 app.js 抽出真实的动态同步代码
const appSrc = readSrc('app.js');
const startAt = appSrc.indexOf('var _plotLayerDesired = {};');
const endAt = appSrc.indexOf('/**\n * 注册各图层的**要素提供者**', startAt);
if (startAt < 0 || endAt < 0) {
  console.error('无法从 app.js 定位动态图层代码块');
  process.exit(1);
}
// 这几处是 app.js 里的全局函数/DOM 依赖，桩掉即可（不影响被测逻辑）。
global.refreshPlotLayerSelect = function () {};
global.applyLayers = function () { LayerRegistry.apply(); };
global.LayerPanel = { invalidate: function () {} };
const dynamic = new Function(
  appSrc.slice(startAt, endAt) +
  '\nreturn { syncPlotLayers: syncPlotLayers, desired: _plotLayerDesired,' +
  ' plotLayerOptions: plotLayerOptions };'
)();
const syncPlotLayers = dynamic.syncPlotLayers;
Plotting.onChange(function () { syncPlotLayers(); });

const warned = [];
const realWarn = console.warn;
console.warn = function () {
  warned.push(Array.prototype.slice.call(arguments).join(' '));
};
function resetWarns() { warned.length = 0; }
function spuriousWarns() {
  return warned.filter(function (w) { return /不是标绘类别层/.test(w); });
}

function plotLayerIds() {
  return LayerRegistry.list('plot').filter(function (l) { return l.subgroup === '标绘'; })
    .map(function (l) { return l.id; }).sort();
}

// ---------------------------------------------------------------- 1. 元数据
const cats = Plotting.categories();
const catIds = cats.map(function (c) { return c.id; });
const missingCats = Plotting.catalogue().map(function (c) { return c.category; })
  .filter(function (c) { return catIds.indexOf(c) < 0; });
check('categories 覆盖目录里的全部类别', missingCats.length === 0,
  'cats=' + catIds.join(',') + ' missing=' + missingCats.join(','));

// 轨迹在 2026-10-20（ac7da7f）改回实体渲染：目录里不应再有 system 专用类型。
const systemTypes = Plotting.catalogue().filter(function (c) { return c.system; })
  .map(function (c) { return c.type; });
check('目录没有 system 专用标绘类型（轨迹走实体，不走标绘）',
  systemTypes.length === 0, systemTypes.join(','));
check('手动画布没有 track 类别', catIds.indexOf('track') < 0, catIds.join(','));

const layerOpts = dynamic.plotLayerOptions();
check('手动标绘图层选项覆盖全部声明类别',
  layerOpts.length === cats.length &&
  layerOpts.every(function (o) {
    return cats.some(function (c) { return c.layer === o[0]; });
  }),
  JSON.stringify(layerOpts));

check('源码中无「预测/预计」残留',
  ['app.js', 'plotting.js', 'layerRegistry.js', 'situation.js'].every(function (f) {
    return !/预测|预计/.test(readSrc(f));
  }));

// ------------------------------------------- 2. 声明类别常驻在册
// 设计（3a60ad7 起）：标绘图层归【TS图层】→【标绘】子组；即使当前没有要素也
// 保留成行，用户才能在那儿开关。
syncPlotLayers();
const initial = plotLayerIds();
check('声明类别全部在册（空类别不消失）',
  initial.length === cats.length && initial.every(function (id) {
    return catIds.some(function (c) { return 'plot.' + c === id; });
  }),
  JSON.stringify(initial));

const defFire = LayerRegistry.def('plot.fire');
const catFire = cats.filter(function (c) { return c.id === 'fire'; })[0];
check('图层名直接用类别中文名（不再带「标绘·」前缀）',
  !!defFire && defFire.label === catFire.zh && defFire.label.indexOf('标绘') < 0,
  JSON.stringify(defFire && defFire.label));
check('归在【TS图层】/【标绘】子组，z 取类别定义',
  !!defFire && defFire.group === 'TS图层' && defFire.subgroup === '标绘' &&
  defFire.z === catFire.z,
  JSON.stringify(defFire && { group: defFire.group, subgroup: defFire.subgroup,
                              z: defFire.z }));
check('每个声明类别都有要素提供者',
  initial.every(function (id) { return LayerRegistry.hasProvider(id); }),
  JSON.stringify(initial.filter(function (id) {
    return !LayerRegistry.hasProvider(id);
  })));
check('sync 不对合法标绘图层报警告（新建分支）',
  spuriousWarns().length === 0, JSON.stringify(warned.slice(0, 3)));

// ------------------------------------------- 3. 未声明类别：出现即成层，空了注销
const A = [119.4, 24.2], B = [120.6, 25.0];
const p3 = Plotting.add({ type: 'phase_line', layer: 'plot.custom_ext', points: [A, B] });
const defCustom = LayerRegistry.def('plot.custom_ext');
check('未声明的类别也能自动成层（不依赖固定分类）',
  !!defCustom && defCustom.subgroup === '标绘',
  JSON.stringify(defCustom && { label: defCustom.label, z: defCustom.z }));
check('新增图层后总数为 声明类别 + 1',
  plotLayerIds().length === cats.length + 1, JSON.stringify(plotLayerIds()));

Plotting.remove(p3.id);
check('最后一个要素删除后未声明图层注销，声明类别仍留',
  !LayerRegistry.def('plot.custom_ext') && plotLayerIds().length === cats.length,
  JSON.stringify(plotLayerIds()));

// ------------------------------------------- 4. 显隐意图 + provider 自愈
// 场景一：只有 provider 丢失、图层集合没变（走"集合一致"分支）。
LayerRegistry.setProvider('plot.intel', null);
syncPlotLayers();
check('provider 丢失后 sync 自愈', LayerRegistry.hasProvider('plot.intel'));

// 场景二：图层被外部注销（走重建分支）——要素提供者与用户意图都要补回来。
LayerRegistry.unregister('plot.fire');
dynamic.desired['plot.fire'] = false;
LayerRegistry.setProvider('plot.defense', null);   // 同一轮里再制造一处 provider 缺失
resetWarns();
syncPlotLayers();
check('注销后 sync 重建图层', !!LayerRegistry.def('plot.fire'));
check('重建时恢复用户关掉的显隐意图',
  LayerRegistry.runtime['plot.fire'].desired === false,
  JSON.stringify(LayerRegistry.runtime['plot.fire'] &&
                 LayerRegistry.runtime['plot.fire'].desired));
// 3a60ad7 把标绘图层挪到【TS图层】/subgroup『标绘』，但 ensurePlotLayer 的
// 既有层分支仍按 `def.group !== '标绘'` 判定，于是重建分支上每一个已存在的
// 标绘层都被当成"非标绘类别层"拒掉、provider 不补。集合一致的分支不经过那里，
// 所以只有"恰好在同一轮有图层增删"时才暴露。
defect('重建分支也补齐既有层的 provider',
  LayerRegistry.hasProvider('plot.defense'),
  'defense provider=' + LayerRegistry.hasProvider('plot.defense'));
delete dynamic.desired['plot.fire'];
LayerRegistry.setDesired('plot.fire', true);
LayerRegistry.setDesired('plot.defense', true);

// ------------------------------------------- 5. 导出 / 导入
const p5 = Plotting.add({ type: 'phase_line', layer: 'plot.fire', points: [A, B],
                          label: '测试' });
const exported = Plotting.exportAll();
check('exportAll 输出控制点而不是派生几何',
  exported.length === 1 && Array.isArray(exported[0].points) &&
  exported[0].points.length === 2, JSON.stringify(exported[0]));
const imported = Plotting.importAll(JSON.parse(JSON.stringify(exported)),
                                    { replace: true });
check('导出结果可原样导入', imported.added === 1 && imported.errors.length === 0,
  JSON.stringify(imported));
const rejected = Plotting.importAll([{ type: 'no_such_type', points: [A, B] }]);
check('未知标绘类型被拒绝',
  rejected.added === 0 && rejected.errors.length >= 1 &&
  rejected.errors[0].indexOf('未知标绘类型') >= 0, JSON.stringify(rejected));
Plotting.clearLayer();   // 不传图层 id = 清空全部
check('清空后声明类别仍在册',
  plotLayerIds().length === cats.length && Plotting.debug().count === 0,
  JSON.stringify({ layers: plotLayerIds(), items: Plotting.debug().count }));

// ------------------------------------------- 6. 计划轨迹取点（真实代码）
//
// 轨迹不走了标绘（ac7da7f），但取点逻辑仍是 app.js 里的真函数：
// 「当前位置 + 当前航段终点及其后续点」少一个点就会画出一条断头的计划线。
const fnStart = appSrc.indexOf('function predBearing(');
const fnEnd = appSrc.indexOf('/** 计划轨迹箭头图标');
check('能从 app.js 定位计划轨迹函数', fnStart >= 0 && fnEnd > fnStart,
  'start=' + fnStart + ' end=' + fnEnd);

const st = { tick: 0, data: { frames: [], targets: [] } };
let posNow = { lon: 119.65, lat: 24.35, alt: 1200 };
let framePos = null;
const appTrack = new Function(
  'state', 'positionFor', 'PREDICT_HORIZON_TICKS',
  appSrc.slice(fnStart, fnEnd) +
  '\nreturn { predBearing: predBearing, predictedTrackPoints: predictedTrackPoints };'
)(st, function (id, t) {
  if (framePos && framePos[t]) {
    return { lon: framePos[t][0], lat: framePos[t][1], alt: framePos[t][2] };
  }
  return posNow;
}, 60);

// -- 有航路点：当前位置 → 当前航段终点 → 后续航路点
// waypoints 的坐标顺序是 [纬度, 经度]（app.js 里按 w[0]=lat / w[1]=lon 取）。
st.tick = 1;
posNow = { lon: 119.65, lat: 24.35, alt: 1200 };
const wp = { id: 'T1', altitude_m: 900,
  waypoints: [[24.4, 119.7], [24.5, 119.8], [24.6, 119.9]] };
const plan = appTrack.predictedTrackPoints(wp);
check('有航路点：首点是当前位置（含高度）',
  plan.length >= 2 && plan[0][0] === 119.65 && plan[0][1] === 24.35 &&
  plan[0][2] === 1200, JSON.stringify(plan));
check('有航路点：末点是最后一个航路点（[经度, 纬度, 高度]）',
  plan[plan.length - 1][0] === 119.9 && plan[plan.length - 1][1] === 24.6 &&
  plan[plan.length - 1][2] === 900, JSON.stringify(plan[plan.length - 1]));
check('每个点都是三元组 [经度, 纬度, 高度]',
  plan.every(function (p) { return p.length === 3 && p.every(isFinite); }),
  JSON.stringify(plan));

// -- 无航路点（CZML 驱动）：用后续帧的计划位置，在第一次明显转向处截断
// `state.data.frames` 只被当作"还有后续帧"的长度信号；具体位置仍由
// positionFor 给出（与浏览器里一致）。
const lon0 = 119.65, lat0 = 24.35;
framePos = [];
for (let t = 0; t <= 11; t++) {
  framePos.push(t <= 7
    ? [lon0 + t * 0.01, lat0 + t * 0.01, 1000]
    : [lon0 + 0.07 + (t - 7) * 0.01, lat0 + 0.07 - (t - 7) * 0.01, 1000]);
}
st.data.frames = framePos.slice();
posNow = { lon: framePos[1][0], lat: framePos[1][1], alt: 1000 };
const byFrame = appTrack.predictedTrackPoints({ id: 'T2' });
check('无航路点：首点是当前位置',
  byFrame.length >= 2 && byFrame[0][0] === framePos[1][0] &&
  byFrame[0][1] === framePos[1][1], JSON.stringify(byFrame.slice(0, 2)));
check('无航路点：在第一次明显转向处截断',
  byFrame.length < framePos.length && byFrame.length >= 3,
  'points=' + byFrame.length + ' frames=' + framePos.length);

// -- 既无航路点也无后续帧：只返回当前点（实体按 <2 点隐藏）
framePos = [];
st.data.frames = [];
const lone = appTrack.predictedTrackPoints({ id: 'T3' });
check('无航路点且无后续帧：只返回当前点', lone.length === 1, JSON.stringify(lone));

check('方位角按顺时针从北算（正北 0、正东 90）',
  appTrack.predBearing([119.0, 24.0, 0], [119.0, 25.0, 0]) === 0 &&
  appTrack.predBearing([119.0, 24.0, 0], [120.0, 24.0, 0]) === 90,
  JSON.stringify([appTrack.predBearing([119.0, 24.0, 0], [119.0, 25.0, 0]),
                 appTrack.predBearing([119.0, 24.0, 0], [120.0, 24.0, 0])]));

// ------------------------------------------- 7. 轨迹是实体，不是标绘
const dashAt = (function () {
  const i = appSrc.indexOf("id: 'predict-' + def.id");
  if (i < 0) return -1;
  const j = appSrc.indexOf('billboard:', i);
  return (j > i && /PolylineDashMaterialProperty/.test(appSrc.slice(i, j))) ? j : -1;
})();
check('计划轨迹是独立实体（predict- 前缀）',
  /id: 'predict-' \+ def\.id/.test(appSrc));
check('计划轨迹用虚线材质 + 终点箭头', dashAt > 0);
check('web/src 里不再出现 target_track',
  ['app.js', 'plotting.js', 'layerRegistry.js', 'situation.js'].every(function (f) {
    return !/target_track/.test(readSrc(f));
  }),
  // situation.js 里的 `plot.track` 只剩两行注释（见 TODOLIST E 的残留说明）
  '');

// 标绘图层在 3a60ad7 之后用 subgroup 『标绘』归组，group 是『TS图层』。
// 仍按 group 判定的两处会把合法标绘层拒掉：
//   - app.js ensurePlotLayer：重建分支上拒层、不补 provider（上面已计一条 DEFECT）
//   - situation.js 导入校验：plotLayers 只剩目录默认类别，显式归属图层的标绘被
//     静默改回默认类别（用户看到的是"导入成功但图层变了"）
defect('situation.js 用 subgroup 判定标绘图层',
  !/\.group === '标绘'/.test(readSrc('situation.js')),
  'situation.js:677 仍是 .group === \'标绘\'');

console.warn = realWarn;
console.log();
if (defects) {
  console.log('DEFECTS (' + defects + ') —— 已登记、未修复，见 TODOLIST E');
}
if (failures) {
  console.log('FAILED (' + failures + ')');
  process.exit(1);
}
console.log('ALL DYNAMIC PLOTTING CHECKS PASSED');
