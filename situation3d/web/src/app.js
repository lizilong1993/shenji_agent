/* situation3d — 三维ZCTS生成系统前端
 *
 * 职责：
 *   1. 加载引擎输出的逐帧TS与三级推理结果，在 Cesium 三维地球上回放；
 *   2. 用军标符号呈现海/陆/空三域的红蓝双方目标；
 *   3. 点击目标时在中间窗口右上角弹出信息面板（型号 / ID / 经纬高 / 三级TS）；
 *   4. 热点区域按真实经纬范围绘制，点击后联动右侧"地理位置信息"；
 *   5. 模型开关逐项对应一个真实生效的渲染行为。
 *
 * 数据流：engine (per-tick state + inference) -> data/<case>.json -> 本文件
 *        -> Cesium entities
 *
 * 离线：底图/高程/建筑瓦片全部来自 web/tiles/，运行期不访问外网。
 */

'use strict';

// 截图与深链用的确定性钩子：
//   ?case=<id>      指定演示案例
//   ?tick=300       首帧前定位到某帧
//   ?autoplay=1     立即开始推演
//   ?panel=R01      打开指定目标的信息面板
//   ?hotspot=0      选中第 N 个热点区域
//   ?terrain=1      打开地形模型
//   ?overlay=1      打开 WSF3D 建筑高度图层
//   ?buildings=1    打开 WSF3D 三维建筑体块
//   ?sensors=1      打开传感器覆盖
const params = new URLSearchParams(window.location.search);

const LEVEL_CLASS = { 1: 'lv1', 2: 'lv2', 3: 'lv3', 4: 'lv4' };
// ---- B（方向 1）：轨迹显示窗口 / 选中目标轨迹标绘 / 选中掩码 -------------------
// TRAIL_WINDOW_TICKS：历史轨迹**全量保留**，只在渲染时切片。窗口只影响"看多少"，
// 不影响"清空后从第 0 帧重建"的回退逻辑（回退靠 rec.lastTick，不靠窗口）。
const TRAIL_WINDOW_TICKS = 120;
// 目标轨迹来源：默认取"计划航路剩余段"，是一段**计划轨迹**（虚线）；
// 'kinematic' 才是航向外推备选。两种模式都走同一套标绘渲染，不再另建独立轨迹实体。
const TRACK_MODE = 'planned';
// 计划轨迹的前推窗口（帧）。无航路点的数据集用它从帧序列里取未来位置。
const PREDICT_HORIZON_TICKS = 60;
// 航向外推的视野（备选，默认不走 TRACK_MODE=planned；切换前未在远程验证过）。TICK_SECONDS = 30。
const TRACK_HORIZON_TICKS = 60;
const TICK_SECONDS = 30;
// 掩码内环的最小挖空半径（度）：目标刚起步、轨迹只有几帧时也看得见那个点。
const FOCUS_MASK_PAD_DEG = 0.05;
// 选中掩码的图元 id：左键拾取被它挡住时要能认出它（它不是可选对象）。
const FOCUS_MASK_ID = 'overlay-focus-mask';

// ---- D (direction 2): occupation faces + minimap mask ----------------------
// Four distinguishable status colours. The engine owns the judgement; the
// front end only paints frames[i]["occupation"] and never reads plotting.
const OCCUPATION_LAYER_ID = 'situation.occupation';
const OCCUPATION_FACE_HEIGHT_M = 1000;
// I09：视觉编码的**唯一来源是 `web/src/theme.css`**，此处只做一次 Cesium 包装。
// 原先这 4 个颜色是硬编码字面量，与 style.css 的 --red/--blue 各写一份
// （实测同一个"红方"有两个值：#d0342c vs #c8352c），改一处不影响另一处。
// `Theme` 读不到 CSS 时会回落到与其内置默认（与 theme.css 一致），故不会静默变黑。
function themeColor(key) {
  if (typeof window !== 'undefined' && window.Theme) {
    return Cesium.Color.fromCssColorString(window.Theme.color(key));
  }
  // 无 Theme（极端情况）时不静默：抛出以便被 pageErrors 捕获，而不是画成黑色
  throw new Error('Theme module unavailable; cannot resolve colour: ' + key);
}

const OCCUPATION_STATUS_COLORS = {
  red: themeColor('faction-red'),
  blue: themeColor('faction-blue'),
  contested: themeColor('faction-contested'),
  neutral: themeColor('faction-neutral'),
};

const state = {
  // 动态图层管理的**单要素**状态。与图层开关、阵营/域过滤是三层，取与（见 isVisible）。
  //   hiddenTargets   单独关掉某个目标（可再打开）
  //   removedTargets  从当前TS移除（会话内；重新加载案例即恢复，数据集文件不动）
  //   targetOverrides 目标可改的**只有它的 TS 信息**（名称/阵营/域/装备/型号），
  //                  几何与航迹来自数据集，改它们等于改数据
  hiddenTargets: {},
  removedTargets: {},
  targetOverrides: {},
  // 航路与热点同理：整层开关回答"这一类看不看"，单个要素开关回答"这一条看不看"。
  hiddenRoutes: {},
  hiddenHotspots: {},
  caseIndex: null,
  caseId: null,
  data: null,
  tick: 0,
  tickFloat: 0,
  playing: false,
  ticksPerSecond: 1,
  targets: {},              // id -> { def, entity, trail: [], lastTick, sensorEntity }
  layers: { targets: true, targetsRed: true, targetsBlue: true,
            tracks: true, routes: true, hotspots: true, labels: true,
            targetPanel: true, overview: true, focusMask: false, occupation: true },
  filters: { red: true, blue: true, sea: true, land: true, air: true },
  selected: null,
  focusMaskEntity: null, // 选中掩码图元（懒建：没选中就先不建）
  focusMaskHole: null,   // 掩码内环经纬边界（focusDebug 报出去用）
  hotspotEntities: [],
  occupationEntities: [],
  hotspotRows: [],
  routeEntities: [],
  activeHotspot: -1,        // index into data.hotspots, -1 = none
  geoPinned: false,         // true 时"地理位置信息"锁定在选中的热点区域
  ramp: null,
  terrainOn: false,
  wsf3dOverlayLayer: null,
  buildingsPrimitive: null,
  buildingsAoi: -1,
  buildingsKey: null,       // "aoiIdx:lodIdx"，当前已生成的体块组合
  buildingsCache: {},       // 已生成的体块，按组合缓存复用
  buildingBusy: false,
  // 已 add 进场景、但几何还没 ready 的体块（隐藏状态）。见 refreshBuildings 里的
  // waitReady 轮询：Primitive 必须先在场景里才会推进异步几何准备。
  pendingPrimitive: null,
  buildingTimer: null,
  imageryLayer: null,
  // 最近一次 LayerRegistry.apply() 的结果（取证/数据健康面板用）。
  // 图层句柄本身不由 state 持有——注册表才是唯一的顺序与归属来源。
  lastLayerApply: null,
  // 最近一次成功生成/取用的体块组合统计（结构化），供 debugBuildings() 取证。
  lastBuildingsStats: null,
  cursorCarto: null,        // 光标处的地面点，用于"地理位置信息"读数
  lastError: null,          // 最近一次未捕获异常
  urbanLayers: null,        // 城区 z12-z14 高清影像层（按相机高度挂载）
  urbanAoi: [],             // 城区 AOI 列表
  urbanEnabled: true,       // 城区高清影像是否启用（图层管理「底图」）
  globalLayer: null,        // 离线全球层（z0-6，Esri 瓦片树）
  esriLayer: null,          // 在线 Esri 底图（浏览器直连，无代理）
  hotspotLayers: [],        // 热点区域高清层（每框一个 provider）
  hotspotImageryEnabled: true,  // 热点高清影像是否启用（与 urbanEnabled 同构）
  hotspotImageryMounted: false, // 当前是否因相机高度而挂着（区别于"用户想不想要"）
  offlineDetailLayers: [],  // 已挂的离线高倍段实例 [{zoom, layer}]
  offlineDetailZooms: [],   // 已挂的层级（用于判断"要不要动"）
  offlineDetailEnabled: true,   // 离线高倍段是否启用（图层管理「底图」）
  imageryStats: null,       // 底图瓦片请求统计（诊断用）
  inference: null,          // inference_frames，与 frames 逐帧对齐

  // 目标视角。id 是当前相机绑定的目标，mode 是相机驾驶模式：
  //   off    未绑定（camera.transform 必须是单位阵）
  //   glide  选中后的平滑过渡（逐帧沿测地线插值）
  //   center 普通选中模式：lookAt 锁定目标，目标始终居中，缩放围绕目标
  //   chase  第三人称斜后方跟随：相机绕到目标运动方向的后上方并沿前进方向观察
  // enabled 只对应「目标跟随」复选框（chase）；suspend 在 chase 模式下用户
  // 按下鼠标时置位，让相机在拖拽期间完全交给用户，松手后阻尼回到跟随位。
  follow: {
    enabled: false, id: null, mode: 'off', suspend: false, press: null,
    resumeAt: 0, glide: null,
    focus: null,          // 当前 lookAt 的原点（上一目标位置），用于平移跟随
    lastPos: null,        // 上一帧目标位置（航向兜底用）
    pose: { heading: 0, pitch: 0, range: 0 },
    chaseRange: null,     // chase 模式的距离；滚轮缩放直接改它
    lastNow: 0,
    lastLockMs: 0,        // 最近一次自动改相机的时间（建筑体块防抖用）
    notified: false,
  },

  // 地图组件显隐（图层管理第四类）
  components: { compass: true, scalebar: true, viewpoint: true },

  base: null,               // data/base.json：与案例无关的基础数据
  baseCamera: null,         // 空态默认机位
  baseRegion: null,         // 空态下标题栏显示的来源区域
};

let viewer = null;
let lastFrameMs = 0;
let lastGeoRefresh = 0;
let camChangedCount = 0;      // camera.changed 触发次数（诊断用）

// base.json 缺失时的兜底机位：**中国全景**（2026-10-16 由"台海全景"改）。
// 需求是"初始化时把视点中心定位在中国"；base.json 里的 camera 是同一组值，
// 两者保持一致，这样 base.json 缺失时行为不变。加载具体案例仍飞回案例机位。
const FALLBACK_CAMERA = { lon: 104.0, lat: 35.5, height: 8000000, heading: 0, pitch: -90 };

// 已移除的 id（连同对应的面板一起）：targetList / targetCountHint（左栏目标列表）、
// healthSummary / healthBody / healthRefresh（数据健康栏）、
// targetPanel / tpTitle / tpBody / tpClose（中部浮动目标面板，内容并入右栏目标意图栏）、
// mvLon / mvLat / mvAlt（视点三行合并为单行 mvAll）、
// menuLayers / menuView / menuSituation（顶栏下拉）、btnOverview / btnShareScene、
// switchNote / followNote / hint（说明性文字，按"界面不写操作说明"移除）、
// geoLon / geoLat / geoRegion / geoLevel / hotspotBody（区域TS栏只保留案例范围的整体
// TS，逐热点与光标所辖区域的那些读数一并去掉）。
const dom = {};
['statusText', 'statusDot', 'scenarioName',
 'geoSpan',
 'rlValue', 'rlNote', 'rlIntent', 'rlPhase',
 'overallBody', 'targetBody',
 'menuCase', 'menuCaseList',
 'tickLabel', 'timeline', 'clockLabel',
 'countRed', 'countBlue', 'countSea', 'countLand', 'countAir',
 'mapCompass', 'compRose', 'mapViewpoint', 'mvAll', 'modelPlaces',
 'modelTianditu', 'modelImagery', 'modelTerrain', 'modelBuildings',
 'mapScalebar', 'sbBar', 'sbLabel', 'followTarget',
].forEach(function (id) { dom[id] = document.getElementById(id); });

// 已移除读数的**惰性桩**。
//
// `selectHotspot()` / `updateGeoReadout()` 里还有若干"跟随光标/热点"的读数写入，那些
// 元素已随区域TS栏的收敛而去掉。与其在有限的改动里逐处删（容易漏一处、漏了就是
// 每次移动鼠标抛一次 TypeError），不如在这里给一个显式的空对象：写入静默丢失，
// 控制流（尤其是 selectHotspot 里的相机动作）完全不受影响。
//
// 这是有意为之的取舍，不是疏忽——**桩名里带 Removed 前缀**，一眼能看出它不对应任何
// DOM，将来清理这些死写入时也容易定位。
['geoLon', 'geoLat', 'geoRegion', 'geoLevel', 'hotspotBody', 'switchNote', 'followNote']
  .forEach(function (id) {
    if (!dom[id]) {
      dom[id + 'Removed'] = true;
      dom[id] = { textContent: '', className: '', innerHTML: '', appendChild: function () {} };
    }
  });

// --------------------------------------------------------------------------
// 瓦片加载进度
// --------------------------------------------------------------------------

/**
 * 瓦片加载进度回调。
 *
 * 顶部的"正在加载瓦片… 剩余 N"徽标已按需求**删除**（2026-10-17）；这个回调留下
 * 来是因为它还担着两件与提示无关的事：
 *   1. 告诉 TileCache"底图是否空闲" —— 这是**预加载与渲染之间的调度闸门**，
 *      删掉它预加载就失去唯一的启动时机；
 *   2. 空闲后创建鸟瞰图 —— 启动期两个 WebGL 上下文会互抢连接与 CPU。
 *
 * 队列计数的"过期纠正"也留着：`tileLoadProgressEvent` 只在队列**变化**时触发，
 * 而实测存在"最后几块是被移除/取消而不是加载完成"的路径 —— 事件停在"剩余 2"，
 * 同一时刻 `globe.tilesLoaded === true`、三条队列全空、零在途请求（2026-10-15 用
 * CDP 摊开核对过）。空闲判定若照它的字面值走，就永远判不出空闲、预热永不启动。
 */
function onTileLoadProgress(queued) {
  if (queued > 0 && viewer && viewer.scene && viewer.scene.globe
      && viewer.scene.globe.tilesLoaded) {
    queued = 0;
  }

  if (window.TileCache && TileCache.setIdle) TileCache.setIdle(queued === 0);

  // 底图空闲后再创建鸟瞰图：启动期两个 WebGL 上下文会互抢连接与 CPU。
  // 面板默认带 hidden，只有 Viewer 真正建好才显示，避免先出现一个白框。
  if (queued === 0 && state.layers.overview && window.Overview) {
    try { Overview.show(); } catch (e) { console.warn('[overview]', e.message); }
  }
}

function humanBytes(n) {
  if (n === null || n === undefined) return '—';
  const units = ['B', 'KB', 'MB', 'GB'];
  let v = n;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
  return v.toFixed(i === 0 ? 0 : 1) + ' ' + units[i];
}

/**
 * 刷新"瓦片缓存"面板。
 *
 * 这个面板存在的意义是把缓存效果变成可见的数字：浏览过程中"回服务器取"
 * 基本不再增长，就说明浏览过的瓦片确实走了本地。
 */

// --------------------------------------------------------------------------
// 通用工具
// --------------------------------------------------------------------------

/**
 * `?diag=1` 时在窗口左上角叠加一张运行时读数。
 *
 * 存在的理由：Cesium 的底图层级、地形加载和瓦片失败都不会抛错，只会安静地
 * 退化成模糊画面。这些数字把"看起来不对"变成"哪一项不对"。
 */
function renderDiagnostics() {
  if (params.get('diag') !== '1') return;
  if (!viewer) return;      // manifest 未就绪时定时器已在跑
  let box = document.getElementById('diagBox');
  if (!box) {
    box = document.createElement('pre');
    box.id = 'diagBox';
    box.style.cssText = 'position:absolute;left:12px;top:52px;z-index:70;margin:0;' +
      'padding:8px 10px;max-width:520px;border:1px solid #b9c6d8;border-radius:4px;' +
      'background:rgba(255,255,255,.96);font:11px/1.5 monospace;color:#1d2430;' +
      'white-space:pre-wrap;';
    document.querySelector('.globe-wrap').appendChild(box);
  }
  const im = state.imageryStats || { requests: 0, errors: 0, maxLevel: -1, minLevel: -1, levels: {} };
  const lines = [];
  lines.push('Cesium ' + Cesium.VERSION);
  lines.push('camera height   ' + (viewer.camera.positionCartographic.height / 1000).toFixed(1) + ' km');
  // ⚠ 必须写清这组计数是**哪一层**的。`imageryStats` 只给**离线底图** provider 包了
  // 计数（见 initViewer 里的 `OfflineAssets.instrument`）；天地图走服务端代理 `/tdt/`，
  // **不计入这里**。而 `imagery.offline` 默认是关的（默认底图是天地图），于是
  // 画面上明明有瓦片在加载，这里却显示 `requests 0 / levels none` —— 查"底图为什么糊"
  // 的人会照着这个 0 往"影像根本没加载"的方向查，正好查反。
  // 实测：把 `imagery.offline` 打开并压到 300 km 后，计数立刻变成
  // requests 57 / levels {1..8} / maxLevel 8。
  lines.push((im.label || 'imagery') + '  requests ' + im.requests
    + '  errors ' + im.errors + '   （仅离线底图层；天地图经 /tdt/ 代理，不计入）');
  lines.push('imagery levels  min=' + im.minLevel + ' max=' + im.maxLevel);
  const hist = Object.keys(im.levels).sort(function (a, b) { return Number(a) - Number(b); })
    .map(function (k) { return 'z' + k + ':' + im.levels[k]; }).join(' ');
  lines.push('imagery hist    ' + (hist || '(none)'));
  if (im.lastError) lines.push('imagery error   ' + String(im.lastError).slice(0, 90));
  lines.push('terrain         ' + (state.terrainOn ? 'on' : 'ellipsoid'));
  lines.push('tiles rendered  ' + viewer.scene.globe.tilesLoaded);
  const layerInfo = [];
  for (let i = 0; i < viewer.imageryLayers.length; i++) {
    const L = viewer.imageryLayers.get(i);
    layerInfo.push(i + ':' + (L.show ? 'on' : 'off') + '@' +
      (L.imageryProvider && L.imageryProvider.maximumLevel));
  }
  lines.push('imagery layers  ' + viewer.imageryLayers.length + ' [' + layerInfo.join(' ') +
             ']  urban ' + (state.urbanLayers ? 'on' : 'off'));
  // 按需挂载的决策过程：光看"现在挂了哪几层"看不出**为什么**是这几层。
  lines.push('detail bands    L=' + cameraImageryLevel() + ' (渲染中 ' + surfaceRenderLevel()
    + ')  挂 z[' + ((state.offlineDetailZooms || []).join(',') || '-') + ']  可用 '
    + (window.OfflineAssets ? (OfflineAssets.offlineDetailBands() || []).length : 0) + ' 层');
  lines.push('hotspot imagery ' + (state.hotspotImageryMounted ? 'on' : 'off')
    + '  阈值 ' + (HOTSPOT_IMAGERY_TRIGGER_M / 1000) + ' km');
  lines.push('ground atmos    ' + viewer.scene.globe.showGroundAtmosphere +
             '  fog ' + viewer.scene.fog.enabled +
             '  cache ' + (window.__situation3dNoTileCache ? 'off' : 'on'));
  lines.push('screenSpaceErr  ' + viewer.scene.globe.maximumScreenSpaceError);
  // Cesium 的底图层级跟随"表面瓦片"层级，而表面瓦片层级由屏幕空间误差决定。
  // 这里直接读出真正在渲染的表面瓦片层级，避免再靠推断。
  try {
    const surface = viewer.scene.globe._surface;
    const tiles = surface && surface._tilesToRender ? surface._tilesToRender : [];
    let maxLevel = -1;
    let sumLevel = 0;
    tiles.forEach(function (t) {
      if (t.level > maxLevel) maxLevel = t.level;
      sumLevel += t.level;
    });
    lines.push('surface tiles   ' + tiles.length + '  maxLevel ' + maxLevel +
               '  meanLevel ' + (tiles.length ? (sumLevel / tiles.length).toFixed(1) : '-'));
  } catch (e) {
    lines.push('surface tiles   n/a');
  }
  const fr = viewer.camera.frustum;
  lines.push('fovy/aspect     ' + (fr.fovy ? (fr.fovy * 180 / Math.PI).toFixed(1) + '°' : '?') +
             '  resScale ' + viewer.resolutionScale);
  const tc = window.TileCache ? TileCache.stats : null;
  if (tc) {
    lines.push('tile cache      hit ' + tc.memoryHits + '  net ' + tc.network +
               '  err ' + tc.errors + '  mem ' + tc.memoryCount + '/' + tc.memoryLimit +
               '  disk-preload ' + tc.preloaded + '  idle ' + tc.idleFlag);
    lines.push('prefetch        取回 ' + tc.prefetched + '  在途 ' + tc.prefetchInflightCount +
               '  排队 ' + tc.prefetchQueueCount + '  丢弃 ' + tc.prefetchDropped +
               '  404跳过 ' + tc.prefetchMiss404Count + '（已知没有 ' + tc.prefetchMissedCount + '）' +
               '  ' + (window.TileCache.prefetchConfig ? JSON.stringify(TileCache.prefetchConfig()) : ''));
    if (tc.lastError) lines.push('cache error     ' + String(tc.lastError).slice(0, 120));
  }
  // 目标视角与参考系状态。跟随时 camera.transform 被 lookAt 锁在目标 ENU 上，
  // 这是**预期状态**；只有未跟随时才必须是单位阵，否则 setView / flyTo 会把
  // orientation 按残留的 ENU 解释，缩放与贴地钳制失效（可以穿到地面以下）。
  lines.push('follow          ' + (state.follow.id || '-') +
             '  mode ' + state.follow.mode +
             '  chase ' + state.follow.enabled +
             '  suspend ' + state.follow.suspend +
             '  locked ' + !viewer.camera.transform.equals(Cesium.Matrix4.IDENTITY));
  lines.push('camChanged      ' + camChangedCount + ' 次');
  if (window.Overview) lines.push('overview        ' + Overview.debugInfo());
  if (window.Places) lines.push('places          ' + Places.debugInfo());
  lines.push('base            ' + (state.base ? 'ok' : 'MISSING') +
             '  baseCamera ' + (state.baseCamera ? 'ok' : '-') +
             '  urbanAoi ' + (state.urbanAoi ? state.urbanAoi.length : 0));
  lines.push('components      compass ' + state.components.compass +
             '  scalebar ' + state.components.scalebar +
             '  viewpoint ' + state.components.viewpoint);
  if (state.lastError) lines.push('LAST ERROR      ' + String(state.lastError).slice(0, 140));
  box.textContent = lines.join('\n');
}

// --------------------------------------------------------------------------
// 运行时错误捕获
// --------------------------------------------------------------------------

// Cesium 的失败大多是"安静"的：瓦片请求失败、图层挂载异常都只会让画面变差
// 而不抛到界面上。这里把未捕获异常和 Promise 拒绝记下来并显示在状态栏与
// 诊断浮层，避免只能靠猜。
window.addEventListener('error', function (e) {
  state.lastError = (e.message || 'error') + ' @ ' +
    String(e.filename || '').split('/').pop() + ':' + (e.lineno || 0);
  setStatus('运行错误: ' + state.lastError, 'err');
});

window.addEventListener('unhandledrejection', function (e) {
  const r = e.reason;
  state.lastError = 'Promise 拒绝: ' + ((r && r.message) || String(r));
  setStatus('运行错误: ' + state.lastError, 'err');
});

/**
 * 复选框当前状态。
 *
 * 一律现取，不用 `dom.*` 缓存：那个缓存里的复选框引用实测会与文档脱钩
 * （`isConnected === false`），于是"开关是开的"被读成"关的"，建筑体块一路静默
 * 短路——不报错、没有任何提示，看起来就是"这个功能没做"。
 */
/**
 * 读一个开关的当前状态。
 *
 * **两个来源**：DOM 里还有节点就读它；节点已经没有（已迁进注册表的
 * `modelImagery`/`modelTerrain`/`modelWsf3dOverlay`/`modelSensor`/`modelWeather`/
 * `modelBuildings`）就读注册表的**意图**。
 *
 * 不这样做的话这些 id 会一律返回 `false`，而本函数有 8 处调用点在问
 * `checkboxOn('modelBuildings')` —— 后果是 WSF3D 建筑体块**整条链路静默死掉**
 * （`refreshBuildings()` 的每个入口都被它挡着），而且不报错。
 */
/**
 * 复选框 DOM id → 图层 id。
 *
 * **这两个不是一回事**：图层 id 是 `buildings.blocks`，它的复选框 id 是
 * `modelBuildings`。`LayerRegistry.def()` 只按**图层 id** 查，所以拿复选框 id 去问
 * 它永远得到 null —— 一开始我就是这么写的，结果 `?buildings=1` 与
 * `checkboxOn('modelBuildings')` 全都静默失效（不报错，只是没反应）。
 */
function layerIdForCheckbox(id) {
  if (!window.LayerRegistry || !LayerRegistry.LAYERS) return null;
  const hit = LayerRegistry.LAYERS.filter(function (l) { return l.checkbox === id; })[0];
  return hit ? hit.id : null;
}

/**
 * 读一个开关的当前状态。
 *
 * **两个来源**：DOM 里还有节点就读它；节点已经没有（例如 `modelBuildings`，
 * 它的复选框已经不在 HTML 里）就按复选框 id 反查图层、读注册表的**意图**。
 *
 * 否则这些 id 会一律返回 `false`，而本函数有 8 处调用点在问
 * `checkboxOn('modelBuildings')` —— 后果是 WSF3D 建筑体块**整条链路静默死掉**
 * （`refreshBuildings()` 的每个入口都被它挡着），而且不报错。
 */
function checkboxOn(id) {
  const node = document.getElementById(id);
  if (node) return !!node.checked;
  const lid = layerIdForCheckbox(id);
  if (lid) return !!LayerRegistry.runtime[lid].desired;
  return false;
}

function setStatus(text, kind) {
  dom.statusText.textContent = text;
  dom.statusDot.className = 'dot' + (kind ? ' ' + kind : '');
}

/**
 * 图层状态说明。**已按要求改为空实现**。
 *
 * 「去掉整个系统 UI 中的说明性文字，用户只能通过用户手册了解系统操作」——这 40 多处
 * 调用写的都是"某图层为什么这样显示"的解释，正是要拿掉的东西。
 *
 * 保留函数本身而不是删掉 40 个调用点：这些调用分布在每个 setXxx 里，逐个删既容易漏、
 * 又会让"这里本该有一句状态说明"的意图消失；留一个注明原因的空实现，将来若要恢复
 * 状态提示，改一处即可。真实故障仍然有出口——`setStatus()`（顶栏状态点）与
 * `#tdtNote` 不受影响。
 */
function setSwitchNote(text) { /* 见上方说明：说明性文字已从界面移除 */ }

/** 同 setSwitchNote：跟随模式的说明文字已从界面移除。 */
function setFollowNote(text) { /* 见 setSwitchNote 的说明 */ }

function el(tag, className, html) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (html !== undefined) node.innerHTML = html;
  return node;
}

function colorFor(affiliation) {
  // I09：取值来自 `theme.css`（经 `Theme.faction()`），不再是硬编码字面量。
  // `Theme.faction()` 覆盖 red/blue/green/其余→neutral，比原先的二元三元式更完整：
  // 原先任何非 'red' 都会被画成蓝方色，**绿方与未知阵营也会变成蓝方**。
  if (typeof window !== 'undefined' && window.Theme) {
    return Cesium.Color.fromCssColorString(window.Theme.faction(affiliation));
  }
  throw new Error('Theme module unavailable; cannot resolve affiliation colour');
}

function levelColor(level) {
  return ['#7d8a99', '#4a9c6d', '#d8a12a', '#d9772a', '#c8352c'][level] || '#7d8a99';
}

function fmtDeg(v, pos, neg) {
  if (v === null || v === undefined || !isFinite(v)) return '—';
  return Math.abs(v).toFixed(6) + '°' + (v >= 0 ? pos : neg);
}

// --------------------------------------------------------------------------
// 视图初始化
// --------------------------------------------------------------------------

function initViewer() {
  // Cesium 1.107 起 Viewer 的 imageryProvider 选项被移除且静默忽略，
  // 必须用 baseLayer，否则球体只剩默认底色。下面的断言把静默退化变成显式失败。
  //
  // 图层顺序（index 0 在最底层，Cesium 由下往上叠）：
  //   [0] 离线影像垫底（z=5）  →  [1] 天地图影像（z=10，在线，经 /tdt 代理）
  //   [2] 天地图中文注记（z=90，在线）
  //   离线全球 z0-6 → 战区 z0-11 → 热点 z0-12 → 城区 z13-17，全部叠在 [1] 之上
  //
  // 离线与天地图**不再互斥**：离线垫在下面当兜底，天地图有瓦片的地方盖住它。
  // 这样 key 失效 / 离线 / 瓦片在途时露出的是离线影像，而不是球体底色（rgb(0,0,128)
  // ——以前底图"变蓝"就是这个）。顺序由 layerRegistry 的 z 决定，不在这里手工调整。
    // 标签写成 `imagery(offline)` 而不是 `imagery`：这组计数只给**离线底图** provider
  // 包了计数（下面的 instrument），天地图走服务端代理 `/tdt/`、不计入。
  // 标签含糊过一次——`?diag=1` 上写着"imagery requests 0"，而画面上天地图瓦片明明在加载，
  // 查"底图为什么糊"的人会照着这个 0 往"影像根本没加载"的方向查，正好查反。
  state.imageryStats = OfflineAssets.newStats('imagery(offline)');
  state.tiandituLayer = new Cesium.ImageryLayer(
    OfflineAssets.createTiandituProvider('img'));
  state.tiandituLabelLayer = new Cesium.ImageryLayer(
    OfflineAssets.createTiandituLabelProvider());

  viewer = new Cesium.Viewer('cesiumContainer', {
    baseLayer: state.tiandituLayer,
    terrainProvider: new Cesium.EllipsoidTerrainProvider(),
    baseLayerPicker: false,
    geocoder: false,
    homeButton: false,
    sceneModePicker: false,
    navigationHelpButton: false,
    animation: false,
    timeline: false,
    fullscreenButton: false,
    infoBox: false,
    selectionIndicator: false,
    shouldAnimate: false,
  });

  // 按设备像素比渲染，别按 CSS 像素。
  //
  // Cesium 的 `useBrowserRecommendedResolution` 默认是 **true**，含义是"忽略
  // devicePixelRatio，按 CSS 像素渲染再放大"。在 Windows 缩放 125%/150%（很常见）
  // 或任何 HiDPI 屏上，后果是**整幅画面连同所有文字一起发虚**——地名、目标标牌、
  // 天地图注记全都糊。这不是"字号太小"能解释的，也不是加粗描边能救的。
  //
  // 实测本项目的无头环境 devicePixelRatio=1，所以这个改动在自动截图里看不出来，
  // 但在用户的机器上很可能是清晰度的主因。代价是 GPU 负载上升；若某台机器吃不消，
  // 把这一行改回 true 即可（只是又回到发虚的状态）。
  viewer.useBrowserRecommendedResolution = false;

  // 隐藏 Cesium 底部那条「Data attribution」署名条（按界面反馈：底部那道满宽的长条）。
  //
  // ⚠ 这不是"没有署名"：在线底图是天地图（服务端代理的 credit 见 assets.js），
  //   离线底图是 **Esri World Imagery，其许可要求署名**（清单里记的是
  //   "free with attribution; review before operational use"）。署名文字仍然保留在
  //   图层的 credit 元数据里，并作为「底图」分组标题的 title 属性可查（悬停可见）。
  //   若将来对外发布，应当把 Esri 的署名放回可见位置——那时改这一行即可。
  if (viewer.cesiumWidget && viewer.cesiumWidget.creditContainer) {
    viewer.cesiumWidget.creditContainer.style.display = 'none';
  }

  if (viewer.imageryLayers.length === 0) {
    throw new Error('底图图层未创建：Cesium ' + Cesium.VERSION + ' 需要 baseLayer 选项');
  }

  // 注记层紧贴天地图影像之上（index 1）。
  viewer.imageryLayers.add(state.tiandituLabelLayer);

  // 在线 Esri 底图：与天地图并列的**可选底图**（默认关，见注册表 esri.online）。
  // 浏览器直连 Esri（其瓦片带 CORS 头，无需服务端代理，见 assets.js 的说明）；
  // 与其它底图一样**始终存在**，切换只动 .show，不重建 provider。
  state.esriLayer = new Cesium.ImageryLayer(OfflineAssets.createEsriProvider());
  viewer.imageryLayers.add(state.esriLayer);
  // 离线全球层：默认关闭（天地图是默认底图），但**始终存在**，切回离线时只是
  // 把 show 打开，不再重建 provider——避免重复初始化与重复请求。
  // 反过来，离线底图是唯一"离线环境下一定有图"的图层，所以不能删。
  state.globalLayer = viewer.imageryLayers.addImageryProvider(
    OfflineAssets.instrument(OfflineAssets.createGlobalImageryProvider(),
                             state.imageryStats));
  state.globalLayer.show = false;

  // 战区层：台海 AOI 到 z11
  state.imageryLayer = viewer.imageryLayers.addImageryProvider(
    OfflineAssets.instrument(OfflineAssets.createImageryProvider(), state.imageryStats));
  state.imageryLayer.show = false;

  // 热点层：每个热点框一个 provider，到 z12（三亚/冲绳/关岛完全在战区之外，
  // 只能靠这一层）。**不在这里挂**：它由 syncHotspotImagery() 按相机高度决定，
  // 战区级视图（>400 km）下 9 个 provider 的重复代价换不来任何可见收益。
  state.hotspotLayers = state.hotspotLayers || [];

  // ---- 图层注册表 ----------------------------------------------------------
  // 到这里所有**常驻** imagery 层都已挂上（天地图影像/注记、离线全球/战区、
  // 热点）。把顺序交给注册表声明，不再靠 add 的先后 + 事后 raiseToTop：
  //
  //   以前注记被热点/城区影像压住，是加了一次 `raiseToTop(tiandituLabelLayer)`
  //   才救回来的——每加一个新影像层都要重新检查这一处。现在注记在 LAYERS 里
  //   z=90，排在所有底图/高清影像之上，顺序由数据决定。
  //
  // 栈与栈之间 Cesium 没有统一 z 轴（见 layerRegistry.js 文件头），所以三个
  // sink 分别写各自的栈：
  //   imagery   → .show（顺序由 reorderImagery 按 z 重排 index）
  //   primitive → .show（mesh 前后由深度缓冲决定，与 index 无关）
  //   label     → Places.setEnabled（内部再做屏幕空间仲裁）
  state.layerHealth = LayerRegistry.bind(viewer, {
    imagery: function (id, want) {
      // `state.urbanEnabled` 原先只由 `bindCheck('modelUrbanImagery', …)` 维护，而那条
      // 绑定在复选框删除后已经不会触发 —— 于是用户在面板里关掉城区影像，相机高度
      // 一变化它又会被挂回来（挂载判据读的正是 state.urbanEnabled）。
      if (id === 'urban.imagery') state.urbanEnabled = !!want;
      // 同 urban：热点高清与离线高倍段都是**按需挂载**的，挂载判据要读它们，
      // 所以开关一变就记下来 —— 否则用户在面板里关掉，相机一动作又被挂回来。
      if (id === 'hotspot.imagery') state.hotspotImageryEnabled = !!want;
      // 高倍段是**常挂、只切 show** 的（见 ensureOfflineDetailLayers）。这里不能走
      // 下面的通用分支：那会把已挂载的**所有** band 一起显示出来，把"按相机只显示
      // 相关层级"的闸门冲掉。开关从关到开时立即按相机重算一次（不等 1.5 s 周期）。
      if (id === 'imagery.offline_detail') {
        const was = state.offlineDetailEnabled;
        state.offlineDetailEnabled = !!want;
        if (!want) setOfflineDetailShown([]);
        else if (!was) setOfflineDetailShown(offlineDetailWantedZooms());
        return;
      }

      // 这里本来还想把 `setTiandituLabel` / `setHotspotImagery` 里的提示文案接过来，
      // 查证后**没有必要**：`setSwitchNote()` 本身就是一个空函数
      // （`function setSwitchNote(text) { /* 说明性文字已从界面移除 */ }`）——
      // 那些文案早就按"界面不写说明性文字"的要求被有意丢弃了，从来没有显示过。
      // 所以那两个 setter 里除 `setDesired` + `applyLayers`（都已由面板的 onChange
      // 覆盖）之外**没有任何可观察行为**，删掉即可，不必搬运提示。
      // imagery.offline 是两级（全球 z0-6 + 战区 z7-11），都在槽位里，
      // 由下面的通用分支统一翻，不需要特判。
      // 热点层是**一层多实例**（每个热点框一个 provider 各自带 rectangle），
      // 槽位里每个实例都登记了，同样交给通用分支。
      const layers = LayerRegistry.layersOf(id);
      if (layers && layers.length) {
        layers.forEach(function (l) { l.show = !!want; });
        return;
      }
      // 城区层按相机高度动态挂载，卸载后槽位为空，此时无事可做。
    },
    primitive: function (id, want) {
      if (id === 'buildings.blocks') {
        // 直接走内部实现（`applyBuildingsVisibility`）：它只把状态落到场景上、不碰
      // `desired`，所以不会形成回环。`desired` 由注册表那条行统一写 —— 面板的
      // onChange、URL 钩子、恢复默认都走同一条路。
        applyBuildingsVisibility(want);
      } else if (id === 'tiles.google3d') {
        setOnlineTilesetVisible(want);
      } else if (id === 'tiles3d.tw_coastal') {
        // 本地白膜瓦片：挂/卸交给 tiles3d.js，意图与顺序仍归注册表。
        if (window.Local3D) Local3D.setEnabled('tiles3d.tw_coastal', want);
      }
    },
    label: function (id, want) {
      if (id === 'places.labels' && window.Places) Places.setEnabled(!!want);
    },
    // 标绘：与其它栈一样，注册表只说"该不该显示"，怎么显示由 sink 决定。
    // 标绘实例的增删/几何在 Plotting 里，这里只切它所属类别图层的显隐。
    plot: function (id, want) {
      if (window.Plotting) Plotting.setLayerVisible(id, !!want);
    },
  });

  // Plotting 需要 viewer 才能建实体。**必须在 bind 之后显式交进去**——
  // 漏了这一步的表现是"add() 成功、但场景里没有任何东西"（实测 0 个实体），
  // 而且不报错：drawOne 里的 `if (!viewer) return 0` 是静默返回。
  if (window.Plotting) Plotting.bind(viewer);
  // 热点/占领由帧号驱动重建，但**初始的图层意图在下面才写入**：先把开关置开，
  // 否则首帧的 createHotspots() 会因 layers.hotspots=false 建出空集合，
  // 而后的 key（帧号+开关+数量）不再变化，热点就永远不出现（实测）。
  state.layers.hotspots = true;
  // 标绘图层由**实际存在的标绘要素**动态驱动：Plotting 每次增删改后通知这里，
  // 由 syncPlotLayers() 增删对应的 kind:'plot' 图层（见该函数的说明）。
  if (window.Plotting && Plotting.onChange) {
    Plotting.onChange(function () { syncPlotLayers(); });
    syncPlotLayers();
  }

  // 常驻 imagery 槽位登记（动态层在挂载时登记，见 registerImagerySlot）。
  registerImagerySlot('tianditu.base', state.tiandituLayer);
  registerImagerySlot('tianditu.label', state.tiandituLabelLayer);
  // 在线 Esri 底图（默认关，但常驻，顺序由注册表的 z=11 决定）
  registerImagerySlot('esri.online', state.esriLayer);
  // 离线底图是**两级**：全球层 z0-6 + 战区层 z7-11，共用一个 imagery.offline 意图。
  // 两个都要登记——只登记全球层的话，战区层会留在顺序之外，反而压到注记上面。
  registerImagerySlot('imagery.offline', [state.globalLayer, state.imageryLayer]);
  // 离线底图的**高倍段**（z14+）与热点高清**都不在启动时挂**：它们由
  // syncOfflineDetail() / syncHotspotImagery() 按"当前实际渲染层级 / 相机高度"
  // 决定，首次由 applySituationData 与 camera.changed 触发（见那两处的说明）。
  // 启动时挂上任何一层，Cesium 都会立刻为每个表面瓦片去要一遍深层瓦片 —— 那正是
  // "半个屏幕在加载"的来源。
  // 高倍段（z14+）在这里**一次性挂好并隐藏**：之后只切 show，不再 add/remove
  // （原因见 ensureOfflineDetailLayers）。启动时挂不会引起用户可见的重建。
  ensureOfflineDetailLayers();
  registerImagerySlot('imagery.offline_detail', state.offlineDetailLayers);
  // 热点影像是**一层多实例**（每个热点框一个 provider），挂载时才登记。
  registerImagerySlot('hotspot.imagery', state.hotspotLayers);

  registerLayerProviders();

  // 初始意图：**遍历注册表**取 `default`，不再逐个去读 HTML 复选框。
  //
  // 面板改由注册表生成之后，HTML 里已经没有这些复选框了；默认值的唯一来源必须是
  // 注册表，否则就会出现"面板画出来是关的、而代码以为它开着"这类两边不一致。
  // 原先逐个 `checkboxOn('modelXxx')` 的写法还漏过一个：`urban.imagery` 从头到尾
  // 没被同步过，HTML 里勾着、注册表里却是关的。
  // Step C: register the relation-line layer before the default pass below, so
  // `default: true` travels the same setDesired/apply path as every other row.
  registerRelationsLayer();
  LayerRegistry.LAYERS.forEach(function (d) {
    LayerRegistry.setDesired(d.id, d.default === true);
  });
  // 立刻应用一次：把 imagery 的 index 按 LAYERS 的 z 排好。这一步取代了以前那句
  // raiseToTop(tiandituLabelLayer)——注记现在由 z=90 直接排到最上。
  applyLayers();

  // 完全离线：不触碰 Ion 资产，不使用 Bing/在线地形服务。
  viewer.scene.globe.enableLighting = false;
  viewer.scene.globe.showGroundAtmosphere = true;
  viewer.scene.skyAtmosphere.show = true;
  viewer.scene.fog.enabled = false;
  viewer.scene.globe.depthTestAgainstTerrain = false;

  // 关闭 Cesium 自带的双击跟踪与默认缩放行为，避免与信息面板抢事件。
  viewer.cesiumWidget.screenSpaceEventHandler.removeInputAction(
    Cesium.ScreenSpaceEventType.LEFT_DOUBLE_CLICK);

  if (window.Overview) Overview.init(viewer);
  if (window.Places) Places.init(viewer);
  // 在线三维实景只是登记 viewer，真正的联网在建 tileset 时才发生（见 bindUI），
  // 这样"没配置令牌"的部署不会在开页面时就发一次注定 401 的请求。
  if (window.Buildings3D) Buildings3D.init(viewer);
  // 本地 3D Tiles 只登记 viewer；tileset 在用户勾选时才开始加载（见 tiles3d.js）。
  if (window.Local3D) Local3D.init(viewer);
}

/**
 * 飞到某个经纬点，并让该点**正处视窗中心**。
 *
 * 旧实现 `flyTo({destination: fromDegrees(lon, lat, h)})` 把**相机位置**当成了
 * 目标位置：带俯仰角时视窗中心落在目标前方，目标跑到屏幕侧下方（实测 900 km
 * 下偏 200 px）。这里保持同一套"相机在目标上方 h 米"的机位语义，但改为
 * `headingPitchRange`（以目标为原点的球坐标）——目标点恒在视窗中心。
 */
function flyTo(lon, lat, height, pitchDeg) {
  // 先解除跟随：camera.lookAt 会把 camera.transform 锁在目标 ENU 上，而
  // setView / flyTo 不会自动解锁，orientation 会被按那个局部系解释。
  detachFollow('视角跳转');
  const pitch = Cesium.Math.toRadians(pitchDeg === undefined ? -70 : pitchDeg);
  const h = Math.max(1000, Number(height) || 180000);
  const target = Cesium.Cartesian3.fromDegrees(lon, lat, 0);
  // 斜距：目标点正上方 h、俯仰 pitch 时的视线距离 = h / sin(-pitch)。
  const range = Math.abs(Math.cos(pitch)) < 1e-3 ? h : h / Math.sin(-pitch);
  // ⚠ 用 `setView` 而不是 `flyTo` / `flyToBoundingSphere`：实测本页的
  // `RequestAnimationFrame` 渲染循环在无头/低帧率下不会推进 Cesium 的飞行
  // 动画（`flyTo` 立即完成但机位不变），定位会静默失效。setView 是同步的，
  // 且 `lookAt` 自带"目标居视窗中心"的语义。
  viewer.camera.lookAt(target, new Cesium.HeadingPitchRange(0, pitch, range));
  // lookAt 会把相机锁在目标 ENU 上；这里必须**解除锁定**（transform 置单位阵），
  // 否则后续旋转/缩放会被解释为绕目标公转。判定不能用 `Matrix4.IDENTITY` ——
  // 那是同一个常量对象，clone 进去等于没改；用"取出来比较是否仍是上一次的锁"更稳。
  if (!Cesium.Matrix4.equals(viewer.camera.transform, Cesium.Matrix4.IDENTITY)) {
    viewer.camera.lookAtTransform(Cesium.Matrix4.IDENTITY);
  }
}

// --------------------------------------------------------------------------
// 图层自动定位：用户打开图层后，平滑飞到该图层的空间范围
// --------------------------------------------------------------------------
//
// 需求（用户 2026-10-19）：选择并加载图层后，视点自动平滑移到图层中心，并按
// 覆盖范围和视窗尺寸给出合适缩放，让内容完整、清晰地显示。
//
// 为什么用 `flyTo({destination: Rectangle})` 而不是自己算高度：
// Cesium 内部的 `Camera.getRectangleCameraCoordinates` 会按**当前视窗宽高比**和
// 视锥计算能完整容纳矩形的位置，这正是"自适应缩放"要的；且 `flyTo` 自带平滑
// 动画。自己用 span → 高度公式反而在不同宽高比下会裁掉内容。
//
// 触发点只有**面板的 onChange**（用户显式打开图层）。URL 钩子、批量「全部显示/
// 隐藏/恢复默认」、以及按相机高度自动挂载/卸载（syncHotspot/syncUrban/
// syncOfflineDetail）都**不触发飞行** —— 否则相机自己动一下就被自己的 LOD 逻辑
// 拽走，交互会打架。
const LAYER_FOCUS_MAX_SPAN_DEG = 60;   // 跨度超过它视为全球底图，不做自动定位
// 正在进行的图层定位飞行。token 用于**作废旧飞行的回调**：连续点两个图层时，
// Cesium 会取消上一个飞行并触发它的 cancel 回调，若没有 token，取消回调会把
// 新飞行的状态一起清掉（随后用户点画布就无法中断新飞行）。
let layerFlight = { token: 0, id: null };

/** 把 [w,s,e,n]（度）并入累加器；非法值忽略。 */
function _unionBbox(acc, b) {
  if (!Array.isArray(b) || b.length !== 4) return;
  const w = Number(b[0]), s = Number(b[1]), e = Number(b[2]), n = Number(b[3]);
  if (![w, s, e, n].every(isFinite)) return;
  acc.w = Math.min(acc.w, w);
  acc.s = Math.min(acc.s, s);
  acc.e = Math.max(acc.e, e);
  acc.n = Math.max(acc.n, n);
}

/**
 * 不做自动定位的图层：它们的显示/挂载由**相机高度**驱动
 * （syncHotspotImagery / syncUrbanImagery / syncOfflineDetail）。
 * 若在打开时把相机拖到"看全整个图层"的高度，很可能越过这些层自己的挂载阈值，
 * 落地后立刻被卸载 —— 用户看到的是"飞过去又消失"。把它们排除，交给原有的
 * 相机驱动逻辑，才是 2026-10-17 那轮"按相机挂/卸"的既定设计。
 */
const LAYER_FOCUS_EXCLUDED = {
  'hotspot.imagery': true,
  'urban.imagery': true,
  'imagery.offline_detail': true,
};

/**
 * 某图层的空间范围 [W,S,E,N]（度）。取不到或等于全球范围时返回 null。
 *
 * 优先用注册表声明的 `bbox`（primitive/globe 层没有 provider 矩形）；否则
 * 合并已挂载 provider 的 `rectangle`（例如 WSF3D 覆盖层）。
 */
function layerExtent(id) {
  if (LAYER_FOCUS_EXCLUDED[id]) return null;
  const def = window.LayerRegistry && LayerRegistry.def ? LayerRegistry.def(id) : null;
  if (def && Array.isArray(def.bbox) && def.bbox.length === 4) {
    return def.bbox.slice();
  }
  const acc = { w: Infinity, s: Infinity, e: -Infinity, n: -Infinity };
  const layers = (window.LayerRegistry && LayerRegistry.layersOf)
    ? LayerRegistry.layersOf(id) : [];
  layers.forEach(function (L) {
    const r = L && L.imageryProvider && L.imageryProvider.rectangle;
    if (!r) return;
    _unionBbox(acc, [Cesium.Math.toDegrees(r.west), Cesium.Math.toDegrees(r.south),
                     Cesium.Math.toDegrees(r.east), Cesium.Math.toDegrees(r.north)]);
  });
  if (!isFinite(acc.w)) return null;
  if (acc.e < acc.w) return null;                      // 跨 180°：本项目没有这种层
  if (acc.e - acc.w > LAYER_FOCUS_MAX_SPAN_DEG
      || acc.n - acc.s > LAYER_FOCUS_MAX_SPAN_DEG) {
    return null;                                       // 全球底图：不抢视点
  }
  return [acc.w, acc.s, acc.e, acc.n];
}

/**
 * 平滑飞到图层范围。返回 true 表示真的发起了飞行。
 *
 * 外扩 12%（最小 0.004°）作为留白，避免内容贴边；`flyTo` 的 `cancel` 回调与
 * 画布上的 pointer/wheel 中断保证用户随时能接管，不会"点了一下图层就被锁死视点"。
 */
function focusLayer(id) {
  if (!viewer || !window.Cesium) return false;
  const ext = layerExtent(id);
  if (!ext) return false;
  const w = ext[0], s = ext[1], e = ext[2], n = ext[3];
  const padX = Math.max((e - w) * 0.12, 0.004);
  const padY = Math.max((n - s) * 0.12, 0.004);
  const rect = Cesium.Rectangle.fromDegrees(w - padX, s - padY, e + padX, n + padY);
  detachFollow('图层定位');
  const token = ++layerFlight.token;
  layerFlight.id = id;
  try {
    viewer.camera.flyTo({
      destination: rect,
      duration: 1.6,
      easingFunction: Cesium.EasingFunction.QUINTIC_IN_OUT,
      complete: function () {
        if (token === layerFlight.token) layerFlight.id = null;
      },
      cancel: function () {
        if (token === layerFlight.token) layerFlight.id = null;
      },
    });
  } catch (e2) {
    if (token === layerFlight.token) layerFlight.id = null;
    return false;
  }
  return true;
}

/** 用户一碰画布就取消正在进行的图层定位飞行（不干扰其它交互）。 */
function cancelLayerFlightOnInput() {
  if (!layerFlight.id) return;
  layerFlight.id = null;
  layerFlight.token++;                 // 让被取消飞行的回调作废
  try { viewer.camera.cancelFlight(); } catch (e) { /* 没有活动飞行时忽略 */ }
}

function applyCamera(camera) {
  if (!camera) return;
  detachFollow('复位机位');
  viewer.camera.setView({
    destination: Cesium.Cartesian3.fromDegrees(camera.lon, camera.lat, camera.height),
    orientation: {
      heading: Cesium.Math.toRadians(camera.heading || 0),
      pitch: Cesium.Math.toRadians(camera.pitch === undefined ? -75 : camera.pitch),
      roll: 0,
    },
  });
}

// --------------------------------------------------------------------------
// 数据加载
// --------------------------------------------------------------------------

/**
 * 案例索引来自**服务端扫描固定目录**（`/api/situations`），不再是生成出来的
 * `web/data/cases.json`：往目录里放一份 TS 就多一个案例、删掉就少一个，
 * 既不用跑生成器，也不用改任何索引文件。
 *
 * `errors` **不吞掉**：坏文件必须报出来，否则现象只是"某个案例莫名不见了"，
 * 而没人会想到是文件本身解析失败。空目录是正常状态，但也要说清"没有可用的 TS"，
 * 不能让菜单一片空白还查不出原因。
 */
function loadCaseIndex() {
  return fetch('/api/situations', { cache: 'no-store' })
    .then(function (r) {
      if (!r.ok) throw new Error('TS 目录扫描失败: HTTP ' + r.status);
      return r.json();
    })
    .then(function (index) {
      state.caseIndex = index;
      var errs = index.errors || [];
      if (errs.length) {
        console.warn('[situations] ' + errs.length + ' 份文件未被载入：', errs);
      }
      if (!index.cases || !index.cases.length) {
        console.warn('[situations] 目录 ' + (index.dir || '')
          + ' 下没有可用的 TS（放一份 *.json 进去即成为一个案例）');
      }
      renderCaseMenu();
      return index;
    });
}

function loadCase(caseId) {
  const entry = state.caseIndex.cases.find(function (c) { return c.id === caseId; });
  if (!entry) throw new Error('未知案例: ' + caseId);

  setStatus('加载TS数据…');
  // 2026-10-19：案例拆成 core / inference 两个分片（见 serve.py 的 _case_parts）。
  // taiwan_landing.json 39.6 MB 里 inference_frames 占 35 MB / 1441 帧，而首屏
  // 渲染只需要 targets/frames。先拿 core 立即出场景，inference 在后台补进右栏；
  // 接口不可用时（旧服务、直接开静态文件）回退整包，行为与以前一致。
  return fetch('/api/situation?id=' + encodeURIComponent(caseId) + '&part=core',
               { cache: 'no-store' })
    .then(function (r) {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.json();
    })
    .then(function (data) {
      applySituationData(data, caseId);
      state.inference = null;
      if (dom.rlNote) dom.rlNote.textContent = '推理结果加载中…';
      // 不 return：推理帧后台加载，loadCase 在 core 应用完就 resolve（首屏不等它）。
      loadCaseInference(caseId);
      return data;
    })
    .catch(function (splitErr) {
      console.warn('[situations] 分片接口不可用，回退整包：', splitErr);
      return fetch('./' + entry.file, { cache: 'no-store' })
        .then(function (r) {
          if (!r.ok) throw new Error('TS数据加载失败: HTTP ' + r.status);
          return r.json();
        })
        .then(function (data) { return applySituationData(data, caseId); });
    });
}

/**
 * 后台取推理帧并补进场景状态。
 *
 * 只做"补数据"：场景、实体、时间轴已经由 core 建好；这里完成后刷新右栏与
 * 目标状态面板。切走案例时（state.caseId 变了）丢弃晚到的结果。
 */
function loadCaseInference(caseId) {
  return fetch('/api/situation?id=' + encodeURIComponent(caseId) + '&part=inference',
               { cache: 'no-store' })
    .then(function (r) {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.json();
    })
    .then(function (arr) {
      if (state.caseId !== caseId) return;          // 已切换案例
      state.inference = Array.isArray(arr) ? arr : (arr && arr.inference_frames) || null;
      if (state.data && !state.data.inference_frames) {
        state.data.inference_frames = state.inference;
      }
      updateInferenceUi();
      markScene('inference');
      setStatus('TS已更新（推理帧已载入）', 'ok');
    })
    .catch(function (e) {
      if (state.caseId !== caseId) return;
      if (dom.rlNote) dom.rlNote.textContent = '推理结果加载失败：' + e.message;
      if (window.console && console.warn) console.warn('[inference] 加载失败', e);
    });
}

/**
 * 把一份TS数据装载进场景。
 *
 * 从 loadCase 里拆出来，是为了让**前端生成的TS**（TS输入）走完全同一条管线：
 * 取数据与装载数据分开，两条来源共用装载逻辑，就不会出现"案例能跑、自建TS
 * 少初始化了某样东西"的偏差。
 */
/**
 * Remount the hotspot imagery layers for the current case's theatre.
 *
 * They are built once in `initViewer` from `manifest.hotspot_imagery.boxes`, which
 * is case-independent, so after switching to the Ladakh case the ten Taiwan Strait
 * hotspot providers stayed mounted and kept requesting tiles at 117-145E.
 *
 * Select by **identity, not geometry**. A bounding-box test cannot do this job:
 * `manifest.hotspot_imagery.boxes` are anonymous coordinate arrays, and the Taiwan
 * case deliberately keeps four hotspots *outside* its theatre box (三亚, 冲绳岛,
 * 关岛, 羽田) that the code comments say are reachable only through this layer, so
 * filtering by box would hide four intended layers.
 *
 * The case dataset, however, carries `hotspots[].name` and `.bbox`, so the boxes
 * the app should mount can be derived by matching those bboxes against the
 * manifest's list. Boxes matching no declared hotspot belong to another theatre
 * and are skipped.
 */
function remountHotspotImagery() {
  if (!viewer || !window.OfflineAssets) return 0;
  // 相机太高时**不挂**：热点高清是 z0-12 的一层多实例（每个热点框一个 provider，
  // 台海案例有 9 个），常挂时每个表面瓦片都要为每个框去要一遍 —— 在 1939 km 的
  // 战区级视图下纯属浪费（那些瓦片本来也不会被看到）。降到阈值以下再由
  // syncHotspotImagery() 挂回来。
  if (!hotspotImageryWanted()) {
    unmountHotspotImagery();
    return 0;
  }
  unmountHotspotImagery();

  const boxes = (OfflineAssets.manifest().hotspot_imagery || {}).boxes || [];
  const declared = (state.data && state.data.hotspots) || [];
  let want = boxes;

  if (declared.length) {
    const key = function (b) {
      return [b[0], b[1], b[2], b[3]].map(function (v) {
        return Math.round(Number(v) * 1000) / 1000;
      }).join(',');
    };
    const wanted = {};
    declared.forEach(function (h) { if (h && h.bbox) wanted[key(h.bbox)] = h.name; });
    const matched = boxes.filter(function (b) { return wanted[key(b)] !== undefined; });
    if (matched.length) {
      want = matched;
    } else {
      // No coordinate match: mount everything rather than nothing. Hiding
      // hotspots would look identical to "this case has no hotspots", and the
      // manifest and dataset come from different tools so a rounding mismatch is
      // plausible.
      //
      // Measured, this branch is what currently runs for the Ladakh case, because
      // that theatre's hotspots have not been baked into the manifest at all
      // (`manifest.hotspot_imagery.boxes` holds 9 Taiwan boxes and 0 of the 7
      // declared Ladakh ones, so the match count is 0). The ten Taiwan providers
      // therefore stay mounted while the Ladakh case is shown. That is a
      // consequence of the missing bake, not of this selection logic: once
      // `fetch_base_imagery.py` has baked the india_china parks, the boxes will
      // exist, the match will succeed and only the declared ones will mount.
      state.hotspotMatch = { declared: declared.length, boxes: boxes.length,
                             matched: 0, fallback: true };
      if (window.console && console.warn) {
        console.warn('no manifest hotspot box matches this case\'s declared '
                     + 'hotspots (' + declared.length + ' declared, ' + boxes.length
                     + ' in manifest); mounting all. Expected until the theatre '
                     + 'has been baked.');
      }
    }
  } else {
    state.hotspotMatch = { declared: 0, boxes: boxes.length, matched: boxes.length,
                           fallback: false };
  }

  state.hotspotLayers = OfflineAssets.attachHotspotImagery(viewer, want);
  registerImagerySlot('hotspot.imagery', state.hotspotLayers);
  state.hotspotImageryMounted = true;
  return state.hotspotLayers.length;
}

// --------------------------------------------------------------------------
// 按需挂载：热点高清影像 / 离线高倍段
// --------------------------------------------------------------------------
//
// 两者都遵循 urban.imagery 已经用过的那条做法：**相机用不到的层就不挂**。
// 为什么这件事重要：每多挂一个 imagery 层，Cesium 就要为**每个表面瓦片**去要一遍
// 该层的瓦片 —— 代价是相乘的。实测（2026-10-15）把高倍段常挂之后，1939 km 机位
// 下 Cesium 准备了 28.7 万个影像瓦片，加载徽标停在"剩余 2"半小时不动。
//
// 判据不写死高度：**高倍段按"当前实际在渲染的表面层级"选层**（同一个数据源
// 诊断面板已经在用），所以相机一动就自动对齐，不需要维护"多少米对应哪一级"的换算表。
// 热点影像用高度阈值即可 —— 它是战区级的一整层，本来就没有层级概念。

// 热点高清影像的挂载阈值。取 400 km：战区级视图（数百公里以上）看不到热点框内的
// 米级细节，而那一层是 9 个 provider，挂着就是 9 份重复代价。
const HOTSPOT_IMAGERY_TRIGGER_M = 400000;

function hotspotImageryWanted() {
  if (!state.hotspotImageryEnabled) return false;
  if (!viewer) return false;
  // 地球不可见时这张影像也画不出来 —— 别为不可见的东西取瓦片（与高倍段同一条规则）。
  if (viewer.scene && viewer.scene.globe && !viewer.scene.globe.show) return false;
  // 相机需要的层级已经明显超过这一层的最高层级时不挂：它在这个高度只能提供
  // **用不上的粗瓦片**。实测 8 km（需要 z14）时热点层（z≤12）仍全量拉取，
  // 15 秒内 /tiles/imagery/ 138 次请求、9 个长帧、p99 150 ms —— 这就是 8 km 的卡顿源。
  if (layerTooCoarseForCamera(hotspotLayerMaxZoom())) return false;
  return viewer.camera.positionCartographic.height <= HOTSPOT_IMAGERY_TRIGGER_M;
}

/** 热点层的最高层级（manifest 里声明，默认 12）。 */
function hotspotLayerMaxZoom() {
  try {
    const m = window.OfflineAssets ? OfflineAssets.manifest() : null;
    return ((m && m.hotspot_imagery) || {}).max_zoom || 12;
  } catch (e) {
    return 12;
  }
}

/** 摘掉全部热点高清层并注销槽位（幂等）。 */
function unmountHotspotImagery() {
  (state.hotspotLayers || []).forEach(function (L) {
    try { viewer.imageryLayers.remove(L, true); } catch (e) { /* already gone */ }
  });
  state.hotspotLayers = [];
  state.hotspotImageryMounted = false;
  if (window.LayerRegistry) LayerRegistry.dropImagery('hotspot.imagery');
}

/** 相机高度跨过阈值时挂/卸热点高清影像。与 syncUrbanImagery 同构。 */
function syncHotspotImagery() {
  if (!viewer) return;
  const want = hotspotImageryWanted();
  if (want === state.hotspotImageryMounted) return;   // 状态没变，不动场景
  if (want) {
    remountHotspotImagery();
    applyLayers();
  } else {
    unmountHotspotImagery();
    applyLayers();
  }
}

/**
 * 当前视野"该用哪一级影像"。由相机几何算出，**不依赖渲染器状态**。
 *
 * 为什么不用 `globe._surface._tilesToRender`：那是渲染器的内部状态，在两种情况下
 * 拿不到有用值 —— 页面刚起来还没画第一帧、以及软件光栅化（无头）环境里整帧都没渲染。
 * 实测：无头 CDP 下它恒为 -1，于是按它选层的实现**一层都不会挂**（低空该有高清影像
 * 却没有）。相机几何是随时可算的，与环境无关，也便于用脚本断言。
 *
 * 公式：地面分辨率 = 2·h·tan(fovy/2) / 视口高度（米/像素），
 *       层级 = log2(156543.033928 / 地面分辨率)（156543.033928 是 z0 在赤道上的米/像素）。
 * 它给的是量级正确的层级；调用方再留 ±2/±3 的层带余量，所以不必追求逐帧精确。
 */
/**
 * 这一层在**当前相机层级**下是否已经无力贡献。
 *
 * 判据只用相机，不用 `surfaceRenderLevel()` —— 理由与高倍段选层完全一样
 * （见 offlineDetailWantedZooms 的长注释）：挂/卸会让表面重建，用渲染读数当判据
 * 就会自激。这里取 `cam >= maxZoom + 1`：相机已经需要比这一层更细的细节时，
 * 它的瓦片只能被拉伸显示，却仍然要为每个表面瓦片算一遍（台海热点层有 9 个 provider）。
 * 让位不会留空洞 —— 天地图 img/cia 是**全球底图**，这些框内同样有瓦片。
 * 实测（8 km/台北，cam=13）：热点层（z≤12）在此闸门下落位，城区层（z≤13）保留
 * （13 正是这一屏需要的层级）。
 */
function layerTooCoarseForCamera(maxZoom) {
  if (!(maxZoom > 0)) return false;
  const cam = cameraImageryLevel();
  if (cam < 0) return false;          // 拿不到层级不闸（保守）
  return cam >= maxZoom + 1;
}

function cameraImageryLevel() {
  try {
    if (!viewer) return -1;
    const cam = viewer.camera;
    const h = cam.positionCartographic.height;
    if (!(h > 0)) return -1;
    const canvas = viewer.scene.canvas;
    const px = Math.max(1, canvas.clientHeight || canvas.height || 800);
    const fovy = (cam.frustum && cam.frustum.fovy) ? cam.frustum.fovy : (60 * Math.PI / 180);
    const groundRes = (2 * h * Math.tan(fovy / 2)) / px;
    if (!(groundRes > 0)) return -1;
    const lvl = Math.log2(156543.033928 / groundRes);
    if (!isFinite(lvl)) return -1;
    return Math.max(0, Math.min(22, Math.round(lvl)));
  } catch (e) {
    return -1;
  }
}

/**
 * 当前**实际在渲染**的表面瓦片层级；拿不到时返回 -1。
 *
 * 只作为相机估算的**上调**参考：渲染器真在画更深层时，说明视野确实需要更细的影像。
 */
function surfaceRenderLevel() {
  try {
    const surface = viewer && viewer.scene && viewer.scene.globe && viewer.scene.globe._surface;
    const tiles = surface && surface._tilesToRender ? surface._tilesToRender : [];
    if (!tiles.length) return -1;
    let mx = -1;
    tiles.forEach(function (t) { if (t.level > mx) mx = t.level; });
    return mx;
  } catch (e) {
    return -1;
  }
}

// 高倍段层带：围绕**正在渲染的层级 S** 取 [S-1, S+1]。
//
// 2026-10-17 第三次订正（前两版都被实测否掉了）：
//   · 第一版 `max(相机, 渲染) ± (2,3)`：20 km 时挂 z14/15/16 → 一个 z11 表面瓦片要
//     64 张 z14 才盖满 → **969 次请求 / 144 s / fps 2**。
//   · 第二版以 S 为锚但**只在 camera.changed 时重算**：S 是相机停稳后才追上来的，
//     之后没有事件 → 窗口永远差一档 → 4 km 下 mounted 恒为空（高倍段等于不存在）。
//   · 第三版改成相机口径 + S 做刹车：8 km 时表面只渲染 z12 却挂了 z14
//     （一个 z12 瓦片要 16 张 z14）→ 87 次 /tiles_db 请求 + 反复重建 → fps 7.6。
// 现在的组合（第四版，前三版都被实测否掉，过程见 offlineDetailWantedZooms 的注释）：
//   · 相机口径 [cam-2, cam+1] 决定层带在哪一段（稳定、不受渲染快慢影响）；
//   · 表面**峰值** `S_peak + 1` 决定能不能真的挂（不预订表面还画不到的细节）；
//   · 已挂的层粘滞（除非相机走远或比表面峰值深超过 2 级）—— 打破挂/卸自激；
//   · 加上"读不到 S 就保持现状"与 1.5 s 周期重算两条护栏。
const DETAIL_BAND_CAM_BELOW = 2;
const DETAIL_BAND_CAM_ABOVE = 1;
const DETAIL_BAND_S_ABOVE = 1;

// 高倍段集合变更的阻尼：上一次算出的候选集合。连着两次算出同一结果才真正切换，
// 避免表面层级抖动导致的"挂→重建→卸"自激（见 syncOfflineDetail 的说明）。
let pendingDetailZooms = null;

/**
 * 当前应挂的高倍段层级（从 manifest 的 bands 里挑）。
 *
 * **判据只看相机层级**，不看 `surfaceRenderLevel()`。这是被实测逼出来的最终形态：
 *
 *   挂/卸一层影像会让 Cesium 重建表面影像，重建期间 `_tilesToRender` 掉到极小值
 *   （本环境实测掉到 0 并要 ~10 s 才重新精化回来）。于是任何"以渲染层级为判据"的规则
 *   都会用自己的副作用当输入，形成自激：挂 z14 → 表面重建 → S 掉 → 判为"不该挂"→ 卸 →
 *   又重建 → S 掉 → 再挂… 实测 4 km 下 applyLayers 11 次、p95 帧时 300 ms、
 *   挂上又消失反复出现；而按相机判时同一个机位**一次决策、之后不动**。
 *
 * 规则（含 1 级迟滞，避免在阈值上抖动）：
 *   · 挂载：`cam - 2 <= z <= cam`（视野确实需要这一档，且不比它粗太多）
 *   · 保持：`cam - 2 <= z <= cam + 1`（比需要细 1 级也留着，别在阈值上来回切）
 *   · 于是 20 km（cam 13）**一层都不挂** —— 原版"预订 z14/15/16"造成的 969 次请求
 *     在这里从规则上就不可能发生；8 km（cam 14）挂 z14；4 km（cam 15）挂 z14,15。
 *
 * 代价（明说）：相机层级比表面实际渲染层级深时，会为"表面还没画到的细节"下单 ——
 * 但那是**相机（人的意图）说要看那么细**，而不是渲染器的一时快慢决定的。
 */
function offlineDetailWantedZooms() {
  if (!state.offlineDetailEnabled || !window.OfflineAssets) return [];
  // 地球不可见（例如被在线三维实景层让位）时一张都不订：栅格图层根本画不出来。
  // 实测该状态下 12 s 内仍有 232 次高倍段请求。
  if (!viewer || !viewer.scene || !viewer.scene.globe || !viewer.scene.globe.show) return [];
  const all = OfflineAssets.offlineDetailBands();
  if (!all || !all.length) return [];
  const cam = cameraImageryLevel();
  if (cam < 0) return (state.offlineDetailZooms || []).slice();
  const mountLo = cam - DETAIL_BAND_CAM_BELOW;
  const keepLo = cam - DETAIL_BAND_CAM_BELOW;
  const keepHi = cam + DETAIL_BAND_CAM_ABOVE;

  // ① 已挂且仍在容许带内 → 原样保持（迟滞区：比需要细 1 级也不急着卸）。
  const mounted = (state.offlineDetailZooms || []).slice();
  if (mounted.length && mounted.every(function (z) { return z >= keepLo && z <= keepHi; })) {
    return mounted;
  }
  // ② 重算：只挂"相机需要的档"。
  return all
    .filter(function (b) { return b.zoom >= mountLo && b.zoom <= cam; })
    .map(function (b) { return b.zoom; })
    .sort(function (a, b) { return a - b; });
}

/**
 * 把 manifest 声明的全部高倍段**挂一次、常驻**，初始全部隐藏。
 *
 * 为什么不再随相机 add/remove（2026-10-19 改）
 * --------------------------------------------
 * Cesium 对 imageryLayers 的 add/remove 会让表面瓦片**整体重建**：实测在 1.5 km
 * 挂上 z14/15/16 三个 band 后，表面层级从 14 掉回 0/1/2，再花十几秒才重新细化，
 * 期间画面持续卡顿。相机每跨一档就重建一次，就是"越低越卡"的直接来源。
 *
 * 改成常挂之后，层级切换只写 `ImageryLayer.show`（隐藏层在 Cesium 的
 * `_createTileImagerySkeletons` 里直接跳过，不产生任何瓦片请求），既不重建表面，
 * 也不会反复登记/注销槽位。
 *
 * 低层级爆炸的防护在 assets.js 的 attachOfflineDetailBand：provider 从 z0 起，
 * 低层级用祖先瓦片/透明占位兜住，所以"挂上时表面层级还很低"不会再枚举整层瓦片。
 */
function ensureOfflineDetailLayers() {
  if (!viewer || !window.OfflineAssets) return;
  if (state.offlineDetailLayers && state.offlineDetailLayers.length) return;
  const bands = OfflineAssets.offlineDetailBands();
  if (!bands || !bands.length) return;
  const recs = [];
  bands.forEach(function (band) {
    const rec = OfflineAssets.attachOfflineDetailBand(viewer, band);
    if (!rec) return;
    rec.layer.show = false;
    recs.push(rec);
  });
  state.offlineDetailLayers = recs;
  registerImagerySlot('imagery.offline_detail',
    recs.map(function (r) { return r.layer; }));
}

/** 只写 show：`zooms` 是当前应该显示的高倍段层级集合。 */
function setOfflineDetailShown(zooms) {
  const keep = {};
  (zooms || []).forEach(function (z) { keep[z] = true; });
  (state.offlineDetailLayers || []).forEach(function (rec) {
    rec.layer.show = !!keep[rec.zoom];
  });
  state.offlineDetailZooms = (zooms || []).slice();
}

/**
 * 按相机层级切换高倍段显隐（不再增删图层）。
 *
 * 变更仍保留"连续两次算出同一结果才执行"的阻尼：`_tilesToRender` 在相机移动中
 * 会短暂读不到，直接跟随会抖。代价是层级变化晚一拍生效（约 1.5~3 s）。
 */
function syncOfflineDetail(force) {
  if (!viewer || !window.OfflineAssets) return;
  ensureOfflineDetailLayers();
  if (!state.offlineDetailEnabled) {
    if ((state.offlineDetailZooms || []).length) setOfflineDetailShown([]);
    return;
  }
  const want = offlineDetailWantedZooms();
  const have = state.offlineDetailZooms || [];
  if (want.length === have.length && want.every(function (z, i) { return z === have[i]; })) {
    pendingDetailZooms = null;
    return;
  }
  if (!force) {
    if (!pendingDetailZooms || pendingDetailZooms.join(',') !== want.join(',')) {
      pendingDetailZooms = want;
      return;
    }
  }
  pendingDetailZooms = null;
  setOfflineDetailShown(want);
}

/**
 * Point the offline regional layers at the case's theatre.
 *
 * The two theatres (Taiwan Strait, India-China western sector) are disjoint, and
 * a single manifest bounding box covering both would span most of Asia: the
 * providers would advertise imagery across that whole rectangle and spend the
 * session chasing tiles that were never baked. Rebuilding the region layer on a
 * theatre change keeps the declared rectangle equal to what actually exists.
 *
 * The global basemap is deliberately *not* rebuilt - its coverage is global at
 * low zoom and is shared by both theatres.
 */
function switchTheatre(theatre) {
  if (!theatre || !window.OfflineAssets || !OfflineAssets.setTheatre) return;
  const prev = OfflineAssets.setTheatre(theatre);
  const changed = prev !== theatre;
  if (changed && (!viewer || !state.imageryLayer)) return;
  if (!changed) {
    // The theatre is unchanged, but the *case* may differ and the hotspot layer is
    // selected from the case's declared hotspots. Without this, the first load of
    // a non-default case kept the previously mounted hotspot set (measured: the
    // Ladakh case showed 9 Taiwan hotspot providers because the initial mount in
    // `initViewer` is case-independent and this path returned early).
    // 这里同样要重排：remountHotspotImagery 会把新层 add 到 imageryLayers 末尾，
    // 不 apply 就会临时压在注记之上（相机不动时这个"临时"可能一直不消失）。
    remountHotspotImagery();
    applyLayers();
    return;
  }
  try {
    const wasShown = state.imageryLayer.show;
    viewer.imageryLayers.remove(state.imageryLayer, true);
    state.imageryLayer = viewer.imageryLayers.addImageryProvider(
      OfflineAssets.instrument(OfflineAssets.createImageryProvider(),
                               state.imageryStats));
    state.imageryLayer.show = wasShown;
    registerImagerySlot('imagery.offline', [state.globalLayer, state.imageryLayer]);
    // Hotspot layers belong to a theatre too, so they are rebuilt here rather
    // than left pointing at the previous one.
    remountHotspotImagery();
    if (state.layers && state.layers.imagery !== undefined) {
      state.imageryLayer.show = !!state.layers.imagery;
    }
    markScene('switchTheatre');
    // ⚠ 必须显式 apply：上面 remove + addImageryProvider 把战区影像层加到了
    // imageryLayers **末尾**，而 markScene 只驱动 syncEntities，不会重排 imagery。
    // 不补这一句，"切案例后底图压住注记/热点"会一直保持到下一次相机触发
    // syncOfflineDetail 才自愈——这正是"动态加载后层级变了"的现象之一。
    applyLayers();
  } catch (e) {
    // A failed rebuild must not abort loading the case: the data, targets and
    // panels are independent of the basemap rectangle. Report it on the status
    // bar and console rather than swallowing it, because "no imagery" and
    // "imagery announced over the wrong rectangle" look identical on screen.
    setStatus('战区影像层重建失败：' + e.message, 'warn');
    if (window.console && console.warn) console.warn('switchTheatre failed', e);
  }
}

function applySituationData(data, caseId) {
  {
      if (!data.frames || !data.frames.length) throw new Error('TS数据为空');

      teardownScene();

      state.data = data;
      state.caseId = caseId || data.case_id || null;
      state.inference = data.inference_frames || null;
      // 案例数据换了：面板结构与战损缓存都必须重建。"前端生成的TS"的
      // case_id 固定为 custom，只靠 case_id 命中会残留上一份编成。
      if (window.StatusPanels && StatusPanels._resetCache) StatusPanels._resetCache();
      if (window.TargetStatus && TargetStatus.reset) TargetStatus.reset();
      state.tick = 0;
      state.tickFloat = 0;
      state.selected = null;
      state.activeHotspot = -1;
      state.geoPinned = false;
      state.buildingsAoi = -1;
      // Switch the theatre *after* `state.data` is set and *before* any provider
      // is built for the new case. Both halves matter: the regional imagery/WSF3D
      // rectangles must follow the case, and the hotspot layer selection reads
      // `state.data.hotspots`, which is empty until the assignment above.
      switchTheatre(data.theatre);
      markScene('applySituationData');
      // 数据集自带本体的压缩形态，优先用它（离线可用），接口只是兜底。
      if (window.IntentUi) IntentUi.adoptFromModel(state.data && state.data.inference_model);

      if (data.symbology) Symbology.setCatalog(data.symbology);

  // 城区 AOI 以 base.json 为准（与案例无关）；旧数据集里也带一份，作为回退。
  if (!state.urbanAoi || !state.urbanAoi.length) {
    state.urbanAoi = data.urban_aoi || [];
  }
  syncUrbanImagery();
  // 同一批"按相机决定挂不挂"的层在这里一起对齐：案例加载后必须重判一次，
  // 否则用 ?case= 深链直接进来时会停在"启动时的高度"这个错误判断上。
  syncHotspotImagery();
  syncOfflineDetail();

      dom.scenarioName.textContent = data.region + ' · ' + data.case_name;
      document.title = '三维ZCTS生成系统 — ' + data.case_name;

      createHotspots();
      createRouteEntities();
      createTargetEntities();
      attachMotionAndFx(data);
  if (typeof applyTargetLod === "function") applyTargetLod(true);
      applyCamera(data.camera);

      renderHotspotTable();
      renderTargetCounts();
      renderTransport();
      clearTargetSelection();
      setTick(0);
      updateInferenceUi();
      applyAutoPlots(data);

      setTransportEnabled(true);
      if (dom.followTarget) {
        dom.followTarget.disabled = false;
        dom.followTarget.title = '';
      }
      setFollowNote('选中目标即居中跟随；勾选「目标跟随」切换第三人称斜后方视角。');
      // 原先这里打印「就绪 · N 个目标 · M 帧 · K 个热点区域」。按界面重构要求
      // 移除该概略文本；状态栏只保留**动态**信息（加载进度、错误、操作反馈），
      // 空闲时不占用视线。案例身份已由顶栏 `#scenarioName` 显示，不再重复。
      setStatus('', 'ok');
      setSwitchNote('已加载「' + data.case_name + '」。各开关的生效状态会在此说明。');
      // 装载完TS后补一次建筑体块。refreshBuildings() 原本只挂在"选中热点 /
      // 解除跟随 / 目标高度变化"上，于是一进页面就带 `?buildings=1`（或先开开关
      // 再选案例）时体块不会出现——除非用户再去点一个热点。色带未就绪时
      // `?buildings=1` 的回调会补这一次。
      if (checkboxOn('modelBuildings') && state.ramp) refreshBuildings();
      return data;
  }
}

/**
 * 清空**单要素**状态：逐个目标的隐藏/移除/信息覆盖，以及逐条航路 / 热点。
 *
 * 必须挂在 `teardownScene()` 上，**不能只挂在 `clearSituation()` 上**：换案例走的是
 * `loadCase → applySituationData → teardownScene()`，这条路**不经过** clearSituation。
 * 只清在 clearSituation 里的话，"在台海把 R01 隐藏、R02 移除 → 切到中印"之后这两条
 * 仍然留着（实测：切案例后 `hiddenTargets` 里还有 R01、`removedTargets` 里还有 R02）。
 * 而 id 在不同案例里会重名（每个案例都有自己的 R01），一旦撞上，用户从没在新案例里
 * 动过那个目标，它却怎么也找不到 —— 正是"切了案例状态还在"的那类幽灵 bug。
 *
 * 这里**原地清键**而不是换新对象：万一有地方捕获了这些对象的引用，换对象会让它
 * 继续读写旧的那份，状态就分叉了。
 */
function resetElementState() {
  [state.hiddenTargets, state.removedTargets, state.targetOverrides,
   state.hiddenRoutes, state.hiddenHotspots].forEach(function (bag) {
    Object.keys(bag).forEach(function (k) { delete bag[k]; });
  });
}

/**
 * 场景清空后恢复标绘。
 *
 * `viewer.entities.removeAll()` 只清图元，不清 `Plotting.byId` 里的记录；不重画的话，
 * 「标绘图层由实际要素驱动」会自相矛盾——面板显示 N 条、场景里 0 条。
 * 目标轨迹是选中目标的派生数据（此时选中已清空），先移除；用户标绘按控制点重画。
 */
function redrawPlotsAfterClear() {
  // `updateFocusMask` 只在引用为空时才新建实体；removeAll 已经删掉了实体，
  // 这里必须把引用清掉，否则下一次选中目标会去改一个已不在场景里的对象，
  // 掩码整场会话再也不出现（而 focusDebug 还报 maskExists=true）。
  state.focusMaskEntity = null;
  state.focusMaskHole = null;
  if (!window.Plotting) return 0;
  return Plotting.redrawAll();
}

function teardownScene() {
  if (!viewer) return;
  detachFollow('TS已切换');
  clearTargetSelection();
  if (window.TargetCombatFx) TargetCombatFx.dispose();
  if (window.TargetFx) TargetFx.dispose();
  if (window.TargetMotion) TargetMotion.dispose(viewer);
  viewer.entities.removeAll();
  redrawPlotsAfterClear();
  state.targets = {};
  state.hotspotEntities = [];
  state.occupationEntities = [];
  state.hotspotRows = [];
  state.routeEntities = [];
  // 场景都拆了，单要素状态自然跟着作废（换案例 / 回空态两条路都经过这里）。
  resetElementState();
  if (state.buildingsPrimitive) {
    viewer.scene.primitives.remove(state.buildingsPrimitive);
    state.buildingsPrimitive = null;
  }
  state.buildingsAoi = -1;
}

// --------------------------------------------------------------------------
// 与案例无关的基础数据（data/base.json）
// --------------------------------------------------------------------------
//
// 空态（不加载任何TS）下前端仍需要两样东西：城区 AOI（生成 WSF3D 建筑体块）
// 与军标目录（渲染符号）。它们原本只随案例数据集下发，所以提到这里单独加载。

function loadBase() {
  return fetch('./data/base.json', { cache: 'no-store' })
    .then(function (r) {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.json();
    })
    .catch(function (e) {
      // 缺失不阻塞启动：军标退回内置兜底，城区体块给出明确提示。
      console.warn('[base] 基础数据加载失败：' + e.message);
      return null;
    });
}

function applyBase(base) {
  state.base = base || null;
  if (base && base.symbology) Symbology.setCatalog(base.symbology);
  state.urbanAoi = (base && base.urban_aoi) || [];
  state.baseCamera = (base && base.camera) || null;
  state.baseRegion = (base && base.region) || '未加载TS';
  // 预加载参数：base.json 给基线，URL 参数可覆盖（见 applyPrefetchParams）。
  if (window.TileCache && TileCache.configurePrefetch) {
    if (base && base.prefetch) TileCache.configurePrefetch(base.prefetch);
    applyPrefetchParams();
  }
  syncUrbanImagery();
}

/**
 * 预加载的 URL 覆盖项。默认值在 tilecache.js 的 PREFETCH_DEFAULTS 里；
 * 这里只做"现场可调"，并且**默认什么都不写就是默认行为**。
 *
 *   ?prefetch=off                     关闭预加载（回到纯按需加载）
 *   ?prefetch=children                只预取下一档（拉近用的提前量）
 *   ?prefetch=children,parents        默认值
 *   ?prefetch=all                     再加上 siblings（同层相邻一圈，会越出视口）
 *   ?prefetchConcurrency=4            并发（默认 2）
 *   ?prefetchIdleMs=3000              空闲多久才开始预取（默认 1200）
 *   ?prefetchPerSec=30                每秒最多几张（默认 12）
 *   ?prefetchQueue=2048               队列上限（默认 512）
 */
function applyPrefetchParams() {
  if (!window.TileCache || !TileCache.configurePrefetch) return;
  let q = '';
  try { q = String(window.location.search || ''); } catch (e) { q = ''; }
  if (!q) return;
  const get = function (name) {
    // 值里排除 `&` 与 `/`：`?prefetch=off` 后面若还跟着路径（例如被脚本拼成
    // `?prefetch=off/`）也应当被当作 `off`，否则会静默降级成"空 range"——
    // 功能上等于关，但配置里读出来还是 enabled:true，验收时对不上。
    const m = new RegExp('[?&]' + name + '=([^&/]*)').exec(q);
    return m ? decodeURIComponent(m[1]) : null;
  };
  const cfg = {};
  const pf = get('prefetch');
  if (pf !== null) {
    if (pf === 'off' || pf === '0' || pf === 'false') cfg.enabled = false;
    else if (pf === 'all') { cfg.enabled = true; cfg.range = ['children', 'parents', 'siblings']; }
    else if (pf === 'on' || pf === '1') cfg.enabled = true;
    else {
      // 显式给出范围（如 `?prefetch=children`）本身就意味着"要开"：
      // 否则会静默不生效 —— 实测 `?prefetch=children` 曾把 range 设好却仍是
      // enabled:false（因为 base.json 里默认关），看起来像"这个参数没用"。
      cfg.enabled = true;
      cfg.range = pf.split(',').map(function (s) { return s.trim(); }).filter(Boolean);
    }
  }
  const num = { prefetchConcurrency: 'concurrency', prefetchIdleMs: 'idleDelayMs',
                prefetchPerSec: 'maxPerSecond', prefetchQueue: 'queueLimit',
                prefetchAhead: 'levelsAhead' };
  Object.keys(num).forEach(function (p) {
    const v = get(p);
    if (v !== null && isFinite(Number(v)) && Number(v) > 0) cfg[num[p]] = Number(v);
  });
  TileCache.configurePrefetch(cfg);
}

/**
 * 预加载的候选 URL：**只取视口内、另一档层级**的瓦片。
 *
 * 为什么由这里算而不是 tilecache 算：几何信息（相机视锥、各 provider 的 tilingScheme
 * 与矩形）都在这一侧；tilecache 只负责"排队 + 限流 + 取回来 + 存缓存"。
 *
 * 规则（对每个 `show === true` 的影像层）：
 *   · 缓存里没有、且不在途的 URL 才交给 TileCache.prefetch（重复的由它再过滤一次）；
 *   · 层级取 `S+1`（children，拉近时的提前量）与 `S-1`（parents，缩小时的过渡），
 *     `S` = 当前**实际渲染**的表面层级；越出 provider 的 min/max 就跳过；
 *   · 瓦片范围由视口矩形的四角 + 中心用 `tilingScheme.positionToTileXY`（公开 API）
 *     算出，再与**图层自身的矩形**求交 —— 图层不在视野里就整层跳过，
 *     这正是"瓦片只在视口内"的保证；
 *   · 一次最多 256 张，避免宽视野下一次丢进上千个 URL。
 */
function collectPrefetchUrls() {
  if (!viewer || !window.TileCache || !TileCache.prefetch) return [];
  const cfg = TileCache.prefetchConfig ? TileCache.prefetchConfig() : null;
  if (!cfg || !cfg.enabled) return [];
  const scene = viewer.scene;
  if (!scene || !scene.globe || !scene.globe.show) return [];
  const S = surfaceRenderLevel();
  if (S < 0) return [];
  const rect = viewer.camera.computeViewRectangle();
  if (!rect) return [];
  const want = {};
  (cfg.range || []).forEach(function (r) { want[r] = true; });
  const out = [];
  for (let i = 0; i < viewer.imageryLayers.length && out.length < 256; i++) {
    const layer = viewer.imageryLayers.get(i);
    if (!layer.show || !layer.imageryProvider) continue;
    const p = layer.imageryProvider;
    const tpl = p.url || p._url;
    if (typeof tpl !== 'string' || tpl.indexOf('{') < 0) continue;
    const ts = p.tilingScheme;
    if (!ts || !ts.positionToTileXY) continue;
    // 图层矩形与视野不相交 → 整层跳过（高倍段/热点/城区都是窄矩形，这条最省）
    if (p.rectangle && !Cesium.Rectangle.intersection(rect, p.rectangle)) continue;
    const levels = [];
    if (want.children) {
      // 预取深度可配（默认 2）。为什么默认不是 1：Cesium 自己的四叉树精化在
      // SSE=2 下**本来就会把下一级取回来**（实测：200 km 档预取了 z8，下一档 100 km
      // 因此 0 请求）—— 深度 1 的预取和它重叠，等于白做。深度 2 才是"渲染器不会
      // 主动做、而用户下一格就可能要"的那一档。
      const ahead = Math.max(1, Math.min(4, (cfg.levelsAhead | 0) || 2));
      for (let d = 1; d <= ahead; d++) levels.push(S + d);
    }
    if (want.parents) levels.push(S - 1);
    levels.forEach(function (z) {
      if (z < 0) return;
      if (p.minimumLevel !== undefined && z < p.minimumLevel) return;
      if (p.maximumLevel !== undefined && z > p.maximumLevel) return;
      const lr = p.rectangle || Cesium.Rectangle.MAX_VALUE;
      const view = Cesium.Rectangle.intersection(rect, lr) || rect;
      const pts = [[view.west, view.north], [view.east, view.north],
                   [view.west, view.south], [view.east, view.south],
                   [(view.west + view.east) / 2, (view.north + view.south) / 2]];
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      pts.forEach(function (pt) {
        let t = null;
        try { t = ts.positionToTileXY(Cesium.Cartographic.fromDegrees(pt[0], pt[1]), z); } catch (e) { t = null; }
        if (!t) return;
        x0 = Math.min(x0, t.x); x1 = Math.max(x1, t.x);
        y0 = Math.min(y0, t.y); y1 = Math.max(y1, t.y);
      });
      if (!isFinite(x0)) return;
      for (let y = y0; y <= y1 && out.length < 256; y++) {
        for (let x = x0; x <= x1 && out.length < 256; x++) {
          // 模板里 {z}/{x}/{y} 的顺序各源不同（Esri 是 {z}/{y}/{x}），
          // 逐个替换占位符即可，与顺序无关。
          out.push(tpl.replace('{z}', z).replace('{x}', x).replace('{y}', y));
        }
      }
    });
  }
  return out;
}

/**
 * 预加载的触发闸门：**底图已加载完 + 相机停稳**才收一次候选 URL。
 *
 * 三条都满足才触发，缺一不可：还在加载时预取会抢那 6 条连接；相机在动时预取的是
 * 马上要作废的视野；地球不可见时预取的瓦片根本画不出来。
 */
function startPrefetchLoop() {
  let lastMove = Date.now();
  if (viewer && viewer.camera && viewer.camera.changed) {
    viewer.camera.changed.addEventListener(function () { lastMove = Date.now(); });
  }
  setInterval(function () {
    if (!viewer || !viewer.scene || !viewer.scene.globe) return;
    if (!viewer.scene.globe.show || !viewer.scene.globe.tilesLoaded) return;
    const cfg = window.TileCache && TileCache.prefetchConfig ? TileCache.prefetchConfig() : null;
    if (!cfg || !cfg.enabled) return;
    if (Date.now() - lastMove < (cfg.idleDelayMs || 1200)) return;   // 相机还在动
    const urls = collectPrefetchUrls();
    if (urls.length) TileCache.prefetch(urls);
  }, 2000);
}

// --------------------------------------------------------------------------
// 空态与TS生命周期
// --------------------------------------------------------------------------

/** 播放/步进/时间轴/导出在空态下无意义，统一在此开关。 */
function setTransportEnabled(on) {
  ['btnPlay', 'btnRun', 'btnReset', 'btnStepBack', 'btnStepFwd', 'btnExport'].forEach(function (id) {
    const node = document.getElementById(id);
    if (node) node.disabled = !on;
  });
  if (dom.timeline) dom.timeline.disabled = !on;
}

/**
 * 回到空态：不加载任何TS，只保留底图 / 地形 / WSF3D / 地图组件。
 *
 * 幂等——启动时与「清除当前TS」共用同一条路径，避免两套清场逻辑漂移。
 */
function clearSituation() {
  if (!viewer) return;
  setPlaying(false);
  if (state.buildingTimer) { clearTimeout(state.buildingTimer); state.buildingTimer = null; }
  teardownScene();                     // 内含 detachFollow + 清空实体

  state.data = null;
  state.caseId = null;
  state.inference = null;
  // 鸟瞰图随 TS 一起清空，否则会残留上一场的图标与取景。
  if (window.Overview && Overview.clear) Overview.clear();
  // 整体TS面板也要回到空态；只清 state.data 的话 updateInferenceUi 会因
  // state.data 为空而跳过渲染，上一场的图标/血条会一直留在右栏。
  if (window.StatusPanels) {
    StatusPanels._resetCache();
    StatusPanels.renderOverall(dom.overallBody, null, null, 0);
  }
  if (window.TargetStatus) TargetStatus.reset();
  state.tick = 0;
  state.tickFloat = 0;
  state.selected = null;
  state.activeHotspot = -1;
  state.geoPinned = false;
  state.buildingsKey = null;
  state.buildingsAoi = -1;
  // 单要素状态（hiddenTargets / removedTargets / targetOverrides / hiddenRoutes /
  // hiddenHotspots）由上面的 teardownScene() → resetElementState() 一并清掉。放在那里
  // 而不是这里，是因为**换案例不走 clearSituation**，只清这里等于没清（见该函数说明）。
  markScene('clearSituation');

  dom.scenarioName.textContent = state.baseRegion || '未加载TS';
  document.title = '三维ZCTS生成系统';

  renderTargetCounts();
  renderHotspotTable();
  renderTransport();
  updateInferenceUi();
  clearHotspotSelection();
  clearTargetSelection();
  setTransportEnabled(false);

  if (dom.followTarget) {
    dom.followTarget.disabled = true;
    dom.followTarget.checked = false;
    dom.followTarget.title = '未加载TS';
  }
  setFollowNote('未加载TS，无法跟随。请先从「演示案例」选择案例。');

  // 机位兜底：base.json 缺失时也要回到台海，否则会停在 Cesium 的默认视图
  // （DEFAULT_VIEW_RECTANGLE 的中心在 82.5°W / 35°N，看起来像"相机飞走了"）。
  applyCamera(state.baseCamera || FALLBACK_CAMERA);
  setStatus('空态 · 未加载TS', 'ok');
  setSwitchNote('未加载TS：底图 / 地形 / WSF3D / 地图组件仍可用。请从顶栏「演示案例」选择案例。');
}

// --------------------------------------------------------------------------
// 案例菜单
// --------------------------------------------------------------------------

function renderCaseMenu() {
  const list = dom.menuCaseList;
  list.innerHTML = '';

  state.caseIndex.cases.forEach(function (c) {
    const item = el('button', 'menu-item', '');
    item.type = 'button';
    item.setAttribute('role', 'menuitem');
    item.dataset.caseId = c.id;

    const head = el('span', 'mi-head', c.name);
    // 案例条目现在**只留案例名**。历史上这里还有两行副文本：编成计数
    //（「16 个目标 · 11 个热点区域」）与案例描述，按反馈先后去掉——菜单是选案例用的，
    // 不是读说明的地方，两行副文本把菜单撑高一倍还挤掉视线。
    // 完整描述仍挂在 title 上：悬停可看，但不占版面（不是"显示的说明文字"）。
    if (c.description) item.title = c.name + '：' + c.description;
    item.appendChild(head);

    item.addEventListener('click', function () {
      closeCaseMenu();
      if (c.id === state.caseId) return;
      loadCase(c.id).then(function () { applyCaseHooks(); }).catch(function (err) {
        console.error(err);
        setStatus('案例加载失败: ' + err.message, 'err');
      });
    });
    list.appendChild(item);
  });

  // 清除当前TS，回到空态（底图/地形/WSF3D/地图组件保留）
  const sep = el('div', 'menu-sep');
  list.appendChild(sep);
  const clearItem = el('button', 'menu-item menu-item-quiet', '');
  clearItem.type = 'button';
  clearItem.setAttribute('role', 'menuitem');
  // 同样只留标题：「回到空态：保留底图 / 地形 / WSF3D / 地图组件」那行说明也去掉。
  clearItem.appendChild(el('span', 'mi-head', '清除当前TS'));
  clearItem.title = '回到空态：保留底图 / 地形 / WSF3D / 地图组件';
  clearItem.addEventListener('click', function () {
    closeCaseMenu();
    if (!state.data) return;
    clearSituation();
  });
  list.appendChild(clearItem);
}

const MENUS = [];

/**
 * 顶栏下拉菜单的通用开合。各菜单互斥，点空白处全部收起。
 *
 * `opts.keepOpen` 是一个谓词：为真时**外部点击不收这个菜单**（Esc 仍能收，
 * 走 `closeAllMenus(true)` 的强制通道 —— 两者语义不同：外部点击是"你可能想干别的"，
 * Esc 是"我明确要关它"）。
 *
 * ⚠ 2026-10-16 修一个真实缺陷：原先只在**按钮**上 `stopPropagation`，而
 * `document` 上挂着 `click → closeAllMenus`。于是**点面板内部的任何控件**
 * （类型下拉、红蓝方下拉、输入框…）都会冒泡到 document，把面板自己关掉 ——
 * 表现就是"在标绘过程中点选项会直接退出标绘工具"。
 * 现在给面板本身也加 `stopPropagation`：面板内部的操作不再影响菜单开合。
 */
function bindMenu(buttonId, listId, opts) {
  const btn = document.getElementById(buttonId);
  const list = document.getElementById(listId);
  if (!btn || !list) return;
  MENUS.push({ btn: btn, list: list, keepOpen: (opts && opts.keepOpen) || null });
  // 面板内的点击不冒泡到 document：否则 document 的 closeAllMenus 会把面板关掉。
  // 用 capture 之外的两个事件都拦：select 在部分浏览器上只发 pointerdown/click 之一。
  ['click', 'pointerdown', 'mousedown'].forEach(function (ev) {
    list.addEventListener(ev, function (e) { e.stopPropagation(); });
  });
  btn.addEventListener('click', function (e) {
    e.stopPropagation();
    const wasOpen = list.classList.contains('open');
    closeAllMenus();
    if (!wasOpen) {
      list.classList.add('open');
      btn.setAttribute('aria-expanded', 'true');
    } else {
      // 显式点按钮 = 收起，即便这个菜单声明了 keepOpen。
      list.classList.remove('open');
      btn.setAttribute('aria-expanded', 'false');
    }
  });
}

/**
 * 收起菜单。`force = true` 时忽略 `keepOpen`（Esc 走这条）。
 */
function closeAllMenus(force) {
  MENUS.forEach(function (m) {
    if (!force && m.keepOpen && m.list.classList.contains('open') && m.keepOpen()) return;
    m.list.classList.remove('open');
    m.btn.setAttribute('aria-expanded', 'false');
  });
}

function openCaseMenu() {
  closeAllMenus();
  Array.prototype.forEach.call(dom.menuCaseList.children, function (n) {
    n.classList.toggle('active', n.dataset.caseId === state.caseId);
  });
  dom.menuCaseList.classList.add('open');
  dom.menuCase.setAttribute('aria-expanded', 'true');
}

function closeCaseMenu() { closeAllMenus(); }

// --------------------------------------------------------------------------
// 图层管理的批量操作
// --------------------------------------------------------------------------
//
// 四大类下的全部开关。目标跟随是"模式开关"而不是图层，刻意不列入——把它一起
// 关掉等于隐式解除跟随，语义混乱。

const LAYER_GROUPS = [
  { name: 'TS图层', ids: ['layerHotspots', 'layerRoutes'] },
  { name: '目标图层', ids: ['layerTargetsRed', 'layerTargetsBlue', 'layerLabels', 'layerTracks', 'modelSensor',
                            'layerTargetPanel', 'filterRed', 'filterBlue', 'filterSea',
                            'filterLand', 'filterAir'] },
  { name: '底图', ids: ['modelTianditu', 'modelTiandituLabel', 'modelImagery',
                        'modelHotspotImagery', 'modelUrbanImagery',
                        'modelPlaces', 'modelTerrain',
                        'modelWsf3dOverlay', 'modelBuildings', 'modelWeather'] },
  { name: '地图组件', ids: ['compCompass', 'compScalebar', 'compViewpoint'] },
];

const LAYER_DEFAULTS = {};

/**
 * 记录每个开关的默认值，供「恢复默认」用。
 *
 * 默认值现在有**两个来源**，必须都读：
 *   1. 仍在 HTML 里的开关（TS图层/目标图层/地图组件）→ 读 DOM 的 checked；
 *   2. 已由面板生成的图层（底图/标绘）→ 读注册表的 `default`。
 *
 * 以前只有来源 1。底图/标绘搬进生成式面板之后，`getElementById` 一律返回 null，
 * `LAYER_DEFAULTS` 里就没有这些条目，「恢复默认」会取到 `!!undefined === false` ——
 * 那不是"恢复默认"，是把底图和标绘**全部关掉**。这类"移除 DOM 之后批量功能静默
 * 失效"的问题不会报错，只会表现为"按钮点了没反应"。
 */
function captureLayerDefaults() {
  LayerRegistry.LAYERS.forEach(function (d) {
    const node = d.checkbox ? document.getElementById(d.checkbox) : null;
    LAYER_DEFAULTS[d.id] = node ? node.checked : (d.default === true);
  });
  // 非图层注册表的开关（阵营/域过滤等）仍按 DOM 取。
  LAYER_GROUPS.forEach(function (g) {
    g.ids.forEach(function (id) {
      if (id in LAYER_DEFAULTS) return;
      const node = document.getElementById(id);
      if (node) LAYER_DEFAULTS[id] = node.checked;
    });
  });
}

/**
 * 以编程方式切换开关。
 *
 * 有 DOM 节点就派发 change 复用既有 handler；**没有节点说明这一行由面板生成**，
 * 此时必须走注册表那条路 —— 否则批量命令（全部显示/隐藏/恢复默认）对底图与标绘
 * 会**静默无效**，表现为"点了没反应"。
 */
function setSwitch(id, on) {
  const node = document.getElementById(id);
  if (node) {
    if (node.disabled || node.checked === on) return;
    node.checked = on;
    node.dispatchEvent(new Event('change', { bubbles: true }));
    return;
  }
  // 按**复选框 id 反查**图层再写意图 —— 直接用 id 问 def() 是查不到的（见
  // layerIdForCheckbox 的说明）
  const lid = layerIdForCheckbox(id);
  if (lid) {
    LayerRegistry.setDesired(lid, on);
    applyLayers();
    return;
  }
  // 兜底：调用方（applyLayerCommand）传的是**图层 id**（面板生成的图层没有
  // DOM 节点、checkbox 字段也可能是惰性的）。不兜底时「全部显示/恢复默认」
  // 对这些图层静默无效。
  if (LayerRegistry.def && LayerRegistry.def(id)) {
    LayerRegistry.setDesired(id, on);
    applyLayers();
  }
}

/**
 * cmd: 'show' 全开 | 'hide' 只关TS与目标图层 | 'default' 恢复默认。
 *
 * 「全部隐藏」刻意不关底图与地图组件——把底图关掉只会得到一片黑球，不是用户
 * 想要的结果。
 */
function applyLayerCommand(cmd) {
  const only = cmd === 'hide' ? ['TS图层', '目标图层'] : null;
  const ids = [];
  LAYER_GROUPS.forEach(function (g) {
    if (only && only.indexOf(g.name) < 0) return;
    g.ids.forEach(function (id) { if (ids.indexOf(id) < 0) ids.push(id); });
  });
  // 面板生成的图层（底图/标绘）不在 LAYER_GROUPS 里，必须按 `group` 一并纳入，
  // 否则「全部显示」对它们无效 —— 这类漏项不会报错，只会表现为按钮没反应。
  LayerRegistry.LAYERS.forEach(function (d) {
    if (only && only.indexOf(d.group) < 0) return;
    if (ids.indexOf(d.id) < 0) ids.push(d.id);
  });
  ids.forEach(function (id) {
    setSwitch(id, cmd === 'show' ? true : cmd === 'hide' ? false : !!LAYER_DEFAULTS[id]);
  });
  setStatus(cmd === 'show' ? '图层已全部显示'
    : cmd === 'hide' ? 'TS与目标图层已隐藏'
    : '图层已恢复默认', 'ok');
}

// --------------------------------------------------------------------------
// 场景构建
// --------------------------------------------------------------------------

function positionFor(id, tick) {
  // 空态守卫放在最前面：这个函数被 Cesium 的 CallbackProperty 每帧求值，
  // 一旦抛异常就是每帧一次未捕获错误，直接污染状态栏。
  if (!state.data || !state.data.frames) return undefined;
  const frames = state.data.frames;
  const frame = frames[Math.max(0, Math.min(tick, frames.length - 1))];
  return frame.targets.find(function (t) { return t.id === id; });
}

function isVisible(def) {
  if (!state.layers.targets) return false;
  // 可见目标按阵营拆成两行：红方 / 蓝方目标各自可单独关。
  if (def.affiliation === 'red' && !state.layers.targetsRed) return false;
  if (def.affiliation === 'blue' && !state.layers.targetsBlue) return false;
  // 单要素层面：图层开关 → 阵营/域过滤 → **单要素开关/移除**，三层取与。
  // 走同一条通路（实体 show）而不是另建一套隐藏机制：目标实体、标牌、航路都由
  // isVisible/syncEntities 统一驱动，另建一套就会出现"隐藏了但航路还在"。
  if (state.removedTargets[def.id]) return false;
  if (state.hiddenTargets[def.id]) return false;
  if (def.affiliation === 'red' && !state.filters.red) return false;
  if (def.affiliation === 'blue' && !state.filters.blue) return false;
  if (!state.filters[def.domain]) return false;
  return true;
}

// ---------------------------------------------------------------------------
// 车/船 2D 图标 ⇄ 3D 模型（按视点高度，判据在 web/src/entityLod.js）
// ---------------------------------------------------------------------------
//
// 判据与场景分离：EntityLod.decide() 是纯函数（已有 Node 用例
// tools/entity_lod_test.js）。这里只负责「把结果落到场景」。
//
// 为什么不在 isVisible() 里做：isVisible 表达的是**用户意图**（图层开关、阵营/
// 域过滤、单要素隐藏），而形态切换是**相机距离的函数**，两者混在一起会让
// 「关了目标图层」和「离得远」变成同一件事，无法分别推理。所以这里只写
// billboard.show / model.show，**不碰 isVisible**，避免与既有通路打架。

/** 模型 scale：资产已按米归一，恒为 1；判不出模型时返回 undefined。 */
function modelScaleForTarget(def) {
  if (!window.EntityLod || !window.EntityLod.scaleFor) return undefined;
  const s = window.EntityLod.scaleFor(def);
  return (s && isFinite(s) && s > 0) ? s : undefined;
}

/** 该目标要用的 3D 模型 URL；拿不准返回 null（保留 2D）。 */
function modelUrlForTarget(def) {
  if (!window.EntityLod) return null;
  return window.EntityLod.modelUrlFor(def);
}

/** 当前相机高度（米）。取不到时返回 undefined —— decide() 会保持现状。 */
function cameraHeightMeters() {
  try {
    return viewer.camera.positionCartographic.height;
  } catch (e) {
    return undefined;
  }
}

/**
 * 把当前相机高度对应的形态落到所有目标实体上。
 *
 * 只在**形态真的变了**时写 show：每次 camera.changed 都遍历 31 个实体写属性，
 * 会让 Cesium 每帧重建这些实体的渲染状态。所以记住上一次的形态，没变就直接返回。
 */
var lastTargetForm = null;
// 哪些目标真的挂了 3D 模型。**由 createTargetEntities() 登记**，不去读实体属性猜：
// 实测 Cesium 的 `entity.model.uri` 读出来是 Property 对象而不是字符串真值，
// 用它做 `!!` 判断会在首次解算时误判为「没有模型」，表现为「刚进低空时图标和
// 模型同时显示」，要等下一次 camera.changed 才自愈。登记表没有这个歧义。
var targetHasModel = {};

function applyTargetLod(force) {
  if (!window.EntityLod || !state.data || !state.data.targets) return;
  const d = window.EntityLod.decide(cameraHeightMeters(), lastTargetForm || window.EntityLod.MODE_ICON);
  if (!force && !d.changed) return;
  lastTargetForm = d.mode;
  const useModel = (d.mode === window.EntityLod.MODE_MODEL);
  state.data.targets.forEach(function (def) {
    const rec = state.targets[def.id];
    if (!rec || !rec.entity) return;
    const hasModel = !!targetHasModel[def.id];
    // 没有模型的实体（车辆、空中目标）**永远保留 2D 图标**——不能因为
    // 切到「模型」形态就把图标藏掉，那会让这些目标整个消失（实测：31 个目标
    // 里只有 12 个有模型，另 19 个是陆/空目标）。
    const showModel = useModel && hasModel;
    // 图标**始终显示**：切到 3D 模型后图标不再隐藏，而是抬到模型头顶上方
    // （见 applyTargetBillboardAnchor）。目标点本身仍由模型表达。
    if (rec.entity.billboard) {
      rec.entity.billboard.show = true;
      applyTargetBillboardAnchor(rec.entity, showModel);
    }
    if (hasModel && rec.entity.model) rec.entity.model.show = showModel;
  });
}

/**
 * 2D 图标的锚定方式：图标形态沿用"居中贴目标点"（与历史行为一致）；模型形态
 * 改为**底边锚定 + 抬高 52 px**，使图标稳定落在模型/标签的上方。
 *
 * 为什么用 pixelOffset 而不是 eyeOffset：图标是屏幕空间图元，pixelOffset 不受
 * 相机距离/朝向影响，缩放时不会跳；eyeOffset 是视空间偏移，相机俯仰变化时会
 * 让图标在模型两侧滑动。52 px 取"标签（底边 24 px + 约 18 px 高）之上再留
 * 10 px"，模型再高也不会遮住图标。
 */
function applyTargetBillboardAnchor(entity, modelMode) {
  const bb = entity && entity.billboard;
  if (!bb) return;
  // syncEntities 每个脏帧都会走这里；形态没变就不重复写属性，避免 Cesium
  // 把实体反复标脏。
  if (entity._iconAnchorModel === modelMode) return;
  entity._iconAnchorModel = modelMode;
  // The symbol sits at the model centre and must stay above it; the old
  // "lift above the model head" offset hid the symbol behind the mesh.
  bb.verticalOrigin = Cesium.VerticalOrigin.CENTER;
  bb.pixelOffset = new Cesium.Cartesian2(0, 0);
  bb.disableDepthTestDistance = Number.POSITIVE_INFINITY;
}

function createTargetEntities() {
  state.data.targets.forEach(function (def) {
    const color = colorFor(def.affiliation);

    const entity = viewer.entities.add({
      id: 'target-' + def.id,
      name: def.id + ' ' + def.name,
      position: new Cesium.CallbackProperty(function () {
        const s = positionFor(def.id, state.tick);
        if (!s) return undefined;
        return Cesium.Cartesian3.fromDegrees(s.lon, s.lat, s.alt);
      }, false),
      billboard: {
        image: Symbology.symbolUri(def.icon, def.affiliation),
        // **固定屏幕尺寸**：34 CSS px，不随相机缩放（去掉 scaleByDistance）。
        // 之前近处 1.5×、远处 0.5×，拉远后目标牌小到读不出字。
        width: 34,
        height: 34,
        // 锚点/偏移由 applyTargetBillboardAnchor() 按形态切换：图标形态保持不变
        // （居中、贴目标点），模型形态改到底边锚定并抬高，落在模型"头顶"之上。
        verticalOrigin: Cesium.VerticalOrigin.CENTER,
        pixelOffset: new Cesium.Cartesian2(0, 0),
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
      // 车/船的 3D 模型（低空显示，见 EntityLod）。**默认 show:false**：
      // 高空由 billboard 表达，跨过阈值后由 applyTargetLod() 切过来。
      // uri 可能是 undefined（形态类别拿不准 / 尚无车辆资产）——那时 Cesium 不建
      // 模型，实体只留 2D，这正是我们要的「拿不准就保留图标」。
      model: (function () {
        const url = modelUrlForTarget(def);
        if (!url) return undefined;
        targetHasModel[def.id] = true;   // 供 applyTargetLod 判断，不读 Property
        return {
          uri: url,
          // 新资产在 tools/modelgen 管线里已按公开全长归一为 **1 单位 = 1 米**
          // （舰长/机长/车长含炮管），所以 scale 恒为 1；保留调用是为了兼容
          // scaleFor() 的契约（拿不准时返回 undefined，不写 scale）。
          scale: modelScaleForTarget(def),
          // 阵营着色：按 affiliation 做**混合着色**（MIX, 0.2），保留模型自身的
          // 材质/纹理层次，又让红方/蓝方在低空一眼可分。
          color: color,
          colorBlendMode: Cesium.ColorBlendMode.MIX,
          colorBlendAmount: 0.2,
          // 像素下限只作**极远处的可辨性兜底**，不能大到把距离语义抹平：
          // 原先 24 与 2D 图标的 34 同量级，会把 3D 恒定成 24 px。
          minimumPixelSize: 6,
          maximumScale: 20000,
          // 车轮节点旋转（targetFx.js）：只给轮式车辆，节点名/轴/半径来自
          // EntityLod.WHEEL_NODES；船/飞机/履带车返回 undefined，不改变模型。
          nodeTransformations: window.TargetFx ? TargetFx.modelNodeTransformations(def) : undefined,
          show: false,
        };
      })(),
      label: {
        text: def.id + ' ' + def.platform_class,
        font: '12px "Microsoft YaHei", "Noto Sans CJK SC", sans-serif',
        fillColor: Cesium.Color.fromCssColorString('#1d2430'),
        outlineColor: Cesium.Color.WHITE,
        // 12px 配 3px 描边会把汉字糊成一团——汉字笔画本身才 1px 出头，3px 的白边
        // 从两侧吃掉笔画，字怀就填死了。降到 1.5 才留得住字形。
        outlineWidth: 1.5,
        style: Cesium.LabelStyle.FILL_AND_OUTLINE,
        pixelOffset: new Cesium.Cartesian2(0, -24),
        verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
        showBackground: true,
        backgroundColor: Cesium.Color.fromCssColorString('rgba(255,255,255,0.86)'),
        // 标牌固定字号：不随距离缩放（去掉 scaleByDistance）。
      },
    });

    // 历史轨迹：截至当前帧，目标已经走过的航迹。
    entity.polyline = {
      positions: new Cesium.CallbackProperty(function () {
        const rec = state.targets[def.id];
        // 完整历史（第 0 帧 → 当前帧）。窗口只影响掩码/取证读数，不影响绘制。
        return rec ? rec.trail : [];
      }, false),
      width: 2,
      material: color.withAlpha(0.85),
      clampToGround: false,
    };

    // 计划轨迹：虚线 + 终点箭头，由「历史轨迹」图层统一管理（不再单独成标绘层）。
    // 与历史实线是两个独立图元，视觉上区分"已发生"与"计划"。
    const predict = viewer.entities.add({
      id: 'predict-' + def.id,
      show: false,
      polyline: {
        positions: new Cesium.CallbackProperty(function () {
          const pts = predictedTrackPoints(def);
          return pts.map(function (p) { return Cesium.Cartesian3.fromDegrees(p[0], p[1], p[2] || 0); });
        }, false),
        width: 2,
        material: new Cesium.PolylineDashMaterialProperty({
          color: color.withAlpha(0.9),
          dashLength: 14,
        }),
        clampToGround: false,
      },
      // 终点箭头：billboard 转成屏幕像素，指向计划的终点（不随缩放变形）。
      position: new Cesium.CallbackProperty(function () {
        const pts = predictedTrackPoints(def);
        return pts.length >= 2
          ? Cesium.Cartesian3.fromDegrees(pts[pts.length - 1][0], pts[pts.length - 1][1],
                                          pts[pts.length - 1][2] || 0)
          : undefined;
      }, false),
      billboard: {
        image: PREDICT_ARROW_URI,
        width: 16,
        height: 16,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
        // 用 polylines 的 material 颜色没法复用到 billboard，直接用阵营色近似
        color: color,
      },
    });

    // 传感器覆盖范围，默认隐藏。
    const sensorEntity = viewer.entities.add({
      id: 'sensor-' + def.id,
      show: false,
      position: new Cesium.CallbackProperty(function () {
        const s = positionFor(def.id, state.tick);
        if (!s) return undefined;
        return Cesium.Cartesian3.fromDegrees(s.lon, s.lat, s.alt);
      }, false),
      ellipse: {
        semiMajorAxis: def.sensor_range_km * 1000,
        semiMinorAxis: def.sensor_range_km * 1000,
        material: color.withAlpha(0.07),
        outline: true,
        outlineColor: color.withAlpha(0.55),
        outlineWidth: 1,
        height: 0,
      },
    });

    // 血条：所有出现目标 icon 的地方都要带血条（用户 2026-10-21 要求）。
    // 主 3D 图里图标 34px、center 锚定，血条 TOP 锚定在 +19px（17 半高 + 2 间距）。
    // 透明度在 TargetStatus.barUri 里按档位编码。
    const hpBar = viewer.entities.add({
      id: "hp-" + def.id,
      position: new Cesium.CallbackProperty(function () {
        const s = positionFor(def.id, state.tick);
        if (!s) return undefined;
        return Cesium.Cartesian3.fromDegrees(s.lon, s.lat, s.alt);
      }, false),
      billboard: {
        image: window.TargetStatus ? TargetStatus.barUri(100, 0, 30, 4) : undefined,
        width: 30,
        height: 4,
        verticalOrigin: Cesium.VerticalOrigin.TOP,
        pixelOffset: new Cesium.Cartesian2(0, 19),
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
    });

    // 历史轨迹 = 上面的实线 polyline（全程）；计划轨迹 = 上面的 predict 实体
    // （虚线 + 终点箭头）。两者都由「历史轨迹」图层统一管理。
    state.targets[def.id] = {
      def: def,
      entity: entity,
      predict: predict,
      sensorEntity: sensorEntity,
      hpBar: hpBar,
      trail: [],
      lastTick: -1,
    };
  });
}

/** targetFx.js 需要的只读环境（不在特效模块里直接摸 app.js 的闭包）。 */
function targetFxApi() {
  return {
    isVisible: function (def) { return isVisible(def); },
    sample: function (id) { return positionFor(id, state.tick); },
    entity: function (id) { const rec = state.targets[id]; return rec ? rec.entity : null; },
    clockTime: function () { return viewer ? viewer.clock.currentTime : null; },
    playing: function () { return !!state.playing; },
    cameraHeight: cameraHeightMeters,
    tick: function () { return state.tick; },
    tickFloat: function () { return state.tickFloat; },
    selected: function () { return state.selected; },
  };
}

/**
 * 装载 CZML 运动源并初始化特效。
 *
 * 实体先由 createTargetEntities() 建好，CZML 采样属性在异步 load 完成后挂上去
 * （见 targetMotion.js）。特效与运动都从 viewer.clock 取时间，而 viewer.clock
 * 由 loop() 里的 tickFloat 每帧同步，三条链路共用同一个时间源。
 */
function attachMotionAndFx(data) {
  if (window.TargetFx) {
    try { TargetFx.init(viewer, data, targetFxApi()); }
    catch (e) { console.warn('[fx] 初始化失败：', e); }
  }
  if (window.TargetCombatFx) {
    try { TargetCombatFx.init(viewer, data, targetFxApi()); }
    catch (e) { console.warn("[combat-fx] 初始化失败：", e); }
  }
  if (window.TargetMotion) {
    TargetMotion.load(viewer, data).then(function (st) {
      if (!st || state.data !== data) return;
      markScene('czml-ready');
    }).catch(function (e) {
      // CZML 装载失败时保留 createTargetEntities 里的 CallbackProperty 兜底位置，
      // 功能退化但场景不空。
      console.warn('[czml] 装载失败，保留逐帧位置：', e);
    });
  }
}

function createRouteEntities() {
  state.data.targets.forEach(function (def) {
    if (!def.waypoints || def.waypoints.length < 2) return;
    const flat = [];
    def.waypoints.forEach(function (wp) { flat.push(wp[1], wp[0], def.altitude_m || 0); });
    const entity = viewer.entities.add({
      id: 'route-' + def.id,
      polyline: {
        positions: Cesium.Cartesian3.fromDegreesArrayHeights(flat),
        width: 1.5,
        material: new Cesium.PolylineDashMaterialProperty({
          color: colorFor(def.affiliation).withAlpha(0.5),
          dashLength: 12,
        }),
        clampToGround: false,
      },
    });
    state.routeEntities.push(entity);
  });
}

// --------------------------------------------------------------------------
// 动态热点 / 动态占领区（按推演数据判定，不依赖预设矩形）
// --------------------------------------------------------------------------
//
// 判定依据（数据里实际具备的字段，见 data/situations/*.json）：
//   · 位置        frames[i].targets[*].lon/lat/alt
//   · 敌对关系    inference_frames[i].targets[*].threat / alert / threat_terms
//                 （threat_terms 给 proximity / coverage / base，能区分"为什么高"）
//   · 交战判定    双方 engage_km 之和（与关系线用同一口径）
// 热点 = **双方兵力都存在的交战/对峙聚集区**：
//   对每个敌对目标对，若距离 ≤ engage 半径和 → 强度 = 双方 threat 均值 × 1.2
//                                        否则若距离 ≤ 双方影响半径和 → 强度 = 均值 × 0.5
//   再把强度按网格聚合成"簇"（相邻格点合并），簇中心即热点位置；
//   半径由簇内目标的空间散度决定（覆盖实际接触面，而不是固定框）。
// 占领区 = 每个目标按其类型给出**影响半径**（舰/防空/炮兵/地面），
//   半径按该目标当前 threat 缩放（0.5x~1x）；同阵营叠加，交集处判为争夺。

const HOTSPOT_GRID_DEG = 0.35;      // 聚合网格（约 35 km）
const HOTSPOT_MIN_SCORE = 0.6;      // 低于此强度不入热点
const HOTSPOT_MAX = 12;             // 上限，防止战区级铺满

// 每类目标的影响半径（km）——取"能覆盖到的战场范围"，不是射程本身。
function influenceRadiusKm(def) {
  const ic = Number(def.influence_km) || 0;
  if (ic > 0) return ic;
  if (def.domain === 'air') return 160;
  if (def.domain === 'sea') return 100;
  return 80;
}

/** 当前帧的 threat 表（0 表示没有推理数据）。 */
function threatOf(id) {
  const inf = inferenceAt(state.tick);
  const t = inf && inf.targets && inf.targets[id];
  return { threat: t ? (Number(t.threat) || 0) : 0,
           alert: t ? (Number(t.alert) || 0) : 0 };
}

/** 当前帧全部目标的 {id, def, pos, side, threat}。 */
function liveTargets() {
  const out = [];
  Object.keys(state.targets).forEach(function (id) {
    const rec = state.targets[id];
    const pos = positionFor(id, state.tick);
    if (!rec || !pos) return;
    const th = threatOf(id);
    out.push({ id: id, def: rec.def, pos: pos, threat: th.threat });
  });
  return out;
}

/** km → 经/纬度跨度（局部近似，足够画覆盖区）。 */
function kmToLonLat(km, lat) {
  const mLon = 111.32 * Math.max(0.2, Math.cos(lat * Math.PI / 180));
  return { lon: km / mLon, lat: km / 110.54 };
}

/**
 * 动态热点。
 * 返回 `[{ id, name, center:[lat,lon], radiusKm, score, contacts }]`。
 */
// --------------------------------------------------------------------------
// 面状要素的存储格式：GeoJSON
// --------------------------------------------------------------------------
//
// 圆、交火区这类面状要素在**内部存储与对外交换**时统一用 GeoJSON：
//   · 存储：每个热点/占领区带上 `feature`（Polygon，WGS84 经纬度），
//     不再只有"圆心 + 半径"这种私有形式；
//   · 交换：SituationHost.plotFeatureCollection() 把标绘/热点/占领导出为
//     FeatureCollection，QGIS、其它系统可直接读；
//   · 渲染仍走 Entity（贴地面 + 贴地边界），不用 GeoJsonDataSource：
//     后者每次更新整体重建，逐帧 31 个目标会明显掉帧。
// 只支持 WGS84，不做其它 CRS 转换。

/** 经纬度环 → GeoJSON Polygon Feature（环自动闭合）。 */
function ringToFeature(ring, props) {
  const r = ring.slice();
  if (r.length < 3) return null;
  if (r[0][0] !== r[r.length - 1][0] || r[0][1] !== r[r.length - 1][1]) r.push(r[0]);
  return {
    type: 'Feature',
    geometry: { type: 'Polygon', coordinates: [r] },
    properties: props || {},
  };
}

/** 读取 Feature 的外环（MultiPolygon 取第一片）。非法输入返回空数组。 */
function featureRing(feature) {
  if (!feature || feature.type !== 'Feature' || !feature.geometry) return [];
  const g = feature.geometry;
  if (g.type === 'Polygon' && g.coordinates && g.coordinates[0]) return g.coordinates[0];
  if (g.type === 'MultiPolygon' && g.coordinates && g.coordinates[0] && g.coordinates[0][0]) {
    return g.coordinates[0][0];
  }
  return [];
}

/** Feature 的属性/几何是否可用于贴地面。 */
function featureValid(feature) {
  const r = featureRing(feature);
  return r.length >= 4 && r.every(function (c) {
    return c && isFinite(c[0]) && isFinite(c[1]);
  });
}

function computeHotspots() {
  const list = liveTargets();
  const cells = {};        // "gi|gj" -> { score, pts:[], contacts }
  for (let i = 0; i < list.length; i++) {
    for (let j = i + 1; j < list.length; j++) {
      const A = list[i], B = list[j];
      if (A.def.affiliation === B.def.affiliation) continue;
      const km = relKm(A.pos, B.pos);
      const engR = (Number(A.def.engage_km) || 0) + (Number(B.def.engage_km) || 0);
      const infR = influenceRadiusKm(A.def) + influenceRadiusKm(B.def);
      // 强度基础：有推理威胁就用它；数据集没有推理帧（如 CZML 驱动的新案例）时
      // 退回**纯几何**判据——至少"进入交战射程"本身就是一个战场信号。
      const tA = A.threat > 0 ? A.threat : 40;
      const tB = B.threat > 0 ? B.threat : 40;
      let score = 0;
      if (engR > 0 && km <= engR) score = ((tA + tB) / 2) * 1.2;
      else if (km <= infR) score = ((tA + tB) / 2) * 0.5;
      if (score < HOTSPOT_MIN_SCORE) continue;
      // 热点落在两点中点；相近的交战对并入同一格
      const lat = (A.pos.lat + B.pos.lat) / 2, lon = (A.pos.lon + B.pos.lon) / 2;
      const key = Math.round(lat / HOTSPOT_GRID_DEG) + '|' + Math.round(lon / HOTSPOT_GRID_DEG);
      if (!cells[key]) cells[key] = { score: 0, pts: [], contacts: [] };
      cells[key].score += score;
      cells[key].pts.push({ lat: lat, lon: lon, a: A, b: B });
      cells[key].contacts.push(km <= engR ? 'engage' : 'proximity');
    }
  }
  const out = Object.keys(cells).map(function (key) {
    const c = cells[key];
    const lat = c.pts.reduce(function (s, p) { return s + p.lat; }, 0) / c.pts.length;
    const lon = c.pts.reduce(function (s, p) { return s + p.lon; }, 0) / c.pts.length;
    // 半径 = 簇内接触点的空间散度 + 余量；上限收到 80 km——之前 120 km 的圆
    // 在战区级视图上几乎盖住半个海峡，反而看不出"热点在哪里"。
    let maxD = 0;
    c.pts.forEach(function (p) {
      maxD = Math.max(maxD, relKm({ lat: lat, lon: lon }, p));
    });
    const radiusKm = Math.max(15, Math.min(80, maxD * 1.3 + 12));
    const engageN = c.contacts.filter(function (x) { return x === 'engage'; }).length;
    // 名称只写性质：坐标就写在图上，写进标签只是噪声。
    const name = engageN > 0 ? '交火区' : '对峙区';
    const ring = circleRing(lat, lon, radiusKm, 28);
    return { id: key, name: name, center: [lat, lon], radiusKm: radiusKm,
             score: c.score, contacts: c.pts.length, engaging: engageN,
             // 存储格式：GeoJSON Polygon（与渲染同源，导出即可交换）
             feature: ringToFeature(ring, {
               kind: 'hotspot', level: engageN > 0 ? 'engage' : 'proximity',
               score: Math.round(c.score * 10) / 10, radiusKm: radiusKm,
             }) };
  });
  out.sort(function (a, b) { return b.score - a.score; });
  // ---- 合并重叠簇 ----
  // 网格聚合按 0.35° 切分，同一片交战区常被切成两三个相邻格，各自画一个
  // 30–60 km 的圆互相大面积重叠，读起来像"两个不同的交火区"，其实是同一处。
  // 这里做一次贪心吸收：后者的圆心落进前者的圆内（或反之）就并入前者——
  // 圆心按强度加权、半径取能同时包住两者的最小半径（距离 + 对方半径）。
  let merged = out.slice();
  // 反复吸收直到稳定：一次遍历只能合并相邻簇，合并后的新圆心会重新覆盖第三个簇，
  // 必须迭代到"任意两个圆心都不落在对方半径内"为止（实测 11 个热点里仍有两两重叠）。
  for (let pass = 0; pass < 12; pass++) {
    let changed = false;
    const next = [];
    merged.forEach(function (h) {
      for (let i = 0; i < next.length; i++) {
        const m = next[i];
        const d = relKm({ lat: m.center[0], lon: m.center[1] },
                         { lat: h.center[0], lon: h.center[1] });
        // 任一方圆内包含对方圆心 → 同一处战场，合并。
        if (d < Math.max(m.radiusKm, h.radiusKm)) {
          const wSum = m.score + h.score;
          const lat = (m.center[0] * m.score + h.center[0] * h.score) / wSum;
          const lon = (m.center[1] * m.score + h.center[1] * h.score) / wSum;
          m.center = [lat, lon];
          m.radiusKm = Math.min(100, Math.max(m.radiusKm, d + h.radiusKm));
          m.score += h.score;
          m.contacts += h.contacts;
          m.engaging += h.engaging;
          m.merged = (m.merged || 1) + 1;
          changed = true;
          return;
        }
      }
      next.push(h);
    });
    merged = next;
    if (!changed) break;
  }
  merged.forEach(function (m) {
    m.feature = ringToFeature(circleRing(m.center[0], m.center[1], m.radiusKm, 28), {
      kind: 'hotspot', level: m.engaging > 0 ? 'engage' : 'proximity',
      score: Math.round(m.score * 10) / 10, radiusKm: m.radiusKm,
      merged: m.merged || 1,
    });
  });
  merged.sort(function (a, b) { return b.score - a.score; });
  return merged.slice(0, HOTSPOT_MAX);
}

/**
 * 动态占领区：每个目标输出一个圆形影响区（贴地），同阵营叠加。
 * 返回 `[{ side, lat, lon, radiusKm, id }]`；渲染时按阵营着色。
 */
function computeOccupationArea() {
  const list = liveTargets();
  return list.map(function (t) {
    const r = influenceRadiusKm(t.def);
    // threat 越高覆盖越实：0 威胁也保留 55% 的"存在范围"
    const k = 0.55 + 0.45 * Math.min(1, Math.max(0, t.threat) / 70);
    const ring = circleRing(t.pos.lat, t.pos.lon, r * k, 20);
    return { id: t.id, side: t.def.affiliation, lat: t.pos.lat, lon: t.pos.lon,
             radiusKm: r * k,
             feature: ringToFeature(ring, {
               kind: 'occupation', side: t.def.affiliation, target: t.id,
             }) };
  });
}

/**
 * 重建热点 / 占领区域图元。
 *
 * 旧实现是"数据集预设的 18 个固定矩形"，既不是按战况生成，也无法反映变化。
 * 现在两者都按当前帧动态计算：
 *   热点 —— 交战/对峙聚集区（computeHotspots），用标绘体系里的 `recon_area`
 *          多边形贴地绘制（圆按 24 段近似），同一图层由 Plotting 管理；
 *   占领 —— 每个目标的影响圆（computeOccupationArea），归「占领区域」图层。
 * 只在**帧号或选中/过滤变化**时重建，避免每帧反复建实体。
 */
function dynamicAreasKey() {
  return [state.tick, state.layers.hotspots ? 1 : 0, state.layers.occupation ? 1 : 0,
          state.filters.red ? 1 : 0, state.filters.blue ? 1 : 0,
          state.filters.sea ? 1 : 0, state.filters.land ? 1 : 0,
          state.filters.air ? 1 : 0].join('|');
}
var _dynAreasKey = null;

function rebuildDynamicAreas(force) {
  if (!viewer) return;
  // ⚠ 实测：本函数在 `createHotspots()` 里被调用时，`state.tick` 与 `layers` 已就绪；
  // 但**不能**只在 key 相同时短路——开局会把"0 个热点"写进 key，之后数据补齐
  // 也不会重算。因此 force=true 是唯一的确定性入口（案例加载、图层开关、手工刷新）。
  if (force) _dynAreasKey = null;
  // 局部 import 一次就能全部重建，开销很小；只有**播放推进**（每秒几十次）才需要
  // 用 key 短路。之前只在"帧号+开关+数量"变化时重建，而开局时数量恰好都是 0
  // （数据集未就绪），key 不再变化，热点就永远不出现（实测）。
  const key = dynamicAreasKey();
  if (!force && key === _dynAreasKey) return;
  // 结果集本身就进 key：开局时 inference 往往还没就绪，强度全 0 → 热点为空；
  // 只在帧号/开关上判断会让"数据到了但没重建"永久空着（实测）。
  const spots = computeHotspots();
  const setKey = key + '#' + spots.length;
  if (!force && setKey === _dynAreasKey) return;
  _dynAreasKey = setKey;
  rebuildHotspotEntities(spots);
  // ⚠ 「占领区域」已按用户要求**整块下线**（2026-10-20）：图层、渲染、小地图掩码
  // 与要素列表都不再创建。`computeOccupationArea` / `rebuildOccupationEntities`
  // 保留未删，确认不再需要时可整段删除；恢复方式：去掉此注释并恢复图层行。
  // rebuildOccupationEntities(areas);
}

/** 圆 → 贴地多边形（经纬度近似）。 */
function circleRing(lat, lon, radiusKm, steps) {
  const n = steps || 24;
  const d = kmToLonLat(radiusKm, lat);
  const ring = [];
  for (let i = 0; i < n; i++) {
    const a = (i / n) * Math.PI * 2;
    ring.push([lon + Math.cos(a) * d.lon, lat + Math.sin(a) * d.lat]);
  }
  return ring;
}

function rebuildHotspotEntities(spots) {
  state.hotspotEntities.forEach(function (e) {
    try { viewer.entities.remove(e); } catch (err) { /* 已移除 */ }
  });
  state.hotspotEntities = [];
  if (!state.layers.hotspots) return;
  spots = spots || computeHotspots();
  spots.forEach(function (hs, i) {
    const ring = hs.feature ? featureRing(hs.feature)
                            : circleRing(hs.center[0], hs.center[1], hs.radiusKm, 28);
    const color = hs.engaging > 0
      ? Cesium.Color.fromCssColorString('#d84a3a')
      : Cesium.Color.fromCssColorString('#f0a020');
    const entity = viewer.entities.add({
      id: 'hotspot-' + i,
      name: hs.name,
      polygon: {
        hierarchy: new Cesium.PolygonHierarchy(
          Cesium.Cartesian3.fromDegreesArray(ring.reduce(function (a, p) {
            return a.concat([p[0], p[1]]); }, []))),
        material: color.withAlpha(0.18),
        classificationType: Cesium.ClassificationType.BOTH,
      },
      polyline: {
        positions: Cesium.Cartesian3.fromDegreesArray(ring.concat([ring[0]]).reduce(function (a, p) {
          return a.concat([p[0], p[1]]); }, [])),
        width: 2, clampToGround: true, material: color,
      },
      // 热点**不写文字**：面本身已经说明了位置和范围，名称只在图层要素列表里
      // 用于定位；图上再写一遍只会和密集的目标标牌抢空间。
      position: Cesium.Cartesian3.fromDegrees(hs.center[1], hs.center[0]),
    });
    entity.hsData = hs;
    state.hotspotEntities.push(entity);
  });
  state.hotspotRows = spots.map(function (hs, i) {
    return { key: 'hotspot-' + i, name: hs.name, level_label: hs.engaging > 0 ? '交火' : '对峙',
             score: hs.score, radiusKm: hs.radiusKm, center: hs.center };
  });
  if (window.LayerPanel) LayerPanel.invalidate();
}

function rebuildOccupationEntities(areas) {
  state.occupationEntities.forEach(function (e) {
    try { viewer.entities.remove(e); } catch (err) { /* 已移除 */ }
  });
  state.occupationEntities = [];
  if (!state.layers.occupation) return;
  areas = areas || computeOccupationArea();
  areas.forEach(function (a, i) {
    if (a.side === 'red' && !state.filters.red) return;
    if (a.side === 'blue' && !state.filters.blue) return;
    const color = a.side === 'blue' ? OCCUPATION_STATUS_COLORS.blue
                                    : OCCUPATION_STATUS_COLORS.red;
    const ring = a.feature ? featureRing(a.feature)
                           : circleRing(a.lat, a.lon, a.radiusKm, 20);
    const flat = ring.reduce(function (acc, p) { return acc.concat([p[0], p[1]]); }, []);
    const closed = ring.concat([ring[0]]);
    const closedFlat = closed.reduce(function (acc, p) { return acc.concat([p[0], p[1]]); }, []);
    const ent = viewer.entities.add({
      id: 'occupation-' + i,
      name: a.id + ' 控制区',
      polygon: {
        hierarchy: new Cesium.PolygonHierarchy(Cesium.Cartesian3.fromDegreesArray(flat)),
        // 31 个影响圆会互相叠加：填充压得很淡（只用来表示"覆盖到的范围"），
        // 边界单独用贴地细线描出来，叠在一起时才看得出各自的轮廓。
        material: color.withAlpha(0.07),
        classificationType: Cesium.ClassificationType.BOTH,
      },
      polyline: {
        positions: Cesium.Cartesian3.fromDegreesArray(closedFlat),
        width: 1,
        clampToGround: true,
        material: color.withAlpha(0.55),
      },
    });
    ent.occTarget = a.id;
    state.occupationEntities.push(ent);
  });
}

// 兼容旧调用点：原先 createHotspots() 同时建热点与占领面。
// 现在统一走重建入口（force=true 时忽略帧号缓存）。
function createHotspots() {
  // 启动时立即建一次（`state.layers.hotspots` 已在 initViewer 里预置为 true，
  // 见那里的说明）；帧号变化后再由 updateOccupationAreas 按 key 增量重建。
  _dynAreasKey = null;
  rebuildDynamicAreas(true);
}

/** 方向 4 取证出口：动态热点数量（旧数据字段已不适用，保留结构）。 */
function hotspotFacilities() {
  var out = { data: {}, anchors: {}, labels: {} };
  (state.hotspotEntities || []).forEach(function (e) {
    var kind = e.hsData ? (e.hsData.engaging > 0 ? 'engage' : 'proximity') : 'other';
    out.data[kind] = (out.data[kind] || 0) + 1;
    if (e.label) out.labels[kind] = (out.labels[kind] || 0) + 1;
  });
  return out;
}

// --------------------------------------------------------------------------
// 逐帧更新
// --------------------------------------------------------------------------

/**
 * 声明"目标/路线/热点的可见性需要重算"。
 *
 * 所有会改变 syncEntities 结果的地方都必须调它：时间轴、播放推进、数据装载、
 * 图层与过滤开关、选中变化。漏一处 = "改了不生效、刷新才生效"，所以这里包一层
 * 带名字的简写，调用点一眼能看出自己该不该调。
 */
function markScene(reason) {
  if (window.RenderGovernor) RenderGovernor.mark(reason);
}

/**
 * 计划轨迹的显隐：单独抽出来，供 syncEntities 与**选中/取消选中**直接调用。
 * 不能只依赖 syncEntities 的脏标记：它由帧循环里的 RenderGovernor 驱动，
 * 在低帧率（或无头）环境下可能要等上好几秒，表现为"点了目标、计划线迟到"。
 */
// --------------------------------------------------------------------------
// 屏幕空间标签避让（目标标牌 + 关系文字）
// --------------------------------------------------------------------------
//
// Cesium 不会在实体之间做标签避让：目标密集时标牌叠成一团，关系文字的
// 中点也常落在别的标牌上。这里统一做一次屏幕空间矩形相交判定：
//
//   优先级  选中目标 > 关系文字 > 其它目标标牌
//   规则    被更高优先级的矩形压住就隐藏这一帧（**不删除实体**，只置 show）
//   代价    每帧约 N 次 worldToWindowCoordinates（N≈60），按 200 ms 节流
//
// 被隐藏的标签：label 对象 -> 连续"仍然不重叠"的轮数。
// 规则是**一重叠就藏、连续两轮不重叠才恢复**——既满足"隐藏直到不重叠"，
// 也避免两个标签在边界附近互相顶掉造成闪烁。

var _declutterHidden = new WeakMap();
var _declutterLastMs = 0;
var DECLUTTER_THROTTLE_MS = 150;
var DECLUTTER_PAD = 2;
var DECLUTTER_CLEAR_PASSES = 2;

/**
 * 估一个标签的屏幕矩形（Cesium 没给 label 的包围盒，按字号与字数近似）。
 *
 * `pos` 是**实体锚点**；标签相对锚点还有 pixelOffset 与 verticalOrigin：
 *   目标标牌：pixelOffset(0,-24)、verticalOrigin.BOTTOM → 底边在 anchorY-24
 *   关系文字：pixelOffset(0,0)、CENTER → 中心在 anchorY
 */
function labelRect(pos, text, size, anchorOffsetY, verticalCenter) {
  var chars = String(text || '').length;
  // 中文按 1.0 字宽、ASCII 按 0.55；背景 padding 各 4/2 px。
  var w = 0;
  for (var i = 0; i < chars; i++) {
    w += String(text).charCodeAt(i) > 255 ? size : size * 0.55;
  }
  w += 10;
  var h = size + 8;
  var cy = pos.y + (anchorOffsetY || 0);
  return { x: pos.x - w / 2, y: verticalCenter ? cy - h / 2 : cy - h, w: w, h: h };
}

function rectsOverlap(a, b) {
  return !(a.x + a.w + DECLUTTER_PAD < b.x || b.x + b.w + DECLUTTER_PAD < a.x ||
           a.y + a.h + DECLUTTER_PAD < b.y || b.y + b.h + DECLUTTER_PAD < a.y);
}

function declutterLabels(force) {
  if (!viewer || !window.Cesium || !Cesium.SceneTransforms) return null;
  var now = (typeof performance !== 'undefined' ? performance.now() : Date.now());
  if (!force && now - _declutterLastMs < DECLUTTER_THROTTLE_MS) return null;
  _declutterLastMs = now;

  var time = viewer.clock.currentTime;
  var items = [];

  Object.keys(state.targets).forEach(function (id) {
    var rec = state.targets[id];
    var ent = rec && rec.entity;
    if (!ent || !ent.show || !ent.label) return;
    // ⚠ 这里**不看 label.show**：它可能正是上一轮被我们关掉的。
    var text = ent.label.text;
    text = (text && text.getValue) ? text.getValue() : text;
    if (!text) return;
    var p = ent.position && ent.position.getValue ? ent.position.getValue(time) : null;
    if (!p) return;
    var w = Cesium.SceneTransforms.worldToWindowCoordinates(viewer.scene, p);
    if (!w) return;
    items.push({ label: ent.label, text: text, size: 12,
                 pri: state.selected === id ? 3 : 1, w: w, offY: -24, vc: false });
  });

  relView.ents.forEach(function (e) {
    if (!e.show || !e.label) return;
    var text = e.label.text;
    text = (text && text.getValue) ? text.getValue() : text;
    if (!text) return;
    var p = e.position && e.position.getValue ? e.position.getValue(time) : null;
    if (!p) return;
    var w = Cesium.SceneTransforms.worldToWindowCoordinates(viewer.scene, p);
    if (!w) return;
    items.push({ label: e.label, text: text, size: 12, pri: 2, w: w, offY: 0, vc: true });
  });

  items.sort(function (a, b) { return (b.pri - a.pri) || (a.w.y - b.w.y); });

  var placed = [];
  var hidden = 0, shown = 0;
  items.forEach(function (it) {
    var r = labelRect(it.w, it.text, it.size, it.offY, it.vc);
    var fits = true;
    if (it.w.x < -r.w || it.w.y < -r.h ||
        it.w.x > viewer.canvas.clientWidth + r.w || it.w.y > viewer.canvas.clientHeight + r.h) {
      fits = false;                          // 视窗外同样按"不可见"处理
    } else {
      for (var i = 0; i < placed.length; i++) {
        if (rectsOverlap(r, placed[i])) { fits = false; break; }
      }
    }

    if (fits) {
      var n = (_declutterHidden.get(it.label) || 0) + 1;
      if (_declutterHidden.has(it.label) && n < DECLUTTER_CLEAR_PASSES) {
        // 刚空出来一轮还不够：继续藏，防抖。
        _declutterHidden.set(it.label, n);
        it.label.show = false; hidden++;
        return;
      }
      _declutterHidden.delete(it.label);
      it.label.show = true; shown++;
      placed.push(r);
    } else {
      _declutterHidden.set(it.label, 0);
      it.label.show = false; hidden++;
    }
  });

  return { total: items.length, shown: shown, hidden: hidden,
           relHidden: relView.ents.filter(function (e) {
             return e.label && _declutterHidden.has(e.label); }).length };
}

function refreshPredictEntity(id) {
  const rec = state.targets[id];
  if (!rec || !rec.predict) return;
  const pred = (state.selected === id && isVisible(rec.def) && state.layers.tracks)
    ? predictedTrackPoints(rec.def) : [];
  // The arrow asset points north. Cesium billboard rotation is CCW, so a
  // clockwise compass bearing is applied as -bearing.
  if (rec.predict.billboard) {
    var rot = 0;
    if (pred.length >= 2) {
      var a = pred[pred.length - 2], b = pred[pred.length - 1];
      rot = -predBearing(a, b) * Math.PI / 180;
    }
    rec.predict.billboard.rotation = rot;
    rec.predict.billboard.alignedAxis = Cesium.Cartesian3.UNIT_Z;
  }
  setEntityShow(rec.predict, pred.length >= 2);
}

function syncEntities() {
  Object.keys(state.targets).forEach(function (id) {
    const rec = state.targets[id];
    const def = rec.def;
    const visible = isVisible(def);
    // 形态（2D 图标 / 3D 模型）必须在这里一起解算：syncEntities 会在每次脏标记
    // 重跑时重写 billboard.show。现在图标**在两种形态下都显示**（2026-10-19 按
    // 用户要求），差别只在锚点：模型形态由 applyTargetBillboardAnchor 抬到头顶。
    // 两处（这里与 applyTargetLod）必须用同一套判据，否则锚点会在一次重算后
    // 退回"居中"，表现为低空时图标忽然落回模型身上。
    const useModel = !!(window.EntityLod && lastTargetForm === window.EntityLod.MODE_MODEL
      && targetHasModel[id]);

    rec.entity.show = visible;
    // 标签避让的隐藏是**另一层状态**：syncEntities 每次脏帧都会重写 show，
    // 不乘上它就会把隐藏的标签重新点亮（实测表现为标牌闪烁）。
    rec.entity.label.show = visible && state.layers.labels
      && !_declutterHidden.has(rec.entity.label);
    // 图标始终显示（模型形态抬到头顶），原因见 applyTargetBillboardAnchor。
    rec.entity.billboard.show = visible;
    applyTargetBillboardAnchor(rec.entity, useModel);
    if (rec.entity.model) rec.entity.model.show = visible && useModel;
    rec.sensorEntity.show = visible && state.filters.sensors === true;

    // 主 3D 图标底部血条：与整体TS/鸟瞰图同一 TargetStatus 口径。
    // 只在档位或 0.1% 血量变化时换图，避免每帧重建 SVG。
    if (rec.hpBar) {
      const hpSt = (window.TargetStatus && state.data)
        ? TargetStatus.at(state.data, id, state.tick) : null;
      const hpKey = hpSt ? (hpSt.status + "|" + Math.round(hpSt.hp * 10)) : "none";
      if (rec.hpKey !== hpKey) {
        rec.hpKey = hpKey;
        rec.hpBar.billboard.image = hpSt
          ? TargetStatus.barUri(hpSt.hp, hpSt.status, 30, 4)
          : undefined;
      }
      rec.hpBar.show = visible;
    }

    // 航迹单调累积。回退时不是简单清空，而是从第 0 帧重建，
    // 否则在暂停状态拖动时间轴会导致航迹永久消失。
    if (state.tick < rec.lastTick) {
      rec.trail.length = 0;
      rec.lastTick = -1;
    }
    if (visible && state.layers.tracks && state.tick > rec.lastTick) {
      for (let t = rec.lastTick + 1; t <= state.tick; t++) {
        const s = positionFor(id, t);
        if (s) rec.trail.push(Cesium.Cartesian3.fromDegrees(s.lon, s.lat, s.alt));
      }
    }
    rec.lastTick = state.tick;

    if (!state.layers.tracks) rec.trail.length = 0;
    // 历史轨迹**全程实线**：不按窗口切片。窗口只用于掩码取值（updateFocusMask），
    // 绘制给用户看的是从第 0 帧到当前帧的完整航迹（需求：不要只画固定步长）。
    rec.entity.polyline.show = visible && state.layers.tracks && rec.trail.length > 1;
    // 计划轨迹与历史轨迹同属「历史轨迹」图层；只给选中目标画，避免满屏虚线。
    refreshPredictEntity(id);
  });
  // Auto plots must follow every tick even when the tick was set by a deep
  // link or a jump that bypasses setTick(); this is a cheap no-op when nothing
  // changed because updateAutoPlots only rebuilds dynamic items on tick change.
  updateAutoPlots(state.tick);
  // 掩码遮的是"整图减去选中目标与轨迹"：选中、时间、图层任一变化都要重画。
  updateFocusMask();

  // 图层开关 ∩ 该要素未被单独关掉。实体 id 里带着身份（`route-<目标id>` /
  // `hotspot-<序号>`），直接从中解析，不再另存一份映射 —— 另存一份就多一处会不同步。
  state.routeEntities.forEach(function (e) {
    var id = typeof e.id === 'string' ? e.id : '';
    var key = id.indexOf('route-') === 0 ? id.slice(6) : null;
    e.show = !!state.layers.routes && !(key && state.hiddenRoutes[key]);
  });
  // 热点/占领区都贴地，不再用"固定高度 + 低于该视点不可见"的旧判据。
  state.hotspotEntities.forEach(function (e) {
    var id = typeof e.id === 'string' ? e.id : '';
    var key = id.indexOf('hotspot-') === 0 ? id.slice(8) : null;
    e.show = !!state.layers.hotspots && !(key && state.hiddenHotspots[key]);
  });
  updateOccupationAreas();
  if (state.selected) refreshRelations();
}
/**
 * 轨迹显示窗口：历史轨迹全量保留（回退到早期帧仍能重建），只在渲染时切片。
 * 未选中 → 最近 TRAIL_WINDOW_TICKS 帧；选中 → 全量（选中态给的是"历史实线"）。
 */
function trailWindowOf(rec, id) {
  const t = rec.trail;
  if (state.selected === id || t.length <= TRAIL_WINDOW_TICKS) return t;
  return t.slice(t.length - TRAIL_WINDOW_TICKS);
}

/**
 * 计划轨迹：从当前位置沿计划航路剩余段，或是航向外推。
 * 返回 `[lon, lat, alt]`：虚线 + 终点箭头都基于它，与历史实线分开绘制。
 */
/** Bearing from point A to B in degrees, clockwise from north. Both are [lon, lat, alt]. */
function predBearing(a, b) {
  var cos = Math.cos((a[1] || 0) * Math.PI / 180) || 1;
  return (Math.atan2((b[0] - a[0]) * cos, b[1] - a[1]) * 180 / Math.PI + 360) % 360;
}

function predictedTrackPoints(def) {
  const s = positionFor(def.id, state.tick);
  if (!s) return [];
  const out = [[s.lon, s.lat, s.alt]];
  if (!def.waypoints || def.waypoints.length < 2) {
    // 没有航路点（例如 CZML 驱动的新案例）：用**后续帧的计划位置**作为计划轨迹。
    // 密集案例只取前 10 分钟，并在第一次明显转向处截断——否则闭环巡逻/折返
    // 会在 30 分钟窗口里画成一整圈多边形，看起来像"乱画"。
    const frames = (state.data && state.data.frames) || [];
    // Predict to the next turning point, not a fixed wall-clock horizon.
    // The cap is only a safety valve for a perfectly straight path.
    const dense = !!(state.data && (state.data.campaign_contract_version ||
      ((state.data.targets || []).length > 50)));
    const safetyTicks = dense ? 240 : PREDICT_HORIZON_TICKS;
    if (frames.length > state.tick + 1) {
      const end = Math.min(frames.length - 1, state.tick + safetyTicks);
      var prevBearing = null;
      var lastPt = null;
      for (let t = state.tick + 1; t <= end; t++) {
        const p = positionFor(def.id, t);
        if (!p) continue;
        const pt = [p.lon, p.lat, p.alt];
        if (lastPt) {
          const cos = Math.cos(lastPt[1] * Math.PI / 180) || 1;
          if (Math.hypot((pt[0] - lastPt[0]) * cos, pt[1] - lastPt[1]) < 0.00002) continue;
        }
        if (lastPt && prevBearing !== null && out.length >= 3) {
          const b = predBearing(lastPt, pt);
          const diff = Math.abs(((b - prevBearing + 540) % 360) - 180);
          // Include the turn point itself, then stop: the prediction is
          // "current leg -> next turn", not a fixed 10-minute slice.
          if (diff > 30) { out.push(pt); break; }
        }
        if (lastPt) prevBearing = predBearing(lastPt, pt);
        out.push(pt);
        lastPt = pt;
      }
    }
    return out;
  }
  const alt = def.altitude_m || 0;
  // 取**当前所在航段**的下一段：两点航路时，用"最近航路点"会在中点前先往回折。
  const cosLat = Math.cos(s.lat * Math.PI / 180) || 1;
  let near = 1, best = Infinity;
  for (let i = 1; i < def.waypoints.length; i++) {
    const a = def.waypoints[i - 1], b = def.waypoints[i];
    const ax = (a[1] - s.lon) * cosLat, ay = a[0] - s.lat;
    const bx = (b[1] - s.lon) * cosLat, by = b[0] - s.lat;
    const dx = bx - ax, dy = by - ay;
    const len2 = dx * dx + dy * dy;
    let tt = len2 > 1e-12 ? -(ax * dx + ay * dy) / len2 : 0;
    tt = Math.max(0, Math.min(1, tt));
    const d = Math.hypot(ax + dx * tt, ay + dy * tt);
    if (d < best) { best = d; near = i; }
  }
  for (let j = near; j < def.waypoints.length; j++) {
    const w = def.waypoints[j];
    out.push([w[1], w[0], alt]);
  }
  return out;
}

/** 计划轨迹箭头图标：16×16 的白色三角，按阵营色着色（billboard.color）。 */
const PREDICT_ARROW_URI = (function () {
  const svg = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16">'
    + '<path d="M8 1 L14.5 14 L8 10.5 L1.5 14 Z" fill="#ffffff" '
    + 'stroke="#ffffff" stroke-width="1" stroke-linejoin="round"/></svg>';
  return 'data:image/svg+xml;base64,' + btoa(svg);
})();

/**
 * 幂等开关：ntity.show = x 每次都会换一个新的属性对象，逐帧写会让地面分类图元
 * 反复重建、实测整块掩码永远画不出来。只在状态真的变化时写一次。
 */
function setEntityShow(entity, on) {
  const cur = entity.show;
  const val = (cur && typeof cur.getValue === "function") ? cur.getValue() : cur;
  if (!!val !== !!on) entity.show = !!on;
}

/**
 * 选中掩码：外环全图、内环挖空「选中目标 + 它的轨迹显示窗口」。
 * 图层关着、或没有选中目标 → 隐藏（没选中时掩码没有意义，也不留常驻整图图元）。
 * ⚠ 世界级地面分类多边形在 headless 下要 ~10 s 才画出来（截图验收请给足 settle，别误判"没渲染"）。
 */
function updateFocusMask() {
  const id = state.selected;
  const rec = id ? state.targets[id] : null;
  const s = rec ? positionFor(id, state.tick) : null;
  if (!state.layers.focusMask || !s) {
    if (state.focusMaskEntity) setEntityShow(state.focusMaskEntity, false);
    state.focusMaskHole = null;
    return;
  }
  let w = s.lon - FOCUS_MASK_PAD_DEG;
  let e = s.lon + FOCUS_MASK_PAD_DEG;
  let so = s.lat - FOCUS_MASK_PAD_DEG;
  let n = s.lat + FOCUS_MASK_PAD_DEG;
  const pts = trailWindowOf(rec, id);
  for (let i = 0; i < pts.length; i++) {
    const c = Cesium.Cartographic.fromCartesian(pts[i]);
    if (!c) continue;
    const lon = Cesium.Math.toDegrees(c.longitude);
    const lat = Cesium.Math.toDegrees(c.latitude);
    if (lon < w) w = lon;
    if (lon > e) e = lon;
    if (lat < so) so = lat;
    if (lat > n) n = lat;
  }
  // 点集算完再统一外扩，四面留一样的边距；洞最小就是 FOCUS_MASK_PAD_DEG。
  w -= FOCUS_MASK_PAD_DEG;
  e += FOCUS_MASK_PAD_DEG;
  so -= FOCUS_MASK_PAD_DEG;
  n += FOCUS_MASK_PAD_DEG;
  const outer = Cesium.Cartesian3.fromDegreesArray([-180, -90, 180, -90, 180, 90, -180, 90]);
  const inner = Cesium.Cartesian3.fromDegreesArray([w, so, e, so, e, n, w, n]);
  const hierarchy = new Cesium.PolygonHierarchy(outer, [new Cesium.PolygonHierarchy(inner)]);
  const hole = state.focusMaskHole;
  // 洞没动就不重建 hierarchy：地面分类图元每帧重建会导致整帧纹丝不动（实测）。
  const moved = !hole || Math.abs(hole.w - w) > 1e-7 || Math.abs(hole.e - e) > 1e-7 ||
    Math.abs(hole.s - so) > 1e-7 || Math.abs(hole.n - n) > 1e-7;
  if (!state.focusMaskEntity) {
    state.focusMaskEntity = viewer.entities.add({
      id: FOCUS_MASK_ID,
      name: '选中掩码',
      show: false,
      polygon: {
        hierarchy: hierarchy,
        material: Cesium.Color.fromCssColorString('#0b1a2a').withAlpha(0.45),
        // 地面分类：普通 height=0 多边形与球面同深度，实测整帧亮度和不开掩码一样（没渲染出来）。
        classificationType: Cesium.ClassificationType.BOTH,
        outline: false,
      },
    });
  } else if (moved) {
    state.focusMaskEntity.polygon.hierarchy = hierarchy;
  }
  setEntityShow(state.focusMaskEntity, true);
  state.focusMaskHole = { w: w, e: e, s: so, n: n };
}
/**
 * 轨迹窗口 / 目标轨迹标绘 / 掩码的取证读数：外部（自检、验收）只认这些数字 ——
 * 少了这个出口，"画得对不对"就只能靠肉眼，窗口与掩码是否生效也无从断言。
 */
function focusDebug() {
  const id = state.selected;
  const rec = id ? state.targets[id] : null;
  let maxV = 0;
  let sum = 0;
  Object.keys(state.targets).forEach(function (tid) {
    const n = trailWindowOf(state.targets[tid], tid).length;
    if (n > maxV) maxV = n;
    sum += n;
  });
  const mat = rec ? rec.entity.polyline.material : null;
  // 显隐可能是 Property 包装（setEntityShow 写的是布尔），两种形态都要读对。
  const predShown = !!(rec && rec.predict &&
    (rec.predict.show && rec.predict.show.getValue ? rec.predict.show.getValue() : rec.predict.show));
  const pred = predShown ? predictedTrackPoints(rec.def) : [];
  return {
    focusMask: !!state.layers.focusMask,
    selected: id || null,
    tick: state.tick,
    // 轨迹窗口只影响掩码与取证读数；绘制给用户的历史线是**全程实线**。
    trailWindowTicks: TRAIL_WINDOW_TICKS,
    trailFull: rec ? rec.trail.length : 0,
    trailDrawn: rec ? trailWindowOf(rec, id).length : 0,
    maxTrailVertices: maxV,
    trailVertexSum: sum,
    historySolid: !!mat && !(mat instanceof Cesium.PolylineDashMaterialProperty),
    // 计划轨迹：虚线 + 终点箭头；与历史实线同属「历史轨迹」图层。
    predictPoints: pred.length,
    predictDashed: !!(rec && rec.predict && rec.predict.polyline &&
      rec.predict.polyline.material instanceof Cesium.PolylineDashMaterialProperty),
    predictArrow: !!(rec && rec.predict && rec.predict.billboard),
    maskExists: !!state.focusMaskEntity,
    maskShow: !!(state.focusMaskEntity && state.focusMaskEntity.show),
    maskHole: state.focusMaskHole,
  };
}

function occupationFrame() {
  const frames = (state.data && state.data.frames) || [];
  if (!frames.length) return null;
  return frames[Math.max(0, Math.min(state.tick, frames.length - 1))];
}

/**
 * 占领区域：**已下线**（2026-10-20 用户要求，显示与功能一并停用）。
 *
 * 原实现按目标影响圆逐帧重建贴地面，并把阵营状态推给鸟瞰图掩码。现在整块功能
 * 不再调用；保留空函数是为了让 syncEntities、图层 handler 等旧调用点无需全部改动，
 * 恢复时把 `rebuildOccupationEntities/computeOccupationArea` 的调用放回即可。
 */
function updateOccupationAreas() {
  // ⚠ 必须 force：调用点在 syncEntities 末尾，`state.tick` 要到下一帧同步才更新；
  // 若按 key 短路，首帧会用"第 0 帧的空热点"锁死 key，之后播放/拖时间轴都不再重建
  // （实测 tick=720 时热点仍为 0）。重建只在脏帧发生一次，不是逐帧开销。
  // 占领区域已下线（原 rebuildOccupationEntities 调用保留注释）。
  rebuildDynamicAreas(true);
}

/**
 * Direction 2 evidence outlet: tick, the exact per-hotspot status map of the
 * current frame, its histogram, and the minimap mask readings.
 */
function occupationDebug() {
  // 占领区域已下线：只报"未启用"，不再读图元。
  return { tick: state.tick, enabled: false, entities: 0, visible: 0 };
}

function syncSelection() {
  Object.keys(state.targets).forEach(function (id) {
    const rec = state.targets[id];
    rec.entity.billboard.image = Symbology.symbolUri(
      rec.def.icon, rec.def.affiliation, { selected: state.selected === id });
  });
  // 鸟瞰图同步选中高亮：同一次点击在两个视图里看起来必须是同一个目标。
  if (window.Overview && Overview.setSelected) Overview.setSelected(state.selected);
  // 选中/取消选中会改变掩码的洞：立即重画一次，不等下一帧。
  updateFocusMask();
}

// --------------------------------------------------------------------------
// 三级TS推理的界面呈现
// --------------------------------------------------------------------------

function inferenceAt(tick) {
  if (!state.inference) return null;
  return state.inference[Math.max(0, Math.min(tick, state.inference.length - 1))];
}

/**
 * 当前帧出现最多的主意图（全局）。横幅用它回答"ZC上在打什么"。
 *
 * 只统计**主**意图：次要意图（本体标 secondary-only 的 FIRE_SUPPORT /
 * TARGET_GUIDANCE）计入会把"火力支援"顶成主要活动，掩盖真实主任务。
 */
function regionIntentCounts(inf) {
  const counts = {};
  Object.keys(inf.targets).forEach(function (tid) {
    const e = inf.targets[tid];
    if (!e.primary_intent) return;
    if (window.IntentUi && IntentUi.isSecondaryOnly(e.primary_intent)) return;
    counts[e.primary_intent] = (counts[e.primary_intent] || 0) + 1;
  });
  return Object.keys(counts)
    .map(function (c) {
      return { code: c, zh: window.IntentUi ? IntentUi.nameZh(c) : c, n: counts[c] };
    })
    .sort(function (a, b) { return b.n - a.n; })
    .slice(0, 3);
}

function updateInferenceUi() {
  const inf = inferenceAt(state.tick);
  // 鸟瞰图与整体TS同一节奏：目标位置、战损状态、血条都在这里推一次。
  // 播放推进、拖动时间轴、切案例、换选中都会经过本函数，不需要另挂定时器。
  if (window.Overview && state.data && Overview.updateSituation) {
    Overview.updateSituation(state.data, state.tick);
  }
  if (!inf) {
    if (dom.rlValue) dom.rlValue.textContent = '—';
    if (dom.rlNote) dom.rlNote.textContent = '未加载推理结果';
    // 没有推理帧时整体TS栏仍然渲染，只是战损估计停在"未受高告警"的正常档。
    if (window.StatusPanels && dom.overallBody && state.data) {
      StatusPanels.renderOverall(dom.overallBody, state.data, null, state.tick);
    }
    return;
  }

  const lv = inf.overall.level;
  // 区域TS只写在**右栏**。地图上那份「区域级TS」横幅已移除（内容与之重复，
  // 且占着地图），所以这里不再有"两份显示、需要同步"的问题——原先 rl* 是从
  // level* 复制过来的，现在直接算一份。
  const noteText = inf.overall.note +
    ' 平均威胁 ' + inf.overall.mean_threat + '，最高 ' + inf.overall.max_threat +
    '，对抗域 ' + inf.overall.engaged_pairs + ' 个。';
  if (dom.rlValue) {
    dom.rlValue.textContent = lv + ' 级 · ' + inf.overall.level_label;
    dom.rlValue.className = 'rl-value ' + (LEVEL_CLASS[lv] || '');
  }
  if (dom.rlNote) dom.rlNote.textContent = noteText;

  // 整体TS栏：红蓝双方各类型目标的整体战损状态（图标 + 紧贴底部的血条）。
  // 只更新既有 DOM 的档位/血量，结构在案例切换时才重建——见 statusPanels.js。
  if (window.StatusPanels && dom.overallBody && state.data) {
    StatusPanels.renderOverall(dom.overallBody, state.data, inf, state.tick);
  }

  // 主导意图：等级回答"多紧张"，意图回答"在打什么"，两者并列才算把区域TS说全。
  if (dom.rlIntent) {
    const top = regionIntentCounts(inf);
    dom.rlIntent.textContent = top.length
      ? top.map(function (x) { return x.zh + ' ×' + x.n; }).join('、')
      : '无带意图判定的目标';
  }

  // 作战阶段：登陆作战是分阶段的，同一个意图在不同阶段含义不同，
  // 所以阶段与等级、意图并列显示，而不是让人自己去推时间轴。
  if (dom.rlPhase) {
    const fr = state.data && state.data.frames && state.data.frames[state.tick];
    // Gate on the dataset *declaring* phases. The engine attaches a phase to every
    // frame for the Taiwan landing operation, and the India-China case legitimately
    // declares none (a standing confrontation has no landing sequence) - but its
    // frames still carry the field, so reading `fr.phase` alone displayed
    // "集结装载 1/6" on the Ladakh case: a Taiwan landing stage on the wrong
    // continent.
    const declared = (state.data && state.data.phases) || [];
    const ph = (declared.length && fr) ? fr.phase : null;
    // Per-unit phase comes precomputed from the engine (`frame.unit_phase`). A unit
    // may lag the operation (still crossing while the first wave is ashore) or have
    // finished (a transport holds at 航渡). That lag is the point of showing it, so
    // it sits beside the operation's stage rather than replacing it. The cap rule
    // deliberately is NOT re-derived here: it depends on the engine's
    // `PHASE_TARGETS` table, and a second copy would drift.
    const up = (ph && state.selected && fr.unit_phase)
      ? fr.unit_phase[state.selected] : null;
    const lags = up && up.id && up.id !== ph.id;
    dom.rlPhase.textContent = ph
      ? (ph.name + ' ' + (ph.index + 1) + '/' + (ph.count || declared.length)
         + (lags ? ' · 该目标：' + up.name : ''))
      : '';
    dom.rlPhase.title = ph
      ? ('本阶段已持续 ' + ph.ticks_in + ' 帧（自第 ' + ph.since_tick + ' 帧）'
         + (lags ? '；所选目标仍处于「' + up.name + '」，落后于作战阶段' : ''))
      : '该战区未定义阶段';
  }

  // 热点区域已改为**动态生成**，不再对应数据集里的预设 regions 索引；
  // 旧的热点表格也已移除。这里只在行对象真的带 levelCell 时刷新（兼容旧形态），
  // 否则跳过——曾经因为动态行没有 levelCell 在这里抛异常，导致整个选中流程失败。
  (state.hotspotRows || []).forEach(function (row, i) {
    if (!row || !row.levelCell) return;
    const r = inf.regions && inf.regions[i];
    if (!r) return;
    row.levelCell.textContent = r[0] + ' ' + levelLabel(r[0]);
    row.levelCell.className = 'lv-cell ' + (LEVEL_CLASS[r[0]] || '');
    row.levelCell.title = '红方 ' + r[1] + ' 个目标 / 蓝方 ' + r[2] + ' 个目标';
  });
  renderHotspotIntents();

  if (state.geoPinned && state.activeHotspot >= 0) {
    const r = inf.regions[state.activeHotspot];
    if (r) {
      dom.geoLevel.textContent = r[0] + ' ' + levelLabel(r[0]);
      dom.geoLevel.className = 'lv-cell ' + (LEVEL_CLASS[r[0]] || '');
    }
  }

  if (state.selected) renderTargetStatus(state.selected);
}

function levelLabel(code) {
  const model = state.data && state.data.inference_model;
  if (model && model.level_codes) return model.level_codes[code] || '';
  return '';
}

function codeLabel(section, code) {
  const model = state.data && state.data.inference_model;
  if (!model) return '';
  const table = model[section];
  return table ? (table[code] || '') : '';
}

/**
 * 意图链路的取证入口。
 *
 * 「意图有没有生效」光看画面判断不了：面板可能显示的是旧字段、本体可能没加载、
 * 当前帧可能整帧都是 UNKNOWN。这里把每个短路条件逐条报出来。
 */
function debugIntent() {
  const inf = inferenceAt(state.tick);
  const per = {};
  if (inf) {
    Object.keys(inf.targets).forEach(function (tid) {
      const e = inf.targets[tid];
      per[tid] = {
        primary: e.primary_intent || null,
        status: e.intent_status || null,
        set: (e.intent_set || []).map(function (x) { return x.code; }),
        missing: e.missing_evidence || [],
        // CES 链路：集合大小、来源（model/rule）、不足度、是否沿用上一评分帧
        setSize: (typeof e.intent_set_size === 'number') ? e.intent_set_size : null,
        source: e.intent_source || null,
        insufficiency: (typeof e.insufficiency === 'number') ? e.insufficiency : null,
        carried: e.intent_carried === true,
        evalTick: (e.intent_eval_tick !== undefined) ? e.intent_eval_tick : null,
      };
    });
  }
  const summary = state.data && state.data.intent_summary;
  return {
    ontology: window.IntentUi ? IntentUi.debug() : null,
    // 数据集层面的 CES 链路状态（模型版本、嵌套违例、步长、评分帧数）
    datasetIntentSet: (state.data && state.data.intent_set) || null,
    datasetIntentSource: (state.data && state.data.intent_source) || null,
    intentModelVersion: (state.data && state.data.intent_model_version) || null,
    modelHasOntology: !!(state.data && state.data.inference_model
      && state.data.inference_model.intent_ontology),
    hasData: !!state.data,
    tick: state.tick,
    hasInferenceFrame: !!inf,
    targetsThisFrame: per,
    datasetRuleHits: summary ? summary.rule_hits : null,
    datasetStatusCounts: summary ? summary.status_counts : null,
    underivableCount: summary && summary.underivable ? summary.underivable.length : null,
    panelHasIntentBlock: !!(dom.targetBody && dom.targetBody.querySelector
      && dom.targetBody.querySelector('.tp-intent')),
  };
}

// --------------------------------------------------------------------------
// 目标计数
// --------------------------------------------------------------------------
//
// 左栏的「目标列表」已按界面重构移除，但「目标图层」里的红/蓝/海/陆/空计数仍然
// 有用，所以保留计数更新、不再渲染列表。
//
// 不能只是把原函数留着不调：它会写 `dom.targetList`，而该元素已从 DOM 移除，
// 一旦被调用就会在 null 上取 innerHTML 直接抛异常。因此改为只更新计数。

function renderTargetCounts() {
  const counts = { red: 0, blue: 0, sea: 0, land: 0, air: 0 };
  const targets = (state.data && state.data.targets) || [];
  targets.forEach(function (def) {
    // 计数按**编成**统计，不随地图可见性过滤变化——过滤是"看得见什么"，
    // 而这里回答的是"编成里有多少"。
    counts[def.affiliation]++;
    counts[def.domain]++;
  });
  if (dom.countRed) dom.countRed.textContent = counts.red;
  if (dom.countBlue) dom.countBlue.textContent = counts.blue;
  if (dom.countSea) dom.countSea.textContent = counts.sea;
  if (dom.countLand) dom.countLand.textContent = counts.land;
  if (dom.countAir) dom.countAir.textContent = counts.air;
}

// --------------------------------------------------------------------------
// 热点区域表格 —— 已移除
// --------------------------------------------------------------------------
//
// 「区域TS」栏按你的要求只显示**整个TS案例范围**的整体TS，不再逐热点罗列。
// 表格 DOM（#hotspotBody）已从 index.html 删除，但相机联动仍然需要"点击热点区域"的
// 能力，所以 `createHotspots()` 与 `selectHotspot()` 都保留；这里只把**建表**这一步
// 停掉：不再构造 11~18 行 DOM，也不再把行挂进 state.hotspotRows。
//
// 保留函数本身（而不是删掉调用点）是为了让"表格曾经在这里、且被有意去掉"这件事留在
// 代码里；否则以后有人看到 selectHotspot 里的 active 行逻辑会一头雾水。

function renderHotspotTable() {
  // 热点区域表格已随界面重构移除（#hotspotBody 不存在）；热点已改为
  // **按战况动态生成**，表格的数据源（数据集预设矩形）也不再成立。
  // 保留函数与调用点，避免改动面扩大；实际只清空行缓存。
  if (!document.getElementById('hotspotBody')) {
    state.hotspotRows = [];
    return;
  }
}

/**
 * 刷新热点表里的意图列。
 *
 * 区域级主导意图 = 该热点框内所有目标当前主意图的众数。逐帧调用，
 * 与 `updateInferenceUi()` 同一时机（时间轴/播放推进）。
 */
function renderHotspotIntents() {
  const inf = inferenceAt(state.tick);
  if (!inf || !state.hotspotRows.length) return;
  state.hotspotRows.forEach(function (rec, i) {
    if (!rec.intentCell) return;
    const region = state.data.hotspots[i];
    const counts = {};
    Object.keys(inf.targets).forEach(function (tid) {
      const e = inf.targets[tid];
      if (!e.primary_intent) return;
      const def = state.targets[tid] && state.targets[tid].def;
      if (!def || !region) return;
      const s = positionFor(tid, state.tick);
      if (!s) return;
      if (s.lat < region.bbox[1] || s.lat > region.bbox[3] ||
          s.lon < region.bbox[0] || s.lon > region.bbox[2]) return;
      counts[e.primary_intent] = (counts[e.primary_intent] || 0) + 1;
    });
    const keys = Object.keys(counts).sort(function (a, b) { return counts[b] - counts[a]; });
    if (!keys.length) {
      rec.intentCell.textContent = '—';
      rec.intentCell.title = '该区域内当前没有带意图判定的目标';
      return;
    }
    const top = keys.slice(0, 2).map(function (k) {
      return window.IntentUi ? IntentUi.nameZh(k) : k;
    });
    rec.intentCell.textContent = top.join(' / ');
    rec.intentCell.title = keys.map(function (k) {
      return (window.IntentUi ? IntentUi.nameZh(k) : k) + ' ×' + counts[k];
    }).join('、');
  });
}

/**
 * 选中一个热点区域：地图飞行到该范围，并联动更新右侧"地理位置信息"。
 *
 * 这是需求里"点击某一热点区域时需联动更新地理信息位置"的实现点：面板被
 * 锁定到热点范围（中心经纬、经纬跨度、区域TS等级），直到用户点击
 * "位置信息"按钮回到光标跟随模式。
 */
function selectHotspot(index, fly) {
  // 热点现在是**动态生成**的：先取当前帧的热点行，找不到再退回数据集预设
  // （旧数据集/旧链接仍可能带 index）。
  const row = (state.hotspotRows || [])[index];
  if (row && row.center) {
    detachFollow('热点区域跳转');
    state.activeHotspot = index;
    state.geoPinned = true;
    if (dom.geoRegion) dom.geoRegion.textContent = row.name;
    if (fly) flyTo(row.center[1], row.center[0], (row.radiusKm || 40) * 3000);
    state.hotspotRows.forEach(function (rec, i) {
      if (rec.row) rec.row.classList.toggle('active', i === index);
    });
    return;
  }
  const hs = state.data && state.data.hotspots && state.data.hotspots[index];
  if (!hs) return;
  detachFollow('热点区域跳转');

  state.activeHotspot = index;
  state.geoPinned = true;

  dom.geoRegion.textContent = hs.name;
  dom.geoLon.textContent = fmtDeg(hs.center[1], 'E', 'W') +
    '（' + hs.bbox[0].toFixed(6) + ' – ' + hs.bbox[2].toFixed(6) + '）';
  dom.geoLat.textContent = fmtDeg(hs.center[0], 'N', 'S') +
    '（' + hs.bbox[1].toFixed(6) + ' – ' + hs.bbox[3].toFixed(6) + '）';
  dom.geoSpan.textContent = hs.span_deg[0].toFixed(6) + '° × ' + hs.span_deg[1].toFixed(6) + '°';

  const inf = inferenceAt(state.tick);
  if (inf && inf.regions[index]) {
    const r = inf.regions[index];
    dom.geoLevel.textContent = r[0] + ' ' + levelLabel(r[0]);
    dom.geoLevel.className = 'lv-cell ' + (LEVEL_CLASS[r[0]] || '');
  }

  state.hotspotRows.forEach(function (rec, i) {
    rec.row.classList.toggle('active', i === index);
  });

  const w = hs.bbox[0], s = hs.bbox[1], e = hs.bbox[2], n = hs.bbox[3];
  if (fly) {
    viewer.camera.flyTo({
      destination: Cesium.Rectangle.fromDegrees(w, s, e, n),
      duration: 1.2,
    });
  }

  // 建筑体块跟随当前热点：若该热点附近有预下载的高清 WSF3D 城市区，
  // 就切换到那个 AOI，否则提示用户。
  if (checkboxOn('modelBuildings')) refreshBuildings();
}

/**
 * 最具体的热点区域：多个热点会互相包含（台湾海峡套着平潭、澎湖、台北、高雄），
 * 取面积最小的那个，读数才不会总是被最大的框吃掉。
 */
function hotspotAt(lat, lon) {
  if (!state.data) return -1;
  let best = -1;
  let bestArea = Infinity;
  state.data.hotspots.forEach(function (hs, i) {
    const b = hs.bbox;
    if (lat < b[1] || lat > b[3] || lon < b[0] || lon > b[2]) return;
    const area = (b[2] - b[0]) * (b[3] - b[1]);
    if (area < bestArea) { bestArea = area; best = i; }
  });
  return best;
}

/**
 * 刷新"地理位置信息"。光标跟随模式下由光标位置驱动；没有光标读数时（例如
 * 刚加载、或纯相机飞行）退回到视图中心的地面交点，避免面板一片"—"。
 */
function updateGeoReadout() {
  // 空态下没有热点可判，但相机经纬与视高照常刷新——否则一打开页面面板全是"—"。
  if (state.geoPinned || !viewer) return;

  // 「视高」一行已移除：相机高度由地图组件「视点坐标」提供，这里只管光标落点。

  let carto = state.cursorCarto;
  if (!carto) {
    const canvas = viewer.scene.canvas;
    const ray = viewer.camera.getPickRay(
      new Cesium.Cartesian2(canvas.clientWidth / 2, canvas.clientHeight / 2));
    const point = ray ? viewer.scene.globe.pick(ray, viewer.scene) : null;
    if (!point) return;
    carto = Cesium.Cartographic.fromCartesian(point);
  }

  const lon = Cesium.Math.toDegrees(carto.longitude);
  const lat = Cesium.Math.toDegrees(carto.latitude);
  dom.geoLon.textContent = fmtDeg(lon, 'E', 'W');
  dom.geoLat.textContent = fmtDeg(lat, 'N', 'S');

  const idx = state.data ? hotspotAt(lat, lon) : -1;
  if (idx >= 0) {
    const hs = state.data.hotspots[idx];
    const b = hs.bbox;
    dom.geoRegion.textContent = hs.name;
    dom.geoSpan.textContent = (b[2] - b[0]).toFixed(6) + '° × ' + (b[3] - b[1]).toFixed(6) + '°';
    const inf = inferenceAt(state.tick);
    if (inf && inf.regions[idx]) {
      const r = inf.regions[idx];
      dom.geoLevel.textContent = r[0] + ' ' + levelLabel(r[0]);
      dom.geoLevel.className = 'lv-cell ' + (LEVEL_CLASS[r[0]] || '');
    }
  } else {
    dom.geoRegion.textContent = state.data ? '不在热点区域内' : (state.baseRegion || '未加载TS');
    dom.geoSpan.textContent = '—';
    dom.geoLevel.textContent = '—';
    dom.geoLevel.className = 'lv-cell';
  }
}

function clearHotspotSelection() {
  state.activeHotspot = -1;
  state.geoPinned = false;
  state.hotspotRows.forEach(function (rec) { rec.row.classList.remove('active'); });
  dom.geoRegion.textContent = '光标跟随';
  dom.geoSpan.textContent = '—';
  dom.geoLevel.textContent = '—';
  dom.geoLevel.className = 'lv-cell';
}

// --------------------------------------------------------------------------
// 目标TS栏（右侧栏中段）
// --------------------------------------------------------------------------
//
// 原先是中部地图右上角的浮动面板 `#targetPanel`。该位置与新增的鸟瞰图重叠，且
// 右栏已专设「目标TS栏」，因此浮动面板整体移除，内容迁到这里——目标信息只在一处。


// ---------------------------------------------------------------- relations
// Step C: relation lines for the selected unit.
//   * friendly link : both ends share attributes.formation_id -> SOLID line
//   * hostile  link : weapon-coverage overlap -> DASHED line. Default criterion is
//                     engage_km (sum of the two engage radii >= distance). If a case
//                     yields no engage-based hostile pair at all we fall back to
//                     influence_km and report criterion = 'influence'.
// Mid-line text: Cesium label, font size in px, i.e. already screen spaced. That is
// the same guarantee plotting.refreshScreenSpaceDecorations rebuilds for its own
// `array` family (it only walks items registered inside plotting.js), so no extra
// rebuild hook is needed here.
var REL_LAYER_ID = 'situation.relations';
var relView = { ents: [], own: 0, hostile: 0, hostileInf: 0, peers: [], criterion: 'engage', show: true,
               caseKey: '', caseEngage: false };

function relAttr(def, key) {
  var a = (def && def.attributes) || {};
  var v = a[key];
  return (v === undefined || v === null) ? '' : String(v);
}

function relNum(def, key) {
  var v = def ? Number(def[key]) : NaN;
  return isFinite(v) ? v : 0;
}

function relKm(a, b) {
  var R = 6371.0088, rad = Math.PI / 180;
  var dla = (b.lat - a.lat) * rad, dlo = (b.lon - a.lon) * rad;
  var h = Math.sin(dla / 2) * Math.sin(dla / 2)
        + Math.cos(a.lat * rad) * Math.cos(b.lat * rad) * Math.sin(dlo / 2) * Math.sin(dlo / 2);
  return 2 * R * Math.asin(Math.min(1, Math.sqrt(h)));
}

/**
 * 关系线配色（方案见汇报）：
 *   own    本方（同编队/同阵营）→ 绿色 #2F9E44：直观"协同/友军"，且不与红方
 *          阵营色冲突（自动选择目标时本方也是红方时，用绿色才不会读成"敌对"）。
 *   hostile 敌方 → 红色 #C0392B：与阵营色一致，最符合"威胁"直觉。
 *   inferred 仅有推理依据的关系（见 relAddLine 的 dashed 参数）用浅一档的色，
 *          配合虚线，双重编码"证据性质"。
 */
function relCss(kind, inferred) {
  if (kind === 'own') return inferred ? '#74B98A' : '#2F9E44';
  return inferred ? '#E0705F' : '#C0392B';
}

function relColor(kind, inferred) {
  return Cesium.Color.fromCssColorString(relCss(kind, inferred));
}

function relClear() {
  relView.ents.forEach(function (e) {
    try { viewer.entities.remove(e); } catch (err) { /* already gone */ }
  });
  relView.ents.length = 0;
  relView.own = 0; relView.hostile = 0; relView.hostileInf = 0; relView.peers.length = 0;
}

// 关系标签的可读性上限：同阵营目标可能很多（20 个红方），每对都写字必然糊成一片。
// 只在**较近**的关系上写标签，远的只画线（线本身仍表达"有关联"）。
var REL_LABEL_MAX_KM = 260;

function relAddLine(from, to, dashed, color, text) {
  // ⚠ 两端**各自的真实高度**，不能用平均高度：飞机 8500 m 连水面舰艇 0 m 时，
  // 平均高度画出来的线两端各悬空 4 km，视觉上完全不贴目标（实测截图）。
  var altFrom = Math.max(0, Number(from.alt) || 0);
  var altTo = Math.max(0, Number(to.alt) || 0);
  var alt = (altFrom + altTo) / 2;      // 只用于标签中点
  // 过远的关联只画线不写字（见 REL_LABEL_MAX_KM）。
  if (relKm(from, to) > REL_LABEL_MAX_KM) text = '';
  // **没有文字就不建 label 实体**：空字符串的标签在 Cesium 里仍占一个图元，
  // 也会让"每个实体都有标签"的统计变得没有意义。
  var ent = viewer.entities.add({
    polyline: {
      positions: Cesium.Cartesian3.fromDegreesArrayHeights(
        [from.lon, from.lat, altFrom, to.lon, to.lat, altTo]),
      width: 2,
      material: dashed
        ? new Cesium.PolylineDashMaterialProperty({ color: color, dashLength: 16 })
        : new Cesium.ColorMaterialProperty(color),
    },
    position: text
      ? Cesium.Cartesian3.fromDegrees((from.lon + to.lon) / 2, (from.lat + to.lat) / 2, alt)
      : undefined,
    label: text ? {
      text: text, font: '12px "Microsoft YaHei", "Noto Sans CJK SC", sans-serif',
      fillColor: color, style: Cesium.LabelStyle.FILL,
      showBackground: true, backgroundColor: Cesium.Color.fromCssColorString('#ffffffd9'),
      backgroundPadding: new Cesium.Cartesian2(4, 2),
      horizontalOrigin: Cesium.HorizontalOrigin.CENTER,
      verticalOrigin: Cesium.VerticalOrigin.CENTER,
      disableDepthTestDistance: Number.POSITIVE_INFINITY,
      // 拉远后标签自然淡出，避免战区级视图上标签挤成一片（线仍然可见）。
      translucencyByDistance: new Cesium.NearFarScalar(300000, 1.0, 900000, 0.0),
      scaleByDistance: new Cesium.NearFarScalar(150000, 1.0, 700000, 0.75),
    } : undefined,
  });
  ent.show = !!relView.show;
  relView.ents.push(ent);
}


// Case-level criterion gate: the engage criterion is the default, but a case where
// NO opposing pair at all overlaps under engage would draw nothing, so we fall back
// to influence for the whole case (stable across selections; the debug output says
// which one is in force). Measured on the first frame so the answer does not drift
// with playback.
function relCaseEngageOk() {
  var key = String((state.data && state.data.case_id) || state.caseId || '');
  if (relView.caseKey !== key) {
    var ids = Object.keys(state.targets), hit = false;
    for (var i = 0; i < ids.length && !hit; i++) {
      var da = state.targets[ids[i]].def, pa = positionFor(ids[i], 0);
      if (!da || !pa) continue;
      for (var j = i + 1; j < ids.length; j++) {
        var db = state.targets[ids[j]].def, pb = positionFor(ids[j], 0);
        if (!db || !pb || da.affiliation === db.affiliation) continue;
        if (relKm(pa, pb) <= relNum(da, 'engage_km') + relNum(db, 'engage_km')) { hit = true; break; }
      }
    }
    relView.caseKey = key;
    relView.caseEngage = hit;
  }
  return relView.caseEngage;
}

var PAIR_ZH = { lead: '长机', wingman: '僚机' };

/**
 * 关系文案。
 *
 * 友方：编队内写 `协同 · 长机→僚机`；同阵营其余目标**不写字**（绿线已表达本方）。
 * 敌方：不再只写"敌对 XX km"，而是把**系统实际具备的**证据说清楚：
 *   · 进入交战射程（双方 engage 半径之和）→ `交战威胁 · 距离 X km`
 *   · 仅进入影响/探测范围 → `探测跟踪 · 距离 X km`
 * 证据性质（实测 / 推理）**不写进文字**：实线=实测、虚线=推理，线型已经表达。
 * 队伍代号（air-red-1 等）只用于内部，不再直接展示。
 */
function relText(kind, evidence, fromDef, toDef, km) {
  if (kind === 'own') {
    var fr = relAttr(fromDef, 'pair_role'), tr = relAttr(toDef, 'pair_role');
    if (fr && tr) {
      // 只有**编队内**的协同关系值得写出来（长机/僚机是数据里真实存在的字段）。
      return '协同 · ' + (PAIR_ZH[fr] || fr) + '→' + (PAIR_ZH[tr] || tr);
    }
    // 同阵营但没有编队字段：绿线已经表达了"本方"，不再写字。
    return '';
  }
  var d = Math.round(km);
  var verb = evidence === 'measured' ? '交战威胁' : '探测跟踪';
  return verb + ' · 距离 ' + d + ' km';
}

var _relKey = null;

/**
 * 关系线只在**相关内容变化**时重建：选中目标、帧号、过滤开关。
 *
 * 原先它挂在每次 syncEntities 后面，而 syncEntities 每个脏帧都会跑（播放时
 * 每帧一次）。整组线被清掉重建，标签也随之销毁重建——正好和标签避让的隐藏
 * 状态打架，表现为标牌闪烁、关系文字一闪一闪。
 */
function relationsKey() {
  return [state.selected || '', state.tick,
          state.filters.red ? 1 : 0, state.filters.blue ? 1 : 0,
          state.filters.sea ? 1 : 0, state.filters.land ? 1 : 0,
          state.filters.air ? 1 : 0].join('|');
}

function refreshRelations(force) {
  if (!window.Cesium || !viewer || !state || !state.targets) return;
  var key = relationsKey();
  if (!force && key === _relKey) return;
  _relKey = key;
  relClear();
  var sid = state.selected;
  if (!sid || !state.targets[sid]) return;
  var sdef = state.targets[sid].def;
  var spos = positionFor(sid, state.tick);
  if (!sdef || !spos) return;
  var fid = relAttr(sdef, 'formation_id');
  // Dense cases (80 targets) turn the relation layer into a hairball: one
  // selected target can produce ~79 lines and 34 distance labels, rebuilt
  // every tick. Keep only the closest hostile relations in that case; old
  // cases are untouched.
  var dense = !!(state.data && (state.data.campaign_contract_version ||
    ((state.data.targets || []).length > 50)));
  var eng = [], inf = [];
  Object.keys(state.targets).forEach(function (id) {
    if (id === sid) return;
    var def = state.targets[id] && state.targets[id].def;
    var pos = def ? positionFor(id, state.tick) : null;
    if (!pos) return;
    // 友方：同编队（formation_id 一致）或同阵营。
    // **同编队关系是数据集明确声明的 → 实测、实线、绿色。**
    if (fid && relAttr(def, 'formation_id') === fid) {
      if (!dense) {
        relAddLine(spos, pos, false, relColor('own', false),
                   relText('own', 'measured', sdef, def, relKm(spos, pos)));
        relView.own++;
      }
      relView.peers.push(id);
      return;
    }
    if (def.affiliation === sdef.affiliation) {
      if (!dense) {
        relAddLine(spos, pos, false, relColor('own', false),
                   relText('own', 'measured', sdef, def, relKm(spos, pos)));
        relView.own++;
      }
      return;
    }
    var km = relKm(spos, pos);
    if (km <= relNum(sdef, 'engage_km') + relNum(def, 'engage_km')) eng.push({ id: id, km: km });
    if (km <= relNum(sdef, 'influence_km') + relNum(def, 'influence_km')) inf.push({ id: id, km: km });
  });
  var useInf = (!relCaseEngageOk() && inf.length > 0);
  relView.criterion = useInf ? 'influence' : 'engage';
  relView.hostileInf = inf.length;
  var drawn = useInf ? inf : eng;
  if (dense) {
    // Closest hostile contacts are the ones worth drawing; the rest are
    // available in the target panel / inference data but not as screen lines.
    drawn = drawn.slice().sort(function (a, b) { return a.km - b.km; }).slice(0, 6);
  }
  relView.dense = dense;
  // 敌方：距离阈值推出来的关系是**推理**（数据集没有声明交火），虚线 + 浅一档的颜色。
  drawn.forEach(function (o) {
    var def = state.targets[o.id].def;
    relAddLine(spos, positionFor(o.id, state.tick), true, relColor('hostile', true),
               relText('hostile', 'inferred', sdef, def, o.km));
  });
  relView.hostile = drawn.length;
}

function setRelationsVisible(on) {
  relView.show = !!on;
  relView.ents.forEach(function (e) { e.show = !!on; });
  if (on) refreshRelations(true);
}

// Activity state: the code(s) the UI is actually showing right now, taken from
// inference_frames (parent-class semantics per the plan). Selected unit -> its own
// codes at this tick; nothing selected -> de-duplicated codes of the visible units.
// `activityVocabulary` reports the six parent classes the ontology offers, so a test
// can tell "which values are on screen" from "where the values come from".
function relActivityCode(code) {
  if (!code) return '';
  var o = (window.IntentUi && IntentUi._ontology) ? IntentUi._ontology() : null;
  if (!o) return code;
  var intents = o.intents || [];
  for (var i = 0; i < intents.length; i++) {
    if (intents[i].code === code) return intents[i].parent_code || code;
  }
  return code;   // already a parent class, or a status layer code
}

function activityCodes() {
  var seen = {}, keys = [];
  function add(code) {
    var c = relActivityCode(code);
    if (c && !seen[c]) { seen[c] = 1; keys.push(c); }
  }
  var inf = inferenceAt(state.tick);
  if (!inf || !inf.targets) return keys;
  if (state.selected && inf.targets[state.selected]) {
    var own = inf.targets[state.selected];
    (own.intent_set || []).forEach(function (x) { add(x && x.code); });
    add(own.primary_intent);
    return keys;
  }
  Object.keys(inf.targets).forEach(function (tid) {
    var rec = state.targets[tid];
    if (!rec || !isVisible(rec.def)) return;
    var e = inf.targets[tid];
    (e.intent_set || []).forEach(function (x) { add(x && x.code); });
    add(e.primary_intent);
  });
  return keys;
}

function activityVocabulary() {
  var out = [];
  try {
    var o = (window.IntentUi && IntentUi._ontology) ? IntentUi._ontology() : null;
    var ps = (o && o.parents) || [];
    ps.forEach(function (p) { if (p && p.code) out.push(p.code); });
  } catch (err) { /* ontology not loaded yet */ }
  return out;
}

function activityName(code) {
  if (!code) return '';
  if (!window.IntentUi) return code;
  var st = IntentUi.statusInfo ? IntentUi.statusInfo(code) : null;
  if (st && st.zh) return st.zh;
  var par = IntentUi.parentZh ? IntentUi.parentZh(code) : '';
  if (par) return par;
  var o = IntentUi._ontology ? IntentUi._ontology() : null;
  var ps = (o && o.parents) || [];
  for (var i = 0; i < ps.length; i++) {
    if (ps[i] && ps[i].code === code) return ps[i].name_zh || code;
  }
  return IntentUi.nameZh ? IntentUi.nameZh(code) : code;
}

// "text really drawn": every visible relation line must carry exactly one mid-line
// label. Counted off the entity list rather than eyeballed.
function lineLabelStats() {
  // 只有**近距**关系才带文字（见 REL_LABEL_MAX_KM），所以"每线必有字"不再成立：
  // 改为"每线要么有文字、要么是一段无文字的连线"，并单独报出带字数量。
  var lines = 0, labeled = 0;
  relView.ents.forEach(function (e) {
    if (!e.show) return;
    lines++;
    var t = e.label && e.label.text;
    try { t = (t && t.getValue) ? t.getValue() : t; } catch (err) { t = null; }
    if (t) labeled++;
  });
  return { ok: labeled <= lines, lines: lines, labeled: labeled,
           labelMaxKm: REL_LABEL_MAX_KM };
}

function relationsDebug() {
  var ls = lineLabelStats();
  return {
    selected: state.selected || null,
    ownLines: relView.own,
    hostileLines: relView.hostile,
    hostileLinesInfluence: relView.hostileInf,
    total: relView.own + relView.hostile,
    formationMembers: relView.peers.slice(),
    activityCodes: activityCodes(),
    activityVocabulary: activityVocabulary(),
    linesVisible: ls.lines,
    lineLabelOk: ls.ok,
    lineLabelCount: ls.labeled,
    criterion: relView.criterion,
  };
}

function registerRelationsLayer() {
  if (!window.LayerRegistry || !LayerRegistry.register) return;
  if (LayerRegistry.def && LayerRegistry.def(REL_LAYER_ID)) return;
  LayerRegistry.register({
    id: REL_LAYER_ID, kind: 'plot', z: 45, default: true,
    label: '\u76ee\u6807\u5173\u7cfb', group: 'TS\u56fe\u5c42',
    handler: function (id, want) { setRelationsVisible(want); },
  });
}

/**
 * 点击目标后：清掉当前场景里的标绘（用户手绘 + 预览），把视觉让给关系线。
 * 标绘**记录**不删（图层/要素列表还在），只是隐藏；取消选中时恢复。
 */
/**
 * Case auto-plots: data-driven system plots that appear/disappear by tick and
 * update from the current frame/inference. They use the existing Plotting/layer
 * panel, so there is no new UI surface. Manual plots remain independent.
 */
var autoPlotState = { data: null, ids: [], visible: {}, lastDynamic: -1 };

function autoPlotDest(lon, lat, bearingDeg, km) {
  var R = 6371.0;
  var brg = bearingDeg * Math.PI / 180;
  var la1 = lat * Math.PI / 180, lo1 = lon * Math.PI / 180;
  var d = km / R;
  var la2 = Math.asin(Math.sin(la1) * Math.cos(d) +
                      Math.cos(la1) * Math.sin(d) * Math.cos(brg));
  var lo2 = lo1 + Math.atan2(Math.sin(brg) * Math.sin(d) * Math.cos(la1),
                             Math.cos(d) - Math.sin(la1) * Math.sin(la2));
  return [lo2 * 180 / Math.PI, la2 * 180 / Math.PI];
}

function autoPlotDetach() {
  if (!window.Plotting) return;
  autoPlotState.ids.forEach(function (id) {
    try { Plotting.remove(id); } catch (e) { /* already gone */ }
  });
  autoPlotState = { data: null, ids: [], visible: {}, lastDynamic: -1 };
}

function applyAutoPlots(data) {
  autoPlotDetach();
  if (!window.Plotting || !data || !Array.isArray(data.auto_plots)) return;
  autoPlotState.data = data.auto_plots;
  data.auto_plots.forEach(function (spec) {
    try {
      Plotting.add({
        id: spec.id, type: spec.type, layer: spec.layer,
        points: spec.points, label: spec.label,
        hostility: spec.hostility, system: true,
      });
      autoPlotState.ids.push(spec.id);
    } catch (e) {
      console.warn('[autoPlot] add failed', spec && spec.id, e);
    }
  });
  updateAutoPlots(state.tick || 0);
}

function updateAutoPlots(tick) {
  if (!autoPlotState.data || !window.Plotting) return;
  var t = Math.max(0, Math.round(tick || 0));
  autoPlotState.data.forEach(function (spec) {
    var a = spec.from_tick == null ? 0 : Number(spec.from_tick);
    var b = spec.to_tick == null ? 1e9 : Number(spec.to_tick);
    var vis = t >= a && t < b;
    if (autoPlotState.visible[spec.id] !== vis) {
      try { Plotting.setVisible(spec.id, vis); } catch (e) { /* gone */ }
      autoPlotState.visible[spec.id] = vis;
    }
    if (!vis || !spec.dynamic) return;
    if (t === autoPlotState.lastDynamic) return;
    try {
      if (spec.dynamic === 'aew_observation') {
        var p = positionFor(spec.actor, t);
        if (!p) return;
        var brg = Number(p.heading) || 0;
        var half = Number(spec.half_angle_deg) || 30;
        var range = Number(spec.range_km) || 80;
        Plotting.update(spec.id, { points: [
          [p.lon, p.lat],
          autoPlotDest(p.lon, p.lat, brg, range),
          autoPlotDest(p.lon, p.lat, brg - half, range),
        ] });
      } else if (spec.dynamic === 'closest_groups') {
        var inf = inferenceAt(t);
        if (!inf || !inf.groups) return;
        var red = null, blue = null;
        inf.groups.forEach(function (g) {
          if (g.domain !== 'land') return;
          if (g.affiliation === 'red') red = g;
          if (g.affiliation === 'blue') blue = g;
        });
        if (!red || !blue) return;
        Plotting.update(spec.id, { points: [
          [red.centroid[1], red.centroid[0]],
          [blue.centroid[1], blue.centroid[0]],
        ] });
      } else if (spec.dynamic === 'objective_progress') {
        var fr = state.data.frames[Math.min(t, state.data.frames.length - 1)];
        var prog = fr && fr.objectives && fr.objectives[spec.objective_id];
        if (prog == null) return;
        Plotting.update(spec.id, { label: spec.label + ' ' + Math.round(prog * 100) + '%' });
      }
    } catch (e) {
      console.warn('[autoPlot] update failed', spec.id, e);
    }
  });
  autoPlotState.lastDynamic = t;
}

function setPlottingHidden(hidden) {
  if (!window.Plotting || !Plotting.debug) return;
  var items = Plotting.debug().items || [];
  for (var i = 0; i < items.length; i++) {
    var id = items[i].id;
    // Auto plots are situation overlays; selecting a target must not hide them.
    if (items[i].system) continue;
    if (hidden) {
      if (items[i].visible !== false) Plotting.setVisible(id, false);
    } else if (items[i].visible === false) {
      Plotting.setVisible(id, true);
    }
  }
}

function selectTarget(id, fly, height) {
  const prev = state.selected;
  state.selected = id;
  if (prev && prev !== id) refreshPredictEntity(prev);
  if (id) refreshPredictEntity(id);
  setPlottingHidden(!!id);
  // 选中即绑定相机：目标平滑移到视窗中心并持续跟随；勾选「目标跟随」时
  // 直接用第三人称斜后方位姿。fly=true（面板「定位」）强制重新居中，
  // height 沿用"目标上方 N 米"的定位语义。
  if (!id) {
    detachFollow('取消选中');
  } else if (state.targets[id]) {
    beginTargetView(id, { force: !!fly, height: fly ? height : null });
  }
  syncSelection();
  renderTargetCounts();
  renderTargetStatus(id);
  refreshRelations(true);
  // The banner shows the operation's phase plus the selected unit's own phase when
  // the two differ, so it has to be refreshed on selection - otherwise picking a
  // unit that lags leaves the previous unit's comparison on screen.
  updateInferenceUi();
  markScene('selectTarget');
}

/** 清除选中态；相机随之解除绑定。Esc 与「解除跟随」都会走到这里。 */
function clearTargetSelection() {
  const prev = state.selected;
  state.selected = null;
  if (prev) refreshPredictEntity(prev);
  if (state.follow.id || state.follow.mode !== 'off') detachFollow('取消选中');
  setPlottingHidden(false);
  syncSelection();
  renderTargetStatus(null);
  refreshRelations(true);
  markScene('clearTargetSelection');
}

function kvRow(label, value, extraClass) {
  return '<div class="tp-kv"><span>' + label + '</span><b' +
    (extraClass ? ' class="' + extraClass + '"' : '') + '>' + value + '</b></div>';
}

/**
 * 渲染右栏目标TS栏：状态信息 / 活动规律 / 可能的意图。
 *
 * 渲染本身在 `statusPanels.js`，这里只负责把状态取齐（目标定义、当前帧状态、
 * 一级推理结果、帧序列）后交出去。
 *
 * `state.layers.targetPanel`（「目标元数据面板」开关）仍然生效：关掉后点击目标只做
 * 选中与高亮，右栏显示提示而不展开详情。
 */
function renderTargetStatus(id) {
  const host = dom.targetBody;
  if (!host) return;
  if (!window.StatusPanels) return;
  if (id === null || id === undefined || !state.layers.targetPanel) {
    StatusPanels.renderTargetStatus(host, null, null, null, null, null, 0);
    return;
  }
  const rec = state.targets[id];
  if (!rec) {
    StatusPanels.renderTargetStatus(host, null, null, null, null, null, 0);
    return;
  }
  const s = positionFor(id, state.tick);
  const inf = inferenceAt(state.tick);
  const l1 = (inf && inf.targets[id]) || null;
  StatusPanels.renderTargetStatus(
    host, rec.def, s || null, l1, l1, state.data && state.data.frames, state.tick,
    state.data);
  // Step C: activity state (ontology parent class) beside the unit's own phase.
  var fr = (state.data && state.data.frames) ? state.data.frames[state.tick] : null;
  var up = (fr && fr.unit_phase && fr.unit_phase[id]) || null;
  var act = activityName(l1 && l1.primary_intent);
  if (dom.targetBody && (act || up)) {
    var actRow = document.createElement('div');
    actRow.className = 'tp-kv tp-activity';
    var actKey = document.createElement('span');
    actKey.textContent = '\u6d3b\u52a8\u72b6\u6001';
    var actVal = document.createElement('b');
    actVal.textContent = (act || '\u2014') + (up ? (' \u00b7 \u9636\u6bb5 ' + (up.name || up.id || '')) : '');
    actRow.appendChild(actKey);
    actRow.appendChild(actVal);
    dom.targetBody.appendChild(actRow);
  }
}

// --------------------------------------------------------------------------
// 模型开关：每一项都对应一个真实生效的渲染行为
// --------------------------------------------------------------------------

function setTerrain(on) {
  if (on) {
    setSwitchNote('正在加载离线 DEM 高程（Terrarium，z0–z12）…');
    const provider = OfflineAssets.createTerrainProvider();
    viewer.terrainProvider = provider;
    viewer.scene.globe.depthTestAgainstTerrain = true;
    state.terrainOn = true;
    // **不要用 provider.readyPromise**：这个 Cesium 构建里
    // CustomHeightmapTerrainProvider 没有 readyPromise（和 Primitive 一样，
    // 全文件只有 KTX2 转码器带这个名字）。原来写
    // `provider.readyPromise.then(...)` 会直接抛
    // "Cannot read properties of undefined (reading 'then')"，
    // 被 change 事件冒泡成未捕获异常——**地形开关看起来点了没反应，还报错**。
    // 改用轮询 provider.ready，对两种构建都成立。
    setSwitchNote('地形模型已启用：离线 Terrarium 高程，最高 z12。');
    const waitReady = function (tries) {
      if (provider.ready) {
        setSwitchNote('地形模型已启用：离线 Terrarium 高程，最高 z12（相机高度读数已按椭球高）。');
        return;
      }
      if (tries <= 0) return;
      setTimeout(function () { waitReady(tries - 1); }, 250);
    };
    if (provider.ready) waitReady(0);
    else waitReady(60);          // 最多等 15 s，等不到也保持可用状态
  } else {
    // 互斥解算后另一个地形源可能正生效（网格地形赢、本层被判 false）——
    // 这时不能把 provider 打回椭球，否则赢家的地形会被后执行的我们抹掉。
    if (!terrainEffectiveExcept('terrain.model')) {
      viewer.terrainProvider = new Cesium.EllipsoidTerrainProvider();
      viewer.scene.globe.depthTestAgainstTerrain = false;
      state.terrainOn = false;
    }
    setSwitchNote('地形模型已关闭：使用椭球面，高程不参与渲染。');
  }
}

/** 除 `exceptId` 外，是否还有地形图层在互斥解算后生效。 */
function terrainEffectiveExcept(exceptId) {
  try {
    const eff = LayerRegistry.resolve();
    return ['terrain.model', 'terrain.mesh'].some(function (id) {
      return id !== exceptId && !!eff[id];
    });
  } catch (e) {
    return false;
  }
}

/**
 * 台湾 20m DEM（quantized-mesh）地形开关。
 *
 * 与 setTerrain() 并列而不是替换：`CesiumTerrainProvider` 只在有数据的区域
 * （台湾，z0–z14）返回真实高程，区域外 Cesium 按 layer.json 的 available 直接
 * 判为无数据、不再发请求；需要全球低层地形时切回「地形模型」即可。
 *
 * `fromUrl` 返回 Promise。老构建没有静态入口时退回构造器，两者行为一致。
 * 代次 `terrainMeshToken` 保证"点了又关"时晚到的 provider 不会写进场景。
 */
var terrainMeshToken = 0;
function setTerrainMesh(on) {
  const token = ++terrainMeshToken;
  if (!on) {
    if (!terrainEffectiveExcept('terrain.mesh')) {
      viewer.terrainProvider = new Cesium.EllipsoidTerrainProvider();
      viewer.scene.globe.depthTestAgainstTerrain = false;
      state.terrainOn = false;
    }
    setSwitchNote('网格地形已关闭。');
    return;
  }
  setSwitchNote('正在加载台湾 20m 网格地形（quantized-mesh，z0–z14）…');
  const url = './tiles3d/tw_dem_20m/';
  let start;
  try {
    start = Cesium.CesiumTerrainProvider.fromUrl
      ? Cesium.CesiumTerrainProvider.fromUrl(url, {
        requestVertexNormals: true, requestWaterMask: false,
      })
      : Promise.resolve(new Cesium.CesiumTerrainProvider({
        url: url, requestVertexNormals: true, requestWaterMask: false,
      }));
  } catch (e) {
    start = Promise.reject(e);
  }
  start.then(function (provider) {
    if (token !== terrainMeshToken) return;      // 已切走
    viewer.terrainProvider = provider;
    viewer.scene.globe.depthTestAgainstTerrain = true;
    state.terrainOn = true;
    LayerRegistry.setFeed('terrain.mesh', 'ok', null);
    setSwitchNote('网格地形已启用：台湾 20m DEM（quantized-mesh，z0–z14）。');
  }).catch(function (e) {
    if (token !== terrainMeshToken) return;
    LayerRegistry.setFeed('terrain.mesh', 'failed',
      '网格地形加载失败：' + ((e && e.message) || e));
    setSwitchNote('网格地形加载失败：' + ((e && e.message) || e) + '（保持原地形）。');
  });
}

function setWsf3dOverlay(on) {
  LayerRegistry.setDesired('wsf3d.overlay', !!on);
  // 专题层是懒挂载的：挂上之后必须登记槽位，否则它在 imageryLayers 里是 add 到
  // 末尾的，会直接落在天地图注记之上把它盖住（实测就是这样：注记在 index 15，
  // 未登记的 WSF3D 层在 16）。登记 + applyLayers 之后会被排到注记之下。
  if (on && !state.wsf3dOverlayLayer) {
    state.wsf3dOverlayLayer = OfflineAssets.attachWsf3dOverlay(
      viewer, 'buildingheight', 0.62);
    registerImagerySlot('wsf3d.overlay', state.wsf3dOverlayLayer);
  }
  applyLayers();
  if (on) {
    setSwitchNote('建筑高度图层已启用：WSF3D v02 建筑高度（90 m），' +
      '叠加于影像底图之上、天地图注记之下，黑色像元为无数据已剔除。');
  } else {
    setSwitchNote('建筑高度图层已关闭。');
  }
}

/**
 * 在线三维实景（Google Photorealistic 3D Tiles）开关，带自动回落。
 *
 * 设计要点：这是**在线**图层，断网、没令牌、令牌过期都会失败。失败时不能让用户
 * 对着一片空洞——本机的离线体块必须顶上，所以这里在失败路径上主动把
 * `modelBuildings` 打开，并把"为什么回落到本机"写进提示。
 *
 * 两个图层是并列的，不是替换关系：用户可以只开本机、只开在线，或者都不开。
 */
function setBuildingsOnline(on) {
  if (!window.Buildings3D) {
    setSwitchNote('在线三维实景模块未加载（src/buildings3d.js）。');
    return;
  }
  LayerRegistry.setDesired('tiles.google3d', !!on);
  window.Buildings3D.setEnabled(on).then(function (info) {
    // 在线层不可用 → 把**本机**体块的意图打开，让注册表解算后自然落到体块上。
    // 不在这里直接改场景：注册表才是唯一落地点。
    if (info.shouldFallback) {
      const why = info.detail || '在线图层不可用';
      const alreadyLocal = checkboxOn('modelBuildings');
      LayerRegistry.setDesired('buildings.blocks', true);
      LayerRegistry.setFeed('tiles.google3d',
        info.phase === 'unconfigured' ? 'unsupported' : 'failed', why);
      applyLayers();
      syncBuildingsOnlineNote(info);
      // 写 `.checked` 在节点已删除之后是**空操作**，于是"在线实景不可用就自动回落到
      // 本机体块"这条兜底静默失效。改走 setSwitch（有节点派发 change / 没节点写注册表意图）。
      if (!alreadyLocal) setSwitch('modelBuildings', true);
      setSwitchNote('在线三维实景不可用（' + why + '），' +
        (alreadyLocal ? '已继续使用本机建筑体块。' : '已自动回落到本机建筑体块。'));
      return;
    }
    if (info.live) {
      LayerRegistry.setFeed('tiles.google3d', 'ok', info.detail);
    }
    applyLayers();
    syncBuildingsOnlineNote(info);
    if (info.live) {
      setSwitchNote('在线三维实景已开启（' + info.detail + '）。' +
        '注：该模式会隐藏本机 DEM，且瓦片无法离线缓存。');
    } else if (!info.enabled) {
      setSwitchNote('在线三维实景已关闭。');
    }
  }).catch(function (e) {
    setSwitchNote('在线三维实景开启失败：' + (e && e.message ? e.message : e));
  });
}

/**
 * 在线实景链路的状态。
 *
 * 原先把它写成 `#ionTokenNote` 里的一整句「状态：连接失败（无法加载在线实景：等待
 * Cesium ion 超时（12 秒）：本机可能无法访问 Google 瓦片服务）」。按"去掉界面上的
 * 说明性文字"的要求移除；随后令牌输入框也整块删了（token 改由 .env 提供），
 * 所以现在**只剩控制台一条出口**——`Buildings3D.debug()` 里有完整的 phase 与 detail。
 * 界面上不再有"在线实景通没通"的任何指示，这是移除令牌 UI 的必然结果。
 */
function syncBuildingsOnlineNote(info) {
  if (info.phase === 'failed') {
    console.warn('[ion] 在线实景连接失败：' + (info.detail || ''));
  }
}

/**
 * 建筑体块的**实际显隐**（注册表的 primitive sink 调它）。
 *
 * 纯"把状态落到场景上"，不碰 desired——desired 由注册表的 `buildings.blocks` 行写。
 * 两者分开是为了避免回环：sink 若再去写 desired，就会又触发一轮 apply。
 */
function applyBuildingsVisibility(on) {
  if (!on) {
    if (state.buildingsPrimitive) state.buildingsPrimitive.show = false;
    // **必须清掉 buildingsKey**。refreshBuildings 的开头是
    //   if (key === state.buildingsKey && state.buildingsPrimitive) return;
    // 用来避免重复生成；但关掉开关时只把 show 置 false、key 留着，重新打开就会
    // 命中这个提前返回：体块已经被隐藏，却以为"当前组合已经生成好了"，
    // 于是**关了再开就再也看不到体块**（实测 primitiveShow 一直是 false）。
    // 清掉 key 后，重新打开会走缓存命中分支并重新显示。
    state.buildingsKey = null;
    setSwitchNote('建筑体块已关闭。');
    return;
  }
  if (!state.ramp) {
    setSwitchNote('建筑体块需要色带标定文件 tiles/wsf3d/height_ramp.json，正在加载…');
    OfflineAssets.loadHeightRamp().then(function (ramp) {
      state.ramp = ramp;
      // 色带到位后重新解算一次，sink 会再进来，此时 ramp 已有。
      LayerRegistry.setFeed('buildings.blocks', 'ok', null);
      refreshBuildings();
    }).catch(function (e) {
      setSwitchNote('建筑体块不可用：' + e.message +
        '（请先运行 tools/calibrate_wsf3d.py）');
      LayerRegistry.setFeed('buildings.blocks', 'failed', e.message);
      setSwitch('modelBuildings', false);
    });
    return;
  }
  refreshBuildings();
}

/**
 * 登记 imagery 槽位。**多实例层要把每个实例都登记进来**，只登记第 0 个会让其余
 * 实例落在顺序范围之外、继续压在注记上面（这正是"注记被热点影像盖住"没被真正
 * 修掉的原因）。接受单个 layer 或 layer 数组。
 */
function registerImagerySlot(id, layerOrLayers) {
  if (!layerOrLayers) return layerOrLayers;
  const arr = Array.isArray(layerOrLayers) ? layerOrLayers : [layerOrLayers];
  arr.forEach(function (l) { if (l) LayerRegistry.addImagery(id, l); });
  return layerOrLayers;
}

/** 在线实景 tileset 的显隐。mesh 层的前后由深度缓冲决定，这里只管 show。 */
function setOnlineTilesetVisible(on) {
  if (window.Buildings3D && Buildings3D.setVisible) Buildings3D.setVisible(!!on);
}

/**
 * 解算并应用图层状态。**所有**意图变更都走这里，不要各自去改场景。
 *
 * 幂等，但不要在帧循环里无脑调：reorderImagery 会动 imageryLayers 的 index，
 * 每帧动一次会让 Cesium 反复重建表面瓦片顺序。
 */
function applyLayers() {
  // 天地图注记（cva）与本机地名层**都画中文地名**，叠在一起就是重影——实测截图里
  // 杭州、浙江、福州、南昌都能看到清清楚楚的两份，位置还差几个像素。这是"地名看不清"
  // 里最容易被误判成"分辨率不够"的一条。
  //
  // 规则：cva 想要显示时，本机地名层让位（cva 是天地图的官方注记，排布更专业；
  // 本机那层在离线时仍然可用，因为那时 cva 本来就不生效）。
  //
  // **必须在 apply() 之前改 desired**：这样注册表算出的 effective 与实际一致，
  // 健康面板不会出现"开关开着却没在显示"的假状态；在 apply 之后再关就会留下它。
  // 同时把复选框也改掉——界面不能与事实不符。
  // 「有天地图注记就不要离线地名」这条规则**已经移进注册表**（`tianditu.label` 与
  // `places.labels` 互相列进 `exclusives`，见 layerRegistry.js）。
  //
  // 原先这里是硬编码：apply 之前直接把 `places.labels` 的 `desired` 改 false。
  // 那是个**单向开关** —— 用户后来关掉天地图注记，离线地名**再也回不来**，
  // 因为没有任何路径会把它重新打开（实测：cva 关掉后 places 仍是
  // desired:false / effective:false / Places.loaded:false）。
  // 交给注册表的互斥之后，desired 保持为用户意图、判负方自动回归，
  // 面板还会显示"被顶掉"及原因 —— 而不是让用户面对一个"开了却没东西"的开关。

  var out = LayerRegistry.apply();
  state.lastLayerApply = out;
  syncLayerCheckboxes();
  if (out.skipped && out.skipped.length) {
    // 不静默：sink 缺失或抛异常都会在这里露出来。
    console.warn('[layers] apply 跳过：', out.skipped);
  }
  // 图层一变就刷新健康面板。展开着的时候才真渲染，收起时只标脏，
  // 避免每次点开关都白跑一遍 DOM 组装。
  markHealthDirty();
  return out;
}

/**
 * 数据健康诊断。
 *
 * 左栏的「数据健康」面板已随界面重构移除，但诊断结论仍然必须保留出口——否则
 * "开关开着、图层却没生效"这类问题就没有任何证据可查了。因此保留
 * `DataHealth.report()` 的调用口，改为按需查询：`SituationHost.health()`。
 *
 * 原先"面板展开时才渲染、收起时只标脏"的缓存逻辑一并删除：没有面板可刷新了，
 * 留着只会让人以为某处还在渲染它。
 */
function healthReport() {
  if (!window.DataHealth) return { available: false, reason: 'dataHealth.js 未加载' };
  try {
    return DataHealth.report();
  } catch (e) {
    return { available: false, reason: String((e && e.message) || e) };
  }
}

/**
 * 图层变化后的钩子。
 *
 * 面板已移除，诊断改为按需查询，这里不再需要标脏；保留空函数是为了让
 * `applyLayers()` 里的调用点语义明确（"图层变了、诊断结论需要重算"），
 * 而不是把一个"看起来还会渲染"的调用悄悄删掉。
 */
function markHealthDirty() { /* 面板已移除：诊断按需查询，不缓存 */ }

/**
 * 把仲裁结果同步回复选框。
 *
 * 互斥会**替用户关掉**一个层（例如打开离线底图会把天地图影像顶掉）。这时复选框
 * 若还停在勾选状态，界面就在撒谎——用户看到天地图勾着、画面却是离线影像。
 * 这里只改 .checked，**不派发 change**，否则会再走一遍 setDesired 造成回环。
 */
/**
 * 把注册表的**实际**可见性同步回 DOM 复选框。
 *
 * 只对**还剩 DOM 节点**的图层有效 —— 迁进生成式面板的那些层没有静态复选框，
 * 它们的行由 `layerPanel.js` 按 `desired` 重绘（面板签名里带 `feed`，所以互斥导致的
 * `hidden` 变化也会触发重建）。`if (box && …)` 的守卫是必须的：迁移之后
 * `getElementById` 会对这些 id 返回 null，没有守卫就是每轮 apply 抛一次异常。
 *
 * 注意这里同步的是 `effective`（事实）而不是 `desired`（意图）：静态复选框是旧的
 * 交互方式，让用户看到"实际有没有在显示"；面板那边相反，显示 `desired` 并把
 * 被顶掉的原因写在旁边的标记里 —— 两种呈现各有各的道理，不要互相看齐。
 */
function syncLayerCheckboxes() {
  LayerRegistry.status().forEach(function (s) {
    if (!s.checkbox) return;
    const box = document.getElementById(s.checkbox);
    if (box && box.checked !== !!s.effective) box.checked = !!s.effective;
  });
}

/** 图层顺序与状态的取证入口。 */
function debugLayers() {
  return {
    registry: LayerRegistry.status(),
    stacks: LayerRegistry._debug(),
    lastApply: state.lastLayerApply,
    // 渲染调度：能看到"省掉了多少帧无意义重算"以及最近是谁触发的重算。
    governor: window.RenderGovernor ? RenderGovernor.debug() : null,
    imageryCount: viewer && viewer.imageryLayers ? viewer.imageryLayers.length : -1,
    terrainDepthTest: viewer ? viewer.scene.globe.depthTestAgainstTerrain : null,
    globeShow: viewer ? viewer.scene.globe.show : null,
  };
}

/** 屏幕中心射线打到地面的经纬；打不到时退回相机自身的地面投影。 */
function cameraGroundCarto() {
  const canvas = viewer.scene.canvas;
  const ray = viewer.camera.getPickRay(
    new Cesium.Cartesian2(canvas.clientWidth / 2, canvas.clientHeight / 2));
  const point = ray ? viewer.scene.globe.pick(ray, viewer.scene) : null;
  return point ? Cesium.Cartographic.fromCartesian(point) : viewer.camera.positionCartographic;
}

/**
 * 离相机地面点最近的城区 AOI。空态（没有热点）时用它选体块区域。
 *
 * 用 haversine 而不是 |Δlat|+|Δlon|：在纬度 25° 处 1° 经度的地面距离只有 1° 纬度的
 * 约 91%，曼哈顿度量会把经度方向的差异算大。
 */
function nearestAoiToCamera() {
  const aois = state.urbanAoi || [];
  if (!aois.length) return -1;
  const g = cameraGroundCarto();
  const clat = Cesium.Math.toDegrees(g.latitude);
  const clon = Cesium.Math.toDegrees(g.longitude);
  const cosLat = Math.cos(clat * Math.PI / 180);
  const R = 6371000;

  let best = -1;
  let bestD = Infinity;
  aois.forEach(function (aoi, i) {
    const alat = (aoi.bbox[1] + aoi.bbox[3]) / 2;
    const alon = (aoi.bbox[0] + aoi.bbox[2]) / 2;
    const dy = (alat - clat) * Math.PI / 180 * R;
    const dx = (alon - clon) * Math.PI / 180 * R * cosLat;
    const d = Math.sqrt(dx * dx + dy * dy);
    if (d < bestD) { bestD = d; best = i; }
  });
  return bestD <= 200000 ? best : -1;   // AOI 边长约 13 km，200 km 外就不硬套
}

function pickAoiForHotspot(index) {
  const aois = state.urbanAoi || [];
  if (!aois.length) return -1;
  // 空态没有热点，交给"离相机最近的 AOI"分支（见 refreshBuildings）
  if (index < 0 || !state.data) return -1;

  const hs = state.data.hotspots[index];
  if (!hs) return -1;
  const clat = hs.center[0], clon = hs.center[1];

  let best = -1;
  let bestD = Infinity;
  aois.forEach(function (aoi, i) {
    const alat = (aoi.bbox[1] + aoi.bbox[3]) / 2;
    const alon = (aoi.bbox[0] + aoi.bbox[2]) / 2;
    const d = Math.abs(alat - clat) + Math.abs(alon - clon);
    if (d < bestD) { bestD = d; best = i; }
  });
  // 距离太远就不硬套：宁可明说没有覆盖，也不要给错城市的建筑。
  return bestD <= 1.0 ? best : -1;
}

/**
 * 按当前相机高度选择 LOD 档位并（必要时）重新生成建筑体块。
 *
 * 这是 LOD 的落点：相机高时用粗网格快速出图，压低后才做全分辨率聚合。
 * 已生成的组合留在 buildingsCache 里，来回缩放不会重复计算。
 */
function refreshBuildings() {
  // 开关状态**从 DOM 现取**。`dom.modelBuildings` 这个缓存引用实测会失效
  // （isConnected 为 false），拿它判断"开关开没开"会把已经勾上的开关读成没勾，
  // 于是整个体块功能静默不工作（现象：`?buildings=1` 完全没有体块，也不报错）。
  // ⚠ 这里必须走 `checkboxOn()`。上面那段注释说的是"不要用 `dom.modelBuildings`
  // 缓存引用"，于是改成了 `getElementById` 现取 —— 而**复选框本身已经随面板迁移
  // 从 HTML 删除了**，现取同样得到 null，`if (!box …) return` 于是**恒真**：
  // 建筑体块整条链路静默不工作，现象与注释里描述的一模一样（`?buildings=1` 没有
  // 体块、也不报错）。`checkboxOn()` 在没有节点时会反查注册表，两种取法都不会再错。
  if (!checkboxOn('modelBuildings')) return;
  // 不再依赖 state.data：空态下用"离相机最近的城区 AOI"生成体块。
  if (!state.ramp) return;

  const aois = state.urbanAoi || [];
  let aoiIdx = pickAoiForHotspot(state.activeHotspot);
  if (aoiIdx < 0) aoiIdx = nearestAoiToCamera();
  if (aoiIdx < 0) {
    setSwitchNote('当前热点区域内没有预下载的高分辨率 WSF3D 城市区数据；' +
      '可选城市区：' + aois.map(function (a) { return a.name; }).join('、') + '。');
    return;
  }

  const height = viewer.camera.positionCartographic.height;
  const lod = OfflineAssets.lodForHeight(height);
  const key = aoiIdx + ':' + lod;
  if (key === state.buildingsKey && state.buildingsPrimitive) return;

  const cached = state.buildingsCache[key];
  if (cached) {
    swapBuildings(cached, aoiIdx, lod);
    return;
  }
  if (state.buildingBusy) return;

  const aoi = aois[aoiIdx];
  state.buildingBusy = true;
  const t0 = Date.now();
  setSwitchNote('正在按 LOD「' + OfflineAssets.BUILDING_LOD[lod].key + '」生成「' +
    aoi.name + '」三维建筑体块…');

  OfflineAssets.buildBuildingBlocks({
    bbox: aoi.bbox,
    ramp: state.ramp,
    zoom: aoi.zoom || 13,
    lod: lod,
  }).then(function (result) {
    if (!result.primitive) {
      state.buildingBusy = false;
      setSwitchNote('「' + aoi.name + '」没有生成体块：' + JSON.stringify(result.stats));
      return;
    }
    // **先入场景（隐藏）再等 ready，最后才显示。**
    //
    // Primitive 的异步几何准备（asynchronous: true）只会在它**参与渲染**时才推进：
    // 实测把 265 个非凸挤出多边形留在场景外等 readyPromise，等 120 s 都不落地；
    // 而 add 进场景后同一批几何 0.9–1.3 s 就绪。原先的 swapBuildings 是"构造 →
    // await 完成 → 才 add"，于是永远等不到，`buildingBusy` 卡在 true，体块再也不
    // 会生成——现象就是"开着建筑体块却什么都没有，也不报错"。
    //
    // 隐藏着 add 不会影响画面，ready 之后再显示，仍然满足"渲染切换不闪半成品"。
    const prim = result.primitive;
    prim.show = false;
    viewer.scene.primitives.add(prim);
    const ms = Date.now() - t0;
    // 等 ready 用轮询而不是 readyPromise：**这个 Cesium 构建里 Primitive 没有
    // readyPromise**（全文件只出现 4 次，都属于 KTX2 转码器；Primitive 用的是
    // readyEvent）。之前写 `prim.readyPromise.then(...)` 直接抛
    // "Cannot read properties of undefined (reading 'then')"，被 catch 吞成
    // "建筑体块几何准备失败"，体块永远不显示。用 `prim.ready` 判断对两种构建都成立。
    state.pendingPrimitive = prim;
    const waitReady = function () {
      if (prim.isDestroyed && prim.isDestroyed()) return;
      if (!prim.ready) {
        setTimeout(waitReady, 150);
        return;
      }
      state.buildingBusy = false;
      if (state.pendingPrimitive === prim) state.pendingPrimitive = null;
      if (!checkboxOn('modelBuildings')) {
        // 等待期间用户把开关关了：直接丢弃，别把图层留在场景里。
        viewer.scene.primitives.remove(prim);
        return;
      }
      state.buildingsCache[key] = { primitive: prim, stats: result.stats };
      swapBuildings(state.buildingsCache[key], aoiIdx, lod, ms, true);
    };
    setTimeout(waitReady, 150);
  }).catch(function (e) {
    state.buildingBusy = false;
    console.error(e);
    setSwitchNote('建筑体块生成失败：' + e.message);
  });
}

/**
 * @param {boolean} alreadyInScene 该 primitive 已经由调用方 add 进场景（并且是隐藏
 *   状态）时为 true。默认 false，只用于缓存命中那条路径——缓存里的 primitive 已经
 *   在场景里了，切换时要么保留、要么先移除再 add。
 */
function swapBuildings(entry, aoiIdx, lod, ms, alreadyInScene) {
  if (state.buildingsPrimitive && state.buildingsPrimitive !== entry.primitive) {
    viewer.scene.primitives.remove(state.buildingsPrimitive);
    state.buildingsPrimitive = null;
  }
  state.buildingsPrimitive = entry.primitive;
  state.buildingsAoi = aoiIdx;
  state.buildingsKey = aoiIdx + ':' + lod;
  if (!alreadyInScene && viewer.scene.primitives.contains(entry.primitive) === false) {
    viewer.scene.primitives.add(entry.primitive);
  }
  // ready 之后才显示：隐藏状态下 add 进来只是为了推进异步几何准备。
  entry.primitive.show = true;

  const aoi = state.urbanAoi[aoiIdx];
  const st = entry.stats;
  // 结构化留一份：debugBuildings() 直接读它，不再从下面那条中文提示里抠数字。
  state.lastBuildingsStats = {
    aoi: aoi ? aoi.name : null, aoiIdx: aoiIdx, lod: lod,
    lodKey: st.lod, grid: st.grid, step: st.step,
    blocks: st.blocks, skipped: st.skipped,
    buildMs: (ms === undefined ? null : ms), fromCache: (ms === undefined),
  };
  setSwitchNote('建筑体块已显示：「' + aoi.name + '」LOD ' + st.lod +
    '（网格 ' + st.grid + '×' + st.grid + '，采样步长 ' + st.step + '），' +
    st.blocks + ' 个体块' + (ms === undefined ? '（缓存）' : '，用时 ' + ms + ' ms') +
    '。相机高度变化时会自动切换 LOD。');
}

function setWeather(on) {
  viewer.scene.fog.enabled = on;
  viewer.scene.fog.density = on ? 0.00025 : 0.0002;
  viewer.scene.skyAtmosphere.show = true;
  viewer.scene.globe.showGroundAtmosphere = on;
  viewer.scene.haze = on;
  setSwitchNote(on
    ? '气象模型已启用：地面大气与雾效参与渲染（影响远景能见度）。'
    : '气象模型已关闭：关闭地面大气与雾效。');
}

function setSensors(on) {
  state.filters.sensors = on;
  const n = state.data ? state.data.targets.length : 0;
  setSwitchNote(on
    ? '传感器覆盖已启用：显示 ' + n + ' 个目标的传感器探测半径（按真实雷达量级）。'
    : '传感器覆盖已关闭。');
}

// --------------------------------------------------------------------------
// 城区高清影像的 LOD 挂载
// --------------------------------------------------------------------------

// 相机高于这个高度就不需要 z12–z14 影像：那时底图停在 z7 左右，城区层只会
// 白白请求一批用不上的瓦片。降到阈值以下再挂上，飞高时摘掉。
const URBAN_IMAGERY_TRIGGER_M = 200000;

/**
 * 按相机高度挂载/卸载城区高清影像层。
 *
 * 这是"浏览过的地图不再重复取"之外的另一个 LOD 收益点：城区的 4 个 AOI 在
 * 剧院尺度下完全用不到，提前挂上会让 Cesium 为每个表面瓦片都去要一遍
 * z12+ 瓦片，既慢又必然 404。
 */
function syncUrbanImagery() {
  if (!viewer || !state.urbanAoi || !state.urbanAoi.length) return;
  const height = viewer.camera.positionCartographic.height;
  // 地球不可见时城区层同样画不出来，不挂（见 hotspotImageryWanted 的同类守卫）。
  const globeOn = !(viewer.scene && viewer.scene.globe) || viewer.scene.globe.show;
  // 相机需要的层级已明显超过城区的最高层级时也不挂：这一层是城市尺度的粗层
  // （z≤13），在 8 km 这类"需要 z14"的高度给不出细节却仍会全量拉取 —— 那是
  // 实测到的 8 km 卡顿源之一（与热点层同因，见 layerTooCoarseForCamera）。
  const urbanTop = (window.OfflineAssets && OfflineAssets.urbanTreeMaxZoom)
    ? Math.min(((OfflineAssets.manifest().urban_imagery_detail || {}).max_zoom || 99),
               OfflineAssets.urbanTreeMaxZoom)
    : 13;
  const want = state.urbanEnabled && globeOn && !layerTooCoarseForCamera(urbanTop)
    && height <= URBAN_IMAGERY_TRIGGER_M;

  if (want && !state.urbanLayers) {
    state.urbanLayers = OfflineAssets.attachUrbanImagery(viewer, state.urbanAoi);
    // 挂载后必须登记槽位并重排：新层是 add 到 imageryLayers 末尾的，不重排就会
    // 落在天地图注记（z=90）之上，把注记重新盖住。
    if (state.urbanLayers && state.urbanLayers.length) {
      registerImagerySlot('urban.imagery', state.urbanLayers.map(function (r) { return r.layer; }));
    }
    applyLayers();
    return;
  }
  if (!want && state.urbanLayers) {
    state.urbanLayers.forEach(function (rec) {
      viewer.imageryLayers.remove(rec.layer, true);
    });
    state.urbanLayers = null;
    LayerRegistry.dropImagery('urban.imagery');
    applyLayers();
  }
}

function setImagery(on) {
  // 「影像底图（离线）」管全球层 + 战区层；热点层与城区层各有自己的开关。
  // 与天地图**不再互斥**（2026-09-29 起）：离线层 z=5 垫在天地图 z=10 下面，
  // 两者可以同时开——天地图没有瓦片的地方露出离线影像，而不是球体底色。
  // 注记层依旧不受影响——注记是"文字覆盖"而不是底图，用户可以要离线影像 +
  // 天地图注记。
  LayerRegistry.setDesired('imagery.offline', !!on);
  applyLayers();
  setSwitchNote(on
    ? '离线影像底图已启用：全球/战区层（离线 Esri World Imagery），垫在天地图之下。'
    : '离线影像底图已关闭；天地图取不到瓦片的地方会露出球体底色。');
}

/**
 * 天地图底图开关（默认开）。
 *
 * 天地图影像与它的中文注记是**两个 layer**：影像本身不含地名，不叠 `cva` 注记层
 * 地图上一个字都没有。所以打开这个开关会同时打开注记；注记另有一个独立开关
 * 用来在需要时单独关掉（与天地图注记互斥，见 layerRegistry.js 的 exclusives）。
 *
 * 自 2026-09-29 起天地图与**离线底图不再互斥**：离线影像垫在下面（z=5），天地图
 * 盖在上面（z=10）。所以这里不再需要"关掉一个才能开另一个"的说法；key 失效时露出
 * 的是离线影像而不是球体底色。key 只存在服务端 `.env`，前端连它的尾号都拿不到。
 */
function setTianditu(on) {
  LayerRegistry.setDesired('tianditu.base', !!on);
  // 注记跟随影像一起开：不叠 cva 层，天地图上就一个字都没有，用户会以为地名层坏了。
  // 但关闭影像时**不动注记**——用户可能就是想"离线影像 + 天地图注记"。
  if (on) LayerRegistry.setDesired('tianditu.label', true);
  applyLayers();
  setSwitchNote(on
    ? '天地图已启用（在线影像 + 中文注记，经服务端代理取瓦片，key 存于 .env）。'
      + '离线影像仍垫在下面作为兜底。'
    : '天地图影像已关闭（注记层保持当前状态）；可见底图为离线影像。');
}

/**
 * 探测天地图是否真的可用，结论**只写控制台**。
 *
 * 为什么必须真发一次请求：`/api/tdt/status` 只能说明"服务端配了 key"，说明不了
 * "key 还能用"。配额耗尽（429）与被禁用（301005）都只有发了请求才知道——而这个项目
 * 已经吃过一次亏：旧 key 配额耗尽后底图静默变成一片深蓝，很久没人发现。
 *
 * 界面上原先有一条 `#tdtNote` 把这个结论显示在「底图」面板里，按"去掉界面上的说明性
 * 文字"的要求移除了。**失败仍然不会静默**：这里打 console.warn，服务端还有
 * `/api/tdt/status` 与 `/tdt/` 代理的失败日志可以查。三条出口都在，只是不再占界面。
 *
 * 失败**不关闭**天地图图层：留着它，取不到瓦片的地方自然露出下面的离线影像。
 */
function probeTiandituStatus() {
  const report = function (text, level) {
    if (level === 'warn') console.warn('[tdt] ' + text);
    else console.info('[tdt] ' + text);
  };
  /**
   * 天地图不可用时，把离线影像层**自动打开**兜底。
   *
   * 离线层现在默认关（默认底图是天地图 + 注记，画面才统一）。但默认关掉它就丢掉了
   * 上一轮刚补上的兜底：key 一挂、或者断网，整张底图会重新变成球体底色 rgb(0,0,128)
   * ——就是之前"底图变深蓝"那个症状。所以探活失败时在这里自动打开它，并把复选框
   * 同步勾上，让界面与实际一致（否则用户会看到"没勾却在显示"的假状态）。
   */
  const fallbackToOffline = function (why) {
    report('天地图不可用，已自动启用离线影像兜底：' + why, 'warn');
    const box = document.getElementById('modelImagery');
    if (box && !box.checked) box.checked = true;
    setImagery(true);
  };
  fetch('/api/tdt/status', { cache: 'no-store' })
    .then(function (r) { return r.ok ? r.json() : null; })
    .catch(function () { return null; })
    .then(function (st) {
      if (st && st.configured === false) {
        fallbackToOffline('服务端未配置 TIANDITU_KEY');
        return null;
      }
      return OfflineAssets.probeTianditu().then(function (res) {
        if (res.ok) {
          report('天地图可用（经服务端代理取瓦片，key 存于 .env，不下发到浏览器）。');
        } else {
          fallbackToOffline('HTTP ' + res.status
            + (res.reason ? '：' + res.reason : '')
            + '（常见原因：301005 key 已被禁用 / 429 配额耗尽 / '
            + '301012·301013 key 类型与调用方式不符）');
        }
        return null;
      });
    })
    .catch(function (e) {
      fallbackToOffline('状态检查失败 ' + (e && e.message ? e.message : e));
    });
}

// --------------------------------------------------------------------------
// 时间轴
// --------------------------------------------------------------------------

// --------------------------------------------------------------------------
// 地图组件：指南针 / 视点坐标 / 比例尺
// --------------------------------------------------------------------------
//
// Cesium 1.145 没有内置指南针，也没有内置比例尺（内置 widget 只有
// HomeButton / SceneModePicker / BaseLayerPicker / Geocoder /
// NavigationHelpButton / Timeline / Animation / FullscreenButton / VRButton /
// InfoBox / SelectionIndicator / ProjectionPicker / CreditDisplay），三项全部自建。
//
// 三项都与案例无关：只用 viewer 与 state.components，因此未加载TS时同样可用。

const _enuM = new Cesium.Matrix4();
const _enuInv = new Cesium.Matrix4();
const _enuDir = new Cesium.Cartesian3();

/**
 * 相机朝向（弧度）。返回 heading 与 pitch，均在相机所在点的 ENU 局部系下。
 *
 * 不用 `camera.heading`：那个 getter 内部会临时改写 `camera._transform` 并调用
 * 两次 `updateMembers`，10 Hz 高频读取有额外开销。这里用等价的自算实现。
 */
function cameraOrientation() {
  const cam = viewer.camera;
  Cesium.Transforms.eastNorthUpToFixedFrame(cam.positionWC, Cesium.Ellipsoid.WGS84, _enuM);
  Cesium.Matrix4.inverse(_enuM, _enuInv);
  // 方向是向量，必须用 multiplyByPointAsVector（ByPoint 会叠加平移分量）
  Cesium.Matrix4.multiplyByPointAsVector(_enuInv, cam.directionWC, _enuDir);
  Cesium.Cartesian3.normalize(_enuDir, _enuDir);
  return {
    headingRad: Math.atan2(_enuDir.x, _enuDir.y),   // ENU: x=东, y=北
    pitchRad: Math.asin(Math.max(-1, Math.min(1, _enuDir.z))),
  };
}

/**
 * 指南针：盘面 N 指向屏幕上的北，顶端蓝色指标固定代表"屏幕上方 = 相机朝向"。
 *
 * 另一种语义"指针指向相机朝向"在这里没有信息量——屏幕上方本身就是相机朝向，
 * 指针会恒指顶端。
 *
 * 写入值必须是**连续角度**，不能直接写 `atan2` 的结果。
 * `cameraOrientation().headingRad` 由 `atan2` 得到，落在 (-180°, 180°]：航向跨过
 * 正南时会从 +179.9 跳到 -179.9。CSS 对 `rotate()` 的角度是按数值**线性**插值的，
 * 于是 `transition` 会让指针多转 359.8° —— 看上去就是指针猛地晃一下。
 * 这里改为累加"最短角差"，写入值因此始终连续；跨 ±180 只是数值上的事，
 * 不再产生绕圈。
 *
 * 关于"缩放时指针晃动"：**三条候选机制都实测为 0，未能复现**。
 *
 * 曾经按"俯视时 atan2 病态"的假设改过一次（改用视线/屏幕上方向量取水平分量较大者
 * + 圆周平滑），实测被推翻并在 −86° 姿态引入了 10° 跳变，已回退（见提交 d9ef78c）。
 * 此后不再猜，改为**采集实测数据**：见下方 COMPASS_TRACE 与
 * `SituationHost.compassTrace()` / `compassTraceStats()`。
 *
 * 已测的路径与结果：
 *   · 脚本化 setView + 动画 zoomIn/zoomOut，pitch −89.5/−86/−70、heading 0/35/120，
 *     各 40 次采样 → 角度极差 ≤ 0.08°
 *   · **真实滚轮事件** 60 次（交替 ±120，相机高度 900 km → 601 km，确实缩放了），
 *     240 个采样 → 角度极差 **0**、单步最大跳变 **0**、容器位移 **0 px**、尺寸变化 **0 px**
 *
 * 也就是说：写进 DOM 的角度、元素的位置、元素的尺寸**三者都没动**。剩下的解释只有
 * "针在如实反映交互过程中相机航向的微小变化"（那属于正确行为，要消除只能给针加平滑/
 * 迟滞，是**取舍**不是修 bug），或者来自这里驱动不出的路径。若将来又有人报晃动，
 * 先调 `compassTraceStats()`：这三个数只要有一个不为 0，方向就明确了。
 *
 * ⚠ 本 trace 里 `hdg` 字段曾经与 `raw` 用了**不同的区间**（[0,360) vs (-180,180]），
 * 于是 `maxCameraHeadingStepDeg` 报出 360° 的**假跳变**。诊断自己在撒谎，比没有诊断更糟。
 * 已统一归一化。
 */
let _compassWrittenDeg = null;

const COMPASS_TRACE_MAX = 240;
const COMPASS_TRACE = [];

function compassSample(rawDeg, writtenDeg) {
  // 量**容器** `#mapCompass` 而不是被旋转的 `#compRose`：旋转元素的轴对齐包围盒
  // 会随角度胀缩（84px 的方块转会得到 84~119px 的盒子），拿它当"位置移动"会得到
  // 5~14px 的假位移，把人引到"布局抖动"这个错误方向上去——本轮就先踩了这个坑。
  // 容器没有 transform，它的矩形变化才是真的布局/重排。
  const box = dom.mapCompass || dom.compRose;
  const r = box.getBoundingClientRect();
  const c = viewer.camera;
  COMPASS_TRACE.push({
    t: Date.now(),
    raw: Math.round(rawDeg * 100) / 100,
    written: Math.round(writtenDeg * 100) / 100,
    elLeft: Math.round(r.left),
    elTop: Math.round(r.top),
    elW: Math.round(r.width),
    elH: Math.round(r.height),
    // ⚠ 必须和 `raw` 用**同一个区间**。`raw` 来自 `Math.atan2`（区间 (-180, 180]），
    // 而 `camera.heading` 的区间是 [0, 360) —— 同一个物理朝向在两者里可能分别是 0 与
    // 360，直接入 trace 会让 `maxCameraHeadingStepDeg` 报出一个 **360° 的假跳变**
    // （实测就是这样）。那是诊断自己在撒谎，会把下一个查"指针晃动"的人引到一个
    // 不存在的问题上去 —— 而这条 trace 存在的唯一目的就是查它。
    hdg: Math.round((((Cesium.Math.toDegrees(c.heading) + 540) % 360) - 180) * 100) / 100,
    pitch: Math.round(Cesium.Math.toDegrees(c.pitch) * 100) / 100,
    heightM: Math.round(c.positionCartographic.height),
  });
  if (COMPASS_TRACE.length > COMPASS_TRACE_MAX) COMPASS_TRACE.shift();
}

function updateCompass() {
  if (!dom.compRose) return;
  const rawDeg = Cesium.Math.toDegrees(cameraOrientation().headingRad);
  if (_compassWrittenDeg === null) {
    _compassWrittenDeg = rawDeg;
  } else {
    // 归一到 (-180, 180] 后累加：写入值连续，跨 ±180 不产生绕圈。
    let d = rawDeg - _compassWrittenDeg;
    d = ((d + 540) % 360) - 180;
    _compassWrittenDeg += d;
    // 长时间运行后把数值收敛回 ±360 内（不影响视觉，只避免数值越滚越大）
    if (Math.abs(_compassWrittenDeg) > 3600) _compassWrittenDeg = _compassWrittenDeg % 360;
  }
  dom.compRose.style.transform = 'rotate(' + (-_compassWrittenDeg) + 'deg)';
  compassSample(rawDeg, _compassWrittenDeg);
}

/** 视点坐标：相机自身的经纬高。高度是椭球高（WGS84），不是海拔。 */
function updateViewpoint() {
  if (!dom.mvAll) return;
  const c = viewer.camera.positionCartographic;
  const lon = Cesium.Math.toDegrees(c.longitude);
  const lat = Cesium.Math.toDegrees(c.latitude);
  const alt = c.height < 1000
    ? c.height.toFixed(0) + ' m'
    : (c.height / 1000).toFixed(c.height < 100000 ? 2 : 1) + ' km';
  // 单行显示（底部居中）。用 | 分隔而不是空格，读数紧凑且便于对齐比较。
  dom.mvAll.textContent = fmtDeg(lon, 'E', 'W') + '|' + fmtDeg(lat, 'N', 'S') + '|' + alt;
  dom.mvAll.parentNode.title = '视点经纬高：椭球高（WGS84 椭球面），非海拔';
}

/**
 * 屏幕点对应的地面经纬。三级取点：
 *   1. scene.globe.pick —— 只对**已渲染瓦片**求交，瓦片没加载完会返回 undefined
 *   2. IntersectionTests.rayEllipsoid —— 解析求交，不依赖瓦片
 *   3. 都没有则返回 null（由调用方兜底）
 */
function groundCartoAt(px, py) {
  const ray = viewer.camera.getPickRay(new Cesium.Cartesian2(px, py));
  if (!ray) return null;
  const hit = viewer.scene.globe.pick(ray, viewer.scene);
  if (hit) return Cesium.Cartographic.fromCartesian(hit);
  const t = Cesium.IntersectionTests.rayEllipsoid(ray, Cesium.Ellipsoid.WGS84);
  if (!t) return null;
  return Cesium.Cartographic.fromCartesian(
    Cesium.Ray.getPoint(ray, t.start, new Cesium.Cartesian3()));
}

const SCALE_PROBE_PX = 100;

/**
 * 比例尺标签的取值方式。
 *
 *   'exact' —— 取横条实际代表的地面距离（2 位有效数字）。**默认**。
 *   'nice'  —— 取 1/2/5×10ⁿ 的整数档。
 *
 * 为什么默认精确：横条宽度现在是**固定**的（`style.css` 的 `--sb-width`），
 * 于是标签是唯一定义比例的东西。若取整数档，条长与文本最多可差约 ±40%
 * （`n` 略低于 3.5 时取 2，偏差 43%），而比例尺是测量参照，差这么多不能用。
 * 整数档读起来更顺口，所以留成开关备选。
 */
const SCALE_LABEL_MODE = 'exact';

/** 地面距离 → 比例尺文本。≥1 km 用 km，否则用 m；保留 2 位有效数字。 */
function formatScaleDistance(meters) {
  if (!isFinite(meters) || meters <= 0) return '—';
  if (meters >= 1000) {
    const km = meters / 1000;
    return (km >= 10 ? km.toFixed(0) : km.toFixed(1)) + ' km';
  }
  return (meters >= 10 ? Math.round(meters) : meters.toFixed(1)) + ' m';
}

/**
 * 比例尺：在**比例尺实际绘制处**取两个屏幕点量地面距离，换算每像素米数。
 *
 * 取样点不能用屏幕中心：3D 倾斜视图下近端与远端的每像素米数可差数倍，
 * 用中心值会让画在左下的横条与实际地面长度对不上。
 * 探针跨度固定 100 px（与横条的固定宽度解耦），单趟即可算出每像素米数；
 * 横条代表的距离再用它**自己的实际宽度**换算 —— 这样改了 CSS 宽度也不必同步改探针。
 */
function updateScalebar() {
  if (!dom.sbBar || !dom.sbLabel) return;
  const canvas = viewer.scene.canvas;
  const cr = canvas.getBoundingClientRect();
  const br = dom.sbBar.getBoundingClientRect();
  if (!cr.width || !br.width) return;

  const x0 = br.left - cr.left;
  const y = (br.top + br.bottom) / 2 - cr.top;
  const a = groundCartoAt(x0, y);
  const b = groundCartoAt(x0 + SCALE_PROBE_PX, y);

  let mPerPx;
  if (a && b) {
    // 用测地线直接量测，不需要手工乘 cos(lat)：纬度效应已隐含在几何里。
    mPerPx = new Cesium.EllipsoidGeodesic(a, b).surfaceDistance / SCALE_PROBE_PX;
  } else {
    // 兜底：视锥解析式。近似"垂直俯视 + 平地"。
    const frustum = viewer.camera.frustum;
    const h = canvas.clientHeight;
    if (!frustum.fovy || !h) { dom.mapScalebar.classList.add('hidden'); return; }
    mPerPx = 2 * viewer.camera.positionCartographic.height * Math.tan(frustum.fovy / 2) / h;
  }
  if (!isFinite(mPerPx) || mPerPx <= 0) return;

  // 横条代表的距离 = 每像素米数 × 横条**自己的**宽度（不写宽度，宽度由 CSS 定）。
  const span = mPerPx * br.width;
  let meters = span;
  if (SCALE_LABEL_MODE === 'nice') {
    const pow = Math.pow(10, Math.floor(Math.log10(span)));
    const n = span / pow;
    meters = (n < 1.5 ? 1 : n < 3.5 ? 2 : n < 7.5 ? 5 : 10) * pow;
  }
  dom.sbLabel.textContent = formatScaleDistance(meters);
}

let compTick = 0;

/** 统一刷新三个组件。全关时零开销。 */
function updateMapComponents() {
  if (!viewer || !state.components) return;
  if (!state.components.compass && !state.components.viewpoint && !state.components.scalebar) return;
  compTick++;
  if (state.components.compass) updateCompass();
  if (state.components.viewpoint) updateViewpoint();
  // 比例尺每次要发 2 条射线，10 Hz 没必要，压到约 3 Hz
  if (state.components.scalebar && compTick % 3 === 0) updateScalebar();
}

const COMPONENT_NODES = {
  compass: 'mapCompass',
  scalebar: 'mapScalebar',
  viewpoint: 'mapViewpoint',
};

// --------------------------------------------------------------------------
// 鸟瞰图（第二个 Cesium 2D 视图，实现见 src/overview.js）
// --------------------------------------------------------------------------

/**
 * 显示/隐藏鸟瞰图。
 *
 * 首次显示时才创建 Viewer（惰性），避免启动期与主视图抢那 6 条/域的连接。
 * 隐藏只加 class 不销毁：瓦片已缓存，再次显示无需重载。
 */
function setOverview(on) {
  state.layers.overview = !!on;
  // 顶栏的「鸟瞰图」按钮已移除，可见性改由左栏「地图组件 / TS图层」里的
  // `#layerOverview` 复选框驱动，勾选状态与 state 保持同步。
  const box = document.getElementById('layerOverview');
  if (box) box.checked = !!on;

  if (!on) { Overview.hide(); return; }
  if (!window.Overview) return;
  try {
    Overview.show();
    setSwitchNote('全局鸟瞰已显示：按战场范围自动取景，标出红蓝全部目标与战损状态。点击目标可跳转主视角。');
  } catch (e) {
    state.layers.overview = false;
    if (box) box.checked = false;
    setSwitchNote('全局鸟瞰不可用：' + e.message);
  }
}

function setComponent(name, on) {
  state.components[name] = !!on;
  const node = document.getElementById(COMPONENT_NODES[name]);
  if (node) node.classList.toggle('hidden', !on);
  if (on && viewer) updateMapComponents();
}

// --------------------------------------------------------------------------
// 目标视角：普通居中跟随（center）与第三人称斜后方跟随（chase）
// --------------------------------------------------------------------------
//
// 两种模式共用同一套参考系：camera.lookAt + HeadingPitchRange 把相机锁在
// 目标 ENU 上。锁定的好处正是需求要的语义 —— 目标始终在视窗中心，拖动是绕
// 目标公转；代价是解除时必须 lookAtTransform(Matrix4.IDENTITY)（见
// detachFollow 的说明）。
//
// ⚠ 缩放**不能交给 Cesium 的默认滚轮**：1.145 的 ScreenSpaceCameraController
// 在 3D 下按光标位置缩放，光标偏离目标时目标会被推离中心。本模块在 canvas 的
// capture 阶段接管 wheel（center/chase 都直接改"相机→目标"的斜距再 lookAt），
// 因此缩放中心恒为目标。没有绑定目标时才交回 Cesium。
//
//   center  普通选中模式：跟随目标平移，保留用户自己的方位/俯仰/距离；
//           拖动=绕目标公转，滚轮=以目标为中心缩放。
//   chase   第三人称斜后方：相机绕到目标运动方向的后上方，视线沿目标前进
//           方向；航向/俯仰/距离用指数阻尼收敛，转弯、加减速时不会跳。
//
// HeadingPitchRange 的语义（由 1.145 源码 offsetFromHeadingPitchRange 推出）：
//   offset_ENU = ( -r·cos p·sin h, -r·cos p·cos h, -r·sin p )
//   ⇒ heading 是**相机视线方位角**，相机**位置**方位角 = heading + 180°。
// 所以 chase 模式 heading = 目标航向 时，相机恰好在目标正后方并沿前进方向观察。

const TARGET_VIEW = {
  sea:  { range:  6000, pitchDeg: -32 },   // 选中后的默认观察距离/俯角
  land: { range:  6000, pitchDeg: -36 },
  air:  { range: 22000, pitchDeg: -22 },
};

const CHASE_VIEW = {
  sea:  { range:  3500, pitchDeg: -20 },   // 高于目标约 1.2 km、后方 3.3 km
  land: { range:  4000, pitchDeg: -28 },   // 高于目标约 1.9 km
  air:  { range: 15000, pitchDeg: -16 },   // 高于目标约 4.1 km
};

// 距离钳制：太小会穿进模型，太大失去"跟随"的意义。
const FOLLOW_RANGE_MIN = 300;
const FOLLOW_RANGE_MAX = 120000;
// 目标移动超过该距离才重下 lookAt（米）。静止时完全不碰相机。
const FOLLOW_MOVE_EPS = 0.02;
// 初始居中过渡：相机沿起止位置间的测地线插值，同时转到目标/跟随位。
const GLIDE_MS = 1200;
// chase 模式下拖拽结束到恢复跟随的停顿（毫秒），让用户能停在手动视角上看一眼。
const CHASE_RESUME_DELAY_MS = 700;

const _followHpr = new Cesium.HeadingPitchRange(0, 0, 0);
const _rigEnu = new Cesium.Matrix4();
const _rigInv = new Cesium.Matrix4();
const _rigLocal = new Cesium.Cartesian3();
const _rigOffset = new Cesium.Cartesian3();
const _rigEnd = new Cesium.Cartesian3();
const _rigPos = new Cesium.Cartesian3();
const _rigGeo = new Cesium.EllipsoidGeodesic();
const _rigMid = new Cesium.Cartographic();
const _rigEndCarto = new Cesium.Cartographic();
const _rigHpr = { heading: 0, pitch: 0, range: 0 };

function wrapPi(a) {
  a = (a + Math.PI) % (Math.PI * 2);
  if (a < 0) a += Math.PI * 2;
  return a - Math.PI;
}

/** 从 from 转到 to 的最短角差，落在 (-π, π]。 */
function shortestAngle(from, to) {
  return wrapPi(to - from);
}

function clampRange(r) {
  return Math.max(FOLLOW_RANGE_MIN, Math.min(FOLLOW_RANGE_MAX, r));
}

// 用户缩放不受"跟随预设"的 120 km 上限约束，只防相机穿进模型或飞出地球。
const ZOOM_RANGE_MIN = 10;
const ZOOM_RANGE_MAX = 20000000;

function clampZoomRange(r) {
  return Math.max(ZOOM_RANGE_MIN, Math.min(ZOOM_RANGE_MAX, r));
}

/** 滚轮增量（像素）。deltaMode: 0=pixel, 1=行, 2=页（Firefox 常用行模式）。 */
function wheelPixels(e) {
  return e.deltaMode === 1 ? e.deltaY * 16 : e.deltaMode === 2 ? e.deltaY * 400 : e.deltaY;
}

function easeInOut(t) {
  return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
}

/**
 * 目标**当前显示位置**：优先取 CZML 在 viewer.clock.currentTime 的采样
 * （帧间连续插值），拿不到再退回整数帧坐标。相机与模型因此走同一个时间源，
 * 近景不会出现"模型在动、相机按 tick 跳步"的相对蹭动。
 */
function liveTargetPosition(id) {
  if (!viewer) return null;
  const rec = state.targets[id];
  if (!rec) return null;
  try {
    if (rec.entity && rec.entity.position) {
      const p = rec.entity.position.getValue(viewer.clock.currentTime);
      if (p) return Cesium.Cartesian3.clone(p, new Cesium.Cartesian3());
    }
  } catch (e) { /* 采样属性尚未挂上 / 时钟无效：走帧坐标兜底 */ }
  const s = positionFor(id, state.tick);
  if (!s) return null;
  return Cesium.Cartesian3.fromDegrees(s.lon, s.lat, s.alt);
}

/** 由目标点与视线 heading/pitch/range 求相机世界位置。 */
function offsetPosition(target, heading, pitch, range, result) {
  Cesium.Transforms.eastNorthUpToFixedFrame(target, Cesium.Ellipsoid.WGS84, _rigEnu);
  _rigOffset.x = -range * Math.cos(pitch) * Math.sin(heading);
  _rigOffset.y = -range * Math.cos(pitch) * Math.cos(heading);
  _rigOffset.z = -range * Math.sin(pitch);
  return Cesium.Matrix4.multiplyByPoint(_rigEnu, _rigOffset, result);
}

/**
 * 当前相机相对某目标点的球坐标（视线 heading/pitch + 斜距）。
 * 自己算而不用 camera.heading/pitch：这两个 getter 会临时改写
 * camera._transform 并两次 updateMembers，逐帧读的代价不划算（README §8.1 同结论）。
 */
function hprFromCameraTo(target, out) {
  Cesium.Transforms.eastNorthUpToFixedFrame(target, Cesium.Ellipsoid.WGS84, _rigEnu);
  Cesium.Matrix4.inverseTransformation(_rigEnu, _rigInv);
  Cesium.Matrix4.multiplyByPoint(_rigInv, viewer.camera.positionWC, _rigLocal);
  const range = Cesium.Cartesian3.magnitude(_rigLocal);
  out = out || _rigHpr;
  if (range < 1e-3) {
    out.heading = 0;
    out.pitch = -Math.PI / 2;
    out.range = FOLLOW_RANGE_MIN;
    return out;
  }
  out.range = range;
  out.pitch = Math.asin(Cesium.Math.clamp(-_rigLocal.z / range, -1, 1));
  out.heading = Math.atan2(-_rigLocal.x, -_rigLocal.y);
  if (out.heading < 0) out.heading += Math.PI * 2;
  return out;
}

/** 目标航向（弧度）。数据没有 heading 时用前后两帧位置求方位，再退回当前视线。 */
function targetHeadingRad(id, s, fallback) {
  const h = s && Number(s.heading);
  if (isFinite(h)) return Cesium.Math.toRadians(((h % 360) + 360) % 360);
  const f = state.follow;
  if (f.lastPos) {
    const p = liveTargetPosition(id);
    if (p) {
      const a = Cesium.Cartographic.fromCartesian(f.lastPos);
      const b = Cesium.Cartographic.fromCartesian(p);
      const dLon = b.longitude - a.longitude;
      const dLat = b.latitude - a.latitude;
      if (Math.abs(dLon) > 1e-9 || Math.abs(dLat) > 1e-9) {
        return Math.atan2(dLon * Math.cos(b.latitude), dLat);
      }
    }
  }
  return fallback;
}

/** 被跟随目标是否仍然可跟随（存在、可见、有位置）。 */
function followTargetUsable(id) {
  const rec = state.targets[id];
  if (!rec) return false;
  if (!isVisible(rec.def)) return false;
  return !!liveTargetPosition(id);
}

/** 把相机锁到目标 ENU 参考系的指定球坐标位置。 */
function lockAt(pos, hpr) {
  const f = state.follow;
  _followHpr.heading = hpr.heading;
  _followHpr.pitch = hpr.pitch;
  _followHpr.range = Math.max(1, hpr.range);
  viewer.camera.lookAt(pos, _followHpr);
  f.lastLockMs = performance.now();
  if (!f.focus) f.focus = new Cesium.Cartesian3();
  Cesium.Cartesian3.clone(pos, f.focus);
  if (!f.lastPos) f.lastPos = new Cesium.Cartesian3();
  Cesium.Cartesian3.clone(pos, f.lastPos);
}

/**
 * center 模式的一帧：把相机按"相对目标的同一球坐标"平移到新目标位置。
 * 只读几何、不做缓动 —— 目标连续移动时相机平移也连续；用户拖动/缩放后
 * 读到的就是新的 heading/pitch/range，因此不会与交互抢相机。
 */
function applyCenterFrame(pos) {
  const f = state.follow;
  if (f.focus && Cesium.Cartesian3.distance(pos, f.focus) < FOLLOW_MOVE_EPS) return;
  const ref = f.focus || pos;
  const cur = hprFromCameraTo(ref);
  lockAt(pos, cur);
}

/** chase 模式的一帧：目标位置精确跟随，姿态用指数阻尼收敛到后上方。 */
function applyChaseFrame(nowMs, pos, rec, s) {
  const f = state.follow;
  const v = CHASE_VIEW[rec.def.domain] || CHASE_VIEW.sea;

  if (!f.lastNow) f.lastNow = nowMs;
  // 不限死在一帧的 1/10：headless / 低帧率下（例如 5 fps）限成 0.1 s 会让阻尼
  // 实际慢一倍，转弯看起来"拖泥带水"。指数阻尼对任意 dt 稳定，只防切回标签页
  // 后一次跳太远，故上限取 0.5 s。
  const dt = Math.max(0, Math.min(0.5, (nowMs - f.lastNow) / 1000));
  f.lastNow = nowMs;

  const desiredHeading = targetHeadingRad(f.id, s, f.pose.heading);
  const desiredPitch = Cesium.Math.toRadians(v.pitchDeg);
  const desiredRange = f.chaseRange || v.range;

  const kH = 1 - Math.exp(-dt / 0.45);
  const kP = 1 - Math.exp(-dt / 0.50);
  const kR = 1 - Math.exp(-dt / 0.65);

  f.pose.heading = wrapPi(f.pose.heading + shortestAngle(f.pose.heading, desiredHeading) * kH);
  f.pose.pitch += (desiredPitch - f.pose.pitch) * kP;
  f.pose.range += (desiredRange - f.pose.range) * kR;

  const moved = !f.focus || Cesium.Cartesian3.distance(pos, f.focus) >= FOLLOW_MOVE_EPS;
  const settling =
    Math.abs(shortestAngle(f.pose.heading, desiredHeading)) > 0.002 ||
    Math.abs(f.pose.pitch - desiredPitch) > 0.002 ||
    Math.abs(f.pose.range - desiredRange) > 0.5;
  if (!moved && !settling) return;

  lockAt(pos, f.pose);
}

/** glide 过渡的一帧：位置沿测地线插值，姿态按最短角插值，目标逐渐居中。 */
function applyGlideFrame(nowMs) {
  const f = state.follow;
  const g = f.glide;
  if (!g) return;
  const raw = Math.max(0, Math.min(1, (nowMs - g.t0) / g.duration));
  const e = easeInOut(raw);

  const pos = liveTargetPosition(f.id);
  if (!pos) { f.glide = null; f.mode = g.finalMode; return; }

  offsetPosition(pos, g.endHeading, g.endPitch, g.endRange, _rigEnd);
  const endCarto = Cesium.Cartographic.fromCartesian(_rigEnd, Cesium.Ellipsoid.WGS84, _rigEndCarto);

  let dest;
  if (Cesium.Cartesian3.distance(g.startPos, _rigEnd) < 0.5) {
    dest = Cesium.Cartesian3.clone(_rigEnd, _rigPos);
  } else {
    _rigGeo.setEndPoints(g.startCarto, endCarto);
    const mid = _rigGeo.interpolateUsingFraction(e, _rigMid);
    const height = g.startCarto.height + (endCarto.height - g.startCarto.height) * e;
    dest = Cesium.Cartesian3.fromRadians(
      mid.longitude, mid.latitude, height, Cesium.Ellipsoid.WGS84, _rigPos);
  }
  const heading = wrapPi(g.startHeading + shortestAngle(g.startHeading, g.endHeading) * e);
  const pitch = g.startPitch + (g.endPitch - g.startPitch) * e;
  viewer.camera.setView({ destination: dest, orientation: { heading: heading, pitch: pitch, roll: 0 } });
  f.lastLockMs = nowMs;

  if (raw >= 1) finishGlide(pos);
}

function finishGlide(pos) {
  const f = state.follow;
  const g = f.glide;
  if (!g) return;
  const cur = hprFromCameraTo(pos);
  f.glide = null;
  f.mode = g.finalMode;
  f.pose.heading = cur.heading;
  f.pose.pitch = cur.pitch;
  f.pose.range = cur.range;
  f.chaseRange = f.mode === 'chase' ? clampRange(g.endRange) : null;
  f.lastNow = 0;
  lockAt(pos, cur);
  setStatus((f.mode === 'chase' ? '第三人称跟随：' : '目标居中跟随：') + f.id, 'ok');
}

/** 用户在地图上一按下就结束过渡；相机停在当前机位并立刻锁定目标。 */
function finishGlideNow() {
  const f = state.follow;
  if (f.mode !== 'glide' || !f.glide) return;
  const g = f.glide;
  const pos = liveTargetPosition(f.id);
  if (!pos) { f.glide = null; f.mode = g.finalMode; return; }
  finishGlide(pos);
}

/** 选中目标后的目标位姿：center 保留来向、chase 取目标航向的正后方。 */
function endHprFor(rec, cur, mode, height) {
  const dom = rec.def.domain;
  if (mode === 'chase') {
    const v = CHASE_VIEW[dom] || CHASE_VIEW.sea;
    const s = positionFor(rec.def.id, state.tick);
    return {
      heading: targetHeadingRad(rec.def.id, s, cur.heading),
      pitch: Cesium.Math.toRadians(v.pitchDeg),
      range: v.range,
    };
  }
  const v = TARGET_VIEW[dom] || TARGET_VIEW.sea;
  if (height > 0) {
    const pitch = Cesium.Math.toRadians(-55);
    return { heading: cur.heading, pitch: pitch, range: height / Math.sin(-pitch) };
  }
  const maxPitch = dom === 'air' ? Cesium.Math.toRadians(20) : Cesium.Math.toRadians(-5);
  return {
    heading: cur.heading,
    pitch: Cesium.Math.toRadians(Cesium.Math.clamp(
      Cesium.Math.toDegrees(cur.pitch), -85, Cesium.Math.toDegrees(maxPitch))),
    range: clampRange(Math.min(Math.max(cur.range, v.range * 0.8), v.range * 4)),
  };
}

/**
 * 绑定/重绑定相机到某个目标（点选、面板「定位」、深链共用）。
 *
 *   force  同一目标也重新做一次居中过渡（定位按钮）。
 *   height 以"目标上方 height 米"为目标机位（面板定位的原有语义）。
 */
function beginTargetView(id, opts) {
  opts = opts || {};
  const f = state.follow;
  const rec = state.targets[id];
  if (!rec || !state.data || !viewer) return;
  const pos = liveTargetPosition(id);
  if (!pos) return;
  const mode = f.enabled ? 'chase' : 'center';
  const same = f.id === id && (f.mode === 'center' || f.mode === 'chase');
  if (same && !opts.force) {
    if (f.mode !== mode) setFollowMode(f.enabled);
    return;
  }

  f.id = id;
  f.focus = null;        // 旧目标的注视点不能带到新目标上（航向兜底会读错方位）
  f.lastPos = null;
  f.suspend = false;
  f.press = null;
  f.resumeAt = 0;
  f.notified = false;
  f.lastNow = 0;

  viewer.camera.cancelFlight();
  // 换目标时可能还锁在旧目标的 ENU 上：先解锁，否则下面的世界坐标计算会混用参考系。
  viewer.camera.lookAtTransform(Cesium.Matrix4.IDENTITY);

  const cur = hprFromCameraTo(pos);
  const end = endHprFor(rec, cur, mode, opts.height);
  const startPos = Cesium.Cartesian3.clone(viewer.camera.positionWC, new Cesium.Cartesian3());
  f.glide = {
    t0: performance.now(),
    duration: GLIDE_MS,
    startCarto: Cesium.Cartographic.clone(
      Cesium.Cartographic.fromCartesian(startPos), new Cesium.Cartographic()),
    startPos: startPos,
    startHeading: viewer.camera.heading,
    startPitch: viewer.camera.pitch,
    endHeading: end.heading,
    endPitch: end.pitch,
    endRange: end.range,
    finalMode: mode,
  };
  f.mode = 'glide';
  f.chaseRange = mode === 'chase' ? clampRange(end.range) : null;
}

/**
 * 勾选/取消「目标跟随」：只切换相机驾驶模式，不解除与目标的绑定。
 * 取消勾选 → 回到普通居中模式（保持当前视角）；勾选 → 绕到目标斜后方。
 */
function setFollowMode(enabled) {
  const f = state.follow;
  f.enabled = !!enabled;
  if (!f.id) return;
  if (f.mode === 'glide') {
    // 过渡中点开关：先收敛到当前机位（不改变绑定），再按新模式接管，
    // 否则会先按旧模式的终点飞完、再被新姿态拉一次，多出一段可见的摆动。
    const pos = liveTargetPosition(f.id);
    if (pos) finishGlide(pos);
    else { if (f.glide) f.glide.finalMode = f.enabled ? 'chase' : 'center'; return; }
  }
  if (f.enabled) {
    if (f.mode !== 'center') return;
    const pos = liveTargetPosition(f.id);
    if (!pos) return;
    const cur = hprFromCameraTo(pos);
    f.pose.heading = cur.heading;
    f.pose.pitch = cur.pitch;
    f.pose.range = cur.range;
    const rec = state.targets[f.id];
    const v = CHASE_VIEW[(rec && rec.def.domain) || 'sea'] || CHASE_VIEW.sea;
    f.chaseRange = v.range;
    f.mode = 'chase';
    f.suspend = false;
    f.resumeAt = 0;
    f.lastNow = 0;
    setStatus('第三人称跟随：' + f.id, 'ok');
  } else if (f.mode === 'chase') {
    f.mode = 'center';
    f.suspend = false;
    f.resumeAt = 0;
    const pos = liveTargetPosition(f.id);
    if (pos) lockAt(pos, hprFromCameraTo(pos));
    setStatus('目标居中跟随：' + f.id, 'ok');
  }
}

/** 每帧驱动相机。挂在 scene.preRender 上，本帧生效。 */
function applyCameraRig(nowMs) {
  const f = state.follow;
  if (!viewer || !state.data || !f.id) return;
  const rec = state.targets[f.id];
  if (!rec || !isVisible(rec.def)) { detachFollow('目标已隐藏'); return; }

  if (f.mode === 'glide') { applyGlideFrame(nowMs); return; }

  const pos = liveTargetPosition(f.id);
  if (!pos) return;

  const s = positionFor(f.id, state.tick);
  if (s && s.reached_end && !f.notified) {
    f.notified = true;
    setStatus('目标 ' + f.id + ' 已到达航路终点，跟随保持', 'ok');
  }

  if (f.mode === 'chase') {
    if (f.suspend) {
      // 拖拽期间仍让目标保持居中（只是不强制后上方姿态）；松手后平滑归位。
      if (f.resumeAt && !f.press && nowMs >= f.resumeAt) {
        f.suspend = false;
        f.resumeAt = 0;
        const cur = hprFromCameraTo(pos);
        f.pose.heading = cur.heading;
        f.pose.pitch = cur.pitch;
        f.pose.range = cur.range;
        f.chaseRange = clampRange(cur.range);
        f.lastNow = nowMs;
      } else {
        applyCenterFrame(pos);
        return;
      }
    }
    applyChaseFrame(nowMs, pos, rec, s);
  } else {
    applyCenterFrame(pos);
  }
}

/**
 * 解除绑定/跟随。
 *
 * **必须** lookAtTransform(Matrix4.IDENTITY)：camera.lookAt 会把 camera.transform
 * 设成目标 ENU 矩阵，ScreenSpaceCameraController 随即把 _globe 置 undefined、
 * _ellipsoid 换成单位球 —— 缩放与贴地钳制失效、可以穿到地面以下。而
 * setView / flyTo 自己**不会**解锁（它们保存并恢复原 transform），所以所有
 * 视角跳转入口都要先经过这里，否则 orientation 会被按目标 ENU 解释。
 */
function detachFollow(reason) {
  const f = state.follow;
  const was = f.id;
  f.id = null;
  f.mode = 'off';
  f.suspend = false;
  f.press = null;
  f.resumeAt = 0;
  f.glide = null;
  f.focus = null;
  f.lastPos = null;
  f.chaseRange = null;
  f.notified = false;
  f.lastNow = 0;
  f.lastLockMs = 0;
  f.enabled = false;
  if (dom.followTarget) dom.followTarget.checked = false;
  if (viewer) viewer.camera.lookAtTransform(Cesium.Matrix4.IDENTITY);
  if (was) {
    setStatus('已解除目标跟随（' + was + '）' + (reason ? '：' + reason : ''), 'ok');
    setFollowNote('已解除。再次点击目标回到居中跟随；勾选「目标跟随」切换第三人称视角。');
    // 跟随期间 camera.changed 每帧触发，建筑体块的防抖被反复重置而从不执行；
    // 解除后补一次，让当前机位下的体块 LOD 归位。
    if (checkboxOn('modelBuildings')) refreshBuildings();
  }
}

/** 过滤/显隐变化后检查绑定是否仍然成立。 */
function guardFollow() {
  if (state.follow.id && !followTargetUsable(state.follow.id)) {
    detachFollow('目标已隐藏或过滤');
  }
}

/**
 * 目标视角下的用户输入。
 *
 * 判据：相机驱动只改相机状态、**从不派发 DOM 事件**，所以画布上任何
 * pointer/wheel 事件都来自用户。不能用 camera.changed 或 moveStart/moveEnd——
 * 它们内部会比较 camera.transform，而跟随每帧都在改 transform，会被持续触发。
 *
 * center：拖动/滚轮全部交给 Cesium —— 参考系原点就是目标，所以公转与缩放
 *         天然围绕目标，不需要也不应该解除跟随。
 * chase：按下暂时让权（suspend），期间仍保持目标居中；拖拽松手 0.7 s 后
 *         阻尼回到后上方，滚轮缩放的距离被采用为新的跟随距离。
 */
const DRAG_THRESHOLD_PX = 4;

function bindFollowInput() {
  const canvas = viewer.scene.canvas;

  // 松手/取消的统一收尾。拖拽过 → 稍后恢复 chase；纯点击 → 立即恢复。
  function releaseFollowPress(resume) {
    const f = state.follow;
    const press = f.press;
    if (!press) return;
    f.press = null;
    if (f.mode !== 'chase') return;
    if (resume || press.moved) {
      f.resumeAt = performance.now() + CHASE_RESUME_DELAY_MS;
    } else {
      f.suspend = false;
      f.resumeAt = 0;
    }
  }

  canvas.addEventListener('pointerdown', function (e) {
    const f = state.follow;
    if (!f.id) return;
    if (f.mode === 'glide') finishGlideNow();
    f.press = { x: e.clientX, y: e.clientY, moved: false };
    if (f.mode === 'chase') f.suspend = true;
    // 指针捕获：拖出画布/窗口再松手也能收到 pointerup；不支持时忽略。
    try { canvas.setPointerCapture(e.pointerId); } catch (err) { /* 老浏览器 */ }
    // 不 stopPropagation：ScreenSpaceCameraController 仍需收到事件
  }, true);

  window.addEventListener('pointermove', function (e) {
    const f = state.follow;
    const press = f.press;
    if (!press) return;
    if (Math.abs(e.clientX - press.x) + Math.abs(e.clientY - press.y) < DRAG_THRESHOLD_PX) return;
    press.moved = true;
  }, true);

  window.addEventListener('pointerup', function () {
    releaseFollowPress(false);
  }, true);

  // pointercancel（触摸被系统打断）与窗口失焦不会发 pointerup；没有兜底时
  // f.press 清不掉，chase 会永久停在 suspend。失焦按"拖拽结束"处理，回来后恢复。
  canvas.addEventListener('pointercancel', function () {
    releaseFollowPress(true);
  }, true);
  window.addEventListener('blur', function () {
    releaseFollowPress(true);
  }, true);
  canvas.addEventListener('lostpointercapture', function () {
    if (state.follow.press) releaseFollowPress(true);
  }, true);

  canvas.addEventListener('wheel', function (e) {
    const f = state.follow;
    if (!f.id) return;
    if (f.mode === 'glide') finishGlideNow();
    if (f.mode !== 'chase' && f.mode !== 'center') return;

    // 两种模式都由 rig 自己缩放，并拦下 Cesium 默认滚轮。Cesium 1.145 的
    // controller 在 3D 下**按光标位置**缩放：光标不在目标上时，目标会被推离
    // 视窗中心。这里一律按"相机→目标"的视线改变距离，目标永远留在中心。
    e.preventDefault();
    e.stopPropagation();
    // Cesium 的 controller 监听在同一个 canvas 上；stopPropagation 不会拦住
    // 同元素上的其它监听，必须 stopImmediatePropagation，否则两套缩放同帧打架。
    if (e.stopImmediatePropagation) e.stopImmediatePropagation();

    const dy = wheelPixels(e);
    if (!dy) return;
    const factor = Math.exp(dy * 0.0015);
    const pos = liveTargetPosition(f.id);
    if (!pos) return;

    if (f.mode === 'center') {
      // 保留用户当前方位/俯仰，只改到目标的距离；lookAt 的目标点不变，
      // 因此无论光标在屏幕哪里，缩放中心都是目标。
      const cur = hprFromCameraTo(pos);
      cur.range = clampZoomRange(cur.range * factor);
      lockAt(pos, cur);
      return;
    }

    const rec = state.targets[f.id];
    const v = CHASE_VIEW[(rec && rec.def.domain) || 'sea'] || CHASE_VIEW.sea;
    const base = f.chaseRange || f.pose.range || v.range;
    f.chaseRange = clampRange(base * factor);
    f.lastNow = 0;                    // 让阻尼以当前时刻重新计时
    // suspend（拖拽/松手后的停顿窗口）里 rig 走的是 center 分支，不会消费
    // chaseRange；这里立即按当前视线方向应用新距离，滚轮才是"所见即所得"。
    if (f.suspend) {
      const cur = hprFromCameraTo(pos);
      cur.range = f.chaseRange;
      lockAt(pos, cur);
    }
  }, { capture: true, passive: false });
}

/** 取证：目标在视窗中的位置、相机与目标的几何关系。供自检/验收断言。 */
function followDebug() {
  const f = state.follow;
  const out = {
    id: f.id, mode: f.mode, chase: !!f.enabled, suspend: !!f.suspend,
    press: !!f.press, resumeAt: f.resumeAt,
    chaseRange: f.chaseRange, pose: { heading: f.pose.heading, pitch: f.pose.pitch, range: f.pose.range },
    target: null, camera: null, screen: null, viewport: null,
    locked: !!(viewer && !viewer.camera.transform.equals(Cesium.Matrix4.IDENTITY)),
  };
  if (!viewer) return out;
  const cv = viewer.scene.canvas;
  out.viewport = { w: cv.clientWidth, h: cv.clientHeight };
  if (!f.id) return out;
  const s = positionFor(f.id, state.tick);
  const pos = liveTargetPosition(f.id);
  if (s) {
    out.target = { lon: s.lon, lat: s.lat, alt: s.alt, heading: s.heading };
  }
  if (!pos) return out;
  const carto = Cesium.Cartographic.fromCartesian(pos);
  const cam = viewer.camera.positionCartographic;
  out.camera = {
    lon: Cesium.Math.toDegrees(cam.longitude),
    lat: Cesium.Math.toDegrees(cam.latitude),
    height: cam.height,
    distance: Cesium.Cartesian3.distance(viewer.camera.positionWC, pos),
  };
  try {
    const st = Cesium.SceneTransforms.worldToWindowCoordinates
      || Cesium.SceneTransforms.wgs84ToWindowCoordinates;
    if (st) {
      const p = st(viewer.scene, pos);
      if (p) {
        out.screen = {
          x: p.x, y: p.y,
          dx: p.x - out.viewport.w / 2,
          dy: p.y - out.viewport.h / 2,
        };
      }
    }
  } catch (e) { /* 场景尚未渲染 */ }
  if (carto) {
    out.targetCarte = { height: carto.height };
    out.targetLive = {
      lon: Cesium.Math.toDegrees(carto.longitude),
      lat: Cesium.Math.toDegrees(carto.latitude),
      alt: carto.height,
    };
  }
  return out;
}

// 供 overview.js 等其他模块调用（它们拿不到本文件的闭包）
window.Follow = {
  start: function (id) { beginTargetView(id, { force: true }); },
  detach: detachFollow,
  isActive: function () { return !!state.follow.id; },
  mode: function () { return state.follow.mode; },
  debug: followDebug,
};

function renderTransport() {
  if (!state.data || !state.data.frames) {
    dom.timeline.max = '0';
    dom.timeline.value = '0';
    dom.tickLabel.textContent = '— / —';
    dom.clockLabel.textContent = '—';
    return;
  }
  const frames = state.data.frames;
  const frame = frames[Math.min(state.tick, frames.length - 1)];
  dom.timeline.max = String(frames.length - 1);
  dom.timeline.value = String(state.tick);
  dom.tickLabel.textContent = state.tick + ' / ' + (frames.length - 1);

  const start = new Date(state.data.start_time_utc);
  const t = new Date(start.getTime() + frame.t_seconds * 1000);
  dom.clockLabel.textContent = t.toISOString().replace('T', ' ').slice(0, 19) + 'Z';
}

function setTick(tick) {
  if (!state.data || !state.data.frames) { state.tick = 0; state.tickFloat = 0; return; }
  const max = state.data.frames.length - 1;
  state.tick = Math.max(0, Math.min(max, Math.round(tick)));
  state.tickFloat = state.tick;
  // 立刻把 CZML 时钟对到该帧：?tick / 时间轴 / 步进后紧接着的 selectTarget
  // 会读 entity.position 的当前时钟采样；等下一帧 loop 再同步的话，深链会先
  // 按旧时钟的位置开始过渡（可见的"先飞错点再回来"）。
  if (window.TargetMotion) TargetMotion.syncClock(state.tickFloat);
  markScene('setTick');
}

function setPlaying(on) {
  state.playing = on;
  document.getElementById('btnPlay').textContent = on ? '❚❚ 暂停' : '▶ 播放';
  document.getElementById('btnRun').textContent = on ? '推演中…' : 'TS自动推演';
}

function loop(nowMs) {
  const dt = lastFrameMs ? (nowMs - lastFrameMs) / 1000 : 0;
  lastFrameMs = nowMs;

  if (!viewer) { requestAnimationFrame(loop); return; }

  // 图层管理面板：**每帧比对签名，变了才重建**。放在帧循环里而不是定时器里，
  // 是为了让"加了标绘/切了案例"这类外部变化在同一帧就反映出来，不用等下一次轮询。
  if (window.LayerPanel) LayerPanel.tick();

  if (state.playing && state.data) {
    state.tickFloat += dt * state.ticksPerSecond;
    const max = state.data.frames.length - 1;
    if (state.tickFloat >= max) {
      state.tickFloat = max;
      setPlaying(false);
    }
    const next = Math.floor(state.tickFloat);
    if (next !== state.tick) {
      state.tick = next;
      updateAutoPlots(state.tick);
      renderTargetCounts();
      renderTransport();
      updateInferenceUi();
      // 播放推进也是一次状态变化：不标记的话画面会停在原地（只在拖动时间轴时更新）。
      markScene('playback-tick');
    }
  }

  // 运动/特效与时间轴同源：先把连续帧号写进 Cesium 时钟，CZML 采样再按该
  // 时刻插值；特效在同一帧读取同一个时钟。
  if (window.TargetMotion) TargetMotion.syncClock(state.tickFloat);
  if (window.TargetFx) TargetFx.frame(dt);
  if (window.TargetCombatFx) TargetCombatFx.frame(dt);

  if (RenderGovernor.take()) syncEntities();
  // 标签避让按时间节流（不受 RenderGovernor 的脏标记影响：相机微动也要重排）。
  declutterLabels(false);

  // 相机静止时 camera.changed 不会再触发，而它第一次触发时场景可能尚未渲染
  // （globe.pick 返回 null），所以这里兜底按秒刷新一次。
  if (nowMs - lastGeoRefresh > 1000) {
    lastGeoRefresh = nowMs;
    updateGeoReadout();
  }

  requestAnimationFrame(loop);
}

// --------------------------------------------------------------------------
// 导出
// --------------------------------------------------------------------------

/**
 * 用当前的 state.data 重建实体与面板。
 *
 * TS管理里增删目标后要调它：目标实体是按 data.targets 建的，改了数据必须重建，
 * 但不需要重新取数据，也不该走 applySituationData（那会重置帧号与机位）。
 */
function reloadSituation() {
  if (!state.data || !viewer) return;
  detachFollow('TS已更新');
  clearTargetSelection();
  viewer.entities.removeAll();
  redrawPlotsAfterClear();
  state.targets = {};
  state.hotspotEntities = [];
  state.occupationEntities = [];
  state.hotspotRows = [];
  state.routeEntities = [];
  state.selected = null;

  createHotspots();
  createRouteEntities();
  createTargetEntities();
  attachMotionAndFx(state.data);
  if (typeof applyTargetLod === "function") applyTargetLod(true);
  // 编成可能被 TS 管理改过：鸟瞰图的目标图元与取景要跟着重建。
  if (window.Overview && Overview.setData) Overview.setData(state.data);
  // 面板/战损缓存按"案例+目标数"命中；编成改了但数量没变时也要重建。
  if (window.StatusPanels && StatusPanels._resetCache) StatusPanels._resetCache();
  if (window.TargetStatus && TargetStatus.reset) TargetStatus.reset();
  updateOccupationAreas();
  renderHotspotTable();
  renderTargetCounts();
  renderTransport();
  updateInferenceUi();
  setStatus('TS已更新 · ' + state.data.targets.length + ' 个目标', 'ok');
}

/** 全帧 CSV：逐帧逐目标展开，便于在表格软件里做轨迹分析。 */
function exportCsv() {
  if (!state.data || !state.data.frames) {
    setStatus('未加载TS，无可导出数据', 'err');
    return;
  }
  const d = state.data;
  const ids = d.targets.map(function (t) { return t.id; });
  const head = ['tick', 't_seconds'];
  ids.forEach(function (id) {
    head.push(id + '_lon', id + '_lat', id + '_alt', id + '_heading', id + '_speed_kn');
  });
  const rows = [head.join(',')];
  d.frames.forEach(function (f) {
    const byId = {};
    f.targets.forEach(function (t) { byId[t.id] = t; });
    const row = [f.tick, f.t_seconds];
    ids.forEach(function (id) {
      const t = byId[id];
      row.push(t ? t.lon : '', t ? t.lat : '', t ? t.alt : '',
               t ? t.heading : '', t ? t.speed_kn : '');
    });
    rows.push(row.join(','));
  });
  const blob = new Blob(['\ufeff' + rows.join('\n')], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = (d.case_id || 'situation') + '-frames.csv';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(function () { URL.revokeObjectURL(url); }, 2000);
  setStatus('已导出全帧 CSV（' + (d.frames.length - 1) + ' 帧 × ' + ids.length + ' 目标）', 'ok');
}

// TS模块（src/situation.js）拿不到本文件的闭包，用这个接口回调。
window.SituationHost = {
  state: state,
  // I08 取证出口：统一事件模型的字段、队列、过滤与导出。
  // 事件是"看得见的问题"，故必须能被外部（自检/探针）读到，而不是只画在界面上。
  eventsDebug: function () { return window.EventFeed ? window.EventFeed.debug() : null; },
  eventsFilter: function (opts) { return window.EventFeed ? window.EventFeed.filter(opts) : []; },
  eventsExportCSV: function (opts) { return window.EventFeed ? window.EventFeed.exportCSV(opts) : ""; },
  eventsExportJSON: function (opts) { return window.EventFeed ? window.EventFeed.exportJSON(opts) : "{}"; },
  eventsPush: function (e) { return window.EventFeed ? window.EventFeed.push(e) : null; },
  relationsDebug: relationsDebug,
  selectTarget: function (id, fly) { selectTarget(id || null, !!fly); },
  viewer: function () { return viewer; },
  // 运动/特效取证：CZML 采样数、时钟、尾流/尾气/轮角状态。
  motionDebug: function () { return window.TargetMotion ? TargetMotion.stats() : null; },
  fxDebug: function () { return window.TargetFx ? TargetFx.debug() : null; },
  // 取证用：建筑体块的短路条件逐条报出来。`?buildings=1` 不生效时，
  // "到底卡在哪个 return"光看结果猜不出来。
  // 注意这里**从 DOM 现取**复选框，不用 dom 缓存：实测两者会不一致
  // （DOM 上是 true、dom.modelBuildings.checked 是 false），拿缓存的引用去判断
  // "开关开没开"会把已经打开的开关读成关的。
  debugBuildings: function () {
    // ⚠ `checked` 必须走 `checkboxOn()`，**不能**直接读 `getElementById('modelBuildings')`：
    // 那个复选框已经随面板迁移从 HTML 删除，节点永远为 null，于是这个诊断字段
    // 恒为 false —— 实测"`?buildings=1` 已经生效（注册表 desired=true、effective=true）"
    // 时它仍报 false，会把人引向错误的方向。`checkboxOn()` 会在没有节点时反查注册表。
    const aoiIdx = nearestAoiToCamera();
    return {
      checked: checkboxOn('modelBuildings'),
      checkedViaDomCache: !!(dom.modelBuildings && dom.modelBuildings.checked),
      cachedNodeIsLive: !!(dom.modelBuildings && dom.modelBuildings.isConnected),
      rampLoaded: !!state.ramp,
      rampSource: state.ramp ? String(state.ramp.source || '').slice(0, 60) : null,
      buildingBusy: !!state.buildingBusy,
      aoiCount: (state.urbanAoi || []).length,
      activeHotspot: state.activeHotspot,
      hasData: !!state.data,
      nearestAoi: aoiIdx,
      nearestAoiName: aoiIdx >= 0 ? state.urbanAoi[aoiIdx].name : null,
      hotspotAoi: pickAoiForHotspot(state.activeHotspot),
      cameraHeightKm: +(viewer.camera.positionCartographic.height / 1000).toFixed(2),
      lod: OfflineAssets.lodForHeight(viewer.camera.positionCartographic.height),
      buildingsKey: state.buildingsKey,
      cacheKeys: Object.keys(state.buildingsCache || {}),
      hasPrimitive: !!state.buildingsPrimitive,
      // 体块统计**结构化**报出来。不要靠 switchNote 里的中文抠数字：那条提示会被
      // 在线层回落/关闭等别的分支覆盖，从文案里正则取数非常脆（本轮就吃过这个亏）。
      buildingsStats: state.lastBuildingsStats || null,
      pendingPrimitive: !!state.pendingPrimitive,
      switchNote: dom.switchNote ? dom.switchNote.textContent : null,
      // 在线实景那条链路（默认图层）的取证：断网/无令牌/令牌过期都只能从
      // phase 与 detail 分辨，光看画面是"没有建筑"还是"建筑没加载出来"分不清。
      online: window.Buildings3D ? Buildings3D.debug() : null,
      // 本地 3D Tiles 的取证：每个数据集是否挂载、已加载瓦片数、失败原因。
      local3d: window.Local3D ? Local3D.debug() : null,
    };
  },
  setStatus: setStatus,
  // 标绘绘制模式是否激活。**必须能读到**：绘制模式下左键被改成"落控制点"，
  // 若模式退不出去，用户就再也选不中目标了；而这件事从界面上只能间接看出来
  // （按钮文案变回来），从外部断言不了。
  plotDrawActive: function () { return !!plotDraw; },
  // 图层顺序取证：imagery 的真实 index 顺序 + 每个图层的意图/事实。
  // 「哪个层压在哪个层上面」光看画面猜不出来，必须能读到 index。
  debugLayers: debugLayers,
  /**
   * 面状要素的 GeoJSON 导出：用户标绘（Plotting）+ 当前帧热点 + 占领区。
   * 所有面都是 Polygon（WGS84），可直接在 QGIS 等工具里打开。
   */
  plotFeatureCollection: function () {
    const fc = (window.Plotting && Plotting.toFeatureCollection)
      ? Plotting.toFeatureCollection()
      : { type: 'FeatureCollection', features: [] };
    (state.hotspotRows || []).forEach(function (h) {
      const e = (state.hotspotEntities || []).filter(function (en) {
        return en.hsData && en.hsData.name === h.name;
      })[0];
      if (e && e.hsData.feature) fc.features.push(e.hsData.feature);
    });
    // 占领区域已下线，不再进入导出。
    return fc;
  },
  /** GeoJSON 导入（用户标绘）。返回 {added, errors}。 */
  importPlotFeatures: function (fc) {
    return (window.Plotting && Plotting.fromFeatureCollection)
      ? Plotting.fromFeatureCollection(fc)
      : { added: 0, errors: ['Plotting 未加载'] };
  },
  /** 标签避让取证：本轮参与判定/显示/隐藏的数量。 */
  declutterDebug: function () { return declutterLabels(true); },
  /** 动态热点/占领区的取证出口：计算值 vs 场景图元数（数据集可能有/无威胁数据）。 */
  dynamicAreasDebug: function () {
    const spots = computeHotspots();
    return {
      hotspots: spots.length,
      hotspotEntities: (state.hotspotEntities || []).length,
      hotspotRows: (state.hotspotRows || []).length,
      // 占领区域已下线（保留字段，恒为 0/enabled:false，避免旧自检报错）。
      occupation: 0,
      occupationEnabled: false,
      occupationEntities: 0,
      occupationVisible: 0,
      threatAvailable: Object.keys(state.targets).some(function (id) {
        return threatOf(id).threat > 0;
      }),
      errors: window.__pageErrors || [],
    };
  },
  // 动态标绘图层取证：注册表里实际存在的标绘类别图层 vs 要素所属图层。
  // "图层由实际要素驱动"不能靠肉眼判断，必须能读出两侧的集合。
  debugPlotLayers: function () {
    var items = (window.Plotting ? Plotting.debug().items : []);
    var used = {};
    items.forEach(function (it) { used[it.layer] = (used[it.layer] || 0) + 1; });
    return {
      categories: (window.Plotting && Plotting.categories) ? Plotting.categories() : [],
      registered: LayerRegistry.list('plot').filter(function (l) { return l.group === '标绘'; })
        .map(function (l) {
          return { id: l.id, label: l.label, z: l.z,
                   count: LayerRegistry.elements(l.id).length,
                   desired: !!LayerRegistry.runtime[l.id].desired };
        }),
      used: used,
      items: items.map(function (it) {
        return { id: it.id, type: it.type, layer: it.layer, system: it.system };
      }),
    };
  },
  // 图层自动定位：取证/测试入口（面板开关是唯一会触发飞行的用户路径）。
  focusLayer: focusLayer,
  layerExtent: layerExtent,
  layerFlight: function () { return { id: layerFlight.id, token: layerFlight.token }; },
  // 预加载的取证出口：配置、候选 URL 与缓存侧统计。
  // 没有它就只能靠"感觉快了"，而本 feature 的验收要求给数（请求数/耗时/帧时）。
  prefetchConfig: function () {
    return window.TileCache && TileCache.prefetchConfig ? TileCache.prefetchConfig() : null;
  },
  prefetchCandidates: function () { return collectPrefetchUrls(); },
  prefetchStats: function () {
    return window.TileCache ? TileCache.snapshot() : null;
  },
  // 数据健康诊断的出口。左栏「数据健康」面板已随界面重构移除，但诊断结论仍需可查：
  // 否则"开关开着、图层没生效"就失去了唯一的证据出口。
  health: healthReport,
  // 指南针实测采集：把"指针抖动"从主观描述变成可量的数。
  // 关键是把**角度变化**与**元素位移**分开统计：
  //   maxWrittenStep 大 → 写入的旋转角在跳（指南针自身的角度问题）
  //   maxRawStep 大而 maxWrittenStep 小 → 原始航向在跳，但连续化把它压住了
  //   maxElementMove 大 → 抖动的是整个指南针的**位置**（布局/重排问题），
  //                      与旋转无关，此前几轮一直在错的维度上找
  compassTrace: function () { return COMPASS_TRACE.slice(); },
  compassTraceStats: function () {
    let maxWritten = 0, maxRawWrap = 0, maxRawNorm = 0, maxEl = 0, maxHdg = 0, maxResize = 0;
    for (let i = 1; i < COMPASS_TRACE.length; i++) {
      const a = COMPASS_TRACE[i], b = COMPASS_TRACE[i - 1];
      if (a.t - b.t > 1500) continue;              // 跨过长空档（例如切案例）不算跳变
      maxWritten = Math.max(maxWritten, Math.abs(a.written - b.written));
      // raw 的**未归一化**差值：±180 处的锯齿会在这里显示成约 360，
      // 这正是"直接写 atan2 结果"会让 CSS 多转一圈的证据。
      maxRawWrap = Math.max(maxRawWrap, Math.abs(a.raw - b.raw));
      let d = a.raw - b.raw;
      d = ((d + 540) % 360) - 180;
      maxRawNorm = Math.max(maxRawNorm, Math.abs(d));
      maxHdg = Math.max(maxHdg, Math.abs(a.hdg - b.hdg));
      maxEl = Math.max(maxEl, Math.abs(a.elLeft - b.elLeft), Math.abs(a.elTop - b.elTop));
      maxResize = Math.max(maxResize, Math.abs(a.elW - b.elW), Math.abs(a.elH - b.elH));
    }
    const r2 = function (v) { return Math.round(v * 100) / 100; };
    return {
      samples: COMPASS_TRACE.length,
      maxWrittenStepDeg: r2(maxWritten),
      maxRawWrapStepDeg: r2(maxRawWrap),
      maxRawNormStepDeg: r2(maxRawNorm),
      maxCameraHeadingStepDeg: r2(maxHdg),
      maxElementMovePx: maxEl,
      maxElementResizePx: maxResize,
    };
  },
  compassTraceReset: function () { COMPASS_TRACE.length = 0; _compassWrittenDeg = null; },
  // 界面骨架的取证入口：各面板与地图组件的实际几何位置。
  // 组件重排（鸟瞰图右上、指南针其下、比例尺左下、视点底部居中）靠截图判断不可靠
  // ——本项目已确认 --shot 会给出相同图。这里输出可断言的矩形，让验收有据可依。
  uiLayout: function () {
    function rect(id) {
      const el = document.getElementById(id);
      if (!el) return null;
      const r = el.getBoundingClientRect();
      const hidden = el.classList.contains('hidden') || r.width === 0;
      return {
        hidden: hidden,
        left: Math.round(r.left), top: Math.round(r.top),
        right: Math.round(r.right), bottom: Math.round(r.bottom),
        width: Math.round(r.width), height: Math.round(r.height),
        centerX: Math.round((r.left + r.right) / 2),
      };
    }
    const wrap = document.getElementById('cesiumContainer');
    return {
      viewport: { w: window.innerWidth, h: window.innerHeight },
      globe: wrap ? rect('cesiumContainer') : null,
      overview: rect('overviewWrap'),
      compass: rect('mapCompass'),
      locate: rect('btnLocate'),
      scalebar: rect('mapScalebar'),
      viewpoint: rect('mapViewpoint'),
      legend: rect('legend'),
      hint: rect('hint'),
      panels: {
        left: rect('layerGroups') || (function () {
          const el = document.querySelector('.layer-groups');
          if (!el) return null;
          const r = el.getBoundingClientRect();
          return { height: Math.round(r.height), width: Math.round(r.width) };
        })(),
        right: (function () {
          const el = document.querySelector('.panel-right');
          if (!el) return null;
          const r = el.getBoundingClientRect();
          return { left: Math.round(r.left), width: Math.round(r.width),
                   height: Math.round(r.height) };
        })(),
      },
    };
  },
  // 热点层选框结果：declared=案例声明的热点数，boxes=manifest 里的框数，
  // matched=按坐标匹配上的数量，fallback=true 表示没匹配上而全挂（战区未烘制时如此）。
  // 没有这个读数时，"热点层挂错战区"与"该战区还没烘制"看起来完全一样。
  facilityDebug: hotspotFacilities,
    hotspotMatch: function () { return state.hotspotMatch || null; },
  // 意图链路取证：本体是否加载、当前帧每目标的意图集合与状态、判不出的原因。
  debugIntent: debugIntent,
  applyLayers: applyLayers,
  // 场景文档/深链：取证与测试入口。
  applySceneDoc: applySceneDoc,
  shareScene: shareScene,
  sceneProviders: applySceneProviders,
  /** 瓦片请求统计，给数据健康面板用（state 在闭包里，外面拿不到）。 */
  imageryStats: function () { return state.imageryStats; },
  /** 数据健康面板要用到的其它 state 片段。 */
  urbanInfo: function () {
    return {
      enabled: state.urbanEnabled,
      mounted: !!state.urbanLayers,
      count: (state.urbanLayers || []).length,
      aoiCount: (state.urbanAoi || []).length,
    };
  },
  applySituationData: applySituationData,
  reload: reloadSituation,
  clearSituation: clearSituation,
  exportFrame: exportData,
  exportCsv: exportCsv,
  // 选中入口（与点选/面板"定位"走**同一条路径**）：自检自动化要选中/取消选中时用它。
  // 另抄一份状态改动就会漏掉 markScene 与掩码重算 —— 那正是本仓修过 4 起的老毛病。
  selectTarget: function (id, fly, height) {
    selectTarget(id || null, !!fly, height);
    return state.selected;
  },
  // 轨迹窗口 / 目标轨迹标绘 / 掩码的取证读数：外部（自检、验收）只认这些数字，
  // "画得对不对"不能靠肉眼。
  focusDebug: focusDebug,
  followDebug: followDebug,
  // 取证/自检用：手动推进一帧相机（now 可注入虚拟时间）。正常渲染时由
  // scene.preRender 每帧调用；无头浏览器帧循环停摆时，自检用它在明确的
  // 时间步上验证阻尼/过渡，而不是靠 sleep 等帧。
  followStep: function (now) {
    applyCameraRig(typeof now === 'number' ? now : performance.now());
    return followDebug();
  },
  occupationDebug: occupationDebug,
};

function exportData() {
  if (!state.data || !state.data.frames) {
    setStatus('未加载TS，无可导出数据', 'err');
    return;
  }
  const frames = state.data.frames;
  const inf = inferenceAt(state.tick);
  const payload = {
    schema: state.data.schema,
    case_id: state.data.case_id,
    case_name: state.data.case_name,
    scenario_id: state.data.scenario_id,
    region: state.data.region,
    provenance: state.data.provenance,
    tick_seconds: state.data.tick_seconds,
    exported_tick: state.tick,
    tick_count: frames.length - 1,
    inference_model: state.data.inference_model,
    targets: state.data.targets.map(function (d) {
      const s = positionFor(d.id, state.tick);
      const l1 = inf ? inf.targets[d.id] : null;
      return {
        id: d.id,
        name: d.name,
        model: d.model,
        platform_class: d.platform_class,
        domain: d.domain,
        affiliation: d.affiliation,
        icon: d.icon,
        lat: s ? s.lat : null,
        lon: s ? s.lon : null,
        alt_m: s ? s.alt : null,
        speed_kn: s ? s.speed_kn : null,
        heading_deg: s ? s.heading : null,
        inference_l1: l1,
      };
    }),
    groups: inf ? inf.groups : [],
    regions: inf ? state.data.hotspots.map(function (h, i) {
      return { name: h.name, bbox: h.bbox, inference: inf.regions[i] };
    }) : [],
    overall: inf ? inf.overall : null,
  };

  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'situation_' + state.data.case_id + '_tick' + state.tick + '.json';
  a.click();
  URL.revokeObjectURL(a.href);
}

// --------------------------------------------------------------------------
// 事件绑定
// --------------------------------------------------------------------------

// --------------------------------------------------------------------------
// 动态标绘图层：由**实际存在的标绘要素**驱动
// --------------------------------------------------------------------------
//
// 注册表里不再预置固定的标绘类别（原先写死 plot.mobility/offense/defense/fire/intel）。
// 现在：
//   · Plotting 每次 add/remove/update/import/clear 后通过 onChange 通知这里；
//   · 这里按 `Plotting.debug().items` 里出现的 layer 集合，补注册缺失的类别图层、
//     注销已经没有要素的类别图层；
//   · 图层名称/顺序来自 `Plotting.categories()`（CATEGORIES 元数据），新增类别
//     不需要改注册表。
//
// 只有"图层的集合"变化时才动注册表；同一图层内要素增删只让面板按数量徽标自然
// 刷新（layerPanel.tick 的签名里带了 count），避免每次描点都重建图层。

// 图层会因"最后一个要素被删"而注销；用户对某个类别的显隐意图要跨注销保留，
// 否则重画一条同类标绘时，用户关掉的层会自己弹回来。
var _plotLayerDesired = {};

/** 标绘类型的中文名（catalogue 是静态目录，现取即可，不依赖初始化顺序）。 */
function plotTypeZh(type) {
  var zh = type;
  if (window.Plotting) {
    Plotting.catalogue().forEach(function (c) { if (c.type === type) zh = c.zh; });
  }
  return zh;
}

/** 标绘类别下拉项：`[['plot.offense','进攻'], …]`（创建/改归属共用）。 */
function plotLayerOptions() {
  var out = [];
  if (!window.Plotting || !Plotting.categories) return out;
  // 只由系统类型组成的类别（目标轨迹）不是手动标绘的归属选项
  var anyManual = {};
  Plotting.catalogue().forEach(function (c) {
    if (!c.system) anyManual[c.category] = true;
  });
  Plotting.categories().forEach(function (c) {
    if (!anyManual[c.id]) return;
    out.push([c.layer, c.zh]);
  });
  return out;
}

/**
 * 给某个标绘类别图层注册要素提供者。
 *
 * 标绘可**删除、可更改**。更改允许动控制点/标注/归属/归属图层；控制点在表单里按
 * "每行 经度, 纬度"编辑 —— `update()` 存的是控制点本身，不是派生出来的环/线。
 * 系统生成的目标轨迹（`it.system`）是只读要素：只给「定位」，
 * 不给删除/改属性/单要素开关（否则改完下一个 tick 就被系统覆盖）。
 */
function registerPlotLayerProvider(layerId) {
  if (!window.LayerRegistry || !window.Plotting) return;
  var touch = function () { if (window.LayerPanel) LayerPanel.invalidate(); };
  LayerRegistry.setProvider(layerId, function () {
    return Plotting.debug().items.filter(function (it) { return it.layer === layerId; })
      .map(function (it) {
        var zh = plotTypeZh(it.type);
        if (it.system) {
          return {
            key: it.id,
            label: it.label || zh,
            sub: zh + ' · 系统生成',
            visible: it.visible !== false,
            actions: ['locate'],
            onLocate: function () { Plotting.focus(it.id); },
          };
        }
        return {
          key: it.id,
          label: it.label || zh,
          sub: zh + ' · ' + it.points + ' 点',
          visible: it.visible !== false,
          actions: ['toggle', 'locate', 'edit', 'delete'],
          onToggle: function (on) { Plotting.setVisible(it.id, on); touch(); },
          onLocate: function () { Plotting.focus(it.id); },
          onDelete: function () { Plotting.remove(it.id); touch(); },
          editFields: [
            { key: 'label', label: '标注', value: it.label || '' },
            { key: 'hostility', label: '归属', value: it.hostility || 'friendly',
              options: [['friendly', '红方（我军）'], ['enemy', '蓝方（敌军）']] },
            { key: 'layer', label: '归属图层', value: it.layer,
              options: plotLayerOptions() },
            { key: 'points', label: '控制点', value: '', type: 'text' },
          ],
          onSave: function (v) {
            var patch = { label: v.label, hostility: v.hostility, layer: v.layer };
            if (v.points && v.points.trim()) {
              var pts = v.points.split(/[;\n]+/).map(function (s) {
                var a = s.split(/[,\s]+/).filter(Boolean).map(Number);
                return (a.length === 2 && a.every(isFinite)) ? a : null;
              }).filter(Boolean);
              if (!pts.length) throw new Error('控制点格式应为「每行 经度, 纬度」');
              patch.points = pts;
            }
            Plotting.update(it.id, patch);
            touch();
          },
        };
      });
  });
}

/** 确保某个标绘类别图层已注册（按首次出现自动成层）。 */
function ensurePlotLayer(layerId) {
  if (!window.LayerRegistry || !window.Plotting) return;
  var def = LayerRegistry.def(layerId);
  if (def) {
    if (def.group !== '标绘') {
      // 程序化调用把标绘写进 relations/occupation 这类同 kind 但不同用途的图层：
      // 不接管，显式说出来（导入路径已被 situation.js 的校验挡住）。
      if (window.console && console.warn) {
        console.warn('[plot] 标绘要素的归属图层不是标绘类别层：' + layerId);
      }
      return;
    }
    // 已存在：只补可能缺失的 provider（外部注销过 provider 时也能自愈）。
    if (!LayerRegistry.hasProvider(layerId)) registerPlotLayerProvider(layerId);
    return;
  }
  var cat = Plotting.categoryOfLayer(layerId);
  LayerRegistry.register({
    id: layerId, kind: 'plot', z: cat.z,
    label: cat.zh, group: '标绘', default: true,
  });
  // register() 的 runtime 初始 desired=false。新类别默认可见，但**用户之前关掉过的
  // 类别要恢复为关**（图层会因清空而注销，显隐意图不能跟着丢）。
  LayerRegistry.setDesired(layerId, _plotLayerDesired[layerId] !== false);
  registerPlotLayerProvider(layerId);
}

/**
 * 把"实际存在哪些标绘图层"同步到注册表。
 *
 * 每次都比较**两个集合**（要素所属图层 vs 已注册的标绘类别图层），而不是靠一个
 * 提前写入的签名短路：外部 `LayerRegistry.unregister()` 或中途异常之后，下一次
 * 同步仍能把缺失的图层补回来。集合没变时只补 provider，不做重排。
 */
function syncPlotLayers() {
  if (!window.LayerRegistry || !window.Plotting) return null;
  var used = {};
  Plotting.debug().items.forEach(function (it) { used[it.layer] = true; });
  // Keep every declared plot category registered, even when it currently has
  // no item: the user expects 【TS图层】→【标绘】 to list 自动/手动 plot entries
  // as manageable rows instead of the whole group disappearing.
  Plotting.categories().forEach(function (c) { used[c.layer] = true; });
  var usedIds = Object.keys(used).sort();
  var registered = LayerRegistry.list('plot').filter(function (l) { return l.group === '标绘'; })
    .map(function (l) { return l.id; }).sort();
  var missing = usedIds.filter(function (id) { return registered.indexOf(id) < 0; });
  var extra = registered.filter(function (id) { return usedIds.indexOf(id) < 0; });

  if (!missing.length && !extra.length) {
    // 集合一致：只补可能丢失的 provider，不重排 imagery / 不重建面板
    usedIds.forEach(function (id) {
      if (!LayerRegistry.hasProvider(id)) registerPlotLayerProvider(id);
    });
    return null;
  }

  usedIds.forEach(ensurePlotLayer);
  extra.forEach(function (id) { LayerRegistry.unregister(id); });
  refreshPlotLayerSelect();
  applyLayers();
  if (window.LayerPanel) LayerPanel.invalidate();
  return { layers: usedIds };
}

/**
 * 注册各图层的**要素提供者** —— 动态图层管理的数据源。
 *
 * 面板只认一个契约：`() => [{key, label, sub, visible, actions, ...回调}]`。
 * 提供者把"这一层现在有哪些要素"翻译成它，面板负责画出来。
 * 于是**新增一类可列要素只需要加一个提供者**，不用碰面板代码。
 *
 * 每个提供者都要能在数据未加载时返回 `null`（空态）：面板把它显示成"没有要素列表"，
 * 而不是 0 —— 那是两件不同的事。
 */
function registerLayerProviders() {
  if (!window.LayerRegistry) return;

  var AFF_ZH = { red: '红方', blue: '蓝方' };
  var DOM_ZH = { sea: '水面', land: '陆上', air: '空中' };

  function touch() { if (window.LayerPanel) LayerPanel.invalidate(); }

  // ---- 标绘：提供者按**实际存在的类别图层**注册 --------------------------
  //
  // 标绘图层不是启动时写死的五行，而是由实际要素动态成层（见 syncPlotLayers）。
  // 这里只给当前已经存在的标绘图层补齐提供者；新图层创建时由
  // registerPlotLayerProvider() 补上。
  if (window.Plotting) {
    LayerRegistry.list('plot').forEach(function (d) {
      if (d.group === '标绘') registerPlotLayerProvider(d.id);
    });
  }

  // ---- 目标：红方 / 蓝方两个图层，各列出目标型号 ---------------------------
  //
  // 需求：可见目标按红/蓝拆成两行；列表里优先显示**型号**（platform_class），
  // 而不是 "R01 红方驱逐舰编队指挥舰" 这种以编号开头的长名。
  // 这里只改**图层里的展示名**：`t.id` 与 `t.name` 都不动，面板编辑仍按原字段走。
  function targetModelName(t, ov) {
    return (ov.model !== undefined ? ov.model : t.model)
      || t.platform_class || (AFF_ZH[t.affiliation] || '') || t.id;
  }

  function targetItem(t) {
    var ov = state.targetOverrides[t.id] || {};
    var def = { id: t.id, name: ov.name || t.name,
                affiliation: ov.affiliation || t.affiliation,
                domain: ov.domain || t.domain };
    var model = targetModelName(t, ov);
    var side = AFF_ZH[def.affiliation] || def.affiliation;
    return {
      key: t.id,
      // 展示名 = 型号；编号放在副标题里（仍然能找到唯一目标）。
      label: model,
      // 阵营已是图层名（红方目标/蓝方目标），副标题不再重复它。
      sub: t.id + ' · ' + (DOM_ZH[def.domain] || def.domain),
      visible: isVisible(t),
      actions: ['toggle', 'locate', 'edit', 'delete'],
      onToggle: function (on) {
        if (on) delete state.hiddenTargets[t.id];
        else state.hiddenTargets[t.id] = true;
        markScene('singleTarget');
        touch();
      },
      onLocate: function () { selectTarget(t.id, true, 60000); },
      onDelete: function () {
        // **从当前 TS 移除**：会话内状态。数据集文件不动，重新加载案例即恢复。
        state.removedTargets[t.id] = true;
        if (state.selected === t.id) selectTarget(null, false);
        markScene('removeTarget');
        touch();
      },
      editFields: [
        { key: 'name', label: '名称', value: def.name },
        { key: 'affiliation', label: '阵营', value: def.affiliation,
          options: [['red', '红方'], ['blue', '蓝方']] },
        { key: 'domain', label: '域', value: def.domain,
          options: [['sea', '水面'], ['land', '陆上'], ['air', '空中']] },
        { key: 'model', label: '型号', value: (ov.model !== undefined ? ov.model : t.model) || '' },
      ],
      onSave: function (v) {
        // 只改 **TS 信息**。几何与航迹来自数据集帧，改它们等于改数据 ——
        // 你明确要求目标要素"只有其 TS 信息可更改"。
        state.targetOverrides[t.id] = {
          name: v.name, affiliation: v.affiliation, domain: v.domain, model: v.model,
        };
        applyTargetOverrides();
        markScene('targetOverride');
        renderTargetStatus(state.selected);
        touch();
      },
    };
  }

  ['layerTargetsRed', 'layerTargetsBlue'].forEach(function (lid) {
    LayerRegistry.setProvider(lid, function () {
      var d = state.data;
      if (!d || !d.targets) return null;
      var side = lid === 'layerTargetsRed' ? 'red' : 'blue';
      return d.targets
        .filter(function (t) {
          if (state.removedTargets[t.id]) return false;
          var ov = state.targetOverrides[t.id] || {};
          return (ov.affiliation || t.affiliation) === side;
        })
        .map(targetItem);
    });
  });

  // ---- 计划航路：每条航路对应一个目标 -------------------------------------
  LayerRegistry.setProvider('layerRoutes', function () {
    var d = state.data;
    if (!d || !d.targets) return null;
    return d.targets
      .filter(function (t) { return !state.removedTargets[t.id]; })
      .map(function (t) {
        return {
          key: t.id,
          label: '航路 ' + t.id,
          sub: DOM_ZH[t.domain] || t.domain,
          visible: !!state.layers.routes && !state.hiddenRoutes[t.id]
                   && !state.removedTargets[t.id],
          actions: ['toggle', 'locate'],
          onToggle: function (on) {
            if (on) delete state.hiddenRoutes[t.id]; else state.hiddenRoutes[t.id] = true;
            markScene('singleRoute');
            touch();
          },
          onLocate: function () { selectTarget(t.id, false); },
        };
      });
  });

  // ---- 热点区域 -----------------------------------------------------------
  LayerRegistry.setProvider('layerHotspots', function () {
    var d = state.data;
    if (!d) return null;
    // 热点改为按战况动态生成：列表直接反映**当前帧**的热点（交火/对峙聚集区）。
    return (state.hotspotRows || []).map(function (h, i) {
      return {
        key: 'hotspot-' + i,
        label: h.name || ('热点 ' + (i + 1)),
        // 名称本身就是「交火区/对峙区」，副标题只补范围，不再重复等级。
        sub: h.radiusKm ? '半径 ' + Math.round(h.radiusKm) + ' km' : '',
        visible: !!state.layers.hotspots && !state.hiddenHotspots[i],
        actions: ['toggle', 'locate'],
        onToggle: function (on) {
          if (on) delete state.hiddenHotspots[i]; else state.hiddenHotspots[i] = true;
          markScene('singleHotspot');
          touch();
        },
        onLocate: function () {
          var e = state.hotspotEntities[i];
          if (!e || !e.hsData) return;
          flyTo(e.hsData.center[1], e.hsData.center[0], e.hsData.radiusKm * 3000);
        },
      };
    });
  });

  // ---- 地名：只读统计 -----------------------------------------------------
  // 不逐个列出候选词条：地名是**按相机高度分级 + 屏幕空间去重**后动态增删的
  // （`Places` 内部有一个对象池），逐条列出来的话数量每帧都在跳，既没用又费。
  // 这里给"加载了多少、当前高度下有多少候选、实际放了多少个"三个数。
  LayerRegistry.setProvider('places.labels', function () {
    if (!window.Places || typeof Places.debugPlaces !== 'function') return null;
    var d = Places.debugPlaces();
    if (!d || !d.loaded) return null;
    return [{
      key: 'places-summary',
      label: '离线地名',
      sub: '候选 ' + (d.candidates || 0) + ' / 已放置 ' + (d.shown || 0) + ' / 共 ' + (d.total || 0),
      visible: !!state.layers.labels,
      actions: [],
    }];
  });

  // ---- 建筑：只读统计 -----------------------------------------------------
  //
  // 两个图层来自**两套不同实现**，取证入口也不一样，不能共用一个：
  //   buildings.blocks  离线体块，由 assets.js 挤出后挂在 `state.buildingsPrimitive`
  //   tiles.google3d    在线实景，状态在 `Buildings3D.debug()`（注意它**没有**
  //                     blocksLive/onlineLive 这类字段，实测返回的是
  //                     enabled/phase/hasTileset/tilesetReady/tilesetShow/primitiveCount）
  LayerRegistry.setProvider('buildings.blocks', function () {
    // 计数取自 `state.lastBuildingsStats.blocks` —— 那是代码自己为取证留的结构化统计
    // （见 refreshBuildings 末尾的注释："不再从中文提示里抠数字"）。
    //
    // ⚠ 别从 `state.buildingsPrimitive.geometryInstances` 去数：实测该属性是
    // **undefined**（体块不是按 geometryInstances 建的），于是 provider 永远返回 null，
    // 面板上这一行的数量**恒为 0** —— 而体块其实已经建出来了（`hasPrimitive: true`）。
    // 一个"看起来是空的"图层比没有这一行更糟。
    var st = state.lastBuildingsStats;
    if (!st) return null;
    return [{
      key: 'offline-blocks',
      label: '离线建筑体块',
      sub: (st.aoi || '—') + ' · LOD ' + st.lod + ' · 网格 ' + st.grid + '×' + st.grid
           + ' · ' + st.blocks + ' 个体块' + (st.fromCache ? '（缓存）' : ''),
      visible: !!(state.buildingsPrimitive && state.buildingsPrimitive.show),
      actions: [],
    }];
  });
  LayerRegistry.setProvider('tiles.google3d', function () {
    if (!window.Buildings3D || typeof Buildings3D.debug !== 'function') return null;
    var d = Buildings3D.debug();
    if (!d || !d.hasTileset) return null;
    return [{
      key: 'online-tileset',
      label: '在线三维实景瓦片集',
      sub: '就绪 ' + (d.tilesetReady ? '是' : '否') + ' / 显示 ' + (d.tilesetShow ? '是' : '否')
           + ' / globe ' + (d.globeShowing ? '开' : '关'),
      visible: !!d.tilesetShow,
      actions: [],
    }];
  });

  // ---- 台湾沿海城市建筑白膜：15 个 AOI 子条目（只读状态 + 定位）----------
  // 数据：tiles3d_db/tw_coastal_v2.zip（根 tileset 的 children 即这 15 个子 tileset）。
  // 条目只做两件事：显示该 AOI 的栋数/挂载状态；点「定位」飞过去（城市级视距，
  // 白膜在 ~50 km 内才会加载叶子瓦片，见根/子 tileset 的 geometricError 设计）。
  // 不做逐 AOI 显隐：Cesium3DTileset 不支持对子 tileset 单独 show，按 AOI 拆 15 个
  // tileset 实例的复杂度不值得，先给定位入口。
  const TW_COASTAL_AOIS = [
    { key: 'keelung',        label: '基隆',          buildings: 9302,  bbox: [121.700, 25.105, 121.780, 25.175] },
    { key: 'taipei_port',    label: '台北港（八里）', buildings: 2321,   bbox: [121.330, 25.125, 121.420, 25.200] },
    { key: 'taoyuan_coast',  label: '桃园海岸',      buildings: 11718,  bbox: [121.080, 25.010, 121.220, 25.120] },
    { key: 'hsinchu',        label: '新竹',          buildings: 16252,  bbox: [120.940, 24.770, 121.030, 24.850] },
    { key: 'taichung_port',  label: '台中港',        buildings: 17076,  bbox: [120.480, 24.230, 120.620, 24.320] },
    { key: 'changhua',       label: '彰化（鹿港）',   buildings: 19638,  bbox: [120.340, 24.020, 120.490, 24.120] },
    { key: 'mailiao',        label: '麦寮',          buildings: 7529,  bbox: [120.160, 23.740, 120.280, 23.830] },
    { key: 'budai',          label: '布袋',          buildings: 3388,   bbox: [120.110, 23.340, 120.230, 23.430] },
    { key: 'anping',         label: '安平（台南）',   buildings: 23269, bbox: [120.130, 22.960, 120.230, 23.040] },
    { key: 'kaohsiung',      label: '高雄港／左营',   buildings: 31109, bbox: [120.240, 22.540, 120.400, 22.680] },
    { key: 'fangliao',       label: '枋寮',          buildings: 2345,   bbox: [120.540, 22.320, 120.640, 22.400] },
    { key: 'suao',           label: '苏澳',          buildings: 8181,  bbox: [121.810, 24.540, 121.910, 24.640] },
    { key: 'hualien',        label: '花莲',          buildings: 47411, bbox: [121.560, 23.930, 121.660, 24.030] },
    { key: 'taitung',        label: '台东（富冈）',   buildings: 5285,  bbox: [121.110, 22.740, 121.210, 22.840] },
    { key: 'penghu',         label: '澎湖（马公）',   buildings: 5711,   bbox: [119.520, 23.520, 119.630, 23.610] },
  ];
  LayerRegistry.setProvider('tiles3d.tw_coastal', function () {
    const on = !!(window.Local3D && Local3D.enabled && Local3D.enabled('tiles3d.tw_coastal'));
    return TW_COASTAL_AOIS.map(function (a) {
      return {
        key: a.key,
        label: a.label,
        sub: a.buildings.toLocaleString() + ' 栋 · ' + (on ? '图层已挂载' : '图层未挂载（先打开本层）'),
        visible: on,
        actions: ['locate'],
        onLocate: function () {
          if (!viewer) return;
          // 选中任何目标都会把 camera.transform 锁在目标 ENU 上；这里必须先解绑，
          // 否则 flyTo 保存并恢复该 transform，镜头会被 rig 每帧拽回目标。
          detachFollow('图层定位');
          viewer.camera.flyTo({
            destination: Cesium.Rectangle.fromDegrees(a.bbox[0], a.bbox[1], a.bbox[2], a.bbox[3]),
            duration: 1.6,
          });
        },
      };
    });
  });

  // ---- 影像：一层可能多实例（热点影像一个热点框一个 provider）------------
  // **只读展示**：挂载/卸载由相机高度驱动，面板不提供删除，只说清"现在挂了几个"。
  LayerRegistry.list('imagery').forEach(function (d) {
    LayerRegistry.setProvider(d.id, function () {
      var rows = (LayerRegistry._debug().imagery || []).filter(function (r) {
        return r.slot === d.id;
      });
      if (!rows.length) return null;
      return rows.map(function (r, i) {
        return {
          key: d.id + '#' + i,
          label: rows.length > 1 ? ('实例 ' + (i + 1)) : '底图实例',
          sub: 'index ' + r.index + (r.show ? ' · 显示' : ' · 隐藏'),
          visible: !!r.show,
          actions: [],
        };
      });
    });
  });
}

/**
 * 把目标的信息覆盖**原地写回** `state.data.targets`。
 *
 * 为什么不保留一张覆盖表、在每个读取点各查一次：名称/图标被左栏标牌、右栏
 * 「整体TS」、意图面板等多处使用，逐个改查表迟早漏一处，表现就是
 * "改了名字但某一栏还是旧的"。写回数据集对象之后，所有读取点自然一致。
 */
function applyTargetOverrides() {
  var d = state.data;
  if (!d || !d.targets) return;
  d.targets.forEach(function (t) {
    var ov = state.targetOverrides[t.id];
    if (!ov) return;
    if (ov.name) t.name = ov.name;
    if (ov.affiliation) t.affiliation = ov.affiliation;
    if (ov.domain) t.domain = ov.domain;
    if (ov.model !== undefined) t.model = ov.model;
  });
}
/**
 * Left/right panel chrome: collapse buttons and drag-to-resize handles.
 * Widths live in CSS variables on #workspace, so responsive rules remain the
 * baseline and a user drag only overrides that one dimension at runtime.
 */
function bindPanelChrome() {
  var ws = document.getElementById('workspace');
  if (!ws) return;
  var left = document.getElementById('panelLeft');
  var right = document.getElementById('panelRight');
  var toggleL = document.getElementById('toggleLeftPanel');
  var toggleR = document.getElementById('toggleRightPanel');
  if (!left || !right || !toggleL || !toggleR) return;

  function setWidth(side, px) {
    px = Math.max(180, Math.min(620, Math.round(px)));
    ws.style.setProperty(side === 'left' ? '--panel-left-w' : '--panel-right-w', px + 'px');
  }
  function syncToggles() {
    var lc = ws.classList.contains('left-collapsed');
    var rc = ws.classList.contains('right-collapsed');
    toggleL.textContent = lc ? '›' : '‹';
    toggleR.textContent = rc ? '‹' : '›';
    toggleL.title = lc ? '展开左栏' : '收起左栏';
    toggleR.title = rc ? '展开右栏' : '收起右栏';
  }
  function resizeViewer() {
    if (viewer) setTimeout(function () { try { viewer.resize(); } catch (e) {} }, 200);
  }
  toggleL.addEventListener('click', function () {
    ws.classList.toggle('left-collapsed');
    syncToggles(); resizeViewer();
  });
  toggleR.addEventListener('click', function () {
    ws.classList.toggle('right-collapsed');
    syncToggles(); resizeViewer();
  });

  var drag = null;
  function onDown(e) {
    var side = this.getAttribute('data-resize');
    if (!side || (side === 'left' && ws.classList.contains('left-collapsed')) ||
        (side === 'right' && ws.classList.contains('right-collapsed'))) return;
    e.preventDefault();
    drag = { side: side, x: e.clientX,
             w: (side === 'left' ? left : right).getBoundingClientRect().width };
    ws.classList.add('resizing');
    this.classList.add('dragging');
    // Listen on window: the pointer leaves the 6 px handle almost immediately.
    window.addEventListener('pointermove', onMove);
    window.addEventListener('mousemove', onMove);
    window.addEventListener('pointerup', onUp, { once: true });
    window.addEventListener('mouseup', onUp, { once: true });
  }
  function onMove(e) {
    if (!drag) return;
    var dx = e.clientX - drag.x;
    setWidth(drag.side, drag.side === 'left' ? drag.w + dx : drag.w - dx);
  }
  function onUp() {
    if (!drag) return;
    drag = null;
    ws.classList.remove('resizing');
    window.removeEventListener('pointermove', onMove);
    window.removeEventListener('mousemove', onMove);
    document.querySelectorAll('.panel-resizer').forEach(function (n) { n.classList.remove('dragging'); });
    resizeViewer();
  }
  document.querySelectorAll('.panel-resizer').forEach(function (n) {
    n.addEventListener('pointerdown', onDown);
    n.addEventListener('mousedown', onDown);
  });
  syncToggles();
}

function bindUI() {
  const bindCheck = function (id, handler) {
    const node = document.getElementById(id);
    if (!node) return;
    // 顺手登记进 dom 缓存，省得新增开关时忘了往 dom 的 id 列表里补一行。
    // （注意：判断开关状态一律用 checkboxOn() 现取 DOM，不要信这个缓存引用——
    // 实测它会与文档脱钩。）
    dom[id] = node;
    node.addEventListener('change', function (e) { handler(e.target.checked); });
  };

  // 图层管理面板：**由注册表生成**（见 web/src/layerPanel.js）。
  //
  // 它取代了原先"每个图层在 HTML 里写一个复选框 + 在这里 bindCheck 一次"的写法 ——
  // 那种写法有两个真相来源，注册表里加了图层面板上看不到。现在面板只是注册表的
  // 投影：开关写回 setDesired，可见性走同一条 applyLayers()。
  //
  // 图层可见性变化**直接调 applyLayers()**，与 setImagery / setTianditu 同一约定。
  // 不要只 markScene()：RenderGovernor.take() 只驱动 syncEntities()（目标实体），
  // 不含图层栈，只标记的话 sink 根本不会被调用——开关点了画面不变（实测如此）。
  if (window.LayerPanel) {
    LayerPanel.bind(document.getElementById('layerPanel'), {
      onChange: function (id, want) {
        // 动态标绘图层会因清空而注销；把用户意图记在 app 侧，重建时恢复。
        var d = LayerRegistry.def(id);
        if (d && d.group === '标绘') _plotLayerDesired[id] = !!want;
        LayerRegistry.setDesired(id, want);
        applyLayers();
        if (want) {
          // 打开图层后平滑定位到它的空间范围（见 focusLayer；没有范围/全球
          // 底图/相机驱动的按需层都会返回 false，不影响开关本身）。
          focusLayer(id);
        } else if (layerFlight.id === id) {
          // 飞行途中把同一图层关掉：这次定位已无意义，立即交还视点控制。
          cancelLayerFlightOnInput();
        }
      },
    });
    // 用户一碰画布就中断图层定位飞行：Cesium 的飞行默认不会因为拖动而停，
    // 不主动取消会让用户"点了一下图层就抢不回视点"。
    if (viewer && viewer.scene && viewer.scene.canvas) {
      ['pointerdown', 'wheel', 'touchstart'].forEach(function (ev) {
        viewer.scene.canvas.addEventListener(ev, cancelLayerFlightOnInput,
                                            { passive: true });
      });
    }
  }

  // 迁移进注册表的图层行：**不再逐个 bindCheck**，改为把回调体挂成逐层 `handler`。
  //
  // 这样它们与底图/标绘走**同一条 apply 通路**，面板才能把它们当普通图层行渲染并
  // 统计要素数量。回调体是从这里原样搬过去的，行为不变。
  [
    // 开关立即重建/清除热点（不等下一帧 syncEntities；无头/低帧率下会滞后）。
    ['layerHotspots', function (v) {
      state.layers.hotspots = v;
      rebuildDynamicAreas(true);
      markScene('layerHotspots');
    }],
    ['layerRoutes', function (v) { state.layers.routes = v; markScene('layerRoutes'); }],
    ['layerTargetsRed', function (v) {
      state.layers.targetsRed = v;
      state.layers.targets = !!(state.layers.targetsRed || state.layers.targetsBlue);
      markScene('layerTargets'); guardFollow();
    }],
    ['layerTargetsBlue', function (v) {
      state.layers.targetsBlue = v;
      state.layers.targets = !!(state.layers.targetsRed || state.layers.targetsBlue);
      markScene('layerTargets'); guardFollow();
    }],
    ['layerLabels', function (v) { state.layers.labels = v; markScene('layerLabels'); }],
    ['layerTracks', function (v) { state.layers.tracks = v; markScene('layerTracks'); }],
    ['modelSensor', function (v) { setSensors(v); }],
    // WSF3D 建筑高度图层：**图层的创建写在 `setWsf3dOverlay` 里**，
    // 而那个函数只有 URL 钩子会调。该行在注册表里是 `kind:'imagery'` 且
    // **没有 handler** —— 于是面板路径（`setDesired` + `apply()`）走到 imagery sink 时，
    // `layersOf('wsf3d.overlay')` 是**空的**（图层从未创建），什么都不做。
    // 实测：置 desired 为 true 后，场景里仍然没有这一层（`layersOf` 返回空）
    // —— **面板上“建筑高度图层（WSF3D）”那个复选框是空转的**。
    //
    // ⚠ **不能直接把 `setWsf3dOverlay` 挂成 handler**：它末尾调 `applyLayers()`，
    // 而 handler 就是在 `apply()` 里被调用的 —— **无限递归**。
    // 这里只做两件事：需要时创建（并登记槽位）、设 `show`。
    // 顺序重排由 `apply()` 在所有 writer 跑完后统一做（`out.imagery = reorderImagery(...)`）。
    ['wsf3d.overlay', function (v) {
      if (v && !state.wsf3dOverlayLayer) {
        state.wsf3dOverlayLayer = OfflineAssets.attachWsf3dOverlay(
          viewer, 'buildingheight', 0.62);
        registerImagerySlot('wsf3d.overlay', state.wsf3dOverlayLayer);
      }
      const L = LayerRegistry.layersOf('wsf3d.overlay') || [];
      L.forEach(function (l) { l.show = !!v; });
    }],
    ['layerOverview', function (v) { setOverview(v); }],
    ['layerTargetPanel', function (v) {
      state.layers.targetPanel = v;
      if (!v) clearTargetSelection();
      else if (state.selected) renderTargetStatus(state.selected);
    }],
    ['compCompass', function (v) { setComponent('compass', v); }],
    ['compScalebar', function (v) { setComponent('scalebar', v); }],
    ['compViewpoint', function (v) { setComponent('viewpoint', v); }],
    // 地形/气象：它们原先只是 HTML 复选框（DOM-only），迁移「底图」组时漏了，
    // 现在补进注册表并在这里接上原本 bindCheck 的回调体。
    ['terrain.model', function (v) { setTerrain(v); }],
    ['terrain.mesh', function (v) { setTerrainMesh(v); }],
    ['weather.model', function (v) { setWeather(v); }],
  // B4 选中掩码：开关只改 state.layers.focusMask，重画交给 updateFocusMask()。
  // 必须走注册表 + apply() 的 handler：绕过注册表直接调 setter，会被 apply() 改回去。
  ['overlay.focusMask', function (v) { state.layers.focusMask = v; updateFocusMask(); }],
  // D (direction 2): occupation faces + minimap mask. The handler writes the
  // state flag and repaints immediately; apply() alone would leave stale shows.
  // 占领区域已下线：图层行与 handler 一并停用（恢复时恢复这两处）。
  // [OCCUPATION_LAYER_ID, function (v) { state.layers.occupation = v; updateOccupationAreas(); }],
  ].forEach(function (pair) {
    var d = LayerRegistry.def(pair[0]);
    if (d) d.handler = function (id, want) { pair[1](want); };
  });

  const refilter = function () { renderTargetCounts(); markScene('filter'); guardFollow(); };
  bindCheck('filterRed', function (v) { state.filters.red = v; refilter(); });
  bindCheck('filterBlue', function (v) { state.filters.blue = v; refilter(); });
  bindCheck('filterSea', function (v) { state.filters.sea = v; refilter(); });
  bindCheck('filterLand', function (v) { state.filters.land = v; refilter(); });
  bindCheck('filterAir', function (v) { state.filters.air = v; refilter(); });


  // 顶栏「鸟瞰图」按钮已移除，改由上面的 #layerOverview 复选框控制；
  // 组件自带的 × 仍可关闭。
  const ovHide = document.getElementById('ovHide');
  if (ovHide) {
    // ⚠ **必须走注册表**。原先这里是 `setOverview(false)`，
    // 而 `setOverview` 只写 `state.layers.overview` —— **绕过注册表**。
    // 后果：点×能把鸟瞰图关掉，但注册表那一行的 desired 仍是 true，
    // 随后任何一次 `apply()` 都会调 handler 把它**弹回来** ——
    // 与 `?overview=1` / `?places=1` 完全同型。
    //
    // 而且这条路径不是 URL 钩子，是 **UI 按钮**：
    // 上一轮只查了钩子，所以漏掉了它。类别不限于钩子。
    //
    // 走 setSwitch 后三者一致：面板复选框取消勾选、注册表 desired=false、
    // 场景里鸟瞰图真的隐藏（且不会再自己弹回来）。
    // ⚠不能用 `setSwitch('layerOverview', ...)`：`setSwitch` 按 **checkbox id**
    // 反查图层（`layerIdForCheckbox`），而 `layerOverview` 那一行**没有 `checkbox` 字段**，
    // 反查返回 null —— 调用**静默空转**。实测：点×后 desired 仍为 true、isVisible 仍为 true。
    // （我先前就是这么修的，又踩了一次——**推断 `setSwitch` 能用之前应先查它的映射依据**。）
    //
    // 直接走注册表，与 `?overview=1` 那处修法一致（那处已实测可行）。
    ovHide.addEventListener('click', function () {
      LayerRegistry.setDesired('layerOverview', false);
      LayerRegistry.apply();
    });
  }

  // 地图上的「区域级TS」横幅与其开关已移除，读数只在右栏「区域TS」显示。
  // 「目标元数据面板」开关现在控制右栏「目标TS栏」是否展开详情。

  // 地名是懒加载的：初始 checked 不会触发 change，但 initViewer 里已经写过一次
  // desired 并 apply 过（label sink → Places.setEnabled），这里不再重复调用。
  // （原先这段相同代码写了两遍，属于历史重复，一并去掉。）



  // 地图组件的三个开关已迁移进注册表（上面的 handler 数组），静态复选框也已从
  // index.html 删除 —— 这里不能再 bindCheck，否则面板里点了没反应（节点不存在时
  // bindCheck 直接 return，不报错）。

  bindCheck('followTarget', function (v) {
    if (v && !state.data) {
      state.follow.enabled = false;
      dom.followTarget.checked = false;
      setFollowNote('未加载TS，无法跟随。请先从「演示案例」选择案例。');
      return;
    }
    // 只切换驾驶模式，不解除与目标的绑定：取消勾选回到普通居中跟随。
    setFollowMode(!!v);
    if (!v) {
      setFollowNote('已回到目标居中模式；再次勾选切换第三人称斜后方跟随。');
    } else if (!state.follow.id) {
      setStatus('目标跟随已开启 · 点击目标开始跟随', 'ok');
      setFollowNote('勾选后点击任一目标，相机进入第三人称斜后方跟随。');
    }
  });

  // 指南针点击复位正北
  if (dom.mapCompass) {
    dom.mapCompass.addEventListener('click', function () {
      detachFollow('复位正北');
      const cam = viewer.camera;
      const c = cam.positionCartographic;
      const o = cameraOrientation();
      // pitch 接近 ±90° 时 heading 无视觉意义，clamp 到可辨范围内
      const pitch = Math.max(-Math.PI * 0.494, Math.min(-Math.PI / 36, o.pitchRad));
      cam.setView({
        destination: Cesium.Cartesian3.fromRadians(c.longitude, c.latitude, c.height),
        orientation: { heading: 0, pitch: pitch, roll: 0 },
      });
    });
  }

  // ⚠ 这里原先有 7 个 `bindCheck('modelXxx', …)`（Wsf3dOverlay / Buildings /
  // BuildingsOnline / Weather / Imagery / Tianditu / TiandituLabel）。**全是死代码**：
  // 这些复选框随「底图」组一起并进了生成式图层面板，`getElementById` 对它们一律返回
  // null，而 `bindCheck` 的首行就是 `if (!node) return` —— 不报错、也不接线。
  // 留着比删掉更糟：读代码的人会以为这条通路是活的。开关现在由注册表的 handler
  // 驱动（见 LAYERS 里各行的 `handler` 与 `registerLayerProviders`）。
  //
  // 同理，下面两处原先读的是 `dom.*` 缓存引用 —— 缓存只在 bindCheck 成功时才填，
  // 节点没了它永远是 undefined，条件恒为 false。改走 `checkboxOn()`（它会反查注册表）；
  // 注意 `checkboxOn('modelImagery')` 在本项目里**一直是 false**：离线底图的初始
  // `checked` 从来没打过（见原始 HTML），所以这一句在此之前也没生效过。
  if (checkboxOn('modelImagery')) setImagery(true);
  // 天地图是默认底图，`initViewer` 里已把它设成 baseLayer。这里补一句是为了把
  // 在线/离线的提示文案写清楚 —— 用户需要知道当前底图是哪一种。
  if (checkboxOn('modelTianditu')) setTianditu(true);

  // 天地图可用性探测。必须**真的发一次请求**：`/api/tdt/status` 只能说明"配了 key"，
  // 说明不了"key 还能用"——配额耗尽（429）与被禁用（301005）都只有请求了才知道。
  // 之前 key 失效时底图静默变深蓝，就是因为没有任何人问过这个问题。
  probeTiandituStatus();

  // 军标资产探测：确认 web/assets/symbols/ 里有预生成的 SVG。没有就退回内置生成器。
  // 放在这里（而非首次用到符号时）是为了**在画图标之前**定下来，否则会先用静态路径
  // 建好一波 billboard，等探测失败再逐个改回来。资产缺失时 `<img>`/billboard 是静默
  // 失败的，页面上只会看到空白，所以这里必须显式判断并在退回后重画一遍。
  if (window.Symbology && Symbology.verifyAssets) {
    Symbology.verifyAssets().then(function (ok) {
      if (ok) return;
      // 退回内置生成器：已建好的 billboard 与已渲染的整体TS栏都要重画。
      syncSelection();
      if (window.StatusPanels && StatusPanels._resetCache) {
        StatusPanels._resetCache();
        updateInferenceUi();
      }
    });
  }

  // 数据健康面板与「分享场景」按钮已随界面重构移除。
  // 诊断改为按需查询 SituationHost.health()；`?scene=` 深链仍然可用（sceneDoc 未动），
  // 只是不再有顶栏按钮来生成它。

  // Cesium ion 令牌的**输入与保存**已随 UI 块一起移除：token 现在只从服务端 .env 的
  // CESIUM_ION_TOKEN 读取，前端通过 /api/ion/token 取用（buildings3d.js 负责）。
  // 服务端的 POST 写入接口也一并删掉——唯一的调用方就是那个按钮，留着就是死代码。
  //
  // 附带修掉一个潜在 ReferenceError：原处理器 catch 分支里写了 `if (note) …`，
  // 而 `note`（#ionTokenNote）在上一轮移除说明文字时已经删掉，一旦保存失败就会抛。
  // 整块移除后这个问题自然消失。

  // 在线实景是默认图层（index.html 里 checked）：HTML 的 checked 同样不会触发
  // change，而它又不像天地图那样在 initViewer 里已经建好，所以必须显式跑一次。
  // 提示文案由 setBuildingsOnline 的成功/回落分支负责写。
  if (checkboxOn('modelBuildingsOnline')) setBuildingsOnline(true);

  // 顶栏仍有两个下拉（TS案例）；TS注册/管理/下载与图层批量操作已改为直挂按钮，
  // 不再需要 bindMenu。
  bindMenu('menuCase', 'menuCaseList');
  if (dom.menuCase) dom.menuCase.addEventListener('click', function () { openCaseMenu(); });
  // 标绘面板：与 TS案例 同一套开合，但它是**工具面板**，不因外部点击而收起
  // （keepOpen 恒真）——只有点【标绘】按钮或按 Esc 才关。
  // 面板内部控件的 stopPropagation 由 bindMenu 统一加上，所以点类型/红蓝方
  // 下拉不会再把它关掉。
  bindMenu('menuPlot', 'menuPlotPanel', { keepOpen: function () { return true; } });
  document.addEventListener('click', function () { closeAllMenus(); });

  // TS菜单：输入 / 管理 / 数据下载，三标签页共用一个模态框
  Array.prototype.forEach.call(
    document.querySelectorAll('[data-situation-cmd]'), function (btn) {
      btn.addEventListener('click', function () {
        closeAllMenus();
        if (window.SituationPanel) SituationPanel.open(btn.dataset.situationCmd);
      });
    });
  if (window.SituationPanel) SituationPanel.bind();

  // 图层管理批量操作
  Array.prototype.forEach.call(
    document.querySelectorAll('[data-layer-cmd]'), function (btn) {
      btn.addEventListener('click', function () {
        closeAllMenus();
        applyLayerCommand(btn.dataset.layerCmd);
      });
    });
  document.addEventListener('keydown', function (e) {
    // 浮动目标面板已移除，Esc 改为清除选中态 + 复位目标TS栏。
    // Esc 走**强制**关菜单：它区别于"外部点击"——后者对工具面板（标绘）是豁免的，
    // 而按 Esc 是用户明确要关掉。
    if (e.key === 'Escape') { closeAllMenus(true); clearTargetSelection(); }
  });

  document.getElementById('btnPlay').addEventListener('click', function () { setPlaying(!state.playing); });
  document.getElementById('btnRun').addEventListener('click', function () { setPlaying(!state.playing); });
  document.getElementById('btnReset').addEventListener('click', function () {
    setPlaying(false);
    setTick(0);
    Object.keys(state.targets).forEach(function (id) {
      state.targets[id].trail.length = 0;
      state.targets[id].lastTick = -1;
    });
    renderTargetCounts();
    renderTransport();
    updateInferenceUi();
  });

  document.getElementById('speedSelect').addEventListener('change', function (e) {
    state.ticksPerSecond = Number(e.target.value);
  });

  dom.timeline.addEventListener('input', function (e) {
    setPlaying(false);
    setTick(Number(e.target.value));
    renderTargetCounts();
    renderTransport();
    updateInferenceUi();
  });

  document.getElementById('btnStepBack').addEventListener('click', function () {
    setPlaying(false); setTick(state.tick - 1); renderTargetCounts(); renderTransport(); updateInferenceUi();
  });
  document.getElementById('btnStepFwd').addEventListener('click', function () {
    setPlaying(false); setTick(state.tick + 1); renderTargetCounts(); renderTransport(); updateInferenceUi();
  });


  // "位置信息"：回到光标跟随模式，同时把视角拉回案例默认机位。
  document.getElementById('btnLocate').addEventListener('click', function () {
    clearHotspotSelection();
    const cam = (state.data && state.data.camera) || state.baseCamera;
    if (cam) applyCamera(cam);
  });

  captureLayerDefaults();

  bindPicking();
  bindPanelChrome();
  bindPlotPanel();
  bindFollowInput();

  // 跟随相机挂在 preRender：Scene.render 的顺序是 preUpdate → update →
  // preRender → draw，在这里改相机本帧生效。
  viewer.scene.preRender.addEventListener(function () { applyCameraRig(performance.now()); });

  // 兜底：底图瓦片队列可能长时间不归零（视野大、瓦片多），给鸟瞰图一个延时创建。
  // 4 s 足以让主底图完成第一轮细化，不至于和它抢连接。
  setTimeout(function () {
    if (state.layers.overview && window.Overview && !Overview.isVisible()) {
      try { Overview.show(); } catch (e) { console.warn('[overview]', e.message); }
    }
  }, 4000);

  // 地图组件统一刷新（10 Hz；比例尺内部压到约 3 Hz）
  setInterval(updateMapComponents, 100);
  updateMapComponents();

  // 相机移动时也刷新一次读数（节流到 4 Hz），否则刚打开页面时面板全是"—"。
  let lastGeoUpdate = 0;
  // 底图加载进度。queued 为 0 表示当前视野的瓦片已经全部就位。
  viewer.scene.globe.tileLoadProgressEvent.addEventListener(onTileLoadProgress);

  // 预加载循环：空闲且相机停稳时，把"视口内、另一档层级"的瓦片提前取回来。
  // 默认开启，`?prefetch=off` 可关（见 applyPrefetchParams）。
  startPrefetchLoop();

  // 按需层的周期性复核。两件事都需要它：
  //   1. `surfaceRenderLevel()`（正在渲染的层级）**必然晚于相机停稳**，而重算原先只挂在
  //      camera.changed 上 —— 相机一停就没有事件，最后一次重算用的是滞后一档的层级，
  //      实测 4 km 下高倍段恒为空（离线高清影像等于不存在）。
  //   2. 热点/城区的"还有没有用"闸门同样以 S 为判据。
  // 1.5 秒一次即可：三个同步在集合没变时立刻返回，代价是一次比较。
  setInterval(function () {
    if (!viewer || !viewer.scene || !viewer.scene.globe) return;
    if (!viewer.scene.globe.show) return;
    syncOfflineDetail();
    syncHotspotImagery();
    syncUrbanImagery();
  }, 1500);

  // 地球可见性看门狗：`globe.show` 只被在线三维实景层（buildings3d.js）改，改完没有
  // 事件通知到这里；而"不可见就不该取瓦片"与"恢复可见要把按需层挂回来"两条都依赖它。
  // 1 秒比一次布尔值，代价可忽略；只在**真的变了**时才重排图层。
  let lastGlobeShow = null;
  setInterval(function () {
    if (!viewer || !viewer.scene || !viewer.scene.globe) return;
    const on = !!viewer.scene.globe.show;
    if (on === lastGlobeShow) return;
    lastGlobeShow = on;
    syncUrbanImagery();
    syncHotspotImagery();
    syncOfflineDetail();
    applyLayers();
  }, 1000);

  viewer.camera.changed.addEventListener(function () {
    camChangedCount++;
    const now = Date.now();
    if (now - lastGeoUpdate < 250) return;
    lastGeoUpdate = now;
    updateGeoReadout();
    syncUrbanImagery();
    syncHotspotImagery();
    syncOfflineDetail();
    guardFollow();
    if (window.Places) Places.onCameraChanged(false);
    // 相机高度跨过 LOD 阈值时才重建，且加一段防抖，避免飞行过程中反复触发。
    // 相机 rig 正在逐帧改相机时（跟随运动目标 / chase 收敛 / glide），防抖会被
    // 反复重置而**永不执行**，所以跳过；detachFollow() 收尾时会补一次。
    // ⚠ 判据不能用 `state.follow.id`：现在"选中即绑定"，静止目标被选中后用户
    // 自己缩放相机时 rig 并不改相机，仍应刷新体块 LOD。用 lastLockMs 更准确。
    const rigBusy = !!state.follow.id &&
      performance.now() - (state.follow.lastLockMs || 0) < 450;
    if (!rigBusy && checkboxOn('modelBuildings')) {
      if (state.buildingTimer) clearTimeout(state.buildingTimer);
      state.buildingTimer = setTimeout(refreshBuildings, 400);
    }
    // 车/船的 2D/3D 形态（判据在 web/src/entityLod.js）。放在这个监听器内部
    // 即可被上面的 250 ms 节流覆盖：applyTargetLod 只在**形态真的变了**时遍历
    // 实体，所以相机连续移动时每 250 ms 至多一次空转判断，不会每帧重写实体状态。
    applyTargetLod();
  });

  // 目标实体建好时相机可能还没动过，先算一次让初始形态与当前高度一致
  // （否则会一律停在默认的 2D）。案例数据异步到位，故补两次延迟重试，
  // 与 places.js 处理同类时序问题的做法一致。
  applyTargetLod(true);
  setTimeout(function () { applyTargetLod(true); }, 1200);
  setTimeout(function () { applyTargetLod(true); }, 3000);
}

// --------------------------------------------------------------------------
// 标绘面板：地图手动绘制 + 按坐标添加
// --------------------------------------------------------------------------
//
// 为什么需要这一块：`plotting.js` 的引擎与 24 类几何早就完成了，但**没有任何界面入口**
// —— `Plotting.add()` 只能在控制台里调。人工介入的标绘（分界线、火力扇面、禁飞区…）
// 一条也落不了图。这里补上两条路径，共用同一套「类型 / 归属 / 图层」选择。
//
// 参数录入**没有**放进 TS注册 模态框：绘制按钮必须紧挨图层开关，而两条路径共用同一组
// 控件，分到两处等于把类型/归属/图层各写一遍。要挪过去是一处 DOM 搬家。

let plotDraw = null;         // {type, layer, side, label, points:[Carto], previewId}
let plotDraftEntities = [];

function plotCatalogue() {
  return (window.Plotting && Plotting.catalogue()) || [];
}

function plotDef(type) {
  return plotCatalogue().filter(function (c) { return c.type === type; })[0] || null;
}

function setPlotHint(text, cls) {
  const el = document.getElementById('plotHint');
  if (!el) return;
  el.textContent = text || '';
  el.className = 'switch-note' + (cls ? ' ' + cls : '');
}

/** 草稿控制点标记。用普通 point 图元，不走 Plotting —— 它们不是标绘的一部分。 */
function clearPlotDraft() {
  plotDraftEntities.forEach(function (e) {
    try { viewer.entities.remove(e); } catch (err) { /* 已随场景销毁 */ }
  });
  plotDraftEntities = [];
  if (plotDraw && plotDraw.previewId && window.Plotting) {
    try { Plotting.remove(plotDraw.previewId); } catch (err) { /* 已移除 */ }
    plotDraw.previewId = null;
  }
}

function plotDrawReset() {
  clearPlotDraft();
  plotDraw = null;
  const btn = document.getElementById('plotDraw');
  const cancel = document.getElementById('plotCancel');
  if (btn) btn.textContent = '地图绘制';
  if (cancel) cancel.style.display = 'none';
  setPlotHint('');
}

/** 已落点数 → 提示文案。把「还需几个点」说清楚，否则用户不知道什么时候能完成。 */
function syncPlotHint() {
  if (!plotDraw) return;
  const def = plotDef(plotDraw.type);
  const ctrl = (def && def.ctrl) || { min: 2, max: null };
  const n = plotDraw.points.length;
  const btn = document.getElementById('plotDraw');
  if (n < ctrl.min) {
    setPlotHint('还需 ' + (ctrl.min - n) + ' 个控制点（已点 ' + n + '）', 'warn');
    if (btn) btn.textContent = '绘制中…';
  } else {
    const maxTxt = ctrl.max ? ('／' + ctrl.max) : '';
    setPlotHint('已点 ' + n + maxTxt + ' 个，可「完成」', 'ok');
    if (btn) btn.textContent = '完成绘制';
  }
}

/** 点数够了下限就生成一次预览，让用户看到形状再决定是否完成。 */
function syncPlotPreview() {
  if (!plotDraw || !window.Plotting) return;
  const def = plotDef(plotDraw.type);
  const ctrl = (def && def.ctrl) || { min: 2, max: null };
  if (plotDraw.points.length < ctrl.min) {
    if (plotDraw.previewId) {
      try { Plotting.remove(plotDraw.previewId); } catch (err) { /* 已移除 */ }
      plotDraw.previewId = null;
    }
    return;
  }
  // 续画时**更新**同一条预览，而不是 remove + add：后者会让动态标绘图层在
  // "有要素/没要素"之间反复注册与注销，面板行跟着闪。
  if (plotDraw.previewId) {
    try {
      Plotting.update(plotDraw.previewId, {
        layer: plotDraw.layer, points: plotDraw.points,
        hostility: plotDraw.side, label: plotDraw.label || null,
      });
      return;
    } catch (err) {
      plotDraw.previewId = null;   // 预览被外部清掉：下面重建一次
    }
  }
  try {
    const rec = Plotting.add({
      type: plotDraw.type, layer: plotDraw.layer, points: plotDraw.points,
      hostility: plotDraw.side, label: plotDraw.label || null,
    });
    plotDraw.previewId = rec.id;
  } catch (err) { /* 预览失败不打断绘制，完成时会再报一次 */ }
}

function plotCartoToLonLat(c) {
  return [Cesium.Math.toDegrees(c.longitude), Cesium.Math.toDegrees(c.latitude)];
}

function addPlotPoint(carto) {
  if (!plotDraw || !carto) return;
  plotDraw.points.push(plotCartoToLonLat(carto));
  const def = plotDef(plotDraw.type);
  const ctrl = (def && def.ctrl) || { min: 2, max: null };
  const pal = (window.Plotting && Plotting.palette) ? Plotting.palette() : { friendly: '#c8102e', enemy: '#123a6b' };
  const col = Cesium.Color.fromCssColorString(plotDraw.side === 'enemy' ? pal.enemy : pal.friendly);
  plotDraftEntities.push(viewer.entities.add({
    position: Cesium.Cartesian3.fromDegrees(plotDraw.points[plotDraw.points.length - 1][0],
                                            plotDraw.points[plotDraw.points.length - 1][1]),
    point: {
      pixelSize: 9, color: col, outlineColor: Cesium.Color.WHITE, outlineWidth: 2,
      disableDepthTestDistance: Number.POSITIVE_INFINITY,
    },
  }));
  if (ctrl.max && plotDraw.points.length >= ctrl.max) { syncPlotPreview(); syncPlotHint(); finishPlotDraw(); return; }
  syncPlotPreview();
  syncPlotHint();
}

function finishPlotDraw() {
  if (!plotDraw) return;
  const def = plotDef(plotDraw.type);
  const ctrl = (def && def.ctrl) || { min: 2, max: null };
  if (plotDraw.points.length < ctrl.min) {
    setPlotHint('控制点不足：' + plotDraw.type + ' 至少需要 ' + ctrl.min + ' 个', 'warn');
    return;
  }
  const pts = plotDraw.points.slice();
  const spec = {
    type: plotDraw.type, layer: plotDraw.layer, points: pts,
    hostility: plotDraw.side, label: plotDraw.label || null,
  };
  // 预览已经是一份真标绘；完成时先撤掉预览再正式落一条，避免重复
  if (plotDraw.previewId) {
    try { Plotting.remove(plotDraw.previewId); } catch (err) { /* 已移除 */ }
    plotDraw.previewId = null;
  }
  try {
    Plotting.add(spec);
    setPlotHint('已添加 ' + (def ? def.zh : plotDraw.type) + '（' + pts.length + ' 个控制点）', 'ok');
  } catch (err) {
    setPlotHint('添加失败：' + err.message, 'warn');
  }
  plotDraftEntities.forEach(function (e) { try { viewer.entities.remove(e); } catch (err) {} });
  plotDraftEntities = [];
  plotDraw = null;
  const btn = document.getElementById('plotDraw');
  const cancel = document.getElementById('plotCancel');
  if (btn) btn.textContent = '地图绘制';
  if (cancel) cancel.style.display = 'none';
}

/**
 * 解析坐标文本：每行一个点，`lon, lat`（逗号或空白分隔）。
 * 逐条报错而不是只报第一条 —— 粘一份大清单时反复试是最烦的。
 */
function parsePlotPoints(text) {
  const lines = String(text || '').split(/\r?\n/);
  const pts = [];
  const errs = [];
  lines.forEach(function (raw, i) {
    const s = raw.trim();
    if (!s || s.charAt(0) === '#') return;
    const parts = s.split(/[,\s]+/).filter(Boolean);
    if (parts.length !== 2) { errs.push('第 ' + (i + 1) + ' 行：需要两个数（经度, 纬度）'); return; }
    const lon = Number(parts[0]);
    const lat = Number(parts[1]);
    if (!isFinite(lon) || !isFinite(lat)) { errs.push('第 ' + (i + 1) + ' 行：不是有效数字'); return; }
    if (lon < -180 || lon > 180 || lat < -90 || lat > 90) {
      errs.push('第 ' + (i + 1) + ' 行：坐标越界（' + lon + ', ' + lat + '）'); return;
    }
    pts.push([lon, lat]);
  });
  return { points: pts, errors: errs };
}


/**
 * 刷新标绘工具的「归属图层」下拉。
 *
 * 选项来自 `Plotting.categories()`，**不是**注册表里的图层清单：图层面板只列
 * 当前有要素的类别，而创建入口要允许往任意类别里画（包括暂时为空的类别）。
 * 注册动态图层后由 syncPlotLayers 调它，保证下拉随类别扩展自动更新。
 */
function refreshPlotLayerSelect() {
  const layerSel = document.getElementById('plotLayer');
  if (!layerSel || !window.Plotting) return;
  const cur = layerSel.value;
  const opts = plotLayerOptions();
  layerSel.innerHTML = '';
  opts.forEach(function (o) {
    const n = document.createElement('option');
    n.value = o[0];
    n.textContent = o[1];
    layerSel.appendChild(n);
  });
  if (cur && opts.some(function (o) { return o[0] === cur; })) {
    layerSel.value = cur;
  } else if (opts.length) {
    layerSel.value = opts[0][0];
  }
}

function bindPlotPanel() {
  const typeSel = document.getElementById('plotType');
  const layerSel = document.getElementById('plotLayer');
  const sideSel = document.getElementById('plotSide');
  const labelIn = document.getElementById('plotLabel');
  const pointsIn = document.getElementById('plotPoints');
  const drawBtn = document.getElementById('plotDraw');
  const cancelBtn = document.getElementById('plotCancel');
  const addBtn = document.getElementById('plotAddByText');
  if (!typeSel || !window.Plotting) return;

  const cat = plotCatalogue();
  typeSel.innerHTML = '';
  cat.forEach(function (c) {
    // 系统生成的类型（目标轨迹）由选中目标自动维护，不进入手动标绘入口
    if (c.system) return;
    const o = document.createElement('option');
    o.value = c.type;
    // 标准来源直接写在选项里 —— 每个类型是中式还是美式补齐，选的时候就该看见
    o.textContent = c.zh + '（' + (c.sourceStandard || '?') + '）';
    typeSel.appendChild(o);
  });
  refreshPlotLayerSelect();
  const syncLayerDefault = function () {
    const d = plotDef(typeSel.value);
    if (d && d.defaultLayer) layerSel.value = d.defaultLayer;
  };
  typeSel.addEventListener('change', syncLayerDefault);
  syncLayerDefault();

  drawBtn.addEventListener('click', function () {
    if (plotDraw) { finishPlotDraw(); return; }
    const d = plotDef(typeSel.value);
    if (!d) return;
    plotDraw = {
      type: typeSel.value, layer: layerSel.value,
      side: sideSel.value === 'enemy' ? 'enemy' : 'friendly',
      label: (labelIn.value || '').trim(), points: [], previewId: null,
    };
    if (cancelBtn) cancelBtn.style.display = '';
    syncPlotHint();
  });

  if (cancelBtn) {
    cancelBtn.style.display = 'none';
    cancelBtn.addEventListener('click', function () {
      plotDrawReset();
      setPlotHint('已取消绘制');
    });
  }

  addBtn.addEventListener('click', function () {
    const res = parsePlotPoints(pointsIn.value);
    if (res.errors.length) { setPlotHint('坐标有误：' + res.errors.join('；'), 'warn'); return; }
    const d = plotDef(typeSel.value);
    const ctrl = (d && d.ctrl) || { min: 2, max: null };
    if (res.points.length < ctrl.min) {
      setPlotHint('控制点不足：' + (d ? d.zh : typeSel.value) + ' 至少需要 ' + ctrl.min + ' 个（当前 ' + res.points.length + '）', 'warn');
      return;
    }
    try {
      Plotting.add({
        type: typeSel.value, layer: layerSel.value, points: res.points,
        hostility: sideSel.value === 'enemy' ? 'enemy' : 'friendly',
        label: (labelIn.value || '').trim() || null,
      });
      setPlotHint('已按坐标添加 ' + (d ? d.zh : typeSel.value) + '（' + res.points.length + ' 个控制点）', 'ok');
    } catch (err) {
      setPlotHint('添加失败：' + err.message, 'warn');
    }
  });

}
function bindPicking() {
  const handler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);

  // 光标跟随模式下的经纬与视高读数。热点区域被选中时面板锁定，不再跟随。
  handler.setInputAction(function (movement) {
    const ray = viewer.camera.getPickRay(movement.endPosition);
    const cartesian = ray ? viewer.scene.globe.pick(ray, viewer.scene) : null;
    state.cursorCarto = cartesian ? Cesium.Cartographic.fromCartesian(cartesian) : null;
    updateGeoReadout();
  }, Cesium.ScreenSpaceEventType.MOUSE_MOVE);

  // 记录按下位置：用来区分"点击"和"拖拽"（拖动地图不应改变选中态）。
  var downPos = null;
  handler.setInputAction(function (m) {
    downPos = Cesium.Cartesian2.clone(m.position);
  }, Cesium.ScreenSpaceEventType.LEFT_DOWN);

  handler.setInputAction(function (click) {
    // 绘制模式下左键**只落控制点，不选中目标**：否则点在地图上会误切换选中，
    // 把正在画的标绘打断。模式退出后恢复原有拾取行为。
    if (plotDraw) {
      const carto = groundCartoAt(click.position.x, click.position.y);
      if (carto) addPlotPoint(carto);
      downPos = null;
      return;
    }
    // 按下与松开相距超过 6 px 视为拖拽（旋转/平移地图），不改变选中态。
    if (downPos) {
      const dx = click.position.x - downPos.x;
      const dy = click.position.y - downPos.y;
      downPos = null;
      if (Math.sqrt(dx * dx + dy * dy) > 6) return;
    }
    const picked = viewer.scene.pick(click.position);
    const id = picked && picked.id && typeof picked.id.id === 'string' ? picked.id.id : null;
    if (!id) {
      // 点到空白（天空/地面）：取消目标选中，关系线随之清除。
      if (state.selected) clearTargetSelection();
      return;
    }

    if (id.indexOf('target-') === 0) {
      selectTarget(id.slice('target-'.length), false);
    } else if (id.indexOf('hotspot-') === 0) {
      // 热点同样不是目标：先清掉目标选中，避免“目标关系线 + 热点高亮”同时存在。
      if (state.selected) clearTargetSelection();
      selectHotspot(Number(id.slice('hotspot-'.length)), false);
    } else if (state.selected) {
      // 点到其它非目标图元（标牌、关系线、航路、热点面…）：同样取消选中。
      clearTargetSelection();
    }
  }, Cesium.ScreenSpaceEventType.LEFT_CLICK);
}

// --------------------------------------------------------------------------
// URL 钩子（截图取证与深链）
// --------------------------------------------------------------------------
//
// 拆成两半：与案例无关的（底图/组件/机位）在 clearSituation 之前就要生效，
// 依赖案例数据的（跟随/热点/帧号）必须等数据加载完。顺序上无关钩子先跑，
// 所以 ?cam= 与 ?follow= 同时出现时 follow 胜出。

function applyCaseIndependentHooks() {
  // TS面板：?situation=input|manage|download（取证与深链用）
  if (params.has('situation') && window.SituationPanel) {
    SituationPanel.open(params.get('situation') || 'input');
  }

  // TS面板：?situation=input|manage|download（取证与深链用）
  if (params.has('situation') && window.SituationPanel) {
    SituationPanel.open(params.get('situation') || 'input');
  }

  // 本地地名：?places=0|1
  // （原先这行注释写的是「鸟瞰图：?overview=0|1」—— 复制粘贴留下的错注释。）
  if (params.has('places')) {
    const on = params.get('places') !== '0';
    // ⚠ **必须走注册表**，与紧邻的那批开关同理（见下面「这些开关的 DOM 复选框
    // 已经不存在」那段注释）。原先这里两处都不对：
    //   · `document.getElementById('modelPlaces')` —— 节点**已经不存在**
    //     （迁进了注册表面板），恒为 null，那一行是死代码；
    //   · `Places.setEnabled(on)` —— 它是 **label sink 的落点**（见 `sinks` 里
    //     `if (id === 'places.labels' …) Places.setEnabled(!!want)`），**不是注册表意图**。
    //     直接调它只改了"效果"，随后任何一次 `apply()` 都会按注册表那一行
    //     （`places.labels`，desired 仍是 false）调 `setEnabled(false)` 关回去。
    //
    // 实测（`?case=taiwan_landing&places=1`）：地名**确实被加载了**
    // （`total 7342`），但 `enabled false`、且 `runtime['places.labels'].desired === false`
    // —— 参数**静默失效**，与 `?overview=1` 完全同型。
    setSwitch('modelPlaces', on);
  }

  if (params.has('overview')) {
    const on = params.get('overview') !== '0';
    // ⚠ **必须走注册表**。原先这里是 `setOverview(on)`，而 `setOverview` 只写
    // `state.layers.overview` —— **绕过注册表**。随后任何一次 `apply()` 都会按注册表
    // 那一行（`layerOverview`，desired 仍是 false）把它改回来，于是
    // `?overview=1` **静默失效**：实测 URL 上带着 `overview=1`，而
    // `LayerRegistry.runtime['layerOverview'].desired === false`、`Overview.isVisible() === false`。
    //
    // 这与下面 `components` 那段注释说的是**同一件事**（"直接调 setComponent 会绕过注册表，
    // 面板上的复选框仍显示旧状态 —— 就是'界面与事实不符'"）—— 当时 `components` 改对了，
    // **overview 这一处漏改了**。
    //
    // 走 setDesired + apply 后由该行的 handler 去调 `setOverview`，DOM/state/注册表三者一致。
    LayerRegistry.setDesired('layerOverview', on);
    LayerRegistry.apply();
  }

  // 地图组件：?components=0|1 全关/全开；?components=a,b 白名单；?components=-a 排除
  if (params.has('components')) {
    const raw = params.get('components');
    // 一律走 setSwitch：它会在有 DOM 节点时派发 change，没有节点时改注册表意图。
    // 直接调 setComponent 会**绕过注册表**，面板上的复选框仍显示旧状态 ——
    // 就是"界面与事实不符"。
    const CIDS = ['compCompass', 'compScalebar', 'compViewpoint'];
    const CNAMES = ['compass', 'scalebar', 'viewpoint'];
    if (raw === '0') {
      CIDS.forEach(function (id) { setSwitch(id, false); });
    } else if (raw === '1' || raw === '') {
      CIDS.forEach(function (id) { setSwitch(id, true); });
    } else {
      const parts = raw.split(',').map(function (x) { return x.trim(); }).filter(Boolean);
      const exclude = parts.length && parts[0].charAt(0) === '-';
      CNAMES.forEach(function (nm, k) {
        setSwitch(CIDS[k], exclude ? parts.indexOf('-' + nm) < 0 : parts.indexOf(nm) >= 0);
      });
    }
  }
  [['compass', 'compass'], ['scalebar', 'scalebar'], ['viewpoint', 'viewpoint']].forEach(function (pair) {
    if (!params.has(pair[0])) return;
    const on = params.get(pair[0]) !== '0';
    // 同上：走 setSwitch，保证注册表意图与面板一致
    setSwitch('comp' + pair[0].charAt(0).toUpperCase() + pair[1].slice(1), on);
  });

  // ⚠ 这些开关的 DOM 复选框**已经不存在**（都迁进了注册表面板）。原先这里直接写
  // `.checked`，对 null 取属性会抛 TypeError；而它跑在启动链里，异常被最外层的
  // `.catch` 吞掉 —— 症状是**整个案例都不加载**（`state.data` 始终为 null、机位停在
  // Cesium 默认值），`#statusText` 才写着"启动失败"，而 `pageErrors` 是**空的**。
  // 用 `setSwitch` 走"有节点派发 change / 没节点改注册表意图"这条统一通路。
  // 下面紧跟的显式 setter 保留：两者幂等，且 setter 可能还做了别的事。
  if (params.get('imagery') === '0') {
    setSwitch('modelImagery', false);
    setImagery(false);
  }
  // 离线高倍段（manifest.offline_detail_imagery，z14+）。与 ?overlay= / ?terrain= 同一套写法：
  // 走 setSwitch，注册表意图与面板保持一致。加上它是因为**其它每一层都有 URL 钩子**，
  // 没有钩子这一层就无法在深链与自检里被单独开关（A/B 截图验证正是靠它）。
  if (params.has('detail')) {
    setSwitch('modelOfflineDetail', params.get('detail') !== '0');
    applyLayers();
  }
  if (params.get('terrain') === '1') {
    setSwitch('modelTerrain', true);
    setTerrain(true);
  }
  if (params.get('overlay') === '1') {
    setSwitch('modelWsf3dOverlay', true);
    setWsf3dOverlay(true);
  }
  if (params.get('sensors') === '1') {
    // ⚠ `modelSensor` 是**图层 id**，而它那一行**没有 `checkbox` 字段**。
    // `setSwitch` 按 **checkbox id** 反查图层（`layerIdForCheckbox`），于是反查为 null、
    // **静默空转**；而旁边直接调的 `setSensors(true)` 又会被随后的 `apply()` 按注册表
    // （desired 仍是 false）改回去。实测：`?sensors=1` 下 desired=True 的图层数
    // **20 -> 20 没变**，即参数完全没生效。
    //
    // （对比：旁边 `?overlay=1` 的 `wsf3d.overlay` **有** `checkbox: 'modelWsf3dOverlay'`，
    // 实测 False -> True，是好的。差别就在有没有那个字段。）
    //
    // 直接走注册表，与 `?overview=1` / 鸟瞰图×按钮两处修法一致。
    LayerRegistry.setDesired('modelSensor', true);
    LayerRegistry.apply();
  }
  if (params.get('weather') === '0') {
    setSwitch('modelWeather', false);
    setWeather(false);
  }
  if (params.get('weather') === '1') {
    setSwitch('modelWeather', true);
    setWeather(true);
  }
  if (params.get('buildings') === '1') {
    // 原先只写 `.checked`、**不调 setter**，靠 change 事件驱动。节点没了之后
    // 这一条等于什么都没做 —— 现在走 setSwitch，注册表的 handler 会把体块打开。
    setSwitch('modelBuildings', true);
  }
  // 诊断/取证用的机位钩子：?cam=lon,lat,height[,pitch]
  if (params.has('cam')) {
    const parts = params.get('cam').split(',').map(Number);
    if (parts.length >= 3 && parts.slice(0, 3).every(Number.isFinite)) {
      viewer.camera.cancelFlight();
      viewer.camera.setView({
        destination: Cesium.Cartesian3.fromDegrees(parts[0], parts[1], parts[2]),
        orientation: {
          heading: 0,
          pitch: Cesium.Math.toRadians(parts.length > 3 ? parts[3] : -70),
          roll: 0,
        },
      });
    }
  }
  // 建筑体块放最后：它按**当前机位**挑最近的城区 AOI 并按高度选 LOD，
  // 所以必须在 ?cam= 之后跑。之前放在机位钩子之前，取证时选到的是默认机位
  // 对应的城市，`?cam=` 定位到厦门却拿到台北的体块（或什么都不生成）。
  // 色带标定是异步的，取到之后再生成一次。
  if (params.get('buildings') === '1') {
    OfflineAssets.loadHeightRamp().then(function (ramp) {
      state.ramp = ramp;
      refreshBuildings();
    }).catch(function (e) {
      setSwitchNote('建筑体块不可用：' + e.message);
    });
  }
}

function applySceneProviders() {
  // 图层意图的真相在注册表里，不在 URL 里——所以 capture 直接读注册表，
  // 而不是回头去解析 ?scene=。defaults 用 HTML 的初始 checked，这样
  // "与默认相同"的层不会进 URL，链接天然更短。
  const defaults = {};
  LayerRegistry.LAYERS.forEach(function (l) {
    if (!l.checkbox) return;
    const box = document.getElementById(l.checkbox);
    if (box) defaults[l.id] = !!box.getAttribute('checked') || box.defaultChecked;
  });
  return {
    layerStatus: function () { return LayerRegistry.status(); },
    layerDefaults: defaults,
    tick: state.tick,
    caseId: state.caseId,
  };
}

/**
 * 从 ?scene= 还原整个场景。
 *
 * **必须在 applyCaseHooks() 之后调用**：clearSituation / applySituationData 会把机位
 * 重置为默认值、把图层意图写回初始状态，先应用会被立刻覆盖掉。
 */
function applySceneDoc() {
  if (!window.SceneDoc) return null;
  const doc = SceneDoc.fromUrl();
  if (!doc) return null;

  const out = { applied: true, camera: false, layers: 0, tick: null, caseId: doc.c || null };

  const cam = SceneDoc.saneCamera(doc.m);
  if (cam && viewer) {
    // 场景还原会覆盖机位；必须在解锁参考系后再 setView，否则 orientation
    // 会被按目标 ENU 解释（见 detachFollow 的纪律说明）。
    if (state.follow.id) detachFollow('场景还原机位');
    out.camera = SceneDoc.applyCamera(viewer, cam);
  }

  // 图层：只改"意图"，然后走一次统一的 apply，互斥与顺序仍由注册表解算。
  const l = doc.l;
  if (l && typeof l === 'object') {
    Object.keys(l).forEach(function (id) {
      if (!LayerRegistry.def(id)) return;      // 链接可能来自旧版本，忽略未知 id
      LayerRegistry.setDesired(id, !!l[id]);
      out.layers++;
    });
    applyLayers();
    // 复选框要与意图一致：syncLayerCheckboxes 只同步"实际生效"的层，而这里
    // 还要让用户看到自己没勾的层确实是没勾的（例如链接里关掉的在线实景）。
    LayerRegistry.LAYERS.forEach(function (def2) {
      if (!def2.checkbox) return;
      const box = document.getElementById(def2.checkbox);
      const r = LayerRegistry.runtime[def2.id];
      if (box && r) box.checked = !!r.desired;
    });
  }

  if (typeof doc.t === 'number' && doc.t >= 0) {
    setTick(doc.t);
    renderTransport();
    updateInferenceUi();
    out.tick = state.tick;
  }

  setStatus('已从链接还原场景（图层 ' + out.layers + ' 项'
    + (out.tick === null ? '' : '，时间轴 #' + out.tick) + '）', 'ok');
  markScene('applySceneDoc');
  return out;
}

/** 把当前场景变成可分享链接并复制到剪贴板。 */
function shareScene() {
  if (!window.SceneDoc) { setStatus('场景文档模块未加载', 'err'); return null; }
  const url = window.location.origin + SceneDoc.shareUrl(viewer, applySceneProviders());
  const done = function () { setStatus('场景链接已复制到剪贴板', 'ok'); };
  const fail = function () {
    // 剪贴板不可用（非 https / 无权限）时退化成把链接显示出来让用户手抄。
    setStatus('复制失败，链接：' + url, 'err');
  };
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(url).then(done, fail);
  } else {
    fail();
  }
  return url;
}

/** 依赖案例数据的钩子。必须在 loadCase 之后调用。 */
function applyCaseHooks() {
  // ?tick 必须先于 ?follow / ?panel：chase 的过渡终点要用该帧的航向，
  // 晚于定位执行会让深链按第 0 帧航向飞到反侧，再被阻尼甩回来。
  if (params.has('tick')) {
    const seek = Number(params.get('tick'));
    if (Number.isFinite(seek)) {
      setTick(seek);
      renderTargetCounts();
      renderTransport();
      updateInferenceUi();
    }
  }
  // 目标跟随：?follow=<目标ID>。放在这里是因为它需要 state.targets 已建好。
  if (params.has('follow')) {
    const fid = params.get('follow');
    if (state.targets[fid]) {
      if (dom.followTarget) dom.followTarget.checked = true;
      state.follow.enabled = true;
      beginTargetView(fid, { force: true });
      // ?unfollow=1 立刻解除一次，用于验证 detachFollow 里的参考系解锁
      // （camera.lookAt 会把 transform 锁在目标 ENU 上，必须显式还原成单位阵）。
      if (params.get('unfollow') === '1') detachFollow('验证解除');
    }
  }
  if (params.has('hotspot')) {
    const i = Number(params.get('hotspot'));
    if (state.data && Number.isFinite(i) && state.data.hotspots[i]) selectHotspot(i, true);
  }
  if (params.has('panel')) {
    const id = params.get('panel');
    if (state.targets[id]) {
      // 选中即居中；深链沿用"目标上方 120 km"的定位语义。
      selectTarget(id, true, 120000);
    }
  }
  if (params.get('autoplay') === '1') setPlaying(true);
}

// --------------------------------------------------------------------------
// 启动
// --------------------------------------------------------------------------

function main() {
  // 必须先读 manifest：底图 provider 的覆盖范围由它决定，否则会声明覆盖
  // 全球、为域外瓦片反复 404，并卡在最低层级不再细化。
  if (params.get('cache') === '0') {
    window.__situation3dNoTileCache = true;
  }

  OfflineAssets.loadManifest().then(function (m) {
    setStatus('初始化三维地球…（离线瓦片 ' + m.bbox.join('/') + '）');
    initViewer();
    // bindUI 提前到这里：它注册的 tileLoadProgressEvent / camera.changed /
    // preRender / ScreenSpaceEventHandler 都不依赖案例数据。放在 loadCase 之后
    // 会让磁盘预热、城区影像挂载、地图组件与跟随驱动全部被推迟到案例加载完。
    bindUI();

    // 色带标定按需加载，缺失不影响其它功能。
    OfflineAssets.loadHeightRamp().then(function (ramp) {
      state.ramp = ramp;
    }).catch(function () {
      state.ramp = null;
    });

    return Promise.all([loadBase(), loadCaseIndex()]);
  }).then(function (res) {
    applyBase(res[0]);

    // ?case= 优先；没有的话看 ?scene= 里带的案例 id——场景链接应当是自足的，
    // 打开就能回到"当时那个案例的那一帧"，不需要用户再补一个 ?case=。
    const sceneDoc = window.SceneDoc ? SceneDoc.fromUrl() : null;
    const wanted = params.get('case') || (sceneDoc && sceneDoc.c) || null;
    if (!wanted) {
      // 默认不加载任何TS：只保留底图 / 地形 / WSF3D / 地图组件。
      // 注意顺序：clearSituation 里的 applyCamera(默认机位) 会覆盖 ?cam=，而
      // applyCaseIndependentHooks（含 ?cam= 钩子）在它之后跑，所以空态下
      // ?cam= 能生效。这两步的顺序不能调换——调换后空态机位永远是默认值，
      // 取证时表现为"URL 里的坐标完全不起作用"（实测过：指定 121.57°E 却停在 2.1°E）。
      clearSituation();
    }
    applyCaseIndependentHooks();
    if (!wanted) return null;
    // 目录为空时 `default` 是 null —— 此时**不加载任何案例**，与"没有指定 case"
    // 走同一条空态路径，而不是拿 null 去 loadCase 报"未知案例"。
    const known = (state.caseIndex && state.caseIndex.cases) || [];
    const id = known.some(function (c) { return c.id === wanted; })
      ? wanted : (state.caseIndex.default || null);
    if (!id) {
      console.warn('[situations] 找不到案例「' + wanted + '」，且目录中没有默认案例');
      return null;
    }
    return loadCase(id);
  }).then(function () {
    applyCaseHooks();
    // 场景还原放最后：它会覆盖机位与图层意图，必须在案例装载把它们重置之后跑。
    state.sceneApplied = applySceneDoc();
    syncSelection();
    updateGeoReadout();
  }).catch(function (err) {
    console.error(err);
    setStatus('启动失败: ' + err.message, 'err');
  });

  setInterval(renderDiagnostics, 500);
  requestAnimationFrame(loop);
}

main();
