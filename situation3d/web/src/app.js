/* situation3d — 三维战场态势生成系统前端
 *
 * 职责：
 *   1. 加载引擎输出的逐帧态势与三级推理结果，在 Cesium 三维地球上回放；
 *   2. 用军标符号呈现海/陆/空三域的红蓝双方目标；
 *   3. 点击目标时在中间窗口右上角弹出信息面板（型号 / ID / 经纬高 / 三级态势）；
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

const state = {
  caseIndex: null,
  caseId: null,
  data: null,
  tick: 0,
  tickFloat: 0,
  playing: false,
  ticksPerSecond: 30,
  targets: {},              // id -> { def, entity, trail: [], lastTick, sensorEntity }
  layers: { targets: true, tracks: true, routes: true, hotspots: true, labels: true,
            levelBanner: true, targetPanel: true, overview: true },
  filters: { red: true, blue: true, sea: true, land: true, air: true },
  selected: null,
  hotspotEntities: [],
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
  globalLayer: null,        // 全球底图层（z0-6）
  hotspotLayers: [],        // 热点区域高清层（每框一个 provider）
  imageryStats: null,       // 底图瓦片请求统计（诊断用）
  inference: null,          // inference_frames，与 frames 逐帧对齐

  // 目标跟随。enabled 是图层管理里的开关，id 是当前被跟随的目标；
  // suspend 在用户按下鼠标时置位，让相机在拖拽期间完全交给用户。
  follow: { enabled: false, id: null, suspend: false, press: null, last: null, notified: false },

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

// base.json 缺失时的兜底机位（台海全景）
const FALLBACK_CAMERA = { lon: 119.6, lat: 24.4, height: 900000, heading: 0, pitch: -75 };

const dom = {};
['statusText', 'statusDot', 'scenarioName', 'targetList', 'hotspotBody',
 'geoLon', 'geoLat', 'geoRegion', 'geoSpan', 'geoLevel',
 'levelValue', 'levelNote', 'levelIntent', 'targetPanel', 'tpTitle', 'tpBody', 'tpClose',
 'menuCase', 'menuCaseList', 'menuLayers', 'menuLayerList', 'menuView', 'menuViewList',
 'switchNote', 'tickLabel', 'timeline', 'clockLabel', 'levelBanner', 'btnOverview',
 'countRed', 'countBlue', 'countSea', 'countLand', 'countAir', 'targetCountHint',
 'loadingBadge', 'loadingText',
 'mapCompass', 'compRose', 'mapViewpoint', 'mvLon', 'mvLat', 'mvAlt', 'modelPlaces',
 'modelTianditu', 'modelImagery', 'modelTerrain', 'modelBuildings',
 'mapScalebar', 'sbBar', 'sbLabel', 'followTarget', 'followNote',
].forEach(function (id) { dom[id] = document.getElementById(id); });

// --------------------------------------------------------------------------
// 瓦片加载进度
// --------------------------------------------------------------------------

let loadingSettleTimer = null;

/**
 * 显示/隐藏底图加载提示。
 *
 * Cesium 的瓦片细化是渐进的，底图会从粗到细逐级出现。没有提示时，这段
 * 时间看起来就像"卡住了"。这里用 tileLoadProgressEvent 的排队数驱动徽标，
 * 归零后再等一小段时间确认稳定，避免在两次细化之间反复闪烁。
 */
function setLoadingTiles(queued) {
  // 底图空闲时才让缓存开始预热：预热与当前视野的渲染共用同一批连接和 CPU，
  // 同时进行会把 Cesium 的影像请求挤掉，表现为底图一直画不出来。
  if (window.TileCache && TileCache.setIdle) TileCache.setIdle(queued === 0);

  // 底图空闲后再创建鸟瞰图：启动期两个 WebGL 上下文会互抢连接与 CPU。
  // 面板默认带 hidden，只有 Viewer 真正建好才显示，避免先出现一个白框。
  if (queued === 0 && state.layers.overview && window.Overview) {
    try { Overview.show(); } catch (e) { console.warn('[overview]', e.message); }
  }

  if (!dom.loadingBadge) return;
  if (loadingSettleTimer) {
    clearTimeout(loadingSettleTimer);
    loadingSettleTimer = null;
  }
  if (queued > 0) {
    dom.loadingText.textContent = '正在加载瓦片… 剩余 ' + queued;
    dom.loadingBadge.classList.remove('hidden');
    return;
  }
  loadingSettleTimer = setTimeout(function () {
    dom.loadingBadge.classList.add('hidden');
    loadingSettleTimer = null;
  }, 600);
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
  lines.push('imagery requests ' + im.requests + '  errors ' + im.errors);
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
               '  preload ' + tc.preloaded + '  warming ' + tc.warmingCount +
               '  idle ' + tc.idleFlag);
    if (tc.lastError) lines.push('cache error     ' + String(tc.lastError).slice(0, 120));
  }
  // 跟随与参考系状态。unlocked 必须是 true：camera.lookAt 会把 camera.transform
  // 锁在目标 ENU 上，而 setView / flyTo 不会自动解锁，锁定状态下缩放与贴地钳制
  // 会失效（可以穿到地面以下）。解除跟随必须让这一项回到 true。
  lines.push('follow          ' + (state.follow.id || '-') +
             '  enabled ' + state.follow.enabled +
             '  suspend ' + state.follow.suspend +
             '  unlocked ' + viewer.camera.transform.equals(Cesium.Matrix4.IDENTITY));
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
function checkboxOn(id) {
  const node = document.getElementById(id);
  return !!(node && node.checked);
}

function setStatus(text, kind) {
  dom.statusText.textContent = text;
  dom.statusDot.className = 'dot' + (kind ? ' ' + kind : '');
}

function setSwitchNote(text) {
  dom.switchNote.textContent = text;
}

function el(tag, className, html) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (html !== undefined) node.innerHTML = html;
  return node;
}

function colorFor(affiliation) {
  return affiliation === 'red'
    ? Cesium.Color.fromCssColorString('#d0342c')
    : Cesium.Color.fromCssColorString('#1f5fbf');
}

function levelColor(level) {
  return ['#7d8a99', '#4a9c6d', '#d8a12a', '#d9772a', '#c8352c'][level] || '#7d8a99';
}

function fmtDeg(v, pos, neg) {
  if (v === null || v === undefined || !isFinite(v)) return '—';
  return Math.abs(v).toFixed(4) + '°' + (v >= 0 ? pos : neg);
}

// --------------------------------------------------------------------------
// 视图初始化
// --------------------------------------------------------------------------

function initViewer() {
  // Cesium 1.107 起 Viewer 的 imageryProvider 选项被移除且静默忽略，
  // 必须用 baseLayer，否则球体只剩默认底色。下面的断言把静默退化变成显式失败。
  //
  // 图层顺序（index 0 在最底层，Cesium 由下往上叠）：
  //   [0] 天地图影像（在线，默认底图）
  //   [1] 天地图中文注记（在线）
  //   [2] 离线全球 z0-6  →  [3] 战区 z0-11  →  [4..] 热点 z0-12  →  城区 z13-17
  // 天地图与"离线影像底图"是**两套互斥的底图方案**（都铺满视野，同时开会互相
  // 遮挡），由 setTianditu/setImagery 互相对冲；热点/城区/专题层始终叠在最上面，
  // 与用哪套底图无关。
  state.imageryStats = OfflineAssets.newStats('imagery');
  state.tiandituLayer = new Cesium.ImageryLayer(
    OfflineAssets.createTiandituProvider('img', 'w'));
  state.tiandituLabelLayer = new Cesium.ImageryLayer(
    OfflineAssets.createTiandituLabelProvider('w'));

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

  if (viewer.imageryLayers.length === 0) {
    throw new Error('底图图层未创建：Cesium ' + Cesium.VERSION + ' 需要 baseLayer 选项');
  }

  // 注记层紧贴天地图影像之上（index 1）。
  viewer.imageryLayers.add(state.tiandituLabelLayer);
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

  // 热点层：每个热点框一个 provider，到 z12。三亚/冲绳/关岛完全在战区之外，
  // 只能靠这一层；与战区重叠的部分因 URL 相同会命中 TileCache 内存缓存。
  const hboxes = (OfflineAssets.manifest().hotspot_imagery || {}).boxes || [];
  state.hotspotLayers = OfflineAssets.attachHotspotImagery(viewer, hboxes);

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
        // 直接走内部实现：若改走 setBuildings 会再写一次 desired，形成回环。
        applyBuildingsVisibility(want);
      } else if (id === 'tiles.google3d') {
        setOnlineTilesetVisible(want);
      }
    },
    label: function (id, want) {
      if (id === 'places.labels' && window.Places) Places.setEnabled(!!want);
    },
  });

  // 常驻 imagery 槽位登记（动态层在挂载时登记，见 registerImagerySlot）。
  registerImagerySlot('tianditu.base', state.tiandituLayer);
  registerImagerySlot('tianditu.label', state.tiandituLabelLayer);
  // 离线底图是**两级**：全球层 z0-6 + 战区层 z7-11，共用一个 imagery.offline 意图。
  // 两个都要登记——只登记全球层的话，战区层会留在顺序之外，反而压到注记上面。
  registerImagerySlot('imagery.offline', [state.globalLayer, state.imageryLayer]);
  // 热点影像是**一层多实例**（每个热点框一个 provider），必须全部登记。
  registerImagerySlot('hotspot.imagery', state.hotspotLayers);

  // 初始意图：与 index.html 里各复选框的 checked 一致。HTML 的 checked 只是初始
  // 显示，不会触发 change，所以必须在这里显式同步一次，否则注册表会以为全都关着。
  LayerRegistry.setDesired('tianditu.base', checkboxOn('modelTianditu'));
  LayerRegistry.setDesired('tianditu.label', checkboxOn('modelTiandituLabel'));
  LayerRegistry.setDesired('imagery.offline', checkboxOn('modelImagery'));
  LayerRegistry.setDesired('hotspot.imagery', checkboxOn('modelHotspotImagery'));
  LayerRegistry.setDesired('wsf3d.overlay', checkboxOn('modelWsf3dOverlay'));
  LayerRegistry.setDesired('buildings.blocks', checkboxOn('modelBuildings'));
  LayerRegistry.setDesired('tiles.google3d', checkboxOn('modelBuildingsOnline'));
  LayerRegistry.setDesired('places.labels', checkboxOn('modelPlaces'));
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
}

function flyTo(lon, lat, height, pitchDeg) {
  // 先解除跟随：camera.lookAt 会把 camera.transform 锁在目标 ENU 上，而
  // setView / flyTo 不会自动解锁，orientation 会被按那个局部系解释。
  detachFollow('视角跳转');
  viewer.camera.flyTo({
    destination: Cesium.Cartesian3.fromDegrees(lon, lat, height),
    orientation: { heading: 0, pitch: Cesium.Math.toRadians(pitchDeg === undefined ? -70 : pitchDeg), roll: 0 },
    duration: 1.2,
  });
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

function loadCaseIndex() {
  return fetch('./data/cases.json', { cache: 'no-store' })
    .then(function (r) {
      if (!r.ok) throw new Error('案例索引加载失败: HTTP ' + r.status);
      return r.json();
    })
    .then(function (index) {
      state.caseIndex = index;
      renderCaseMenu();
      return index;
    });
}

function loadCase(caseId) {
  const entry = state.caseIndex.cases.find(function (c) { return c.id === caseId; });
  if (!entry) throw new Error('未知案例: ' + caseId);

  setStatus('加载态势数据…');
  return fetch('./' + entry.file, { cache: 'no-store' })
    .then(function (r) {
      if (!r.ok) throw new Error('态势数据加载失败: HTTP ' + r.status);
      return r.json();
    })
    .then(function (data) { return applySituationData(data, caseId); });
}

/**
 * 把一份态势数据装载进场景。
 *
 * 从 loadCase 里拆出来，是为了让**前端生成的态势**（态势输入）走完全同一条管线：
 * 取数据与装载数据分开，两条来源共用装载逻辑，就不会出现"案例能跑、自建态势
 * 少初始化了某样东西"的偏差。
 */
function applySituationData(data, caseId) {
  {
      if (!data.frames || !data.frames.length) throw new Error('态势数据为空');

      teardownScene();

      state.data = data;
      state.caseId = caseId || data.case_id || null;
      state.inference = data.inference_frames || null;
      state.tick = 0;
      state.tickFloat = 0;
      state.selected = null;
      state.activeHotspot = -1;
      state.geoPinned = false;
      state.buildingsAoi = -1;
      markScene('applySituationData');
      // 数据集自带本体的压缩形态，优先用它（离线可用），接口只是兜底。
      if (window.IntentUi) IntentUi.adoptFromModel(state.data && state.data.inference_model);

      if (data.symbology) Symbology.setCatalog(data.symbology);

  // 城区 AOI 以 base.json 为准（与案例无关）；旧数据集里也带一份，作为回退。
  if (!state.urbanAoi || !state.urbanAoi.length) {
    state.urbanAoi = data.urban_aoi || [];
  }
  syncUrbanImagery();

      dom.scenarioName.textContent = data.region + ' · ' + data.case_name;
      document.title = '三维战场态势生成系统 — ' + data.case_name;

      createHotspots();
      createRouteEntities();
      createTargetEntities();
      applyCamera(data.camera);

      renderHotspotTable();
      renderTargetList();
      renderTransport();
      closeTargetPanel();
      setTick(0);
      updateInferenceUi();

      setTransportEnabled(true);
      if (dom.followTarget) {
        dom.followTarget.disabled = false;
        dom.followTarget.title = '';
      }
      setFollowNote('开启后点击任一目标，相机以随航向旋转的追踪视角持续跟随；拖动地图即自动解除。');
      setStatus('就绪 · ' + data.targets.length + ' 个目标 · ' +
                (data.frames.length - 1) + ' 帧 · ' + data.hotspots.length + ' 个热点区域', 'ok');
      setSwitchNote('已加载「' + data.case_name + '」。各开关的生效状态会在此说明。');
      if (window.Overview && window.Overview.setCenter) {
        const cam = data.camera || {};
        if (isFinite(cam.lon) && isFinite(cam.lat)) Overview.setCenter(cam.lon, cam.lat);
      }
      // 装载完态势后补一次建筑体块。refreshBuildings() 原本只挂在"选中热点 /
      // 解除跟随 / 目标高度变化"上，于是一进页面就带 `?buildings=1`（或先开开关
      // 再选案例）时体块不会出现——除非用户再去点一个热点。色带未就绪时
      // `?buildings=1` 的回调会补这一次。
      if (checkboxOn('modelBuildings') && state.ramp) refreshBuildings();
      return data;
  }
}

function teardownScene() {
  if (!viewer) return;
  detachFollow('态势已切换');
  closeTargetPanel();
  viewer.entities.removeAll();
  state.targets = {};
  state.hotspotEntities = [];
  state.hotspotRows = [];
  state.routeEntities = [];
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
// 空态（不加载任何态势）下前端仍需要两样东西：城区 AOI（生成 WSF3D 建筑体块）
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
  state.baseRegion = (base && base.region) || '未加载态势';
  syncUrbanImagery();
}

// --------------------------------------------------------------------------
// 空态与态势生命周期
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
 * 回到空态：不加载任何态势，只保留底图 / 地形 / WSF3D / 地图组件。
 *
 * 幂等——启动时与「清除当前态势」共用同一条路径，避免两套清场逻辑漂移。
 */
function clearSituation() {
  if (!viewer) return;
  setPlaying(false);
  if (state.buildingTimer) { clearTimeout(state.buildingTimer); state.buildingTimer = null; }
  teardownScene();                     // 内含 detachFollow + 清空实体

  state.data = null;
  state.caseId = null;
  state.inference = null;
  state.tick = 0;
  state.tickFloat = 0;
  state.selected = null;
  state.activeHotspot = -1;
  state.geoPinned = false;
  state.buildingsKey = null;
  state.buildingsAoi = -1;
  markScene('clearSituation');

  dom.scenarioName.textContent = state.baseRegion || '未加载态势';
  document.title = '三维战场态势生成系统';

  renderTargetList();
  renderHotspotTable();
  renderTransport();
  updateInferenceUi();
  clearHotspotSelection();
  closeTargetPanel();
  setTransportEnabled(false);

  if (dom.followTarget) {
    dom.followTarget.disabled = true;
    dom.followTarget.checked = false;
    dom.followTarget.title = '未加载态势';
  }
  setFollowNote('未加载态势，无法跟随。请先从「演示案例」选择案例。');

  // 机位兜底：base.json 缺失时也要回到台海，否则会停在 Cesium 的默认视图
  // （DEFAULT_VIEW_RECTANGLE 的中心在 82.5°W / 35°N，看起来像"相机飞走了"）。
  applyCamera(state.baseCamera || FALLBACK_CAMERA);
  setStatus('空态 · 未加载态势', 'ok');
  setSwitchNote('未加载态势：底图 / 地形 / WSF3D / 地图组件仍可用。请从顶栏「演示案例」选择案例。');
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
    const sub = el('span', 'mi-sub',
      c.target_count + ' 个目标 · ' + c.hotspot_count + ' 个热点区域');
    const desc = el('span', 'mi-desc', c.description);
    // 描述在菜单里只显示一行（CSS 截断），完整文字放 title，hover 可看全。
    if (c.description) item.title = c.name + '：' + c.description;
    item.appendChild(head);
    item.appendChild(sub);
    item.appendChild(desc);

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

  // 清除当前态势，回到空态（底图/地形/WSF3D/地图组件保留）
  const sep = el('div', 'menu-sep');
  list.appendChild(sep);
  const clearItem = el('button', 'menu-item menu-item-quiet', '');
  clearItem.type = 'button';
  clearItem.setAttribute('role', 'menuitem');
  clearItem.appendChild(el('span', 'mi-head', '清除当前态势'));
  clearItem.appendChild(el('span', 'mi-sub', '回到空态：保留底图 / 地形 / WSF3D / 地图组件'));
  clearItem.addEventListener('click', function () {
    closeCaseMenu();
    if (!state.data) return;
    clearSituation();
  });
  list.appendChild(clearItem);
}

const MENUS = [];

/** 顶栏下拉菜单的通用开合。三个菜单互斥，点空白处全部收起。 */
function bindMenu(buttonId, listId) {
  const btn = document.getElementById(buttonId);
  const list = document.getElementById(listId);
  if (!btn || !list) return;
  MENUS.push({ btn: btn, list: list });
  btn.addEventListener('click', function (e) {
    e.stopPropagation();
    const wasOpen = list.classList.contains('open');
    closeAllMenus();
    if (!wasOpen) {
      list.classList.add('open');
      btn.setAttribute('aria-expanded', 'true');
    }
  });
}

function closeAllMenus() {
  MENUS.forEach(function (m) {
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
  { name: '态势图层', ids: ['layerHotspots', 'layerRoutes', 'layerLevelBanner'] },
  { name: '目标图层', ids: ['layerTargets', 'layerLabels', 'layerTracks', 'modelSensor',
                            'layerTargetPanel', 'filterRed', 'filterBlue', 'filterSea',
                            'filterLand', 'filterAir'] },
  { name: '底图', ids: ['modelTianditu', 'modelTiandituLabel', 'modelImagery',
                        'modelHotspotImagery', 'modelUrbanImagery',
                        'modelPlaces', 'modelTerrain',
                        'modelWsf3dOverlay', 'modelBuildings', 'modelWeather'] },
  { name: '地图组件', ids: ['compCompass', 'compScalebar', 'compViewpoint'] },
];

const LAYER_DEFAULTS = {};

function captureLayerDefaults() {
  LAYER_GROUPS.forEach(function (g) {
    g.ids.forEach(function (id) {
      const node = document.getElementById(id);
      if (node) LAYER_DEFAULTS[id] = node.checked;
    });
  });
}

/** 以编程方式切换开关，并派发 change 事件复用既有 handler。 */
function setSwitch(id, on) {
  const node = document.getElementById(id);
  if (!node || node.disabled || node.checked === on) return;
  node.checked = on;
  node.dispatchEvent(new Event('change', { bubbles: true }));
}

/**
 * cmd: 'show' 全开 | 'hide' 只关态势与目标图层 | 'default' 恢复默认。
 *
 * 「全部隐藏」刻意不关底图与地图组件——把底图关掉只会得到一片黑球，不是用户
 * 想要的结果。
 */
function applyLayerCommand(cmd) {
  const groups = cmd === 'hide'
    ? LAYER_GROUPS.filter(function (g) { return g.name === '态势图层' || g.name === '目标图层'; })
    : LAYER_GROUPS;
  groups.forEach(function (g) {
    g.ids.forEach(function (id) {
      setSwitch(id, cmd === 'show' ? true : cmd === 'hide' ? false : !!LAYER_DEFAULTS[id]);
    });
  });
  setStatus(cmd === 'show' ? '图层已全部显示'
    : cmd === 'hide' ? '态势与目标图层已隐藏'
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
  if (def.affiliation === 'red' && !state.filters.red) return false;
  if (def.affiliation === 'blue' && !state.filters.blue) return false;
  if (!state.filters[def.domain]) return false;
  return true;
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
        width: 34,
        height: 34,
        verticalOrigin: Cesium.VerticalOrigin.CENTER,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
        scaleByDistance: new Cesium.NearFarScalar(20000, 1.5, 2000000, 0.5),
      },
      label: {
        text: def.id + ' ' + def.platform_class,
        font: '12px "Microsoft YaHei", "Noto Sans CJK SC", sans-serif',
        fillColor: Cesium.Color.fromCssColorString('#1d2430'),
        outlineColor: Cesium.Color.WHITE,
        outlineWidth: 3,
        style: Cesium.LabelStyle.FILL_AND_OUTLINE,
        pixelOffset: new Cesium.Cartesian2(0, -24),
        verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
        showBackground: true,
        backgroundColor: Cesium.Color.fromCssColorString('rgba(255,255,255,0.86)'),
        scaleByDistance: new Cesium.NearFarScalar(20000, 1.0, 2000000, 0.0),
      },
    });

    // 历史轨迹：截至当前帧，目标已经走过的航迹。
    entity.polyline = {
      positions: new Cesium.CallbackProperty(function () {
        const rec = state.targets[def.id];
        return rec ? rec.trail : [];
      }, false),
      width: 2,
      material: color.withAlpha(0.85),
      clampToGround: false,
    };

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

    state.targets[def.id] = {
      def: def,
      entity: entity,
      sensorEntity: sensorEntity,
      trail: [],
      lastTick: -1,
    };
  });
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

function createHotspots() {
  state.data.hotspots.forEach(function (hs, i) {
    const w = hs.bbox[0], s = hs.bbox[1], e = hs.bbox[2], n = hs.bbox[3];
    const centerLat = hs.center[0], centerLon = hs.center[1];

    // 热点区域按真实经纬范围绘制成矩形，而不是一个点 —— 面板展示的是"范围"，
    // 一个点无法表示一条海峡或一个群岛。
    const entity = viewer.entities.add({
      id: 'hotspot-' + i,
      name: hs.name,
      rectangle: {
        coordinates: Cesium.Rectangle.fromDegrees(w, s, e, n),
        material: Cesium.Color.fromCssColorString('#f0a020').withAlpha(0.10),
        outline: true,
        outlineColor: Cesium.Color.fromCssColorString('#f0a020').withAlpha(0.85),
        outlineWidth: 1,
        height: 0,
      },
      position: Cesium.Cartesian3.fromDegrees(centerLon, centerLat, 0),
      label: {
        text: hs.name,
        font: '12px "Microsoft YaHei", "Noto Sans CJK SC", sans-serif',
        fillColor: Cesium.Color.fromCssColorString('#8a5a00'),
        outlineColor: Cesium.Color.WHITE,
        outlineWidth: 3,
        style: Cesium.LabelStyle.FILL_AND_OUTLINE,
        pixelOffset: new Cesium.Cartesian2(0, 0),
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
    });
    state.hotspotEntities.push(entity);
  });
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

function syncEntities() {
  Object.keys(state.targets).forEach(function (id) {
    const rec = state.targets[id];
    const def = rec.def;
    const visible = isVisible(def);

    rec.entity.show = visible;
    rec.entity.label.show = visible && state.layers.labels;
    rec.entity.billboard.show = visible;
    rec.sensorEntity.show = visible && state.filters.sensors === true;

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
    rec.entity.polyline.show = visible && state.layers.tracks && rec.trail.length > 1;
  });

  state.routeEntities.forEach(function (e) { e.show = state.layers.routes; });
  state.hotspotEntities.forEach(function (e) { e.show = state.layers.hotspots; });
}

function syncSelection() {
  Object.keys(state.targets).forEach(function (id) {
    const rec = state.targets[id];
    rec.entity.billboard.image = Symbology.symbolUri(
      rec.def.icon, rec.def.affiliation, { selected: state.selected === id });
  });
}

// --------------------------------------------------------------------------
// 三级态势推理的界面呈现
// --------------------------------------------------------------------------

function inferenceAt(tick) {
  if (!state.inference) return null;
  return state.inference[Math.max(0, Math.min(tick, state.inference.length - 1))];
}

/**
 * 当前帧出现最多的主意图（全局）。横幅用它回答"战场上在打什么"。
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
  if (!inf) {
    dom.levelValue.textContent = '—';
    dom.levelNote.textContent = '未加载推理结果';
    return;
  }

  const lv = inf.overall.level;
  dom.levelValue.textContent = lv + ' 级 · ' + inf.overall.level_label;
  dom.levelValue.className = 'lb-value ' + (LEVEL_CLASS[lv] || '');
  dom.levelNote.textContent = inf.overall.note +
    ' 平均威胁 ' + inf.overall.mean_threat + '，最高 ' + inf.overall.max_threat +
    '，对抗域 ' + inf.overall.engaged_pairs + ' 个。';

  // 区域级横幅补上"主导意图"：等级回答"多紧张"，意图回答"在打什么"，
  // 两者并列才算把区域态势说全。
  if (dom.levelIntent) {
    const top = regionIntentCounts(inf);
    dom.levelIntent.textContent = top.length
      ? top.map(function (x) { return x.zh + ' ×' + x.n; }).join('、')
      : '无带意图判定的目标';
  }

  // 热点区域表格里的态势等级与意图随之刷新。
  state.hotspotRows.forEach(function (row, i) {
    const r = inf.regions[i];
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

  if (state.selected) renderTargetPanel(state.selected);
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
      };
    });
  }
  const summary = state.data && state.data.intent_summary;
  return {
    ontology: window.IntentUi ? IntentUi.debug() : null,
    modelHasOntology: !!(state.data && state.data.inference_model
      && state.data.inference_model.intent_ontology),
    hasData: !!state.data,
    tick: state.tick,
    hasInferenceFrame: !!inf,
    targetsThisFrame: per,
    datasetRuleHits: summary ? summary.rule_hits : null,
    datasetStatusCounts: summary ? summary.status_counts : null,
    underivableCount: summary && summary.underivable ? summary.underivable.length : null,
    panelHasIntentBlock: !!(dom.tpBody && dom.tpBody.querySelector
      && dom.tpBody.querySelector('.tp-intent')),
  };
}

// --------------------------------------------------------------------------
// 目标列表
// --------------------------------------------------------------------------

function renderTargetList() {
  const host = dom.targetList;
  host.innerHTML = '';

  const counts = { red: 0, blue: 0, sea: 0, land: 0, air: 0 };
  let shown = 0;
  const inf = inferenceAt(state.tick);
  const targets = (state.data && state.data.targets) || [];

  targets.forEach(function (def) {
    counts[def.affiliation]++;
    counts[def.domain]++;
    if (!isVisible(def)) return;
    shown++;

    const s = positionFor(def.id, state.tick);
    const l1 = inf && inf.targets[def.id];

    const div = el('div', 'target-item ' + def.affiliation +
      (state.selected === def.id ? ' active' : ''));

    const head = el('div', 'ti-head');
    head.appendChild(el('span', 'ti-id', def.id));
    head.appendChild(el('span', 'ti-name', def.name));
    if (l1) {
      head.appendChild(el('span', 'ti-threat ' + (LEVEL_CLASS[l1.alert + 1] || ''),
        '威胁 ' + l1.threat));
    }

    const sub = el('div', 'ti-sub',
      (s ? fmtDeg(s.lat, 'N', 'S') + ' ' + fmtDeg(s.lon, 'E', 'W') : '—') +
      (s ? ' · ' + s.speed_kn + ' kn · ' + s.heading.toFixed(0) + '°' : ''));

    const meta = el('div', 'ti-meta',
      (def.model || def.platform_class) + ' · ' + def.role +
      (l1 ? ' · ' + l1.state_label : ''));

    div.appendChild(head);
    div.appendChild(sub);
    div.appendChild(meta);
    div.addEventListener('click', function () { selectTarget(def.id, true); });
    host.appendChild(div);
  });

  dom.countRed.textContent = counts.red;
  dom.countBlue.textContent = counts.blue;
  dom.countSea.textContent = counts.sea;
  dom.countLand.textContent = counts.land;
  dom.countAir.textContent = counts.air;
  dom.targetCountHint.textContent = shown + ' / ' + targets.length;

  if (!targets.length) {
    host.appendChild(el('div', 'empty-state', '未加载态势。请从顶栏「演示案例」选择案例。'));
  }
}

// --------------------------------------------------------------------------
// 热点区域表格
// --------------------------------------------------------------------------

function renderHotspotTable() {
  const body = dom.hotspotBody;
  body.innerHTML = '';
  state.hotspotRows = [];
  const hotspots = (state.data && state.data.hotspots) || [];

  if (!hotspots.length) {
    const tr = el('tr', 'hotspot-row');
    const td = el('td', 'hs-empty', '未加载态势');
    td.colSpan = 5;
    tr.appendChild(td);
    body.appendChild(tr);
    return;
  }

  hotspots.forEach(function (hs, i) {
    const tr = el('tr', 'hotspot-row');
    tr.dataset.index = String(i);

    const levelCell = el('td', 'lv-cell', '—');
    // 主导意图列：区域级"这里在打什么"，与区域等级并列。
    const intentCell = el('td', 'hs-intent', '—');
    tr.appendChild(el('td', 'hs-idx', String(i + 1)));
    tr.appendChild(el('td', 'hs-name', hs.name));
    // 经纬范围直接取自 bbox，与地图上绘制的矩形同源，不会再出现"显示范围与
    // 真实区域不一致"的问题。
    tr.appendChild(el('td', 'hs-range',
      hs.bbox[0].toFixed(2) + '–' + hs.bbox[2].toFixed(2) + '°E<br>' +
      hs.bbox[1].toFixed(2) + '–' + hs.bbox[3].toFixed(2) + '°N'));
    tr.appendChild(levelCell);
    tr.appendChild(intentCell);

    tr.addEventListener('click', function () { selectHotspot(i, true); });
    body.appendChild(tr);
    state.hotspotRows.push({ row: tr, levelCell: levelCell, intentCell: intentCell, hotspot: hs });
  });
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
 * 锁定到热点范围（中心经纬、经纬跨度、区域态势等级），直到用户点击
 * "位置信息"按钮回到光标跟随模式。
 */
function selectHotspot(index, fly) {
  const hs = state.data.hotspots[index];
  if (!hs) return;
  detachFollow('热点区域跳转');

  state.activeHotspot = index;
  state.geoPinned = true;

  dom.geoRegion.textContent = hs.name;
  dom.geoLon.textContent = fmtDeg(hs.center[1], 'E', 'W') +
    '（' + hs.bbox[0].toFixed(2) + ' – ' + hs.bbox[2].toFixed(2) + '）';
  dom.geoLat.textContent = fmtDeg(hs.center[0], 'N', 'S') +
    '（' + hs.bbox[1].toFixed(2) + ' – ' + hs.bbox[3].toFixed(2) + '）';
  dom.geoSpan.textContent = hs.span_deg[0].toFixed(2) + '° × ' + hs.span_deg[1].toFixed(2) + '°';

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
    dom.geoSpan.textContent = (b[2] - b[0]).toFixed(2) + '° × ' + (b[3] - b[1]).toFixed(2) + '°';
    const inf = inferenceAt(state.tick);
    if (inf && inf.regions[idx]) {
      const r = inf.regions[idx];
      dom.geoLevel.textContent = r[0] + ' ' + levelLabel(r[0]);
      dom.geoLevel.className = 'lv-cell ' + (LEVEL_CLASS[r[0]] || '');
    }
  } else {
    dom.geoRegion.textContent = state.data ? '不在热点区域内' : (state.baseRegion || '未加载态势');
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
// 目标信息面板（中间窗口右上角）
// --------------------------------------------------------------------------

function selectTarget(id, fly) {
  state.selected = id;
  const s = positionFor(id, state.tick);
  if (state.follow.enabled) {
    // 跟随开启时，点击目标是"换一个跟随对象"，不能再走 flyTo —— 两者会互抢相机
    startFollow(id);
  } else if (fly && s) {
    flyTo(s.lon, s.lat, 180000);
  }
  syncSelection();
  renderTargetList();
  renderTargetPanel(id);
  markScene('selectTarget');
}

function closeTargetPanel() {
  state.selected = null;
  dom.targetPanel.classList.add('hidden');
  syncSelection();
  markScene('closeTargetPanel');
}

function kvRow(label, value, extraClass) {
  return '<div class="tp-kv"><span>' + label + '</span><b' +
    (extraClass ? ' class="' + extraClass + '"' : '') + '>' + value + '</b></div>';
}

function renderTargetPanel(id) {
  // 「目标元数据面板」开关关掉后，点击目标只做选中与高亮，不弹面板。
  if (!state.layers.targetPanel) { dom.targetPanel.classList.add('hidden'); return; }
  const rec = state.targets[id];
  if (!rec) return;
  const def = rec.def;
  const s = positionFor(id, state.tick);
  const inf = inferenceAt(state.tick);
  const l1 = inf ? inf.targets[id] : null;
  const group = inf ? inf.groups.find(function (g) {
    return g.affiliation === def.affiliation && g.domain === def.domain;
  }) : null;

  // 该目标所在的热点区域（取包含它的第一个），用于展示区域级态势。
  let regionIdx = -1;
  if (s) {
    state.data.hotspots.forEach(function (hs, i) {
      if (regionIdx >= 0) return;
      if (s.lat >= hs.bbox[1] && s.lat <= hs.bbox[3] &&
          s.lon >= hs.bbox[0] && s.lon <= hs.bbox[2]) regionIdx = i;
    });
  }
  const region = regionIdx >= 0 ? state.data.hotspots[regionIdx] : null;
  const r3 = regionIdx >= 0 && inf ? inf.regions[regionIdx] : null;

  dom.tpTitle.textContent = def.id + ' ' + def.name;

  let html = '';
  html += '<div class="tp-symbol">' +
    '<img src="' + Symbology.symbolUri(def.icon, def.affiliation, { selected: true }) + '" alt="军标">' +
    '<div class="tp-symbol-meta"><b>' + def.model + '</b><span>' + def.model_note + '</span>' +
    '<span>' + Symbology.iconLabel(def.icon) + ' · ' +
    (def.affiliation === 'red' ? '红方' : '蓝方') + ' · ' +
    (Symbology.domainLabel(def.domain) || def.domain) + '</span></div></div>';

  // 意图摘要**置顶**：ZHY 关注的是"这个目标在干什么"。以前意图只是「一级」里的
  // 一行，要读完一堆属性才看到；现在它排在最前，一/二/三级明细仍在下方按原样给出
  // （未删减，只是不再抢视线）。
  if (window.IntentUi) {
    html += IntentUi.renderIntentBlock(l1, def);
  }

  // 目标要素：保留但降为一个折叠区。意图摘要已经占了最上方，属性/一/二/三级明细
  // 仍在原处、内容未删减，只是不再与意图抢第一眼。
  html += '<details class="tp-fold" open><summary>目标要素</summary>';
  html += kvRow('目标 ID', def.id);
  html += kvRow('型号', def.model);
  html += kvRow('平台类别', def.platform_class + ' · ' + def.role);
  html += kvRow('经度', s ? fmtDeg(s.lon, 'E', 'W') : '—');
  html += kvRow('纬度', s ? fmtDeg(s.lat, 'N', 'S') : '—');
  html += kvRow('高度', s ? s.alt.toFixed(0) + ' m' : '—');
  html += kvRow('航向 / 航速', s ? s.heading.toFixed(0) + '° / ' + s.speed_kn + ' ' + def.speed_unit : '—');
  html += kvRow('传感器', def.sensor_name + '（' + def.sensor_range_km.toFixed(0) + ' km）');
  html += '</details>';

  if (l1) {
    const terms = l1.threat_terms;
    html += '<details class="tp-fold" open><summary>一级 · 目标级态势</summary>';
    html += kvRow('状态', l1.state_label);
    html += kvRow('威胁度', l1.threat + ' / 100', 'lv-' + (l1.alert + 1));
    html += kvRow('告警等级', l1.alert_label, 'lv-' + (l1.alert + 1));
    html += kvRow('最近对方目标',
      l1.nearest_id ? (l1.nearest_id + ' · ' + l1.nearest_km + ' km') : '—');
    html += kvRow('被对方探测', l1.covered_by + ' 部传感器');
    html += '<div class="tp-reason">判定依据：装备权重 ' + terms.base +
      ' + 接近度 ' + terms.proximity + ' + 覆盖度 ' + terms.coverage + '</div>';
    html += '</details>';
  }

  if (group) {
    html += '<details class="tp-fold"><summary>二级 · 编组级态势</summary>';
    html += kvRow('所属编组', (group.affiliation === 'red' ? '红方' : '蓝方') +
      (Symbology.domainLabel(group.domain) || group.domain) + '编组（' + group.count + ' 个目标）');
    html += kvRow('编队形态', group.posture_label + '（跨度 ' + group.spread_km + ' km）');
    html += kvRow('对方同域编组', group.opposing_key || '无');
    html += kvRow('最小距离', group.min_distance_km === null ? '—' : group.min_distance_km + ' km');
    html += kvRow('对抗关系', group.relation_label, 'lv-' + (group.relation + 1));
    html += '</details>';
  }

  html += '<details class="tp-fold"><summary>三级 · 区域级态势</summary>';
  if (region && r3) {
    html += kvRow('所在区域', region.name);
    html += kvRow('区域等级', r3[0] + ' 级 · ' + levelLabel(r3[0]), 'lv-' + r3[0]);
    html += kvRow('兵力对比', '红方 ' + r3[1] + ' / 蓝方 ' + r3[2]);
    html += kvRow('主导方', codeLabel('dominance_codes', r3[3]));
  } else {
    html += kvRow('所在区域', '不在已登记的热点区域内');
  }
  if (inf) {
    html += kvRow('全局态势', inf.overall.level + ' 级 · ' + inf.overall.level_label,
      'lv-' + inf.overall.level);
  }
  html += '</details>';

  html += '<details class="tp-fold"><summary>装备参数</summary>';
  Object.keys(def.attributes || {}).forEach(function (k) {
    html += kvRow(k, def.attributes[k]);
  });
  html += '</details>';

  dom.tpBody.innerHTML = html;
  dom.targetPanel.classList.remove('hidden');
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
    viewer.terrainProvider = new Cesium.EllipsoidTerrainProvider();
    viewer.scene.globe.depthTestAgainstTerrain = false;
    state.terrainOn = false;
    setSwitchNote('地形模型已关闭：使用椭球面，高程不参与渲染。');
  }
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
      if (!alreadyLocal) {
        const box = document.getElementById('modelBuildings');
        if (box) box.checked = true;
      }
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

/** 把在线层状态同步到令牌输入框下方的提示，以及离线体块的可用性感知。 */
function syncBuildingsOnlineNote(info) {
  const note = document.getElementById('ionTokenNote');
  if (!note) return;
  const phaseText = {
    idle: '未启用',
    unconfigured: '未配置令牌',
    loading: '正在连接…',
    live: '已连接',
    failed: '连接失败',
  }[info.phase] || info.phase;
  note.textContent = '状态：' + phaseText +
    (info.detail ? '（' + info.detail + '）' : '');
  const box = document.getElementById('ionTokenInput');
  if (box) box.classList.toggle('bad', info.phase === 'failed');
}

/**
 * 建筑体块的**实际显隐**（注册表的 primitive sink 调它）。
 *
 * 纯"把状态落到场景上"，不碰 desired——desired 由 setBuildings 写。
 * 两者分开是为了避免回环：sink 若再调 setBuildings，就会又写一次 desired。
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
      const box = document.getElementById('modelBuildings');
      if (box) box.checked = false;
    });
    return;
  }
  refreshBuildings();
}

/**
 * 建筑体块开关：只写「意图」，实际显隐交给注册表解算。
 *
 * 以前的写法是直接改场景（show / buildingsKey / 重建），于是"在线实景开着的时候
 * 体块该不该显示"这类**跨图层**的判断只能散在各个 setter 里手工维护。现在意图与
 * 落地分开，互斥规则集中在 layerRegistry.js。
 */
function setBuildings(on) {
  LayerRegistry.setDesired('buildings.blocks', !!on);
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

var healthDirty = true;

/** 标脏并（在面板展开时）立刻重渲染。 */
function markHealthDirty() {
  healthDirty = true;
  var el = document.getElementById('healthPanel');
  if (el && el.open) renderHealthPanel();
}

/**
 * 把仲裁结果同步回复选框。
 *
 * 互斥会**替用户关掉**一个层（例如打开离线底图会把天地图影像顶掉）。这时复选框
 * 若还停在勾选状态，界面就在撒谎——用户看到天地图勾着、画面却是离线影像。
 * 这里只改 .checked，**不派发 change**，否则会再走一遍 setDesired 造成回环。
 */
function syncLayerCheckboxes() {
  LayerRegistry.status().forEach(function (s) {
    if (!s.checkbox) return;
    const box = document.getElementById(s.checkbox);
    if (box && box.checked !== !!s.effective) box.checked = !!s.effective;
  });
}

/**
 * 渲染"数据健康"面板。
 *
 * 这个面板只回答一个问题：**"我以为开着的图层，到底有没有真的在显示"**。
 * 所以正文只列 attention（想开但没生效）的条目，其余图层收在下面的计数里；
 * 全部正常时给一行"全部正常"，不制造噪音。
 */
function renderHealthPanel() {
  const body = document.getElementById('healthBody');
  const sum = document.getElementById('healthSummary');
  if (!body || !window.DataHealth) return;

  const r = DataHealth.report();
  if (sum) sum.textContent = r.healthy ? '全部正常' : (r.attention.length + ' 项待查');

  const rows = [];
  if (!r.attention.length) {
    rows.push('<div class="health-ok">全部图层均按开关生效。</div>');
  } else {
    r.attention.forEach(function (a) {
      rows.push('<div class="health-item"><b>' + a.label + '</b>' +
        '<span class="health-feed">' + a.feed + '</span>' +
        '<div class="health-detail">' + (a.detail || '已开但未生效') + '</div></div>');
    });
  }

  // 摘要行：让"看起来正常但其实有问题"的数字浮出来（瓦片错误率、重叠丢弃率）。
  const t = r.tiles || {};
  const p = r.places || {};
  const g = r.governor || {};
  const bits = [];
  if (t.totalRequests) {
    bits.push('影像请求 ' + t.totalRequests + (t.totalErrors ? '（错误 ' + t.totalErrors + '）' : ''));
  }
  if (t.sources && t.sources[0] && t.sources[0].maxLevel >= 0) {
    bits.push('影像层级 ' + t.sources[0].minLevel + '–' + t.sources[0].maxLevel);
  }
  if (p.loaded) bits.push('地名 ' + p.placed + '/' + p.candidates + '（重叠丢弃 ' + p.overlapped + '）');
  if (r.buildings && r.buildings.buildingsStats) {
    const bs = r.buildings.buildingsStats;
    bits.push('体块 ' + bs.blocks + ' 个 / LOD ' + bs.lodKey);
  }
  if (g.skipRatio !== null && g.skipRatio !== undefined) {
    bits.push('渲染跳过率 ' + (g.skipRatio * 100).toFixed(1) + '%');
  }
  if (r.viewport) bits.push('视高 ' + r.viewport.heightKm + ' km');
  rows.push('<div class="health-meta">' + bits.join(' · ') + '</div>');

  body.innerHTML = rows.join('');
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
  const box = document.getElementById('modelBuildings');
  if (!box || !box.checked) return;
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
  const want = state.urbanEnabled && height <= URBAN_IMAGERY_TRIGGER_M;

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
  // 与天地图的互斥**不再在这里手工对冲**：LAYERS 里 imagery.offline 与
  // tianditu.base 互相列进 exclusives，由注册表解算谁让位。
  // 注记层依旧不受影响——注记是"文字覆盖"而不是底图，用户可以要离线影像 +
  // 天地图注记。
  LayerRegistry.setDesired('imagery.offline', !!on);
  applyLayers();
  setSwitchNote(on
    ? '离线影像底图已启用：全球/战区层（离线 Esri World Imagery）。'
    : '离线影像底图已关闭。');
}

/**
 * 天地图注记层开关。
 *
 * 单独一个开关的理由：注记是**栅格**，会盖住热点/城区的高清影像（栅格叠栅格
 * 没有"谁更重要"这回事，只有顺序）。有些场景（例如要看城市细节）不想要注记，
 * 所以给它独立开关，默认开。
 *
 * 顺序上注记在 LAYERS 里是 z=90，排在所有底图/高清影像之上——这是它**应该**
 * 在的位置（文字压在影像上才符合直觉）；以前是被热点影像盖住之后用一次
 * raiseToTop 救回来的，现在由声明直接保证。
 */
function setTiandituLabel(on) {
  LayerRegistry.setDesired('tianditu.label', !!on);
  applyLayers();
  setSwitchNote(on ? '天地图注记已显示（置于所有影像层之上）。'
                   : '天地图注记已隐藏。');
}

/**
 * 天地图底图开关（默认开）。
 *
 * 天地图影像与它的中文注记是**两个 layer**：影像本身不含地名，不叠 `cva` 注记层
 * 地图上一个字都没有。所以打开这个开关会同时打开注记；注记另有一个独立开关
 * （setTiandituLabel）用来在需要时单独关掉。
 */
function setTianditu(on) {
  LayerRegistry.setDesired('tianditu.base', !!on);
  // 注记跟随影像一起开：不叠 cva 层，天地图上就一个字都没有，用户会以为地名层坏了。
  // 但关闭影像时**不动注记**——用户可能就是想"离线影像 + 天地图注记"。
  if (on) LayerRegistry.setDesired('tianditu.label', true);
  applyLayers();
  const key = OfflineAssets.tiandituKey ? OfflineAssets.tiandituKey() : '';
  setSwitchNote(on
    ? '天地图已启用（在线影像 + 中文注记，key …' + String(key).slice(-6) +
      '）。离线影像底图已暂时关闭，可在同组里切回。'
    : '天地图影像已关闭（注记层保持当前状态）。');
}

function setHotspotImagery(on) {
  LayerRegistry.setDesired('hotspot.imagery', !!on);
  applyLayers();
  const n = (state.hotspotLayers || []).length;
  setSwitchNote(on
    ? '热点区域高清影像已启用：' + n + ' 个热点框（z0–12，约 19 m/px）。'
    : '热点区域高清影像已关闭。');
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
// 三项都与案例无关：只用 viewer 与 state.components，因此未加载态势时同样可用。

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
 */
function updateCompass() {
  if (!dom.compRose) return;
  const deg = Cesium.Math.toDegrees(cameraOrientation().headingRad);
  dom.compRose.style.transform = 'rotate(' + (-deg) + 'deg)';
}

/** 视点坐标：相机自身的经纬高。高度是椭球高（WGS84），不是海拔。 */
function updateViewpoint() {
  if (!dom.mvLon) return;
  const c = viewer.camera.positionCartographic;
  const lon = Cesium.Math.toDegrees(c.longitude);
  const lat = Cesium.Math.toDegrees(c.latitude);
  dom.mvLon.textContent = fmtDeg(lon, 'E', 'W');
  dom.mvLat.textContent = fmtDeg(lat, 'N', 'S');
  dom.mvAlt.textContent = c.height < 1000
    ? c.height.toFixed(0) + ' m'
    : (c.height / 1000).toFixed(c.height < 100000 ? 2 : 1) + ' km';
  dom.mvAlt.title = '椭球高（WGS84 椭球面），非海拔';
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
 * 比例尺：在**比例尺实际绘制处**取两个屏幕点量地面距离，换算每像素米数，
 * 再按 1/2/5×10ⁿ 取整到人类友好的刻度。
 *
 * 取样点不能用屏幕中心：3D 倾斜视图下近端与远端的每像素米数可差数倍，
 * 用中心值会让画在左上的横条与实际地面长度对不上。
 * 探针跨度固定 100 px（与待求的 sbBar 宽度解耦），单趟即可算出结果。
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

  const raw = mPerPx * SCALE_PROBE_PX;
  const pow = Math.pow(10, Math.floor(Math.log10(raw)));
  const n = raw / pow;
  const nice = n < 1.5 ? 1 : n < 3.5 ? 2 : n < 7.5 ? 5 : 10;
  const meters = nice * pow;

  dom.sbBar.style.width = Math.max(24, Math.round(meters / mPerPx)) + 'px';
  dom.sbLabel.textContent = meters >= 1000
    ? (meters / 1000).toFixed(meters % 1000 === 0 ? 0 : 1) + ' km'
    : Math.round(meters) + ' m';
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
  const box = document.getElementById('layerOverview');
  if (box) box.checked = !!on;
  const btn = dom.btnOverview;
  if (btn) btn.classList.toggle('primary', !!on);

  if (!on) { Overview.hide(); return; }
  if (!window.Overview) return;
  try {
    Overview.show();
    setSwitchNote('鸟瞰图已显示：真实地球缩略图，黄框为主相机当前视野。点击鸟瞰图可跳转主视角。');
  } catch (e) {
    state.layers.overview = false;
    if (box) box.checked = false;
    setSwitchNote('鸟瞰图不可用：' + e.message);
  }
}

function setComponent(name, on) {
  state.components[name] = !!on;
  const node = document.getElementById(COMPONENT_NODES[name]);
  if (node) node.classList.toggle('hidden', !on);
  if (on && viewer) updateMapComponents();
}

// --------------------------------------------------------------------------
// 目标跟随
// --------------------------------------------------------------------------
//
// 用 camera.lookAt + HeadingPitchRange 逐帧驱动，不用 viewer.trackedEntity：
// 后者的 entity.viewFrom **只在首帧生效**（跟踪中改视角必须解绑重绑），且 3D
// 跟踪会强制 enableTilt = false，做不到"随航向旋转 + 俯仰可控"。
//
// HeadingPitchRange 的语义（由 1.145 源码 offsetFromHeadingPitchRange 推出）：
//   offset_ENU = ( -r·cos p·sin h, -r·cos p·cos h, -r·sin p )
//   ⇒ heading 是**相机视线方位角**，相机**位置**方位角 = heading + 180°。
// 所以要让相机位于目标的右后上方：heading = 目标航向 - 偏角。

const FOLLOW_VIEW = {
  sea:  { range:  3500, pitchDeg: -22, rightDeg: 30 },   // 高于目标约 1.3 km
  land: { range:  4000, pitchDeg: -30, rightDeg: 30 },   // 高于目标约 2.0 km，俯视阵地
  air:  { range: 15000, pitchDeg: -18, rightDeg: 30 },   // 高于目标约 4.6 km
};

const _followHpr = new Cesium.HeadingPitchRange(0, 0, 0);

function setFollowNote(text) {
  if (dom.followNote) dom.followNote.textContent = text;
}

/** 被跟随目标是否仍然可跟随（存在、可见、有位置）。 */
function followTargetUsable(id) {
  const rec = state.targets[id];
  if (!rec) return false;
  if (!isVisible(rec.def)) return false;
  return !!positionFor(id, state.tick);
}

/**
 * 每帧更新跟随相机。挂在 scene.preRender 上，本帧生效。
 *
 * 目标静止（暂停、或已到航路终点）时直接返回：既省掉每帧的 lookAt，
 * 也不再触发 camera.changed，避免无谓的级联刷新。
 */
function applyFollowFrame(force) {
  const f = state.follow;
  if (!f.enabled || !f.id || f.suspend || !viewer || !state.data) return;

  const rec = state.targets[f.id];
  const s = rec ? positionFor(f.id, state.tick) : null;
  if (!rec || !s || !isVisible(rec.def)) { detachFollow('目标已隐藏'); return; }

  if (s.reached_end && !f.notified) {
    f.notified = true;
    setStatus('目标 ' + f.id + ' 已到达航路终点，跟随保持', 'ok');
  }

  const pos = Cesium.Cartesian3.fromDegrees(s.lon, s.lat, s.alt);
  if (!force && f.last && Cesium.Cartesian3.distance(pos, f.last) < 0.5) return;
  if (!f.last) f.last = new Cesium.Cartesian3();
  Cesium.Cartesian3.clone(pos, f.last);

  const v = FOLLOW_VIEW[rec.def.domain] || FOLLOW_VIEW.sea;
  _followHpr.heading = Cesium.Math.toRadians((s.heading || 0) - v.rightDeg);
  _followHpr.pitch = Cesium.Math.toRadians(v.pitchDeg);
  _followHpr.range = v.range;
  viewer.camera.lookAt(pos, _followHpr);
}

function startFollow(id) {
  const f = state.follow;
  if (!f.enabled || !state.targets[id] || !state.data) return;
  f.id = id;
  f.suspend = false;
  f.press = null;
  f.last = null;
  f.notified = false;
  viewer.camera.cancelFlight();      // 取消可能在飞的 flyTo，否则两者互相覆盖
  applyFollowFrame(true);
  setStatus('正在跟随 ' + id + ' · 拖动地图可解除', 'ok');
  setFollowNote('正在跟随 ' + id + '。拖动地图或滚轮缩放即自动解除。');
}

/**
 * 解除跟随。
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
  f.suspend = false;
  f.press = null;
  f.last = null;
  f.notified = false;
  f.enabled = false;
  if (dom.followTarget) dom.followTarget.checked = false;
  if (viewer) viewer.camera.lookAtTransform(Cesium.Matrix4.IDENTITY);
  if (was) {
    setStatus('已解除目标跟随（' + was + '）' + (reason ? '：' + reason : ''), 'ok');
    setFollowNote('已解除跟随。勾选后可再次点击目标跟随。');
    // 跟随期间 camera.changed 每帧触发，建筑体块的防抖被反复重置而从不执行；
    // 解除后补一次，让当前机位下的体块 LOD 归位。
    if (checkboxOn('modelBuildings')) refreshBuildings();
  }
}

/** 过滤/显隐变化后检查跟随是否仍然成立。 */
function guardFollow() {
  if (state.follow.id && !followTargetUsable(state.follow.id)) {
    detachFollow('目标已隐藏或过滤');
  }
}

/**
 * "拖动即解除"的检测。
 *
 * 判据：跟随代码只改相机状态、**从不派发 DOM 事件**，所以画布上任何
 * pointer/wheel 事件都来自用户。不能用 camera.changed 或 moveStart/moveEnd——
 * 它们内部会比较 camera.transform，而跟随每帧都在改 transform，会被持续触发。
 *
 * 按下即 suspend（拖拽第一像素起相机就归用户），抬起时若位移未超阈值则恢复
 * 跟随，这样"点击另一个目标"不会被误判成拖动。
 */
const DRAG_THRESHOLD_PX = 4;

function bindFollowInput() {
  const canvas = viewer.scene.canvas;

  canvas.addEventListener('pointerdown', function (e) {
    if (!state.follow.id) return;
    state.follow.press = { x: e.clientX, y: e.clientY };
    state.follow.suspend = true;
    // 不 stopPropagation：ScreenSpaceCameraController 仍需收到事件
  }, true);

  window.addEventListener('pointermove', function (e) {
    const press = state.follow.press;
    if (!press) return;
    if (Math.abs(e.clientX - press.x) + Math.abs(e.clientY - press.y) < DRAG_THRESHOLD_PX) return;
    state.follow.press = null;
    detachFollow('拖动地图');
  }, true);

  window.addEventListener('pointerup', function () {
    if (!state.follow.press) return;
    state.follow.press = null;
    state.follow.suspend = false;   // 未超阈值 ⇒ 是点击，恢复跟随
  }, true);

  canvas.addEventListener('wheel', function () {
    detachFollow('滚轮缩放');
  }, { capture: true, passive: true });
}

// 供 overview.js 等其他模块调用（它们拿不到本文件的闭包）
window.Follow = {
  start: startFollow,
  detach: detachFollow,
  isActive: function () { return !!state.follow.id; },
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
  markScene('setTick');
}

function setPlaying(on) {
  state.playing = on;
  document.getElementById('btnPlay').textContent = on ? '❚❚ 暂停' : '▶ 播放';
  document.getElementById('btnRun').textContent = on ? '推演中…' : '态势自动推演';
}

function loop(nowMs) {
  const dt = lastFrameMs ? (nowMs - lastFrameMs) / 1000 : 0;
  lastFrameMs = nowMs;

  if (!viewer) { requestAnimationFrame(loop); return; }

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
      renderTargetList();
      renderTransport();
      updateInferenceUi();
      // 播放推进也是一次状态变化：不标记的话画面会停在原地（只在拖动时间轴时更新）。
      markScene('playback-tick');
    }
  }

  if (RenderGovernor.take()) syncEntities();

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
 * 态势管理里增删目标后要调它：目标实体是按 data.targets 建的，改了数据必须重建，
 * 但不需要重新取数据，也不该走 applySituationData（那会重置帧号与机位）。
 */
function reloadSituation() {
  if (!state.data || !viewer) return;
  detachFollow('态势已更新');
  closeTargetPanel();
  viewer.entities.removeAll();
  state.targets = {};
  state.hotspotEntities = [];
  state.hotspotRows = [];
  state.routeEntities = [];
  state.selected = null;

  createHotspots();
  createRouteEntities();
  createTargetEntities();
  renderHotspotTable();
  renderTargetList();
  renderTransport();
  updateInferenceUi();
  setStatus('态势已更新 · ' + state.data.targets.length + ' 个目标', 'ok');
}

/** 全帧 CSV：逐帧逐目标展开，便于在表格软件里做轨迹分析。 */
function exportCsv() {
  if (!state.data || !state.data.frames) {
    setStatus('未加载态势，无可导出数据', 'err');
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

// 态势模块（src/situation.js）拿不到本文件的闭包，用这个接口回调。
window.SituationHost = {
  state: state,
  viewer: function () { return viewer; },
  // 取证用：建筑体块的短路条件逐条报出来。`?buildings=1` 不生效时，
  // "到底卡在哪个 return"光看结果猜不出来。
  // 注意这里**从 DOM 现取**复选框，不用 dom 缓存：实测两者会不一致
  // （DOM 上是 true、dom.modelBuildings.checked 是 false），拿缓存的引用去判断
  // "开关开没开"会把已经打开的开关读成关的。
  debugBuildings: function () {
    const box = document.getElementById('modelBuildings');
    const aoiIdx = nearestAoiToCamera();
    return {
      checked: !!(box && box.checked),
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
      onlineNote: (function () {
        const n = document.getElementById('ionTokenNote');
        return n ? n.textContent : null;
      })(),
    };
  },
  setStatus: setStatus,
  // 图层顺序取证：imagery 的真实 index 顺序 + 每个图层的意图/事实。
  // 「哪个层压在哪个层上面」光看画面猜不出来，必须能读到 index。
  debugLayers: debugLayers,
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
};

function exportData() {
  if (!state.data || !state.data.frames) {
    setStatus('未加载态势，无可导出数据', 'err');
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

  bindCheck('layerTargets', function (v) { state.layers.targets = v; markScene('layerTargets'); guardFollow(); });
  bindCheck('layerTracks', function (v) { state.layers.tracks = v; markScene('layerTracks'); });
  bindCheck('layerRoutes', function (v) { state.layers.routes = v; markScene('layerRoutes'); });
  bindCheck('layerHotspots', function (v) { state.layers.hotspots = v; markScene('layerHotspots'); });
  bindCheck('layerLabels', function (v) { state.layers.labels = v; markScene('layerLabels'); });

  const refilter = function () { renderTargetList(); markScene('filter'); guardFollow(); };
  bindCheck('filterRed', function (v) { state.filters.red = v; refilter(); });
  bindCheck('filterBlue', function (v) { state.filters.blue = v; refilter(); });
  bindCheck('filterSea', function (v) { state.filters.sea = v; refilter(); });
  bindCheck('filterLand', function (v) { state.filters.land = v; refilter(); });
  bindCheck('filterAir', function (v) { state.filters.air = v; refilter(); });

  bindCheck('layerOverview', setOverview);

  if (dom.btnOverview) {
    dom.btnOverview.disabled = false;
    dom.btnOverview.addEventListener('click', function () { setOverview(!state.layers.overview); });
  }
  const ovHide = document.getElementById('ovHide');
  if (ovHide) ovHide.addEventListener('click', function () { setOverview(false); });

  bindCheck('layerLevelBanner', function (v) {
    state.layers.levelBanner = v;
    if (dom.levelBanner) dom.levelBanner.classList.toggle('hidden', !v);
  });
  bindCheck('layerTargetPanel', function (v) {
    state.layers.targetPanel = v;
    if (!v) closeTargetPanel();
  });
  bindCheck('modelPlaces', function (v) {
    LayerRegistry.setDesired('places.labels', !!v);
    applyLayers();
    setSwitchNote(v ? '地名图层已启用：Natural Earth 离线地名，按相机高度分级显示并做屏幕空间去重。'
                    : '地名图层已关闭。');
  });
  // 地名是懒加载的：初始 checked 不会触发 change，但 initViewer 里已经写过一次
  // desired 并 apply 过（label sink → Places.setEnabled），这里不再重复调用。
  // （原先这段相同代码写了两遍，属于历史重复，一并去掉。）

  bindCheck('modelHotspotImagery', setHotspotImagery);

  bindCheck('modelUrbanImagery', function (v) {
    state.urbanEnabled = v;
    LayerRegistry.setDesired('urban.imagery', !!v);
    // 城区层不是"改了 show 就完事"：它按相机高度动态挂载/卸载，所以要走
    // syncUrbanImagery；里面挂载后会自动 apply，让新层排到注记之下。
    syncUrbanImagery();
    applyLayers();
    setSwitchNote(v ? '城区高清影像已启用（相机低于 200 km 时挂载，z13–z17，最高约 1.2 m/px）。'
                    : '城区高清影像已关闭。');
  });

  bindCheck('compCompass', function (v) { setComponent('compass', v); });
  bindCheck('compScalebar', function (v) { setComponent('scalebar', v); });
  bindCheck('compViewpoint', function (v) { setComponent('viewpoint', v); });

  bindCheck('followTarget', function (v) {
    state.follow.enabled = !!v;
    if (!v) {
      detachFollow('');
    } else if (!state.data) {
      state.follow.enabled = false;
      dom.followTarget.checked = false;
      setFollowNote('未加载态势，无法跟随。请先从「演示案例」选择案例。');
    } else {
      setFollowNote('已开启跟随：点击任一目标开始跟随，拖动地图即解除。');
      setStatus('目标跟随已开启 · 点击目标开始跟随', 'ok');
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

  bindCheck('modelTerrain', setTerrain);
  bindCheck('modelWsf3dOverlay', setWsf3dOverlay);
  bindCheck('modelBuildings', setBuildings);
  bindCheck('modelBuildingsOnline', setBuildingsOnline);
  bindCheck('modelWeather', setWeather);
  bindCheck('modelSensor', setSensors);
  bindCheck('modelImagery', setImagery);
  bindCheck('modelTianditu', setTianditu);
  bindCheck('modelTiandituLabel', setTiandituLabel);
  // 天地图是默认底图：HTML 的 checked 只是初始显示，change 不会自动触发，
  // 而 initViewer 里已经把它设成 baseLayer，所以这里不需要再调用一次；
  // 但把提示文案写清楚，否则用户不知道当前底图是在线还是离线。
  if (dom.modelTianditu && dom.modelTianditu.checked) setTianditu(true);

  // 数据健康面板：展开时才算、才渲染；点"重新检查"强制刷新。
  const healthPanel = document.getElementById('healthPanel');
  if (healthPanel) {
    healthPanel.addEventListener('toggle', function () {
      if (healthPanel.open && healthDirty) { renderHealthPanel(); healthDirty = false; }
    });
  }
  const healthRefresh = document.getElementById('healthRefresh');
  if (healthRefresh) {
    healthRefresh.addEventListener('click', function () {
      healthDirty = false;
      renderHealthPanel();
    });
  }

  const shareBtn = document.getElementById('btnShareScene');
  if (shareBtn) shareBtn.addEventListener('click', function () { shareScene(); });

  // ion 令牌输入：保存到服务端 web/data/ion.json（该文件被 gitignore）。
  // 输入框里的值**不回显**到任何提示里，保存成功后只报"已保存"。
  const ionSave = document.getElementById('ionTokenSave');
  const ionInput = document.getElementById('ionTokenInput');
  if (ionSave && ionInput && window.Buildings3D) {
    ionSave.addEventListener('click', function () {
      const note = document.getElementById('ionTokenNote');
      const v = (ionInput.value || '').trim();
      if (note) note.textContent = '正在保存…';
      window.Buildings3D.saveToken(v).then(function () {
        ionInput.value = '';
        ionInput.classList.remove('bad');
        ionInput.classList.add('ok');
        if (note) note.textContent = '令牌已保存。重开「在线三维实景建筑」即可生效。';
        // 令牌换了以后旧的失败状态不该继续留着，否则会一直显示上一次的报错。
        window.Buildings3D.forgetToken();
        if (checkboxOn('modelBuildingsOnline')) setBuildingsOnline(true);
      }).catch(function (e) {
        ionInput.classList.remove('ok');
        ionInput.classList.add('bad');
        if (note) note.textContent = '保存失败：' + (e && e.message ? e.message : e);
      });
    });
  }

  // 在线实景是默认图层（index.html 里 checked）：HTML 的 checked 同样不会触发
  // change，而它又不像天地图那样在 initViewer 里已经建好，所以必须显式跑一次。
  // 提示文案由 setBuildingsOnline 的成功/回落分支负责写。
  if (checkboxOn('modelBuildingsOnline')) setBuildingsOnline(true);

  // 三个顶栏下拉菜单统一开合，互斥，点空白处收起。
  bindMenu('menuCase', 'menuCaseList');
  bindMenu('menuSituation', 'menuSituationList');
  bindMenu('menuLayers', 'menuLayerList');
  bindMenu('menuView', 'menuViewList');
  if (dom.menuCase) dom.menuCase.addEventListener('click', function () { openCaseMenu(); });
  document.addEventListener('click', function () { closeAllMenus(); });

  // 态势菜单：输入 / 管理 / 数据下载，三标签页共用一个模态框
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
    if (e.key === 'Escape') { closeCaseMenu(); closeTargetPanel(); }
  });

  dom.tpClose.addEventListener('click', closeTargetPanel);

  document.getElementById('btnPlay').addEventListener('click', function () { setPlaying(!state.playing); });
  document.getElementById('btnRun').addEventListener('click', function () { setPlaying(!state.playing); });
  document.getElementById('btnReset').addEventListener('click', function () {
    setPlaying(false);
    setTick(0);
    Object.keys(state.targets).forEach(function (id) {
      state.targets[id].trail.length = 0;
      state.targets[id].lastTick = -1;
    });
    renderTargetList();
    renderTransport();
    updateInferenceUi();
  });

  document.getElementById('speedSelect').addEventListener('change', function (e) {
    state.ticksPerSecond = Number(e.target.value);
  });

  dom.timeline.addEventListener('input', function (e) {
    setPlaying(false);
    setTick(Number(e.target.value));
    renderTargetList();
    renderTransport();
    updateInferenceUi();
  });

  document.getElementById('btnStepBack').addEventListener('click', function () {
    setPlaying(false); setTick(state.tick - 1); renderTargetList(); renderTransport(); updateInferenceUi();
  });
  document.getElementById('btnStepFwd').addEventListener('click', function () {
    setPlaying(false); setTick(state.tick + 1); renderTargetList(); renderTransport(); updateInferenceUi();
  });


  // "位置信息"：回到光标跟随模式，同时把视角拉回案例默认机位。
  document.getElementById('btnLocate').addEventListener('click', function () {
    clearHotspotSelection();
    const cam = (state.data && state.data.camera) || state.baseCamera;
    if (cam) applyCamera(cam);
  });

  captureLayerDefaults();

  bindPicking();
  bindFollowInput();

  // 跟随相机挂在 preRender：Scene.render 的顺序是 preUpdate → update →
  // preRender → draw，在这里改相机本帧生效。
  viewer.scene.preRender.addEventListener(function () { applyFollowFrame(false); });

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
  viewer.scene.globe.tileLoadProgressEvent.addEventListener(setLoadingTiles);

  viewer.camera.changed.addEventListener(function () {
    camChangedCount++;
    const now = Date.now();
    if (now - lastGeoUpdate < 250) return;
    lastGeoUpdate = now;
    updateGeoReadout();
    syncUrbanImagery();
    guardFollow();
    if (window.Overview) Overview.onMainCameraChanged();
    if (window.Places) Places.onCameraChanged(false);
    // 相机高度跨过 LOD 阈值时才重建，且加一段防抖，避免飞行过程中反复触发。
    // 跟随期间相机每帧都在动，防抖会被反复重置而**永不执行**，所以直接跳过；
    // detachFollow() 收尾时会补一次 refreshBuildings()。
    if (!state.follow.id && checkboxOn('modelBuildings')) {
      if (state.buildingTimer) clearTimeout(state.buildingTimer);
      state.buildingTimer = setTimeout(refreshBuildings, 400);
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

  handler.setInputAction(function (click) {
    const picked = viewer.scene.pick(click.position);
    const id = picked && picked.id && typeof picked.id.id === 'string' ? picked.id.id : null;
    if (!id) return;

    if (id.indexOf('target-') === 0) {
      selectTarget(id.slice('target-'.length), false);
    } else if (id.indexOf('hotspot-') === 0) {
      selectHotspot(Number(id.slice('hotspot-'.length)), false);
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
  // 态势面板：?situation=input|manage|download（取证与深链用）
  if (params.has('situation') && window.SituationPanel) {
    SituationPanel.open(params.get('situation') || 'input');
  }

  // 态势面板：?situation=input|manage|download（取证与深链用）
  if (params.has('situation') && window.SituationPanel) {
    SituationPanel.open(params.get('situation') || 'input');
  }

  // 鸟瞰图：?overview=0|1
  if (params.has('places')) {
    const on = params.get('places') !== '0';
    const box = document.getElementById('modelPlaces');
    if (box) box.checked = on;
    if (window.Places) Places.setEnabled(on);
  }

  if (params.has('overview')) setOverview(params.get('overview') !== '0');

  // 地图组件：?components=0|1 全关/全开；?components=a,b 白名单；?components=-a 排除
  if (params.has('components')) {
    const raw = params.get('components');
    if (raw === '0') {
      ['compass', 'scalebar', 'viewpoint'].forEach(function (n) { setComponent(n, false); });
      if (dom.compCompass) dom.compCompass.checked = false;
      if (dom.compScalebar) dom.compScalebar.checked = false;
      if (dom.compViewpoint) dom.compViewpoint.checked = false;
    } else if (raw === '1' || raw === '') {
      ['compass', 'scalebar', 'viewpoint'].forEach(function (n) { setComponent(n, true); });
    } else {
      const parts = raw.split(',').map(function (x) { return x.trim(); }).filter(Boolean);
      const exclude = parts.length && parts[0].charAt(0) === '-';
      ['compass', 'scalebar', 'viewpoint'].forEach(function (n) {
        const on = exclude ? parts.indexOf('-' + n) < 0 : parts.indexOf(n) >= 0;
        setComponent(n, on);
        const box = document.getElementById('comp' + n.charAt(0).toUpperCase() + n.slice(1));
        if (box) box.checked = on;
      });
    }
  }
  [['compass', 'compass'], ['scalebar', 'scalebar'], ['viewpoint', 'viewpoint']].forEach(function (pair) {
    if (!params.has(pair[0])) return;
    const on = params.get(pair[0]) !== '0';
    setComponent(pair[1], on);
    const box = document.getElementById('comp' + pair[0].charAt(0).toUpperCase() + pair[0].slice(1));
    if (box) box.checked = on;
  });

  if (params.get('imagery') === '0') {
    document.getElementById('modelImagery').checked = false;
    setImagery(false);
  }
  if (params.get('terrain') === '1') {
    document.getElementById('modelTerrain').checked = true;
    setTerrain(true);
  }
  if (params.get('overlay') === '1') {
    document.getElementById('modelWsf3dOverlay').checked = true;
    setWsf3dOverlay(true);
  }
  if (params.get('sensors') === '1') {
    document.getElementById('modelSensor').checked = true;
    setSensors(true);
  }
  if (params.get('weather') === '0') {
    document.getElementById('modelWeather').checked = false;
    setWeather(false);
  }
  if (params.get('weather') === '1') {
    document.getElementById('modelWeather').checked = true;
    setWeather(true);
  }
  if (params.get('buildings') === '1') {
    document.getElementById('modelBuildings').checked = true;
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
  if (cam && viewer) out.camera = SceneDoc.applyCamera(viewer, cam);

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
  // 目标跟随：?follow=<目标ID>。放在这里是因为它需要 state.targets 已建好。
  if (params.has('follow')) {
    const fid = params.get('follow');
    if (state.targets[fid]) {
      if (dom.followTarget) dom.followTarget.checked = true;
      state.follow.enabled = true;
      startFollow(fid);
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
      selectTarget(id, false);
      const s = positionFor(id, state.tick);
      if (s) flyTo(s.lon, s.lat, 120000);
    }
  }
  if (params.has('tick')) {
    const seek = Number(params.get('tick'));
    if (Number.isFinite(seek)) {
      setTick(seek);
      renderTargetList();
      renderTransport();
      updateInferenceUi();
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
      // 默认不加载任何态势：只保留底图 / 地形 / WSF3D / 地图组件。
      // 注意顺序：clearSituation 里的 applyCamera(默认机位) 会覆盖 ?cam=，而
      // applyCaseIndependentHooks（含 ?cam= 钩子）在它之后跑，所以空态下
      // ?cam= 能生效。这两步的顺序不能调换——调换后空态机位永远是默认值，
      // 取证时表现为"URL 里的坐标完全不起作用"（实测过：指定 121.57°E 却停在 2.1°E）。
      clearSituation();
    }
    applyCaseIndependentHooks();
    if (!wanted) return null;
    const id = state.caseIndex.cases.some(function (c) { return c.id === wanted; })
      ? wanted : state.caseIndex.default;
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
