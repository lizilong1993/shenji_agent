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
  layers: { targets: true, tracks: true, routes: true, hotspots: true, labels: true },
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
  buildingTimer: null,
  imageryLayer: null,
  cursorCarto: null,        // 光标处的地面点，用于"地理位置信息"读数
  lastError: null,          // 最近一次未捕获异常
  urbanLayers: null,        // 城区 z12-z14 高清影像层（按相机高度挂载）
  urbanAoi: [],             // 城区 AOI 列表
  imageryStats: null,       // 底图瓦片请求统计（诊断用）
  inference: null,          // inference_frames，与 frames 逐帧对齐
};

let viewer = null;
let lastFrameMs = 0;
let lastGeoRefresh = 0;

const dom = {};
['statusText', 'statusDot', 'scenarioName', 'targetList', 'hotspotBody',
 'geoLon', 'geoLat', 'geoAlt', 'geoRegion', 'geoSpan', 'geoLevel',
 'levelValue', 'levelNote', 'targetPanel', 'tpTitle', 'tpBody', 'tpClose',
 'menuCase', 'menuCaseList', 'switchNote', 'tickLabel', 'timeline', 'clockLabel',
 'countRed', 'countBlue', 'countSea', 'countLand', 'countAir', 'targetCountHint',
 'cacheMemory', 'cacheDisk', 'cacheNetwork', 'cacheStored',
 'loadingBadge', 'loadingText',
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
async function refreshCacheStats() {
  if (!window.TileCache || !dom.cacheMemory) return;
  try {
    const s = await TileCache.snapshot();
    dom.cacheMemory.textContent = s.memoryHits + ' 次命中 / ' + s.memory + ' 张在内存';
    dom.cacheDisk.textContent = (s.disk === null ? '—' : s.disk + ' 张') +
      '，已预热 ' + s.preloaded + ' 张';
    dom.cacheNetwork.textContent = s.network + ' 次' +
      (s.warming ? '（预热中 ' + s.warming + '）' : '') +
      (s.errors ? '，失败 ' + s.errors : '');
    dom.cacheStored.textContent = humanBytes(s.usageBytes) + ' / 上限 ' +
      (s.disk === null ? '—' : s.diskLimit + ' 张') +
      (s.persistent === false ? '（持久化不可用，仅内存）' : '');
  } catch (e) {
    // 缓存统计不可用不应影响主流程。
  }
}

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
  state.imageryStats = OfflineAssets.newStats('imagery');
  const imageryProvider = OfflineAssets.instrument(
    OfflineAssets.createImageryProvider(), state.imageryStats);
  state.imageryLayer = new Cesium.ImageryLayer(imageryProvider);

  viewer = new Cesium.Viewer('cesiumContainer', {
    baseLayer: state.imageryLayer,
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

  // 完全离线：不触碰 Ion 资产，不使用 Bing/在线地形服务。
  viewer.scene.globe.enableLighting = false;
  viewer.scene.globe.showGroundAtmosphere = true;
  viewer.scene.skyAtmosphere.show = true;
  viewer.scene.fog.enabled = false;
  viewer.scene.globe.depthTestAgainstTerrain = false;

  // 关闭 Cesium 自带的双击跟踪与默认缩放行为，避免与信息面板抢事件。
  viewer.cesiumWidget.screenSpaceEventHandler.removeInputAction(
    Cesium.ScreenSpaceEventType.LEFT_DOUBLE_CLICK);
}

function flyTo(lon, lat, height, pitchDeg) {
  viewer.camera.flyTo({
    destination: Cesium.Cartesian3.fromDegrees(lon, lat, height),
    orientation: { heading: 0, pitch: Cesium.Math.toRadians(pitchDeg === undefined ? -70 : pitchDeg), roll: 0 },
    duration: 1.2,
  });
}

function applyCamera(camera) {
  if (!camera) return;
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
    .then(function (data) {
      if (!data.frames || !data.frames.length) throw new Error('态势数据为空');

      teardownScene();

      state.data = data;
      state.caseId = caseId;
      state.inference = data.inference_frames || null;
      state.tick = 0;
      state.tickFloat = 0;
      state.selected = null;
      state.activeHotspot = -1;
      state.geoPinned = false;
      state.buildingsAoi = -1;

      if (data.symbology) Symbology.setCatalog(data.symbology);

  state.urbanAoi = data.urban_aoi || [];
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

      setStatus('就绪 · ' + data.targets.length + ' 个目标 · ' +
                (data.frames.length - 1) + ' 帧 · ' + data.hotspots.length + ' 个热点区域', 'ok');
      setSwitchNote('已加载案例「' + data.case_name + '」。各开关的生效状态会在此说明。');
      return data;
    });
}

function teardownScene() {
  if (!viewer) return;
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
    item.appendChild(head);
    item.appendChild(sub);
    item.appendChild(desc);

    item.addEventListener('click', function () {
      closeCaseMenu();
      if (c.id === state.caseId) return;
      loadCase(c.id).catch(function (err) {
        console.error(err);
        setStatus('案例加载失败: ' + err.message, 'err');
      });
    });
    list.appendChild(item);
  });
}

function openCaseMenu() {
  dom.menuCaseList.classList.add('open');
  dom.menuCase.setAttribute('aria-expanded', 'true');
  Array.prototype.forEach.call(dom.menuCaseList.children, function (n) {
    n.classList.toggle('active', n.dataset.caseId === state.caseId);
  });
}

function closeCaseMenu() {
  dom.menuCaseList.classList.remove('open');
  dom.menuCase.setAttribute('aria-expanded', 'false');
}

// --------------------------------------------------------------------------
// 场景构建
// --------------------------------------------------------------------------

function positionFor(id, tick) {
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

  // 热点区域表格里的态势等级随之刷新。
  state.hotspotRows.forEach(function (row, i) {
    const r = inf.regions[i];
    if (!r) return;
    row.levelCell.textContent = r[0] + ' ' + levelLabel(r[0]);
    row.levelCell.className = 'lv-cell ' + (LEVEL_CLASS[r[0]] || '');
    row.levelCell.title = '红方 ' + r[1] + ' 个目标 / 蓝方 ' + r[2] + ' 个目标';
  });

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

// --------------------------------------------------------------------------
// 目标列表
// --------------------------------------------------------------------------

function renderTargetList() {
  const host = dom.targetList;
  host.innerHTML = '';

  const counts = { red: 0, blue: 0, sea: 0, land: 0, air: 0 };
  let shown = 0;
  const inf = inferenceAt(state.tick);

  state.data.targets.forEach(function (def) {
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
  dom.targetCountHint.textContent = shown + ' / ' + state.data.targets.length;
}

// --------------------------------------------------------------------------
// 热点区域表格
// --------------------------------------------------------------------------

function renderHotspotTable() {
  const body = dom.hotspotBody;
  body.innerHTML = '';
  state.hotspotRows = [];

  state.data.hotspots.forEach(function (hs, i) {
    const tr = el('tr', 'hotspot-row');
    tr.dataset.index = String(i);

    const levelCell = el('td', 'lv-cell', '—');
    tr.appendChild(el('td', 'hs-idx', String(i + 1)));
    tr.appendChild(el('td', 'hs-name', hs.name));
    // 经纬范围直接取自 bbox，与地图上绘制的矩形同源，不会再出现"显示范围与
    // 真实区域不一致"的问题。
    tr.appendChild(el('td', 'hs-range',
      hs.bbox[0].toFixed(2) + '–' + hs.bbox[2].toFixed(2) + '°E<br>' +
      hs.bbox[1].toFixed(2) + '–' + hs.bbox[3].toFixed(2) + '°N'));
    tr.appendChild(levelCell);

    tr.addEventListener('click', function () { selectHotspot(i, true); });
    body.appendChild(tr);
    state.hotspotRows.push({ row: tr, levelCell: levelCell, hotspot: hs });
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
  if (dom.modelBuildings && dom.modelBuildings.checked) refreshBuildings();
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
  if (state.geoPinned || !state.data) return;

  const height = viewer.camera.positionCartographic.height;
  dom.geoAlt.textContent = (height / 1000).toFixed(1) + ' km';

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

  const idx = hotspotAt(lat, lon);
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
    dom.geoRegion.textContent = '不在热点区域内';
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
  if (fly && s) flyTo(s.lon, s.lat, 180000);
  syncSelection();
  renderTargetList();
  renderTargetPanel(id);
}

function closeTargetPanel() {
  state.selected = null;
  dom.targetPanel.classList.add('hidden');
  syncSelection();
}

function kvRow(label, value, extraClass) {
  return '<div class="tp-kv"><span>' + label + '</span><b' +
    (extraClass ? ' class="' + extraClass + '"' : '') + '>' + value + '</b></div>';
}

function renderTargetPanel(id) {
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
    (state.data.symbology.domains[def.domain] || def.domain) + '</span></div></div>';

  html += '<div class="tp-section"><h4>目标要素</h4>';
  html += kvRow('目标 ID', def.id);
  html += kvRow('型号', def.model);
  html += kvRow('平台类别', def.platform_class + ' · ' + def.role);
  html += kvRow('经度', s ? fmtDeg(s.lon, 'E', 'W') : '—');
  html += kvRow('纬度', s ? fmtDeg(s.lat, 'N', 'S') : '—');
  html += kvRow('高度', s ? s.alt.toFixed(0) + ' m' : '—');
  html += kvRow('航向 / 航速', s ? s.heading.toFixed(0) + '° / ' + s.speed_kn + ' ' + def.speed_unit : '—');
  html += kvRow('传感器', def.sensor_name + '（' + def.sensor_range_km.toFixed(0) + ' km）');
  html += '</div>';

  if (l1) {
    const terms = l1.threat_terms;
    html += '<div class="tp-section"><h4>一级 · 目标级态势</h4>';
    html += kvRow('状态', l1.state_label);
    html += kvRow('意图', l1.intent_label);
    html += kvRow('威胁度', l1.threat + ' / 100', 'lv-' + (l1.alert + 1));
    html += kvRow('告警等级', l1.alert_label, 'lv-' + (l1.alert + 1));
    html += kvRow('最近对方目标',
      l1.nearest_id ? (l1.nearest_id + ' · ' + l1.nearest_km + ' km') : '—');
    html += kvRow('被对方探测', l1.covered_by + ' 部传感器');
    html += '<div class="tp-reason">判定依据：装备权重 ' + terms.base +
      ' + 接近度 ' + terms.proximity + ' + 覆盖度 ' + terms.coverage + '</div>';
    html += '</div>';
  }

  if (group) {
    html += '<div class="tp-section"><h4>二级 · 编组级态势</h4>';
    html += kvRow('所属编组', (group.affiliation === 'red' ? '红方' : '蓝方') +
      (state.data.symbology.domains[group.domain] || group.domain) + '编组（' + group.count + ' 个目标）');
    html += kvRow('编队形态', group.posture_label + '（跨度 ' + group.spread_km + ' km）');
    html += kvRow('对方同域编组', group.opposing_key || '无');
    html += kvRow('最小距离', group.min_distance_km === null ? '—' : group.min_distance_km + ' km');
    html += kvRow('对抗关系', group.relation_label, 'lv-' + (group.relation + 1));
    html += '</div>';
  }

  html += '<div class="tp-section"><h4>三级 · 区域级态势</h4>';
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
  html += '</div>';

  html += '<div class="tp-section"><h4>装备参数</h4>';
  Object.keys(def.attributes || {}).forEach(function (k) {
    html += kvRow(k, def.attributes[k]);
  });
  html += '</div>';

  dom.tpBody.innerHTML = html;
  dom.targetPanel.classList.remove('hidden');
}

// --------------------------------------------------------------------------
// 模型开关：每一项都对应一个真实生效的渲染行为
// --------------------------------------------------------------------------

function setTerrain(on) {
  if (on) {
    setSwitchNote('正在加载离线 DEM 高程（Terrarium，z0–z10）…');
    const provider = OfflineAssets.createTerrainProvider();
    viewer.terrainProvider = provider;
    viewer.scene.globe.depthTestAgainstTerrain = true;
    const done = function () {
      state.terrainOn = true;
      setSwitchNote('地形模型已启用：离线 Terrarium 高程，最高 z10（约 39 km/瓦片）。');
    };
    if (provider.ready) done();
    else provider.readyPromise.then(done).catch(function (e) {
      setSwitchNote('地形加载失败：' + e.message);
    });
  } else {
    viewer.terrainProvider = new Cesium.EllipsoidTerrainProvider();
    viewer.scene.globe.depthTestAgainstTerrain = false;
    state.terrainOn = false;
    setSwitchNote('地形模型已关闭：使用椭球面，高程不参与渲染。');
  }
}

function setWsf3dOverlay(on) {
  if (on) {
    if (!state.wsf3dOverlayLayer) {
      state.wsf3dOverlayLayer = OfflineAssets.attachWsf3dOverlay(
        viewer, 'buildingheight', 0.62);
    }
    state.wsf3dOverlayLayer.show = true;
    setSwitchNote('建筑高度图层已启用：WSF3D v02 建筑高度（90 m），' +
      '叠加于影像底图之上，黑色像元为无数据已剔除。');
  } else if (state.wsf3dOverlayLayer) {
    state.wsf3dOverlayLayer.show = false;
    setSwitchNote('建筑高度图层已关闭。');
  }
}

function setBuildings(on) {
  if (!on) {
    if (state.buildingsPrimitive) state.buildingsPrimitive.show = false;
    setSwitchNote('建筑体块已关闭。');
    return;
  }
  if (!state.ramp) {
    setSwitchNote('建筑体块需要色带标定文件 tiles/wsf3d/height_ramp.json，正在加载…');
    OfflineAssets.loadHeightRamp().then(function (ramp) {
      state.ramp = ramp;
      setBuildings(true);
    }).catch(function (e) {
      setSwitchNote('建筑体块不可用：' + e.message +
        '（请先运行 tools/calibrate_wsf3d.py）');
      document.getElementById('modelBuildings').checked = false;
    });
    return;
  }
  refreshBuildings();
}

function pickAoiForHotspot(index) {
  const aois = state.data.urban_aoi || [];
  if (!aois.length) return -1;
  if (index < 0) return 0;

  const hs = state.data.hotspots[index];
  if (!hs) return 0;
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
  if (!dom.modelBuildings || !dom.modelBuildings.checked) return;
  if (!state.ramp || !state.data) return;

  const aois = state.data.urban_aoi || [];
  const aoiIdx = pickAoiForHotspot(state.activeHotspot);
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
    state.buildingBusy = false;
    const ms = Date.now() - t0;
    if (!result.primitive) {
      setSwitchNote('「' + aoi.name + '」没有生成体块：' + JSON.stringify(result.stats));
      return;
    }
    state.buildingsCache[key] = { primitive: result.primitive, stats: result.stats };
    swapBuildings(state.buildingsCache[key], aoiIdx, lod, ms);
  }).catch(function (e) {
    state.buildingBusy = false;
    console.error(e);
    setSwitchNote('建筑体块生成失败：' + e.message);
  });
}

function swapBuildings(entry, aoiIdx, lod, ms) {
  if (state.buildingsPrimitive) {
    viewer.scene.primitives.remove(state.buildingsPrimitive);
    state.buildingsPrimitive = null;
  }
  state.buildingsPrimitive = entry.primitive;
  state.buildingsAoi = aoiIdx;
  state.buildingsKey = aoiIdx + ':' + lod;
  viewer.scene.primitives.add(entry.primitive);

  const aoi = state.data.urban_aoi[aoiIdx];
  const st = entry.stats;
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
  const want = height <= URBAN_IMAGERY_TRIGGER_M;

  if (want && !state.urbanLayers) {
    state.urbanLayers = OfflineAssets.attachUrbanImagery(viewer, state.urbanAoi);
    return;
  }
  if (!want && state.urbanLayers) {
    state.urbanLayers.forEach(function (rec) {
      viewer.imageryLayers.remove(rec.layer, true);
    });
    state.urbanLayers = null;
  }
}

function setImagery(on) {
  if (state.imageryLayer) state.imageryLayer.show = on;
  setSwitchNote(on
    ? '影像底图已启用：离线 Esri World Imagery 瓦片（z0–z11）。'
    : '影像底图已关闭：球体仅显示几何与态势要素。');
}

// --------------------------------------------------------------------------
// 时间轴
// --------------------------------------------------------------------------

function renderTransport() {
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
  const max = state.data.frames.length - 1;
  state.tick = Math.max(0, Math.min(max, Math.round(tick)));
  state.tickFloat = state.tick;
}

function setPlaying(on) {
  state.playing = on;
  document.getElementById('btnPlay').textContent = on ? '❚❚ 暂停' : '▶ 播放';
  document.getElementById('btnRun').textContent = on ? '推演中…' : '态势自动推演';
}

function loop(nowMs) {
  const dt = lastFrameMs ? (nowMs - lastFrameMs) / 1000 : 0;
  lastFrameMs = nowMs;

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
    }
  }

  syncEntities();

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

function exportData() {
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
    if (node) node.addEventListener('change', function (e) { handler(e.target.checked); });
  };

  bindCheck('layerTargets', function (v) { state.layers.targets = v; });
  bindCheck('layerTracks', function (v) { state.layers.tracks = v; });
  bindCheck('layerRoutes', function (v) { state.layers.routes = v; });
  bindCheck('layerHotspots', function (v) { state.layers.hotspots = v; });
  bindCheck('layerLabels', function (v) { state.layers.labels = v; });

  const refilter = function () { renderTargetList(); };
  bindCheck('filterRed', function (v) { state.filters.red = v; refilter(); });
  bindCheck('filterBlue', function (v) { state.filters.blue = v; refilter(); });
  bindCheck('filterSea', function (v) { state.filters.sea = v; refilter(); });
  bindCheck('filterLand', function (v) { state.filters.land = v; refilter(); });
  bindCheck('filterAir', function (v) { state.filters.air = v; refilter(); });

  bindCheck('modelTerrain', setTerrain);
  bindCheck('modelWsf3dOverlay', setWsf3dOverlay);
  bindCheck('modelBuildings', setBuildings);
  bindCheck('modelWeather', setWeather);
  bindCheck('modelSensor', setSensors);
  bindCheck('modelImagery', setImagery);

  dom.menuCase.addEventListener('click', function (e) {
    e.stopPropagation();
    if (dom.menuCaseList.classList.contains('open')) closeCaseMenu();
    else openCaseMenu();
  });
  document.addEventListener('click', function (e) {
    if (!dom.menuCaseList.contains(e.target) && e.target !== dom.menuCase) closeCaseMenu();
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

  document.getElementById('btnExport').addEventListener('click', exportData);

  document.getElementById('btnClearCache').addEventListener('click', function () {
    if (!window.TileCache) return;
    TileCache.clear().then(function () {
      setSwitchNote('瓦片缓存已清空，下次浏览会重新从服务器取。');
      refreshCacheStats();
    });
  });

  // "位置信息"：回到光标跟随模式，同时把视角拉回案例默认机位。
  document.getElementById('btnLocate').addEventListener('click', function () {
    clearHotspotSelection();
    if (state.data && state.data.camera) applyCamera(state.data.camera);
  });

  document.getElementById('btnSituation').addEventListener('click', function () {
    dom.targetList.scrollTop = 0;
  });

  document.getElementById('btnWeather').addEventListener('click', function () {
    const node = document.getElementById('modelWeather');
    node.checked = !node.checked;
    setWeather(node.checked);
  });

  bindPicking();

  // 相机移动时也刷新一次读数（节流到 4 Hz），否则刚打开页面时面板全是"—"。
  let lastGeoUpdate = 0;
  // 底图加载进度。queued 为 0 表示当前视野的瓦片已经全部就位。
  viewer.scene.globe.tileLoadProgressEvent.addEventListener(setLoadingTiles);

  viewer.camera.changed.addEventListener(function () {
    const now = Date.now();
    if (now - lastGeoUpdate < 250) return;
    lastGeoUpdate = now;
    updateGeoReadout();
    syncUrbanImagery();
    // 相机高度跨过 LOD 阈值时才重建，且加一段防抖，避免飞行过程中反复触发。
    if (dom.modelBuildings && dom.modelBuildings.checked) {
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

function applyUrlHooks() {
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
  if (params.has('hotspot')) {
    const i = Number(params.get('hotspot'));
    if (Number.isFinite(i) && state.data.hotspots[i]) selectHotspot(i, true);
  }
  if (params.get('buildings') === '1') {
    document.getElementById('modelBuildings').checked = true;
    // 建筑体块依赖色带标定，异步完成后才会出现。
    OfflineAssets.loadHeightRamp().then(function (ramp) {
      state.ramp = ramp;
      refreshBuildings();
    }).catch(function (e) {
      setSwitchNote('建筑体块不可用：' + e.message);
    });
  }
  if (params.has('panel')) {
    const id = params.get('panel');
    if (state.targets[id]) {
      selectTarget(id, false);
      const s = positionFor(id, state.tick);
      if (s) flyTo(s.lon, s.lat, 120000);
    }
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
    return continueStartup();
  }).catch(function (err) {
    console.error(err);
    setStatus('初始化失败: ' + err.message, 'err');
  });

  setInterval(renderDiagnostics, 500);
  setInterval(refreshCacheStats, 1500);
  setTimeout(refreshCacheStats, 300);
  requestAnimationFrame(loop);
}

function continueStartup() {

  // 色带标定按需加载，缺失不影响其它功能。
  OfflineAssets.loadHeightRamp().then(function (ramp) {
    state.ramp = ramp;
  }).catch(function () {
    state.ramp = null;
  });

  loadCaseIndex()
    .then(function (index) {
      const wanted = params.get('case') || index.default;
      return loadCase(index.cases.some(function (c) { return c.id === wanted; })
        ? wanted : index.default);
    })
    .then(function () {
      bindUI();
      applyUrlHooks();
      syncSelection();
      updateGeoReadout();
    })
    .catch(function (err) {
      console.error(err);
      setStatus('启动失败: ' + err.message, 'err');
    });

}

main();
