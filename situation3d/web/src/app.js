/* situation3d — 三维战场态势生成系统前端
 *
 * 职责：加载引擎输出的逐帧态势数据，在 Cesium 三维地球上回放。
 * 数据流：engine (per-tick state) -> data/*.json -> 本文件 -> Cesium entities
 *
 * 当前实现为纯前端回放器：位置由引擎逐帧给出，前端不做轨迹外推。
 */

'use strict';

const DATA_URL = './data/demo_taiwan_strait.json';

// Deterministic capture hook, used by the verification harness and for deep links:
//   ?tick=300     seek to a frame before the first render
//   ?autoplay=1   start playback immediately
const params = new URLSearchParams(window.location.search);

const state = {
  data: null,
  tick: 0,
  tickFloat: 0,
  playing: false,
  ticksPerSecond: 30,
  targets: {},          // id -> { def, entity, trail: [], lastTick }
  layers: { targets: true, tracks: true, routes: true, hotspots: true, labels: true },
  filters: { red: true, blue: true, sea: true, air: true },
  selected: null,
  hotspotEntities: [],
  routeEntities: [],
};

let viewer = null;
let lastFrameMs = 0;

// --------------------------------------------------------------------------
// bootstrap
// --------------------------------------------------------------------------

function setStatus(text, kind) {
  document.getElementById('statusText').textContent = text;
  const dot = document.getElementById('statusDot');
  dot.className = 'dot' + (kind ? ' ' + kind : '');
}

function buildImageryProvider() {
  // Natural Earth II ships with CesiumJS and is served from the vendored copy,
  // so the globe renders with no external tile service.
  return new Cesium.UrlTemplateImageryProvider({
    url: Cesium.buildModuleUrl('Assets/Textures/NaturalEarthII') + '/{z}/{x}/{y}.jpg',
    tilingScheme: new Cesium.GeographicTilingScheme(),
    maximumLevel: 2,
    credit: 'Natural Earth II',
  });
}

function initViewer() {
  // Cesium 1.107 removed the Viewer `imageryProvider` option; it is now silently
  // ignored and the globe falls back to its default baseColor. `baseLayer` is the
  // supported replacement, and the assertion below turns a silent regression into
  // a loud one.
  viewer = new Cesium.Viewer('cesiumContainer', {
    baseLayer: new Cesium.ImageryLayer(buildImageryProvider()),
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

  // Fully offline: no Ion assets, no Bing/terrain services.
  viewer.scene.globe.enableLighting = false;
  viewer.scene.globe.showGroundAtmosphere = true;
  viewer.scene.skyAtmosphere.show = true;
  viewer.scene.fog.enabled = false;

  // Camera opens over the Taiwan Strait.
  viewer.camera.setView({
    destination: Cesium.Cartesian3.fromDegrees(119.6, 24.4, 900000),
    orientation: { heading: 0, pitch: Cesium.Math.toRadians(-75), roll: 0 },
  });
}

// --------------------------------------------------------------------------
// data loading
// --------------------------------------------------------------------------

async function loadData() {
  const resp = await fetch(DATA_URL, { cache: 'no-store' });
  if (!resp.ok) throw new Error('态势数据加载失败: HTTP ' + resp.status);
  state.data = await resp.json();
  if (!state.data.frames || !state.data.frames.length) {
    throw new Error('态势数据为空');
  }
}

// --------------------------------------------------------------------------
// rendering
// --------------------------------------------------------------------------

function colorFor(affiliation) {
  return affiliation === 'red'
    ? Cesium.Color.fromCssColorString('#c8352c')
    : Cesium.Color.fromCssColorString('#1f5fbf');
}

function positionFor(id, tick) {
  const frame = state.data.frames[Math.min(tick, state.data.frames.length - 1)];
  return frame.targets.find((t) => t.id === id);
}

function isVisible(def) {
  if (!state.layers.targets) return false;
  if (def.affiliation === 'red' && !state.filters.red) return false;
  if (def.affiliation === 'blue' && !state.filters.blue) return false;
  if (def.domain === 'sea' && !state.filters.sea) return false;
  if (def.domain === 'air' && !state.filters.air) return false;
  return true;
}

function createTargetEntities() {
  state.data.targets.forEach((def) => {
    const color = colorFor(def.affiliation);
    const isAir = def.domain === 'air';

    const entity = viewer.entities.add({
      id: 'target-' + def.id,
      position: new Cesium.CallbackProperty(() => {
        const s = positionFor(def.id, state.tick);
        if (!s) return undefined;
        return Cesium.Cartesian3.fromDegrees(s.lon, s.lat, s.alt);
      }, false),
      point: {
        pixelSize: isAir ? 9 : 11,
        color: color,
        outlineColor: Cesium.Color.WHITE,
        outlineWidth: 2,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
      label: {
        text: def.id + ' ' + def.platform_class,
        font: '12px "Microsoft YaHei", sans-serif',
        fillColor: Cesium.Color.fromCssColorString('#1d2430'),
        outlineColor: Cesium.Color.WHITE,
        outlineWidth: 3,
        style: Cesium.LabelStyle.FILL_AND_OUTLINE,
        pixelOffset: new Cesium.Cartesian2(0, -22),
        verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
        showBackground: true,
        backgroundColor: Cesium.Color.fromCssColorString('rgba(255,255,255,0.82)'),
      },
    });

    // Growing trail: everything the platform has covered up to the current tick.
    entity.polyline = {
      positions: new Cesium.CallbackProperty(() => {
        const trail = state.targets[def.id] ? state.targets[def.id].trail : [];
        return trail;
      }, false),
      width: 2,
      material: color.withAlpha(0.85),
      clampToGround: false,
    };

    state.targets[def.id] = { def, entity, trail: [], lastTick: -1 };
  });
}

function createRouteEntities() {
  state.data.targets.forEach((def) => {
    if (!def.waypoints || def.waypoints.length < 2) return;
    const flat = [];
    def.waypoints.forEach((wp) => flat.push(wp[1], wp[0], def.altitude_m || 0));
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
  state.data.hotspots.forEach((hs) => {
    const entity = viewer.entities.add({
      id: 'hotspot-' + hs.name,
      position: Cesium.Cartesian3.fromDegrees(hs.lon, hs.lat, 0),
      point: {
        pixelSize: 7,
        color: Cesium.Color.fromCssColorString('#f0a020'),
        outlineColor: Cesium.Color.WHITE,
        outlineWidth: 2,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
      label: {
        text: hs.name,
        font: '11px "Microsoft YaHei", sans-serif',
        fillColor: Cesium.Color.fromCssColorString('#6b7688'),
        outlineColor: Cesium.Color.WHITE,
        outlineWidth: 2,
        style: Cesium.LabelStyle.FILL_AND_OUTLINE,
        pixelOffset: new Cesium.Cartesian2(0, 14),
        verticalOrigin: Cesium.VerticalOrigin.TOP,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
    });
    state.hotspotEntities.push(entity);
  });
}

// --------------------------------------------------------------------------
// per-frame update
// --------------------------------------------------------------------------

function syncEntities() {
  Object.values(state.targets).forEach((rec) => {
    const def = rec.def;
    const visible = isVisible(def);

    rec.entity.show = visible;
    rec.entity.label.show = visible && state.layers.labels;
    rec.entity.point.show = visible;

    // Trail accumulates monotonically. On a rewind it is rebuilt from tick 0
    // rather than merely cleared, otherwise dragging the timeline backwards
    // while paused would leave the trail permanently empty.
    if (state.tick < rec.lastTick) {
      rec.trail.length = 0;
      rec.lastTick = -1;
    }
    if (visible && state.layers.tracks && state.tick > rec.lastTick) {
      for (let t = rec.lastTick + 1; t <= state.tick; t++) {
        const s = positionFor(def.id, t);
        if (s) rec.trail.push(Cesium.Cartesian3.fromDegrees(s.lon, s.lat, s.alt));
      }
    }
    rec.lastTick = state.tick;

    if (!state.layers.tracks) rec.trail.length = 0;

    rec.entity.polyline.show = visible && state.layers.tracks && rec.trail.length > 1;
  });

  state.routeEntities.forEach((e) => { e.show = state.layers.routes; });
  state.hotspotEntities.forEach((e) => { e.show = state.layers.hotspots; });
}

function renderTargetList() {
  const host = document.getElementById('targetList');
  host.innerHTML = '';

  let red = 0;
  let blue = 0;

  state.data.targets.forEach((def) => {
    if (def.affiliation === 'red') red++;
    if (def.affiliation === 'blue') blue++;
    if (!isVisible(def)) return;

    const s = positionFor(def.id, state.tick);
    const div = document.createElement('div');
    div.className = 'target-item ' + def.affiliation + (state.selected === def.id ? ' active' : '');
    div.innerHTML =
      '<div class="ti-head"><span>' + def.id + ' ' + def.name + '</span></div>' +
      '<div class="ti-sub">' +
        (s ? s.lat.toFixed(4) + '°N ' + s.lon.toFixed(4) + '°E' : '—') +
        (s ? ' · ' + s.speed_kn + ' kn · ' + s.heading.toFixed(0) + '°' : '') +
      '</div>' +
      '<div class="ti-meta">' + def.platform_class + ' · ' + def.role + '</div>';
    div.addEventListener('click', () => selectTarget(def.id));
    host.appendChild(div);
  });

  document.getElementById('countRed').textContent = red;
  document.getElementById('countBlue').textContent = blue;
}

function renderHotspotTable() {
  const body = document.getElementById('hotspotBody');
  body.innerHTML = '';
  state.data.hotspots.forEach((hs, i) => {
    const tr = document.createElement('tr');
    tr.innerHTML =
      '<td>' + (i + 1) + '</td>' +
      '<td>' + hs.name + '</td>' +
      '<td>' + hs.lat.toFixed(4) + 'N<br>' + hs.lon.toFixed(4) + 'E</td>';
    tr.addEventListener('click', () => flyTo(hs.lon, hs.lat, 400000));
    body.appendChild(tr);
  });
}

function renderTransport() {
  const frames = state.data.frames;
  const frame = frames[Math.min(state.tick, frames.length - 1)];
  document.getElementById('timeline').max = String(frames.length - 1);
  document.getElementById('timeline').value = String(state.tick);
  document.getElementById('tickLabel').textContent = state.tick + ' / ' + (frames.length - 1);

  const start = new Date(state.data.start_time_utc);
  const t = new Date(start.getTime() + frame.t_seconds * 1000);
  document.getElementById('clockLabel').textContent =
    t.toISOString().replace('T', ' ').slice(0, 19) + 'Z';
}

function selectTarget(id) {
  state.selected = id;
  const s = positionFor(id, state.tick);
  if (s) flyTo(s.lon, s.lat, 220000);
  renderTargetList();
}

function flyTo(lon, lat, height) {
  viewer.camera.flyTo({
    destination: Cesium.Cartesian3.fromDegrees(lon, lat, height),
    orientation: { heading: 0, pitch: Cesium.Math.toRadians(-70), roll: 0 },
    duration: 1.2,
  });
}

function setTick(tick) {
  const max = state.data.frames.length - 1;
  state.tick = Math.max(0, Math.min(max, Math.round(tick)));
  state.tickFloat = state.tick;
}

// --------------------------------------------------------------------------
// main loop
// --------------------------------------------------------------------------

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
    }
  }

  syncEntities();
  requestAnimationFrame(loop);
}

function setPlaying(on) {
  state.playing = on;
  document.getElementById('btnPlay').textContent = on ? '❚❚ 暂停' : '▶ 播放';
  document.getElementById('btnRun').textContent = on ? '推演中…' : '态势自动推演';
}

// --------------------------------------------------------------------------
// wiring
// --------------------------------------------------------------------------

function bindUI() {
  const bind = (id, handler) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('change', handler);
  };

  bind('layerTargets', (e) => { state.layers.targets = e.target.checked; });
  bind('layerTracks', (e) => { state.layers.tracks = e.target.checked; });
  bind('layerRoutes', (e) => { state.layers.routes = e.target.checked; });
  bind('layerHotspots', (e) => { state.layers.hotspots = e.target.checked; });
  bind('layerLabels', (e) => { state.layers.labels = e.target.checked; });

  bind('filterRed', (e) => { state.filters.red = e.target.checked; renderTargetList(); });
  bind('filterBlue', (e) => { state.filters.blue = e.target.checked; renderTargetList(); });
  bind('filterSea', (e) => { state.filters.sea = e.target.checked; renderTargetList(); });
  bind('filterAir', (e) => { state.filters.air = e.target.checked; renderTargetList(); });

  document.getElementById('btnPlay').addEventListener('click', () => setPlaying(!state.playing));
  document.getElementById('btnRun').addEventListener('click', () => setPlaying(!state.playing));
  document.getElementById('btnReset').addEventListener('click', () => {
    setPlaying(false);
    setTick(0);
    Object.values(state.targets).forEach((r) => { r.trail.length = 0; r.lastTick = -1; });
    renderTargetList();
    renderTransport();
  });

  document.getElementById('speedSelect').addEventListener('change', (e) => {
    state.ticksPerSecond = Number(e.target.value);
  });

  document.getElementById('timeline').addEventListener('input', (e) => {
    setPlaying(false);
    setTick(Number(e.target.value));
    renderTargetList();
    renderTransport();
  });

  document.getElementById('btnStepBack').addEventListener('click', () => {
    setPlaying(false); setTick(state.tick - 1); renderTargetList(); renderTransport();
  });
  document.getElementById('btnStepFwd').addEventListener('click', () => {
    setPlaying(false); setTick(state.tick + 1); renderTargetList(); renderTransport();
  });

  document.getElementById('btnExport').addEventListener('click', exportData);

  document.getElementById('btnLocate').addEventListener('click', () => {
    viewer.camera.flyTo({
      destination: Cesium.Cartesian3.fromDegrees(119.6, 24.4, 900000),
      orientation: { heading: 0, pitch: Cesium.Math.toRadians(-75), roll: 0 },
      duration: 1.2,
    });
  });

  document.getElementById('btnSituation').addEventListener('click', () => {
    const t = document.getElementById('targetList');
    t.scrollTop = 0;
  });

  document.getElementById('btnWeather').addEventListener('click', () => {
    const on = !viewer.scene.skyAtmosphere.show;
    viewer.scene.skyAtmosphere.show = on;
    viewer.scene.globe.showGroundAtmosphere = on;
  });

  document.getElementById('modelTerrain').addEventListener('change', (e) => {
    viewer.scene.globe.depthTestAgainstTerrain = e.target.checked;
  });
  document.getElementById('modelWeather').addEventListener('change', (e) => {
    viewer.scene.fog.enabled = e.target.checked;
  });
  document.getElementById('modelSensor').addEventListener('change', (e) => {
    viewer.scene.globe.showGroundAtmosphere = e.target.checked;
  });

  // Mouse position readout.
  const handler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);
  handler.setInputAction((movement) => {
    const cartesian = viewer.camera.pickEllipsoid(movement.endPosition, viewer.scene.globe.ellipsoid);
    if (!cartesian) return;
    const carto = Cesium.Cartographic.fromCartesian(cartesian);
    document.getElementById('geoLon').textContent = Cesium.Math.toDegrees(carto.longitude).toFixed(4) + '°E';
    document.getElementById('geoLat').textContent = Cesium.Math.toDegrees(carto.latitude).toFixed(4) + '°N';
  }, Cesium.ScreenSpaceEventType.MOUSE_MOVE);

  handler.setInputAction((click) => {
    const picked = viewer.scene.pick(click.position);
    if (picked && picked.id && typeof picked.id.id === 'string' && picked.id.id.startsWith('target-')) {
      selectTarget(picked.id.id.replace('target-', ''));
    }
  }, Cesium.ScreenSpaceEventType.LEFT_CLICK);
}

function exportData() {
  const frames = state.data.frames;
  const payload = {
    scenario_id: state.data.scenario_id,
    scenario_name: state.data.scenario_name,
    region: state.data.region,
    provenance: state.data.provenance,
    tick_seconds: state.data.tick_seconds,
    exported_tick: state.tick,
    tick_count: frames.length - 1,
    targets: state.data.targets.map((d) => {
      const s = positionFor(d.id, state.tick);
      return {
        id: d.id,
        name: d.name,
        platform_class: d.platform_class,
        domain: d.domain,
        affiliation: d.affiliation,
        lat: s ? s.lat : null,
        lon: s ? s.lon : null,
        alt_m: s ? s.alt : null,
        speed_kn: s ? s.speed_kn : null,
        heading_deg: s ? s.heading : null,
      };
    }),
  };

  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'situation_export_tick' + state.tick + '.json';
  a.click();
  URL.revokeObjectURL(a.href);
}

async function main() {
  try {
    setStatus('加载态势数据…');
    initViewer();
    await loadData();

    document.getElementById('scenarioName').textContent = state.data.region + ' · ' + state.data.scenario_name;

    createHotspots();
    createRouteEntities();
    createTargetEntities();

    setTick(0);
    renderHotspotTable();
    renderTargetList();
    renderTransport();
    bindUI();

    if (params.has('tick')) {
      const seek = Number(params.get('tick'));
      if (Number.isFinite(seek)) setTick(seek);
    }
    if (params.get('autoplay') === '1') setPlaying(true);
    renderTargetList();
    renderTransport();

    setStatus('就绪 · ' + state.data.targets.length + ' 个目标 · ' + (state.data.frames.length - 1) + ' 帧', 'ok');
  } catch (err) {
    console.error(err);
    setStatus('启动失败: ' + err.message, 'err');
  }

  requestAnimationFrame(loop);
}

main();
