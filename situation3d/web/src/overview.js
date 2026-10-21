/* situation3d — 鸟瞰图（全局二维态势视图）
 *
 * 定位
 * ----
 * 它是**整个ZC的二维态势图**：加载案例后按全部目标的实际航迹范围自动取景，
 * 显示红蓝双方全部参战目标，并用与「整体TS」同一套战损状态规范（见
 * targetStatus.js）表达每个目标当前还能不能打。
 *
 * 早先的实现是"固定 4000 km 的台海缩略图 + 主视窗黄框"：换到中印案例时仍在
 * 显示台海，而且图上没有任何目标。现在：
 *   - 取景由 `setData()` 从 frames 计算（不再跟随 data.camera 的单点）；
 *   - 每个目标一个图标 + 紧贴底部的血条（损毁另有 X 遮罩），逐帧与主图/整体TS同步；
 *   - 点击目标可选中并飞到主视角；点击空白仍跳到该地理位置。
 *
 * 仍然保留的设计（都有实测原因，不要随手改）
 * ------------------------------------------
 * 1. **SCENE2D，不用 3D**：鸟瞰图本来就是俯视；而且第二个 WebGL 上下文在无头
 *    SwiftShader 下用 3D 会被彻底饿死（实测画布全黑），2D 则稳定。
 * 2. **必须用 `baseLayer`**：Cesium 1.107 起 `Viewer` 的 `imageryProvider` 选项被
 *    移除且静默忽略，不传 baseLayer 会回落到 Ion 在线底图（离线环境直接报错）。
 * 3. **不共用主视图的 provider 实例**：主视图的 provider 被 TileCache 与
 *    instrument 各包了一层，共用会让鸟瞰图的请求与 404 计入主视图的
 *    `imageryStats`，`?diag=1` 全部失真。
 * 4. **视野框用普通数组赋值，不用 `CallbackProperty`**：避免场景被判定为持续时变。
 * 5. **不用 `requestRenderMode`**：无头 SwiftShader 下会停在黑屏。
 * 6. **隐藏/显示要走 resize**：`display:none` 让容器尺寸归零，而 Cesium 只监听
 *    window.resize，重新显示不会自愈；show() 必须 resize 后重新取景。
 *
 * 图像资产
 * --------
 * 图标复用 `Symbology.symbolUri` 的静态军标；血条与状态角标由
 * `TargetStatus.barUri/markerUri` 生成 data URI，按 (档位, 血量百分比) 缓存，
 * 不会逐帧重新编码。
 */
'use strict';

(function (global) {
  // 主图低于这个高度时不画视野框：那时它退化成一个点，没有信息量。
  const FOOTPRINT_MIN_HEIGHT_M = 20000;
  // 视野框更新节流。跟随时主相机每帧都在动，放宽到 500 ms。
  const FOOTPRINT_THROTTLE_MS = 500;

  // 目标图标 / 血条 / 损毁遮罩的像素尺寸与偏移。血条**紧贴图标底边**：
  // 图标底边与血条顶边都落在锚点上，二者之间没有空隙。
  const ICON_PX = 24;
  const MARK_PX = 24;               // 损毁暗色底与图标同大，不压住自己的血条
  const BAR_W = 24;
  const BAR_H = 4;
  const ICON_OFFSET_Y = 0;          // 图标底边距锚点
  const ICON_CENTER_Y = ICON_OFFSET_Y - ICON_PX / 2;
  const BAR_OFFSET_Y = -2;          // 让血条顶到军标可见底边（图标资产底部有 ~2px 透明留白）

  // 固定的兜底取景（没有案例数据时）。加载案例后一定由 setData 覆盖。
  const DEFAULT_EXTENT = { w: 117.0, s: 21.0, e: 123.0, n: 27.0 };

  let viewer = null;          // 鸟瞰图 Viewer
  let mainViewer = null;      // 主 3D Viewer
  let footprintEntity = null;
  let lastFootprintAt = 0;
  let lastFootprintKey = 'init';   // 'init' | 'none' | 四个交点的实际坐标
  let footprintUpdates = 0;        // 黄框真正被重画的次数（诊断与自检用）
  let created = false;

  let data = null;            // 当前案例数据（targets / frames / inference_frames）
  let extent = { w: DEFAULT_EXTENT.w, s: DEFAULT_EXTENT.s, e: DEFAULT_EXTENT.e, n: DEFAULT_EXTENT.n };
  let lastTick = 0;
  let selected = null;
  let targetEnts = {};        // 目标 id -> { def, icon, mark, bar, status, hpBucket }

  // Direction 2: occupation mask. Kept for compatibility; the layer is offline.
  let maskEntities = [];
  let maskVertices = 0;
  let maskUpdates = 0;
  let maskState = { tick: null, occupation: {}, hotspots: [], visible: false, key: '' };

  // ── I09：语义色唯一来源 ──────────────────────────────────────────────
  // 阵营色由 `theme.css` 的 `--faction-*` 定义，经 `Theme` 读取。
  // 下面的字面量**只是 `Theme` 不可用时的兜底**（与 `theme.js` 的
  // "CSS 变量优先、内置默认兜底" 同一口径）。
  function _semantic(key, fallback) {
    try {
      if (window.Theme && typeof window.Theme.color === 'function') {
        var v = window.Theme.color(key);
        if (v) return v;
      }
    } catch (e) { /* Theme 不可用时用兜底，不让渲染整体失败 */ }
    return fallback;
  }

  const MASK_COLORS = {
    red: _semantic('faction-red', '#E74C3C'),
    blue: _semantic('faction-blue', '#2E86C1'),
    contested: _semantic('faction-contested', '#d8a12a'),
    neutral: _semantic('faction-neutral', '#95A5A6'),
  };

  // 损毁目标的图标压暗色。只压暗、不改色相，兵种图形仍然可辨。
  const DESTROYED_TINT = Cesium.Color.fromCssColorString('#AEB6C2');

  function maskColor(status) {
    return Cesium.Color.fromCssColorString(MASK_COLORS[status] || MASK_COLORS.neutral);
  }

  function container() { return document.getElementById('overviewContainer'); }

  function _statusApi() { return global.TargetStatus || null; }

  // ── 取景 ────────────────────────────────────────────────────────────

  /**
   * 由案例全部帧的全部目标位置计算战场范围（经纬度包围盒）。
   *
   * 用**整段航迹**而不是当前帧：鸟瞰图要回答"整场仗在哪打"，如果取景跟着
   * 当前帧目标走，播放时地图会不断缩放，比例尺失去可比性。
   * 没有 frames 时退回案例相机点周围 ±1.5°。
   */
  function computeExtent(d) {
    let w = Infinity, s = Infinity, e = -Infinity, n = -Infinity;
    (d && d.frames || []).forEach(function (fr) {
      (fr.targets || []).forEach(function (p) {
        if (!isFinite(p.lon) || !isFinite(p.lat)) return;
        if (p.lon < w) w = p.lon;
        if (p.lon > e) e = p.lon;
        if (p.lat < s) s = p.lat;
        if (p.lat > n) n = p.lat;
      });
    });
    if (!isFinite(w) || !isFinite(e)) {
      const cam = (d && d.camera) || {};
      const lon = isFinite(cam.lon) ? cam.lon : 119.60;
      const lat = isFinite(cam.lat) ? cam.lat : 24.40;
      w = lon - 1.5; e = lon + 1.5; s = lat - 1.5; n = lat + 1.5;
    }
    // 最小跨度 0.4°：目标几乎静止时也要有一个可读的战场范围，而不是把镜头
    // 推到一栋楼那么近。先按最小跨度扩到中心对称，再加 8% 边距。
    const spanLon = Math.max(e - w, 0.4);
    const spanLat = Math.max(n - s, 0.4);
    const cx = (w + e) / 2, cy = (s + n) / 2;
    w = cx - spanLon / 2; e = cx + spanLon / 2;
    s = cy - spanLat / 2; n = cy + spanLat / 2;
    const padLon = spanLon * 0.08;
    const padLat = spanLat * 0.08;
    return { w: w - padLon, s: s - padLat, e: e + padLon, n: n + padLat };
  }

  /** 把鸟瞰图对准当前战场范围。隐藏状态下容器为 0×0，不做取景。 */
  function applyExtent() {
    if (!viewer || !extent || !isVisible()) return;
    viewer.camera.setView({
      destination: Cesium.Rectangle.fromDegrees(extent.w, extent.s, extent.e, extent.n),
    });
    viewer.scene.requestRender();
  }

  function setExtent(ex) {
    if (!ex || !isFinite(ex.w) || !isFinite(ex.s) || !isFinite(ex.e) || !isFinite(ex.n)) return;
    extent = ex;
    applyExtent();
  }

  /**
   * 兼容旧接口：数据尚未加载时用相机点做一个兜底取景。
   * 已加载案例时以 `setData` 算出的战场范围为准，不因主相机移动而改变比例尺。
   */
  function setCenter(lon, lat) {
    if (!isFinite(lon) || !isFinite(lat)) return;
    if (!data) setExtent({ w: lon - 1.5, s: lat - 1.5, e: lon + 1.5, n: lat + 1.5 });
  }

  // ── 案例数据 / 目标图元 ─────────────────────────────────────────────

  /**
   * 装载案例。会重建目标图元并重新取景；在 viewer 尚未创建时只保存数据，
   * 等 ensure() 时再建（鸟瞰图是惰性创建的，默认可能一直关着）。
   */
  function setData(d) {
    if (!d) return;
    data = d;
    extent = computeExtent(d);
    lastTick = 0;
    selected = null;
    if (viewer) {
      clearTargetEntities();
      buildTargetEntities();
    }
    applyExtent();
    updateSituation(d, lastTick);
  }

  function clearTargetEntities() {
    if (!viewer) return;
    Object.keys(targetEnts).forEach(function (id) {
      const rec = targetEnts[id];
      [rec.icon, rec.mark, rec.bar].forEach(function (e) {
        if (e) { try { viewer.entities.remove(e); } catch (err) { /* already gone */ } }
      });
    });
    targetEnts = {};
  }

  /** 清空案例（TS 被卸载）：移除目标图元并回到兜底取景，避免残留上一场的图标。 */
  function clear() {
    data = null;
    extent = {
      w: DEFAULT_EXTENT.w, s: DEFAULT_EXTENT.s, e: DEFAULT_EXTENT.e, n: DEFAULT_EXTENT.n,
    };
    lastTick = 0;
    selected = null;
    clearTargetEntities();
    applyExtent();   // 隐藏时由 show() 里的 applyExtent 补上
  }

  function buildTargetEntities() {
    if (!viewer || !data) return;
    (data.targets || []).forEach(function (def) {
      const icon = viewer.entities.add({
        id: 'ov-target-' + def.id,
        name: def.name || def.id,
        position: Cesium.Cartesian3.fromDegrees(0, 0, 0),
        billboard: {
          image: Symbology.symbolUri(def.icon, def.affiliation, { domain: def.domain }),
          width: ICON_PX, height: ICON_PX,
          verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
          pixelOffset: new Cesium.Cartesian2(0, ICON_OFFSET_Y),
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
      });
      icon.ovTargetId = def.id;

      // 损毁遮罩：暗色底 + 大号 "X"（label 比 billboard 后绘制，密集叠压时
      // 叉号仍浮在最上层）。受损/严重受损仅靠血条，不画状态环/角标。
      const mark = viewer.entities.add({
        id: 'ov-mark-' + def.id,
        position: Cesium.Cartesian3.fromDegrees(0, 0, 0),
        show: false,
        billboard: {
          image: _statusApi() ? _statusApi().markerUri(3) : undefined,
          width: MARK_PX, height: MARK_PX,
          verticalOrigin: Cesium.VerticalOrigin.CENTER,
          pixelOffset: new Cesium.Cartesian2(0, ICON_CENTER_Y),
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
        label: {
          text: 'X',
          font: 'bold 26px sans-serif',
          fillColor: Cesium.Color.WHITE,
          outlineColor: Cesium.Color.fromCssColorString('#7F1D1D'),
          outlineWidth: 4,
          style: Cesium.LabelStyle.FILL_AND_OUTLINE,
          horizontalOrigin: Cesium.HorizontalOrigin.CENTER,
          verticalOrigin: Cesium.VerticalOrigin.CENTER,
          pixelOffset: new Cesium.Cartesian2(0, ICON_CENTER_Y),
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
      });
      mark.ovTargetId = def.id;

      const bar = viewer.entities.add({
        id: 'ov-bar-' + def.id,
        position: Cesium.Cartesian3.fromDegrees(0, 0, 0),
        billboard: {
          image: _statusApi() ? _statusApi().barUri(100, 0, BAR_W, BAR_H) : undefined,
          width: BAR_W, height: BAR_H,
          verticalOrigin: Cesium.VerticalOrigin.TOP,
          pixelOffset: new Cesium.Cartesian2(0, BAR_OFFSET_Y),
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
      });
      bar.ovTargetId = def.id;

      targetEnts[def.id] = {
        def: def, icon: icon, mark: mark, bar: bar,
        status: -1, hpBucket: -1, barKey: null,
      };
    });
  }

  /**
   * 逐帧同步：位置、战损状态、血条。
   *
   * 由 app.js 在 `updateInferenceUi()` 里调用（播放推进、拖动时间轴、切案例、
   * 切换选中都会经过这里），与主图和整体TS同一个节奏。
   */
  function updateSituation(d, tick) {
    if (d && d !== data) setData(d);
    if (!data) return;
    lastTick = Math.max(0, tick | 0);
    // 隐藏时只记 tick，不写图元；show() 会用当前 tick 补一次。
    if (!isVisible()) return;
    if (!viewer) return;

    const frames = data.frames || [];
    if (!frames.length) return;
    const fr = frames[Math.min(lastTick, frames.length - 1)];
    const pos = {};
    (fr.targets || []).forEach(function (p) { pos[p.id] = p; });
    const api = _statusApi();

    Object.keys(targetEnts).forEach(function (id) {
      const rec = targetEnts[id];
      const p = pos[id];
      if (p) {
        const c = Cesium.Cartesian3.fromDegrees(p.lon, p.lat, 0);
        rec.icon.position = c;
        rec.mark.position = c;
        rec.bar.position = c;
      }

      let hp = 100, status = 0;
      if (api) {
        const st = api.at(data, id, lastTick);
        hp = st.hp; status = st.status;
      }

      if (status !== rec.status) {
        rec.status = status;
        rec.mark.show = (status === 3);
        if (status === 3) rec.mark.billboard.image = api.markerUri(3);
        rec.icon.billboard.color = (status === 3) ? DESTROYED_TINT : Cesium.Color.WHITE;
      }

      // 缓存键必须同时包含**档位**与 0.1% 精度的血量：只按整数血量比较时，
      // 70.1→70.0、35.1→35.0、0.1→0.0 这类跨档跃迁的取整值不变，
      // 血条会滞后数帧，损毁瞬间也不会切成深色空槽（与整体TS不一致）。
      const barKey = status + '|' + Math.round(hp * 10);
      if (barKey !== rec.barKey) {
        rec.barKey = barKey;
        rec.hpBucket = Math.round(hp);
        rec.bar.billboard.image = api
          ? api.barUri(hp, status, BAR_W, BAR_H)
          : undefined;
      }
    });

    viewer.scene.requestRender();
  }

  /** 主图选中变化时同步高亮（app.js 的 syncSelection 调用）。 */
  function setSelected(id) {
    selected = id || null;
    if (!viewer) return;
    Object.keys(targetEnts).forEach(function (tid) {
      const rec = targetEnts[tid];
      rec.icon.billboard.image = Symbology.symbolUri(
        rec.def.icon, rec.def.affiliation,
        { selected: tid === selected, domain: rec.def.domain });
    });
    viewer.scene.requestRender();
  }

  // ── 主视窗视野框 ────────────────────────────────────────────────────

  /**
   * 主相机的地面覆盖多边形。
   * 四角各发一条射线打地面。看向天空或瓦片未渲染时 `globe.pick` 会返回 undefined，
   * 此时退到"相机地面点的切平面"求交；仍失败则返回 null（不画框）。
   */
  function mainViewFootprint() {
    const scene = mainViewer.scene;
    const canvas = scene.canvas;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    const corners = [[0, 0], [w, 0], [w, h], [0, h]];
    const pts = [];

    for (let i = 0; i < 4; i++) {
      const ray = mainViewer.camera.getPickRay(new Cesium.Cartesian2(corners[i][0], corners[i][1]));
      if (!ray) return null;
      let hit = scene.globe.pick(ray, scene);
      if (!hit) hit = planeFallback(ray);
      if (!hit) return null;
      pts.push(hit);
    }
    return pts;
  }

  /** 把射线交到"相机地面点的切平面"，用于看向天空时也能得到有限远点。 */
  function planeFallback(ray) {
    const g = mainViewer.camera.positionCartographic;
    const origin = Cesium.Cartesian3.fromRadians(g.longitude, g.latitude, 0);
    const normal = Cesium.Ellipsoid.WGS84.geodeticSurfaceNormal(origin, new Cesium.Cartesian3());
    const plane = Cesium.Plane.fromPointNormal(origin, normal);
    return Cesium.IntersectionTests.rayPlane(ray, plane);
  }

  function syncFootprint(force) {
    if (!viewer || !mainViewer) return;
    const now = Date.now();
    if (!force && now - lastFootprintAt < FOOTPRINT_THROTTLE_MS) return;
    lastFootprintAt = now;

    const height = mainViewer.camera.positionCartographic.height;
    const pts = height < FOOTPRINT_MIN_HEIGHT_M ? null : mainViewFootprint();
    if (!footprintEntity) return;

    // key 由四个交点的实际坐标（按米取整）构成——不要退回 'set'/'none'，
    // 那样首次赋值后黄框就永久冻结（历史 bug，见 footprintsDebug 注释）。
    const key = pts
      ? pts.map(function (p) {
          return Math.round(p.x) + ',' + Math.round(p.y) + ',' + Math.round(p.z);
        }).join(';')
      : 'none';
    if (key === lastFootprintKey && !force) return;
    lastFootprintKey = key;
    footprintUpdates++;

    // 直接赋普通数组（ConstantProperty），不要用 CallbackProperty —— 见文件头第 4 条
    if (pts) {
      footprintEntity.polyline.positions = pts;
      footprintEntity.show = true;
    } else {
      footprintEntity.show = false;
    }
    viewer.scene.requestRender();
  }

  // ── 占领掩码（图层已下线，保留兼容） ───────────────────────────────

  function buildEntities() {
    footprintEntity = viewer.entities.add({
      id: 'overview-footprint',
      polyline: {
        positions: [],
        width: 2,
        arcType: Cesium.ArcType.NONE,   // 直连，避免测地插值在缩略图上抖动
        material: Cesium.Color.fromCssColorString('#ffd24a'),
        clampToGround: false,
      },
    });
    rebuildMaskEntities();
    applyMask();
    buildTargetEntities();
  }

  function rebuildMaskEntities() {
    if (!viewer) return;
    maskEntities.forEach(function (e) {
      try { viewer.entities.remove(e); } catch (err) { /* already gone */ }
    });
    maskEntities = [];
    maskState.hotspots.forEach(function (hs, i) {
      const w = hs.bbox[0], s = hs.bbox[1], e = hs.bbox[2], n = hs.bbox[3];
      const ent = viewer.entities.add({
        id: 'overview-occupation-' + i,
        name: (hs.name || ('hotspot-' + i)) + ' 占领掩码',
        show: false,
        polygon: {
          hierarchy: new Cesium.PolygonHierarchy(
            Cesium.Cartesian3.fromDegreesArray([w, s, e, s, e, n, w, n])),
          material: maskColor('neutral').withAlpha(0.5),
          outline: true,
          outlineColor: maskColor('neutral'),
          outlineWidth: 1,
        },
      });
      ent.maskStatus = null;
      maskEntities.push(ent);
    });
  }

  function applyMask() {
    if (!viewer) return;
    const on = !!maskState.visible;
    maskEntities.forEach(function (ent, i) {
      const hs = maskState.hotspots[i];
      const status = (hs && maskState.occupation[hs.name]) || 'neutral';
      if (ent.maskStatus !== status) {
        ent.maskStatus = status;
        ent.polygon.material = maskColor(status).withAlpha(0.5);
        ent.polygon.outlineColor = maskColor(status);
      }
      ent.show = on;
    });
    viewer.scene.requestRender();
  }

  function updateOccupation(tick, occupation, hotspots, visible) {
    maskState.tick = tick;
    maskState.occupation = occupation || {};
    maskState.hotspots = hotspots || [];
    maskState.visible = !!visible;
    maskUpdates++;
    maskVertices = 4 * maskState.hotspots.length;
    const key = maskState.hotspots.map(function (h) { return h.name || ''; }).join('|');
    if (viewer) {
      if (key !== maskState.key || maskEntities.length !== maskState.hotspots.length) {
        maskState.key = key;
        rebuildMaskEntities();
      }
      applyMask();
    } else {
      maskState.key = key;
    }
  }

  function occupationMaskDebug() {
    let shown = 0;
    maskEntities.forEach(function (e) { if (e && e.show) shown++; });
    return {
      tick: maskState.tick,
      vertices: maskVertices,
      entities: maskEntities.length,
      shown: shown,
      updates: maskUpdates,
      visible: maskState.visible,
    };
  }

  // ── 交互 ────────────────────────────────────────────────────────────

  function bindClick() {
    const handler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);
    handler.setInputAction(function (click) {
      // 先看是否点中目标：点中则走主视图的选中 + 飞行，而不是按地面跳转。
      const picked = viewer.scene.pick(click.position);
      const ent = picked && picked.id;
      if (ent && ent.ovTargetId && global.SituationHost && SituationHost.selectTarget) {
        SituationHost.selectTarget(ent.ovTargetId, true);
        return;
      }

      const ray = viewer.camera.getPickRay(click.position);
      let point = ray ? viewer.scene.globe.pick(ray, viewer.scene) : null;
      if (!point) {
        // SCENE2D 下 globe.pick 实测可能拿不到点；与 targetDebug 同口径改用
        // pickEllipsoid 兜底，避免"点空白没反应"。
        const ellipsoid = (viewer.scene.globe && viewer.scene.globe.ellipsoid)
          || Cesium.Ellipsoid.WGS84;
        point = viewer.camera.pickEllipsoid(click.position, ellipsoid);
      }
      if (!point) return;
      const carto = Cesium.Cartographic.fromCartesian(point);
      // 跳转前解除跟随：跟随锁定的是目标 ENU 参考系，主相机不能同时被两处驱动
      if (global.Follow && global.Follow.isActive()) global.Follow.detach('鸟瞰图跳转');
      const h = Math.max(mainViewer.camera.positionCartographic.height, 600000);
      mainViewer.camera.cancelFlight();
      mainViewer.camera.flyTo({
        destination: Cesium.Cartesian3.fromRadians(carto.longitude, carto.latitude, h),
        orientation: {
          heading: mainViewer.camera.heading,
          pitch: mainViewer.camera.pitch,
          roll: 0,
        },
        duration: 1.2,
      });
    }, Cesium.ScreenSpaceEventType.LEFT_CLICK);
  }

  // ── 生命周期 ────────────────────────────────────────────────────────

  /** 惰性创建。首次调用时容器必须已经可见（非 display:none），否则画布会是 0×0。 */
  function ensure() {
    if (created || !mainViewer) return false;
    const el = container();
    if (!el) return false;
    created = true;

    viewer = new Cesium.Viewer('overviewContainer', {
      baseLayer: new Cesium.ImageryLayer(OfflineAssets.createOverviewImageryProvider()),
      terrainProvider: new Cesium.EllipsoidTerrainProvider(),
      sceneMode: Cesium.SceneMode.SCENE2D,
      baseLayerPicker: false, geocoder: false, homeButton: false,
      sceneModePicker: false, navigationHelpButton: false,
      navigationInstructionsInitiallyVisible: false,
      animation: false, timeline: false, fullscreenButton: false, vrButton: false,
      infoBox: false, selectionIndicator: false, shouldAnimate: false,
    });

    if (viewer.imageryLayers.length === 0) {
      throw new Error('鸟瞰图底图未创建：Cesium ' + Cesium.VERSION + ' 需要 baseLayer 选项');
    }

    const scene = viewer.scene;
    scene.globe.maximumScreenSpaceError = 4;
    scene.globe.enableLighting = false;
    scene.globe.showGroundAtmosphere = false;
    scene.globe.baseColor = Cesium.Color.fromCssColorString('#dfe6ef');
    scene.skyAtmosphere.show = false;
    if (scene.skyBox) scene.skyBox.show = false;
    if (scene.sun) scene.sun.show = false;
    if (scene.moon) scene.moon.show = false;
    scene.fog.enabled = false;
    scene.screenSpaceCameraController.enableInputs = false;   // 取景由案例范围决定
    viewer.cesiumWidget.screenSpaceEventHandler.removeInputAction(
      Cesium.ScreenSpaceEventType.LEFT_DOUBLE_CLICK);

    viewer.resolutionScale = 0.8;
    // 组件底部不显示 Cesium / 影像归属文字（用户 2026-10-21 要求）。
    // creditContainer 是 CesiumWidget 的 DOM 容器；CSS 里另有一条兜底规则。
    const credit = viewer.cesiumWidget.creditContainer;
    if (credit) credit.style.display = 'none';
    buildEntities();
    bindClick();
    applyExtent();
    updateSituation(data, lastTick);
    syncFootprint(true);
    return true;
  }

  function show() {
    const el = document.getElementById('overviewWrap');
    if (!el) return;
    el.classList.remove('hidden');
    if (!created) ensure();
    if (!viewer) return;
    // display:none 会让容器尺寸归零，而 Cesium 只监听 window.resize，必须手动补
    viewer.resize();
    applyExtent();
    updateSituation(data, lastTick);
    syncFootprint(true);
    // 布局可能还没落定，下一帧再对齐一次
    setTimeout(function () {
      if (!viewer) return;
      viewer.resize();
      applyExtent();
      syncFootprint(true);
    }, 0);
  }

  function hide() {
    const el = document.getElementById('overviewWrap');
    if (el) el.classList.add('hidden');
  }

  function isVisible() {
    const el = document.getElementById('overviewWrap');
    return !!el && !el.classList.contains('hidden');
  }

  function onMainCameraChanged() {
    if (!viewer || !isVisible()) return;
    syncFootprint(false);
  }

  // ── 诊断 ────────────────────────────────────────────────────────────

  function debugInfo() {
    if (!viewer) return 'not created';
    const c = viewer.scene.canvas;
    return 'created  targets ' + Object.keys(targetEnts).length +
           '  layers ' + viewer.imageryLayers.length +
           '  canvas ' + c.width + 'x' + c.height +
           '  extent ' + (extent
             ? [extent.w, extent.s, extent.e, extent.n].map(function (v) { return v.toFixed(2); }).join(',')
             : '—') +
           '  visible ' + isVisible();
  }

  /**
   * 目标图元取证。自检据此断言：
   *   - 目标数量 = targets 数量；
   *   - 每个目标的位置与当前帧 frames 一致；
   *   - 血条/角标/颜色与 TargetStatus 的档位一致；
   *   - 取景包含全部目标。
   */
  function targetDebug() {
    const out = { caseId: data ? data.case_id : null, tick: lastTick, extent: extent, targets: {} };
    if (!viewer || !data) return out;
    const cam = viewer.camera.positionCartographic;
    out.camera = { lon: Cesium.Math.toDegrees(cam.longitude), lat: Cesium.Math.toDegrees(cam.latitude),
                   height: cam.height };
    out.viewRect = (function () {
      // SCENE2D 下 `camera.computeViewRectangle()` 实测返回 undefined（传不传椭球
      // 都一样），所以改用四角反投影：2D 仍是正射，pickEllipsoid 对四角都能给出
      // 椭球面交点，取经纬度的包围盒就是当前可见范围。
      const ellipsoid = (viewer.scene && viewer.scene.globe && viewer.scene.globe.ellipsoid)
        || Cesium.Ellipsoid.WGS84;
      const canvas = viewer.scene.canvas;
      const w = canvas.clientWidth, h = canvas.clientHeight;
      const corners = [[0, 0], [w, 0], [w, h], [0, h]];
      const cartos = [];
      for (let i = 0; i < corners.length; i++) {
        const p = viewer.camera.pickEllipsoid(
          new Cesium.Cartesian2(corners[i][0], corners[i][1]), ellipsoid);
        if (!p) return null;
        cartos.push(Cesium.Cartographic.fromCartesian(p));
      }
      let wDeg = Infinity, sDeg = Infinity, eDeg = -Infinity, nDeg = -Infinity;
      cartos.forEach(function (c) {
        const lon = Cesium.Math.toDegrees(c.longitude);
        const lat = Cesium.Math.toDegrees(c.latitude);
        if (lon < wDeg) wDeg = lon;
        if (lon > eDeg) eDeg = lon;
        if (lat < sDeg) sDeg = lat;
        if (lat > nDeg) nDeg = lat;
      });
      return { w: wDeg, s: sDeg, e: eDeg, n: nDeg };
    })();
    const frames = data.frames || [];
    const fr = frames[Math.min(lastTick, frames.length - 1)] || { targets: [] };
    const pos = {};
    (fr.targets || []).forEach(function (p) { pos[p.id] = p; });
    Object.keys(targetEnts).forEach(function (id) {
      const rec = targetEnts[id];
      const p = rec.icon.position && rec.icon.position.getValue
        ? rec.icon.position.getValue(viewer.clock.currentTime) : null;
      const carto = p ? Cesium.Cartographic.fromCartesian(p) : null;
      out.targets[id] = {
        show: rec.icon.show !== false,
        barShow: rec.bar.show !== false,
        markShow: !!rec.mark.show,
        barImage: rec.bar.billboard.image ? String(rec.bar.billboard.image.getValue()) : '',
        iconTint: rec.icon.billboard.color && rec.icon.billboard.color.getValue
          ? rec.icon.billboard.color.getValue().toCssHexString() : null,
        frameLon: pos[id] ? pos[id].lon : null,
        frameLat: pos[id] ? pos[id].lat : null,
        lon: carto ? Cesium.Math.toDegrees(carto.longitude) : null,
        lat: carto ? Cesium.Math.toDegrees(carto.latitude) : null,
        status: rec.status,
        hp: rec.hpBucket,
        selected: id === selected,
      };
    });
    return out;
  }

  function footprintDebug() {
    const c = viewer ? viewer.camera.positionCartographic : null;
    return {
      updates: footprintUpdates,
      keyLen: lastFootprintKey.length,
      keyHead: lastFootprintKey.slice(0, 60),
      shown: !!(footprintEntity && footprintEntity.show),
      vertices: footprintEntity && footprintEntity.polyline.positions
        ? (footprintEntity.polyline.positions.getValue() || []).length : 0,
      cameraLon: c ? c.longitude : null,
      cameraLat: c ? c.latitude : null,
      cameraHeight: c ? c.height : null,
    };
  }

  // 窗口跨过 1280px 断点时组件尺寸会变（268×190 ↔ 200×160），2D 相机高度
  // 不会自动跟随；重新取景一次，保证战场范围仍然完整可见。
  let _resizeTimer = null;
  window.addEventListener('resize', function () {
    if (!viewer) return;
    clearTimeout(_resizeTimer);
    _resizeTimer = setTimeout(function () {
      if (!viewer || !isVisible()) return;
      viewer.resize();
      applyExtent();
      syncFootprint(true);
    }, 200);
  });

  global.Overview = {
    init: function (main) { mainViewer = main; },
    // 诊断出口：自检需要读场景图元（label 是否真的生成/可见），
    // 而不是只看实体属性。
    _viewer: function () { return viewer; },
    debugInfo: debugInfo,
    targetDebug: targetDebug,
    footprintDebug: footprintDebug,
    updateOccupation: updateOccupation,
    occupationMaskDebug: occupationMaskDebug,
    setData: setData,
    clear: clear,
    updateSituation: updateSituation,
    setSelected: setSelected,
    ensure: ensure,
    show: show,
    hide: hide,
    isVisible: isVisible,
    onMainCameraChanged: onMainCameraChanged,
    setExtent: setExtent,
    setCenter: setCenter,
    resize: function () {
      if (viewer) { viewer.resize(); applyExtent(); }
    },
  };
})(window);
