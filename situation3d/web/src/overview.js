/* situation3d — 鸟瞰图（第二个 Cesium 2D 视图）
 *
 * 需求：一个可显示/隐藏的鸟瞰图，用真实地球缩略图展示全局态势，并与主相机联动。
 *
 * 为什么是独立的第二个 Viewer 而不是自绘 canvas
 * --------------------------------------------
 * 用户明确要求"真实地球缩略图"。自绘 canvas 只能画示意框，给不出真实影像；而
 * 第二个 Cesium Viewer 在 SCENE2D 下只渲染一块 240×172 的画布，配 requestRenderMode
 * 后稳态渲染次数接近 0，代价可控。
 *
 * 三条必须遵守的约束（都已核实）
 * ------------------------------
 * 1. **必须用 `baseLayer`**：Cesium 1.107 起 `Viewer` 的 `imageryProvider` 选项被移除
 *    且**静默忽略**，不传 baseLayer 会回落到 Ion 在线底图，既违反离线要求也会因缺
 *    token 报错。
 * 2. **不共用主视图的 provider 实例**：主视图的 provider 被 `TileCache.attach` 与
 *    `instrument` 各包了一层，共用会让 2D 的请求与 404 计入主视图的 `imageryStats`，
    `?diag=1` 全部失真；`errorEvent` 也是共享的 Event，无法按 viewer 区分。这里
 *    自建一个 `maximumLevel: 6` 的 provider，只靠 HTTP 缓存（/tiles/ 已是 immutable）。
 * 3. **视野框用普通数组赋值，不用 `CallbackProperty`**：前者是恒定属性，不会让场景
 *    被判定为持续时变。
 *
 * 关于按需渲染（`requestRenderMode`）：实测在无头 SwiftShader 下该视图会停在黑屏
 * （瓦片已加载完但画布未重绘）。这块画布只有约 190×116，占主视图像素的 1.5%，
 * 不值得为它承担按需渲染的不确定性，因此保持默认的每帧渲染。
 *
 * 隐藏/显示的一个坑：`display:none` 会让容器尺寸归零，而 Cesium 只监听 window.resize，
 * 重新显示不会自愈。所以 show() 必须走 resize() + setView()。
 *
 * 已知现象（无头软件光栅化环境）：主相机压到几公里时这块画布会变黑，而 900 km /
 * 6000 km 时正常。相关性指向 SwiftShader 下的资源争用——主视图在城区要上传大量
 * z16-17 纹理，第二个 WebGL 上下文被饿死，而不是这里的逻辑有问题（视野框在低空
 * 本来就按设计不画）。真实 GPU 环境下待复核。
 *
 * 已知现象（无头软件光栅化环境）：主相机压到几公里时这块画布会变黑，而 900 km /
 * 6000 km 时正常。相关性指向 SwiftShader 下的资源争用——主视图在城区要上传大量
 * z16-17 纹理，第二个 WebGL 上下文被饿死，而不是这里的逻辑有问题（视野框在低空
 * 本来就按设计不画）。真实 GPU 环境下待复核。
 */

'use strict';

(function (global) {
  // 主视图低于这个高度时不画视野框：此时视野框会退化成极小的矩形，且频繁走
  // 切平面兜底，既无信息量又费算力。
  const FOOTPRINT_MIN_HEIGHT_M = 20000;
  // 视野框更新节流。跟随时主相机每帧都在动，放宽到 500 ms 以免拖累第二个上下文。
  const FOOTPRINT_THROTTLE_MS = 500;

  let overviewRect = null;    // 鸟瞰图固定的地理范围
  let viewer = null;          // 2D 鸟瞰图 Viewer
  let mainViewer = null;      // 主 3D Viewer
  let footprintEntity = null;
  let lastFootprintAt = 0;
  let lastFootprintKey = 'init';   // 'init' | 'none' | 'set'，用于跳过无变化的重复赋值
  let created = false;

  const _enuM = new Cesium.Matrix4();
  const _enuInv = new Cesium.Matrix4();

  function container() { return document.getElementById('overviewContainer'); }

  /**
   * 主相机的地面覆盖多边形。
   *
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

    // 状态没变就不重复赋值与重绘。低空时视野框会被关闭，此时相机每动一下都重新
    // 赋值空数组并 requestRender 是纯浪费，也可能让第二个上下文被反复打断。
    const key = pts ? 'set' : 'none';
    if (key === lastFootprintKey && !force) return;
    lastFootprintKey = key;

    // 直接赋普通数组（ConstantProperty），不要用 CallbackProperty —— 见文件头第 3 条
    if (pts) {
      footprintEntity.polyline.positions = pts;
      footprintEntity.show = true;
    } else {
      footprintEntity.show = false;
    }
    viewer.scene.requestRender();
  }

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
  }

  function bindClick() {
    const handler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);
    handler.setInputAction(function (click) {
      const ray = viewer.camera.getPickRay(click.position);
      const point = ray ? viewer.scene.globe.pick(ray, viewer.scene) : null;
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

  /**
   * 惰性创建。首次调用时容器必须已经可见（非 display:none），否则画布会是 0×0。
   */
  function ensure() {
    if (created || !mainViewer) return false;
    const el = container();
    if (!el) return false;
    created = true;

    const bbox = (global.OfflineAssets && OfflineAssets.manifest().bbox) || [115, 18, 125, 29];
    // 每边外扩 15%，让主视野框在默认机位下不会被裁掉
    const padLon = (bbox[2] - bbox[0]) * 0.15;
    const padLat = (bbox[3] - bbox[1]) * 0.15;
    overviewRect = Cesium.Rectangle.fromDegrees(
      bbox[0] - padLon, bbox[1] - padLat, bbox[2] + padLon, bbox[3] + padLat);

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
    scene.globe.maximumScreenSpaceError = 8;      // 缩略图不需要细节
    scene.globe.enableLighting = false;
    scene.globe.showGroundAtmosphere = false;
    scene.globe.baseColor = Cesium.Color.fromCssColorString('#dfe6ef');
    scene.skyAtmosphere.show = false;
    if (scene.skyBox) scene.skyBox.show = false;
    if (scene.sun) scene.sun.show = false;
    if (scene.moon) scene.moon.show = false;
    scene.fog.enabled = false;
    scene.screenSpaceCameraController.enableInputs = false;   // 机位完全由主相机决定
    viewer.cesiumWidget.screenSpaceEventHandler.removeInputAction(
      Cesium.ScreenSpaceEventType.LEFT_DOUBLE_CLICK);

    viewer.resolutionScale = 0.8;
    buildEntities();
    bindClick();
    fitExtent();
    return true;
  }

  /**
   * 把 2D 相机 fit 到固定范围。
   *
   * **必须在容器尺寸稳定之后调用**：SCENE2D 的 `setView(Rectangle)` 会按调用那一刻的
   * 画布尺寸反算正交视锥宽度，如果容器刚解除 `display:none` 就 setView，画布尺寸
   * 还没生效，地球会缩在一角、其余留白。所以这里先 resize 再 setView，并且
   * 在 show() 里用一次 setTimeout(0) 补一次，确保布局已完成。
   */
  function fitExtent() {
    if (!viewer || !overviewRect) return;
    viewer.resize();
    viewer.camera.setView({ destination: overviewRect });
    viewer.scene.requestRender();
  }

  function show() {
    const el = document.getElementById('overviewWrap');
    if (!el) return;
    el.classList.remove('hidden');
    if (!created) ensure();
    if (!viewer) return;
    // display:none 会让容器尺寸归零，而 Cesium 只监听 window.resize，必须手动补
    fitExtent();
    syncFootprint(true);
    // 布局可能还没落定，下一帧再 fit 一次
    setTimeout(function () { fitExtent(); syncFootprint(true); }, 0);
  }

  function hide() {
    const el = document.getElementById('overviewWrap');
    if (el) el.classList.add('hidden');
  }

  function isVisible() {
    const el = document.getElementById('overviewWrap');
    return !!el && !el.classList.contains('hidden');
  }

  /** 主相机移动后调用（由 app.js 的 camera.changed 驱动）。 */
  function onMainCameraChanged() {
    if (!viewer || !isVisible()) return;
    syncFootprint(false);
  }

  /** 诊断读数：2D 视图是否建好、影像层是否在、画布尺寸是否正常。 */
  function debugInfo() {
    if (!viewer) return 'not created';
    const c = viewer.scene.canvas;
    return 'created  layers ' + viewer.imageryLayers.length +
           '  canvas ' + c.width + 'x' + c.height +
           '  tilesLoaded ' + viewer.scene.globe.tilesLoaded +
           '  visible ' + isVisible();
  }

  global.Overview = {
    init: function (main) { mainViewer = main; },
    debugInfo: debugInfo,
    fitExtent: fitExtent,
    ensure: ensure,
    show: show,
    hide: hide,
    isVisible: isVisible,
    onMainCameraChanged: onMainCameraChanged,
    resize: function () { if (viewer) { viewer.resize(); viewer.scene.requestRender(); } },
  };
})(window);
