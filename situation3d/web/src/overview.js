/* situation3d — 鸟瞰图（与主图共用视点的俯瞰视图）
 *
 * 设计：**和主图同一个视点，只把高度放大若干倍**。
 *
 * 早先的实现是一个固定范围的 SCENE2D 缩略图（永远框着台海 AOI）。问题有两个：
 * 一是它和主图没有关系，主图飞到关岛时它还在显示台海；二是 2D 正交视图看不出
 * 立体态势。现在改成共用经纬与朝向、只抬高高度，于是它天然就是"整个战场"的俯瞰：
 *
 * **高度固定**，中心锁定战场（由 setCenter 指定，默认台海）。这样它始终是同一张
 * "战场全图"，比例尺稳定、可比较；主图怎么飞都不影响它的尺度，只在图上留下一个
 * 黄色视野框表示"主图现在看的是哪一块"。
 *
 * 4000 km 在 44° 垂直视场下对应约 30° 跨度，正好覆盖三亚到关岛的西太战场。
 *
 * 用 SCENE2D 而不是 3D：鸟瞰图本来就是俯视，2D 正交视图正好；而且第二个 WebGL
 * 上下文在无头 SwiftShader 下用 3D 会被彻底饿死（实测画布全黑），2D 则稳定。
 *
 * 黄色闭合折线仍然保留：它标出主图当前视野在鸟瞰图里覆盖的范围，是两者的对应关系。
 *
 * 四个必须遵守的约束（都已核实）
 * ----------------------------
 * 1. **必须用 `baseLayer`**：Cesium 1.107 起 `Viewer` 的 `imageryProvider` 选项被移除
 *    且**静默忽略**，不传 baseLayer 会回落到 Ion 在线底图，既违反离线要求也会因缺
 *    token 报错。
 * 2. **不共用主视图的 provider 实例**：主视图的 provider 被 `TileCache.attach` 与
 *    `instrument` 各包了一层，共用会让鸟瞰图的请求与 404 计入主视图的 `imageryStats`，
 *    `?diag=1` 全部失真；`errorEvent` 也是共享的 Event，无法按 viewer 区分。
 * 3. **视野框用普通数组赋值，不用 `CallbackProperty`**：前者是恒定属性，不会让场景
 *    被判定为持续时变。
 * 4. **不用 `requestRenderMode`**：实测在无头 SwiftShader 下该视图会停在黑屏（瓦片
 *    已加载完但画布未重绘）。这块画布只占主视图约 1.5% 像素，不值得承担按需渲染的
 *    不确定性。
 *
 * 隐藏/显示的一个坑：`display:none` 会让容器尺寸归零，而 Cesium 只监听 window.resize，
 * 重新显示不会自愈。所以 show() 必须走 resize() + setView()。
 *
 * 已知现象（无头软件光栅化环境）：主相机压到几公里时这块画布会变黑，而 900 km /
 * 6000 km 时正常。相关性指向 SwiftShader 下的资源争用——主视图在城区要上传大量
 * z16-17 纹理，第二个 WebGL 上下文被饿死，而不是这里的逻辑有问题。真实 GPU 环境
 * 下待复核。
 */

'use strict';

(function (global) {
  // 主图低于这个高度时不画视野框：那时它退化成一个点，没有信息量。
  const FOOTPRINT_MIN_HEIGHT_M = 20000;
  // 视野框更新节流。跟随时主相机每帧都在动，放宽到 500 ms。
  const FOOTPRINT_THROTTLE_MS = 500;

  // 固定的鸟瞰高度。不再跟随主图——鸟瞰图的价值在于"始终是同一张战场全图"，
  // 高度一变比例尺就变，两张图之间没法比较。
  const FIXED_HEIGHT_M = 4000000;

  // 战场中心。默认台海，加载态势后由 app.js 用态势的 AOI 中心调用 setCenter 覆盖。
  let center = { lon: 119.60, lat: 24.40 };

  let viewer = null;          // 鸟瞰图 Viewer
  let mainViewer = null;      // 主 3D Viewer
  let footprintEntity = null;
  let lastFootprintAt = 0;
  let lastFootprintKey = 'init';   // 'init' | 'none' | 'set'，用于跳过无变化的重复赋值
  let created = false;

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

  /**
   * 与主图共用视点：经纬与朝向照搬，只把高度放大。
   *
   * `camera.heading` / `camera.pitch` 的 getter 有副作用（内部临时改写
   * `camera._transform` 并两次 updateMembers），所以只在节流后的同步里各读一次。
   */
  /** 把鸟瞰图对准固定高度与战场中心。 */
  function syncCamera() {
    if (!viewer) return;
    // SCENE2D 下 destination 的高度决定正交视锥的宽度，也就是"看多大范围"。
    // 2D 是正俯视，heading/pitch 无意义，不传。
    viewer.camera.setView({
      destination: Cesium.Cartesian3.fromRadians(center.lon, center.lat, FIXED_HEIGHT_M),
    });
    viewer.scene.requestRender();
  }

  /**
   * 设定战场中心。加载态势后由 app.js 调用，参数是 AOI / 战区中心。
   * 与主相机无关——主相机只通过视野框在这张图上留下痕迹。
   */
  function setCenter(lon, lat) {
    if (!isFinite(lon) || !isFinite(lat)) return;
    center.lon = Cesium.Math.toRadians(lon);
    center.lat = Cesium.Math.toRadians(lat);
    syncCamera();
    syncFootprint(true);
  }

  function syncFootprint(force) {
    if (!viewer || !mainViewer) return;
    const now = Date.now();
    if (!force && now - lastFootprintAt < FOOTPRINT_THROTTLE_MS) return;
    lastFootprintAt = now;

    const height = mainViewer.camera.positionCartographic.height;
    const pts = height < FOOTPRINT_MIN_HEIGHT_M ? null : mainViewFootprint();
    if (!footprintEntity) return;

    // 状态没变就不重复赋值与重绘。
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

    viewer = new Cesium.Viewer('overviewContainer', {
      baseLayer: new Cesium.ImageryLayer(OfflineAssets.createOverviewImageryProvider()),
      terrainProvider: new Cesium.EllipsoidTerrainProvider(),
      // SCENE2D：鸟瞰图本来就是俯视，2D 正交视图正好；而且第二个 WebGL 上下文
      // 在无头 SwiftShader 下用 3D 会被彻底饿死（实测画布全黑），2D 则稳定。
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
    scene.screenSpaceCameraController.enableInputs = false;   // 机位完全由主相机决定
    viewer.cesiumWidget.screenSpaceEventHandler.removeInputAction(
      Cesium.ScreenSpaceEventType.LEFT_DOUBLE_CLICK);

    viewer.resolutionScale = 0.8;
    buildEntities();
    bindClick();
    syncCamera();
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
    syncCamera();
    syncFootprint(true);
    // 布局可能还没落定，下一帧再对齐一次
    setTimeout(function () {
      if (!viewer) return;
      viewer.resize();
      syncCamera();
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

  /**
   * 主相机移动后调用（由 app.js 的 camera.changed 驱动）。
   *
   * 只更新视野框——鸟瞰图的相机是固定的，不跟主图动。这样它才是一张稳定的
   * 战场全图，而不是"主图的小号复制"。
   */
  function onMainCameraChanged() {
    if (!viewer || !isVisible()) return;
    syncFootprint(false);
  }

  /** 诊断读数。 */
  function debugInfo() {
    if (!viewer) return 'not created';
    const c = viewer.scene.canvas;
    const h = viewer.camera.positionCartographic.height / 1000;
    return 'created  layers ' + viewer.imageryLayers.length +
           '  canvas ' + c.width + 'x' + c.height +
           '  fixedHeight ' + h.toFixed(0) + ' km' +
           '  visible ' + isVisible();
  }

  global.Overview = {
    init: function (main) { mainViewer = main; },
    debugInfo: debugInfo,
    ensure: ensure,
    show: show,
    hide: hide,
    isVisible: isVisible,
    onMainCameraChanged: onMainCameraChanged,
    syncCamera: syncCamera,
    setCenter: setCenter,
    resize: function () {
      if (viewer) { viewer.resize(); syncCamera(); }
    },
  };
})(window);
