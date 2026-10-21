/* situation3d — 目标属性（选中目标的 3D 侧视预览 + 实时属性）
 *
 * 定位
 * ----
 * 「目标图层」下的可选子图层。勾选后点击场景里的目标，在右下角 Legend 上方弹出
 * 属性面板：上半部是目标的三维模型（与主场景同源），下半部是基础/运动/装备/
 * 自定义属性与战损状态。取消勾选即隐藏并销毁预览视图。
 *
 * 为什么是"第二个 Cesium Viewer"
 * -------------------------------
 * 面板要紧贴主视图、又要保证模型的外观/朝向/战损与主场景逐帧一致，唯一不复制
 * 状态的实现就是**镜像主实体**：同一个 GLB URI、同一个 scale、同一个 CZML 采样
 * 位置/四元数（按主 Viewer 的时钟取值）、同一份战损着色（targetCombatFx 直接改写
 * 主实体的 model.color 等属性，这里每帧读回）、同一套关节回调（nodeTransformations
 * 读的是目标特效模块里共享的 rigAngles / wheelStates）。任何一处都不另存"面板版
 * 本"，所以不会出现两套状态漂移。
 *
 * 性能
 * ----
 * 预览 Viewer 只在「图层勾选 + 选中目标」同时成立时才创建，取消选中/取消勾选
 * 立即 destroy，GLB 走浏览器缓存；requestRenderMode 下仅在播放、换帧或交互时
 * 请求一帧，暂停时零帧渲染。主场景只多一次属性读取与相机 setView（每帧微秒级）。
 *
 * 视角（用户 2026-10-21 指定）
 * ----------------------------
 * **始终侧视图**：相机固定在目标右舷（模型局部 +Y）方向、略高于水平面，看向
 * 目标；目标转向时机位跟着绕，因此永远是"侧面剪影"，滚轮可缩放，不提供自由
 * 旋转（避免偏离侧视语义）。
 *
 * 依赖：Cesium、TargetStatus、EntityLod；TargetFx / TargetCombatFx 可选（缺省时
 * 没有关节动画，但预览仍可用）。
 */
(function (root) {
  'use strict';

  var C = root.Cesium;

  var mainViewer = null;      // 主 3D Viewer
  var api = null;             // app.js 注入的只读环境（见 targetPropsApi）
  var enabled = false;        // 「目标属性」图层开关
  var currentId = null;       // 当前选中目标 id（与 state.selected 同步）
  var viewer = null;          // 预览用第二 Viewer
  var previewEnt = null;
  var proxyEnt = null;        // 代理炮塔（与主场景同款，近景可见）
  var groundEnt = null;       // 海面/地面参照盘（体现下沉/坠地）
  var zoom = 1;               // 滚轮缩放系数
  var renderCount = 0;
  var attemptedFor = null;    // 当前预览是为哪个目标建的（无模型目标也要记住，避免反复重试）
  var resizeBound = false;
  var wheelBound = false;
  var fallbackTimer = null;   // requestRenderMode 看门狗（见 buildPreview）
  var lastSig = '';           // 上一帧的位姿/着色签名，暂停时据此决定是否重绘

  var lastHudMs = 0;
  var lastPreviewMs = 0;

  var scratchM3 = new C.Matrix3();
  var scratchR = new C.Cartesian3();
  var scratchU = new C.Cartesian3();
  var scratchP = new C.Cartesian3();
  var scratchD = new C.Cartesian3();

  var dom = {};
  var cells = {};             // 动态字段 key -> <b> 节点

  function $(id) {
    if (!dom[id]) dom[id] = document.getElementById(id);
    return dom[id];
  }

  function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }

  // ------------------------------------------------------------------
  // 数据读取（全部来自主场景/案例数据，不保存副本）
  // ------------------------------------------------------------------

  function recOf(id) {
    return api && api.rec ? api.rec(id || currentId) : null;
  }

  function timeNow() {
    return mainViewer ? mainViewer.clock.currentTime : null;
  }

  /** Property 或裸值统一取值（targetCombatFx 在裸值与 ConstantProperty 间切换）。 */
  function readProp(p, t, dflt) {
    if (p === undefined || p === null) return dflt;
    if (typeof p.getValue === 'function') {
      try {
        var v = p.getValue(t || timeNow());
        return (v === undefined || v === null) ? dflt : v;
      } catch (e) {
        return dflt;
      }
    }
    return p;
  }

  function frameAt(tick) {
    var data = api && api.data ? api.data() : null;
    if (!data) return null;
    var frames = data.frames || [];
    if (!frames.length) return null;
    var fr = frames[Math.min(Math.max(0, tick | 0), frames.length - 1)];
    if (!fr) return null;
    var list = fr.targets || [];
    for (var i = 0; i < list.length; i++) {
      if (list[i].id === currentId) return list[i];
    }
    return null;
  }

  function statusOf() {
    var data = api && api.data ? api.data() : null;
    if (!data || !currentId) return null;
    try {
      return root.TargetStatus ? TargetStatus.at(data, currentId, api.tick()) : null;
    } catch (e) {
      return null;
    }
  }

  function formClass(def) {
    return root.EntityLod ? EntityLod.formClassOf(def) : null;
  }

  function modelLength(def) {
    var cls = formClass(def);
    return (root.EntityLod && EntityLod.REAL_LENGTH_M[cls]) || 15;
  }

  function sideLabel(affiliation) {
    return affiliation === 'blue' ? '蓝方' : '红方';
  }

  function domainLabel(domain) {
    var d = String(domain || '').toLowerCase();
    return d === 'sea' || d === 'naval' || d === 'maritime' ? '海上'
      : d === 'air' ? '空中' : d === 'land' ? '地面' : (domain || '—');
  }

  function motionLabel(def, s) {
    var speed = Number(s && s.speed_kn);
    if (!isFinite(speed)) speed = 0;
    var d = String(def && def.domain || '').toLowerCase();
    if (s && s.reached_end) return '已到达终点';
    if (speed < 0.2) return d === 'air' ? '悬停/静止' : (d === 'sea' ? '停泊' : '静止');
    if (d === 'air') return '飞行';
    if (d === 'sea' || d === 'naval' || d === 'maritime') return '航行';
    return '机动';
  }

  function fmtNum(v, digits) {
    var n = Number(v);
    if (!isFinite(n)) return '—';
    return n.toFixed(digits === undefined ? 1 : digits);
  }

  function fmtAlt(v) {
    var n = Number(v);
    if (!isFinite(n)) return '—';
    return Math.abs(n) >= 1000 ? (n / 1000).toFixed(2) + ' km' : n.toFixed(0) + ' m';
  }

  function fmtKm(v) {
    var n = Number(v);
    return isFinite(n) ? n.toFixed(n >= 100 ? 0 : 1) + ' km' : '—';
  }

  // ------------------------------------------------------------------
  // 预览 Viewer
  // ------------------------------------------------------------------

  function ensureViewer() {
    if (viewer) return true;
    var host = $('tprView');
    if (!host) return false;
    try {
      viewer = new C.Viewer(host, {
        baseLayer: false,
        terrainProvider: new C.EllipsoidTerrainProvider(),
        sceneMode: C.SceneMode.SCENE3D,
        skyBox: false,
        skyAtmosphere: false,
        baseLayerPicker: false, geocoder: false, homeButton: false,
        sceneModePicker: false, navigationHelpButton: false,
        navigationInstructionsInitiallyVisible: false,
        animation: false, timeline: false, fullscreenButton: false, vrButton: false,
        infoBox: false, selectionIndicator: false, shouldAnimate: false,
        requestRenderMode: true,
        maximumRenderTimeChange: Infinity,
      });
    } catch (e) {
      console.warn('[target-props] 预览视图创建失败：', e);
      viewer = null;
      return false;
    }

    var scene = viewer.scene;
    scene.globe.show = false;
    // 用户 2026-10-21：预览背景与系统 UI 同为浅色（白底），不再用深色视口。
    scene.backgroundColor = C.Color.WHITE;
    if (scene.skyBox) scene.skyBox.show = false;
    if (scene.sun) scene.sun.show = false;
    if (scene.moon) scene.moon.show = false;
    scene.fog.enabled = false;
    scene.screenSpaceCameraController.enableInputs = false;
    // 与主视图一致：按 devicePixelRatio 渲染，Windows 125%/150% 缩放下不发虚。
    viewer.useBrowserRecommendedResolution = false;
    viewer.cesiumWidget.screenSpaceEventHandler.removeInputAction(
      C.ScreenSpaceEventType.LEFT_DOUBLE_CLICK);
    var credit = viewer.cesiumWidget.creditContainer;
    if (credit) credit.style.display = 'none';

    // 固定"头灯"式方向光：模型朝向变化时不受地球时刻/太阳方位影响。
    scene.light = new C.DirectionalLight({
      direction: C.Cartesian3.clone(C.Cartesian3.UNIT_Z),
    });

    scene.postRender.addEventListener(function () { renderCount++; });

    if (!resizeBound) {
      resizeBound = true;
      window.addEventListener('resize', function () {
        if (!viewer) return;
        viewer.resize();
        try { viewer.scene.requestRender(); } catch (e) { /* 已销毁 */ }
      });
    }

    return true;
  }

  function requestRender() {
    // display:none 时容器为 0×0，Cesium 渲染会报错；隐藏态不请求。
    if (viewer && enabled && currentId) viewer.scene.requestRender();
  }

  /**
   * 滚轮缩放只绑定一次（#tprView 是静态 DOM，Viewer 会被反复销毁重建；
   * 在 ensureViewer 里绑定会让监听器随重建次数累积、缩放倍率被连乘）。
   */
  function bindWheel() {
    if (wheelBound) return;
    var host = $('tprView');
    if (!host) return;
    wheelBound = true;
    host.addEventListener('wheel', function (ev) {
      if (!enabled || !currentId || !viewer) return;
      ev.preventDefault();
      ev.stopPropagation();
      zoom = clamp(zoom * (ev.deltaY > 0 ? 1.12 : 0.89), 0.45, 3.0);
      // 先应用新机位再请求渲染，否则 requestRenderMode 下可能画出旧视角。
      updateCamera();
      requestRender();
    }, { passive: false });
  }

  function stopFallback() {
    if (fallbackTimer) { clearInterval(fallbackTimer); fallbackTimer = null; }
  }

  function removePreviewEntities() {
    if (previewEnt) { try { viewer && viewer.entities.remove(previewEnt); } catch (e) {} previewEnt = null; }
    if (proxyEnt) { try { viewer && viewer.entities.remove(proxyEnt); } catch (e) {} proxyEnt = null; }
    if (groundEnt) { try { viewer && viewer.entities.remove(groundEnt); } catch (e) {} groundEnt = null; }
  }

  function destroyPreview() {
    removePreviewEntities();
    stopFallback();
    attemptedFor = null;
    lastSig = '';
    if (viewer) {
      try { viewer.destroy(); } catch (e) {}
      viewer = null;
    }
    renderCount = 0;
  }

  /** 主实体的位置/朝向在本帧的取值（CZML 采样，与主场景同一时钟）。 */
  function poseNow(rec) {
    var t = timeNow();
    var pos = rec && rec.entity && rec.entity.position
      ? readProp(rec.entity.position, t, null) : null;
    var q = rec && rec.entity && rec.entity.orientation
      ? readProp(rec.entity.orientation, t, null) : null;
    if (pos && !q) {
      // CZML 尚未就绪/没有姿态采样时按帧里的航向兜底，避免预览停在世界原点朝向。
      var s = frameAt(api.tick());
      if (s && root.TargetMotion && TargetMotion.orientationQuat) {
        var fwd = root.EntityLod
          ? EntityLod.forwardAxis(EntityLod.formClassOf(rec.def)) : '-X';
        q = TargetMotion.orientationQuat(pos, C.Math.toRadians(Number(s.heading) || 0),
          0, 0, fwd, new C.Quaternion());
      }
    }
    return { pos: pos, q: q, t: t };
  }

  /**
   * 位姿 + 着色签名。暂停时主场景仍可能推进「击毁下沉/坠地」与命中闪光
   * （targetCombatFx 用 Date.now() 插值），预览必须跟着重绘，不能只看 tick。
   */
  function signature(rec, pose) {
    if (!pose || !pose.pos) return 'no-pose';
    var t = timeNow();
    var m = rec && rec.entity ? rec.entity.model : null;
    var col = readProp(m && m.color, t, null);
    var q = pose.q;
    return pose.pos.x.toFixed(2) + ',' + pose.pos.y.toFixed(2) + ',' + pose.pos.z.toFixed(2)
      + '|' + (q ? (q.x.toFixed(4) + ',' + q.y.toFixed(4) + ',' + q.z.toFixed(4) + ',' + q.w.toFixed(4)) : '-')
      + '|' + zoom.toFixed(3)
      + '|' + (col && col.toCssHexString ? col.toCssHexString() : '')
      + '|' + readProp(m && m.colorBlendAmount, t, '')
      + '|' + readProp(m && m.silhouetteSize, t, '');
  }

  function buildPreview() {
    var rec = recOf();
    if (!rec || !rec.def) return;
    var def = rec.def;
    var url = root.EntityLod ? EntityLod.modelUrlFor(def) : null;
    if (!url) {
      // 没有 GLB 的目标：整个 3D 视口不显示（用户 2026-10-21 要求），
      // 也不创建第二个 WebGL 上下文，只保留属性字段。
      if (viewer) destroyPreview();
      attemptedFor = currentId;
      setViewVisible(false);
      return;
    }
    // 视口可能是上一目标（无模型）时隐藏的：先显示再建 Viewer，
    // 否则 Cesium 拿到的容器是 0×0。
    setViewVisible(true);
    if (!ensureViewer()) { showNoModel('预览视图不可用'); return; }
    attemptedFor = currentId;
    removePreviewEntities();
    hideNoModel();
    zoom = 1;
    lastSig = '';

    // 节点动画与主场景共用同一套共享状态（旋翼/雷达/发射筒 + 车轮）。
    var nodeTransforms = undefined;
    try {
      if (root.TargetFx && TargetFx.modelNodeTransformations) {
        nodeTransforms = TargetFx.modelNodeTransformations(def);
      } else if (root.TargetCombatFx && TargetCombatFx.nodeTransformations) {
        nodeTransforms = TargetCombatFx.nodeTransformations(def);
      }
    } catch (e) { nodeTransforms = undefined; }

    var silhouetteDefault = C.Color.BLACK;

    previewEnt = viewer.entities.add({
      id: 'tpr-preview',
      position: new C.CallbackProperty(function () {
        var rec2 = recOf();
        if (!rec2) return undefined;
        return readProp(rec2.entity.position, timeNow(), undefined);
      }, false),
      orientation: new C.CallbackProperty(function () {
        var rec2 = recOf();
        if (!rec2) return undefined;
        var q = readProp(rec2.entity.orientation, timeNow(), undefined);
        if (q) return q;
        var pose = rec2 ? poseNow(rec2) : null;
        return pose ? pose.q : undefined;
      }, false),
      model: {
        uri: url,
        scale: readProp(rec.entity.model && rec.entity.model.scale, null, 1) || 1,
        minimumPixelSize: 0,
        maximumScale: 20000,
        // 阵营着色/战损着色都从主实体的 model 属性读回，保证同一帧同一颜色。
        color: new C.CallbackProperty(function () {
          var rec2 = recOf();
          return readProp(rec2 && rec2.entity.model && rec2.entity.model.color, timeNow(),
            C.Color.WHITE);
        }, false),
        colorBlendMode: C.ColorBlendMode.MIX,
        colorBlendAmount: new C.CallbackProperty(function () {
          var rec2 = recOf();
          return readProp(rec2 && rec2.entity.model && rec2.entity.model.colorBlendAmount,
            timeNow(), 0.2);
        }, false),
        silhouetteColor: new C.CallbackProperty(function () {
          var rec2 = recOf();
          return readProp(rec2 && rec2.entity.model && rec2.entity.model.silhouetteColor,
            timeNow(), silhouetteDefault);
        }, false),
        silhouetteSize: new C.CallbackProperty(function () {
          var rec2 = recOf();
          return readProp(rec2 && rec2.entity.model && rec2.entity.model.silhouetteSize,
            timeNow(), 0);
        }, false),
        nodeTransformations: nodeTransforms,
      },
    });

    try {
      var ready = previewEnt.model && previewEnt.model.readyEvent;
      if (ready && ready.addEventListener) {
        ready.addEventListener(function () { requestRender(); });
      }
    } catch (e) { /* readyEvent 不存在也不影响显示 */ }

    buildProxy(def);
    buildGround(def);

    // 模型异步加载不会自动触发 requestRenderMode 的帧，补几次。
    [120, 400, 900, 1600].forEach(function (ms) {
      setTimeout(requestRender, ms);
    });

    // 看门狗：2.5 s 内一帧都没画出来（慢 GLB / 上下文饥饿）就退化为 5 fps
    // 周期请求，直到真正出帧；避免 requestRenderMode 下永久黑屏。
    var r0 = renderCount;
    setTimeout(function () {
      if (!viewer || !previewEnt || fallbackTimer) return;
      if (renderCount > r0) return;
      fallbackTimer = setInterval(function () {
        if (!viewer) { stopFallback(); return; }
        requestRender();
        if (renderCount > r0 + 2) stopFallback();
      }, 200);
    }, 2500);
  }

  /** 代理炮塔：与主场景的 fxgun-* 同模型、同偏移、同后坐（共享 recoil 状态）。 */
  function buildProxy(def) {
    if (!root.TargetCombatFx || !TargetCombatFx.proxyState) return;
    var ps = null;
    try { ps = TargetCombatFx.proxyState(def); } catch (e) { ps = null; }
    if (!ps || !ps.model) return;
    var off = ps.offset;
    proxyEnt = viewer.entities.add({
      id: 'tpr-proxy-gun',
      position: new C.CallbackProperty(function () {
        var rec2 = recOf();
        if (!rec2) return undefined;
        var pose = poseNow(rec2);
        if (!pose.pos) return undefined;
        C.Cartesian3.fromElements(off[0], off[1], off[2], scratchP);
        if (pose.q) {
          C.Matrix3.fromQuaternion(pose.q, scratchM3);
          C.Matrix3.multiplyByVector(scratchM3, scratchP, scratchP);
        }
        return C.Cartesian3.add(pose.pos, scratchP, new C.Cartesian3());
      }, false),
      orientation: new C.CallbackProperty(function () {
        var rec2 = recOf();
        return rec2 ? readProp(rec2.entity.orientation, timeNow(), undefined) : undefined;
      }, false),
      model: {
        uri: ps.model,
        scale: 1,
        minimumPixelSize: 0,
        nodeTransformations: {
          fx_barrel: new C.NodeTransformationProperty({
            translation: new C.CallbackProperty(function () {
              var st = null;
              try { st = TargetCombatFx.proxyState(def); } catch (e) {}
              var recoil = st ? st.recoil : 0;
              return new C.Cartesian3(0, -recoil, 0);
            }, false),
          }),
        },
      },
    });
  }

  /** 海面/地面参照盘：击毁后主实体下沉/坠地，参照盘不动，侧视里能看出"沉下去"。 */
  function buildGround(def) {
    var d = String(def.domain || '').toLowerCase();
    var isSea = d === 'sea' || d === 'naval' || d === 'maritime';
    var isLand = d === 'land';
    if (!isSea && !isLand) return;           // 空中目标不画参照面
    var L = modelLength(def);
    var radius = Math.max(8, L * (isSea ? 1.6 : 1.1));
    // 浅色视口里的参照盘用 UI 语义色淡染：海上=蓝、地面=灰，透明度压到不抢模型。
    var fill = isSea ? 'rgba(31,95,191,0.10)' : 'rgba(107,118,136,0.10)';
    var line = isSea ? 'rgba(31,95,191,0.45)' : 'rgba(107,118,136,0.40)';
    groundEnt = viewer.entities.add({
      id: 'tpr-ground',
      position: new C.CallbackProperty(function () {
        var rec2 = recOf();
        if (!rec2) return undefined;
        var pose = poseNow(rec2);
        if (!pose.pos) return undefined;
        var g = C.Cartographic.fromCartesian(pose.pos);
        if (!g) return undefined;
        // 海面取零高程；地面取目标当前高度（高原案例里地面不是 0 m）。
        var alt = isSea ? -0.2 : Math.max(0, g.height - 0.2);
        return C.Cartesian3.fromRadians(g.longitude, g.latitude, alt);
      }, false),
      ellipse: {
        semiMajorAxis: new C.CallbackProperty(function () { return radius; }, false),
        semiMinorAxis: new C.CallbackProperty(function () { return radius; }, false),
        height: new C.CallbackProperty(function () {
          var rec2 = recOf();
          if (!rec2) return 0;
          var pose = poseNow(rec2);
          if (!pose.pos) return 0;
          var g = C.Cartographic.fromCartesian(pose.pos);
          return isSea ? -0.2 : (g ? Math.max(0, g.height - 0.2) : 0);
        }, false),
        material: C.Color.fromCssColorString(fill),
        outline: true,
        outlineColor: C.Color.fromCssColorString(line),
        outlineWidth: 1,
      },
    });
  }

  // ------------------------------------------------------------------
  // 相机：始终侧视（右舷），随目标转向
  // ------------------------------------------------------------------

  function updateCamera() {
    if (!viewer || !currentId) return;
    var rec = recOf();
    if (!rec || !rec.def) return;
    var pose = poseNow(rec);
    if (!pose.pos) return;
    var q = pose.q;
    if (!q) return;

    C.Matrix3.fromQuaternion(q, scratchM3);
    C.Matrix3.multiplyByVector(scratchM3, C.Cartesian3.UNIT_Y, scratchR);   // 模型右侧
    C.Matrix3.multiplyByVector(scratchM3, C.Cartesian3.UNIT_Z, scratchU);   // 模型上方

    var L = modelLength(rec.def);
    var dist = Math.max(9, L * 1.55) * zoom;
    // 相机 = 目标 + 右舷*dist + 上方*dist*0.16；
    // 视点 = 目标上方 dist*0.11 —— 模型落在视口中下部（约 70% 高度），
    // 下方留给水面/地面参照盘，而不是死居中（用户 2026-10-21 要求）。
    C.Cartesian3.multiplyByScalar(scratchR, dist, scratchP);
    C.Cartesian3.multiplyByScalar(scratchU, dist * 0.16, scratchD);
    C.Cartesian3.add(pose.pos, scratchP, scratchP);
    C.Cartesian3.add(scratchP, scratchD, scratchP);

    var look = C.Cartesian3.multiplyByScalar(scratchU, dist * 0.11, new C.Cartesian3());
    C.Cartesian3.add(pose.pos, look, look);
    C.Cartesian3.subtract(look, scratchP, scratchD);
    C.Cartesian3.normalize(scratchD, scratchD);
    var up = C.Cartesian3.clone(scratchU, new C.Cartesian3());
    viewer.camera.setView({
      destination: scratchP,
      orientation: { direction: scratchD, up: up },
    });

    // 头灯方向跟随相机，避免目标朝向不同导致一面全黑。
    if (viewer.scene.light && viewer.scene.light.direction) {
      C.Cartesian3.clone(viewer.camera.direction, viewer.scene.light.direction);
    }
  }

  // ------------------------------------------------------------------
  // 面板 DOM
  // ------------------------------------------------------------------

  function row(label) {
    var r = document.createElement('div');
    r.className = 'tpr-row';
    var k = document.createElement('span');
    k.textContent = label;
    var v = document.createElement('b');
    v.textContent = '—';
    r.appendChild(k);
    r.appendChild(v);
    return { row: r, val: v };
  }

  function section(title) {
    var h = document.createElement('div');
    h.className = 'tpr-sec';
    h.textContent = title;
    return h;
  }

  function addRow(host, label, key) {
    var r = row(label);
    if (key) cells[key] = r.val;
    host.appendChild(r.row);
    return r;
  }

  function buildFields(def) {
    var host = $('tprFields');
    if (!host) return;
    host.textContent = '';
    cells = {};

    host.appendChild(section('基本'));
    addRow(host, '类别', 'cls');
    addRow(host, '型号', 'model');
    if (def.model_note) addRow(host, '型号备注', 'note');
    addRow(host, '阵营', 'side');
    addRow(host, '域', 'domain');
    addRow(host, '角色', 'role');

    host.appendChild(section('状态与运动'));
    addRow(host, '战损', 'damage');
    addRow(host, '生命值', 'hp');
    addRow(host, '警戒', 'alert');
    addRow(host, '运动状态', 'motion');
    addRow(host, '航速', 'speed');
    addRow(host, '航向', 'heading');
    addRow(host, '高度', 'alt');
    addRow(host, '位置', 'pos');
    addRow(host, '已航里程', 'travel');

    host.appendChild(section('装备'));
    addRow(host, '主传感器', 'sensor');
    addRow(host, '探测范围', 'sensorKm');
    addRow(host, '影响范围', 'influenceKm');
    addRow(host, '交战距离', 'engageKm');

    var attrs = def.attributes || {};
    var keys = Object.keys(attrs);
    if (keys.length) {
      host.appendChild(section('自定义属性'));
      keys.forEach(function (k) {
        var r = row(k);
        var v = attrs[k];
        r.val.textContent = (v === undefined || v === null || v === '') ? '—' : String(v);
        host.appendChild(r.row);
      });
    }

    // 静态字段
    setCell('cls', def.platform_class || def.icon_label || '—');
    setCell('model', def.model || '—');
    if (cells.note) setCell('note', def.model_note || '—');
    setCell('side', sideLabel(def.affiliation));
    setCell('domain', domainLabel(def.domain));
    setCell('role', def.role || '—');
    setCell('sensor', def.sensor_name || '—');
    setCell('sensorKm', fmtKm(def.sensor_range_km));
    setCell('influenceKm', fmtKm(def.influence_km));
    setCell('engageKm', fmtKm(def.engage_km));
  }

  function setCell(key, text) {
    var node = cells[key];
    if (!node || node.textContent === text) return false;
    node.textContent = text;
    return true;
  }

  function buildBadges(def) {
    var host = $('tprBadges');
    if (!host) return;
    host.textContent = '';
    var side = document.createElement('span');
    side.className = 'tpr-tag ' + (def.affiliation === 'blue' ? 'tpr-blue' : 'tpr-red');
    side.textContent = sideLabel(def.affiliation);
    host.appendChild(side);
    var dom2 = document.createElement('span');
    dom2.className = 'tpr-tag';
    dom2.textContent = domainLabel(def.domain);
    host.appendChild(dom2);
    var model = document.createElement('span');
    model.className = 'tpr-tag tpr-model';
    model.textContent = def.model || def.platform_class || '';
    if (!model.textContent) return;
    host.appendChild(model);
  }

  function updateState() {
    var host = $('tprState');
    if (!host) return;
    var st = statusOf();
    var tier = root.TargetStatus ? TargetStatus.tier(st ? st.status : 0)
      : { label: '—', color: '#4B5563' };
    var hp = st ? st.hp : 100;
    var pct = st ? clamp(st.status === 3 ? 0 : hp, 0, 100) : 100;

    // 状态行每帧都会被调用；内容没变时不重建 DOM（暂停时避免持续重排）。
    var key = (st ? st.status : -1) + '|' + Math.round(hp * 10) + '|' + tier.label;
    if (host.dataset.key === key) return false;
    host.dataset.key = key;

    host.textContent = '';
    var pill = document.createElement('span');
    pill.className = 'tpr-pill';
    pill.textContent = st ? st.label : '—';
    pill.style.background = tier.color;
    host.appendChild(pill);

    var barWrap = document.createElement('span');
    barWrap.className = 'tpr-hp';
    var bar = document.createElement('i');
    bar.style.width = pct.toFixed(1) + '%';
    bar.style.background = tier.color;
    barWrap.appendChild(bar);
    host.appendChild(barWrap);

    var hpText = document.createElement('span');
    hpText.className = 'tpr-hp-text';
    hpText.textContent = st ? ('HP ' + fmtNum(st.hp, 1) + '%') : 'HP —';
    host.appendChild(hpText);
    return true;
  }

  function showNoModel(msg) {
    var host = $('tprView');
    if (!host) return;
    setViewVisible(true);
    host.classList.add('tpr-view-empty');
    var node = host.querySelector('.tpr-view-note');
    if (!node) {
      node = document.createElement('div');
      node.className = 'tpr-view-note';
      host.appendChild(node);
    }
    node.textContent = msg || '';
  }

  function hideNoModel() {
    var host = $('tprView');
    if (!host) return;
    host.classList.remove('tpr-view-empty');
    var node = host.querySelector('.tpr-view-note');
    if (node) node.remove();
  }

  /** 无模型目标隐藏整个 3D 视口（含黑底与提示），面板只留属性字段。 */
  function setViewVisible(v) {
    var host = $('tprView');
    if (host) host.classList.toggle('hidden', !v);
  }

  // ------------------------------------------------------------------
  // 显隐与刷新
  // ------------------------------------------------------------------

  function showPanel(show) {
    var host = $('targetProps');
    if (host) host.classList.toggle('hidden', !show);
    if (!show) return;
    var rec = currentId ? recOf() : null;
    var has = !!(rec && rec.def);
    var body = $('tprBody');
    if (body) body.classList.toggle('hidden', !has);
    var empty = $('tprEmpty');
    if (empty) empty.classList.toggle('hidden', has);
    if (!has) {
      // 取消选中 / 记录缺失：标题与状态必须复位，不能挂着上一个目标的读数。
      var name = $('tprName');
      var idn = $('tprId');
      var st = $('tprState');
      if (name) name.textContent = '目标属性';
      if (idn) idn.textContent = '';
      if (st) st.dataset.key = '';
      cells = {};
    }
  }

  function syncVisibility() {
    if (!enabled) {
      showPanel(false);
      destroyPreview();
      return;
    }
    showPanel(true);
    if (!currentId) {
      destroyPreview();
      return;
    }
    var rec = recOf();
    if (!rec || !rec.def) { destroyPreview(); return; }
    showPanel(true);
    if (attemptedFor !== currentId) buildPreview();
    updateCamera();
    updateStatic();
    updateHud(true);
    requestRender();
  }

  function updateStatic() {
    var rec = recOf();
    if (!rec || !rec.def) return;
    var def = rec.def;
    var name = $('tprName');
    var idn = $('tprId');
    if (name) name.textContent = def.name || def.id || '目标属性';
    if (idn) idn.textContent = def.id || '';
    buildBadges(def);
    buildFields(def);
    updateState();
  }

  /** 动态字段随帧更新；节流 100 ms，暂停/拖动时间轴也能立即刷新一次。 */
  function updateHud(force) {
    if (!currentId) return;
    var now = (root.performance && performance.now) ? performance.now() : Date.now();
    if (!force && now - lastHudMs < 100) return;
    lastHudMs = now;

    var rec = recOf();
    if (!rec || !rec.def) return;
    var def = rec.def;
    var s = frameAt(api.tick());
    var st = statusOf();
    var changed = false;

    changed = setCell('damage', st ? st.label : '—') || changed;
    changed = setCell('hp', st ? fmtNum(st.hp, 1) + ' %' : '—') || changed;
    var alertText = '—';
    if (st && (st.alertLabel || st.alert !== null && st.alert !== undefined)) {
      alertText = st.alertLabel || ('等级 ' + st.alert);
    }
    changed = setCell('alert', alertText) || changed;
    changed = setCell('motion', motionLabel(def, s)) || changed;
    if (s) {
      var unit = def.speed_unit || 'kn';
      var sp = unit === 'km/h' ? s.speed_kn * 1.852 : s.speed_kn;
      changed = setCell('speed', fmtNum(sp, 1) + ' ' + unit) || changed;
      changed = setCell('heading', fmtNum(s.heading, 1) + '°') || changed;
      changed = setCell('alt', fmtAlt(s.alt)) || changed;
      changed = setCell('pos', Number(s.lon).toFixed(6) + ', ' + Number(s.lat).toFixed(6)) || changed;
      changed = setCell('travel', (Number(s.travelled_m || 0) / 1000).toFixed(2) + ' km') || changed;
    } else {
      ['speed', 'heading', 'alt', 'pos', 'travel'].forEach(function (k) {
        changed = setCell(k, '—') || changed;
      });
    }
    changed = updateState() || changed;
    // 暂停时属性没变就不重绘预览视图；播放中由 frame() 每帧请求渲染。
    if (changed) requestRender();
  }

  // ------------------------------------------------------------------
  // 对外接口
  // ------------------------------------------------------------------

  function init(mv, a) {
    mainViewer = mv;
    api = a;
    bindWheel();
  }

  function setEnabled(on) {
    var next = !!on;
    if (next === enabled) return;
    enabled = next;
    syncVisibility();
  }

  function setTarget(id) {
    var next = id || null;
    if (next === currentId) {
      // 右栏/要素列表的「定位」会对同一目标重复调 selectTarget：只刷新读数，
      // 不重建字段 DOM（否则用户滚到装备区的滚动位置会被打回顶部）。
      updateHud(true);
      requestRender();
      return;
    }
    currentId = next;
    lastHudMs = 0;
    cells = {};
    // 换目标必须重建预览实体（模型 URI 在创建时定死），但复用同一个 Viewer。
    attemptedFor = null;
    removePreviewEntities();
    if (!enabled) return;
    syncVisibility();
  }

  function reset() {
    currentId = null;
    cells = {};
    lastHudMs = 0;
    destroyPreview();
    if (enabled) syncVisibility(); else showPanel(false);
  }

  function frame() {
    if (!enabled || !currentId) return;
    var rec = recOf();
    if (!rec) return;
    var playing = !!(api && api.playing && api.playing());
    var sig = signature(rec, poseNow(rec));
    if (playing || sig !== lastSig) {
      lastSig = sig;
      updateCamera();
      // 暂停时位姿/着色变化（击毁下沉、命中闪光、CZML 晚到）也要重绘一帧。
      if (!playing) requestRender();
    }
    updateHud(false);
    // 播放中预览最多 ~22 fps：侧视小窗不需要跟主场景同帧率，
    // 主场景继续 60 fps，第二 WebGL 上下文的开销被削掉一半以上。
    if (playing) {
      var now = (root.performance && performance.now) ? performance.now() : Date.now();
      if (now - lastPreviewMs >= 45) {
        lastPreviewMs = now;
        requestRender();
      }
    }
  }

  function debug() {
    var out = {
      enabled: enabled, currentId: currentId, viewerCreated: !!viewer,
      zoom: zoom, renders: renderCount,
    };
    if (!viewer || !currentId) return out;
    var rec = recOf();
    var t = timeNow();
    var mainModel = rec && rec.entity ? rec.entity.model : null;
    var pvModel = previewEnt ? previewEnt.model : null;
    var mainPos = rec ? readProp(rec.entity.position, t, null) : null;
    var pvPos = previewEnt ? readProp(previewEnt.position, t, null) : null;
    var mainQ = rec ? readProp(rec.entity.orientation, t, null) : null;
    var pvQ = previewEnt ? readProp(previewEnt.orientation, t, null) : null;

    function css(c) { return c && c.toCssHexString ? c.toCssHexString() : null; }

    out.model = {
      uri: readProp(pvModel && pvModel.uri, t, null),
      mainColor: css(readProp(mainModel && mainModel.color, t, null)),
      previewColor: css(readProp(pvModel && pvModel.color, t, null)),
      mainAmount: readProp(mainModel && mainModel.colorBlendAmount, t, null),
      previewAmount: readProp(pvModel && pvModel.colorBlendAmount, t, null),
      mainSilSize: readProp(mainModel && mainModel.silhouetteSize, t, null),
      previewSilSize: readProp(pvModel && pvModel.silhouetteSize, t, null),
      nodes: pvModel && pvModel.nodeTransformations
        ? Object.keys(pvModel.nodeTransformations) : [],
      proxy: !!proxyEnt, ground: !!groundEnt,
    };
    var viewEl = $('tprView');
    out.viewVisible = !!(viewEl && !viewEl.classList.contains('hidden'));
    if (viewer && previewEnt && mainPos) {
      var w = C.SceneTransforms.worldToWindowCoordinates(viewer.scene, mainPos);
      var cv = viewer.scene.canvas;
      out.modelScreen = w ? {
        x: +(w.x / Math.max(1, cv.clientWidth)).toFixed(3),
        y: +(w.y / Math.max(1, cv.clientHeight)).toFixed(3),
      } : null;
    }
    out.posDelta = (mainPos && pvPos) ? C.Cartesian3.distance(mainPos, pvPos) : null;
    out.orientDot = (mainQ && pvQ)
      ? Math.abs(mainQ.x * pvQ.x + mainQ.y * pvQ.y + mainQ.z * pvQ.z + mainQ.w * pvQ.w) : null;

    if (mainQ) {
      C.Matrix3.fromQuaternion(mainQ, scratchM3);
      var right = C.Matrix3.multiplyByVector(scratchM3, C.Cartesian3.UNIT_Y, new C.Cartesian3());
      var fwdLocal = C.Matrix3.multiplyByVector(scratchM3, C.Cartesian3.UNIT_X, new C.Cartesian3());
      var dir = viewer.camera.direction;
      out.camera = {
        dist: mainPos ? C.Cartesian3.distance(viewer.camera.positionWC, mainPos) : null,
        sideDot: C.Cartesian3.dot(dir, right),        // 侧视应为 ±1
        fwdDot: C.Cartesian3.dot(dir, fwdLocal),      // 侧视应≈0
      };
    }
    return out;
  }

  root.TargetProps = {
    init: init,
    setEnabled: setEnabled,
    setTarget: setTarget,
    reset: reset,
    frame: frame,
    debug: debug,
  };
})(window);
