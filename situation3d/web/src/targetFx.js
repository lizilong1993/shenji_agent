/**
 * targetFx.js — 目标动态特效（车轮滚动 / 船舶尾流 / 飞机尾气）
 *
 * 分工：CZML（targetMotion.js）只负责**位置与姿态**；本文件负责"随运动状态
 * 变化的渲染效果"，三类目标各用自己的合适机制：
 *
 *   车轮 —— Entity.model.nodeTransformations：直接驱动 GLB 里的 wheel 节点
 *           绕局部 Y 轴旋转。旋转角速度与当前速度成正比（时间轴为 900 倍
 *           压缩，按真实滚动会变成频闪，因此以"视觉同步"为基准并设上限）。
 *           履带式坦克（armor.glb）没有轮节点，改由 glTF 自带的
 *           Tank_Forward 动画表达，避免轮子对不上履带。
 *
 *   尾流 —— 随船拖尾折线：在 Cesium 时钟下回溯这条船过去若干时刻的 CZML
 *           采样位置，形成一条略高于水面、由宽到窄的水花带。比给 31 个
 *           目标各挂一套粒子系统便宜，且尾流方向天然跟随航迹转弯。
 *
 *   尾气 —— 机尾拖尾折线：同尾流一样回溯 CZML 采样位置，飞机按轨迹飞行时
 *           自然拉出一条灰白尾迹；长度随速度变化，暂停/拖动时同源冻结。
 *           （不采用 ParticleSystem 的原因见下方"飞机尾气"一节。）
 *
 * 所有效果都从**同一条时间源**（viewer.clock，由 targetMotion 同步到
 * tickFloat）取状态，因此暂停、拖动时间轴时不会各行其是。
 */
(function (root, factory) {
  var api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.TargetFx = api;
}(typeof window !== 'undefined' ? window : null, function () {
  'use strict';

  var WAKE_SEGMENTS = 14;
  var WHEEL_GAIN = 3.0;           // 视觉同步系数（见文件头），上限见 WHEEL_MAX
  var WHEEL_MAX_RAD_S = 30;
  var SPEED_TO_MPS = 0.514444;    // 节 → 米/秒

  var ctx = null;                 // { viewer, data, api }
  var wheelStates = {};           // 目标 id -> { angle, radius }
  var wakeSpecs = [];             // [{ id, def, outer, inner }]
  var exhaustSpecs = [];          // [{ id, def, ps }]
  var modelCache = {};            // 目标 id -> Cesium.Model（仅带 glTF 动画的模型需要）
  var lastModelScan = 0;
  var lastAnimSync = 0;

  var scratchUp = new Cesium.Cartesian3();
  var scratchBack = new Cesium.Cartesian3();
  var scratchTail = new Cesium.Cartesian3();
  var scratchFwd = new Cesium.Cartesian3();

  function formClassOf(def) {
    return window.EntityLod ? window.EntityLod.formClassOf(def) : null;
  }

  function hasModel(def) {
    return !!(window.EntityLod && window.EntityLod.modelUrlFor(def));
  }

  function isSea(def) {
    var d = String(def.domain || '').toLowerCase();
    return d === 'sea' || d === 'naval' || d === 'maritime';
  }

  function isAir(def) {
    return String(def.domain || '').toLowerCase() === 'air';
  }

  /** 模型局部前向轴（与 targetMotion 的姿态约定一致）。 */
  function localForward(formClass, result) {
    var axis = window.EntityLod && window.EntityLod.forwardAxis
      ? window.EntityLod.forwardAxis(formClass) : '+Y';
    switch (axis) {
      case '-Y': return Cesium.Cartesian3.fromElements(0, -1, 0, result);
      case '+X': return Cesium.Cartesian3.fromElements(1, 0, 0, result);
      case '-X': return Cesium.Cartesian3.fromElements(-1, 0, 0, result);
      case '+Y':
      default: return Cesium.Cartesian3.fromElements(0, 1, 0, result);
    }
  }

  // ------------------------------------------------------------------
  // 车轮：entity.model.nodeTransformations
  // ------------------------------------------------------------------

  /**
   * 由 app.js 在创建实体时调用；返回 wheel 节点的 NodeTransformationProperty 表。
   * 没有轮式资产的形态（船/飞机/坦克）返回 undefined，不改变原模型。
   */
  function modelNodeTransformations(def) {
    var formClass = formClassOf(def);
    var wheels = (window.EntityLod && window.EntityLod.wheelNodes)
      ? window.EntityLod.wheelNodes(formClass) : [];
    var out = {};
    if (wheels.length && !wheelStates[def.id]) {
      wheelStates[def.id] = { angle: 0, radius: wheels[0].radius || 0.6 };
    }
    wheels.forEach(function (w) {
      var q = new Cesium.Quaternion();
      var axis = w.axis === 'X' ? Cesium.Cartesian3.UNIT_X
        : (w.axis === 'Z' ? Cesium.Cartesian3.UNIT_Z : Cesium.Cartesian3.UNIT_Y);
      out[w.name] = new Cesium.NodeTransformationProperty({
        rotation: new Cesium.CallbackProperty(function () {
          var st = wheelStates[def.id];
          return Cesium.Quaternion.fromAxisAngle(axis, st ? st.angle : 0, q);
        }, false),
      });
    });
    if (window.TargetCombatFx && TargetCombatFx.nodeTransformations) {
      var extra = TargetCombatFx.nodeTransformations(def);
      if (extra) Object.keys(extra).forEach(function (k) { out[k] = extra[k]; });
    }
    return Object.keys(out).length ? out : undefined;
  }

  function updateWheels(dt) {
    var playing = !!(ctx.api.playing && ctx.api.playing());
    Object.keys(wheelStates).forEach(function (id) {
      var st = wheelStates[id];
      var sample = ctx.api.sample ? ctx.api.sample(id) : null;
      var speedKn = sample ? Math.max(0, Number(sample.speed_kn) || 0) : 0;
      if (!playing || speedKn < 0.2) return;        // 暂停或静止：车轮保持
      var mps = speedKn * SPEED_TO_MPS;
      var omega = Math.min(WHEEL_MAX_RAD_S, (mps / (st.radius || 0.6)) * WHEEL_GAIN);
      st.angle = (st.angle + omega * dt) % (Math.PI * 2);
    });
  }

  // ------------------------------------------------------------------
  // 模型查找（履带坦克的 glTF 动画 + 暂停时冻结所有模型动画）
  // ------------------------------------------------------------------

  /**
   * ModelVisualizer 把 Model 图元放进 scene.primitives，并以实体作为 id。
   * 这里按"实体 id"取回 Model，用于两类官方 API：activeAnimations 与 getNode。
   * 找不到时每 1 s 重扫一次（模型是异步加载的）。
   */
  function scanModels(now) {
    if (now - lastModelScan < 1000) return;
    lastModelScan = now;
    var prims = ctx.viewer.scene.primitives;
    for (var i = 0; i < prims.length; i++) {
      var pr = prims.get(i);
      if (!pr || !pr.activeAnimations) continue;
      var ent = pr.id;
      var eid = ent && ent.id ? String(ent.id) : null;
      if (eid && eid.indexOf('target-') === 0) modelCache[eid.slice(7)] = pr;
    }
  }

  /**
   * 履带模型有 4 段动画（前/后/左转/右转），Cesium 会同时播放、互相打架。
   * ModelAnimation.multiplier 在 1.145 是只读 getter，但内部 `_multiplier` 仍是
   * 每帧参与积分的字段；本文件直接写它来"只放前进、其余冻结"。
   * 不用 removeAll/addAll：ModelVisualizer 会在自身 update 里按 runAnimations
   * 重新 addAll，和我们的增删互相覆盖（实测 clips 数在 4 与 0 之间来回跳）。
   */
  function tuneArmor(model) {
    if (model.__fxTuned) return;
    var anims = model.activeAnimations;
    for (var i = 0; i < anims.length; i++) {
      var a = anims.get(i);
      if (!a) continue;
      var forward = /Tank_Forward/i.test(a.name || '');
      a._multiplier = forward ? 1.4 : 0;
      a._loop = Cesium.ModelAnimationLoop.REPEAT;
    }
    model.__fxTuned = true;
  }

  function setArmorRunning(model, on) {
    var anims = model.activeAnimations;
    for (var i = 0; i < anims.length; i++) {
      var a = anims.get(i);
      if (!a) continue;
      a._multiplier = (on && /Tank_Forward/i.test(a.name || '')) ? 1.4 : 0;
    }
    model.__fxArmorOn = on;
  }

  function updateModelAnimations(now) {
    if (now - lastAnimSync < 250) return;
    lastAnimSync = now;
    var playing = !!(ctx.api.playing && ctx.api.playing());
    (ctx.data.targets || []).forEach(function (def) {
      var model = modelCache[def.id];
      if (!model) return;
      var formClass = formClassOf(def);
      if (formClass === 'armor') {
        tuneArmor(model);
        var sample = ctx.api.sample ? ctx.api.sample(def.id) : null;
        var speed = sample ? Number(sample.speed_kn) || 0 : 0;
        setArmorRunning(model, playing && speed > 0.2);
        return;
      }
      // 旋翼 / 雷达：暂停时把内部倍率置 0（不增删动画，避免相位跳变），
      // 播放时恢复 1。只对确实带动画的模型逐一写，空集合直接跳过。
      for (var i = 0; i < model.activeAnimations.length; i++) {
        var an = model.activeAnimations.get(i);
        if (an) an._multiplier = playing ? 1 : 0;
      }
      model.__fxPlaying = playing;
    });
  }

  // ------------------------------------------------------------------
  // 船舶尾流
  // ------------------------------------------------------------------

  function makeWake(def) {
    var formClass = formClassOf(def);
    var lenM = (window.EntityLod && window.EntityLod.REAL_LENGTH_M[formClass]) || 60;
    var id = def.id;
    var cache = { time: null, pts: [] };

    function positionsFor() {
      var t = ctx.api.clockTime ? ctx.api.clockTime() : null;
      if (!t) return [];
      if (cache.time && Cesium.JulianDate.equals(cache.time, t)) return cache.pts;
      var rec = ctx.api.entity ? ctx.api.entity(id) : null;
      var prop = rec && rec.position;
      if (!prop) { cache.pts = []; return cache.pts; }
      var sample = ctx.api.sample ? ctx.api.sample(id) : null;
      var speedKn = sample ? Math.max(0, Number(sample.speed_kn) || 0) : 0;
      if (speedKn < 0.5) { cache.pts = []; cache.time = Cesium.JulianDate.clone(t, new Cesium.JulianDate()); return cache.pts; }

      var mps = speedKn * SPEED_TO_MPS;
      var span = Math.max(12, Math.min(900, (lenM * 6) / Math.max(0.5, mps)));
      var step = span / (WAKE_SEGMENTS - 1);
      var hdg = Cesium.Math.toRadians(sample.heading || 0);

      // 船尾方向：航向的反向（在经纬面上用 ENU 东/北合成，避免极点问题）
      var cart = new Cesium.Cartographic();
      Cesium.Cartographic.fromCartesian(
        Cesium.Cartesian3.fromDegrees(sample.lon, sample.lat, 0), Cesium.Ellipsoid.WGS84, cart);
      var enu = Cesium.Transforms.eastNorthUpToFixedFrame(
        Cesium.Cartesian3.fromRadians(cart.longitude, cart.latitude, 0));
      Cesium.Cartesian3.fromElements(enu[0], enu[1], enu[2], scratchFwd);      // 东
      Cesium.Cartesian3.multiplyByScalar(scratchFwd, Math.sin(hdg), scratchFwd);
      Cesium.Cartesian3.fromElements(enu[4], enu[5], enu[6], scratchBack);    // 北
      Cesium.Cartesian3.multiplyByScalar(scratchBack, Math.cos(hdg), scratchBack);
      Cesium.Cartesian3.add(scratchFwd, scratchBack, scratchFwd);              // 前向
      Cesium.Cartesian3.multiplyByScalar(scratchFwd, -lenM * 0.45, scratchBack);

      var pts = [];
      var tmp = new Cesium.JulianDate();
      for (var k = 0; k < WAKE_SEGMENTS; k++) {
        Cesium.JulianDate.addSeconds(t, -k * step, tmp);
        var p = prop.getValue(tmp);
        if (!p) break;
        var q = Cesium.Cartesian3.add(p, scratchBack, new Cesium.Cartesian3());
        var up = Cesium.Ellipsoid.WGS84.geodeticSurfaceNormal(q, scratchUp);
        Cesium.Cartesian3.add(q, Cesium.Cartesian3.multiplyByScalar(up, 1.5, up), q);
        pts.push(q);
      }
      cache.pts = pts;
      cache.time = Cesium.JulianDate.clone(t, new Cesium.JulianDate());
      return pts;
    }

    var outer = ctx.viewer.entities.add({
      id: 'wake-' + id + '-outer',
      show: false,
      polyline: {
        positions: new Cesium.CallbackProperty(positionsFor, false),
        width: 20,
        material: new Cesium.PolylineGlowMaterialProperty({
          color: Cesium.Color.fromCssColorString('#d7ecff').withAlpha(0.12),
          glowPower: 0.18,
        }),
        clampToGround: false,
      },
    });
    var inner = ctx.viewer.entities.add({
      id: 'wake-' + id + '-inner',
      show: false,
      polyline: {
        positions: new Cesium.CallbackProperty(positionsFor, false),
        width: 5,
        material: new Cesium.PolylineGlowMaterialProperty({
          color: Cesium.Color.fromCssColorString('#ffffff').withAlpha(0.28),
          glowPower: 0.35,
        }),
        clampToGround: false,
      },
    });
    wakeSpecs.push({ id: id, def: def, outer: outer, inner: inner });
  }

  function updateWakes() {
    var camH = ctx.api.cameraHeight ? ctx.api.cameraHeight() : 0;
    wakeSpecs.forEach(function (w) {
      var sample = ctx.api.sample ? ctx.api.sample(w.id) : null;
      var speed = sample ? Math.max(0, Number(sample.speed_kn) || 0) : 0;
      var on = !!(ctx.api.isVisible && ctx.api.isVisible(w.def))
        && speed > 0.8
        && !(camH > 80000);
      // 只画相机附近的船：几公里到几十公里外的尾流会在地平线上连成一条亮线
      if (on && sample && ctx.viewer) {
        var cam = ctx.viewer.camera.positionWC;
        var p = Cesium.Cartesian3.fromDegrees(sample.lon, sample.lat, sample.alt || 0,
          Cesium.Ellipsoid.WGS84, scratchTail);
        on = Cesium.Cartesian3.distance(cam, p) < 60000;
      }
      w.outer.show = on;
      w.inner.show = on;
    });
  }

  // ------------------------------------------------------------------
  // 飞机尾气 / 尾迹
  // ------------------------------------------------------------------
  //
  // 为什么不是 Cesium.ParticleSystem：本系统时间轴是 900 倍压缩（30 s/帧、
  // 30 帧/秒）。ParticleSystem 的发射按场景时钟的帧间隔累计，而粒子寿命按
  // 真实秒计时，两者在这种倍率下不再一致——实测 8 架飞机各 3~6 万粒子、
  // 帧率 3.0 → 0.9 fps、堆内存 1.2 GB。改用与船舶尾流同构的**拖尾折线**：
  // 直接回溯 CZML 采样位置，长度只由速度与时间跨度决定，开销恒定，
  // 并且与暂停/拖动时间轴天然同步。视觉上是一条贴着机尾的灰白尾迹。
  var EXHAUST_SEGMENTS = 14;

  function makeExhaust(def) {
    var formClass = formClassOf(def);
    var lenM = (window.EntityLod && window.EntityLod.REAL_LENGTH_M[formClass]) || 15;
    var id = def.id;
    var cache = { time: null, pts: [] };

    function positionsFor() {
      var t = ctx.api.clockTime ? ctx.api.clockTime() : null;
      if (!t) return [];
      if (cache.time && Cesium.JulianDate.equals(cache.time, t)) return cache.pts;
      var rec = ctx.api.entity ? ctx.api.entity(id) : null;
      var prop = rec && rec.position;
      if (!prop) { cache.pts = []; return cache.pts; }
      var sample = ctx.api.sample ? ctx.api.sample(id) : null;
      var speedKn = sample ? Math.max(0, Number(sample.speed_kn) || 0) : 0;
      if (speedKn < 20) { cache.pts = []; cache.time = Cesium.JulianDate.clone(t, new Cesium.JulianDate()); return cache.pts; }

      var mps = speedKn * SPEED_TO_MPS;
      var span = Math.max(6, Math.min(60, 8000 / Math.max(1, mps)));
      var step = span / (EXHAUST_SEGMENTS - 1);
      var hdg = Cesium.Math.toRadians(sample.heading || 0);

      var base = Cesium.Cartesian3.fromDegrees(sample.lon, sample.lat, sample.alt || 0);
      var enu = Cesium.Transforms.eastNorthUpToFixedFrame(base);
      Cesium.Cartesian3.fromElements(enu[0], enu[1], enu[2], scratchFwd);
      Cesium.Cartesian3.multiplyByScalar(scratchFwd, Math.sin(hdg), scratchFwd);
      Cesium.Cartesian3.fromElements(enu[4], enu[5], enu[6], scratchBack);
      Cesium.Cartesian3.multiplyByScalar(scratchBack, Math.cos(hdg), scratchBack);
      Cesium.Cartesian3.add(scratchFwd, scratchBack, scratchFwd);            // 航向
      Cesium.Cartesian3.multiplyByScalar(scratchFwd, -lenM * 0.45, scratchBack); // 机尾偏移

      var pts = [];
      var tmp = new Cesium.JulianDate();
      for (var k = 0; k < EXHAUST_SEGMENTS; k++) {
        Cesium.JulianDate.addSeconds(t, -k * step, tmp);
        var p = prop.getValue(tmp);
        if (!p) break;
        var q = Cesium.Cartesian3.add(p, scratchBack, new Cesium.Cartesian3());
        pts.push(q);
      }
      cache.pts = pts;
      cache.time = Cesium.JulianDate.clone(t, new Cesium.JulianDate());
      return pts;
    }

    var core = ctx.viewer.entities.add({
      id: 'exhaust-' + id + '-core',
      show: false,
      polyline: {
        positions: new Cesium.CallbackProperty(positionsFor, false),
        width: 3,
        material: new Cesium.PolylineGlowMaterialProperty({
          color: Cesium.Color.fromCssColorString('#eef5ff').withAlpha(0.32),
          glowPower: 0.35,
        }),
        clampToGround: false,
      },
    });
    var plume = ctx.viewer.entities.add({
      id: 'exhaust-' + id + '-plume',
      show: false,
      polyline: {
        positions: new Cesium.CallbackProperty(positionsFor, false),
        width: 10,
        material: new Cesium.PolylineGlowMaterialProperty({
          color: Cesium.Color.fromCssColorString('#cdd9e6').withAlpha(0.10),
          glowPower: 0.2,
        }),
        clampToGround: false,
      },
    });
    exhaustSpecs.push({ id: id, def: def, core: core, plume: plume });
  }

  function updateExhaust() {
    var camH = ctx.api.cameraHeight ? ctx.api.cameraHeight() : 0;
    exhaustSpecs.forEach(function (ex) {
      var sample = ctx.api.sample ? ctx.api.sample(ex.id) : null;
      var speed = sample ? Math.max(0, Number(sample.speed_kn) || 0) : 0;
      var on = !!(ctx.api.isVisible && ctx.api.isVisible(ex.def))
        && speed > 20 && !(camH > 60000);
      if (on && sample && ctx.viewer) {
        var cam = ctx.viewer.camera.positionWC;
        var p = Cesium.Cartesian3.fromDegrees(sample.lon, sample.lat, sample.alt || 0,
          Cesium.Ellipsoid.WGS84, scratchTail);
        on = Cesium.Cartesian3.distance(cam, p) < 80000;
      }
      ex.core.show = on;
      ex.plume.show = on;
    });
  }

  // ------------------------------------------------------------------
  // 生命周期
  // ------------------------------------------------------------------

  /**
   * @param viewer
   * @param data
   * @param api  {
   *   isVisible(def), sample(id), entity(id),
   *   clockTime(), playing(), cameraHeight()
   * }
   */
  function init(viewer, data, api) {
    dispose();
    ctx = { viewer: viewer, data: data, api: api || {} };
    wheelStates = {};
    wakeSpecs = [];
    exhaustSpecs = [];
    modelCache = {};
    lastModelScan = 0;
    lastAnimSync = 0;

    (data.targets || []).forEach(function (def) {
      // 轮节点变换在 createTargetEntities 时已挂到实体上，其回调按 id 读取
      // wheelStates；init 清空后必须在这里补回状态，否则车轮永远停在 0°。
      var wheels = (window.EntityLod && window.EntityLod.wheelNodes)
        ? window.EntityLod.wheelNodes(formClassOf(def)) : [];
      if (wheels.length) {
        wheelStates[def.id] = { angle: 0, radius: wheels[0].radius || 0.6 };
      }
      if (isSea(def) && hasModel(def)) makeWake(def);
      if (isAir(def) && hasModel(def)) makeExhaust(def);
    });
  }

  /** 每帧（rAF）调用；dt 为墙钟秒数。 */
  function frame(dt) {
    if (!ctx) return;
    var now = (typeof performance !== 'undefined' && performance.now)
      ? performance.now() : Date.now();
    scanModels(now);
    updateWheels(Math.min(0.1, Math.max(0, dt || 0)));
    updateModelAnimations(now);
    updateWakes();
    updateExhaust();
  }

  function dispose() {
    if (ctx && ctx.viewer) {
      wakeSpecs.forEach(function (w) {
        try {
          if (w.outer) ctx.viewer.entities.remove(w.outer);
          if (w.inner) ctx.viewer.entities.remove(w.inner);
        } catch (e) { /* 场景已拆 */ }
      });
      exhaustSpecs.forEach(function (ex) {
        try {
          if (ex.core) ctx.viewer.entities.remove(ex.core);
          if (ex.plume) ctx.viewer.entities.remove(ex.plume);
        } catch (e) { /* 已拆 */ }
      });
    }
    ctx = null;
    wheelStates = {};
    wakeSpecs = [];
    exhaustSpecs = [];
    modelCache = {};
  }

  function debug() {
    return {
      ready: !!ctx,
      wakes: wakeSpecs.map(function (w) { return { id: w.id, show: !!w.outer.show }; }),
      exhausts: exhaustSpecs.map(function (e) {
        return { id: e.id, show: !!(e.core && e.core.show) };
      }),
      wheels: Object.keys(wheelStates).map(function (id) {
        return { id: id, angle: +wheelStates[id].angle.toFixed(3) };
      }),
      models: Object.keys(modelCache),
      anims: Object.keys(modelCache).map(function (id) {
        var m = modelCache[id];
        var names = [];
        var mults = [];
        for (var i = 0; i < (m.activeAnimations ? m.activeAnimations.length : 0); i++) {
          var a = m.activeAnimations.get(i);
          if (a) { names.push(a.name); mults.push(+(a._multiplier || 0).toFixed(2)); }
        }
        return { id: id, clips: names, mults: mults, armor: !!m.__fxTuned, playing: m.__fxPlaying };
      }).filter(function (x) { return x.clips.length || x.armor; }),
    };
  }

  return {
    init: init,
    frame: frame,
    dispose: dispose,
    debug: debug,
    modelNodeTransformations: modelNodeTransformations,
  };
}));
