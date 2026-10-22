/**
 * targetMotion.js — 基于 CZML 的目标位置 / 姿态统一运动控制
 *
 * 职责（单一）：把案例的逐帧目标状态（lon/lat/alt + 航向）编译成一份 CZML
 * 文档，交给 Cesium 的 CzmlDataSource 采样，再把这些采样属性挂到既有的目标
 * 实体上。**场景里所有目标的「在哪、朝哪」都只从这一条链路来**，不再由
 * app.js 的 CallbackProperty 按整数帧直接写坐标。
 *
 * 为什么用 CZML 而不是继续每帧 setValue：
 *   · Cesium 的时间轴、属性系统与插值都在 CZML 采样属性里，位置在帧与帧
 *     之间（本系统 30 s 一帧、30 帧/秒播放）按时间线性插值，运动连续；
 *   · 姿态与位置共用同一套采样时间，暂停 / 拖动时间轴时二者天然同步；
 *   · 数据规模（31 目标 × 1441 帧）在加载时一次性编译成 Float64 数组，
 *     运行期不再逐帧重算。
 *
 * 姿态约定：
 *   · 航向取帧数据的 heading（罗盘角，0=北、顺时针）；缺失时用相邻位置
 *     的方位角兜底，因此"沿轨迹"优先，数据只是它的离散表达；
 *   · 空中目标按高度变化给俯仰、按航向变化率给滚转（转弯压坡度）；
 *     水面/地面目标只保留航向（俯仰/滚转 = 0）；
 *   · 2026-10-20 起资产统一在 tools/modelgen 里归一（Blender +Y 为艏/头），
 *     实测 Cesium 1.145 下对应模型局部 -X，由 EntityLod.forwardAxis()
 *     给出，这里按轴构造基向量，而不是给所有资产硬套一个 heading。
 *
 * 该文件不创建任何特效图元；车轮 / 尾流 / 尾气见 targetFx.js。
 */
(function (root, factory) {
  var api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.TargetMotion = api;
}(typeof window !== 'undefined' ? window : null, function () {
  'use strict';

  var TICK_SECONDS_FALLBACK = 30;
  var PITCH_LIMIT_RAD = Cesium.Math.toRadians(20);
  var ROLL_LIMIT_RAD = Cesium.Math.toRadians(35);
  var MIN_SPEED_MPS = 0.6;              // 低于此速度视为静止：不按轨迹推姿态

  // ---- 复用对象（编译期单线程，够用且避免 4 万次采样时疯狂 GC） ----
  // 每个变量只承担一个角色，**不要互相复用**：姿态计算里 F/U0/R 与中间量
  // 共用同一个 Cartesian3 时，pitch=0 那一步会把前向向量写成零（已踩过）。
  var scratchM4 = new Cesium.Matrix4();
  var scratchM3 = new Cesium.Matrix3();
  var scratchE = new Cesium.Cartesian3();
  var scratchN = new Cesium.Cartesian3();
  var scratchU = new Cesium.Cartesian3();
  var scratchFh = new Cesium.Cartesian3();
  var scratchF = new Cesium.Cartesian3();
  var scratchU0 = new Cesium.Cartesian3();
  var scratchR = new Cesium.Cartesian3();
  var scratchRp = new Cesium.Cartesian3();
  var scratchUp = new Cesium.Cartesian3();
  var scratchT = new Cesium.Cartesian3();
  var scratchX = new Cesium.Cartesian3();
  var scratchY = new Cesium.Cartesian3();
  var scratchZ = new Cesium.Cartesian3();
  var scratchClock = new Cesium.JulianDate();

  // 当前已装载的案例运动源。只保留"最新一份"：切案例时旧的立即销毁，
  // 避免异步 load 晚到时把上一案例的属性挂到新场景上。
  var current = null;
  var tokenSeq = 0;

  function forwardAxisOf(formClass) {
    if (window.EntityLod && typeof window.EntityLod.forwardAxis === 'function') {
      return window.EntityLod.forwardAxis(formClass);
    }
    return '+Y';
  }

  /** 罗盘角（度）转弧度；0=北、顺时针。 */
  function compassRad(deg) {
    return Cesium.Math.toRadians(deg || 0);
  }

  /** 两个经纬点的初始方位角（度，罗盘角）。 */
  function bearingDeg(lon1, lat1, lon2, lat2) {
    var p1 = Cesium.Math.toRadians(lat1);
    var p2 = Cesium.Math.toRadians(lat2);
    var dl = Cesium.Math.toRadians(lon2 - lon1);
    var y = Math.sin(dl) * Math.cos(p2);
    var x = Math.cos(p1) * Math.sin(p2) - Math.sin(p1) * Math.cos(p2) * Math.cos(dl);
    return (Cesium.Math.toDegrees(Math.atan2(y, x)) + 360) % 360;
  }

  /** 角度差归一到 (-180, 180]，用于跨 0°/360° 的航向变化率。 */
  function shortestDeltaDeg(a, b) {
    var d = (b - a) % 360;
    if (d > 180) d -= 360;
    if (d <= -180) d += 360;
    return d;
  }

  function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }

  /** 水平距离（米，球面近似；只用于俯仰角，量级足够）。 */
  function groundDistance(lon1, lat1, lon2, lat2) {
    var kx = Math.cos(Cesium.Math.toRadians((lat1 + lat2) * 0.5)) * 111320;
    var ky = 110540;
    var dx = (lon2 - lon1) * kx;
    var dy = (lat2 - lat1) * ky;
    return Math.hypot(dx, dy);
  }

  /**
   * 由位置 + 罗盘航向 + 俯仰/滚转构造 ECEF 姿态四元数。
   *
   * fwdAxis 是资产机头在 **Cesium 模型局部系（Y-up 转 Z-up 之后）** 的方向：
   *   '+Y' = 局部 +Y 为机头（程序化 Blender 资产），'+X' = MiG，'-Y' = Kenney 船。
   * 先把局部轴映射到 (F=前, R=右, U=上)，再写成列向量矩阵求四元数。
   */
  function orientationQuat(pos, headingRad, pitchRad, rollRad, fwdAxis, result) {
    var m = Cesium.Transforms.eastNorthUpToFixedFrame(pos, Cesium.Ellipsoid.WGS84, scratchM4);
    Cesium.Cartesian3.fromElements(m[0], m[1], m[2], scratchE);   // 东
    Cesium.Cartesian3.fromElements(m[4], m[5], m[6], scratchN);   // 北
    Cesium.Cartesian3.fromElements(m[8], m[9], m[10], scratchU);  // 天

    // 水平前向 Fh：从北按罗盘顺时针转 heading
    Cesium.Cartesian3.multiplyByScalar(scratchE, Math.sin(headingRad), scratchFh);
    Cesium.Cartesian3.multiplyByScalar(scratchN, Math.cos(headingRad), scratchT);
    Cesium.Cartesian3.add(scratchFh, scratchT, scratchFh);

    // 俯仰：F = Fh·cos p + U·sin p；U0 = U·cos p − Fh·sin p
    var cp = Math.cos(pitchRad);
    var sp = Math.sin(pitchRad);
    Cesium.Cartesian3.multiplyByScalar(scratchFh, cp, scratchF);
    Cesium.Cartesian3.multiplyByScalar(scratchU, sp, scratchT);
    Cesium.Cartesian3.add(scratchF, scratchT, scratchF);

    Cesium.Cartesian3.multiplyByScalar(scratchU, cp, scratchU0);
    Cesium.Cartesian3.multiplyByScalar(scratchFh, -sp, scratchT);
    Cesium.Cartesian3.add(scratchU0, scratchT, scratchU0);

    // 右向 R = F × U0；滚转绕 F 旋转 (R, U0)
    Cesium.Cartesian3.cross(scratchF, scratchU0, scratchR);
    var cr = Math.cos(rollRad);
    var sr = Math.sin(rollRad);
    Cesium.Cartesian3.multiplyByScalar(scratchR, cr, scratchRp);
    Cesium.Cartesian3.multiplyByScalar(scratchU0, -sr, scratchT);
    Cesium.Cartesian3.add(scratchRp, scratchT, scratchRp);

    Cesium.Cartesian3.multiplyByScalar(scratchU0, cr, scratchUp);
    Cesium.Cartesian3.multiplyByScalar(scratchR, sr, scratchT);
    Cesium.Cartesian3.add(scratchUp, scratchT, scratchUp);

    var X, Y, Z;
    switch (fwdAxis) {
      case '-Y':
        Y = Cesium.Cartesian3.negate(scratchF, scratchY);
        Z = Cesium.Cartesian3.clone(scratchUp, scratchZ);
        X = Cesium.Cartesian3.cross(Y, Z, scratchX);
        break;
      case '+X':
        X = Cesium.Cartesian3.clone(scratchF, scratchX);
        Z = Cesium.Cartesian3.clone(scratchUp, scratchZ);
        Y = Cesium.Cartesian3.cross(Z, X, scratchY);
        break;
      case '-X':
        X = Cesium.Cartesian3.negate(scratchF, scratchX);
        Z = Cesium.Cartesian3.clone(scratchUp, scratchZ);
        Y = Cesium.Cartesian3.cross(Z, X, scratchY);
        break;
      case '+Y':
      default:
        Y = Cesium.Cartesian3.clone(scratchF, scratchY);
        Z = Cesium.Cartesian3.clone(scratchUp, scratchZ);
        X = Cesium.Cartesian3.cross(Y, Z, scratchX);
        break;
    }

    Cesium.Matrix3.fromColumnMajorArray([
      X.x, X.y, X.z,
      Y.x, Y.y, Y.z,
      Z.x, Z.y, Z.z,
    ], scratchM3);
    return Cesium.Quaternion.fromRotationMatrix(scratchM3, result);
  }

  /**
   * 为目标编出姿态采样：航向 / 俯仰 / 滚转。
   * 航迹是姿态的**唯一依据**：航向优先用帧里的 heading 字段（引擎已经沿道路/
   * 航路算好），缺失时用前后点的方位角；俯仰/滚转只对空中目标生效。
   */
  function orientationSamples(def, samples, tickSeconds) {
    var n = samples.length;
    var formClass = window.EntityLod ? window.EntityLod.formClassOf(def) : null;
    var fwdAxis = forwardAxisOf(formClass);
    var isAir = String(def.domain || '').toLowerCase() === 'air';
    var headings = new Array(n);
    var pitches = new Array(n);
    var rolls = new Array(n);
    var i;

    // 航向：heading 字段优先；不合法时用相邻位置方位角
    for (i = 0; i < n; i++) {
      var s = samples[i];
      var h = Number(s.heading);
      if (isFinite(h)) {
        headings[i] = ((h % 360) + 360) % 360;
      } else {
        var a = samples[Math.max(0, i - 1)];
        var b = samples[Math.min(n - 1, i + 1)];
        headings[i] = (a !== b && (a.lon !== b.lon || a.lat !== b.lat))
          ? bearingDeg(a.lon, a.lat, b.lon, b.lat)
          : (i > 0 ? headings[i - 1] : 0);
      }
    }

    // 空中：由高度差得俯仰，由航向变化率得坡度
    for (i = 0; i < n; i++) {
      pitches[i] = 0;
      rolls[i] = 0;
      if (!isAir) continue;
      var sp = samples[i];
      var speed = Math.max(0, Number(sp.speed_kn) || 0) * 0.514444;
      if (speed < MIN_SPEED_MPS) continue;          // 静止/到达终点：保持平飞

      var i0 = Math.max(0, i - 1);
      var i1 = Math.min(n - 1, i + 1);
      var dt = Math.max(1e-6, (i1 - i0) * tickSeconds);
      var gd = groundDistance(samples[i0].lon, samples[i0].lat,
                              samples[i1].lon, samples[i1].lat);
      var dAlt = (samples[i1].alt || 0) - (samples[i0].alt || 0);
      if (gd > 1) pitches[i] = clamp(Math.atan2(dAlt, gd), -PITCH_LIMIT_RAD, PITCH_LIMIT_RAD);

      var dh = shortestDeltaDeg(headings[i0], headings[i1]);
      var yawRate = Cesium.Math.toRadians(dh) / dt;
      rolls[i] = clamp(Math.atan2(speed * yawRate, 9.81), -ROLL_LIMIT_RAD, ROLL_LIMIT_RAD);
    }

    // 滚转做两遍 3 点平滑：heading 在航路拐点会阶跃，直接取切线会让机身
    // 在一帧内横滚 30°，看起来像被拍了一下。
    if (isAir) {
      for (var pass = 0; pass < 2; pass++) {
        var smooth = rolls.slice();
        for (i = 0; i < n; i++) {
          var lo = Math.max(0, i - 1);
          var hi = Math.min(n - 1, i + 1);
          smooth[i] = (rolls[lo] + rolls[i] * 2 + rolls[hi]) / 4;
        }
        rolls = smooth;
      }
    }

    var out = { fwdAxis: fwdAxis, headings: headings, pitches: pitches, rolls: rolls };
    return out;
  }

  /**
   * 编译一份 CZML 文档（纯函数，便于 Node 侧断言结构）。
   *
   * 只放"运动"属性：position + orientation。标牌/模型/图标仍由 app.js 建，
   * 这样低空模型/高空图标的既有 LOD 逻辑一行都不用改，运动却统一由 CZML 驱动。
   */
  function buildCzml(data) {
    var startIso = data.start_time_utc || new Date().toISOString();
    var start = Cesium.JulianDate.fromIso8601(startIso);
    var frames = data.frames || [];
    var tickSeconds = Number(data.tick_seconds) || TICK_SECONDS_FALLBACK;
    var defs = data.targets || [];
    var idIndex = {};
    defs.forEach(function (d, i) { idIndex[d.id] = i; });

    // 先按目标归拢帧样本，避免对每个目标重复扫描全部帧
    var perTarget = defs.map(function () { return []; });
    frames.forEach(function (f, fi) {
      var t = Number(f.t_seconds);
      if (!isFinite(t)) t = fi * tickSeconds;
      (f.targets || []).forEach(function (s) {
        var ti = idIndex[s.id];
        if (ti === undefined) return;
        perTarget[ti].push({ s: s, t: t });
      });
    });

    var domainOf = {};
    defs.forEach(function (d) { domainOf[d.id] = d; });

    var doc = [{
      id: 'document',
      name: (data.case_name || 'situation') + ' target motion',
      version: '1.0',
      clock: {
        interval: startIso + '/' + Cesium.JulianDate.toIso8601(
          Cesium.JulianDate.addSeconds(start, Math.max(0, (frames.length - 1) * tickSeconds), scratchClock)),
        currentTime: startIso,
        multiplier: 1,
        range: 'CLAMPED',
        step: 'SYSTEM_CLOCK_MULTIPLIER',
      },
    }];

    var lastT = 0;
    defs.forEach(function (def, di) {
      var rows = perTarget[di];
      if (!rows.length) return;
      var samples = rows.map(function (r) { return r.s; });
      var pos = [];
      var quat = [];
      var orient = orientationSamples(def, samples, tickSeconds);
      var q = new Cesium.Quaternion();
      for (var i = 0; i < samples.length; i++) {
        var s = samples[i];
        var t = rows[i].t;
        lastT = Math.max(lastT, t);
        var cart = Cesium.Cartesian3.fromDegrees(s.lon, s.lat, s.alt || 0);
        pos.push(t, s.lon, s.lat, s.alt || 0);
        orientationQuat(cart,
          compassRad(orient.headings[i]),
          orient.pitches[i],
          orient.rolls[i],
          orient.fwdAxis, q);
        // CZML unitQuaternion 的多采样格式是 5 元组 [t, x, y, z, w]
        quat.push(t, q.x, q.y, q.z, q.w);
      }
      doc.push({
        id: 'motion-' + def.id,
        name: def.id + ' motion',
        position: {
          epoch: startIso,
          cartographicDegrees: pos,
          interpolationAlgorithm: 'LINEAR',
          interpolationDegree: 1,
        },
        orientation: {
          epoch: startIso,
          unitQuaternion: quat,
          interpolationAlgorithm: 'LINEAR',
          interpolationDegree: 1,
        },
      });
    });

    doc[0].clock.interval = startIso + '/' + Cesium.JulianDate.toIso8601(
      Cesium.JulianDate.addSeconds(start, lastT, scratchClock));
    return doc;
  }

  /**
   * 装载并绑定到既有目标实体。
   *
   * @param viewer  Cesium.Viewer
   * @param data    案例数据（含 frames/targets/tick_seconds/start_time_utc）
   * @returns Promise<TargetMotion stats>
   */
  function load(viewer, data) {
    dispose(viewer);
    var token = ++tokenSeq;
    var t0 = (typeof performance !== 'undefined' && performance.now)
      ? performance.now() : Date.now();
    var czml = buildCzml(data);
    var buildMs = ((typeof performance !== 'undefined' && performance.now)
      ? performance.now() : Date.now()) - t0;
    var ds = new Cesium.CzmlDataSource('target-motion');
    var start = Cesium.JulianDate.fromIso8601(data.start_time_utc ||
      Cesium.JulianDate.toIso8601(viewer.clock.startTime));
    var tickSeconds = Number(data.tick_seconds) || TICK_SECONDS_FALLBACK;
    var end = Cesium.JulianDate.addSeconds(start,
      Math.max(0, ((data.frames || []).length - 1) * tickSeconds), new Cesium.JulianDate());

    return ds.load(czml).then(function () {
      if (token !== tokenSeq) { ds.destroy(); return null; }
      viewer.dataSources.add(ds);
      var bound = 0;
      (data.targets || []).forEach(function (def) {
        var me = ds.entities.getById('motion-' + def.id);
        if (!me) return;
        var ve = viewer.entities.getById('target-' + def.id);
        if (ve) {
          ve.position = me.position;
          ve.orientation = me.orientation;
          bound++;
        }
        var se = viewer.entities.getById('sensor-' + def.id);
        if (se && me.position) se.position = me.position;
        // 血条同样是"挂在目标身上"的图元：不绑同一 motion position 就仍按整数帧
        // 取位，高速目标播放/暂停到半帧时血条会落在图标身后几十像素。
        var he = viewer.entities.getById('hp-' + def.id);
        if (he && me.position) he.position = me.position;
      });
      // 时钟只作为采样查询的游标：由 app 的 tickFloat 驱动，禁掉 Cesium 自己的推进
      viewer.clock.startTime = Cesium.JulianDate.clone(start, viewer.clock.startTime);
      viewer.clock.stopTime = Cesium.JulianDate.clone(end, viewer.clock.stopTime);
      viewer.clock.currentTime = Cesium.JulianDate.clone(start, viewer.clock.currentTime);
      viewer.clock.clockRange = Cesium.ClockRange.CLAMPED;
      viewer.clock.shouldAnimate = false;
      viewer.clock.multiplier = 1;

      current = {
        viewer: viewer, data: data, ds: ds,
        start: start, end: end, tickSeconds: tickSeconds,
        entities: bound, samples: (data.frames || []).length,
        source: 'czml', ready: true,
        buildMs: +buildMs.toFixed(1),
        loadMs: +(((typeof performance !== 'undefined' && performance.now)
          ? performance.now() : Date.now()) - t0).toFixed(1),
      };
      return stats();
    });
  }

  /** 由 app 的连续帧号（tickFloat）同步 Cesium 时钟；每帧调用一次。 */
  function syncClock(tickFloat) {
    if (!current || !current.viewer) return;
    var sec = Math.max(0, tickFloat) * current.tickSeconds;
    Cesium.JulianDate.addSeconds(current.start, sec, scratchClock);
    Cesium.JulianDate.clone(scratchClock, current.viewer.clock.currentTime);
  }

  /** 当前 Cesium 时钟（没有 CZML 源时返回 null）。 */
  function clockTime() {
    return current ? current.viewer.clock.currentTime : null;
  }

  function isReady() {
    return !!(current && current.ready);
  }

  function stats() {
    if (!current) return null;
    return {
      ready: current.ready,
      source: current.source,
      entities: current.entities,
      samples: current.samples,
      buildMs: current.buildMs,
      loadMs: current.loadMs,
      start: Cesium.JulianDate.toIso8601(current.start),
      end: Cesium.JulianDate.toIso8601(current.end),
      currentTime: Cesium.JulianDate.toIso8601(current.viewer.clock.currentTime),
    };
  }

  function dispose(viewer) {
    if (!current) return;
    try {
      if (current.ds && viewer && viewer.dataSources.contains(current.ds)) {
        viewer.dataSources.remove(current.ds, true);
      }
    } catch (e) { /* 场景已拆时忽略 */ }
    current = null;
    tokenSeq++;   // 使尚未 resolve 的 load 作废
  }

  return {
    buildCzml: buildCzml,
    load: load,
    syncClock: syncClock,
    clockTime: clockTime,
    isReady: isReady,
    stats: stats,
    dispose: dispose,
    forwardAxisOf: forwardAxisOf,
    bearingDeg: bearingDeg,
    orientationQuat: orientationQuat,   // 诊断页/用例直接复用同一套姿态数学
  };
}));
