/**
 * 场景文档：把"当前看到的这一幅画面"变成一段可分享的 URL，并能从 URL 还原。
 *
 * 为什么需要
 * ----------
 * 在此之前只有 `?cam=lon,lat,height[,pitch]` 一个机位钩子，且它**只**带机位：
 *
 *   - 图层开关状态全丢——分享出去的链接打开后是默认图层组合，跟当时看到的不是一回事；
 *   - 时间轴位置丢失——态势回放的"第几帧"没法复现；
 *   - 案例靠另一个参数 `?case=` 单独给，两者要各写一次才拼得出一个完整场景。
 *
 * 于是"把现在这个画面发给别人看"这件事做不了。排查问题时也一样：描述了一堆现象，
 * 对方没法一键复现到同一帧、同一机位、同一图层组合。
 *
 * 做法
 * ----
 * 一个 `?scene=` 参数把机位 + 图层意图 + 时间轴 + 案例打包进去，内容是
 * base64url(JSON)。用短键（m/v/l/t/c）压缩体积，且**只存与默认值不同的项**，
 * 常见场景下自然很短。
 *
 * 为什么不是"只存 URL 里能读到的参数"
 * ----------------------------------
 * 因为图层意图的真相在 LayerRegistry 里（含互斥解算），不在 URL 里。若只回放 URL
 * 参数，用户在这次会话里手动关掉的图层会还原不回来。所以 capture 直接读注册表。
 *
 * 还原顺序很重要
 * --------------
 * `apply()` 必须**在案例装载之后**调用：clearSituation / applySituationData 会把
 * 机位重置成默认值、并把图层意图写回初始状态，先应用会被覆盖掉。app.js 里
 * 由 applySceneHook() 在 applyCaseHooks() 之后调用，注释里写明了原因。
 */
(function () {
  'use strict';

  var PARAM = 'scene';
  // 短键：URL 里一个字都算长度，长键在给人贴链接时会明显变长。
  var K = { camera: 'm', layers: 'l', tick: 't', case: 'c', version: 'v' };

  function b64urlEncode(str) {
    // 先按 UTF-8 编码再转 base64，避免 btoa 遇到非 ASCII（中文案例 id）直接抛。
    var bytes = new TextEncoder().encode(str);
    var bin = '';
    for (var i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
    return btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
  }

  function b64urlDecode(s) {
    var t = String(s).replace(/-/g, '+').replace(/_/g, '/');
    // 补 padding 到 4 的倍数。**不能写 `-(t.length % 4)`**：JS 的 % 保留符号，
    // 长度本身就是 4 的倍数时（146 % 4 == 2 也有坑，长度 148 时得 0 没问题，
    // 但长度 147 时 -(3) = -3 → repeat 直接抛 RangeError: Invalid count value）。
    // 正确写法是 (4 - len % 4) % 4，结果恒为非负。
    t += '='.repeat((4 - (t.length % 4)) % 4);
    var bin = atob(t);
    var bytes = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new TextDecoder().decode(bytes);
  }

  /**
   * 把角度归一到 (-180, 180]。
   *
   * Cesium 的 heading/roll 会随操作累积：实测绕一圈之后 roll 读数是 360 而不是 0，
   * 直接记进 URL 会让链接变长，JSON 里也读着像异常值。归一化后 360 → 0，
   * 语义完全等价（setView 传 360° 与传 0° 结果相同）。
   */
  function normDeg(d) {
    var v = d % 360;
    if (v > 180) v -= 360;
    if (v <= -180) v += 360;
    // 去掉 -0：JSON 里合法但看着别扭
    return v === 0 ? 0 : Math.round(v * 10) / 10;
  }

  /** 相机状态 → 紧凑数组 [lon, lat, heightM, headingDeg, pitchDeg, rollDeg]。 */
  function captureCamera(viewer) {
    var c = viewer.camera;
    var p = c.positionCartographic;
    return [
      +Cesium.Math.toDegrees(p.longitude).toFixed(6),
      +Cesium.Math.toDegrees(p.latitude).toFixed(6),
      Math.round(p.height),
      normDeg(Cesium.Math.toDegrees(c.heading)),
      normDeg(Cesium.Math.toDegrees(c.pitch)),
      normDeg(Cesium.Math.toDegrees(c.roll)),
    ];
  }

  /**
   * 采集当前场景。
   *
   * providers 由 app.js 注入，因为案例 id / 时间轴 / 图层意图都在它的闭包里。
   * 注入而不是到处挂全局，是为了让这个模块能独立测试。
   */
  function capture(viewer, providers) {
    var p = providers || {};
    var doc = {};
    doc[K.version] = 1;
    if (viewer) doc[K.camera] = captureCamera(viewer);

    // 只记"与默认不同"的图层意图。默认值由 LayerRegistry 的定义顺序推不出来，
    // 所以让调用方给一份 defaults；没给就全记（体积大一点但正确）。
    var layers = {};
    var status = (p.layerStatus && p.layerStatus()) || [];
    var defaults = p.layerDefaults || null;
    status.forEach(function (s) {
      if (defaults && defaults[s.id] === s.desired) return;
      layers[s.id] = s.desired ? 1 : 0;
    });
    if (Object.keys(layers).length) doc[K.layers] = layers;

    if (p.tick !== undefined && p.tick !== null && p.tick !== 0) doc[K.tick] = p.tick;
    if (p.caseId) doc[K.case] = p.caseId;
    return doc;
  }

  function encode(doc) {
    return b64urlEncode(JSON.stringify(doc));
  }

  function decode(raw) {
    var s = String(raw || '').trim();
    if (!s) return null;
    try {
      var doc = JSON.parse(b64urlDecode(s));
      if (!doc || typeof doc !== 'object') return null;
      return doc;
    } catch (e) {
      // 非法/被截断的链接不该让页面炸掉——返回 null 由调用方忽略。
      return null;
    }
  }

  /** 校验并夹紧相机数组，避免 URL 里的离谱数值把相机甩到奇怪的地方。 */
  function saneCamera(a) {
    if (!Array.isArray(a) || a.length < 3) return null;
    var lon = Number(a[0]), lat = Number(a[1]), h = Number(a[2]);
    if (!isFinite(lon) || !isFinite(lat) || !isFinite(h)) return null;
    if (lat < -90 || lat > 90) return null;
    if (lon < -180 || lon > 180) return null;
    // 高度钳到 [1 m, 3e7 m]：太低会钻进地形，太高会超出 far 平面。
    h = Math.max(1, Math.min(3e7, h));
    return {
      lon: lon, lat: lat, height: h,
      heading: isFinite(Number(a[3])) ? Number(a[3]) : 0,
      pitch: isFinite(Number(a[4])) ? Number(a[4]) : -70,
      roll: isFinite(Number(a[5])) ? Number(a[5]) : 0,
    };
  }

  /** 还原机位。不触发 flyTo——分享链接打开时应该直接就位，不是先飞一遍。 */
  function applyCamera(viewer, cam) {
    if (!viewer || !cam) return false;
    viewer.camera.cancelFlight();
    viewer.camera.setView({
      destination: Cesium.Cartesian3.fromDegrees(cam.lon, cam.lat, cam.height),
      orientation: {
        heading: Cesium.Math.toRadians(cam.heading),
        pitch: Cesium.Math.toRadians(cam.pitch),
        roll: Cesium.Math.toRadians(cam.roll),
      },
    });
    return true;
  }

  /**
   * 从 URL 读取场景参数。
   *
   * 放在 URL 的 hash 之外（普通 query），这样服务端日志里也能看到被分享的场景。
   */
  function fromUrl(search) {
    var q = new URLSearchParams(search === undefined ? window.location.search : search);
    return decode(q.get(PARAM));
  }

  function toUrl(doc, search) {
    var q = new URLSearchParams(search === undefined ? window.location.search : search);
    if (!doc) q.delete(PARAM);
    else q.set(PARAM, encode(doc));
    return window.location.pathname + '?' + q.toString();
  }

  /** 生成本次会话的可分享链接（用当前 URL 的其它参数做基底，保留 ?case= 等）。 */
  function shareUrl(viewer, providers) {
    return toUrl(capture(viewer, providers));
  }

  window.SceneDoc = {
    PARAM: PARAM,
    capture: capture,
    encode: encode,
    decode: decode,
    fromUrl: fromUrl,
    toUrl: toUrl,
    shareUrl: shareUrl,
    applyCamera: applyCamera,
    saneCamera: saneCamera,
    // 暴露给测试：确认编码能往返
    _roundTrip: function (doc) { return decode(encode(doc)); },
  };
})();