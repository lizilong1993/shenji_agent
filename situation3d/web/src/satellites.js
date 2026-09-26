/**
 * 卫星过境图层：星下点 + 过境时刻表。
 *
 * 数据全部来自**服务端离线 TLE 目录**（engine/data/tle/，由 tools/fetch_tle.py
 * 抓一次落盘）。前端不解析 TLE、不做轨道计算：SGP4 在引擎侧（engine/sgp4.py，
 * 有官方参考向量校验），前端只呈现。这样只有一份轨道实现，不会两边漂移。
 *
 * 图元归属：本图层用 Entity（不贴地的点与标签），与影像、建筑体块不在同一条
 * 渲染管线上，因此**不参与 imagery 的 z 序**；它靠 disableDepthTestDistance 保证
 * 星下点与标签不被地形或三维实景挡住，这一点与地名层一致（见 layerRegistry.js
 * 文件头关于三条顺序轴的说明）。
 *
 * 星历时效会**如实显示**：服务端返回 age_days / stale，界面据此提示。拿过期根数
 * 算出的过境时刻看着"精确"却可能偏几分钟，把警告藏起来比显示它更糟。
 */
(function () {
  'use strict';

  var LAYER_ID = 'satellites.passes';

  var state = {
    viewer: null,
    enabled: false,
    entities: [],
    labelPool: null,
    lastResult: null,
    lastError: null,
    loading: false,
    // 查询参数。跟随相机地面点，但只在显式刷新时才重新请求——
    // SGP4 要跑约 900 个目标，不能每帧或每次拖动都打一次。
    opts: { hours: 6, minElev: 15, limit: 60, groups: '' },
  };

  function setStatus(text, kind) {
    if (window.SituationHost && SituationHost.setStatus) SituationHost.setStatus(text, kind);
  }

  function setNote(text) {
    var n = document.getElementById('satNote');
    if (n) n.textContent = text;
  }

  /** 相机地面点，用作过境查询的观测点。 */
  function observerPoint() {
    var v = state.viewer;
    if (!v) return null;
    var c = v.camera.positionCartographic;
    var lat = Cesium.Math.toDegrees(c.latitude);
    var lon = Cesium.Math.toDegrees(c.longitude);
    // 相机在极区或高度异常时经纬没有意义，夹一下避免发出离谱查询。
    if (!isFinite(lat) || !isFinite(lon)) return null;
    return { lat: Math.max(-85, Math.min(85, lat)), lon: lon };
  }

  function clearEntities() {
    var v = state.viewer;
    if (!v) return;
    state.entities.forEach(function (e) {
      try { v.entities.remove(e); } catch (err) { /* 已经不在集合里 */ }
    });
    state.entities = [];
  }

  /** 把一次查询结果画成星下点 + 标签。 */
  function draw(result) {
    var v = state.viewer;
    if (!v) return 0;
    clearEntities();
    var passes = result.passes || [];
    // 只画"还没发生"的过境，历史过境画出来没有意义。
    var now = Date.now();
    var drawn = 0;
    passes.forEach(function (p) {
      var rise = Date.parse(p.rise_utc);
      if (!isFinite(rise) || rise < now - 60000) return;
      var e = v.entities.add({
        id: 'sat-pass-' + p.norad_id + '-' + p.rise_utc,
        name: p.satellite,
        position: Cesium.Cartesian3.fromDegrees(p.peak_lon, p.peak_lat, p.peak_alt_km * 1000),
        point: {
          pixelSize: 7,
          color: Cesium.Color.fromCssColorString('#7ad1ff'),
          outlineColor: Cesium.Color.fromCssColorString('#0d2b3e'),
          outlineWidth: 1.5,
          // 不贴地：星下点必须始终可见，否则被地形/实景挡掉就成了"图层没生效"。
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
        label: {
          text: p.satellite + '  ' + p.max_elevation_deg + '°',
          font: '11px "Microsoft YaHei", "Noto Sans CJK SC", sans-serif',
          fillColor: Cesium.Color.fromCssColorString('#0d2b3e'),
          outlineColor: Cesium.Color.WHITE,
          outlineWidth: 3,
          style: Cesium.LabelStyle.FILL_AND_OUTLINE,
          pixelOffset: new Cesium.Cartesian2(0, -16),
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
          // 过境点密集时先隐去标签，避免糊成一片；点还在，可点选。
          scaleByDistance: new Cesium.NearFarScalar(1e6, 1.0, 1.2e7, 0.0),
        },
      });
      state.entities.push(e);
      drawn++;
    });
    return drawn;
  }

  function describe(result) {
    var groups = result.groups || {};
    var stale = Object.keys(groups).filter(function (g) { return groups[g].stale; });
    var parts = ['未来 ' + result.window.hours + ' 小时内过境 ' + result.pass_count + ' 次'];
    if (result.skipped_count) {
      // 深空目标不计入：说明清楚，否则"为什么没有那些星"会被当成 bug。
      parts.push('深空目标 ' + result.skipped_count + ' 个未用近地模型计算');
    }
    var ages = Object.keys(groups).map(function (g) { return groups[g].age_days; });
    if (ages.length) {
      var maxAge = Math.max.apply(null, ages);
      parts.push('星历最新 ' + maxAge.toFixed(1) + ' 天前');
    }
    if (stale.length) parts.push('⚠ 星历偏旧：' + stale.join('/'));
    return parts.join(' · ');
  }

  function query() {
    if (!state.viewer) return Promise.resolve(null);
    var obs = observerPoint();
    if (!obs) { setNote('相机位置无效，无法查询过境。'); return Promise.resolve(null); }
    var o = state.opts;
    var url = '/api/satellites/passes?lat=' + obs.lat.toFixed(4) +
      '&lon=' + obs.lon.toFixed(4) + '&hours=' + o.hours +
      '&minElev=' + o.minElev + '&limit=' + o.limit +
      (o.groups ? '&group=' + encodeURIComponent(o.groups) : '');
    state.loading = true;
    setNote('正在计算过境（SGP4，约 900 个目标）…');
    // 服务端要跑完整组 TLE，给足超时；失败必须显式，不能静默留空。
    var ctrl = window.AbortController ? new AbortController() : null;
    var timer = ctrl ? setTimeout(function () { ctrl.abort(); }, 45000) : null;
    return fetch(url, { cache: 'no-store', signal: ctrl ? ctrl.signal : undefined })
      .then(function (r) {
        if (!r.ok) return r.json().catch(function () { return {}; }).then(function (j) {
          throw new Error(j.error || ('HTTP ' + r.status));
        });
        return r.json();
      })
      .then(function (result) {
        state.lastResult = result;
        state.lastError = null;
        var n = draw(result);
        setNote(describe(result) + (n !== result.pass_count
          ? '（已画 ' + n + ' 个未来的）' : ''));
        return result;
      })
      .catch(function (e) {
        state.lastError = String(e && e.message ? e.message : e);
        setNote('过境查询失败：' + state.lastError);
        return null;
      })
      .then(function (r) {
        state.loading = false;
        if (timer) clearTimeout(timer);
        return r;
      });
  }

  function setEnabled(on) {
    state.enabled = !!on;
    if (!on) {
      clearEntities();
      setNote('卫星过境图层已关闭。');
      return Promise.resolve({ enabled: false });
    }
    return query().then(function (r) {
      return { enabled: true, ok: !!r, passCount: r ? r.pass_count : 0 };
    });
  }

  function init(viewer) {
    state.viewer = viewer;
    return state;
  }

  /** 取证用。 */
  function debug() {
    var r = state.lastResult;
    return {
      enabled: state.enabled,
      loading: state.loading,
      entityCount: state.entities.length,
      lastError: state.lastError,
      opts: state.opts,
      observer: observerPoint(),
      passCount: r ? r.pass_count : null,
      skippedCount: r ? r.skipped_count : null,
      groups: r ? r.groups : null,
      generatedUtc: r ? r.generated_utc : null,
    };
  }

  function setOption(key, value) {
    if (key in state.opts) state.opts[key] = value;
  }

  window.SatelliteLayer = {
    LAYER_ID: LAYER_ID,
    init: init,
    setEnabled: setEnabled,
    refresh: query,
    setOption: setOption,
    debug: debug,
  };
})();