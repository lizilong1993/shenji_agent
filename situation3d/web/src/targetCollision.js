/**
 * targetCollision.js — 基于 GLB 包围盒的 2D OBB 碰撞检测。
 *
 * 只回答“谁和谁重叠、重叠多深”，不移动任何目标：变道和导航由外部算法
 * 消费 `window.TargetCollision.checkAt(time)` 的结果。
 *
 * 数据来源：`targets[].collision_bbox`（生成器从归一化 GLB 量出的长/宽/高，
 * 米）；`targets[].lane` 给出车道宽和车道中心偏移。位置与航向从 CZML 插值
 * 实体读取，所以暂停、拖动时间轴和播放中的结果一致。
 */
(function (root, factory) {
  var api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.TargetCollision = api;
}(typeof window !== 'undefined' ? window : null, function () {
  'use strict';

  var ctx = null;
  var defs = {};
  var lastCheck = 0;
  var lastResult = [];
  var scratchTime = new Cesium.JulianDate();
  var scratchCarto = new Cesium.Cartographic();

  function init(viewer, data, api) {
    ctx = { viewer: viewer, data: data, api: api || {} };
    defs = {};
    (data.targets || []).forEach(function (def) { defs[def.id] = def; });
    lastResult = [];
    lastCheck = 0;
  }

  function lonLatAlt(position) {
    var carto = Cesium.Cartographic.fromCartesian(position, undefined, scratchCarto);
    if (!carto) return null;
    return {
      lon: Cesium.Math.toDegrees(carto.longitude),
      lat: Cesium.Math.toDegrees(carto.latitude),
      alt: carto.height || 0,
    };
  }

  function headingOf(id, def, time) {
    var entity = ctx.viewer.entities.getById('target-' + id);
    if (!entity || !entity.position) return Number((ctx.api.sample && ctx.api.sample(id) || {}).heading) || 0;
    var p0 = entity.position.getValue(time);
    var p1 = entity.position.getValue(Cesium.JulianDate.addSeconds(time, 0.5, scratchTime));
    if (!p0 || !p1) {
      var s = ctx.api.sample ? ctx.api.sample(id) : null;
      return Number(s && s.heading) || 0;
    }
    var a = lonLatAlt(p0), b = lonLatAlt(p1);
    if (!a || !b) return 0;
    var dy = (b.lat - a.lat) * 110540;
    var dx = (b.lon - a.lon) * 111320 * Math.cos(Cesium.Math.toRadians(a.lat));
    if (Math.hypot(dx, dy) < 0.05) {
      var row = ctx.api.sample ? ctx.api.sample(id) : null;
      return Number(row && row.heading) || 0;
    }
    return (Math.atan2(dx, dy) * 180 / Math.PI + 360) % 360;
  }

  function corners(box, refLat, refLon) {
    var x = (box.lon - refLon) * 111320 * Math.cos(Cesium.Math.toRadians(refLat));
    var y = (box.lat - refLat) * 110540;
    var h = box.heading * Math.PI / 180;
    var fx = Math.sin(h), fy = Math.cos(h);
    var rx = Math.cos(h), ry = -Math.sin(h);
    var hl = Math.max(0, box.length_m) / 2;
    var hw = Math.max(0, box.width_m) / 2;
    var out = [];
    [[1, 1], [1, -1], [-1, -1], [-1, 1]].forEach(function (s) {
      out.push([x + fx * hl * s[0] + rx * hw * s[1],
                y + fy * hl * s[0] + ry * hw * s[1]]);
    });
    return out;
  }

  function overlap(a, b) {
    var refLat = (a.lat + b.lat) / 2;
    var refLon = (a.lon + b.lon) / 2;
    var ca = corners(a, refLat, refLon);
    var cb = corners(b, refLat, refLon);
    var axes = [];
    [a, b].forEach(function (box) {
      var h = box.heading * Math.PI / 180;
      axes.push([Math.sin(h), Math.cos(h)]);
      axes.push([Math.cos(h), -Math.sin(h)]);
    });
    var depth = Infinity;
    for (var i = 0; i < axes.length; i++) {
      var ax = axes[i][0], ay = axes[i][1];
      var amin = Infinity, amax = -Infinity, bmin = Infinity, bmax = -Infinity;
      for (var j = 0; j < 4; j++) {
        var pa = ca[j][0] * ax + ca[j][1] * ay;
        var pb = cb[j][0] * ax + cb[j][1] * ay;
        if (pa < amin) amin = pa;
        if (pa > amax) amax = pa;
        if (pb < bmin) bmin = pb;
        if (pb > bmax) bmax = pb;
      }
      if (amax < bmin || bmax < amin) return null;
      depth = Math.min(depth, amax - bmin, bmax - amin);
    }
    return Math.round(depth * 1000) / 1000;
  }

  /**
   * @param {Cesium.JulianDate} time
   * @param {string[]} domains 默认只检测地面车辆；船/机可显式传入。
   */
  function checkAt(time, domains) {
    if (!ctx) return [];
    domains = domains || ['land'];
    var boxes = [];
    Object.keys(defs).forEach(function (id) {
      var def = defs[id];
      if (domains.indexOf(def.domain) < 0) return;
      var bbox = def.collision_bbox;
      if (!bbox || !(bbox.length_m > 0) || !(bbox.width_m > 0)) return;
      var row = ctx.api.sample ? ctx.api.sample(id) : null;
      if (row && row.mode === 'embarked') return;
      var entity = ctx.viewer.entities.getById('target-' + id);
      if (!entity || !entity.position) return;
      var pos = entity.position.getValue(time);
      var geo = pos && lonLatAlt(pos);
      if (!geo) return;
      boxes.push({
        id: id, lat: geo.lat, lon: geo.lon,
        heading: headingOf(id, def, time),
        length_m: Number(bbox.length_m), width_m: Number(bbox.width_m),
        height_m: Number(bbox.height_m) || 0,
      });
    });
    var hits = [];
    for (var i = 0; i < boxes.length; i++) {
      for (var j = i + 1; j < boxes.length; j++) {
        var depth = overlap(boxes[i], boxes[j]);
        if (depth !== null) {
          hits.push({ a: boxes[i].id, b: boxes[j].id, depth_m: depth });
        }
      }
    }
    return hits;
  }

  function tick(nowMs) {
    if (!ctx || !ctx.viewer || !ctx.viewer.clock) return;
    if (nowMs - lastCheck < 500) return;
    lastCheck = nowMs;
    lastResult = checkAt(ctx.viewer.clock.currentTime);
  }

  function debug() {
    return {
      enabled: !!ctx,
      lastCount: lastResult.length,
      last: lastResult.slice(0, 20),
      lane: (ctx && ctx.data && ctx.data.targets || [])
        .filter(function (d) { return d.lane; })
        .slice(0, 3)
        .map(function (d) { return { id: d.id, lane: d.lane, bbox: d.collision_bbox }; }),
    };
  }

  return {
    init: init,
    tick: tick,
    checkAt: checkAt,
    overlap: overlap,
    debug: debug,
  };
}));
