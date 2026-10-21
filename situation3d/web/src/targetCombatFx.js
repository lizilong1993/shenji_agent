/**
 * targetCombatFx.js — 交战特效（开火 / 命中 / 战损 / 选中目标扫描）
 *
 * 数据来源：案例 `events[]` 中 type=fire/impact/destroy 的事件
 * （由 tools/combat_events.py 合成，或后续引擎直接产出）。
 * 战损档位仍以 targetStatus.js 为唯一来源；本模块只做渲染，不改 HP。
 *
 * 设计原则：
 *   · 不把动画烘焙进 GLB：开火时刻由事件决定，glTF 剪辑无法按事件选播；
 *   · 全部用 Billboard / 曳光线（Cesium 时钟默认暂停，ParticleSystem 不发射）；
 *   · 所有粒子/实体池化，按相机距离与数量上限裁剪；
 *   · 暂停/拖动时间轴时，扫描扇区与弹道按 tickFloat 回放，不另起时钟。
 */
(function (root, factory) {
  var api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.TargetCombatFx = api;
}(typeof window !== 'undefined' ? window : null, function () {
  'use strict';

  var ctx = null;
  var data = null;
  var eventsByTick = {};
  var defById = {};
  var impactByFire = {};
  var lastTick = -1;
  var combatEnabled = false;
  var transients = [];          // {entity, until}
  var projectiles = [];         // {entity, trail, fire, impact}
  var smoke = {};               // id -> {birth, ents[]}
  var smokeState = {};          // id -> level
  var scan = null;              // {entity, material, id, range}
  var destroyed = {};           // id -> true
  var hitFlash = {};            // id -> {until, prevColor, prevSize}
  var statusCache = {};
  var origStyle = {};
  var lastStatusTick = -1;
  var scanAngle = 0;
  var lastCameraCheck = 0;
  var visibleScanIds = 0;

  var MAX_SMOKE = 10;
  var MAX_TRANSIENTS = 80;
  var FLASH_MS = 350;

  var scratchPos = new Cesium.Cartesian3();
  var scratchRot = new Cesium.Matrix3();
  var scratchOff = new Cesium.Cartesian3();

  function apiTick() { return ctx && ctx.api.tick ? ctx.api.tick() : 0; }
  function apiTickFloat() { return ctx && ctx.api.tickFloat ? ctx.api.tickFloat() : apiTick(); }
  function apiSelected() { return ctx && ctx.api.selected ? ctx.api.selected() : null; }
  function e(id) { return ctx.api.entity ? ctx.api.entity(id) : null; }
  function sample(id) { return ctx.api.sample ? ctx.api.sample(id) : null; }

  /** 按 Viewer 时钟（tickFloat 插值）取当前位置；sample() 只到整数 tick。 */
  function currentGeo(id) {
    var rec = e(id);
    if (rec && rec.position) {
      try {
        var c = rec.position.getValue(ctx.viewer.clock.currentTime);
        var g = Cesium.Cartographic.fromCartesian(c);
        if (g) return { lon: Cesium.Math.toDegrees(g.longitude), lat: Cesium.Math.toDegrees(g.latitude), alt: g.height || 0 };
      } catch (err) {}
    }
    return sample(id);
  }

  function makeFlashImage() {
    var c = document.createElement('canvas');
    c.width = c.height = 64;
    var g = c.getContext('2d');
    var grd = g.createRadialGradient(32, 32, 2, 32, 32, 30);
    grd.addColorStop(0, 'rgba(255,255,235,1)');
    grd.addColorStop(0.25, 'rgba(255,220,120,0.95)');
    grd.addColorStop(0.6, 'rgba(255,120,30,0.55)');
    grd.addColorStop(1, 'rgba(255,60,0,0)');
    g.fillStyle = grd;
    g.fillRect(0, 0, 64, 64);
    return c.toDataURL('image/png');
  }

  function makeSmokeImage() {
    var c = document.createElement('canvas');
    c.width = c.height = 64;
    var g = c.getContext('2d');
    var grd = g.createRadialGradient(32, 32, 2, 32, 32, 31);
    grd.addColorStop(0, 'rgba(90,90,90,0.75)');
    grd.addColorStop(0.55, 'rgba(70,70,70,0.35)');
    grd.addColorStop(1, 'rgba(50,50,50,0)');
    g.fillStyle = grd;
    g.fillRect(0, 0, 64, 64);
    return c.toDataURL('image/png');
  }

  function makeFireImage() {
    var c = document.createElement('canvas');
    c.width = c.height = 64;
    var g = c.getContext('2d');
    var grd = g.createRadialGradient(32, 36, 2, 32, 32, 30);
    grd.addColorStop(0, 'rgba(255,250,200,1)');
    grd.addColorStop(0.3, 'rgba(255,170,50,0.9)');
    grd.addColorStop(0.7, 'rgba(230,70,10,0.45)');
    grd.addColorStop(1, 'rgba(120,20,0,0)');
    g.fillStyle = grd;
    g.fillRect(0, 0, 64, 64);
    return c.toDataURL('image/png');
  }

  var IMG = { flash: null, smoke: null, fire: null };

  function ensureImages() {
    // 静态 PNG（web/assets/fx）比 canvas data URI 更稳定，也便于美术替换
    IMG.flash = './assets/fx/flash.png';
    IMG.smoke = './assets/fx/smoke.png';
    IMG.fire = './assets/fx/fire.png';
  }

  // ------------------------------------------------------------------
  // 生命周期
  // ------------------------------------------------------------------
  function init(viewer, dataset, api) {
    ctx = { viewer: viewer, api: api };
    data = dataset || {};
    combatEnabled = !!(data.combat && data.combat.model);
    defById = {};
    origStyle = {};
    (data.targets || []).forEach(function (d) {
      defById[d.id] = d;
      var rec = e(d.id);
      if (rec && rec.model) {
        origStyle[d.id] = { color: rec.model.color, amount: rec.model.colorBlendAmount,
          silColor: rec.model.silhouetteColor, silSize: rec.model.silhouetteSize };
      }
    });
    ensureImages();
    indexEvents();
    resetAll();
    lastTick = apiTick();
    rebuildState(lastTick);
  }

  function indexEvents() {
    eventsByTick = {};
    impactByFire = {};
    (data.events || []).forEach(function (ev) {
      if (!ev || !ev.type) return;
      if (ev.type !== 'fire' && ev.type !== 'impact' && ev.type !== 'destroy') return;
      var t = Number(ev.tick);
      if (!isFinite(t)) return;
      (eventsByTick[t] || (eventsByTick[t] = [])).push(ev);
      if (ev.type === 'impact') {
        (ev.cause_event_ids || []).forEach(function (fid) { impactByFire[fid] = ev; });
      }
    });
  }

  function dispose() {
    if (ctx && ctx.viewer) {
      transients.forEach(function (x) { ctx.viewer.entities.remove(x.entity); });
      projectiles.forEach(function (p) {
        ctx.viewer.entities.remove(p.entity);
        if (p.trail) ctx.viewer.entities.remove(p.trail);
      });
      Object.keys(smoke).forEach(function (id) {
        (smoke[id].ents || []).forEach(function (ent) { ctx.viewer.entities.remove(ent); });
      });
      if (scan && scan.entity) ctx.viewer.entities.remove(scan.entity);
    }
    transients = [];
    projectiles = [];
    smoke = {};
    smokeState = {};
    destroyed = {};
    hitFlash = {};
    scan = null;
    ctx = null;
    data = null;
    eventsByTick = {};
    impactByFire = {};
    lastTick = -1;
  }

  function resetAll() {
    if (!ctx) return;
    transients.forEach(function (x) { ctx.viewer.entities.remove(x.entity); });
    projectiles.forEach(function (p) {
      ctx.viewer.entities.remove(p.entity);
      if (p.trail) ctx.viewer.entities.remove(p.trail);
    });
    Object.keys(smoke).forEach(function (id) {
      (smoke[id].ents || []).forEach(function (ent) { ctx.viewer.entities.remove(ent); });
    });
    if (scan && scan.entity) ctx.viewer.entities.remove(scan.entity);
    transients = [];
    projectiles = [];
    smoke = {};
    smokeState = {};
    destroyed = {};
    hitFlash = {};
    statusCache = {};
    lastStatusTick = -1;
    scan = null;
    // 恢复模型着色（重播/回拖时）
    Object.keys(origStyle).forEach(function (id) { restoreStyle(id); });
  }

  function clearPersistentState() {
    Object.keys(smoke).forEach(function (id) {
      (smoke[id].ents || []).forEach(function (ent) { ctx.viewer.entities.remove(ent); });
    });
    smoke = {};
    smokeState = {};
    destroyed = {};
    statusCache = {};
    lastStatusTick = -1;
    Object.keys(origStyle).forEach(function (id) { restoreStyle(id); });
  }

  function rebuildState(tick) {
    if (!combatEnabled) { statusCache = {}; return; }
    // 回拖/跳转：按已发生的事件恢复“已损毁/受损”的持久状态，不补播瞬时特效。
    (data.events || []).forEach(function (ev) {
      if (!ev || Number(ev.tick) > tick) return;
      if (ev.type === 'destroy' && ev.target_id) destroyed[ev.target_id] = true;
    });
    (data.targets || []).forEach(function (d) {
      var st = statusOf(d.id, tick);
      statusCache[d.id] = st;
      if (!st) { restoreStyle(d.id); return; }
      applyDamage(d.id, st, tick);
      if (st >= 3) destroyed[d.id] = true;
    });
  }

  function statusOf(id, tick) {
    if (window.TargetStatus && window.TargetStatus.at) {
      try {
        var s = TargetStatus.at(data, id, tick);
        return s ? s.status : 0;
      } catch (e) { return 0; }
    }
    return destroyed[id] ? 3 : 0;
  }


  // ------------------------------------------------------------------
  // 模型关节（nodeTransformations）：旋翼 / 雷达 / 发射筒
  //
  // 这些节点在 GLB 里已有名字（自建型号与 Z-10 的旋翼盘），直接用
  // NodeTransformationProperty 驱动；不新增节点。旋转轴用 Cesium 局部系：
  // 模型前向 -X、上 +Z、右 +Y。节点原点多在模型原点，故 translation 做
  // pivot 补偿 t = p - R*p，避免绕世界原点甩出去。
  // ------------------------------------------------------------------
  var RIGS = {
    z10: {
      parts: [
        { node: 'Object_9', axis: 'Z', pivot: [-0.005, 0, 0.755], speed: 9.0 },
        { node: 'Object_10', axis: 'Y', pivot: [6.624, 0.463, 0.674], speed: 22.0 },
      ],
    },
    hq16: {
      parts: [{ node: 'hq16_radar', axis: 'Z', pivot: [6.5, 0, 3.0], speed: 0.9 }],
    },
    yj62: {
      parts: [
        { node: 'yj62_tube0', axis: 'Y', pivot: [3.0, -1.35, 3.35], lift: true },
        { node: 'yj62_tube1', axis: 'Y', pivot: [3.0, 0, 3.35], lift: true },
        { node: 'yj62_tube2', axis: 'Y', pivot: [3.0, 1.35, 3.35], lift: true },
      ],
    },
    m1a2: {
      parts: [{ node: 'Object_7', axis: 'X', recoil: true }],
    },
  };
  var rigAngles = { z10: 0, hq16: 0, yj62: 0, m1a2: 0 };

  function rigKey(def) {
    var text = ((def && def.model) || '') + ' ' + ((def && def.platform_class) || '');
    if (/直-?10|z-?10/i.test(text)) return 'z10';
    if (/红旗-?16|hq-?16|ly-?80/i.test(text)) return 'hq16';
    if (/鹰击-?62|yj-?62/i.test(text)) return 'yj62';
    if (/m1a2/i.test(text)) return 'm1a2';
    return null;
  }

  function axisOf(name) {
    return name === 'X' ? Cesium.Cartesian3.UNIT_X
      : (name === 'Y' ? Cesium.Cartesian3.UNIT_Y : Cesium.Cartesian3.UNIT_Z);
  }

  function pivotTranslation(pivot, q) {
    var r = Cesium.Matrix3.fromQuaternion(q, new Cesium.Matrix3());
    var rp = Cesium.Matrix3.multiplyByVector(r, Cesium.Cartesian3.fromElements(pivot[0], pivot[1], pivot[2]), new Cesium.Cartesian3());
    return new Cesium.Cartesian3(pivot[0] - rp.x, pivot[1] - rp.y, pivot[2] - rp.z);
  }

  /** 由 app.js 的 createTargetEntities 调用；没有关节返回 undefined。 */
  function nodeTransformations(def) {
    var key = rigKey(def);
    if (!key || !RIGS[key]) return undefined;
    var out = {};
    RIGS[key].parts.forEach(function (part) {
      var axis = axisOf(part.axis);
      if (part.recoil) {
        out[part.node] = new Cesium.NodeTransformationProperty({
          translation: new Cesium.CallbackProperty(function () {
            return new Cesium.Cartesian3(rigAngles[key], 0, 0);
          }, false),
        });
        return;
      }
      out[part.node] = new Cesium.NodeTransformationProperty({
        rotation: new Cesium.CallbackProperty(function () {
          var a = part.lift ? -rigAngles.yj62 : (part.sweep ? rigAngles[key] : rigAngles[key]);
          return Cesium.Quaternion.fromAxisAngle(axis, a, new Cesium.Quaternion());
        }, false),
        translation: new Cesium.CallbackProperty(function () {
          var a = part.lift ? -rigAngles.yj62 : (part.sweep ? rigAngles[key] : rigAngles[key]);
          var q = Cesium.Quaternion.fromAxisAngle(axis, a, new Cesium.Quaternion());
          return pivotTranslation(part.pivot, q);
        }, false),
      });
    });
    return out;
  }

  // ------------------------------------------------------------------
  // 帧循环
  // ------------------------------------------------------------------
  function frame(dt) {
    if (!ctx || !data) return;
    var tick = apiTick();
    if (tick < lastTick || tick - lastTick > 8) {
      // 拖动/跳转：重置瞬时层，按新时刻恢复持久状态
      clearTransientLayer();
      clearPersistentState();
      lastTick = tick;
      rebuildState(tick);
    } else {
      for (var t = lastTick + 1; t <= tick; t++) processTick(t);
      lastTick = tick;
    }
    if (ctx.api.playing && ctx.api.playing()) {
      var d = Math.max(0, dt || 0);
      scanAngle += d * 1.1;
      rigAngles.z10 = (rigAngles.z10 + d * 9.0) % (Math.PI * 2);
      rigAngles.hq16 = (rigAngles.hq16 + d * 0.9) % (Math.PI * 2);
      rigAngles.yj62 = Math.max(0, rigAngles.yj62 - d * 0.18);
      rigAngles.m1a2 = Math.max(0, rigAngles.m1a2 - d * 0.9);
    }
    updateProjectiles();
    updateTransients();
    updateDamageStates(tick);
    updateSmoke(tick);
    updateScan(tick);
    updateHitFlashes();
  }

  function restoreStyle(id) {
    var rec = e(id);
    var st = origStyle[id];
    if (!rec || !rec.model || !st) return;
    rec.model.color = st.color;
    rec.model.colorBlendAmount = st.amount;
    rec.model.silhouetteColor = st.silColor;
    rec.model.silhouetteSize = st.silSize;
  }

  function updateDamageStates(tick) {
    if (!combatEnabled) return;
    if (tick === lastStatusTick) return;
    lastStatusTick = tick;
    (data.targets || []).forEach(function (d) {
      var st = statusOf(d.id, tick);
      if (st === statusCache[d.id]) return;
      statusCache[d.id] = st;
      if (st >= 1) applyDamage(d.id, st, tick);
      else restoreStyle(d.id);
      if (st >= 3 && !destroyed[d.id]) {
        destroyed[d.id] = true;
        var s = sample(d.id);
        if (s) spawnExplosion(Cesium.Cartesian3.fromDegrees(s.lon, s.lat, (s.alt || 0) + 2));
        ensureSmoke(d.id, 4);
      }
    });
  }

  function clearTransientLayer() {
    transients.forEach(function (x) { ctx.viewer.entities.remove(x.entity); });
    projectiles.forEach(function (p) {
      ctx.viewer.entities.remove(p.entity);
      if (p.trail) ctx.viewer.entities.remove(p.trail);
    });
    transients = [];
    projectiles = [];
    Object.keys(hitFlash).forEach(function (id) { restoreHit(id); });
  }

  function processTick(tick) {
    var list = eventsByTick[tick];
    if (!list) return;
    for (var i = 0; i < list.length; i++) {
      var ev = list[i];
      if (ev.type === 'fire') onFire(ev);
      else if (ev.type === 'impact') onImpact(ev);
      else if (ev.type === 'destroy') onDestroy(ev);
    }
  }

  // ------------------------------------------------------------------
  // 枪口位置
  // ------------------------------------------------------------------
  function formClassOf(def) {
    return window.EntityLod ? EntityLod.formClassOf(def) : null;
  }

  function muzzleOffset(def) {
    var form = formClassOf(def) || '';
    var len = (window.EntityLod && EntityLod.REAL_LENGTH_M[form]) || 12;
    var fwd = len * 0.42, up = len * 0.12;
    if (/destroyer|frigate|corvette|patrol/.test(form)) { fwd = len * 0.55; up = len * 0.08; }
    else if (form === 'tank' || form === 'light_tank' || form === 'spg') { fwd = len * 0.46; up = len * 0.18; }
    else if (form === 'sam' || form === 'mlrs' || form === 'anti_ship') { fwd = len * 0.30; up = len * 0.34; }
    else if (form === 'fighter') { fwd = len * 0.44; up = len * 0.08; }
    else if (form === 'helicopter') { fwd = len * 0.38; up = len * 0.10; }
    else if (form === 'howitzer') { fwd = len * 0.45; up = len * 0.20; }
    return { fwd: fwd, up: up };
  }

  function muzzlePosition(id, def) {
    var rec = e(id);
    if (!rec) return null;
    var time = ctx.viewer.clock.currentTime;
    var pos = null;
    try { pos = rec.position && rec.position.getValue(time); } catch (err) {}
    if (!pos) {
      var s = sample(id);
      if (s) pos = Cesium.Cartesian3.fromDegrees(s.lon, s.lat, s.alt || 0);
    }
    if (!pos) return null;
    var off = muzzleOffset(def || {});
    // 模型局部：前向 -X、上 +Z（见 entityLod.forwardAxis 的实测说明）
    Cesium.Cartesian3.fromElements(-off.fwd, 0, off.up, scratchOff);
    var q = null;
    try { q = rec.orientation && rec.orientation.getValue(time); } catch (err) {}
    if (q) {
      Cesium.Matrix3.fromQuaternion(q, scratchRot);
      Cesium.Matrix3.multiplyByVector(scratchRot, scratchOff, scratchOff);
      Cesium.Cartesian3.add(pos, scratchOff, scratchPos);
      return Cesium.Cartesian3.clone(scratchPos);
    }
    var s2 = sample(id);
    if (s2) {
      var enu = Cesium.Transforms.eastNorthUpToFixedFrame(
        Cesium.Cartesian3.fromDegrees(s2.lon, s2.lat, s2.alt || 0));
      var hdg = Cesium.Math.toRadians(s2.heading || 0);
      var fx = Math.sin(hdg), fy = Math.cos(hdg);
      var dx = off.fwd;
      var wx = enu[0] * fx + enu[4] * fy;
      var wy = enu[1] * fx + enu[5] * fy;
      var wz = enu[2] * fx + enu[6] * fy;
      var upx = enu[8], upy = enu[9], upz = enu[10];
      return new Cesium.Cartesian3(
        pos.x + wx * dx + upx * off.up,
        pos.y + wy * dx + upy * off.up,
        pos.z + wz * dx + upz * off.up);
    }
    return pos;
  }

  // ------------------------------------------------------------------
  // 开火 / 命中 / 击毁
  // ------------------------------------------------------------------
  function onFire(ev) {
    var shooterDef = defById[(ev.actors && ev.actors[0]) || ev.shooter_id];
    if (shooterDef) {
      var rk = rigKey(shooterDef);
      if (rk === 'yj62') rigAngles.yj62 = 0.22;
      else if (rk === 'm1a2') rigAngles.m1a2 = 0.35;
    }
    if (ctx.api.isVisible && !ctx.api.isVisible(defById[(ev.actors && ev.actors[0]) || ev.shooter_id] || {})) return;
    var shooter = (ev.actors && ev.actors[0]) || ev.shooter_id;
    if (!shooter) return;
    var m = muzzlePosition(shooter, defById[shooter]);
    if (!m) return;
    spawnFlash(m, 1.0);
    var impact = impactByFire[ev.id];
    if (impact) {
      var isMissile = /导弹|火箭弹/.test(ev.weapon || '');
      spawnProjectile(ev, m, impact, isMissile);
    }
  }

  function onImpact(ev) {
    var loc = ev.location;
    if (!loc) return;
    var p = Cesium.Cartesian3.fromDegrees(loc[0], loc[1], loc[2] || 0);
    spawnFlash(p, ev.result === 'hit' ? 1.25 : 0.7);
    if (ev.result === 'hit' && ev.target_id) {
      flashHit(ev.target_id);
      // 命中点小烟团
      spawnPuff(p);
    }
  }

  function onDestroy(ev) {
    var id = ev.target_id || (ev.actors && ev.actors[0]);
    if (!id) return;
    var loc = ev.location;
    if (!loc) {
      var s = sample(id);
      if (!s) return;
      loc = [s.lon, s.lat, s.alt || 0];
    }
    destroyed[id] = true;
    var p = Cesium.Cartesian3.fromDegrees(loc[0], loc[1], loc[2] || 0);
    spawnExplosion(p);
    ensureSmoke(id, 4);
    applyDamage(id, 3, apiTick());
  }

  function flashHit(id) {
    var rec = e(id);
    if (!rec || !rec.model) return;
    if (hitFlash[id]) return;
    var prevColor = rec.model.silhouetteColor;
    var prevSize = rec.model.silhouetteSize;
    hitFlash[id] = { prevColor: prevColor, prevSize: prevSize, until: Date.now() + 260 };
    rec.model.silhouetteColor = new Cesium.ConstantProperty(Cesium.Color.fromCssColorString('#ff3b2f'));
    rec.model.silhouetteSize = new Cesium.ConstantProperty(2.0);
  }

  function restoreHit(id) {
    var rec = e(id);
    var h = hitFlash[id];
    if (rec && rec.model && h) {
      rec.model.silhouetteColor = h.prevColor;
      rec.model.silhouetteSize = h.prevSize;
    }
    delete hitFlash[id];
  }

  function updateHitFlashes() {
    var now = Date.now();
    Object.keys(hitFlash).forEach(function (id) {
      if (hitFlash[id].until <= now) restoreHit(id);
    });
  }

  function applyDamage(id, status, tick) {
    if (!status) return;
    var rec = e(id);
    if (rec && rec.model) {
      var amount = status >= 3 ? 0.75 : (status === 2 ? 0.55 : 0.35);
      var col = status >= 3 ? '#1a1a1a' : '#3b3b3b';
      rec.model.color = new Cesium.ConstantProperty(Cesium.Color.fromCssColorString(col));
      rec.model.colorBlendMode = Cesium.ColorBlendMode.MIX;
      rec.model.colorBlendAmount = amount;
    }
    if (status >= 1) ensureSmoke(id, status >= 3 ? 4 : status);
  }

  // ------------------------------------------------------------------
  // 瞬时实体（闪光 / 弹道 / 烟团）
  // ------------------------------------------------------------------
  function addTransient(entity) {
    transients.push({ entity: entity, until: Date.now() + FLASH_MS });
    if (transients.length > MAX_TRANSIENTS) {
      var old = transients.shift();
      ctx.viewer.entities.remove(old.entity);
    }
  }

  function spawnFlash(pos, scale) {
    var ent = ctx.viewer.entities.add({
      position: pos,
      billboard: {
        image: IMG.flash,
        scale: 0.55 * scale,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
        scaleByDistance: new Cesium.NearFarScalar(200, 1.4, 20000, 0.35),
      },
    });
    addTransient(ent);
  }

  function spawnPuff(pos) {
    var ent = ctx.viewer.entities.add({
      position: pos,
      billboard: {
        image: IMG.smoke,
        scale: 0.9,
        color: Cesium.Color.fromAlpha(Cesium.Color.WHITE, 0.55),
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
    });
    transients.push({ entity: ent, until: Date.now() + 700 });
    if (transients.length > MAX_TRANSIENTS) {
      var old = transients.shift();
      ctx.viewer.entities.remove(old.entity);
    }
  }

  function spawnProjectile(fire, startPos, impact, isMissile) {
    var endLoc = impact.location;
    if (!endLoc) return;
    var endPos = Cesium.Cartesian3.fromDegrees(endLoc[0], endLoc[1], endLoc[2] || 0);
    var t0 = Number(fire.tick);
    var t1 = Math.max(t0 + 0.25, Number(impact.tick));
    var missile = isMissile;

    var posProp = new Cesium.CallbackProperty(function (time) {
      var f = missile ? apiTickFloat() : t1;
      var k = Math.max(0, Math.min(1, (f - t0) / (t1 - t0)));
      if (!missile) k = 1;
      return Cesium.Cartesian3.lerp(startPos, endPos, k, new Cesium.Cartesian3());
    }, false);

    var ent = ctx.viewer.entities.add({
      position: posProp,
      billboard: missile ? {
        image: IMG.fire,
        scale: 0.22,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      } : {
        image: IMG.flash,
        scale: 0.10,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
    });
    var trail = ctx.viewer.entities.add({
      polyline: {
        positions: new Cesium.CallbackProperty(function () {
          var cur = posProp.getValue(ctx.viewer.clock.currentTime);
          return cur ? [startPos, cur] : [];
        }, false),
        width: missile ? 2.0 : 1.2,
        material: new Cesium.PolylineGlowMaterialProperty({
          color: missile ? Cesium.Color.fromCssColorString('#ffb35c').withAlpha(0.55)
            : Cesium.Color.fromCssColorString('#ffe7a8').withAlpha(0.35),
          glowPower: 0.4,
        }),
      },
    });
    projectiles.push({ entity: ent, trail: trail, fire: fire, impact: impact, t1: t1, missile: missile });
    if (projectiles.length > 24) {
      var old = projectiles.shift();
      ctx.viewer.entities.remove(old.entity);
      if (old.trail) ctx.viewer.entities.remove(old.trail);
    }
  }

  function updateProjectiles() {
    var f = apiTickFloat();
    for (var i = projectiles.length - 1; i >= 0; i--) {
      var p = projectiles[i];
      if (f > p.t1 + 0.3) {
        ctx.viewer.entities.remove(p.entity);
        if (p.trail) ctx.viewer.entities.remove(p.trail);
        projectiles.splice(i, 1);
      }
    }
  }

  function updateTransients() {
    var now = Date.now();
    for (var i = transients.length - 1; i >= 0; i--) {
      if (transients[i].until <= now) {
        ctx.viewer.entities.remove(transients[i].entity);
        transients.splice(i, 1);
      }
    }
  }

  // ------------------------------------------------------------------
  // 烟柱 / 爆炸（Billboard 池）
  //
  // 不用 Cesium.ParticleSystem：viewer.clock.shouldAnimate 在 app 里是 false
  // （时间轴由 tickFloat 手动同步），ParticleSystem 不会发射粒子。Billboard 的
  // CallbackProperty 由本模块按 tickFloat/墙钟驱动，暂停时也能保持一致。
  // ------------------------------------------------------------------
  function makePlumePuff(id, plume, k) {
    return ctx.viewer.entities.add({
      position: new Cesium.CallbackProperty(function () {
        var s = currentGeo(id);
        if (!s) return undefined;
        var t = apiTickFloat();
        var rise = Math.min(28, Math.max(0, t - plume.birth) * 0.35) + 3 + k * 7;
        return Cesium.Cartesian3.fromDegrees(s.lon, s.lat, (s.alt || 0) + rise);
      }, false),
      billboard: {
        image: IMG.smoke,
        scale: new Cesium.CallbackProperty(function () {
          var grow = Math.min(1, Math.max(0, apiTickFloat() - plume.birth) / 12);
          return (1.0 + k * 0.45) * (0.7 + grow * 0.7);
        }, false),
        color: new Cesium.CallbackProperty(function () {
          var lv = smokeState[id] || 1;
          var a = lv >= 3 ? 0.52 : (lv === 2 ? 0.40 : 0.28);
          return Cesium.Color.fromAlpha(Cesium.Color.fromCssColorString('#8f8f8f'), a * (1 - k * 0.20));
        }, false),
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
    });
  }

  function ensureSmoke(id, level) {
    if (smoke[id]) {
      smokeState[id] = Math.max(smokeState[id] || 0, level);
      return;
    }
    var ids = Object.keys(smoke);
    if (ids.length >= MAX_SMOKE) {
      var first = ids[0];
      (smoke[first].ents || []).forEach(function (ent) { ctx.viewer.entities.remove(ent); });
      delete smoke[first];
      delete smokeState[first];
    }
    var plume = { id: id, birth: apiTickFloat(), ents: [] };
    for (var k = 0; k < 3; k++) plume.ents.push(makePlumePuff(id, plume, k));
    smoke[id] = plume;
    smokeState[id] = level;
  }

  function updateSmoke(tick) {
    var camH = ctx.api.cameraHeight ? ctx.api.cameraHeight() : 0;
    var cam = ctx.viewer.camera.positionWC;
    Object.keys(smoke).forEach(function (id) {
      var plume = smoke[id];
      var s = currentGeo(id);
      var def = defById[id] || {};
      var on = !!(s && camH < 60000 && (ctx.api.isVisible ? ctx.api.isVisible(def) : true));
      if (on) {
        var p = Cesium.Cartesian3.fromDegrees(s.lon, s.lat, s.alt || 0);
        on = Cesium.Cartesian3.distance(cam, p) < 60000;
      }
      (plume.ents || []).forEach(function (ent) { ent.show = on; });
    });
  }

  function spawnExplosion(pos) {
    var born = Date.now();
    var N = 10;
    for (var i = 0; i < N; i++) {
      (function (i) {
        var ang = (i / N) * Math.PI * 2 + (i % 3) * 0.4;
        var r = 6 + (i % 4) * 7;
        var up = 4 + (i % 5) * 5;
        var ent = ctx.viewer.entities.add({
          position: new Cesium.CallbackProperty(function () {
            var k = Math.min(1, (Date.now() - born) / 1100);
            return new Cesium.Cartesian3(
              pos.x + Math.cos(ang) * r * k,
              pos.y + Math.sin(ang) * r * k,
              pos.z + up * k + 6 * k * k);
          }, false),
          billboard: {
            image: i < 5 ? IMG.fire : IMG.smoke,
            scale: new Cesium.CallbackProperty(function () {
              var k = Math.min(1, (Date.now() - born) / 1100);
              return 0.7 + k * (i < 5 ? 2.6 : 2.0);
            }, false),
            color: new Cesium.CallbackProperty(function () {
              var k = Math.min(1, (Date.now() - born) / 1100);
              return Cesium.Color.fromAlpha(Cesium.Color.WHITE, (1 - k) * 0.9);
            }, false),
            disableDepthTestDistance: Number.POSITIVE_INFINITY,
          },
        });
        transients.push({ entity: ent, until: born + 1200 });
      })(i);
    }
    var ring = ctx.viewer.entities.add({
      position: pos,
      ellipse: {
        semiMajorAxis: 28,
        semiMinorAxis: 28,
        fill: false,
        outline: true,
        outlineColor: Cesium.Color.fromCssColorString('#ffd28a').withAlpha(0.65),
        outlineWidth: 3,
        height: 2,
      },
    });
    transients.push({ entity: ring, until: born + 700 });
  }

  // ------------------------------------------------------------------
  // 选中目标的扫描扇区（仅选中）
  // ------------------------------------------------------------------
  function scanHierarchy() {
    if (!scan) return new Cesium.PolygonHierarchy([]);
    var s = currentGeo(scan.id);
    if (!s) return new Cesium.PolygonHierarchy([]);
    var R = scan.range;
    var ang = scanAngle;
    var lat = s.lat, lon = s.lon;
    var alt = scan.alt || 80;
    var pts = [Cesium.Cartesian3.fromDegrees(lon, lat, alt)];
    var half = 0.55; // 约 63° 扇区
    for (var i = 0; i <= 18; i++) {
      var a = ang + (-half + 2 * half * i / 18);
      var east = R * Math.sin(a);
      var north = R * Math.cos(a);
      var dlon = east / (111320 * Math.cos(lat * Math.PI / 180));
      var dlat = north / 110540;
      pts.push(Cesium.Cartesian3.fromDegrees(lon + dlon, lat + dlat, alt));
    }
    return new Cesium.PolygonHierarchy(pts);
  }

  function ensureScan() {
    if (scan) return;
    var ent = ctx.viewer.entities.add({
      id: 'combat-scan',
      show: false,
      polygon: {
        hierarchy: new Cesium.CallbackProperty(scanHierarchy, false),
        material: Cesium.Color.fromCssColorString('#4fd8ff').withAlpha(0.20),
        outline: true,
        outlineColor: Cesium.Color.fromCssColorString('#59d7ff').withAlpha(0.75),
        perPositionHeight: true,
      },
      ellipse: {
        semiMajorAxis: 1000,
        semiMinorAxis: 1000,
        fill: false,
        outline: true,
        outlineColor: Cesium.Color.fromCssColorString('#59d7ff').withAlpha(0.55),
        outlineWidth: 2,
        height: 80,
      },
      position: Cesium.Cartesian3.fromDegrees(0, 0, 80),
    });
    scan = { entity: ent, id: null, range: 0 };
  }

  function updateScan(tick) {
    var id = apiSelected();
    if (!id) { if (scan) scan.entity.show = false; return; }
    var def = defById[id];
    var rangeKm = Number(def && def.sensor_range_km) || 0;
    if (!def || rangeKm <= 0) { if (scan) scan.entity.show = false; return; }
    ensureScan();
    if (scan.id !== id) {
      scan.id = id;
      scan.range = Math.min(rangeKm, 60) * 1000;
      scan.entity.ellipse.semiMajorAxis = scan.range;
      scan.entity.ellipse.semiMinorAxis = scan.range;
    }
    var s = currentGeo(id);
    if (!s) { scan.entity.show = false; return; }
    var alt = (s.alt || 0) + (def.domain === 'air' ? 0 : 40);
    scan.entity.position = Cesium.Cartesian3.fromDegrees(s.lon, s.lat, alt);
    scan.entity.show = true;
    scan.alt = alt;
    scan.entity.ellipse.height = alt;
  }

  function debug() {
    return {
      events: (data && data.events || []).length,
      transients: transients.length,
      projectiles: projectiles.length,
      smoke: Object.keys(smoke).length,
      scan: scan ? { id: scan.id, show: scan.entity.show, range: scan.range } : null,
      destroyed: Object.keys(destroyed),
      rig: { z10: rigAngles.z10, hq16: rigAngles.hq16, yj62: rigAngles.yj62 },
      smokeInfo: Object.keys(smoke).map(function (id) {
        var sm = sample(id);
        return { id: id, level: smokeState[id], lon: sm && sm.lon, lat: sm && sm.lat, alt: sm && sm.alt };
      }),
    };
  }

  return {
    init: init,
    frame: frame,
    nodeTransformations: nodeTransformations,
    dispose: dispose,
    debug: debug,
  };
}));
