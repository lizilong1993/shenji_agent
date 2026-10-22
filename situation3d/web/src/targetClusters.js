/* situation3d — 编队图标（屏幕空间聚合）
 *
 * 目的：同一阵营的目标在屏幕上挤在一起时，逐个画 34 px 图标只会得到一团糊。
 * 这里按**屏幕距离**把同阵营目标聚成一个编队图标，显示阵营帧 + 数量 + 主类别；
 * 相机拉近、图标之间的屏幕距离拉开后自动拆回单目标。
 *
 * 几条硬规则（避免聚合本身变成新的噪声源）：
 *   - **红蓝不混聚**：computeClusters 先按 affiliation 分组，跨阵营永不合并；
 *   - **选中目标不聚合**：用户正在看的目标必须保留独立图标；
 *   - **滞回**：加入阈值 joinPx < 拆散阈值 splitPx。相机在阈值附近轻微移动时，
 *     成员不会一进一出地闪；
 *   - **不串联**：成员到簇心的最大距离受 maxSpreadPx 限制。单链式聚类会把一整
 *     排目标串成一个长条，这里不允许；
 *   - **节流 + 稳定 id**：120 ms 重算一次；新簇优先复用上一轮成员的 id，实体
 *     不重建，图标/标牌只在内容变化时换图。
 *
 * 纯函数 computeClusters() 可以在 Node 里直接测（tools/target_clusters_test.js），
 * Cesium 集成只在下面的 controller 段。
 */

'use strict';

(function (root, factory) {
  var api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.TargetClusters = api;
}(typeof window !== 'undefined' ? window : null, function () {
  'use strict';

  var DEFAULTS = {
    // 目标图标 34 px、编队图标 44 px。成员离簇心 <= 38 px 时，两者图标必然
    // 有重叠（17 + 22 = 39），所以 joinPx 不能小于这个量级。
    joinPx: 38,        // 新成员加入簇的距离阈值（屏幕像素，到簇心）
    splitPx: 46,       // 已有成员离簇心超过这个距离就拆散（滞回）
    maxSpreadPx: 62,   // 成员到簇心的最大距离，防止链式串联
    throttleMs: 120,   // 重算节流
    sizePx: 44,        // 编队图标屏幕尺寸
  };

  function dist(a, b) {
    var dx = a.x - b.x, dy = a.y - b.y;
    return Math.sqrt(dx * dx + dy * dy);
  }

  function centroid(items) {
    var x = 0, y = 0;
    for (var i = 0; i < items.length; i++) { x += items[i].x; y += items[i].y; }
    return { x: x / items.length, y: y / items.length };
  }

  function spread(items, c) {
    var m = 0;
    for (var i = 0; i < items.length; i++) {
      var d = dist(items[i], c);
      if (d > m) m = d;
    }
    return m;
  }

  function overlap(a, b) {
    var n = 0;
    for (var i = 0; i < a.length; i++) if (b.indexOf(a[i]) >= 0) n++;
    return n;
  }

  function makeCluster(affiliation, members, present, id) {
    var items = members.map(function (id) { return present[id]; });
    var c = centroid(items);
    return {
      id: id || null, affiliation: affiliation, members: members.slice(),
      x: c.x, y: c.y, count: members.length,
      _spread: spread(items, c),
    };
  }

  function recompute(cl, present) {
    var items = cl.members.map(function (id) { return present[id]; });
    var c = centroid(items);
    cl.x = c.x; cl.y = c.y; cl.count = cl.members.length;
    cl._spread = spread(items, c);
  }

  /**
   * Pure clusterer.
   *
   * @param {Array<{id:string, affiliation:string, x:number, y:number}>} items
   *        visible targets with their **current window coordinates**.
   * @param {Array<{id:string, affiliation:string, members:string[]}>} prev
   *        clusters from the previous pass (used for hysteresis + stable ids).
   * @param {object} [options]  joinPx / splitPx / maxSpreadPx.
   * @returns {Array<{id, affiliation, members, x, y, count}>} clusters of >= 2.
   */
  function computeClusters(items, prev, options) {
    var o = {};
    Object.keys(DEFAULTS).forEach(function (k) { o[k] = DEFAULTS[k]; });
    if (options) Object.keys(options).forEach(function (k) { o[k] = options[k]; });
    prev = prev || [];

    var byAff = {};
    items.forEach(function (it) {
      if (!it || !it.id || !it.affiliation) return;
      (byAff[it.affiliation] = byAff[it.affiliation] || []).push(it);
    });

    var out = [];
    Object.keys(byAff).forEach(function (aff) {
      var arr = byAff[aff].slice().sort(function (a, b) { return a.id < b.id ? -1 : 1; });
      var present = {};
      arr.forEach(function (it) { present[it.id] = it; });

      var clusters = [];
      var assigned = {};

      // 1) 上一轮成员优先：只受 splitPx 约束（滞回）。
      prev.forEach(function (pc) {
        if (pc.affiliation !== aff) return;
        var members = (pc.members || []).filter(function (id) { return present[id]; });
        if (members.length < 2) return;
        var cl = makeCluster(aff, members, present, pc.id);
        if (cl._spread <= o.splitPx) {
          clusters.push(cl);
          members.forEach(function (id) { assigned[id] = cl; });
        }
      });

      // 2) 其余目标贪心加入最近的簇；受 joinPx 与 spread 双重约束。
      arr.forEach(function (it) {
        if (assigned[it.id]) return;
        var best = null, bestD = Infinity;
        clusters.forEach(function (cl) {
          var d = dist(it, cl);
          if (d > o.joinPx || d >= bestD) return;
          var all = cl.members.concat([it.id]);
          var c = centroid(all.map(function (id) { return present[id]; }));
          if (spread(all.map(function (id) { return present[id]; }), c) > o.maxSpreadPx) return;
          best = cl; bestD = d;
        });
        if (best) {
          best.members.push(it.id);
          recompute(best, present);
          assigned[it.id] = best;
        } else {
          var cl = makeCluster(aff, [it.id], present, null);
          clusters.push(cl);
          assigned[it.id] = cl;
        }
      });

      // 3) 合并靠得很近的簇（只合并已经成形的簇；单个目标之间仍按 joinPx 判）。
      //    阈值取编队图标直径：两个簇心相距小于 44 px 时，两个编队图标本身就会
      //    叠在一起，必须合掉。
      var mergePx = Math.max(o.joinPx, DEFAULTS.sizePx);
      var merged = true, guard = 0;
      while (merged && guard++ < 16) {
        merged = false;
        for (var i = 0; i < clusters.length; i++) {
          if (clusters[i].members.length < 2) continue;
          for (var j = i + 1; j < clusters.length; j++) {
            if (clusters[j].members.length < 2) continue;
            if (dist(clusters[i], clusters[j]) > mergePx) continue;
            var all = clusters[i].members.concat(clusters[j].members);
            var its = all.map(function (id) { return present[id]; });
            var c = centroid(its);
            if (spread(its, c) > o.maxSpreadPx) continue;
            clusters[i].members = all;
            recompute(clusters[i], present);
            clusters[j].members = [];
            merged = true;
          }
        }
      }

      clusters = clusters.filter(function (cl) { return cl.members.length >= 2; });

      // 4) 稳定 id：与上一轮成员重合度 >= 50% 就沿用上一轮的 id。
      var used = {};
      clusters.forEach(function (cl) {
        var bestId = null, bestScore = 0;
        prev.forEach(function (pc) {
          if (pc.affiliation !== aff || used[pc.id]) return;
          var s = overlap(cl.members, pc.members || []) /
                  Math.max(1, Math.min(cl.members.length, (pc.members || []).length));
          if (s > bestScore) { bestScore = s; bestId = pc.id; }
        });
        if (bestScore >= 0.5 && bestId) cl.id = bestId;
        if (!cl.id || used[cl.id]) {
          cl.id = aff + ':' + cl.members.slice().sort()[0];
        }
        // 极端情况下两个簇取到同一个 leader：追加序号保证唯一。
        var base = cl.id, n = 2;
        while (used[cl.id]) { cl.id = base + '#' + (n++); }
        used[cl.id] = true;
      });

      out = out.concat(clusters.map(function (cl) {
        return {
          id: cl.id, affiliation: cl.affiliation, members: cl.members,
          x: cl.x, y: cl.y, count: cl.count, spread: cl._spread,
        };
      }));
    });
    return out;
  }

  // ------------------------------------------------------------------------
  // 屏幕空间避让（红蓝编队图标不能互相压住）
  // ------------------------------------------------------------------------

  /**
   * Push overlapping clusters of **different affiliations** apart in screen space.
   *
   * 红蓝不能合并成一个图标（语义上必须分开），但两个编队图标在接触区可以完全
   * 叠在一起，后画的那个会把另一个的数量盖住。这里只改动图标的屏幕偏移
   * （pixelOffset），不改变编队的地理锚点；每个簇最多偏移 maxOffset px。
   * 同阵营的簇已经在 clusterer 里按图标直径合并过，不再处理。
   */
  function separateClusters(list, minDist, maxOffset) {
    var maxOff = maxOffset || 30;
    list.forEach(function (cl) {
      cl.ox = cl.x; cl.oy = cl.y; cl.dx = 0; cl.dy = 0;
    });
    for (var iter = 0; iter < 12; iter++) {
      var moved = false;
      for (var i = 0; i < list.length; i++) {
        for (var j = i + 1; j < list.length; j++) {
          var a = list[i], b = list[j];
          if (a.affiliation === b.affiliation) continue;
          var dx = b.x - a.x, dy = b.y - a.y;
          var d = Math.sqrt(dx * dx + dy * dy);
          if (d >= minDist) continue;
          if (d < 1e-3) { dx = (a.id < b.id ? -1 : 1); dy = 0; d = 1; }
          var push = (minDist - d) / 2;
          a.dx -= dx / d * push; a.dy -= dy / d * push;
          b.dx += dx / d * push; b.dy += dy / d * push;
          [a, b].forEach(function (cl) {
            cl.dx = Math.max(-maxOff, Math.min(maxOff, cl.dx));
            cl.dy = Math.max(-maxOff, Math.min(maxOff, cl.dy));
            cl.x = cl.ox + cl.dx;
            cl.y = cl.oy + cl.dy;
          });
          moved = true;
        }
      }
      if (!moved) break;
    }
    return list;
  }

  // ------------------------------------------------------------------------
  // Cesium controller
  // ------------------------------------------------------------------------

  var viewer = null;
  var cfg = null;
  var clusters = [];
  var byTarget = {};            // target id -> cluster id
  var entities = {};            // cluster id -> Cesium.Entity
  var lastMs = 0;
  var lastSig = '';
  var lastUpdated = 0;
  var anchorCache = {};         // cluster id -> anchor key
  var stats = { passes: 0, items: 0, offscreen: 0, noPosition: 0, lastError: null };
  function now() {
    return (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
  }

  /** Cesium 会把 boolean 包成 ConstantProperty：读实体开关时必须取值。 */
  function shownOf(v) {
    return (v && typeof v.getValue === 'function') ? v.getValue() : v;
  }

  function init(v, options) {
    viewer = v;
    cfg = options || {};
    reset();
  }

  function reset() {
    Object.keys(entities).forEach(function (k) {
      try { viewer.entities.remove(entities[k]); } catch (e) { /* 已移除 */ }
    });
    entities = {};
    clusters = [];
    byTarget = {};
    anchorCache = {};
    lastSig = '';
    lastMs = 0;
  }

  function membersById() {
    return cfg && cfg.getTargets ? cfg.getTargets() : {};
  }

  function makeEntity(cid) {
    var ent = viewer.entities.add({
      id: 'cluster-' + cid,
      name: '编队 ' + cid,
      position: Cesium.Cartesian3.fromDegrees(0, 0, 0),
      billboard: {
        image: undefined,
        width: DEFAULTS.sizePx,
        height: DEFAULTS.sizePx,
        verticalOrigin: Cesium.VerticalOrigin.CENTER,
        pixelOffset: new Cesium.Cartesian2(0, 0),
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
      label: {
        text: '',
        font: '12px "Microsoft YaHei", "Noto Sans CJK SC", sans-serif',
        fillColor: Cesium.Color.fromCssColorString('#1d2430'),
        outlineColor: Cesium.Color.WHITE,
        outlineWidth: 1.5,
        style: Cesium.LabelStyle.FILL_AND_OUTLINE,
        pixelOffset: new Cesium.Cartesian2(0, -30),
        verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
        showBackground: true,
        backgroundColor: Cesium.Color.fromCssColorString('rgba(255,255,255,0.86)'),
      },
    });
    entities[cid] = ent;
    return ent;
  }

  /** Category composition of one cluster: {list, dominant, mixed, domain}. */
  function composition(members) {
    var targets = membersById();
    var counts = {};
    var domains = {};
    members.forEach(function (id) {
      var rec = targets[id];
      if (!rec || !rec.def) return;
      var cat = (window.Symbology && Symbology.categoryOf)
        ? Symbology.categoryOf(rec.def.icon, rec.def.domain) : 'unknown';
      counts[cat] = (counts[cat] || 0) + 1;
      if (rec.def.domain) domains[rec.def.domain] = true;
    });
    var list = Object.keys(counts).map(function (k) { return { key: k, count: counts[k] }; });
    list.sort(function (a, b) { return (b.count - a.count) || (a.key < b.key ? -1 : 1); });
    list.forEach(function (c) {
      c.label = (window.Symbology && Symbology.categoryLabel)
        ? Symbology.categoryLabel(c.key) : c.key;
    });
    var domainKeys = Object.keys(domains);
    return {
      list: list, dominant: list.length ? list[0].key : 'unknown',
      mixed: list.length > 1,
      // 单一域用该域的帧形（与单目标军标一致）；混域用中性阵营帧。
      domain: domainKeys.length === 1 ? domainKeys[0] : null,
    };
  }

  function labelText(cluster, comp) {
    var affLabel = (window.Symbology && Symbology.affiliationOf)
      ? Symbology.affiliationOf(cluster.affiliation).label : cluster.affiliation;
    var head = affLabel + '×' + cluster.count;
    if (!comp.list.length) return head;
    if (!comp.mixed) return head + ' · ' + comp.list[0].label;
    var parts = comp.list.slice(0, 2).map(function (c) {
      return c.label + '×' + c.count;
    });
    return head + ' · ' + parts.join(' ') + (comp.list.length > 2 ? ' 等' : '');
  }

  function update(force) {
    if (!viewer || !cfg) return null;
    var t = now();
    if (!force && t - lastMs < DEFAULTS.throttleMs) return null;
    lastMs = t;

    var targets = membersById();
    var selected = cfg.getSelected ? cfg.getSelected() : null;
    var scene = viewer.scene;
    var canvas = scene.canvas;
    // worldToWindowCoordinates 的返回值是 **canvas 相对坐标**（已实测：
    // 相机正下方的地面点投到 canvas 中心 476,333），因此与 clientWidth/Height 比较。
    var cw = canvas.clientWidth, ch = canvas.clientHeight;
    var items = [];
    var offscreen = 0, noPosition = 0;
    var time = viewer.clock.currentTime;

    Object.keys(targets).forEach(function (id) {
      var rec = targets[id];
      if (!rec || !rec.entity || !shownOf(rec.entity.show)) return;
      if (id === selected) return;            // 选中目标保持独立图标
      var pos = rec.entity.position && rec.entity.position.getValue
        ? rec.entity.position.getValue(time) : null;
      if (!pos) { noPosition++; return; }
      var w = Cesium.SceneTransforms.worldToWindowCoordinates(scene, pos);
      if (!w || !isFinite(w.x) || !isFinite(w.y)) { noPosition++; return; }
      // 视窗外不参与聚合（它们本来也看不见）。
      if (w.x < -40 || w.y < -40 || w.x > cw + 40 || w.y > ch + 40) {
        offscreen++;
        return;
      }
      var carto = Cesium.Cartographic.fromCartesian(pos);
      items.push({
        id: id, affiliation: rec.def.affiliation, x: w.x, y: w.y,
        lon: Cesium.Math.toDegrees(carto.longitude),
        lat: Cesium.Math.toDegrees(carto.latitude),
        alt: carto.height,
      });
    });

    var next = computeClusters(items, clusters, DEFAULTS);

    // 位置缓存：成员的地理位置随帧变化，聚合每 120 ms 才重算一次。
    var geo = {};
    items.forEach(function (it) { geo[it.id] = it; });
    next.forEach(function (cl) {
      var s = { lon: 0, lat: 0, alt: 0, n: 0, maxAlt: 0 };
      cl.members.forEach(function (id) {
        var g = geo[id];
        if (!g) return;
        s.lon += g.lon; s.lat += g.lat; s.alt += g.alt; s.n++;
        if (g.alt > s.maxAlt) s.maxAlt = g.alt;
      });
      cl.anchor = s.n ? {
        lon: s.lon / s.n, lat: s.lat / s.n,
        alt: s.maxAlt > 200 ? s.maxAlt : s.alt / s.n,
      } : null;
      // 成员到簇心的最大水平距离（米），点击展开时用它算合适的相机高度。
      var maxR = 0;
      if (cl.anchor) {
        cl.members.forEach(function (id) {
          var g = geo[id];
          if (!g) return;
          var dx = (g.lon - cl.anchor.lon) * 111320 *
                   Math.cos(cl.anchor.lat * Math.PI / 180);
          var dy = (g.lat - cl.anchor.lat) * 110570;
          var d = Math.sqrt(dx * dx + dy * dy);
          if (d > maxR) maxR = d;
        });
      }
      cl.radiusM = maxR;
    });

    // 图标画在 anchor 的投影处（不是成员屏幕坐标的均值）：成员高度差大时两者
    // 可能相差几十像素。碰撞避让、点击命中都以这个位置为准。
    next.forEach(function (cl) {
      var screen = null;
      if (cl.anchor) {
        var ap = Cesium.Cartesian3.fromDegrees(cl.anchor.lon, cl.anchor.lat, cl.anchor.alt);
        screen = Cesium.SceneTransforms.worldToWindowCoordinates(scene, ap);
      }
      cl.x = (screen && isFinite(screen.x)) ? screen.x : cl.x;
      cl.y = (screen && isFinite(screen.y)) ? screen.y : cl.y;
    });
    separateClusters(next, DEFAULTS.sizePx + 2, 30);

    var newByTarget = {};
    next.forEach(function (cl) {
      cl.members.forEach(function (id) { newByTarget[id] = cl.id; });
    });

    var sig = next.map(function (cl) {
      return cl.id + ':' + cl.members.slice().sort().join(',');
    }).join('|') + '#' + Object.keys(newByTarget).sort().join(',');
    var labelsOn = cfg.labelsEnabled ? !!cfg.labelsEnabled() : true;

    var live = {};
    next.forEach(function (cl) {
      live[cl.id] = true;
      var ent = entities[cl.id] || makeEntity(cl.id);
      var comp = composition(cl.members);
      var iconKey = cl.affiliation + '|' + cl.count + '|' + comp.dominant + '|' +
                    (comp.mixed ? 1 : 0) + '|' + (comp.domain || 'any');
      if (ent._iconKey !== iconKey) {
        ent._iconKey = iconKey;
        ent.billboard.image = (window.Symbology && Symbology.clusterUri)
          ? Symbology.clusterUri({
              affiliation: cl.affiliation, count: cl.count,
              icon: comp.dominant, mixed: comp.mixed, domain: comp.domain,
            })
          : undefined;
      }
      var text = labelText(cl, comp);
      if (ent._labelText !== text) {
        ent._labelText = text;
        ent.label.text = text;
      }
      ent.label.show = labelsOn &&
        !(cfg.labelHidden && cfg.labelHidden(ent.label));
      var a = cl.anchor;
      if (a) {
        var key = a.lon.toFixed(6) + ',' + a.lat.toFixed(6) + ',' + a.alt.toFixed(1);
        if (anchorCache[cl.id] !== key) {
          anchorCache[cl.id] = key;
          ent.position = Cesium.Cartesian3.fromDegrees(a.lon, a.lat, a.alt);
        }
      }
      // 红蓝编队图标的屏幕避让量（见 separateClusters）。图标与标牌同步偏移，
      // 标牌仍在其图标上方 30 px。
      var offX = Math.round(cl.dx || 0), offY = Math.round(cl.dy || 0);
      var offKey = offX + ',' + offY;
      if (ent._offKey !== offKey) {
        ent._offKey = offKey;
        ent.billboard.pixelOffset = new Cesium.Cartesian2(offX, offY);
        ent.label.pixelOffset = new Cesium.Cartesian2(offX, offY - 30);
      }
      ent.show = true;
      cl._comp = comp;
    });

    Object.keys(entities).forEach(function (cid) {
      if (live[cid]) return;
      entities[cid].show = false;
      delete anchorCache[cid];
    });

    byTarget = newByTarget;
    clusters = next;
    lastUpdated = Date.now();
    stats.passes++;
    stats.items = items.length;
    stats.offscreen = offscreen;
    stats.noPosition = noPosition;
    stats.viewport = [cw, ch];

    if (sig !== lastSig) {
      lastSig = sig;
      if (cfg.onUpdated) cfg.onUpdated();
    }
    return clusters;
  }

  function frame(nowMs) {
    // 聚合只是显示增强：任何异常都不能让主渲染循环停摆。
    try {
      update(false);
    } catch (e) {
      stats.lastError = String((e && e.message) || e);
      if (window.console && console.warn) console.warn('[clusters]', e);
    }
  }

  function isClustered(id) {
    return !!byTarget[id];
  }

  function clusterOf(id) {
    return byTarget[id] || null;
  }

  function zoomTo(clusterId) {
    var cl = null;
    clusters.forEach(function (c) { if (c.id === clusterId) cl = c; });
    if (!cl || !cl.anchor) return false;
    var maxR = cl.radiusM || 0;
    // 让编队的直径大约占视口高度的 1/3：地面跨幅 ≈ 3×直径，相机高度 ≈ 跨幅。
    // 空中编队还要保证相机在编队**上方**（否则点完以后目标跑到视锥外）。
    var h = Math.max(1000, (cl.anchor.alt || 0) + 1200, maxR * 2 * 2.4);
    if (cfg.onZoom) cfg.onZoom(cl.anchor.lon, cl.anchor.lat, h, cl.id);
    return true;
  }

  function zoomToEntity(entityId) {
    var id = String(entityId || '');
    if (id.indexOf('cluster-') !== 0) return false;
    return zoomTo(id.slice('cluster-'.length));
  }

  /**
   * 点击命中：返回覆盖该屏幕点的编队 id（没有则 null）。
   *
   * 为什么要单独算一遍：`scene.pick` 返回的是**最上层**图元，编队图标旁边
   * 还可能压着单目标图标/标牌，pick 未必拿到编队。点击入口先问这里，保证
   * 点在编队图标上时总是展开编队。
   */
  function clusterAt(pos) {
    if (!pos) return null;
    var best = null, bestD = Infinity;
    clusters.forEach(function (cl) {
      var dx = pos.x - cl.x, dy = pos.y - cl.y;
      var d = Math.sqrt(dx * dx + dy * dy);
      if (d <= DEFAULTS.sizePx * 0.6 && d < bestD) { best = cl.id; bestD = d; }
    });
    return best;
  }

  /** 可见编队标牌的屏幕位置，供 declutterLabels 做统一避让。 */
  function labelItems() {
    if (!viewer) return [];
    var out = [];
    var time = viewer.clock.currentTime;
    clusters.forEach(function (cl) {
      var ent = entities[cl.id];
      if (!ent || !shownOf(ent.show) || !ent.label) return;
      var pos = ent.position && ent.position.getValue ? ent.position.getValue(time) : null;
      if (!pos) return;
      var w = Cesium.SceneTransforms.worldToWindowCoordinates(viewer.scene, pos);
      if (!w || !isFinite(w.x)) return;
      out.push({
        label: ent.label, text: ent._labelText || '', size: 12,
        pri: 2,
        w: { x: w.x + (cl.dx || 0), y: w.y + (cl.dy || 0) },
        offY: -30, vc: false,
      });
    });
    return out;
  }

  function debug() {
    return {
      enabled: !!viewer,
      updated: lastUpdated,
      clusters: clusters.map(function (c) {
        return {
          id: c.id, affiliation: c.affiliation, count: c.count,
          members: c.members.slice(), spread: Math.round(c.spread),
          screen: [Math.round(c.x), Math.round(c.y)],
          offset: [Math.round(c.dx || 0), Math.round(c.dy || 0)],
          categories: c._comp ? c._comp.list : [],
        };
      }),
      hiddenTargets: Object.keys(byTarget).length,
      entityCount: Object.keys(entities).length,
      stats: stats,
      thresholds: {
        joinPx: DEFAULTS.joinPx, splitPx: DEFAULTS.splitPx,
        maxSpreadPx: DEFAULTS.maxSpreadPx, throttleMs: DEFAULTS.throttleMs,
      },
    };
  }

  return {
    computeClusters: computeClusters,
    separateClusters: separateClusters,
    DEFAULTS: DEFAULTS,
    init: init,
    reset: reset,
    frame: frame,
    update: update,
    isClustered: isClustered,
    clusterOf: clusterOf,
    zoomTo: zoomTo,
    zoomToEntity: zoomToEntity,
    clusterAt: clusterAt,
    labelItems: labelItems,
    debug: debug,
  };
}));
