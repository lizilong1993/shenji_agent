/**
 * 数据健康：把"每个图层到底有没有真的拿到数据"汇总到一处。
 *
 * 为什么需要
 * ----------
 * 出问题时的现象永远是同一句话："**开关是开的，但画面上什么都没有**"。而这句话
 * 对应至少六七种完全不同的原因，散在六七个模块里：
 *
 *   - 令牌没配（在线实景）        → buildings3d.js 的 phase = unconfigured
 *   - 网络不通（Google 瓦片）      → phase = failed，且**没有异常**
 *   - 互斥被顶掉（另一套底图开着）  → layerRegistry 的 feed = hidden
 *   - 数据没到（地名 / 体块还在算）  → feed = pending
 *   - 色带文件缺失（体块）          → 加载失败但开关还勾着
 *   - 瓦片 404 被静默吞掉           → imageryStats.errors 涨、画面缺块
 *   - 相机不在覆盖范围内            → 候选集为空，代码"正常"跑完
 *
 * 光看画面区分不出这些，只能一个个翻模块。这个模块把它们收成一个视图，并且
 * **区分"意图"与"事实"**：开关是意图，feed 是事实。两者不一致的地方就是问题所在。
 *
 * 依赖：LayerRegistry（图层状态）、OfflineAssets（瓦片统计）、以及可选的
 * Buildings3D / RenderGovernor / Places / SituationHost。
 */
(function () {
  'use strict';

  function safe(fn, fallback) {
    try { return fn(); } catch (e) { return fallback; }
  }

  /** 汇总瓦片/影像请求统计。 */
  function tileStats() {
    var out = { sources: [], totalRequests: 0, totalErrors: 0 };
    // 统计对象挂在 app.js 的闭包 state 上，只能经 SituationHost 取。
    // （不要去找 OfflineAssets.imageryStats——那个并不存在，本轮踩过这个坑。）
    var s = safe(function () { return window.SituationHost.imageryStats(); }, null);
    if (s) {
      out.sources.push({
        label: s.label, requests: s.requests, errors: s.errors,
        minLevel: s.minLevel, maxLevel: s.maxLevel, lastError: s.lastError,
      });
      out.totalRequests += (s.requests || 0);
      out.totalErrors += (s.errors || 0);
    }
    var u = safe(function () { return window.SituationHost.urbanInfo(); }, null);
    if (u) out.urban = u;
    return out;
  }

  /**
   * 图层级健康。每条包含：意图（desired）、事实（effective/feed）、以及
   * 由谁负责提供数据。feed 与 desired 不一致的会被标出来。
   */
  function layers() {
    var status = safe(function () { return window.LayerRegistry.status(); }, []) || [];
    return status.map(function (s) {
      // 一致时 feed 保持模块自报的值；不一致时以互斥/失败为准说明原因。
      return {
        id: s.id, label: s.label, group: s.group, kind: s.kind,
        desired: s.desired, effective: s.effective,
        feed: s.feed, detail: s.feedDetail,
        // 需要用户注意：想开但没生效
        attention: !!s.desired && !s.effective,
      };
    });
  }

  function online() {
    return safe(function () { return window.Buildings3D.debug(); }, null);
  }

  function governor() {
    return safe(function () { return window.RenderGovernor.debug(); }, null);
  }

  function places() {
    var d = safe(function () { return window.Places.debugPlaces(); }, null);
    if (!d) return null;
    var st = d.lastStats || {};
    return {
      loaded: d.loaded, enabled: d.enabled, total: d.total,
      placed: d.lastPlaced, candidates: st.candidates,
      overlapped: st.overlapped, maxRank: st.maxRank,
      camHeightKm: d.camHeightKm,
      // 一个都没放下且候选集非空 = 布局被挤没了，值得提醒
      starved: !!d.enabled && d.loaded && d.lastPlaced === 0 && (st.candidates || 0) > 0,
    };
  }

  function buildings() {
    return safe(function () { return window.SituationHost.debugBuildings(); }, null);
  }

  /** 相机与场景基本情况：很多"没东西"其实是相机不在数据范围内。 */
  function viewport() {
    var v = safe(function () { return window.SituationHost.viewer(); }, null);
    if (!v) return null;
    var c = v.camera.positionCartographic;
    return {
      lon: +(Cesium.Math.toDegrees(c.longitude)).toFixed(4),
      lat: +(Cesium.Math.toDegrees(c.latitude)).toFixed(4),
      heightKm: +(c.height / 1000).toFixed(2),
      imageryLayers: v.imageryLayers ? v.imageryLayers.length : null,
      globeShow: v.scene.globe.show,
      depthTestAgainstTerrain: v.scene.globe.depthTestAgainstTerrain,
      terrainReady: v.terrainProvider ? !!v.terrainProvider.ready : null,
    };
  }

  /**
   * 把上述各块汇总成一份报告，并给出需要用户注意的条目。
   *
   * `attention` 是这份报告的重点：只在"用户以为开着、实际没显示"时列出，
   * 正常的层不打扰。
   */
  function report() {
    var L = layers();
    var on = online();
    var att = [];

    // 在线实景与离机体块互斥：如果真的失败并已回落到本机，那条"被顶掉"是**预期
    // 行为**（用户要的"自动回落"），不是问题，不该报成待查。
    // 反之，若在线层没失败、用户却手动开了本机体块把它顶掉，那就是真的冲突
    // （开关开着却没生效），要报出来。
    var onlineFailed = !!on && (on.phase === 'failed' || on.phase === 'unconfigured');

    L.forEach(function (l) {
      if (!l.attention) return;
      if (l.id === 'tiles.google3d' && l.feed === 'hidden' && onlineFailed) return;
      att.push({ id: l.id, label: l.label, feed: l.feed, detail: l.detail });
    });

    if (on && on.phase === 'failed') {
      // 只有"没回落到本机"时才算待查：回落到体块之后，这条降级为提示。
      var fellBack = L.some(function (l) { return l.id === 'buildings.blocks' && l.effective; });
      if (!fellBack) {
        att.push({ id: 'tiles.google3d', label: '在线三维实景', feed: 'failed',
                   detail: on.detail });
      }
    }
    var b = buildings();
    if (b && b.checked && !b.hasPrimitive && !b.pendingPrimitive && !b.buildingBusy
        && b.rampLoaded) {
      att.push({ id: 'buildings.blocks', label: '建筑体块', feed: 'pending',
                 detail: '开关已开、色带已加载，但还没有生成体块（相机可能不在城区 AOI 内）' });
    }
    var p = places();
    if (p && p.starved) {
      att.push({ id: 'places.labels', label: '地名', feed: 'pending',
                 detail: '候选 ' + p.candidates + ' 个但一个都没放下（屏幕空间不足）' });
    }
    var t = tileStats();
    if (t.totalErrors > 0 && t.totalRequests > 0) {
      var rate = t.totalErrors / t.totalRequests;
      if (rate > 0.02) {
        att.push({ id: 'imagery', label: '影像瓦片', feed: 'failed',
                   detail: '错误率 ' + (rate * 100).toFixed(1) + '%（'
                     + t.totalErrors + '/' + t.totalRequests + '）'
                     + (t.sources[0] && t.sources[0].lastError
                        ? '：' + t.sources[0].lastError : '') });
      }
    }
    return {
      at: new Date().toISOString(),
      attention: att,
      healthy: att.length === 0,
      layers: L,
      tiles: t,
      online: on,
      places: p,
      buildings: b,
      governor: governor(),
      viewport: viewport(),
    };
  }

  /** 紧凑的一行摘要，给面板标题用。 */
  function summary() {
    var r = report();
    if (r.healthy) return '全部正常';
    return r.attention.length + ' 项需要处理';
  }

  window.DataHealth = {
    report: report,
    summary: summary,
    layers: layers,
    tileStats: tileStats,
    viewport: viewport,
  };
})();