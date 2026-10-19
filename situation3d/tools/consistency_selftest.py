#!/usr/bin/env python3
"""意图↔效果一致性自检。

为什么需要它：这个项目里最危险的缺陷形态是**图层行在注册表里存在、
`desired` 也能置上，但场景里什么都不发生**。它不报错、`pageErrors` 为空、
案例照样加载 —— 只能靠“翻转意图、再读效果”来抓。

已抓到的一例（2026-10-08）：`wsf3d.overlay` 的图层创建写在 `setWsf3dOverlay()` 里，
而那个函数只有 URL 钩子会调 —— 面板上“建筑高度图层（WSF3D）”那个复选框**一直是空转的**。
修了之后就需要这条自检把它钉住。

用法：python3 tools/consistency_selftest.py [--base http://127.0.0.1:8848] [--case taiwan_landing]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSPECT = os.path.join(ROOT, "tools", "inspect_remote.py")

# `inspect_remote.py` 的 --timeout-ms（等瓦片队列排空）与 --wait-ms（排空后静置）
# 是**相加**的；外层超时必须比两者之和还宽。当初就是这里算错了才把
# `?terrain=1` 误报成“probe 无结果”。
QUEUE_MS = 60000
WAIT_MS = 18000

FAILURES = []


def check(label, ok, detail=""):
    print(("  ok   " if ok else "  FAIL ") + label + (("  " + detail) if detail else ""))
    if not ok:
        FAILURES.append(label + (("  " + detail) if detail else ""))


# 逐对翻转 `desired`，再读**场景/模块自己的状态**。
# 只看 `desired` 是不够的 —— 缺陷恰恰就在“desired 对了、效果没发生”。
PROBE = r"""
(async function () {
  var t0 = Date.now();
  while (Date.now() - t0 < 20000 && !window.SituationHost) {
    await new Promise(function (r) { setTimeout(r, 300); });
  }
  await new Promise(function (r) { setTimeout(r, 7000); });
  // rAF in this headless browser runs at ~2.7 fps (measured ~0.37 s per frame), so
  // frame-count waits are ~20x slower than intended and the probe blew its eval
  // timeout. Time-based waits keep the run inside its budget; the polling loop
  // below still allows ~8 s per row for async effects to appear.
  function frames(n) { return new Promise(function (res) { setTimeout(res, Math.min(n * 150, 600)); }); }
  if (!window.SituationHost || !window.LayerRegistry) { return { fatal: "no host/registry" }; }
  var out = { rows: [], fatal: null, stateLayers: (SituationHost.state.layers || null) };

  // 每个图层的“效果读数”。统一返回**数量**（布尔就用 0/1），
  // 因为断言统一是“先关再开、数量必须上升”。
  //
  // ⚠ **不要写绝对断言**（例如“置开后全部 show === true”）：
  // `state.filters`（红/蓝/海/陆/空）也会独立隐藏实体，绝对断言会**假失败**。
  // 这个代码库自己的注释就写着：自检里的假失败比没有自检更糟。
  function countShown(arr) {
    if (!arr) return 0;
    var list = Array.isArray(arr) ? arr : Object.keys(arr).map(function (k) { return arr[k]; });
    var n = 0;
    list.forEach(function (e) { if (e && e.show) n++; });
    return n;
  }
  function imageryShown(id) {
    var L = LayerRegistry.layersOf(id) || [];
    var n = 0;
    L.forEach(function (x) { if (x && x.show) n++; });
    return n;
  }

  function modPlaces() {
    var s = window.Places ? String(Places.debugInfo()) : "";
    var m = s.match(/enabled\s+(true|false)/);
    // `load()` 未完成时 `debugInfo()` 返回 "not loaded" —— 那时**该图层确实没显示**，
    // 所以返 0 而不是 -1（-1 会被当成“无法取证”直接报失败）；
    // 置开后的轮询会等它真的加载完。
    return m ? (m[1] === "true" ? 1 : 0) : 0;
  }

  // Count scene-side entities as well: that separates "state changed but the
  // scene did not" from "the probe read the previous frame".
  function viewerShown(prefix) {
    var v = window.SituationHost.viewer();
    if (!v || !v.entities || !v.entities.values) return -1;
    var n = 0;
    v.entities.values.forEach(function (e) {
      if (e && e.show && String(e.id || "").indexOf(prefix) === 0) n++;
    });
    return n;
  }
  // `relationsDebug().linesVisible` is the scene-side truth: 0 while the layer is
  // off, and equal to `total` (the logical line count) when the layer is on and a
  // lead is selected. The failure detail prints linesVisible and total together.
  function relationsSnapshot() {
    var H = window.SituationHost;
    if (!H || typeof H.relationsDebug !== 'function') return null;
    var d = H.relationsDebug();
    return { selected: d.selected, linesVisible: d.linesVisible, total: d.total };
  }
  function relationsShown() {
    var H = window.SituationHost;
    if (!H || typeof H.relationsDebug !== 'function') return -1;
    var d = H.relationsDebug();
    if (!d.selected && typeof H.selectTarget === 'function') {
      var defs = (ST.data && ST.data.targets) || [];
      for (var i = 0; i < defs.length; i++) {
        var a = (defs[i] && (defs[i].attributes || defs[i].attrs)) || {};
        if (a.pair_role === 'lead' && (defs[i].id || defs[i].target_id)) {
          H.selectTarget(defs[i].id || defs[i].target_id);
          break;
        }
      }
      d = H.relationsDebug();
    }
    return typeof d.linesVisible === 'number' ? d.linesVisible : -1;
  }
  // Occupation areas expose their entities on the state, so the layer flip is read
  // directly: 0 while the layer is off, one visible polygon per hotspot when on.
  function occupationShown() {
    var list = (ST && ST.occupationEntities) || [];
    var n = 0;
    list.forEach(function (e) { if (e && e.show) n += 1; });
    return n;
  }
  var ST = SituationHost.state;
  // The mask only takes effect while a target is selected, so make sure one is
  // selected before flipping desired; otherwise "visible after turning it on"
  // stays 0, which would be a test-harness false failure.
  function focusMaskShown() {
    var H = window.SituationHost;
    if (!H || typeof H.focusDebug !== "function") return -1;
    var d = H.focusDebug();
    if (!d.selected && typeof H.selectTarget === "function") {
      var tarr = (ST.data && ST.data.targets) || ST.targets || [];
      if (!Array.isArray(tarr)) tarr = Object.keys(tarr).map(function (k) { return tarr[k]; });
      var t0 = tarr.length ? tarr[0] : null;
      if (t0 && (t0.id || t0.target_id)) H.selectTarget(t0.id || t0.target_id);
      d = H.focusDebug();
    }
    return d.maskShow ? 1 : 0;
  }
  // 标绘图层（`kind:'plot'`）的显隐读取口。
  //
  // 为什么用它、而不是数场景里的几何：标绘的 sink 是 `Plotting.setLayerVisible()`
  // （见 `app.js` 的 plot handler），它只写 `layerVisible[layerId]` 再调
  // `applyVisibility()`；而**很多标绘类别默认没有要素**，数几何会得到「开关都=0」
  // 的恒真/恒假断言——那正是本文件注释里警告过的**假失败**。所以这里读的是
  // 「sink 有没有把意图传下去」这一层，这正是注册表↔模块之间最容易断的接缝。
  //
  // 返回 0/1：自检的契约是「关=0、开>0」，布尔转 0/1 正好满足。
  function plotFlagShown(layerId) {
    if (!window.Plotting || typeof window.Plotting.debug !== "function") return -1;
    // ⚠ 用 `debug()`，**不是 `catalogue()`**：实测 `catalogue()` 返回的是
    // **24 条标绘类型定义**（`type`/`zh`/`sourceStandard`… 的数组），**没有 `visible`**
    // ——我第一版就读错成 `catalogue().visible`，于是恒为 undefined→0，
    // 6 条断言全报「开后效果没出现」的**假失败**。`debug()` 才是状态：
    // 返回 `{bound, visible:{layerId:bool}, count, items, resolution, …}`。
    var d = window.Plotting.debug();
    var v = (d && d.visible) || {};
    // 只看 sink 标志会漏掉 applyVisibility 写坏实体 show 的情况：该层至少要有一个
    // 要素，且所有图元真的处于显示状态，才算"开了"。
    var items = (d && d.items ? d.items : []).filter(function (it) {
      return it.layer === layerId;
    });
    var shown = items.length > 0 && items.every(function (it) {
      return it.shown !== false;
    });
    return (v[layerId] && shown) ? 1 : 0;
  }

  // 目标图层族的显隐读取口。
  //
  // **替代原先的 `countTargetsShown()`**：旧实现数 `w.entity.show` 是否真值，
  // 而 `entity.show` 是 Cesium 的 **Boolean Property 对象** —— 对象恒真，
  // 于是它恒等于目标总数（实测 31），**与图层开关无关**：一条恒真的假断言。
  // 现读 handler 的同步产物 `state.layers.<flag>`，随开关真实翻转（实测 0<->31）。
  //
  // 依据：`layerTargets`/`layerLabels`/`layerTracks` 走的是**逐层 handler**
  // （`app.js`：`d.handler = function (id, want) { pair[1](want); }`），
  // handler 里写 `state.layers.*` 并 `markScene()`，再由 `syncEntities()` 落到 `entity.show`。
  // `state.layers.*` 是**同步**写入的，不受帧节流影响，故 read() 的「关=0/开>0」立即可满足。
  function targetLayerShown(flag) {
    var L = (ST.layers || {});
    return L[flag] ? 1 : 0;
  }

  var SPECS = [
    // —— 已有的 4 项（上一轮已验证）——
    { id: "places.labels",   what: "离线地名（模块自己缓存的 enabled）", read: modPlaces },
    { id: "wsf3d.overlay",   what: "建筑高度图层（场景 show 计数）", read: function () { return imageryShown("wsf3d.overlay"); } },
    { id: "tianditu.label",  what: "天地图注记（场景 show 计数）", read: function () { return imageryShown("tianditu.label"); } },
    { id: "imagery.offline", what: "离线底图（两级 show 计数）", read: function () { return imageryShown("imagery.offline"); } },
    // —— 本轮新增：取数口已核实——
    { id: "tianditu.base",   what: "天地图底图（场景 show 计数）", read: function () { return imageryShown("tianditu.base"); } },
    { id: "layerHotspots",   what: "热点区域（实体 show 计数）", read: function () { return countShown(ST.hotspotEntities); } },
    { id: "layerRoutes",     what: "计划航路（实体 show 计数）", entityPrefix: "route-", layerFlag: "routes", budget: 8000, read: function () { return countShown(ST.routeEntities); } },
    { id: "layerOverview",   what: "鸟瞰图（第二个 view 的可见性）", read: function () {
        return (window.Overview && typeof Overview.isVisible === "function") ? (Overview.isVisible() ? 1 : 0) : -1; } },
    { id: "weather.model",   what: "气象模型（场景雾开关）", read: function () {
        var v = SituationHost.viewer(); return v ? (v.scene.fog.enabled ? 1 : 0) : -1; } },
    { id: "overlay.focusMask", what: "选中掩码（focusDebug().maskShow，必要时自动选中）", budget: 12000, read: focusMaskShown },
    { id: "situation.relations", what: "关联关系线（选中长机后的可见折线增量）", budget: 12000, read: relationsShown, snapshot: relationsSnapshot },
    { id: "situation.occupation", what: "占领区域（occupationEntities 可见数）", budget: 8000, read: occupationShown },
    // —— 标绘栈（`kind:'plot'`）：行由下方向**目录里实际存在的类别**生成 ——
    // 不再写死五个类别：新增/改名类别时自检自动跟随，符合"图层由要素驱动、
    // 不用固定分类限制扩展"的要求。
    // —— 目标图层族（此前 layerTargets 因读错对象而被标注「暂不纳入」，见上）——
    { id: "layerTargets", what: "可见目标（state.layers.targets，handler 同步产物）",
      read: function () { return targetLayerShown("targets"); } },
    { id: "layerLabels",  what: "目标标牌（state.layers.labels）",
      read: function () { return targetLayerShown("labels"); } },
    { id: "layerTracks",  what: "历史轨迹（state.layers.tracks）",
      read: function () { return targetLayerShown("tracks"); } },
  ];

  // 标绘图层是**动态**的：没有要素就不会注册。先从目录里取"有手动类型"的类别，
  // 每类补一行 SPECS 并各造一条要素；循环结束后全部删除，恢复现场。
  // 类别列表**从目录推导**，不写死五个——新增类别时自检自动跟上。
  function plotLayerIds() {
    return LayerRegistry.list('plot').filter(function (l) { return l.group === '标绘'; })
      .map(function (l) { return l.id; }).sort();
  }
  var catalogueNow = Plotting.catalogue();
  var seedCats = Plotting.categories().filter(function (c) {
    return catalogueNow.some(function (x) { return !x.system && x.defaultLayer === c.layer; });
  });
  out.seedLayers = seedCats.map(function (c) { return c.layer; });
  seedCats.forEach(function (c) {
    SPECS.push({
      id: c.layer,
      what: '标绘·' + c.zh + '（sink 是否收到意图）',
      read: (function (lid) {
        return function () { return plotFlagShown(lid); };
      })(c.layer),
    });
  });
  out.plotLayersBefore = plotLayerIds();
  var PLOT_SEEDS = [];
  seedCats.forEach(function (c) {
    try {
      var cat = catalogueNow.filter(function (x) {
        return !x.system && x.defaultLayer === c.layer;
      })[0];
      if (!cat) return;
      var pts = (cat.family === 'zone')
        ? [[119.4, 24.2], [119.55, 24.2], [119.55, 24.35], [119.4, 24.35]]
        : [[119.4, 24.2], [120.6, 25.0]];
      PLOT_SEEDS.push(Plotting.add({ type: cat.type, layer: c.layer, points: pts }).id);
    } catch (e) { /* 造不出来就让对应的 plot.* 行按失败报出 */ }
  });
  await frames(5);
  out.plotSeeds = PLOT_SEEDS.length;
  out.plotLayers = plotLayerIds();

  for (var i = 0; i < SPECS.length; i++) {
    var s = SPECS[i], row = { id: s.id, what: s.what };
    row.entityPrefix = s.entityPrefix || null;
    row.layerFlag = s.layerFlag || null;
    row.desired0 = (LayerRegistry.runtime[s.id] || {}).desired;
    // 先**关**再**开**：这样不受初始值影响（有些图层默认就是开的）。
    try { LayerRegistry.setDesired(s.id, false); } catch (e) { row.error = String(e); }
    LayerRegistry.apply();
    await frames(4);
    // §11.3: a fake failure is as much a bug as a fake pass - never paper over it
    // with a longer fixed settle or a re-run. Poll to the expected value, fail only
    // on timeout, and keep the last raw frame for the failure detail.
    var budget = s.budget || 8000;
    row.nOff = s.read();
    row.eOff = s.entityPrefix ? viewerShown(s.entityPrefix) : null;
    for (var _of = 0; _of * 250 < budget &&
         !((row.nOff === 0 || row.nOff === -1) && (row.eOff === null || row.eOff === 0)); _of++) {
      await new Promise(function (r) { setTimeout(r, 250); });
      row.nOff = s.read();
      if (s.entityPrefix) { row.eOff = viewerShown(s.entityPrefix); }
    }
    row.offExtra = s.snapshot ? s.snapshot() : null;
    row.offFrame = {
      desired: (LayerRegistry.runtime[s.id] || {}).desired,
      layerFlag: s.layerFlag ? (ST.layers || {})[s.layerFlag] : null,
      stateShown: row.nOff, entityShown: row.eOff
    };
    // 面板复选框：`layerPanel.js` 里 `box.id = "layerRow-" + status.id`。
    // **这是本自检原本漏掉的一环** —— 意图对了、场景也变了，
    // 但面板上的勾还是旧的，就是“界面与事实不符”。本会话修过好几处这类。
    row.panelOff = null;
    // 面板按帧差分重建；无头 SwiftShader 下 rAF 只有 ~2.7 fps，固定 12×150 ms 会偶发
    // 超时（实测同步 102-320 ms，但帧率抖动时窗口不够）。轮询时**主动调 tick()**
    // ——那是生产环境的刷新函数，只是不再依赖本环境的帧率；上限放宽到 10 s。
    for (var _po = 0; _po < 40 && row.panelOff !== false; _po++) {
      if (window.LayerPanel && LayerPanel.tick) LayerPanel.tick();
      var _bo = document.getElementById("layerRow-" + s.id);
      row.panelOff = _bo ? !!_bo.checked : null;
      if (row.panelOff === false) break;
      await new Promise(function (r) { setTimeout(r, 250); });
    }
    try { LayerRegistry.setDesired(s.id, true); } catch (e) { row.error = String(e); }
    LayerRegistry.apply();
    // 开上后**轮询到出现为止**：有些效果是异步的
    // （例：`places.labels` 走 `Places.load()`，未完成时 `debugInfo()` 返回 "not loaded"）。
    // 固定帧数会把“还没加载完”误报成“效果没出现”。
    // 但**不能无限等** —— eval 有自己的超时，所以设上限。
    row.nOn = s.read();
    row.eOn = s.entityPrefix ? viewerShown(s.entityPrefix) : null;
    for (var _try = 0; _try * 250 < budget &&
         !(row.nOn > 0 && (row.eOn === null || row.eOn > 0)); _try++) {
      await new Promise(function (r) { setTimeout(r, 250); });
      row.nOn = s.read();
      if (s.entityPrefix) { row.eOn = viewerShown(s.entityPrefix); }
    }
    row.onExtra = s.snapshot ? s.snapshot() : null;
    row.onFrame = {
      desired: (LayerRegistry.runtime[s.id] || {}).desired,
      layerFlag: s.layerFlag ? (ST.layers || {})[s.layerFlag] : null,
      stateShown: row.nOn, entityShown: row.eOn
    };
    // 面板读数**必须在轮询之后**：面板是按签名差分重建的，
    // 不会在 `apply()` 返回的同一刻就织好。我一度在 `apply()`
    // 后立即读，结果 **10 个图层 × 2 个案例全部同向失败**。
    // 那种整齐划一的失败就是“测试写错了”的信号，不是 10 个真缺陷 ——
    // 我又用一个只测单个图层、打印原始值的聚焦探针把它排除了。
    // 无条件等几帧：上面的轮询**只在读数为 0/-1 时才跑**，
    // 而 imagery 类的 `.show` 是 sink 同步设的、立即就非零，于是一帧都不过。
    // 面板却是按签名差分在**帧里**重建的 —— 不等就读到旧值。
    if (window.LayerPanel && LayerPanel.tick) LayerPanel.tick();
    row.panelOn = null;
    for (var _pn = 0; _pn < 40 && row.panelOn !== true; _pn++) {
      if (window.LayerPanel && LayerPanel.tick) LayerPanel.tick();
      var _bn = document.getElementById("layerRow-" + s.id);
      row.panelOn = _bn ? !!_bn.checked : null;
      if (row.panelOn === true) break;
      await new Promise(function (r) { setTimeout(r, 250); });
    }
    row.desired1 = (LayerRegistry.runtime[s.id] || {}).desired;
    // 复原初始意图
    try { LayerRegistry.setDesired(s.id, row.desired0); } catch (e) { row.error = String(e); }
    LayerRegistry.apply();
    await frames(3);
    // The mask row selects a target to make the mask visible; clear it again
    // so the reload audit below does not see the extra mask entity.
    if (typeof SituationHost.selectTarget === "function") {
      try { SituationHost.selectTarget(null); } catch (e) { row.error = String(e); }
    }
    out.rows.push(row);
  }
  // 清掉种子标绘：空类别图层应随之注销（before == after）
  PLOT_SEEDS.forEach(function (id) {
    try { Plotting.remove(id); } catch (e) { /* 已被图层翻转清掉 */ }
  });
  await frames(6);
  out.plotLayersAfter = plotLayerIds();
  // —— 互斥争裁与**恢复**（`exclusives` 的运行时行为）——
  // 本会话修过互斥的对称性与默认值，但**从没验过运行时**。
  // `resetHiddenFeed()` 的注释恰好警告过失效模式：关掉上层后下层仍标着 `hidden`，
  // 出现“开关明明开着、状态却显示被顶掉”的假象。
  out.exclusive = (function () {
    function rt(id) { var r = LayerRegistry.runtime[id]; return { d: r.desired, f: r.feed }; }
    function placesOn() {
      var s = window.Places ? String(Places.debugInfo()) : "";
      if (s.indexOf("not loaded") >= 0) return false;
      var m = s.match(/enabled\s+(true|false)/);
      return m ? (m[1] === "true") : null;
    }
    var r = {};
    // 记下原始意图，最后复原
    r.d0Label = rt("tianditu.label").d; r.d0Places = rt("places.labels").d;
    LayerRegistry.setDesired("tianditu.label", false);
    LayerRegistry.setDesired("places.labels", true);
    LayerRegistry.apply();
    r.alone = placesOn();                       // 应 true
    LayerRegistry.setDesired("tianditu.label", true);
    LayerRegistry.apply();
    r.under = { placesFeed: rt("places.labels").f, placesOn: placesOn(),
                labelFeed: rt("tianditu.label").f };   // places 应被顶掉
    LayerRegistry.setDesired("tianditu.label", false);
    LayerRegistry.apply();
    r.recovered = { placesFeed: rt("places.labels").f, placesOn: placesOn() };
    r.placesDesiredKept = rt("places.labels").d;  // 被顶掉期间 desired 不应被改写
    // 复原
    LayerRegistry.setDesired("tianditu.label", r.d0Label);
    LayerRegistry.setDesired("places.labels", r.d0Places);
    LayerRegistry.apply();
    return r;
  })();

  // —— 重复重载下的状态积累（只在重复时才暴露的那类泄漏）——
  // 本会话早期把 `resetElementState()` 从 `clearSituation()` 挑进 `teardownScene()`，
  // 因为切换路径是 `loadCase → applySituationData → teardownScene()`；但当时**只做了单点验证**。
  // 累积型泄漏只在反复重载时现形，所以这里重载两次比对。
  out.reloads = [];
  function cnt() {
    var S = SituationHost.state, v = SituationHost.viewer();
    return {
      caseId: S.caseId,
      targetObjs: Object.keys(S.targets || {}).length,
      hotspotEntities: (S.hotspotEntities || []).length,
      routeEntities: (S.routeEntities || []).length,
      viewerEntities: v.entities.values.length,
      imageryLayers: v.imageryLayers.length,
    };
  }
  out.reloads.push(cnt());
  for (var _r = 0; _r < 2; _r++) {
    try {
      var _pr = window.SituationHost.reload();
      if (_pr && _pr.then) { await _pr; }
    } catch (e) { out.reloadErr = String(e && e.message || e); }
    await new Promise(function (r) { setTimeout(r, 1500); });
    out.reloads.push(cnt());
  }

  return out;
})()
"""


def run_probe(base, case, tmpdir):
    out = os.path.join(tmpdir, "consistency.json")
    url = "%s/?case=%s" % (base, case)
    cmd = [sys.executable, INSPECT, "--url", url, "--out", out,
           "--timeout-ms", str(QUEUE_MS), "--wait-ms", str(WAIT_MS),
           "--eval-async", "--eval", PROBE]
    # ⚠ 这里必须把 **eval 自己的耗时**也算进去。
    # `--timeout-ms`（等瓦片队列排空）与 `--wait-ms`（排空后静置）是**相加**的，
    # 而 eval 是在它们**之后**才跑。扩到 10 个图层后 eval 约 90 s
    # （10 层 × 26 帧，headless 下 ~5fps），而原来只留了 60 s 余量 ——
    # 结果是“两次都拿不到结果”。这与本仓库 `deeplink_selftest.py`
    # 里记载的那个坑是**同一个**（当初 `?terrain=1` 就是这么误报的）。
    budget = (QUEUE_MS + WAIT_MS) / 1000.0 + 180
    try:
        subprocess.run(cmd, cwd=ROOT, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=budget)
    except subprocess.TimeoutExpired:
        return None, ["probe timed out"]
    try:
        with open(out, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError) as exc:
        return None, ["probe output unreadable: %s" % exc]
    return doc.get("eval"), doc.get("pageErrors") or []


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8848")
    # 默认**两个案例都跑**。为什么：两个案例的图层集合与规模不同
    # （台海 31 目标/11 热点；中印 10 目标/7 热点、且**没有城区 AOI**），
    # 只跑一个案例会把“只在某个案例下才坏”的问题漏掉。
    # 而且两个案例的计数**应该不同**（实测：31 vs 10、11 vs 7）——
    # 这也是“读的是真实场景数据、不是常量”的侧面证据。
    ap.add_argument("--case", action="append", default=None,
                    help="可重复；省略时默认跑 taiwan_landing 与 india_china_west")
    args = ap.parse_args(argv)

    cases = args.case or ["taiwan_landing", "india_china_west"]
    tmpdir = tempfile.mkdtemp(prefix="consistency-")
    for ci, case in enumerate(cases):
        print("意图↔效果一致性自检  %s/?case=%s" % (args.base, case))
        print()
        _one(args.base, case, tmpdir, ci)
        print()
    if FAILURES:
        print("FAILED (%d):" % len(FAILURES))
        for f in FAILURES:
            print("  - " + f)
        return 1
    print("ALL CHECKS PASSED")
    return 0


def _one(base, case, tmpdir, ci):
    ev, page_errors = run_probe(base, case, tmpdir)
    if ev is None:
        ev, page_errors = run_probe(base, case, tmpdir)
    if ev is None:
        check("页面能跑起来", False, "两次都拿不到结果（服务未启动？）")
        print()
        print("FAILED (%d):" % len(FAILURES))
        for f in FAILURES:
            print("  - " + f)
        return 1

    if ev.get("fatal"):
        check("页面能跑起来", False, str(ev.get("fatal")))
        return 1
    check("案例正常加载（无启动异常）", not page_errors,
          json.dumps(page_errors[:2], ensure_ascii=False))

    # 互斥争裁与恢复
    ex = ev.get("exclusive")
    if ex:
        check("互斥：单独开 places 时它是可见的", ex.get("alone") is True)
        u = ex.get("under") or {}
        check("互斥：再开 label 后 places 被顶掉（feed=hidden）",
              u.get("placesFeed") == "hidden", "实际 feed=%s" % u.get("placesFeed"))
        check("互斥：被顶掉后 places 真的隐藏了", u.get("placesOn") is False)
        check("互斥：上层 label 自己是好的", u.get("labelFeed") == "ok")
        check("互斥：**关掉上层后下层能恢复**（这正是曾经担心的失效点）",
              (ex.get("recovered") or {}).get("placesFeed") == "ok"
              and (ex.get("recovered") or {}).get("placesOn") is True,
              json.dumps(ex.get("recovered"), ensure_ascii=False))
        check("互斥：被顶掉期间不会改写用户的意图",
              ex.get("placesDesiredKept") is True)

    # 重复重载：计数不应增长
    rl = ev.get("reloads") or []
    if len(rl) >= 2:
        b = rl[0]
        for k in ("targetObjs", "hotspotEntities", "routeEntities", "viewerEntities", "imageryLayers"):
            vals = [x.get(k) for x in rl]
            check("重重载后 %s 不增长" % k, vals[-1] <= vals[0],
                  "各次=%s" % vals)
        check("重重载后案例不变", all(x.get("caseId") == b.get("caseId") for x in rl),
              b.get("caseId"))

    rows = ev.get("rows") or []
    # 动态标绘图层：五个种子类别各有要素后必须出现；删除后回到初始集合。
    seeded = ev.get("plotLayers") or []
    before = ev.get("plotLayersBefore") or []
    after = ev.get("plotLayersAfter") or []
    want = ev.get("seedLayers") or []
    check("标绘图层由实际要素动态创建（目录里每个手动类别都在册）",
          bool(want) and all(x in seeded for x in want)
          and (ev.get("plotSeeds") or 0) == len(want),
          "seed=%s layers=%s n=%r" % (want, seeded, ev.get("plotSeeds")))
    check("标绘要素删除后空类别图层注销（集合回到初始）",
          after == before, "before=%s after=%s" % (before, after))
    if not rows:
        check("取到至少一行取证", False, "rows 为空")
    for row in rows:
        rid = row.get("id")
        # 这条只证“置 true 确实写进了注册表”；真正的断言在下面的数量比较。
        check("%s  desired 真的被置开" % rid, row.get("desired1") is True,
              "初值=%s" % row.get("desired0"))
        n_off, n_on = row.get("nOff"), row.get("nOn")
        if n_off is None or n_on is None or n_off < 0 or n_on < 0:
            check("%s  能取到效果读数" % rid, False,
                  "读数为 None/不可用（图层未创建、模块未就绪）")
            continue
        frame = row.get("offFrame") or {}
        check("%s  off: effect disappears" % rid, n_off == 0,
              "stateShown=%s last=%s" % (n_off, json.dumps(frame, sort_keys=True)))
        p_off, p_on = row.get("panelOff"), row.get("panelOn")
        if p_off is None or p_on is None:
            check("%s  面板里有对应的勾选框" % rid, False,
                  "layerRow-%s 不存在" % rid)
        else:
            check("%s  面板勾选如实反映意图（关）" % rid, p_off is False)
            check("%s  面板勾选如实反映意图（开）" % rid, p_on is True)
        frame_on = row.get("onFrame") or {}
        check("%s  on: effect appears" % rid, n_on > n_off,
              "stateShown %s -> %s last=%s" % (n_off, n_on, json.dumps(frame_on, sort_keys=True)))
        ex = row.get("onExtra")
        if ex is not None:
            check("%s  on: linesVisible == total" % rid,
                  ex.get("linesVisible") is not None and ex.get("linesVisible") == ex.get("total"),
                  "linesVisible=%r total=%r selected=%r off=%s" % (
                      ex.get("linesVisible"), ex.get("total"), ex.get("selected"),
                      json.dumps(row.get("offExtra"), sort_keys=True)))
        e_off, e_on = row.get("eOff"), row.get("eOn")
        if e_off is not None or e_on is not None:
            pre = row.get("entityPrefix")
            check("%s  scene entities hidden (%s*)" % (rid, pre), e_off == 0,
                  "entityShown=%s last=%s" % (e_off, json.dumps(frame, sort_keys=True)))
            check("%s  scene entities shown (%s*)" % (rid, pre), (e_on or 0) > 0,
                  "entityShown=%s last=%s" % (e_on, json.dumps(frame_on, sort_keys=True)))
if __name__ == "__main__":
    raise SystemExit(main())
