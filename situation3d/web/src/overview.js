/* situation3d — 全局鸟瞰（二维态势视图）
 *
 * 定位
 * ----
 * 它是**整个战场的二维态势图**：加载案例后按全部目标的实际航迹范围自动取景，
 * 显示红蓝双方全部参战目标，并用与「整体TS」同一套战损状态规范（见
 * targetStatus.js）表达每个目标当前还能不能打。
 *
 * 三个关键行为（2026-10-21 第二轮反馈）
 * -------------------------------------
 * 1. **未加载案例时是世界视图**，不预设台海；加载后 `setData()` 才按案例取景。
 * 2. **密集目标聚合**：与主图共用 `web/src/targetClusters.js` 的屏幕空间聚类，
 *    同一阵营目标合并成一个编队图标（阵营帧 + 数量 + 主类别），红蓝不混聚；
 *    选中目标不参与聚合，保证当前目标始终可见。
 * 3. 原来的"主视窗黄色视野框"已移除：它是旧缩略图时代用来表示主相机范围的，
 *    在全局态势图里既遮挡目标又容易被误读成战场边界。
 *
 * 仍然保留的设计（都有实测原因，不要随手改）
 * ------------------------------------------
 * 1. **SCENE2D，不用 3D**：鸟瞰图本来就是俯视；而且第二个 WebGL 上下文在无头
 *    SwiftShader 下用 3D 会被彻底饿死（实测画布全黑），2D 则稳定。
 * 2. **必须用 `baseLayer`**：Cesium 1.107 起 `Viewer` 的 `imageryProvider` 选项被
 *    移除且静默忽略，不传 baseLayer 会回落到 Ion 在线底图（离线环境直接报错）。
 * 3. **不共用主视图的 provider 实例**：主视图的 provider 被 TileCache 与
 *    instrument 各包了一层，共用会让鸟瞰图的请求与 404 计入主视图的
 *    `imageryStats`，`?diag=1` 全部失真。
 * 4. **不用 `requestRenderMode`**：无头 SwiftShader 下会停在黑屏。
 * 5. **隐藏/显示要走 resize**：`display:none` 让容器尺寸归零，而 Cesium 只监听
 *    window.resize，重新显示不会自愈；show() 必须 resize 后重新取景。
 */
'use strict';

(function (global) {
  // 未加载案例时的中立视图：整个世界，不暗示任何具体战区。
  const DEFAULT_EXTENT = { w: -179.9, s: -70, e: 179.9, n: 80 };

  // 目标图标 / 血条 / 损毁遮罩的像素尺寸与偏移。血条**紧贴图标底边**：
  // 图标底边与血条顶边都落在锚点上，二者之间没有空隙。
  const ICON_PX = 24;
  const MARK_PX = 24;               // 损毁暗色底与图标同大，不压住自己的血条
  const BAR_W = 24;
  const BAR_H = 4;
  const ICON_OFFSET_Y = 0;          // 图标底边距锚点
  const ICON_CENTER_Y = ICON_OFFSET_Y - ICON_PX / 2;
  const BAR_OFFSET_Y = -2;          // 让血条顶到军标可见底边（图标资产底部有 ~2px 透明留白）

  // 聚合阈值：与主图同一套屏幕空间聚类（见 web/src/targetClusters.js）。
  // 鸟瞰图的单目标图标只有 24 px，所以阈值比主图小一档；编队图标 34 px。
  const CLUSTER_JOIN_PX = 34;       // 新成员加入簇的距离（到簇心）
  const CLUSTER_SPLIT_PX = 42;      // 已有成员离簇心超过该值拆散（滞回）
  const CLUSTER_SPREAD_PX = 56;     // 簇内最大半径，防止链式串联
  const CLUSTER_ICON_PX = 34;
  const CLUSTER_BAR_W = 30;
  const CLUSTER_MAX_OFFSET_PX = 26; // 红蓝编队图标的屏幕避让上限

  let viewer = null;          // 鸟瞰图 Viewer
  let mainViewer = null;      // 主 3D Viewer
  let created = false;

  let data = null;            // 当前案例数据（targets / frames / inference_frames）
  let extent = { w: DEFAULT_EXTENT.w, s: DEFAULT_EXTENT.s, e: DEFAULT_EXTENT.e, n: DEFAULT_EXTENT.n };
  let lastTick = 0;
  let selected = null;
  let targetEnts = {};        // 目标 id -> { def, icon, mark, bar, status, hpBucket, barKey }
  let clusterSlots = [];      // [{ main, bar }]
  let clusterState = [];      // 当前帧的聚合结果（诊断/自检用）
  let prevClusters = [];      // 上一轮聚合（滞回 + 稳定 id，来自 TargetClusters）
  let lastLayout = null;      // 最近一次聚合/避让迭代的诊断
  let selectionEnt = null;    // 选中目标的黄色定位环（最后绘制，压住聚合也不怕）

  // 损毁目标的图标压暗色。只压暗、不改色相，兵种图形仍然可辨。
  const DESTROYED_TINT = Cesium.Color.fromCssColorString('#AEB6C2');

  function container() { return document.getElementById('overviewContainer'); }
  function _statusApi() { return global.TargetStatus || null; }

  // ── 取景 ────────────────────────────────────────────────────────────

  /**
   * 由案例全部帧的全部目标位置计算战场范围（经纬度包围盒）。
   *
   * 用**整段航迹**而不是当前帧：鸟瞰图要回答"整场仗在哪打"，如果取景跟着
   * 当前帧目标走，播放时地图会不断缩放，比例尺失去可比性。
   */
  function computeExtent(d) {
    let w = Infinity, s = Infinity, e = -Infinity, n = -Infinity;
    (d && d.frames || []).forEach(function (fr) {
      (fr.targets || []).forEach(function (p) {
        if (!isFinite(p.lon) || !isFinite(p.lat)) return;
        if (p.lon < w) w = p.lon;
        if (p.lon > e) e = p.lon;
        if (p.lat < s) s = p.lat;
        if (p.lat > n) n = p.lat;
      });
    });
    if (!isFinite(w) || !isFinite(e)) {
      // 没有可计算的目标位置：保持世界视图，不再回退到台海（用户明确要求
      // 未加载/无数据时显示全球二维地图）。
      return { w: DEFAULT_EXTENT.w, s: DEFAULT_EXTENT.s, e: DEFAULT_EXTENT.e, n: DEFAULT_EXTENT.n };
    }
    // 最小跨度 0.4°：目标几乎静止时也要有一个可读的战场范围，而不是把镜头
    // 推到一栋楼那么近。先按最小跨度扩到中心对称，再加 8% 边距。
    const spanLon = Math.max(e - w, 0.4);
    const spanLat = Math.max(n - s, 0.4);
    const cx = (w + e) / 2, cy = (s + n) / 2;
    w = cx - spanLon / 2; e = cx + spanLon / 2;
    s = cy - spanLat / 2; n = cy + spanLat / 2;
    const padLon = spanLon * 0.08;
    const padLat = spanLat * 0.08;
    return { w: w - padLon, s: s - padLat, e: e + padLon, n: n + padLat };
  }

  /** 把鸟瞰图对准当前战场范围。隐藏状态下容器为 0×0，不做取景。 */
  function applyExtent() {
    if (!viewer || !extent || !isVisible()) return;
    viewer.camera.setView({
      destination: Cesium.Rectangle.fromDegrees(extent.w, extent.s, extent.e, extent.n),
    });
    viewer.scene.requestRender();
  }

  function setExtent(ex) {
    if (!ex || !isFinite(ex.w) || !isFinite(ex.s) || !isFinite(ex.e) || !isFinite(ex.n)) return;
    extent = ex;
    applyExtent();
  }

  // ── 案例数据 / 目标图元 ─────────────────────────────────────────────

  /**
   * 装载案例。会重建目标图元与聚合槽并重新取景；在 viewer 尚未创建时只保存数据，
   * 等 ensure() 时再建（鸟瞰图是惰性创建的，默认可能一直关着）。
   */
  function setData(d) {
    if (!d) return;
    data = d;
    extent = computeExtent(d);
    lastTick = 0;
    selected = null;
    clusterState = [];
    if (viewer) {
      clearTargetEntities();
      buildTargetEntities();
    }
    applyExtent();
    updateSituation(d, lastTick);
  }

  function clearTargetEntities() {
    if (!viewer) return;
    Object.keys(targetEnts).forEach(function (id) {
      const rec = targetEnts[id];
      [rec.icon, rec.mark, rec.bar].forEach(function (e) {
        if (e) { try { viewer.entities.remove(e); } catch (err) { /* already gone */ } }
      });
    });
    targetEnts = {};
    clusterSlots.forEach(function (slot) {
      [slot.main, slot.bar].forEach(function (e) {
        if (e) { try { viewer.entities.remove(e); } catch (err) { /* already gone */ } }
      });
    });
    clusterSlots = [];
    clusterState = [];
    prevClusters = [];
    lastLayout = null;
    if (selectionEnt) {
      try { viewer.entities.remove(selectionEnt); } catch (err) { /* already gone */ }
      selectionEnt = null;
    }
  }

  /** 清空案例（TS 被卸载）：移除全部目标/聚合图元并回到世界视图。 */
  function clear() {
    data = null;
    extent = {
      w: DEFAULT_EXTENT.w, s: DEFAULT_EXTENT.s, e: DEFAULT_EXTENT.e, n: DEFAULT_EXTENT.n,
    };
    lastTick = 0;
    selected = null;
    clearTargetEntities();
    applyExtent();   // 隐藏时由 show() 里的 applyExtent 补上
  }

  function buildTargetEntities() {
    if (!viewer || !data) return;
    (data.targets || []).forEach(function (def) {
      const icon = viewer.entities.add({
        id: 'ov-target-' + def.id,
        name: def.name || def.id,
        position: Cesium.Cartesian3.fromDegrees(0, 0, 0),
        billboard: {
          image: Symbology.symbolUri(def.icon, def.affiliation, { domain: def.domain }),
          width: ICON_PX, height: ICON_PX,
          verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
          pixelOffset: new Cesium.Cartesian2(0, ICON_OFFSET_Y),
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
      });
      icon.ovTargetId = def.id;

      // 损毁遮罩：暗色底 + 大号 "X"（label 比 billboard 后绘制，密集叠压时
      // 叉号仍浮在最上层）。受损/严重受损仅靠血条，不画状态环/角标。
      const mark = viewer.entities.add({
        id: 'ov-mark-' + def.id,
        position: Cesium.Cartesian3.fromDegrees(0, 0, 0),
        show: false,
        billboard: {
          image: _statusApi() ? _statusApi().markerUri(3) : undefined,
          width: MARK_PX, height: MARK_PX,
          verticalOrigin: Cesium.VerticalOrigin.CENTER,
          pixelOffset: new Cesium.Cartesian2(0, ICON_CENTER_Y),
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
        label: {
          text: 'X',
          font: 'bold 26px sans-serif',
          fillColor: Cesium.Color.WHITE,
          outlineColor: Cesium.Color.fromCssColorString('#7F1D1D'),
          outlineWidth: 4,
          style: Cesium.LabelStyle.FILL_AND_OUTLINE,
          horizontalOrigin: Cesium.HorizontalOrigin.CENTER,
          verticalOrigin: Cesium.VerticalOrigin.CENTER,
          pixelOffset: new Cesium.Cartesian2(0, ICON_CENTER_Y),
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
      });
      mark.ovTargetId = def.id;

      const bar = viewer.entities.add({
        id: 'ov-bar-' + def.id,
        position: Cesium.Cartesian3.fromDegrees(0, 0, 0),
        billboard: {
          image: _statusApi() ? _statusApi().barUri(100, 0, BAR_W, BAR_H) : undefined,
          width: BAR_W, height: BAR_H,
          verticalOrigin: Cesium.VerticalOrigin.TOP,
          pixelOffset: new Cesium.Cartesian2(0, BAR_OFFSET_Y),
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
      });
      bar.ovTargetId = def.id;

      targetEnts[def.id] = {
        def: def, icon: icon, mark: mark, bar: bar,
        status: -1, hpBucket: -1, barKey: null,
      };
    });
    buildClusterSlots();
  }

  /** 聚合槽池：不相交的 >=2 成员组最多 floor(n/2) 个，留 1 个余量。 */
  function buildClusterSlots() {
    const n = (data && data.targets || []).length;
    const maxSlots = Math.max(1, Math.ceil(n / 2));
    clusterSlots = [];
    for (let i = 0; i < maxSlots; i++) {
      const main = viewer.entities.add({
        id: 'ov-cluster-' + i,
        position: Cesium.Cartesian3.fromDegrees(0, 0, 0),
        show: false,
        billboard: {
          image: Symbology.clusterUri({ affiliation: 'blue', count: 2, icon: 'unknown' }),
          width: CLUSTER_ICON_PX, height: CLUSTER_ICON_PX,
          verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
          pixelOffset: new Cesium.Cartesian2(0, 0),
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
      });
      const bar = viewer.entities.add({
        id: 'ov-cluster-bar-' + i,
        position: Cesium.Cartesian3.fromDegrees(0, 0, 0),
        show: false,
        billboard: {
          image: _statusApi() ? _statusApi().barUri(100, 0, CLUSTER_BAR_W, BAR_H) : undefined,
          width: CLUSTER_BAR_W, height: BAR_H,
          verticalOrigin: Cesium.VerticalOrigin.TOP,
          pixelOffset: new Cesium.Cartesian2(0, BAR_OFFSET_Y),
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
      });
      clusterSlots.push({ main: main, bar: bar });
    }
    // 选中环最后创建：实体绘制顺序在聚合之后，选中目标被聚合图标压住时
    // 仍能看到黄色定位环。
    selectionEnt = viewer.entities.add({
      id: 'ov-selection',
      position: Cesium.Cartesian3.fromDegrees(0, 0, 0),
      show: false,
      billboard: {
        image: _statusApi() ? _statusApi().selectionUri() : undefined,
        width: 40, height: 40,
        verticalOrigin: Cesium.VerticalOrigin.CENTER,
        pixelOffset: new Cesium.Cartesian2(0, ICON_CENTER_Y),
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
    });
  }

  // ── 聚合 ────────────────────────────────────────────────────────────
  //
  // 2026-10-22：改用与主图同一套屏幕空间聚类（web/src/targetClusters.js）：
  //   * 先按阵营分组，红蓝永不混聚；
  //   * 滞回（join 34 / split 42）+ spread 上限 56，避免链式串联与边界闪烁；
  //   * 稳定 id/成员由上一轮继承，播放时不会整组跳变；
  //   * 红蓝编队图标重叠时用 TargetClusters.separateClusters 做屏幕避让。
  // 旧的完整链接分组、mixed 组与圆形聚合图标已删除。

  /** 编队的大类构成：dominant/mixed/domain 与主图完全同口径。 */
  function clusterComposition(ids) {
    const counts = {}, domains = {};
    ids.forEach(function (id) {
      const def = targetEnts[id] && targetEnts[id].def;
      if (!def) return;
      const cat = Symbology.categoryOf(def.icon, def.domain);
      counts[cat] = (counts[cat] || 0) + 1;
      if (def.domain) domains[def.domain] = true;
    });
    const list = Object.keys(counts).map(function (k) {
      return { key: k, count: counts[k], label: Symbology.categoryLabel(k) };
    }).sort(function (a, b) { return (b.count - a.count) || (a.key < b.key ? -1 : 1); });
    const dk = Object.keys(domains);
    return {
      list: list, dominant: list.length ? list[0].key : 'unknown',
      mixed: list.length > 1, domain: dk.length === 1 ? dk[0] : null,
    };
  }

  /**
   * 逐帧同步：位置、战损状态、血条、聚合。
   *
   * 由 app.js 在 `updateInferenceUi()` 里调用（播放推进、拖动时间轴、切案例、
   * 切换选中都会经过这里），与主图和整体TS同一个节奏。
   */
  function updateSituation(d, tick) {
    if (d && d !== data) setData(d);
    if (!data) return;
    lastTick = Math.max(0, tick | 0);
    // 隐藏时只记 tick，不写图元；show() 会用当前 tick 补一次。
    if (!isVisible()) return;
    if (!viewer) return;

    const frames = data.frames || [];
    if (!frames.length) return;
    const fr = frames[Math.min(lastTick, frames.length - 1)];
    const pos = {};
    (fr.targets || []).forEach(function (p) { pos[p.id] = p; });
    const api = _statusApi();

    // 先算每个目标的屏幕坐标（聚合判据与视图缩放无关，用实际画布像素）。
    // 先算本帧状态：聚合候选要按**本帧**状态排除损毁目标，
    // 不能读上一帧的 rec.status（案例刚加载时它还是 -1，会把损毁目标藏进聚合）。
    const statusNow = {};
    Object.keys(targetEnts).forEach(function (id) {
      statusNow[id] = api ? api.at(data, id, lastTick) : { hp: 100, status: 0 };
    });

    const screen = {};
    Object.keys(targetEnts).forEach(function (id) {
      const p = pos[id];
      if (!p) return;
      const c = Cesium.Cartesian3.fromDegrees(p.lon, p.lat, 0);
      const w = Cesium.SceneTransforms.worldToWindowCoordinates(viewer.scene, c);
      if (w) screen[id] = { x: w.x, y: w.y };
    });

    // 选中目标不参与聚合（始终单独可见，并用最后绘制的黄环标记）。
    // 聚类与主图同一纯函数：先按阵营分组（红蓝不混聚），滞回 + spread 上限；
    // 上一轮成员用于稳定 id，避免播放时整组跳变。
    const clusterCandidates = Object.keys(screen).filter(function (id) { return id !== selected; });
    const items = clusterCandidates.map(function (id) {
      return {
        id: id, affiliation: targetEnts[id].def.affiliation,
        x: screen[id].x, y: screen[id].y,
      };
    });
    const clusterApi = global.TargetClusters || null;
    const clusterOpts = {
      joinPx: CLUSTER_JOIN_PX,
      splitPx: CLUSTER_SPLIT_PX,
      maxSpreadPx: CLUSTER_SPREAD_PX,
      sizePx: CLUSTER_ICON_PX,
    };
    let groups = (clusterApi && clusterApi.computeClusters)
      ? clusterApi.computeClusters(items, prevClusters, clusterOpts)
      : [];
    // 两遍收敛：第一遍结果作为第二遍的 prev，保证同一帧重复更新得到同一分组。
    if (clusterApi && clusterApi.computeClusters) {
      groups = clusterApi.computeClusters(items, groups, clusterOpts);
    }
    // 滞回/稳定 id 以**纯函数的输出**为准（不含下面的矩形布局合并），
    // 否则按投影矩形补做的合并会让下一轮 sticky 判据把大队丢弃重算。
    prevClusters = groups.map(function (g) {
      return { id: g.id, affiliation: g.affiliation, members: g.members.slice() };
    });
    groups.sort(function (a, b) { return (b.count - a.count) || (a.id < b.id ? -1 : 1); });
    // 编队图标画在**平均经纬度**的投影处（不是成员屏幕坐标的均值）；成员跨
    // 大范围时两者可能差几十像素，避让/取证/合并都必须以图标实际位置为准。
    function setGroupAnchor(g) {
      let lon = 0, lat = 0, n = 0;
      g.members.forEach(function (id) {
        const p = pos[id];
        if (p) { lon += p.lon; lat += p.lat; n++; }
      });
      g.lon = n ? lon / n : 0;
      g.lat = n ? lat / n : 0;
      if (n) {
        const c = Cesium.Cartesian3.fromDegrees(g.lon, g.lat, 0);
        const w = Cesium.SceneTransforms.worldToWindowCoordinates(viewer.scene, c);
        if (w && isFinite(w.x)) { g.x = w.x; g.y = w.y; }
      }
    }
    groups.forEach(setGroupAnchor);
    // 布局收敛：同阵营按**图标实际矩形**合并，红蓝按矩形推离。两步会互相影响
    // （推离可能把红簇推进另一个红簇），所以合并/推离迭代到稳定为止。
    // 避让框比图标本身多 4/12 px：34×42 的图标之间至少留 2 px 间隙，
    // 避免"刚好贴边"被自检的 +1 px 容差判成相交。
    const boxW = CLUSTER_ICON_PX + 4, boxH = CLUSTER_ICON_PX + 12;
    // 合并判据留 margin：红蓝避让最多把两个同色簇推近 52 px，按锚点位置
    // 提前合并可以避免"推离后又相交"的振荡。
    const mergeW = CLUSTER_ICON_PX + 8, mergeH = CLUSTER_ICON_PX + 18;
    function sameAffOverlap(w, h) {
      for (let i = 0; i < groups.length; i++) {
        for (let j = i + 1; j < groups.length; j++) {
          if (groups[i].affiliation !== groups[j].affiliation) continue;
          if (Math.abs(groups[i].x - groups[j].x) < w &&
              Math.abs(groups[i].y - groups[j].y) < h) return true;
        }
      }
      return false;
    }
    function mergeGroupPair(i, j) {
      groups[i].members = groups[i].members.concat(groups[j].members);
      groups[i].count = groups[i].members.length;
      groups.splice(j, 1);
      setGroupAnchor(groups[i]);
    }
    for (let pass = 0; pass < 16; pass++) {
      groups.forEach(setGroupAnchor);
      let merged = false, again = true;
      while (again) {
        again = false;
        for (let i = 0; i < groups.length && !again; i++) {
          for (let j = i + 1; j < groups.length; j++) {
            if (groups[i].affiliation !== groups[j].affiliation) continue;
            if (Math.abs(groups[i].x - groups[j].x) >= mergeW ||
                Math.abs(groups[i].y - groups[j].y) >= mergeH) continue;
            mergeGroupPair(i, j);
            merged = true;
            again = true;
            break;
          }
        }
      }
      // 红蓝编队图标不合并（语义上必须分开），但要在屏幕上推离，避免互相压住。
      // 矩形按 34×42（图标 + 下方血条）判定；斜向排列时圆距离判据不够。
      if (clusterApi && clusterApi.separateClusters) {
        clusterApi.separateClusters(groups, {
          w: boxW, h: boxH, maxOffset: CLUSTER_MAX_OFFSET_PX,
        });
      }
      // 推离可能把两个同色簇挤到一起：按**推离后**的实际位置合并一对，
      // 然后重跑一轮（合并会减少组数，必然收敛）。
      let forced = false;
      for (let i = 0; i < groups.length && !forced; i++) {
        for (let j = i + 1; j < groups.length; j++) {
          if (groups[i].affiliation !== groups[j].affiliation) continue;
          if (Math.abs(groups[i].x - groups[j].x) >= boxW ||
              Math.abs(groups[i].y - groups[j].y) >= boxH) continue;
          mergeGroupPair(i, j);
          forced = true;
          merged = true;
          break;
        }
      }
      const overlap = sameAffOverlap(mergeW, mergeH);
      lastLayout = { pass: pass, groups: groups.length, merged: merged, overlap: overlap };
      if (forced) continue;
      if (!merged && !overlap) break;
    }
    // 视口边界钳制：屏幕避让不能把最边缘的编队图标推出画布。
    const cw = viewer.scene.canvas.clientWidth, ch = viewer.scene.canvas.clientHeight;
    const halfW = CLUSTER_ICON_PX / 2 + 2;
    groups.forEach(function (g) {
      if (!isFinite(g.ox)) return;
      const minDx = halfW + 1 - g.ox, maxDx = cw - halfW - 1 - g.ox;
      const minDy = CLUSTER_ICON_PX + 4 - g.oy;
      const maxDy = ch - 12 - g.oy;
      if (minDx <= maxDx) g.dx = Math.max(minDx, Math.min(maxDx, g.dx));
      if (minDy <= maxDy) g.dy = Math.max(minDy, Math.min(maxDy, g.dy));
      g.x = g.ox + g.dx;
      g.y = g.oy + g.dy;
    });
    const residual = sameAffOverlap(boxW, boxH);
    lastLayout.overlap = residual;
    if (residual && global.console && console.warn) {
      console.warn('[overview] cluster layout did not converge (same-affiliation overlap)');
    }
    const clustered = {};
    groups.forEach(function (g) { g.members.forEach(function (id) { clustered[id] = true; }); });

    // 更新个体目标。
    Object.keys(targetEnts).forEach(function (id) {
      const rec = targetEnts[id];
      const p = pos[id];
      if (p) {
        const c = Cesium.Cartesian3.fromDegrees(p.lon, p.lat, 0);
        rec.icon.position = c;
        rec.mark.position = c;
        rec.bar.position = c;
      }

      const stNow = statusNow[id] || { hp: 100, status: 0 };
      let hp = stNow.hp, status = stNow.status;

      if (status !== rec.status) {
        rec.status = status;
        if (status === 3) rec.mark.billboard.image = api.markerUri(3);
        rec.icon.billboard.color = (status === 3) ? DESTROYED_TINT : Cesium.Color.WHITE;
      }

      // 缓存键必须同时包含**档位**与 0.1% 精度的血量：只按整数血量比较时，
      // 70.1→70.0、35.1→35.0、0.1→0.0 这类跨档跃迁的取整值不变，
      // 血条会滞后数帧，损毁瞬间也不会切成深色空槽（与整体TS不一致）。
      const barKey = status + '|' + Math.round(hp * 10);
      if (barKey !== rec.barKey) {
        rec.barKey = barKey;
        rec.hpBucket = Math.round(hp);
        rec.bar.billboard.image = api
          ? api.barUri(hp, status, BAR_W, BAR_H)
          : undefined;
      }

      const hide = clustered[id] === true;
      rec.icon.show = !hide;
      rec.bar.show = !hide;
      rec.mark.show = !hide && status === 3;
    });

    // 更新聚合槽（按组大小降序，大组优先占槽）。
    clusterState = [];
    groups.forEach(function (g, i) {
      const ids = g.members;
      const slot = clusterSlots[i];
      if (!slot) {
        // 池不够时不能把成员静默藏掉：恢复成员个体显示并留痕。
        console.warn('[overview] cluster slots exhausted, showing group individually');
        ids.forEach(function (id) {
          const r = targetEnts[id];
          if (!r) return;
          r.icon.show = true;
          r.bar.show = true;
          r.mark.show = r.status === 3;
        });
        return;
      }
      let lon = g.lon, lat = g.lat, hpSum = 0, statusWorst = 0, deadN = 0;
      ids.forEach(function (id) {
        const st = statusNow[id] || { hp: 100, status: 0 };
        hpSum += st.hp;
        statusWorst = Math.max(statusWorst, st.status);
        if (st.status === 3) deadN++;
      });
      const allDead = deadN === ids.length;
      const avgHp = allDead ? 0 : hpSum / ids.length;
      const avgStatus = allDead ? 3 : (api ? api.statusOf(avgHp) : 0);
      const comp = clusterComposition(ids);
      // 编队图标：阵营帧 + 数量 + 主类别（与主图同一种图标）。
      const image = Symbology.clusterUri({
        affiliation: g.affiliation, count: ids.length,
        icon: comp.dominant, mixed: comp.mixed, domain: comp.domain,
        deadCount: deadN,
      });

      const c = Cesium.Cartesian3.fromDegrees(lon, lat, 0);
      slot.main.position = c;
      slot.bar.position = c;
      slot.main.billboard.image = image;
      // 整组损毁与单目标同口径：压暗色（颜色不改计数/类别）。
      slot.main.billboard.color = allDead ? DESTROYED_TINT : Cesium.Color.WHITE;
      // 红蓝编队图标的屏幕避让量；标牌/血条跟着图标一起偏。
      // 不取整：避让量可能正是"刚好 44 px"的临界值，取整会留下 1 px 相交。
      const dx = g.dx || 0, dy = g.dy || 0;
      slot.main.billboard.pixelOffset = new Cesium.Cartesian2(dx, dy);
      slot.bar.billboard.pixelOffset = new Cesium.Cartesian2(dx, BAR_OFFSET_Y + dy);
      slot.main.ovTargetId = ids[0];
      slot.main.ovClusterIds = ids.slice();
      slot.bar.ovTargetId = ids[0];
      slot.bar.ovClusterIds = ids.slice();
      slot.main.show = true;
      // 颜色用最差档（避免低血目标被平均成绿色），长度用平均值。
      slot.bar.billboard.image = api
        ? api.barUri(avgHp, avgStatus, CLUSTER_BAR_W, BAR_H, statusWorst)
        : undefined;
      slot.bar.show = true;

      clusterState.push({
        ids: ids.slice(), count: ids.length, affiliation: g.affiliation,
        lon: lon, lat: lat, dx: dx, dy: dy,
        categories: comp.list, dominant: comp.dominant, domain: comp.domain,
        hp: Math.round(avgHp * 10) / 10, status: avgStatus,
        worstStatus: statusWorst, mixed: comp.mixed, destroyed: allDead,
        deadCount: deadN, deadBadge: deadN > 0,
        deadIds: ids.filter(function (id) { return (statusNow[id] || {}).status === 3; }),
      });
    });
    for (let i = groups.length; i < clusterSlots.length; i++) {
      clusterSlots[i].main.show = false;
      clusterSlots[i].bar.show = false;
    }

    // 选中环最后更新：即使所选目标被聚合图标压住，黄环也可见。
    if (selectionEnt) {
      const sp = selected ? pos[selected] : null;
      if (sp) {
        selectionEnt.position = Cesium.Cartesian3.fromDegrees(sp.lon, sp.lat, 0);
        selectionEnt.show = true;
      } else {
        selectionEnt.show = false;
      }
    }

    viewer.scene.requestRender();
  }

  /** 主图选中变化时同步高亮（app.js 的 syncSelection 调用）。 */
  function setSelected(id) {
    selected = id || null;
    if (!viewer) return;
    Object.keys(targetEnts).forEach(function (tid) {
      const rec = targetEnts[tid];
      rec.icon.billboard.image = Symbology.symbolUri(
        rec.def.icon, rec.def.affiliation,
        { selected: tid === selected, domain: rec.def.domain });
    });
    // 选中变化会改变"谁不参与聚合"，必须立即重算一次；否则暂停时
    // 鸟瞰图会保留上一轮的聚合布局，直到下一次 tick 才纠正。
    if (data && isVisible()) updateSituation(data, lastTick);
    viewer.scene.requestRender();
  }

  // ── 交互 ────────────────────────────────────────────────────────────

  function bindClick() {
    const handler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);
    handler.setInputAction(function (click) {
      const picked = viewer.scene.pick(click.position);
      const ent = picked && picked.id;
      if (ent && ent.ovClusterIds && global.SituationHost && SituationHost.selectTarget) {
        // 聚合组：选中代表目标并让主视图飞过去，主 3D 里能看到组内个体。
        SituationHost.selectTarget(ent.ovTargetId, true);
        return;
      }
      if (ent && ent.ovTargetId && global.SituationHost && SituationHost.selectTarget) {
        SituationHost.selectTarget(ent.ovTargetId, true);
        return;
      }

      const ray = viewer.camera.getPickRay(click.position);
      let point = ray ? viewer.scene.globe.pick(ray, viewer.scene) : null;
      if (!point) {
        const ellipsoid = (viewer.scene.globe && viewer.scene.globe.ellipsoid)
          || Cesium.Ellipsoid.WGS84;
        point = viewer.camera.pickEllipsoid(click.position, ellipsoid);
      }
      if (!point) return;
      const carto = Cesium.Cartographic.fromCartesian(point);
      if (global.Follow && global.Follow.isActive()) global.Follow.detach('鸟瞰图跳转');
      const h = Math.max(mainViewer.camera.positionCartographic.height, 600000);
      mainViewer.camera.cancelFlight();
      mainViewer.camera.flyTo({
        destination: Cesium.Cartesian3.fromRadians(carto.longitude, carto.latitude, h),
        orientation: {
          heading: mainViewer.camera.heading,
          pitch: mainViewer.camera.pitch,
          roll: 0,
        },
        duration: 1.2,
      });
    }, Cesium.ScreenSpaceEventType.LEFT_CLICK);
  }

  // ── 生命周期 ────────────────────────────────────────────────────────

  /** 惰性创建。首次调用时容器必须已经可见（非 display:none），否则画布会是 0×0。 */
  function ensure() {
    if (created || !mainViewer) return false;
    const el = container();
    if (!el) return false;
    created = true;

    viewer = new Cesium.Viewer('overviewContainer', {
      baseLayer: new Cesium.ImageryLayer(OfflineAssets.createOverviewImageryProvider()),
      terrainProvider: new Cesium.EllipsoidTerrainProvider(),
      sceneMode: Cesium.SceneMode.SCENE2D,
      baseLayerPicker: false, geocoder: false, homeButton: false,
      sceneModePicker: false, navigationHelpButton: false,
      navigationInstructionsInitiallyVisible: false,
      animation: false, timeline: false, fullscreenButton: false, vrButton: false,
      infoBox: false, selectionIndicator: false, shouldAnimate: false,
    });

    if (viewer.imageryLayers.length === 0) {
      throw new Error('鸟瞰图底图未创建：Cesium ' + Cesium.VERSION + ' 需要 baseLayer 选项');
    }

    const scene = viewer.scene;
    scene.globe.maximumScreenSpaceError = 4;
    scene.globe.enableLighting = false;
    scene.globe.showGroundAtmosphere = false;
    scene.globe.baseColor = Cesium.Color.fromCssColorString('#dfe6ef');
    scene.skyAtmosphere.show = false;
    if (scene.skyBox) scene.skyBox.show = false;
    if (scene.sun) scene.sun.show = false;
    if (scene.moon) scene.moon.show = false;
    scene.fog.enabled = false;
    scene.screenSpaceCameraController.enableInputs = false;   // 取景由案例范围决定
    viewer.cesiumWidget.screenSpaceEventHandler.removeInputAction(
      Cesium.ScreenSpaceEventType.LEFT_DOUBLE_CLICK);

    viewer.resolutionScale = 0.8;
    // 组件底部不显示 Cesium / 影像归属文字（用户 2026-10-21 要求）。
    const credit = viewer.cesiumWidget.creditContainer;
    if (credit) credit.style.display = 'none';

    buildTargetEntities();
    bindClick();
    applyExtent();
    updateSituation(data, lastTick);
    return true;
  }

  function show() {
    const el = document.getElementById('overviewWrap');
    if (!el) return;
    el.classList.remove('hidden');
    if (!created) ensure();
    if (!viewer) return;
    // display:none 会让容器尺寸归零，而 Cesium 只监听 window.resize，必须手动补
    viewer.resize();
    applyExtent();
    updateSituation(data, lastTick);
    // 布局可能还没落定，下一帧再对齐一次
    setTimeout(function () {
      if (!viewer) return;
      viewer.resize();
      applyExtent();
      updateSituation(data, lastTick);
    }, 0);
  }

  function hide() {
    const el = document.getElementById('overviewWrap');
    if (el) el.classList.add('hidden');
  }

  function isVisible() {
    const el = document.getElementById('overviewWrap');
    return !!el && !el.classList.contains('hidden');
  }

  // 窗口跨过 1280px 断点时组件尺寸会变（268×190 ↔ 200×160），2D 相机高度
  // 不会自动跟随；重新取景一次，保证战场范围仍然完整可见。
  let _resizeTimer = null;
  window.addEventListener('resize', function () {
    if (!viewer) return;
    clearTimeout(_resizeTimer);
    _resizeTimer = setTimeout(function () {
      if (!viewer || !isVisible()) return;
      viewer.resize();
      applyExtent();
      updateSituation(data, lastTick);
    }, 200);
  });

  // ── 诊断 ────────────────────────────────────────────────────────────

  function debugInfo() {
    if (!viewer) return 'not created';
    const c = viewer.scene.canvas;
    return 'created  targets ' + Object.keys(targetEnts).length +
           '  clusters ' + clusterState.length +
           '  layers ' + viewer.imageryLayers.length +
           '  canvas ' + c.width + 'x' + c.height +
           '  extent ' + (extent
             ? [extent.w, extent.s, extent.e, extent.n].map(function (v) { return v.toFixed(6); }).join(',')
             : '—') +
           '  visible ' + isVisible();
  }

  /**
   * 目标/聚合取证。
   *   - targets：每个目标的位置、显示状态、血条图；被聚合的目标 show=false；
   *   - clusters：当前帧的聚合组（成员、质心、平均血量/状态）。
   */
  function targetDebug() {
    const out = { caseId: data ? data.case_id : null, tick: lastTick, extent: extent,
                  targets: {}, clusters: clusterState.slice(),
                  layout: lastLayout,
                  selectionShown: !!(selectionEnt && selectionEnt.show) };
    if (!viewer || !data) return out;
    const cam = viewer.camera.positionCartographic;
    out.camera = { lon: Cesium.Math.toDegrees(cam.longitude), lat: Cesium.Math.toDegrees(cam.latitude),
                   height: cam.height };
    out.viewRect = (function () {
      const ellipsoid = (viewer.scene && viewer.scene.globe && viewer.scene.globe.ellipsoid)
        || Cesium.Ellipsoid.WGS84;
      const canvas = viewer.scene.canvas;
      const w = canvas.clientWidth, h = canvas.clientHeight;
      const corners = [[0, 0], [w, 0], [w, h], [0, h]];
      const cartos = [];
      for (let i = 0; i < corners.length; i++) {
        const p = viewer.camera.pickEllipsoid(
          new Cesium.Cartesian2(corners[i][0], corners[i][1]), ellipsoid);
        if (!p) return null;
        cartos.push(Cesium.Cartographic.fromCartesian(p));
      }
      let wDeg = Infinity, sDeg = Infinity, eDeg = -Infinity, nDeg = -Infinity;
      cartos.forEach(function (c) {
        const lon = Cesium.Math.toDegrees(c.longitude);
        const lat = Cesium.Math.toDegrees(c.latitude);
        if (lon < wDeg) wDeg = lon;
        if (lon > eDeg) eDeg = lon;
        if (lat < sDeg) sDeg = lat;
        if (lat > nDeg) nDeg = lat;
      });
      return { w: wDeg, s: sDeg, e: eDeg, n: nDeg };
    })();
    const frames = data.frames || [];
    const fr = frames[Math.min(lastTick, frames.length - 1)] || { targets: [] };
    const pos = {};
    (fr.targets || []).forEach(function (p) { pos[p.id] = p; });
    // 显示项矩形（屏幕像素）：供自检断言去重叠。
    out.items = [];
    function pushItem(kind, idOrIds, lon, lat, w, h, dx, dy) {
      const c = Cesium.Cartesian3.fromDegrees(lon, lat, 0);
      const sp = Cesium.SceneTransforms.worldToWindowCoordinates(viewer.scene, c);
      if (sp) out.items.push({
        kind: kind, id: idOrIds,
        x: sp.x + (dx || 0), y: sp.y + (dy || 0), w: w, h: h,
      });
    }
    Object.keys(targetEnts).forEach(function (id) {
      const rec = targetEnts[id];
      const p = rec.icon.position && rec.icon.position.getValue
        ? rec.icon.position.getValue(viewer.clock.currentTime) : null;
      const carto = p ? Cesium.Cartographic.fromCartesian(p) : null;
      if (rec.icon.show !== false && carto) {
        pushItem('target', id, Cesium.Math.toDegrees(carto.longitude),
                 Cesium.Math.toDegrees(carto.latitude), ICON_PX, ICON_PX + 8);
      }
      out.targets[id] = {
        show: rec.icon.show !== false,
        barShow: rec.bar.show !== false,
        markShow: !!rec.mark.show,
        barImage: rec.bar.billboard.image ? String(rec.bar.billboard.image.getValue()) : '',
        iconTint: rec.icon.billboard.color && rec.icon.billboard.color.getValue
          ? rec.icon.billboard.color.getValue().toCssHexString() : null,
        frameLon: pos[id] ? pos[id].lon : null,
        frameLat: pos[id] ? pos[id].lat : null,
        lon: carto ? Cesium.Math.toDegrees(carto.longitude) : null,
        lat: carto ? Cesium.Math.toDegrees(carto.latitude) : null,
        status: rec.status,
        hp: rec.hpBucket,
        selected: id === selected,
        clustered: rec.icon.show === false,
      };
    });
    clusterState.forEach(function (c) {
      pushItem('cluster', c.ids, c.lon, c.lat, CLUSTER_ICON_PX, CLUSTER_ICON_PX + 8, c.dx, c.dy);
    });
    return out;
  }

  global.Overview = {
    init: function (main) { mainViewer = main; },
    _viewer: function () { return viewer; },
    debugInfo: debugInfo,
    targetDebug: targetDebug,
    setData: setData,
    clear: clear,
    updateSituation: updateSituation,
    setSelected: setSelected,
    ensure: ensure,
    show: show,
    hide: hide,
    isVisible: isVisible,
    setExtent: setExtent,
    resize: function () {
      if (viewer) { viewer.resize(); applyExtent(); }
    },
  };
})(window);
