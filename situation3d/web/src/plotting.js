/**
 * 态势标绘：把「中式军队标号为主、美军标补齐」的标绘图形做成**图层管理模块里的一类图层**。
 *
 * 三条设计约束（来自项目负责人）
 * ------------------------------
 * 1. **中国军标为基准**：能对上的图形一律按中式标号；中式未覆盖或缺失的，用美军标
 *    （FM 1-02.2 / APP-06）补齐。
 * 2. **每类图形必须可溯源**：`CATALOGUE` 里每一项都带 `std` 字段（`'CN'` / `'US'`）
 *    与 `ref`（具体依据），`Plotting.catalogue()` 可整体导出——这是为了后续替换或收敛
 *    时能一眼看出"哪些是美式补齐的"，而不是散在代码注释里。
 * 3. **纳入既有图层管理**：标绘不是独立通路。每个标绘类别就是 `LayerRegistry` 里的
 *    一个 `kind:'plot'` 图层，顺序靠 `z`、显隐靠 `setDesired` + 本模块的
 *    `setLayerVisible` sink、增删复用 `register`/`list`。
 *
 * 为什么标绘不是"图片"
 * -------------------
 * 单位符号是**点状**的，可以离线成 SVG；标绘（突击方向、钳击、集结地域、火力扇面…）是
 * **多点图形**，形状由控制点算出来。所以这里没有一张图片资源：
 *
 *     控制点 → 几何生成函数 → Cesium 原生几何（多边形/折线） → 场景
 *
 * 这也意味着"增加一类标绘"= 增加一个几何函数，不是增加一个素材文件。
 *
 * 几何算法族
 * ----------
 * 标绘类型很多，但构造方式收敛到少数几族。新增类型时先归类，再复用该族的构造：
 *
 *   arrow  双/多控制点箭头（燕尾箭头、突击方向、钳击箭头、突破方向）
 *   zone   闭合区域（集结/待机地域、登陆场、封锁区、雷场）
 *   fan    扇形/环带（火力打击范围、雷达探测范围、预警扇区）
 *   band   双侧平行线走廊（行军走廊、空中走廊、火力封锁带）
 *   line   单/多段折线（战斗分界线、协调线、阶段线）
 *
 * 依赖：window.Cesium。加载顺序在 Cesium.js 之后、app.js 之前。
 */
(function () {
  'use strict';

  // ---------------------------------------------------------------------
  // 图形目录：每项都标注标准来源，供后续替换/收敛
  // ---------------------------------------------------------------------
  //
  // `std` 取值：
  //   'CN'  中式军队标号（基准）
  //   'US'  美军标补齐（FM 1-02.2 / APP-06），**是补齐项，将来有中式依据时应替换**
  //
  // `ref` 写清依据到什么程度：中式写类别与教材体系，美式写 FM/APP-06 的图形名。
  // 不写具体 GJB 编号——公开渠道无法核实编号，写了反而是编造。
  var CATALOGUE = {
    // ---- 进攻类 ----
    main_attack: {
      zh: '主要突击方向', family: 'arrow', std: 'CN',
      ref: '中式军队标号·进攻类（带箭头的突击方向）',
      us: 'Main Axis of Advance (FM 1-02.2)',
      params: { width: 22000, tailSpread: 0.55, headSpread: 1.0 },
    },
    feint_attack: {
      zh: '助攻/佯动方向', family: 'arrow', std: 'CN',
      ref: '中式军队标号·进攻类',
      us: 'Supporting Axis of Advance (FM 1-02.2)',
      params: { width: 14000, tailSpread: 0.5, headSpread: 0.9, dashed: true },
    },
    pincer: {
      zh: '钳击箭头', family: 'arrow', std: 'CN',
      ref: '中式军队标号·进攻类（两翼钳击）',
      us: 'Encirclement / Double Envelopment (FM 1-02.2)',
      params: { width: 16000, tailSpread: 0.45, headSpread: 0.95 },
    },
    breakthrough: {
      zh: '突破方向', family: 'arrow', std: 'CN',
      ref: '中式军队标号·进攻类',
      us: 'Breach (FM 1-02.2)',
      params: { width: 12000, tailSpread: 0.6, headSpread: 1.0 },
    },
    // ---- 机动/集结类 ----
    assembly_area: {
      zh: '集结地域', family: 'zone', std: 'CN',
      ref: '中式军队标号·地域类',
      us: 'Assembly Area (FM 1-02.2)',
      params: { dash: true },
    },
    waiting_area: {
      zh: '待机地域', family: 'zone', std: 'CN',
      ref: '中式军队标号·地域类',
      us: 'Attack Position / Assembly Area (FM 1-02.2)',
      params: { dash: true },
    },
    landing_beach: {
      zh: '登陆场（滩头）', family: 'zone', std: 'CN',
      ref: '中式军队标号·登陆作战类',
      us: 'Landing Beach / Beachhead (APP-06)',
      params: {},
    },
    // ---- 防御/障碍类 ----
    defense_line: {
      zh: '防御阵地（防线）', family: 'band', std: 'CN',
      ref: '中式军队标号·防御类',
      us: 'Defense Line / Battle Position (FM 1-02.2)',
      params: { width: 9000, teeth: true },
    },
    minefield: {
      zh: '雷场（障碍区）', family: 'zone', std: 'CN',
      ref: '中式军队标号·障碍物类',
      us: 'Minefield / Obstacle Belt (FM 1-02.2)',
      params: { hatch: true },
    },
    // ---- 火力类 ----
    fire_range: {
      zh: '火力打击范围', family: 'fan', std: 'CN',
      ref: '中式军队标号·火力类（扇形射击范围）',
      us: 'Field of Fire / Weapons Free Area (FM 1-02.2)',
      params: { radius: 90000, spreadDeg: 70 },
    },
    fire_barrage: {
      zh: '火力封锁线', family: 'band', std: 'CN',
      ref: '中式军队标号·火力类',
      us: 'Final Protective Fire / Linear Barrage (FM 1-02.2)',
      params: { width: 12000 },
    },
    // ---- 侦察/空域类：中式公开依据不足，以下明确标为美式补齐 ----
    recon_area: {
      zh: '侦察区域', family: 'zone', std: 'US',
      ref: 'FM 1-02.2 Reconnaissance Area（中式对应标号公开依据不足，暂用美式）',
      us: 'Reconnaissance Area (FM 1-02.2)',
      params: { dash: true },
    },
    no_fly_zone: {
      zh: '禁飞区', family: 'zone', std: 'US',
      ref: 'APP-06 No-Fly Zone（中式对应标号公开依据不足，暂用美式）',
      us: 'No-Fly Zone (APP-06)',
      params: { hatch: true },
    },
    air_corridor: {
      zh: '空中走廊', family: 'band', std: 'US',
      ref: 'FM 1-02.2 Air Corridor（中式对应标号公开依据不足，暂用美式）',
      us: 'Air Corridor (FM 1-02.2)',
      params: { width: 15000, dashed: true },
    },
    // ---- 分界线/协调线：中式有对应，规则与美式接近 ----
    boundary: {
      zh: '战斗分界线', family: 'line', std: 'CN',
      ref: '中式军队标号·分界线类',
      us: 'Boundary / Boundary Line (FM 1-02.2)',
      params: { dash: false },
    },
    coordination_line: {
      zh: '协调线', family: 'line', std: 'CN',
      ref: '中式军队标号·协调线类',
      us: 'Coordinating Line (FM 1-02.2)',
      params: { dash: true },
    },
  };

  var COLOR = { red: '#d0342c', blue: '#1f5fbf', neutral: '#6b7a8d' };

  var viewer = null;
  var byId = {};              // 标绘实例 id -> 记录
  var seq = 0;
  var layerVisible = {};      // 标绘类别图层 id -> 是否可见（由 LayerRegistry 的 sink 驱动）

  function bind(v) { viewer = v || null; return !!viewer; }

  // ---------------------------------------------------------------------
  // 几何族
  // ---------------------------------------------------------------------

  /** 米 → 经纬度偏移。纬度每度约 111320 m；经度按纬度收缩。 */
  function offset(lon, lat, dEast, dNorth) {
    var lat2 = lat + dNorth / 111320;
    var lon2 = lon + dEast / (111320 * Math.max(0.15, Math.cos(lat * Math.PI / 180)));
    return [lon2, lat2];
  }

  function normDeg(d) { var x = d % 360; return x < 0 ? x + 360 : x; }

  /**
   * 双控制点燕尾箭头（中式突击方向的基本形）。
   *
   * 构造：主轴 A→B；在 A 端按 `tailSpread` 张开尾宽，在 B 端按 `headSpread` 张开箭头；
   * 箭身沿轴向。这是"折线箭头类"的最简形式，钳击/突破都是它的参数化变体。
   */
  function arrowFromTwoPoints(a, b, p) {
    var width = p.width || 20000;
    var tail = width * (p.tailSpread === undefined ? 0.5 : p.tailSpread) / 2;
    var head = width * (p.headSpread === undefined ? 1.0 : p.headSpread) / 2;
    var dx = b[0] - a[0], dy = b[1] - a[1];
    var len = Math.sqrt(dx * dx + dy * dy) || 1e-9;
    // 轴的单位方向（东、北 分量）
    var ux = dx / len, uy = dy / len;
    var px = -uy, py = ux;                    // 左法向

    // 箭头头部占全长的比例：中式标号箭头较饱满
    var headLenRatio = 0.28;
    var neckT = 1 - headLenRatio;
    var neck = [a[0] + dx * neckT, a[1] + dy * neckT];

    function at(pt, sEast, sNorth) {
      // 把"沿轴向的偏移 + 沿法向的偏移"换算成经纬度
      var e = ux * sEast + px * sNorth;
      var n = uy * sEast + py * sNorth;
      return offset(pt[0], pt[1], e, n);
    }

    var ring = [
      at(a, 0, tail),          // 尾左
      at(neck, 0, tail),       // 颈左
      at(neck, 0, head),       // 箭翼左
      at(b, 0, 0),             // 箭尖
      at(neck, 0, -head),      // 箭翼右
      at(neck, 0, -tail),      // 颈右
      at(a, 0, -tail),         // 尾右
    ];
    return ring;
  }

  /** 把两段折线各自生成燕尾箭头（钳击箭头 = 两条 arrow 共起点）。 */
  function pincerRings(pts, p) {
    if (pts.length < 3) return [arrowFromTwoPoints(pts[0], pts[1], p)];
    var mid = pts[1];
    return [
      arrowFromTwoPoints(mid, pts[0], p),
      arrowFromTwoPoints(mid, pts[2], p),
    ];
  }

  /** 走廊/双侧平行线族：沿折线两侧各偏移 half 宽度。 */
  function bandRings(pts, p) {
    var half = (p.width || 10000) / 2;
    var left = [], right = [];
    for (var i = 0; i < pts.length; i++) {
      var prev = pts[Math.max(0, i - 1)], next = pts[Math.min(pts.length - 1, i + 1)];
      var dx = next[0] - prev[0], dy = next[1] - prev[1];
      var len = Math.sqrt(dx * dx + dy * dy) || 1e-9;
      var px = -(dy / len), py = dx / len;       // 左法向（经纬度空间近似）
      var cosLat = Math.max(0.15, Math.cos(pts[i][1] * Math.PI / 180));
      left.push([pts[i][0] + px * half / (111320 * cosLat), pts[i][1] + py * half / 111320]);
      right.push([pts[i][0] - px * half / (111320 * cosLat), pts[i][1] - py * half / 111320]);
    }
    return [left.concat(right.reverse())];
  }

  /** 扇形族：以圆心 + 半径 + 起始方位/张角生成扇面多边形。 */
  function fanRing(center, p) {
    var r = p.radius || 80000;
    var spread = p.spreadDeg || 60;
    var start = (p.startDeg === undefined ? -spread / 2 : p.startDeg);
    var steps = Math.max(8, Math.round(spread / 4));
    var ring = [center.slice()];
    for (var i = 0; i <= steps; i++) {
      var brg = start + spread * i / steps;
      var rad = brg * Math.PI / 180;
      // 方位角自北起顺时针
      ring.push(offset(center[0], center[1], Math.sin(rad) * r, Math.cos(rad) * r));
    }
    return ring;
  }

  /** 闭合区域族：控制点直接成环。 */
  function zoneRing(pts) { return pts.slice(); }

  /** 按族生成环（返回数组：一项标绘可能产生多个环，如钳击箭头）。 */
  function ringsFor(typeId, pts) {
    var def = CATALOGUE[typeId];
    if (!def) throw new Error('未知标绘类型：' + typeId);
    var p = def.params || {};
    switch (def.family) {
      case 'arrow':
        if (typeId === 'pincer') return pincerRings(pts, p);
        return [arrowFromTwoPoints(pts[0], pts[pts.length - 1], p)];
      case 'band': return bandRings(pts, p);
      case 'fan': return [fanRing(pts[0], p)];
      case 'zone': return [zoneRing(pts)];
      case 'line': return [pts.slice()];
      default: throw new Error('未知几何族：' + def.family);
    }
  }

  // ---------------------------------------------------------------------
  // 场景写入
  // ---------------------------------------------------------------------

  /** 一个标绘实例 = 若干 Cesium 实体（环/线 + 可选标注）。 */
  function drawOne(rec) {
    if (!viewer) return 0;
    var def = CATALOGUE[rec.type];
    var rings = ringsFor(rec.type, rec.points);
    var color = Cesium.Color.fromCssColorString(rec.color || COLOR.red);
    var isLine = def.family === 'line';
    var n = 0;

    rings.forEach(function (ring) {
      var positions = ring.map(function (c) { return Cesium.Cartesian3.fromDegrees(c[0], c[1]); });
      if (isLine) {
        rec.entities.push(viewer.entities.add({
          polyline: {
            positions: positions,
            width: 3,
            material: new Cesium.PolylineDashMaterialProperty({
              color: color, dashLength: 16,
            }),
            clampToGround: true,
          },
        }));
      } else {
        rec.entities.push(viewer.entities.add({
          polygon: {
            hierarchy: new Cesium.PolygonHierarchy(positions),
            material: color.withAlpha(def.params && def.params.hatch ? 0.30 : 0.38),
            outline: true,
            outlineColor: color,
            outlineWidth: 2,
            classificationType: Cesium.ClassificationType.BOTH,
          },
        }));
      }
      n++;
    });

    if (rec.label) {
      var mid = rec.points[Math.floor(rec.points.length / 2)];
      rec.entities.push(viewer.entities.add({
        position: Cesium.Cartesian3.fromDegrees(mid[0], mid[1]),
        label: {
          text: rec.label,
          font: 'bold 14px "Microsoft YaHei", sans-serif',
          fillColor: Cesium.Color.WHITE,
          outlineColor: Cesium.Color.fromCssColorString('#0f1720'),
          outlineWidth: 3,
          style: Cesium.LabelStyle.FILL_AND_OUTLINE,
          verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
      }));
    }
    return n;
  }

  function clearRec(rec) {
    if (!viewer) return;
    rec.entities.forEach(function (e) {
      try { viewer.entities.remove(e); } catch (err) { /* 已随场景销毁 */ }
    });
    rec.entities.length = 0;
  }

  // ---------------------------------------------------------------------
  // 对外接口
  // ---------------------------------------------------------------------

  /**
   * 增加一个标绘。`spec = {id?, type, layer, points, color?, label?}`。
   *
   * `layer` 必须是 LayerRegistry 里 kind:'plot' 的图层 id —— 标绘实例归属某个标绘
   * 类别图层，显隐随该图层走，不自己管可见性。
   */
  function add(spec) {
    var def = CATALOGUE[spec.type];
    if (!def) throw new Error('未知标绘类型：' + spec.type);
    if (!spec.points || spec.points.length < 2) throw new Error('标绘至少需要 2 个控制点');
    var id = spec.id || ('plot-' + (++seq));
    if (byId[id]) throw new Error('标绘 id 重复：' + id);
    var rec = {
      id: id, type: spec.type, layer: spec.layer || 'plot.offense',
      std: def.std, ref: def.ref,
      points: spec.points.map(function (c) { return [c[0], c[1]]; }),
      color: spec.color || (spec.hostility === 'blue' ? COLOR.blue : COLOR.red),
      label: spec.label || null,
      entities: [],
    };
    byId[id] = rec;
    drawOne(rec);
    applyVisibility();
    return rec;
  }

  function remove(id) {
    var rec = byId[id];
    if (!rec) return false;
    clearRec(rec);
    delete byId[id];
    return true;
  }

  function clearLayer(layerId) {
    Object.keys(byId).forEach(function (id) {
      if (!layerId || byId[id].layer === layerId) remove(id);
    });
  }

  /** 由 LayerRegistry 的 plot sink 调用：切换某个标绘类别图层的显隐。 */
  function setLayerVisible(layerId, on) {
    layerVisible[layerId] = !!on;
    applyVisibility();
  }

  function applyVisibility() {
    Object.keys(byId).forEach(function (id) {
      var rec = byId[id];
      var on = layerVisible[rec.layer] !== false;   // 未声明过就默认可见
      rec.entities.forEach(function (e) { e.show = on; });
    });
  }

  /** 按图形目录整体导出「标准来源」清单，供文档/界面标注。 */
  function catalogue() {
    return Object.keys(CATALOGUE).map(function (k) {
      var d = CATALOGUE[k];
      return { type: k, zh: d.zh, family: d.family, std: d.std, ref: d.ref, us: d.us || null };
    });
  }

  window.Plotting = {
    CATALOGUE: CATALOGUE,
    catalogue: catalogue,
    bind: bind,
    add: add,
    remove: remove,
    clearLayer: clearLayer,
    setLayerVisible: setLayerVisible,
    // 供测试直接验证几何族（不依赖 Cesium 场景）
    _geometry: {
      arrowFromTwoPoints: arrowFromTwoPoints,
      pincerRings: pincerRings,
      bandRings: bandRings,
      fanRing: fanRing,
      ringsFor: ringsFor,
    },
    debug: function () {
      var list = Object.keys(byId).map(function (id) {
        return { id: id, type: byId[id].type, layer: byId[id].layer,
                 std: byId[id].std, entities: byId[id].entities.length };
      });
      return { bound: !!viewer, visible: layerVisible, count: list.length, items: list };
    },
  };
})();