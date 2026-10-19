/**
 * 目标（车/船/飞机）的 2D 图标 ⇄ 3D 模型切换（按视点高度）。
 *
 * 需求（用户 2026-10-10，2026-10-19 修订阈值）：
 *   - 2D 图标只在视点高度**高于**阈值时显示；
 *   - 高度低于约 **3000 m** 后，把 2D 图标换成 3D 模型。
 *
 * 为什么单独一个模块、而不写进 app.js
 * ----------------------------------
 * 切换判据是**纯函数**（高度 + 当前形态 -> 目标形态），把它与 Cesium 场景隔离后
 * 就能用 Node 直接跑用例（本机无前端测试框架）。tools/entity_lod_test.js 就是
 * 这套用例；decide() 里不允许出现任何 Cesium 依赖。
 *
 * 阈值与滞回
 * ----------
 *   - **下降**到 3000 m 及以下 -> 转 3D；
 *   - **上升**须超过 3600 m -> 才转回 2D（20% 迟滞）。
 * 两个阈值（而非一个）是为了**滞回**：单阈值时相机在阈值附近抖动会让形态反复
 * 切换，而 headless 软渲染下 rAF 只有约 2.7 fps，每次切换都要装卸模型，代价尤其高。
 *
 * 3000 m 是用户给出的初值（"低于约 3000 米"），仍属**可调参数**：改就改这里的
 * 常量，别散落到调用点；entity_lod_test.js 用 DEFAULTS 本身做边界断言。
 *
 * 跨栈提醒：本模块只管「形态」，不管图层顺序。标绘/目标的 z 序在
 * layerRegistry.js（见那里的文件头：Cesium 没有跨栈的统一 z）。
 */
(function (root, factory) {
  var api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.EntityLod = api;
}(typeof window !== 'undefined' ? window : null, function () {
  'use strict';

  var DEFAULTS = {
    enter3dHeight: 3000,
    exit3dHeight: 3600,
  };

  var MODE_ICON = 'icon';
  var MODE_MODEL = 'model';

  /**
   * 纯判据：给定高度与当前形态，返回**应该**处于的形态。
   *
   * 设计要点：
   *   - 高度非法（NaN/undefined/负数）时**保持当前形态**，不抛异常也不乱切——
   *     相机尚未初始化时 height 可能是 undefined，那时切到 3D 是错的。
   *   - 滞回：已在 3D 时要**高于** exit3d 才回 2D；已在 2D 时要**低于** enter3d
   *     才去 3D。两个条件都严格，避免等高时反复。
   */
  function decide(heightMeters, currentMode, cfg) {
    var c = cfg || DEFAULTS;
    var cur = (currentMode === MODE_MODEL) ? MODE_MODEL : MODE_ICON;

    if (typeof heightMeters !== 'number' || !isFinite(heightMeters) || heightMeters < 0) {
      return { mode: cur, changed: false, reason: 'height-invalid' };
    }

    if (cur === MODE_MODEL) {
      if (heightMeters > c.exit3dHeight) {
        return { mode: MODE_ICON, changed: true, reason: 'rose-above-exit3d' };
      }
      return { mode: MODE_MODEL, changed: false, reason: 'within-hysteresis' };
    }

    if (heightMeters <= c.enter3dHeight) {
      return { mode: MODE_MODEL, changed: true, reason: 'descended-to-enter3d' };
    }
    return { mode: MODE_ICON, changed: false, reason: 'above-enter3d' };
  }

  /**
   * 目标类型 -> 3D 模型 URL。**找不到就返回 null**，调用点据此保留 2D 图标。
   *
   * 2026-10-19 起覆盖海/陆/空全部 15 类目标中的 13 个形态：
   *   · 海面：Kenney Watercraft Kit（CC0，既有 5 个模型 + 巡逻舰复用快艇）
   *   · 坦克：Quaternius Animated Tanks（CC0，含行进/转向动画）
   *   · 战斗机：OpenGameArt "Mig"（CC0，由 .blend 转 glb）
   *   · 预警机/直升机/装甲车/自行火炮/防空发射车/岸舰导弹车：由
   *     `tools/modelgen/build_models.py` 程序化生成（Blender 无头，米制建模，
   *     旋翼/雷达带 glTF 动画）。许可证见 web/assets/models/LICENSE-*。
   *
   * 映射只按「形态类别」而非具体型号：实体带真实型号参数（052D、054A…），但这些
   * CC0/低模资产不足以表达具体型号；型号级替换是后续「重点细化」的范畴。
   */
  var MODEL_MAP = {
    // 海面（Kenney CC0 水运套件，既有）
    ship_large: './assets/models/ship-large.glb',
    ship_medium: './assets/models/ship-small.glb',
    ship_cargo: './assets/models/ship-cargo-a.glb',
    boat_speed: './assets/models/boat-speed-a.glb',
    boat_tug: './assets/models/boat-tug-a.glb',
    patrol: './assets/models/boat-speed-a.glb',
    // 地面/空中（2026-10-19 新增：Quaternius CC0 坦克 + OGA CC0 MiG-15 +
    // tools/modelgen 程序化生成的同风格低模。许可证见 web/assets/models/LICENSE-*）
    fighter: './assets/models/fighter.glb',
    aew: './assets/models/awacs.glb',
    helicopter: './assets/models/helicopter.glb',
    armor: './assets/models/armor.glb',
    apc: './assets/models/apc.glb',
    artillery: './assets/models/artillery.glb',
    sam: './assets/models/sam_launcher.glb',
    coastal_missile: './assets/models/coastal_missile.glb',
  };

  function modelUrlFor(formClass) {
    return MODEL_MAP[formClass] || null;
  }

  /**
   * 各形态的**真实长度（米）**，以及资产的最长轴（**单位**，实测自 GLB 包围盒）。
   *
   * 为什么需要这张表：Kenney 这类 low-poly 包**不按米制建模**——`ship-large.glb`
   * 的最长轴只有 **13.1 单位**。Cesium 的 `Entity.model` 默认 **1 单位 = 1 米**，
   * 所以不设 `scale` 时，一艘驱逐舰会被当成 **13.1 m** 的船（真实约 155 m，差一个
   * 数量级）。这会让模型小到 24 px 以下，于是被 `minimumPixelSize` **恒定夹在 24 px**，
   * 「按高度切换 2D/3D」在视觉上就失去意义——阈值也就无从定稿。
   *
   * 所以：**按形态给出真实长度，除以资产单位长度得到 scale**。数值来源是公开的
   * 舰艇全长量级（取整到 5 m），**是量级而非精确型别尺寸**——本批资产是民用
   * low-poly，本来也表达不了具体型号（见 MODEL_MAP 注释）。
   */
  var REAL_LENGTH_M = {
    ship_large: 155,    // 驱逐舰量级（052D 约 157 m）
    ship_medium: 135,   // 护卫舰量级（054A 约 134 m）
    ship_cargo: 130,    // 补给/运输舰量级（登陆舰同量级）
    boat_speed: 35,     // 导弹艇量级（沱江级约 65 m，快艇更短，取 35）
    boat_tug: 30,       // 拖船
    patrol: 65,         // 巡逻舰/轻护（沱江级约 65 m）
    // 地面/空中：程序化模型按米建模，导入资产按公开量级取整
    fighter: 10.1,      // MiG-15 全长 10.1 m
    aew: 46.6,          // E-3 量级全长
    helicopter: 17.7,   // AH-64 量级全长（含旋翼）
    armor: 9.5,         // 主战坦克含炮管量级
    apc: 7.7,           // 轮式装甲车
    artillery: 11.0,    // 自行火炮含炮管
    sam: 12.0,          // 防空导弹发射车
    coastal_missile: 10.4, // 岸舰导弹发射车
  };

  // 资产最长轴（单位）。**实测值**，来自解 GLB accessor 的 min/max。
  // 换了资产必须同步更新，否则尺度会错——entity_lod_test.js 会校验这张表与
  // MODEL_MAP 的键集合一致。
  var MODEL_UNITS = {
    ship_large: 13.1,
    ship_medium: 10.6,
    ship_cargo: 10.5,
    boat_speed: 3.9,
    boat_tug: 3.5,
    patrol: 3.9,          // 与 boat_speed 同一资产
    // 2026-10-19 实测（Blender 导出的世界包围盒最长轴，单位）
    fighter: 3.75,
    aew: 40.7,
    helicopter: 16.78,
    armor: 17.216,
    apc: 7.7,
    artillery: 11.007,
    sam: 10.575,
    coastal_missile: 10.4,
  };

  // ---- 运动姿态 / 车轮节点（2026-10-19，CZML 运动改造） ------------------
  //
  // **模型机头在 Cesium 局部系里的轴向**。Cesium 把 glTF 的 Y-up 转成 Z-up：
  // glTF -Z → 局部 +Y，glTF +X → +X，glTF +Z → -Y。三批资产的建模/导出习惯
  // 不同，不能套同一个"前方"：
  //   · 程序化 Blender 资产（build_models.py）：机身/炮管沿 Blender +Y → glTF -Z
  //     → 局部 **+Y**；
  //   · Kenney Watercraft（UnityGLTF）：船头沿 glTF +Z → 局部 **-Y**；
  //   · OGA MiG-15：机身沿 glTF +X → 局部 **+X**；
  //   · Quaternius 坦克（FBX2glTF）：车身长轴沿局部 **+X**。
  // 这个轴只影响"姿态四元数如何把机头对上航向"，不影响缩放表。
  var FORWARD_AXIS = {
    // 2026-10-19 逐资产标定（俯视、heading=90°，东=屏幕右）：
    //   · Kenney 船：船头在局部 +X；
    //   · OGA MiG：机头在局部 -X；
    //   · Quaternius 坦克：炮管/车头在局部 -Y；
    //   · Blender 程序化资产（build_models.py：APC/火炮/防空车/岸舰车/预警机/直升机）：
    //     车头/机头在局部 -X。
    // 这张表由 tools 下的轴向标定页逐项截图确认；换资产必须重新标定。
    ship_large: '+X', ship_medium: '+X', ship_cargo: '+X',
    boat_speed: '+X', boat_tug: '+X', patrol: '+X',
    fighter: '-X', aew: '-X', helicopter: '-X',
    armor: '-Y', apc: '-X', artillery: '-X',
    sam: '-X', coastal_missile: '-X',
  };

  /** 轮式车辆的轮节点（GLB 节点名与局部旋转轴/半径）。履带式没有轮节点。 */
  var WHEEL_NODES = {
    apc: { prefix: 'wheel', count: 8, axis: 'Y', radius: 0.72 },
    sam: { prefix: 'wheel', count: 6, axis: 'Y', radius: 0.75 },
    coastal_missile: { prefix: 'wheel', count: 6, axis: 'Y', radius: 0.75 },
    artillery: { prefix: 'roadwheel', count: 8, axis: 'Y', radius: 0.42 },
  };

  function forwardAxis(formClass) {
    return FORWARD_AXIS[formClass] || '+Y';
  }

  /**
   * 轮节点列表（GLB 里轮子的实际节点名）。程序化导出把第一个命名为
   * `wheel`，后续是 `wheel.001` … `wheel.007`，沿用这个命名规则生成。
   */
  function wheelNodes(formClass) {
    var spec = WHEEL_NODES[formClass];
    if (!spec) return [];
    var out = [];
    for (var i = 0; i < spec.count; i++) {
      out.push({
        name: i === 0 ? spec.prefix : spec.prefix + '.' + ('00' + i).slice(-3),
        axis: spec.axis,
        radius: spec.radius,
      });
    }
    return out;
  }

  /** 该形态模型应设的 scale；缺任一侧数据时返回 null（调用点不加 scale）。 */
  function scaleFor(formClass) {
    var real = REAL_LENGTH_M[formClass];
    var units = MODEL_UNITS[formClass];
    if (!real || !units) return null;
    return real / units;
  }

  /**
   * 由实体定义推断形态类别。**只做保守映射**：拿不准就返回 null（保留 2D）。
   *
   * def.form 若由数据显式给出则优先用它；否则按 domain + 平台类别粗判。
   */
  function formClassOf(def) {
    if (!def) return null;
    if (def.form && MODEL_MAP[def.form]) return def.form;
    var d = String(def.domain || '').toLowerCase();
    var p = String(def.platform_class || '') + String(def.icon || '');

    // 空中（2026-10-19 起有资产）。判序：预警机（圆盘）→ 直升机（旋翼）→ 战斗机。
    if (d === 'air') {
      if (/预警|aew|awacs/i.test(p)) return 'aew';
      if (/直升机|helicopter|rotor|武装直升机/i.test(p)) return 'helicopter';
      if (/战斗机|歼击机|fighter|air.?superiority|多用途/i.test(p)) return 'fighter';
      return null;
    }
    // 地面：岸舰导弹必须排在"导弹/防空"之前判，否则会被 SAM 的规则截走。
    if (d === 'land') {
      if (/岸舰|岸防|coastal|anti.?ship/i.test(p)) return 'coastal_missile';
      if (/防空|地空|sam|surface.?to.?air/i.test(p)) return 'sam';
      if (/炮兵|火炮|artillery|howitzer/i.test(p)) return 'artillery';
      if (/两栖战车|步战|步兵战车|apc|ifv|amphibious/i.test(p)) return 'apc';
      if (/装甲|坦克|armor|tank|机步/i.test(p)) return 'armor';
      return null;
    }
    if (d !== 'sea' && d !== 'naval' && d !== 'maritime') return null;
    if (/货|补给|cargo|transport|登陆|两栖舰|landing/i.test(p)) return 'ship_cargo';
    if (/快艇|导弹艇|patrol.?boat|missile.?boat/i.test(p)) return 'boat_speed';
    if (/拖|tug/i.test(p)) return 'boat_tug';
    // 实测（??case=taiwan_landing，31 个目标）拿到的 platform_class 只有「驱逐舰/护卫舰/
    // 补给舰」这类**类别词**，没有「052D」这种型号串——所以按类别词给形态：驱逐舰取大舰，
    // 护卫舰取中舰。判具体型号要等资产能表达型号（见 MODEL_MAP 注释）。
    if (/驱逐舰|destroyer/i.test(p)) return 'ship_large';
    if (/护卫舰|巡防舰|frigate|corvette/i.test(p)) return 'ship_medium';
    if (/巡逻舰|patrol|轻护/i.test(p)) return 'patrol';
    if (/舰|ship/i.test(p)) return /大|large|基隆/i.test(p) ? 'ship_large' : 'ship_medium';
    return null;
  }

  return {
    MODE_ICON: MODE_ICON,
    MODE_MODEL: MODE_MODEL,
    DEFAULTS: DEFAULTS,
    decide: decide,
    modelUrlFor: modelUrlFor,
    scaleFor: scaleFor,
    REAL_LENGTH_M: REAL_LENGTH_M,
    MODEL_UNITS: MODEL_UNITS,
    formClassOf: formClassOf,
    forwardAxis: forwardAxis,
    wheelNodes: wheelNodes,
    FORWARD_AXIS: FORWARD_AXIS,
    WHEEL_NODES: WHEEL_NODES,
    MODEL_MAP: MODEL_MAP,
  };
}));
