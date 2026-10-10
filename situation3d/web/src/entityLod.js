/**
 * 车/船的 2D 图标 ⇄ 3D 模型切换（按视点高度）。
 *
 * 需求（用户 2026-10-10）：
 *   - 2D 图标只在鸟瞰图与主视图中、**视点高度超过阈值**时显示；
 *   - 主视图高度低于阈值后，把 2D 图标换成 3D 模型。
 *
 * 为什么单独一个模块、而不写进 app.js
 * ----------------------------------
 * 切换判据是**纯函数**（高度 + 当前形态 -> 目标形态），把它与 Cesium 场景隔离后
 * 就能用 Node 直接跑用例（本机无前端测试框架）。tools/entity_lod_test.js 就是
 * 这套用例；decide() 里不允许出现任何 Cesium 依赖。
 *
 * 阈值与滞回
 * ----------
 *   - **下降**跨过 20 km -> 转 3D；
 *   - **上升**须超过 24 km -> 才转回 2D。
 * 两个阈值（而非一个）是为了**滞回**：单阈值时相机在阈值附近抖动会让形态反复
 * 切换，而 headless 软渲染下 rAF 只有约 2.7 fps（见 tools/consistency_selftest.py），
 * 每次切换都要装卸模型，代价尤其高。
 *
 * 这两个数是**设计初值，不是实测结论**——依据是 places.js 最近档为 60 km、
 * 建筑体块在 <=60 km 进入 close 档；实现后须用探针截图核对「20 km 时模型的屏幕
 * 占比是否足以辨识」再定稿。所以要改就改这里的常量，别散落到调用点。
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
    enter3dHeight: 20000,
    exit3dHeight: 24000,
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
   * 目前接入的是任务三落地的 8 个 CC0 船模型（Kenney Watercraft Kit，见
   * web/assets/models/Kenney_WatercraftKit_License.txt）。**车辆尚无资产**——
   * 这是有意留白：资产路线（补 CC0 军用模型 / 程序化低模）未定，先不做假映射。
   *
   * 映射只按「形态类别」而非具体型号：我们的实体带真实型号参数（052D、054A…），
   * 但这批 CC0 资产是民用 low-poly，**不足以表达具体型号**。等资产路线定了再细化。
   */
  var MODEL_MAP = {
    ship_large: './assets/models/ship-large.glb',
    ship_medium: './assets/models/ship-small.glb',
    ship_cargo: './assets/models/ship-cargo-a.glb',
    boat_speed: './assets/models/boat-speed-a.glb',
    boat_tug: './assets/models/boat-tug-a.glb',
  };

  function modelUrlFor(formClass) {
    return MODEL_MAP[formClass] || null;
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
    if (d !== 'sea' && d !== 'naval' && d !== 'maritime') return null;
    var p = String(def.platform_class || '') + String(def.icon || '');
    if (/货|补给|cargo|transport/i.test(p)) return 'ship_cargo';
    if (/快艇|导弹艇|patrol|missile/i.test(p)) return 'boat_speed';
    if (/拖|tug/i.test(p)) return 'boat_tug';
    // 实测（??case=taiwan_landing，31 个目标）拿到的 platform_class 只有「驱逐舰/护卫舰/
    // 补给舰」这类**类别词**，没有「052D」这种型号串——所以按类别词给形态：驱逐舰取大舰，
    // 护卫舰取中舰。判具体型号要等资产能表达型号（见 MODEL_MAP 注释）。
    if (/驱逐舰|destroyer/i.test(p)) return 'ship_large';
    if (/护卫舰|巡防舰|frigate|corvette/i.test(p)) return 'ship_medium';
    if (/舰|ship/i.test(p)) return /大|large|基隆/i.test(p) ? 'ship_large' : 'ship_medium';
    return null;
  }

  return {
    MODE_ICON: MODE_ICON,
    MODE_MODEL: MODE_MODEL,
    DEFAULTS: DEFAULTS,
    decide: decide,
    modelUrlFor: modelUrlFor,
    formClassOf: formClassOf,
    MODEL_MAP: MODEL_MAP,
  };
}));
