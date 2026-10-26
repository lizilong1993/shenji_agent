/**
 * 目标 2D 图标 ⇄ 3D 模型切换（按视点高度）与**型号 → 资产**解析。
 *
 * 2026-10-20 重构：资产从"按平台大类的低模"升级为"按具体型号的高模"，
 * 并保留大类回退。资产来源见 web/assets/models/CREDITS.md：
 *   - Sketchfab CC-BY / Free Standard 实机模型（054A/071/Perry/99A/M1A2/
 *     F-16C/歼-16/阵风/E-2D/直-10/MQ-9/PL-03 等），经 Blender 减面、贴图压缩、
 *     按公开全长缩放到 **1 单位 = 1 米**；
 *   - 无公开资产的型号（056A/903/磐石/沱江/基隆/红旗-16/鹰击-62）由
 *     tools/modelgen/build_custom.py 按公开尺寸自建。
 *
 * 因此：**所有模型 scale 都是 1、机头/舰艏在 Blender 里一律 +Y**。导出后
 * 实测 Cesium 1.145 的局部轴映射为 glTF +Z→局部 +X、glTF +X→局部 +Y，而
 * Blender +Y 导出为 glTF -Z，所以模型在 Cesium 局部系的前向是 **-X**
 * （forwardAxis() 返回 '-X'）。旧版的尺度表与逐资产轴表已删除。
 *
 * 选型规则（用户 2026-10-20）：
 *   1. 先用 def.model（如"052D 型驱逐舰"）匹配精确型号；
 *   2. 没有精确资产就回退到 formClassOf() 得到的**大类模型**；
 *   3. 连大类也判不出 -> null，保留 2D 图标。
 *
 * 判据 decide() 仍是纯函数，tools/entity_lod_test.js 直接跑 Node 用例。
 * 阈值与滞回见 DEFAULTS：低于 3000 m 进 3D，高于 3600 m 回 2D（20% 迟滞）。
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
   * - 高度非法（NaN/undefined/负数）时保持当前形态；
   * - 滞回：已在 3D 时要高于 exit3d 才回 2D；已在 2D 时要低于 enter3d 才去 3D。
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

  /** 大类 -> 资产。**每个大类都必须有回退**，新类别进来时靠它兜底。 */
  var MODEL_MAP = {
    // 水面
    destroyer: './assets/models/type_052c.glb',
    frigate: './assets/models/type_054a.glb',
    corvette: './assets/models/type_056a.glb',
    patrol: './assets/models/type_056a.glb',
    amphibious: './assets/models/type_071.glb',
    replenishment: './assets/models/type_903.glb',
    boat_speed: './assets/models/type_056a.glb',
    // 地面
    tank: './assets/models/type_ztz99a.glb',
    light_tank: './assets/models/type_ztq15.glb',
    amphibious_vehicle: './assets/models/type_ztd05.glb',
    spg: './assets/models/type_plz05.glb',
    howitzer: './assets/models/type_m777.glb',
    mlrs: './assets/models/type_phl03.glb',
    sam: './assets/models/type_hq9.glb',
    anti_ship: './assets/models/type_yj62.glb',
    truck: './assets/models/type_eq240.glb',
    // 空中
    fighter: './assets/models/type_j16.glb',
    aew: './assets/models/type_kj2000.glb',
    helicopter: './assets/models/type_z10.glb',
    uav: './assets/models/type_mq9.glb',
  };

  /**
   * 精确型号 -> 资产。匹配对象是 def.model（缺失时再看 platform_class + name）。
   * 顺序敏感：更具体的写在前面。注释里的"近似"是公开资产缺失时的同族替代，
   * 不是同型号——换资产时请同步改这里。
   */
  var TYPE_RULES = [
    // 水面舰
    [/054a/i, 'type_054a'],
    [/903|综合补给/i, 'type_903'],
    [/075|两栖攻击|lhd/i, 'type_075'],
    [/071|船坞登陆/i, 'type_071'],
    [/055|renhai|南昌/i, 'type_055'],
    [/056a|轻型护卫|江岛/i, 'type_056a'],
    [/成功级|perry|佩里/i, 'type_perry'],
    [/沱江|tuo.?chiang|ta.?chiang/i, 'type_tuochiang'],
    [/基隆|kidd/i, 'type_kidd'],
    [/磐石|panshi/i, 'type_panshi'],
    // 052D 无公开可下载资产，用同族 052C 近似（文中注释、README 同步说明）
    [/052d|luyang.?iii/i, 'type_052c'],
    // 地面
    [/15\s*式|ztq-?15/i, 'type_ztq15'],
    [/99a|ztz-?99/i, 'type_ztz99a'],
    [/m1a2/i, 'type_m1a2'],
    [/m60a?3/i, 'type_m60a3'],
    [/cm-?11|m48h/i, 'type_m60a3'],
    [/m48/i, 'type_m48a5'],
    [/05\s*式|ztd-?05|zbd-?05/i, 'type_ztd05'],
    [/plz-?05/i, 'type_plz05'],
    [/m109/i, 'type_m109a6'],
    [/m777/i, 'type_m777'],
    [/phl-?03/i, 'type_phl03'],
    [/红旗-?9|hq-?9/i, 'type_hq9'],
    [/红旗-?16|hq-?16|ly-?80/i, 'type_hq16'],
    [/爱国者|patriot|pac-?3/i, 'type_pac3'],
    [/鹰击-?62|yj-?62/i, 'type_yj62'],
    [/himars|m142|海马斯/i, 'type_himars'],
    [/运输|车队|ural|卡车|truck|eq240|东风/i, 'type_eq240'],
    // 空中
    [/歼-?20|j-?20(?!\d)/i, 'type_j20'],
    [/歼-?16|j-?16|su-?30/i, 'type_j16'],
    [/f-?16/i, 'type_f16c'],
    [/idf|经国|f-?ck-?1/i, 'type_idf'],
    [/阵风|rafale/i, 'type_rafale'],
    [/kj-?2000|空警-?2000/i, 'type_kj2000'],
    [/空警-?500|kj-?500/i, 'type_kj500'],
    [/p-?3c|p-?3\b|orion|猎户座/i, 'type_p3c'],
    [/e-?2|鹰眼|hawkeye/i, 'type_e2d'],
    [/ah-?64|apache|阿帕奇/i, 'type_ah64e'],
    [/直-?10|z-?10/i, 'type_z10'],
    [/无人机|uav|drone|mq-?9|翼龙|彩虹/i, 'type_mq9'],
    [/直-?9|z-?9|米-?17|mi-?17|黑鹰|uh-?60/i, 'type_z10'],
  ];

  function normText(s) {
    return String(s || '').toLowerCase();
  }

  /** 精确型号资产；无匹配返回 null。 */
  function exactModelUrl(def) {
    if (!def || typeof def !== 'object') return null;
    var text = normText(def.model) + ' ' + normText(def.platform_class) + ' ' +
      normText(def.name) + ' ' + normText(def.icon);
    for (var i = 0; i < TYPE_RULES.length; i++) {
      if (TYPE_RULES[i][0].test(text)) return './assets/models/' + TYPE_RULES[i][1] + '.glb';
    }
    return null;
  }

  /**
   * 解析资产 URL。参数可以是：
   *   - def 对象：精确型号 -> 大类 -> null；
   *   - 大类字符串（旧调用/测试）：直接查 MODEL_MAP。
   */
  function modelUrlFor(arg) {
    if (typeof arg === 'string') return MODEL_MAP[arg] || null;
    var exact = exactModelUrl(arg);
    if (exact) return exact;
    return MODEL_MAP[formClassOf(arg)] || null;
  }

  /**
   * 运行时缩放：资产都按米归一，所以恒为 1。保留函数是为了 app.js 的调用契约
   * （返回 undefined 时不写 scale 属性）。
   */
  function scaleFor(arg) {
    if (typeof arg === 'string') return MODEL_MAP[arg] ? 1 : null;
    if (!arg) return null;
    return modelUrlFor(arg) ? 1 : null;
  }

  /**
   * 由实体定义推断**大类**（资产回退用）。只做保守映射：拿不准返回 null。
   */
  function formClassOf(def) {
    if (!def) return null;
    if (def.form && MODEL_MAP[def.form]) return def.form;
    var d = String(def.domain || '').toLowerCase();
    var p = String(def.platform_class || '') + ' ' + String(def.icon || '') + ' ' +
      String(def.model || '');

    if (d === 'air') {
      // 无人机要先于预警机判：实测部分无人机数据的 icon 是 aew。
      if (/无人机|uav|drone|侦察机/i.test(p)) return 'uav';
      if (/预警|aew|awacs/i.test(p)) return 'aew';
      if (/直升机|helicopter|rotor|旋翼/i.test(p)) return 'helicopter';
      if (/战斗机|歼击机|fighter|air.?superiority|多用途|阵风|rafale/i.test(p)) return 'fighter';
      return null;
    }
    if (d === 'land') {
      // 岸舰导弹必须在"防空"之前判，否则"导弹"会把它截走。
      if (/岸舰|岸防|coastal|anti.?ship/i.test(p)) return 'anti_ship';
      if (/防空|地空|sam|surface.?to.?air|爱国者|红旗/i.test(p)) return 'sam';
      if (/火箭炮|mlrs|phl/i.test(p)) return 'mlrs';
      if (/榴弹炮|howitzer|m777/i.test(p)) return 'howitzer';
      if (/炮兵|火炮|artillery|自行炮|plz/i.test(p)) return 'spg';
      if (/两栖战车|两栖突击|步战|步兵战车|apc|ifv|aav|zbd|ztd/i.test(p)) return 'amphibious_vehicle';
      if (/运输|车队|卡车|truck|logistic/i.test(p)) return 'truck';
      if (/15\s*式|轻型坦克|light.?tank|ztq/i.test(p)) return 'light_tank';
      if (/装甲|坦克|armor|tank|机步/i.test(p)) return 'tank';
      return null;
    }
    if (d !== 'sea' && d !== 'naval' && d !== 'maritime') return null;
    // 登陆/两栖必须先于补给判：实测数据里 071 的 icon 是 "replenishment"。
    if (/登陆|两栖|landing|amphibious|071/i.test(p)) return 'amphibious';
    if (/补给|replenish|oiler|903/i.test(p)) return 'replenishment';
    if (/巡逻|轻护|导弹艇|corvette|patrol|056|沱江/i.test(p)) return 'corvette';
    if (/驱逐|destroyer/i.test(p)) return 'destroyer';
    if (/护卫|frigate/i.test(p)) return 'frigate';
    if (/舰|ship|艇|boat/i.test(p)) return 'frigate';
    return null;
  }

  /**
   * 资产生成时艏/头朝 Blender +Y。实测 Cesium 1.145 的模型局部轴映射是
   * glTF +Z -> 局部 +X（东）、glTF +X -> 局部 +Y（北），而 Blender +Y 导出为
   * glTF -Z，所以**模型艏/头落在 Cesium 局部 -X**。
   * targetFx/targetMotion 用这张表把"局部前向"对到航向；必须返回 '-X'，
   * 否则所有目标会横着走（2026-10-20 用轴向测试 GLB 在 8848 上实测）。
   */
  function forwardAxis() {
    return '-X';
  }

  /**
   * 资产自身艏/头朝向修正（度，绕模型竖轴；正=顺时针）。
   *
   * 归一化管线按“Blender +Y 为头”导出，但个别 Sketchfab/自建资产导入时
   * 车头落在 Blender -Y（导出后 glTF +Z），与统一约定相反。实测：
   *   · type_ztd05（ZTD-05）：炮管朝 +Z；
   *   · type_eq240（东风 EQ240）：驾驶室朝 +Z。
   * 两者在 Cesium 局部系里都对应 +X，因此姿态需要额外转 180°。
   * 其余资产保持 0；新增资产如看着倒着走，在这里登记 URL 即可。
   */
  var MODEL_YAW_FIX_DEG = {
    './assets/models/type_ztd05.glb': 180,
    './assets/models/type_eq240.glb': 180,
    './assets/models/type_p3c.glb': 180,
  };

  function modelYawOffset(def) {
    var url = exactModelUrl(def) || modelUrlFor(def);
    return (url && MODEL_YAW_FIX_DEG[url]) || 0;
  }

  /**
   * 高频目标的 LOD1（1.2–3 km 相机高度）。只登记确实做了低模的资产；
   * 其余返回 null，运行时继续用 LOD0。LOD1 与 LOD0 同朝向、同原点。
   */
  var LOD1_MAP = {
    './assets/models/type_ztd05.glb': './assets/models/lod1_ztd05.glb',
    './assets/models/type_eq240.glb': './assets/models/lod1_eq240.glb',
    './assets/models/type_perry.glb': './assets/models/lod1_perry.glb',
    './assets/models/type_j16.glb': './assets/models/lod1_j16.glb',
    './assets/models/type_kj2000.glb': './assets/models/lod1_kj2000.glb',
    './assets/models/type_kj500.glb': './assets/models/lod1_kj500.glb',
    './assets/models/type_e2d.glb': './assets/models/lod1_e2d.glb',
    './assets/models/type_052c.glb': './assets/models/lod1_052c.glb',
  };

  function lod1UrlFor(def) {
    var url = modelUrlFor(def);
    return (url && LOD1_MAP[url]) || null;
  }

  /** 新资产没有可命名的轮节点（轮子烘焙在车身网格里），返回空表。 */
  function wheelNodes() {
    return [];
  }

  // 兼容旧引用：这两张表已不再参与计算（资产是米制、scale=1）。
  // targetFx 的尾流/尾气仍按大类取长度；逐型号长度在资产管线里已归一，
  // 这里给各大类的代表值（米），避免尾流起点落在船体/机身内部。
  var REAL_LENGTH_M = {
    destroyer: 155, frigate: 134, corvette: 89, patrol: 89,
    amphibious: 210, replenishment: 178, boat_speed: 15,
    tank: 11, light_tank: 11, amphibious_vehicle: 9.5, spg: 11.6,
    howitzer: 10.2, mlrs: 11, sam: 12, anti_ship: 13, truck: 6.9,
    fighter: 21.9, aew: 46.6, helicopter: 14.15, uav: 11,
  };
  var MODEL_UNITS = {};
  var FORWARD_AXIS = {};
  var WHEEL_NODES = {};

  return {
    MODE_ICON: MODE_ICON,
    MODE_MODEL: MODE_MODEL,
    DEFAULTS: DEFAULTS,
    decide: decide,
    modelUrlFor: modelUrlFor,
    exactModelUrl: exactModelUrl,
    scaleFor: scaleFor,
    formClassOf: formClassOf,
    forwardAxis: forwardAxis,
    modelYawOffset: modelYawOffset,
    lod1UrlFor: lod1UrlFor,
    wheelNodes: wheelNodes,
    MODEL_MAP: MODEL_MAP,
    TYPE_RULES: TYPE_RULES,
    REAL_LENGTH_M: REAL_LENGTH_M,
    MODEL_UNITS: MODEL_UNITS,
    FORWARD_AXIS: FORWARD_AXIS,
    WHEEL_NODES: WHEEL_NODES,
  };
}));
