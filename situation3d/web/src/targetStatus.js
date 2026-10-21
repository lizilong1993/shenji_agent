/* situation3d — 目标战损状态唯一来源
 *
 * 为什么需要这个模块
 * ------------------
 * 「整体TS」和「鸟瞰图」都要回答同一个问题：**这个目标现在还能不能打**。
 * 两处各写一套阈值/配色，迟早会出现"面板里是绿的、地图上是黄的"。
 * 这里把「血量 -> 四档状态 -> 颜色/文字」收敛成一份定义，两个界面只做渲染。
 *
 * 数据从哪来（必须说清，否则会被读成实测战损）
 * --------------------------------------------
 * 案例数据里**没有血量/损毁字段**：`frames[].targets[]` 只有经纬度、航速、
 * 航向等运动量，`inference_frames[].targets[]` 只有威胁分与告警等级
 * （engine/inference.py 的 ALERT_CODES：0 正常 / 1 注意 / 2 警告 / 3 严重）。
 *
 * 因此这里的血量是**战损估计**，不是仿真出的实测值：目标初始 100 HP，
 * 处于高告警（>=2）的每一帧按固定速率累计损伤：
 *
 *     alert 2（警告） -> 0.10 HP / 帧（30 s）
 *     alert 3（严重） -> 0.30 HP / 帧（30 s）
 *
 * 速率、阈值都写在本文件顶部，且 `debug()` 可原样读出，评审时可以逐条核对。
 * 它是**单调的估计**：一旦为 0 在该时刻之后保持损毁；回拖时间轴会按那一时刻
 * 从头重算（显示当时更高的血量），前推到损毁之后仍为损毁。
 * 若将来引擎产出真实战损，只需把 `_apply` 改为读数据字段，两个界面无需改动。
 *
 * 四档（与需求一致）
 * ------------------
 *   0 正常       HP > 70
 *   1 受损       35 < HP <= 70
 *   2 严重受损   0 < HP <= 35
 *   3 损毁       HP == 0
 *
 * 视觉约定（图标本体不改，状态只由血条与损毁遮罩表达）
 * ------------------------------------------------------
 *   - 血条**紧贴图标底部**，颜色与长度表达档位；
 *   - 损毁档血条为空槽、图标压暗并叠加白色 X（深红描边）遮罩；
 *   - 不在图标周围画状态环/角标（2026-10-21 用户要求），阵营色与兵种图形
 *     始终保留，类型与敌我仍然可辨。
 */
'use strict';

(function (global) {
  // 每帧损伤速率（HP / frame）。帧长 30 s，见 engine/scenario.py TICK_SECONDS。
  var RATE = { 2: 0.10, 3: 0.30 };

  // 状态阈值：hp 小于等于该值进入该档。损毁单独按 hp <= 0 判定。
  var THRESHOLDS = { severe: 35, damaged: 70 };

  var TIERS = [
    { code: 0, key: 'normal', label: '正常', color: '#2E9E5B' },
    { code: 1, key: 'damaged', label: '受损', color: '#D9A514' },
    { code: 2, key: 'severe', label: '严重受损', color: '#E2662A' },
    { code: 3, key: 'destroyed', label: '损毁', color: '#4B5563' },
  ];

  function tier(code) { return TIERS[Math.max(0, Math.min(3, code | 0))]; }

  // 计算缓存：case + 推理数组 + 帧数任一变化即重建；tick 只做增量推进。
  var cache = {
    caseId: null,
    infRef: null,
    targetsRef: null,
    framesLen: 0,
    tick: 0,
    hp: {},
    destroyedAt: {},
  };

  function _reset(data) {
    cache.caseId = data.case_id || '';
    cache.infRef = data.inference_frames || null;
    cache.targetsRef = data.targets || null;
    cache.framesLen = (data.frames || []).length;
    cache.tick = 0;
    cache.hp = {};
    cache.destroyedAt = {};
    (data.targets || []).forEach(function (t) {
      cache.hp[t.id] = 100;
    });
  }

  function _ensure(data) {
    var inf = data.inference_frames || null;
    if (cache.caseId !== (data.case_id || '') ||
        cache.infRef !== inf ||
        cache.targetsRef !== (data.targets || null) ||
        cache.framesLen !== (data.frames || []).length) {
      _reset(data);
    }
  }

  /** 把第 t 帧的告警累计进血量。只有 >=2 的告警才造成损伤。 */
  function _apply(t) {
    var inf = cache.infRef;
    if (!inf || !inf.length || t >= inf.length) return;   // 推理帧之后不臆造损伤
    var fr = inf[t];
    if (!fr || !fr.targets) return;
    var ids = Object.keys(cache.hp);
    for (var i = 0; i < ids.length; i++) {
      var id = ids[i];
      var hp = cache.hp[id];
      if (hp <= 0) continue;
      var entry = fr.targets[id];
      if (!entry) continue;
      var rate = RATE[entry.alert | 0];
      if (!rate) continue;
      // 每步量化到 0.1：不量化时二进制浮点会留下 ~1e-13 的正残差，
      // 出现"显示 0%、状态仍是严重受损"的第三种状态，正好违背"损毁与低血量可区分"。
      hp = Math.max(0, Math.round((hp - rate) * 10) / 10);
      cache.hp[id] = hp;
      if (hp === 0) cache.destroyedAt[id] = t;
    }
  }

  function _advance(tick) {
    if (tick < cache.tick) {
      // 回拖：从头重算。1441 帧 × 数十目标对浏览器只是毫秒级开销。
      cache.tick = 0;
      cache.destroyedAt = {};
      Object.keys(cache.hp).forEach(function (id) { cache.hp[id] = 100; });
    }
    // 从第 1 帧开始累计：第 0 帧是案例初始状态，视为尚未交战、不扣血。
    for (var t = cache.tick + 1; t <= tick; t++) _apply(t);
    cache.tick = tick;
  }

  function statusOf(hp) {
    if (hp <= 0) return 3;
    if (hp <= THRESHOLDS.severe) return 2;
    if (hp <= THRESHOLDS.damaged) return 1;
    return 0;
  }

  /**
   * 某目标在某帧的战损估计。
   *
   * @returns {{hp:number, status:number, label:string, destroyedAt:number|null,
   *            alert:number|null, alertLabel:string}}
   */
  function at(data, id, tick) {
    if (!data || !data.targets) {
      return { hp: 100, status: 0, label: TIERS[0].label, destroyedAt: null, alert: null, alertLabel: '' };
    }
    _ensure(data);
    var max = Math.max(0, (data.frames || []).length - 1);
    var t = Math.max(0, Math.min(max, tick | 0));
    _advance(t);
    var hp = (cache.hp[id] === undefined) ? 100 : cache.hp[id];
    var st = statusOf(hp);
    var inf = cache.infRef;
    var entry = (inf && inf.length) ? (inf[Math.min(t, inf.length - 1)] || {}).targets : null;
    var e = entry ? entry[id] : null;
    return {
      hp: Math.round(hp * 10) / 10,
      status: st,
      label: TIERS[st].label,
      destroyedAt: cache.destroyedAt[id] === undefined ? null : cache.destroyedAt[id],
      alert: e ? (typeof e.alert === 'number' ? e.alert : null) : null,
      alertLabel: e && e.alert_label ? e.alert_label : '',
    };
  }

  /** 双方各档计数，供面板摘要与自检读取。 */
  function counts(data, tick) {
    var out = [
      { total: 0, n: [0, 0, 0, 0] },
      { total: 0, n: [0, 0, 0, 0] },
    ];
    (data && data.targets || []).forEach(function (t) {
      var side = t.affiliation === 'blue' ? 1 : 0;
      var st = at(data, t.id, tick).status;
      out[side].total++;
      out[side].n[st]++;
    });
    return { red: out[0], blue: out[1] };
  }

  // ── 可视资产 ────────────────────────────────────────────────────────
  // Cesium 的 billboard 需要图片而不是 DOM；小面板里的 `<img>`/CSS 又更轻。
  // 这里只为鸟瞰图生成两种 data URI：血条与损毁暗色底。
  var barCache = Object.create(null);
  var markCache = Object.create(null);

  function colorOf(status) { return TIERS[Math.max(0, Math.min(3, status | 0))].color; }

  function barUri(hp, status, w, h) {
    status = Math.max(0, Math.min(3, status | 0));
    w = w || 22; h = h || 4;
    hp = Math.max(0, Math.min(100, hp));
    var pct = status === 3 ? 0 : hp / 100;
    var key = status + '|' + Math.round(pct * 100) + '|' + w + 'x' + h;
    if (barCache[key]) return barCache[key];
    // 存活目标至少留 1.5 px 填充：0.1% 血量时也不能变成"看不见颜色的空条"，
    // 否则会和损毁的空槽（深色轨道 + X）混淆。
    var fillW = status === 3 ? 0 : Math.max(1.5, (w - 2) * pct);
    var fill = colorOf(status);
    // 损毁档用深色空槽，与整体TS面板的 `.os-unit.os-s3 .os-hp` 一致；
    // 血条"整条变黑且没有填充"本身就是损毁的第二个视觉信号。
    var track = status === 3 ? '#3F4652' : '#E4E9F0';
    var svg = '<svg xmlns="http://www.w3.org/2000/svg" width="' + w + '" height="' + h + '">'
      + '<rect x="0.5" y="0.5" width="' + (w - 1) + '" height="' + (h - 1) +
        '" rx="1" fill="' + track + '" stroke="#6B7688" stroke-width="1"/>'
      + (fillW > 0
          ? '<rect x="1" y="1" width="' + fillW.toFixed(1) + '" height="' + (h - 2) +
            '" rx="0.5" fill="' + fill + '"/>'
          : '')
      + '</svg>';
    barCache[key] = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svg);
    return barCache[key];
  }

  /**
   * 损毁暗色底（34×34，与图标同心叠加）。仅损毁档使用：
   * 黑暗色半透明底压住图标，叉号由 overview.js 的 label 单独绘制
   * （label 后绘制，密集叠压时仍可见）。受损/严重受损**不画环、不画角标**，
   * 只靠紧贴图标底部的血条颜色与长度表达（2026-10-21 用户要求）。
   */
  function markerUri(status) {
    status = Math.max(0, Math.min(3, status | 0));
    if (status !== 3) {
      if (!markCache[0]) {
        markCache[0] = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(
          '<svg xmlns="http://www.w3.org/2000/svg" width="34" height="34"></svg>');
      }
      return markCache[0];
    }
    if (markCache[3]) return markCache[3];
    var svg = '<svg xmlns="http://www.w3.org/2000/svg" width="34" height="34">'
      + '<rect x="2" y="2" width="30" height="30" rx="4" fill="#1F2937" fill-opacity="0.55"/>'
      + '</svg>';
    markCache[3] = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svg);
    return markCache[3];
  }

  /** 取证：规则、缓存与当前帧的逐目标读数，供自检断言。 */
  function debug(data, tick) {
    var per = {};
    if (data && data.targets) {
      data.targets.forEach(function (t) {
        var r = at(data, t.id, tick);
        per[t.id] = {
          hp: r.hp, status: r.status, label: r.label,
          destroyedAt: r.destroyedAt, alert: r.alert,
        };
      });
    }
    return {
      rule: 'hp -= {2: 0.10, 3: 0.30} per tick with alert >= 2',
      thresholds: THRESHOLDS,
      rate: RATE,
      caseId: cache.caseId,
      tick: cache.tick,
      hasInference: !!cache.infRef,
      counts: data ? counts(data, tick) : null,
      targets: per,
    };
  }

  global.TargetStatus = {
    TIERS: TIERS,
    RATE: RATE,
    THRESHOLDS: THRESHOLDS,
    tier: tier,
    colorOf: colorOf,
    statusOf: statusOf,
    at: at,
    counts: counts,
    barUri: barUri,
    markerUri: markerUri,
    debug: debug,
    reset: function () {
      cache.caseId = null; cache.infRef = null; cache.targetsRef = null;
    },
  };
})(window);
