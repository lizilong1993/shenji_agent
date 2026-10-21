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
 * 因此这里的血量是**案内相对战损估计**，不是仿真出的实测值。规则（全部可取证）：
 *
 *   1. 交战暴露 E = Σ 每帧权重；alert 2 -> 1，alert 3 -> 3，其余 0。
 *   2. 按整段案例的 E_total 排序分配最终档位（占比按编成总数）：
 *        前 6%   -> 损毁（最终 0 HP；仅当最高暴露 >= 800 才启用）
 *        随后 10% -> 严重受损（最终 30..5 HP）
 *        再后 22% -> 受损（最终 68..40 HP）
 *        其余    -> 正常（100 HP）
 *      （少于 5 个目标时只分 1 个严重/1 个受损，不强行制造损毁；见 `_assignFinal`。）
 *   3. 时间展开：HP 随当前暴露按 E(t) / (0.8 * E_total) 线性下降，
 *      到 80% 暴露时落位到最终档；损毁目标在此时刻归零并保持。
 *
 * 为什么用"案内相对"而不是绝对速率：不同案例的交战强度差 10 倍以上
 * （台海 31 目标约 7 千个高告警帧，密集投送 80 目标约 11 万个），绝对速率
 * 要么让低强度案例没有损毁，要么让高强度案例"全军覆没"。排序分配保证任何
 * 案例都能读出四档差异，同时严格按"暴露越多、战损越重"排序。
 *
 * 若将来引擎产出真实战损，只需替换 `_assignFinal`/`at` 的取值，两个界面无需改动。
 *
 * 四档（与需求一致）
 * ------------------
 *   0 正常       HP > 70
 *   1 受损       35 < HP <= 70
 *   2 严重受损   0 < HP <= 35
 *   3 损毁       HP == 0
 *
 * 视觉约定（图标本体不改）
 * ------------------------
 *   - 血条**紧贴图标底部**，颜色与长度表达档位；
 *   - 损毁档血条为空槽、图标压暗并叠加白色 X（深红描边）遮罩；
 *   - 不在图标周围画状态环/角标（2026-10-21 用户要求），阵营色与兵种图形
 *     始终保留；损毁遮罩是唯一覆盖图标本体的状态标识。
 */
'use strict';

(function (global) {
  // 每帧交战权重：告警 2/3 才计入暴露。
  var WEIGHTS = { 2: 1, 3: 3 };

  // 最终档位占比（按整段案例的暴露排序）。
  var BANDS = { destroyed: 0.06, severe: 0.10, damaged: 0.22 };

  // 血量随暴露进度落位到最终档的位置：80% 暴露时到达最终值。
  var SETTLE_AT = 0.8;

  // 只有最高暴露达到该值（约 267 帧严重告警，约 2.2 小时）才允许出现损毁档。
  // 否则低强度对峙（如中印案例最高暴露 642）也会被"排名前 6%"强行判毁。
  var MIN_DESTROY_EXPOSURE = 800;

  var THRESHOLDS = { severe: 35, damaged: 70 };

  // 血条透明度：正常 1.0 -> 损毁 0.3。颜色/长度之外再给一条单调的视觉线索。
  var OPACITY = { 0: 1.0, 1: 0.8, 2: 0.55, 3: 0.3 };

  var TIERS = [
    { code: 0, key: 'normal', label: '正常', color: '#2E9E5B' },
    { code: 1, key: 'damaged', label: '受损', color: '#D9A514' },
    { code: 2, key: 'severe', label: '严重受损', color: '#E2662A' },
    { code: 3, key: 'destroyed', label: '损毁', color: '#4B5563' },
  ];

  function tier(code) { return TIERS[Math.max(0, Math.min(3, code | 0))]; }

  // 计算缓存：case + 推理数组 + 目标数组 + 帧数任一变化即重建。
  var cache = {
    caseId: null,
    infRef: null,
    targetsRef: null,
    framesLen: 0,
    tick: 0,
    exposure: {},        // 当前累计暴露 E(t)
    totalExposure: {},   // 整段案例暴露 E_total
    finalHp: {},         // 最终档位 HP
    destroyedAt: {},
    ranks: {},           // 暴露排名（1 = 最高）
  };

  function _reset(data) {
    cache.caseId = data.case_id || '';
    cache.infRef = data.inference_frames || null;
    cache.targetsRef = data.targets || null;
    cache.framesLen = (data.frames || []).length;
    cache.tick = 0;
    cache.exposure = {};
    cache.totalExposure = {};
    cache.finalHp = {};
    cache.destroyedAt = {};
    cache.ranks = {};

    var ids = (data.targets || []).map(function (t) { return t.id; });
    ids.forEach(function (id) {
      cache.exposure[id] = 0;
      cache.totalExposure[id] = 0;
      cache.finalHp[id] = 100;
    });

    // 一次扫描算出 E_total（不逐帧存数组：80 目标 × 1441 帧只是 11 万次加法）。
    var inf = cache.infRef;
    if (inf && inf.length) {
      for (var t = 1; t < inf.length; t++) {
        var fr = inf[t];
        if (!fr || !fr.targets) continue;
        for (var i = 0; i < ids.length; i++) {
          var e = fr.targets[ids[i]];
          if (e) cache.totalExposure[ids[i]] += (WEIGHTS[e.alert | 0] || 0);
        }
      }
    }
    _assignFinal(ids);
  }

  /** 按 E_total 排序分配最终档位与 HP。 */
  function _assignFinal(ids) {
    var active = ids.filter(function (id) { return cache.totalExposure[id] > 0; });
    active.sort(function (a, b) {
      return (cache.totalExposure[b] - cache.totalExposure[a]) || (a < b ? -1 : 1);
    });
    active.forEach(function (id, i) { cache.ranks[id] = i + 1; });

    // 占比按**编成总数**算（不是按"有暴露的目标数"）：
    //   台海 31 个目标中只有 15 个真正进过高告警，若按 15 算，前 6% 只有 0 个、
    //   严重档只剩 1 个，四档分布会塌缩。编成口径给出 1 损毁 / 3 严重 / 6 受损，
    //   其余（含从未暴露的）正常。
    var n = active.length;
    var roster = ids.length;
    var dN = 0, sN = 0, mN = 0;
    if (roster >= 5) {
      dN = Math.max(1, Math.floor(roster * BANDS.destroyed));
      sN = Math.max(1, Math.floor(roster * BANDS.severe));
      mN = Math.max(1, Math.floor(roster * BANDS.damaged));
    } else {
      sN = n > 1 ? 1 : 0;
      mN = n > 2 ? 1 : 0;
    }
    // 暴露目标太少时，按比例压缩三个 band，并至少给有暴露的目标留一个正常档。
    if (n > 0 && dN + sN + mN >= n) {
      var cap = Math.max(1, n - 1);
      var total = dN + sN + mN;
      dN = Math.floor(dN * cap / total);
      sN = Math.max(0, Math.floor(sN * cap / total));
      mN = Math.max(0, cap - dN - sN);
    }
    // 低强度案例不因排名而制造损毁：最高暴露不足阈值时取消损毁档，
    // 由严重/受损档承接排名靠前的目标。
    if (dN > 0 && active.length &&
        cache.totalExposure[active[0]] < MIN_DESTROY_EXPOSURE) {
      dN = 0;
    }

    active.forEach(function (id, i) {
      if (i < dN) {
        cache.finalHp[id] = 0;
      } else if (i < dN + sN) {
        var k = i - dN;
        cache.finalHp[id] = sN <= 1 ? 30 : 30 - (25 * k / (sN - 1));
      } else if (i < dN + sN + mN) {
        var j = i - dN - sN;
        cache.finalHp[id] = mN <= 1 ? 68 : 68 - (28 * j / (mN - 1));
      } else {
        cache.finalHp[id] = 100;
      }
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

  function _apply(t) {
    var inf = cache.infRef;
    if (!inf || !inf.length || t >= inf.length) return;
    var fr = inf[t];
    if (!fr || !fr.targets) return;
    var ids = Object.keys(cache.exposure);
    for (var i = 0; i < ids.length; i++) {
      var id = ids[i];
      var e = fr.targets[id];
      var w = e ? (WEIGHTS[e.alert | 0] || 0) : 0;
      if (!w) continue;
      cache.exposure[id] += w;
    }
  }

  function _advance(tick) {
    var prevTick = cache.tick;
    if (tick < cache.tick) {
      // 回拖：从第 0 帧重算暴露。1441 帧 × 数十目标只是毫秒级开销。
      cache.tick = 0;
      cache.destroyedAt = {};
      Object.keys(cache.exposure).forEach(function (id) { cache.exposure[id] = 0; });
    }
    // 从第 1 帧开始累计：第 0 帧是案例初始状态，视为尚未交战。
    for (var t = cache.tick + 1; t <= tick; t++) _apply(t);
    cache.tick = tick;

    // 同一帧的多次查询（鸟瞰图/面板/计数各一次）不再重复扫损毁帧。
    if (tick === prevTick) return;

    // 记下首次达到损毁条件的帧（供调试/提示；渲染不依赖它）。
    Object.keys(cache.exposure).forEach(function (id) {
      if (cache.destroyedAt[id] !== undefined) return;
      var total = cache.totalExposure[id] || 0;
      if (cache.finalHp[id] === 0 && total > 0 &&
          cache.exposure[id] >= SETTLE_AT * total) {
        cache.destroyedAt[id] = tick;
      }
    });
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
   *            alert:number|null, alertLabel:string, exposure:number, finalHp:number}}
   */
  function at(data, id, tick) {
    if (!data || !data.targets) {
      return {
        hp: 100, status: 0, label: TIERS[0].label, destroyedAt: null,
        alert: null, alertLabel: '', exposure: 0, finalHp: 100,
      };
    }
    _ensure(data);
    var max = Math.max(0, (data.frames || []).length - 1);
    var t = Math.max(0, Math.min(max, tick | 0));
    _advance(t);

    var total = cache.totalExposure[id] || 0;
    var final = (cache.finalHp[id] === undefined) ? 100 : cache.finalHp[id];
    var hp = 100;
    if (total > 0 && final < 100) {
      var progress = Math.min(1, (cache.exposure[id] || 0) / (SETTLE_AT * total));
      hp = 100 - (100 - final) * progress;
    }
    hp = Math.round(hp * 10) / 10;
    if (final === 0 && total > 0 &&
        (cache.exposure[id] || 0) >= SETTLE_AT * total) {
      hp = 0;
    }
    var st = statusOf(hp);

    var inf = cache.infRef;
    var entry = (inf && inf.length) ? (inf[Math.min(t, inf.length - 1)] || {}).targets : null;
    var e = entry ? entry[id] : null;
    return {
      hp: hp,
      status: st,
      label: TIERS[st].label,
      destroyedAt: cache.destroyedAt[id] === undefined ? null : cache.destroyedAt[id],
      alert: e ? (typeof e.alert === 'number' ? e.alert : null) : null,
      alertLabel: e && e.alert_label ? e.alert_label : '',
      exposure: cache.exposure[id] || 0,
      finalHp: final,
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
  // 这里为鸟瞰图生成 data URI：血条、损毁暗色底、聚合图标与选中环。
  var barCache = Object.create(null);
  var markCache = Object.create(null);

  function colorOf(status) { return TIERS[Math.max(0, Math.min(3, status | 0))].color; }
  function opacityFor(status) { return OPACITY[Math.max(0, Math.min(3, status | 0))] || 1.0; }

  function barUri(hp, status, w, h, colorStatus) {
    status = Math.max(0, Math.min(3, status | 0));
    var cs = (colorStatus === undefined) ? status : Math.max(0, Math.min(3, colorStatus | 0));
    w = w || 22; h = h || 4;
    hp = Math.max(0, Math.min(100, hp));
    var pct = status === 3 ? 0 : hp / 100;
    var key = status + '|' + cs + '|' + Math.round(pct * 100) + '|' + w + 'x' + h;
    if (barCache[key]) return barCache[key];
    // 存活目标至少留 1.5 px 填充：0.1% 血量时也不能变成"看不见颜色的空条"，
    // 否则会和损毁的空槽（深色轨道 + X）混淆。
    var fillW = status === 3 ? 0 : Math.max(1.5, (w - 2) * pct);
    var fill = colorOf(cs);
    // 损毁档用深色空槽，与整体TS面板的 `.os-unit.os-s3 .os-hp` 一致；
    // 血条"整条变黑且没有填充"本身就是损毁的第二个视觉信号。
    var track = cs === 3 ? '#3F4652' : '#E4E9F0';
    var svg = '<svg xmlns="http://www.w3.org/2000/svg" width="' + w + '" height="' + h +
      '" opacity="' + opacityFor(status) + '">'
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

  // 聚合图标：把数量**画进图标内部**，不再外挂角标——外挂标签宽度约 30px，
  // 两个相邻聚合的标签必然互压，还会抢点击。圆形底色按阵营：
  //   red/blue 纯色；mixed 红蓝各半；mixeddead 红蓝各半 + 右上红角；
  //   destroyed 深灰 + 白色 X + 数量。
  var clusterCache = Object.create(null);
  function clusterUri(faction, count) {
    var key = (faction || 'neutral') + '|' + (count | 0);
    if (clusterCache[key]) return clusterCache[key];
    var kind = (faction === 'red' || faction === 'blue' || faction === 'mixed' ||
                faction === 'mixeddead' || faction === 'destroyed') ? faction : 'neutral';
    var base;
    if (kind === 'red') base = '<circle cx="17" cy="17" r="15" fill="#C0392B"/>';
    else if (kind === 'blue') base = '<circle cx="17" cy="17" r="15" fill="#2471A3"/>';
    else if (kind === 'destroyed') base = '<circle cx="17" cy="17" r="15" fill="#4B5563"/>';
    else if (kind === 'mixed' || kind === 'mixeddead') {
      base = '<circle cx="17" cy="17" r="15" fill="#2471A3"/>'
        + '<path d="M17 2 A15 15 0 0 0 17 32 Z" fill="#C0392B"/>';
    } else base = '<circle cx="17" cy="17" r="15" fill="#34495E"/>';
    var corners = (kind === 'mixeddead')
      ? '<path d="M31 2 L34 2 L34 15 Z" fill="#B91C1C"/>'
        + '<path d="M28.5 4.5 L32 8 M32 4.5 L28.5 8" stroke="#FFFFFF" '
        + 'stroke-width="1.8" stroke-linecap="round"/>'
      : '';
    var n = Math.max(1, count | 0);
    var label = kind === 'destroyed' ? ('✕' + n) : String(n);
    var font = label.length >= 3 ? 10 : 12;
    var text = '<text x="17" y="21" text-anchor="middle" font-family="sans-serif" '
      + 'font-size="' + font + '" font-weight="700" fill="#FFFFFF">' + label + '</text>';
    var svg = '<svg xmlns="http://www.w3.org/2000/svg" width="34" height="34">'
      + base
      + '<circle cx="17" cy="17" r="15" fill="none" stroke="#FFFFFF" stroke-width="2"/>'
      + corners + text + '</svg>';
    clusterCache[key] = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svg);
    return clusterCache[key];
  }

  // 选中目标的黄色定位环：聚合图标可能压住选中目标，这个环最后绘制、永远可见。
  var selectionCache = null;
  function selectionUri() {
    if (selectionCache) return selectionCache;
    var svg = '<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40">'
      + '<rect x="2" y="2" width="36" height="36" rx="6" fill="none" '
      + 'stroke="#FFD24A" stroke-width="3"/></svg>';
    selectionCache = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svg);
    return selectionCache;
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
          exposure: r.exposure, totalExposure: cache.totalExposure[t.id] || 0,
          finalHp: r.finalHp, rank: cache.ranks[t.id] || null,
        };
      });
    }
    return {
      rule: 'relative damage by case-wide exposure: E += {alert2:1, alert3:3}; ' +
            'top 6% destroyed, next 10% severe, next 22% damaged; ' +
            'HP settles at 80% exposure',
      bands: BANDS,
      weights: WEIGHTS,
      settleAt: SETTLE_AT,
      minDestroyExposure: MIN_DESTROY_EXPOSURE,
      thresholds: THRESHOLDS,
      caseId: cache.caseId,
      tick: cache.tick,
      hasInference: !!cache.infRef,
      counts: data ? counts(data, tick) : null,
      targets: per,
    };
  }

  global.TargetStatus = {
    TIERS: TIERS,
    WEIGHTS: WEIGHTS,
    BANDS: BANDS,
    THRESHOLDS: THRESHOLDS,
    tier: tier,
    colorOf: colorOf,
    opacityFor: opacityFor,
    statusOf: statusOf,
    at: at,
    counts: counts,
    barUri: barUri,
    markerUri: markerUri,
    clusterUri: clusterUri,
    selectionUri: selectionUri,
    debug: debug,
    reset: function () {
      cache.caseId = null; cache.infRef = null; cache.targetsRef = null;
    },
  };
})(window);
