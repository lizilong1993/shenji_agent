/**
 * 右侧栏的两块新内容：整体TS栏 与 目标TS栏。
 *
 * 为什么单独一个模块
 * ------------------
 * `app.js` 已经 3500 多行，而这两块都是"给定数据渲染 DOM"，与相机、图层、帧循环
 * 无关。放在这里可以独立读、独立改，也避免 app.js 继续膨胀。
 *
 * 一个必须说清的取舍：颜色语义是**告警等级**，不是战损
 * ----------------------------------------------------
 * 参考图里图标颜色编码的是「血量 100%/70%/50%/30%」。本系统**没有任何战损/血量
 * 字段**（`engine/scenario.py` 的 Target 只有经纬度、航速、航向、传感器、交战半径
 * 等），因此这里用每帧推理已有的告警等级（`engine/inference.py` 的 ALERT_CODES：
 * 0 正常 / 1 注意 / 2 警告 / 3 严重）作为颜色维度，图例如实写成对应文字。
 *
 * 若将来真要展示战损，正确做法是给引擎加字段并仿真损伤过程，而不是在这里用别的量
 * 冒充血量——那会让人把"告警等级高"读成"已经被打残了"。
 *
 * 图标不新建 4 套 SVG：复用 `Symbology.symbolUri` 的红/蓝原色图标，用 CSS filter
 * 逐级降饱和压暗（见 style.css 的 `.os-tier-*`），得到与参考图相同的单调渐变观感。
 */
(function () {
  'use strict';

  // 行分组顺序，取自 `symbology.ICONS[*].family`。固定顺序让同一案例每次渲染的行位置
  // 一致；案例里没有的家族不占行。
  var FAMILY_ORDER = [
    '水面舰艇', '辅助舰船', '岸防兵力', '地面防空',
    '地面机动', '地面火力', '作战飞机', '支援飞机',
  ];
  var FAMILY_FALLBACK = '其它';

  // 告警等级 -> 文案与档位。等级取自 `engine/inference.py` 的 ALERT_CODES，
  // 不改写它的措辞，避免两处说法不一致。
  var TIERS = [
    { label: '正常单位', cls: 'os-tier-0' },
    { label: '注意单位', cls: 'os-tier-1' },
    { label: '警告单位', cls: 'os-tier-2' },
    { label: '严重单位', cls: 'os-tier-3' },
  ];

  // 按案例缓存已构建的 DOM，避免每帧重建图标。
  var cache = { key: null, imgs: {} };

  function esc(s) {
    return String(s === null || s === undefined ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function familyOf(icons, iconKey) {
    var meta = (icons || {})[iconKey];
    return (meta && meta.family) || FAMILY_FALLBACK;
  }

  function tierIndex(alert, hasInference) {
    // 没有该帧的推理结果时不猜等级，一律按"正常"渲染并在 title 里说明。
    if (!hasInference || typeof alert !== 'number') return 0;
    return Math.max(0, Math.min(TIERS.length - 1, alert | 0));
  }

  /**
   * 整体TS栏。
   *
   * 红方在上、蓝方在下**纵向堆叠**，而不是参考图里的左右并排：右栏宽 380px，并排后
   * 每侧不足 190px，放不下"家族标签 + 图标条 + 计数"。堆叠保留同样的视觉语言（每方
   * 一个标题、每类一行、右侧计数、底部图例），只是适应窄栏。
   *
   * 展示案例**完整编成**，不跟随地图上的红/蓝/域过滤开关——它是"全局整体状态"汇总，
   * 跟着地图过滤会让人以为编成变了。
   */
  function renderOverall(hostEl, data, inferenceFrame) {
    if (!hostEl) return;
    if (!data || !data.targets || !data.targets.length) {
      hostEl.innerHTML = '<div class="os-empty">未加载TS</div>';
      return;
    }

    var icons = (data.symbology && data.symbology.icons) || {};
    var sig = data.case_id + '|' + data.targets.length;

    if (cache.key !== sig) {
      cache.key = sig;
      cache.imgs = {};
      hostEl.innerHTML = buildOverall(data, icons);
      // 建好结构后把每个目标的 <img> 收进映射，之后每帧只改 class，不重建 DOM。
      Array.prototype.forEach.call(
        hostEl.querySelectorAll('.os-icon[data-target]'), function (img) {
          cache.imgs[img.getAttribute('data-target')] = img;
        });
    }

    var has = !!inferenceFrame;
    Object.keys(cache.imgs).forEach(function (tid) {
      var entry = has ? (inferenceFrame.targets || {})[tid] : null;
      var t = tierIndex(entry ? entry.alert : undefined, has && !!entry);
      var img = cache.imgs[tid];
      img.className = 'os-icon ' + TIERS[t].cls;
      img.title = entry
        ? (tid + ' · ' + (entry.alert_label || TIERS[t].label))
        : (tid + ' · 无该帧推理结果');
    });
  }

  function buildOverall(data, icons) {
    function side(aff, label) {
      var list = data.targets.filter(function (t) { return t.affiliation === aff; });
      var byFam = {};
      list.forEach(function (t) {
        var f = familyOf(icons, t.icon);
        (byFam[f] = byFam[f] || []).push(t);
      });
      var order = FAMILY_ORDER.filter(function (f) { return byFam[f]; });
      Object.keys(byFam).forEach(function (f) { if (order.indexOf(f) < 0) order.push(f); });

      var rows = order.map(function (f) {
        var items = byFam[f];
        var rep = items[0];
        return '<div class="os-row">'
          + '<span class="os-fam">'
          + '<img class="os-fam-ico" src="' + Symbology.symbolUri(rep.icon, aff, {}) + '" alt="">'
          + esc(f) + '</span>'
          + '<span class="os-icons">'
          + items.map(function (t) {
              return '<img class="os-icon os-tier-0" data-target="' + esc(t.id) + '"'
                + ' src="' + Symbology.symbolUri(t.icon, aff, {}) + '" alt="' + esc(t.id) + '">';
            }).join('')
          + '</span>'
          + '<span class="os-count">共 <b>' + items.length + '</b> 单位</span>'
          + '</div>';
      }).join('');

      return '<div class="os-side os-' + aff + '">'
        + '<div class="os-head"><span class="os-title">' + esc(label) + '</span>'
        + '<span class="os-sub">参战单位总览</span></div>'
        + (rows || '<div class="os-empty">无目标</div>')
        + '</div>';
    }

    var legend = '<div class="os-legend">'
      + TIERS.map(function (t, i) {
          return '<span><i class="os-sw os-tier-' + i + '"></i>' + esc(t.label) + '</span>';
        }).join('')
      + '</div>';

    return side('red', '红方') + side('blue', '蓝方') + legend;
  }
  // ------------------------------------------------------------------------
  // 活动规律
  // ------------------------------------------------------------------------

  var R_EARTH = 6371008.8;

  function rad(d) { return d * Math.PI / 180; }

  function haversineM(a, b) {
    var dLat = rad(b.lat - a.lat), dLon = rad(b.lon - a.lon);
    var h = Math.sin(dLat / 2) * Math.sin(dLat / 2)
          + Math.cos(rad(a.lat)) * Math.cos(rad(b.lat))
          * Math.sin(dLon / 2) * Math.sin(dLon / 2);
    return 2 * R_EARTH * Math.asin(Math.min(1, Math.sqrt(h)));
  }

  /**
   * 航向的圆周标准差（度）。
   *
   * 不能用普通标准差：359° 与 1° 相差 2°，普通算法会算成 358°，于是"正在向北转向"
   * 会被读成"航向剧烈抖动"。引擎侧 `intent_features._heading_std` 用同一套公式，
   * 这里保持一致。
   */
  function headingStd(headings) {
    var v = headings.filter(function (h) { return typeof h === 'number'; });
    if (v.length < 2) return 0;
    var s = 0, c = 0;
    v.forEach(function (h) { s += Math.sin(rad(h)); c += Math.cos(rad(h)); });
    s /= v.length; c /= v.length;
    var r = Math.min(1, Math.max(1e-9, Math.sqrt(s * s + c * c)));
    return Math.sqrt(-2 * Math.log(r)) * 180 / Math.PI;
  }

  /**
   * 从帧序列尾窗推导一个目标的活动规律。
   *
   * 样本不足时返回 `{ insufficient: true }` 并显示"样本不足"，不硬给结论——与意图
   * 集合那边同一个原则：证据不够就说不够。
   */
  function deriveActivity(frames, tick, def) {
    var WIN = 30;
    var lo = Math.max(0, (tick | 0) - WIN);
    var pts = [];
    for (var i = lo; i <= tick; i++) {
      var fr = frames && frames[i];
      if (!fr) continue;
      var arr = fr.targets || [];
      for (var k = 0; k < arr.length; k++) {
        if (arr[k].id === def.id) { pts.push(arr[k]); break; }
      }
    }
    if (pts.length < 5) return { insufficient: true, samples: pts.length };

    var speeds = [], headings = [];
    pts.forEach(function (p) {
      if (typeof p.speed_kn === 'number') speeds.push(p.speed_kn);
      if (typeof p.heading === 'number') headings.push(p.heading);
    });
    var avgSpeed = speeds.length
      ? speeds.reduce(function (a, b) { return a + b; }, 0) / speeds.length : 0;

    var path = 0;
    for (var j = 1; j < pts.length; j++) path += haversineM(pts[j - 1], pts[j]);
    var net = haversineM(pts[0], pts[pts.length - 1]);
    var ratio = net > 1 ? path / net : Infinity;

    var last = pts[pts.length - 1];
    // `def.loop` 不在数据集里：`scenario.Target.to_dataset_entry()` 输出 id / name /
    // domain / speed_kn / … / waypoints，**没有 loop 字段**。所以只读 `def.loop`
    // 会让"往复巡逻"永远不可达（实测：闭合环线目标被报成"单向航渡"）。
    // `frames[].targets[].laps` 每帧都有：非闭环目标恒为 0，闭环目标 > 0。
    // 因此以 laps 为主判据，`def.loop` 仅作兜底（若将来把它补进数据集）。
    var looping = (typeof last.laps === 'number' && last.laps > 0) || def.loop === true;

    var mode;
    if (avgSpeed < 5 && net < 5000) mode = '低速待机';
    else if (looping) mode = '往复巡逻';
    else if (net > 1000 && ratio < 1.3) mode = '单向航渡';
    else mode = '区域机动';

    var hs = headingStd(headings);
    var stability = hs < 8 ? '直航' : (hs < 35 ? '缓慢转向' : '频繁转向');

    return {
      insufficient: false,
      samples: pts.length,
      windowFrames: WIN,
      mode: mode,
      looping: looping,
      stability: stability,
      headingStdDeg: Math.round(hs),
      avgSpeedKn: Math.round(avgSpeed * 10) / 10,
      pathKm: Math.round(path / 100) / 10,
      netKm: Math.round(net / 100) / 10,
      travelledKm: (typeof last.travelled_m === 'number')
        ? Math.round(last.travelled_m / 100) / 10 : null,
      laps: (typeof last.laps === 'number' && last.laps > 0) ? last.laps : null,
    };
  }

  // ------------------------------------------------------------------------
  // 目标TS栏
  // ------------------------------------------------------------------------

  function kv(k, v, cls) {
    return '<div class="ts-kv"><span>' + esc(k) + '</span><b'
      + (cls ? ' class="' + esc(cls) + '"' : '') + '>'
      + esc(v === null || v === undefined || v === '' ? '—' : v) + '</b></div>';
  }

  function fmtDeg(v, pos, neg) {
    if (typeof v !== 'number' || !isFinite(v)) return '—';
    return Math.abs(v).toFixed(4) + '°' + (v >= 0 ? pos : neg);
  }

  /**
   * 目标意图栏：**只显示意图集合**（CES 候选集合）与它的判据。
   *
   * 按你的要求把标题从「可能的意图」改成「目标意图」，内容收敛为意图集合本身，
   * 去掉了状态信息 / 活动规律 / 装备参数——这一栏要回答的是"这个目标想干什么"，
   * 其余读数会把它淹掉（`deriveActivity` 与 `headingStd` 因此暂时没有调用方，
   * 保留实现以便将来需要时挂回来）。
   *
   * 意图集合那块**直接复用** `IntentUi.renderIntentBlock`，不重写：集合大小、概率、
   * 状态徽章、判据的呈现已有既定实现与验收，重写只会引入分歧。
   *
   * 头部只保留"这是哪个目标"的识别信息（军标 + 编号 + 名称）——没有它，意图集合
   * 就不知道是谁的。
   */
  function renderTargetStatus(hostEl, def, frameState, l1, entry, frames, tick) {
    if (!hostEl) return;
    if (!def) {
      hostEl.innerHTML = '<div class="ts-empty">未选择目标</div>';
      return;
    }

    var html = '<div class="ts-head">'
      + '<img class="ts-sym" src="' + Symbology.symbolUri(def.icon, def.affiliation, {}) + '" alt="">'
      + '<div class="ts-id"><b>' + esc(def.id) + '</b>'
      + '<span>' + esc(def.name) + '</span></div></div>';

    html += '<div class="ts-sec">';
    if (window.IntentUi) {
      html += IntentUi.renderIntentBlock(entry, def);
    } else {
      html += '<div class="ts-empty">意图模块未加载</div>';
    }
    html += '</div>';

    hostEl.innerHTML = html;
  }

  window.StatusPanels = {
    FAMILY_ORDER: FAMILY_ORDER,
    TIERS: TIERS,
    renderOverall: renderOverall,
    renderTargetStatus: renderTargetStatus,
    deriveActivity: deriveActivity,
    headingStd: headingStd,
    _resetCache: function () { cache.key = null; cache.imgs = {}; },
  };
})();