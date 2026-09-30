/**
 * 战术意图（22 种本体口径）的界面呈现。
 *
 * 为什么单独一个模块
 * ------------------
 * 意图信息有三处呈现（目标面板、热点表、区域横幅），且都要把 code 翻成中文名、
 * 按父类分组、区分主/次意图与判定状态。散在 app.js 里会重复三份同样的码表与
 * 分组逻辑；集中在这里之后，本体改版只需改一处。
 *
 * 本体从服务端取（GET /api/intent/ontology），**前端不内置 22 条码表**：
 * 专家会后细化定义时，界面自动跟随，不会出现"界面里的意图名与本体不一致"。
 *
 * 两类信息在视觉上必须分开
 * ------------------------
 * * **判定状态**（BACKGROUND / UNKNOWN / AMBIGUOUS / OUT_OF_ONTOLOGY）——
 *   回答"这次判定有多可信"；
 * * **威胁度/告警等级**——回答"有多危险"。
 * 两者含义完全不同，混用同一套配色会让人把"证据不足"读成"低威胁"。
 * 所以状态用的是中性/条纹徽章，威胁度沿用原有的红橙黄绿等级色。
 */
(function () {
  'use strict';

  var ontology = null;          // 服务端返回的本体（含 22 意图 + 6 父类）
  var loading = null;           // 去重：并发调用只发一次请求
  var loadError = null;

  // 判定状态的中文名与配色。配色刻意避开威胁度的红橙黄绿：
  // 状态说的是"证据够不够"，不是"危不危险"。
  var STATUS_STYLE = {
    BACKGROUND: { zh: '背景', cls: 'ist-bg', note: '未识别出可判定的战术意图' },
    UNKNOWN: { zh: '证据不足', cls: 'ist-unknown', note: '现有证据不足以判定意图' },
    AMBIGUOUS: { zh: '存在歧义', cls: 'ist-ambiguous', note: '多个意图证据强度相当，未能区分' },
    OUT_OF_ONTOLOGY: { zh: '本体之外', cls: 'ist-ooo', note: '行为不吻合 22 类中的任何一类' },
  };

  function load() {
    if (ontology) return Promise.resolve(ontology);
    if (loading) return loading;
    loading = fetch('/api/intent/ontology', { cache: 'no-store' })
      .then(function (r) {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      })
      .then(function (doc) {
        ontology = doc;
        loadError = null;
        return doc;
      })
      .catch(function (e) {
        loadError = String(e && e.message ? e.message : e);
        return null;
      })
      .then(function (d) { loading = null; return d; });
    return loading;
  }

  /** 从数据集自带的 model 块读本体（离线可用，不依赖接口）。 */
  function adoptFromModel(model) {
    if (!ontology && model && model.intent_ontology) ontology = model.intent_ontology;
    return ontology;
  }

  function intentByCode(code) {
    if (!ontology || !ontology.intents) return null;
    for (var i = 0; i < ontology.intents.length; i++) {
      if (ontology.intents[i].code === code) return ontology.intents[i];
    }
    return null;
  }

  function parentByCode(code) {
    if (!ontology || !ontology.parents) return null;
    for (var i = 0; i < ontology.parents.length; i++) {
      if (ontology.parents[i].code === code) return ontology.parents[i];
    }
    return null;
  }

  function nameZh(code) {
    var it = intentByCode(code);
    return (it && it.name_zh) || code;
  }

  function parentZh(code) {
    var it = intentByCode(code);
    if (!it) return '';
    var p = parentByCode(it.parent_code);
    return p ? p.name_zh : '';
  }

  function isSecondaryOnly(code) {
    var it = intentByCode(code);
    return !!it && String(it.primary_vs_secondary || '').indexOf('secondary') === 0;
  }

  function statusInfo(code) {
    return STATUS_STYLE[code] || null;
  }

  function esc(s) {
    return String(s === null || s === undefined ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  /**
   * 目标意图栏的内容：**只有意图集合与每个意图的置信度**。
   *
   * 按反馈收敛。这里原先还有：模型版本与 |C|、证据不足度 u 与判定阈值 δ、
   * 「多个意图证据强度相当…」的叙述、状态徽章（存在歧义等）、「判定依据」折叠块，
   * 以及按父类分组的主/次列表。那些是给算法调试看的；这一栏要回答的只有两件事——
   * **可能是哪几个意图、各有多可信**。
   *
   * 完整判据没有丢：仍在数据集的 `intent_set[].evidence` 里，需要时从 API 或
   * `SituationHost` 取，不占界面。
   *
   * 空集合仍要显式说明一句，否则一片空白会被当成界面坏了。那不是"说明性文字"，
   * 而是结论本身：22 种意图里有 6 种因缺战场态字段（bridge_* 恒为 0）无法推导。
   */
  function renderIntentBlock(entry, def) {
    var set = (entry && entry.intent_set) || [];
    if (!set.length) {
      return '<div class="tpi-empty">无可支持的意图</div>';
    }
    // 模型链路给 `score`（概率），规则链路给 `weight`；两者都认——早先只读 weight，
    // 于是走模型时置信度永远显示 0%，是个看起来没坏、实际完全错误的读数。
    function scoreOf(e) {
      if (!e) return 0;
      if (typeof e.score === 'number') return e.score;
      if (typeof e.weight === 'number') return e.weight;
      return 0;
    }
    // 按置信度降序：读的人第一个看到的应该是最可能的那个。
    var rows = set.slice().sort(function (a, b) { return scoreOf(b) - scoreOf(a); });
    var html = '';
    rows.forEach(function (e) {
      var pct = Math.round(scoreOf(e) * 100);
      html += '<div class="tpi-row">'
        + '<span class="tpi-name">' + esc(nameZh(e.code)) + '</span>'
        + '<span class="tpi-bar"><i style="width:' + pct + '%"></i></span>'
        + '<span class="tpi-pct">' + pct + '%</span>'
        + '</div>';
    });
    return html;
  }

  /** 按父类分组渲染意图项，主/次用不同字重。 */
  function renderGrouped(prim, sec) {
    var byParent = {};
    var order = [];
    function push(e, roleLabel) {
      var it = intentByCode(e.code);
      var pc = (it && it.parent_code) || 'OTHER';
      if (!byParent[pc]) { byParent[pc] = []; order.push(pc); }
      byParent[pc].push({ e: e, roleLabel: roleLabel });
    }
    prim.forEach(function (e) { push(e, '主'); });
    sec.forEach(function (e) { push(e, '次'); });

    var html = '';
    order.forEach(function (pc) {
      var p = parentByCode(pc);
      html += '<div class="tpi-group"><span class="tpi-group-name">'
        + esc(p ? p.name_zh : '其它') + '</span>';
      byParent[pc].forEach(function (item) {
        var e = item.e;
        // 模型给出的是概率，显示在标签上；规则给权重。两者都不是"准确率"，
        // 所以只在标签内以百分比呈现，不写成"置信度"以免被读成可信程度。
        var s = (typeof e.score === 'number') ? e.score
              : (typeof e.weight === 'number') ? e.weight : null;
        html += '<span class="tpi-chip' + (item.roleLabel === '次' ? ' tpi-chip-sec' : '')
          + '" title="' + esc((e.evidence || []).join('；')) + '">'
          + esc(nameZh(e.code))
          + (s !== null ? '<i class="tpi-pct">' + Math.round(s * 100) + '%</i>' : '')
          + '<i class="tpi-role">' + esc(item.roleLabel) + '</i>'
          + (e.approximated ? '<i class="tpi-approx" title="用近似判据推出">≈</i>' : '')
          + '</span>';
      });
      html += '</div>';
    });
    return html;
  }

  /** 全局/区域摘要：各意图出现频次，用于横幅与热点表。 */
  function topCounts(intentSet, n) {
    var counts = {};
    (intentSet || []).forEach(function (e) {
      if (e.role === 'secondary' || isSecondaryOnly(e.code)) return;
      counts[e.code] = (counts[e.code] || 0) + 1;
    });
    return Object.keys(counts)
      .map(function (c) { return { code: c, zh: nameZh(c), n: counts[c] }; })
      .sort(function (a, b) { return b.n - a.n; })
      .slice(0, n || 3);
  }

  function debug() {
    return {
      loaded: !!ontology,
      loadError: loadError,
      version: ontology ? ontology.version : null,
      intentCount: ontology && ontology.intents ? ontology.intents.length : 0,
      parentCount: ontology && ontology.parents ? ontology.parents.length : 0,
      statusLayers: ontology ? (ontology.status_layers || []) : [],
      underivable: ontology && ontology.underivable ? ontology.underivable.length : null,
    };
  }

  window.IntentUi = {
    load: load,
    adoptFromModel: adoptFromModel,
    nameZh: nameZh,
    parentZh: parentZh,
    isSecondaryOnly: isSecondaryOnly,
    statusInfo: statusInfo,
    renderIntentBlock: renderIntentBlock,
    topCounts: topCounts,
    renderGrouped: renderGrouped,
    debug: debug,
    _ontology: function () { return ontology; },
  };
})();
