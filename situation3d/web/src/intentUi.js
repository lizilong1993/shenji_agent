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
   * 目标面板顶部的「意图摘要」块。
   *
   * 这是本次 UI 改造的核心：ZHY 关注的是"这个目标在干什么"，所以意图集合放在
   * 面板最上方、用最大字号；一/二/三级细节降级到下方折叠区。
   */
  function renderIntentBlock(entry, def) {
    if (!entry) {
      return '<div class="tp-intent"><div class="tpi-head">意图</div>' +
        '<div class="tpi-empty">该帧没有该目标的意图判定数据。</div></div>';
    }
    var set = entry.intent_set || [];
    var st = entry.intent_status;
    var prim = set.filter(function (e) { return e.role !== 'secondary' && !isSecondaryOnly(e.code); });
    var sec = set.filter(function (e) { return e.role === 'secondary' || isSecondaryOnly(e.code); });

    // 模型与规则两条链路给分数字段不同名：CES 模型给 `score`（概率），
    // 规则给 `weight`。早先只读 `weight`，于是走模型时置信度永远显示 0%——
    // 一个看起来"没坏"但完全错误的读数。这里两者都认。
    function scoreOf(e) {
      if (!e) return 0;
      if (typeof e.score === 'number') return e.score;
      if (typeof e.weight === 'number') return e.weight;
      return 0;
    }

    var isModel = entry.intent_source === 'model';
    var setSize = (typeof entry.intent_set_size === 'number')
      ? entry.intent_set_size : set.length;

    var html = '<div class="tp-intent">';

    // ---- 主导意图：一行，最大字号 ----
    html += '<div class="tpi-head">战术意图'
      + '<span class="tpi-src">'
      + (isModel
          ? '模型 ' + esc(entry.model_version || '') + ' · |C|=' + setSize
          : '规则（模型未加载）')
      + '</span></div>';

    if (prim.length) {
      var top = prim[0];
      html += '<div class="tpi-primary">' + esc(nameZh(top.code)) + '</div>';
      html += '<div class="tpi-sub">' + esc(parentZh(top.code))
        + ' · ' + (isModel ? '模型概率 ' : '规则权重 ')
        + Math.round(scoreOf(top) * 100) + '%'
        + (top.approximated ? ' · 近似判据' : '') + '</div>';
    } else if (setSize > 1) {
      // 集合非空但都是次要意图：说清"还没有主意图"，而不是显示"未判定"。
      html += '<div class="tpi-primary tpi-primary-none">尚未确定主意图</div>';
      html += '<div class="tpi-sub">|C|=' + setSize + '，当前只有次要意图候选</div>';
    } else if (st) {
      var si = statusInfo(st);
      html += '<div class="tpi-primary tpi-primary-none">'
        + esc(si ? si.zh : st) + '</div>';
      html += '<div class="tpi-sub">' + esc(si ? si.note : '') + '</div>';
    } else {
      html += '<div class="tpi-primary tpi-primary-none">未判定</div>';
    }

    // ---- 集合收缩：|C| 越小越接近结论 ----
    if (isModel) {
      html += '<div class="tpi-setbar">候选集合 <b>|C| = ' + setSize + '</b>';
      if (setSize === 0) {
        html += '（证据不足，无可支持意图）';
      } else if (setSize === 1) {
        html += '（已收敛到唯一意图）';
      } else {
        html += '（' + setSize + ' 个候选仍无法区分）';
      }
      html += '</div>';
      if (typeof entry.insufficiency === 'number') {
        html += '<div class="tpi-sub">证据不足度 u = '
          + entry.insufficiency.toFixed(3)
          + ' · 判定阈值 δ = '
          + (typeof entry.delta === 'number' ? entry.delta.toFixed(3) : '—')
          + '</div>';
      }
      if (entry.intent_carried) {
        // 步长采样：这一帧沿用上一次评分。不标注会让人以为每帧都在重新判定。
        html += '<div class="tpi-carry">沿用第 '
          + (entry.intent_eval_tick !== undefined ? entry.intent_eval_tick : '?')
          + ' 帧的判定（按步长采样）</div>';
      }
    }

    // ---- 状态徽章（与威胁度配色分开） ----
    if (st) {
      var si2 = statusInfo(st);
      html += '<span class="tpi-badge ' + esc(si2 ? si2.cls : 'ist-unknown') + '">'
        + esc(si2 ? si2.zh : st) + '</span>';
      if (st === 'AMBIGUOUS') {
        html += '<div class="tpi-note">多个意图证据强度相当。等决定性证据出现后集合会收缩，'
          + '这与模型结论一致。</div>';
      }
    }

    // ---- 意图集合，按父类分组 ----
    if (set.length > 1) {
      html += '<div class="tpi-set-title">意图集合（' + set.length + '）</div>';
      html += renderGrouped(prim, sec);
    } else if (set.length === 1 && prim.length) {
      html += '<div class="tpi-set-title">意图集合（1）</div>';
      html += renderGrouped(prim, sec);
    }

    // ---- 证据 ----
    var all = prim.concat(sec);
    if (all.length) {
      html += '<details class="tpi-evidence"><summary>判定依据</summary>';
      all.forEach(function (e) {
        html += '<div class="tpi-ev-item"><b>' + esc(nameZh(e.code)) + '</b>';
        (e.evidence || []).forEach(function (line) {
          html += '<div class="tpi-ev-line">· ' + esc(line) + '</div>';
        });
        html += '</div>';
      });
      html += '</details>';
    }

    // ---- 判不出来时，说清缺什么 ----
    var miss = entry.missing_evidence || [];
    if (!set.length) {
      html += '<div class="tpi-missing">本系统无法判定该类意图所需证据';
      if (miss.length) html += '：' + esc(miss.join('；'));
      html += '。</div>';
    }
    html += '</div>';
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