/* 情报管理：统一列表 / 人工录入 / 情报源管理。
 *
 * 数据全部走同源 /api/intel/*；人工情报是结构化记录，不是仿真真值。
 * 人工优先只影响“展示与选用”，verification 仍由分析员明确设置。
 */
(function (global) {
  'use strict';

  var DOMAINS = [
    ['hazard', '灾害'], ['conflict', '冲突'], ['maritime', '海上'],
    ['aviation', '空中'], ['infrastructure', '基础设施'], ['weather', '气象'],
    ['news', '新闻'], ['geography', '地理'], ['society', '人文社会'],
    ['social', '社交媒体'], ['selfmedia', '自媒体'], ['cyber', '网络'],
    ['health', '卫生'], ['other', '其他']
  ];
  var TYPES = [
    ['target', '目标'], ['aircraft', '飞机'], ['vessel', '舰船'],
    ['ground', '地面目标'], ['earthquake', '地震'], ['fire', '火灾'],
    ['cyclone', '气旋/台风'], ['flood', '洪水'], ['weather', '气象'],
    ['news', '新闻'], ['geography', '地理'], ['society', '人文社会'],
    ['social', '社交媒体'], ['selfmedia', '自媒体'],
    ['infrastructure', '基础设施'], ['threat', '威胁'], ['other', '其他']
  ];
  var VERIFY = [
    ['unverified', '未核实'], ['verified', '已核实'], ['disputed', '有争议']
  ];
  var SOURCE_KINDS = [
    ['auto', '自动连接器'], ['manual', '人工'], ['rss', 'RSS/Atom'],
    ['weather', '气象'], ['web', '网页'], ['social', '社交媒体'],
    ['other', '其他']
  ];

  var state = {
    tab: 'list', overview: null, sources: [], error: null,
    filter: '', priority: 'all', effective: 'all', domain: 'all',
    sourceFilter: '', sourceKind: 'all', sourceEnabled: 'all',
    editing: null, timer: null, loading: false, collecting: false,
    collectStarted: 0, autoTried: {}, lastCase: null, caseRetries: 0,
    docked: false, feedCase: null, feedSeen: {}
  };

  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function modal() {
    return document.getElementById('intelModal');
  }

  function caseId() {
    try {
      var host = global.SituationHost;
      var s = host && host.state;
      return String((s && (s.caseId || (s.data && s.data.case_id)))
        || 'default');
    } catch (e) {
      return 'default';
    }
  }

  function api(path, options) {
    var opts = options || {};
    opts.cache = 'no-store';
    if (opts.body && typeof opts.body !== 'string') {
      opts.headers = Object.assign({ 'Content-Type': 'application/json' },
        opts.headers || {});
      opts.body = JSON.stringify(opts.body);
    }
    return fetch(path, opts).then(function (res) {
      return res.json().catch(function () { return {}; }).then(function (data) {
        if (!res.ok) {
          throw new Error(data.error || ('HTTP ' + res.status));
        }
        return data;
      });
    });
  }

  function notify(level, content, raw) {
    if (global.EventFeed && typeof EventFeed.push === 'function') {
      try {
        EventFeed.push({
          level: level || 'info', source: 'intel', object: 'manual',
          content: content, raw: raw || {}
        });
      } catch (e) { /* event feed must not break the panel */ }
    }
  }

  function label(pairs, value) {
    for (var i = 0; i < pairs.length; i++) {
      if (pairs[i][0] === value) return pairs[i][1];
    }
    return value || '—';
  }

  function timeLabel(value) {
    if (!value) return '—';
    var d = new Date(value);
    if (isNaN(d.getTime())) return String(value);
    return d.toLocaleString('zh-CN', { hour12: false });
  }

  function isoLocal(value) {
    if (!value) return '';
    var d = new Date(value);
    if (isNaN(d.getTime())) return '';
    return new Date(d.getTime() - d.getTimezoneOffset() * 60000)
      .toISOString().slice(0, 16);
  }

  function sourceName(item) {
    return item.source_name || item.source_id || '未知来源';
  }

  function open(tab) {
    state.tab = tab || state.tab || 'list';
    state.editing = null;
    state.domain = 'all';
    state.caseRetries = 0;
    var m = modal();
    if (!m) return;
    m.classList.remove('hidden');
    applyDock();
    refresh();
    onCase(caseId(), situationData());
    startPoll();
  }

  /**
   * 停靠模式：面板贴到右侧，遮罩透明且不吃鼠标事件，地图仍可缩放/点选。
   * 默认仍是居中模态（情报研判需要大版面），停靠是"边看地图边翻情报"时用的。
   * 状态存在 localStorage，避免每次打开都要重设。
   */
  function applyDock() {
    var m = modal();
    var btn = document.getElementById('intelDock');
    if (!m) return;
    m.classList.toggle('docked', !!state.docked);
    if (btn) {
      btn.setAttribute('aria-pressed', state.docked ? 'true' : 'false');
      btn.textContent = state.docked ? '浮层' : '停靠';
    }
  }

  function situationData() {
    try {
      return (global.SituationHost && global.SituationHost.state
        && global.SituationHost.state.data) || null;
    } catch (e) {
      return null;
    }
  }

  function pipelineFresh(pipeline) {
    if (!pipeline || !pipeline.generated_at) return false;
    var stamp = Date.parse(pipeline.generated_at);
    return Number.isFinite(stamp) && (Date.now() - stamp) < 6 * 3600 * 1000;
  }

  function onCase(caseIdValue, data) {
    var cid = String(caseIdValue || '');
    if (!cid || cid === 'default') return;
    state.domain = 'all';
    if (state.autoTried[cid]) return;
    // Case profiles on the server own the AOI; the browser bbox is only a
    // fallback for cases without a profile, so it must not gate collection.
    var bbox = data && data.aoi;
    var body = { case_id: cid, name: (data && data.case_name) || cid };
    if (Array.isArray(bbox) && bbox.length === 4) body.bbox = bbox;
    state.autoTried[cid] = true;
    api('/api/intel/overview?case_id=' + encodeURIComponent(cid))
      .then(function (overview) {
        if (pipelineFresh(overview.pipeline)) return;
        state.collecting = true;
        state.collectStarted = Date.now();
        api('/api/intel/collect', { method: 'POST', body: body })
          .then(function () {
            notify('info', '已启动案例情报采集：' + cid, {});
            setTimeout(refresh, 1500);
            setTimeout(refresh, 6000);
          }).catch(function (err) {
            // Demo mode is read-only; leave the existing view untouched.
            state.collecting = false;
            if (err && /disabled|403/.test(String(err.message))) return;
            delete state.autoTried[cid];
            notify('warn', '自动采集未启动：' + (err.message || err), {});
          });
      })
      .catch(function () {
        state.collecting = false;
        delete state.autoTried[cid];
      });
  }

  function close() {
    var m = modal();
    if (m) m.classList.add('hidden');
    stopPoll();
  }

  function startPoll() {
    stopPoll();
    state.timer = setInterval(function () {
      if (state.tab === 'list' && !state.editing && !state.loading &&
          !isEditingField()) refresh();
    }, 5000);
  }

  function isEditingField() {
    var active = document.activeElement;
    if (!active || !active.closest || !active.closest('#intelBody')) return false;
    return ['INPUT', 'TEXTAREA', 'SELECT'].indexOf(active.tagName) >= 0;
  }

  function stopPoll() {
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
  }

  function refresh() {
    if (!modal() || modal().classList.contains('hidden')) return Promise.resolve();
    if (state.loading) return Promise.resolve();
    state.loading = true;
    var requested = caseId();
    // The panel can be opened before the case finishes loading; poll for the
    // first real case id so collection is not dependent on startup ordering.
    if (requested !== 'default' && requested !== state.lastCase) {
      state.lastCase = requested;
      onCase(requested, situationData());
    }
    var url = '/api/intel/overview?case_id=' + encodeURIComponent(requested);
    return api(url).then(function (data) {
      // A deep link can open the panel before the situation finishes loading;
      // retry briefly so it does not get stuck on the default case.
      var host = global.SituationHost;
      if (requested === 'default' && host && !host.state.caseId &&
          (state.caseRetries || 0) < 10) {
        state.caseRetries = (state.caseRetries || 0) + 1;
        setTimeout(refresh, 800);
        return;
      }
      state.caseRetries = 0;
      if (state.collecting) {
        var stamp = Date.parse((data.pipeline || {}).generated_at || '');
        if ((Number.isFinite(stamp) && stamp >= state.collectStarted - 5000) ||
            (state.collectStarted &&
             Date.now() - state.collectStarted > 10 * 60 * 1000)) {
          state.collecting = false;
        }
      }
      state.overview = data;
      state.sources = data.sources || [];
      state.error = null;
      emitConvergenceToFeed(data);
      render();
    }).catch(function (err) {
      state.error = err.message || String(err);
      renderError();
    }).finally(function () {
      state.loading = false;
    });
  }

  /**
   * 汇聚事件进统一事件流。
   *
   * 情报面板 5 秒轮询一次，同一条汇聚会在很多轮里反复出现；事件流按"每轮
   * 都推一遍"会把真正的新消息冲掉。所以按事件 id 记住推过的，只推新的。
   * 集合要随案例清空——不同案例的 id 空间不同，混在一起会漏报。
   */
  function emitConvergenceToFeed(overview) {
    var block = overview && overview.convergence;
    var events = (block && block.events) || [];
    if (!global.EventFeed || typeof global.EventFeed.push !== 'function') return;
    if (state.feedCase !== (overview && overview.case_id)) {
      state.feedCase = overview && overview.case_id;
      state.feedSeen = {};
    }
    var seen = state.feedSeen || (state.feedSeen = {});
    events.forEach(function (ev) {
      var id = String(ev.id || '');
      if (!id || seen[id]) return;
      seen[id] = true;
      try {
        global.EventFeed.push({
          level: 'warn',
          source: 'intel',
          object: 'convergence',
          content: '区域汇聚：' + (ev.types || []).join(' + ') +
            '（' + (ev.member_count || 0) + ' 条，' + (ev.source_ids || []).join('/') + '）',
          raw: { id: id, score: ev.score, cell: ev.cell,
                 t_start: ev.t_start, t_end: ev.t_end }
        });
      } catch (e) { /* 事件流不能反过来打断情报面板 */ }
    });
  }

  function renderError() {
    var body = document.getElementById('intelBody');
    if (!body) return;
    body.innerHTML = '';
    var box = el('div', 'intel-error', '情报接口失败：' + (state.error || '未知错误'));
    body.appendChild(box);
    var retry = el('button', 'mini', '重试');
    retry.type = 'button';
    retry.addEventListener('click', function () { refresh(); });
    body.appendChild(retry);
  }

  function setTab(tab) {
    state.tab = tab;
    state.editing = null;
    if (tab === 'list') startPoll(); else stopPoll();
    render();
  }

  function render() {
    var body = document.getElementById('intelBody');
    if (!body) return;
    Array.prototype.forEach.call(
      document.querySelectorAll('#intelTabs [data-intel-tab]'),
      function (button) {
        button.classList.toggle('active', button.dataset.intelTab === state.tab);
      });
    body.innerHTML = '';
    if (!state.overview) {
      body.appendChild(el('div', 'intel-empty', '正在加载情报…'));
      return;
    }
    if (state.tab === 'manual') renderManual(body);
    else if (state.tab === 'sources') renderSources(body);
    else if (state.tab === 'board') renderBoard(body);
    else renderList(body);
  }

  var SEVERITY = {
    high: ['高相关度', '#c8352c'],
    medium: ['中相关度', '#c98a00'],
    low: ['低相关度', '#4f86c6']
  };

  function ageLabel(value) {
    if (!value) return '';
    var stamp = Date.parse(value);
    if (!Number.isFinite(stamp)) return '';
    if (stamp > Date.now() + 60000) return '';  // future: keep the absolute time only
    var minutes = Math.round((Date.now() - stamp) / 60000);
    if (minutes < 1) return '刚刚';
    if (minutes < 60) return minutes + ' 分钟前';
    var hours = Math.round(minutes / 60);
    if (hours < 48) return hours + ' 小时前';
    return Math.round(hours / 24) + ' 天前';
  }

  function relevanceOf(item) {
    var props = item.record && item.record.properties;
    var value = props && props.relevance;
    return SEVERITY[value] ? value : '';
  }

  /**
   * 汇聚事件区块：同一网格 24h 内出现三类以上不同情报时，指挥员要一眼看到
   * "哪里在聚集"，而不是自己在几百张卡片里比对。
   *
   * 后端早就算好了（overview.convergence），这里只负责呈现：
   * 每张卡给位置、时段、类型组合、评分与成员；点卡片定位到格中心。
   */
  function renderConvergence(body, overview) {
    var block = overview.convergence || null;
    var events = (block && block.events) || [];
    if (!events.length) return;   // 没有汇聚时不占版面
    var wrap = el('div', 'intel-convergence');
    var head = el('div', 'intel-conv-head');
    head.appendChild(el('span', 'intel-conv-title', '区域汇聚'));
    head.appendChild(el('span', 'intel-conv-count', events.length + ' 处'));
    var rule = (block.rule || {});
    if (rule.window_h) {
      head.appendChild(el('span', 'intel-conv-rule',
        rule.window_h + 'h 内 ' + (rule.min_types || 3) + ' 类以上'));
    }
    wrap.appendChild(head);

    events.slice(0, 6).forEach(function (ev) {
      var card = el('div', 'intel-conv-card');
      var top = el('div', 'intel-conv-top');
      top.appendChild(el('span', 'intel-conv-score',
        Number(ev.score || 0).toFixed(2)));
      top.appendChild(el('span', 'intel-conv-types',
        (ev.types || []).join(' + ')));
      card.appendChild(top);
      card.appendChild(el('div', 'intel-conv-meta',
        (ev.t_start || '').slice(5, 16) + ' → ' + (ev.t_end || '').slice(5, 16) +
        ' · ' + (ev.member_count || 0) + ' 条 · ' +
        (ev.source_ids || []).join('/')));
      // 理由只放第一行："同格 N 类信号"这句话本身说明了一切
      var reason = (ev.reasons || [])[0];
      if (reason) card.appendChild(el('div', 'intel-conv-reason', reason));

      var cell = ev.cell || {};
      if (Number.isFinite(Number(cell.center_lat)) &&
          Number.isFinite(Number(cell.center_lon)) && global.SituationHost &&
          typeof global.SituationHost.focusIntel === 'function') {
        var locate = el('button', 'mini intel-conv-locate', '定位');
        locate.type = 'button';
        locate.title = '定位到这处汇聚所在网格';
        locate.addEventListener('click', function () {
          // 用格边长的一半当半径，圈出整格
          var half = Number(cell.cell_deg || 0) * 111000 / 2;
          global.SituationHost.focusIntel(Number(cell.center_lon),
            Number(cell.center_lat), half, '汇聚 ' + (ev.types || []).length + ' 类');
        });
        card.appendChild(locate);
      }
      card.title = (ev.reasons || []).join('\n');
      wrap.appendChild(card);
    });
    body.appendChild(wrap);
  }

  function renderBrief(body, overview) {
    var counts = overview.counts || {};
    var cache = overview.cache || null;
    var brief = el('div', 'intel-brief');
    var head = el('div', 'intel-brief-head');
    head.appendChild(el('span', 'intel-brief-title', '态势简报'));
    if (cache && cache.observations_7d !== undefined) {
      head.appendChild(el('span', 'intel-brief-fresh',
        '7 天缓存 ' + cache.observations_7d + ' 条'));
    }
    brief.appendChild(head);

    var byId = {};
    (overview.items || []).forEach(function (item) { byId[item.id] = item; });
    var keyItems = (overview.key_items || []).slice(0, 3);
    if (keyItems.length) {
      keyItems.forEach(function (row) {
        var item = byId[row.id];
        if (!item) return;
        var line = el('div', 'intel-brief-row');
        line.appendChild(el('span', 'intel-brief-score',
          Number(row.score || 0).toFixed(2)));
        line.appendChild(el('span', 'intel-brief-text',
          (item.priority === 'manual' ? '【人工】' : '') +
          (item.title || '（无标题）')));
        var reason = (row.reasons || []).slice(0, 2).join('；');
        if (reason) line.appendChild(el('span', 'intel-brief-reason', reason));
        brief.appendChild(line);
      });
    } else {
      brief.appendChild(el('div', 'intel-brief-row intel-note',
        '暂无关键情报；采集完成后自动抽取。'));
    }

    var stat = el('div', 'intel-brief-stats');
    stat.appendChild(el('span', 'intel-chip', '自动 ' + (counts.auto || 0)));
    stat.appendChild(el('span', 'intel-chip', '人工 ' + (counts.manual_active || 0)));
    stat.appendChild(el('span', 'intel-chip', '关联 ' + (counts.links || 0)));
    stat.appendChild(el('span', 'intel-chip', '冲突覆盖 ' +
      (counts.superseded || 0)));
    var ai = overview.ai || {};
    if (ai.counts && ai.counts.total) {
      stat.appendChild(el('span', 'intel-chip', 'AI 建议 ' + ai.counts.total +
        '（待审 ' + (ai.counts.suggested || 0) + '）'));
    }
    var pipeline = overview.pipeline || null;
    if (pipeline && pipeline.generated_at) {
      stat.appendChild(el('span', 'intel-brief-time', '最近采集 ' +
        ageLabel(pipeline.generated_at) + '（' +
        timeLabel(pipeline.generated_at) + '）' +
        (state.collecting ? ' · 正在补充采集…' : '')));
    } else if (state.collecting) {
      stat.appendChild(el('span', 'intel-brief-time', '正在采集…'));
    }
    brief.appendChild(stat);
    body.appendChild(brief);
  }

  function renderFreshness(body, overview) {
    var cache = overview.cache;
    if (!cache || !cache.sources) return;
    var names = {};
    (overview.sources || []).forEach(function (source) {
      names[source.source_id] = source.name || source.source_id;
    });
    var rows = Object.keys(cache.sources).map(function (sourceId) {
      return {
        sourceId: sourceId,
        name: names[sourceId] || sourceId,
        entry: cache.sources[sourceId] || {}
      };
    }).sort(function (a, b) {
      return a.sourceId < b.sourceId ? -1 : a.sourceId > b.sourceId ? 1 : 0;
    });
    if (!rows.length) return;
    var strip = el('div', 'intel-fresh');
    strip.appendChild(el('span', 'intel-fresh-label', '数据源新鲜度'));
    rows.forEach(function (row) {
      var stamp = Date.parse(row.entry.last_fetch_at || '');
      var stale = Number.isFinite(stamp) &&
        (Date.now() - stamp) > 26 * 3600 * 1000;
      var failed = Boolean(row.entry.last_error);
      var chip = el('span', 'intel-fresh-chip' +
        (failed ? ' bad' : stale ? ' stale' : ''),
        row.name + ' · ' + (row.entry.observations || 0) + ' 条 · ' +
        (failed ? '采集失败' : row.entry.last_fetch_at
          ? ageLabel(row.entry.last_fetch_at) : '未采集'));
      if (failed) chip.title = '采集失败，详情见服务端日志';
      strip.appendChild(chip);
    });
    body.appendChild(strip);
  }

  function renderList(body) {
    var overview = state.overview;
    var all = overview.items || [];
    var domainCounts = {};
    all.forEach(function (item) {
      domainCounts[item.domain] = (domainCounts[item.domain] || 0) + 1;
    });

    renderBrief(body, overview);
    renderConvergence(body, overview);
    renderFreshness(body, overview);
    (overview.warnings || []).forEach(function (warning) {
      body.appendChild(el('div', 'intel-warn', warning));
    });

    var toolbar = el('div', 'intel-toolbar intel-toolbar-chips');
    var domains = el('div', 'intel-domains');
    function domainChip(value, text) {
      var button = el('button', 'intel-domain-chip' +
        (state.domain === value ? ' active' : ''), text);
      button.type = 'button';
      button.addEventListener('click', function () {
        state.domain = value;
        render();
      });
      return button;
    }
    domains.appendChild(domainChip('all', '全部 ' + all.length));
    DOMAINS.forEach(function (pair) {
      var count = domainCounts[pair[0]] || 0;
      if (!count && state.domain !== pair[0]) return;
      domains.appendChild(domainChip(pair[0], pair[1] + ' ' + count));
    });
    toolbar.appendChild(domains);

    var filters = el('div', 'intel-toolbar-filters');
    var search = document.createElement('input');
    search.type = 'search';
    search.className = 'intel-input';
    search.placeholder = '搜索标题/内容/来源/类型';
    search.value = state.filter;
    search.addEventListener('change', function () {
      state.filter = search.value.trim();
      render();
    });
    filters.appendChild(search);
    filters.appendChild(selectInput([
      ['all', '全部来源'], ['manual', '仅人工'], ['auto', '仅自动']
    ], state.priority, function (value) {
      state.priority = value; render();
    }));
    filters.appendChild(selectInput([
      ['all', '全部状态'], ['effective', '当前采用'], ['superseded', '已覆盖/撤销']
    ], state.effective, function (value) {
      state.effective = value; render();
    }));
    var add = el('button', 'mini primary', '+ 人工情报');
    add.type = 'button';
    add.addEventListener('click', function () {
      state.editing = null;
      state.tab = 'manual';
      stopPoll();
      render();
    });
    filters.appendChild(add);
    toolbar.appendChild(filters);
    body.appendChild(toolbar);

    var items = all.filter(function (item) {
      if (state.domain !== 'all' && item.domain !== state.domain) return false;
      if (state.priority !== 'all' && item.priority !== state.priority) return false;
      if (state.effective === 'effective' && !item.effective) return false;
      if (state.effective === 'superseded' && item.effective) return false;
      if (state.filter) {
        var text = [
          item.title, item.content, item.source_name, item.source_id,
          item.type, item.domain, item.place
        ].join(' ').toLowerCase();
        if (text.indexOf(state.filter.toLowerCase()) < 0) return false;
      }
      return true;
    });

    if (!items.length) {
      body.appendChild(el('div', 'intel-empty', '没有符合条件的情报'));
      return;
    }
    if (state.domain === 'all') {
      var groups = {};
      var order = [];
      items.forEach(function (item) {
        var key = item.domain || 'other';
        if (!groups[key]) { groups[key] = []; order.push(key); }
        groups[key].push(item);
      });
      order.forEach(function (key) {
        var group = el('div', 'intel-group');
        group.appendChild(el('span', 'intel-group-title', label(DOMAINS, key)));
        group.appendChild(el('span', 'intel-group-count',
          groups[key].length + ' 条'));
        body.appendChild(group);
        groups[key].forEach(function (item) {
          body.appendChild(card(item, overview));
        });
      });
    } else {
      items.forEach(function (item) { body.appendChild(card(item, overview)); });
    }
  }

  // ---- Palantir 借鉴：完整度 / 时效 / 定位精度 / 质量 ----
  //
  // 这四项的共同点是"缺失本身就是信息"：一条没有来源、定位只到省一级、
  // 八小时没更新的情报，和一条刚采集到的精确点位，在研判上是两种东西。
  // 以前这些差别只能靠人把各字段对照一遍才看得出来。

  var QUALITY = {
    measured: ['实测', 'ok'],
    interpolated: ['推断', 'warn'],
    predicted: ['外推', 'warn'],
    stale: ['陈旧', 'bad']
  };

  /** 把米换算成可读的 ±X：<1km 用米，更大用公里。 */
  function uncertaintyLabel(meters) {
    var value = Number(meters);
    if (!Number.isFinite(value) || value <= 0) return '';
    if (value < 1000) return '±' + Math.round(value) + ' m';
    return '±' + (value / 1000).toFixed(value < 10000 ? 1 : 0) + ' km';
  }

  /**
   * 情报完整度：位置 / 来源 / 关联 / 时效 四项固定输出。
   * 缺项不隐藏，写"无"并置灰——隐藏缺失会让人以为这条情报已经齐备。
   */
  function renderCompleteness(item, overview, hasLinks) {
    var box = el('div', 'intel-complete');
    var items = [
      ['位置', !!uncertaintyLabel(item.geo_uncertainty_m) || (
        Number.isFinite(Number(item.lat)) && Number.isFinite(Number(item.lon))
      )],
      ['来源', !!(item.source_name || item.source_id) &&
        item.source_id !== 'unknown'],
      ['关联', !!hasLinks],
      ['时效', !!(item.ingest_time || item.event_time)]
    ];
    items.forEach(function (pair) {
      var chip = el('span', 'intel-complete-chip' + (pair[1] ? '' : ' missing'),
        pair[0] + (pair[1] ? ' ✓' : ' 无'));
      chip.title = pair[1] ? '' : '这项信息缺失，研判时需额外确认';
      box.appendChild(chip);
    });
    return box;
  }

  /** 对象级时效：回答"这条多久没更新"，与"事件发生多久"是两回事。 */
  function renderFreshness(item) {
    var box = el('div', 'intel-fresh-item');
    var quality = item.quality || {};
    var level = String(quality.level || '');
    if (QUALITY[level]) {
      var chip = el('span', 'intel-quality ' + QUALITY[level][1],
        QUALITY[level][0]);
      chip.title = (quality.reasons || []).join('\n');
      box.appendChild(chip);
    }
    var age = quality.age_h;
    if (typeof age === 'number' && Number.isFinite(age)) {
      var half = Number(quality.half_life_h) || 0;
      var stale = half > 0 && age > half;
      var label = age < 1 ? '刚刚' : (age < 48 ? Math.round(age) + ' 小时前'
        : Math.round(age / 24) + ' 天前');
      var node = el('span', 'intel-age' + (stale ? ' stale' : ''),
        '更新于 ' + label);
      if (stale && half) {
        node.title = '已超过该领域半衰期（' + Math.round(half) + ' 小时）';
      }
      box.appendChild(node);
    }
    var radius = uncertaintyLabel(item.geo_uncertainty_m);
    if (radius) {
      var geo = el('span', 'intel-geo',
        '定位 ' + radius + (item.is_centroid ? '（质心）' : ''));
      geo.title = item.is_centroid
        ? '该位置是行政区/国家质心，不是实测点位'
        : '定位误差半径';
      box.appendChild(geo);
    }
    // 定位到地图：把这条观测的误差圈画出来。有坐标才给按钮，
    // 否则点了什么也不会发生。
    var lat = Number(item.lat);
    var lon = Number(item.lon);
    if (Number.isFinite(lat) && Number.isFinite(lon) && global.SituationHost &&
        typeof global.SituationHost.focusIntel === 'function') {
      var locate = el('button', 'mini intel-locate', '定位');
      locate.type = 'button';
      locate.title = '在地图上查看，并按定位误差画出置信圈';
      locate.addEventListener('click', function () {
        global.SituationHost.focusIntel(lon, lat,
          Number(item.geo_uncertainty_m) || 0, item.title || '');
      });
      box.appendChild(locate);
    }
    return box;
  }

  function card(item, overview) {
    var links = (overview.links || []).filter(function (link) {
      return link.from_id === item.id || link.to_id === item.id;
    });
    var relevance = relevanceOf(item);
    var node = el('div', 'intel-card intel-card-wm ' + item.priority +
      (item.effective ? '' : ' superseded') +
      (item.status === 'revoked' ? ' revoked' : ''));
    if (relevance) node.style.borderLeftColor = SEVERITY[relevance][1];
    var head = el('div', 'intel-card-head');
    if (relevance) {
      head.appendChild(el('span', 'intel-sev intel-sev-' + relevance,
        SEVERITY[relevance][0]));
    }
    head.appendChild(el('span', 'intel-badge ' + item.priority,
      item.priority === 'manual' ? '人工' : '自动'));
    if (item.priority === 'manual') {
      if (item.status !== 'revoked') {
        head.appendChild(el('span', 'intel-badge priority', '高优先级'));
        head.appendChild(el('span', 'intel-badge verify-' + item.verification,
          label(VERIFY, item.verification)));
      } else {
        head.appendChild(el('span', 'intel-badge revoked', '历史优先级'));
      }
    } else {
      head.appendChild(el('span', 'intel-badge source', sourceName(item)));
    }
    if (item.effective) head.appendChild(el('span', 'intel-badge on', '采用中'));
    else head.appendChild(el('span', 'intel-badge off',
      item.superseded_by ? '被人工情报覆盖' : '未采用'));
    if (item.status === 'revoked') {
      head.appendChild(el('span', 'intel-badge revoked', '已撤销'));
    }
    node.appendChild(head);

    node.appendChild(el('div', 'intel-title', item.title || '（无标题）'));
    var meta = el('div', 'intel-meta');
    meta.appendChild(el('span', null, timeLabel(item.event_time) +
      (ageLabel(item.event_time) ? ' · ' + ageLabel(item.event_time) : '')));
    meta.appendChild(el('span', null,
      label(DOMAINS, item.domain) + ' / ' + label(TYPES, item.type)));
    if (item.confidence !== null && item.confidence !== undefined) {
      var conf = Number(item.confidence);
      if (Number.isFinite(conf)) {
        meta.appendChild(el('span', 'intel-conf', '置信 ' + conf.toFixed(2)));
      }
    }
    var lat = Number(item.lat);
    var lon = Number(item.lon);
    if (Number.isFinite(lat) && Number.isFinite(lon)) {
      meta.appendChild(el('span', null,
        lat.toFixed(6) + ', ' + lon.toFixed(6) +
        (item.place ? ' · ' + item.place : '')));
    } else if (item.place) {
      meta.appendChild(el('span', null, item.place));
    }
    meta.appendChild(el('span', null, sourceName(item)));
    node.appendChild(meta);

    if (item.content) {
      node.appendChild(el('div', 'intel-content', item.content));
    }
    if (item.priority === 'manual' && item.verification === 'unverified') {
      node.appendChild(el('div', 'intel-warn',
        '人工情报优先级高，但尚未核实；仅作为线索，不作为已证实事实。'));
    }
    if (item.superseded_by) {
      node.appendChild(el('div', 'intel-note',
        '原始自动情报已保留；当前展示优先级让位于人工情报。'));
    }
    if ((item.conflicts || []).length) {
      node.appendChild(el('div', 'intel-note',
        '关联自动情报：' + item.conflicts.join('、')));
    }
    var links = (overview.links || []).filter(function (link) {
      return link.from_id === item.id || link.to_id === item.id;
    });
    if (links.length) {
      node.appendChild(el('div', 'intel-links',
        '跨源关联 ' + links.length + ' 条：' +
        links.slice(0, 2).map(function (link) {
          return link.kind + ' (' + link.score + ')';
        }).join('；')));
      // 人工确认的链接单独标出来：它的分数是 0.99，但更重要的是"这是人定的"。
      var manual = links.filter(function (link) { return link.kind === 'manual'; });
      if (manual.length) {
        node.appendChild(el('div', 'intel-note',
          '其中 ' + manual.length + ' 条已由分析员确认'));
      }
    }

    node.appendChild(renderCompleteness(item, overview, links.length > 0));
    node.appendChild(renderFreshness(item));

    if (item.priority === 'manual') {
      var actions = el('div', 'intel-actions');
      if (item.status !== 'revoked') {
        var edit = el('button', 'mini', '编辑');
        edit.type = 'button';
        edit.addEventListener('click', function () { editManual(item); });
        actions.appendChild(edit);
        var revoke = el('button', 'mini danger', '撤销');
        revoke.type = 'button';
        revoke.addEventListener('click', function () { revokeManual(item); });
        actions.appendChild(revoke);
      } else {
        var restore = el('button', 'mini', '重新启用');
        restore.type = 'button';
        restore.addEventListener('click', function () { restoreManual(item); });
        actions.appendChild(restore);
      }
      appendChildren(actions, cardActions(item));
      node.appendChild(actions);
    } else {
      node.appendChild(cardActions(item));
    }
    return node;
  }

  function selectInput(pairs, value, onChange, className, name) {
    var select = document.createElement('select');
    select.className = 'intel-select' + (className ? ' ' + className : '');
    if (name) select.name = name;
    pairs.forEach(function (pair) {
      var option = document.createElement('option');
      option.value = pair[0];
      option.textContent = pair[1];
      select.appendChild(option);
    });
    select.value = value;
    select.addEventListener('change', function () { onChange(select.value); });
    return select;
  }

  /**
   * 卡片上的快捷动作。Palantir 那套界面的做法是"选了对象就能在对象身上动手"，
   * 而这里以前要点开"人工录入"标签页才能把一条自动情报转成人工条目。
   * 动作只做最常用的两件，避免把卡片撑爆：
   *   - 转人工：带着当前卡片的内容跳进录入表单（原行为保留，但入口前移）
   *   - 标记禁止打击：直接写 NSL 字段，需要填理由
   */
  function appendChildren(target, source) {
    Array.prototype.slice.call(source.childNodes).forEach(function (child) {
      target.appendChild(child);
    });
    return target;
  }

  function cardActions(item) {
    var actions = el('div', 'intel-actions intel-card-actions');
    var toManual = el('button', 'mini', '转人工');
    toManual.type = 'button';
    toManual.title = '以这条自动情报为底稿新建一条人工情报';
    toManual.addEventListener('click', function () { convertToManual(item); });
    actions.appendChild(toManual);

    if (item.priority === 'manual') {
      if (item.no_strike) {
        var tag = el('span', 'intel-badge nsl', '禁止打击');
        if (item.no_strike_reason) tag.title = item.no_strike_reason;
        actions.appendChild(tag);
      } else {
        var nsl = el('button', 'mini', '禁止打击');
        nsl.type = 'button';
        nsl.title = '标记为禁止打击目标（NSL）';
        nsl.addEventListener('click', function () { markNoStrike(item); });
        actions.appendChild(nsl);
      }
    } else if (item.no_strike) {
      var inherited = el('span', 'intel-badge nsl', '禁止打击');
      actions.appendChild(inherited);
    }
    return actions;
  }

  /** 把自动情报作为底稿转成人工条目：字段能带的都带过去，id 不带。 */
  function convertToManual(item) {
    state.editing = {
      case_id: state.lastCase || 'default',
      title: item.title || '',
      domain: item.domain || 'other',
      intel_type: item.type || 'other',
      event_time: item.event_time || '',
      lat: item.lat,
      lon: item.lon,
      place: item.place || '',
      content: item.content || '',
      verification: 'unverified',
      tags: [],
      _converted_from: item.id
    };
    state.tab = 'manual';
    stopPoll();
    render();
  }

  function markNoStrike(item) {
    var reason = global.prompt('禁止打击的理由（例如：已确认为民用医院）', '');
    if (reason === null) return;
    var text = String(reason).trim();
    if (!text) {
      global.alert('理由不能为空：禁止打击必须有可追溯的依据。');
      return;
    }
    api('/api/intel/manual/' + encodeURIComponent(item.id),
      { method: 'POST', body: { no_strike: true, no_strike_reason: text } })
      .then(function () {
        notify('warn', '已标记禁止打击：' + (item.title || item.id),
          { id: item.id, reason: text });
        return refresh();
      }).catch(function (err) {
        global.alert('标记失败：' + err.message);
      });
  }

  function editManual(item) {
    state.editing = item;
    state.tab = 'manual';
    stopPoll();
    render();
  }

  function revokeManual(item) {
    if (!global.confirm('撤销这条人工情报？自动情报会重新成为有效记录。')) return;
    api('/api/intel/manual/' + encodeURIComponent(item.id) + '/revoke',
      { method: 'POST', body: {} }).then(function () {
      notify('warn', '人工情报已撤销：' + (item.title || item.id), { id: item.id });
      return refresh();
    }).catch(function (err) {
      global.alert('撤销失败：' + err.message);
    });
  }

  function restoreManual(item) {
    api('/api/intel/manual/' + encodeURIComponent(item.id),
      { method: 'POST', body: { status: 'active' } }).then(function () {
      notify('info', '人工情报已重新启用：' + (item.title || item.id), { id: item.id });
      return refresh();
    }).catch(function (err) {
      global.alert('启用失败：' + err.message);
    });
  }

  function renderManual(body) {
    var item = state.editing && state.editing.record
      ? state.editing.record : (state.editing || {});
    var editing = Boolean(state.editing);
    var form = el('form', 'intel-form');
    form.addEventListener('submit', function (event) {
      event.preventDefault();
      saveManual(form, editing ? item.intel_id : null);
    });

    var head = el('div', 'intel-form-head');
    head.appendChild(el('h3', null, editing ? '编辑人工情报' : '新增人工情报'));
    head.appendChild(el('div', 'intel-note',
      '人工情报优先展示；未核实状态不会因优先级自动变成事实。'));
    form.appendChild(head);

    var grid = el('div', 'intel-grid');
    form.appendChild(grid);
    grid.appendChild(field('标题 *', input('text', 'title', item.title || '',
      '例如：港口附近出现异常集结')));
    grid.appendChild(field('领域 *', selectInput(DOMAINS,
      item.domain || 'other', function () {}, 'wide', 'domain')));
    grid.appendChild(field('情报类型 *', selectInput(TYPES,
      item.intel_type || item.type || 'other', function () {}, 'wide',
      'intel_type')));
    grid.appendChild(field('时间 *', input('datetime-local', 'event_time',
      isoLocal(item.event_time), '')));
    grid.appendChild(field('纬度 *', input('number', 'lat',
      item.lat === undefined ? '' : Number(item.lat).toFixed(6), '',
      '0.000001')));
    grid.appendChild(field('经度 *', input('number', 'lon',
      item.lon === undefined ? '' : Number(item.lon).toFixed(6), '',
      '0.000001')));
    grid.appendChild(field('地点', input('text', 'place', item.place || '',
      '地名/设施/区域')));
    grid.appendChild(field('来源 *', sourceSelect(item.source_id || 'manual')));
    grid.appendChild(field('核实状态', selectInput(VERIFY,
      item.verification || 'unverified', function () {}, 'wide',
      'verification')));
    grid.appendChild(field('标签', input('text', 'tags',
      (item.tags || []).join(','), '逗号分隔')));

    var content = field('具体内容 *', document.createElement('textarea'));
    var textarea = content.querySelector('textarea');
    textarea.className = 'intel-input intel-textarea';
    textarea.name = 'content';
    textarea.rows = 4;
    textarea.value = item.content || '';
    form.appendChild(content);

    var actions = el('div', 'intel-actions');
    var save = el('button', 'primary', editing ? '保存修改' : '保存人工情报');
    save.type = 'submit';
    actions.appendChild(save);
    var cancel = el('button', 'mini', '取消');
    cancel.type = 'button';
    cancel.addEventListener('click', function () {
      state.editing = null;
      state.tab = 'list';
      startPoll();
      render();
    });
    actions.appendChild(cancel);
    form.appendChild(actions);
    body.appendChild(form);
  }

  function field(labelText, control) {
    var wrap = el('label', 'intel-field');
    wrap.appendChild(el('span', 'intel-field-label', labelText));
    wrap.appendChild(control);
    return wrap;
  }

  function input(type, name, value, placeholder, step) {
    var node = document.createElement('input');
    node.type = type;
    node.name = name;
    node.className = 'intel-input';
    node.value = value === undefined || value === null ? '' : value;
    if (placeholder) node.placeholder = placeholder;
    if (step) node.step = step;
    return node;
  }

  function sourceSelect(selected) {
    var select = document.createElement('select');
    select.name = 'source_id';
    select.className = 'intel-select';
    var seen = {};
    state.sources.forEach(function (source) {
      if (source.enabled === false) return;
      seen[source.source_id] = true;
      var option = document.createElement('option');
      option.value = source.source_id;
      option.textContent = source.name + ' · ' + source.source_id;
      select.appendChild(option);
    });
    if (selected && !seen[selected]) {
      var extra = document.createElement('option');
      extra.value = selected;
      extra.textContent = selected;
      select.appendChild(extra);
    }
    select.value = selected || 'manual';
    return select;
  }

  function saveManual(form, intelId) {
    var data = new FormData(form);
    var payload = {
      case_id: caseId(),
      title: String(data.get('title') || '').trim(),
      domain: data.get('domain'),
      intel_type: data.get('intel_type'),
      event_time: data.get('event_time') ? new Date(data.get('event_time')).toISOString() : '',
      lat: data.get('lat'),
      lon: data.get('lon'),
      place: String(data.get('place') || '').trim(),
      source_id: data.get('source_id') || 'manual',
      source_name: (state.sources.filter(function (s) {
        return s.source_id === data.get('source_id');
      })[0] || {}).name || '人工录入',
      content: String(data.get('content') || '').trim(),
      verification: data.get('verification') || 'unverified',
      tags: String(data.get('tags') || '').split(',').map(function (x) {
        return x.trim();
      }).filter(Boolean)
    };
    if (!payload.title || !payload.content) {
      global.alert('标题和具体内容不能为空');
      return;
    }
    var latText = String(data.get('lat') || '').trim();
    var lonText = String(data.get('lon') || '').trim();
    if (!latText || !lonText) {
      global.alert('经纬度不能为空');
      return;
    }
    var lat = Number(latText), lon = Number(lonText);
    if (!isFinite(lat) || lat < -90 || lat > 90 ||
        !isFinite(lon) || lon < -180 || lon > 180) {
      global.alert('经纬度不合法');
      return;
    }
    payload.lat = lat.toFixed(6);
    payload.lon = lon.toFixed(6);
    var path = intelId ? '/api/intel/manual/' + encodeURIComponent(intelId)
      : '/api/intel/manual';
    api(path, { method: 'POST', body: payload }).then(function (result) {
      state.editing = null;
      state.tab = 'list';
      state.filter = '';
      notify('info', (intelId ? '人工情报已更新：' : '人工情报已新增：') +
        payload.title, { id: result.item && result.item.intel_id });
      startPoll();
      return refresh();
    }).catch(function (err) {
      global.alert('保存失败：' + err.message);
    });
  }

  function renderSources(body) {
    var toolbar = el('div', 'intel-toolbar');
    var search = document.createElement('input');
    search.type = 'search';
    search.className = 'intel-input';
    search.placeholder = '按名称/ID/描述/标签搜索情报源';
    search.value = state.sourceFilter;
    search.addEventListener('change', function () {
      state.sourceFilter = search.value.trim();
      render();
    });
    toolbar.appendChild(search);
    toolbar.appendChild(selectInput([['all', '全部类型']].concat(SOURCE_KINDS),
      state.sourceKind, function (value) {
        state.sourceKind = value; render();
      }));
    toolbar.appendChild(selectInput([
      ['all', '全部状态'], ['true', '启用'], ['false', '停用']
    ], state.sourceEnabled, function (value) {
      state.sourceEnabled = value; render();
    }));
    body.appendChild(toolbar);

    var filtered = (state.sources || []).filter(function (source) {
      if (state.sourceKind !== 'all' && source.kind !== state.sourceKind) return false;
      if (state.sourceEnabled !== 'all' &&
          String(Boolean(source.enabled)) !== state.sourceEnabled) return false;
      var text = [
        source.source_id, source.name, source.description, source.summary,
        source.coverage, (source.features || []).join(' '),
        (source.tags || []).join(' '), (source.domains || []).join(' ')
      ].join(' ').toLowerCase();
      return !state.sourceFilter ||
        text.indexOf(state.sourceFilter.toLowerCase()) >= 0;
    });
    var count = el('div', 'intel-note',
      '情报源 ' + filtered.length + ' / ' + (state.sources || []).length);
    body.appendChild(count);
    var list = el('div', 'intel-source-list');
    filtered.forEach(function (source) {
      var row = el('div', 'intel-source-row');
      var main = el('div', 'intel-source-main');
      main.appendChild(el('b', null, source.name || source.source_id));
      main.appendChild(el('span', 'intel-meta',
        source.source_id + ' · ' +
        (source.builtin ? '内置' : '自定义') + ' · ' +
        label(SOURCE_KINDS, source.kind) +
        ((source.domains || []).length
          ? ' · ' + source.domains.map(function (d) {
            return label(DOMAINS, d);
          }).join('/') : '')));
      main.appendChild(el('span', 'intel-source-summary',
        source.summary || source.description || '未填写简介'));
      var facts = [];
      if (source.coverage) facts.push('覆盖：' + source.coverage);
      if (source.access) facts.push('接入：' + source.access);
      if (source.cadence) {
        facts.push('采集：' + source.cadence +
          (source.cadence_note ? '（' + source.cadence_note + '）' : ''));
      }
      if (source.license) facts.push('许可：' + source.license);
      if (facts.length) {
        main.appendChild(el('span', 'intel-source-facts', facts.join(' · ')));
      }
      // 可信分级：与情报卡片的置信度是两回事（源级 vs 条目级），
      // 放在源信息里，读者才知道置信度那个数字是按什么分档算出来的。
      if (source.tier) {
        main.appendChild(el('span', 'intel-source-tier', '可信分级 ' + source.tier));
      }
      if ((source.features || []).length) {
        var features = el('span', 'intel-source-features');
        source.features.forEach(function (feature) {
          features.appendChild(el('span', 'intel-chip', feature));
        });
        main.appendChild(features);
      }
      row.appendChild(main);
      var enabled = document.createElement('input');
      enabled.type = 'checkbox';
      enabled.checked = source.enabled !== false;
      enabled.title = '启用/停用';
      enabled.addEventListener('change', function () {
        api('/api/intel/sources/' + encodeURIComponent(source.source_id), {
          method: 'POST', body: { enabled: enabled.checked }
        }).then(function () {
          notify('info', '情报源已' + (enabled.checked ? '启用' : '停用') +
            '：' + source.source_id, {});
          return refresh();
        }).catch(function (err) {
          enabled.checked = !enabled.checked;
          global.alert('更新失败：' + err.message);
        });
      });
      row.appendChild(enabled);
      var del = el('button', 'mini danger', '删除');
      del.type = 'button';
      del.disabled = Boolean(source.builtin);
      del.title = source.builtin ? '内置源不能删除，可停用' : '删除自定义源';
      del.addEventListener('click', function () {
        if (!global.confirm('删除情报源 ' + source.source_id + '？')) return;
        api('/api/intel/sources/' + encodeURIComponent(source.source_id) +
          '/delete', { method: 'POST', body: {} }).then(function () {
          notify('warn', '情报源已删除：' + source.source_id, {});
          return refresh();
        }).catch(function (err) {
          global.alert('删除失败：' + err.message);
        });
      });
      row.appendChild(del);
      list.appendChild(row);
    });
    body.appendChild(list);

    var form = el('form', 'intel-form intel-source-form');
    form.addEventListener('submit', function (event) {
      event.preventDefault();
      addSource(form);
    });
    form.appendChild(el('h3', null, '添加情报源'));
    var grid = el('div', 'intel-grid');
    grid.appendChild(field('源 ID *', input('text', 'source_id', '',
      '如 local_news')));
    grid.appendChild(field('名称 *', input('text', 'name', '',
      '如 本地新闻 RSS')));
    grid.appendChild(field('类型', selectInput(SOURCE_KINDS, 'rss',
      function () {}, 'wide', 'kind')));
    grid.appendChild(field('网址', input('text', 'url', '',
      'https://...')));
    grid.appendChild(field('领域', input('text', 'domains', '',
      'news,society（逗号分隔）')));
    grid.appendChild(field('覆盖范围', input('text', 'coverage', '',
      '全球 / 全国 / 城市 / 案例区')));
    grid.appendChild(field('接入方式', input('text', 'access', '',
      '无需 Key / 免费 Key / 需申请')));
    grid.appendChild(field('标签', input('text', 'tags', '',
      '逗号分隔')));
    form.appendChild(grid);
    var summary = field('简要介绍 *', document.createElement('textarea'));
    var summaryArea = summary.querySelector('textarea');
    summaryArea.className = 'intel-input intel-textarea';
    summaryArea.name = 'summary';
    summaryArea.rows = 2;
    summaryArea.placeholder = '说明该源是什么类型、主要提供哪些信息、有什么特点';
    form.appendChild(summary);
    var features = field('特点标签', input('text', 'features', '',
      '逗号分隔，如：实时、全球、无 Key'));
    form.appendChild(features);
    var desc = field('补充描述', document.createElement('textarea'));
    var area = desc.querySelector('textarea');
    area.className = 'intel-input intel-textarea';
    area.name = 'description';
    area.rows = 2;
    form.appendChild(desc);
    var actions = el('div', 'intel-actions');
    var add = el('button', 'primary', '添加');
    add.type = 'submit';
    actions.appendChild(add);
    form.appendChild(actions);
    body.appendChild(form);
  }

  function addSource(form) {
    var data = new FormData(form);
    var payload = {
      source_id: String(data.get('source_id') || '').trim().toLowerCase(),
      name: String(data.get('name') || '').trim(),
      kind: data.get('kind') || 'other',
      url: String(data.get('url') || '').trim(),
      domains: String(data.get('domains') || '').split(',')
        .map(function (x) { return x.trim(); }).filter(Boolean),
      tags: String(data.get('tags') || '').split(',')
        .map(function (x) { return x.trim(); }).filter(Boolean),
      summary: String(data.get('summary') || '').trim(),
      features: String(data.get('features') || '').split(',')
        .map(function (x) { return x.trim(); }).filter(Boolean),
      coverage: String(data.get('coverage') || '').trim(),
      access: String(data.get('access') || '').trim(),
      description: String(data.get('description') || '').trim()
    };
    if (!payload.source_id || !payload.name || !payload.summary) {
      global.alert('源 ID、名称和简要介绍不能为空');
      return;
    }
    api('/api/intel/sources', { method: 'POST', body: payload }).then(function () {
      notify('info', '情报源已添加：' + payload.name, {});
      state.sourceFilter = '';
      return refresh();
    }).catch(function (err) {
      global.alert('添加失败：' + err.message);
    });
  }

  // ---- 目标生命周期看板（PAL-FLOW-01/02 前端） ----
  //
  // 五列就是后端 lifecycle.STAGES 的那五段；这里不自己发明状态名，
  // 列的顺序与允许的迁移都从 overview.lifecycle 里读，后端改词表时前端不会走样。
  var BDA_LABELS = [
    ['destroyed', '已摧毁'], ['damaged', '受损'],
    ['functional', '仍可用'], ['unknown', '无法判定']
  ];

  function renderBoard(body) {
    var block = (state.overview && state.overview.lifecycle) || null;
    if (!block) {
      body.appendChild(el('div', 'intel-empty', '暂无生命周期数据'));
      return;
    }
    var stages = block.stages || [];
    var transitions = block.transitions || {};
    var board = block.board || {};

    var intro = el('div', 'intel-note',
      '目标按五段式推进；只允许表内列出的迁移，跳步会被后端拒绝。' +
      '历史 append-only，每次迁移都留有操作人与备注。');
    body.appendChild(intro);

    var wrap = el('div', 'kanban');
    stages.forEach(function (stage) {
      var column = el('div', 'kanban-col');
      var head = el('div', 'kanban-head');
      head.appendChild(el('span', 'kanban-title', stage.label));
      var cards = board[stage.key] || [];
      head.appendChild(el('span', 'kanban-count', String(cards.length)));
      column.appendChild(head);

      if (!cards.length) {
        column.appendChild(el('div', 'kanban-empty', '（空）'));
      }
      cards.forEach(function (row) {
        column.appendChild(boardCard(row, stage.key, transitions));
      });
      wrap.appendChild(column);
    });
    body.appendChild(wrap);
  }

  function boardCard(row, stageKey, transitions) {
    var card = el('div', 'kanban-card');
    card.appendChild(el('div', 'kanban-id', row.target_id));
    if (row.title) card.appendChild(el('div', 'kanban-name', row.title));
    if (row.assessment) {
      var result = row.assessment.result;
      var label = BDA_LABELS.filter(function (p) { return p[0] === result; });
      card.appendChild(el('div', 'kanban-bda',
        'BDA：' + (label.length ? label[0][1] : result)));
    }
    var allowed = transitions[stageKey] || [];
    if (allowed.length) {
      var actions = el('div', 'kanban-actions');
      allowed.forEach(function (next) {
        // complete 一列只给"重新启用"这一条，评估入口在 in_execution 里
        var button = el('button', 'mini', '→ ' + stageLabelOf(next));
        button.type = 'button';
        button.title = '迁移到「' + stageLabelOf(next) + '」';
        button.addEventListener('click', function () {
          transitTarget(row.target_id, next);
        });
        actions.appendChild(button);
      });
      card.appendChild(actions);
    }
    if (stageKey === 'in_execution') {
      var bda = el('button', 'mini primary', '毁伤评估');
      bda.type = 'button';
      bda.addEventListener('click', function () { openBdaForm(row); });
      card.appendChild(bda);
    }
    return card;
  }

  function stageLabelOf(key) {
    var stages = (state.overview && state.overview.lifecycle
      && state.overview.lifecycle.stages) || [];
    for (var i = 0; i < stages.length; i++) {
      if (stages[i].key === key) return stages[i].label;
    }
    return key;
  }

  function transitTarget(targetId, toStage) {
    var actor = global.prompt('操作人（可留空）', '') || '';
    var note = global.prompt('备注（可留空）', '') || '';
    api('/api/intel/lifecycle', {
      method: 'POST',
      body: { target_id: targetId, to_stage: toStage, actor: actor, note: note }
    }).then(function () {
      notify('info', '目标 ' + targetId + ' → ' + stageLabelOf(toStage), {});
      return refresh();
    }).catch(function (err) {
      // 非法迁移是最常见的失败：后端会说出允许的去处，原样显示
      global.alert('迁移失败：' + err.message);
    });
  }

  function openBdaForm(row) {
    var body = document.getElementById('intelBody');
    if (!body) return;
    var wrap = el('div', 'bda-form');
    wrap.appendChild(el('div', 'bda-title', '毁伤评估：' + row.target_id));
    var result = selectInput(BDA_LABELS, 'destroyed', function () {});
    wrap.appendChild(result);

    var notes = document.createElement('textarea');
    notes.className = 'intel-textarea';
    notes.rows = 3;
    notes.placeholder = '毁伤程度、剩余威胁、附带损伤等';
    wrap.appendChild(notes);

    var media = document.createElement('input');
    media.type = 'text';
    media.className = 'intel-input';
    media.placeholder = '证据文件（多个用逗号分隔，可留空）';
    wrap.appendChild(media);

    var actions = el('div', 'intel-actions');
    var submit = el('button', 'mini primary', '提交评估');
    submit.type = 'button';
    submit.addEventListener('click', function () {
      var files = media.value.split(',').map(function (s) { return s.trim(); })
        .filter(Boolean);
      api('/api/intel/bda', {
        method: 'POST',
        body: {
          target_id: row.target_id, result: result.value,
          notes: notes.value.trim(), media: files
        }
      }).then(function () {
        notify('info', '已提交毁伤评估：' + row.target_id, {});
        return refresh();
      }).catch(function (err) {
        global.alert('提交失败：' + err.message);
      });
    });
    var cancel = el('button', 'mini', '取消');
    cancel.type = 'button';
    cancel.addEventListener('click', function () { render(); });
    actions.appendChild(submit);
    actions.appendChild(cancel);
    wrap.appendChild(actions);
    body.appendChild(wrap);
  }

  function bind() {
    var button = document.getElementById('btnIntel');    if (button) {
      button.addEventListener('click', function () { open('list'); });
    }
    var m = modal();
    if (!m) return;
    var closeButton = document.getElementById('intelClose');
    if (closeButton) closeButton.addEventListener('click', close);
    var dockButton = document.getElementById('intelDock');
    if (dockButton) {
      dockButton.addEventListener('click', function () {
        state.docked = !state.docked;
        try { global.localStorage.setItem('intel.docked',
          state.docked ? '1' : '0'); } catch (e) { /* 隐私模式忽略 */ }
        applyDock();
      });
    }
    try {
      state.docked = global.localStorage.getItem('intel.docked') === '1';
    } catch (e) { state.docked = false; }
    m.addEventListener('click', function (event) {
      if (event.target === m) close();
    });
    document.addEventListener('keydown', function (event) {
      if (event.key === 'Escape' && !m.classList.contains('hidden')) close();
    });
    var tabs = document.getElementById('intelTabs');
    if (tabs) {
      tabs.addEventListener('click', function (event) {
        var target = event.target.closest('[data-intel-tab]');
        if (!target) return;
        setTab(target.dataset.intelTab);
      });
    }
  }

  global.IntelPanel = {
    bind: bind, open: open, close: close, refresh: refresh,
    onCase: onCase,
    _state: function () { return state; },
    _caseId: caseId
  };
})(window);
