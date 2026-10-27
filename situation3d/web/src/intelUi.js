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
    filter: '', priority: 'all', effective: 'all',
    sourceFilter: '', sourceKind: 'all', sourceEnabled: 'all',
    editing: null, timer: null, loading: false, collecting: false,
    collectStarted: 0, autoTried: {}, lastCase: null, caseRetries: 0
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
    state.caseRetries = 0;
    var m = modal();
    if (!m) return;
    m.classList.remove('hidden');
    refresh();
    onCase(caseId(), situationData());
    startPoll();
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
      render();
    }).catch(function (err) {
      state.error = err.message || String(err);
      renderError();
    }).finally(function () {
      state.loading = false;
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
    else renderList(body);
  }

  function renderList(body) {
    var overview = state.overview;
    var items = (overview.items || []).filter(function (item) {
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

    var summary = el('div', 'intel-summary');
    var counts = overview.counts || {};
    summary.appendChild(el('b', null,
      '案例 ' + (overview.case_id || 'default') + ' · 共 ' +
      (overview.items || []).length + ' 条'));
    summary.appendChild(el('span', 'intel-note',
      '人工 ' + (counts.manual_active || 0) +
      ' · 自动 ' + (counts.auto || 0) +
      ' · 冲突覆盖 ' + (counts.superseded || 0) +
      ' · 关联 ' + (counts.links || 0)));
    body.appendChild(summary);
    (overview.warnings || []).forEach(function (warning) {
      body.appendChild(el('div', 'intel-warn', warning));
    });
    var pipeline = overview.pipeline || null;
    if (pipeline && pipeline.stage1 && pipeline.stage2) {
      var s1 = pipeline.stage1, s2 = pipeline.stage2, s3 = pipeline.stage3 || {};
      body.appendChild(el('div', 'intel-pipeline',
        '三阶段：广泛采集 ' + (s1.observations || 0) +
        ' → 清洗保留 ' + (s2.kept || 0) +
        '（无关键词保留 ' + (s2.kept_without_keyword || 0) + '）' +
        ' → 关联 ' + ((s3.counts && s3.counts.links) || 0) +
        ' / 关键 ' + ((s3.counts && s3.counts.key_items) || 0) +
        (state.collecting ? ' · 正在补充采集…' : '')));
    } else if (state.collecting) {
      body.appendChild(el('div', 'intel-pipeline',
        '正在按案例地理范围采集多源情报…'));
    }

    if ((overview.key_items || []).length) {
      var key = el('div', 'intel-key');
      key.appendChild(el('div', 'intel-key-title', '关键情报（自动抽取）'));
      var byId = {};
      (overview.items || []).forEach(function (item) { byId[item.id] = item; });
      overview.key_items.slice(0, 5).forEach(function (row) {
        var item = byId[row.id];
        if (!item) return;
        var line = el('div', 'intel-key-row');
        line.appendChild(el('span', 'intel-key-score', row.score.toFixed(2)));
        line.appendChild(el('span', 'intel-key-text',
          (item.priority === 'manual' ? '[人工] ' : '') + item.title));
        line.appendChild(el('span', 'intel-key-reason',
          (row.reasons || []).join('；')));
        key.appendChild(line);
      });
      body.appendChild(key);
    }

    var toolbar = el('div', 'intel-toolbar');
    var search = document.createElement('input');
    search.type = 'search';
    search.className = 'intel-input';
    search.placeholder = '搜索标题/内容/来源/类型';
    search.value = state.filter;
    search.addEventListener('change', function () {
      state.filter = search.value.trim();
      render();
    });
    toolbar.appendChild(search);
    toolbar.appendChild(selectInput([
      ['all', '全部来源'], ['manual', '仅人工'], ['auto', '仅自动']
    ], state.priority, function (value) {
      state.priority = value; render();
    }));
    toolbar.appendChild(selectInput([
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
    toolbar.appendChild(add);
    body.appendChild(toolbar);

    if (!items.length) {
      body.appendChild(el('div', 'intel-empty', '没有符合条件的情报'));
      return;
    }
    items.forEach(function (item) { body.appendChild(card(item, overview)); });
  }

  function card(item, overview) {
    var node = el('div', 'intel-card ' + item.priority +
      (item.effective ? '' : ' superseded') +
      (item.status === 'revoked' ? ' revoked' : ''));
    var head = el('div', 'intel-card-head');
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
    meta.appendChild(el('span', null,
      label(DOMAINS, item.domain) + ' / ' + label(TYPES, item.type)));
    meta.appendChild(el('span', null, timeLabel(item.event_time)));
    meta.appendChild(el('span', null,
      Number(item.lat).toFixed(6) + ', ' + Number(item.lon).toFixed(6) +
      (item.place ? ' · ' + item.place : '')));
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
    }

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
      node.appendChild(actions);
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
      if (facts.length) {
        main.appendChild(el('span', 'intel-source-facts', facts.join(' · ')));
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

  function bind() {
    var button = document.getElementById('btnIntel');
    if (button) {
      button.addEventListener('click', function () { open('list'); });
    }
    var m = modal();
    if (!m) return;
    var closeButton = document.getElementById('intelClose');
    if (closeButton) closeButton.addEventListener('click', close);
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
