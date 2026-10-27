/* 资源管理：SQLite 资源库的列表、筛选、新增、编辑、固定、校验与删除。
 *
 * 写操作只允许在 test 模式；服务端在 demo 模式会返回 403。删除默认只删
 * 数据库记录，物理文件保留（由服务端 store.remove 保证）。
 *
 * 文件有两种入库方式：
 *   · 路径登记：填写服务器上已存在的绝对路径，服务端计算 sha256/size；
 *   · 网页上传：小文件直接以 application/octet-stream 上传到资源目录。
 * 大目录（envpack、瓦片集）建议用路径登记，避免几十 GB 数据走浏览器。
 */
(function (global) {
  'use strict';

  var QUALITIES = [
    ['measured', 'measured（实测）'],
    ['derived', 'derived（派生）'],
    ['default', 'default（默认）'],
    ['unknown', 'unknown（未知）']
  ];

  var state = {
    loaded: false,
    loading: false,
    error: null,
    notice: '',
    kinds: [],
    entries: [],
    stats: {},
    filterKind: '',
    q: '',
    form: null,
    busy: false
  };

  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function modal() { return document.getElementById('resourceModal'); }
  function body() { return document.getElementById('resourceBody'); }

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
          var err = new Error(data.error || ('HTTP ' + res.status));
          err.status = res.status;
          err.data = data;
          throw err;
        }
        return data;
      });
    });
  }

  function formatBytes(value) {
    var n = Number(value) || 0;
    if (n < 1024) return n + ' B';
    if (n < 1048576) return (n / 1024).toFixed(1) + ' KB';
    if (n < 1073741824) return (n / 1048576).toFixed(1) + ' MB';
    return (n / 1073741824).toFixed(2) + ' GB';
  }

  function kindLabel(kind) {
    for (var i = 0; i < state.kinds.length; i += 1) {
      if (state.kinds[i].kind === kind) {
        return state.kinds[i].description || kind;
      }
    }
    return kind;
  }

  function field(label, control) {
    var wrap = el('label', 'res-field');
    wrap.appendChild(el('span', 'res-label', label));
    wrap.appendChild(control);
    return wrap;
  }

  function textInput(name, value, placeholder) {
    var node = document.createElement('input');
    node.type = 'text';
    node.className = 'res-input';
    node.dataset.field = name;
    if (value !== undefined && value !== null) node.value = String(value);
    if (placeholder) node.placeholder = placeholder;
    return node;
  }

  function textArea(name, value, rows) {
    var node = document.createElement('textarea');
    node.className = 'res-input res-textarea';
    node.rows = rows || 2;
    node.dataset.field = name;
    node.value = value === undefined || value === null ? '' : String(value);
    return node;
  }

  function selectInput(name, options, value) {
    var node = document.createElement('select');
    node.className = 'res-input';
    node.dataset.field = name;
    options.forEach(function (item) {
      var opt = document.createElement('option');
      opt.value = item[0];
      opt.textContent = item[1];
      node.appendChild(opt);
    });
    if (value !== undefined && value !== null) node.value = String(value);
    return node;
  }

  function kindOptions() {
    var out = [];
    state.kinds.forEach(function (kind) {
      out.push([kind.kind, kind.kind + ' — ' + (kind.description || '')]);
    });
    return out;
  }

  function selectedValue(control) {
    return control ? String(control.value || '') : '';
  }

  function readForm() {
    var root = document.getElementById('resForm');
    if (!root) return null;
    var out = {};
    Array.prototype.forEach.call(root.querySelectorAll('[data-field]'), function (node) {
      out[node.dataset.field] = node.value;
    });
    var box = root.querySelector('[data-field="pinned"]');
    if (box) out.pinned = !!box.checked;
    return out;
  }

  function parseOptionalJSON(text, label) {
    var value = String(text || '').trim();
    if (!value) return {};
    try {
      var parsed = JSON.parse(value);
      if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
        throw new Error(label + '必须是 JSON 对象');
      }
      return parsed;
    } catch (err) {
      throw new Error(label + '解析失败：' + err.message);
    }
  }

  function parseBBox(text) {
    var value = String(text || '').trim();
    if (!value) return null;
    var parts = value.split(',').map(function (item) {
      return Number(item.trim());
    });
    if (parts.length !== 4 || parts.some(function (num) { return !isFinite(num); })) {
      throw new Error('范围必须是 west,south,east,north 四个数字');
    }
    return parts;
  }

  function setNotice(text, error) {
    state.notice = text || '';
    state.noticeError = !!error;
    // Do not re-render here: an error while filling the form must not wipe
    // the values the user just typed. Update the notice in place instead.
    var root = body();
    if (!root) return;
    var old = root.querySelector('.res-notice');
    if (old) old.remove();
    var node = noticeNode();
    if (node) root.insertBefore(node, root.firstChild);
  }

  // -- data --------------------------------------------------------------

  function load() {
    state.loading = true;
    state.error = null;
    render();
    return api('/api/resources').then(function (data) {
      state.kinds = data.kinds || [];
      state.entries = data.entries || [];
      state.stats = data.stats || {};
      state.loaded = true;
    }).catch(function (err) {
      state.error = String(err.message || err);
    }).then(function () {
      state.loading = false;
      render();
    });
  }

  function postJSON(path, payload) {
    return api(path, { method: 'POST', body: payload || {} });
  }

  function reloadNotice(text, error) {
    state.notice = text || '';
    state.noticeError = !!error;
    return load();
  }

  // -- rendering ---------------------------------------------------------

  function noticeNode() {
    if (!state.notice) return null;
    return el('div', 'res-notice' + (state.noticeError ? ' error' : ''),
      state.notice);
  }

  function summary() {
    var box = el('div', 'res-summary');
    var total = 0;
    var pinned = 0;
    state.entries.forEach(function (entry) {
      total += Number(entry.size) || 0;
      if (entry.pinned) pinned += 1;
    });
    box.appendChild(el('b', null, '资源库'));
    box.appendChild(el('span', 'res-note',
      'SQLite 管理元数据；删除记录不删除磁盘文件。'));
    var chips = el('div', 'res-chips');
    chips.appendChild(el('span', 'res-chip', '记录 ' + state.entries.length));
    chips.appendChild(el('span', 'res-chip', '合计 ' + formatBytes(total)));
    chips.appendChild(el('span', 'res-chip', '固定 ' + pinned));
    Object.keys(state.stats || {}).sort().forEach(function (kind) {
      var row = state.stats[kind] || {};
      chips.appendChild(el('span', 'res-chip',
        kind + ' ' + (row.count || 0)));
    });
    box.appendChild(chips);
    return box;
  }

  function toolbar() {
    var box = el('div', 'res-toolbar');
    var kind = selectInput('kindFilter',
      [['', '全部类型']].concat(kindOptions()), state.filterKind);
    kind.id = 'resKindFilter';
    kind.dataset.action = 'filter';
    box.appendChild(kind);

    var search = document.createElement('input');
    search.type = 'search';
    search.id = 'resSearch';
    search.className = 'res-input res-search';
    search.placeholder = '搜索 ID / 路径 / 来源';
    search.value = state.q;
    box.appendChild(search);

    var searchBtn = el('button', 'mini', '搜索');
    searchBtn.type = 'button';
    searchBtn.dataset.resAction = 'search';
    box.appendChild(searchBtn);

    var refresh = el('button', 'mini', '刷新');
    refresh.type = 'button';
    refresh.dataset.resAction = 'refresh';
    box.appendChild(refresh);

    var create = el('button', 'mini res-primary', '新增资源');
    create.type = 'button';
    create.dataset.resAction = 'create';
    box.appendChild(create);

    var upload = el('button', 'mini', '上传文件');
    upload.type = 'button';
    upload.dataset.resAction = 'upload';
    box.appendChild(upload);
    return box;
  }

  function filteredEntries() {
    var q = state.q.trim().toLowerCase();
    return state.entries.filter(function (entry) {
      if (state.filterKind && entry.kind !== state.filterKind) return false;
      if (!q) return true;
      return [entry.entry_id, entry.path, entry.source, entry.provider,
        entry.version, entry.license, entry.kind].some(function (value) {
        return String(value || '').toLowerCase().indexOf(q) >= 0;
      });
    });
  }

  function actionButton(label, action, entryId, extraClass) {
    var btn = el('button', 'mini res-action' + (extraClass ? ' ' + extraClass : ''),
      label);
    btn.type = 'button';
    btn.dataset.resAction = action;
    btn.dataset.entryId = entryId;
    return btn;
  }

  function entryRow(entry) {
    var tr = document.createElement('tr');
    var id = el('td', 'res-id-cell');
    id.appendChild(el('b', 'res-id', entry.entry_id));
    id.appendChild(el('div', 'res-sub', entry.path));
    tr.appendChild(id);

    var kind = el('td', 'res-kind-cell');
    kind.appendChild(el('span', 'res-kind', entry.kind));
    kind.appendChild(el('div', 'res-sub', kindLabel(entry.kind)));
    tr.appendChild(kind);

    tr.appendChild(el('td', null, entry.quality || '—'));
    tr.appendChild(el('td', null, entry.as_of || '—'));
    tr.appendChild(el('td', 'res-size', formatBytes(entry.size)));
    tr.appendChild(el('td', null, entry.pinned ? '固定' : '—'));

    var actions = el('td', 'res-actions');
    actions.appendChild(actionButton('编辑', 'edit', entry.entry_id));
    actions.appendChild(actionButton('校验', 'verify', entry.entry_id));
    actions.appendChild(actionButton(entry.pinned ? '取消固定' : '固定',
      'pin', entry.entry_id));
    actions.appendChild(actionButton('删除', 'delete', entry.entry_id,
      'res-danger'));
    tr.appendChild(actions);
    return tr;
  }

  function table() {
    var rows = filteredEntries();
    if (!rows.length) {
      return el('div', 'res-empty', state.entries.length
        ? '当前筛选条件下没有资源。'
        : '资源库为空。可新增路径登记或上传文件。');
    }
    var wrap = el('div', 'res-table-wrap');
    var tableNode = document.createElement('table');
    tableNode.className = 'res-table';
    var head = document.createElement('thead');
    var headRow = document.createElement('tr');
    ['资源 / 路径', '类型', '质量', '数据时间', '大小', '固定', '操作'].forEach(function (name) {
      headRow.appendChild(el('th', null, name));
    });
    head.appendChild(headRow);
    tableNode.appendChild(head);
    var tbody = document.createElement('tbody');
    rows.forEach(function (entry) { tbody.appendChild(entryRow(entry)); });
    tableNode.appendChild(tbody);
    wrap.appendChild(tableNode);
    return wrap;
  }

  function formHeader(title) {
    var head = el('div', 'res-form-head');
    head.appendChild(el('b', null, title));
    var back = el('button', 'mini', '返回列表');
    back.type = 'button';
    back.dataset.resAction = 'cancel';
    head.appendChild(back);
    return head;
  }

  function formActions(submitLabel) {
    var box = el('div', 'res-form-actions');
    var submit = el('button', 'mini res-primary', submitLabel);
    submit.type = 'button';
    submit.dataset.resAction = 'submit';
    box.appendChild(submit);
    var cancel = el('button', 'mini', '取消');
    cancel.type = 'button';
    cancel.dataset.resAction = 'cancel';
    box.appendChild(cancel);
    return box;
  }

  function createEditForm() {
    var edit = state.form.mode === 'edit';
    var entry = state.form.entry || {};
    var form = el('form', 'res-form');
    form.id = 'resForm';
    form.addEventListener('submit', function (event) { event.preventDefault(); });
    form.appendChild(formHeader(edit ? '编辑资源：' + entry.entry_id : '新增资源（路径登记）'));

    var grid = el('div', 'res-form-grid');
    var kind = selectInput('kind', kindOptions(),
      entry.kind || (state.kinds[0] && state.kinds[0].kind) || '');
    if (edit) kind.disabled = true;
    grid.appendChild(field('资源类型', kind));
    grid.appendChild(field('名称', textInput('name', edit ? entry.entry_id.split(':').slice(1).join(':') : '', '用于生成 entry_id，可留空用 ID')));
    grid.appendChild(field('资源 ID', textInput('entry_id', edit ? entry.entry_id : '', 'kind:name；留空自动生成')));
    grid.appendChild(field('服务器绝对路径', textInput('path', entry.path || '', '/remote-home/...')));

    grid.appendChild(field('版本', textInput('version', entry.version || '')));
    grid.appendChild(field('Schema', textInput('schema', entry.schema || '')));
    grid.appendChild(field('Provider', textInput('provider', entry.provider || '')));
    grid.appendChild(field('Source', textInput('source', entry.source || '')));
    grid.appendChild(field('数据时间 as_of', textInput('as_of', entry.as_of || '', '2026-10-27')));
    grid.appendChild(field('抓取时间 fetched_at', textInput('fetched_at', entry.fetched_at || '', '留空自动填')));
    grid.appendChild(field('范围 bbox', textInput('bbox', entry.bbox ? entry.bbox.join(',') : '',
      'west,south,east,north')));
    grid.appendChild(field('许可', textInput('license', entry.license || '')));
    grid.appendChild(field('质量', selectInput('quality', QUALITIES, entry.quality || 'measured')));

    var notes = textArea('notes', JSON.stringify(entry.notes || {}, null, 1), 3);
    grid.appendChild(field('备注 notes（JSON）', notes));

    if (edit) {
      var pin = document.createElement('input');
      pin.type = 'checkbox';
      pin.dataset.field = 'pinned';
      pin.checked = !!entry.pinned;
      var pinWrap = el('label', 'res-check');
      pinWrap.appendChild(pin);
      pinWrap.appendChild(el('span', null, '固定（固定后删除需强制确认）'));
      grid.appendChild(pinWrap);
    } else {
      var replace = document.createElement('input');
      replace.type = 'checkbox';
      replace.dataset.field = 'replace';
      var replaceWrap = el('label', 'res-check');
      replaceWrap.appendChild(replace);
      replaceWrap.appendChild(el('span', null, '同 ID 已存在时覆盖（replace）'));
      grid.appendChild(replaceWrap);
    }

    form.appendChild(grid);
    form.appendChild(el('div', 'res-hint',
      edit ? '修改路径会重新计算 sha256 与大小。' :
        '路径登记只引用现有文件/目录，不复制；服务端会计算 sha256 与大小。'));
    form.appendChild(formActions(edit ? '保存修改' : '登记资源'));
    return form;
  }

  function uploadForm() {
    var form = el('form', 'res-form');
    form.id = 'resForm';
    form.addEventListener('submit', function (event) { event.preventDefault(); });
    form.appendChild(formHeader('上传文件'));
    var grid = el('div', 'res-form-grid');
    grid.appendChild(field('资源类型',
      selectInput('kind', kindOptions(),
        (state.kinds[0] && state.kinds[0].kind) || '')));
    grid.appendChild(field('名称', textInput('name', '', '用于生成 entry_id')));
    grid.appendChild(field('资源 ID', textInput('entry_id', '', '可留空自动生成')));
    var file = document.createElement('input');
    file.type = 'file';
    file.className = 'res-input';
    file.dataset.field = 'file';
    grid.appendChild(field('文件', file));
    grid.appendChild(field('许可', textInput('license', '')));
    grid.appendChild(field('质量', selectInput('quality', QUALITIES, 'measured')));
    grid.appendChild(field('数据时间 as_of', textInput('as_of', '')));
    form.appendChild(grid);
    form.appendChild(el('div', 'res-hint',
      '上传文件写入资源目录的 uploads/ 下；大目录资源请改用“新增资源”登记服务器路径。'));
    form.appendChild(formActions('上传并登记'));
    return form;
  }

  function render() {
    var root = body();
    if (!root) return;
    root.textContent = '';

    if (state.loading && !state.loaded) {
      root.appendChild(el('div', 'res-empty', '正在读取资源库…'));
      return;
    }
    if (state.error) {
      var err = el('div', 'res-notice error', '资源库读取失败：' + state.error);
      root.appendChild(err);
      var retry = el('button', 'mini', '重试');
      retry.type = 'button';
      retry.dataset.resAction = 'refresh';
      root.appendChild(retry);
      return;
    }
    if (noticeNode()) root.appendChild(noticeNode());

    if (state.form) {
      root.appendChild(state.form.mode === 'upload' ? uploadForm() : createEditForm());
      return;
    }

    root.appendChild(toolbar());
    root.appendChild(summary());
    root.appendChild(table());
  }

  // -- operations --------------------------------------------------------

  function startCreate() {
    state.form = { mode: 'create' };
    state.notice = '';
    render();
  }

  function startEdit(entry) {
    state.form = { mode: 'edit', entry: entry };
    state.notice = '';
    render();
  }

  function startUpload() {
    state.form = { mode: 'upload' };
    state.notice = '';
    render();
  }

  function submitCreate() {
    var values = readForm();
    if (!values) return;
    var payload = {
      kind: selectedValue(document.querySelector('#resForm [data-field="kind"]')),
      name: values.name || '',
      path: values.path || '',
      version: values.version || '',
      schema: values.schema || '',
      provider: values.provider || '',
      source: values.source || '',
      as_of: values.as_of || '',
      fetched_at: values.fetched_at || '',
      license: values.license || '',
      quality: values.quality || 'measured',
      replace: !!document.querySelector('#resForm [data-field="replace"]:checked')
    };
    if (values.entry_id) payload.entry_id = values.entry_id;
    try {
      var bbox = parseBBox(values.bbox);
      if (bbox) payload.bbox = bbox;
      payload.notes = parseOptionalJSON(values.notes, 'notes');
    } catch (err) {
      setNotice(err.message, true);
      return;
    }
    state.busy = true;
    postJSON('/api/resources', payload).then(function () {
      state.form = null;
      state.busy = false;
      reloadNotice('资源已登记。');
    }).catch(function (err) {
      state.busy = false;
      setNotice('登记失败：' + err.message, true);
    });
  }

  function submitEdit() {
    var entry = state.form && state.form.entry;
    if (!entry) return;
    var values = readForm();
    if (!values) return;
    var payload = {
      path: values.path || '',
      version: values.version || '',
      schema: values.schema || '',
      provider: values.provider || '',
      source: values.source || '',
      as_of: values.as_of || '',
      fetched_at: values.fetched_at || '',
      license: values.license || '',
      quality: values.quality || 'measured',
      pinned: !!document.querySelector('#resForm [data-field="pinned"]:checked')
    };
    try {
      var bbox = parseBBox(values.bbox);
      payload.bbox = bbox;
      payload.notes = parseOptionalJSON(values.notes, 'notes');
    } catch (err) {
      setNotice(err.message, true);
      return;
    }
    state.busy = true;
    postJSON('/api/resources/' + encodeURIComponent(entry.entry_id) + '/update',
      payload).then(function () {
      state.form = null;
      state.busy = false;
      reloadNotice('资源已更新。');
    }).catch(function (err) {
      state.busy = false;
      setNotice('更新失败：' + err.message, true);
    });
  }

  function submitUpload() {
    var values = readForm();
    if (!values) return;
    var input = document.querySelector('#resForm [data-field="file"]');
    if (!input || !input.files || !input.files[0]) {
      setNotice('请选择要上传的文件。', true);
      return;
    }
    var params = new URLSearchParams();
    params.set('kind', selectedValue(document.querySelector('#resForm [data-field="kind"]')));
    params.set('name', values.name || '');
    params.set('filename', input.files[0].name);
    if (values.entry_id) params.set('entry_id', values.entry_id);
    if (values.license) params.set('license', values.license);
    if (values.quality) params.set('quality', values.quality);
    if (values.as_of) params.set('as_of', values.as_of);
    state.busy = true;
    fetch('/api/resources/upload?' + params.toString(), {
      method: 'POST',
      headers: { 'Content-Type': 'application/octet-stream' },
      body: input.files[0]
    }).then(function (res) {
      return res.json().catch(function () { return {}; }).then(function (data) {
        if (!res.ok) throw new Error(data.error || ('HTTP ' + res.status));
        return data;
      });
    }).then(function () {
      state.form = null;
      state.busy = false;
      reloadNotice('文件已上传并登记。');
    }).catch(function (err) {
      state.busy = false;
      setNotice('上传失败：' + err.message, true);
    });
  }

  function doPin(entry) {
    postJSON('/api/resources/' + encodeURIComponent(entry.entry_id) + '/pin',
      { pinned: !entry.pinned }).then(function () {
      reloadNotice(entry.pinned ? '已取消固定。' : '已固定。');
    }).catch(function (err) {
      setNotice('固定失败：' + err.message, true);
    });
  }

  function doVerify(entry) {
    postJSON('/api/resources/' + encodeURIComponent(entry.entry_id) + '/verify',
      {}).then(function (data) {
      var result = data.verify || {};
      reloadNotice('校验' + (data.ok ? '通过' : '失败') +
        '：' + (result.status || 'unknown'));
    }).catch(function (err) {
      setNotice('校验失败：' + err.message, true);
    });
  }

  function doDelete(entry) {
    var force = false;
    if (!global.confirm('删除资源记录「' + entry.entry_id +
        '」？\n默认只删除数据库记录，磁盘文件保留。')) {
      return;
    }
    function send(forceFlag) {
      return postJSON('/api/resources/' + encodeURIComponent(entry.entry_id) +
        '/delete', { force: forceFlag });
    }
    send(force).catch(function (err) {
      if (err.status === 409 && global.confirm(
          '资源已固定，是否强制删除记录？（磁盘文件仍保留）')) {
        return send(true);
      }
      throw err;
    }).then(function () {
      reloadNotice('资源记录已删除。');
    }).catch(function (err) {
      setNotice('删除失败：' + err.message, true);
    });
  }

  function submitForm() {
    if (!state.form || state.busy) return;
    if (state.form.mode === 'create') submitCreate();
    else if (state.form.mode === 'edit') submitEdit();
    else if (state.form.mode === 'upload') submitUpload();
  }

  function onAction(action, entryId) {
    var entry = state.entries.filter(function (item) {
      return item.entry_id === entryId;
    })[0];
    if (action === 'refresh') return load();
    if (action === 'create') return startCreate();
    if (action === 'upload') return startUpload();
    if (action === 'cancel') {
      state.form = null;
      state.notice = '';
      return render();
    }
    if (action === 'submit') return submitForm();
    if (!entry) return;
    if (action === 'edit') return startEdit(entry);
    if (action === 'pin') return doPin(entry);
    if (action === 'verify') return doVerify(entry);
    if (action === 'delete') return doDelete(entry);
  }

  function bind() {
    var button = document.getElementById('btnResourceManager');
    if (button) button.addEventListener('click', open);
    var closeButton = document.getElementById('resourceClose');
    if (closeButton) closeButton.addEventListener('click', close);
    var m = modal();
    if (m) {
      m.addEventListener('click', function (event) {
        if (event.target === m) close();
      });
    }
    var root = body();
    if (root) {
      root.addEventListener('click', function (event) {
        var node = event.target.closest('[data-res-action]');
        if (!node) return;
        var action = node.dataset.resAction;
        if (action === 'filter') return;
        if (action === 'search') {
          var input = document.getElementById('resSearch');
          state.q = input ? input.value : '';
          return render();
        }
        onAction(action, node.dataset.entryId || '');
      });
      root.addEventListener('change', function (event) {
        if (event.target && event.target.id === 'resKindFilter') {
          state.filterKind = event.target.value || '';
          render();
        }
      });
      root.addEventListener('keydown', function (event) {
        if (event.key === 'Enter' && event.target
            && event.target.id === 'resSearch') {
          event.preventDefault();
          state.q = event.target.value || '';
          render();
        }
      });
    }
    document.addEventListener('keydown', function (event) {
      if (event.key === 'Escape' && m && !m.classList.contains('hidden')) close();
    });
  }

  function open() {
    var m = modal();
    if (!m) return;
    m.classList.remove('hidden');
    if (!state.loaded && !state.loading) load();
    else render();
  }

  function close() {
    var m = modal();
    if (m) m.classList.add('hidden');
  }

  global.ResourceManager = {
    bind: bind,
    open: open,
    close: close,
    refresh: load,
    _state: function () { return state; }
  };
})(window);
