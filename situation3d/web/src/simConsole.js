/* TW2 test console (test mode only). Launches engine runs, compares metrics,
 * and manages the three retention slots: latest / best / pinned.
 * Also: scenario date, launch-time tasks, resource preflight / resolve. */
(function () {
  'use strict';
  var state = { caseId: null, runs: [], timer: null, busy: false,
                resolveRun: null, dateTouched: false };
  var el = {};

  function $(id) { return document.getElementById(id); }
  function esc(v) {
    return String(v == null ? '' : v).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }
  function setStatus(text, bad) {
    el.status.textContent = text || '';
    el.status.className = 'status' + (bad ? ' bad' : '');
  }
  function setResolverStatus(text, bad) {
    el.resolverStatus.textContent = text || '';
    el.resolverStatus.className = 'status' + (bad ? ' bad' : '');
  }

  function api(path, options) {
    return fetch(path, options || {}).then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (body) {
        if (!r.ok) {
          var msg = body && body.error ? body.error : ('HTTP ' + r.status);
          var err = new Error(msg);
          err.detail = body;
          err.status = r.status;
          throw err;
        }
        return body;
      });
    });
  }

  function statusLabel(run) {
    var s = String(run.status || '');
    if (s === 'queued') return '排队';
    if (s === 'waiting_resources') return '等待资源';
    if (s === 'running') {
      var p = run.progress || {};
      if (p.tick != null && p.ticks) return '推演中 ' + p.tick + '/' + p.ticks;
      return '推演中';
    }
    if (s === 'completed') return '完成';
    if (s === 'failed') return '失败';
    if (s === 'cancelled') return '取消';
    return s || '-';
  }

  function rolesHtml(run) {
    var roles = run.roles || [];
    var out = '';
    if (roles.indexOf('latest') >= 0) out += '<span class="pill latest">本次</span>';
    if (roles.indexOf('best') >= 0) out += '<span class="pill best">最优</span>';
    if (roles.indexOf('pinned') >= 0) out += '<span class="pill pinned">指定保留</span>';
    return out || '<span class="hint">—</span>';
  }

  function metrics(run) {
    var m = run.metrics || {};
    var v = run.validation || {};
    var losses = m.losses || {};
    var passes = v.passed === true;
    return {
      events: m.events == null ? '-' : m.events,
      fires: m.fires == null ? '-' : m.fires,
      destroyed: m.destroyed == null ? '-' : m.destroyed,
      losses: (losses.red == null ? '-' : losses.red) + ' / ' + (losses.blue == null ? '-' : losses.blue),
      validation: v.passed == null ? '-' : (passes ? '<span class="ok">通过</span>' : '<span class="bad">未通过</span>')
    };
  }

  function resourceHtml(run) {
    var res = run.resources || {};
    var fetchErr = res.fetch_error ? '<div class="hint bad">' + esc(res.fetch_error) + '</div>' : '';
    if (run.status === 'waiting_resources') {
      var gaps = run.resource_gaps || [];
      var kinds = gaps.map(function (g) { return g.kind; });
      return '<span class="bad">等待资源</span>' +
        (kinds.length ? '<div class="hint">' + esc(kinds.join('、')) + '</div>' : '') +
        fetchErr;
    }
    if (!res.status) return '<span class="hint">—</span>';
    var cls = res.status === 'ready' ? 'ok' : (res.status === 'degraded' ? 'warn' : 'bad');
    var warn = (res.warnings || []).length;
    var derived = (res.entries || []).filter(function (e) {
      return e.quality && e.quality !== 'measured';
    }).length;
    return '<span class="' + cls + '">' + esc(res.status) + '</span>' +
      (warn ? '<div class="hint">⚠ ' + warn + ' 条提示</div>' : '') +
      (derived ? '<div class="hint warn">⚠ ' + derived + ' 项降级数据</div>' : '') +
      fetchErr;
  }

  function render() {
    var rows = state.runs || [];
    if (!rows.length) {
      el.runs.innerHTML = '<tr><td colspan="11" class="hint">还没有运行结果。设置想定日期、任务与 seed 后发起推演。</td></tr>';
      return;
    }
    el.runs.innerHTML = rows.map(function (run) {
      var m = metrics(run);
      var done = run.status === 'completed';
      var waiting = run.status === 'waiting_resources';
      var best = (run.roles || []).indexOf('best') >= 0;
      var pinned = (run.roles || []).indexOf('pinned') >= 0;
      var actions = '<div class="actions">';
      if (waiting) {
        actions += '<button class="ghost" data-act="resolve" data-id="' + esc(run.run_id) + '">处理资源</button>';
        actions += '<button class="danger" data-act="cancel" data-id="' + esc(run.run_id) + '">取消</button>';
      } else {
        actions += '<button class="ghost" data-act="view" data-id="' + esc(run.run_id) + '"' + (done ? '' : ' disabled') + '>查看</button>';
        actions += '<button class="ghost" data-act="best" data-id="' + esc(run.run_id) + '"' + (done && !best ? '' : ' disabled') + '>设为最优</button>';
        if (!pinned) actions += '<button class="ghost" data-act="pin" data-id="' + esc(run.run_id) + '"' + (done ? '' : ' disabled') + '>指定保留</button>';
        else actions += '<button class="ghost" data-act="unpin" data-id="' + esc(run.run_id) + '">取消保留</button>';
        if (run.status === 'queued' || run.status === 'running') {
          actions += '<button class="danger" data-act="cancel" data-id="' + esc(run.run_id) + '">取消</button>';
        } else {
          actions += '<button class="danger" data-act="delete" data-id="' + esc(run.run_id) + '"' + (best || pinned ? ' disabled' : '') + '>删除</button>';
        }
      }
      actions += '</div>';
      return '<tr>' +
        '<td><code>' + esc(run.run_id) + '</code><div class="hint">' + esc(run.created_at || '') + '</div></td>' +
        '<td>' + esc(statusLabel(run)) + '</td>' +
        '<td>' + esc(run.seed == null ? '-' : run.seed) + (run.scenario_date ? '<div class="hint">' + esc(String(run.scenario_date).slice(0, 10)) + '</div>' : '') + '</td>' +
        '<td>' + resourceHtml(run) + '</td>' +
        '<td>' + esc(m.events) + '</td><td>' + esc(m.fires) + '</td><td>' + esc(m.destroyed) + '</td>' +
        '<td>' + esc(m.losses) + '</td><td>' + m.validation + '</td>' +
        '<td>' + rolesHtml(run) + '</td><td>' + actions + '</td></tr>';
    }).join('');
    var active = rows.some(function (r) { return r.status === 'queued' || r.status === 'running'; });
    var waiting = rows.some(function (r) { return r.status === 'waiting_resources'; });
    if (active) schedule(3000);
    else if (waiting) schedule(10000);
  }

  function schedule(ms) {
    if (state.timer) clearTimeout(state.timer);
    state.timer = setTimeout(refresh, ms);
  }

  function refresh() {
    if (!state.caseId) return Promise.resolve();
    return api('/api/sim/runs?case_id=' + encodeURIComponent(state.caseId))
      .then(function (body) {
        state.runs = body.runs || [];
        render();
      })
      .catch(function (err) { setStatus('读取运行列表失败：' + err.message, true); });
  }

  function collectTasks() {
    var rows = Array.prototype.slice.call(
      el.taskList.querySelectorAll('.task-row'));
    return rows.map(function (row, index) {
      function value(k) {
        var node = row.querySelector('[data-k="' + k + '"]');
        return node ? node.value.trim() : '';
      }
      var units = value('units').split(/[,，\s]+/).filter(Boolean).map(Number);
      if (!units.length || units.some(function (u) { return !isFinite(u); })) {
        throw new Error('任务 ' + (index + 1) + '：单位 ID 无效');
      }
      var objective = value('objective');
      var lonText = value('lon');
      var latText = value('lat');
      var hasLon = lonText !== '' && isFinite(parseFloat(lonText));
      var hasLat = latText !== '' && isFinite(parseFloat(latText));
      if (hasLon !== hasLat) throw new Error('任务 ' + (index + 1) + '：经度/纬度需同时填写');
      if (!objective && !hasLon) throw new Error('任务 ' + (index + 1) + '：需要目标名或经纬度');
      var item = {
        id: 'U' + (index + 1),
        units: units,
        task: value('task') || 'move_to',
        start_s: parseFloat(value('start_s') || '0') || 0,
        pace: value('pace') || 'normal'
      };
      if (objective) item.objective = objective;
      if (hasLon) { item.lon = parseFloat(lonText); item.lat = parseFloat(latText); }
      return item;
    });
  }

  function addTaskRow() {
    var empty = el.taskList.querySelector('#task-empty');
    if (empty) empty.remove();
    var row = document.createElement('div');
    row.className = 'task-row';
    row.innerHTML =
      '<input class="units" data-k="units" placeholder="单位ID，逗号分隔">' +
      '<select data-k="task">' +
      '<option value="move_to">move_to</option>' +
      '<option value="seize">seize</option>' +
      '<option value="hold">hold</option></select>' +
      '<input class="obj" data-k="objective" placeholder="目标名（可选）">' +
      '<input class="coord" data-k="lon" type="number" step="0.000001" placeholder="经度">' +
      '<input class="coord" data-k="lat" type="number" step="0.000001" placeholder="纬度">' +
      '<input class="when" data-k="start_s" type="number" value="0" step="30" title="开始秒">' +
      '<select data-k="pace"><option value="normal">normal</option>' +
      '<option value="dash">dash</option><option value="cautious">cautious</option></select>' +
      '<button class="ghost" type="button" data-remove>移除</button>';
    row.querySelector('[data-remove]').addEventListener('click', function () {
      row.remove();
      if (!el.taskList.querySelector('.task-row')) {
        el.taskList.innerHTML = '<div class="hint" id="task-empty">尚未添加任务。</div>';
      }
    });
    el.taskList.appendChild(row);
  }

  function localNowInput() {
    var d = new Date();
    d.setSeconds(0, 0);
    var local = new Date(d.getTime() - d.getTimezoneOffset() * 60000);
    return local.toISOString().slice(0, 16);
  }

  function launch() {
    if (state.busy) return;
    var seed = parseInt(el.seed.value, 10);
    var ticks = parseInt(el.ticks.value, 10);
    if (!isFinite(seed)) { setStatus('seed 必须是整数', true); return; }
    if (!isFinite(ticks) || ticks < 10 || ticks > 2000) { setStatus('时长需在 10–2000 tick', true); return; }
    var tasks;
    try { tasks = collectTasks(); }
    catch (err) { setStatus(err.message, true); return; }
    var body = {
      case_id: state.caseId, seed: seed, ticks: ticks,
      resource_mode: el.resourceMode.value,
      side: parseInt(el.side.value, 10)
    };
    if (el.date.value) {
      var when = new Date(el.date.value);
      if (!isFinite(when.getTime())) { setStatus('想定日期无效', true); return; }
      body.scenario_date = when.toISOString();
    }
    if (tasks.length) {
      body.tasks = tasks;
      body.task_mode = el.taskMode.value;
    }
    state.busy = true;
    el.launch.disabled = true;
    setStatus('提交推演…');
    api('/api/sim/runs', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    }).then(function (res) {
      setStatus('已排队：' + res.run_id);
      return refresh();
    }).catch(function (err) {
      setStatus('发起失败：' + err.message, true);
    }).then(function () {
      state.busy = false;
      el.launch.disabled = false;
    });
  }

  function postRun(runId, action, body) {
    return api('/api/sim/runs/' + encodeURIComponent(runId) + '/' + action, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(Object.assign({ case_id: state.caseId }, body || {}))
    });
  }

  function openResolver(run) {
    state.resolveRun = run;
    el.resolver.style.display = 'block';
    el.resolverRun.textContent = run.run_id + (run.scenario_date ? ' · ' + run.scenario_date : '');
    var gaps = run.resource_gaps || [];
    if (!gaps.length) {
      el.resolverInfo.innerHTML = '<div class="hint">该运行没有资源缺口记录；可刷新列表或重新发起。</div>';
      return;
    }
    el.resolverInfo.innerHTML = gaps.map(function (gap) {
      var rec = gap.recommendation || {};
      var level = gap.criticality === 'critical' ? '<span class="bad">[必须处理]</span>' : '<span class="warn">[提示]</span>';
      var html = '<div class="gap"><strong>' + esc(gap.kind) + '</strong> ' + level + '<br>' +
        esc(gap.impact || '') +
        '<div class="hint">原因：' + esc(gap.reason || '') + '</div>';
      if (rec.path) html += '<div class="hint">路径：<code>' + esc(rec.path) + '</code></div>';
      if (rec.command) html += '<div class="hint">命令：<code>' + esc(rec.command) + '</code></div>';
      if (rec.entry_id) {
        html += '<button class="ghost" data-stale="' + esc(rec.entry_id) + '">接受该缓存数据</button>';
      }
      return html + '</div>';
    }).join('');
  }

  function resolve(decisions) {
    if (!state.resolveRun) return;
    var runId = state.resolveRun.run_id;
    var buttons = el.resolver.querySelectorAll('button');
    for (var i = 0; i < buttons.length; i++) buttons[i].disabled = true;
    setResolverStatus('提交处理…');
    postRun(runId, 'resolve', { decisions: decisions })
      .then(function () {
        setResolverStatus('已重新排队');
        el.resolver.style.display = 'none';
        state.resolveRun = null;
        return refresh();
      })
      .catch(function (err) {
        setResolverStatus('处理失败：' + err.message, true);
        for (var j = 0; j < buttons.length; j++) buttons[j].disabled = false;
      });
  }

  function onAction(evt) {
    var btn = evt.target.closest('button[data-act]');
    if (!btn) return;
    var runId = btn.getAttribute('data-id');
    var act = btn.getAttribute('data-act');
    if (act === 'view') {
      window.open('/?case=' + encodeURIComponent(state.caseId) +
                  '&run=' + encodeURIComponent(runId), '_blank');
      return;
    }
    if (act === 'resolve') {
      var run = (state.runs || []).filter(function (r) { return r.run_id === runId; })[0];
      if (run) openResolver(run);
      return;
    }
    if (act === 'best') {
      if (!confirm('将 ' + runId + ' 设为“最优”？它是唯一允许发布到演示端的运行。')) return;
      var note = prompt('选择备注（可留空）', '');
      if (note === null) return;
      postRun(runId, 'select', { confirm: true, note: note })
        .then(function () { setStatus('已设为最优'); return refresh(); })
        .catch(function (err) { setStatus('操作失败：' + err.message, true); });
      return;
    }
    if (act === 'pin') {
      postRun(runId, 'pin', {})
        .then(function () { setStatus('已指定保留'); return refresh(); })
        .catch(function (err) {
          var pinned = err.detail && err.detail.pinned;
          if (err.status === 409 && pinned) {
            if (!confirm('已有指定保留：' + pinned + '\n替换为 ' + runId + '？')) return;
            postRun(runId, 'pin', { replace: pinned })
              .then(function () { setStatus('已替换指定保留'); return refresh(); })
              .catch(function (e2) { setStatus('操作失败：' + e2.message, true); });
            return;
          }
          setStatus('操作失败：' + err.message, true);
        });
      return;
    }
    if (act === 'unpin') {
      postRun(runId, 'unpin', {})
        .then(function () { setStatus('已取消指定保留'); return refresh(); })
        .catch(function (err) { setStatus('操作失败：' + err.message, true); });
      return;
    }
    if (act === 'cancel') {
      if (!confirm('取消运行 ' + runId + '？')) return;
      postRun(runId, 'cancel', {})
        .then(function () { setStatus('已取消'); return refresh(); })
        .catch(function (err) { setStatus('取消失败：' + err.message, true); });
      return;
    }
    if (act === 'delete') {
      if (!confirm('删除运行 ' + runId + '？该操作不可恢复。')) return;
      postRun(runId, 'delete', { confirm: true })
        .then(function () { setStatus('已删除'); return refresh(); })
        .catch(function (err) { setStatus('删除失败：' + err.message, true); });
    }
  }

  function initResolver() {
    el.resolverInfo.addEventListener('click', function (evt) {
      var btn = evt.target.closest('button[data-stale]');
      if (!btn) return;
      resolve([{ action: 'use_cached_stale',
                 entry_id: btn.getAttribute('data-stale'),
                 note: 'console accept' }]);
    });
    el.resolverFetch.addEventListener('click', function () {
      resolve([{ action: 'fetch_online', note: 'console fetch' }]);
    });
    el.resolverProvide.addEventListener('click', function () {
      var path = prompt('服务器上 envpack 目录的绝对路径：', '');
      if (!path) return;
      resolve([{ action: 'provide_file', kind: 'envpack', path: path,
                 note: 'console provide' }]);
    });
    el.resolverCancel.addEventListener('click', function () {
      if (!confirm('取消该运行？')) return;
      resolve([{ action: 'cancel', note: 'console cancel' }]);
    });
    el.resolverClose.addEventListener('click', function () {
      el.resolver.style.display = 'none';
      state.resolveRun = null;
    });
  }

  function init() {
    el = {
      case: $('case'), seed: $('seed'), ticks: $('ticks'), launch: $('launch'),
      status: $('status'), refresh: $('refresh'), runs: $('runs'),
      date: $('scenario-date'), resourceMode: $('resource-mode'),
      side: $('side'), taskMode: $('task-mode'), taskList: $('task-list'),
      addTask: $('add-task'),
      resolver: $('resolver'), resolverRun: $('resolver-run'),
      resolverInfo: $('resolver-info'), resolverStatus: $('resolver-status'),
      resolverFetch: $('resolver-fetch'), resolverProvide: $('resolver-provide'),
      resolverCancel: $('resolver-cancel'), resolverClose: $('resolver-close')
    };
    el.date.value = localNowInput();
    el.date.addEventListener('change', function () { state.dateTouched = true; });
    document.addEventListener('visibilitychange', function () {
      if (!document.hidden && !state.dateTouched) {
        el.date.value = localNowInput();
      }
    });
    el.launch.addEventListener('click', launch);
    el.refresh.addEventListener('click', refresh);
    el.runs.addEventListener('click', onAction);
    el.addTask.addEventListener('click', addTaskRow);
    el.case.addEventListener('change', function () {
      state.caseId = el.case.value;
      setStatus('');
      refresh();
    });
    initResolver();
    api('/api/sim/cases').then(function (body) {
      var cases = body.cases || [];
      el.case.innerHTML = cases.map(function (c) {
        return '<option value="' + esc(c.id) + '">' + esc(c.id) + '</option>';
      }).join('');
      state.caseId = el.case.value || null;
      if (!state.caseId) { setStatus('测试端没有可推演案例', true); return; }
      setStatus('就绪');
      refresh();
    }).catch(function (err) { setStatus('无法读取案例：' + err.message, true); });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
}());
