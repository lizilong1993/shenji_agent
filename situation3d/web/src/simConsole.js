/* TW2 test console (test mode only). Launches engine runs, compares metrics,
 * and manages the three retention slots: latest / best / pinned. */
(function () {
  'use strict';
  var state = { caseId: null, runs: [], timer: null, busy: false };
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

  function render() {
    var rows = state.runs || [];
    if (!rows.length) {
      el.runs.innerHTML = '<tr><td colspan="10" class="hint">还没有运行结果。设置 seed 与时长后发起推演。</td></tr>';
      return;
    }
    el.runs.innerHTML = rows.map(function (run) {
      var m = metrics(run);
      var done = run.status === 'completed';
      var best = (run.roles || []).indexOf('best') >= 0;
      var pinned = (run.roles || []).indexOf('pinned') >= 0;
      var actions = '<div class="actions">';
      actions += '<button class="ghost" data-act="view" data-id="' + esc(run.run_id) + '"' + (done ? '' : ' disabled') + '>查看</button>';
      actions += '<button class="ghost" data-act="best" data-id="' + esc(run.run_id) + '"' + (done && !best ? '' : ' disabled') + '>设为最优</button>';
      if (!pinned) actions += '<button class="ghost" data-act="pin" data-id="' + esc(run.run_id) + '"' + (done ? '' : ' disabled') + '>指定保留</button>';
      else actions += '<button class="ghost" data-act="unpin" data-id="' + esc(run.run_id) + '">取消保留</button>';
      if (run.status === 'queued' || run.status === 'running') {
        actions += '<button class="danger" data-act="cancel" data-id="' + esc(run.run_id) + '">取消</button>';
      } else {
        actions += '<button class="danger" data-act="delete" data-id="' + esc(run.run_id) + '"' + (best || pinned ? ' disabled' : '') + '>删除</button>';
      }
      actions += '</div>';
      return '<tr>' +
        '<td><code>' + esc(run.run_id) + '</code><div class="hint">' + esc(run.created_at || '') + '</div></td>' +
        '<td>' + esc(statusLabel(run)) + '</td>' +
        '<td>' + esc(run.seed == null ? '-' : run.seed) + '</td>' +
        '<td>' + esc(m.events) + '</td><td>' + esc(m.fires) + '</td><td>' + esc(m.destroyed) + '</td>' +
        '<td>' + esc(m.losses) + '</td><td>' + m.validation + '</td>' +
        '<td>' + rolesHtml(run) + '</td><td>' + actions + '</td></tr>';
    }).join('');
    var active = rows.some(function (r) { return r.status === 'queued' || r.status === 'running'; });
    if (active) schedule(3000);
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

  function launch() {
    if (state.busy) return;
    var seed = parseInt(el.seed.value, 10);
    var ticks = parseInt(el.ticks.value, 10);
    if (!isFinite(seed)) { setStatus('seed 必须是整数', true); return; }
    if (!isFinite(ticks) || ticks < 10 || ticks > 2000) { setStatus('时长需在 10–2000 tick', true); return; }
    state.busy = true;
    el.launch.disabled = true;
    setStatus('提交推演…');
    api('/api/sim/runs', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ case_id: state.caseId, seed: seed, ticks: ticks })
    }).then(function (body) {
      setStatus('已排队：' + body.run_id);
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

  function init() {
    el = {
      case: $('case'), seed: $('seed'), ticks: $('ticks'), launch: $('launch'),
      status: $('status'), refresh: $('refresh'), runs: $('runs')
    };
    el.launch.addEventListener('click', launch);
    el.refresh.addEventListener('click', refresh);
    el.runs.addEventListener('click', onAction);
    el.case.addEventListener('change', function () {
      state.caseId = el.case.value;
      setStatus('');
      refresh();
    });
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
