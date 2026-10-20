/* situation3d — 战役演示界面（campaign.v3）
 *
 * 只对 `schema: situation3d.campaign.v3` 的案例生效；旧 v2 案例完全不改变行为。
 *
 * 三块内容：
 *   1. 底部时间轴上方的**阶段色带**：每阶段一段，点击跳到阶段起点；
 *   2. 右栏顶部「战役态势」：阶段列表、目标点控制、战损、结局卡；
 *   3. 事件流：按阶段分组、可按级别筛选，点击定位到 tick / 目标 / 地点。
 *
 * 依赖：window.SituationHost（app.js 暴露 state/selectTarget/flyTo/setTick）。
 * 不修改 app.js 的渲染逻辑；app.js 只在案例装载与 tick 变化时调用
 * `CampaignUi.onCase(data)` / `CampaignUi.onTick(tick)`。
 */
(function (global) {
  'use strict';

  var data = null;          // campaign.v3 数据集
  var root = null;          // 右栏面板
  var ribbon = null;        // 阶段色带
  var lastStage = -1;
  var eventFilter = 'all';

  function host() { return global.SituationHost; }
  function fmtTime(tickSec) {
    if (!data || !data.start_time_utc) return '';
    var base = new Date(data.start_time_utc);
    var t = new Date(base.getTime() + tickSec * 1000);
    return t.toISOString().slice(11, 19) + 'Z';
  }
  function esc(s) {
    return String(s === undefined || s === null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }
  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  }
  function stageAt(tick) {
    var stages = data.stages || [];
    for (var i = 0; i < stages.length; i++) {
      if (tick < stages[i].end_tick) return i;
    }
    return stages.length ? stages.length - 1 : -1;
  }
  function frameAt(tick) {
    var frames = data.frames || [];
    if (!frames.length) return null;
    var i = Math.min(frames.length - 1, Math.max(0, tick - 1));
    return frames[i];
  }
  function ownerOf(frame, coord) {
    var o = (frame && frame.objectives) || {};
    return o[String(coord)] || 'neutral';
  }

  // ------------------------------------------------------------------
  // 右栏：战役态势
  // ------------------------------------------------------------------
  function ensurePanel() {
    if (root && root.isConnected) return root;
    var col = document.querySelector('.panel.panel-right');
    if (!col) return null;
    root = el('section', 'panel-block campaign-panel');
    root.innerHTML =
      '<h2>战役态势</h2>' +
      '<div class="cp-stages" id="cpStages"></div>' +
      '<div class="cp-objectives" id="cpObjectives"></div>' +
      '<div class="cp-events" id="cpEvents"></div>' +
      '<div class="cp-outcome" id="cpOutcome"></div>';
    col.insertBefore(root, col.firstChild);
    return root;
  }

  function renderStages(tick) {
    var box = root && root.querySelector('#cpStages');
    if (!box) return;
    var cur = stageAt(tick);
    if (cur === lastStage) return;
    lastStage = cur;
    box.innerHTML = '';
    (data.stages || []).forEach(function (st, i) {
      var row = el('div', 'cp-stage' + (i === cur ? ' active' : '') +
        (i < cur ? ' done' : ''));
      row.innerHTML = '<span class="cp-idx">' + (i + 1) + '</span>' +
        '<span class="cp-name">' + esc(st.name) + '</span>' +
        '<span class="cp-ticks">' + fmtTime(st.start_tick * data.tick_seconds) +
        '–' + fmtTime(st.end_tick * data.tick_seconds) + '</span>';
      row.title = (st.summary || '') + '\n点击跳到阶段起点';
      row.addEventListener('click', function () { goToTick(st.start_tick + 1); });
      box.appendChild(row);
    });
  }

  function renderObjectives(tick) {
    var box = root && root.querySelector('#cpObjectives');
    if (!box) return;
    var frame = frameAt(tick);
    var objs = (data.stages && data.stages[0] && data.stages[0].state &&
      data.stages[0].state.objectives) || [];
    var color = { red: '#d0342c', blue: '#1f5fbf', neutral: '#8a94a6' };
    var html = '<div class="cp-sub">目标点</div>';
    objs.forEach(function (o) {
      var owner = ownerOf(frame, o.coord);
      html += '<div class="cp-obj"><i style="background:' +
        (color[owner] || color.neutral) + '"></i>' +
        '<span>' + esc(o.name || o.coord) + '</span>' +
        '<b>' + ({ red: '红方', blue: '蓝方', neutral: '未控' }[owner] || '') + '</b></div>';
    });
    box.innerHTML = html;
  }

  function renderEvents(tick) {
    var box = root && root.querySelector('#cpEvents');
    if (!box) return;
    var events = data.events || [];
    var curStage = stageAt(tick);
    var counts = { all: events.length };
    events.forEach(function (e) {
      counts[e.level || 'info'] = (counts[e.level || 'info'] || 0) + 1;
    });
    var list = events.filter(function (e) {
      if (eventFilter === 'all') return true;
      return (e.level || 'info') === eventFilter;
    });
    var html = '<div class="cp-sub">事件 <span class="cp-count">' + list.length +
      '</span></div>' +
      '<div class="cp-filter">' +
      ['all', 'warn', 'info'].map(function (k) {
        var label = k === 'all' ? '全部' : k === 'warn' ? '告警' : '信息';
        return '<button type="button" data-f="' + k + '"' +
          (eventFilter === k ? ' class="on"' : '') + '>' + label +
          (counts[k] ? ' ' + counts[k] : '') + '</button>';
      }).join('') + '</div>' +
      '<div class="cp-event-list">';
    list.slice(-120).forEach(function (e) {
      var src = e.source === 'script' ? '脚本' : '引擎';
      var active = e.tick <= tick && stageAt(e.tick) === curStage;
      html += '<div class="cp-event lv-' + (e.level || 'info') +
        (active ? ' active' : '') + '" data-ev="' + esc(e.id) + '">' +
        '<span class="cp-t">' + fmtTime(e.tick * data.tick_seconds) + '</span>' +
        '<span class="cp-src">' + src + '</span>' +
        '<span class="cp-d">' + esc(e.description) + '</span></div>';
    });
    html += '</div>';
    box.innerHTML = html;
    box.querySelectorAll('.cp-filter button').forEach(function (b) {
      b.addEventListener('click', function () {
        eventFilter = b.getAttribute('data-f');
        renderEvents(host().state.tick);
      });
    });
    box.querySelectorAll('.cp-event').forEach(function (n) {
      n.addEventListener('click', function () { focusEvent(n.getAttribute('data-ev')); });
    });
  }

  function renderOutcome(tick) {
    var box = root && root.querySelector('#cpOutcome');
    if (!box) return;
    var out = data.outcome;
    var frames = data.frames || [];
    var atEnd = frames.length && tick >= frames.length - 1;
    if (!out || !atEnd) {
      box.innerHTML = '';
      return;
    }
    var eng = out.engine || {};
    var scr = out.script || {};
    var zh = { red: '红方', blue: '蓝方', draw: '平局', undecided: '未决' };
    box.innerHTML =
      '<div class="cp-sub">结局</div>' +
      '<div class="cp-out">引擎裁决：<b>' + (zh[eng.winner] || eng.winner || '—') +
      '</b>（红 ' + ((eng.score || {}).red || 0) + ' / 蓝 ' +
      ((eng.score || {}).blue || 0) + '）</div>' +
      '<div class="cp-out">演示脚本：<b>' + (zh[scr.winner] || scr.winner || '—') +
      '</b>' + (out.match ? ' · 与引擎一致' : ' · 与引擎不一致') + '</div>' +
      '<div class="cp-note">' + esc(out.disclaimer || '') + '</div>';
  }

  // ------------------------------------------------------------------
  // 底部：阶段色带
  // ------------------------------------------------------------------
  function ensureRibbon() {
    if (ribbon && ribbon.isConnected) return ribbon;
    var transport = document.querySelector('footer.transport');
    if (!transport) return null;
    ribbon = el('div', 'cp-ribbon');
    transport.insertBefore(ribbon, transport.firstChild);
    return ribbon;
  }

  function renderRibbon(tick) {
    var box = ensureRibbon();
    if (!box) return;
    var stages = data.stages || [];
    var total = (data.frames || []).length || 1;
    var cur = stageAt(tick);
    box.innerHTML = '';
    stages.forEach(function (st, i) {
      var seg = el('button', 'cp-seg' + (i === cur ? ' active' : '') +
        (i < cur ? ' done' : ''));
      seg.type = 'button';
      seg.style.flex = String(Math.max(1, st.end_tick - st.start_tick));
      seg.title = st.name + '（' + fmtTime(st.start_tick * data.tick_seconds) +
        '–' + fmtTime(st.end_tick * data.tick_seconds) + '）';
      seg.textContent = st.name;
      seg.addEventListener('click', function () { goToTick(st.start_tick + 1); });
      box.appendChild(seg);
    });
    var clock = el('div', 'cp-clock', fmtTime(tick * data.tick_seconds));
    box.appendChild(clock);
  }

  // ------------------------------------------------------------------
  // 交互
  // ------------------------------------------------------------------
  function goToTick(tick) {
    var h = host();
    if (!h || !h.setTick) return;
    h.setTick(tick);
  }

  function focusEvent(id) {
    var e = null;
    (data.events || []).some(function (x) { if (x.id === id) { e = x; return true; } return false; });
    if (!e) return;
    goToTick(e.tick);
    var geo = e.geo || null;
    var h = host();
    if (h && h.flyTo && geo) {
      h.flyTo(geo[0], geo[1], 90000, -55);
    }
    if (e.actors && e.actors.length && h && h.selectTarget) {
      // Frame/target ids are strings in campaign.v3.
      var first = String(e.actors[0]);
      // 只有当前案例里存在的目标才可选
      var exists = (data.targets || []).some(function (t) { return String(t.id) === first; });
      if (exists) h.selectTarget(first, false);
    }
  }

  // ------------------------------------------------------------------
  // 对外接口
  // ------------------------------------------------------------------
  function log(msg) {
    (window.__campaignLog = window.__campaignLog || []).push(msg);
  }

  function onCase(caseData) {
    log('onCase ' + (caseData && caseData.schema || 'null'));
    if (!caseData) {
      data = null;
      removeUi();
      return;
    }
    if (caseData.schema !== 'situation3d.campaign.v3') {
      // A non-campaign case loaded after a campaign one: drop the UI, but do
      // not clear `data` for a null call in the middle of a case switch.
      data = null;
      removeUi();
      return;
    }
    data = caseData;
    lastStage = -1;
    onTick(host().state.tick || 0);
  }

  function removeUi() {
    var old = document.querySelector('.campaign-panel');
    if (old && old.parentNode) old.parentNode.removeChild(old);
    root = null;
    if (ribbon && ribbon.parentNode) ribbon.parentNode.removeChild(ribbon);
    ribbon = null;
    lastStage = -1;
  }

  function onTick(tick) {
    if (!data) return;
    log('onTick ' + tick);
    ensurePanel();
    ensureRibbon();
    renderStages(tick);
    renderObjectives(tick);
    renderEvents(tick);
    renderOutcome(tick);
    renderRibbon(tick);
  }

  function debug() {
    if (!data) return { campaign: false };
    var tick = host().state.tick;
    return {
      campaign: true,
      schema: data.schema,
      caseId: data.case_id,
      tick: tick,
      stage: stageAt(tick),
      stages: (data.stages || []).length,
      events: (data.events || []).length,
      outcome: data.outcome ? data.outcome.engine : null,
      ribbon: !!ribbon,
      panel: !!root,
    };
  }

  global.CampaignUi = {
    onCase: onCase,
    onTick: onTick,
    debug: debug,
    goToTick: goToTick,
    focusEvent: focusEvent,
  };
})(window);
