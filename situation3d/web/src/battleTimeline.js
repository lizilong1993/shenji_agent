/* 战斗时间轴（仅测试端引擎案例启用）。
 *
 * 旧播放条以"帧号 / 总帧数"表示进度：一份预烘焙轨迹一次装载，滑块天然可以拖到
 * 结尾。这里改成面向战斗进程的时间轴：
 *
 *   · 左端 = 案例载入时刻（战斗开始时间），右端 = 推演已经推进到的时刻；
 *   · 右端随引擎实时推演生长，拖动/播放/步进只能落在已发生区间；
 *   · 条上标关键节点（首次任务/交火/毁伤/夺控、每次毁伤与占领变化），
 *     悬停显示事件摘要，点击回到该时刻；
 *   · 回退后按播放，从选定时刻继续回放已发生的战斗过程。
 *
 * 演示端（mode=demo）、非引擎案例、?run= 成品回放都不进入这个模式，
 * 播放条保持原样——这是 8848 演示分支"逻辑例外"的落点。
 */
(function () {
  'use strict';

  var HEALTH = null;
  var ENGINE = null;            // {case_id: true}
  var readyPromise = null;
  var live = null;              // 见 onCaseLoaded()
  var els = {};
  var tipEl = null;
  var markerEls = [];

  function $(id) { return document.getElementById(id); }
  function pad2(n) { return n < 10 ? '0' + n : '' + n; }
  function hmsOf(ms) {
    var d = new Date(ms);
    return pad2(d.getUTCHours()) + ':' + pad2(d.getUTCMinutes()) + ':' +
      pad2(d.getUTCSeconds());
  }
  function isoOf(ms) {
    var d = new Date(ms);
    return d.toISOString().replace('T', ' ').slice(0, 19) + 'Z';
  }
  function elapsedOf(ms) {
    var s = Math.max(0, Math.floor(ms / 1000));
    return 'T+' + pad2(Math.floor(s / 3600)) + ':' +
      pad2(Math.floor((s % 3600) / 60)) + ':' + pad2(s % 60);
  }
  function esc(v) {
    return String(v == null ? '' : v).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }
  function bridge() { return window.TransportBridge; }
  function activeStatus(s) {
    return s === 'queued' || s === 'running' || s === 'starting' ||
      s === 'waiting_resources';
  }

  // ------------------------------------------------------------------------
  // 启动时拿服务模式与可推演案例：/api/health 的 mode 决定是否启用，
  // /api/sim/cases 决定哪些案例能开战（演示端 404，自动降级为空）。
  // ------------------------------------------------------------------------
  function ready() {
    if (!readyPromise) {
      readyPromise = Promise.all([
        fetch('/api/health', { cache: 'no-store' })
          .then(function (r) { return r.json(); })
          .catch(function () { return {}; }),
        fetch('/api/sim/cases', { cache: 'no-store' })
          .then(function (r) { return r.ok ? r.json() : { cases: [] }; })
          .catch(function () { return { cases: [] }; })
      ]).then(function (res) {
        HEALTH = res[0] || {};
        ENGINE = {};
        ((res[1] || {}).cases || []).forEach(function (c) {
          if (c && c.id) ENGINE[c.id] = true;
        });
      });
    }
    return readyPromise;
  }

  // ------------------------------------------------------------------------
  // 案例装载：测试端引擎案例进入"待开战"状态
  // ------------------------------------------------------------------------
  function onCaseLoaded() {
    var B = bridge();
    if (!B) return;
    stopPolling();
    live = null;
    clearMarkers();
    if (els.wrap) {
      els.wrap.classList.remove('tl-live');
      els.wrap.classList.add('tl-off');
    }
    if (els.start) els.start.textContent = '';
    if (els.now) els.now.textContent = '';

    var data = B.state.data;
    if (!data) return;
    var caseId = data.case_id || B.state.caseId;
    var runParam = null;
    try {
      runParam = new URLSearchParams(window.location.search).get('run');
    } catch (e) { runParam = null; }
    if (!HEALTH || HEALTH.mode !== 'test' || !ENGINE || !ENGINE[caseId] ||
        runParam) {
      return;                       // 演示端 / 非引擎案例 / 成品回放：保持旧界面
    }

    // 载入时刻就是战斗开始时刻：向下取整到秒，引擎与时间轴用同一个锚点。
    var startMs = Math.floor(Date.now() / 1000) * 1000;
    live = {
      caseId: caseId,
      startMs: startMs,
      tickSeconds: Number(data.tick_seconds) || 30,
      tickCount: Number(data.tick_count) || 1440,
      status: 'idle',
      runId: null,
      timer: null,
      events: [],
      nodes: [],
      markers: [],
      markerSig: '',
      received: false,
      following: true
    };

    // 旧案例自带的 12 小时轨迹不属于这场战斗：只保留初始帧，后续帧全部
    // 来自新推演的 live/ 增量接口。terminal_tick 置空，让 timelineMaxTick
    // 以"已收到的最后一帧"为上界（新帧到达即右端生长）。
    if (data.frames && data.frames.length > 1) data.frames.length = 1;
    data.events = [];
    data.nodes = [];
    data.terminal_tick = null;
    data.start_time_utc = new Date(startMs).toISOString();
    var nav = document.getElementById('nodeNav');
    if (nav) nav.style.display = 'none';

    if (els.wrap) {
      els.wrap.classList.add('tl-live');
      els.wrap.classList.remove('tl-off');
    }
    els.timeline.min = '0';
    els.timeline.max = '0';
    els.timeline.value = '0';
    els.timeline.disabled = true;
    els.timeline.style.setProperty('--tl-progress', '0%');
    render();
    setButton('idle');
    B.setStatus('已载入「' + (data.case_name || '') +
      '」：点击「TS自动推演」开始战斗', 'ok');
  }

  // ------------------------------------------------------------------------
  // 开战 / 轮询
  // ------------------------------------------------------------------------
  function launch() {
    if (!live) return;
    if (activeStatus(live.status)) return;
    var B = bridge();
    live.status = 'starting';
    setButton('starting');
    B.setStatus('正在启动引擎推演…');
    fetch('/api/sim/runs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        case_id: live.caseId,
        ticks: live.tickCount,
        // 推演可能晚于载入几分钟才开始，但战斗开始时刻仍是"载入时刻"。
        scenario_date: new Date(live.startMs).toISOString(),
        resource_mode: 'auto',
        side: 0
      })
    }).then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (body) {
        if (!r.ok) {
          throw new Error(body && body.error ? body.error : ('HTTP ' + r.status));
        }
        return body;
      });
    }).then(function (body) {
      if (!live) return;
      live.runId = body.run_id;
      live.status = body.status || 'queued';
      setButton('running');
      B.setStatus('推演已排队：' + live.runId);
      startPolling();
    }).catch(function (err) {
      if (!live) return;
      live.status = 'failed';
      setButton('failed');
      B.setStatus('启动推演失败：' + err.message, badStatus());
    });
  }

  function startPolling() {
    stopPolling();
    live.received = false;
    live.events = [];
    live.nodes = [];
    poll();
  }

  function stopPolling() {
    if (live && live.timer) {
      clearTimeout(live.timer);
      live.timer = null;
    }
  }

  function schedule(ms) {
    if (!live) return;
    stopPolling();
    live.timer = setTimeout(poll, ms);
  }

  function poll() {
    if (!live || !live.runId) return;
    var B = bridge();
    var frames = (B.state.data && B.state.data.frames) || [];
    var since = live.received ? frames.length : 0;
    var url = '/api/sim/runs/' + encodeURIComponent(live.runId) + '/live' +
      '?case_id=' + encodeURIComponent(live.caseId) +
      '&frames=' + since + '&events=' + live.events.length;
    fetch(url, { cache: 'no-store' }).then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (body) {
        if (!r.ok) {
          throw new Error(body && body.error ? body.error : ('HTTP ' + r.status));
        }
        return body;
      });
    }).then(function (body) {
      adopt(body);
    }).catch(function (err) {
      if (!live) return;
      B.setStatus('读取推演进度失败：' + err.message, badStatus());
      if (activeStatus(live.status)) schedule(3000);
    });
  }

  function adopt(d) {
    if (!live || d.run_id !== live.runId) return;
    var B = bridge();
    var s = B.state;
    var data = s.data || {};
    if (d.start_time_utc) {
      var ms = Date.parse(d.start_time_utc);
      if (isFinite(ms)) live.startMs = ms;
    }
    if (d.tick_seconds) live.tickSeconds = Number(d.tick_seconds);
    if (d.tick_count) live.tickCount = Number(d.tick_count);

    var frames = d.frames || [];
    var oldFrontier = (data.frames || []).length - 1;
    if (!live.received) {
      if (frames.length) {
        // 第一包用运行库的初始帧整体替换（同一批目标 id，场景已按案例建好）。
        data.frames = frames.slice();
        live.received = true;
        data.start_time_utc = new Date(live.startMs).toISOString();
        s.tick = 0;
        s.tickFloat = 0;
        B.setTick(0);
      }
    } else if (frames.length) {
      if (live.received && typeof d.frame_count === 'number' &&
          d.frame_count < (data.frames || []).length) {
        // 运行器重启后文件被截短：丢掉旧帧，下一轮从 0 重新取。
        live.received = false;
        live.events = [];
        live.nodes = [];
        data.frames = (data.frames || []).slice(0, 1);
        schedule(0);
        return;
      }
      for (var i = 0; i < frames.length; i++) data.frames.push(frames[i]);
    }
    if (d.events && d.events.length) {
      live.events = live.events.concat(d.events);
    }
    if (d.nodes && d.nodes.length) live.nodes = d.nodes;
    // 运行被取消/杀掉时进程来不及写 live/state.json，运行库的终态更可信。
    live.status = (d.status === 'failed' || d.status === 'cancelled' ||
      d.status === 'completed')
      ? d.status
      : (d.live_status || d.status || live.status);
    if (live.status === 'completed') {
      data.terminal_tick = live.tickCount;
    }
    buildMarkers();
    render();

    var frontier = Math.max(0, ((data.frames || []).length - 1));
    if (s.tick > frontier) B.setTick(frontier);
    if (live.following && !s.playing && frontier > oldFrontier &&
        s.tick >= oldFrontier) {
      // 用户本来就站在"当前时刻"上：随推演推进把视点带到新的右端。
      // （步进按钮的监听先于 app.js 执行，s.tick 的实时比较比 following
      //   标志更可靠，避免"刚点后退却被新帧拽回前沿"。）
      B.setTick(frontier);
      render();
    }

    if (live.status === 'completed') {
      stopPolling();
      setButton('done');
      B.setStatus('推演结束：已发生 ' + elapsedOf(frontier * live.tickSeconds * 1000) +
        ' 的战斗可回放', 'ok');
    } else if (live.status === 'failed' || live.status === 'cancelled') {
      stopPolling();
      setButton('failed');
      B.setStatus('推演' + (live.status === 'failed' ? '失败' : '已取消') +
        (d.error ? '：' + d.error : ''), badStatus());
    } else if (live.status === 'waiting_resources') {
      setButton('waiting');
      B.setStatus('推演等待资源：请在推演控制台处理资源缺口', 'warn');
      schedule(5000);
    } else {
      setButton('running');
      if (live.status === 'queued') B.setStatus('推演排队中…');
      else B.setStatus('引擎推演中… 已发生 ' +
        elapsedOf(frontier * live.tickSeconds * 1000));
      schedule(1000);
    }
  }

  function badStatus() { return 'bad'; }

  // ------------------------------------------------------------------------
  // 时间轴渲染
  // ------------------------------------------------------------------------
  function render() {
    if (!live) return;
    var B = bridge();
    var s = B.state;
    var data = s.data || {};
    var frames = data.frames || [];
    var frontier = Math.max(0, frames.length - 1);
    var tick = Math.max(0, Math.min(s.tick, frontier));
    var totalMs = frontier * live.tickSeconds * 1000;

    els.timeline.min = '0';
    els.timeline.max = String(frontier);
    els.timeline.value = String(tick);
    els.timeline.disabled = frontier <= 0;
    var pct = frontier > 0 ? (tick / frontier) * 100 : 0;
    els.timeline.style.setProperty('--tl-progress', pct.toFixed(2) + '%');

    els.start.textContent = '战斗开始 ' + hmsOf(live.startMs);
    els.now.textContent = frontier > 0
      ? '当前 ' + hmsOf(live.startMs + totalMs) : '未开始';
    els.tick.textContent = elapsedOf(tick * live.tickSeconds * 1000);
    els.clock.textContent = isoOf(live.startMs + tick * live.tickSeconds * 1000);
    setLiveNote(totalMs);
    renderMarkers(totalMs);
  }

  /**
   * 右栏「区域TS」的说明行：引擎推演没有预烘焙推理帧，这里改成推演状态，
   * 否则每次拖动时间轴都会被 app.js 的 updateInferenceUi() 覆盖成
   * “未加载推理结果”。放在 render 末尾，保证本轮最后写入。
   */
  function setLiveNote(totalMs) {
    var note = document.getElementById('rlNote');
    if (!note) return;
    if (live.status === 'completed') {
      note.textContent = '引擎推演结束：可回放已发生的 ' +
        elapsedOf(totalMs) + ' 战斗过程';
    } else if (activeStatus(live.status)) {
      note.textContent = '引擎实时推演中：已发生 ' + elapsedOf(totalMs) +
        '，只能查看该时刻之前的过程';
    } else if (live.status === 'failed' || live.status === 'cancelled') {
      note.textContent = '推演未完成：仅能回放已发生部分';
    } else {
      note.textContent = '引擎实时推演：点击「TS自动推演」开始';
    }
  }

  function buildMarkers() {
    if (!live) return;
    var out = [];
    var seen = {};
    function add(m, key) {
      key = key || (String(m.tick) + '|' + m.kind);
      if (seen[key]) return;
      seen[key] = true;
      out.push(m);
    }
    (live.nodes || []).forEach(function (n) {
      var kind = n.type === 'first_destroy' ? 'destroy'
        : (n.type === 'first_objective' ? 'objective'
          : (n.type === 'terminal' ? 'terminal' : 'node'));
      add({
        tick: Number(n.tick) || 0,
        kind: kind,
        title: n.title || '关键节点',
        desc: n.description || '',
        actors: n.actors || []
      });
    });
    (live.events || []).forEach(function (e) {
      var kind = null;
      var title = '';
      if (e.type === 'destroy') { kind = 'destroy'; title = '目标损毁'; }
      else if (e.type === 'objective_progress') { kind = 'objective'; title = '占领变化'; }
      if (!kind) return;
      var tick = Number(e.tick) || 0;
      // 同一 tick 已有精选节点（首次毁伤/首次夺控）时不再重复标事件；
      // 其余同 tick 的多个毁伤各自保留（用事件 id 作键）。
      if (seen[tick + '|' + kind]) return;
      add({
        tick: tick,
        kind: kind,
        title: title,
        desc: e.description || '',
        actors: e.actors || []
      }, tick + '|' + kind + '|' + (e.id || e.target_id || ''));
    });
    out.sort(function (a, b) { return a.tick - b.tick; });
    live.markers = out;
    // 每秒轮询都会走到这里，但多数时候节点集合没变：只有集合变化时才重建
    // DOM，否则悬停中的节点会被替换掉（鼠标一停就闪）。
    var sig = out.map(function (m) {
      return m.tick + '|' + m.kind + '|' + m.title + '|' + m.desc;
    }).join(';');
    if (sig !== live.markerSig) {
      live.markerSig = sig;
      rebuildMarkerEls();
    }
  }

  function rebuildMarkerEls() {
    clearMarkers();
    if (!live) return;
    live.markers.forEach(function (m, i) {
      var btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'tl-marker ' + m.kind;
      btn.dataset.i = String(i);
      btn.setAttribute('aria-label', m.title);
      btn.addEventListener('mouseenter', showTip);
      btn.addEventListener('mouseleave', hideTip);
      btn.addEventListener('focus', showTip);
      btn.addEventListener('blur', hideTip);
      btn.addEventListener('click', function (ev) {
        ev.preventDefault();
        ev.stopPropagation();
        seekToMarker(i);
      });
      els.markers.appendChild(btn);
      markerEls.push(btn);
    });
  }

  function clearMarkers() {
    if (els.markers) els.markers.innerHTML = '';
    markerEls = [];
  }

  function renderMarkers(totalMs) {
    if (!markerEls.length) return;
    var width = els.markers.clientWidth || 0;
    var lastX = -100;
    markerEls.forEach(function (btn, i) {
      var m = live.markers[i];
      if (!m || totalMs <= 0) { btn.style.display = 'none'; return; }
      var atMs = m.tick * live.tickSeconds * 1000;
      if (atMs > totalMs + 1) { btn.style.display = 'none'; return; }
      var pct = Math.max(0, Math.min(100, (atMs / totalMs) * 100));
      // 按像素而不是百分比判重：时间轴很长时 1% 可能是十几分钟，
      // 按百分比会把"同一时段的多个毁伤"整片吞掉。8px 内只留一个圆点，
      // 其余信息仍在事件数据里（悬停卡片只展示最靠左的那个）。
      var x = width ? (pct / 100) * width : pct;
      if (x - lastX < 8) { btn.style.display = 'none'; return; }
      lastX = x;
      btn.style.display = '';
      btn.style.left = pct.toFixed(2) + '%';
    });
  }

  function markerText(i) {
    var B = bridge();
    var m = live.markers[i];
    var data = (B.state && B.state.data) || {};
    var names = {};
    (data.targets || []).forEach(function (t) { names[t.id] = t.name || t.id; });
    var atMs = live.startMs + m.tick * live.tickSeconds * 1000;
    var actors = (m.actors || []).map(function (id) {
      return names[id] ? names[id] + '（' + id + '）' : id;
    }).slice(0, 4).join('、');
    var html = '<div class="tl-tip-title">' + esc(m.title) + '</div>' +
      '<div class="tl-tip-meta">' + esc(isoOf(atMs)) + ' · ' +
      esc(elapsedOf(m.tick * live.tickSeconds * 1000)) + '</div>';
    if (m.desc) html += '<div class="tl-tip-desc">' + esc(m.desc) + '</div>';
    if (actors) html += '<div class="tl-tip-actors">' + esc(actors) + '</div>';
    return html;
  }

  function showTip(ev) {
    if (!live || !tipEl) return;
    var i = Number(ev.currentTarget.dataset.i);
    if (!live.markers[i]) return;
    tipEl.innerHTML = markerText(i);
    tipEl.style.display = 'block';
    tipEl.style.left = '0px';
    tipEl.style.top = '0px';
    var rect = ev.currentTarget.getBoundingClientRect();
    var w = tipEl.offsetWidth;
    var h = tipEl.offsetHeight;
    var left = rect.left + rect.width / 2 - w / 2;
    left = Math.max(8, Math.min(window.innerWidth - w - 8, left));
    var top = rect.top - h - 8;
    if (top < 8) top = rect.bottom + 8;
    tipEl.style.left = left + 'px';
    tipEl.style.top = top + 'px';
  }

  function hideTip() {
    if (tipEl) tipEl.style.display = 'none';
  }

  function seekToMarker(i) {
    var B = bridge();
    var m = live && live.markers[i];
    if (!m) return;
    B.setPlaying(false);
    B.setTick(m.tick);
    B.renderTargetCounts();
    B.updateInferenceUi();
    live.following = m.tick >= B.timelineMaxTick();
    render();
  }

  // ------------------------------------------------------------------------
  // 控件状态与用户交互
  // ------------------------------------------------------------------------
  function setButton(state) {
    var b = els.btnRun;
    if (!b) return;
    if (state === 'starting') {
      b.textContent = '启动中…'; b.disabled = true;
    } else if (state === 'running') {
      b.textContent = '推演中…'; b.disabled = true;
    } else if (state === 'waiting') {
      b.textContent = '等待资源…'; b.disabled = true;
    } else if (state === 'done') {
      b.textContent = '▶ 重新推演'; b.disabled = false;
      b.title = '以本次载入时刻为战斗开始，再启动一次引擎推演';
    } else if (state === 'failed') {
      b.textContent = '▶ TS自动推演'; b.disabled = false;
    } else {
      b.textContent = '▶ TS自动推演'; b.disabled = false;
      b.title = '以案例载入时刻为战斗开始，启动引擎推演';
    }
  }

  function onUserSeek() {
    if (!live) return;
    var v = Number(els.timeline.value);
    var max = Number(els.timeline.max);
    live.following = max > 0 && v >= max;
    hideTip();
  }

  function init() {
    els.timeline = $('timeline');
    els.markers = $('tlMarkers');
    els.start = $('tlStartLabel');
    els.now = $('tlNowLabel');
    els.tick = $('tickLabel');
    els.clock = $('clockLabel');
    els.btnRun = $('btnRun');
    els.wrap = els.timeline ? els.timeline.closest('.tr-timeline') : null;
    if (!els.timeline || !els.markers) return;

    tipEl = document.createElement('div');
    tipEl.className = 'tl-tip';
    tipEl.style.display = 'none';
    document.body.appendChild(tipEl);

    els.timeline.addEventListener('input', onUserSeek);
    var prev = $('btnStepBack');
    var next = $('btnStepFwd');
    if (prev) prev.addEventListener('click', onUserSeek);
    if (next) next.addEventListener('click', onUserSeek);
    var play = $('btnPlay');
    if (play) {
      play.addEventListener('click', function () {
        if (!live) return;
        // 读取的是点击前的播放状态：从"正在播放"点成暂停时，用户是想停在
        // 当前时刻，不再让新帧把他拽走；从暂停点成播放时，只有人已经在右端
        // 才继续保持"跟随最新"。
        var wasPlaying = bridge().state.playing;
        live.following = wasPlaying ? false
          : (Number(els.timeline.value) >= Number(els.timeline.max));
      });
    }
    window.addEventListener('resize', hideTip);
    window.addEventListener('scroll', hideTip, true);
  }

  window.BattleTimeline = {
    ready: ready,
    ownsTransport: function () { return !!live; },
    isLiveCase: function () { return !!live; },
    onCaseLoaded: onCaseLoaded,
    launch: launch,
    render: render
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
}());
