/* situation3d — 统一事件模型（I08 步骤 1/5）
 *
 * 依据 31004 6.2.4 / 6.2.5：信息提示与告警、进程状态展示，
 * 统一事件模型字段为 **时间 / 级别 / 来源 / 对象 / 内容 / 原始数据 JSON**。
 *
 * 为什么需要它
 * ------------
 * 本项目原先各页**自建提示结构**：状态面板拼一句字符串、数据健康页另拼一句、
 * 渲染降级又是第三种。于是同一个问题在三处有三种表述，无法按级别筛、无法导出、
 * 也无法回答「过去 5 分钟出现过哪些告警」。
 *
 * 设计取舍
 * --------
 * 1. **`raw` 必须存在但可为空对象** —— 判据 1 写的是"raw 可为空 JSON 但需存在"。
 *    故 `push()` 会把缺失的 `raw` 补成 `{}`，而不是留 `undefined`。
 * 2. **队列有上限并按策略丢弃**（【风险与回滚】要求事件量增长不得拖慢渲染）。
 *    丢弃时**记一条 `dropped` 计数**而不是装作没发生 —— 静默丢事件等于丢证据。
 * 3. **过滤与导出共用同一数据源**：`filter()` 返回的是队列的视图，
 *    `export()` 内部走同一个 `filter()`，避免"界面上筛出来的"与"导出的"不一致。
 * 4. **级别枚举固定**：`debug/info/warn/error/critical`。未知级别归一为 `info`
 *    并记一次 `coercedLevels`，不让拼错的级别静默变成"看着像 info"的东西。
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.EventFeed = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  var LEVELS = ['debug', 'info', 'warn', 'error', 'critical'];
  var LEVEL_RANK = { debug: 10, info: 20, warn: 30, error: 40, critical: 50 };
  var DEFAULT_MAX_EVENTS = 500;

  var FIELDS = ['time', 'level', 'source', 'object', 'content', 'raw'];

  var _events = [];
  var _seq = 0;
  var _stats = { pushed: 0, dropped: 0, coercedLevels: 0, rejected: 0 };
  var _maxEvents = DEFAULT_MAX_EVENTS;
  var _listeners = [];
  var _sinks = [];      // 订阅者的 source 名，用于 debug 取证

  function _nowIso() {
    try { return new Date().toISOString(); }
    catch (e) { return String(Date.now()); }
  }

  function _normLevel(level) {
    var l = String(level === undefined || level === null ? '' : level).toLowerCase().trim();
    if (LEVELS.indexOf(l) !== -1) return l;
    _stats.coercedLevels += 1;
    return 'info';
  }

  /**
   * 投递一个事件。返回存下来的事件对象（含 `seq`），被拒时返回 `null`。
   *
   * `raw` 缺省补 `{}`（判据 1：可为空 JSON 但需存在）。
   */
  function push(evt) {
    if (!evt || typeof evt !== 'object') {
      _stats.rejected += 1;
      return null;
    }
    var e = {
      seq: ++_seq,
      time: evt.time || _nowIso(),
      level: _normLevel(evt.level),
      source: evt.source === undefined || evt.source === null ? '' : String(evt.source),
      object: evt.object === undefined || evt.object === null ? '' : String(evt.object),
      content: evt.content === undefined || evt.content === null ? '' : String(evt.content),
      raw: (evt.raw === undefined || evt.raw === null) ? {} : evt.raw
    };
    _events.push(e);
    _stats.pushed += 1;

    // 上限与丢弃策略：丢**最旧**的（保留最近发生的事）
    while (_events.length > _maxEvents) {
      _events.shift();
      _stats.dropped += 1;
    }
    for (var i = 0; i < _listeners.length; i++) {
      try { _listeners[i](e); } catch (err) { /* 订阅者自己出错不能拖垮投递 */ }
    }
    return e;
  }

  /** 订阅（供告警页增量渲染）。返回取消订阅函数。 */
  function subscribe(fn, name) {
    if (typeof fn !== 'function') return function () {};
    _listeners.push(fn);
    _sinks.push(String(name || ('sink' + _listeners.length)));
    return function () {
      var i = _listeners.indexOf(fn);
      if (i >= 0) { _listeners.splice(i, 1); _sinks.splice(i, 1); }
    };
  }

  /**
   * 过滤。`opts`:
   *   level      最低级别（含），如 `'warn'` → 只回 warn/error/critical
   *   levels     精确级别数组（与 level 同用时取交集）
   *   source     子串匹配（不区分大小写），**同时匹配 `source` 与 `object`**
   *              —— 实测暴露的问题：`noteConnectivity()` 投的是
   *              `source:'dataHealth'` + `object:'network'`，
   *              调用方自然会写 `{source:'network'}` 去筛网络事件，
   *              而只比 `source` 会得到 **0 条**。两者都是"这条事件来自哪"的
   *              合理标识，故都参与匹配；这比要求调用方知道该写哪个字段更稳。
   *   sinceSeq   只回 seq 大于它的
   *   limit      最多回多少条（从**最新**往回取，返回仍按时间升序）
   */
  function filter(opts) {
    var o = opts || {};
    var minRank = o.level ? (LEVEL_RANK[String(o.level).toLowerCase()] || 0) : 0;
    var exact = null;
    if (o.levels && o.levels.length) {
      exact = {};
      for (var i = 0; i < o.levels.length; i++) exact[_normLevel(o.levels[i])] = true;
    }
    var srcSub = o.source ? String(o.source).toLowerCase() : null;
    var since = (typeof o.sinceSeq === 'number') ? o.sinceSeq : null;

    var out = [];
    for (var k = 0; k < _events.length; k++) {
      var e = _events[k];
      if (LEVEL_RANK[e.level] < minRank) continue;
      if (exact && !exact[e.level]) continue;
      if (srcSub
          && e.source.toLowerCase().indexOf(srcSub) === -1
          && e.object.toLowerCase().indexOf(srcSub) === -1) continue;
      if (since !== null && !(e.seq > since)) continue;
      out.push(e);
    }
    if (o.limit && out.length > o.limit) out = out.slice(out.length - o.limit);
    return out;
  }

  function all() { return _events.slice(); }
  function count() { return _events.length; }
  function latest() { return _events.length ? _events[_events.length - 1] : null; }
  function levels() { return LEVELS.slice(); }

  /** 字段完整率（判据 1 要求 100%）。 */
  function completeness() {
    if (!_events.length) return 1;
    var ok = 0;
    for (var i = 0; i < _events.length; i++) {
      var e = _events[i], good = true;
      for (var f = 0; f < FIELDS.length; f++) {
        var v = e[FIELDS[f]];
        // `raw` 允许是空对象，但**必须存在且类型正确**
        if (f === 5) { if (v === undefined || v === null || typeof v !== 'object') good = false; }
        else if (v === undefined || v === null || v === '') good = false;
      }
      if (good) ok += 1;
    }
    return ok / _events.length;
  }

  // ── 导出（步骤 5：过滤与导出共用同一数据源）─────────────────────────────

  function _csvCell(v) {
    var s = (typeof v === 'string') ? v : JSON.stringify(v);
    if (s === undefined || s === null) s = '';
    // RFC4180：含逗号/引号/换行时加引号并转义引号
    if (/[",\n\r]/.test(s)) s = '"' + s.replace(/"/g, '""') + '"';
    return s;
  }

  function exportCSV(opts) {
    var rows = filter(opts);
    var head = ['seq', 'time', 'level', 'source', 'object', 'content', 'raw'];
    var lines = [head.join(',')];
    for (var i = 0; i < rows.length; i++) {
      var e = rows[i];
      lines.push([e.seq, e.time, e.level, e.source, e.object, e.content,
                  JSON.stringify(e.raw)].map(_csvCell).join(','));
    }
    return lines.join('\n') + '\n';
  }

  function exportJSON(opts) {
    return JSON.stringify({ exported_at: _nowIso(), count: filter(opts).length,
                            events: filter(opts) }, null, 2);
  }

  function clear() { _events = []; }

  function setMaxEvents(n) {
    var v = parseInt(n, 10);
    if (isFinite(v) && v > 0) _maxEvents = v;
    return _maxEvents;
  }

  /** 取证出口（并入 `SituationHost.*Debug()` 体系）。 */
  function debug() {
    return {
      fields: FIELDS.slice(),
      levels: LEVELS.slice(),
      count: _events.length,
      maxEvents: _maxEvents,
      fieldsCompleteRate: completeness(),
      stats: { pushed: _stats.pushed, dropped: _stats.dropped,
               coercedLevels: _stats.coercedLevels, rejected: _stats.rejected },
      sinks: _sinks.slice(),
      latest: latest(),
      byLevel: (function () {
        var m = {};
        for (var i = 0; i < LEVELS.length; i++) m[LEVELS[i]] = 0;
        for (var k = 0; k < _events.length; k++) m[_events[k].level] = (m[_events[k].level] || 0) + 1;
        return m;
      })()
    };
  }

  return {
    FIELDS: FIELDS, LEVELS: LEVELS,
    push: push, subscribe: subscribe, filter: filter,
    all: all, count: count, latest: latest, levels: levels,
    completeness: completeness,
    exportCSV: exportCSV, exportJSON: exportJSON,
    clear: clear, setMaxEvents: setMaxEvents, debug: debug
  };
});
