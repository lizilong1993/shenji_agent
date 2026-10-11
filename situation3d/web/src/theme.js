/* situation3d — 视觉编码取值口（I09）
 *
 * 职责：把 `theme.css` 里的 CSS 变量读成 JS 能用的值，**并只在这一处解析**。
 *
 * 为什么需要它（Cesium 侧的硬约束）
 * --------------------------------
 * 本项目的视觉编码有两类消费者：
 *   - DOM/CSS 消费者直接 `var(--faction-red)` 即可；
 *   - **Cesium 画布**不行：它要的是 `Cesium.Color`，无法从 CSS 取值。
 * 于是原先画布侧把颜色**硬编码在 `app.js:colorFor()`** 里，
 * 与 `style.css` 的 `--red` 各写一份 —— 实测两个"红方"是**不同的值**
 * （画布 `#d0342c` / DOM `#c8352c`），改一处不影响另一处。
 *
 * 本模块把 CSS 定为**唯一来源**，画布侧改为从这里取。
 *
 * 取值策略：**CSS 变量优先，内置默认兜底**
 * --------------------------------------
 * 若直接读 CSS，在 `theme.css` 未加载（例如单测环境无 DOM，或文件缺失）时
 * 会读到空串，画布就会拿 `undefined` 去建色 —— 那是**静默变黑**，最难查。
 * 故每个键都有与 `theme.css` 一致的默认值；读不到就回落，并**记录一次**，
 * 通过 `Theme.debug()` 暴露，让"回落"可见而不是假装一切正常。
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.Theme = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  // 与 theme.css 的 :root 保持一致。改这里必须同步改 theme.css ——
  // `Theme.debug().mismatches` 会对两者做比对并把差异报出来。
  var DEFAULTS = {
    'faction-red': '#E74C3C',
    'faction-blue': '#2E86C1',
    'faction-green': '#2f7d4f',
    'faction-neutral': '#95A5A6',
    'faction-contested': '#d8a12a',
    'faction-unknown': '#F1C40F',
    'strength-0': '#7d8a99',
    'strength-1': '#4a9c6d',
    'strength-2': '#d8a12a',
    'strength-3': '#d9772a',
    'strength-4': '#c8352c',
    'status-border-width': '2px',
    'status-border-normal': '#6b7688',
    'status-border-alert': '#c8352c',
    'status-border-selected': '#1668dc',
    'pulse-period-default': '1.5s',
    'pulse-period-fast': '0.8s',
    'pulse-speed-threshold-kmh': '50',
    'pulse-enabled': '1',
    'hover-offset': '8px',
    'hover-transition': '200ms',
    'ring-strength-base-radius': '6px',
    'ring-strength-step': '2px',
    'ring-max-rings': '4'
  };

  var _fallbackUsed = {};
  var _cache = null;

  function _doc() {
    return (typeof document !== 'undefined') ? document : null;
  }

  function _readVar(name) {
    var d = _doc();
    if (!d || !d.documentElement) return '';
    try {
      var v = getComputedStyle(d.documentElement).getPropertyValue('--' + name);
      return (v === null || v === undefined) ? '' : String(v).trim();
    } catch (e) {
      return '';
    }
  }

  /** 读一个变量：CSS 优先，缺失则回落默认并记录。 */
  function raw(name) {
    if (!(name in DEFAULTS)) throw new Error('Theme: unknown key ' + name);
    var v = _readVar(name);
    if (!v) {
      if (!_fallbackUsed[name]) _fallbackUsed[name] = 0;
      _fallbackUsed[name] += 1;
      return DEFAULTS[name];
    }
    return v;
  }

  function num(name) {
    var v = String(raw(name)).replace(/[^0-9.+-]/g, '');
    var n = parseFloat(v);
    return isFinite(n) ? n : 0;
  }

  /** 解析时长：`1.5s` / `800ms` / `0.8s` → **秒**。 */
  function seconds(name) {
    var s = String(raw(name)).trim().toLowerCase();
    if (/ms$/.test(s)) return parseFloat(s) / 1000;
    return parseFloat(s);
  }

  function px(name) { return num(name); }

  function color(name) { return raw(name); }

  /**
   * 时效外圈脉冲周期（秒）。步骤 4 的规则集中在这里：
   * 实时目标 1.5s；速度 > 阈值（默认 50km/h）时改为 0.8s。
   * **两个取值都来自 CSS 变量**，不是字面量。
   */
  function pulsePeriodSeconds(speedKmh) {
    if (num('pulse-enabled') === 0) return 0;          // 关闭脉冲（降级）
    var fast = seconds('pulse-period-fast');
    var slow = seconds('pulse-period-default');
    var thr = num('pulse-speed-threshold-kmh');
    return (typeof speedKmh === 'number' && isFinite(speedKmh) && speedKmh > thr) ? fast : slow;
  }

  /** 强度内环：返回半径（px）。强度 0 或越界 → 0（不画环）。 */
  function strengthRingRadius(strength) {
    var s = parseInt(strength, 10);
    if (!isFinite(s) || s <= 0) return 0;
    var maxRings = num('ring-max-rings');
    if (s > maxRings) s = maxRings;
    return px('ring-strength-base-radius') + (s - 1) * px('ring-strength-step');
  }

  function faction(affiliation) {
    var a = String(affiliation || '').toLowerCase();
    if (a === 'red') return color('faction-red');
    if (a === 'blue') return color('faction-blue');
    if (a === 'green') return color('faction-green');
    return color('faction-neutral');
  }

  function strengthColor(level) {
    var n = parseInt(level, 10);
    if (!isFinite(n) || n < 0 || n > 4) n = 0;
    return color('strength-' + n);
  }

  /** 主题切换：`setTheme('dark')` / `setTheme(null)`。不改编码语义，只改取值。 */
  function setTheme(name) {
    var d = _doc();
    if (!d || !d.documentElement) return false;
    if (name) d.documentElement.setAttribute('data-theme', String(name));
    else d.documentElement.removeAttribute('data-theme');
    _cache = null;
    return true;
  }

  function currentTheme() {
    var d = _doc();
    if (!d || !d.documentElement) return null;
    return d.documentElement.getAttribute('data-theme') || null;
  }

  /**
   * 取证出口（供 `tools/*selftest*.py` 的 `SituationHost.*Debug()` 体系与 CDP 探针）。
   * 除当前取值外，还给出：
   *   - `fallbackUsed`：哪些键在回落（说明 theme.css 没生效）；
   *   - `mismatches`：CSS 实读值与内置默认**不一致**的键 ——
   *     正常应只有主题相关的键在深色下不同；浅色下应为空。
   */
  function debug() {
    var values = {};
    for (var k in DEFAULTS) values[k] = raw(k);
    var mism = {};
    for (var k2 in DEFAULTS) {
      var cssv = _readVar(k2);
      if (cssv && cssv !== DEFAULTS[k2]) mism[k2] = { css: cssv, builtin: DEFAULTS[k2] };
    }
    return {
      theme: currentTheme(),
      cssAvailable: !!_readVar('faction-red'),
      values: values,
      fallbackUsed: Object.keys(_fallbackUsed),
      mismatches: mism,
      pulse: {
        defaultSeconds: seconds('pulse-period-default'),
        fastSeconds: seconds('pulse-period-fast'),
        thresholdKmh: num('pulse-speed-threshold-kmh'),
        at49: pulsePeriodSeconds(49),
        at51: pulsePeriodSeconds(51)
      },
      hover: { offsetPx: px('hover-offset'), transitionMs: px('hover-transition') },
      ring: { base: px('ring-strength-base-radius'), step: px('ring-strength-step'),
              maxRings: num('ring-max-rings'),
              radiusAt0: strengthRingRadius(0), radiusAt1: strengthRingRadius(1),
              radiusAt4: strengthRingRadius(4), radiusAt9: strengthRingRadius(9) }
    };
  }

  return {
    DEFAULTS: DEFAULTS,
    raw: raw, num: num, seconds: seconds, px: px, color: color,
    faction: faction, strengthColor: strengthColor,
    pulsePeriodSeconds: pulsePeriodSeconds,
    strengthRingRadius: strengthRingRadius,
    setTheme: setTheme, currentTheme: currentTheme,
    debug: debug
  };
});
