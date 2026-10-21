/* situation3d — 军标（MIL-STD-2525 风格）矢量生成
 *
 * 符号构成遵循 MIL-STD-2525 / APP-6 的惯例，并适配红蓝双方对抗着色：
 *
 *   框体(frame)     红方红、蓝方蓝的矩形，白色描边保证在卫星影像上可辨
 *   域修饰符        空中=框体上方弧线；水面=框体下方弧线；陆上=无弧线
 *   图形(glyph)     框内的平台族线画
 *
 * 图标键的权威定义在 engine/symbology.py，前端不再复制一份，而是从数据集
 * 的 symbology 段读取，避免两侧定义漂移。
 *
 * 输出是 SVG data URI，直接交给 Cesium.BillboardGraphics 使用。
 */

'use strict';

(function (global) {
  const SIZE = 64;

  /**
   * 预生成的军标资产目录。
   *
   * 首选路径是**静态 SVG**：由 `tools/gen_symbols.py` 用 milsymbol（MIT，MIL-STD-2525D）
   * 离线生成、提交进仓库，文件名 `<icon>_<affiliation>[_sel].svg`。
   * 好处是渲染确定、可逐个人工核对、浏览器不需要 862 KB 的符号库，且红蓝与域的形状
   * 全部由标准 SIDC 决定——下面这套手绘 GLYPHS 只作为**兜底**：资产缺失时（例如只
   * 部署了源码、没跑生成器）仍然要有图标，不能开天窗。
   *
   * 兜底是**近似**：它保留了 2525 的关键语义（敌方=尖角菱形、友方=矩形、空中=上方弧、
   * 水面与陆上无域修饰符），但不像 milsymbol 那样按域给出方/圆/拱等标准帧形。
   */
  // 生成这些资产时用的 milsymbol 版本（tools/gen_symbols.py 写进 manifest）。
  // 对不上只告警不报错：形状可能变了，但用着总比空白好。
  const MILSYMBOL_EXPECTED = '3.0.4';
  const ASSET_BASE = './assets/symbols/';
  let useAssets = true;
  // manifest 里有资产的图标集合。目录（engine/symbology.py 或 base.json）可能出现
  // **没有对应资产**的键（例如别的分支加过的图标），那种键不能算"已知"，
  // 否则 billboard 会指向 404 的 SVG，页面上是空白且不报错。
  let assetIconSet = null;
  let assetManifest = null;

  // 与 engine/symbology.py 的 AFFILIATIONS 保持一致；这里的值只在数据集
  // 尚未加载时作为兜底。
  // ⚠ 这里必须**列全所有阵营**，否则下一段的 `Object.keys()` 循环覆盖不到它们 ——
  // 实测：原先只列 red/blue，于是 green/neutral/unknown 取不到 `Theme` 的值，
  // 回落到 `#6b7688`（状态边框色），**阵营维度静默失效**。
  // （该缺陷由 quickjs 实际执行 `symbolSvg` 后检查 SVG 的 `fill` 值发现。）
  const FALLBACK_AFFILIATIONS = {
    // 兜底色**必须与 theme.css / theme.js DEFAULTS 一致**（31004 表3 的标准值）；
    // 若留旧值，`Theme` 不可用时就会悄悄换成一个非标准的颜色。
    red: { color: '#E74C3C', ink: '#7d1611', label: '红方' },
    blue: { color: '#2E86C1', ink: '#123a78', label: '蓝方' },
    green: { color: '#2f7d4f', ink: '#123a1c', label: '第三方/中立' },
    neutral: { color: '#95A5A6', ink: '#3a4149', label: '未知/未归属' },
    unknown: { color: '#F1C40F', ink: '#5a4600', label: '未知' },
  };

  // ── I09：语义色唯一来源 ──────────────────────────────────────────────
  // 阵营色由 `theme.css` 的 `--faction-*` 定义，经 `Theme` 读取。
  // 下面的字面量**只是 `Theme` 不可用时的兜底**（与 `theme.js` 的
  // "CSS 变量优先、内置默认兜底" 同一口径）—— 故它们不再是"第二份定义"，
  // 而是与 `theme.js` DEFAULTS 同级的降级值。
  function _semantic(key, fallback) {
    try {
      if (window.Theme && typeof window.Theme.color === 'function') {
        var v = window.Theme.color(key);
        if (v) return v;
      }
    } catch (e) { /* Theme 不可用时用兜底，不让渲染整体失败 */ }
    return fallback;
  }

  let AFFILIATIONS = Object.assign({}, FALLBACK_AFFILIATIONS);
  // 阵营主色**改由 Theme 提供**（`--faction-*`）；`ink` 等非语义字段仍用兜底表。
  Object.keys(AFFILIATIONS).forEach(function (k) {
    AFFILIATIONS[k] = Object.assign({}, AFFILIATIONS[k], {
      color: _semantic('faction-' + k, AFFILIATIONS[k].color),
    });
  });
  let ICONS = {};
  // 域的中文名。放在这里而不是从 state.data.symbology 读，是为了让信息面板
  // 不再依赖案例数据集——空态下同样要能渲染符号说明。
  let DOMAINS = { sea: '水面', land: '陆上', air: '空中' };

  // 框体几何：矩形 36x24，居中略偏下，为域修饰符留出空间。
  const FRAME = { x: 14, y: 20, w: 36, h: 24 };

  /**
   * 域修饰符：**只有空中是框体上方的弧**。
   *
   * 原先这里水面也画了一条下方弧线——那是**潜艇**（subsurface）的修饰符，不是水面
   * 舰艇的。MIL-STD-2525 里水面单位**没有任何域修饰符**（帧形本身就表示水面）。
   * 已核实：DISA 的部件库里 `0_630_0.svg`（敌方+水面）与 `0_610_0.svg`（敌方+陆上）
   * 是同一个纯菱形，水面那一版没有弧。
   * 陆上同样无修饰符。
   */
  const DOMAIN_ARC = {
    air: 'M14,20 A18,9 0 0 1 50,20 Z',
  };

  /**
   * 敌方帧形 = **尖角菱形**（矩形四边中点外凸），友方 = 矩形。
   *
   * 原先红蓝双方都画矩形，只靠颜色区分——而 2525 里**对抗关系是靠帧形编码的**，
   * 只改颜色会让"红方"在形制上仍是友军符号。改后：友军=矩形、敌军=菱形。
   * （标准里敌方空中还是五边形、友军海上还是圆形等更细的划分由 milsymbol 那份
   * 静态资产负责；这里是兜底，取最要紧的两条语义。）
   */
  function framePath(affiliationKey) {
    const x = FRAME.x, y = FRAME.y, w = FRAME.w, h = FRAME.h;
    if (affiliationKey !== 'red') {
      return { tag: 'rect', attrs: 'x="' + x + '" y="' + y + '" width="' + w +
               '" height="' + h + '" rx="2"' };
    }
    const mx = x + w / 2, my = y + h / 2, px = w * 0.26, py = h * 0.26;
    return { tag: 'polygon',
             attrs: 'points="' + [mx, y, x + w + px, my, mx, y + h, x - px, my]
               .join(' ') + '"' };
  }

  // 图形线画。坐标全部落在框体内侧 (x 18..46, y 24..40)。
  const GLYPHS = {
    destroyer: [
      { d: 'M20,35 L44,35 L40,40 L24,40 Z', fill: true },
      { d: 'M32,35 L32,27' },
      { d: 'M27,27 L37,27' },
    ],
    frigate: [
      { d: 'M20,35 L44,35 L40,40 L24,40 Z', fill: true },
      { d: 'M25,35 L39,27' },
    ],
    replenishment: [
      { d: 'M20,35 L44,35 L40,40 L24,40 Z', fill: true },
      { d: 'M32,35 L32,27' },
      { d: 'M26,31 L38,31' },
    ],
    corvette: [
      { d: 'M22,36 L42,36 L38,40 L26,40 Z', fill: true },
      { c: [32, 31, 2.5], fill: true },
    ],
    coastal_missile: [
      { d: 'M32,25 L35,32 L35,39 L29,39 L29,32 Z', fill: true },
      { d: 'M29,36 L25,40' },
      { d: 'M35,36 L39,40' },
    ],
    sam_battalion: [
      { d: 'M26,39 A6,6 0 0 1 38,39', fill: false },
      { d: 'M23,39 L41,39' },
      { d: 'M32,33 L32,26' },
    ],
    armor: [
      { d: 'M25,29 L39,29 L39,37 L25,37 Z', fill: true },
      { d: 'M23,40 L41,40' },
      { d: 'M32,29 L32,24' },
    ],
    artillery: [
      { d: 'M32,39 L41,27' },
      { c: [32, 39, 2.5], fill: true },
      { d: 'M23,39 L41,39' },
    ],
    fighter: [
      { d: 'M32,24 L44,40 L32,35 L20,40 Z', fill: true },
    ],
    aew: [
      { d: 'M32,28 L43,40 L32,36 L21,40 Z', fill: true },
      { d: 'M28,27 A4.5,4.5 0 0 1 36,27 Z', fill: true },
      { d: 'M32,27 L32,24' },
    ],
    helicopter: [
      { d: 'M22,26 L42,26' },
      { d: 'M26,37 L38,37 L36,40 L28,40 Z', fill: true },
      { d: 'M32,26 L32,35' },
      { d: 'M36,33 L40,33' },
    ],
    uav: [
      { d: 'M32,26 L42,32 L32,30 L22,32 Z', fill: true },
      { c: [32, 34, 2], fill: true },
      { d: 'M32,30 L32,34' },
    ],
    landing_ship: [
      { d: 'M20,35 L44,35 L41,40 L23,40 Z', fill: true },
      { d: 'M44,35 L47,30 L48,33 L46,36 Z', fill: true },
      { d: 'M27,30 L36,30 L36,35 L27,35 Z' },
    ],
    transport: [
      { d: 'M21,33 L37,33 L37,39 L21,39 Z', fill: true },
      { d: 'M37,35 L42,35 L42,39 L37,39 Z' },
      { c: [25, 40, 1.8], fill: true },
      { c: [39, 40, 1.8], fill: true },
    ],
    outpost: [
      { d: 'M22,40 L22,33 L32,26 L42,33 L42,40 Z', fill: true },
      { d: 'M30,40 L30,35 L34,35 L34,40' },
    ],
    // 大类兜底（新目标没有专属图标时按域落到这里）
    generic_sea: [
      { d: 'M20,35 L44,35 L40,40 L24,40 Z', fill: true },
      { d: 'M32,35 L32,27' },
      { d: 'M26,40 L38,40' },
    ],
    generic_land: [
      { d: 'M24,26 L40,40' },
      { d: 'M40,26 L24,40' },
      { d: 'M23,27 L27,23' },
      { d: 'M37,23 L41,27' },
    ],
    generic_air: [
      { d: 'M32,24 L44,40 L32,36 L20,40 Z', fill: true },
    ],
    unknown: [
      { c: [32, 34, 4], fill: false },
      { d: 'M28.5,30 C28.5,27 35.5,27 35.5,30 C35.5,32.5 32,32.5 32,35' },
      { c: [32, 39, 1], fill: true },
    ],
  };

  const UNKNOWN_GLYPH = [
    { d: 'M32,25 L32,37' },
    { d: 'M32,40 L32,40.5' },
  ];

  // 专属图标 → 大类（按域）→ unknown。目录里没有的键**不能画空白**：
  // 新目标只来得及写 domain 时，就落到 generic_sea / generic_land / generic_air。
  const GENERIC_BY_DOMAIN = { sea: 'generic_sea', land: 'generic_land', air: 'generic_air' };

  function isKnownIcon(icon) {
    if (!icon) return false;
    // 用静态资产时**必须有资产**才算已知：只存在于目录里的键会渲染成空图。
    if (useAssets && assetIconSet) return assetIconSet.has(icon);
    if (Object.prototype.hasOwnProperty.call(ICONS, icon)) return true;
    return Object.prototype.hasOwnProperty.call(GLYPHS, icon);
  }

  function resolveIcon(icon, domain) {
    if (isKnownIcon(icon)) return icon;
    const g = GENERIC_BY_DOMAIN[domain];
    if (g && isKnownIcon(g)) return g;
    return 'unknown';
  }

  function affiliationOf(key) {
    return AFFILIATIONS[key] || FALLBACK_AFFILIATIONS[key] || {
      color: '#6b7688', ink: '#3b434f', label: '未知',
    };
  }

  function domainOf(icon, fallback) {
    const meta = ICONS[icon];
    if (meta && meta.domain) return meta.domain;
    return fallback || 'land';
  }

  function pathEl(item, color) {
    const fill = item.fill ? color : 'none';
    return '<path d="' + item.d + '" fill="' + fill + '" stroke="' + color +
           '" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>';
  }

  function circleEl(item, color) {
    return '<circle cx="' + item.c[0] + '" cy="' + item.c[1] + '" r="' + item.c[2] +
           '" fill="' + (item.fill ? color : 'none') + '" stroke="' + color +
           '" stroke-width="1.5"/>';
  }

  /**
   * Build the SVG markup for one symbol.
   *
   * @param {string} icon        icon key from engine/symbology.py
   * @param {string} affiliation 'red' | 'blue'
   * @param {object} [opts]      { selected: boolean, dim: boolean }
   * @returns {string} SVG source
   */
  function symbolSvg(icon, affiliation, opts) {
    opts = opts || {};
    icon = resolveIcon(icon, opts.domain);
    const aff = affiliationOf(affiliation);
    const frame = aff.color;
    const ink = '#ffffff';
    const domain = domainOf(icon, opts.domain);
    const glyph = GLYPHS[icon] || UNKNOWN_GLYPH;
    // 帧形由对抗关系决定：友方矩形、敌方尖角菱形（见 framePath 的说明）。
    const fp = framePath(affiliation);

    const parts = [];

    // 域修饰符画在框体下面，避免盖住框体描边。
    const arc = DOMAIN_ARC[domain];
    if (arc) {
      parts.push('<path d="' + arc + '" fill="' + frame + '" stroke="' + ink +
                 '" stroke-width="1.5" stroke-linejoin="round"/>');
    }

    parts.push('<' + fp.tag + ' ' + fp.attrs + ' fill="' + frame +
               '" stroke="' + ink + '" stroke-width="2"/>');
    glyph.forEach(function (item) {
      parts.push(item.d ? pathEl(item, ink) : circleEl(item, ink));
    });

    if (opts.selected) {
      // 选中态：外圈高亮环。敌方帧是菱形，用矩形环会在四角留出空隙，所以按同一个
      // 形状放大一圈来画——环的形状必须跟着帧形走。
      if (fp.tag === 'polygon') {
        const pts = fp.attrs.match(/points="([^"]+)"/)[1].split(/\s+/)
          .map(function (p) { return p.split(',').map(Number); });
        const mx = (FRAME.x + FRAME.w / 2), my = (FRAME.y + FRAME.h / 2);
        const grown = pts.map(function (p) {
          const dx = p[0] - mx, dy = p[1] - my;
          const len = Math.sqrt(dx * dx + dy * dy) || 1;
          const k = (len + 3.5) / len;
          return (mx + dx * k).toFixed(1) + ',' + (my + dy * k).toFixed(1);
        }).join(' ');
        parts.push('<polygon points="' + grown +
                   '" fill="none" stroke="#ffd24a" stroke-width="2.5"/>');
      } else {
        parts.push('<rect x="' + (FRAME.x - 3) + '" y="' + (FRAME.y - 3) + '" width="' +
                   (FRAME.w + 6) + '" height="' + (FRAME.h + 6) +
                   '" rx="3" fill="none" stroke="#ffd24a" stroke-width="2.5"/>');
      }
    }

    const opacity = opts.dim ? 0.35 : 1;
    return '<svg xmlns="http://www.w3.org/2000/svg" width="' + SIZE + '" height="' +
           SIZE + '" viewBox="0 0 ' + SIZE + ' ' + SIZE + '" opacity="' + opacity +
           '">' + parts.join('') + '</svg>';
  }

  function toDataUri(svg) {
    // encodeURIComponent keeps the URI valid without pulling in base64 for
    // what is mostly ASCII markup.
    return 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svg);
  }

  const cache = new Map();

  /**
   * URL for a symbol, memoised: the same (icon, affiliation, selected) triple is
   * requested on every frame otherwise.
   *
   * 默认返回**静态资产**路径（`assets/symbols/<icon>_<affil>[_sel].svg`，由
   * tools/gen_symbols.py 用 milsymbol 预生成并提交）；只有在资产被判定不可用
   * （见 verifyAssets）时才退回内置生成器的 data URI。
   *
   * 返回的是 URL 而不是 data URI，所以 Cesium 的 billboard 与 HTML 的 `<img>` 都能
   * 直接用；缓存也随之变成"路径查询"，比编码 SVG 便宜得多。
   */
  function assetUri(icon, affiliation, opts) {
    opts = opts || {};
    icon = resolveIcon(icon, opts.domain);
    return ASSET_BASE + icon + '_' + affiliation + (opts.selected ? '_sel' : '') + '.svg';
  }

  function symbolUri(icon, affiliation, opts) {
    // 兼容两种调用：`symbolUri(icon, aff, 'air')` 与 `symbolUri(icon, aff, {domain:'air'})`。
    if (typeof opts === 'string') opts = { domain: opts };
    opts = opts || {};
    icon = resolveIcon(icon, opts.domain);
    if (useAssets) return assetUri(icon, affiliation, opts);
    const key = icon + '|' + affiliation + '|' + (opts.selected ? 1 : 0) + '|' + (opts.dim ? 1 : 0);
    let uri = cache.get(key);
    if (!uri) {
      uri = toDataUri(symbolSvg(icon, affiliation, opts));
      cache.set(key, uri);
    }
    return uri;
  }

  /**
   * 检查预生成资产是否可用，并据此决定走静态资产还是内置兜底。
   *
   * 为什么要探测而不是假定存在：`web/assets/symbols/` 是**提交进 git 的**，但若有人
   * 只部署了源码而没跑生成器（或生成器换了 milsymbol 版本后忘了提交产物），图标会
   * 整片消失——而 `<img>`/billboard 加载失败是**静默**的，页面上只会看到空白。
   * 用一次 manifest 请求把这件事变成显式判断，并在退回时打日志。
   *
   * 返回 Promise<boolean>：true = 用静态资产。
   */
  function verifyAssets() {
    return fetch(ASSET_BASE + 'manifest.json', { cache: 'no-store' })
      .then(function (r) {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      })
      .then(function (m) {
        assetManifest = m;
        useAssets = true;
        assetIconSet = new Set();
        Object.keys((m && m.symbols) || {}).forEach(function (k) {
          var mm = /^(.+)_(red|blue)(?:_sel)?$/.exec(k);
          if (mm) assetIconSet.add(mm[1]);
        });
        // 版本对不上说明产物不是当前 milsymbol 生成的，仍可用但要留痕。
        if (m && m.milsymbol && m.milsymbol !== MILSYMBOL_EXPECTED) {
          console.warn('[symbology] 符号资产由 milsymbol ' + m.milsymbol +
            ' 生成，期望 ' + MILSYMBOL_EXPECTED + '；请重跑 tools/gen_symbols.py');
        }
        return true;
      })
      .catch(function (e) {
        useAssets = false;
        cache.clear();
        console.warn('[symbology] 预生成军标资产不可用（' + e.message +
          '），回退到内置生成器；跑 tools/gen_symbols.py 可生成资产');
        return false;
      });
  }

  /** 资产清单（verifyAssets 成功后可用），供诊断用。 */
  function assets() {
    return { enabled: useAssets, manifest: assetManifest };
  }

  /** 手动切换资产/兜底模式（verifyAssets 会自己调；测试与诊断也用它）。 */
  function setAssetMode(on) {
    useAssets = !!on;
    if (!useAssets) assetIconSet = null;
    cache.clear();
  }

  /** Adopt the catalogue published by the engine. */
  function setCatalog(catalog) {
    if (!catalog) return;
    if (catalog.affiliations) AFFILIATIONS = catalog.affiliations;
    if (catalog.icons) ICONS = catalog.icons;
    if (catalog.domains) DOMAINS = catalog.domains;
    cache.clear();
  }

  function domainLabel(domain) {
    return DOMAINS[domain] || domain || '未知';
  }

  function iconLabel(icon, domain) {
    const meta = ICONS[resolveIcon(icon, domain)];
    return meta ? meta.label : '未知目标';
  }

  function iconAbbrev(icon, domain) {
    const meta = ICONS[resolveIcon(icon, domain)];
    return meta ? meta.abbrev : 'UNK';
  }

  global.Symbology = {
    symbolSvg: symbolSvg,
    symbolUri: symbolUri,
    assetUri: assetUri,
    verifyAssets: verifyAssets,
    setAssetMode: setAssetMode,
    assets: assets,
    setCatalog: setCatalog,
    iconLabel: iconLabel,
    iconAbbrev: iconAbbrev,
    domainLabel: domainLabel,
    domainOf: domainOf,
    resolveIcon: resolveIcon,
    GENERIC_BY_DOMAIN: GENERIC_BY_DOMAIN,
    affiliationOf: affiliationOf,
  };
})(window);
