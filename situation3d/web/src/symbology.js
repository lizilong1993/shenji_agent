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

  // 与 engine/symbology.py 的 AFFILIATIONS 保持一致；这里的值只在数据集
  // 尚未加载时作为兜底。
  const FALLBACK_AFFILIATIONS = {
    red: { color: '#d0342c', ink: '#7d1611', label: '红方' },
    blue: { color: '#1f5fbf', ink: '#123a78', label: '蓝方' },
  };

  let AFFILIATIONS = Object.assign({}, FALLBACK_AFFILIATIONS);
  let ICONS = {};

  // 框体几何：矩形 36x24，居中略偏下，为域修饰符留出空间。
  const FRAME = { x: 14, y: 20, w: 36, h: 24 };

  // 域修饰符：空中是框体上方的弧，水面是下方的弧。
  const DOMAIN_ARC = {
    air: 'M14,20 A18,9 0 0 1 50,20 Z',
    sea: 'M14,44 A18,9 0 0 0 50,44 Z',
  };

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
  };

  const UNKNOWN_GLYPH = [
    { d: 'M32,25 L32,37' },
    { d: 'M32,40 L32,40.5' },
  ];

  function affiliationOf(key) {
    return AFFILIATIONS[key] || FALLBACK_AFFILIATIONS[key] || {
      color: '#6b7688', ink: '#3b434f', label: '未知',
    };
  }

  function domainOf(icon) {
    const meta = ICONS[icon];
    if (meta && meta.domain) return meta.domain;
    return 'land';
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
    const aff = affiliationOf(affiliation);
    const frame = aff.color;
    const ink = '#ffffff';
    const domain = domainOf(icon);
    const glyph = GLYPHS[icon] || UNKNOWN_GLYPH;

    const parts = [];

    // 域修饰符画在框体下面，避免盖住框体描边。
    const arc = DOMAIN_ARC[domain];
    if (arc) {
      parts.push('<path d="' + arc + '" fill="' + frame + '" stroke="' + ink +
                 '" stroke-width="1.5" stroke-linejoin="round"/>');
    }

    parts.push('<rect x="' + FRAME.x + '" y="' + FRAME.y + '" width="' + FRAME.w +
               '" height="' + FRAME.h + '" rx="2" fill="' + frame +
               '" stroke="' + ink + '" stroke-width="2"/>');

    glyph.forEach(function (item) {
      parts.push(item.d ? pathEl(item, ink) : circleEl(item, ink));
    });

    if (opts.selected) {
      // 选中态：外圈高亮环，与框体保持 3px 间距。
      parts.push('<rect x="' + (FRAME.x - 3) + '" y="' + (FRAME.y - 3) + '" width="' +
                 (FRAME.w + 6) + '" height="' + (FRAME.h + 6) +
                 '" rx="3" fill="none" stroke="#ffd24a" stroke-width="2.5"/>');
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
   * Data URI for a symbol, memoised: the same (icon, affiliation, selected)
   * triple is requested on every frame otherwise.
   */
  function symbolUri(icon, affiliation, opts) {
    opts = opts || {};
    const key = icon + '|' + affiliation + '|' + (opts.selected ? 1 : 0) + '|' + (opts.dim ? 1 : 0);
    let uri = cache.get(key);
    if (!uri) {
      uri = toDataUri(symbolSvg(icon, affiliation, opts));
      cache.set(key, uri);
    }
    return uri;
  }

  /** Adopt the catalogue published by the engine. */
  function setCatalog(catalog) {
    if (!catalog) return;
    if (catalog.affiliations) AFFILIATIONS = catalog.affiliations;
    if (catalog.icons) ICONS = catalog.icons;
    cache.clear();
  }

  function iconLabel(icon) {
    const meta = ICONS[icon];
    return meta ? meta.label : '未知目标';
  }

  function iconAbbrev(icon) {
    const meta = ICONS[icon];
    return meta ? meta.abbrev : 'UNK';
  }

  global.Symbology = {
    symbolSvg: symbolSvg,
    symbolUri: symbolUri,
    setCatalog: setCatalog,
    iconLabel: iconLabel,
    iconAbbrev: iconAbbrev,
    domainOf: domainOf,
    affiliationOf: affiliationOf,
  };
})(window);
