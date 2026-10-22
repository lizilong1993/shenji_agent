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

  // ── 大类 ────────────────────────────────────────────────────────────────
  // 目标图标按**大类**共用图形（DDG/FFG/PC 都是"舰艇"）。这张表必须与
  // tools/gen_symbols.py 的 GLYPH_OF 保持一致：那里决定静态资产里画什么，
  // 这里决定编队图标里用哪个小图形、以及标牌上的类别名。
  const CATEGORY_OF = {
    destroyer: 'sea_surface', frigate: 'sea_surface', corvette: 'sea_surface',
    transport_ship: 'sea_surface',
    landing_ship: 'amphibious',
    replenishment: 'auxiliary',
    coastal_missile: 'coastal_missile',
    sam_battalion: 'air_defense',
    armor: 'armor', light_vehicle: 'armor',
    artillery: 'artillery',
    transport: 'transport', truck: 'transport', engineer_vehicle: 'transport',
    outpost: 'outpost',
    fighter: 'fighter', aew: 'aew', helicopter: 'helicopter', uav: 'uav',
    generic_sea: 'generic_sea', generic_land: 'generic_land',
    generic_air: 'generic_air', unknown: 'unknown',
  };

  const CATEGORY_LABELS = {
    sea_surface: '舰艇', amphibious: '两栖', auxiliary: '补给',
    coastal_missile: '岸导', air_defense: '防空', armor: '装甲',
    artillery: '炮兵', transport: '运输', outpost: '阵地',
    fighter: '战斗机', aew: '预警机', helicopter: '直升机', uav: '无人机',
    generic_sea: '水面目标', generic_land: '陆上目标', generic_air: '空中目标',
    unknown: '未知',
  };

  // 编队图标中心的小类别图形。与 gen_symbols.py 的 GLYPH_BODIES 同一套路径，
  // 但这里只用于 44 px 编队图标里约 20 px 的图形，因此保留原样即可（都是实心剪影）。
  const CLUSTER_GLYPHS = {
    sea_surface:
      '<path d="M40 330 L452 330 L500 356 L452 404 L78 404 Q48 404 40 378 Z"/>' +
      '<path d="M150 250 L300 250 L330 330 L150 330 Z"/>' +
      '<path d="M190 210 L280 210 L300 250 L170 250 Z"/>' +
      '<path d="M248 160 L262 160 L262 250 L248 250 Z"/>' +
      '<path d="M306 190 L342 190 L352 250 L298 250 Z"/>' +
      '<path d="M356 296 L410 296 L418 316 L356 316 Z"/>' +
      '<path d="M410 302 L470 302 L470 309 L410 309 Z"/>',
    amphibious:
      '<path d="M60 320 L420 320 L470 372 L420 408 L100 408 Q70 400 60 370 Z"/>' +
      '<path d="M470 372 L512 300 L520 320 L488 392 Z"/>' +
      '<path d="M110 230 L250 230 L280 320 L110 320 Z"/>' +
      '<path d="M120 190 L220 190 L240 230 L120 230 Z"/>' +
      '<path d="M150 140 L162 140 L162 190 L150 190 Z"/>',
    auxiliary:
      '<path d="M50 350 L450 350 L490 378 L440 410 L90 410 Q60 405 50 380 Z"/>' +
      '<path d="M352 250 L430 250 L430 350 L340 350 Z"/>' +
      '<path d="M88 300 L168 300 L168 344 L88 344 Z"/>' +
      '<path d="M186 300 L266 300 L266 344 L186 344 Z"/>' +
      '<path d="M284 300 L330 300 L330 344 L284 344 Z"/>' +
      '<path d="M382 180 L392 180 L392 250 L382 250 Z"/>' +
      '<path d="M338 195 L434 195 L434 205 L338 205 Z"/>',
    coastal_missile:
      '<path d="M200 330 L420 330 L444 384 L180 384 Z"/>' +
      '<path d="M120 270 L210 270 L210 384 L120 384 Z"/>' +
      '<path d="M270 90 L300 68 L436 286 L408 304 Z"/>' +
      '<path d="M300 68 L322 44 L344 60 L330 84 Z"/>' +
      '<path d="M318 70 L350 48 L484 272 L456 290 Z"/>' +
      '<path d="M236 392 m-28 0 a28 28 0 1 0 56 0 a28 28 0 1 0 -56 0"/>' +
      '<path d="M380 392 m-28 0 a28 28 0 1 0 56 0 a28 28 0 1 0 -56 0"/>',
    air_defense:
      '<path fill-rule="evenodd" d="' +
      'M256 60 m-64 0 a64 64 0 1 0 128 0 a64 64 0 1 0 -128 0 ' +
      'M256 60 m-40 0 a40 40 0 1 0 80 0 a40 40 0 1 0 -80 0"/>' +
      '<path d="M248 104 L264 104 L264 220 L248 220 Z"/>' +
      '<path fill-rule="evenodd" d="' +
      'M176 220 L336 220 L336 322 L176 322 Z ' +
      'M198 242 L218 242 L218 300 L198 300 Z ' +
      'M230 242 L250 242 L250 300 L230 300 Z ' +
      'M262 242 L282 242 L282 300 L262 300 Z ' +
      'M294 242 L314 242 L314 300 L294 300 Z"/>' +
      '<path d="M156 330 L356 330 L380 384 L132 384 Z"/>' +
      '<path d="M190 392 m-26 0 a26 26 0 1 0 52 0 a26 26 0 1 0 -52 0"/>' +
      '<path d="M330 392 m-26 0 a26 26 0 1 0 52 0 a26 26 0 1 0 -52 0"/>',
    armor:
      '<path fill-rule="evenodd" d="' +
      'M90 336 L430 336 Q470 336 470 372 Q470 410 430 410 L90 410 ' +
      'Q50 410 50 372 Q50 336 90 336 Z ' +
      'M110 373 m-18 0 a18 18 0 1 0 36 0 a18 18 0 1 0 -36 0 ' +
      'M190 373 m-18 0 a18 18 0 1 0 36 0 a18 18 0 1 0 -36 0 ' +
      'M270 373 m-18 0 a18 18 0 1 0 36 0 a18 18 0 1 0 -36 0 ' +
      'M350 373 m-18 0 a18 18 0 1 0 36 0 a18 18 0 1 0 -36 0 ' +
      'M420 373 m-18 0 a18 18 0 1 0 36 0 a18 18 0 1 0 -36 0"/>' +
      '<path d="M90 304 L450 304 L450 336 L90 336 Z"/>' +
      '<path fill-rule="evenodd" d="' +
      'M160 262 L330 262 L356 304 L140 304 Z ' +
      'M232 276 m-12 0 a12 12 0 1 0 24 0 a12 12 0 1 0 -24 0"/>' +
      '<path d="M352 278 L500 278 L500 292 L352 292 Z"/>',
    artillery:
      '<path d="M170 300 L380 300 L380 344 L130 344 Z"/>' +
      '<path fill-rule="evenodd" d="' +
      'M170 390 m-70 0 a70 70 0 1 0 140 0 a70 70 0 1 0 -140 0 ' +
      'M170 390 m-38 0 a38 38 0 1 0 76 0 a38 38 0 1 0 -76 0"/>' +
      '<path fill-rule="evenodd" d="' +
      'M330 390 m-50 0 a50 50 0 1 0 100 0 a50 50 0 1 0 -100 0 ' +
      'M330 390 m-24 0 a24 24 0 1 0 48 0 a24 24 0 1 0 -48 0"/>' +
      '<path d="M340 282 L512 160 L520 180 L352 302 Z"/>' +
      '<path d="M180 350 L50 470 L68 488 L200 372 Z"/>',
    transport:
      '<path d="M90 228 L340 228 L340 342 L90 342 Z"/>' +
      '<path d="M340 258 L432 258 L454 342 L340 342 Z"/>' +
      '<path d="M80 342 L454 342 L454 372 L80 372 Z"/>' +
      '<path fill-rule="evenodd" d="' +
      'M160 400 m-40 0 a40 40 0 1 0 80 0 a40 40 0 1 0 -80 0 ' +
      'M160 400 m-18 0 a18 18 0 1 0 36 0 a18 18 0 1 0 -36 0"/>' +
      '<path fill-rule="evenodd" d="' +
      'M360 400 m-40 0 a40 40 0 1 0 80 0 a40 40 0 1 0 -80 0 ' +
      'M360 400 m-18 0 a18 18 0 1 0 36 0 a18 18 0 1 0 -36 0"/>',
    outpost:
      '<path fill-rule="evenodd" d="' +
      'M86 372 A170 170 0 0 1 426 372 Z ' +
      'M176 332 L336 332 L336 358 L176 358 Z"/>' +
      '<path d="M70 372 L442 372 L442 416 L70 416 Z"/>' +
      '<path d="M248 120 L264 120 L264 232 L248 232 Z"/>' +
      '<path d="M200 196 L312 196 L312 232 L200 232 Z"/>',
    fighter:
      '<path d="M256 24 L280 170 L470 300 L470 330 L282 262 L282 420 ' +
      'L330 470 L330 494 L256 466 L182 494 L182 470 L230 420 L230 262 ' +
      'L42 330 L42 300 L232 170 Z"/>',
    aew:
      '<path d="' +
      'M256 40 C270 90 274 150 274 230 L274 390 C274 435 267 465 256 478 ' +
      'C245 465 238 435 238 390 L238 230 C238 150 242 90 256 40 Z ' +
      'M256 196 L468 296 L468 326 L278 288 L278 208 Z ' +
      'M256 196 L44 296 L44 326 L234 288 L234 208 Z ' +
      'M256 392 L356 446 L356 468 L268 442 Z ' +
      'M256 392 L156 446 L156 468 L244 442 Z"/>' +
      '<path fill-rule="evenodd" d="' +
      'M256 150 m-92 0 a92 92 0 1 0 184 0 a92 92 0 1 0 -184 0 ' +
      'M256 150 m-58 0 a58 58 0 1 0 116 0 a58 58 0 1 0 -116 0"/>',
    helicopter:
      '<path fill-rule="evenodd" d="' +
      'M140 300 Q140 240 210 240 L330 240 Q396 240 396 306 Q396 366 330 366 ' +
      'L210 366 Q140 366 140 300 Z ' +
      'M188 268 L282 268 L282 300 L188 300 Z"/>' +
      '<path d="M330 300 L480 268 L512 268 L512 286 L360 330 Z"/>' +
      '<path d="M250 240 L266 240 L266 196 L250 196 Z"/>' +
      '<path d="M110 186 L430 186 L430 200 L110 200 Z"/>' +
      '<path d="M170 392 L370 392 L370 404 L170 404 Z"/>' +
      '<path d="M208 366 L218 366 L218 392 L208 392 Z"/>' +
      '<path d="M320 366 L330 366 L330 392 L320 392 Z"/>',
    uav:
      '<path d="' +
      'M256 60 C270 80 276 120 276 170 L276 460 C276 486 268 500 256 508 ' +
      'C244 500 236 486 236 460 L236 170 C236 120 242 80 256 60 Z ' +
      'M256 210 L40 300 L40 320 L244 274 L268 274 L472 320 L472 300 Z ' +
      'M256 446 L170 500 L170 520 L250 476 Z ' +
      'M256 446 L342 500 L342 520 L262 476 Z"/>' +
      '<path fill-rule="evenodd" d="' +
      'M256 40 m-34 0 a34 34 0 1 0 68 0 a34 34 0 1 0 -68 0 ' +
      'M256 40 m-14 0 a14 14 0 1 0 28 0 a14 14 0 1 0 -28 0"/>',
    generic_sea:
      '<path d="M60 340 L460 340 L500 366 L456 406 L100 406 Q70 402 60 380 Z"/>' +
      '<path d="M230 280 L320 280 L340 340 L210 340 Z"/>' +
      '<path d="M244 220 L258 220 L258 280 L244 280 Z"/>' +
      '<path d="M300 250 L330 250 L336 280 L294 280 Z"/>',
    generic_land:
      '<path d="M96 96 L136 56 L416 416 L376 456 Z"/>' +
      '<path d="M416 96 L456 136 L136 456 L96 416 Z"/>' +
      '<path d="M96 376 L136 416 L176 376 L136 336 Z"/>' +
      '<path d="M376 376 L416 416 L456 376 L416 336 Z"/>',
    generic_air:
      '<path d="' +
      'M256 50 L276 210 L480 210 L480 240 L276 270 L276 430 L330 470 ' +
      'L330 494 L256 470 L182 494 L182 470 L230 430 L230 270 L32 240 ' +
      'L32 210 L236 210 Z"/>',
    unknown:
      '<path d="' +
      'M256 90 C180 90 140 140 140 200 L196 200 C196 172 218 146 256 146 ' +
      'C292 146 312 166 312 196 C312 226 292 240 256 262 C216 288 206 312 ' +
      '206 360 L206 392 L266 392 L266 366 C266 330 278 314 312 292 ' +
      'C352 266 372 236 372 194 C372 126 320 90 256 90 Z ' +
      'M206 420 L266 420 L266 480 L206 480 Z"/>',
  };

  function categoryOf(icon, domain) {
    // 编队图标传进来的可能已经是**大类键**（composition() 的输出），此时原样返回；
    // 否则按目标图标键映射（CATEGORY_OF 缺项时按域兜底）。
    if (CATEGORY_LABELS[icon]) return icon;
    if (CATEGORY_OF[icon]) return CATEGORY_OF[icon];
    if (domain === 'sea') return 'generic_sea';
    if (domain === 'air') return 'generic_air';
    return 'generic_land';
  }

  function categoryLabel(cat) {
    return CATEGORY_LABELS[cat] || cat || '未知';
  }

  // ── 编队图标 ────────────────────────────────────────────────────────────
  // 44 px 图标：阵营帧 + 顶部大类小图形 + 中央大号数量；混合编队右下加"+"角标。
  // 帧形沿用单目标军标的域语义：红方 海/陆=菱形、空中=五边形；蓝方 陆=矩形、
  // 水面=圆、空中=拱。混域编队没有单一域，用中性阵营帧（红菱形/蓝矩形）。
  const CLUSTER_FRAMES = {
    'red:sea': { tag: 'polygon', attrs: 'points="32,2 62,32 32,62 2,32"' },
    'red:land': { tag: 'polygon', attrs: 'points="32,2 62,32 32,62 2,32"' },
    'red:air': { tag: 'polygon', attrs: 'points="32,2 60,24 60,58 4,58 4,24"' },
    'red:any': { tag: 'polygon', attrs: 'points="32,2 62,32 32,62 2,32"' },
    'blue:sea': { tag: 'circle', attrs: 'cx="32" cy="32" r="28"' },
    'blue:land': { tag: 'rect', attrs: 'x="3" y="9" width="58" height="46" rx="4"' },
    'blue:air': { tag: 'path', attrs: 'd="M4 57 L4 30 Q4 5 32 5 Q60 5 60 30 L60 57 Z"' },
    'blue:any': { tag: 'rect', attrs: 'x="3" y="9" width="58" height="46" rx="4"' },
  };

  function clusterFrameOf(affiliation, domain) {
    var key = affiliation + ':' + (domain || 'any');
    return CLUSTER_FRAMES[key] || CLUSTER_FRAMES[(affiliation === 'red' ? 'red' : 'blue') + ':any'];
  }

  const clusterCache = new Map();

  function clusterSvg(opts) {
    opts = opts || {};
    const aff = affiliationOf(opts.affiliation);
    const frameColor = aff.color;
    const count = Math.max(2, Math.floor(opts.count || 2));
    const cat = categoryOf(opts.icon || 'unknown');
    const fp = clusterFrameOf(opts.affiliation === 'red' ? 'red' : 'blue', opts.domain);
    const parts = [];

    parts.push('<' + fp.tag + ' ' + fp.attrs + ' fill="' + frameColor +
               '" stroke="#ffffff" stroke-width="2.5"/>');

    // 大类小图形：约 18 px 宽，置于帧体上部（y=8 起，避免机头越出帧顶）。
    const glyph = CLUSTER_GLYPHS[cat] || CLUSTER_GLYPHS.unknown;
    parts.push('<g transform="translate(22,8) scale(0.036)" fill="#ffffff" ' +
               'color="#ffffff" stroke="none">' + glyph + '</g>');

    // 数量：3 位数以上缩字号并上移，避免在菱形下角越界。
    const label = count > 99 ? '99+' : String(count);
    const fs = label.length >= 3 ? 12 : (label.length === 2 ? 17 : 21);
    const ty = label.length >= 3 ? 51 : 53;
    parts.push('<text x="32" y="' + ty + '" text-anchor="middle" ' +
               'font-family="Arial, Helvetica, sans-serif" ' +
               'font-weight="bold" font-size="' + fs + '" fill="#ffffff">' +
               label + '</text>');

    if (opts.mixed && count <= 99) {
      // 混编角标：小圆 + 加号，位置在红菱形内部（圆心 46,42 只占 4 px 半径）。
      parts.push('<circle cx="46" cy="42" r="4" fill="' + frameColor +
                 '" stroke="#ffffff" stroke-width="1.4"/>');
      parts.push('<rect x="44.2" y="41.4" width="3.6" height="1.2" fill="#ffffff"/>');
      parts.push('<rect x="45.4" y="40.2" width="1.2" height="3.6" fill="#ffffff"/>');
    }

    return '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" ' +
           'viewBox="0 0 64 64">' + parts.join('') + '</svg>';
  }

  function clusterUri(opts) {
    opts = opts || {};
    const key = [opts.affiliation, opts.count, opts.icon, opts.mixed ? 1 : 0,
                 opts.domain || 'any'].join('|');
    let uri = clusterCache.get(key);
    if (!uri) {
      uri = toDataUri(clusterSvg(opts));
      clusterCache.set(key, uri);
    }
    return uri;
  }

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
    categoryOf: categoryOf,
    categoryLabel: categoryLabel,
    clusterSvg: clusterSvg,
    clusterUri: clusterUri,
  };
})(window);
