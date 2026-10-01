/**
 * 态势标绘：把「中式军队标号为主、美军标补齐」的标绘图形做成**图层管理模块里的一类图层**。
 *
 * 三条设计约束（来自项目负责人）
 * ------------------------------
 * 1. **中国军标为基准**：能对上的图形一律按中式标号；中式未覆盖或缺失的，用美军标
 *    （FM 1-02.2 / APP-06）补齐。
 * 2. **每类图形必须可溯源**：`CATALOGUE` 里每一项都带 `std` 字段（`'CN'` / `'US'`）
 *    与 `ref`（具体依据），`Plotting.catalogue()` 可整体导出——这是为了后续替换或收敛
 *    时能一眼看出"哪些是美式补齐的"，而不是散在代码注释里。
 * 3. **纳入既有图层管理**：标绘不是独立通路。每个标绘类别就是 `LayerRegistry` 里的
 *    一个 `kind:'plot'` 图层，顺序靠 `z`、显隐靠 `setDesired` + 本模块的
 *    `setLayerVisible` sink、增删复用 `register`/`list`。
 *
 * 为什么标绘不是"图片"
 * -------------------
 * 单位符号是**点状**的，可以离线成 SVG；标绘（突击方向、钳击、集结地域、火力扇面…）是
 * **多点图形**，形状由控制点算出来。所以这里没有一张图片资源：
 *
 *     控制点 → 几何生成函数 → Cesium 原生几何（多边形/折线） → 场景
 *
 * 这也意味着"增加一类标绘"= 增加一个几何函数，不是增加一个素材文件。
 *
 * 几何算法族
 * ----------
 * 标绘类型很多，但构造方式收敛到少数几族。新增类型时先归类，再复用该族的构造：
 *
 *   arrow  双/多控制点箭头（燕尾箭头、突击方向、钳击箭头、突破方向）
 *   zone   闭合区域（集结/待机地域、登陆场、封锁区、雷场）
 *   fan    扇形/环带（火力打击范围、雷达探测范围、预警扇区）
 *   band   双侧平行线走廊（行军走廊、空中走廊、火力封锁带）
 *   line   单/多段折线（战斗分界线、协调线、阶段线）
 *
 * 依赖：window.Cesium。加载顺序在 Cesium.js 之后、app.js 之前。
 */
(function () {
  'use strict';

  // ---------------------------------------------------------------------
  // 图形目录：每项都标注标准来源，供后续替换/收敛
  // ---------------------------------------------------------------------
  //
  // `std` 取值：
  //   'CN'  中式军队标号（基准）
  //   'US'  美军标补齐（FM 1-02.2 / APP-06），**是补齐项，将来有中式依据时应替换**
  //
  // `ref` 写清依据到什么程度：中式写类别与教材体系，美式写 FM/APP-06 的图形名。
  // 不写具体 GJB 编号——公开渠道无法核实编号，写了反而是编造。
  // ---------------------------------------------------------------------
  // 图形目录：每项标注「标准来源 + 依据 + 几何族 + 实现参考」，供替换/收敛
  // ---------------------------------------------------------------------
  //
  // 标准来源（`sourceStandard`）：
  //   'CN'     中式依据（《中国人民解放军作战标图规定》/ 中文教材体系）
  //   'US'     中式未覆盖 → 美军标补齐（FM 1-02.2 / APP-06）
  //   'CN+US'  中式定义图形与命名，几何参数/实现细节参考美式
  //
  // ⚠ **诚实性字段 `sourceConfidence`，不要删**：
  //   中式做到的是"**分类体系 + 图形名称 + 造型定性**"可溯源，
  //   做不到"**几何比例参数**"可溯源——《作战标图规定》全文未公开，
  //   中式符号图册属内部发行。所以中式项的 `params` 取自中文开源实现
  //   （xupinhui 2018 算法族）的**经验取值**，不是规范值。
  //   不写这个字段，后人会误以为参数来自规范。
  //
  // `usEquivalent.code` 是 APP-06 六位实体码，跨系统交换的唯一标识。
  var CONV_CN = '分类与命名可溯源；几何参数取自中文开源实现，非规范值';
  var CONV_US = 'FM 1-02.2 / APP-06 有完整公开规范';
  var CN_DOC = '《中国人民解放军作战标图规定》(2013-05-27 总参谋部发布、中央军委批准)';

  var CATALOGUE = {
    // ---- 进攻类 · 折线箭头族（族 A：2 控制点） ----
    assault_direction: {
      zh: '突击方向', family: 'arrow2', sourceStandard: 'CN',
      sourceDoc: CN_DOC, sourceConfidence: CONV_CN,
      algorithmFamily: 'polyline-arrow-2pt',
      cnImplRef: 'c-lei-en/cesiumPlot:arrowDraw/algorithm.ts#getAssaultDirection',
      usEquivalent: { name: 'Direction of Main Attack', code: '140602', spec: 'FM 1-02.2 + APP-06' },
      // 中文开源实测取值：tail/neck/head = 0.20/0.25/0.30，headAngle=π/4，neckAngle≈31.9°
      params: { tailWidthFactor: 0.20, neckWidthFactor: 0.25, headWidthFactor: 0.30,
                headAngle: Math.PI / 4, neckAngle: Math.PI * 0.17741 },
    },
    fine_arrow: {
      zh: '细直箭头（进攻方向）', family: 'arrow2', sourceStandard: 'CN',
      sourceDoc: CN_DOC, sourceConfidence: CONV_CN,
      algorithmFamily: 'polyline-arrow-2pt',
      cnImplRef: 'c-lei-en/cesiumPlot:arrowDraw/algorithm.ts#getFineArrow',
      usEquivalent: { name: 'Direction of Supporting Attack', code: '140603', spec: 'FM 1-02.2 + APP-06' },
      params: { tailWidthFactor: 0.15, neckWidthFactor: 0.20, headWidthFactor: 0.25,
                headAngle: Math.PI / 8.5, neckAngle: Math.PI / 13 },
    },
    straight_arrow: {
      zh: '直箭头', family: 'arrow2', sourceStandard: 'CN',
      sourceDoc: CN_DOC, sourceConfidence: CONV_CN,
      algorithmFamily: 'polyline-arrow-2pt',
      cnImplRef: 'canbaoSama/cesium-draw:drawGraph/drawStraightArrow.ts',
      usEquivalent: { name: 'Direction of Supporting Attack', code: '140603', spec: 'FM 1-02.2 + APP-06' },
      params: { tailWidthFactor: 0.15, neckWidthFactor: 0.20, headWidthFactor: 0.25,
                headAngle: Math.PI / 8.5, neckAngle: Math.PI / 13 },
    },
    // ---- 进攻类 · 箭身型箭头（族 B：≥3 控制点，半角法向偏移） ----
    attack_arrow: {
      zh: '进攻箭头', family: 'arrowN', sourceStandard: 'CN',
      sourceDoc: CN_DOC, sourceConfidence: CONV_CN,
      algorithmFamily: 'polyline-arrow-npt',
      cnImplRef: 'c-lei-en/cesiumPlot:arrowDraw/algorithm.ts#getAttackArrow',
      usEquivalent: { name: 'Axis of Advance (Supporting)', code: '151404', spec: 'FM 1-02.2 + APP-06' },
      params: { headHeightFactor: 0.18, headWidthFactor: 0.30, neckHeightFactor: 0.85,
                neckWidthFactor: 0.15, headTailFactor: 0.8, tailWidthFactor: 0.1 },
    },
    tailed_attack_arrow: {
      zh: '燕尾进攻箭头', family: 'arrowN', sourceStandard: 'CN',
      sourceDoc: CN_DOC, sourceConfidence: CONV_CN,
      algorithmFamily: 'polyline-arrow-npt',
      cnImplRef: 'c-lei-en/cesiumPlot:arrowDraw/algorithm.ts#getTailedAttackArrow',
      usEquivalent: { name: 'Main Axis of Advance', code: '151403', spec: 'FM 1-02.2 + APP-06' },
      params: { headHeightFactor: 0.18, headWidthFactor: 0.30, neckHeightFactor: 0.85,
                neckWidthFactor: 0.15, headTailFactor: 0.8, tailWidthFactor: 0.1,
                swallowTailFactor: 1 },
    },
    squad_combat: {
      zh: '分队战斗行动箭头', family: 'arrowN', sourceStandard: 'CN',
      sourceDoc: CN_DOC, sourceConfidence: CONV_CN,
      algorithmFamily: 'polyline-arrow-npt',
      cnImplRef: 'c-lei-en/cesiumPlot:arrowDraw/algorithm.ts#getSquadCombat',
      usEquivalent: { name: 'Axis of Advance', code: '151404', spec: 'FM 1-02.2 + APP-06' },
      params: { headHeightFactor: 0.18, headWidthFactor: 0.30, neckHeightFactor: 0.85,
                neckWidthFactor: 0.15, tailWidthFactor: 0.1 },
    },
    // ---- 进攻类 · 钳击（族 C） ----
    pincer: {
      zh: '钳击箭头（双箭头）', family: 'pincer', sourceStandard: 'CN',
      sourceDoc: CN_DOC, sourceConfidence: CONV_CN,
      algorithmFamily: 'double-arrow',
      cnImplRef: 'c-lei-en/cesiumPlot:arrowDraw/algorithm.ts#getDoubleArrow；renshuo/cps/Polygon/PincerAttack.ts',
      usEquivalent: { name: 'Encirclement', code: '151800', spec: 'FM 1-02.2 + APP-06' },
      params: { headHeightFactor: 0.25, headWidthFactor: 0.3,
                neckHeightFactor: 0.85, neckWidthFactor: 0.15 },
    },
    // ---- 集结/地域类（族 F：闭合区域） ----
    assembly_area: {
      zh: '集结地域', family: 'zone', sourceStandard: 'CN',
      sourceDoc: CN_DOC, sourceConfidence: CONV_CN,
      algorithmFamily: 'closed-spline-area',
      cnImplRef: 'c-lei-en/cesiumPlot:areaDraw/algorithm.ts#getGatheringPlacePositions',
      usEquivalent: { name: 'Assembly Area', code: '150200', spec: 'FM 1-02.2 + APP-06' },
      params: { splineT: 0.4 },
    },
    waiting_area: {
      zh: '待机地域', family: 'zone', sourceStandard: 'CN',
      sourceDoc: CN_DOC, sourceConfidence: CONV_CN,
      algorithmFamily: 'closed-spline-area',
      cnImplRef: '复用集结地域算法',
      usEquivalent: { name: 'Attack Position', code: '151600', spec: 'FM 1-02.2 + APP-06' },
      params: { splineT: 0.4 },
    },
    // ---- 防御/障碍类 ----
    battle_position: {
      zh: '防御阵地（战斗阵地）', family: 'zone', sourceStandard: 'CN',
      sourceDoc: CN_DOC, sourceConfidence: CONV_CN,
      algorithmFamily: 'area',
      cnImplRef: '无中文开源实现（裸多边形可复用）',
      usEquivalent: { name: 'Battle Position', code: '151200', spec: 'FM 1-02.2 + APP-06' },
      params: {},
    },
    mined_area: {
      zh: '雷场（障碍区）', family: 'zone', sourceStandard: 'CN',
      sourceDoc: CN_DOC, sourceConfidence: CONV_CN,
      algorithmFamily: 'area+repeated-glyph',
      cnImplRef: '无中文开源实现；重复图元阵列需自行实现',
      usEquivalent: { name: 'Mined Area', code: '270800', spec: 'FM 1-02.2 + APP-06' },
      params: { glyphArray: true },
    },
    // ---- 火力类（族 D：扇面） ----
    fire_range: {
      zh: '火力打击范围（扇面）', family: 'fan', sourceStandard: 'CN',
      sourceDoc: CN_DOC, sourceConfidence: CONV_CN,
      algorithmFamily: 'sector-3pt',
      cnImplRef: 'c-lei-en/cesiumPlot:areaDraw/algorithm.ts#getSectorPositions',
      usEquivalent: { name: 'Weapon/Sensor Range Fan, Sector', code: '242200', spec: 'FM 1-02.2 + APP-06' },
      // 中式需"多层不同半径环带"（最大/有效/直瞄），中文开源均无 → 这里用 rangeBands
      params: { rangeBands: [1.0], spreadDeg: 70 },
    },
    circular_range: {
      zh: '火力/探测范围（圆形）', family: 'fan', sourceStandard: 'CN',
      sourceDoc: CN_DOC, sourceConfidence: CONV_CN,
      algorithmFamily: 'sector-3pt',
      cnImplRef: 'c-lei-en/cesiumPlot:areaDraw/algorithm.ts#getSectorPositions',
      usEquivalent: { name: 'Weapon/Sensor Range Fan, Circular', code: '242100', spec: 'FM 1-02.2 + APP-06' },
      params: { spreadDeg: 360 },
    },
    // ---- 走廊类（族 G）——**中文开源完全缺失**，算法借美式 ----
    air_corridor: {
      zh: '空中走廊', family: 'band', sourceStandard: 'CN+US',
      sourceDoc: CN_DOC + '（图形与命名）；几何借 ' + 'FM 1-02.2 走廊类',
      sourceConfidence: '中式分类可溯源；**中文开源无走廊实现**，几何算法取自美式',
      algorithmFamily: 'corridor-offset',
      cnImplRef: '中文六仓库**均无走廊实现**（最接近 renshuo:MultiPartArrow 的法向偏移原语）',
      usEquivalent: { name: 'Air Corridor', code: '170100', spec: 'FM 1-02.2 + APP-06' },
      params: { width: 15000, dashed: true },
    },
    mobility_corridor: {
      zh: '机动走廊', family: 'band', sourceStandard: 'CN+US',
      sourceDoc: CN_DOC + '（图形与命名）；几何借 FM 1-02.2',
      sourceConfidence: '中南开无实现，几何取自美式',
      algorithmFamily: 'corridor-offset',
      cnImplRef: '无（复用走廊原语）',
      usEquivalent: { name: 'Mobility Corridor', code: '142100', spec: 'FM 1-02.2 + APP-06' },
      params: { width: 12000 },
    },
    // ---- 分界线类（族 H：单线 + **边界符号化**） ----
    boundary: {
      zh: '战斗分界线', family: 'line', sourceStandard: 'CN+US',
      sourceDoc: CN_DOC + '（图形与命名）；符号化细节借 FM 1-02.2',
      sourceConfidence: '中式样式（交替着色 + 垂直短划）**无公开规范**，样式取自美式实现',
      algorithmFamily: 'boundary-symbolised',
      cnImplRef: '中文六仓库**均只输出裸折线**，无边界符号化实现',
      usEquivalent: { name: 'Boundary', code: '110100', spec: 'FM 1-02.2 + APP-06' },
      // 教范构造的参数。等级代字用**中式代字**（表 2.9：J军 S师 U旅 T团 Y营 L连 P排 B班 Z组）；
      // 美式对应物是点/杠/X 图形，这里不画。断口与番号距离按要图观感取值——
      // 标准只规定"在适中位置、空以间隔、两侧等齐"，不给数值。
      params: { symbolised: true, echelon: 'T', labelLeft: '红方', labelRight: '蓝方',
                gapHalf: 2500, labelOffset: 6000, dashed: false },
    },
    coordination_line: {
      zh: '协调线', family: 'line', sourceStandard: 'CN',
      sourceDoc: CN_DOC, sourceConfidence: CONV_CN,
      algorithmFamily: 'line',
      cnImplRef: '无（裸折线 + 点划线）',
      usEquivalent: { name: 'Fire Support Coordination Line', code: '260100', spec: 'FM 1-02.2 + APP-06' },
      params: { dash: true },
    },
    phase_line: {
      zh: '阶段线', family: 'line', sourceStandard: 'CN',
      sourceDoc: CN_DOC, sourceConfidence: CONV_CN,
      algorithmFamily: 'line',
      cnImplRef: '无',
      usEquivalent: { name: 'Phase Line', code: '140300', spec: 'FM 1-02.2 + APP-06' },
      params: { dash: true },
    },
    // ---- 以下为**中式标准覆盖、但无中文开源实现参考**：一律标 CN+US ----
    recon_area: {
      zh: '侦察区域', family: 'zone', sourceStandard: 'CN',
      sourceDoc: CN_DOC, sourceConfidence: CONV_CN,
      algorithmFamily: 'area',
      cnImplRef: '无中文开源实现（裸多边形可复用）',
      usEquivalent: { name: 'Reconnaissance Area / Search Area', code: '152200', spec: 'FM 1-02.2 + APP-06' },
      params: { dash: true },
    },
    no_fly_zone: {
      zh: '禁飞区 / 空域协调区', family: 'zone', sourceStandard: 'CN',
      sourceDoc: CN_DOC, sourceConfidence: CONV_CN + '；**空域类必须带高度上下限**，中式高度层规范未公开',
      algorithmFamily: 'area+altitude-layer',
      cnImplRef: '无中文开源实现',
      usEquivalent: { name: 'Airspace Coordination Area (Irregular)', code: '240101', spec: 'FM 1-02.2 + APP-06' },
      params: { hatch: true, altLayer: [0, 10000] },
    },
    maritime_blockade: {
      zh: '海上封锁区', family: 'zone', sourceStandard: 'US',
      sourceDoc: 'FM 1-02.2 **无"海上封锁区"专门图形**；本项为自建图形',
      sourceConfidence: '**自建图形**，几何沿用 APP-06 多边形规范；非标准图形，须显式标注',
      algorithmFamily: 'area',
      cnImplRef: '无',
      usEquivalent: { name: '（无直接对应）最接近 Submarine Action Area', code: '150502', spec: 'FM 1-02.2 + APP-06' },
      params: { custom: true },
    },
  };
  // ---------------------------------------------------------------------
  // 配色：**中式约定与北约相反，这是最容易搞错的一点**
  // ---------------------------------------------------------------------
  // 依据《军事标图技法》（杨晓华，解放军出版社 1995）第二章第一节「标号颜色规定」，
  // 该书引 1981-03-19 总参《军队标号》与 1987-07-07《军队标号(增补规定)》：
  //   红  ——**标示我军**（步兵、装甲兵、骑兵、防化、空降、铁道兵、海空军、导弹部队）；
  //          炮兵(除自行火炮)、工程兵、通信兵、雷达、电子战用**黑**
  //   黑  —— 我军上述兵种 + **队号注记、说明注记、表格、图名、密级、图廓、签署、时间、
  //          指北矢标、框线一律用黑**
  //   深蓝——**标示敌军（全部）**
  //   深绿——不明国籍的舰船飞机、海战范围、水面漂浮物
  //   深黄——核化生武器及相关内容（在轮廓线内侧加衬）
  //   单色标图时：**我军队标用单线，敌军用双线**
  // 即 **蓝在解放军用法里是"敌"、红是"我"**，与 NATO 的 blue=friendly 正好相反。
  // 本文件此前按 NATO 约定写（红=敌、蓝=友），对中式标图是错的——现以中式为默认。
  var CN_COLOR = {
    friendly: '#c8102e',    // 我军：红
    enemy: '#123a6b',       // 敌军：深蓝
    unknown: '#1f5c3a',     // 不明国籍：深绿
    nbc: '#b8860b',         // 核化生：深黄
    annotation: '#101418',  // 注记/图廓/框线：黑
  };
  var NATO_COLOR = { friendly: '#1f5fbf', enemy: '#d0342c', unknown: '#6b7a8d',
                     nbc: '#b8860b', annotation: '#101418' };
  var colorStandard = 'CN';
  function palette() { return colorStandard === 'NATO' ? NATO_COLOR : CN_COLOR; }

  // 兼容旧调用点：按"红/蓝"**字面**取色（注意这**不等于**友/敌，友敌请用 palette()）。
  var COLOR = {
    get red() { return NATO_COLOR.enemy; },
    get blue() { return NATO_COLOR.friendly; },
    neutral: '#6b7a8d',
  };

  var viewer = null;
  var byId = {};              // 标绘实例 id -> 记录
  var seq = 0;
  var layerVisible = {};      // 标绘类别图层 id -> 是否可见（由 LayerRegistry 的 sink 驱动）

  function bind(v) { viewer = v || null; return !!viewer; }

  // ---------------------------------------------------------------------
  // 几何族
  // ---------------------------------------------------------------------

  // ---------------------------------------------------------------------
  // 几何基元：**中文标绘算法族**（xupinhui 2018 算法族，中文 Cesium 标绘的事实标准）
  // ---------------------------------------------------------------------
  //
  // 三个移植陷阱，照直觉改写就会错（调研已确认）：
  //   1. `getThirdPoint` 的偏移基点是**第二个参数 end**，不是 start。反直觉，但所有
  //      箭头都依赖它；按常规直觉写会得到镜像翻转的箭头。
  //   2. `getBaseLength = L^0.99` 是**经验拟合**，让单位长度随图形总长亚线性增长
  //      （短箭头相对更胖、长箭头相对更瘦，符合标绘视觉习惯）。改成 `L` 会明显变观感。
  //      这里保留指数并做成可配置。
  //   3. **必须在局部平面里算**：中文算法把经纬度当平面直角坐标用。台海（22–26°N）
  //      1° 经度 ≈ 100.6 km、1° 纬度 ≈ 110.9 km，各向异性约 10%；中印（28–35°N）更差。
  //      所以先投影到以首控制点为中心的局部 ENU 平面（**米**），算完再投回经纬度。
  //
  // 参数取自中文开源实现的实测常量，**不是规范值**——见 CATALOGUE 的 sourceConfidence。

  var FITTING_COUNT = 30;          // 每段曲线采样数。中文库用 100；30 在视觉上已足够且顶点数降 70%
  var BASE_LEN_EXP = 0.99;         // getBaseLength 的经验指数，保留可配置

  var DEG_LAT_M = 110540;          // 每度纬度约 110.54 km

  function metersPerDegLon(lat) {
    return 111320 * Math.cos(lat * Math.PI / 180);
  }

  /** 经纬度数组 → 以 origin 为中心的局部平面（米）。x=东, y=北。 */
  function toPlane(pts, origin) {
    var mlon = Math.max(1e-6, metersPerDegLon(origin[1]));
    return pts.map(function (c) {
      return { x: (c[0] - origin[0]) * mlon, y: (c[1] - origin[1]) * DEG_LAT_M };
    });
  }

  /** 局部平面（米）→ 经纬度。 */
  function fromPlane(pts, origin) {
    var mlon = Math.max(1e-6, metersPerDegLon(origin[1]));
    return pts.map(function (p) {
      return [origin[0] + p.x / mlon, origin[1] + p.y / DEG_LAT_M];
    });
  }

  function distance(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }

  function wholeDistance(pts) {
    var s = 0;
    for (var i = 1; i < pts.length; i++) s += distance(pts[i - 1], pts[i]);
    return s;
  }

  /** 经验尺度：亚线性，见文件头陷阱 2。 */
  function getBaseLength(pts) {
    return Math.pow(Math.max(1e-6, wholeDistance(pts)), BASE_LEN_EXP);
  }

  function mid(a, b) { return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }; }

  /** 中文算法族的方位角定义（**非**标准数学角，自北起顺时针）。 */
  function getAzimuth(start, end) {
    var dx = end.x - start.x, dy = end.y - start.y;
    var d = Math.hypot(dx, dy);
    if (d < 1e-9) return 0;
    var angle = Math.asin(Math.abs(dy) / d);
    if (dy >= 0 && dx >= 0) return angle + Math.PI;
    if (dy >= 0 && dx < 0) return Math.PI * 2 - angle;
    if (dy < 0 && dx < 0) return angle;
    return Math.PI - angle;
  }

  function getAngleOfThreePoints(a, b, c) {
    var ang = getAzimuth(b, a) - getAzimuth(b, c);
    return ang < 0 ? ang + Math.PI * 2 : ang;
  }

  function isClockWise(p1, p2, p3) {
    return (p3.y - p1.y) * (p2.x - p1.x) > (p2.y - p1.y) * (p3.x - p1.x);
  }

  /**
   * 从 **end** 出发、按方位角偏移距离取点。
   *
   * ⚠ 偏移基点是第二个参数 `end`，不是 `start`（陷阱 1）。`distance` 是**半宽**：
   * 以 ±π/2 对称调用即得 ±半宽的左右两点。
   */
  function getThirdPoint(start, end, angle, dist, clockWise) {
    var az = getAzimuth(start, end);
    var alpha = clockWise ? az + angle : az - angle;
    return { x: end.x + dist * Math.cos(alpha), y: end.y + dist * Math.sin(alpha) };
  }

  // ---------------------------------------------------------------------
  // 族 A：双控制点燕尾/细直箭头（`getFineOrAssault`）—— 7 顶点闭合环
  // ---------------------------------------------------------------------
  function arrow2Ring(p, params) {
    // 角度参数**必须给默认值**：arrowN 族的 params 里没有 headAngle/neckAngle
    // （它的头部尺寸由 headHeight/headWidth 因子决定），但 arrowN 在骨架退化时会
    // 回落到本函数。缺默认值时 `alpha = az ± undefined` → NaN 坐标，
    // 表现为箭头画不出来且**不报错**。实测正好 4 个点变 NaN（两翼的颈/头点）。
    var headAngle = params.headAngle === undefined ? Math.PI / 8.5 : params.headAngle;
    var neckAngle = params.neckAngle === undefined ? Math.PI / 13 : params.neckAngle;
    var len = getBaseLength(p);
    var tailW = len * params.tailWidthFactor;
    var neckW = len * params.neckWidthFactor;
    var headW = len * params.headWidthFactor;
    var A = p[0], B = p[p.length - 1];

    var tailLeft = getThirdPoint(B, A, Math.PI / 2, tailW, true);
    var tailRight = getThirdPoint(B, A, Math.PI / 2, tailW, false);
    var headLeft = getThirdPoint(A, B, headAngle, headW, false);
    var headRight = getThirdPoint(A, B, headAngle, headW, true);
    var neckLeft = getThirdPoint(A, B, neckAngle, neckW, false);
    var neckRight = getThirdPoint(A, B, neckAngle, neckW, true);

    return [tailLeft, neckLeft, headLeft, B, headRight, neckRight, tailRight];
  }

  // ---------------------------------------------------------------------
  // 族 B：多控制点箭身型箭头（`getAttackArrow` / `getTailedAttackArrow`）
  // ---------------------------------------------------------------------
  /** 箭头头部 5 点。headTailFactor / 末段长度两处钳制是防"头过大"。 */
  function arrowHeadPoints(p, params) {
    var len = getBaseLength(p);
    var headHeight = len * params.headHeightFactor;
    var tailWidth = distance(p[0], p[1]);      // 传进来是 tailLeft/tailRight
    if (params.headTailFactor && headHeight > tailWidth * params.headTailFactor) {
      headHeight = tailWidth * params.headTailFactor;
    }
    var headWidth = headHeight * params.headWidthFactor;
    var neckWidth = headHeight * params.neckWidthFactor;
    var n = p.length;
    var lastSeg = distance(p[n - 2], p[n - 1]);
    if (headHeight > lastSeg) headHeight = lastSeg;
    var neckHeight = headHeight * params.neckHeightFactor;

    var headEnd = getThirdPoint(p[n - 2], p[n - 1], 0, headHeight, true);
    var neckEnd = getThirdPoint(p[n - 2], p[n - 1], 0, neckHeight, true);
    var headLeft = getThirdPoint(p[n - 1], headEnd, Math.PI / 2, headWidth, false);
    var headRight = getThirdPoint(p[n - 1], headEnd, Math.PI / 2, headWidth, true);
    var neckLeft = getThirdPoint(p[n - 1], neckEnd, Math.PI / 2, neckWidth, false);
    var neckRight = getThirdPoint(p[n - 1], neckEnd, Math.PI / 2, neckWidth, true);
    return [neckLeft, headLeft, p[n - 1], headRight, neckRight];
  }

  /**
   * 箭身两侧点。**核心是那一行 `w = (...)/sin(halfAngle)`**：
   * 除以半角正弦，使偏移后的两侧边在折线拐点处交于同一条角平分线；
   * 宽度沿累计弧长从 tailWidth/2 线性收敛到 neckWidth/2。
   */
  function arrowBodyPoints(p, neckLeft, neckRight, tailWidthFactor) {
    var n = p.length;
    var allLen = wholeDistance(p);
    var len = getBaseLength(p);
    var tailWidth = len * tailWidthFactor;
    var neckWidth = distance(neckLeft, neckRight);
    var widthDif = (tailWidth - neckWidth) / 2;
    var tempLen = 0, left = [], right = [];
    for (var i = 1; i < n - 1; i++) {
      var halfAngle = getAngleOfThreePoints(p[i - 1], p[i], p[i + 1]) / 2;
      tempLen += distance(p[i - 1], p[i]);
      // 三点共线且同侧时 getAngleOfThreePoints 返回 0 → halfAngle=0 → sin=0 → 除零 → NaN。
      // 这在**用户把控制点画成一条直线**时必然发生（实测 arrowN 三个类型全部产出 NaN）。
      // 钳到一个下限即可：角度极小时按"近乎直行"处理，宽度取当前应有值。
      var sinHalf = Math.sin(halfAngle);
      if (!(Math.abs(sinHalf) > 1e-6)) sinHalf = sinHalf < 0 ? -1e-6 : 1e-6;
      var w = (tailWidth / 2 - (tempLen / allLen) * widthDif) / sinHalf;
      left.push(getThirdPoint(p[i - 1], p[i], Math.PI - halfAngle, w, true));
      right.push(getThirdPoint(p[i - 1], p[i], halfAngle, w, false));
    }
    return { left: left, right: right };
  }

  /** 二次均匀 B 样条（n=2），用于把折线箭身磨圆。 */
  function qbSpline(pts) {
    if (pts.length < 3) return pts.slice();
    var out = [pts[0]];
    var t, k, x, y, factors;
    for (var i = 1; i < pts.length - 1; i++) {
      for (var j = 0; j < FITTING_COUNT; j++) {
        t = j / FITTING_COUNT;
        x = 0; y = 0;
        var seg = [pts[i - 1], pts[i], pts[i + 1]];
        for (k = 0; k < 3; k++) {
          factors = k === 0 ? (t - 1) * (t - 1) / 2
                  : k === 1 ? (-2 * t * t + 2 * t + 1) / 2
                  : t * t / 2;
          x += factors * seg[k].x; y += factors * seg[k].y;
        }
        out.push({ x: x, y: y });
      }
    }
    out.push(pts[pts.length - 1]);
    return out;
  }

  function arrowNRing(p, params, tailed) {
    var tailLeft, tailRight;
    if (p.length >= 3 && isClockWise(p[0], p[1], p[2])) {
      tailLeft = p[1]; tailRight = p[0];
    } else if (p.length >= 2) {
      tailLeft = p[0]; tailRight = p[1];
    } else {
      return arrow2Ring(p, params);
    }
    var midTail = mid(tailLeft, tailRight);
    var bone = [midTail].concat(p.slice(2));
    // 骨架不足 3 点时退化成双控制点箭头。**A/B 必须取不同的两点**：若 tailLeft 与尾部
    // 控制点重合（两控制点正好落在同一位置），getBaseLength 得 0，宽度全为 0，
    // 环退化成一点。这里优先用"尾中点 → 末控制点"，并保证两点不重合。
    if (bone.length < 3) {
      var tip = p[p.length - 1];
      var base = (Math.abs(midTail.x - tip.x) < 1e-9 && Math.abs(midTail.y - tip.y) < 1e-9)
        ? { x: tailRight.x, y: tailRight.y } : midTail;
      return arrow2Ring([base, tip], params);
    }

    var head = arrowHeadPoints([tailLeft, tailRight].concat(bone.slice(1)), params);
    var body = arrowBodyPoints(bone, head[0], head[4], params.tailWidthFactor);
    var left = qbSpline([tailLeft].concat(body.left).concat([head[0]]));
    var right = qbSpline([tailRight].concat(body.right).concat([head[4]]));
    var ring = left.concat(head).concat(right.reverse());
    if (tailed) {
      var allLen = wholeDistance(bone);
      var swallow = getThirdPoint(bone[1], bone[0], 0,
        allLen * params.tailWidthFactor * (params.swallowTailFactor || 1), true);
      ring = ring.concat([swallow, left[0]]);
    }
    return ring;
  }

  /**
   * 族 C 钳击箭头：按 `renshuo` 那套**不构造骨架**的做法——直接放两个雪佛龙箭头
   * + 两侧样条边。优点是箭头宽度是显式属性、不受经纬度畸变影响（在平面米制里算）。
   */
  function pincerRings(p, params) {
    if (p.length < 3) return [arrow2Ring(p, params)];
    var base = p[0], tipR = p[1], tipL = p.length >= 3 ? p[2] : p[1];
    var width = params.width || 12000;

    function chevron(tip, bearing) {
      var innerDeg = 160, outerDeg = 150;
      var innerDis = (width / 2) / Math.cos((innerDeg - 90) * Math.PI / 180);
      var outerDis = width / Math.cos((outerDeg - 90) * Math.PI / 180);
      function dest(b, d) {
        var r = b * Math.PI / 180;
        return { x: tip.x + d * Math.sin(r), y: tip.y + d * Math.cos(r) };
      }
      return [dest(bearing + innerDeg, innerDis), dest(bearing + outerDeg, outerDis),
              tip, dest(bearing - outerDeg, outerDis), dest(bearing - innerDeg, innerDis)];
    }
    var brgR = getAzimuth(base, tipR) * 180 / Math.PI;
    var brgL = getAzimuth(base, tipL) * 180 / Math.PI;
    return [chevron(tipR, brgR), chevron(tipL, brgL)];
  }

  // ---------------------------------------------------------------------
  // 族 D：扇面（3 控制点：圆心 + 定半径点 + 定张角点）
  // ---------------------------------------------------------------------
  function fanRing(p, params) {
    var center = p[0];
    var radius = p.length >= 2 ? distance(p[1], center) : 50000;
    var spread = params.spreadDeg === undefined ? 60 : params.spreadDeg;
    var start;
    if (spread >= 360) { spread = 360; start = 0; }
    else { start = p.length >= 3 ? getAzimuth(p[2], center) * 180 / Math.PI : -spread / 2; }
    var steps = Math.max(8, Math.round(spread / 6));
    var ring = [center];
    for (var i = 0; i <= steps; i++) {
      var brg = (start + spread * i / steps) * Math.PI / 180;
      ring.push({ x: center.x + Math.sin(brg) * radius, y: center.y + Math.cos(brg) * radius });
    }
    return ring;
  }

  // ---------------------------------------------------------------------
  // 边界符号化（族 H 的符号化变体）
  // ---------------------------------------------------------------------
  //
  // 战斗分界线**不是一条虚线**：它按等长段交替着色（表示两侧归属），并在段的分界处
  // 向一侧打垂直短划。调研结论：**六个中文开源仓库全部只输出裸折线，没有任何一个
  // 实现符号化**，所以这一部分是自研；样式规则参考 FM 1-02.2 的 Boundaries（APP-06
  // 110100）。中式《作战标图规定》对分界线形制的规定未公开，此处同样标在
  // `sourceConfidence` 里。
  //
  // 关键工程点：**段长必须"贴合"整条线的长度**（dash fitting）——直接固定段长会在
  // 末端留一段长短不一的残段，看起来像画错了。这里先求段数 n = round(L/segLen)，
  // 再用 actual = L/n 作为实际段长，保证首尾对齐。

  /** 沿折线走 d 米，返回该点与切线方向（平面坐标）。 */
  function pointAtDist(p, d) {
    var acc = 0;
    for (var i = 1; i < p.length; i++) {
      var seg = distance(p[i - 1], p[i]);
      if (acc + seg >= d || i === p.length - 1) {
        var t = seg > 1e-9 ? (d - acc) / seg : 0;
        t = Math.max(0, Math.min(1, t));
        return {
          pt: { x: p[i - 1].x + (p[i].x - p[i - 1].x) * t,
                y: p[i - 1].y + (p[i].y - p[i - 1].y) * t },
          dir: seg > 1e-9 ? { x: (p[i].x - p[i - 1].x) / seg, y: (p[i].y - p[i - 1].y) / seg }
                          : { x: 1, y: 0 },
        };
      }
      acc += seg;
    }
    return { pt: p[p.length - 1], dir: { x: 1, y: 0 } };
  }

  /** 取折线上 [d0, d1] 区间的子折线。 */
  function subLine(p, d0, d1) {
    var out = [pointAtDist(p, d0).pt];
    var acc = 0;
    for (var i = 1; i < p.length; i++) {
      acc += distance(p[i - 1], p[i]);
      if (acc > d0 && acc < d1) out.push(p[i]);
    }
    out.push(pointAtDist(p, d1).pt);
    return out;
  }

    /**
   * 生成战斗分界线的分件。
   *
   * ⚠ **这里推翻了我自己上一版的实现，原因要记下来。**
   * 上一版按"交替色段 + 段界垂直短划"画，并把依据写成 FM 1-02.2 Boundaries。
   * 复核原始文献后发现**不成立**：美军 ADRP 1-02 表 8-1、FM 1-02.2 §5-14、
   * 中文《军事标图技法》、台湾国防部《聯合作戰符號與國軍軍隊符號之差異》——
   * 四处**没有任何一处**用交替色或垂直短划画分界线。那是自定样式，
   * 上一版的注释等于**把自己的约定冒充成了标准**。
   *
   * 教范构造（中美一致）：
   *   1. **一条线**：实线=现行，虚线=计划/待命。《军事标图技法》第二章第五节原文：
   *      「标示军队的实际配置和行动的队标用实线，标示计划（预定）、预备的配置和行动……
   *      用虚线」，并明确「**……战斗分界线等不便使用虚线的队标可用实线**」；
   *   2. 在**线中点断开一次**；
   *   3. 断口里放**部队等级**标记；
   *   4. 两侧**垂直于线**各放一个番号（番号朝北/上方）。台湾国防部文本：
   *      「標記於戰鬥地境線上之符號或註記，須在地境線上適中位置，空以間隔，註記部隊等級
   *      符號，並於等級符號兩側等齊註記部隊番號（番號須朝北或上方）」。
   *
   * **中式用"代字"而非美式点/杠/X 图形**：《军事标图技法》表 2.9 给的是文字代字
   * （`J`军 `S`师 `U`旅 `T`团 `Y`营 `L`连 `P`排 `B`班 `Z`组），所以断口里放**文字**，
   * 不必去画美式那套点杠图形——反而更简单。
   *
   * 等级标记**保持黑色不随归属着色**（参考实现的理由：分界线分隔两个建制，
   * 等级标记不属于任何一方）；线本身用归属色。
   */
  function boundaryPieces(p, params) {
    var total = wholeDistance(p);
    var midD = total / 2;
    var gapHalf = params.gapHalf || 2500;
    var at = pointAtDist(p, midD);
    var rotDeg = Math.atan2(at.dir.y, at.dir.x) * 180 / Math.PI;
    var upright = (rotDeg > 90 || rotDeg < -90) ? rotDeg + 180 : rotDeg;  // 番号保持正立
    var nx = -at.dir.y, ny = at.dir.x;
    var off = params.labelOffset || 6000;
    return {
      lines: [subLine(p, 0, Math.max(0, midD - gapHalf)),
              subLine(p, Math.min(total, midD + gapHalf), total)],
      glyph: { pt: at.pt, rotDeg: upright, text: params.echelon || 'T' },
      labels: [
        { pt: { x: at.pt.x + nx * off, y: at.pt.y + ny * off }, rotDeg: upright, text: params.labelLeft || '' },
        { pt: { x: at.pt.x - nx * off, y: at.pt.y - ny * off }, rotDeg: upright, text: params.labelRight || '' },
      ],
    };
  }

  /** 在平面里算分界线分件，再投回经纬度。 */
  function piecesFor(typeId, lonLatPts) {
    var def = CATALOGUE[typeId];
    if (!def) throw new Error('未知标绘类型：' + typeId);
    var origin = lonLatPts[0];
    var p = toPlane(lonLatPts, origin);
    var b = boundaryPieces(p, def.params || {});
    return {
      lines: b.lines.map(function (l) { return fromPlane(l, origin); }),
      glyph: { pt: fromPlane([b.glyph.pt], origin)[0], rotDeg: b.glyph.rotDeg, text: b.glyph.text },
      labels: b.labels.map(function (l) {
        return { pt: fromPlane([l.pt], origin)[0], rotDeg: l.rotDeg, text: l.text };
      }),
    };
  }
/** 在平面里按族生成环，再投回经纬度。 */
  function ringsFor(typeId, lonLatPts) {
    var def = CATALOGUE[typeId];
    if (!def) throw new Error('未知标绘类型：' + typeId);
    var params = def.params || {};
    var origin = lonLatPts[0];
    var p = toPlane(lonLatPts, origin);
    var fam = def.family, rings;
    switch (fam) {
      case 'arrow2': rings = [arrow2Ring(p, params)]; break;
      case 'arrowN': rings = [arrowNRing(p, params, typeId === 'tailed_attack_arrow')]; break;
      case 'pincer': rings = pincerRings(p, params); break;
      case 'fan': rings = [fanRing(p, params)]; break;
      case 'band': rings = [bandRing(p, params)]; break;
      case 'zone': rings = [p.slice()]; break;
      case 'line': rings = [p.slice()]; break;
      default: throw new Error('未知几何族：' + fam);
    }
    return rings.map(function (r) { return fromPlane(r, origin); });
  }

  /** 走廊/双侧平行线（族 G）：中文开源完全缺失，此处按美式走廊做法逐顶点法向偏移。 */
  function bandRing(p, params) {
    var half = (params.width || 12000) / 2;
    var left = [], right = [];
    for (var i = 0; i < p.length; i++) {
      var a = p[Math.max(0, i - 1)], c = p[Math.min(p.length - 1, i + 1)];
      var dx = c.x - a.x, dy = c.y - a.y;
      var len = Math.hypot(dx, dy) || 1e-9;
      var px = -dy / len, py = dx / len;
      left.push({ x: p[i].x + px * half, y: p[i].y + py * half });
      right.push({ x: p[i].x - px * half, y: p[i].y - py * half });
    }
    return left.concat(right.reverse());
  }
  // ---------------------------------------------------------------------
  // 场景写入
  // ---------------------------------------------------------------------

  /** 一个标绘实例 = 若干 Cesium 实体（环/线 + 可选标注）。 */
  function drawOne(rec) {
    if (!viewer) return 0;
    var def = CATALOGUE[rec.type];
    var color = Cesium.Color.fromCssColorString(rec.color || palette()[rec.hostility === 'enemy' ? 'enemy' : 'friendly']);
    var isLine = def.family === 'line';
    var n = 0;

    // 战斗分界线：**一条线 + 中点断口 + 断口里的等级代字 + 两侧垂直番号**。
    // 不是"交替色段 + 短划"（那是自定样式，见 boundaryPieces 的说明）。
    if (isLine && def.params && def.params.symbolised) {
      var pc = piecesFor(rec.type, rec.points);
      var pal = palette();
      var lineColor = Cesium.Color.fromCssColorString(pal[rec.hostility === 'enemy' ? 'enemy' : 'friendly']);
      var planned = !!def.params.dashed;
      pc.lines.forEach(function (seg) {
        if (seg.length < 2) return;
        var positions = seg.map(function (c) { return Cesium.Cartesian3.fromDegrees(c[0], c[1]); });
        rec.entities.push(viewer.entities.add({
          polyline: {
            positions: positions, width: 4, clampToGround: true,
            // 实线=现行、虚线=计划/待命（《军事标图技法》第二章第五节）
            material: planned
              ? new Cesium.PolylineDashMaterialProperty({ color: lineColor, dashLength: 16 })
              : lineColor,
          },
        }));
        n++;
      });
      // 断口里的等级代字：**黑色，不随归属着色**（它不属于任何一方）
      rec.entities.push(viewer.entities.add({
        position: Cesium.Cartesian3.fromDegrees(pc.glyph.pt[0], pc.glyph.pt[1]),
        label: {
          text: pc.glyph.text,
          font: 'bold 15px "Microsoft YaHei", sans-serif',
          fillColor: Cesium.Color.fromCssColorString(pal.annotation),
          outlineColor: Cesium.Color.WHITE, outlineWidth: 3,
          style: Cesium.LabelStyle.FILL_AND_OUTLINE,
          rotation: -pc.glyph.rotDeg * Math.PI / 180,   // Cesium 的 rotation 是逆时针
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
      }));
      n++;
      pc.labels.forEach(function (lb) {
        if (!lb.text) return;
        rec.entities.push(viewer.entities.add({
          position: Cesium.Cartesian3.fromDegrees(lb.pt[0], lb.pt[1]),
          label: {
            text: lb.text,
            font: 'bold 13px "Microsoft YaHei", sans-serif',
            fillColor: lineColor,
            outlineColor: Cesium.Color.WHITE, outlineWidth: 3,
            style: Cesium.LabelStyle.FILL_AND_OUTLINE,
            rotation: -lb.rotDeg * Math.PI / 180,
            disableDepthTestDistance: Number.POSITIVE_INFINITY,
          },
        }));
        n++;
      });
      if (rec.label) n += addLabel(rec);
      return n;
    }
    var rings = ringsFor(rec.type, rec.points);

    rings.forEach(function (ring) {
      var positions = ring.map(function (c) { return Cesium.Cartesian3.fromDegrees(c[0], c[1]); });
      if (isLine) {
        rec.entities.push(viewer.entities.add({
          polyline: {
            positions: positions,
            width: 3,
            material: new Cesium.PolylineDashMaterialProperty({
              color: color, dashLength: 16,
            }),
            clampToGround: true,
          },
        }));
      } else {
        rec.entities.push(viewer.entities.add({
          polygon: {
            hierarchy: new Cesium.PolygonHierarchy(positions),
            material: color.withAlpha(def.params && def.params.hatch ? 0.30 : 0.38),
            outline: true,
            outlineColor: color,
            outlineWidth: 2,
            classificationType: Cesium.ClassificationType.BOTH,
          },
        }));
      }
      n++;
    });

    if (rec.label) n += addLabel(rec);
    return n;
  }

  /** 控制点中点处放一个标注。抽出来是因为符号化边界分支也要用同一份样式。 */
  function addLabel(rec) {
    if (!viewer || !rec.label) return 0;
    var mid = rec.points[Math.floor(rec.points.length / 2)];
    rec.entities.push(viewer.entities.add({
      position: Cesium.Cartesian3.fromDegrees(mid[0], mid[1]),
      label: {
        text: rec.label,
        font: 'bold 14px "Microsoft YaHei", sans-serif',
        fillColor: Cesium.Color.WHITE,
        outlineColor: Cesium.Color.fromCssColorString('#0f1720'),
        outlineWidth: 3,
        style: Cesium.LabelStyle.FILL_AND_OUTLINE,
        verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
    }));
    return 1;
  }

  function clearRec(rec) {
    if (!viewer) return;
    rec.entities.forEach(function (e) {
      try { viewer.entities.remove(e); } catch (err) { /* 已随场景销毁 */ }
    });
    rec.entities.length = 0;
  }

  // ---------------------------------------------------------------------
  // 对外接口
  // ---------------------------------------------------------------------

  /**
   * 增加一个标绘。`spec = {id?, type, layer, points, color?, label?}`。
   *
   * `layer` 必须是 LayerRegistry 里 kind:'plot' 的图层 id —— 标绘实例归属某个标绘
   * 类别图层，显隐随该图层走，不自己管可见性。
   */
  function add(spec) {
    var def = CATALOGUE[spec.type];
    if (!def) throw new Error('未知标绘类型：' + spec.type);
    if (!spec.points || spec.points.length < 2) throw new Error('标绘至少需要 2 个控制点');
    var id = spec.id || ('plot-' + (++seq));
    if (byId[id]) throw new Error('标绘 id 重复：' + id);
    var rec = {
      id: id, type: spec.type, layer: spec.layer || 'plot.offense',
      std: def.std, ref: def.ref,
      points: spec.points.map(function (c) { return [c[0], c[1]]; }),
      // 颜色**不在这里定**：存归属，取色留给渲染时按当前配色标准解析，
      // 否则 setColorStandard('NATO') 之后已有标绘不会跟着变。
      color: spec.color || null,
      hostility: spec.hostility === 'enemy' ? 'enemy' : 'friendly',
      label: spec.label || null,
      entities: [],
    };
    byId[id] = rec;
    drawOne(rec);
    applyVisibility();
    return rec;
  }

  function remove(id) {
    var rec = byId[id];
    if (!rec) return false;
    clearRec(rec);
    delete byId[id];
    return true;
  }

  function clearLayer(layerId) {
    Object.keys(byId).forEach(function (id) {
      if (!layerId || byId[id].layer === layerId) remove(id);
    });
  }

  /** 由 LayerRegistry 的 plot sink 调用：切换某个标绘类别图层的显隐。 */
  function setLayerVisible(layerId, on) {
    layerVisible[layerId] = !!on;
    applyVisibility();
  }

  function applyVisibility() {
    Object.keys(byId).forEach(function (id) {
      var rec = byId[id];
      var on = layerVisible[rec.layer] !== false;   // 未声明过就默认可见
      rec.entities.forEach(function (e) { e.show = on; });
    });
  }

  /** 按图形目录整体导出「标准来源」清单，供文档/界面标注。 */
  function catalogue() {
    return Object.keys(CATALOGUE).map(function (k) {
      var d = CATALOGUE[k];
      return {
        type: k, zh: d.zh, family: d.family,
        sourceStandard: d.sourceStandard, sourceDoc: d.sourceDoc,
        sourceConfidence: d.sourceConfidence, algorithmFamily: d.algorithmFamily,
        cnImplRef: d.cnImplRef || null, usEquivalent: d.usEquivalent || null,
      };
    });
  }

  window.Plotting = {
    CATALOGUE: CATALOGUE,
    catalogue: catalogue,
    bind: bind,
    add: add,
    remove: remove,
    clearLayer: clearLayer,
    setLayerVisible: setLayerVisible,
    // 配色标准：'CN'（默认，红=我军、深蓝=敌军）或 'NATO'（蓝=友、红=敌）
    setColorStandard: function (s) {
      if (s !== 'CN' && s !== 'NATO') throw new Error('配色标准只能是 CN 或 NATO');
      colorStandard = s; return s;
    },
    colorStandard: function () { return colorStandard; },
    palette: palette,
    // 供测试直接验证几何族（不依赖 Cesium 场景）
    //
    // ⚠ 这里导出的函数名必须与上面**实际定义的名字**一致。改几何函数时最容易漏掉
    // 这个对象：引用一个已删除的名字会在**脚本加载时**抛 ReferenceError，
    // 表现为整个 `window.Plotting` 不存在（而不是某个方法报错），很难从现象倒推。
    _geometry: {
      arrow2Ring: arrow2Ring,
      arrowNRing: arrowNRing,
      pincerRings: pincerRings,
      fanRing: fanRing,
      bandRing: bandRing,
      ringsFor: ringsFor,
      toPlane: toPlane,
      fromPlane: fromPlane,
      getThirdPoint: getThirdPoint,
      getBaseLength: getBaseLength,
      getAzimuth: getAzimuth,
      boundaryPieces: boundaryPieces,
      piecesFor: piecesFor,
      pointAtDist: pointAtDist,
    },
    debug: function () {
      var list = Object.keys(byId).map(function (id) {
        return { id: id, type: byId[id].type, layer: byId[id].layer,
                 std: byId[id].std, entities: byId[id].entities.length };
      });
      return { bound: !!viewer, visible: layerVisible, count: list.length, items: list };
    },
  };
})();