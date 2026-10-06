/**
 * 图层注册表：把「有哪些图层、谁在谁上面、谁和谁互斥」变成**显式数据**。
 *
 * 为什么需要这一层
 * ----------------
 * 在这之前，图层顺序是**运行时副作用的产物**：
 *
 *   - imagery 的叠放顺序 = add 的先后（app.js 334-385 一路 add 下去）；
 *   - 天地图注记被热点/城区影像压住之后，用一次 `raiseToTop` 抢救到顶；
 *   - 「离线底图开 → 关在线底图」「在线底图开 → 关离线底图」这类互斥，
 *     是散在 setImagery / setTianditu 里的 `if (on && ...)` 手工维护。
 *
 * 于是每加一个图层都要回头检查所有已有 setter，漏一处就出现「上层图层盖住
 * 下层」——用户原话「热点区域等图层会遮盖地名图层，这是不符合逻辑的」就是
 * 这个结构的必然结果，不是某一次写漏了。
 *
 * 这里把它反过来：**先声明，再应用**。
 *
 *   - `LAYERS` 是唯一的真相来源，含每个图层的 z 序号与所在栈；
 *   - `setVisible(id, on)` 只记「意图」，不碰场景；
 *   - `apply()` 把意图与约束解算成实际顺序，再写回场景。
 *
 * 「意图」与「事实」分离是关键：`state.desired` 是用户想要的，`state.feed` 是
 * 实际能不能给（数据没到 / 加载失败 / 被覆盖）。两者分开之后，「开了开关但没
 * 东西」就能被准确报出来，而不是静默失败。
 *
 * Cesium 里没有单一的 z-index
 * ---------------------------
 * 三个互相独立的顺序轴，本文件按轴分栈处理，不要试图用一个数字跨轴比较：
 *
 *   1. **imagery 栈**：`viewer.imageryLayers`，**后面的盖住前面的**，
 *      `index` 越大越靠上。只影响栅格影像，与三维几何无关。
 *   2. **scene primitive 栈**：`viewer.scene.primitives`，按加入顺序绘制，
 *      彼此之间靠深度缓冲决定前后（不是靠 index）。
 *   3. **ground vs non-ground**：贴地图元（GroundPrimitive / 带
 *      classificationType 的）与不贴地的图元走不同管线；标签/广告牌属于
 *      不贴地的 translucent pass，默认会被地形遮挡，需要用
 *      `disableDepthTestDistance` 才能"永远在前"。
 *
 * 所以「在地名之上」这种要求，在两个栈里要用不同手段实现：imagery 用 order，
 * 标签用 depthTest。混着设是没效果的。
 *
 * 依赖：window.Cesium（仅在 apply 时才用，加载顺序上须在 Cesium.js 之后）。
 */
(function () {
  'use strict';

  /**
   * z 序号按「栈内从下到上」编号，中间留出空档以便插入新层而不必重排。
   *
   * 参考层级（自下而上）：
   *   在线底图 → 离线底图 → 热点/城区高清影像 → 建筑高度覆盖 → 三维实景/体块
   *   → TS几何（航迹/计划路线）→ 热点区域面 → 目标/标牌 → 地名 → 目标元数据面板
   */
  // `default` 是该图层的**初始意图**。面板由本表生成之后，HTML 的 `checked`
  // 不再是默认值的来源（那些复选框已经删掉了）；用同一份声明，才不会出现
  // "HTML 里勾着、注册表里以为关着"这种两边不一致。
  var LAYERS = [
    // ---- imagery 栈：序号即 imageryLayers 的 index，越大越靠上 ----
    // 离线底图**垫在天地图下面**（z=5 < z=10），两者不再互斥。
    //
    // 原先是互斥的：两边都开时"后动的一方赢"，另一方判为不可见。规则本身没问题，
    // 但前提假设错了——它假设两个底图是**二选一**。实际发生的是：天地图 key 配额
    // 耗尽后返回 429，整张底图变成一片深蓝（地球底色），而离线底图被互斥规则按住，
    // 明明有瓦片却不肯显示。
    //
    // 垫一层就没有这个单点：天地图有瓦片的地方盖住离线层；没瓦片的地方（离线、
    // key 失效、瓦片还在路上）露出离线影像，而不是露出球体底色。代价是天地图加载
    // 期间会短暂看到较粗的离线影像——这比看到空洞好。
    {
      id: 'imagery.offline', kind: 'imagery', z: 5, default: false,
      label: '影像底图（离线）', group: '底图', checkbox: 'modelImagery',
      exclusives: [],
    },
    {
      id: 'tianditu.base', kind: 'imagery', z: 10, default: true,
      label: '天地图影像', group: '底图', checkbox: 'modelTianditu',
      exclusives: [],
    },
    {
      id: 'hotspot.imagery', kind: 'imagery', z: 30, default: true,
      label: '热点区域高清影像', group: '底图', checkbox: 'modelHotspotImagery',
    },
    {
      id: 'urban.imagery', kind: 'imagery', z: 40, default: true,
      label: '城区高清影像（米级）', group: '底图', checkbox: 'modelUrbanImagery',
    },
    {
      id: 'wsf3d.overlay', kind: 'imagery', z: 50, default: false,
      label: '建筑高度图层（WSF3D）', group: '底图', checkbox: 'modelWsf3dOverlay',
    },
    // 注记是栅格影像，必须在**所有**底图/高清影像之上——否则就会被盖住。
    // 以前靠一次 raiseToTop 抢救，现在由 z 序号直接决定，不再需要事后调整。
    {
      id: 'tianditu.label', kind: 'imagery', z: 90, default: true,
      label: '天地图注记', group: '底图', checkbox: 'modelTiandituLabel',
    },

    // ---- scene primitive 栈：mesh 类，两两互斥（同一块地面不能同时铺两套网格）
    {
      id: 'tiles.google3d', kind: 'primitive', z: 10, default: true,
      label: '在线三维实景建筑（Google 3D Tiles）', group: '底图',
      checkbox: 'modelBuildingsOnline',
      // 实测：Google 实景瓦片与本机离线 DEM 几何冲突，启用时必须关掉 globe。
      requires: { globeHidden: true },
      exclusives: ['buildings.blocks'],
    },
    {
      id: 'buildings.blocks', kind: 'primitive', z: 30, default: false,
      label: '建筑体块（三维，本机离线）', group: '底图',
      checkbox: 'modelBuildings',
      exclusives: ['tiles.google3d'],
    },

    // ---- 标签栈：不贴地，靠 disableDepthTestDistance 保证在几何之上 ----
    {
      id: 'places.labels', kind: 'label', z: 20, default: true,
      label: '地名', group: '底图', checkbox: 'modelPlaces',
    },

    // ---- 标绘栈：态势标绘（军队标号）作为一类图层纳管 ----
    //
    // 标绘**不新起独立的图层通路**：它就是 kind:'plot' 的普通图层，顺序靠这里的 z、
    // 显隐靠 setDesired + plot sink、增删靠 register/list。几何与标准来源在
    // `web/src/plotting.js`（`kind` 的 sink 只是把它接到场景）。
    //
    // 按类别分层（对应中式军队标号的类别体系）：机动/集结 → 进攻 → 防御/障碍 →
    // 火力 → 侦察/空域。`z` 只在这个栈内部排序；不同栈之间不可比（见文件头）。
    //
    // 默认**开**：标绘是"开了就该看到"的东西，没有数据时自然什么都没有，不会误导。
    {
      id: 'plot.mobility', kind: 'plot', z: 10, default: true,
      label: '标绘·机动与集结', group: '标绘', checkbox: 'plotMobility',
    },
    {
      id: 'plot.offense', kind: 'plot', z: 20, default: true,
      label: '标绘·进攻', group: '标绘', checkbox: 'plotOffense',
    },
    {
      id: 'plot.defense', kind: 'plot', z: 30, default: true,
      label: '标绘·防御与障碍', group: '标绘', checkbox: 'plotDefense',
    },
    {
      id: 'plot.fire', kind: 'plot', z: 40, default: true,
      label: '标绘·火力', group: '标绘', checkbox: 'plotFire',
    },
    {
      id: 'plot.intel', kind: 'plot', z: 50, default: true,
      label: '标绘·侦察与空域', group: '标绘', checkbox: 'plotIntel',
    },
    // ---- TS图层 / 目标图层：原先只存在于 index.html，注册表里没有 ----
    //
    // 它们用**逐层 `handler`**（在 app.js 里挂上）而不是 kind sink：各自是一次性动作
    // （写 state.layers.*、setSensors、setOverview…），为每个造一个 kind 只是把同一段
    // 代码换个地方写。有了 handler 就能与其它图层**共用同一条 apply 通路**，
    // 面板也才能把它们当成普通图层行来渲染与计数。
    //
    // `default` 与迁移前 HTML 的 checked 一致（`layerOverview` 例外：它由
    // `setOverview` 在启动时按可用性决定，这里给 false 避免"声明开着但实际没挂"）。
    { id: 'layerHotspots', kind: 'situation', z: 10, label: '热点区域',
      group: 'TS图层', default: true },
    { id: 'layerRoutes', kind: 'situation', z: 20, label: '计划航路',
      group: 'TS图层', default: true },
    { id: 'layerOverview', kind: 'situation', z: 30, label: '鸟瞰图',
      group: 'TS图层', default: false },

    // ---- 地图组件 ----
    //
    // 与 TS图层/目标图层同一套：逐层 `handler`（在 app.js 里挂 `setComponent`），
    // `default: true` 与迁移前 HTML 的 checked 一致。
    { id: 'compCompass', kind: 'component', z: 10, label: '指南针',
      group: '地图组件', default: true },
    { id: 'compScalebar', kind: 'component', z: 20, label: '比例尺',
      group: '地图组件', default: true },
    { id: 'compViewpoint', kind: 'component', z: 30, label: '视点读数',
      group: '地图组件', default: true },
    { id: 'layerTargets', kind: 'target', z: 10, label: '可见目标',
      group: '目标图层', default: true },
    { id: 'layerLabels', kind: 'target', z: 20, label: '目标标牌',
      group: '目标图层', default: true },
    { id: 'layerTracks', kind: 'target', z: 30, label: '历史轨迹',
      group: '目标图层', default: true },
    { id: 'modelSensor', kind: 'target', z: 40, label: '传感器覆盖',
      group: '目标图层', default: false },
    { id: 'layerTargetPanel', kind: 'target', z: 50, label: '目标元数据面板',
      group: '目标图层', default: true },

  ];
  var byId = {};
  LAYERS.forEach(function (l) { byId[l.id] = l; });

  var registry = {};

  // ------------------------------------------------------------------------
  // 修订号与要素提供者（动态图层管理的基础）
  // ------------------------------------------------------------------------
  //
  // 面板（`layerPanel.js`）需要在"图层或要素变了"时重建 DOM。做法是**每帧比对
  // 一个签名**，签名里带修订号与各层要素数量 —— 所以这里只要能廉价地回答
  // "变过没有"，不需要事件总线。
  //
  // 修订号覆盖注册表自己的改动；要素数量覆盖**外部数据**的改动（标绘增删、案例
  // 切换、影像层随相机高度挂载/卸载）—— 后者不经过注册表，只有靠数量比对才发现，
  // 这正是签名里要带 count 的原因。

  var revSeq = 0;
  var providers = {};   // layerId -> () => [element] | null

  /** 记一次"变了"。注册表内所有改动都会调它，外部改动靠面板的数量比对兜底。 */
  function notify() { revSeq++; return revSeq; }

  function revision() { return revSeq; }

  /**
   * 注册某层的**要素提供者**。
   *
   * 提供者形如 `() => [{key, label, sub, visible, actions}]`，返回 `null` 表示该层
   * 没有要素列表（面板不显示展开箭头）。
   */
  function setProvider(layerId, fn) {
    if (fn) providers[layerId] = fn; else delete providers[layerId];
    notify();
    return fn || null;
  }

  /**
   * 取某层的要素列表。
   *
   * **必须兜住异常**：面板跑在帧循环里，一个提供者抛错就会变成每帧一次的未捕获
   * 错误，把整个渲染循环和状态栏一起污染（这个仓库已经在 `positionFor` 上踩过一次）。
   * 出错时返回空数组并把原因记进该层的 feedDetail，面板照常渲染其余行。
   */
  function elements(layerId) {
    var fn = providers[layerId];
    if (!fn) return [];
    try {
      var out = fn();
      return Array.isArray(out) ? out : [];
    } catch (e) {
      var r = registry[layerId];
      if (r) { r.feed = 'failed'; r.feedDetail = '要素枚举失败：' + e.message; }
      return [];
    }
  }

  function hasProvider(layerId) { return !!providers[layerId]; }

  /** 要素级显隐：转交给该要素描述符声明的 `onToggle`。 */
  function setElementVisible(layerId, key, on) {
    var fn = providers[layerId];
    if (!fn) return false;
    var list = elements(layerId);
    for (var i = 0; i < list.length; i++) {
      if (list[i].key === key) {
        if (typeof list[i].onToggle !== 'function') return false;
        list[i].onToggle(!!on);
        notify();
        return true;
      }
    }
    return false;
  }

  /**
   * 注册一个图层。重复 id 直接报错——静默覆盖会让"顺序看起来对但实际用了旧定义"，
   * 比直接失败难查得多。
   */
  function register(def) {
    if (byId[def.id]) throw new Error('图层 id 重复注册：' + def.id);
    byId[def.id] = def;
    LAYERS.push(def);
    registry[def.id] = newRuntime(def);
    notify();
    return def;
  }

  /**
   * 注销一个图层。
   *
   * 若该层还有**已挂载的 imagery 实例**则拒绝并抛错：那些 Cesium.ImageryLayer 已经
   * 进了 viewer.imageryLayers，悄悄把定义删掉会留下无主图层 —— 它继续渲染、继续
   * 参与重排，但面板里已经没有对应的开关，用户再也关不掉它。宁可报错让调用方先卸载。
   */
  function unregister(id) {
    var def = byId[id];
    if (!def) return null;
    if (imagerySlots[id] && imagerySlots[id].length) {
      throw new Error('图层 ' + id + ' 还有 ' + imagerySlots[id].length
        + ' 个已挂载的影像实例，请先卸载再注销');
    }
    delete byId[id];
    delete registry[id];
    delete providers[id];
    for (var i = LAYERS.length - 1; i >= 0; i--) {
      if (LAYERS[i].id === id) LAYERS.splice(i, 1);
    }
    notify();
    return def;
  }

  /** runtime 行的构造抽出来，register 与初始化共用一份，避免两处字段漂移。 */
  function newRuntime(def) {
    return {
      def: def,
      desired: false,
      feed: 'ok',
      feedDetail: null,
      stamp: 0,
    };
  }

  function list(kind) {
    return LAYERS.filter(function (l) {
      return (kind ? l.kind === kind : true);
    }).sort(function (a, b) { return a.z - b.z; });
  }

  /** 同一栈内按 z 升序；不跨栈比较（见文件头说明）。 */
  function orderOf(id) {
    var def = byId[id];
    if (!def) return -1;
    var peers = list(def.kind);
    for (var i = 0; i < peers.length; i++) {
      if (peers[i].id === id) return i;
    }
    return -1;
  }

  // 把定义同步到 runtime，runtime 里保存「意图」与「事实」。
  var seq = 0;
  // 事实（feed）的取值：
  //   ok          正常
  //   pending     数据还没到（加载中）
  //   failed      加载失败
  //   hidden      被上层互斥层顶掉了（不是出错，是设计如此）
  //   unsupported 该图层在当前部署下不可用（例如没有令牌的在线层）
  LAYERS.forEach(function (l) { registry[l.id] = newRuntime(l); });

  // -- 意图 ---------------------------------------------------------------

  /**
   * 记录「想不想显示」，**不碰场景**。实际生效由 apply() 负责。
   * 这样开关回调变成纯状态写入，互斥与顺序集中在一处解算。
   */
  function setDesired(id, on) {
    var r = registry[id];
    if (!r) return null;
    r.desired = !!on;
    r.stamp = ++seq;
    notify();
    return r;
  }

  function setFeed(id, feed, detail) {
    var r = registry[id];
    if (!r) return null;
    r.feed = feed;
    r.feedDetail = detail || null;
    return r;
  }

  /**
   * 解算实际可见性：意图 ∩ 未被互斥顶掉。
   *
   * 互斥规则：若 A 想要显示且 A 把 B 列为 exclusives，则 B 即使 desired 为真
   * 也判为 hidden。**不修改 B 的 desired**——用户关掉 A 之后 B 应当自动回来，
   * 这正是之前手工 setter 做不到、需要逐个 `if (on && ...)` 补的地方。
   */
  /**
   * 解算实际可见性：意图 ∩ 未被互斥顶掉。
   *
   * 互斥是**对称**的（天地图影像与离线影像互相列进 exclusives），所以两边都开着时
   * 必须有个明确的仲裁规则，不能靠遍历次序。规则是**后动的一方赢**：
   * 谁最后被用户改动（stamp 最大）谁留下，另一方判为 hidden。
   *
   * 为什么必须有仲裁：如果只是"互相把对方关掉"，两边都会变 false，
   * 用户点一下"离线底图"就直接黑屏——实测就是这样。
   *
   * 判负的一方**只被置为不可见，desired 不动**：用户回头把赢家关掉，输家会自动
   * 回来，不需要用户再手动勾一次。
   */
  function resolve() {
    var effective = {};
    LAYERS.forEach(function (l) { effective[l.id] = !!registry[l.id].desired; });

    // 第一轮：对称冲突仲裁，只记录谁整体让位（双方都想要时比 stamp）。
    var conceded = {};
    LAYERS.forEach(function (l) {
      if (!effective[l.id] || !l.exclusives) return;
      l.exclusives.forEach(function (otherId) {
        if (!effective[otherId] || conceded[otherId]) return;
        var mine = registry[l.id].stamp || 0;
        var theirs = registry[otherId].stamp || 0;
        if (mine >= theirs) {
          conceded[otherId] = l;
        } else {
          conceded[l.id] = byId[otherId];
        }
      });
    });

    LAYERS.forEach(function (l) {
      var r = registry[l.id];
      if (conceded[l.id]) {
        effective[l.id] = false;
        r.feed = 'hidden';
        r.feedDetail = '被「' + conceded[l.id].label + '」顶掉（两者互斥，后者优先）';
      } else {
        effective[l.id] = !!r.desired;
      }
    });
    return effective;
  }

  /**
   * 清掉因互斥临时写下的 feed，再按当前可见性重新判定。
   *
   * 必须每轮 apply 前调用：否则用户关掉互斥的上层之后，下层的 `hidden` 标记会一直
   * 留着，出现「开关明明开着、状态却显示被顶掉」的假象。
   */
  function resetHiddenFeed() {
    LAYERS.forEach(function (l) {
      var r = registry[l.id];
      if (r.feed === 'hidden') {
        r.feed = 'ok';
        r.feedDetail = null;
      }
    });
  }

  /** 每个图层的当前状态快照，供数据健康面板与取证使用。 */
  function status() {
    var effective = resolve();
    return LAYERS.map(function (l) {
      var r = registry[l.id];
      // 数量只在**有提供者**时才算；没有提供者返回 null 而不是 0 ——
      // 0 会被读成"这一层是空的"，那和"这一层没有要素列表"是两件事。
      var expandable = hasProvider(l.id);
      return {
        id: l.id, kind: l.kind, z: l.z, label: l.label, group: l.group,
        checkbox: l.checkbox || null,
        desired: r.desired,
        effective: !!effective[l.id],
        feed: r.feed,
        feedDetail: r.feedDetail,
        expandable: expandable,
        count: expandable ? elements(l.id).length : null,
      };
    });
  }

  // ------------------------------------------------------------------------
  // 应用到场景
  // ------------------------------------------------------------------------
  //
  // 栈与栈之间没有统一的 z 轴（见文件头），所以这里按 kind 分派到不同机制：
  //
  //   imagery    → 用 reorderImagery 重排 imageryLayers 的 index
  //   primitive  → 直接设 .show（mesh 之间的前后由深度缓冲决定，与 index 无关）
  //   label      → 交给 Places.setEnabled（它内部再做屏幕空间仲裁）
  //
  // 一个关键约束：imagery 的 index 是**零散可变**的，因为城区/热点层会根据相机
  // 高度动态挂载和卸载。所以不能只在启动时排一次，每次结构变化都要重排。

  var imagerySlots = {};   // id -> Cesium.ImageryLayer
  var sinks = {};          // kind -> 应用函数
  var ctxViewer = null;    // 由 bind 注入。viewer 在 app.js 里是闭包变量，拿不到全局，
                           // 所以这里不做隐式查找，必须显式交进来。

  /** 通过 viewer 反查某层在 imageryLayers 里的当前 index，找不到返回 -1。 */
  function layerIndex(layer) {
    if (!ctxViewer || !ctxViewer.imageryLayers || !layer) return -1;
    // Cesium 1.145 的 ImageryLayerCollection 自带 indexOf，不用自己扫。
    return ctxViewer.imageryLayers.indexOf(layer);
  }

  /**
   * 把 imagery 槽位按声明顺序重排。
   *
   * 为什么不用 raise / raiseToTop
   * ----------------------------
   * `raise(layer)` 只上移**一个位置**，不是移到顶（实测：注记在 index 1，调一次
   * raise 之后仍在 index 1 附近，因为后面又补了层）。而按 z 升序反复调
   * `raiseToTop` 会把顺序**整个颠倒**——最后一个被抬到顶的反而排在最上面。
   *
   * 可靠的原语是 `add(layer, index)`：它接受显式 index，且插到哪个位置就是哪个
   * 位置。所以这里按目标顺序逐个"锚定"：每个层都插到它应有的下标上，已经在位的
   * 不动（这样不会每轮都触发 Cesium 重建表面瓦片顺序）。
   */
  function reorderImagery(targetIds) {
    if (!ctxViewer || !ctxViewer.imageryLayers) return { moved: 0, skipped: 'no viewer' };
    var coll = ctxViewer.imageryLayers;

    // 摊平成**逐个真实图层**的目标顺序：同类多实例按登记顺序相邻。
    // 只处理已挂载的（城区/热点会按相机高度动态挂载卸载）。
    var mounted = [];
    (targetIds || []).forEach(function (id) {
      layersOf(id).forEach(function (layer) {
        if (layerIndex(layer) >= 0) mounted.push(layer);
      });
    });

    var moved = 0;
    for (var targetPos = 0; targetPos < mounted.length; targetPos++) {
      var layer = mounted[targetPos];
      if (coll.indexOf(layer) === targetPos) continue;   // 已到位，不动
      coll.remove(layer, false);                          // 摘下来但**不销毁**
      coll.add(layer, targetPos);
      moved++;
    }
    return { moved: moved, anchored: mounted.length, order: mounted.map(function (l) { return coll.indexOf(l); }) };
  }

  /**
   * 登记一个 imagery 槽位。
   *
   * **一层多实例的层必须把每个实例都登记进来**（热点影像一个热点框一个 provider，
   * 城区影像同理）。早先只在槽位里放第 0 个当"顺序锚点"，结果其余实例全落在
   * 锚定范围之外——它们 show=true，稳稳压在注记上面，"注记被盖住"这个 bug 其实
   * 没修掉。顺序必须覆盖**每一个真实存在的层**，不能只看每类一个代表。
   */
  function addImagery(id, layer) {
    if (!imagerySlots[id]) imagerySlots[id] = [];
    // 去重：registerImagerySlot 可能因重复挂载被调用多次。
    if (imagerySlots[id].indexOf(layer) < 0) imagerySlots[id].push(layer);
    return layer;
  }

  /** 整组注销（例如城区层因相机高度被卸载）。 */
  function dropImagery(id) {
    delete imagerySlots[id];
  }

  /** 某槽位当前挂载的所有层，已按登记顺序。 */
  function layersOf(id) {
    return imagerySlots[id] || [];
  }

  /**
   * 注入 viewer 与各栈的写回函数。
   *
   * sinks 是 `{ imagery, primitive, label }`，每个形如 `(id, want, runtime) => void`。
   * 缺哪个栈就跳过哪个（apply 会在 skipped 里报出来，不静默）。
   */
  function bind(viewer, handlers) {
    ctxViewer = viewer || null;
    sinks = handlers || {};
    return sinks;
  }

  /**
   * 把解算后的可见性写回场景。
   *
   * 幂等：可以每帧调用，但正常只在意图变化或结构变化时调用（见 renderGovernor
   * 的脏标记），不要放在帧循环里无脑跑。
   */
  function apply() {
    resetHiddenFeed();
    var effective = resolve();
    var out = { desired: 0, applied: 0, skipped: [] };

    LAYERS.forEach(function (l) {
      var r = registry[l.id];
      if (r.desired) out.desired++;

      // 分派顺序：**逐层 handler 优先**，否则按 kind 找 sink。
      //
      // 为什么要有 handler：TS图层/目标图层/地图组件那几行各有一次性动作
      // （`state.layers.x = v; markScene(...)`、`setTerrain(v)`、`setComponent(n,v)`…），
      // 为它们各造一个 kind 与 sink 只是把同一段代码换个地方写。给它们一个
      // handler 就能**和其它图层共用同一条 apply 通路**，而不是在面板之外再维护
      // 一份开关绑定 —— 那正是"第二套图层管理通路"。
      //
      // 未挂载的 imagery（城区/热点在低空才挂）也走 sink：sink 里判空跳过即可，
      // 把"没挂载"和"该隐藏"分开处理是 sink 的责任，注册表不猜。
      var write = (typeof l.handler === 'function') ? l.handler : sinks[l.kind];
      if (!write) {
        out.skipped.push(l.id + '(no handler/sink for ' + l.kind + ')');
        return;
      }
      try {
        write(l.id, !!effective[l.id], r);
        out.applied++;
      } catch (e) {
        out.skipped.push(l.id + '(' + e.message + ')');
      }
    });

    // 顺序必须在可见性之后重排：刚挂上的层还没有正确 index。
    // 只重排「已挂载」的，城区/热点未挂载时自动跳过。
    out.imagery = reorderImagery(list('imagery').map(function (l) { return l.id; }));
    return out;
  }

  window.LayerRegistry = {
    LAYERS: LAYERS,
    list: list,
    def: function (id) { return byId[id] || null; },
    orderOf: orderOf,
    register: register,
    unregister: unregister,
    setDesired: setDesired,
    // 动态图层管理：要素提供者 / 查询 / 显隐 / 修订号
    setProvider: setProvider,
    elements: elements,
    hasProvider: hasProvider,
    setElementVisible: setElementVisible,
    notify: notify,
    revision: revision,
    setFeed: setFeed,
    resolve: resolve,
    status: status,
    runtime: registry,
    apply: apply,
    bind: bind,
    reorderImagery: reorderImagery,
    addImagery: addImagery,
    dropImagery: dropImagery,
    layersOf: layersOf,
    _debug: function () {
      var rows = [];
      Object.keys(imagerySlots).forEach(function (id) {
        imagerySlots[id].forEach(function (layer, n) {
          rows.push({
            id: id + (imagerySlots[id].length > 1 ? ('#' + n) : ''),
            slot: id,
            z: byId[id] ? byId[id].z : -1,
            index: layerIndex(layer),
            show: !!layer.show,
          });
        });
      });
      rows.sort(function (a, b) { return a.index - b.index; });
      return {
        viewerBound: !!ctxViewer,
        sinks: Object.keys(sinks),
        imagery: rows,
        anchored: rows.length,
        unbound: list('imagery').map(function (l) { return l.id; })
          .filter(function (id) { return !imagerySlots[id]; }),
      };
    },
  };
})();