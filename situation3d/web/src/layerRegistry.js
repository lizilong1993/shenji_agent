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
   * **imagery 栈**（自下而上，数值即 `imageryLayers` 的 index 顺序）：
   *   z=10  天地图影像 `tianditu.base`（在线底图，默认开）
   *   z=11  Esri 影像（在线）`esri.online`（浏览器直连 Esri，无需代理）
   *   z=12  Esri 影像（离线）`imagery.offline`（静态瓦片树 z0-11）
   *   z=30  热点区域高清影像 `hotspot.imagery`（按相机高度挂，见 app.js syncHotspotImagery）
   *   z=40  城区高清影像 `urban.imagery`（**只到 z13**）
   *   z=45  离线高清底图 `imagery.offline_detail`（**从 z14 起**，mbtiles 由 /tiles_db 直发）
   *   z=50  建筑高度覆盖 `wsf3d.overlay`（半透明叠加）
   *   z=90  天地图注记 `tianditu.label`（注记永远最上）
   *
   * 三行底图的次序是 **天地图(10) < Esri 在线(11) < Esri 离线(12)**：离线在上，符合
   * 本项目"离线优先"的取向；两两都开时靠 z 决定谁可见，不做互斥解算。
   * （历史：`imagery.offline` 原为 z=5、垫在天地图之下当兜底 —— 结果是勾上它画面
   *   毫无变化。2026-10-16 按"Esri 图层要能作为底图加载管理"的要求改为 z=12，
   *   理由与代价写在下面那一行的注释里。）
   *
   * ⚠ 城区(40) 与 离线高倍段(45) **按 z 分段、互不重叠**：z≤13 读静态瓦片树（Esri），
   * z≥14 读瓦片库（谷歌）。若两者都覆盖 z14-17，同一片城区会被**两个 provider 各取
   * 一遍**（数据源还不同），而画面上只有上面那层可见 —— 纯浪费。分段后规则只剩一条：
   * **看 z 就知道是哪个源**。实现见 assets.js 的 `URBAN_TREE_MAX_ZOOM`。
   *
   * **其它栈**（各自独立，与 imagery 不同轴，见文件头）：
   *   三维实景/体块（primitive）→ TS几何（航迹/计划路线）→ 热点区域面 → 目标/标牌
   *   → 地名 → 目标元数据面板
   *
   * 标绘栈（`kind: 'plot'`）**单独占最高段 z=80—93**，与上面这条链不同轴、不可比：
   *   80 机动/集结 → 83 进攻 → 85 占领区域 → 87 防御/障碍 → 90 火力 → 93 侦察/空域
   * 依据：用户要求「标绘、目标相关要素的 z-index 应处于最高级别范围」。故标绘**整体高于**
   * target 栈（其最高为 50）与所有 imagery/globe 层。改前它们分散在 z=10—50，与影像、
   * 目标层交错，数值上看不出「标绘在最上层」这一意图。
   * 注：Cesium 没有跨栈的统一 z（见文件头），**这条只是数值上的表达**；跨栈遮挡仍由渲染
   * 管线（深度缓冲 / ground-vs-nonground / depthTest）决定。
   */
  // `default` 是该图层的**初始意图**。面板由本表生成之后，HTML 的 `checked`
  // 不再是默认值的来源（那些复选框已经删掉了）；用同一份声明，才不会出现
  // "HTML 里勾着、注册表里以为关着"这种两边不一致。
  //
  // **显示优先级在注册阶段确定**：影像栈的 `z` 就是固定优先级（大范围/粗分辨率
  // 在下，小范围/高分辨率在上）。运行期的挂载/卸载只改变"是否参与显示"，不改变
  // `z`，也不写回任何顺序；同一图层内部的多个实例（热点框、高倍段）由
  // `compareImagery()` 按 provider 声明的 maximumLevel 和覆盖矩形排序，
  // 同样与挂载先后无关。
  var LAYERS = [
    // ---- imagery 栈：序号即 imageryLayers 的 index，越大越靠上 ----
    // 【2026-10-16 改】离线 Esri 底图由 z=5 提到 **z=12**，并改名「Esri 影像（离线）」。
    //
    // 原先它 z=5、**垫在天地图（z=10）之下**，定位是"兜底层"：天地图有瓦片的地方
    // 盖住它，没瓦片的地方露出它，避免 key 失效时看到球体底色。那个设计本身没错，
    // 但用户的实际观感是——**勾上「影像底图（离线）」画面没有任何变化**，因为
    // 天地图在线底图把它整个盖住了。用户的要求是"Esri 图层也要能作为底图加载并管理"，
    // 所以这里把语义从"兜底层"改为"**可独立加载的底图**"：开它、关天地图即切换。
    //
    // 代价（明说）：Esri 离线层打开时会**盖住天地图影像**；要看天地图就关掉它。
    // 保留了原设计真正想防的那件事——它仍在**天地图之上**却默认关闭，所以默认视图
    // 与以前一致（天地图在线影像）；只有用户主动打开它时才改变。
    //
    // 互斥规则仍然不用：两边都开时靠 z 决定谁在上，不判"谁不可见"。
    //
    // 三行底图的相对次序：天地图(10) < Esri 在线(11) < Esri 离线(12) —— 离线优先，
    // 与本项目"离线兜底"的取向一致。
    {
      id: 'imagery.offline', kind: 'imagery', z: 12, default: false,
      label: 'Esri 影像（离线）', group: '底图', checkbox: 'modelImagery',
      exclusives: [],
    },
    {
      id: 'esri.online', kind: 'imagery', z: 11, default: false,
      label: 'Esri 影像（在线）', group: '底图', checkbox: 'modelEsriOnline',
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
    // 离线底图的**高倍段**（z14+），矩形来自 manifest 的逐层实测覆盖（见
    // assets.js 的 attachOfflineDetail）。它比 `imagery.offline`（z=5）高、
    // 比 wsf3d 覆盖层（z=50）低：
    //
    //   * 放在 z=5 是不行的——那一层在天地图（z=10）**下面**，是"断网兜底"的
    //     定位；新烘焙的 z14-z20 是要被看见的高清影像，而且城区层（z=40）
    //     会把它盖住；
    //   * 放在 z=50 之上也不行——会盖住半透明的建筑高度覆盖层与注记的层次关系。
    {
      id: 'imagery.offline_detail', kind: 'imagery', z: 45, default: true,
      label: '离线高清底图（z14+）', group: '底图', checkbox: 'modelOfflineDetail',
      exclusives: [],
    },
    {
      id: 'wsf3d.overlay', kind: 'imagery', z: 50, default: false,
      label: '建筑高度图层（WSF3D）', group: '底图', checkbox: 'modelWsf3dOverlay',
    },
    // 注记是栅格影像，必须在**所有**底图/高清影像之上——否则就会被盖住。
    // 以前靠一次 raiseToTop 抢救，现在由 z 序号直接决定，不再需要事后调整。
    {
      id: 'tianditu.label', kind: 'imagery', z: 90, default: true,
      // 有天地图注记时就不需要离线地名 —— 两套注记叠在一起只会互相压字。
      //
      // ⚠ 这条规则原先**不在这里**，而是 `app.js` 里的一段硬编码：apply 之前直接把
      // `places.labels` 的 `desired` 改成 false。那是个**单向开关** —— 用户后来把
      // 天地图注记关掉，离线地名**再也回不来**（没有任何路径会把它重新打开）。
      // 改成注册表的互斥之后：`desired` 是用户意图、不被改写，判负的一方只被置为
      // 不可见，赢家一关它就自动回来；面板上还会显示"被顶掉"以及原因。
      exclusives: ['places.labels'],
      label: '天地图注记', group: '底图', checkbox: 'modelTiandituLabel',
    },

    // ---- 地形 / 气象 ----
    //
    // ⚠ 这两行是**补回来的**：它们原先只是 HTML 里的复选框，在把「底图」组迁进
    // 生成式面板时**被漏掉了** —— 于是节点没了、注册表里也没有对应行，
    // 两个开关从界面上彻底消失（地形模型、气象模型都点不到了）。
    // 迁移时"照抄注册表已有的行"这种做法，天生看不见那些**只存在于 HTML** 的开关。
    //
    // `checkbox` 字段仍然写上：URL 钩子（`?terrain=` / `?weather=`）靠它把
    // 复选框 id 反查成图层 id（见 app.js 的 layerIdForCheckbox）。
    {
      id: 'terrain.model', kind: 'globe', z: 10, label: '地形模型',
      group: '底图', default: false, checkbox: 'modelTerrain',
      // 与 20m 网格互斥：Cesium 只有一个 terrainProvider，两者同时开时后动的赢
      // （见 resolve() 的对称仲裁）。对称互斥在另一边也有声明。
      exclusives: ['terrain.mesh'] },
    {
      // 2026-10-19 上传的台湾 20m DEM（quantized-mesh，z0–z14，zip 直发）。与
      // 上面的 Terrarium 高程度图是**并列的两个地形源**，不是替换：前者全球
      // 低层 + 台海区域 z14，后者只在台湾区域有数据但精度是 20 m 网格。
      id: 'terrain.mesh', kind: 'globe', z: 11, label: '地形模型（台湾 20m 网格）',
      group: '底图', default: false, checkbox: 'modelTerrainMesh',
      // 自动定位用的空间范围 [W,S,E,N]（度）。来源：tiles3d_db/tw_dem_20m.zip
      // 的 layer.json available（z14: x 27308–27489, y 10184–10494，EPSG:4326 2× 根瓦片）。
      bbox: [120.015, 21.885, 122.014, 25.302],
      exclusives: ['terrain.model'] },
    { id: 'weather.model', kind: 'globe', z: 20, label: '气象模型',
      group: '底图', default: false, checkbox: 'modelWeather' },
    // ---- scene primitive 栈：mesh 类，两两互斥（同一块地面不能同时铺两套网格）
    {
      // 【2026-10-16 改】默认由 **true → false**。
      // 原因（实测）：这一层挂载成功后会**把地球关掉**（`globe.show=false`，为避免在线
      // 实景与本机 DEM 几何冲突）。而「天地图影像」「天地图注记」「Esri 影像」等都是
      // **画在地球上的栅格图层** —— 地球一关，它们全都消失。默认打开它，等于默认状态
      // 下没有影像、没有注记，用户看到的现象就是"注记图层没反应"。
      // 关掉它、需要时再打开：栅格底图与注记照常工作；关闭/失败时 globe 会由
      // `buildings3d.js` 的 `syncGlobeVisibility()` 恢复（同批改动）。
      // 想恢复"默认开"就把这一行改回 true —— 但请一并接受"默认看不到天地图与注记"。
      id: 'tiles.google3d', kind: 'primitive', z: 10, default: false,
      label: '在线三维实景建筑（Google 3D Tiles）', group: '底图',
      checkbox: 'modelBuildingsOnline',
      // 实测：Google 实景瓦片与本机离线 DEM 几何冲突，启用时必须关掉 globe。
      //
      // ⚠ 这里原先写的是 `requires: { globeHidden: true }` —— **那是一句空话**：
      // `resolve()` 只处理 `exclusives`，全仓**没有任何代码读 `requires`**（它只出现过
      // 这唯一一次）。真正关掉 globe 的地方在 `buildings3d.js`：
      // 挂载时 `syncGlobeVisibility()` 让 globe 让位（因为 tileset 正在显示）；
      // 取消勾选 / 加载失败 / 卸载时它再把 globe 还回来。
      //
      // 之所以不改成"在 resolve() 里实现它"：`resolve()` 算的是**可见性**，
      // 而 `globeHidden` 是**场景副作用**，还带保存/恢复语义 —— 放进去反而更绕。
      // 之所以要删掉那行：**一条不生效的声明比没有声明更危险** —— 将来有人以为注册表
      // 管着它，把 `buildings3d.js` 里的两行删了，冲突就会悄悄回来。
      exclusives: ['buildings.blocks', 'tiles3d.tw_coastal'],
    },
    {
      id: 'buildings.blocks', kind: 'primitive', z: 30, default: false,
      label: '建筑体块（三维，本机离线）', group: '底图',
      checkbox: 'modelBuildings',
      exclusives: ['tiles.google3d', 'tiles3d.tw_coastal'],
    },
    {
      // 2026-10-20 新增：台湾沿海 15 个 AOI 的 LOD1 白膜（OSM 轮廓 + GBA 高度，
      // 114,785 栋，b3dm 瓦片）。数据为 tiles3d_db/tw_coastal.zip，由 serve.py 直发。
      // 默认关：15 套子 tileset 的解析与根瓦片加载不应拖慢首屏。
      // bbox 取 15 AOI 外包范围（含澎湖）；打开时相机飞到台湾全岛概览。
      id: 'tiles3d.tw_coastal', kind: 'primitive', z: 40, default: false,
      label: '台湾沿海城市建筑白膜', group: '底图', checkbox: 'modelTwCoastal',
      bbox: [119.520, 22.320, 121.910, 25.200],
      exclusives: ['tiles.google3d', 'buildings.blocks'],
    },


    // ---- 标签栈：不贴地，靠 disableDepthTestDistance 保证在几何之上 ----
    {
      // `default: false` —— 默认**不开**离线地名，因为天地图注记默认开着，两者互斥。
      // 这一点必须靠 default 表达，不能靠"默认 true 再在运行时把它改掉"：
      // 互斥的仲裁规则是"**后动的一方赢**"，而启动时 `setDesired` 按 LAYERS 顺序
      // 依次打 stamp —— 于是"谁在数组里靠后"就决定了谁赢。`places.labels` 排在
      // `tianditu.label` 之后，若它 default 也是 true，**默认胜出的会是离线地名**，
      // 把原本"天地图注记优先"的意图整个反过来（实测就是这样）。
      id: 'places.labels', kind: 'label', z: 20, default: false,
      // 与天地图注记**对称**互斥（两边互相列进 exclusives）。对称是必须的：
      // 注册表的仲裁规则是"后动的一方赢"，只写单向的话，用户后来手动打开离线地名
      // 也不会赢 —— 而"我刚点的开关应该生效"是更强的预期。
      exclusives: ['tianditu.label'],
      label: '地名', group: '底图', checkbox: 'modelPlaces',
    },

    // ---- 标绘栈：军队标号标绘作为一类图层纳管 ----
    //
    // 标绘**不新起独立的图层通路**：它就是 kind:'plot' 的普通图层，顺序靠这里的 z、
    // 显隐靠 setDesired + plot sink、增删靠 register/list。几何与标准来源在
    // `web/src/plotting.js`（`kind` 的 sink 只是把它接到场景）。
    //
    // ⚠ 这里**不再预置固定的标绘类别图层**。图层由实际存在的标绘要素动态驱动：
    // `app.js` 的 syncPlotLayers() 在 Plotting 增删要素后，按要素的 layer 字段
    // `register` 缺失的类别、`unregister` 已经没有要素的类别。这样：
    //   · 没有要素的类别不会在图层面板里显示成一行"0"；
    //   · 新增标绘类型/类别时不需要回来改注册表；
    //   · 用户手动标绘与系统自动生成的标绘（如目标轨迹）走同一条驱动逻辑。
    // 类别名称与栈内顺序来自 `Plotting.categories()`（CATEGORIES 元数据）。
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
      group: 'TS图层', default: false },
    { id: 'layerRoutes', kind: 'situation', z: 20, label: '计划航路',
      group: 'TS图层', default: false },
    { id: 'layerOverview', kind: 'situation', z: 30, label: '全局鸟瞰',
      group: 'TS图层', default: false },
    // 方向 2：占领区域 —— **已下线**（2026-10-20 用户要求，显示与功能一并停用）。
    // 原实现：引擎逐帧下发 frames[i]["occupation"]，主视图面图元与小地图掩码由
    // app.js 的 handler 驱动。恢复时取消下面这行的注释，并恢复 app.js 里的
    // occupation handler / updateOccupationAreas 实现。
    // { id: 'situation.occupation', kind: 'plot', z: 85, label: '占领区域',
    //   group: 'TS图层', default: true },

    // 可见目标按阵营拆成两行（需求：红方目标 / 蓝方目标）；两行是对同一个
    // `state.layers.targets` 的**分面视图**，总开关语义由 app.js 的 handler
    // 取"任一行开 = 显示该类目标"实现。
    { id: 'layerTargetsRed', kind: 'target', z: 9, label: '红方目标',
      group: '目标图层', default: true },
    { id: 'layerTargetsBlue', kind: 'target', z: 11, label: '蓝方目标',
      group: '目标图层', default: true },
    // 目标标牌：开关对画面无可感知影响（标牌只在避让时隐藏），按用户要求
    // 2026-10-21 从图层面板移除，只保留内部开关。
    { id: 'layerLabels', kind: 'target', z: 20, label: '目标标牌', panel: false,
      group: '目标图层', default: true },
    // 「历史轨迹」图层同时管理历史实线与计划虚线（虚线 + 终点箭头）。
    { id: 'layerTracks', kind: 'target', z: 30, label: '历史轨迹',
      group: '目标图层', default: true },
  // B4 选中掩码：外环全图 + 内环挖空「选中目标 + 它的轨迹范围」。
  // handler 在 app.js 的映射表里（state.layers.focusMask → updateFocusMask）。
  // 2026-10-21 按用户要求从面板移除（看不出效果）；内部开关保留。
  { id: 'overlay.focusMask', kind: 'target', z: 35, label: '选中掩码', panel: false,
    group: '目标图层', default: false },

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
        subgroup: l.subgroup || null,
        panel: l.panel !== false,   // false 的行不在图层面板里显示（内部开关保留）
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
  // 登记序号：**只**用于"同 z、同分辨率、同覆盖范围"时的最终并列裁决。
  // 它不参与跨层排序（跨层顺序永远由 z 决定），所以挂载先后不会改变层级。
  var imagerySeq = 0;

  /**
   * 同层多实例的稳定顺序键 —— **只读 provider 的静态声明**，与挂载先后无关。
   *
   * 一个注册表图层可能对应多个真实 imagery 实例（热点 9 个框、城区 4 个 AOI、
   * 高倍段多个 z）。以前它们的相对顺序 = `imagerySlots[id]` 的 push 顺序 =
   * **挂载顺序**，于是"先挂 z15 后补 z14"就会把 z14（粗）排到 z15（细）上面，
   * 与"小范围高分辨率在上"相反。重挂一次，顺序还可能整个翻过来。
   *
   * 这里按注册时声明的空间属性排序：
   *   1. `maximumLevel` 小的在下（粗分辨率垫底）；
   *   2. 同层级时覆盖范围大的在下（大范围垫底，小范围高分辨率覆盖其上）；
   *   3. 再按矩形西南角、最后按登记序号，保证完全可重复。
   */
  function imageryRank(layer) {
    var p = layer && layer.imageryProvider;
    var max = (p && typeof p.maximumLevel === 'number') ? p.maximumLevel : 30;
    var r = p && p.rectangle;
    var area = 0, west = 0, south = 0;
    if (r && typeof r.west === 'number') {
      west = r.west;
      south = r.south;
      var w = Math.abs(r.east - r.west);
      var h = Math.abs(r.north - r.south);
      // 跨 180° 的矩形 east-west 为负；取补一次即可（本项目的框都不跨）。
      if (r.east < r.west) w = Math.abs(r.east + 2 * Math.PI - r.west);
      area = w * h;
    }
    return { max: max, area: area, west: west, south: south };
  }

  function compareImagery(a, b) {
    var ra = imageryRank(a), rb = imageryRank(b);
    if (ra.max !== rb.max) return ra.max - rb.max;       // 粗的在下
    if (ra.area !== rb.area) return rb.area - ra.area;   // 大范围在下
    if (ra.west !== rb.west) return ra.west - rb.west;
    if (ra.south !== rb.south) return ra.south - rb.south;
    return (a._l3dSeq || 0) - (b._l3dSeq || 0);
  }
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
    if (imagerySlots[id].indexOf(layer) < 0) {
      if (!layer._l3dSeq) layer._l3dSeq = ++imagerySeq;
      imagerySlots[id].push(layer);
    }
    return layer;
  }

  /** 整组注销（例如城区层因相机高度被卸载）。 */
  function dropImagery(id) {
    delete imagerySlots[id];
  }

  /**
   * 某槽位当前挂载的所有层，**按稳定顺序**返回（粗/大范围在前，细/小范围在后）。
   *
   * 返回副本：调用方（reorderImagery、面板）不应通过它改内部数组；排序本身
   * 也不写回 `imagerySlots`，保证"动态挂卸只影响可见性，不改变登记顺序"。
   */
  function layersOf(id) {
    var arr = imagerySlots[id] || [];
    if (arr.length < 2) return arr.slice();
    return arr.slice().sort(compareImagery);
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

    // 自检：锚定完成后，已登记影像层必须**正好**占据 0..n-1。若不是，说明声明的 z
    // 顺序与实际栈顺序不一致 —— 面板上的层级关系就是错的，而画面**不会报错**
    // （只表现为"某个东西莫名被盖住"）。这条把"看不出来的错"变成一条可见告警。
    var ord = out.imagery && out.imagery.order;
    if (ord && ord.length) {
      out.imagery.consistent = ord.every(function (v, i) { return v === i; });
      if (!out.imagery.consistent) {
        out.skipped.push('imagery 栈顺序与注册表 z 不一致（期望 0..' + (ord.length - 1)
                         + '，实际 ' + ord.join(',') + '）');
      }
    }
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
        layersOf(id).forEach(function (layer, n) {
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
    // 同层多实例的稳定排序（纯函数，tools/imagery_order_test.js 直接测）。
    _compareImagery: compareImagery,
  };
})();