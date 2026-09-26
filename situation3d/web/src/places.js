/* situation3d — 地名图层
 *
 * 数据来源与离线存储
 * ------------------
 * **Natural Earth 1:10m Populated Places**，由 `tools/fetch_places.py` 抓取并转换，
 * 落盘在 `web/tiles/places/places.json`（约 926 KB，gzip 后约 250 KB，随静态服务
 * 一起走 HTTP 缓存）。
 *
 * 选它的三个理由：**公有领域**（无需署名，不引入授权风险）；自带 `SCALERANK`
 * 0–10 的重要性分级，正是分级显示需要的 LOD 提示；含中文字段，省去翻译。
 *
 * 渲染与交互效果
 * --------------
 * 用 `Cesium.LabelCollection` 而不是 `viewer.entities`：7000 多个地名如果都建成
 * Entity，光是实体管理开销就压垮帧率。这里维护一个**固定大小的标签池**，每次
 * 更新只把"该显示的那几十个"重新绑定到池对象上——不新建、不销毁。
 *
 * 两级筛选：
 *
 *  1. **按相机高度分档**（`LOD` 表）：拉远时只留 rank 0–1 的大城市（68 个），
 *     压低后才逐级放开到 rank 8（约 6800 个）。
 *  2. **屏幕空间贪心去重**：按重要性排序后依次放置，与已放置标签的屏幕包围盒
 *     重叠的跳过。这一步是必需的——按 rank 筛选后，长三角/珠三角这种密集区仍会
 *     糊成一团。
 *
 * 缩放/平移时的更新策略
 * --------------------
 * 挂在 `camera.changed` 上，**节流 250 ms**，并且只有"相机高度跨过 LOD 档位"或
 * "移动超过阈值"时才重算；静止时完全不工作。投影上千个点到屏幕坐标是这里唯一的
 * 重活，所以不放在每帧的 preRender 里。
 */

'use strict';

(function (global) {
  const DATA_URL = './tiles/places/places.json';

  // 标签池上限。屏幕就那么大，超过这个数的标签必然互相重叠，多绑只是浪费。
  const MAX_LABELS = 260;

  // 相机高度 → 允许显示的最大 scalerank。高度单位米。
  // 每档的条目数（全球）：rank≤1 68 个、≤2 186、≤3 522、≤4 1128、
  // ≤6 2443、≤7 5525、≤8 6768。
  //
  // **档位必须从低到高排列**（越近放得越宽）。原先是从高到低：
  //   { maxHeight: 6000000, rank: 1 }, { 3000000, 2 }, ...
  // 而 rankForHeight 是"第一个满足 h <= maxHeight 的就返回"，于是 6000 km 那条
  // 永远先命中，**任何相机高度都返回 rank 1**——30 km 也不例外。
  //
  // 实测后果：屏幕放到市级/街道级时，候选集仍然只有全球那 68 个 rank-1 地名
  // （对马六甲/台海一带大约是"台北、上海、香港"这几个），县城乡镇一概不出现，
  // 而且因为只剩这几个名字可用，它们反复占据屏幕、大量互相重叠（实测 30 km 时
  // 候选 68、放下 18、重叠 18）。反过来在 5000 km 全球视角又放开到 rank 1，
  // 数字上碰巧是对的，所以只看全球视角发现不了。
  //
  // 现在按"越近越宽"排：h 先命中最低的那条，得到该距离下合理的信息量。
  const LOD = [
    { maxHeight: 60000, rank: 8 },
    { maxHeight: 100000, rank: 7 },
    { maxHeight: 300000, rank: 6 },
    { maxHeight: 700000, rank: 4 },
    { maxHeight: 1500000, rank: 3 },
    { maxHeight: 3000000, rank: 2 },
    { maxHeight: Infinity, rank: 1 },
  ];

  // 屏幕包围盒的估算：中文 13px 字号下，一个汉字约 13px 宽，英文约 7px。
  const CHAR_PX_CJK = 13;
  const CHAR_PX_LATIN = 7;
  const LINE_H_PX = 17;
  const PAD_PX = 2;

  /**
   * 每个 scalerank 档位的**标签预算**。数组下标即 rank。
   *
   * 为什么要有每档配额
   * ------------------
   * 之前是"按 rank 截断候选集 → 单遍贪心铺满 260 个槽"。问题是候选集里低等级的
   * 地名数量远多于高等级（rank≤1 全球 68 个，rank≤4 就有 1128 个，rank≤8 上万个），
   * 于是一旦相机压低、候选集放开，**大量村镇会先把 260 个槽吃干净**，而"北京/上海/
   * 台北"这类最该出现的名字可能在第 261 个之后，直接一屏都看不到——标签图层看起来
   * 就是"满屏都是小地方、没有大城市"。
   *
   * 配额把预算按等级切块：高等级先拿，低等级用剩下的。这样稀疏时行为不变（各档都
   * 用不完，下面的回收 pass 会把余量还回去），密集时才体现出"优先显示重要的"。
   *
   * 数值按"1800x900 画布大约能放 260 个标签"估的：高等级少而精，低等级多而密。
   */
  const RANK_BUDGET = [0, 40, 36, 42, 46, 0, 40, 30, 26];

  /** 没有单独配额时的兜底每档上限。 */
  const RANK_BUDGET_DEFAULT = 30;

  const UPDATE_THROTTLE_MS = 250;

  let viewer = null;
  let all = null;            // 全部地名
  let collection = null;     // LabelCollection
  let pool = [];             // 复用的 Label 对象
  let loaded = false;
  let loading = null;
  let enabled = true;
  let lastUpdateAt = 0;
  let lastKey = '';
  let updating = false;    // update() 防重入
  // 上一次 update 的账目（取证用：候选数 / 放下数 / 被裁掉数 / 重叠数 / 画布尺寸）
  let lastStats = null;
  // 上一次真正放下了几个标签。用来区分"算过但一个都放不下"和"算过且有结果"：
  // 前者需要尽快重试（数据可能还没到位），后者才允许被跳过。
  let lastPlaced = 0;
  const scratch = new Cesium.Cartesian3();

  function toWindow(scene, cartesian) {
    const fn = Cesium.SceneTransforms.worldToWindowCoordinates ||
               Cesium.SceneTransforms.wgs84ToWindowCoordinates;
    return fn ? fn.call(Cesium.SceneTransforms, scene, cartesian, scratch) : undefined;
  }

  function labelWidth(text) {
    let w = 0;
    for (let i = 0; i < text.length; i++) {
      w += text.charCodeAt(i) > 0x2e80 ? CHAR_PX_CJK : CHAR_PX_LATIN;
    }
    return w;
  }

  function rankForHeight(h) {
    for (let i = 0; i < LOD.length; i++) {
      if (h <= LOD[i].maxHeight) return LOD[i].rank;
    }
    return LOD[LOD.length - 1].rank;
  }

  function ensureCollection() {
    if (collection || !viewer) return;
    collection = viewer.scene.primitives.add(new Cesium.LabelCollection({
      scene: viewer.scene,
    }));
    for (let i = 0; i < MAX_LABELS; i++) {
      pool.push(collection.add({
        text: '',
        font: '13px "Microsoft YaHei", "PingFang SC", sans-serif',
        style: Cesium.LabelStyle.FILL_AND_OUTLINE,
        // 深色字 + 白色描边：陆地和海面都读得清
        fillColor: Cesium.Color.fromCssColorString('#2b3440'),
        outlineColor: Cesium.Color.fromCssColorString('#ffffff').withAlpha(0.92),
        outlineWidth: 2.4,
        verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
        // **不要关掉深度测试**。原来这里是 Number.POSITIVE_INFINITY（即"永远画在
        // 最前面"），后果是地名无视地球遮挡——地球背面的、被山体挡住的、被三维
        // 建筑体块遮住的标签全都会浮在最上层，看起来就是"地名浮在最上面"。
        // 去掉之后标签回归正常的深度排序。
        //
        // 但**遮挡反过来也会出问题**：热点/城区高清影像与三维建筑体块都贴在地表附近，
        // 地名一旦与它们同深度就会被压住——用户反馈"热点区域等图层会遮盖地名图层"。
        // 所以用 eyeOffset 把标签沿**视线方向**抬起一段（约 400 m 等效量）：
        //   - 它是视空间偏移，相机怎么转都朝观察者一侧，标签始终可见；
        //   - 只改绘制位置，不改锚点经纬度，贴地关系不变；
        //   - 深度排序仍然生效，地球背面/地平线以下的标签照旧不显示。
        eyeOffset: new Cesium.Cartesian3(0, 0, -400),
        scaleByDistance: new Cesium.NearFarScalar(2.0e5, 1.0, 1.2e7, 0.75),
        show: false,
      }));
    }
  }

  function load() {
    if (loaded) return Promise.resolve(all);
    if (loading) return loading;
    loading = fetch(DATA_URL, { cache: 'no-store' })
      .then(function (r) {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      })
      .then(function (payload) {
        all = (payload && payload.places) || [];
        loaded = true;
        // 数据里已按 (scalerank, -人口) 排好序，直接按下标取前 N 个即可
        return all;
      })
      .catch(function (e) {
        console.warn('[places] 地名数据加载失败：' + e.message +
          '（请先运行 tools/fetch_places.py）');
        all = [];
        loaded = true;
        return all;
      });
    return loading;
  }

  /**
   * 重算该显示哪些地名。
   *
   * 先按高度档位截断候选集（数据已排序，取前 N 个就是最重要的 N 个），再做屏幕
   * 空间贪心去重。只改池对象的属性，不增删标签。
   */
  function update() {
    if (!viewer || !enabled || updating) return;
    // 兜底建池：正常情况下 setEnabled 已经建好，但 update 可能被更早的路径调到。
    ensureCollection();
    if (!collection) return;
    updating = true;
    try {
      updateInner();
    } finally {
      // 一定要复位：否则一次异常就让地名图层永久停止更新，且毫无提示。
      updating = false;
    }
  }

  function updateInner() {
    const scene = viewer.scene;
    const canvas = scene.canvas;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (!w || !h) return;

    const camHeight = viewer.camera.positionCartographic.height;
    const maxRank = rankForHeight(camHeight);

    // 候选集：按 rank 截断（数组已排序，用一次线性扫描比 filter 省一次分配）
    const cand = [];
    for (let i = 0; i < all.length; i++) {
      if (all[i].r > maxRank) break;
      cand.push(all[i]);
    }

    const placed = [];
    let used = 0;
    let offScreen = 0;
    let overlapped = 0;
    const globe = scene.globe;
    const carto = new Cesium.Cartographic();

    /**
     * 尝试放下**一个**地名。放得下返回 true。
     *
     * 抽成函数是因为要按等级配额分多轮调用（见下方 RANK_BUDGET），
     * 而"投影 → 视口剔除 → 屏幕空间取交"这段逻辑每轮都一样，不能复制粘贴。
     */
    function placeOne(p) {
      // 锚点高度取**地形表面**，不是椭球面（高度 0）。
      //
      // 地名在真实地理里的位置是"地面上的那一点"。地形打开后（本项目的
      // CustomHeightmapTerrainProvider 会抬升地表），按椭球面高度放置的标签在
      // 山地会**陷进山体或浮在山谷上方**，垂直方向上系统性地偏一个地形高。
      //
      // **必须先把该点自己的经纬度写进 carto**：`globe.getHeight()` 不传参时返回的
      // 是**相机地面点**的地形高，不写就会把整层地名都按相机脚下的高度抬升。
      carto.longitude = Cesium.Math.toRadians(p.x);
      carto.latitude = Cesium.Math.toRadians(p.y);
      carto.height = 0;
      const gh = globe ? globe.getHeight(carto) : undefined;
      const height = (gh === undefined || gh === null) ? 0 : gh;
      // 投影用**独立对象**：scratch 是共享的，若位置与投影输出共用它，
      // 上一次的投影结果会被下一次的 fromDegrees 覆盖掉。
      const pos = Cesium.Cartesian3.fromDegrees(p.x, p.y, height);
      const win = toWindow(scene, pos);
      if (!win) { offScreen++; return false; }
      if (win.x < -40 || win.y < -20 || win.x > w + 40 || win.y > h + 20) {
        offScreen++;
        return false;
      }

      const lw = labelWidth(p.n) + PAD_PX * 2;
      const lh = LINE_H_PX + PAD_PX * 2;
      const x0 = win.x - lw / 2;
      const y1 = win.y;              // verticalOrigin=BOTTOM，标签在锚点上方
      const y0 = y1 - lh;

      let hit = false;
      for (let k = 0; k < placed.length; k++) {
        const q = placed[k];
        if (x0 < q.x1 && x0 + lw > q.x0 && y0 < q.y1 && y0 + lh > q.y0) { hit = true; break; }
      }
      if (hit) { overlapped++; return false; }
      placed.push({ x0: x0, x1: x0 + lw, y0: y0, y1: y0 + lh });

      const lbl = pool[used++];
      lbl.text = p.n;
      // 标签本身也放在地形高度上：这样即使相机极度倾斜，标签锚点也与地面同高，
      // 不会因为"贴图在地形上、标签在椭球上"而分离。
      lbl.position = pos;
      lbl.show = true;
      return true;
    }

    function budgetFor(rank) {
      const b = RANK_BUDGET[rank];
      return (typeof b === 'number' && b > 0) ? b : RANK_BUDGET_DEFAULT;
    }

    // 第 1 轮：按等级从高到低分档。每档只在**本档配额**内放，配额用完就跳到下一档。
    // 这样"最重要的先占位"，不会出现低等级地名把槽吃干净、高等级一个都放不下。
    let firstPassTotal = 0;
    const placedByRank = {};
    for (let r = 1; r <= maxRank && used < MAX_LABELS; r++) {
      const quota = budgetFor(r);
      let placedThisRank = 0;
      for (let i = 0; i < cand.length && used < MAX_LABELS; i++) {
        const p = cand[i];
        if (p.r !== r) continue;
        if (placedThisRank >= quota) break;
        if (placeOne(p)) placedThisRank++;
      }
      placedByRank[r] = (placedByRank[r] || 0) + placedThisRank;
      firstPassTotal += placedThisRank;
    }

    // 第 2 轮：回收。各档配额是"上限"不是"预留"——稀疏地区高等级地名很少，
    // 余量不能浪费。这里再按等级顺序补一遍，把剩余的槽用掉。
    // 密集地区这一轮几乎什么都不放（槽已在第 1 轮用完），稀疏地区则与旧行为一致。
    if (used < MAX_LABELS) {
      for (let r = 1; r <= maxRank && used < MAX_LABELS; r++) {
        for (let i = 0; i < cand.length && used < MAX_LABELS; i++) {
          const p = cand[i];
          if (p.r !== r) continue;
          if (placeOne(p)) placedByRank[r] = (placedByRank[r] || 0) + 1;
        }
      }
    }

    for (let i = used; i < pool.length; i++) pool[i].show = false;
    lastPlaced = used;
    lastStats = { candidates: cand.length, used: used, offScreen: offScreen,
                  overlapped: overlapped, canvas: [w, h],
                  // 首轮（受配额约束）放下的数量。若它明显小于 used，说明这一屏
                  // 主要靠回收轮填满，即高等级地名本身就不够多。
                  quotaPass: firstPassTotal,
                  placedByRank: placedByRank,
                  maxRank: maxRank };
  }

  /**
   * 相机变动后调用。节流，并在"这一帧的标签布局与上一帧相同"时直接跳过。
   *
   * **key 必须包含相机朝向（heading / pitch），不只是位置与高度。**
   * 标签的屏幕位置取决于完整的相机姿态：绕目标原地旋转（位置一米没动）时，地面
   * 上每个点的屏幕坐标都会大幅平移。早期版本只把"位置 + 高度档位"放进 key，
   * 于是旋转、俯仰、倾斜这些**不改变位置的**操作全部命中"key 没变 → 跳过重算"，
   * 标签就留在上一帧的屏幕位置上不动——看起来就是"地图在转、地名钉在屏幕上"。
   *
   * 用世界坐标无法发现这个问题：标签本来就锚在真实经纬度上，旋转时世界坐标
   * 本来就不该变。**屏幕坐标才是"显示在哪儿"**。
   */
  function onCameraChanged(force) {
    if (!enabled || !loaded) return;
    if (updating) return;   // 防重入：update() 里会做上千次投影，不能被打断
    const now = Date.now();

    const c = viewer.camera.positionCartographic;
    const rank = rankForHeight(c.height);
    const key = rank + '|' +
      Math.round(c.longitude * 200) + ',' + Math.round(c.latitude * 200) + ',' +
      Math.round(c.height / 100) + '|' +
      Math.round(viewer.camera.heading * 114.6) + ',' +
      Math.round(viewer.camera.pitch * 114.6);
    if (!force && key === lastKey) return;

    lastUpdateAt = now;
    lastKey = key;
    update();
  }

  function setEnabled(on) {
    enabled = !!on;
    if (!viewer) return;
    if (!enabled) {
      if (collection) collection.show = false;
      return;
    }
    load().then(function () {
      if (!viewer) return;
      ensureCollection();
      if (collection) collection.show = true;
      onCameraChanged(true);
      // 兜底重试：首次 update 可能发生在相机/地形还没就位时（一个标签都放不下），
      // 而那种情况下往往不会再有第二次 camera.changed。这里补两次延迟重试。
      setTimeout(function () { if (enabled && lastPlaced === 0) onCameraChanged(true); }, 800);
      setTimeout(function () { if (enabled && lastPlaced === 0) onCameraChanged(true); }, 2500);
    });
  }

  function init(v) {
    viewer = v;
  }

  function debugInfo() {
    if (!loaded) return 'not loaded';
    return 'total ' + (all ? all.length : 0) + '  shown ' +
      (pool.filter(function (l) { return l.show; }).length) + '  enabled ' + enabled;
  }

  /**
   * 取证用：把"为什么一个地名都没显示"的每个短路条件逐条报出来。
   *
   * 地名这条链路有五六处可以静默返回（数据没到、开关没开、池没建、画布尺寸为 0、
   * 相机看着别处、投影落在屏幕外、被去重挡掉），只盯着"页面上没有字"是查不出来的。
   */
  function debugPlaces() {
    const canvas = viewer && viewer.scene ? viewer.scene.canvas : null;
    let camHeight = null;
    if (viewer) camHeight = viewer.camera.positionCartographic.height;
    const maxRank = camHeight === null ? null : rankForHeight(camHeight);
    let candCount = 0;
    if (all) {
      for (let i = 0; i < all.length; i++) {
        if (all[i].r > maxRank) break;
        candCount++;
      }
    }
    let sample = null;
    if (all && all.length) {
      const p = all[0];
      const pos = Cesium.Cartesian3.fromDegrees(p.x, p.y, 0);
      const win = toWindow(viewer.scene, pos);
      sample = { name: p.n, lon: p.x, lat: p.y, rank: p.r, onScreen: !!win,
                 win: win ? [Math.round(win.x), Math.round(win.y)] : null };
    }
    return {
      loaded: loaded, total: all ? all.length : 0, enabled: enabled,
      hasViewer: !!viewer, collection: !!collection,
      poolSize: pool.length,
      shown: pool.filter(function (l) { return l.show; }).length,
      lastPlaced: lastPlaced, lastKey: lastKey, lastStats: lastStats,
      updating: updating,
      canvas: canvas ? [canvas.clientWidth, canvas.clientHeight] : null,
      camHeightKm: camHeight === null ? null : Math.round(camHeight / 1000),
      maxRank: maxRank, candidates: candCount,
      firstPlaceSample: sample,
      cameraLonLat: viewer ? [
        +Cesium.Math.toDegrees(viewer.camera.positionCartographic.longitude).toFixed(2),
        +Cesium.Math.toDegrees(viewer.camera.positionCartographic.latitude).toFixed(2),
      ] : null,
    };
  }

  global.Places = {
    init: init,
    setEnabled: setEnabled,
    onCameraChanged: onCameraChanged,
    debugInfo: debugInfo,
    debugPlaces: debugPlaces,
    forceUpdate: function () { update(); },
    load: load,
  };
})(window);
