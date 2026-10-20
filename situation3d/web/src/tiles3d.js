/**
 * 本地三维瓦片图层（3D Tiles，glb 城市模型）。
 *
 * 为什么单独一个文件
 * ------------------
 * 与 `buildings3d.js`（在线 Google 实景，走 Cesium ion）刻意分开：那条链路有三条
 * 硬约束（必须在线、必须关地球、不能缓存），而本文件是**完全离线**的第三方导出包，
 * 约束完全不同，混在一起会让"哪条链路失败"难以分辨。
 *
 * 数据从哪来
 * ----------
 * 2026-10-19 上传的台湾城市 3D Tiles（台北 101 / 台中 / 桃园）与 TW2024 样例。
 * 导出包是几十万个小文件（glb + json），本机文件系统建小文件极慢，因此**不摊开**，
 * 由 `server/serve.py` 的 `/tiles3d/<dataset>/<路径>` 直接从 zip 中央目录取条目
 * （与 `/tiles_db/` 直发 mbtiles 同一个思路）。前端只需要相对 URL。
 *
 * 图层怎么挂
 * ----------
 * 每个数据集是一个 `kind:'primitive'` 的注册表图层，开关由 `layerPanel` 生成；
 * 本模块的 `setEnabled()` 只做"挂/卸 Cesium3DTileset"这一件事，意图与顺序仍归
 * `layerRegistry.js` 管（不要绕过注册表直接调本模块）。
 *
 * 依赖：window.Cesium、window.LayerRegistry。
 */
(function () {
  'use strict';

  /**
   * 图层 id → 数据集。id 必须与 layerRegistry.js 里的条目一致。
   * `url` 指向 tileset.json；Cesium 会以它为根解析相对 URI，因此不能带查询串。
   */
  var SOURCES = {
    'tiles3d.taipei': {
      label: '台北 3D Tiles',
      url: './tiles3d/taibei101/tileset.json',
    },
    'tiles3d.taichung': {
      label: '台中 3D Tiles',
      url: './tiles3d/taizhong/tileset.json',
    },
    'tiles3d.taoyuan': {
      label: '桃园 3D Tiles',
      url: './tiles3d/taoyuan/tileset.json',
    },
    'tiles3d.tw2024': {
      label: 'TW2024 三维样例',
      url: './tiles3d/tw2024/tileset.json',
    },
    // 2026-10-20：OSM 轮廓 + GBA 高度生成的 15 个沿海 AOI LOD1 白膜（tw_coastal.zip）。
    'tiles3d.tw_coastal': {
      label: '台湾沿海白膜（LOD1）',
      url: './tiles3d/tw_coastal/tileset.json',
    },
  };

  /**
   * 运行时优化参数（对应《3DTiles 模型太大、层级太深，到底怎么优化》第五节）。
   *
   * 数据侧（纹理降分辨率、剪枝、geometricError 重排）是离线流水线的活，见
   * `tools/tileset_stats.py` 的体检/演练输出；这里只处理**运行时**能直接拿到的收益：
   *
   *   maximumScreenSpaceError 16 → 24
   *       每屏挑选的瓦片数约降 2.25 倍（SSE 是屏幕误差上界，越大越少选），
   *       对城市级倾斜摄影 24 在 1.5 km 视距下与 16 肉眼差别很小，但显存/带宽
   *       明显下降。文章建议 32；本项目仍要看塔楼轮廓，取折中值 24。
   *   （skipLevelOfDetail 本数据集**不可用**：实测打开后 12 s 内只访问 3 个节点、
   *   0 个内容就绪 —— 本项目的 tileset 是"根节点无内容 + 子节点全为外部 tileset"
   *   的深链（见 tools/tileset_stats.py 的 depth 输出），跳级遍历找不到可渲染的
   *   祖先内容就停住了。保持 false，层级问题交给数据侧合并/剪枝。）
   *   cacheBytes / maximumCacheOverflowBytes
   *       显式给缓存封顶（256 MB / 128 MB）：默认 512 MB 对本项目的城市数据
   *       偏大，切换视角时更容易触发 GC 尖峰；封顶后卸载由 Cesium 自己按 LRU 做。
   *   dynamicScreenSpaceError / foveatedScreenSpaceError
   *       高空远瓦片降精度、屏幕边缘降精度。Cesium 1.145 默认已开，这里显式写出，
   *       避免换构建后默认值变化导致性能回退而无人察觉。
   */
  var TILESET_OPTIONS = {
    maximumScreenSpaceError: 24,
    skipLevelOfDetail: false,
    baseScreenSpaceError: 1024,
    skipScreenSpaceErrorFactor: 16,
    dynamicScreenSpaceError: true,
    foveatedScreenSpaceError: true,
    cacheBytes: 256 * 1024 * 1024,
    maximumCacheOverflowBytes: 128 * 1024 * 1024,
  };

  var viewerRef = null;
  var state = {};    // id -> {phase, tileset, detail}
  var seq = {};      // id -> 请求代次；关掉后晚到的 Promise 必须作废

  function recOf(id) {
    if (!state[id]) {
      state[id] = { phase: 'idle', tileset: null, detail: null };
    }
    return state[id];
  }

  function feed(id, status, detail) {
    if (window.LayerRegistry && LayerRegistry.setFeed) {
      LayerRegistry.setFeed(id, status, detail || null);
    }
  }

  function init(viewer) {
    viewerRef = viewer || null;
    return !!viewerRef;
  }

  /**
   * 挂上或卸下某个数据集。
   *
   * 幂等：已挂载时再次 on 直接返回；off 时若无 tileset 也安全。
   * 晚到的 `fromUrl` 用代次作废：用户"点了又关"时，Promise 仍会 resolve，
   * 若不作废就会在关闭之后把 tileset 又挂回场景（开关显示关、画面却在加载）。
   */
  function setEnabled(id, on) {
    var src = SOURCES[id];
    if (!src) return Promise.resolve(null);
    var rec = recOf(id);

    if (!on) {
      seq[id] = (seq[id] || 0) + 1;      // 作废进行中的加载
      if (rec.tileset) {
        try { viewerRef.scene.primitives.remove(rec.tileset); } catch (e) { /* 已不在场景 */ }
        try { rec.tileset.destroy(); } catch (e) { /* destroy 失败不阻塞关闭 */ }
        rec.tileset = null;
      }
      rec.phase = 'idle';
      rec.detail = null;
      feed(id, 'ok', null);
      return Promise.resolve(null);
    }

    if (rec.tileset) return Promise.resolve(rec.tileset);
    if (!viewerRef || !window.Cesium || !Cesium.Cesium3DTileset) {
      rec.phase = 'failed';
      rec.detail = 'Cesium3DTileset 不可用';
      feed(id, 'unsupported', rec.detail);
      return Promise.resolve(null);
    }

    var token = (seq[id] = (seq[id] || 0) + 1);
    rec.phase = 'loading';
    rec.detail = null;
    feed(id, 'pending', '三维瓦片加载中');

    var start;
    try {
      // fromUrl 是 Cesium 1.104+ 的静态入口；极老的构建没有它，退回构造器。
      start = Cesium.Cesium3DTileset.fromUrl
        ? Cesium.Cesium3DTileset.fromUrl(src.url, TILESET_OPTIONS)
        : Promise.resolve(new Cesium.Cesium3DTileset({ url: src.url }));
    } catch (e) {
      start = Promise.reject(e);
    }

    return start.then(function (tileset) {
      if (seq[id] !== token) {           // 加载期间已被关掉
        try { tileset.destroy(); } catch (e) { /* 忽略 */ }
        return null;
      }
      rec.tileset = viewerRef.scene.primitives.add(tileset);
      rec.phase = 'live';
      feed(id, 'ok', null);
      return tileset;
    }).catch(function (e) {
      if (seq[id] !== token) return null;
      rec.phase = 'failed';
      rec.detail = (e && e.message) || String(e);
      feed(id, 'failed', '三维瓦片加载失败：' + rec.detail);
      return null;
    });
  }

  function enabled(id) {
    var rec = state[id];
    return !!(rec && rec.tileset);
  }

  /** 取证出口：每个数据集的阶段与已加载瓦片统计（验收与排错用）。 */
  function debug() {
    var out = {};
    Object.keys(SOURCES).forEach(function (id) {
      var r = recOf(id);
      out[id] = {
        phase: r.phase,
        detail: r.detail,
        url: SOURCES[id].url,
        mounted: !!r.tileset,
        tilesLoaded: r.tileset ? r.tileset.tilesLoaded : null,
        maximumScreenSpaceError: r.tileset ? r.tileset.maximumScreenSpaceError : null,
      };
    });
    return out;
  }

  window.Local3D = {
    SOURCES: SOURCES,
    init: init,
    setEnabled: setEnabled,
    enabled: enabled,
    debug: debug,
  };
})();
