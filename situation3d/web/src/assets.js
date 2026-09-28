/* situation3d — 离线资源层（含 LOD）
 *
 * 平台要求完全离线运行，因此底图、高程、建筑三类数据都预先下载为
 * XYZ(Web Mercator) 瓦片，由 tools/fetch_assets.py 落盘：
 *
 *   tiles/imagery/{z}/{x}/{y}.jpg        卫星影像底图（Esri World Imagery）
 *   tiles/terrain/{z}/{x}/{y}.png        高程，Terrarium 编码（Mapzen/AWS）
 *   tiles/wsf3d/{layer}/{z}/{x}/{y}.png  World Settlement Footprint 3D（DLR）
 *   tiles/manifest.json                  实际覆盖范围与层级区间
 *
 * LOD 设计要点
 * ------------
 * 1. **覆盖范围来自 manifest，而不是猜。** 底图只下载了台海 AOI，如果
 *    provider 仍声明覆盖全球，Cesium 会为域外瓦片持续发请求、持续 404，
 *    并因此卡在 z0–z1 不再细化（实测：500 次重试、整幅底图退化成渐变）。
 *    把 `rectangle` 设成 manifest 里的真实范围，请求全部命中已下载瓦片。
 * 2. **每族瓦片有明确的层级区间。** 影像 z0–11（全域）+ 城区 z12–14；
 *    高程 z0–10；WSF3D z0–10（面）+ 城区 z13（三维体块）。
 * 3. **高程超过下载层级时复用祖先瓦片**，按需降采样，不会出现空洞。
 * 4. **建筑体块分级生成**：按相机高度选择聚合网格与采样步长，远看用粗网格
 *    快速出图，近看才做全分辨率聚合；已生成的层级缓存复用。
 * 5. 三族瓦片解码结果都有 LRU 上限，长时间推演不会无限占内存。
 */

'use strict';

(function (global) {
  const TILE_PX = 256;

  // 瓦片 URL 模板集中在这里：provider 与缓存层必须用同一个模板，否则缓存
  // 键与 Cesium 请求的瓦片会对不上。
  const IMAGERY_URL = './tiles/imagery/{z}/{x}/{y}.jpg';
  const WSF3D_URL = './tiles/wsf3d/{layer}/{z}/{x}/{y}.png';
  const GH3D_URL = './tiles/gh3d/{layer}/{z}/{x}/{y}.png';

  // CustomHeightmapTerrainProvider 的高度图宽度/高度：文档要求 2^n+1，
  // 这样正好得到 2^n 个网格四边形。源瓦片是 256x256，解码后重采样到 257。
  const HEIGHTMAP_PX = 257;

  // 兜底覆盖范围，与 tools/fetch_assets.py 的 DEFAULT_BBOX 一致。
  // manifest.json 存在时以 manifest 为准。
  const FALLBACK = {
    bbox: [115.0, 18.0, 125.0, 29.0],
    imagery: { min_zoom: 0, max_zoom: 11 },
    terrain: { min_zoom: 0, max_zoom: 10 },
    wsf3d: { max_zoom: 10 },
    urban_imagery: { min_zoom: 12, max_zoom: 14 },
    // 三维体块的高度栅格来源；见 blockSource()。缺省 wsf3d。
    building_blocks: { source: 'wsf3d', layer: 'buildingheight' },
  };

  // 解码结果缓存上限（瓦片数）。一瓦片约 256 KB（Float32 高程）或 256 KB
  // （RGBA），200 片约 50 MB，够覆盖视野又不会撑爆内存。
  const CACHE_LIMIT = 200;

  function lruGet(cache, key) {
    if (!cache.has(key)) return undefined;
    const value = cache.get(key);
    // 重新插入以刷新 LRU 次序。
    cache.delete(key);
    cache.set(key, value);
    return value;
  }

  function lruSet(cache, key, value) {
    cache.set(key, value);
    while (cache.size > CACHE_LIMIT) {
      const oldest = cache.keys().next();
      if (oldest.done) break;
      cache.delete(oldest.value);
    }
  }

  let MANIFEST = null;

  function loadManifest() {
    if (MANIFEST) return Promise.resolve(MANIFEST);
    return fetch('./tiles/manifest.json', { cache: 'no-store' })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (m) {
        MANIFEST = Object.assign({}, FALLBACK, m || {});
        if (!Array.isArray(MANIFEST.bbox) || MANIFEST.bbox.length !== 4) {
          MANIFEST.bbox = FALLBACK.bbox;
        }
        MANIFEST.imagery = Object.assign({}, FALLBACK.imagery, m && m.imagery);
        MANIFEST.terrain = Object.assign({}, FALLBACK.terrain, m && m.terrain);
        MANIFEST.urban_imagery = Object.assign({}, FALLBACK.urban_imagery,
          m && m.urban_imagery);
        return MANIFEST;
      })
      .catch(function () {
        MANIFEST = Object.assign({}, FALLBACK);
        return MANIFEST;
      });
  }

  function manifest() { return MANIFEST || FALLBACK; }

  function aoiRectangle(bbox) {
    return Cesium.Rectangle.fromDegrees(bbox[0], bbox[1], bbox[2], bbox[3]);
  }

  // ------------------------------------------------------------------------
  // 请求统计（诊断用）
  // ------------------------------------------------------------------------

  /**
   * Wrap a provider so the tile levels it actually asks for can be inspected.
   *
   * Cesium picks the imagery level from the surface tile's screen-space error,
   * and when the result looks coarser than expected there is no public API that
   * says which level was chosen. Counting the requests settles it.
   */
  function instrument(provider, stats) {
    const original = provider.requestImage.bind(provider);
    provider.requestImage = function (x, y, level, request) {
      stats.requests++;
      if (level > stats.maxLevel) stats.maxLevel = level;
      if (level < stats.minLevel) stats.minLevel = level;
      stats.levels[level] = (stats.levels[level] || 0) + 1;
      return original(x, y, level, request);
    };
    if (provider.errorEvent && provider.errorEvent.addEventListener) {
      provider.errorEvent.addEventListener(function (err) {
        stats.errors++;
        stats.lastError = (err && err.message) || String(err);
      });
    }
    provider.__stats = stats;
    return provider;
  }

  function newStats(label) {
    return { label: label, requests: 0, errors: 0, maxLevel: -1, minLevel: 99,
             levels: {}, lastError: null };
  }

  // ------------------------------------------------------------------------
  // 影像底图
  // ------------------------------------------------------------------------

  /**
   * 是否绕过前端缓存。
   *
   * 取证用的开关：`?cache=0` 时瓦片交给 Cesium 自己的加载器，用来把
   * "底图不对" 定位到缓存层还是场景配置，而不是靠猜。
   */
  function cacheEnabled() {
    return !global.__situation3dNoTileCache;
  }

  function attachCached(provider, urlTemplate) {
    return cacheEnabled() ? TileCache.attach(provider, { urlTemplate: urlTemplate })
                          : provider;
  }

  // ------------------------------------------------------------------------
  // 天地图（在线）
  // ------------------------------------------------------------------------

  /**
   * 天地图 WMTS 服务。
   *
   * 与其它 provider 的关键差异：**这是在线服务，不经过 TileCache**。
   * 其它图层都是离线瓦片（`./tiles/...`），走前端缓存是为了"浏览过的瓦片不再回
   * 服务器"；天地图本来就必须联网，缓存它的字节只会占 IndexedDB 配额，还会让
   * "离线可用"这条边界变得含糊。所以这里直接交给 Cesium 的加载器。
   *
   * 两套坐标：`w` 是 Web Mercator（与 Cesium 的 WebMercatorTilingScheme 一致），
   * `c` 是经纬度直投（GeographicTilingScheme）。这里用 `w`。
   *
   * 影像服务本身是**不带注记**的纯底图，中文注记在 `cva`（矢量注记）里，必须叠一层，
   * 否则地图上一个地名都没有。两者是不同 layer，由 createTiandituLabelProvider 提供。
   */
  const TIANDITU_BASE = 'http://t0.tianditu.gov.cn';
  // 申请的 key。可用 `?tdtkey=` 覆盖，便于换 key 时不必改代码。
  const TIANDITU_KEY_DEFAULT = '8af77812ad311e785fa7d8c0539da6d5';

  function tiandituKey() {
    try {
      const q = new URLSearchParams(global.location.search);
      return q.get('tdtkey') || TIANDITU_KEY_DEFAULT;
    } catch (e) {
      return TIANDITU_KEY_DEFAULT;
    }
  }

  function tiandituUrl(layer, suffix, ext) {
    return TIANDITU_BASE + '/' + layer + '_' + suffix + '/wmts' +
      '?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0' +
      '&LAYER=' + layer + '&STYLE=default' +
      '&TILEMATRIXSET=' + suffix + '&FORMAT=' + (ext || 'tiles') +
      '&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}&tk=' + tiandituKey();
  }

  /**
   * @param {string} kind 'img' 影像 / 'vec' 矢量底图 / 'ter' 地形晕渲
   * @param {string} suffix 'w' WebMercator / 'c' 经纬度
   */
  function createTiandituProvider(kind, suffix) {
    const s = suffix || 'w';
    const scheme = s === 'c' ? new Cesium.GeographicTilingScheme()
                             : new Cesium.WebMercatorTilingScheme();
    return new Cesium.UrlTemplateImageryProvider({
      url: tiandituUrl(kind || 'img', s),
      tilingScheme: scheme,
      minimumLevel: 1,
      maximumLevel: 18,
      credit: '天地图（国家地理信息公共服务平台）',
    });
  }

  /** 中文注记层：影像底图必须叠这一层才有地名。 */
  function createTiandituLabelProvider(suffix) {
    const s = suffix || 'w';
    const scheme = s === 'c' ? new Cesium.GeographicTilingScheme()
                             : new Cesium.WebMercatorTilingScheme();
    return new Cesium.UrlTemplateImageryProvider({
      url: tiandituUrl('cva', s),
      tilingScheme: scheme,
      minimumLevel: 1,
      maximumLevel: 18,
      credit: '天地图注记',
    });
  }

  /**
   * 低层底图（原称"全球层"）：给拉远/平移出战区时留一层底，避免只剩球体底色。
   *
   * **覆盖范围必须用实测值，不能声明全球。** 这批瓦片的名字叫"全球 z0-6"，但实际
   * 只覆盖台海区域——实测各层瓦片数是 1/1/1/2/5/7/12/34/99/339/1242/4851（z0-z11），
   * 是一条金字塔而不是全球。provider 不传 rectangle 就是全球，于是拉开视野后
   * Cesium 会为域外请求瓦片，每个都 404（实测打开离线底图时
   * `imagery/2/3/0.jpg`、`1/1/1.jpg`、`6/54/25.jpg` 一连串 404）。
   *
   * 真实范围由 tools/analyze_tile_coverage.py 扫描实际瓦片算出并写入
   * `manifest.coverage.imagery.rectangle`，这里读它；拿不到时退回 bbox。
   */
  function measuredRect(kind, fallbackBbox) {
    const m = manifest();
    const cov = m.coverage && m.coverage[kind];
    if (cov) {
      // 覆盖形态现在是"低层全球 + 高层区域"两段：
      //   底图（低层）层用 global_rectangle（补过 z0-z6 全球瓦片后就是全球）
      //   区域层用 region_rectangle
      // 拿不到就退回 basemap / 全部并集 / bbox。
      const r = cov.global_rectangle || cov.basemap_rectangle ||
                cov.region_rectangle || cov.rectangle;
      if (Array.isArray(r) && r.length === 4) return aoiRectangle(r);
    }
    return fallbackBbox ? aoiRectangle(fallbackBbox) : undefined;
  }

  /**
   * 低层底图：z0 起、到"全球瓦片齐全的最高层级"为止（补过 z0-z6 全球后就是 z6）。
   *
   * 覆盖矩形来自 manifest.coverage.imagery.global_rectangle——补齐全全球瓦片后
   * 它等于全球；在补之前它是台海区域。这样 provider 声明的范围始终与实际瓦片一致，
   * 不会出现"声明全球却处处 404"。
   */
  function createGlobalImageryProvider() {
    const m = manifest();
    const cov = (m.coverage && m.coverage.imagery) || {};
    const provider = new Cesium.UrlTemplateImageryProvider({
      url: IMAGERY_URL,
      tilingScheme: new Cesium.WebMercatorTilingScheme(),
      minimumLevel: 0,
      // 底图层只负责低层级；再往上交给战区/热点/城区层。设成 zoom_max(17)
      // 会让它在覆盖之外的 z7+ 反复 404。
      maximumLevel: (cov.global_max_zoom !== undefined && cov.global_max_zoom > 0)
        ? cov.global_max_zoom
        : ((m.imagery_global && m.imagery_global.max_zoom) || 6),
      rectangle: measuredRect('imagery', m.bbox),
      credit: 'Esri World Imagery (low zoom)',
    });
    return attachCached(provider, IMAGERY_URL);
  }

  /**
   * 热点区域高清影像：每个热点框一个 provider，限制在自己的 rectangle 内。
   *
   * 这些框里有三个（三亚 / 冲绳 / 关岛）完全在战区 bbox 之外，战区层覆盖不到，
   * 必须靠这一层。层级从 0 起：在框内即使拉远也应该走这一层，而不是掉到全球层的 z6。
   * 与战区层重叠的部分会因为 URL 相同而命中 TileCache 的内存缓存，代价很小。
   */
  function attachHotspotImagery(viewer, boxes, maxZoom) {
    const m = manifest();
    const top = maxZoom || (m.hotspot_imagery && m.hotspot_imagery.max_zoom) || 12;
    const layers = [];
    (boxes || []).forEach(function (box, i) {
      const provider = attachCached(new Cesium.UrlTemplateImageryProvider({
        url: IMAGERY_URL,
        tilingScheme: new Cesium.WebMercatorTilingScheme(),
        minimumLevel: 0,
        maximumLevel: top,
        rectangle: aoiRectangle(box),
        credit: 'Esri World Imagery (hotspot ' + (i + 1) + ')',
      }), IMAGERY_URL);
      layers.push(viewer.imageryLayers.addImageryProvider(provider));
    });
    return layers;
  }

  /**
   * Theatre bounding boxes, for when the manifest does not carry them.
   *
   * Two theatres are disjoint (Taiwan Strait and the India-China western sector),
   * and a single manifest `bbox` covering both would span most of Asia, so the
   * offline providers would declare coverage over a vast area with no tiles and
   * spend the session on 404s. Each provider therefore takes the *active*
   * theatre's box, which `setTheatre` switches when a case is loaded.
   *
   * Kept in sync with `engine/scenario.py` (INDIA_CHINA_HOTSPOT_BOXES /
   * `_HOTSPOT_BOXES`) and `tools/fetch_roads.py` THEATRES. The manifest wins when
   * it provides `theatres`, so baking a theatre updates the front end without a
   * code change; this table is the fallback that keeps the app working before the
   * manifest has been regenerated.
   */
  var THEATRE_BBOX = {
    taiwan_strait: [115.5, 20.0, 122.5, 26.5],
    india_china: [77.0, 31.5, 81.5, 36.5],
  };
  var activeTheatre = 'taiwan_strait';

  function theatreList() {
    var m = manifest();
    if (m.theatres && typeof m.theatres === 'object') {
      var out = {};
      Object.keys(m.theatres).forEach(function (k) {
        var t = m.theatres[k] || {};
        if (Array.isArray(t.bbox) && t.bbox.length === 4) out[k] = t.bbox;
      });
      if (Object.keys(out).length) return out;
    }
    return THEATRE_BBOX;
  }

  function getTheatre() { return activeTheatre; }

  /**
   * Select the theatre whose coverage the regional layers should use.
   *
   * Returns the previous name so the caller can skip a rebuild when a case is
   * reloaded for the same theatre.
   */
  function setTheatre(name) {
    var prev = activeTheatre;
    if (name && theatreList()[name]) activeTheatre = name;
    return prev;
  }

  function theatreBbox(name) {
    var list = theatreList();
    return list[name || activeTheatre] || manifest().bbox;
  }

  /**
   * Does a hotspot box intersect the active theatre?
   *
   * NOT currently used to filter the mounted hotspot layers, and that is
   * deliberate rather than an oversight. `manifest.hotspot_imagery.boxes` are
   * anonymous coordinate arrays (no names), and the Taiwan case intentionally
   * keeps four hotspots *outside* its theatre box - 三亚, 冲绳岛, 关岛, 羽田 -
   * which the code comments call out as reachable only through this layer. A
   * bounding-box filter therefore cannot tell "hotspot of the other theatre" from
   * "hotspot deliberately outside this theatre", and applying it hid four
   * intended layers.
   *
   * The leak it was meant to fix is real: measured, switching to the Ladakh case
   * left ten Taiwan Strait hotspot providers mounted and requesting tiles at
   * 117-145E. The fix requires hotspot identity, so the next step is to have
   * `tools/finalize_manifest.py` emit `hotspot_imagery.boxes` as
   * `{name, bbox}` (it already knows the names) and to select boxes by the case's
   * declared `hotspot_names` instead of by geometry. This helper is kept for that
   * change and for callers that only have a box.
   */
  function boxInTheatre(bbox) {
    if (!bbox || bbox.length !== 4) return true;
    var t = theatreBbox();
    if (!t || t.length !== 4) return true;
    var overlap = !(bbox[2] < t[0] || bbox[0] > t[2] || bbox[3] < t[1] || bbox[1] > t[3]);
    return overlap;
  }

  function createImageryProvider() {
    const m = manifest();
    // Regional imagery is restricted to the active theatre, not the manifest bbox.
    const bbox = theatreBbox();
    const provider = new Cesium.UrlTemplateImageryProvider({
      url: IMAGERY_URL,
      tilingScheme: new Cesium.WebMercatorTilingScheme(),
      minimumLevel: m.imagery.min_zoom || 0,
      maximumLevel: m.imagery.max_zoom || 11,
      // 关键：声明真实覆盖范围，域外不再发请求。
      rectangle: aoiRectangle(bbox),
      credit: 'Esri World Imagery (Maxar / Earthstar Geographics)',
    });
    // 走前端缓存：取过的瓦片不再回服务器。
    return attachCached(provider, IMAGERY_URL);
  }

  /**
   * High-zoom imagery restricted to the urban boxes.
   *
   * Each provider carries its own `rectangle`, so Cesium only asks for z12-z14
   * tiles inside the cities. Without that limit the extra levels would 404
   * across the whole theatre and the provider would spend the session chasing
   * tiles that do not exist.
   */
  function attachUrbanImagery(viewer, aois) {
    const m = manifest();
    const layers = [];
    (aois || []).forEach(function (aoi) {
      const provider = attachCached(new Cesium.UrlTemplateImageryProvider({
        url: IMAGERY_URL,
        tilingScheme: new Cesium.WebMercatorTilingScheme(),
        // 重点区域补到 z17（约 1.2 m/px，米级）；旧 manifest 只有 z12-14
        minimumLevel: (m.urban_imagery_detail && m.urban_imagery_detail.min_zoom) ||
                      m.urban_imagery.min_zoom || 12,
        maximumLevel: (m.urban_imagery_detail && m.urban_imagery_detail.max_zoom) ||
                      m.urban_imagery.max_zoom || 14,
        rectangle: aoiRectangle(aoi.bbox),
        credit: 'Esri World Imagery (urban detail)',
      }), IMAGERY_URL);
      const layer = viewer.imageryLayers.addImageryProvider(provider);
      layers.push({ name: aoi.name, layer: layer });
    });
    return layers;
  }

  /**
   * 鸟瞰图专用影像 provider。
   *
   * **不与主视图共用实例**：主视图的 provider 被 TileCache.attach 与 instrument
   * 各包了一层，共用会让 2D 的请求与 404 计入主视图的 imageryStats，?diag=1 全部
   * 失真；errorEvent 也是共享 Event，无法按 viewer 区分。这里只要 z0–z6，靠 HTTP
   * 缓存（/tiles/ 已是 immutable）即可，不挂 TileCache。
   */
  function createOverviewImageryProvider() {
    const m = manifest();
    return new Cesium.UrlTemplateImageryProvider({
      url: IMAGERY_URL,
      tilingScheme: new Cesium.WebMercatorTilingScheme(),
      minimumLevel: 0,
      maximumLevel: (m.imagery_global && m.imagery_global.max_zoom) || 6,
      // **不限制 rectangle**：鸟瞰图与主图同视点但高度是它的 5 倍，视野覆盖整个
      // 战场（甚至整个地球）。限定在台海 AOI 会让它在看别处时整片空白——实测
      // 主图 900 km 时鸟瞰在 4500 km、pitch -75°，视线落点在北边约 35°N，
      // 正好在 AOI（18-29°N）之外，于是画布全黑。
      credit: 'Esri World Imagery (overview)',
    });
  }

  // ------------------------------------------------------------------------
  // 地形高程
  // ------------------------------------------------------------------------

  const terrainCache = new Map();

  function imageFromUrl(url) {
    return new Promise(function (resolve, reject) {
      const im = new Image();
      im.crossOrigin = 'anonymous';
      im.onload = function () { resolve(im); };
      im.onerror = function () { reject(new Error('image load failed: ' + url)); };
      im.src = url;
    });
  }

  const decodeCanvas = document.createElement('canvas');
  decodeCanvas.width = TILE_PX;
  decodeCanvas.height = TILE_PX;
  const decodeCtx = decodeCanvas.getContext('2d', { willReadFrequently: true });

  /**
   * Decode one Terrarium tile into a 256x256 Float32Array of metres.
   *
   * Terrarium encoding: height = (R * 256 + G + B / 256) - 32768.
   * Tiles outside the downloaded zoom range resolve to null so the caller can
   * fall back to an ancestor tile instead of failing the whole terrain.
   */
  function loadTerrainTile(z, x, y) {
    const key = z + '/' + x + '/' + y;
    const hit = lruGet(terrainCache, key);
    if (hit !== undefined) return hit;

    const promise = imageFromUrl('./tiles/terrain/' + key + '.png')
      .then(function (im) {
        decodeCtx.clearRect(0, 0, TILE_PX, TILE_PX);
        decodeCtx.drawImage(im, 0, 0, TILE_PX, TILE_PX);
        const d = decodeCtx.getImageData(0, 0, TILE_PX, TILE_PX).data;
        const out = new Float32Array(TILE_PX * TILE_PX);
        for (let i = 0, p = 0; i < out.length; i++, p += 4) {
          out[i] = (d[p] * 256 + d[p + 1] + d[p + 2] / 256) - 32768;
        }
        return out;
      })
      .catch(function () { return null; });

    lruSet(terrainCache, key, promise);
    return promise;
  }

  /**
   * Build the heightmap CustomHeightmapTerrainProvider expects.
   *
   * Levels beyond the downloaded range are served from the nearest available
   * ancestor tile, nearest-neighbour sampled into the requested sub-extent.
   * That is exactly what a terrain pyramid does, and it keeps the globe solid
   * when the camera drops below the last downloaded level instead of leaving a
   * hole.
   */
  /**
   * 高程的最大可用层级。地形现在是分级覆盖的：全球 z0-10，战区与热点补到 z12，
   * 重点区域补到 z14。取其中最大者，再由 heightmapFor 逐级上溯决定实际用哪一级。
   */
  function terrainMaxLevel() {
    const m = manifest();
    return Math.max(
      (m.terrain && m.terrain.max_zoom) || 10,
      (m.theatre_terrain && m.theatre_terrain.max_zoom) || 0,
      (m.hotspot_terrain && m.hotspot_terrain.max_zoom) || 0,
      (m.urban_terrain && m.urban_terrain.max_zoom) || 0);
  }

  /**
   * 构造高度图，**逐级上溯**直到找到一张真实存在的瓦片。
   *
   * 地形是分级覆盖的：重点区域有 z14，热点有 z12，别处只有 z10。如果只按
   * "请求层级取不到就返回平地"，那么在覆盖范围之外放大时地形会**塌成平面**，
   * 反而比粗层级更差。所以这里从目标层级一路向上找，用最近的有数据的祖先瓦片
   * 降采样——这正是地形金字塔本身的行为。
   */
  /**
   * 该瓦片是否落在实测的地形覆盖范围内。
   *
   * 不在范围内就直接返回平地、不发请求——地形瓦片集和影像一样是台海金字塔，
   * 域外请求必然 404（实测 `tiles/terrain/1/0/0.png`、`1/0/1.png` 都是 404）。
   * 用实测矩形短路，比让 Cesium 反复请求再失败干净。
   */
  function terrainTileCovered(x, y, level) {
    const m = manifest();
    const cov = m.coverage && m.coverage.terrain;
    if (!cov) return true;
    const perZoom = cov.per_zoom && cov.per_zoom[String(level)];
    if (!perZoom) return false;            // 这一级一张都没有 → 直接给平地，别发请求
    const r = perZoom.regional_rectangle || perZoom.full_rectangle;
    if (!Array.isArray(r) || r.length !== 4) return true;
    const w = tileFracToLon(x, level);
    const e = tileFracToLon(x + 1, level);
    const n = tileFracToLat(y, level);
    const s = tileFracToLat(y + 1, level);
    return !(e < r[0] || w > r[2] || n < r[1] || s > r[3]);
  }

  /** 平地高度图（无数据时的兜底）。 */
  function flatHeightmap() {
    return new Float32Array(HEIGHTMAP_PX * HEIGHTMAP_PX);
  }

  async function heightmapFor(x, y, level) {
    const maxLevel = terrainMaxLevel();
    let lv = Math.min(level, maxLevel);
    let tx = x;
    let ty = y;
    if (level > maxLevel) {
      const shift = level - maxLevel;
      tx = x >> shift;
      ty = y >> shift;
    }

    // 逐级上溯，但**只对覆盖范围内的层级发请求**。
    // 地形是"低层全球 + 高层区域"：台海之外在 z10 以上没有瓦片，
    // 不加这层判断就会一路 404（每个失败请求还要等一次往返）。
    // 上溯到有瓦片的祖先层级后降采样——这正是地形金字塔的行为，
    // 既避免 404，也不会让台海之外的地形塌成平面（那里本来就有 z0-z6 的全球瓦片）。
    for (; lv >= 0; lv--) {
      if (terrainTileCovered(tx, ty, lv)) {
        const src = await loadTerrainTile(lv, tx, ty);
        if (src) {
          const span = Math.pow(2, level - lv);     // 该瓦片跨越多少个 level 级瓦片
          const ox = (x - tx * span) / span;
          const oy = (y - ty * span) / span;
          return subSample(src, ox, oy, 1 / span);
        }
      }
      tx >>= 1;
      ty >>= 1;
    }
    return flatHeightmap();
  }

  function subSample(src, ox, oy, span) {
    const out = new Float32Array(HEIGHTMAP_PX * HEIGHTMAP_PX);
    const last = TILE_PX - 1;
    for (let j = 0; j < HEIGHTMAP_PX; j++) {
      const v = oy + (j / (HEIGHTMAP_PX - 1)) * span;
      let sj = Math.round(v * last);
      if (sj < 0) sj = 0; else if (sj > last) sj = last;
      const rowOff = sj * TILE_PX;
      const dstOff = j * HEIGHTMAP_PX;
      for (let i = 0; i < HEIGHTMAP_PX; i++) {
        const u = ox + (i / (HEIGHTMAP_PX - 1)) * span;
        let si = Math.round(u * last);
        if (si < 0) si = 0; else if (si > last) si = last;
        out[dstOff + i] = src[rowOff + si];
      }
    }
    return out;
  }

  function createTerrainProvider() {
    return new Cesium.CustomHeightmapTerrainProvider({
      width: HEIGHTMAP_PX,
      height: HEIGHTMAP_PX,
      tilingScheme: new Cesium.WebMercatorTilingScheme(),
      callback: function (x, y, level) { return heightmapFor(x, y, level); },
      credit: 'Elevation: Mapzen / AWS Open Data (SRTM, 30 m)',
    });
  }

  // ------------------------------------------------------------------------
  // World Settlement Footprint 3D / 3D-GloBFP 建筑高度栅格
  // ------------------------------------------------------------------------

  /**
   * Draped WSF3D layer.
   *
   * The DLR WMS renders the product with its own colour ramp and paints
   * no-data as opaque black, so the layer is composited with colorToAlpha to
   * knock the black back out; otherwise the whole globe would be covered.
   */
  function createWsf3dOverlayProvider(layerName) {
    const m = manifest();
    const template = WSF3D_URL.replace('{layer}', layerName);
    const provider = new Cesium.UrlTemplateImageryProvider({
      url: template,
      tilingScheme: new Cesium.WebMercatorTilingScheme(),
      minimumLevel: 0,
      maximumLevel: m.wsf3d.max_zoom || 10,
      rectangle: aoiRectangle(theatreBbox()),
      credit: 'World Settlement Footprint 3D v02, DLR EOC',
    });
    return attachCached(provider, template);
  }

  function attachWsf3dOverlay(viewer, layerName, alpha) {
    const provider = createWsf3dOverlayProvider(layerName);
    const layer = viewer.imageryLayers.addImageryProvider(provider);
    layer.alpha = alpha === undefined ? 0.6 : alpha;
    if ('colorToAlpha' in layer) {
      layer.colorToAlpha = new Cesium.Color(0, 0, 0, 1);
      layer.colorToAlphaThreshold = 0.04;
    }
    return layer;
  }

  /** The colour-ramp calibration produced by tools/calibrate_wsf3d.py. */
  function loadHeightRamp() {
    return fetch(rampUrl(), { cache: 'no-store' })
      .then(function (r) {
        if (!r.ok) throw new Error(rampUrl() + ' HTTP ' + r.status);
        return r.json();
      })
      .then(function (ramp) {
        // 调色板查表：颜色是离散的调色板项，逐项实测比线性拟合准。
        const table = new Map();
        const entries = (ramp.lookup || []).slice();
        entries.forEach(function (e) {
          table.set(e.rgb[0] + ',' + e.rgb[1] + ',' + e.rgb[2], e.h);
        });
        ramp.decode = function (r, g, b) {
          const exact = table.get(r + ',' + g + ',' + b);
          if (exact !== undefined) return exact;
          // 未命中的调色板项取 RGB 距离最近的条目。
          let best = 0;
          let bestD = Infinity;
          for (let i = 0; i < entries.length; i++) {
            const e = entries[i].rgb;
            const d = (e[0] - r) * (e[0] - r) + (e[1] - g) * (e[1] - g) +
                      (e[2] - b) * (e[2] - b);
            if (d < bestD) { bestD = d; best = i; }
          }
          return entries.length ? entries[best].h : 0;
        };
        ramp.isNoData = function (r, g, b) {
          // WSF3D 的无数据是纯黑；3D-GloBFP 瓦片用调色板 index 0 且 tRNS=0，
          // 浏览器解码后同样是 (0,0,0)，所以判据一致。
          return r === 0 && g === 0 && b === 0;
        };
        return ramp;
      });
  }

  // ------------------------------------------------------------------------
  // 由 WSF3D 栅格生成三维建筑体块（分级 LOD）
  // ------------------------------------------------------------------------

  /**
   * LOD 档位。相机越高，聚合网格越粗、采样步长越大，出图越快。
   *
   * maxHeight 相机高度上限（米）；grid 每轴格数；step 像素采样步长。
   * 以 0.12°（约 13 km）的城区 AOI 为例：
   *   grid 64  → 约 200 m 体块
   *   grid 96  → 约 135 m
   *   grid 128 → 约 100 m
   *   grid 144 → 约 90 m，与源数据分辨率相当
   *
   * **档位必须从"最近"往"最远"排**（maxHeight 递增）。
   * lodForHeight 是"第一个满足 h <= maxHeight 的就返回"，所以表的顺序必须与
   * 高度的递增方向一致。原先的表是反的：
   *
   *   { far, 400000 }, { mid, 160000 }, { near, 60000 }, { close, Infinity }
   *
   * 于是 60 km 以下与 60–400 km **全部命中 far**，而 close 只在**高于 400 km**
   * 时才命中（即最远反而用最细的网格，且那时 AOI 已经小到看不见）。
   * 表现就是"不论怎么推近，体块永远是 far 那一档的粗方块"——实测 9 km / 12 km /
   * 30 km 高度下 lodForHeight 都返回 0（far）。地名图层（places.js 的 LOD）有
   * **完全相同**的方向错误，一并修掉了。
   *
   * 关键约束：grid 加密**必须**同步抬 maxBlocks。块数是按 gy/gx 行优先顺序填充的，
   * 一旦撞上 maxBlocks 就 continue——那会从北/西侧开始整片丢失方块，比均匀稀疏
   * 难看得多（一片地少东西、另一片满）。
   */
  const BUILDING_LOD = [
    { key: 'close', maxHeight: 60000,    grid: 144, step: 1, minHeight: 4, maxBlocks: 9000 },
    { key: 'near',  maxHeight: 160000,   grid: 128, step: 2, minHeight: 4, maxBlocks: 9000 },
    { key: 'mid',   maxHeight: 400000,   grid: 96,  step: 2, minHeight: 5, maxBlocks: 7000 },
    { key: 'far',   maxHeight: Infinity, grid: 64,  step: 2, minHeight: 6, maxBlocks: 5200 },
  ];


  function lodForHeight(heightM) {
    for (let i = 0; i < BUILDING_LOD.length; i++) {
      if (heightM <= BUILDING_LOD[i].maxHeight) return i;
    }
    return BUILDING_LOD.length - 1;
  }

  /**
   * 三维体块的高度栅格来源。
   *
   * 两个源同构（`<layer>/<z>/<x>/<y>.png` + 一张颜色→高度查表），所以切换只
   * 需要换目录与查表文件：
   *
   *   wsf3d  DLR WSF3D，90 m 逐聚落平均高度，全球连续但对城市偏粗
   *   gh3d   3D-GloBFP 逐建筑 Height 栅格化，17 m 像元取像元内最大高度，
   *          天际线更真，但覆盖不均（中国大陆可用，台湾近乎为空）
   *
   * 由 manifest 的 `building_blocks.source` 选择，缺省保持 wsf3d，这样即使
   * gh3d 瓦片没构建出来，体块也不会整块消失。
   */
  function blockSource() {
    const cfg = manifest().building_blocks || {};
    return {
      dir: cfg.source === 'gh3d' ? './tiles/gh3d' : './tiles/wsf3d',
      layer: cfg.layer || 'buildingheight',
      ramp: cfg.ramp || (cfg.source === 'gh3d'
        ? './tiles/gh3d/height_ramp_gh3d.json'
        : './tiles/wsf3d/height_ramp.json'),
    };
  }

  function rampUrl() { return blockSource().ramp; }

  const wsf3dTileCache = new Map();

  /**
   * 高程瓦片 → Float32Array 高度。**按瓦片缓存**（LRU 48 片 ≈ 12 MB）。
   *
   * 加这层缓存的原因是实测出来的：同一个 AOI 反复生成体块（切 LOD、开关体块、
   * 相机高度跨档）时，每片瓦片都要重新 `new Image()` + `drawImage` +
   * `getImageData`，而 `getImageData` 在无头软件光栅化下是同步回读，16 片就要
   * 约 5 s。缓存之后重复生成几乎是瞬时的。
   */
  const wsf3dHeightCache = new Map();
  const WSF3D_CACHE_LIMIT = 48;

  function decodeWsf3dTile(png) {
    return Cesium.defined(png) ? (function () {
      const out = new Float32Array(TILE_PX * TILE_PX);
      for (let i = 0, p = 0; i < out.length; i++, p += 4) {
        const r = png[p], g = png[p + 1], b = png[p + 2];
        // (0,0,0) 是两种源的统一无数据表示：WSF3D 是黑色，gh3d 是 index 0 + tRNS。
        out[i] = (r === 0 && g === 0 && b === 0)
          ? 0
          : (r * 256 + g + b / 256) - 32768;
      }
      return out;
    })() : null;
  }

  function loadWsf3dTile(z, x, y) {
    const key = z + '/' + x + '/' + y;
    const hit = lruGet(wsf3dTileCache, key);
    if (hit !== undefined) return hit;
    const src = blockSource();
    const promise = imageFromUrl(src.dir + '/' + src.layer + '/' + key + '.png')
      .then(function (im) {
        decodeCtx.clearRect(0, 0, TILE_PX, TILE_PX);
        decodeCtx.drawImage(im, 0, 0, TILE_PX, TILE_PX);
        return decodeCtx.getImageData(0, 0, TILE_PX, TILE_PX).data;
      })
      .catch(function () { return null; });
    lruSet(wsf3dTileCache, key, promise);
    return promise;
  }

  /** 解码并缓存单片瓦片的高度（按瓦片而非按请求缓存）。 */
  const DECODE_CANVASES = 4;
  const decodePool = [];
  let decodePoolCursor = 0;

  /**
   * 取一块解码画布。
   *
   * 单块共享画布会把所有瓦片解码串行化：`drawImage` + `getImageData` 在无头软件
   * 光栅化下每片要几百毫秒（同步回读），16 片就是 5 s 以上。用几块画布错开，
   * 让浏览器把解码摊到多帧上。**每块画布同一时刻只被一个任务占用**，取用与归还
   * 都在同一次同步执行里，不存在同一块画布被两个任务交叉使用的情况。
   */
  function acquireDecodeCanvas() {
    for (let i = 0; i < DECODE_CANVASES; i++) {
      const idx = (decodePoolCursor + i) % DECODE_CANVASES;
      if (!decodePool[idx]) {
        const cv = document.createElement('canvas');
        cv.width = cv.height = TILE_PX;
        decodePool[idx] = { canvas: cv, busy: false };
      }
      if (!decodePool[idx].busy) {
        decodePool[idx].busy = true;
        decodePoolCursor = (idx + 1) % DECODE_CANVASES;
        return decodePool[idx];
      }
    }
    return null;   // 全忙：调用方回退到共享画布
  }

  function releaseDecodeCanvas(slot) {
    if (slot) slot.busy = false;
  }

  function loadWsf3dHeights(z, x, y) {
    const src = blockSource();
    const key = src.dir + '/' + src.layer + '/' + z + '/' + x + '/' + y;
    const hit = lruGet(wsf3dHeightCache, key);
    if (hit !== undefined) return hit;
    const promise = imageFromUrl(src.dir + '/' + src.layer + '/' + z + '/' + x + '/' + y + '.png')
      .then(function (im) {
        const slot = acquireDecodeCanvas();
        const cv = slot ? slot.canvas : decodeCanvas;
        const cx = slot ? cv.getContext('2d', { willReadFrequently: true }) : decodeCtx;
        try {
          cx.clearRect(0, 0, TILE_PX, TILE_PX);
          cx.drawImage(im, 0, 0, TILE_PX, TILE_PX);
          return decodeWsf3dTile(cx.getImageData(0, 0, TILE_PX, TILE_PX).data);
        } finally {
          releaseDecodeCanvas(slot);
        }
      })
      .catch(function () { return null; });
    while (wsf3dHeightCache.size >= WSF3D_CACHE_LIMIT) {
      const oldest = wsf3dHeightCache.keys().next();
      if (oldest.done) break;
      wsf3dHeightCache.delete(oldest.value);
    }
    wsf3dHeightCache.set(key, promise);
    return promise;
  }

  /**
   * 体块高度 → 颜色。
   *
   * 用高度色带而不是"照抄栅格像元颜色"：栅格颜色本身也是高度色带（WSF3D 与
   * 3D-GloBFP 的瓦片都是），直接取色时同一个色带的相邻色阶几乎看不出差别，
   * **体块之间没有高度层次**，于是整片看起来像一块扁平的板。这里按高度重新
   * 映射一次，低矮的偏冷、高的偏暖，相邻体块的层次肉眼可辨。
   */
  const BLOCK_RAMP = [
    [0, 0x3f, 0x6b, 0x8a],
    [10, 0x55, 0x8c, 0x9e],
    [20, 0x7d, 0xa8, 0x8c],
    [40, 0xb9, 0xc0, 0x84],
    [70, 0xd8, 0xb0, 0x66],
    [110, 0xd9, 0x8a, 0x55],
    [180, 0xc2, 0x5b, 0x4e],
    [300, 0x8f, 0x3a, 0x6b],
  ];

  function blockColour(height, alpha) {
    const a = alpha === undefined ? 1.0 : alpha;
    const top = BLOCK_RAMP[BLOCK_RAMP.length - 1];
    if (height <= BLOCK_RAMP[0][0]) {
      const c0 = BLOCK_RAMP[0];
      return new Cesium.Color(c0[1] / 255, c0[2] / 255, c0[3] / 255, a);
    }
    for (let i = 0; i < BLOCK_RAMP.length - 1; i++) {
      const lo = BLOCK_RAMP[i], hi = BLOCK_RAMP[i + 1];
      if (height <= hi[0]) {
        const t = (height - lo[0]) / (hi[0] - lo[0]);
        return new Cesium.Color(
          (lo[1] + (hi[1] - lo[1]) * t) / 255,
          (lo[2] + (hi[2] - lo[2]) * t) / 255,
          (lo[3] + (hi[3] - lo[3]) * t) / 255, a);
      }
    }
    return new Cesium.Color(top[1] / 255, top[2] / 255, top[3] / 255, a);
  }

  function lonlatToTileFrac(lon, lat, z) {
    const n = Math.pow(2, z);
    const x = (lon + 180) / 360 * n;
    const latRad = lat * Math.PI / 180;
    const y = (1 - Math.asinh(Math.tan(latRad)) / Math.PI) / 2 * n;
    return { x: x, y: y };
  }

  function tileFracToLon(x, z) { return x / Math.pow(2, z) * 360 - 180; }

  function tileFracToLat(y, z) {
    const n = Math.PI - 2 * Math.PI * y / Math.pow(2, z);
    return 180 / Math.PI * Math.atan(0.5 * (Math.exp(n) - Math.exp(-n)));
  }

  /**
   * Extrude the WSF3D building-height raster into 3D blocks at a given LOD.
   *
   * The raster is aggregated onto a regular grid over the area of interest and
   * every cell above `minHeight` becomes one extruded polygon. Blocks are
   * batched into a single Cesium.Primitive so the whole district costs one draw
   * call, and each block keeps the colour of the raster cell it came from so
   * the 3D view and the draped overlay read as the same data.
   *
   * @param {object} opts
   *   bbox  [west, south, east, north]
   *   ramp  calibration from loadHeightRamp()。**现在只作为"图层是否就绪"的门禁**：
   *         颜色改由 blockColour(高度) 现算，不再读栅格像元颜色（见 blockColour 的
   *         注释）。保留这个参数是为了让调用方的就绪判断与提示文案不必改动。
   *   lod   index into BUILDING_LOD
   *   zoom  建筑高度瓦片的 zoom
   * @returns {Promise<{primitive, stats}>}
   */
  /**
   * 在瓦片像素坐标系里取一个采样点的高度：取 2x2 邻域**最大**值。
   *
   * 为什么不双线性插值
   * ------------------
   * 双线性会把相邻像元加权平均，在建筑边缘处**造出**源数据里不存在的高度
   * （一栋 8 m 的房子挨着 120 m 的塔楼，会插出 64 m 的中间值），凭空多出一圈
   * 假建筑。取邻域最大值只从真实存在的像元里挑，不会无中生有；再叠加逐格取最大
   * （见下方 heights 的说明）就能保住天际线里最高的那几栋。
   *
   * 对 far/mid 档（step>1）同样成立：跳着读的时候要的是"这一带最高的那栋"，
   * 而不是把周围平均掉。
   */
  function heightAt(data, px, py) {
    if (!data) return 0;
    let best = data[py * TILE_PX + px];
    const rowFree = py + 1 < TILE_PX;
    const colFree = px + 1 < TILE_PX;
    // 同一片瓦片内用下标偏移就是右邻/下邻。px+1 或 py+1 越界时不能读：那已经跨到
    // 相邻瓦片，而这片瓦片的数组里没有下一片的数据，下标会绕回本片的另一端。
    if (colFree) {
      const v = data[py * TILE_PX + px + 1];
      if (v > best) best = v;
    }
    if (rowFree) {
      const v = data[(py + 1) * TILE_PX + px];
      if (v > best) best = v;
    }
    if (rowFree && colFree) {
      const v = data[(py + 1) * TILE_PX + px + 1];
      if (v > best) best = v;
    }
    return best;
  }

  async function buildBuildingBlocks(opts) {
    const bbox = opts.bbox;
    const ramp = opts.ramp;
    const lodIndex = opts.lod === undefined ? BUILDING_LOD.length - 1 : opts.lod;
    const lod = BUILDING_LOD[Math.max(0, Math.min(BUILDING_LOD.length - 1, lodIndex))];
    const grid = lod.grid;
    const step = lod.step;
    const minHeight = lod.minHeight;
    const maxBlocks = lod.maxBlocks;
    const z = opts.zoom || 13;

    const west = bbox[0], south = bbox[1], east = bbox[2], north = bbox[3];

    // 覆盖 AOI 所需的瓦片范围（北西角与南东角）。
    const nw = lonlatToTileFrac(west, north, z);
    const se = lonlatToTileFrac(east, south, z);
    const tx0 = Math.floor(nw.x), tx1 = Math.floor(se.x);
    const ty0 = Math.floor(nw.y), ty1 = Math.floor(se.y);

    const jobs = [];
    for (let tx = tx0; tx <= tx1; tx++) {
      for (let ty = ty0; ty <= ty1; ty++) {
        // loadWsf3dHeights 而不是 loadWsf3dTile：前者按瓦片缓存解码后的高度，
        // 后者每次都重新 getImageData（无头软件光栅化下 16 片约 5 s）。
        jobs.push(loadWsf3dHeights(z, tx, ty).then(function (d) {
          return { tx: tx, ty: ty, data: d };
        }));
      }
    }
    const tiles = await Promise.all(jobs);
    const usable = tiles.filter(function (t) { return !!t.data; });
    if (!usable.length) {
      return { primitive: null, stats: { reason: 'no building-height tiles at z' + z, blocks: 0, lod: lod.key } };
    }

    // 把 AOI 划成 grid x grid 个格子，逐格取落在其中像元的**最大**高度。
    // 取最大而不是平均：一个 CBD 格子里 1 栋 150 m 塔楼 + 20 栋 8 m 建筑，
    // 平均值只有 15 m，塔楼会被抹掉，天际线就没了。
    const heights = new Float32Array(grid * grid);
    const counts = new Uint16Array(grid * grid);

    usable.forEach(function (tile) {
      const d = tile.data;
      // 瓦片在墨卡托下是均匀的，但 AOI 的经纬范围不是，所以逐像元反算
      // 经纬度再定位到格子上，避免高纬度处的偏移。
      for (let py = step >> 1; py < TILE_PX; py += step) {
        const fy = tile.ty + (py + 0.5) / TILE_PX;
        const lat = tileFracToLat(fy, z);
        if (lat < south || lat > north) continue;
        const gy = Math.min(grid - 1, Math.floor((north - lat) / (north - south) * grid));
        for (let px = step >> 1; px < TILE_PX; px += step) {
          const fx = tile.tx + (px + 0.5) / TILE_PX;
          const lon = tileFracToLon(fx, z);
          if (lon < west || lon > east) continue;
          // d 已经是"每像元高度（米）"，无数据为 0——解码在 loadWsf3dHeights 里
          // 做过一次并被缓存，这里不再重复解码颜色。
          const hm = heightAt(tile.data, px, py);
          if (!(hm > 0)) continue;
          const gx = Math.min(grid - 1, Math.floor((lon - west) / (east - west) * grid));
          const idx = gy * grid + gx;
          // 取本格内的**最大**高度：平均会把塔楼抹平（见上方注释）。
          if (hm > heights[idx]) heights[idx] = hm;
          counts[idx] += 1;
        }
      }
    });

    const dLon = (east - west) / grid;
    const dLat = (north - south) / grid;
    const instances = [];
    let skipped = 0;

    for (let gy = 0; gy < grid; gy++) {
      for (let gx = 0; gx < grid; gx++) {
        const idx = gy * grid + gx;
        const n = counts[idx];
        if (!n) continue;
        const h = heights[idx];
        if (h < minHeight) { skipped++; continue; }
        if (instances.length >= maxBlocks) { skipped++; continue; }

        const w = west + gx * dLon;
        const e = w + dLon;
        const nn = north - gy * dLat;
        const ss = nn - dLat;
        const positions = Cesium.Cartesian3.fromDegreesArray([w, nn, e, nn, e, ss, w, ss]);
        // 高度色带：低矮偏冷、高的偏暖，相邻体块的层次肉眼可辨（见 BLOCK_RAMP）。
        const colour = blockColour(h, 1.0);

        instances.push(new Cesium.GeometryInstance({
          geometry: new Cesium.PolygonGeometry({
            polygonHierarchy: new Cesium.PolygonHierarchy(positions),
            height: 0,
            extrudedHeight: h,
            // 不透明体块：PER_INSTANCE_COLOR 的 vertexFormat 与下面 appearance
            // 的 flat:false 必须配对，否则法线不在顶点格式里，光照无从计算。
            vertexFormat: Cesium.PerInstanceColorAppearance.VERTEX_FORMAT,
          }),
          attributes: {
            color: Cesium.ColorGeometryInstanceAttribute.fromColor(colour),
          },
        }));
      }
    }

    if (!instances.length) {
      return { primitive: null, stats: { blocks: 0, skipped: skipped, lod: lod.key, reason: 'all cells below threshold' } };
    }

    const primitive = new Cesium.Primitive({
      geometryInstances: instances,
      // flat: false 是"看起来像立体"的关键。flat: true 让每个几何体的光照法线
      // 恒等于屏幕法线，光照**完全不起作用**，所有面同色——不管挤出多高，读起来
      // 都像一块贴在地面上的色板。改成 false 之后顶面与侧墙各自算法线，开
      // globe.enableLighting（见 app.js）才有明暗，才有体积感。
      appearance: new Cesium.PerInstanceColorAppearance({
        flat: false,
        translucent: false,
        closed: true,
      }),
      // 体块是小而多的几何体，异步编译避免一次卡住；释放由 swapBuildings 负责。
      asynchronous: true,
      allowPicking: false,
    });

    return {
      primitive: primitive,
      stats: {
        blocks: instances.length,
        skipped: skipped,
        grid: grid,
        step: step,
        lod: lod.key,
        zoom: z,
        tiles: usable.length,
        bbox: bbox,
      },
    };
  }

  global.OfflineAssets = {
    loadManifest: loadManifest,
    manifest: manifest,
    instrument: instrument,
    newStats: newStats,
    createImageryProvider: createImageryProvider,
    // Theatre selection: regional imagery/WSF3D rectangles follow the active
    // theatre, so loading a Ladakh case stops the Taiwan Strait box from being
    // declared (and vice versa).
    setTheatre: setTheatre,
    getTheatre: getTheatre,
    theatreBbox: theatreBbox,
    theatreList: theatreList,
    createGlobalImageryProvider: createGlobalImageryProvider,
    createTiandituProvider: createTiandituProvider,
    createTiandituLabelProvider: createTiandituLabelProvider,
    tiandituKey: tiandituKey,
    attachHotspotImagery: attachHotspotImagery,
    terrainMaxLevel: terrainMaxLevel,
    createOverviewImageryProvider: createOverviewImageryProvider,
    attachUrbanImagery: attachUrbanImagery,
    createTerrainProvider: createTerrainProvider,
    createWsf3dOverlayProvider: createWsf3dOverlayProvider,
    attachWsf3dOverlay: attachWsf3dOverlay,
    loadHeightRamp: loadHeightRamp,
    buildBuildingBlocks: buildBuildingBlocks,
    BUILDING_LOD: BUILDING_LOD,
    lodForHeight: lodForHeight,
  };
})(window);
