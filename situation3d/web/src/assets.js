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

  /**
   * 全球底图：任意位置都有影像，避免把镜头拉远或平移出去后只剩底色。
   *
   * 只到 z6（约 0.7 km/px）——再往下铺满全球会多出上万张瓦片，而战区与热点另有
   * 更高层级的图层覆盖。rectangle 取全球，这是唯一一个不限制范围的影像层。
   */
  function createGlobalImageryProvider() {
    const m = manifest();
    const provider = new Cesium.UrlTemplateImageryProvider({
      url: IMAGERY_URL,
      tilingScheme: new Cesium.WebMercatorTilingScheme(),
      minimumLevel: 0,
      maximumLevel: (m.imagery_global && m.imagery_global.max_zoom) || 6,
      credit: 'Esri World Imagery (global)',
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

  function createImageryProvider() {
    const m = manifest();
    const provider = new Cesium.UrlTemplateImageryProvider({
      url: IMAGERY_URL,
      tilingScheme: new Cesium.WebMercatorTilingScheme(),
      minimumLevel: m.imagery.min_zoom || 0,
      maximumLevel: m.imagery.max_zoom || 11,
      // 关键：声明真实覆盖范围，域外不再发请求。
      rectangle: aoiRectangle(m.bbox),
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

    for (; lv >= 0; lv--) {
      const src = await loadTerrainTile(lv, tx, ty);
      if (src) {
        const span = Math.pow(2, level - lv);       // 该瓦片跨越多少个 level 级瓦片
        const ox = (x - tx * span) / span;
        const oy = (y - ty * span) / span;
        return subSample(src, ox, oy, 1 / span);
      }
      tx >>= 1;
      ty >>= 1;
    }
    return new Float32Array(HEIGHTMAP_PX * HEIGHTMAP_PX);
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
  // World Settlement Footprint 3D
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
      rectangle: aoiRectangle(m.bbox),
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
   *   grid 24  → 约 540 m 体块，远看只看轮廓
   *   grid 48  → 约 270 m
   *   grid 96  → 约 135 m
   *   grid 144 → 约 90 m，与 WSF3D 原生分辨率相当
   */
  const BUILDING_LOD = [
    { key: 'far',   maxHeight: 400000, grid: 24,  step: 8, minHeight: 6, maxBlocks: 900 },
    { key: 'mid',   maxHeight: 150000, grid: 48,  step: 4, minHeight: 5, maxBlocks: 2200 },
    { key: 'near',  maxHeight: 60000,  grid: 96,  step: 2, minHeight: 4, maxBlocks: 5000 },
    { key: 'close', maxHeight: Infinity, grid: 144, step: 1, minHeight: 4, maxBlocks: 9000 },
  ];

  function lodForHeight(heightM) {
    for (let i = 0; i < BUILDING_LOD.length; i++) {
      if (heightM <= BUILDING_LOD[i].maxHeight) return i;
    }
    return BUILDING_LOD.length - 1;
  }

  const wsf3dTileCache = new Map();

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
   *   ramp  calibration from loadHeightRamp()
   *   lod   index into BUILDING_LOD
   *   zoom  WSF3D tile zoom to read
   * @returns {Promise<{primitive, stats}>}
   */
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
        jobs.push(loadWsf3dTile(z, tx, ty).then(function (d) {
          return { tx: tx, ty: ty, data: d };
        }));
      }
    }
    const tiles = await Promise.all(jobs);
    const usable = tiles.filter(function (t) { return !!t.data; });
    if (!usable.length) {
      return { primitive: null, stats: { reason: 'no WSF3D tiles at z' + z, blocks: 0, lod: lod.key } };
    }

    // 把 AOI 划成 grid x grid 个格子，逐格取落在其中的 WSF3D 像元的平均高度
    // 与代表色。step 控制抽样密度：粗 LOD 少读像素，出图快得多。
    const heights = new Float32Array(grid * grid);
    const counts = new Uint16Array(grid * grid);
    const colours = new Array(grid * grid);

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
          const off = (py * TILE_PX + px) * 4;
          const r = d[off], g = d[off + 1], b = d[off + 2];
          if (ramp.isNoData(r, g, b)) continue;
          const gx = Math.min(grid - 1, Math.floor((lon - west) / (east - west) * grid));
          const idx = gy * grid + gx;
          heights[idx] += ramp.decode(r, g, b);
          counts[idx] += 1;
          if (!colours[idx]) colours[idx] = 'rgb(' + r + ',' + g + ',' + b + ')';
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
        const h = heights[idx] / n;
        if (h < minHeight) { skipped++; continue; }
        if (instances.length >= maxBlocks) { skipped++; continue; }

        const w = west + gx * dLon;
        const e = w + dLon;
        const nn = north - gy * dLat;
        const ss = nn - dLat;
        const positions = Cesium.Cartesian3.fromDegreesArray([w, nn, e, nn, e, ss, w, ss]);
        const colour = Cesium.Color.fromCssColorString(colours[idx] || '#8fa9c4');

        instances.push(new Cesium.GeometryInstance({
          geometry: new Cesium.PolygonGeometry({
            polygonHierarchy: new Cesium.PolygonHierarchy(positions),
            height: 0,
            extrudedHeight: h,
            vertexFormat: Cesium.PerInstanceColorAppearance.VERTEX_FORMAT,
          }),
          attributes: {
            color: Cesium.ColorGeometryInstanceAttribute.fromColor(colour.withAlpha(0.92)),
          },
        }));
      }
    }

    if (!instances.length) {
      return { primitive: null, stats: { blocks: 0, skipped: skipped, lod: lod.key, reason: 'all cells below threshold' } };
    }

    const primitive = new Cesium.Primitive({
      geometryInstances: instances,
      appearance: new Cesium.PerInstanceColorAppearance({
        flat: true,
        translucent: false,
        closed: true,
      }),
      asynchronous: true,
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
    createGlobalImageryProvider: createGlobalImageryProvider,
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
