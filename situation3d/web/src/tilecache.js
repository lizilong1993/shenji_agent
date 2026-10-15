/* situation3d — 瓦片前端缓存
 *
 * 目标：浏览过的地图瓦片不再重复向服务器取。第一次取到后落进 IndexedDB，
 * 之后（包括刷新页面、重新打开浏览器）从本地出。
 *
 * 设计原则：**缓存永远不能挡住渲染。**
 * ------------------------------------
 * 第一版把缓存放在渲染路径上（requestImage 里 await 取图），结果很糟：只要
 * 取图链路的任一环卡住（IndexedDB 打开、fetch、图片解码），Cesium 就会因为
 * 等这一张瓦片而完全不渲染表面瓦片，整幅底图变黑。实测 `surface tiles 0`、
 * `imagery requests 1`——一次请求就把整条流水线堵死了。
 *
 * 所以现在分成两条互不影响的路径：
 *
 *   渲染路径（同步）  requestImage 只查内存 LRU，命中就直接返回；未命中就
 *                     交给 Cesium 自己的加载器。全程没有任何 await。
 *   预热路径（异步）  后台把瓦片取回来、解码、写进内存 LRU 与 IndexedDB。
 *                     它慢、它失败、它卡住，都只影响"下次能不能命中"，不
 *                     影响这一次能不能画出来。
 *
 * 持久化命中靠启动时的**后台预热**搬进内存：页面加载后把 IndexedDB 里的记录
 * 分批读回内存 LRU，于是刷新后再次浏览同一区域时，渲染路径直接从内存命中，
 * 一个网络请求都不发。这样既拿到了跨会话缓存，渲染路径又保持纯同步。
 *
 * 另一层保障是 HTTP 缓存：server/serve.py 对 /tiles/ 返回
 * `Cache-Control: public, max-age=604800, immutable` + ETag，浏览器自身也会
 * 复用已取过的瓦片。应用层缓存与 HTTP 缓存互为补充，而不是互相替代。
 */

'use strict';

(function (global) {
  const DB_NAME = 'situation3d-tiles';
  const DB_VERSION = 1;
  const STORE = 'tiles';

  // 内存层：一张 256x256 RGBA canvas 约 256 KB，512 张约 128 MB。这个上限
  // 同时决定了"刷新后能立刻本地命中"的瓦片数量。
  const MEMORY_LIMIT = 512;
  // 磁盘层：按条目数封顶，避免不同瓦片体积差异导致容量不可控。
  const DISK_LIMIT = 6000;

  const DB_OPEN_TIMEOUT_MS = 1500;
  const IDB_OP_TIMEOUT_MS = 2000;
  // 预热取图的整体上限。预热不阻塞渲染，所以这里超时只是放弃这张瓦片。
  const WARM_BUDGET_MS = 8000;
  // 持久化不可用时的重试间隔。
  const DB_RETRY_INTERVAL_MS = 15000;
  // 启动预热的批量与并发。预热的唯一目的是"下次本地命中"，它比当前视野的
  // 渲染次要得多，所以一次只搬一小批，且必须等底图空闲。
  const PRELOAD_CONCURRENCY = 1;
  const PRELOAD_BATCH = 16;
  // 预热取图的最大并发。浏览器每域只开 6 条连接，预热一旦无节制地并发，
  // Cesium 自己的影像请求就会饿死——底图变黑，而日志里什么错都没有。
  const WARM_CONCURRENCY = 2;

  // 预热队列的**上界**。必须有：每一张影像未命中都会 warm() 一次，而队列只在
  // "底图空闲"时才排空 —— 一旦场景长期不空闲（战区级视图下 Cesium 会为每个表面
  // 瓦片准备成千上万张深层瓦片），队列与 warming 集合就会无界增长。实测
  // 2026-10-15：1939 km 机位下一次页面加载积了 **287,099** 条 URL，全是内存里的
  // 死重量。超上界的一律不排队（也不占 warming），并计入 warmDropped 供诊断。
  const WARM_QUEUE_LIMIT = 1024;
  // 空闲判定从"queued === 0"放宽为"连续这么多毫秒没有新的瓦片活动"：
  // queued 归零是**充分**条件但不是必要条件 —— 只要有一个表面瓦片长期不完成，
  // 它就永远不归零，预热于是永远不开始，形成"越卡越不预热"的死锁。
  const IDLE_SETTLE_MS = 1500;

  const memory = new Map();     // url -> canvas（Map 保持插入序，用作 LRU）
  const warming = new Set();    // 正在预热的 url，避免重复取
  const stats = {
    memoryHits: 0,
    diskHits: 0,
    network: 0,
    errors: 0,
    stored: 0,
    evicted: 0,
    preloaded: 0,
    warmDropped: 0,             // 因队列满而未排队的**尝试**数：同一张瓦片每次重试都会再计一次
                                // （被丢弃的 URL 不进 warming 集合，所以不记忆，这是有意的：
                                //  记住它们等于把刚刚才裁掉的内存又长回来）
    // 内存里当前的瓦片数。放在 stats 上是为了让诊断浮层能直接读，而不必
    // 走一次异步的 snapshot()。
    get memoryCount() { return memory.size; },
    get memoryLimit() { return MEMORY_LIMIT; },
    get warmingCount() { return warming.size; },
    get warmQueueCount() { return warmQueue.length; },
    get idleFlag() { return sceneIdle; },
    lastError: null,
    errorLog: [],
  };

  let dbPromise = null;
  let diskCount = null;
  let persistentUnavailable = false;
  let dbRetryAfter = 0;
  let preloadStarted = false;

  // ----------------------------------------------------------------------
  // 内存 LRU
  // ----------------------------------------------------------------------

  function memoryGet(url) {
    if (!memory.has(url)) return null;
    const canvas = memory.get(url);
    memory.delete(url);
    memory.set(url, canvas);
    return canvas;
  }

  function memoryPut(url, canvas) {
    if (memory.has(url)) memory.delete(url);
    memory.set(url, canvas);
    while (memory.size > MEMORY_LIMIT) {
      const oldest = memory.keys().next();
      if (oldest.done) break;
      memory.delete(oldest.value);
      stats.evicted++;
    }
  }

  // ----------------------------------------------------------------------
  // IndexedDB
  // ----------------------------------------------------------------------

  /**
   * Open the persistent store, giving up quickly if it does not cooperate.
   *
   * Even though nothing on the render path waits for this, a store that never
   * opens would keep every preload attempt parked, so the open is bounded and
   * retried on a cooldown rather than once per tile.
   */
  function openDb() {
    if (dbPromise) return dbPromise;
    if (persistentUnavailable && Date.now() < dbRetryAfter) return Promise.resolve(null);

    const attempt = new Promise(function (resolve) {
      if (!global.indexedDB) { resolve(null); return; }
      let req;
      try {
        req = indexedDB.open(DB_NAME, DB_VERSION);
      } catch (e) {
        resolve(null);
        return;
      }
      req.onupgradeneeded = function () {
        const db = req.result;
        if (!db.objectStoreNames.contains(STORE)) {
          const store = db.createObjectStore(STORE, { keyPath: 'url' });
          // 按写入时间建索引，淘汰时取最旧的。
          store.createIndex('at', 'at');
        }
      };
      req.onsuccess = function () { resolve(req.result); };
      req.onerror = function () { resolve(null); };
      req.onblocked = function () { resolve(null); };
    });

    const timeout = new Promise(function (resolve) {
      setTimeout(function () { resolve(null); }, DB_OPEN_TIMEOUT_MS);
    });

    dbPromise = Promise.race([attempt, timeout]).then(function (db) {
      if (!db) {
        dbPromise = null;
        persistentUnavailable = true;
        dbRetryAfter = Date.now() + DB_RETRY_INTERVAL_MS;
      } else {
        persistentUnavailable = false;
      }
      return db;
    });
    return dbPromise;
  }

  function tx(db, mode) { return db.transaction(STORE, mode).objectStore(STORE); }

  function withTimeout(promise, fallback) {
    return Promise.race([
      promise,
      new Promise(function (resolve) {
        setTimeout(function () { resolve(fallback); }, IDB_OP_TIMEOUT_MS);
      }),
    ]);
  }

  function idbGet(db, url) {
    return new Promise(function (resolve) {
      const req = tx(db, 'readonly').get(url);
      req.onsuccess = function () { resolve(req.result || null); };
      req.onerror = function () { resolve(null); };
    });
  }

  function idbPut(db, record) {
    return new Promise(function (resolve) {
      const req = tx(db, 'readwrite').put(record);
      req.onsuccess = function () { resolve(true); };
      req.onerror = function () { resolve(false); };
    });
  }

  function idbCount(db) {
    return new Promise(function (resolve) {
      const req = tx(db, 'readonly').count();
      req.onsuccess = function () { resolve(req.result); };
      req.onerror = function () { resolve(0); };
    });
  }

  function idbClear(db) {
    return new Promise(function (resolve) {
      const req = tx(db, 'readwrite').clear();
      req.onsuccess = function () { resolve(true); };
      req.onerror = function () { resolve(false); };
    });
  }

  /** 按写入时间淘汰最旧的条目，直到回到上限以内。 */
  function idbTrim(db) {
    return new Promise(function (resolve) {
      const store = tx(db, 'readwrite');
      const index = store.index('at');
      const over = (diskCount === null ? 0 : diskCount) - DISK_LIMIT;
      if (over <= 0) { resolve(0); return; }
      let removed = 0;
      const cursorReq = index.openCursor();
      cursorReq.onsuccess = function () {
        const cursor = cursorReq.result;
        if (!cursor || removed >= over) {
          if (diskCount !== null) diskCount -= removed;
          resolve(removed);
          return;
        }
        cursor.delete();
        removed++;
        cursor.continue();
      };
      cursorReq.onerror = function () { resolve(removed); };
    });
  }

  // ----------------------------------------------------------------------
  // 解码
  // ----------------------------------------------------------------------

  /**
   * Decode a blob into a canvas.
   *
   * createImageBitmap is the fast path, but it is not available in every
   * headless/software-rasteriser configuration, so the Image + object URL route
   * is kept as a fallback rather than losing the tile.
   */
  function blobToCanvas(blob) {
    if (global.createImageBitmap) {
      return createImageBitmap(blob).then(function (bitmap) {
        const canvas = document.createElement('canvas');
        canvas.width = bitmap.width;
        canvas.height = bitmap.height;
        canvas.getContext('2d').drawImage(bitmap, 0, 0);
        bitmap.close();
        return canvas;
      }).catch(function (e) {
        noteError('createImageBitmap: ' + e.message);
        return blobToCanvasViaImage(blob);
      });
    }
    return blobToCanvasViaImage(blob);
  }

  function blobToCanvasViaImage(blob) {
    return new Promise(function (resolve, reject) {
      const objectUrl = URL.createObjectURL(blob);
      const im = new Image();
      im.onload = function () {
        const canvas = document.createElement('canvas');
        canvas.width = im.naturalWidth || 256;
        canvas.height = im.naturalHeight || 256;
        canvas.getContext('2d').drawImage(im, 0, 0);
        URL.revokeObjectURL(objectUrl);
        resolve(canvas);
      };
      im.onerror = function () {
        URL.revokeObjectURL(objectUrl);
        reject(new Error('image decode failed'));
      };
      im.src = objectUrl;
    });
  }

  function noteError(message) {
    stats.lastError = message;
    if (stats.errorLog.length < 5) stats.errorLog.push(message);
  }

  // ----------------------------------------------------------------------
  // 预热路径：取图、解码、入内存、落盘
  // ----------------------------------------------------------------------

  async function fetchTile(url, db) {
    stats.network++;
    const response = await fetch(url);
    if (!response.ok) {
      const err = new Error('HTTP ' + response.status + ' ' + url);
      err.status = response.status;
      throw err;
    }
    const blob = await response.blob();
    if (db) {
      idbPut(db, { url: url, blob: blob, at: Date.now() }).then(function (ok) {
        if (!ok) return;
        stats.stored++;
        if (diskCount !== null) diskCount++;
        if (diskCount !== null && diskCount > DISK_LIMIT) idbTrim(db);
      });
    }
    return blob;
  }

  /**
   * 取一张瓦片并写入内存（以及尽可能落盘）。
   *
   * 只被预热路径调用，因此可以放心地 await；超时只是放弃这张瓦片。
   */
  async function loadTile(url) {
    const db = await openDb();

    if (db) {
      const record = await withTimeout(idbGet(db, url), null);
      if (record && record.blob) {
        try {
          const canvas = await blobToCanvas(record.blob);
          stats.diskHits++;
          memoryPut(url, canvas);
          return canvas;
        } catch (e) {
          stats.errors++;
        }
      }
    }

    let lastError = null;
    for (let i = 0; i < 2; i++) {
      try {
        const blob = await fetchTile(url, db);
        const canvas = await blobToCanvas(blob);
        memoryPut(url, canvas);
        return canvas;
      } catch (e) {
        lastError = e;
        noteError((e && e.status ? 'HTTP ' + e.status : (e && e.message) || String(e)) +
                  ' ' + url);
        if (e.status === 404) break;
        await new Promise(function (r) { setTimeout(r, 200 * (i + 1)); });
      }
    }
    stats.errors++;
    throw lastError || new Error('tile fetch failed: ' + url);
  }

  // 预热队列。只有底图空闲（tileLoadProgressEvent 归零，或安静到 IDLE_SETTLE_MS）
  // 时才会排空，保证预热永远不和当前视野的渲染抢连接。
  const warmQueue = [];
  let warmInflight = 0;
  let sceneIdle = false;
  let lastTileActivity = Date.now();

  /**
   * 通知缓存当前底图是否已经加载完。
   *
   * 由 app.js 的 tileLoadProgressEvent 驱动。这是预热与渲染之间的调度闸门：
   * 底图还在加载时，任何额外请求都是净损失。
   */
  function setIdle(idle) {
    if (!idle) lastTileActivity = Date.now();   // 有瓦片活动：重新计时
    sceneIdle = !!idle;
    if (!sceneIdle) return;
    lastTileActivity = Date.now();
    drainWarmQueue();
    if (!preloadStarted) {
      preloadStarted = true;
      preloadBatch().catch(function () {});
    }
  }

  // 兜底闸门：即使 setIdle(true) 一直没来（表面瓦片长期不完成 → queued 永不归零），
  // 只要安静了 IDLE_SETTLE_MS 就允许排空。每秒只读一个时间戳，代价可忽略；
  // 真正取图的并发仍受 WARM_CONCURRENCY 限制，所以它不会反噬渲染。
  setInterval(function () {
    if (!sceneIdle && Date.now() - lastTileActivity >= IDLE_SETTLE_MS) {
      sceneIdle = true;
      drainWarmQueue();
    }
  }, 1000);

  /** 后台预热一张瓦片。同一张瓦片不会重复取，失败静默。 */
  function warm(url) {
    if (memory.has(url) || warming.has(url)) return;
    if (warmQueue.length >= WARM_QUEUE_LIMIT) {
      // 队列已满：**不排队、也不占 warming**。丢掉的是"下次能不能命中"，
      // 不影响这一帧能不能画出来 —— 这正是缓存设计原则里那条边界。
      stats.warmDropped++;
      return;
    }
    warming.add(url);
    warmQueue.push(url);
    if (sceneIdle) drainWarmQueue();
  }

  function drainWarmQueue() {
    while (sceneIdle && warmInflight < WARM_CONCURRENCY && warmQueue.length) {
      runWarm(warmQueue.shift());
    }
  }

  function runWarm(url) {
    warmInflight++;
    let timer = null;
    const budget = new Promise(function (_resolve, reject) {
      timer = setTimeout(function () {
        reject(new Error('warm budget exceeded'));
      }, WARM_BUDGET_MS);
    });

    Promise.race([loadTile(url), budget])
      .catch(function () { /* 预热失败不影响任何东西 */ })
      .then(function () {
        if (timer) clearTimeout(timer);
        warming.delete(url);
        warmInflight--;
        // 队列排空后停手，等下一次空闲通知再继续，避免持续占用连接。
        if (warmInflight === 0 && !warmQueue.length) sceneIdle = false;
        drainWarmQueue();
      });
  }

  // ----------------------------------------------------------------------
  // 启动时把磁盘缓存搬回内存
  // ----------------------------------------------------------------------

  /**
   * 分批把 IndexedDB 里的瓦片读回内存 LRU。
   *
   * 这是"刷新后仍然本地命中"的实现：渲染路径只查内存，所以持久化的内容
   * 必须在后台先搬进来。限量、限并发，且完全在后台进行。
   */
  let preloadQueue = null;

  /**
   * 把 IndexedDB 里的瓦片搬回内存 LRU，一次一小批。
   *
   * 这是"刷新后仍然本地命中"的实现：渲染路径只查内存，所以持久化的内容
   * 必须先进内存。但解码 canvas 是实打实的 CPU 开销，在软件光栅化环境下
   * 会直接拖慢渲染，所以：
   *   - 只在底图空闲时进行（由 app.js 的 tileLoadProgressEvent 驱动）；
   *   - 每次空闲只搬 PRELOAD_BATCH 张，串行解码；
   *   - 任何一批失败都不影响渲染。
   */
  async function preloadBatch() {
    const db = await openDb();
    if (!db) return;

    if (preloadQueue === null) {
      preloadQueue = await withTimeout(new Promise(function (resolve) {
        const out = [];
        const cursorReq = tx(db, 'readonly').index('at').openCursor(null, 'prev');
        cursorReq.onsuccess = function () {
          const cursor = cursorReq.result;
          if (!cursor || out.length >= MEMORY_LIMIT) { resolve(out); return; }
          out.push(cursor.value.url);
          cursor.continue();
        };
        cursorReq.onerror = function () { resolve(out); };
      }), []);
      diskCount = await withTimeout(idbCount(db), 0);
    }

    let moved = 0;
    while (preloadQueue.length && moved < PRELOAD_BATCH) {
      const url = preloadQueue.shift();
      if (memory.has(url)) continue;
      try {
        const record = await withTimeout(idbGet(db, url), null);
        if (!record || !record.blob) continue;
        memoryPut(url, await blobToCanvas(record.blob));
        stats.preloaded++;
        moved++;
      } catch (e) {
        // 单张坏瓦片跳过即可。
      }
    }
    return moved;
  }

  // ----------------------------------------------------------------------
  // 接到 Cesium provider 上
  // ----------------------------------------------------------------------

  /**
   * 用缓存包装 provider 的 requestImage。
   *
   * 注意这里的形状：**没有任何 await**。命中内存就返回 canvas，未命中就立刻
   * 走 Cesium 原生加载，同时后台预热。渲染能不能画出来，与缓存是否正常无关。
   */
  function attach(provider, opts) {
    const template = opts.urlTemplate;
    const original = provider.requestImage.bind(provider);

    provider.requestImage = function (x, y, level, request) {
      if (!template) return original(x, y, level, request);

      const url = template
        .replace('{z}', level)
        .replace('{x}', x)
        .replace('{y}', y);

      const hit = memoryGet(url);
      if (hit) {
        stats.memoryHits++;
        // 必须包成 Promise：Cesium 1.145 的 ImageryLayer._requestImagery().doRequest()
        // 对 requestImage() 的返回值**直接调用 .then()**（只有 defined() 守卫，没有
        // Promise.resolve() 包装）。同步返回 canvas 会让它在渲染循环内抛
        // "a.then is not a function"，CesiumWidget 随即判定渲染已停止、整幅底图变黑。
        return Promise.resolve(hit);
      }

      // 未命中：原生路径负责这一次的渲染，预热负责下一次的命中。
      warm(url);
      return original(x, y, level, request);
    };

    provider.__tileCache = true;
    return provider;
  }

  /** 当前缓存概况，用于界面展示与诊断。 */
  async function snapshot() {
    const db = await openDb();
    if (db && diskCount === null) {
      diskCount = await withTimeout(idbCount(db), null);
    }
    let quota = null;
    if (navigator.storage && navigator.storage.estimate) {
      try { quota = await navigator.storage.estimate(); } catch (e) { quota = null; }
    }
    return {
      memory: memory.size,
      memoryLimit: MEMORY_LIMIT,
      disk: diskCount,
      diskLimit: DISK_LIMIT,
      warming: warming.size,
      preloaded: stats.preloaded,
      queued: warmQueue.length,
      idle: sceneIdle,
      usageBytes: quota ? quota.usage : null,
      quotaBytes: quota ? quota.quota : null,
      hits: stats.memoryHits + stats.diskHits,
      memoryHits: stats.memoryHits,
      diskHits: stats.diskHits,
      network: stats.network,
      errors: stats.errors,
      evicted: stats.evicted,
      persistent: !persistentUnavailable,
      lastError: stats.lastError,
      errorLog: stats.errorLog,
    };
  }

  async function clear() {
    memory.clear();
    const db = await openDb();
    if (db) {
      await idbClear(db);
      diskCount = 0;
    }
    stats.memoryHits = stats.diskHits = stats.network = stats.errors = 0;
    stats.stored = stats.evicted = stats.preloaded = 0;
    warmQueue.length = 0;
    warming.clear();
    stats.lastError = null;
    stats.errorLog = [];
    return true;
  }

  global.TileCache = {
    attach: attach,
    setIdle: setIdle,
    preloadBatch: preloadBatch,
    warm: warm,

    snapshot: snapshot,
    clear: clear,
    stats: stats,
  };
})(window);
