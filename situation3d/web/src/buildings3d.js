/**
 * 在线三维实景建筑层（Google Photorealistic 3D Tiles，经 Cesium ion 转发）。
 *
 * 为什么单独一个文件
 * ------------------
 * 本项目的建筑体块是**离线自产**的（3D-GloBFP 逐建筑高度 → z13 PNG → 挤出体块，
 * 见 assets.js 的 buildBuildingBlocks）。效果上在线实景明显更好，所以按要求把它
 * 集成进来并设为默认；但离线那套**必须完整保留**，两者是并列的两个图层，不是一个
 * 替换另一个。放在独立文件里，是为了让"在线"这条路径的失败面收敛在这个闭包里，
 * 不让它污染离线体块那条已经调通的链路。
 *
 * 三条不能绕过的硬约束（都实测过，不是推理）
 * ------------------------------------------
 * 1. **必须走 ion。** `Cesium3DTileset.fromIonAssetId(2275207)` 时 Google 的 API key
 *    由 ion 下发（实测 endpoint 返回的 url 里带 `key=AIzaSy...`，那是 ion 的 key，
 *    不是我们的）。我们**不持有** Google key，也不该持有。
 * 2. **必须关掉 globe。** Google 实景瓦片与本机离线 DEM 在高缩放级别几何冲突
 *    （gods-eye-view 也是同样处理）。关掉后本机 DEM 与离线影像在该模式下不再显示，
 *    这是这个图层的固有代价，必须让用户看得见（见 app.js 的提示文案）。
 * 3. **无法离线缓存。** ion 对该资产的回答是 `"archivable":false, "exportable":false`；
 *    瓦片 URL 带一次性 `?session=`（去掉直接 403）；响应头是
 *    `Cache-Control: private, must-revalidate`；且其使用条款绑定 Google Map Tiles API
 *    Policy（禁止预取/缓存/存储）。所以这个图层**永远是在线的**，断网必然回落。
 *    不要试图把它的瓦片写进 web/tiles/——session 会失效，只会拿到 403 空瓦片。
 *
 * 状态机
 * ------
 *   idle          未启用
 *   unconfigured  服务端没有令牌，未发起任何网络请求
 *   loading       正在建 tileset
 *   live          正常显示
 *   failed        建 tileset 失败或瓦片加载出错，调用方应回落到离机体块
 *
 * 依赖：window 上的 Cesium 与 fetch。无 ES module（index.html 逐个 <script src>）。
 */
(function () {
  'use strict';

  var ASSET_ID = 2275207;

  /**
   * 等 ion 建 tileset 的上界。ion 的 endpoint 调用本身很快（实测约 1 s），
   * 真正可能挂住的是随后拉 Google root.json 那一步，所以这个值取得比较紧：
   * 12 s 还不回就是链路上某个环节不通，早点回落比让用户干等好。
   */
  var LOAD_TIMEOUT_MS = 12000;

  var state = {
    viewer: null,
    tileset: null,
    enabled: false,
    phase: 'idle',
    detail: null,
    /** 建 tileset 期间被再次调用时用来丢弃过期结果。 */
    epoch: 0,
    token: null,
    tokenChecked: false,
    globeWasShowing: true,
  };

  var listeners = [];
  function emit() {
    for (var i = 0; i < listeners.length; i++) {
      try { listeners[i](stateInfo()); } catch (e) { /* 监听方自己出错不影响图层 */ }
    }
  }

  function destroyed(v) {
    return !v || (typeof v.isDestroyed === 'function' && v.isDestroyed());
  }

  function stateInfo() {
    return {
      enabled: state.enabled,
      phase: state.phase,
      detail: state.detail,
      assetId: ASSET_ID,
      hasTileset: !!state.tileset,
      live: state.phase === 'live',
      /** 调用方据此决定是否回落到离机体块。 */
      shouldFallback: state.enabled &&
        (state.phase === 'unconfigured' || state.phase === 'failed'),
    };
  }

  function setPhase(phase, detail) {
    state.phase = phase;
    state.detail = detail || null;
    emit();
  }

  // -- 令牌 ---------------------------------------------------------------

  /**
   * 取 ion 访问令牌。
   *
   * 顺序：localStorage（用户在界面上填过）→ 服务端 /api/ion/token。
   * 服务端的值来自 web/data/ion.json，该文件被 .gitignore 覆盖，不进仓库。
   * 令牌只放在内存与 localStorage，**不写日志、不回显**。
   */
  function fetchToken() {
    if (state.tokenChecked) return Promise.resolve(state.token);
    state.tokenChecked = true;

    var cached = null;
    try { cached = window.localStorage.getItem('situation3d.ionToken'); } catch (e) { cached = null; }
    if (cached) {
      state.token = cached;
      return Promise.resolve(cached);
    }

    return fetch('/api/ion/token', { cache: 'no-store' })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (j) {
        state.token = (j && j.token) ? String(j.token) : null;
        return state.token;
      })
      .catch(function () {
        // 服务端没有这个接口（例如用别的静态服务器）也算"没配置"，不抛。
        state.token = null;
        return null;
      });
  }

  /** 界面里保存令牌用：先落到服务端，再落到 localStorage。 */
  function saveToken(token) {
    var t = String(token || '').trim();
    if (!t) return Promise.reject(new Error('令牌为空'));
    return fetch('/api/ion/token', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token: t }),
    }).then(function (r) {
      if (!r.ok) {
        return r.json().catch(function () { return {}; }).then(function (j) {
          throw new Error((j && j.error) || ('服务端拒绝（HTTP ' + r.status + '）'));
        });
      }
      try { window.localStorage.setItem('situation3d.ionToken', t); } catch (e) { /* 隐私模式下会失败 */ }
      state.token = t;
      state.tokenChecked = true;
      return t;
    });
  }

  /** 清掉本地缓存的令牌，让下一次重新走服务端。 */
  function forgetToken() {
    try { window.localStorage.removeItem('situation3d.ionToken'); } catch (e) { /* ignore */ }
    state.token = null;
    state.tokenChecked = false;
  }

  // -- 生命周期 -----------------------------------------------------------

  function init(viewer) {
    state.viewer = viewer;
    return stateInfo();
  }

  /**
   * 给一个 promise 加超时上界。
   *
   * 为什么必须有这个
   * ----------------
   * 实测：容器里 `api.cesium.com` 1 秒内就回，但 **`tile.googleapis.com` 完全
   * 不可达**。此时 ion 的 endpoint 调用是成功的，Cesium 接着去拉 Google 的
   * root.json，而 `fromIonAssetId` 要等那个 JSON 回来才 resolve——于是它
   * **既不 resolve 也不 reject，就那么挂着**（实测 25 s 后 phase 仍为 loading，
   * pageErrors 为空）。
   *
   * 这种静默挂起比抛异常难查得多：没有报错、没有提示、图层永远转圈。所以这里给
   * 一个硬上界，到点当失败处理，交给调用方回落到离机体块。
   */
  function withTimeout(promise, ms, message) {
    var timer = null;
    var guard = new Promise(function (_resolve, reject) {
      timer = setTimeout(function () { reject(new Error(message)); }, ms);
    });
    function done(v) { if (timer) clearTimeout(timer); return v; }
    return Promise.race([promise, guard]).then(done, function (e) {
      done();
      throw e;
    });
  }

  /**
   * 建 tileset 并挂到场景上。
   *
   * 注意 `Ion.defaultAccessToken` 必须在 `fromIonAssetId` **之前**设好——ion 在拿到
   * 资产 endpoint 时就要用它做鉴权，晚设会直接 401。
   */
  function createTileset(epoch) {
    var viewer = state.viewer;
    if (destroyed(viewer)) {
      setPhase('failed', '视图已销毁');
      return Promise.resolve(null);
    }

    Cesium.Ion.defaultAccessToken = state.token;

    var ready = Cesium.Cesium3DTileset.fromIonAssetId(ASSET_ID, {
      // Google 与 Cesium 的署名必须留在屏幕上，这是使用条款要求，不要关。
      showCreditsOnScreen: true,
    });

    return withTimeout(ready, LOAD_TIMEOUT_MS,
      '等待 Cesium ion 超时（' + Math.round(LOAD_TIMEOUT_MS / 1000) +
      ' 秒）：本机可能无法访问 Google 瓦片服务'
    ).then(function (tileset) {
      // epoch 变了说明期间用户又关掉/重开过，这个结果已经过期，直接丢掉，
      // 否则会留下一个没人管的 primitive 永远挂在场景里。
      if (epoch !== state.epoch) {
        try { tileset.destroy(); } catch (e) { /* ignore */ }
        return null;
      }

      // 瓦片级失败不致命（个别 404 不该让整个图层消失），但要把状态记下来，
      // 否则用户只看到一片空洞而不知道为什么。
      tileset.tileFailed.addEventListener(function (err) {
        if (state.phase === 'live') {
          state.detail = '部分瓦片加载失败：' + (err && err.message ? err.message : '未知原因');
          emit();
        }
      });

      viewer.scene.primitives.add(tileset);
      state.tileset = tileset;

      // 实景瓦片与本机离线 DEM 几何冲突，必须让 globe 让位。
      // 记录原值，关掉这个图层时恢复。
      state.globeWasShowing = viewer.scene.globe.show;
      viewer.scene.globe.show = false;

      // globe 关掉后地面氛围光失效，海面会读成纯黑；给一个接近海色的背景兜底，
      // 保证"关掉 globe"不至于看起来像渲染坏了。
      viewer.scene.globe.showGroundAtmosphere = false;
      viewer.scene.backgroundColor = Cesium.Color.fromCssColorString('#0a1520');

      setPhase('live', 'Google Photorealistic 3D Tiles（在线，不可离线缓存）');
      return tileset;
    }).catch(function (e) {
      if (epoch !== state.epoch) return null;
      var msg = (e && e.message) ? e.message : String(e);
      // 401/403 基本就是令牌无效或过期，单独说清楚，省得用户去查网络。
      if (/401|403|token|Unauthorized|Forbidden/i.test(msg)) {
        setPhase('failed', '令牌被拒绝（' + msg + '）');
      } else {
        setPhase('failed', '无法加载在线实景：' + msg);
      }
      return null;
    });
  }

  function removeTileset() {
    var viewer = state.viewer;
    if (state.tileset && !destroyed(viewer)) {
      try {
        viewer.scene.primitives.remove(state.tileset);
      } catch (e) { /* 已经不在集合里就算了 */ }
    }
    state.tileset = null;
    if (!destroyed(viewer)) {
      // 恢复本机 DEM：这是"关掉在线层就能回到离线能力"的关键一步。
      viewer.scene.globe.show = state.globeWasShowing !== false;
      viewer.scene.globe.showGroundAtmosphere = true;
      viewer.scene.backgroundColor = Cesium.Color.BLACK;
    }
  }

  function setEnabled(on) {
    state.enabled = !!on;

    if (!on) {
      state.epoch++;
      removeTileset();
      setPhase('idle', '在线三维实景已关闭');
      return Promise.resolve(stateInfo());
    }

    var epoch = ++state.epoch;

    if (state.tileset) {
      setPhase('live', 'Google Photorealistic 3D Tiles（在线，不可离线缓存）');
      return Promise.resolve(stateInfo());
    }

    setPhase('loading', '正在连接 Cesium ion…');

    return fetchToken().then(function (token) {
      if (epoch !== state.epoch) return null;
      if (!token) {
        setPhase('unconfigured', '未配置 Cesium ion 令牌（见图层管理里的输入框）');
        return null;
      }
      return createTileset(epoch);
    }).then(function () {
      return stateInfo();
    });
  }

  /** 取证用：把这条链路上的每个短路条件逐条报出来。 */
  function debug() {
    var viewer = state.viewer;
    var live = !destroyed(viewer);
    return {
      enabled: state.enabled,
      phase: state.phase,
      detail: state.detail,
      assetId: ASSET_ID,
      hasToken: !!state.token,
      tokenFromLocalStorage: (function () {
        try { return !!window.localStorage.getItem('situation3d.ionToken'); } catch (e) { return null; }
      })(),
      hasTileset: !!state.tileset,
      globeShowing: live ? viewer.scene.globe.show : null,
      primitiveCount: live ? viewer.scene.primitives.length : null,
      tilesetReady: state.tileset ? !!state.tileset.ready : null,
      tilesetShow: state.tileset ? !!state.tileset.show : null,
      epoch: state.epoch,
    };
  }

  window.Buildings3D = {
    init: init,
    setEnabled: setEnabled,
    state: stateInfo,
    debug: debug,
    saveToken: saveToken,
    forgetToken: forgetToken,
    onStateChange: function (fn) { listeners.push(fn); },
    ASSET_ID: ASSET_ID,
  };
})();