/* 影像数据源：只读展示在线影像源、Sentinel-2 场景与案例缓存。
 *
 * 这里不发起获取、不切换底图——获取仍走 scripts/resource.py，底图切换仍由
 * 图层管理控制。本面板回答的是“系统现在能用哪些影像源、哪些案例已经缓存”，
 * 所以只读即可。
 *
 * 在线源目录与 resources/sources.py 的 IMAGERY_SOURCES 保持一致；案例 recipe
 * 与 resources/case_imagery.py 的 CASES 保持一致。若两处改动，记得同步这里
 * 的展示文案（真正的获取逻辑始终以 Python 侧为准）。
 */
(function (global) {
  'use strict';

  var state = { loaded: false, loading: false, health: null, error: null };

  var XYZ_SOURCES = [
    {
      id: 'esri', label: 'Esri World Imagery', method: 'XYZ',
      maxZoom: 19, priority: 10, quality: 'measured',
      license: 'Esri 使用条款（需署名，作业使用前复核）',
      note: '全球高分辨率影像，作为默认在线瓦片源之一。'
    },
    {
      id: 'eox-s2', label: 'EOX Sentinel-2 cloudless', method: 'XYZ',
      maxZoom: 14, priority: 20, quality: 'measured',
      license: 'CC BY-NC-SA 4.0（仅限非商业用途）',
      note: '2020 年 Sentinel-2 无云合成，适合作为无云兜底底图。'
    },
    {
      id: 'gibs-modis', label: 'NASA GIBS / MODIS Terra', method: 'XYZ',
      maxZoom: 9, priority: 30, quality: 'measured',
      license: '公有领域（美国政府作品）',
      note: '250 m 全球每日真彩色，含云层；适合低倍率快速浏览。'
    }
  ];

  var SCENE_SOURCES = [
    {
      id: 'sentinel-2-l2a', label: 'Sentinel-2 L2A', method: 'STAC + COG',
      license: 'Copernicus Sentinel 数据条款',
      note: 'Earth Search STAC 检索 + AWS sentinel-cogs 窗口裁剪；支持日期/云量筛选、多景 MGRS 镶嵌。'
    }
  ];

  var CASE_RECIPES = [
    {
      key: 'taiwan', label: '台湾 · 安平',
      name: 'taiwan-anping-s2', date: '2026-10-26',
      bbox: '120.055, 22.915, 120.235, 23.025',
      resolution: '20 m', zooms: 'z10–13', mosaic: '否'
    },
    {
      key: 'india_china', label: '中印边界',
      name: 'india-china-s2', date: '2025-12-31',
      bbox: '78.7, 33.35, 79.5, 34.15',
      resolution: '30 m', zooms: 'z10–12', mosaic: '是（4 景）'
    }
  ];

  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function modal() {
    return document.getElementById('imageryModal');
  }

  function currentCaseId() {
    try {
      var host = global.SituationHost;
      var s = host && host.state;
      return String((s && (s.caseId || (s.data && s.data.case_id))) || '');
    } catch (e) {
      return '';
    }
  }

  function loadHealth() {
    if (state.loading) return;
    state.loading = true;
    state.error = null;
    render();
    fetch('/api/health', { cache: 'no-store' })
      .then(function (res) {
        if (!res.ok) throw new Error('HTTP ' + res.status);
        return res.json();
      })
      .then(function (data) {
        state.health = data || {};
        state.loaded = true;
      })
      .catch(function (err) {
        state.error = String(err && err.message || err);
      })
      .then(function () {
        state.loading = false;
        render();
      });
  }

  function resourceCounts() {
    var health = state.health || {};
    var sub = (health.subchecks && health.subchecks.sim_resources) || {};
    return sub.detail || null;
  }

  function section(title, note) {
    var box = el('section', 'ds-section');
    box.appendChild(el('h3', null, title));
    if (note) box.appendChild(el('p', 'ds-note', note));
    return box;
  }

  function sourceCard(src) {
    var card = el('div', 'ds-card');
    var head = el('div', 'ds-card-head');
    head.appendChild(el('b', 'ds-name', src.label));
    head.appendChild(el('span', 'ds-badge', src.method));
    card.appendChild(head);
    var meta = [];
    if (src.maxZoom) meta.push('最高 z' + src.maxZoom);
    if (src.priority !== undefined) meta.push('优先级 ' + src.priority);
    if (src.quality) meta.push('质量 ' + src.quality);
    if (meta.length) card.appendChild(el('div', 'ds-meta', meta.join(' · ')));
    if (src.note) card.appendChild(el('div', 'ds-desc', src.note));
    if (src.license) card.appendChild(el('div', 'ds-license', '许可：' + src.license));
    return card;
  }

  function caseCard(recipe, active) {
    var card = el('div', 'ds-card ds-case' + (active ? ' active' : ''));
    var head = el('div', 'ds-card-head');
    head.appendChild(el('b', 'ds-name', recipe.label));
    head.appendChild(el('span', 'ds-badge ' + (active ? 'on' : ''), active ? '当前案例' : '已配置'));
    card.appendChild(head);
    card.appendChild(el('div', 'ds-meta',
      recipe.name + ' · ' + recipe.date + ' · ' + recipe.resolution + ' · ' + recipe.zooms));
    card.appendChild(el('div', 'ds-desc', '范围 ' + recipe.bbox + '；镶嵌 ' + recipe.mosaic + '。'));
    card.appendChild(el('div', 'ds-license',
      '缓存命中时不再联网；获取/刷新：scripts/resource.py case-imagery --case ' + recipe.key));
    return card;
  }

  function render() {
    var body = document.getElementById('imageryBody');
    if (!body) return;
    body.textContent = '';

    if (state.loading && !state.loaded) {
      body.appendChild(el('div', 'ds-empty', '正在读取资源状态…'));
      return;
    }

    var counts = resourceCounts();
    var summary = el('div', 'ds-summary');
    summary.appendChild(el('b', null, '在线影像源与案例缓存'));
    summary.appendChild(el('span', 'ds-note',
      '在线源用于获取；案例缓存优先复用，避免重复下载。'));
    if (counts) {
      var chips = el('div', 'ds-chips');
      Object.keys(counts).sort().forEach(function (kind) {
        chips.appendChild(el('span', 'ds-chip', kind + ' ' + counts[kind]));
      });
      summary.appendChild(chips);
    } else if (state.error) {
      summary.appendChild(el('span', 'ds-warn', '资源索引状态读取失败：' + state.error));
    }
    body.appendChild(summary);

    var xyz = section('在线瓦片源',
      '按质量、速度与优先级自动探测并逐源回退；获取结果可落成标准 XYZ 瓦片集。');
    var grid = el('div', 'ds-grid');
    XYZ_SOURCES.forEach(function (src) { grid.appendChild(sourceCard(src)); });
    xyz.appendChild(grid);
    body.appendChild(xyz);

    var scene = section('卫星场景源',
      '按 AOI、日期与云量检索原始场景，可直接生成瓦片或参与案例缓存。');
    var sceneGrid = el('div', 'ds-grid');
    SCENE_SOURCES.forEach(function (src) { sceneGrid.appendChild(sourceCard(src)); });
    scene.appendChild(sceneGrid);
    body.appendChild(scene);

    var cache = section('案例影像缓存',
      '案例级 recipe 已登记在 resources/case_imagery.py；重复调用命中缓存，不重新下载。');
    var caseGrid = el('div', 'ds-grid');
    var current = currentCaseId();
    CASE_RECIPES.forEach(function (recipe) {
      var active = (recipe.key === 'taiwan' && current.indexOf('taiwan') >= 0) ||
                   (recipe.key === 'india_china' && current.indexOf('india') >= 0);
      caseGrid.appendChild(caseCard(recipe, active));
    });
    cache.appendChild(caseGrid);
    body.appendChild(cache);

    var offline = section('离线底图库', '演示与断网回放使用；由 tile_dbs 发布状态决定。');
    var dbs = (state.health && state.health.tile_dbs) || [];
    if (dbs.length) {
      var list = el('div', 'ds-chips');
      dbs.forEach(function (db) {
        var name = (db && (db.name || db.id || db.path)) || String(db);
        list.appendChild(el('span', 'ds-chip', name));
      });
      offline.appendChild(list);
    } else {
      offline.appendChild(el('div', 'ds-empty', '未发现已发布的离线瓦片库。'));
    }
    body.appendChild(offline);
  }

  function open() {
    var m = modal();
    if (!m) return;
    m.classList.remove('hidden');
    if (!state.loaded && !state.loading) loadHealth();
    else render();
  }

  function close() {
    var m = modal();
    if (m) m.classList.add('hidden');
  }

  function bind() {
    var button = document.getElementById('btnImagerySources');
    if (button) button.addEventListener('click', open);
    var closeButton = document.getElementById('imageryClose');
    if (closeButton) closeButton.addEventListener('click', close);
    var m = modal();
    if (m) {
      m.addEventListener('click', function (event) {
        if (event.target === m) close();
      });
    }
    document.addEventListener('keydown', function (event) {
      if (event.key === 'Escape' && m && !m.classList.contains('hidden')) close();
    });
  }

  global.ImagerySources = { bind: bind, open: open, close: close, refresh: loadHealth };
})(window);
