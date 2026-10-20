/* situation3d — TS：输入 / 管理 / 数据下载
 *
 * 三件事都围绕"当前TS"，所以共用一个三标签页模态框，而不是三个入口。
 *
 * 设计要点
 * --------
 * 1. **ZC范围支持两种确定方式**：在地球上框选，或直接输入 AOI 经纬度。
 *    框选用的是 `getPickRay` + `globe.pick` 取地面点，与"光标位置信息"同源。
 * 2. **生成的自建TS与案例走完全同一条装载管线**：`SituationHost.applySituationData()`。
 *    案例是 Python 侧离线生成的，自建TS是前端即时生成的，但装载逻辑只有一份，
 *    不会出现"案例能跑、自建TS少初始化了某样东西"。
 * 3. **目标运动是匀速直线**：起点 → 终点按帧线性插值，航向取两点间的真方位角，
 *    航速由距离与总时长反算。这足以驱动三维TS的全部显示与推理链路。
 *
 * 依赖 app.js 暴露的 `window.SituationHost`。
 */

'use strict';

(function (global) {
  const TICK_SECONDS = 30;
  const DEFAULT_HOURS = 12;

  // 可选军标（与 engine/symbology.py 的键一致）
  const ICONS = [
    ['destroyer', '驱逐舰'], ['frigate', '护卫舰'], ['replenishment', '补给舰'],
    ['corvette', '巡逻舰'], ['coastal_missile', '岸舰导弹'], ['sam_battalion', '防空导弹'],
    ['armor', '装甲分队'], ['artillery', '炮兵分队'], ['fighter', '战斗机'], ['aew', '预警机'],
  ];

  const DEFAULT_ICON_BY_DOMAIN = { sea: 'destroyer', land: 'armor', air: 'fighter' };

  let tab = 'input';
  let draft = null;            // { name, aoi:[w,s,e,n], hours, targets:[...] }
  let picking = false;
  let pickHandler = null;
  let pickPreview = null;
  let pickAnchor = null;

  function host() { return global.SituationHost; }

  function el(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  }

  // ------------------------------------------------------------------------
  // 草稿
  // ------------------------------------------------------------------------

  function blankTarget(index, side) {
    const h = host();
    const cam = (h.state.data && h.state.data.camera) || h.state.baseCamera || { lon: 119.6, lat: 24.4 };
    const dx = side === 'red' ? -0.6 : 0.6;
    return {
      id: (side === 'red' ? 'R' : 'B') + String(index).padStart(2, '0'),
      name: side === 'red' ? '红方目标' : '蓝方目标',
      affiliation: side,
      domain: 'sea',
      icon: 'destroyer',
      model: '',
      start: [round4(cam.lon + dx), round4(cam.lat + 0.8)],
      end: [round4(cam.lon + dx), round4(cam.lat - 0.8)],
      altitude: 0,
    };
  }

  function round4(v) { return Math.round(v * 10000) / 10000; }

  /**
   * 从当前TS派生草稿：把已有目标的起止位置与属性带过来，便于"改一改再生成"。
   * 没有TS时给一个台海的两目标最小样例。
   */
  function draftFromCurrent() {
    const h = host();
    const data = h.state.data;
    if (!data || !data.targets || !data.targets.length) {
      return {
        name: '自建TS',
        aoi: [117.5, 23.0, 121.0, 26.0],
        hours: DEFAULT_HOURS,
        targets: [blankTarget(1, 'red'), blankTarget(1, 'blue')],
        // 没有TS时**同样**把屏幕上的标绘收进来。两个分支必须一致：否则
        // "先画标绘、后打开TS面板"走的就是这一条，标绘会在导出里凭空消失 ——
        // 这正是本轮要消灭的那类静默丢失（见下面有TS分支的说明）。
        plots: (window.Plotting && Plotting.exportAll) ? Plotting.exportAll() : [],
      plotFeatures: exportPlotFeatures(),
        plotFeatures: exportPlotFeatures(),
      };
    }
    const frames = data.frames;
    const first = frames[0].targets;
    const last = frames[frames.length - 1].targets;
    const targets = data.targets.map(function (def) {
      const a = first.find(function (t) { return t.id === def.id; }) || {};
      const b = last.find(function (t) { return t.id === def.id; }) || a;
      return {
        id: def.id,
        name: def.name,
        affiliation: def.affiliation,
        domain: def.domain,
        icon: def.icon,
        model: def.model || '',
        start: [round4(a.lon), round4(a.lat)],
        end: [round4(b.lon), round4(b.lat)],
        altitude: Math.round(a.alt || 0),
      };
    });
    const bbox = (data.hotspots && data.hotspots.length)
      ? data.hotspots[0].bbox : [117.5, 23.0, 121.0, 26.0];
    return {
      name: (data.case_name || 'TS') + '（副本）',
      aoi: bbox.slice(),
      hours: DEFAULT_HOURS,
      targets: targets,
      // 草稿是"当前TS的副本"，那么**屏幕上已经画着的标绘也属于它**。不带上就会
      // 出现"重置草稿之后标绘全没了"——用户没删任何东西，却丢了。
      plots: (window.Plotting && Plotting.exportAll) ? Plotting.exportAll() : [],
      plotFeatures: exportPlotFeatures(),
    };
  }

  function ensureDraft() {
    if (!draft) draft = draftFromCurrent();
    return draft;
  }

  // ------------------------------------------------------------------------
  // 由草稿生成TS数据
  // ------------------------------------------------------------------------

  function bearingDeg(a, b) {
    const p1 = Cesium.Math.toRadians(a[1]);
    const p2 = Cesium.Math.toRadians(b[1]);
    const dl = Cesium.Math.toRadians(b[0] - a[0]);
    const y = Math.sin(dl) * Math.cos(p2);
    const x = Math.cos(p1) * Math.sin(p2) - Math.sin(p1) * Math.cos(p2) * Math.cos(dl);
    return (Cesium.Math.toDegrees(Math.atan2(y, x)) + 360) % 360;
  }

  function haversineKm(a, b) {
    const R = 6371;
    const dLat = Cesium.Math.toRadians(b[1] - a[1]);
    const dLon = Cesium.Math.toRadians(b[0] - a[0]);
    const s = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
      Math.cos(Cesium.Math.toRadians(a[1])) * Math.cos(Cesium.Math.toRadians(b[1])) *
      Math.sin(dLon / 2) * Math.sin(dLon / 2);
    return 2 * R * Math.asin(Math.min(1, Math.sqrt(s)));
  }

  /**
   * 把草稿展开成与案例数据集同构的对象。
   *
   * 目标按起点→终点匀速直线运动：每帧线性插值，航向取两点间的真方位角，
   * 航速由距离与总时长反算。字段名与离线数据集完全一致，装载管线才能共用。
   */
  function buildSituation(d) {
    const aoi = d.aoi;
    const ticks = Math.max(2, Math.round(d.hours * 3600 / TICK_SECONDS));

    const defs = d.targets.map(function (t) {
      const km = haversineKm(t.start, t.end);
      const hours = d.hours || DEFAULT_HOURS;
      const kn = km / 1.852 / hours;
      const icon = t.icon || DEFAULT_ICON_BY_DOMAIN[t.domain] || 'armor';
      const iconLabel = (ICONS.find(function (p) { return p[0] === icon; }) || [icon, icon])[1];
      return {
        id: t.id,
        name: t.name,
        affiliation: t.affiliation,
        domain: t.domain,
        icon: icon,
        icon_label: iconLabel,
        model: t.model || '',
        model_note: '',
        platform_class: t.domain === 'sea' ? '水面舰艇' : t.domain === 'air' ? '作战飞机' : '地面部队',
        role: '',
        sensor_name: '',
        sensor_range_km: t.domain === 'air' ? 200 : 120,
        engage_km: t.domain === 'air' ? 120 : 80,
        influence_km: t.domain === 'air' ? 260 : 160,
        speed_kn: Math.round(kn * 10) / 10,
        speed_kmh: Math.round(km / hours * 10) / 10,
        speed_unit: '节',
        altitude_m: t.altitude || 0,
        attributes: { 起点: t.start.join(', '), 终点: t.end.join(', ') },
        // ⚠ `waypoints` 的约定是 **[纬度, 经度]**（与离线数据集、createRouteEntities
        // 和 trackPlotPoints 一致），而 t.start/t.end 是 [经度, 纬度]。这里必须交换，
        // 否则自建 TS 的航路与目标轨迹会把经度当纬度画到错误位置。
        waypoints: [[t.start[1], t.start[0]], [t.end[1], t.end[0]]],
        _start: t.start,
        _end: t.end,
        _km: km,
      };
    });

    const frames = [];
    for (let i = 0; i <= ticks; i++) {
      const u = i / ticks;
      const ft = defs.map(function (def) {
        const lon = def._start[0] + (def._end[0] - def._start[0]) * u;
        const lat = def._start[1] + (def._end[1] - def._start[1]) * u;
        return {
          id: def.id,
          lon: round4(lon),
          lat: round4(lat),
          alt: def.altitude_m,
          heading: Math.round(bearingDeg(def._start, def._end) * 10) / 10,
          speed_kn: def.speed_kn,
          travelled_m: Math.round(def._km * 1000 * u),
          reached_end: u >= 1,
          laps: 0,
        };
      });
      frames.push({ tick: i, t_seconds: i * TICK_SECONDS, targets: ft });
    }

    // 草稿里的 _ 前缀字段只是中间量，不进数据集
    const targets = defs.map(function (def) {
      const clean = {};
      Object.keys(def).forEach(function (k) { if (k.charAt(0) !== '_') clean[k] = def[k]; });
      return clean;
    });

    const cx = (aoi[0] + aoi[2]) / 2;
    const cy = (aoi[1] + aoi[3]) / 2;

    return {
      schema: 'situation3d.situation.v1',
      case_id: 'custom',
      case_name: d.name || '自建TS',
      case_description: '由TS输入在前端生成：' + targets.length + ' 个目标，' +
        ticks + ' 帧，ZC范围 ' + aoi.join(', ') + '。',
      region: '自建ZC',
      scenario_id: null,
      scenario_name: d.name || '自建TS',
      provenance: 'frontend-situation-input',
      provenance_note: '目标按起点→终点匀速直线运动，用于快速搭建TS；不代表真实航路。',
      start_time_utc: new Date().toISOString().replace(/\.\d+Z$/, 'Z'),
      tick_seconds: TICK_SECONDS,
      tick_count: ticks + 1,
      camera: { lon: round4(cx), lat: round4(cy), height: 900000, heading: 0, pitch: -75 },
      // ZC范围本身作为一个热点区域，右侧表格与区域级推理才有内容
      hotspots: [{
        name: (d.name || '自建ZC') + ' 范围',
        bbox: aoi.slice(),
        center: [round4(cy), round4(cx)],
        span_deg: [Math.round((aoi[2] - aoi[0]) * 1000) / 1000,
                   Math.round((aoi[3] - aoi[1]) * 1000) / 1000],
        note: 'TS输入确定的ZC范围',
        kind: 'custom',
      }],
      urban_aoi: [],
      symbology: (host().state.base && host().state.base.symbology) || undefined,
      targets: targets,
      frames: frames,
      inference_frames: null,
      inference_model: null,
      inference_summary: null,
    };
  }

  // ------------------------------------------------------------------------
  // 框选ZC范围
  // ------------------------------------------------------------------------

  function pickLonLat(pos) {
    const v = host().viewer();
    if (!v) return null;
    const ray = v.camera.getPickRay(pos);
    const hit = ray ? v.scene.globe.pick(ray, v.scene) : null;
    if (!hit) return null;
    const c = Cesium.Cartographic.fromCartesian(hit);
    return { lon: Cesium.Math.toDegrees(c.longitude), lat: Cesium.Math.toDegrees(c.latitude) };
  }

  function drawPreview(a, b) {
    const v = host().viewer();
    if (!v) return;
    if (!pickPreview) {
      pickPreview = v.entities.add({
        id: 'sit-aoi-preview',
        rectangle: {
          coordinates: new Cesium.CallbackProperty(function () {
            if (!pickAnchor || !pickCursor) return undefined;
            return Cesium.Rectangle.fromDegrees(
              Math.min(pickAnchor.lon, pickCursor.lon), Math.min(pickAnchor.lat, pickCursor.lat),
              Math.max(pickAnchor.lon, pickCursor.lon), Math.max(pickAnchor.lat, pickCursor.lat));
          }, false),
          material: Cesium.Color.fromCssColorString('#1668dc').withAlpha(0.12),
          outline: true,
          outlineColor: Cesium.Color.fromCssColorString('#1668dc'),
          outlineWidth: 2,
          height: 0,
        },
      });
    }
    pickCursor = b;
  }

  let pickCursor = null;

  function startPicking() {
    const v = host().viewer();
    if (!v || picking) return;
    picking = true;
    pickAnchor = null;
    pickCursor = null;

    // 关掉模态框才能看到地球；框完再自动打开
    close();
    setHint('框选ZC范围：在地球上按住左键拖出一个矩形（Esc 取消）');
    v.scene.canvas.classList.add('sit-picking');

    const handler = new Cesium.ScreenSpaceEventHandler(v.scene.canvas);
    handler.setInputAction(function (click) {
      pickAnchor = pickLonLat(click.position);
      pickCursor = pickAnchor;
    }, Cesium.ScreenSpaceEventType.LEFT_DOWN);
    handler.setInputAction(function (move) {
      if (!pickAnchor) return;
      pickCursor = pickLonLat(move.endPosition);
      drawPreview(pickAnchor, pickCursor);
    }, Cesium.ScreenSpaceEventType.MOUSE_MOVE);
    handler.setInputAction(function () {
      if (!pickAnchor || !pickCursor) { stopPicking(); return; }
      const d = ensureDraft();
      d.aoi = [
        Math.round(Math.min(pickAnchor.lon, pickCursor.lon) * 1000) / 1000,
        Math.round(Math.min(pickAnchor.lat, pickCursor.lat) * 1000) / 1000,
        Math.round(Math.max(pickAnchor.lon, pickCursor.lon) * 1000) / 1000,
        Math.round(Math.max(pickAnchor.lat, pickCursor.lat) * 1000) / 1000,
      ];
      stopPicking();
      open('input');
      setHint('ZC范围已框选：' + d.aoi.join(', '));
    }, Cesium.ScreenSpaceEventType.LEFT_UP);
    pickHandler = handler;

    document.addEventListener('keydown', onPickKey, true);
  }

  function onPickKey(e) {
    if (e.key === 'Escape' && picking) {
      stopPicking();
      open('input');
      setHint('已取消框选。');
    }
  }

  function stopPicking() {
    const v = host().viewer();
    picking = false;
    if (pickHandler) { pickHandler.destroy(); pickHandler = null; }
    if (pickPreview && v) { v.entities.remove(pickPreview); pickPreview = null; }
    if (v) v.scene.canvas.classList.remove('sit-picking');
    document.removeEventListener('keydown', onPickKey, true);
    pickAnchor = null;
    pickCursor = null;
  }

  // ------------------------------------------------------------------------
  // 面板渲染
  // ------------------------------------------------------------------------

  function open(which) {
    tab = which || 'input';
    if (tab === 'input' || tab === 'manage') ensureDraft();
    const modal = document.getElementById('situationModal');
    if (!modal) return;
    modal.classList.remove('hidden');
    render();
  }

  function close() {
    const modal = document.getElementById('situationModal');
    if (modal) modal.classList.add('hidden');
    stopPicking();
  }

  function setHint(text) {
    const h = host();
    if (h && h.setStatus) h.setStatus(text, 'ok');
  }

  function render() {
    const body = document.getElementById('sitBody');
    if (!body) return;
    Array.prototype.forEach.call(
      document.querySelectorAll('#sitTabs .modal-tab'), function (b) {
        b.classList.toggle('active', b.dataset.sitTab === tab);
      });
    body.innerHTML = '';
    // 面板渲染一旦抛异常，之前的表现是"整块内容静默变空"，极难定位。
    // 这里兜住并把错误显示出来——面板渲染失败必须看得见。
    try {
      if (tab === 'input') renderInput(body);
      else if (tab === 'manage') renderManage(body);
      else if (tab === 'brief') renderBrief(body);
      else renderDownload(body);
    } catch (e) {
      console.error('[situation] 面板渲染失败', e);
      const box = el('div', 'sit-error');
      box.appendChild(el('div', null, '面板渲染失败：' + e.message));
      const pre = document.createElement('pre');
      pre.textContent = String(e.stack || '').split('\n').slice(0, 4).join('\n');
      box.appendChild(pre);
      body.appendChild(box);
    }
  }

  function section(parent, title) {
    const s = el('div', 'sit-section');
    s.appendChild(el('h3', null, title));
    parent.appendChild(s);
    return s;
  }

  function numberInput(value, step) {
    const i = document.createElement('input');
    i.type = 'number';
    i.step = step || '0.01';
    i.value = value;
    return i;
  }

  function textInput(value) {
    const i = document.createElement('input');
    i.type = 'text';
    i.value = value;
    return i;
  }

  // ---- 标签页 1：TS输入 ----
  // ------------------------------------------------------------------------
  // 结构化文本注册（JSON / XML）
  // ------------------------------------------------------------------------
  //
  // 两种格式走**同一个归一化函数**，输出与草稿完全同构的对象。这是"JSON 与 XML
  // 真正等价"的保证 —— 自检里直接断言同一份定义的两种写法归一化后深度相等，
  // 而不是分别测各自能跑通（那样两种格式可以悄悄漂移）。
  // XML 用浏览器原生 DOMParser，**不引入任何依赖**。
  //
  // 校验**一次报全**：粘一份大清单时只报第一条，用户要反复试很多轮。

  var iconKeysCache = null;

  // 结构化注册区的**待显示提示**。
  //
  // 为什么需要它：追加/替换之后要 `open('input')` 重刷草稿（来源行与目标表都要更新），
  // 而 `open()` 会**重建整个输入面板的 DOM** —— 先写进旧提示元素的文案连元素一起被丢掉，
  // 用户一个字都看不到。把文案存下来、由重建后的区域自己取走显示。
  // （同一类问题在本文件里已出现过一次：生成TS 的成功提示把还原失败提示覆盖掉了。）
  var pendingStructHint = null;
  function iconKeys() {
    if (iconKeysCache) return iconKeysCache;
    var m = {};
    (typeof ICONS !== 'undefined' ? ICONS : []).forEach(function (p) { m[p[0]] = p[1]; });
    iconKeysCache = m;
    return m;
  }

  function num(v) {
    if (v === null || v === undefined || v === '') return NaN;
    return Number(v);
  }

  function parseCoordPair(lon, lat, where, errors, list) {
    var a = num(lon), b = num(lat);
    if (!isFinite(a) || !isFinite(b)) { errors.push(where + '：经纬度不是有效数字'); return; }
    if (a < -180 || a > 180 || b < -90 || b > 90) {
      errors.push(where + '：坐标越界（' + a + ', ' + b + '）'); return;
    }
    list.push([a, b]);
  }

  /**
   * 把 JSON 或 XML 文本归一化成草稿对象。
   * 返回 `{draft, errors, warnings}`；`errors` 非空时不要应用。
   */
  /**
   * 标绘的 GeoJSON 形式（与 `plots` 并存，互为兼容）：
   *   · 导出：在 TS 定义里额外写 `plotFeatures`（FeatureCollection）；
   *   · 导入：若给了 `plotFeatures` 就优先用它，否则退回旧 `plots` 数组。
   * 两条路都最终落到 `Plotting.add()`，所以类型/点数校验只有一份。
   */
  function exportPlotFeatures() {
    return (window.Plotting && Plotting.toFeatureCollection)
      ? Plotting.toFeatureCollection()
      : { type: 'FeatureCollection', features: [] };
  }

  /** FeatureCollection → 与 `plots` 同构的数组（供 normalizeStructured 校验）。 */
  function featuresToPlots(fc) {
    if (!fc || fc.type !== 'FeatureCollection' || !Array.isArray(fc.features)) return null;
    const out = [];
    fc.features.forEach(function (f) {
      const p = (f && f.properties) || {};
      if (p.kind === 'hotspot' || p.kind === 'occupation') return; // 动态生成，不入 TS
      const g = f && f.geometry;
      let pts = [];
      if (g && g.type === 'Polygon' && g.coordinates && g.coordinates[0]) {
        pts = g.coordinates[0].slice();
        if (pts.length > 1 && pts[0][0] === pts[pts.length - 1][0] &&
            pts[0][1] === pts[pts.length - 1][1]) pts.pop();
      } else if (g && g.type === 'MultiLineString' && g.coordinates && g.coordinates[0]) {
        pts = g.coordinates[0].slice();
      } else if (g && g.type === 'LineString') {
        pts = g.coordinates.slice();
      }
      out.push({ type: p.type, layer: p.layer, hostility: p.hostility,
                 label: p.label || null, points: pts });
    });
    return out;
  }

  function normalizeStructured(text, format) {
    var errors = [], warnings = [], raw = null;

    if (format === 'xml') {
      var doc, perr;
      try {
        doc = new DOMParser().parseFromString(String(text), 'application/xml');
      } catch (e) {
        return { draft: null, errors: ['XML 解析失败：' + e.message], warnings: [] };
      }
      perr = doc.getElementsByTagName('parsererror')[0];
      if (perr) {
        var msg = (perr.textContent || '').replace(/\s+/g, ' ').trim();
        return { draft: null, errors: ['XML 解析失败：' + msg.slice(0, 160)], warnings: [] };
      }
      var root = doc.documentElement;
      if (!root || root.nodeName !== 'situation') {
        return { draft: null,
                 errors: ['根元素应为 <situation>，实际是 <' + (root ? root.nodeName : '空') + '>'],
                 warnings: [] };
      }
      raw = { name: root.getAttribute('name') || undefined,
              hours: root.getAttribute('hours') || undefined,
              aoi: null, targets: [] };
      var aoiEl = root.getElementsByTagName('aoi')[0];
      if (aoiEl) {
        raw.aoi = [aoiEl.getAttribute('west'), aoiEl.getAttribute('south'),
                   aoiEl.getAttribute('east'), aoiEl.getAttribute('north')];
      }
      var tEls = root.getElementsByTagName('target');
      for (var i = 0; i < tEls.length; i++) {
        var te = tEls[i];
        var st = te.getElementsByTagName('start')[0];
        var en = te.getElementsByTagName('end')[0];
        raw.targets.push({
          id: te.getAttribute('id'), name: te.getAttribute('name'),
          affiliation: te.getAttribute('affiliation'), domain: te.getAttribute('domain'),
          icon: te.getAttribute('icon'), model: te.getAttribute('model') || '',
          altitude: te.getAttribute('altitude') || 0,
          start: st ? [st.getAttribute('lon'), st.getAttribute('lat')] : null,
          end: en ? [en.getAttribute('lon'), en.getAttribute('lat')] : null,
        });
      }
      // <plots><plot type=".." layer=".." hostility=".." label=".."><point lon lat/>…</plot></plots>
      // 与 JSON 的 plots 一一对应 —— 两种格式必须能互相转换，否则"支持 XML"就是空话。
      raw.plots = [];
      var plEls = root.getElementsByTagName('plot');
      for (var q = 0; q < plEls.length; q++) {
        var pe = plEls[q];
        var pts = [];
        var ptEls = pe.getElementsByTagName('point');
        for (var w = 0; w < ptEls.length; w++) {
          pts.push([ptEls[w].getAttribute('lon'), ptEls[w].getAttribute('lat')]);
        }
        raw.plots.push({
          type: pe.getAttribute('type'), layer: pe.getAttribute('layer') || undefined,
          hostility: pe.getAttribute('hostility') || undefined,
          label: pe.getAttribute('label') || undefined, points: pts,
        });
      }
    } else {
      try {
        raw = JSON.parse(String(text));
      } catch (e) {
        return { draft: null, errors: ['JSON 解析失败：' + e.message], warnings: [] };
      }
      if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
        return { draft: null, errors: ['顶层必须是一个对象'], warnings: [] };
      }
    }

    // ---- 以下校验与字段处理对两种格式完全相同 ----
    if (!Array.isArray(raw.aoi) || raw.aoi.length !== 4) {
      errors.push('缺少 aoi（需要 west, south, east, north 四个数）');
    }
    var aoi = null;
    if (Array.isArray(raw.aoi) && raw.aoi.length === 4) {
      aoi = raw.aoi.map(num);
      if (aoi.some(function (v) { return !isFinite(v); })) {
        errors.push('aoi 含非数字'); aoi = null;
      } else if (aoi[0] >= aoi[2] || aoi[1] >= aoi[3]) {
        errors.push('aoi 需满足 west < east 且 south < north'); aoi = null;
      }
    }
    if (!Array.isArray(raw.targets) || !raw.targets.length) {
      errors.push('targets 必须是非空数组');
      return { draft: null, errors: errors, warnings: warnings };
    }

    var hours = num(raw.hours);
    if (raw.hours === undefined || raw.hours === null || raw.hours === '') hours = DEFAULT_HOURS;
    else if (!isFinite(hours) || hours <= 0) { errors.push('hours 必须是正数'); }

    var seen = {}, keys = iconKeys(), out = [];
    raw.targets.forEach(function (tg, i) {
      var where = 'targets[' + i + ']' + (tg && tg.id ? '（' + tg.id + '）' : '');
      if (!tg || typeof tg !== 'object') { errors.push(where + '：不是对象'); return; }
      var id = String(tg.id === undefined || tg.id === null ? '' : tg.id).trim();
      if (!id) { errors.push(where + '：缺少 id'); return; }
      if (seen[id]) { errors.push(where + '：id 重复（' + id + '）'); return; }
      seen[id] = true;
      var aff = tg.affiliation === 'blue' ? 'blue' : (tg.affiliation === 'red' ? 'red' : null);
      if (!aff) errors.push(where + '：affiliation 必须是 red 或 blue');
      var dom = (tg.domain === 'sea' || tg.domain === 'land' || tg.domain === 'air') ? tg.domain : null;
      if (!dom) errors.push(where + '：domain 必须是 sea / land / air');
      var icon = tg.icon ? String(tg.icon) : '';
      if (icon && !keys[icon]) {
        warnings.push(where + '：图标「' + icon + '」不在册，已回落按域的默认图标');
        icon = '';
      }
      if (!icon && dom) icon = DEFAULT_ICON_BY_DOMAIN[dom] || 'armor';
      if (!icon) icon = 'armor';
      var start = [], end = [];
      if (!Array.isArray(tg.start) || tg.start.length !== 2) errors.push(where + '：缺少 start');
      else parseCoordPair(tg.start[0], tg.start[1], where + ' 的 start', errors, start);
      if (!Array.isArray(tg.end) || tg.end.length !== 2) errors.push(where + '：缺少 end');
      else parseCoordPair(tg.end[0], tg.end[1], where + ' 的 end', errors, end);
      var alt = num(tg.altitude);
      if (tg.altitude !== undefined && tg.altitude !== null && tg.altitude !== '' && !isFinite(alt)) {
        errors.push(where + '：altitude 不是数字');
      }
      if (start.length && end.length) {
        out.push({ id: id, name: String(tg.name || id), affiliation: aff || 'red',
                   domain: dom || 'land', icon: icon, model: String(tg.model || ''),
                   start: start[0], end: end[0],
                   altitude: isFinite(alt) ? alt : 0 });
      }
    });

    // ---- plots（可选）----
    //
    // **缺失时视为空列表**：这是必须的向后兼容 —— 本轮之前导出的所有 TS 定义都没有
    // 这个字段，不能因为加了一个可选项就让旧文件全部打不开。
    var plotsOut = [];
    var plotCat = {};
    if (window.Plotting && Plotting.catalogue) {
      Plotting.catalogue().forEach(function (c) { plotCat[c.type] = c; });
    }
    var plotLayers = [];
    // 只由系统类型组成的类别（当前是 plot.track）：普通标绘不能写进去。
    // 注意这段必须在注册表过滤**之前**算好——选中目标时 plot.track 已经被
    // syncPlotLayers 动态注册，单看"是否在册"会把系统图层当成合法目标。
    var manualLayerIds = {}, systemOnlyLayers = {};
    Object.keys(plotCat).forEach(function (t) {
      var c = plotCat[t];
      if (c.system) {
        if (!manualLayerIds[c.defaultLayer]) systemOnlyLayers[c.defaultLayer] = true;
      } else {
        manualLayerIds[c.defaultLayer] = true;
        delete systemOnlyLayers[c.defaultLayer];
      }
    });
    if (window.LayerRegistry && LayerRegistry.list) {
      // 只认「标绘」组：situation.relations / situation.occupation 也是 kind:'plot'，
      // 但它们走各自的 handler、没有要素提供者，普通标绘落进去会变成关不掉的孤儿。
      plotLayers = LayerRegistry.list('plot').filter(function (l) {
        return l.group === '标绘' && !systemOnlyLayers[l.id];
      }).map(function (l) { return l.id; });
    }
    // 标绘图层由**实际要素动态成层**：某个类别当前为空时并不在注册表里，
    // 但导入指向它的标绘仍然合法（导入后 Plotting.onChange 会自动建层）。
    // 因此"在册"集合 = 当前已注册的标绘图层 ∪ 目录里**手动类型**的默认类别。
    Object.keys(manualLayerIds).forEach(function (lid) {
      if (plotLayers.indexOf(lid) < 0) plotLayers.push(lid);
    });
    // GeoJSON 入口优先：给了 `plotFeatures` 就以它为准（并忽略旧 `plots`，
    // 避免同一批标绘被两份数据导入两次）。
    if (raw.plotFeatures !== undefined && raw.plotFeatures !== null) {
      const asPlots = featuresToPlots(raw.plotFeatures);
      if (!asPlots) {
        errors.push('plotFeatures 必须是 GeoJSON FeatureCollection');
      } else {
        raw = Object.assign({}, raw, { plots: asPlots });
      }
    }
    if (raw.plots !== undefined && raw.plots !== null) {
      if (!Array.isArray(raw.plots)) {
        errors.push('plots 必须是数组');
      } else {
        raw.plots.forEach(function (pl, i) {
          var where = 'plots[' + i + ']';
          if (!pl || typeof pl !== 'object') { errors.push(where + '：不是对象'); return; }
          var cat = plotCat[pl.type];
          if (!cat) { errors.push(where + '：未知标绘类型「' + pl.type + '」'); return; }
          // 系统生成的标绘（目标轨迹）是选中目标的派生数据，不能写进 TS 定义
          if (cat.system) {
            errors.push(where + '：' + cat.zh + ' 由系统生成，不可写入 TS 定义');
            return;
          }
          var ctrl = cat.ctrl || { min: 2, max: null };
          if (!Array.isArray(pl.points) || pl.points.length < ctrl.min) {
            errors.push(where + '（' + pl.type + '）：至少需要 ' + ctrl.min + ' 个控制点');
            return;
          }
          if (ctrl.max && pl.points.length > ctrl.max) {
            errors.push(where + '（' + pl.type + '）：最多 ' + ctrl.max + ' 个控制点');
            return;
          }
          var pts = [], bad = false;
          pl.points.forEach(function (c, k) {
            if (!Array.isArray(c) || c.length !== 2) {
              errors.push(where + ' 第 ' + (k + 1) + ' 个点：需要 [经度, 纬度]'); bad = true; return;
            }
            var lo = num(c[0]), la = num(c[1]);
            if (!isFinite(lo) || !isFinite(la)) {
              errors.push(where + ' 第 ' + (k + 1) + ' 个点：不是有效数字'); bad = true; return;
            }
            if (lo < -180 || lo > 180 || la < -90 || la > 90) {
              errors.push(where + ' 第 ' + (k + 1) + ' 个点：坐标越界（' + lo + ', ' + la + '）');
              bad = true; return;
            }
            pts.push([lo, la]);
          });
          if (bad) return;
          var layer = pl.layer;
          // 图层不在册 → 回落该类型的默认图层并**警告**（与未知 icon 同一策略：
          // 可恢复的问题不该硬失败，但必须让用户看见）
          if (layer && plotLayers && plotLayers.indexOf(layer) < 0) {
            warnings.push(where + '：图层「' + layer + '」不在册，已回落 '
              + (cat.defaultLayer || 'plot.offense'));
            layer = cat.defaultLayer;
          }
          plotsOut.push({
            type: pl.type,
            layer: layer || cat.defaultLayer || 'plot.offense',
            hostility: pl.hostility === 'enemy' ? 'enemy' : 'friendly',
            label: pl.label || null,
            points: pts,
          });
        });
      }
    }

    if (errors.length) return { draft: null, errors: errors, warnings: warnings };
    return {
      draft: { name: String(raw.name || '结构化注册TS'), aoi: aoi, hours: hours,
               targets: out, plots: plotsOut },
      errors: [], warnings: warnings,
    };
  }

  /** 结构化注册区：粘贴 JSON 或 XML → 预览 → 追加/替换草稿。 */
  function structuredSection(body) {
    const s = section(body, '结构化文本注册（JSON / XML）');
    const row = el('div', 'sit-row');
    const fmt = el('select', 'sit-select');
    [['json', 'JSON'], ['xml', 'XML']].forEach(function (o) {
      const op = document.createElement('option');
      op.value = o[0]; op.textContent = o[1];
      fmt.appendChild(op);
    });
    row.appendChild(fmt);
    row.appendChild(el('span', 'sit-hint',
      '粘贴一份完整定义（与「导出TS定义」同构）。两种格式等价，可互相转换。'));
    s.appendChild(row);

    const ta = document.createElement('textarea');
    ta.className = 'sit-textarea';
    ta.rows = 6;
    ta.placeholder = '{\n  "name": "……",\n  "aoi": [119.0, 23.0, 121.0, 26.0],\n'
      + '  "hours": 12,\n  "targets": [\n    {"id":"R01","name":"…","affiliation":"red",\n'
      + '     "domain":"sea","icon":"destroyer","start":[119.6,24.4],"end":[119.6,23.5]}\n  ]\n}';
    s.appendChild(ta);

    const row2 = el('div', 'sit-row');
    const bPrev = el('button', 'mini', '解析并预览');
    bPrev.type = 'button';
    const bAdd = el('button', 'mini', '追加到草稿');
    bAdd.type = 'button';
    const bRep = el('button', 'mini', '替换草稿');
    bRep.type = 'button';
    [bPrev, bAdd, bRep].forEach(function (x) { row2.appendChild(x); });
    s.appendChild(row2);
    const out = el('div', 'sit-hint');
    // 面板刚被重建时，把上一次操作留下的提示接过来显示（见 pendingStructHint 的说明）
    if (pendingStructHint) { out.textContent = pendingStructHint; pendingStructHint = null; }
    s.appendChild(out);

    function report(res) {
      if (res.errors.length) {
        out.textContent = '解析失败（' + res.errors.length + ' 项）：' + res.errors.join('；');
        return null;
      }
      const d = res.draft;
      const cnt = {};
      d.targets.forEach(function (x) { cnt[x.domain] = (cnt[x.domain] || 0) + 1; });
      const aff = { red: 0, blue: 0 };
      d.targets.forEach(function (x) { aff[x.affiliation]++; });
      out.textContent = '解析成功：' + d.targets.length + ' 个目标（红 ' + aff.red + ' / 蓝 ' + aff.blue
        + '；海 ' + (cnt.sea || 0) + ' 陆 ' + (cnt.land || 0) + ' 空 ' + (cnt.air || 0) + '）'
        // 标绘条数也要报出来：不报的话"解析成功"只证明了目标那半边，
        // 用户没法从预览看出标绘到底有没有被读到。
        + '，标绘 ' + ((d.plots && d.plots.length) || 0) + ' 条'
        + '，AOI ' + d.aoi.join(', ') + '，时长 ' + d.hours + ' h'
        + (res.warnings.length ? '。警告：' + res.warnings.join('；') : '');
      return d;
    }

    function parsed() {
      if (!ta.value.trim()) { out.textContent = '请先粘贴 JSON 或 XML 文本。'; return null; }
      return report(normalizeStructured(ta.value, fmt.value));
    }

    bPrev.addEventListener('click', function () { parsed(); });
    bAdd.addEventListener('click', function () {
      const nd = parsed();
      if (!nd) return;
      const d = ensureDraft();
      const have = {};
      d.targets.forEach(function (x) { have[x.id] = true; });
      const dup = nd.targets.filter(function (x) { return have[x.id]; }).map(function (x) { return x.id; });
      if (dup.length) {
        out.textContent = '追加失败：草稿里已有同 id 目标 ' + dup.join(', ')
          + '。请先改 id，或用「替换草稿」。';
        return;
      }
      d.targets = d.targets.concat(nd.targets);
      // 标绘跟着一起追加：`nd` 是一份**完整定义**，它的 plots 若在这里丢掉，
      // 就是"预览说读到了 N 条、追加完却一条都没有"。标绘没有 id，直接拼接即可
      // （生成时的 importAll 会重新发 id，不会与已有标绘冲突）。
      if (nd.plots && nd.plots.length) d.plots = (d.plots || []).concat(nd.plots);
      // 先存后刷：`open()` 会重建面板，直接写 out 的文案会随之消失（见 pendingStructHint）
      pendingStructHint = '已追加 ' + nd.targets.length + ' 个目标'
        + (nd.plots && nd.plots.length ? '、' + nd.plots.length + ' 条标绘' : '')
        + '，草稿现共 ' + d.targets.length + ' 个目标、'
        + ((d.plots && d.plots.length) || 0) + ' 条标绘。';
      open('input');
    });
    bRep.addEventListener('click', function () {
      const nd = parsed();
      if (!nd) return;
      draft = nd;
      pendingStructHint = '已替换草稿：' + nd.targets.length + ' 个目标、'
        + ((nd.plots && nd.plots.length) || 0) + ' 条标绘。';
      open('input');
    });
  }
  function renderInput(body) {
    const d = ensureDraft();

    // 数据来源一行：草稿是从当前TS派生的还是最小样例，一眼能看出来
    const src = el('div', 'sit-hint');
    const data = host().state.data;
    src.textContent = '草稿来源：' + (data
      ? '当前TS「' + data.case_name + '」（' + data.targets.length + ' 个目标）'
      : '最小样例（当前未加载TS）') + ' · 草稿目标 ' + d.targets.length + ' 个';
    body.appendChild(src);

    // 结构化文本注册：与手工录入并列的**第二条入口**，两条都落到同一份草稿上。
    structuredSection(body);

    // ZC范围
    const s1 = section(body, 'ZC范围');
    const row1 = el('div', 'sit-row');
    const btnPick = el('button', 'mini', '在地图上框选');
    btnPick.type = 'button';
    btnPick.addEventListener('click', startPicking);
    row1.appendChild(btnPick);
    row1.appendChild(el('span', 'sit-hint', '或直接输入 AOI 经纬度（度）：'));
    s1.appendChild(row1);

    const row2 = el('div', 'sit-row');
    const labels = ['西', '南', '东', '北'];
    const aoiInputs = d.aoi.map(function (v, i) {
      const lab = el('label', null, labels[i]);
      const inp = numberInput(v);
      inp.addEventListener('change', function () {
        d.aoi[i] = Number(inp.value);
      });
      lab.appendChild(inp);
      row2.appendChild(lab);
      return inp;
    });
    s1.appendChild(row2);
    s1.appendChild(el('div', 'sit-hint',
      '当前：' + d.aoi.join(', ') + '（跨度 ' +
      (d.aoi[2] - d.aoi[0]).toFixed(2) + '° × ' + (d.aoi[3] - d.aoi[1]).toFixed(2) + '°）'));

    // TS名称与时长
    const s2 = section(body, 'TS参数');
    const rowN = el('div', 'sit-row');
    const nameLab = el('label', null, '名称');
    const nameInp = textInput(d.name);
    nameInp.addEventListener('change', function () { d.name = nameInp.value; });
    nameLab.appendChild(nameInp);
    rowN.appendChild(nameLab);
    const hoursLab = el('label', null, '推演时长（小时）');
    const hoursInp = numberInput(d.hours, '1');
    hoursInp.addEventListener('change', function () { d.hours = Math.max(1, Number(hoursInp.value)); });
    hoursLab.appendChild(hoursInp);
    rowN.appendChild(hoursLab);
    s2.appendChild(rowN);

    // 目标编成
    const s3 = section(body, '目标编成');
    const table = el('table', 'sit-table');
    table.innerHTML = '<thead><tr><th>ID</th><th>名称</th><th>阵营</th><th>域</th>' +
      '<th>军标</th><th>起点 经/纬</th><th>终点 经/纬</th><th>高度 m</th><th></th></tr></thead>';
    const tb = el('tbody');
    table.appendChild(tb);

    d.targets.forEach(function (t, idx) {
      const tr = el('tr');
      function cell(node) { const td = el('td'); td.appendChild(node); tr.appendChild(td); return td; }
      const idInp = textInput(t.id);
      idInp.addEventListener('change', function () { t.id = idInp.value.trim(); });
      cell(idInp);
      const nmInp = textInput(t.name);
      nmInp.addEventListener('change', function () { t.name = nmInp.value; });
      cell(nmInp);
      const affSel = selectFrom([['red', '红方'], ['blue', '蓝方']], t.affiliation);
      affSel.addEventListener('change', function () { t.affiliation = affSel.value; });
      cell(affSel);
      const domSel = selectFrom([['sea', '海上'], ['land', '陆上'], ['air', '空中']], t.domain);
      domSel.addEventListener('change', function () {
        t.domain = domSel.value;
        t.icon = DEFAULT_ICON_BY_DOMAIN[t.domain];
        render();
      });
      cell(domSel);
      const iconSel = selectFrom(ICONS, t.icon);
      iconSel.addEventListener('change', function () { t.icon = iconSel.value; });
      cell(iconSel);
      cell(pairInputs(t.start));
      cell(pairInputs(t.end));
      const altInp = numberInput(t.altitude, '100');
      altInp.addEventListener('change', function () { t.altitude = Number(altInp.value); });
      cell(altInp);
      const del = el('button', 'mini', '删除');
      del.type = 'button';
      del.addEventListener('click', function () { d.targets.splice(idx, 1); render(); });
      cell(del);
      tb.appendChild(tr);          // ← 少了这一行，行就永远不会进表格
    });
    s3.appendChild(table);

    const rowAdd = el('div', 'sit-row');
    const addRed = el('button', 'mini', '+ 红方目标');
    addRed.type = 'button';
    addRed.addEventListener('click', function () {
      d.targets.push(blankTarget(countSide(d.targets, 'red') + 1, 'red')); render();
    });
    const addBlue = el('button', 'mini', '+ 蓝方目标');
    addBlue.type = 'button';
    addBlue.addEventListener('click', function () {
      d.targets.push(blankTarget(countSide(d.targets, 'blue') + 1, 'blue')); render();
    });
    rowAdd.appendChild(addRed);
    rowAdd.appendChild(addBlue);
    s3.appendChild(rowAdd);

    // 图层
    const s4 = section(body, '图层');
    s4.appendChild(el('div', 'sit-hint',
      '新TS生成后按这里的勾选初始化图层。也可随时在左栏「图层管理」里改。'));
    const rowL = el('div', 'sit-row');
    [['layerHotspots', '热点区域'], ['layerRoutes', '计划航路'],
     ['layerTracks', '历史轨迹'], ['layerLabels', '目标标牌'],
     ['layerTargetPanel', '目标元数据面板']].forEach(function (p) {
      const src = document.getElementById(p[0]);
      const lab = el('label', 'check');
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = src ? src.checked : true;
      cb.dataset.sitLayer = p[0];
      lab.appendChild(cb);
      lab.appendChild(document.createTextNode(' ' + p[1]));
      rowL.appendChild(lab);
    });
    s4.appendChild(rowL);

    // 动作
    const acts = el('div', 'sit-actions');
    const gen = el('button', 'primary', '生成并初始化双方TS');
    gen.type = 'button';
    gen.addEventListener('click', function () { generate(); });
    const reset = el('button', 'mini', '重置为当前TS');
    reset.type = 'button';
    reset.addEventListener('click', function () { draft = draftFromCurrent(); render(); });
    acts.appendChild(gen);
    acts.appendChild(reset);
    body.appendChild(acts);

    function pairInputs(pair) {
      const wrap = el('span');
      const a = numberInput(pair[0]); a.style.width = '76px';
      const b = numberInput(pair[1]); b.style.width = '76px';
      a.addEventListener('change', function () { pair[0] = Number(a.value); });
      b.addEventListener('change', function () { pair[1] = Number(b.value); });
      wrap.appendChild(a); wrap.appendChild(document.createTextNode(' / ')); wrap.appendChild(b);
      return wrap;
    }
  }

  function countSide(targets, side) {
    return targets.filter(function (t) { return t.affiliation === side; }).length;
  }

  function selectFrom(pairs, value) {
    const s = document.createElement('select');
    pairs.forEach(function (p) {
      const o = document.createElement('option');
      o.value = p[0];
      o.textContent = p[1];
      if (p[0] === value) o.selected = true;
      s.appendChild(o);
    });
    return s;
  }

  // ---- 标签页 2：TS管理 ----
  function renderManage(body) {
    const h = host();
    const data = h.state.data;
    const s1 = section(body, '当前TS');
    if (!data) {
      s1.appendChild(el('div', 'sit-hint', '未加载TS。到「TS输入」生成一个，或从顶栏「TS案例」选一个。'));
      return;
    }
    const info = [
      ['名称', data.case_name],
      ['来源', data.provenance === 'frontend-situation-input' ? 'TS输入（前端生成）' : ('案例 ' + (data.case_id || ''))],
      ['目标数', data.targets.length],
      ['帧数', (data.frames.length - 1) + '（步长 ' + data.tick_seconds + ' s）'],
      ['热点区域', (data.hotspots || []).length],
      ['ZC范围', (data.hotspots && data.hotspots[0]) ? data.hotspots[0].bbox.join(', ') : '—'],
    ];
    info.forEach(function (kv) {
      const r = el('div', 'kv');
      r.appendChild(el('span', null, kv[0]));
      r.appendChild(el('b', null, String(kv[1])));
      s1.appendChild(r);
    });

    const s2 = section(body, '目标编成');
    const table = el('table', 'sit-table');
    table.innerHTML = '<thead><tr><th>ID</th><th>名称</th><th>阵营</th><th>域</th>' +
      '<th>军标</th><th>当前经纬</th><th>航向/航速</th><th></th></tr></thead>';
    const tb = el('tbody');
    const cur = data.frames[Math.min(h.state.tick, data.frames.length - 1)].targets;
    data.targets.forEach(function (def) {
      const st = cur.find(function (t) { return t.id === def.id; }) || {};
      const tr = el('tr');
      [def.id, def.name,
       def.affiliation === 'red' ? '红方' : '蓝方',
       def.domain === 'sea' ? '海上' : def.domain === 'air' ? '空中' : '陆上',
       def.icon_label || def.icon,
       (st.lon !== undefined ? st.lon + ', ' + st.lat : '—'),
       (st.heading !== undefined ? st.heading + '° / ' + st.speed_kn + ' 节' : '—'),
      ].forEach(function (v) { tr.appendChild(el('td', null, String(v))); });
      const td = el('td');
      const del = el('button', 'mini', '删除');
      del.type = 'button';
      del.addEventListener('click', function () {
        data.targets = data.targets.filter(function (x) { return x.id !== def.id; });
        data.frames.forEach(function (f) {
          f.targets = f.targets.filter(function (x) { return x.id !== def.id; });
        });
        h.reload();
        render();
        setHint('已删除目标 ' + def.id + '。');
      });
      td.appendChild(del);
      tr.appendChild(td);
      tb.appendChild(tr);
    });
    table.appendChild(tb);
    s2.appendChild(table);

    const acts = el('div', 'sit-actions');
    const edit = el('button', 'primary', '在TS输入中编辑');
    edit.type = 'button';
    edit.addEventListener('click', function () { open('input'); });
    const clear = el('button', 'mini', '清空TS');
    clear.type = 'button';
    clear.addEventListener('click', function () { h.clearSituation(); close(); });
    acts.appendChild(edit);
    acts.appendChild(clear);
    body.appendChild(acts);
  }

  // ---- 标签页 3：TS数据下载 ----
  /**
   * TS简报页。
   *
   * 简报是**引擎侧**生成的（engine/brief.py），随数据集一起下发，前端只负责呈现。
   * 不在前端重算的理由：它要引用案例与目标定义，而且必须与画面上显示的等级出自
   * 同一份推理输出——前端另算一遍迟早会和引擎的规则漂移。
   */
  function renderBrief(body) {
    const h = host();
    const data = h.state.data;
    if (!data) {
      const s = section(body, 'TS简报');
      s.appendChild(el('div', 'sit-hint',
        '未加载TS。请先从顶栏「TS案例」选择一个案例，再回来看简报。'));
      return;
    }
    const brief = data.brief;
    if (!brief) {
      const s = section(body, 'TS简报');
      s.appendChild(el('div', 'sit-hint',
        '该数据集里没有简报字段。简报由 engine/brief.py 生成并随数据集下发，' +
        '请重新运行 scripts/gen_demo_trajectory.py 生成数据。'));
      return;
    }
    if (brief.empty) {
      const s = section(body, 'TS简报');
      s.appendChild(el('div', 'sit-hint', brief.headline || '该TS没有可用帧。'));
      return;
    }

    const head = section(body, brief.case_name || brief.case_id || 'TS简报');
    head.appendChild(el('div', 'brief-headline', brief.headline || ''));
    const meta = el('div', 'sit-hint',
      '峰值 ' + (brief.peak_level_label || '—') +
      '（第 ' + brief.peak_frame + ' 帧' +
      (brief.peak_clock ? '，' + brief.peak_clock : '') + '） · 结束 ' +
      (brief.final_level_label || '—') + ' · 趋势 ' + (brief.trend || '—') +
      ' · 共 ' + brief.frame_count + ' 帧');
    head.appendChild(meta);

    (brief.sections || []).forEach(function (sec) {
      const s = section(body, sec.title || '');
      (sec.lines || []).forEach(function (line) {
        s.appendChild(el('div', 'brief-line', line));
      });
    });

    // 原文可复制：给人贴进报告用，也是"简报到底说了什么"的最终依据。
    const s = section(body, '原文');
    const area = document.createElement('textarea');
    area.className = 'brief-text';
    area.readOnly = true;
    area.value = brief.text || '';
    s.appendChild(area);
    const row = el('div', 'sit-row');
    const copy = el('button', 'mini', '复制原文');
    copy.type = 'button';
    copy.addEventListener('click', function () {
      area.select();
      try {
        if (document.execCommand) document.execCommand('copy');
        setHint('简报原文已复制。');
      } catch (e) {
        setHint('复制失败，请手动选择文本。');
      }
    });
    row.appendChild(copy);
    s.appendChild(row);
  }

  function renderDownload(body) {
    const h = host();
    const s = section(body, '下载');
    const hasData = !!h.state.data;

    const items = [
      ['当前帧 JSON', '导出当前时刻的完整TS快照：全部目标的位置、航向、航速与三级推理结果。',
        function () { h.exportFrame(); }, hasData],
      ['全帧 CSV', '把整段推演的逐帧目标状态导成表格，便于在 Excel 里做轨迹分析。',
        function () { h.exportCsv(); }, hasData],
      ['TS定义 JSON', '导出"TS输入"的定义（ZC范围 + 目标起止 + 图层），可以再导入复现。',
        function () { exportDefinition(); }, !!draft],
      ['导入TS定义', '从上面导出的定义文件恢复一份TS输入草稿。',
        function () { importDefinition(); }, true],
    ];

    items.forEach(function (it) {
      const row = el('div', 'sit-row');
      const btn = el('button', 'mini', it[0]);
      btn.type = 'button';
      btn.disabled = !it[3];
      btn.addEventListener('click', it[2]);
      row.appendChild(btn);
      row.appendChild(el('span', 'sit-hint', it[1]));
      s.appendChild(row);
    });

    const s2 = section(body, '说明');
    s2.appendChild(el('div', 'sit-hint',
      '原来的「导出数据」按钮已并入这里——导出是TS数据的一个出口，' +
      '和其他TS操作放在一起更好找。'));
  }

  function download(name, text, mime) {
    const blob = new Blob([text], { type: mime || 'application/json;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(function () { URL.revokeObjectURL(url); }, 2000);
  }

  function exportDefinition() {
    const d = ensureDraft();
    download('situation-definition.json', JSON.stringify(d, null, 2));
    setHint('已导出TS定义。');
  }

  function importDefinition() {
    const inp = document.createElement('input');
    inp.type = 'file';
    // 粘贴与文件走**同一条归一化路径**，避免两种入口对同一份文件给出不同结果。
    //
    // ⚠ 这里原先写的是 `…给出不同结果\n    inp.accept = '…'`，那个 `\n` 是**字面反斜杠 n**
    // 而不是换行，于是整条赋值被吃进 `//` 注释里 —— 文件选择框一直**没有任何过滤器**，
    // 而且不报错，只能从"选文件时能选到 .txt"这种间接现象发现。
    inp.accept = '.json,.xml,application/json,application/xml,text/xml';
    inp.addEventListener('change', function () {
      const f = inp.files && inp.files[0];
      if (!f) return;
      const reader = new FileReader();
      reader.onload = function () {
        const text = String(reader.result);
        const isXml = /\.xml$/i.test(f.name) || /^\s*</.test(text);
        const res = normalizeStructured(text, isXml ? 'xml' : 'json');
        if (res.errors.length) {
          setHint('导入失败：' + res.errors.join('；'));
          return;
        }
        draft = res.draft;
        open('input');
        setHint('已导入TS定义「' + (res.draft.name || '未命名') + '」（' + res.draft.targets.length
          + ' 个目标）' + (res.warnings.length ? '；' + res.warnings.join('；') : ''));
      };
      reader.readAsText(f);
    });
    inp.click();
  }

  // ------------------------------------------------------------------------
  // 生成并初始化
  // ------------------------------------------------------------------------

  function generate() {
    const d = ensureDraft();
    if (!d.targets.length) { setHint('至少要有一个目标。'); return; }
    const bad = d.targets.find(function (t) {
      return !t.id || !isFinite(t.start[0]) || !isFinite(t.start[1]) ||
        !isFinite(t.end[0]) || !isFinite(t.end[1]);
    });
    if (bad) { setHint('目标「' + (bad.id || '未命名') + '」的经纬度不完整。'); return; }
    const ids = d.targets.map(function (t) { return t.id; });
    if (new Set(ids).size !== ids.length) { setHint('目标 ID 有重复。'); return; }

    let data;
    try {
      data = buildSituation(d);
    } catch (e) {
      setHint('生成失败：' + e.message);
      return;
    }

    // 图层：把输入页的勾选同步到左栏开关，让"生成后按这里的勾选初始化"成立
    Array.prototype.forEach.call(
      document.querySelectorAll('[data-sit-layer]'), function (cb) {
        const target = document.getElementById(cb.dataset.sitLayer);
        if (target && target.checked !== cb.checked) {
          target.checked = cb.checked;
          target.dispatchEvent(new Event('change', { bubbles: true }));
        }
      });

    close();
    try {
      host().applySituationData(data, 'custom');

      // **重放标绘**：生成的 dataset 是引擎产出的，里面没有 plots 字段（标绘属于
      // 草稿，不属于数据集）。而 `applySituationData` 会清场重建，屏幕上的标绘会被
      // 一起清掉 —— 不重放的话"改个参数点生成，画好的标绘全没了"。
      //
      // 为什么不是"把 plots 塞进 data"：dataset 的字段集由引擎（engine/、离线数据集）
      // 定义，前端往里加一个它不认识的字段，只会让"前端自建TS"和"离线案例"两种数据集
      // 出现结构差异，装载管线就不再是同一份了。标绘留在草稿里、生成后重放，才与
      // "草稿=当前态势的一份副本"这个定义一致。
      //
      // ⚠ 这一步的失败**必须说出来**：原先写的是先 setHint('标绘未能全部还原…')、
      // 紧接着无条件 setHint('已生成并初始化…')，后一句把前一句立刻覆盖掉，
      // 标绘少了几条时用户一个字都看不到。现在并进同一条提示。
      let plotNote = '';
      if (window.Plotting && Plotting.importAll) {
        const rp = Plotting.importAll(d.plots || [], { replace: true });
        plotNote = ' · 标绘 ' + rp.added + ' 条';
        if (rp.errors.length) {
          plotNote += '（' + rp.errors.length + ' 条未能还原：' + rp.errors.join('；') + '）';
        }
        if (window.LayerPanel) LayerPanel.invalidate();
      }

      setHint('已生成并初始化「' + data.case_name + '」：' + data.targets.length +
        ' 个目标 · ' + (data.frames.length - 1) + ' 帧 · ZC范围 ' + d.aoi.join(', ') + plotNote);
    } catch (e) {
      setHint('装载失败：' + e.message);
    }
  }

  // ------------------------------------------------------------------------
  // 接线
  // ------------------------------------------------------------------------

  function bind() {
    const modal = document.getElementById('situationModal');
    if (!modal) return;

    const closeBtn = document.getElementById('sitClose');
    if (closeBtn) closeBtn.addEventListener('click', close);
    // 点遮罩关闭（点面板内部不关）
    modal.addEventListener('click', function (e) {
      if (e.target === modal) close();
    });
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' && !picking && !modal.classList.contains('hidden')) close();
    });

    const tabs = document.getElementById('sitTabs');
    if (tabs) {
      tabs.addEventListener('click', function (e) {
        const b = e.target.closest('[data-sit-tab]');
        if (!b) return;
        tab = b.dataset.sitTab;
        if (tab === 'input' || tab === 'manage') ensureDraft();
        render();
      });
    }
  }

  global.SituationPanel = {
    bind: bind, open: open, close: close,
    // 结构化注册的取证入口：**必须能从外部调到归一化函数**，否则"JSON 与 XML
    // 等价"这条只能分别测各自能跑通 —— 而两种格式可以悄悄漂移还不被发现。
    _normalize: normalizeStructured,
    _draftCount: function () { return draft && draft.targets ? draft.targets.length : 0; },
    _draftIds: function () {
      return draft && draft.targets ? draft.targets.map(function (x) { return x.id; }) : [];
    },
  };
})(window);
