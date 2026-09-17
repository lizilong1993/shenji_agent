/* situation3d — 态势：输入 / 管理 / 数据下载
 *
 * 三件事都围绕"当前态势"，所以共用一个三标签页模态框，而不是三个入口。
 *
 * 设计要点
 * --------
 * 1. **战场范围支持两种确定方式**：在地球上框选，或直接输入 AOI 经纬度。
 *    框选用的是 `getPickRay` + `globe.pick` 取地面点，与"光标位置信息"同源。
 * 2. **生成的自建态势与案例走完全同一条装载管线**：`SituationHost.applySituationData()`。
 *    案例是 Python 侧离线生成的，自建态势是前端即时生成的，但装载逻辑只有一份，
 *    不会出现"案例能跑、自建态势少初始化了某样东西"。
 * 3. **目标运动是匀速直线**：起点 → 终点按帧线性插值，航向取两点间的真方位角，
 *    航速由距离与总时长反算。这足以驱动三维态势的全部显示与推理链路。
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
   * 从当前态势派生草稿：把已有目标的起止位置与属性带过来，便于"改一改再生成"。
   * 没有态势时给一个台海的两目标最小样例。
   */
  function draftFromCurrent() {
    const h = host();
    const data = h.state.data;
    if (!data || !data.targets || !data.targets.length) {
      return {
        name: '自建态势',
        aoi: [117.5, 23.0, 121.0, 26.0],
        hours: DEFAULT_HOURS,
        targets: [blankTarget(1, 'red'), blankTarget(1, 'blue')],
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
      name: (data.case_name || '态势') + '（副本）',
      aoi: bbox.slice(),
      hours: DEFAULT_HOURS,
      targets: targets,
    };
  }

  function ensureDraft() {
    if (!draft) draft = draftFromCurrent();
    return draft;
  }

  // ------------------------------------------------------------------------
  // 由草稿生成态势数据
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
        waypoints: [t.start, t.end],
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
      case_name: d.name || '自建态势',
      case_description: '由态势输入在前端生成：' + targets.length + ' 个目标，' +
        ticks + ' 帧，战场范围 ' + aoi.join(', ') + '。',
      region: '自建战场',
      scenario_id: null,
      scenario_name: d.name || '自建态势',
      provenance: 'frontend-situation-input',
      provenance_note: '目标按起点→终点匀速直线运动，用于快速搭建态势；不代表真实航路。',
      start_time_utc: new Date().toISOString().replace(/\.\d+Z$/, 'Z'),
      tick_seconds: TICK_SECONDS,
      tick_count: ticks + 1,
      camera: { lon: round4(cx), lat: round4(cy), height: 900000, heading: 0, pitch: -75 },
      // 战场范围本身作为一个热点区域，右侧表格与区域级推理才有内容
      hotspots: [{
        name: (d.name || '自建战场') + ' 范围',
        bbox: aoi.slice(),
        center: [round4(cy), round4(cx)],
        span_deg: [Math.round((aoi[2] - aoi[0]) * 1000) / 1000,
                   Math.round((aoi[3] - aoi[1]) * 1000) / 1000],
        note: '态势输入确定的战场范围',
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
  // 框选战场范围
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
    setHint('框选战场范围：在地球上按住左键拖出一个矩形（Esc 取消）');
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
      setHint('战场范围已框选：' + d.aoi.join(', '));
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

  // ---- 标签页 1：态势输入 ----
  function renderInput(body) {
    const d = ensureDraft();

    // 数据来源一行：草稿是从当前态势派生的还是最小样例，一眼能看出来
    const src = el('div', 'sit-hint');
    const data = host().state.data;
    src.textContent = '草稿来源：' + (data
      ? '当前态势「' + data.case_name + '」（' + data.targets.length + ' 个目标）'
      : '最小样例（当前未加载态势）') + ' · 草稿目标 ' + d.targets.length + ' 个';
    body.appendChild(src);

    // 战场范围
    const s1 = section(body, '战场范围');
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

    // 态势名称与时长
    const s2 = section(body, '态势参数');
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
      '新态势生成后按这里的勾选初始化图层。也可随时在左栏「图层管理」里改。'));
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
    const gen = el('button', 'primary', '生成并初始化双方态势');
    gen.type = 'button';
    gen.addEventListener('click', function () { generate(); });
    const reset = el('button', 'mini', '重置为当前态势');
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

  // ---- 标签页 2：态势管理 ----
  function renderManage(body) {
    const h = host();
    const data = h.state.data;
    const s1 = section(body, '当前态势');
    if (!data) {
      s1.appendChild(el('div', 'sit-hint', '未加载态势。到「态势输入」生成一个，或从顶栏「态势案例」选一个。'));
      return;
    }
    const info = [
      ['名称', data.case_name],
      ['来源', data.provenance === 'frontend-situation-input' ? '态势输入（前端生成）' : ('案例 ' + (data.case_id || ''))],
      ['目标数', data.targets.length],
      ['帧数', (data.frames.length - 1) + '（步长 ' + data.tick_seconds + ' s）'],
      ['热点区域', (data.hotspots || []).length],
      ['战场范围', (data.hotspots && data.hotspots[0]) ? data.hotspots[0].bbox.join(', ') : '—'],
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
    const edit = el('button', 'primary', '在态势输入中编辑');
    edit.type = 'button';
    edit.addEventListener('click', function () { open('input'); });
    const clear = el('button', 'mini', '清空态势');
    clear.type = 'button';
    clear.addEventListener('click', function () { h.clearSituation(); close(); });
    acts.appendChild(edit);
    acts.appendChild(clear);
    body.appendChild(acts);
  }

  // ---- 标签页 3：态势数据下载 ----
  function renderDownload(body) {
    const h = host();
    const s = section(body, '下载');
    const hasData = !!h.state.data;

    const items = [
      ['当前帧 JSON', '导出当前时刻的完整态势快照：全部目标的位置、航向、航速与三级推理结果。',
        function () { h.exportFrame(); }, hasData],
      ['全帧 CSV', '把整段推演的逐帧目标状态导成表格，便于在 Excel 里做轨迹分析。',
        function () { h.exportCsv(); }, hasData],
      ['态势定义 JSON', '导出"态势输入"的定义（战场范围 + 目标起止 + 图层），可以再导入复现。',
        function () { exportDefinition(); }, !!draft],
      ['导入态势定义', '从上面导出的定义文件恢复一份态势输入草稿。',
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
      '原来的「导出数据」按钮已并入这里——导出是态势数据的一个出口，' +
      '和其他态势操作放在一起更好找。'));
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
    setHint('已导出态势定义。');
  }

  function importDefinition() {
    const inp = document.createElement('input');
    inp.type = 'file';
    inp.accept = '.json,application/json';
    inp.addEventListener('change', function () {
      const f = inp.files && inp.files[0];
      if (!f) return;
      const reader = new FileReader();
      reader.onload = function () {
        try {
          const parsed = JSON.parse(String(reader.result));
          if (!parsed.aoi || !Array.isArray(parsed.targets)) throw new Error('缺少 aoi 或 targets');
          draft = parsed;
          open('input');
          setHint('已导入态势定义「' + (parsed.name || '未命名') + '」。');
        } catch (e) {
          setHint('导入失败：' + e.message);
        }
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
      setHint('已生成并初始化「' + data.case_name + '」：' + data.targets.length +
        ' 个目标 · ' + (data.frames.length - 1) + ' 帧 · 战场范围 ' + d.aoi.join(', '));
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

  global.SituationPanel = { bind: bind, open: open, close: close };
})(window);
