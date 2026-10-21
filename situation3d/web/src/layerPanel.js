/**
 * 图层管理面板：**由注册表生成的 DOM**，不是写死的一堆复选框。
 *
 * 为什么要有这个文件
 * ------------------
 * 在这之前，「有哪些图层」这件事有两个互相独立的副本：
 *
 *   1. `layerRegistry.js` 的 `LAYERS`（底图与标绘）；
 *   2. `index.html` 里手写的一堆 `<label class="check">`（TS图层/目标图层/地图组件）。
 *
 * 两份副本的必然结果是：**在注册表里加了图层，面板上看不到**；在 HTML 里改了名字，
 * 注册表那边不知道。而"图层管理应该是动态的"这个要求，靠改其中一份是做不到的。
 *
 * 这里把面板变成注册表的**只读投影**：注册表里有什么就画什么，`register()` 立刻
 * 多一行、`unregister()` 立刻少一行，不需要碰 HTML。
 *
 * 刷新策略：签名比对，变了才重建
 * ------------------------------
 * `tick()` 每帧被调用一次，先算一个**签名**再决定要不要重建 DOM。签名里带
 * 修订号与各层要素数量 —— 后者是必要的，因为要素增删**不经过注册表**
 * （标绘由 `Plotting` 管、目标由案例数据管），只有靠数量比对才发现。
 *
 * 未展开的层**不把要素算进签名、也不枚举要素**：台海案例有 31 个目标 + 31 条航路，
 * 每帧枚举一遍是白烧 CPU。展开态通常只有 0~3 层。
 *
 * 依赖：window.LayerRegistry（须先加载）。
 */
(function () {
  'use strict';

  var root = null;
  var opts = {};
  var expanded = {};      // layerId -> true
  var editing = {};       // layerId|key -> true
  var lastSig = null;

  /** 组的显示顺序。注册表里出现但不在表内的组，排在最后。 */
  var GROUP_ORDER = ['TS图层', '目标图层', '底图', '标绘', '地图组件'];

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined && text !== null) n.textContent = text;
    return n;
  }

  function groupRank(g) {
    var i = GROUP_ORDER.indexOf(g);
    return i < 0 ? 99 : i;
  }

  // ------------------------------------------------------------------------
  // 签名
  // ------------------------------------------------------------------------

  /**
   * 面板内容的摘要。**只包含会影响 DOM 的东西**：层序、意图、状态、数量，
   * 以及**已展开层**的要素键与可见性。
   */
  function signature(rows) {
    var parts = ['rev:' + LayerRegistry.revision()];
    rows.forEach(function (s) {
      parts.push([s.group, s.subgroup || '', s.id, s.desired ? 1 : 0, s.feed, s.count,
                  s.expandable ? 1 : 0, expanded[s.id] ? 1 : 0].join('|'));
      if (expanded[s.id]) {
        LayerRegistry.elements(s.id).forEach(function (e) {
          parts.push('  ' + e.key + '|' + (e.visible ? 1 : 0));
        });
      }
    });
    var groups = [];
    rows.forEach(function (s) { if (groups.indexOf(s.group) < 0) groups.push(s.group); });
    parts.push('groups:' + groups.slice().sort(function (a, b) {
      return groupRank(a) - groupRank(b);
    }).join(','));
    return parts.join('\n');
  }

  // ------------------------------------------------------------------------
  // 渲染
  // ------------------------------------------------------------------------

  /** 一行图层：复选框 + 名称 + 数量徽标 + （可展开时）箭头。 */
  var LAYER_HINTS = {
    layerLabels: '目标标牌：目标编号 + 具体型号的屏幕标签',
    layerTargetProps: '目标属性：选中目标的 3D 侧视预览与实时属性（右下角图例上方）',
    'overlay.focusMask': '选中掩码：压暗全图，只保留选中目标及其轨迹范围',
    modelSensor: '传感器覆盖：按目标 sensor_range_km 绘制作用半径',
    layerTargetPanel: '目标元数据面板：选中目标后，在右栏显示状态、意图与参数',
    layerTracks: '历史轨迹：已发生轨迹的实线（计划虚线也归此层）',
    'situation.relations': '目标关系：选中目标与本方/对方目标的关系线',
    layerHotspots: '热点区域：按接触关系动态生成的热点',
    layerRoutes: '计划航路：目标预设航路折线',
  };

  function layerRow(status) {
    var row = el('div', 'lp-row');

    var label = el('label', 'check lp-check');
    var box = document.createElement('input');
    box.type = 'checkbox';
    box.checked = !!status.desired;
    box.id = 'layerRow-' + status.id;
    box.addEventListener('change', function () {
      if (typeof opts.onChange === 'function') opts.onChange(status.id, box.checked);
      invalidate();
    });
    label.appendChild(box);
    var nameEl = el('span', 'lp-label', status.label);
    if (LAYER_HINTS[status.id]) nameEl.title = LAYER_HINTS[status.id];
    label.appendChild(nameEl);
    row.appendChild(label);

    // 数量徽标：`null` = 这一层没有要素列表，不显示徽标（不是显示 0）。
    if (status.count !== null && status.count !== undefined) {
      row.appendChild(el('span', 'lp-count', String(status.count)));
    }

    // 状态：只在**不正常**时出标记，正常不占视觉。
    // hidden 不算异常（互斥是设计如此），但要说清是被谁顶掉的——否则就是
    // "开关开着却没东西"那种查不出来的现象。
    if (status.feed === 'hidden' || status.feed === 'failed' ||
        status.feed === 'pending' || status.feed === 'unsupported') {
      var tag = el('span', 'lp-feed lp-feed-' + status.feed,
        status.feed === 'hidden' ? '被顶掉'
          : status.feed === 'failed' ? '出错'
          : status.feed === 'pending' ? '加载中' : '不可用');
      if (status.feedDetail) tag.title = status.feedDetail;
      row.appendChild(tag);
    }

    if (status.expandable) {
      var btn = el('button', 'lp-expand', expanded[status.id] ? '\u25be' : '\u25b8');
      btn.type = 'button';
      btn.title = expanded[status.id] ? '收起要素' : '展开要素';
      btn.addEventListener('click', function () {
        expanded[status.id] = !expanded[status.id];
        invalidate();
      });
      row.appendChild(btn);
    }
    return row;
  }

  /**
   * 行内编辑表单。
   *
   * 为什么内联在要素行下面而不是弹模态框：改一个名称/归属却要盖住整张地图，
   * 用户没法一边看态势一边改。内联表单保持地图可见。
   */
  function editForm(layerId, item) {
    var wrap = el('div', 'lp-form');
    var inputs = {};
    (item.editFields || []).forEach(function (f) {
      var row = el('label', 'lp-form-row');
      row.appendChild(el('span', null, f.label));
      var inp;
      if (f.options) {
        inp = document.createElement('select');
        inp.className = 'plot-sel';
        f.options.forEach(function (o) {
          var op = document.createElement('option');
          op.value = String(o[0]); op.textContent = o[1];
          if (String(o[0]) === String(f.value)) op.selected = true;
          inp.appendChild(op);
        });
      } else {
        inp = document.createElement('input');
        inp.className = 'plot-input';
        inp.type = f.type || 'text';
        inp.value = (f.value === null || f.value === undefined) ? '' : f.value;
      }
      inputs[f.key] = inp;
      row.appendChild(inp);
      wrap.appendChild(row);
    });
    var btns = el('div', 'plot-row');
    var ok = el('button', 'mini', '保存');
    var no = el('button', 'mini', '取消');
    ok.type = 'button'; no.type = 'button';
    ok.addEventListener('click', function () {
      var values = {};
      Object.keys(inputs).forEach(function (k) { values[k] = inputs[k].value; });
      // onSave 会做真正的写入（`Plotting.update` 会对控制点数、合法性抛错）。
      // 这里必须兜住：面板的点击处理器一旦抛错就是未捕获错误，而用户的操作
      // （点保存）看起来"没反应"，实际错误只在控制台里。
      try {
        item.onSave(values);
        editing[layerId + '|' + item.key] = false;
      } catch (e) {
        alert('保存失败：' + e.message);
      }
      invalidate();
    });
    no.addEventListener('click', function () {
      editing[layerId + '|' + item.key] = false;
      invalidate();
    });
    btns.appendChild(ok); btns.appendChild(no);
    wrap.appendChild(btns);
    return wrap;
  }

  /** 一行要素：名称 + 副标题 + 该要素支持的操作。 */
  function elementRow(layerId, item) {
    var box = el('div', 'lp-elem-box');
    var row = el('div', 'lp-elem');

    if (item.actions && item.actions.indexOf('toggle') >= 0 && typeof item.onToggle === 'function') {
      var t = document.createElement('input');
      t.type = 'checkbox';
      t.className = 'lp-elem-toggle';
      t.checked = item.visible !== false;
      t.title = '单独显示/隐藏该要素';
      t.addEventListener('change', function () {
        item.onToggle(t.checked);
        invalidate();
      });
      row.appendChild(t);
    } else {
      // 占位对齐，让没有开关的要素行与有开关的左边对齐
      row.appendChild(el('span', 'lp-elem-toggle'));
    }

    var text = el('span', 'lp-elem-text');
    text.appendChild(el('b', null, item.label));
    if (item.sub) text.appendChild(el('i', 'lp-elem-sub', item.sub));
    if (item.note) text.appendChild(el('i', 'lp-elem-note', item.note));
    row.appendChild(text);

    var ACTIONS = {
      locate: ['\u2316', '定位', 'onLocate'],
      edit: ['\u270e', '编辑', 'onEdit'],
      delete: ['\ud83d\uddd1', '删除', 'onDelete'],
    };
    (item.actions || []).forEach(function (a) {
      if (a === 'toggle') return;
      var m = ACTIONS[a];
      if (!m) return;
      // 「编辑」有两种形态：带 `editFields` 的走行内表单（由面板渲染），
      // 没有表单的才要求提供者给 `onEdit`。**只按 `onEdit` 判存在会把表单形态的
      // 编辑按钮整个漏掉** —— 表现为"标绘能看不能改"，而且不报错。
      var hasForm = (a === 'edit' && item.editFields && item.editFields.length);
      if (!hasForm && typeof item[m[2]] !== 'function') return;
      var btn = el('button', 'lp-act', m[0]);
      btn.type = 'button';
      btn.title = m[1];
      btn.addEventListener('click', function () {
        // 「编辑」优先走**行内表单**（要素自带 editFields 时）；没有表单的才交给
        // 要素自己的 onEdit —— 这样"能编辑什么"由提供者声明，面板不必知道细节。
        if (a === 'edit' && item.editFields) {
          var k = layerId + '|' + item.key;
          editing[k] = !editing[k];
          invalidate();
          return;
        }
        item[m[2]]();
        invalidate();
      });
      row.appendChild(btn);
    });
    box.appendChild(row);
    if (editing[layerId + '|' + item.key] && item.editFields) {
      box.appendChild(editForm(layerId, item));
    }
    return box;
  }

  function appendLayerRow(host, s) {
    host.appendChild(layerRow(s));
    if (expanded[s.id]) {
      var list = LayerRegistry.elements(s.id);
      var wrap = el('div', 'lp-elems');
      if (!list.length) {
        wrap.appendChild(el('div', 'lp-empty', '（暂无要素）'));
      } else {
        list.forEach(function (it) { wrap.appendChild(elementRow(s.id, it)); });
      }
      host.appendChild(wrap);
    }
  }

  /** 海/陆/空过滤：只放在目标图层条目之前。 */
  function domainFilterRow() {
    var wrap = el('div', 'lp-filter-row');
    var cur = (typeof opts.getFilters === 'function') ? (opts.getFilters() || {}) : {};
    [['sea', '海上'], ['land', '陆上'], ['air', '空中']].forEach(function (p) {
      var lab = el('label', 'check');
      var box = document.createElement('input');
      box.type = 'checkbox';
      box.checked = cur[p[0]] !== false;
      box.setAttribute('data-domain-filter', p[0]);
      box.addEventListener('change', function () {
        if (typeof opts.onFilter === 'function') opts.onFilter(p[0], box.checked);
        invalidate();
      });
      lab.appendChild(box);
      lab.appendChild(el('span', null, p[1]));
      wrap.appendChild(lab);
    });
    return wrap;
  }

  function render() {
    if (!root) return;
    // 记下滚动位置：重建会把 scrollTop 打回 0，用户展开一个要素列表就"跳回顶部"。
    var scroller = root.scrollParent;
    var keepTop = scroller ? scroller.scrollTop : 0;

    var rows = LayerRegistry.status();
    var groups = [];
    rows.forEach(function (s) {
      if (groups.indexOf(s.group) < 0) groups.push(s.group);
    });
    groups.sort(function (a, b) { return groupRank(a) - groupRank(b); });

    root.innerHTML = '';
    groups.forEach(function (g) {
      var box = el('details', 'layer-group panel-block');
      box.open = true;
      box.appendChild(el('summary', null, g));
      var groupRows = rows.filter(function (s) { return s.group === g; });
      var direct = groupRows.filter(function (s) { return !s.subgroup && s.panel !== false; });
      direct.sort(function (a, b) { return (a.z - b.z) || (a.id < b.id ? -1 : 1); })
        .forEach(function (s) { appendLayerRow(box, s); });
      var subs = {};
      groupRows.filter(function (s) { return !!s.subgroup && s.panel !== false; }).forEach(function (s) {
        (subs[s.subgroup] = subs[s.subgroup] || []).push(s);
      });
      Object.keys(subs).forEach(function (sg) {
        var sub = el('details', 'layer-subgroup');
        sub.open = true;
        sub.appendChild(el('summary', null, sg));
        subs[sg].sort(function (a, b) { return (a.z - b.z) || (a.id < b.id ? -1 : 1); })
          .forEach(function (s) { appendLayerRow(sub, s); });
        box.appendChild(sub);
      });
      root.appendChild(box);
    });

    if (scroller) scroller.scrollTop = keepTop;
    lastSig = signature(rows);
  }

  /** 让下一次 tick() 重建。任何会改变面板内容的操作后调用它。 */
  function invalidate() { lastSig = null; }

  /** 每帧调用：签名没变就什么都不做。 */
  function tick() {
    if (!root || !window.LayerRegistry) return false;
    var sig = signature(LayerRegistry.status());
    if (sig === lastSig) return false;
    render();
    return true;
  }

  function bind(rootEl, options) {
    root = rootEl || null;
    opts = options || {};
    // 左栏的滚动容器：面板重建时要把 scrollTop 还原回去
    root.scrollParent = root ? root.closest('.panel-left') : null;
    invalidate();
    tick();
    return !!root;
  }

  window.LayerPanel = {
    bind: bind,
    render: render,
    tick: tick,
    invalidate: invalidate,
    expandedKeys: function () { return Object.keys(expanded); },
    _signature: function () { return signature(LayerRegistry.status()); },
  };
})();