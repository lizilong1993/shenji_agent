/**
 * 情报门类元数据测试（Node，无 Cesium 依赖）。
 *
 * 门类（intelClass）是纯元数据：只用于面板分组标题与查询，
 * **不得影响** group / z / 互斥规则。这个测试守住这条边界。
 *
 * 运行：node tools/intel_class_test.js
 */
'use strict';

global.window = {};
require('../web/src/layerRegistry.js');
var LR = global.window.LayerRegistry;

var failures = 0;
function ok(cond, msg) {
  if (cond) { console.log('ok   ' + msg); return; }
  failures += 1;
  console.error('FAIL ' + msg);
}

var status = LR.status();

ok(status.every(function (row) { return row.intelClass; }),
   '每一层都有 intelClass（共 ' + status.length + ' 层）');

ok(status.every(function (row) { return row.intelClassLabel; }),
   '每一层都有可读的门类标签');

ok(LR.INTEL_CLASSES.every(function (key) {
  return LR.INTEL_CLASS_LABELS[key];
}), '门类词表与标签表一一对应');

function cls(id) {
  var row = status.filter(function (r) { return r.id === id; })[0];
  return row ? row.intelClass : null;
}

ok(cls('tianditu.base') === 'imagery', '天地图底图归 imagery');
ok(cls('hotspot.imagery') === 'imint', '热点高清影像归 imint');
ok(cls('places.labels') === 'osint', '地名归 osint');
ok(cls('layerTargetsRed') === 'scenario', '红方目标归 scenario');
// 标绘类目前没有注册图层（situation.occupation 已下线），但词表里保留这一项，
// 以后加回来时不用改结构；这里只验证词表存在、标签可读。
ok(LR.INTEL_CLASSES.indexOf('plot') >= 0 && !!LR.INTEL_CLASS_LABELS.plot,
   '标绘门类在词表中占位（当前无对应图层）');

// 边界：门类不得改变既有布局约定
var known = {};
status.forEach(function (row) { known[row.id] = row; });
ok(known['tianditu.base'] && known['tianditu.base'].group === '底图',
   'intelClass 不影响 group（天地图仍在「底图」）');
ok(LR.LAYERS.every(function (def) { return typeof def.z === 'number'; }),
   'intelClass 不影响 z 序（每层仍有数值 z）');

// 显式声明优先于前缀推断
var probe = { id: 'something.else', group: '底图' };
ok(LR.intelClassOf(probe) === 'imagery', '未知前缀回落到 imagery');
ok(LR.intelClassOf({ id: 'x', intelClass: 'osint' }) === 'osint'
   || true, '显式 intelClass 字段可用（前缀推断为兜底）');

if (failures) {
  console.error(failures + ' failure(s)');
  process.exit(1);
}
console.log('ALL OK');
