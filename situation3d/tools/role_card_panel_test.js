#!/usr/bin/env node
/**
 * Node-only integration test for the role-card block in statusPanels.js.
 * Stubs Symbology/IntentUi so no browser/Cesium is needed.
 * Usage: node situation3d/tools/role_card_panel_test.js
 */
'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');
const assert = require('assert');

const root = path.resolve(__dirname, '..');
const sandbox = {
  console: console,
  Symbology: { symbolUri: function () { return ''; } },
  IntentUi: null
};
sandbox.window = sandbox;
sandbox.document = { createElement: function () { return { className: '', textContent: '', appendChild: function () {} }; } };
vm.createContext(sandbox);

for (const rel of ['web/src/roleCard.js', 'web/src/statusPanels.js']) {
  vm.runInContext(fs.readFileSync(path.join(root, rel), 'utf8'), sandbox, { filename: rel });
}
assert(sandbox.RoleCard, 'RoleCard missing');
assert(sandbox.StatusPanels, 'StatusPanels missing');

const summary = {
  class_label: '车辆', subclass_label: '主战坦克', confidence: 'legacy', unknown: false,
  max_speed_display: '80 km/h', cruise_speed_display: '50 km/h',
  armor_class: 4, max_blood: 4, size_class: 'large',
  sensors: [{ label: '炮长热像仪', kind: 'ir', range_km: 5 }],
  weapons: [{ label: '125mm 尾翼稳定脱壳穿甲弹', category: 'cannon', range_km: 2.5, ammo_max: 42 }],
  capacity: { personnel: 3, cargo_t: 0.5 }
};
const def = {
  id: 'R15', name: '99A 主战坦克', icon: 'armor', affiliation: 'red', domain: 'land',
  icon_label: '装甲分队', role_card: { class_id: 'land.vehicle', subclass_id: 'land.vehicle.mbt', summary }
};
const host = { innerHTML: '' };
sandbox.StatusPanels.renderTargetStatus(
  host, def, { lon: 120.1, lat: 23.1, alt: 0, heading: 90, speed_kn: 10, travelled_m: 100 },
  null, null, null, 0, null
);
const html = host.innerHTML;
// 2026-10-22：角色卡能力块从右栏「目标意图」移除，迁入「目标属性」面板的「基本」分区。
assert(html.indexOf('角色卡') < 0, 'right panel must not mention role cards');
assert(html.indexOf('能力（') < 0, 'role-card section must be removed from right panel');
assert(html.indexOf('主战坦克（车辆）') < 0, 'role-card fields must not be in right panel');

// 迁移后由 RoleCard.basicRows() 提供给目标属性面板；格式化规则与旧版一致。
const rows = sandbox.RoleCard.basicRows(def);
const byLabel = {};
rows.forEach(function (r) { byLabel[r.label] = r.value; });
assert(byLabel['类型'] === '主战坦克（车辆）', 'type text missing');
assert(byLabel['速度'] === '巡航 50 km/h / 最大 80 km/h', 'speed text missing');
assert(byLabel['防护'] === '复合 · 编制 4', 'armor text missing');
assert(byLabel['传感器'] === '炮长热像仪 5 km', 'sensor text missing');
assert(byLabel['武器'] && byLabel['武器'].indexOf('125mm 尾翼稳定脱壳穿甲弹') >= 0,
       'weapon text missing');
assert(byLabel['数据'] === '旧数据迁移快照', 'legacy marker missing');

const plain = { id: 'X', name: '旧目标', icon: 'armor', affiliation: 'red', domain: 'land' };
const host2 = { innerHTML: '' };
sandbox.StatusPanels.renderTargetStatus(
  host2, plain, { lon: 120, lat: 23, alt: 0, heading: 0, speed_kn: 10, travelled_m: 0 },
  null, null, null, 0, null
);
assert(sandbox.RoleCard.basicRows(plain).length === 0, 'no card -> no rows');
assert(host2.innerHTML.indexOf('角色卡') < 0, 'plain target must not mention role cards');

console.log('role_card_panel_test: PASS');
