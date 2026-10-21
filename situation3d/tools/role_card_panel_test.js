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
assert(html.indexOf('能力（角色卡）') >= 0, 'role-card section missing');
assert(html.indexOf('主战坦克（车辆）') >= 0, 'type text missing');
assert(html.indexOf('巡航 50 km/h / 最大 80 km/h') >= 0, 'speed text missing');
assert(html.indexOf('复合 · 编制 4') >= 0, 'armor text missing');
assert(html.indexOf('炮长热像仪 5 km') >= 0, 'sensor text missing');
assert(html.indexOf('125mm 尾翼稳定脱壳穿甲弹') >= 0, 'weapon text missing');
assert(html.indexOf('旧数据迁移快照') >= 0, 'legacy marker missing');

const plain = { id: 'X', name: '旧目标', icon: 'armor', affiliation: 'red', domain: 'land' };
const host2 = { innerHTML: '' };
sandbox.StatusPanels.renderTargetStatus(
  host2, plain, { lon: 120, lat: 23, alt: 0, heading: 0, speed_kn: 10, travelled_m: 0 },
  null, null, null, 0, null
);
assert(host2.innerHTML.indexOf('能力（角色卡）') < 0, 'section must not appear without card');

console.log('role_card_panel_test: PASS');
