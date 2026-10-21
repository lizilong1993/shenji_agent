#!/usr/bin/env node
/**
 * Node-only unit test for web/src/roleCard.js (no browser/Cesium needed).
 * Usage: node situation3d/tools/role_card_test.js
 */
'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');
const assert = require('assert');

const file = path.resolve(__dirname, '..', 'web', 'src', 'roleCard.js');
const src = fs.readFileSync(file, 'utf8');
const sandbox = { window: {} };
vm.createContext(sandbox);
vm.runInContext(src, sandbox);
const RC = sandbox.window.RoleCard;

assert(RC, 'RoleCard global missing');
assert.strictEqual(RC.summary(null), null);
assert.strictEqual(RC.summary({}), null);

const card = {
  class_id: 'land.vehicle',
  subclass_id: 'land.vehicle.mbt',
  confidence: 'legacy',
  summary: {
    class_label: '车辆',
    subclass_label: '主战坦克',
    confidence: 'legacy',
    unknown: false,
    max_speed_display: '80 km/h',
    cruise_speed_display: '50 km/h',
    armor_class: 4,
    max_blood: 4,
    sensors: [{ label: '炮长热像仪', kind: 'ir', range_km: 5 }],
    weapons: [{ label: '125mm 尾翼稳定脱壳穿甲弹', category: 'cannon', range_km: 2.5, ammo_max: 42 }],
    capacity: { personnel: 3, cargo_t: 0.5 }
  }
};
const rc = RC.summary({ role_card: card });
assert.strictEqual(rc.subclass_label, '主战坦克');
assert.strictEqual(RC.typeText(rc), '主战坦克（车辆）');
assert.strictEqual(RC.speedText(rc), '巡航 50 km/h / 最大 80 km/h');
assert.strictEqual(RC.armorText(rc), '复合 · 编制 4');
assert.strictEqual(RC.sensorText(rc), '炮长热像仪 5 km');
assert.strictEqual(RC.weaponText(rc), '125mm 尾翼稳定脱壳穿甲弹 3 km ×42');
assert.strictEqual(RC.capacityText(rc), '3 人 / 0.5 t');

// Fallback summary without role_card.summary (uses effective only).
const fallback = RC.summary({
  role_card: {
    class_id: 'sea.surface',
    subclass_id: 'sea.surface.destroyer',
    confidence: 'high',
    effective: { motion: { max_speed_ms: 15.4 }, protection: { armor_class: 0, max_blood: 4 }, capacity: {} }
  }
});
assert(fallback, 'fallback summary missing');
assert.strictEqual(fallback.subclass_label, 'sea.surface.destroyer');
assert.strictEqual(RC.typeText(fallback), 'sea.surface.destroyer');
assert.strictEqual(RC.speedText(fallback), '—');
assert.strictEqual(RC.armorText(fallback), '无装甲 · 编制 4');

// Unknown badge is carried through.
const unknown = RC.summary({ role_card: { summary: {
  class_label: '车辆', subclass_label: '车辆（未知小类）', unknown: true, confidence: 'unknown',
  max_speed_display: '40 km/h', cruise_speed_display: '30 km/h',
  armor_class: 1, max_blood: 1, sensors: [], weapons: [], capacity: {}
} } });
assert.strictEqual(RC.typeText(unknown), '车辆（未知小类）');
assert.strictEqual(RC.sensorText(unknown), '');
assert.strictEqual(RC.weaponText(unknown), '');

console.log('role_card_test: PASS');
