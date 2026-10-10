#!/usr/bin/env node
/**
 * entityLod.js 的判据用例（纯 Node，无框架）。
 *
 * 为什么有这个文件：任务三的切换逻辑此前只有「设计文档」，没有任何可执行的验证。
 * decide() 是纯函数，正适合用 Node 直接跑——本机无前端测试框架，但 Node 可用。
 *
 * 用法：node tools/entity_lod_test.js
 */
'use strict';

const path = require('path');
const L = require(path.join(__dirname, '..', 'web', 'src', 'entityLod.js'));

let pass = 0;
const failures = [];

function eq(actual, expected, label) {
  if (actual === expected) { pass++; return; }
  failures.push(`${label}: 期望 ${JSON.stringify(expected)}，实际 ${JSON.stringify(actual)}`);
}

// --- 1. 基本阈值方向 ---
eq(L.decide(5000, L.MODE_ICON).mode, L.MODE_MODEL, '低空(5km) 从 icon 应转 model');
eq(L.decide(50000, L.MODE_ICON).mode, L.MODE_ICON, '高空(50km) 保持 icon');
eq(L.decide(20000, L.MODE_ICON).mode, L.MODE_MODEL, '恰好 20km 应转 model（<=）');
eq(L.decide(20001, L.MODE_ICON).mode, L.MODE_ICON, '20.001km 不转（严格边界）');

// --- 2. 滞回带：20—24km 内**保持当前形态**，不来回切 ---
eq(L.decide(22000, L.MODE_ICON).mode, L.MODE_ICON, '滞回带内(icon) 不切');
eq(L.decide(22000, L.MODE_MODEL).mode, L.MODE_MODEL, '滞回带内(model) 不切');
eq(L.decide(22000, L.MODE_ICON).changed, false, '滞回带内(icon) changed=false');
eq(L.decide(22000, L.MODE_MODEL).changed, false, '滞回带内(model) changed=false');
eq(L.decide(24000, L.MODE_MODEL).mode, L.MODE_MODEL, '恰好 24km 仍 model（需 >）');
eq(L.decide(24001, L.MODE_MODEL).mode, L.MODE_ICON, '超过 24km 才回 icon');

// --- 3. 非法高度：保持当前形态，且不报 changed ---
eq(L.decide(undefined, L.MODE_ICON).mode, L.MODE_ICON, 'undefined 高度保持 icon');
eq(L.decide(undefined, L.MODE_MODEL).mode, L.MODE_MODEL, 'undefined 高度保持 model');
eq(L.decide(NaN, L.MODE_ICON).changed, false, 'NaN 不触发切换');
eq(L.decide(-1, L.MODE_ICON).changed, false, '负高度不触发切换');
eq(L.decide(undefined, L.MODE_ICON).reason, 'height-invalid', '非法高度 reason');

// --- 4. 未知 currentMode 按 icon 处理（防御性）---
eq(L.decide(50000, 'bogus').mode, L.MODE_ICON, '未知形态按 icon');
eq(L.decide(5000, null).mode, L.MODE_MODEL, 'null 形态按 icon 后可下降转 model');

// --- 5. 自定义阈值（确认常量可覆盖，便于日后定稿）---
const cfg = { enter3dHeight: 1000, exit3dHeight: 2000 };
eq(L.decide(1500, L.MODE_ICON, cfg).mode, L.MODE_ICON, '自定义阈值下 1500m 不转');
eq(L.decide(500, L.MODE_ICON, cfg).mode, L.MODE_MODEL, '自定义阈值下 500m 转 model');
eq(L.decide(2500, L.MODE_MODEL, cfg).mode, L.MODE_ICON, '自定义阈值下 2500m 回 icon');

// --- 6. 模型映射：拿不准必须返回 null（保留 2D），不得瞎映射 ---
eq(L.modelUrlFor('ship_large'), './assets/models/ship-large.glb', 'ship_large 映射');
eq(L.modelUrlFor('nope'), null, '未知类别返回 null');
eq(L.modelUrlFor(undefined), null, 'undefined 类别返回 null');
eq(L.formClassOf(null), null, 'null 定义返回 null');
eq(L.formClassOf({ domain: 'air' }), null, '空中目标无资产 -> null（不得映射成船）');
eq(L.formClassOf({ domain: 'land', platform_class: '主战坦克' }), null, '车辆暂无资产 -> null');
eq(L.formClassOf({ domain: 'sea', platform_class: '052D 驱逐舰' }), 'ship_large', '052D -> ship_large');
eq(L.formClassOf({ domain: 'sea', platform_class: '054A 护卫舰' }), 'ship_medium', '054A -> ship_medium');
eq(L.formClassOf({ domain: 'sea', platform_class: '903 补给舰' }), 'ship_cargo', '903 -> ship_cargo');
eq(L.formClassOf({ domain: 'sea', platform_class: '沱江级 导弹艇' }), 'boat_speed', '沱江级 -> boat_speed');
eq(L.formClassOf({ domain: 'sea', form: 'boat_tug' }), 'boat_tug', '显式 form 优先');
eq(L.formClassOf({ domain: 'sea', platform_class: '驱逐舰' }), 'ship_large', '驱逐舰(实测类别词) -> ship_large');
eq(L.formClassOf({ domain: 'sea', platform_class: '护卫舰' }), 'ship_medium', '护卫舰(实测类别词) -> ship_medium');
eq(L.formClassOf({ domain: 'land', platform_class: '岸舰导弹' }), null, '岸舰导弹属陆域 -> null（不得当船）');
eq(L.formClassOf({ domain: 'land', platform_class: '地面防空' }), null, '地面防空 -> null');

// --- 7. 每一个映射到的模型文件都应真实存在（防路径写错）---
const fs = require('fs');
for (const [k, url] of Object.entries(L.MODEL_MAP)) {
  const p = path.join(__dirname, '..', 'web', url.replace(/^\.\//, ''));
  if (fs.existsSync(p)) { pass++; } else { failures.push(`${k} 映射的模型文件不存在: ${p}`); }
}


// --- 8. 尺度：任意单位资产必须按真实长度缩放（否则模型被 minimumPixelSize 夹死）---
eq(L.scaleFor('ship_large') > 10, true, 'ship_large 需放大 10 倍以上（13.1 单位 -> 155 m）');
eq(Math.round(L.scaleFor('ship_medium') * 10) / 10, 12.7, 'ship_medium scale=135/10.6');
eq(L.scaleFor('nope'), null, '未知形态 scale=null');
eq(L.scaleFor(undefined), null, 'undefined 形态 scale=null');
for (const k of Object.keys(L.MODEL_MAP)) {
  if (L.scaleFor(k) === null) { failures.push(k + ' 在 MODEL_MAP 里但没有 scale（尺度表与映射表不同步）'); }
  else { pass++; }
}

console.log(`entityLod 判据：${pass} 通过, ${failures.length} 失败`);
if (failures.length) {
  failures.forEach(f => console.log('  FAIL ' + f));
  process.exit(1);
}
console.log('ALL CHECKS PASSED');
