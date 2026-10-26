#!/usr/bin/env node
/**
 * entityLod.js 的判据与型号映射用例（纯 Node，无框架）。
 *
 * 用法：node tools/entity_lod_test.js
 */
'use strict';

const path = require('path');
const fs = require('fs');
const L = require(path.join(__dirname, '..', 'web', 'src', 'entityLod.js'));

let pass = 0;
const failures = [];

function eq(actual, expected, label) {
  if (actual === expected) { pass++; return; }
  failures.push(`${label}: 期望 ${JSON.stringify(expected)}，实际 ${JSON.stringify(actual)}`);
}

// --- 1. 阈值方向（低于 3000 m 显示 3D，3600 m 以上回 2D）---
eq(L.DEFAULTS.enter3dHeight, 3000, '进入 3D 的阈值是 3000 m');
eq(L.DEFAULTS.exit3dHeight, 3600, '退出 3D 的阈值是 3600 m');
eq(L.decide(2000, L.MODE_ICON).mode, L.MODE_MODEL, '低空(2km) 从 icon 应转 model');
eq(L.decide(50000, L.MODE_ICON).mode, L.MODE_ICON, '高空(50km) 保持 icon');
eq(L.decide(3000, L.MODE_ICON).mode, L.MODE_MODEL, '恰好 3000 m 应转 model（<=）');
eq(L.decide(3001, L.MODE_ICON).mode, L.MODE_ICON, '3001 m 不转（严格边界）');

// --- 2. 滞回带 ---
eq(L.decide(3300, L.MODE_ICON).mode, L.MODE_ICON, '滞回带内(icon) 不切');
eq(L.decide(3300, L.MODE_MODEL).mode, L.MODE_MODEL, '滞回带内(model) 不切');
eq(L.decide(3600, L.MODE_MODEL).mode, L.MODE_MODEL, '恰好 3600 m 仍 model（需 >）');
eq(L.decide(3601, L.MODE_MODEL).mode, L.MODE_ICON, '超过 3600 m 才回 icon');

// --- 3. 非法高度 ---
eq(L.decide(undefined, L.MODE_ICON).mode, L.MODE_ICON, 'undefined 高度保持 icon');
eq(L.decide(NaN, L.MODE_ICON).changed, false, 'NaN 不触发切换');
eq(L.decide(-1, L.MODE_ICON).changed, false, '负高度不触发切换');
eq(L.decide(undefined, L.MODE_ICON).reason, 'height-invalid', '非法高度 reason');

// --- 4. 预设阈值可覆盖 ---
const cfg = { enter3dHeight: 1000, exit3dHeight: 2000 };
eq(L.decide(1500, L.MODE_ICON, cfg).mode, L.MODE_ICON, '自定义阈值下 1500m 不转');
eq(L.decide(500, L.MODE_ICON, cfg).mode, L.MODE_MODEL, '自定义阈值下 500m 转 model');

// --- 5. 大类回退：所有大类都必须有模型，未知返回 null ---
eq(L.modelUrlFor('nope'), null, '未知大类返回 null');
eq(L.modelUrlFor(undefined), null, 'undefined 返回 null');
eq(L.modelUrlFor({ domain: 'air' }), null, '判不出大类 -> null（保留 2D 图标）');
for (const k of Object.keys(L.MODEL_MAP)) {
  eq(typeof L.modelUrlFor(k), 'string', `大类 ${k} 有模型`);
}

// --- 6. formClassOf：红蓝双方实测数据 ---
eq(L.formClassOf({ domain: 'sea', platform_class: '驱逐舰', model: '052D 型驱逐舰' }), 'destroyer', '052D -> destroyer');
eq(L.formClassOf({ domain: 'sea', platform_class: '护卫舰', model: '054A 型护卫舰' }), 'frigate', '054A -> frigate');
eq(L.formClassOf({ domain: 'sea', platform_class: '巡逻舰', model: '沱江级巡逻舰' }), 'corvette', '沱江 -> corvette');
eq(L.formClassOf({ domain: 'sea', platform_class: '登陆舰', model: '071 型综合登陆舰' }), 'amphibious', '登陆舰 -> amphibious');
eq(L.formClassOf({ domain: 'sea', platform_class: '补给舰', model: '903 型综合补给舰' }), 'replenishment', '补给舰 -> replenishment');
eq(L.formClassOf({ domain: 'land', platform_class: '装甲分队', model: '99A 主战坦克' }), 'tank', '99A -> tank');
eq(L.formClassOf({ domain: 'land', platform_class: '装甲分队', model: '15 式轻型坦克' }), 'light_tank', '15 式 -> light_tank');
eq(L.formClassOf({ domain: 'land', platform_class: '两栖战车', model: '05 式两栖突击车' }), 'amphibious_vehicle', '05 式 -> amphibious_vehicle');
eq(L.formClassOf({ domain: 'land', platform_class: '炮兵分队', model: 'PLZ-05 自行加榴炮' }), 'spg', 'PLZ-05 -> spg');
eq(L.formClassOf({ domain: 'land', platform_class: '炮兵分队', model: 'M777 超轻型榴弹炮' }), 'howitzer', 'M777 -> howitzer');
eq(L.formClassOf({ domain: 'land', platform_class: '炮兵分队', model: 'PHL-03 远程火箭炮' }), 'mlrs', 'PHL-03 -> mlrs');
eq(L.formClassOf({ domain: 'land', platform_class: '地面防空', model: '红旗-9B 远程防空导弹' }), 'sam', 'HQ-9B -> sam');
eq(L.formClassOf({ domain: 'land', platform_class: '岸舰导弹', model: '鹰击-62 岸舰导弹' }), 'anti_ship', 'YJ-62 -> anti_ship');
eq(L.formClassOf({ domain: 'land', platform_class: '运输队', model: '高原运输车队' }), 'truck', '运输队 -> truck');
eq(L.formClassOf({ domain: 'air', platform_class: '战斗机', model: '歼-16 多用途战斗机' }), 'fighter', '歼-16 -> fighter');
eq(L.formClassOf({ domain: 'air', platform_class: '预警机', model: '空警-500 预警机' }), 'aew', '空警-500 -> aew');
eq(L.formClassOf({ domain: 'air', platform_class: '武装直升机', model: '直-10 武装直升机' }), 'helicopter', '直-10 -> helicopter');
eq(L.formClassOf({ domain: 'air', platform_class: '无人机', model: '中高空长航时无人机' }), 'uav', '无人机 -> uav');
eq(L.formClassOf({ domain: 'air', platform_class: '无人机', icon: 'aew' }), 'uav', '无人机(icon=aew) 仍 -> uav');

// --- 7. 精确型号优先，缺失型号回退大类 ---
const exact = [
  ['052D 型驱逐舰', 'type_052c.glb'],
  ['054A 型护卫舰', 'type_054a.glb'],
  ['055 型驱逐舰', 'type_055.glb'],
  ['075 型两栖攻击舰', 'type_075.glb'],
  ['M60A3 主战坦克', 'type_m60a3.glb'],
  ['M48H 主战坦克', 'type_m60a3.glb'],
  ['M142 HIMARS', 'type_himars.glb'],
  ['歼-20 战斗机', 'type_j20.glb'],
  ['P-3C 反潜巡逻机', 'type_p3c.glb'],
  ['AH-64E 阿帕奇', 'type_ah64e.glb'],
  ['903 型综合补给舰', 'type_903.glb'],
  ['071 型综合登陆舰', 'type_071.glb'],
  ['056A 型轻型护卫舰', 'type_056a.glb'],
  ['成功级护卫舰', 'type_perry.glb'],
  ['沱江级巡逻舰', 'type_tuochiang.glb'],
  ['基隆级驱逐舰', 'type_kidd.glb'],
  ['磐石级油弹补给舰', 'type_panshi.glb'],
  ['99A 主战坦克', 'type_ztz99a.glb'],
  ['M1A2T 主战坦克', 'type_m1a2.glb'],
  ['CM-11 主战坦克', 'type_m60a3.glb'],
  ['15 式轻型坦克', 'type_ztq15.glb'],
  ['05 式两栖突击车', 'type_ztd05.glb'],
  ['PLZ-05 自行加榴炮', 'type_plz05.glb'],
  ['M109A6 自行榴弹炮', 'type_m109a6.glb'],
  ['M777 超轻型榴弹炮', 'type_m777.glb'],
  ['PHL-03 远程火箭炮', 'type_phl03.glb'],
  ['红旗-9B 远程防空导弹', 'type_hq9.glb'],
  ['红旗-16 中程防空导弹', 'type_hq16.glb'],
  ['爱国者-3 防空导弹', 'type_pac3.glb'],
  ['鹰击-62 岸舰导弹', 'type_yj62.glb'],
  ['歼-16 多用途战斗机', 'type_j16.glb'],
  ['F-16V 战斗机', 'type_f16c.glb'],
  ['IDF 经国号战斗机', 'type_idf.glb'],
  ['阵风战斗机', 'type_rafale.glb'],
  ['空警-500 预警机', 'type_kj500.glb'],
  ['KJ-2000 预警机', 'type_kj2000.glb'],
  ['E-2K 预警机', 'type_e2d.glb'],
  ['直-10 武装直升机', 'type_z10.glb'],
  ['中高空长航时无人机', 'type_mq9.glb'],
  ['高原运输车队', 'type_eq240.glb'],
];
for (const [model, file] of exact) {
  const def = { domain: 'sea', platform_class: '', model };
  const d2 = Object.assign({}, def);
  // 让 domain 按关键词猜测（这里只关心 model 命中）
  d2.domain = /无人机|预警机|战斗机|直升机/.test(model) ? 'air'
    : /坦克|战车|炮|导弹|车队/.test(model) ? 'land' : 'sea';
  const got = L.modelUrlFor(d2);
  eq(got, './assets/models/' + file, `型号 ${model} -> ${file}`);
}

// --- 8. 模型文件真实存在 ---
const urls = new Set(Object.values(L.MODEL_MAP));
for (const rule of L.TYPE_RULES) urls.add('./assets/models/' + rule[1] + '.glb');
for (const url of urls) {
  const p = path.join(__dirname, '..', 'web', url.replace(/^\.\//, ''));
  if (fs.existsSync(p)) pass++;
  else failures.push(`映射的模型文件不存在: ${p}`);
}

// --- 8b. 尾流/尾气长度表：每个大类都要有代表长度 ---
for (const k of Object.keys(L.MODEL_MAP)) {
  if (L.REAL_LENGTH_M[k] > 0) pass++;
  else failures.push(k + ' 缺少 REAL_LENGTH_M（targetFx 尾流会退化）');
}

// --- 9. 尺度：资产按米归一，scale 恒为 1 ---
for (const k of Object.keys(L.MODEL_MAP)) {
  eq(L.scaleFor(k), 1, `${k} scale=1`);
}
eq(L.scaleFor('nope'), null, '未知大类 scale=null');
eq(L.scaleFor(undefined), null, 'undefined scale=null');

// --- 10. 朝向：全部 +Y；无轮节点 ---
for (const k of Object.keys(L.MODEL_MAP)) {
  eq(L.forwardAxis(k), '-X', `${k} forwardAxis=-X`);
}
eq(L.wheelNodes('tank').length, 0, '新资产没有独立轮节点');
eq(L.wheelNodes('nope').length, 0, '未知大类没有轮节点');

console.log(`entityLod 判据：${pass} 通过, ${failures.length} 失败`);
if (failures.length) {
  failures.forEach(f => console.log('  FAIL ' + f));
  process.exit(1);
}
console.log('ALL CHECKS PASSED');
