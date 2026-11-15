#!/usr/bin/env node
/**
 * target_status_test.js — 战损档位必须读引擎自己的 HP。
 *
 * 背景（2026-11-15 实测）：`TargetStatus.at()` 只在案例声明
 * `provenance.runtime === 'engine_run'` 时才读每帧 HP，而四个随包案例里
 * 两个战役数据集写的是字符串 `"fictional_demo"` / `"simulated"`，于是全部
 * 落到按推理暴露量反推的老启发式上：
 *
 *   taiwan_campaign        47520 行中 62.3 % 档位错误（含 28.3 % 已击毁仍判"正常"）
 *   india_china_west       28800 行中 61.6 % 档位错误
 *
 * 两个写入方的字段名与量纲还不一样：
 *   scripts/run_situation_case.py  strength = blood / max_blood（0-1 比值）
 *   campaign.py                   strength = u.strength（0-100，战损时不变）
 *                                 blood    = 整数强度点，忠实反映战损
 *
 * 这里锁住的行为：
 *   1. 三种写入方都能读出正确的 0-100 HP；
 *   2. `alive === false` / `status === 'destroyed'` 优先于数值，判为击毁；
 *   3. 没有 HP 字段的纯移动案例（taiwan_dense_transfer_v1）仍走老启发式，
 *      不受这次改动影响；
 *   4. 档位阈值与血条口径一致。
 */
'use strict';

const path = require('path');
const fs = require('fs');

global.window = global;
require(path.join(__dirname, '..', 'web', 'src', 'targetStatus.js'));

const TS = global.TargetStatus;
const DATA = path.join(__dirname, '..', 'web', 'data', 'situations');

let failures = 0;
function check(name, cond, detail) {
  if (cond) { console.log('  ok   ' + name); return; }
  failures++;
  console.log('  FAIL ' + name + (detail ? '  —  ' + detail : ''));
}

function load(caseName) {
  return JSON.parse(fs.readFileSync(path.join(DATA, caseName + '.json'), 'utf8'));
}

/**
 * 全帧比对：HP 必须与引擎给的数值逐行相等，存活判定必须与 alive 一致。
 * 返回 {hpBad, deadBad, rows}。
 */
function audit(data) {
  const maxBlood = {};
  (data.targets || []).forEach(function (t) { maxBlood[t.id] = t.max_blood; });
  let hpBad = 0, deadBad = 0, rows = 0;
  (data.frames || []).forEach(function (fr, t) {
    (fr.targets || []).forEach(function (r) {
      if (!maxBlood[r.id]) return;          // 该案例这一行没有 HP 字段
      rows++;
      const got = TS.at(data, r.id, t);
      const want = Math.round(r.blood / maxBlood[r.id] * 1000) / 10;
      if (Math.abs(got.hp - want) > 0.05) hpBad++;
      if ((r.status === 'destroyed') !== (got.status === 3)) deadBad++;
    });
  });
  return { hpBad: hpBad, deadBad: deadBad, rows: rows };
}

console.log('1. 战役数据集：HP 与存活判定逐行对齐引擎');
['taiwan_campaign', 'india_china_west'].forEach(function (name) {
  const data = load(name);
  const r = audit(data);
  check(name + ' 有 HP 行', r.rows > 1000, 'rows=' + r.rows);
  check(name + ' HP 全部一致', r.hpBad === 0, r.hpBad + '/' + r.rows);
  check(name + ' 存活判定全部一致', r.deadBad === 0, r.deadBad + '/' + r.rows);
});

console.log('2. 已击毁单位不再画满血条');
(function () {
  const data = load('taiwan_campaign');
  const def = data.targets.find(function (t) { return t.max_blood === 4; });
  let tick = -1;
  for (let t = 0; t < data.frames.length; t++) {
    const row = data.frames[t].targets.find(function (r) { return r.id === def.id; });
    if (row && row.status === 'destroyed') { tick = t; break; }
  }
  check('找到一条击毁记录', tick >= 0);
  if (tick >= 0) {
    const at = TS.at(data, def.id, tick);
    check('击毁后 hp=0', at.hp === 0, 'hp=' + at.hp);
    check('击毁后档位=3', at.status === 3, 'status=' + at.status);
  }
})();

console.log('3. 纯移动案例仍走老启发式');
(function () {
  const data = load('taiwan_dense_transfer_v1');
  const id = data.targets[0].id;
  const at = TS.at(data, id, 100);
  check('无 HP 字段时不报 NaN', typeof at.hp === 'number' && isFinite(at.hp), 'hp=' + at.hp);
  check('档位在合法范围', at.status >= 0 && at.status <= 3, 'status=' + at.status);
})();

console.log('4. 三档阈值（THRESHOLDS.damaged=70、severe=35，语义是 hp <= 阈值）');
(function () {
  const cases = [[100, 0], [98, 0], [71, 0], [70, 1], [36, 1], [35, 2], [1, 2], [0, 3]];
  cases.forEach(function (pair) {
    check('hp ' + pair[0] + ' -> 档位 ' + pair[1],
          TS.statusOf(pair[0]) === pair[1], '得到 ' + TS.statusOf(pair[0]));
  });
})();

console.log('');
if (failures) {
  console.log('FAILED: ' + failures + ' 项');
  process.exit(1);
}
console.log('PASSED');
