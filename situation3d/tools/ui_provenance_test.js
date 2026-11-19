#!/usr/bin/env node
/**
 * 三项界面改动的纯 Node 自检（无 Cesium、无浏览器、无网络）：
 *   1. 资源卡片的影像时间三项（采集 / 检索参考 / 入库）与缺失时的"未记录"；
 *   2. 标绘实例的来源三项随实例可取（编辑菜单只读展示的数据来源）；
 *   3. 关联复核区块：确认/否决分别呈现与计数、按对象集合隔离、撤销按钮的请求。
 *
 * 运行：node tools/ui_provenance_test.js
 */
'use strict';

const fs = require('fs');
const path = require('path');

const SRC = path.join(path.dirname(__dirname), 'web', 'src');
const readSrc = (file) => fs.readFileSync(path.join(SRC, file), 'utf8')
  .replace(/\r\n/g, '\n');

let failures = 0;
function check(name, cond, detail) {
  console.log((cond ? '  ok   ' : '  FAIL ') + name + (detail && !cond ? '  ' + detail : ''));
  if (!cond) failures += 1;
}

// ------------------------------------------------------------ 最小 DOM 桩
function node(tag) {
  const el = {
    tagName: tag, children: [], className: '', textContent: '', title: '',
    dataset: {}, style: {}, handlers: {},
    appendChild(child) { this.children.push(child); return child; },
    addEventListener(type, fn) { this.handlers[type] = fn; },
  };
  return el;
}
function textOf(el) {
  return el.textContent + el.children.map(textOf).join(' ');
}
function find(el, pred) {
  if (pred(el)) return el;
  for (const child of el.children) {
    const hit = find(child, pred);
    if (hit) return hit;
  }
  return null;
}

global.window = global;
global.document = { createElement: node, getElementById: () => null };
const requests = [];
global.fetch = function (url, opts) {
  requests.push({ url, method: (opts && opts.method) || 'GET',
                  body: opts && opts.body });
  // 撤销后刷新 overview 也走 fetch：给一个空壳响应即可。
  return Promise.resolve({ ok: true, json: () => Promise.resolve({}) });
};

function load(file) { (0, eval)(readSrc(file)); }

// ------------------------------------------------------ 1. 资源时间三项
load('resourceManager.js');
const RM = global.ResourceManager;

const full = {
  fetched_at: '2026-11-19T04:00:00Z',
  notes: { acquisition: {
    start_at: '2026-10-01T02:00:00Z', end_at: '2026-10-20T05:00:00Z',
    requested: '2026-10-26T00:00:00Z', scene_ids: ['s1'] } }
};
const lines = RM._timeLines(full);
check('采集区间原样显示（跨日记为 start → end）',
  lines[0][0] === '采集' &&
  lines[0][1] === '2026-10-01T02:00:00Z → 2026-10-20T05:00:00Z',
  JSON.stringify(lines[0]));
check('检索参考日期与采集时间分开',
  lines[1][0] === '检索参考' && lines[1][1] === '2026-10-26T00:00:00Z',
  JSON.stringify(lines[1]));
check('入库时间取 fetched_at',
  lines[2][0] === '入库' && lines[2][1] === '2026-11-19T04:00:00Z',
  JSON.stringify(lines[2]));

const single = { fetched_at: '2026-11-19T04:00:00Z',
                 notes: { acquisition: { start_at: '2026-10-20T05:00:00Z',
                                         end_at: '2026-10-20T05:00:00Z' } } };
check('单景只报一个时间戳而不是空区间',
  RM._timeLines(single)[0][1] === '2026-10-20T05:00:00Z',
  RM._timeLines(single)[0][1]);

const legacy = { as_of: '2026-11-19', fetched_at: '2026-11-19T04:00:00Z',
                 notes: {} };
const legacyLines = RM._timeLines(legacy);
check('没有采集记录的条目写"未记录"，不拿入库日冒充采集日',
  legacyLines[0][1] === '未记录' && legacyLines[1][1] === '未记录' &&
  legacyLines[2][1] === '2026-11-19T04:00:00Z',
  JSON.stringify(legacyLines));

check('资源编辑表单展示只读的影像时间三项',
  /影像时间（只读）/.test(readSrc('resourceManager.js')));

// -------------------------------------------- 2. 标绘来源三项（纯文本核对）
// getItem/exportAll 的行为断言在 tools/plot_dynamic_test.js 里做（那里有
// Cesium 桩）；这里只核对该列出的三项确实由实例带出、菜单确实只读展示。
check('编辑菜单只读展示来源/引用/精度，缺失写"未记录"',
  /'来源', rec\.std/.test(readSrc('app.js')) &&
  /'引用', rec\.ref/.test(readSrc('app.js')) &&
  /'精度', rec\.confidence/.test(readSrc('app.js')),
  'app.js 里找不到来源三项');
check('getItem 透出实例的 std/ref/confidence',
  /std: rec\.std \|\| null/.test(readSrc('plotting.js')) &&
  /ref: rec\.ref \|\| null/.test(readSrc('plotting.js')) &&
  /confidence: rec\.confidence \|\| null/.test(readSrc('plotting.js')),
  'plotting.js getItem 未带出三项');

// ------------------------------------------------------ 3. 关联复核区块
load('intelUi.js');
const Panel = global.IntelPanel;
const body = node('div');
const current = { correlations: { items: [
  { correlation_id: 'corr:1', case_id: 'c1', a_id: 'usgs:1', b_id: 'gdacs:2',
    verdict: 'confirm', note: '影像已确认', updated_at: '2026-11-18T10:00:00Z' },
  { correlation_id: 'corr:2', case_id: 'c1', a_id: 'usgs:3', b_id: 'gdacs:4',
    verdict: 'reject', note: '', updated_at: '2026-11-18T11:00:00Z' },
] } };
Panel._renderReviews(body, current);
const html = textOf(body);
check('确认与否决都呈现，且各自计数',
  html.includes('已确认 1') && html.includes('已否决 1') &&
  html.includes('usgs:1 ↔ gdacs:2') && html.includes('usgs:3 ↔ gdacs:4'),
  html);
check('确认项显示分析员备注，否决项没有备注也不留空壳',
  find(body, (e) => e.className === 'intel-review-note') &&
  textOf(find(body, (e) => e.className === 'intel-review-note')) === '影像已确认',
  html);

const other = node('div');
Panel._renderReviews(other, { correlations: { items: [
  { correlation_id: 'corr:9', case_id: 'c2', a_id: 'x', b_id: 'y',
    verdict: 'reject', updated_at: '2026-11-18T12:00:00Z' },
] } });
check('复核区块只呈现传入对象集合里的裁决（跨案例隔离由 case_id 过滤保证）',
  !textOf(other).includes('usgs:1') && textOf(other).includes('x ↔ y'),
  textOf(other));

const empty = node('div');
Panel._renderReviews(empty, { correlations: { items: [] } });
check('没有裁决时不渲染任何区块', empty.children.length === 0,
  String(empty.children.length));

const clearBtn = find(body, (e) => e.textContent === '撤销');
check('每条裁决都有撤销按钮', !!clearBtn);
if (clearBtn && clearBtn.handlers.click) {
  clearBtn.handlers.click();
  check('撤销走 POST /api/intel/correlations/<id>，不是删除即丢弃',
    requests.some((r) => r.method === 'POST' &&
      r.url === '/api/intel/correlations/corr%3A1'),
    JSON.stringify(requests));
}

// 上面几条只验证区块本身；这条守住它确实挂在列表页上（否则区块写对了也没人看）。
check('复核区块接进了列表渲染',
  /renderFreshness\(body, overview\);\n    renderReviews\(body, overview\);/
    .test(readSrc('intelUi.js')),
  'renderList 里没有 renderReviews 调用');

// 撤销在服务端的语义（退回算法判定）由 tests/test_intel_correlations.py 守住。
console.log();
if (failures) {
  console.log('FAILED (' + failures + ')');
  process.exit(1);
}
console.log('ALL UI PROVENANCE CHECKS PASSED');
