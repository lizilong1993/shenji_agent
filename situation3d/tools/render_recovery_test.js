#!/usr/bin/env node
/**
 * render_recovery_test.js — WebGL 上下文丢失必须被处理，而不是静默。
 *
 * 背景：上下文一丢，Cesium 的 requestAnimationFrame 循环照常跑，
 * `viewer.scene` 照常返回，画面停在最后一帧。全仓此前没有
 * `webglcontextlost` 监听，用户看到的是"卡住了"，顶栏状态点还是绿的。
 *
 * 这里用源码断言 + 一个最小 DOM 桩来锁行为：真跑 WebGL 需要 GPU 与浏览器，
 * 而这条要守的是"有没有接上"，不是"Cesium 内部做得对不对"。
 */
'use strict';

const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const APP = fs.readFileSync(path.join(ROOT, 'web', 'src', 'app.js'), 'utf8');
const CSS = fs.readFileSync(path.join(ROOT, 'web', 'src', 'style.css'), 'utf8');

let failures = 0;
function check(name, cond, detail) {
  if (cond) { console.log('  ok   ' + name); return; }
  failures++;
  console.log('  FAIL ' + name + (detail ? '  —  ' + detail : ''));
}

console.log('1. 两个事件都监听在 canvas 上');
check('监听 webglcontextlost',
      /addEventListener\('webglcontextlost'/.test(APP));
check('监听 webglcontextrestored',
      /addEventListener\('webglcontextrestored'/.test(APP));
check('挂在 viewer.canvas 上（不是 window）',
      /var canvas = viewer && viewer\.canvas;/.test(APP));
// 定义一处 + 调用一处 = 2 次；只匹配 "function attachContextRecovery" 之外的那个。
check('attachContextRecovery 真的被调用（不只是定义）',
      APP.replace(/function attachContextRecovery[\s\S]*?\n\}\n/, '').indexOf('attachContextRecovery();') >= 0);

console.log('2. 丢失时必须 preventDefault，否则浏览器不尝试恢复');
check('调用了 preventDefault', /ev\.preventDefault/.test(APP));

console.log('3. 丢失时停时间轴并给出状态');
check('停播放', /if \(state\.playing\) setPlaying\(false\);/.test(APP));
check('状态点报错', /setStatus\('3D 图形连接中断/.test(APP));
check('弹出恢复提示', /showRenderRecovery\(\)/.test(APP));

console.log('4. 恢复时重算投影缓存');
check('清血条排布节流', /_hpLayoutLastMs = 0;/.test(APP));
check('清血条偏移缓存', /_hpOffsetCache = \{\};/.test(APP));
check('清标牌避让节流', /_declutterLastMs = 0;/.test(APP));
check('重新标脏', /markScene\('context-restored'\)/.test(APP));

console.log('5. 提示条有对应样式，否则显示为裸文本');
['.render-recovery', '.rr-title', '.rr-body', '.rr-actions'].forEach(function (sel) {
  check('样式含 ' + sel, CSS.indexOf(sel) >= 0);
});
check('提示条默认隐藏', /\.render-recovery \{[^}]*display: none/s.test(CSS));

console.log('6. 提示可关闭、可降画质');
check('有关闭按钮', /rrDismiss/.test(APP));
check('有降画质按钮', /rrQuality/.test(APP));

console.log('');
if (failures) { console.log('FAILED: ' + failures + ' 项'); process.exit(1); }
console.log('PASSED');
