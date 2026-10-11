#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""I08/I09 **运行时**测试的 JS 执行引擎封装。

## 做什么

用 `quickjs` **真正加载并执行** `theme.js` / `eventFeed.js`，
并给全局 `getComputedStyle` 造一个可控 shim，从而**驱动真实的 CSS 读取路径**。

## 引擎从哪来

`scripts/setup_js_engine.sh` 把它装到仓库内 `.tools/jsengine/`（**已 gitignore**，13 MB）。
本模块**自动把该目录加入 `sys.path`**，故测试无需额外设置 `PYTHONPATH`。

**缺引擎时**：`quickjs` 导入失败 → 调用方 `pytest.importorskip` → **跳过并说明**，
**不是静默通过**。

## ⚠ 能力边界（必须写明）

shim **不是浏览器**：它只实现被真实代码用到的
`getComputedStyle(...).getPropertyValue(var)` 一个接口。
故它证明的是「**JS 的读取与计算逻辑正确**」，
**不是**「浏览器里的渲染/布局正确」——
判据里"界面上逐项可复现""截图评审"仍**必须浏览器**。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "situation3d" / "web" / "src"
_ENGINE_DIR = REPO / ".tools" / "jsengine"
if _ENGINE_DIR.is_dir() and str(_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(_ENGINE_DIR))

# ── shim：只实现被真实代码用到的接口，不假装是浏览器
SHIM = r"""
var __cssVars = {};
var __warned = [];
var __attributes = {};
function getComputedStyle(el) {
  return { getPropertyValue: function (name) { return __cssVars[name] || ''; } };
}
var document = {
  documentElement: {
    style: { getPropertyValue: function (n) { return __cssVars[n] || ''; } },
    setAttribute: function (k, v) { __attributes[k] = v; },
    getAttribute: function (k) { return __attributes[k] || null; },
  },
  querySelector: function () { return null; },
  querySelectorAll: function () { return []; },
  addEventListener: function () {},
  createElement: function () {
    return { style: {}, setAttribute: function () {}, appendChild: function () {} };
  },
  body: { appendChild: function () {}, classList: { add: function () {}, remove: function () {} } },
};
var window = this;
window.document = document;
window.getComputedStyle = getComputedStyle;
window.console = { warn: function (m) { __warned.push(String(m)); },
                   log: function () {}, error: function () {} };
window.addEventListener = function () {};
window.matchMedia = function () {
  return { matches: false, addEventListener: function () {} };
};
window.localStorage = { getItem: function () { return null; }, setItem: function () {} };
window.__setVar = function (n, v) { __cssVars[n] = v; };
window.__warned = __warned;
"""


def engine_available() -> bool:
    """引擎是否可用（供测试决定 skip 还是 fail）。"""
    try:
        import quickjs  # noqa: F401, PLC0415
        return True
    except ImportError:
        return False


class JsHarness:
    """加载真实 JS 文件并暴露可调用的接口。"""

    def __init__(self, files=("theme.js", "eventFeed.js")):
        import quickjs  # noqa: PLC0415

        self._ctx = quickjs.Context()
        self._ctx.eval(SHIM)
        self.loaded = []
        for name in files:
            p = SRC / name
            if not p.is_file():
                continue
            self._ctx.eval(p.read_text(encoding="utf-8"))
            self.loaded.append(name)

    # ── 底层
    def eval(self, expr):
        return self._ctx.eval(expr)

    def set_var(self, name, value):
        """设置 CSS 变量（等价于改 `theme.css` 后 `getComputedStyle` 返回新值）。"""
        self._ctx.eval("window.__setVar(%s, %s);"
                       % (json.dumps(name), json.dumps(value)))

    def load_css_vars(self, css_path: Path) -> int:
        """把 `theme.css` 的 `:root` 变量灌进 shim —— 等价于"浏览器加载了该 CSS"。"""
        css_path = Path(css_path)
        if not css_path.is_file():
            return 0
        text = css_path.read_text(encoding="utf-8")
        m = re.search(r":root\s*\{(.*?)\n\}", text, re.S)
        if not m:
            return 0
        n = 0
        for k, v in re.findall(r"(--[a-z0-9-]+)\s*:\s*([^;]+);", m.group(1)):
            self.set_var(k, v.strip())
            n += 1
        return n

    # ── Theme 接口
    def _args(self, args):
        return ",".join(json.dumps(a) if isinstance(a, str) else repr(a) for a in args)

    def theme(self, fn, *args):
        return self.eval("window.Theme.%s(%s)" % (fn, self._args(args)))

    def theme_json(self, fn, *args):
        return json.loads(self.eval("JSON.stringify(window.Theme.%s(%s))"
                                    % (fn, self._args(args))))

    # ── EventFeed 接口
    def feed(self, *statements):
        for s in statements:
            self.eval(s)

    def feed_json(self, expr):
        return json.loads(self.eval("JSON.stringify(%s)" % expr))

    @property
    def warned(self):
        return json.loads(self.eval("JSON.stringify(window.__warned)"))
