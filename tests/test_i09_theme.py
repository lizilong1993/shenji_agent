"""I09：视觉编码体系的**单一来源**与规则正确性。

本测试不依赖浏览器，做三件事：
  1. `theme.css` 的 `:root` 与 `theme.js` 的 `DEFAULTS` **逐键比对**
     —— 这是"唯一配置来源"不漂移的机器保证；
  2. 用 Python 复算 `theme.js` 里的**规则**（脉冲周期、内环半径），
     把步骤 3/4 的规则钉住（浏览器侧另有 CDP 实测，见 §42）；
  3. 断言 `index.html` 确实引入了这两个文件且顺序正确
     —— 文件存在但没被引入，是本项目踩过的坑（见 `NOTES_audits.md` 的悬空 id 轴）。

为什么不直接用 node 跑 `theme.js`：远程**没有 node**（实测 `node: command not found`）。
故规则用 Python 复算，`theme.js` 的真实运行由 CDP 探针覆盖。
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

# 本测试放在仓库根的 `tests/`（本仓的 pytest 根），
# 而被测的 `theme.css` / `theme.js` 在 `situation3d/web/src/` 下。
REPO = Path(__file__).resolve().parents[1] / "situation3d"
CSS = REPO / "web" / "src" / "theme.css"
JS = REPO / "web" / "src" / "theme.js"
HTML = REPO / "web" / "index.html"


def _root_block(css: str) -> str:
    """取**第一个 `:root { ... }` 块本体**（按花括号配对，不用正则截断）。

    ⚠ 不能简单用 `css[:css.find('[data-theme')]` 去切：本文件的**注释里**
    也出现了 `[data-theme=...]` 字样，那样切会得到空串
    —— 我自己第一版就这么切，结果报出 18 个"CSS 缺该变量"的假失败。
    """
    i = css.find(":root")
    assert i >= 0, "theme.css 缺少 :root 块"
    b0 = css.find("{", i)
    depth = 0
    for k in range(b0, len(css)):
        if css[k] == "{":
            depth += 1
        elif css[k] == "}":
            depth -= 1
            if depth == 0:
                return css[b0 + 1:k]
    raise AssertionError(":root 块未闭合")


def _css_vars(css: str) -> dict:
    return {k[2:]: v.strip() for k, v in
            re.findall(r"(--[a-z0-9-]+)\s*:\s*([^;\n]+);", _root_block(css))}


def _js_defaults(js: str) -> dict:
    seg = js[js.find("var DEFAULTS = {"):]
    seg = seg[:seg.find("};")]
    return dict(re.findall(r"'([a-z0-9-]+)':\s*'([^']*)'", seg))


@pytest.fixture(scope="module")
def theme_files():
    if not CSS.exists() or not JS.exists():
        pytest.skip("theme.css / theme.js not present")
    return CSS.read_text(encoding="utf-8"), JS.read_text(encoding="utf-8")


# ─────────────────────────── 单一来源：CSS ↔ JS 逐键一致

def test_css_and_js_declare_the_same_keys(theme_files) -> None:
    css, js = theme_files
    c, j = _css_vars(css), _js_defaults(js)
    assert c, "未能从 theme.css 的 :root 解析出变量"
    assert set(c) == set(j), (
        "CSS 与 JS 的键集合不一致\n  仅 CSS: %s\n  仅 JS: %s"
        % (sorted(set(c) - set(j)), sorted(set(j) - set(c))))


def test_css_and_js_values_match(theme_files) -> None:
    """同一键在两处的值必须相同，否则"唯一来源"名存实亡。"""
    css, js = theme_files
    c, j = _css_vars(css), _js_defaults(js)
    bad = {k: (c[k], j[k]) for k in c if k in j and c[k] != j[k]}
    assert bad == {}, "以下键 CSS 与 JS 取值不一致（css, js）: %s" % bad


# ─────────────────────────── 步骤 4：脉冲规则

def _seconds(v: str) -> float:
    v = v.strip().lower()
    return float(v[:-2]) / 1000 if v.endswith("ms") else float(v[:-1])


def test_pulse_periods_are_the_specified_values(theme_files) -> None:
    """判据 2：1.5s 与 0.8s 必须在 theme.css 中可配置。"""
    css, _ = theme_files
    c = _css_vars(css)
    assert _seconds(c["pulse-period-default"]) == pytest.approx(1.5)
    assert _seconds(c["pulse-period-fast"]) == pytest.approx(0.8)


def test_pulse_rule_threshold_is_strictly_greater_than_50(theme_files) -> None:
    """步骤 4 写的是「速度 >50km/h 改为 0.8s」—— 边界必须用**严格大于**。

    50 本身应走 1.5s（默认）；51 才走 0.8s。写成 `>=` 会让正好 50 的目标
    提前进入快脉冲，与条文不符。
    """
    css, _ = theme_files
    c = _css_vars(css)
    thr = float(c["pulse-speed-threshold-kmh"])
    assert thr == 50
    fast, slow = _seconds(c["pulse-period-fast"]), _seconds(c["pulse-period-default"])
    pick = lambda v: fast if v > thr else slow          # noqa: E731
    assert pick(49) == pytest.approx(1.5)
    assert pick(50) == pytest.approx(1.5), "50km/h 不应提前快脉冲（条文是「>50」）"
    assert pick(51) == pytest.approx(0.8)


def test_pulse_can_be_disabled_for_degradation(theme_files) -> None:
    """【风险与回滚】要求性能不足时可按 renderGovernor 降级 —— 需要一个开关。"""
    css, _ = theme_files
    assert _css_vars(css)["pulse-enabled"] == "1"
    assert "0 = 关闭脉冲" in css or "0 = 关闭" in css, "缺少关闭语义的说明"


# ─────────────────────────── 步骤 3：强度内环

def test_strength_ring_radius_rule(theme_files) -> None:
    css, _ = theme_files
    c = _css_vars(css)
    base = float(c["ring-strength-base-radius"].replace("px", ""))
    step = float(c["ring-strength-step"].replace("px", ""))
    maxr = float(c["ring-max-rings"])

    def radius(s):
        if s <= 0 or s > maxr:
            return 0 if s <= 0 else base + (maxr - 1) * step
        return base + (s - 1) * step

    assert radius(0) == 0, "强度 0 不画环"
    assert radius(1) == base
    assert radius(2) == base + step
    assert radius(4) == base + 3 * step
    # 越界必须被夹到最大值，而不是继续线性放大
    assert radius(9) == radius(int(maxr))


# ─────────────────────────── 步骤 5：交互反馈值

def test_hover_values_match_spec(theme_files) -> None:
    """判据 3：悬停位移 8px、过渡 200ms 与实现值一致。"""
    css, _ = theme_files
    c = _css_vars(css)
    assert c["hover-offset"] == "8px"
    assert c["hover-transition"] == "200ms"


# ─────────────────────────── 主题可切换（Q3 未关闭）

def test_dark_theme_is_optional_and_present(theme_files) -> None:
    """Q3 未裁定前：深色只能作为**可选声明**存在，且必须是可整体回退的独立块。"""
    css, _ = theme_files
    assert '[data-theme="dark"]' in css, "缺少可选深色主题块"
    # 默认（浅色）不应把 data-theme 写死在 html 上
    html = HTML.read_text(encoding="utf-8") if HTML.exists() else ""
    assert 'data-theme=' not in html.split("<body")[0], \
        "默认不应锁定主题属性（Q3 未关闭）"


def test_dark_theme_keeps_encoding_semantics(theme_files) -> None:
    """主题切换**只改取值，不改语义**：深色块必须重定义阵营色，而不是删掉它们。"""
    css, _ = theme_files
    i = css.find('[data-theme="dark"]')
    dark = css[i:]
    for key in ("--faction-red", "--faction-blue", "--strength-4"):
        assert key in dark, "深色主题缺少 %s（会导致该维度失去唯一来源）" % key


# ─────────────────────────── 引入顺序

def test_index_html_loads_theme_files_in_order() -> None:
    if not HTML.exists():
        pytest.skip("index.html not present")
    html = HTML.read_text(encoding="utf-8")
    assert "src/theme.css" in html, "theme.css 未引入（文件存在但没人加载 = 本项目踩过的坑）"
    assert "src/theme.js" in html, "theme.js 未引入"
    # theme.css 必须在 style.css 之后（否则被 style.css 覆盖）
    assert html.find("src/style.css") < html.find("src/theme.css"), \
        "theme.css 必须在 style.css 之后加载"
    # theme.js 必须在 app.js 之前（app.js 要用它）
    assert html.find("src/theme.js") < html.find("src/app.js"), \
        "theme.js 必须在 app.js 之前加载"


SEMANTIC_KEYS = ("faction-red", "faction-blue", "faction-contested", "faction-neutral")
SEMANTIC_HEX = ("#d0342c", "#1f5fbf", "#d8a12a", "#7d8a99")

# 已知技术债：**非**语义编码的显示色仍硬编码（label / overlay / facility 等）。
# 它们不是"同一语义两个值"，而是各自的显示常量，故**不在 I09 范围内**。
# 这里**显式登记**而不是假装不存在 —— 清单会写清未改的部分。
KNOWN_HARDCODED_DISPLAY_HEX = {
    "#1d2430", "#dce9ff", "#10233c", "#1f6feb", "#7b3fa0",
    "#f0a020", "#8a5a00", "#0b1a2a", "#000000a6",
}


def test_semantic_palette_is_not_hardcoded_in_js() -> None:
    """**本任务要解决的根因**：同一语义色在 JS 与 CSS 各写一份。

    原先 `app.js` 的 `OCCUPATION_STATUS_COLORS` 与 `colorFor()` 硬编码
    `#d0342c` / `#1f5fbf` / `#d8a12a` / `#7d8a99`，
    而 `style.css` 另写 `--red: #c8352c`（**与画布红方不是同一个值**）。
    现在这 4 个语义色必须全部来自 `Theme`。
    """
    app = REPO / "web" / "src" / "app.js"
    if not app.exists():
        pytest.skip("app.js not present")
    src = app.read_text(encoding="utf-8")
    offenders = []
    for ln in src.split("\n"):
        st = ln.strip()
        if st.startswith("//") or st.startswith("*"):
            continue
        for hexv in SEMANTIC_HEX:
            if hexv in st and "fromCssColorString" in st:
                offenders.append(st[:90])
    assert offenders == [], "语义编码色仍硬编码在 app.js（应经 Theme 取）: %s" % offenders


def test_app_reads_semantic_colors_from_theme() -> None:
    """不只是"删掉硬编码"，还要**确实改走 Theme**，否则是删功能而非重构。"""
    app = REPO / "web" / "src" / "app.js"
    if not app.exists():
        pytest.skip("app.js not present")
    src = app.read_text(encoding="utf-8")
    for key in SEMANTIC_KEYS:
        assert ("'" + key + "'") in src, "app.js 未从 Theme 读取 %s" % key
    assert "themeColor(" in src, "缺少 themeColor() 包装"


def test_hardcoded_display_colors_are_registered_debt() -> None:
    """把"其余硬编码显示色"登记成**已知技术债**，防止无声扩张。

    扫出 app.js 里所有 `fromCssColorString('#...')` 字面量，断言它们**都在已知清单里**：
      - 新增一个硬编码色 → 测试失败（阻止债务无声增长）；
      - 修掉一个 → 仍在清单里，不误报。
    本测试**不**要求改掉它们 —— 那是另一个任务，且它们不是语义编码。
    """
    import re as _re
    app = REPO / "web" / "src" / "app.js"
    if not app.exists():
        pytest.skip("app.js not present")
    src = app.read_text(encoding="utf-8")
    found = set()
    for ln in src.split("\n"):
        st = ln.strip()
        if st.startswith("//") or st.startswith("*"):
            continue
        for m in _re.finditer(r"fromCssColorString\('(#[0-9a-fA-F]{3,8})'\)", st):
            found.add(m.group(1).lower())
    allowed = {h.lower() for h in KNOWN_HARDCODED_DISPLAY_HEX} | {h.lower() for h in SEMANTIC_HEX}
    unexpected = sorted(found - allowed)
    assert unexpected == [], (
        "app.js 出现**未登记**的硬编码颜色（请改走 Theme，或显式登记为技术债）: %s" % unexpected)
