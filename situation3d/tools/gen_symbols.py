#!/usr/bin/env python3
"""Pre-generate the platform symbol SVGs with milsymbol.

Output (committed):
    web/assets/symbols/<icon>_<affiliation>.svg
    web/assets/symbols/<icon>_<affiliation>_sel.svg
    web/assets/symbols/manifest.json

Why pre-generate instead of shipping milsymbol
----------------------------------------------
* The bundle is 862 KB; the 40 SVG files total a few tens of KB.
* Rendering becomes deterministic and reviewable: generate once, look at the
  contact sheet, commit. A renderer change can never alter the deployed look.
* The browser does not need a JS symbol library at runtime.

Why the browser does the generating
-----------------------------------
There is **no node/npm in the deployment container** (python3.8, stdlib only, no
pip), and project code must not run on the developer machine. The browser is the
only JS engine available, so this tool drives the existing headless-Chrome
harness (`tools/inspect_remote.py`) against `web/tools/symbol-gen.html` and takes
the SVG strings back over the CDP eval channel.

Usage:
    python3 tools/fetch_milsymbol.py       # once: vendor the bundle
    python3 server/serve.py --port 8848 &  # the harness needs a served page
    python3 tools/gen_symbols.py
    python3 tools/gen_symbols.py --dry-run # generate and report, write nothing
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

WEB = os.path.join(ROOT, "web")
OUT_DIR = os.path.join(WEB, "assets", "symbols")
PAGE_URL = "http://127.0.0.1:8848/tools/symbol-gen.html"

# ---------------------------------------------------------------------------
# SIDC table -- MIL-STD-2525D, 20 digits.
# ---------------------------------------------------------------------------
# Layout (verified field by field; see the assertion in _sidc()):
#   1-2 version/coding scheme | 3 context | 4 affiliation | 5-6 symbol set
#   7 status | 8 HQ/TF | 9-10 echelon | 11-16 entity | 17-18 mod1 | 19-20 mod2
#
# Affiliation digit: 3 = Friend (blue force here), 6 = Hostile (red force).
#
# The entity codes were checked against the mil-std-2525 TSV tables
# (Sea surface / Land unit / Air) by the research pass.
#
# NOTE ON 岸舰导弹: MIL-STD-2525D has **no dedicated coastal anti-ship missile
# battery entity**. 130700 (Land Unit -> Fires -> Missile) is the closest unit
# symbol and is an explicit approximation, not a standard entity. Flagged here
# so it is never mistaken for one.
PLATFORMS = {
    "destroyer":       ("30", "120203"),   # Sea Surface / Destroyer
    "frigate":         ("30", "120204"),   # Sea Surface / Frigate
    "replenishment":   ("30", "130110"),   # Sea Surface / Oiler, Replenishment
    "corvette":        ("30", "120205"),   # Sea Surface / Corvette
    "coastal_missile": ("10", "130700"),   # Land Unit / Fires / Missile  (近似)
    "sam_battalion":   ("10", "130102"),   # Land Unit / Air Defense / Missile
    "armor":           ("10", "120500"),   # Land Unit / Armor
    "artillery":       ("10", "130300"),   # Land Unit / Field Artillery
    "fighter":         ("01", "110104"),   # Air / Fixed Wing / Fighter
    "aew":             ("01", "110116"),   # Air / Fixed Wing / AEW
    # 以下是 2026-10-21 补齐的兵种：原先直升机借用战斗机、登陆舰借用补给舰、
    # 无人机借用预警机、运输队/哨所借用装甲，图标与平台语义不符。
    "helicopter":      ("01", "110300"),   # Air / Rotary Wing
    "uav":             ("01", "110200"),   # Air / Unmanned
    "landing_ship":    ("30", "120900"),   # Sea Surface / Landing Ship
    "transport":       ("10", "120300"),   # Land Unit / Transport
    "outpost":         ("10", "130000"),   # Land Unit / Installation
}

# blue force = Friend, red force = Hostile. Colour comes from the SIDC itself
# (milsymbol's affiliation palette), so the two sides can never drift apart.
AFFILIATIONS = {"red": "6", "blue": "3"}

# Canvas: `size` is the icon-octagon width, `padding` the margin around it.
# Identical for every symbol so all files share one viewBox -- that is what lets
# the front end draw them all at one fixed pixel size with no per-icon maths.
SIZE = 40
PADDING = 12

# ---------------------------------------------------------------------------
# 观感：帧形用标准、配色用应用自己的、帧内用专业绘制的图形
# ---------------------------------------------------------------------------
# 为什么不再用 milsymbol 的默认外观（三个实测出来的原因）：
#   1. 默认是北约淡彩（友方 rgb(128,224,255)、敌方 rgb(255,128,128)），在卫星影像上
#      发白，而且与应用其它地方的 #1f5fbf / #d0342c 不一致；
#   2. 驱逐舰/战斗机/预警机在标准里是靠**文字**辨认的（生成物里是 Arial 的 "DD"、
#      "F"、"AEW"），不是图形；
#   3. 整体态势栏图标只有 16px，那行文字在 16px 下约 4px 高，完全是糊的。
#
# 所以：**帧形仍由 milsymbol 出**（它是标准权威，帧形已实测正确：敌方菱形／友方
# 方・圆・拱／空中敌方五角形），但 `icon:false` 不画帧内图形，改由下面这张表把
# game-icons（CC-BY-3.0）的专业绘制件注入帧内，配色换成应用自身的饱和红蓝。
FRAME_FILL = {"blue": "#1f5fbf", "red": "#d0342c"}
FRAME_EDGE = "#0f1720"        # 近黑描边：亮/暗影像上都能把符号从背景里分离出来
GLYPH_FILL = "#ffffff"        # 饱和帧上，白色图形的对比最强

# 平台 → game-icons 键。全部**实测存在**（game-icons 4137 个图标里查过）。
# 注：game-icons **没有** frigate / destroyer / patrol，所以舰船四型从
# battleship / iron-hulled-warship / interceptor-ship / cargo-ship 里各取其一；
# 补给舰用 cargo-ship 语义上反而更贴。
ICON_PARTS = {
    "destroyer": "battleship",
    "frigate": "iron-hulled-warship",
    "replenishment": "cargo-ship",
    "corvette": "interceptor-ship",
    "coastal_missile": "missile-launcher",
    "sam_battalion": "missile-pod",
    "armor": "tank",
    "artillery": "field-gun",
    "fighter": "jet-fighter",
    "helicopter": "helicopter",
    "uav": "delivery-drone",
    "transport": "truck",
    "outpost": "bunker",
    "aew": None,
    "landing_ship": None,
    # aew / landing_ship 用自绘图形（见 CUSTOM_PARTS）：game-icons 没有
    # "带雷达罩的预警机"，也没有登陆舰；原先 aew 用 "radar-dish" 画出来是一根
    # 雷达天线，被用户当成"雷达图标"（2026-10-21 实测反馈）。
}

# 自绘图形（512×512 坐标系，与 game-icons 一致）。用于 game-icons 无对应件的平台。
CUSTOM_PARTS = {
    # 预警机：俯视飞机轮廓 + 背部圆盘雷达罩（环形，避免与机身糊成一块）。
    "aew": (
        '<path fill-rule="evenodd" d="'
        'M256 40 C270 90 274 150 274 230 L274 390 C274 435 267 465 256 478 '
        'C245 465 238 435 238 390 L238 230 C238 150 242 90 256 40 Z '
        'M256 196 L468 296 L468 326 L278 288 L278 208 Z '
        'M256 196 L44 296 L44 326 L234 288 L234 208 Z '
        'M256 392 L356 446 L356 468 L268 442 Z '
        'M256 392 L156 446 L156 468 L244 442 Z"/>'
        '<path fill-rule="evenodd" d="'
        'M256 150 m-92 0 a92 92 0 1 0 184 0 a92 92 0 1 0 -184 0 '
        'M256 150 m-58 0 a58 58 0 1 0 116 0 a58 58 0 1 0 -116 0"/>'
    ),
    # 登陆舰：侧视舰体 + 舰艏跳板 + 上层建筑，一眼区别于补给舰。
    "landing_ship": (
        '<path d="M56 300 L452 300 L438 372 Q432 418 380 420 L120 420 '
        'Q76 418 64 372 Z"/>'
        '<path d="M452 300 L512 226 L518 250 L470 322 Z"/>'
        '<path d="M168 216 L286 216 L286 300 L168 300 Z"/>'
        '<path d="M226 168 L240 168 L240 216 L226 216 Z"/>'
        '<path d="M196 186 L214 186 L214 200 L196 200 Z"/>'
    ),
}

GAME_ICONS_TARBALL = "/tmp/iconprobe/gi.tgz"   # tools/fetch_icons.py 会下到这里

# 帧内图形的方框（milsymbol 的画布坐标）。取 74×74 居中于 (100,100)：
# 已按三种帧形验算：菱形在 y=63 处的半宽是 72-37=35 > 37? 临界，故取 74 略留余量；
# 圆的半径 60 > 37√2≈52 —— 圆帧会限制到 ~84，但菱形更紧，所以以菱形为准。
# 第一版取 56 偏保守，实测 20 个符号里 5 个白像素不足 16（火炮、岸舰导弹几乎看不见）。
GLYPH_BOX = 74.0
GLYPH_SRC = 512.0             # game-icons 的源坐标系
GLYPH_ORIGIN = 100.0 - GLYPH_BOX / 2.0


def load_icon_bodies():
    """从 game-icons 包里读出需要的 path 数据。只用标准库（容器无 npm）。"""
    import tarfile
    try:
        tar = tarfile.open(GAME_ICONS_TARBALL)
    except OSError:
        raise SystemExit(
            "[gen] 缺少图标包 %s\n"
            "      跑 tools/fetch_icons.py 下载（容器可直连 registry.npmjs.org），"
            "或手工把 @iconify-json/game-icons 的 tgz 放到该路径。" % GAME_ICONS_TARBALL)
    name = [n for n in tar.getnames() if n.endswith("icons.json")][0]
    data = json.loads(tar.extractfile(name).read())
    icons = data.get("icons", {})
    bodies, missing = {}, []
    for platform, key in ICON_PARTS.items():
        if key is None:
            continue
        node = icons.get(key)
        if not node:
            missing.append("%s(%s)" % (platform, key))
            continue
        bodies[platform] = node["body"]
    # 自绘图形优先（aew / landing_ship）
    for platform, body in CUSTOM_PARTS.items():
        bodies[platform] = body
    if missing:
        raise SystemExit("[gen] game-icons 里找不到：%s" % ", ".join(missing))
    return bodies


def inject_glyph(svg, body):
    """
    把图形塞进帧内。

    game-icons 的 path 是 0 0 512 512 坐标系；用 translate+scale 映射到 GLYPH_BOX
    的方框里。整体一个 `<g>` 设 fill，省得逐个 path 改属性。
    """
    s = GLYPH_BOX / GLYPH_SRC
    # **必须替换 body 里的 `fill="currentColor"`**：Iconify 的 body 自带这个属性，
    # 它优先级高于我加在 `<g>` 上的 fill，而没有 CSS `color` 时 currentColor 解析成
    # **黑色**——于是图形画成黑压在饱和红蓝帧上，几乎看不见。
    # 实测：20 个符号的白色像素数全是 0，黑色像素却有 330~510 个，就是这个原因。
    # 同时给 `<g>` 加 `color`，把残留的 currentColor 也兜住。
    body = body.replace('fill="currentColor"', 'fill="%s"' % GLYPH_FILL)
    g = ('<g transform="translate(%.3f,%.3f) scale(%.6f)" fill="%s" color="%s" '
         'stroke="none">%s</g>'
         % (GLYPH_ORIGIN, GLYPH_ORIGIN, s, GLYPH_FILL, GLYPH_FILL, body))
    return svg.replace("</svg>", g + "</svg>", 1)
SVG_TAG = re.compile(r"<svg\b[^>]*>", re.I)
VIEWBOX = re.compile(r'viewBox="([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)"', re.I)
WH = re.compile(r'\b(width|height)="([\d.]+)"', re.I)


def _sidc(affiliation_digit, symbol_set, entity):
    code = ("10" + "0" + affiliation_digit + symbol_set + "0" + "0"
            + "00" + entity + "00" + "00")
    # The first version of this table was 21 digits: one extra '0' shifted every
    # field after the affiliation, so friendly fighters came out as dashed
    # "unknown" circles and land units as air frames -- and milsymbol renders
    # over-long input **without complaint**, so the error was completely silent.
    # Only the rendered contact sheet caught it. Hence this hard assertion.
    if len(code) != 20:
        raise AssertionError("SIDC 必须是 20 位，得到 %d 位：%r" % (len(code), code))
    return code


def specs():
    """Build the generation spec list and the expected file names."""
    out = []
    for icon, (symbol_set, entity) in PLATFORMS.items():
        for aff, digit in AFFILIATIONS.items():
            code = _sidc(digit, symbol_set, entity)
            # 基础外观：只画帧（`icon:false`，帧内图形由 inject_glyph 注入）、
            # 不画文字放大器（`infoFields:false`，"DD"/"F" 那些字母就是丑的根源之一）、
            # 饱和填充 + 近黑描边。
            base = {
                "size": SIZE, "padding": PADDING,
                "icon": False, "infoFields": False,
                "fillColor": FRAME_FILL[aff],
                "strokeWidth": 4,
                "outlineColor": FRAME_EDGE, "outlineWidth": 2,
            }
            sel = dict(base)
            # 选中态：黄色高亮环（与应用其它地方的选中色一致）
            sel["outlineColor"] = "#ffd400"
            sel["outlineWidth"] = 6
            out.append({"name": "%s_%s" % (icon, aff), "icon": icon,
                        "sidc": code, "selected": False, "opts": base})
            out.append({"name": "%s_%s_sel" % (icon, aff), "icon": icon,
                        "sidc": code, "selected": True, "opts": sel})
    return out


def check_icons_match_engine():
    """
    The icon keys must be exactly the catalogue the engine emits.

    gen_symbols.py can silently fall out of step with engine/symbology.py (a new
    platform type added there, no symbol generated here) and the only symptom
    would be an empty icon in the panel. So compare the two sets and refuse.
    """
    try:
        import engine.symbology as symbology
    except Exception as exc:  # noqa: BLE001
        print("[gen] 无法导入 engine.symbology（%s），跳过一致性检查" % exc)
        return
    engine_keys = set(symbology.ICONS.keys())
    mine = set(PLATFORMS.keys())
    missing = sorted(engine_keys - mine)
    extra = sorted(mine - engine_keys)
    if missing:
        print("[gen] engine/symbology.py 里有而这里没有的平台：%s" % ", ".join(missing))
        print("      -> 请在 PLATFORMS 里补上 SIDC，否则这些目标的图标是空的")
    if extra:
        print("[gen] 这里有而 engine/symbology.py 没有的平台：%s" % ", ".join(extra))
    if missing or extra:
        raise SystemExit(2)
    print("[gen] 平台清单与 engine/symbology.py 一致（%d 个）" % len(engine_keys))


def normalize_pair(normal_svg, selected_svg):
    """
    Rewrite both SVGs to a common square viewBox so the two variants overlay
    exactly.

    Why this is needed: the selected variant carries an outline halo, so
    milsymbol gives it a **larger** canvas. Swapping the image on an existing
    billboard would then draw the selected symbol smaller (same 34 px box, wider
    viewBox) -- the icon would visibly shrink when you click it. Forcing one
    shared canvas removes that, and means the front end needs no per-variant
    size maths at all.
    """
    def parse(svg):
        tag = SVG_TAG.search(svg)
        if not tag:
            raise ValueError("找不到 <svg> 标签")
        vb = VIEWBOX.search(tag.group(0))
        if not vb:
            raise ValueError("找不到 viewBox")
        x, y, w, h = (float(v) for v in vb.groups())
        return x, y, w, h, tag

    nx, ny, nw, nh, ntag = parse(normal_svg)
    sx, sy, sw, sh, stag = parse(selected_svg)

    # Both are centred on the same anchor; take the larger canvas, centred on the
    # normal variant's centre (they agreed to well under a pixel in practice).
    cx, cy = nx + nw / 2.0, ny + nh / 2.0
    scx, scy = sx + sw / 2.0, sy + sh / 2.0
    if abs(cx - scx) > 1.0 or abs(cy - scy) > 1.0:
        print("[gen] 警告：两套变体的中心不一致（normal %.1f,%.1f / sel %.1f,%.1f），"
              "以 normal 为准" % (cx, cy, scx, scy))
    side = max(nw, nh, sw, sh)
    vx, vy = cx - side / 2.0, cy - side / 2.0

    def rewrite(svg, tag):
        new_tag = tag.group(0)
        new_tag = VIEWBOX.sub(
            'viewBox="%s %s %s %s"' % (_n(vx), _n(vy), _n(side), _n(side)),
            new_tag)
        # width/height: drop them and let the viewBox define the aspect. Leaving
        # absolute px would make the two files disagree again.
        new_tag = WH.sub("", new_tag)
        new_tag = re.sub(r"\s+>", ">", new_tag)
        return svg.replace(tag.group(0), new_tag, 1)

    return rewrite(normal_svg, ntag), rewrite(selected_svg, stag), side


def _n(v):
    s = ("%.3f" % v).rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"


def call_browser(spec_list):
    """Drive the headless browser and return the generated symbols."""
    js = ("(function(){var r=window.genSymbols(" + json.dumps(spec_list) + ");"
          "return r;})()")
    js_path = os.path.join(tempfile.gettempdir(), "_gensym.js")
    with open(js_path, "w", encoding="utf-8") as fh:
        fh.write(js)
    out_json = os.path.join(tempfile.gettempdir(), "_gensym.json")

    cmd = [sys.executable, os.path.join(HERE, "inspect_remote.py"),
           "--url", PAGE_URL, "--out", out_json,
           "--wait-ms", "2500", "--eval", js]
    print("[gen] 调用无头浏览器生成 %d 个符号…" % len(spec_list))
    proc = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, timeout=180)
    if proc.returncode != 0:
        sys.stdout.write(proc.stdout.decode("utf-8", "replace"))
        raise SystemExit("[gen] inspect_remote.py 退出码 %d" % proc.returncode)

    with open(out_json, "r", encoding="utf-8") as fh:
        report = json.load(fh)
    if not report.get("evalOk"):
        raise SystemExit("[gen] 页面内生成失败：%s" % report.get("evalError"))
    data = report.get("eval")
    if not isinstance(data, dict) or data.get("error"):
        raise SystemExit("[gen] 生成返回异常：%r" % (data,))
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="只生成并报告，不写文件")
    args = ap.parse_args()

    check_icons_match_engine()

    if not os.path.exists(os.path.join(WEB, "vendor", "milsymbol", "milsymbol.js")):
        raise SystemExit("[gen] 缺少 web/vendor/milsymbol/milsymbol.js，"
                         "先跑 python3 tools/fetch_milsymbol.py")

    bodies = load_icon_bodies()
    spec_list = specs()
    # 生成结果是 milsymbol 的对象（svg/anchor/size/valid），不回带我们传入的 sidc，
    # 所以这里自己留一份 name -> sidc 的映射写进 manifest。
    sidc_of = {s['name']: s['sidc'] for s in spec_list}
    results = call_browser(spec_list)

    written, manifest = 0, {"milsymbol": _vendored_version(), "size": SIZE,
                            "padding": PADDING, "symbols": {}}
    # Pair each normal variant with its selected twin.
    for icon in PLATFORMS:
        for aff in AFFILIATIONS:
            nkey, skey = "%s_%s" % (icon, aff), "%s_%s_sel" % (icon, aff)
            if nkey not in results or skey not in results:
                raise SystemExit("[gen] 缺少生成结果：%s" % nkey)
            n, s = results[nkey], results[skey]
            if not n.get("valid") or not s.get("valid"):
                raise SystemExit("[gen] SIDC 无效：%s" % nkey)
            # 先把专业图形注进帧内，再归一化两套变体的画布
            nsvg = inject_glyph(n["svg"], bodies[icon])
            ssvg = inject_glyph(s["svg"], bodies[icon])
            nsvg, ssvg, side = normalize_pair(nsvg, ssvg)
            for name, svg in ((nkey, nsvg), (skey, ssvg)):
                if not args.dry_run:
                    with open(os.path.join(OUT_DIR, name + ".svg"), "w",
                              encoding="utf-8") as fh:
                        fh.write(svg)
                written += 1
            manifest["symbols"][nkey] = {
                "icon": icon, "affiliation": aff, "sidc": sidc_of[nkey],
                "viewBoxSize": _n(side), "selectedVariant": skey + ".svg",
            }

    if not args.dry_run:
        if not os.path.isdir(OUT_DIR):
            os.makedirs(OUT_DIR)
        with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, ensure_ascii=False, indent=2, sort_keys=True)

    print("[gen] %s %d 个 SVG，共用画布 %s×%s"
          % ("试运行：" if args.dry_run else "写入", written,
             _n(side), _n(side)))
    print("[gen] 平台 %d 个 × 归属 %d 种 × 变体 2 套" % (len(PLATFORMS), len(AFFILIATIONS)))
    return 0


def _vendored_version():
    p = os.path.join(WEB, "vendor", "milsymbol", "VERSION")
    try:
        with open(p, "r", encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return "unknown"


if __name__ == "__main__":
    sys.exit(main())