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
    # 2026-10-22 补齐：engine/symbology.py 里已有、但生成器一直没有 SIDC 的平台。
    # 它们此前只能落到 JS 兜底生成器（形状与静态资产不一致），必须在这里补上。
    # 帧内图形按大类共用（见 GLYPH_OF），SIDC 只决定帧形，所以取同域/同族已知值即可。
    "truck":           ("10", "120300"),   # Land Unit / Transport
    "light_vehicle":   ("10", "120500"),   # Land Unit / Armor（同属地面机动）
    "engineer_vehicle": ("10", "120300"),  # Land Unit / Transport（保障分队）
    "transport_ship":  ("30", "120203"),   # Sea Surface（转运平台）
    # 大类兜底（engine/symbology.resolve_icon 会把未知 icon 落到这里）：
    # 帧形由 domain 决定，实体码复用同域已知值即可。
    "generic_sea":     ("30", "120203"),   # Sea Surface（通用）
    "generic_land":    ("10", "120500"),   # Land Unit（通用）
    "generic_air":     ("01", "110104"),   # Air（通用）
    "unknown":         ("10", "120500"),   # 未知：陆上帧 + 问号图形
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
# 观感：帧形用标准、配色用应用自己的、帧内用统一的大类剪影
# ---------------------------------------------------------------------------
# 为什么不用 milsymbol 的默认外观（三个实测出来的原因）：
#   1. 默认是北约淡彩（友方 rgb(128,224,255)、敌方 rgb(255,128,128)），在卫星影像上
#      发白，而且与应用其它地方的 #1f5fbf / #d0342c 不一致；
#   2. 驱逐舰/战斗机/预警机在标准里是靠**文字**辨认的（生成物里是 Arial 的 "DD"、
#      "F"、"AEW"），不是图形；
#   3. 整体态势栏图标只有 16px，那行文字在 16px 下约 4px 高，完全是糊的。
#
# 2026-10-22 重设计（feat/target-icon-clusters）：
#   * **帧形仍由 milsymbol 出**（标准权威；敌方菱形／友方方・圆・拱／空中敌方五角形）；
#   * 帧内不再用 game-icons —— 它是线性图标、线宽和装帧各不相同，34 px 下细节糊成
#     一片，且驱逐舰/护卫舰/巡逻舰三型几乎分不出。改为下面这套**自绘 512×512 实心
#     剪影**：同一大类共用同一个图形（DDG/FFG/PC 都是"舰艇"），类与类之间的轮廓
#     差异优先于型与型之间的细节差异；
#   * 全部白填充 + 少量镂空，镂空处透出帧色。红蓝底上对比都足够。
#
# 图标键 → 大类。键集合必须与 engine/symbology.py 一致（check_icons_match_engine）。
GLYPH_OF = {
    # 水面
    "destroyer": "sea_surface",
    "frigate": "sea_surface",
    "corvette": "sea_surface",
    "transport_ship": "sea_surface",
    "landing_ship": "amphibious",
    "replenishment": "auxiliary",
    # 地面
    "coastal_missile": "coastal_missile",
    "sam_battalion": "air_defense",
    "armor": "armor",
    "light_vehicle": "armor",
    "artillery": "artillery",
    "transport": "transport",
    "truck": "transport",
    "engineer_vehicle": "transport",
    "outpost": "outpost",
    # 空中
    "fighter": "fighter",
    "aew": "aew",
    "helicopter": "helicopter",
    "uav": "uav",
    # 兜底
    "generic_sea": "generic_sea",
    "generic_land": "generic_land",
    "generic_air": "generic_air",
    "unknown": "unknown",
}

FRAME_FILL = {"blue": "#1f5fbf", "red": "#d0342c"}
FRAME_EDGE = "#0f1720"        # 近黑描边：亮/暗影像上都能把符号从背景里分离出来
GLYPH_FILL = "#ffffff"        # 饱和帧上，白色图形的对比最强

# 大类剪影（512×512 源坐标系）。轮廓统一为"实心 + 少量镂空"，不要细线细节：
# 目标图标在屏幕上只有 34 px，帧内图形实际约 16×16 px，细线会消失。
GLYPH_BODIES = {
    # 水面舰艇（侧视，舰艏向右）：长舰体 + 舰桥 + 桅杆 + 烟囱 + 前主炮。
    "sea_surface": (
        '<path d="M40 330 L452 330 L500 356 L452 404 L78 404 Q48 404 40 378 Z"/>'
        '<path d="M150 250 L300 250 L330 330 L150 330 Z"/>'
        '<path d="M190 210 L280 210 L300 250 L170 250 Z"/>'
        '<path d="M248 160 L262 160 L262 250 L248 250 Z"/>'
        '<path d="M306 190 L342 190 L352 250 L298 250 Z"/>'
        '<path d="M356 296 L410 296 L418 316 L356 316 Z"/>'
        '<path d="M410 302 L470 302 L470 309 L410 309 Z"/>'
    ),
    # 两栖舰：平直甲板 + 右前方放下的艏跳板 + 后部高上层建筑。
    "amphibious": (
        '<path d="M60 320 L420 320 L470 372 L420 408 L100 408 Q70 400 60 370 Z"/>'
        '<path d="M470 372 L512 300 L520 320 L488 392 Z"/>'
        '<path d="M110 230 L250 230 L280 320 L110 320 Z"/>'
        '<path d="M120 190 L220 190 L240 230 L120 230 Z"/>'
        '<path d="M150 140 L162 140 L162 190 L150 190 Z"/>'
    ),
    # 辅助/补给舰：舰体 + 后部上层建筑 + 甲板上三段货舱 + 门吊。
    "auxiliary": (
        '<path d="M50 350 L450 350 L490 378 L440 410 L90 410 Q60 405 50 380 Z"/>'
        '<path d="M352 250 L430 250 L430 350 L340 350 Z"/>'
        '<path d="M88 300 L168 300 L168 344 L88 344 Z"/>'
        '<path d="M186 300 L266 300 L266 344 L186 344 Z"/>'
        '<path d="M284 300 L330 300 L330 344 L284 344 Z"/>'
        '<path d="M382 180 L392 180 L392 250 L382 250 Z"/>'
        '<path d="M338 195 L434 195 L434 205 L338 205 Z"/>'
    ),
    # 岸舰导弹：发射车 + 两枚斜向上的导弹弹体（与防空的竖直弹筒区分）。
    "coastal_missile": (
        '<path d="M200 330 L420 330 L444 384 L180 384 Z"/>'
        '<path d="M120 270 L210 270 L210 384 L120 384 Z"/>'
        '<path d="M270 90 L300 68 L436 286 L408 304 Z"/>'
        '<path d="M300 68 L322 44 L344 60 L330 84 Z"/>'
        '<path d="M318 70 L350 48 L484 272 L456 290 Z"/>'
        '<path d="M236 392 m-28 0 a28 28 0 1 0 56 0 a28 28 0 1 0 -56 0"/>'
        '<path d="M380 392 m-28 0 a28 28 0 1 0 56 0 a28 28 0 1 0 -56 0"/>'
    ),
    # 地面防空：顶置搜索雷达环 + 导弹发射箱（4 个镂空弹筒）+ 底盘。
    "air_defense": (
        '<path fill-rule="evenodd" d="'
        'M256 60 m-64 0 a64 64 0 1 0 128 0 a64 64 0 1 0 -128 0 '
        'M256 60 m-40 0 a40 40 0 1 0 80 0 a40 40 0 1 0 -80 0"/>'
        '<path d="M248 104 L264 104 L264 220 L248 220 Z"/>'
        '<path fill-rule="evenodd" d="'
        'M176 220 L336 220 L336 322 L176 322 Z '
        'M198 242 L218 242 L218 300 L198 300 Z '
        'M230 242 L250 242 L250 300 L230 300 Z '
        'M262 242 L282 242 L282 300 L262 300 Z '
        'M294 242 L314 242 L314 300 L294 300 Z"/>'
        '<path d="M156 330 L356 330 L380 384 L132 384 Z"/>'
        '<path d="M190 392 m-26 0 a26 26 0 1 0 52 0 a26 26 0 1 0 -52 0"/>'
        '<path d="M330 392 m-26 0 a26 26 0 1 0 52 0 a26 26 0 1 0 -52 0"/>'
    ),
    # 装甲分队（侧视，车头向右）：厚履带 + 车体 + 炮塔 + 主炮。
    "armor": (
        '<path fill-rule="evenodd" d="'
        'M90 336 L430 336 Q470 336 470 372 Q470 410 430 410 L90 410 '
        'Q50 410 50 372 Q50 336 90 336 Z '
        'M110 373 m-18 0 a18 18 0 1 0 36 0 a18 18 0 1 0 -36 0 '
        'M190 373 m-18 0 a18 18 0 1 0 36 0 a18 18 0 1 0 -36 0 '
        'M270 373 m-18 0 a18 18 0 1 0 36 0 a18 18 0 1 0 -36 0 '
        'M350 373 m-18 0 a18 18 0 1 0 36 0 a18 18 0 1 0 -36 0 '
        'M420 373 m-18 0 a18 18 0 1 0 36 0 a18 18 0 1 0 -36 0"/>'
        '<path d="M90 304 L450 304 L450 336 L90 336 Z"/>'
        '<path fill-rule="evenodd" d="'
        'M160 262 L330 262 L356 304 L140 304 Z '
        'M232 276 m-12 0 a12 12 0 1 0 24 0 a12 12 0 1 0 -24 0"/>'
        '<path d="M352 278 L500 278 L500 292 L352 292 Z"/>'
    ),
    # 炮兵分队：牵引炮（驻锄/大架 + 两个大车轮 + 上扬身管）。
    "artillery": (
        '<path d="M170 300 L380 300 L380 344 L130 344 Z"/>'
        '<path fill-rule="evenodd" d="'
        'M170 390 m-70 0 a70 70 0 1 0 140 0 a70 70 0 1 0 -140 0 '
        'M170 390 m-38 0 a38 38 0 1 0 76 0 a38 38 0 1 0 -76 0"/>'
        '<path fill-rule="evenodd" d="'
        'M330 390 m-50 0 a50 50 0 1 0 100 0 a50 50 0 1 0 -100 0 '
        'M330 390 m-24 0 a24 24 0 1 0 48 0 a24 24 0 1 0 -48 0"/>'
        '<path d="M340 282 L512 160 L520 180 L352 302 Z"/>'
        '<path d="M180 350 L50 470 L68 488 L200 372 Z"/>'
    ),
    # 运输/保障：厢式卡车（车头向右）+ 两轴车轮。
    "transport": (
        '<path d="M90 228 L340 228 L340 342 L90 342 Z"/>'
        '<path d="M340 258 L432 258 L454 342 L340 342 Z"/>'
        '<path fill-rule="evenodd" d="'
        'M80 342 L454 342 L454 372 L80 372 Z"/>'
        '<path fill-rule="evenodd" d="'
        'M160 400 m-40 0 a40 40 0 1 0 80 0 a40 40 0 1 0 -80 0 '
        'M160 400 m-18 0 a18 18 0 1 0 36 0 a18 18 0 1 0 -36 0"/>'
        '<path fill-rule="evenodd" d="'
        'M360 400 m-40 0 a40 40 0 1 0 80 0 a40 40 0 1 0 -80 0 '
        'M360 400 m-18 0 a18 18 0 1 0 36 0 a18 18 0 1 0 -36 0"/>'
    ),
    # 哨所/阵地：半圆掩体 + 观察缝（镂空）+ 顶部天线。
    "outpost": (
        '<path fill-rule="evenodd" d="'
        'M86 372 A170 170 0 0 1 426 372 Z '
        'M176 332 L336 332 L336 358 L176 358 Z"/>'
        '<path d="M70 372 L442 372 L442 416 L70 416 Z"/>'
        '<path d="M248 120 L264 120 L264 232 L248 232 Z"/>'
        '<path d="M200 196 L312 196 L312 232 L200 232 Z"/>'
    ),
    # 战斗机（俯视）：后掠翼 + 平尾，机头向上。
    "fighter": (
        '<path d="M256 24 L280 170 L470 300 L470 330 L282 262 L282 420 '
        'L330 470 L330 494 L256 466 L182 494 L182 470 L230 420 L230 262 '
        'L42 330 L42 300 L232 170 Z"/>'
    ),
    # 预警机：俯视飞机 + 背部圆盘雷达罩（环），与战斗机的轮廓差异在"盘"。
    "aew": (
        '<path d="'
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
    # 直升机（侧视，机头向右）：机身 + 尾梁 + 旋翼 + 起落橇。
    "helicopter": (
        '<path fill-rule="evenodd" d="'
        'M140 300 Q140 240 210 240 L330 240 Q396 240 396 306 Q396 366 330 366 '
        'L210 366 Q140 366 140 300 Z '
        'M188 268 L282 268 L282 300 L188 300 Z"/>'
        '<path d="M330 300 L480 268 L512 268 L512 286 L360 330 Z"/>'
        '<path d="M250 240 L266 240 L266 196 L250 196 Z"/>'
        '<path d="M110 186 L430 186 L430 200 L110 200 Z"/>'
        '<path d="M170 392 L370 392 L370 404 L170 404 Z"/>'
        '<path d="M208 366 L218 366 L218 392 L208 392 Z"/>'
        '<path d="M320 366 L330 366 L330 392 L320 392 Z"/>'
    ),
    # 无人机（俯视）：细长机身 + 平直长翼 + V 尾 + 螺旋桨环。
    "uav": (
        '<path d="'
        'M256 60 C270 80 276 120 276 170 L276 460 C276 486 268 500 256 508 '
        'C244 500 236 486 236 460 L236 170 C236 120 242 80 256 60 Z '
        'M256 210 L40 300 L40 320 L244 274 L268 274 L472 320 L472 300 Z '
        'M256 446 L170 500 L170 520 L250 476 Z '
        'M256 446 L342 500 L342 520 L262 476 Z"/>'
        '<path fill-rule="evenodd" d="'
        'M256 40 m-34 0 a34 34 0 1 0 68 0 a34 34 0 1 0 -68 0 '
        'M256 40 m-14 0 a14 14 0 1 0 28 0 a14 14 0 1 0 -28 0"/>'
    ),
    # 通用水面：简化舰体 + 单桅 + 烟囱。
    "generic_sea": (
        '<path d="M60 340 L460 340 L500 366 L456 406 L100 406 Q70 402 60 380 Z"/>'
        '<path d="M230 280 L320 280 L340 340 L210 340 Z"/>'
        '<path d="M244 220 L258 220 L258 280 L244 280 Z"/>'
        '<path d="M300 250 L330 250 L336 280 L294 280 Z"/>'
    ),
    # 通用陆上：交叉双剑。
    "generic_land": (
        '<path d="M96 96 L136 56 L416 416 L376 456 Z"/>'
        '<path d="M416 96 L456 136 L136 456 L96 416 Z"/>'
        '<path d="M96 376 L136 416 L176 376 L136 336 Z"/>'
        '<path d="M376 376 L416 416 L456 376 L416 336 Z"/>'
    ),
    # 通用空中：平直翼飞机（俯视）。
    "generic_air": (
        '<path d="'
        'M256 50 L276 210 L480 210 L480 240 L276 270 L276 430 L330 470 '
        'L330 494 L256 470 L182 494 L182 470 L230 430 L230 270 L32 240 '
        'L32 210 L236 210 Z"/>'
    ),
    # 未知：粗体问号。
    "unknown": (
        '<path d="'
        'M256 90 C180 90 140 140 140 200 L196 200 C196 172 218 146 256 146 '
        'C292 146 312 166 312 196 C312 226 292 240 256 262 C216 288 206 312 '
        '206 360 L206 392 L266 392 L266 366 C266 330 278 314 312 292 '
        'C352 266 372 236 372 194 C372 126 320 90 256 90 Z '
        'M206 420 L266 420 L266 480 L206 480 Z"/>'
    ),
}

# 帧内图形的方框（milsymbol 的画布坐标）。取 92×92 居中于 (100,100)：
# 帧体约 144 宽；92 让舰体/车体占到帧宽 ~55%，34 px 图标下图形约 16 px（此前 74
# 的版本只有 ~13 px，舰船类看上去就是一个小方块）。竖向不留满：敌方水上/陆上
# 帧是菱形，92 方框的四个角会越出菱形，但实际图形（舰船、车辆）都是宽扁轮廓，
# 纵向只占 25 px 左右，不会碰到边。
GLYPH_BOX = 92.0
GLYPH_SRC = 512.0             # GLYPH_BODIES 的源坐标系
GLYPH_ORIGIN = 100.0 - GLYPH_BOX / 2.0


def load_icon_bodies():
    """Return icon key -> glyph body for the built-in silhouette set.

    2026-10-22 起不再读 game-icons 包：所有帧内图形都是上面 GLYPH_BODIES 里的
    自绘件，生成器因此不再依赖 /tmp 下的 npm tarball。
    """
    missing = sorted(set(PLATFORMS) - set(GLYPH_OF))
    if missing:
        raise SystemExit("[gen] GLYPH_OF 缺少平台：%s" % ", ".join(missing))
    return {icon: GLYPH_BODIES[cat] for icon, cat in GLYPH_OF.items()}


def inject_glyph(svg, body):
    """
    把图形塞进帧内。

    GLYPH_BODIES 的 path 用 0 0 512 512 坐标系；用 translate+scale 映射到 GLYPH_BOX
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