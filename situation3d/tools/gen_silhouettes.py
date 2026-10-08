#!/usr/bin/env python3
"""目标军标剪影生成器。

为什么是生成器而不是 10 个手画文件：归一化规则（见 web/poc/DIMENSIONS.md）
只有在“**坐标由尺寸算出来**”时才是由构造保证的。手画时每一类都要记得去量，
而我已经在第一版驱逐舰上忘了一次（长高比 2.31 vs 目标 3.44）。

用法：python3 tools/gen_silhouettes.py [--out web/poc] [--check-only]
"""

import argparse
import io
import os
import re

# 视角约定与真实尺寸（米）。与 web/poc/DIMENSIONS.md 对应。
#   舰船 = 侧视（长, 桅顶高）；飞机/车辆 = 俯视（长, 翼展或宽）
DIM = {
    "destroyer":       ("ship",  155.0, 45.0),
    "frigate":         ("ship",  134.0, 35.0),
    "corvette":        ("ship",   89.0, 25.0),
    "replenishment":   ("ship",  178.0, 40.0),
    "fighter":         ("air",    15.0,  9.96),
    "aew":             ("air",    17.6, 24.6),
    "armor":           ("ground", 11.0,  3.4),
    "artillery":       ("ground", 11.6,  3.4),
    "coastal_missile": ("ground", 10.0,  3.0),
    "sam_battalion":   ("ground", 12.0,  3.0),
}

CANVAS = 188.0
NORM = 154.0    # 较大边归一到多少单位
MARGIN = 8.0


def norm(cls):
    """返回 (kind, 归一后的长, 归一后的高/宽, 缩放比)"""
    kind, L, H = DIM[cls]
    s = NORM / max(L, H)
    return kind, L * s, H * s, s


def _fmt(v):
    return ("%.2f" % v).rstrip("0").rstrip(".")


def _path(pts):
    d = "M" + " L".join("%s,%s" % (_fmt(x), _fmt(y)) for x, y in pts) + " Z"
    return d


def svg_ship(cls):
    """侧视船：舶在右。干舷 = 总高的 18%（近乎真实比例）。"""
    _, L, H, _ = norm(cls)
    x0 = (CANVAS - L) / 2.0
    water = (CANVAS + H) / 2.0
    deck = water - H * 0.18
    top = water - H
    def fx(t): return x0 + L * t
    def fy(t): return top + H * t          # t: 0=桅顶 1=水线
    parts = [
        # 船体：尾部到舶部，再沿水线回来
        _path([(fx(0.0), fy(0.18)), (fx(0.90), fy(0.18)), (fx(1.0), fy(0.10)),
               (fx(0.985), fy(0.46)), (fx(0.97), fy(1.0)), (fx(0.0), fy(1.0)),
               (fx(0.0), fy(0.52))]),
        # 主体上层建筑
        _path([(fx(0.38), fy(0.18)), (fx(0.38), fy(0.30)), (fx(0.60), fy(0.30)),
               (fx(0.64), fy(0.18))]),
        # 桅杆
        _path([(fx(0.465), fy(0.30)), (fx(0.465), fy(0.0)), (fx(0.487), fy(0.0)),
               (fx(0.487), fy(0.30))]),
        # 桅横桤
        _path([(fx(0.47), fy(0.19)), (fx(0.60), fy(0.19)), (fx(0.60), fy(0.23)),
               (fx(0.47), fy(0.23))]),
        # 烟囱（后倾）
        _path([(fx(0.66), fy(0.18)), (fx(0.70), fy(0.50)), (fx(0.775), fy(0.50)),
               (fx(0.745), fy(0.18))]),
        # 尾部甲板室
        _path([(fx(0.14), fy(0.18)), (fx(0.14), fy(0.62)), (fx(0.33), fy(0.62)),
               (fx(0.33), fy(0.18))]),
        # 前主炮
        _path([(fx(0.80), fy(0.18)), (fx(0.80), fy(0.72)), (fx(0.93), fy(0.72)),
               (fx(0.93), fy(0.18))]),
        # 直升机甲板标记
        _path([(fx(0.03), fy(0.20)), (fx(0.13), fy(0.20)), (fx(0.13), fy(0.28)),
               (fx(0.03), fy(0.28))]),
    ]
    return parts, (L, H)


def svg_air(cls):
    """俯视飞机：机头在右。长沿 x、翼展沿 y。"""
    _, L, W, _ = norm(cls)
    x0 = (CANVAS - L) / 2.0
    yc = CANVAS / 2.0
    def fx(t): return x0 + L * t
    def fy(t): return yc + W * t          # t: -0.5..0.5
    fus_w = W * 0.11                      # 机身宽
    parts = [
        # 机身（尖头）
        _path([(fx(1.0), fy(0.0)), (fx(0.55), fy(-fus_w / W / 2)),
               (fx(0.0), fy(-fus_w / W / 2)), (fx(0.0), fy(-fus_w / W / 3)),
               (fx(0.0), fy(fus_w / W / 3)), (fx(0.0), fy(fus_w / W / 2)),
               (fx(0.55), fy(fus_w / W / 2))]),
        # 主翼（后掠）
        _path([(fx(0.52), fy(-fus_w / W / 2)), (fx(0.30), fy(-0.50)),
               (fx(0.20), fy(-0.50)), (fx(0.38), fy(-fus_w / W / 2))]),
        _path([(fx(0.52), fy(fus_w / W / 2)), (fx(0.30), fy(0.50)),
               (fx(0.20), fy(0.50)), (fx(0.38), fy(fus_w / W / 2))]),
        # 尾翼
        _path([(fx(0.06), fy(-fus_w / W / 2)), (fx(0.0), fy(-0.22)),
               (fx(0.0), fy(-0.22)), (fx(0.0), fy(-fus_w / W / 2))]),
        _path([(fx(0.06), fy(fus_w / W / 2)), (fx(0.0), fy(0.22)),
               (fx(0.0), fy(0.22)), (fx(0.0), fy(fus_w / W / 2))]),
    ]
    # 预警机：直翼 + 圆盘雷达（这就是它与战斗机的区别）
    if cls == "aew":
        parts.append(_path([(fx(0.46), fy(-0.50)), (fx(0.30), fy(-0.50)),
                            (fx(0.26), fy(-fus_w / W / 2)), (fx(0.42), fy(-fus_w / W / 2))]))
        parts.append(_path([(fx(0.46), fy(0.50)), (fx(0.30), fy(0.50)),
                            (fx(0.26), fy(fus_w / W / 2)), (fx(0.42), fy(fus_w / W / 2))]))
        r = W * 0.19 / CANVAS * CANVAS
        parts.append("<circle cx=\"%s\" cy=\"%s\" r=\"%s\"/>" % (_fmt(fx(0.38)), _fmt(yc), _fmt(W * 0.19)))
    return parts, (L, W)


def svg_ground(cls):
    """俯视地面装备：车头在右。长沿 x、宽沿 y。"""
    _, L, W, _ = norm(cls)
    x0 = (CANVAS - L) / 2.0
    yc = CANVAS / 2.0
    def fx(t): return x0 + L * t
    def fy(t): return yc + W * t
    h = W / 2.0
    def band(a, b, w): return [(fx(a), yc - w), (fx(b), yc - w), (fx(b), yc + w), (fx(a), yc + w)]
    parts = []
    if cls == "armor":
        parts.append(_path([(fx(0.0), fy(-0.50)), (fx(0.90), fy(-0.50)), (fx(1.0), fy(-0.20)),
                            (fx(1.0), fy(0.20)), (fx(0.90), fy(0.50)), (fx(0.0), fy(0.50))]))
        parts.append(_path([(fx(0.30), fy(-0.26)), (fx(0.70), fy(-0.26)), (fx(0.76), fy(0.0)),
                            (fx(0.70), fy(0.26)), (fx(0.30), fy(0.26))]))
        parts.append(_path(band(0.70, 1.0, h * 0.055)))            # 炮管
    elif cls == "artillery":
        parts.append(_path([(fx(0.0), fy(-0.50)), (fx(0.86), fy(-0.50)), (fx(1.0), fy(-0.22)),
                            (fx(1.0), fy(0.22)), (fx(0.86), fy(0.50)), (fx(0.0), fy(0.50))]))
        parts.append(_path([(fx(0.34), fy(-0.28)), (fx(0.66), fy(-0.28)), (fx(0.72), fy(0.0)),
                            (fx(0.66), fy(0.28)), (fx(0.34), fy(0.28))]))
        parts.append(_path(band(0.66, 1.0, h * 0.05)))
    elif cls == "coastal_missile":
        parts.append(_path([(fx(0.0), fy(-0.50)), (fx(0.80), fy(-0.50)), (fx(1.0), fy(-0.24)),
                            (fx(1.0), fy(0.24)), (fx(0.80), fy(0.50)), (fx(0.0), fy(0.50))]))
        for k in (-0.20, 0.20):
            parts.append(_path(band(0.10, 0.72, h * 0.13)))
        parts.append(_path([(fx(0.78), fy(-0.30)), (fx(0.98), fy(-0.30)), (fx(0.98), fy(0.30)),
                            (fx(0.78), fy(0.30))]))
    else:  # sam_battalion
        parts.append(_path([(fx(0.0), fy(-0.50)), (fx(0.78), fy(-0.50)), (fx(1.0), fy(-0.22)),
                            (fx(1.0), fy(0.22)), (fx(0.78), fy(0.50)), (fx(0.0), fy(0.50))]))
        parts.append(_path(band(0.06, 0.66, h * 0.30)))
        parts.append(_path(band(0.30, 0.42, h * 0.42)))            # 发射箱分隔
        parts.append(_path([(fx(0.80), fy(-0.28)), (fx(0.98), fy(-0.28)), (fx(0.98), fy(0.28)),
                            (fx(0.80), fy(0.28))]))
    return parts, (L, W)


def svg_for(cls):
    kind = DIM[cls][0]
    fn = {"ship": svg_ship, "air": svg_air, "ground": svg_ground}[kind]
    parts, _ = fn(cls)
    body = "".join(parts)
    # ⚠ 形状函数**必须自己铺满**目标包围盒（即 x 跨满 L、y 跨满 H/翼展）。
    # 不能靠外层 `transform` 缩放：本文件的 `geom_bbox` 解析的是**路径原始坐标**，
    # **看不见 transform** —— 于是“量到的”和“画出来的”会不一致（我试过，失败）。
    # 让形状直接铺满，校验器才是诚实的。

    return ('<svg xmlns="http://www.w3.org/2000/svg" version="1.2" baseProfile="tiny" '
            'viewBox="0 0 188 188">\n  <g fill="#d0342c" stroke="#0f1720" '
            'stroke-width="2.2" stroke-linejoin="round">\n    ' + body + '\n  </g>\n</svg>\n')


def _geom_extents(text):
    xs, ys = [], []
    for m in re.finditer(r"[ML]\s*([-\d.]+)\s*,\s*([-\d.]+)", text):
        xs.append(float(m.group(1))); ys.append(float(m.group(2)))
    for m in re.finditer(r'circle cx="([-\d.]+)" cy="([-\d.]+)" r="([-\d.]+)"', text):
        cx2, cy2, r2 = float(m.group(1)), float(m.group(2)), float(m.group(3))
        xs += [cx2 - r2, cx2 + r2]; ys += [cy2 - r2, cy2 + r2]
    return (min(xs), min(ys), max(xs), max(ys)) if xs else (0, 0, 0, 0)


def geom_bbox(svg):
    xs, ys = [], []
    for m in re.finditer(r"[ML]\s*([-\d.]+)\s*,\s*([-\d.]+)", svg):
        xs.append(float(m.group(1))); ys.append(float(m.group(2)))
    for m in re.finditer(r'circle cx="([-\d.]+)" cy="([-\d.]+)" r="([-\d.]+)"', svg):
        cx, cy, r = float(m.group(1)), float(m.group(2)), float(m.group(3))
        xs += [cx - r, cx + r]; ys += [cy - r, cy + r]
    return (max(xs) - min(xs), max(ys) - min(ys)) if xs else (0, 0)

# 已发布符号的配色与规格（从 web/assets/symbols/*.svg 里取的，不是猜的）
AFFIL = {"red": "#d0342c", "blue": "#1f5fbf"}
STROKE = "#0f1720"
# 已发布符号用的是 viewBox "6 6 188 188"，而本生成器内部用 0 0 188 188。
# **不一致就不能 drop-in** —— 这是本轮才发现的。
VIEWBOX = "6 6 188 188"


def dropin_svg(cls, affil, selected):
    """方案①的 drop-in 符号：**不套帧**，整个 SVG 就是剪影。"""
    kind = DIM[cls][0]
    fn = {"ship": svg_ship, "air": svg_air, "ground": svg_ground}[kind]
    parts, _ = fn(cls)
    body = "".join(parts)
    # 把内部 0 0 188 188 的坐标平移到 6 6 188 188
    body = '<g transform="translate(6,6)">%s</g>' % body
    fill = AFFIL[affil]
    halo = ''
    if selected:
        # 选中变体：已发布符号用的是“外描边 stroke-width 16 + #ffd400”。
        # 剪影是多个形状组成的，没有单一轮廓，所以**把整组先用黄色粗描一遍再叠正形**，
        # 效果等价于光圈。
        halo = '<g transform="translate(6,6)" fill="none" stroke="#ffd400" stroke-width="14" stroke-linejoin="round">%s</g>' % body.replace('stroke="#0f1720"', 'stroke="#ffd400"')
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" version="1.2" baseProfile="tiny" '
           'viewBox="%s">%s<g transform="translate(6,6)" fill="%s" stroke="%s" '
           'stroke-width="2.2" stroke-linejoin="round">%s</g></svg>') % (
        VIEWBOX, halo, fill, STROKE, body)
    return svg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="web/poc")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--dropin", default=None,
                    help="额外产出可直接替换已发布符号的剪影版（方案①）到该目录")
    args = ap.parse_args()
    if not args.check_only and not os.path.isdir(args.out):
        os.makedirs(args.out)
    print("  %-17s %-6s %8s %8s %8s %8s   %s" % ("类别", "视角", "归一长", "归一高", "图实测", "目标", "状态"))
    bad = []
    for cls in DIM:
        kind, L, H, s = norm(cls)
        target = DIM[cls][1] / DIM[cls][2]
        svg = svg_for(cls)
        bw, bh = geom_bbox(svg)
        # 这里量的是**生成出来的图**，不是我算的数 ——
        # 两者不一致就说明形状函数没按尺寸画。
        ratio = bw / bh if bh else 0
        fits = bw <= CANVAS - 2 and bh <= CANVAS - 2
        if not fits or abs(ratio - target) > 0.06:
            bad.append(cls)
        if not args.check_only:
            with io.open(os.path.join(args.out, cls + "_silhouette.svg"), "w",
                         encoding="utf-8", newline="\n") as fh:
                fh.write(svg)
        print("  %-17s %-6s %8.1f %8.1f %8.2f %8.2f   %s" % (
            cls, kind, L, H, ratio, target, "OK" if (fits and abs(ratio - target) <= 0.06) else "!! 不合格"))
    print()
    print("  溢出或比例不对的:", bad if bad else "（无）")
    if args.dropin:
        if not os.path.isdir(args.dropin):
            os.makedirs(args.dropin)
        n = 0
        for cls in DIM:
            for affil in ("red", "blue"):
                for sel in (False, True):
                    name = "%s_%s%s.svg" % (cls, affil, "_sel" if sel else "")
                    with io.open(os.path.join(args.dropin, name), "w",
                                 encoding="utf-8", newline="") as fh:
                        fh.write(dropin_svg(cls, affil, sel))   # 已发布文件结尾无换行
                    n += 1
        print("  drop-in 符号已写出: %d 个 -> %s" % (n, args.dropin))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
