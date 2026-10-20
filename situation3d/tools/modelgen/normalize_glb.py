# -*- coding: utf-8 -*-
"""
Sketchfab 资产归一化：统一朝向 +Y、按真实尺寸缩放、减面、贴图压缩、导出 GLB。

用法：
  blender -b --python normalize_glb.py -- IN.glb OUT.glb --len 134 --axis Y \
      --tris 45000 --anchor water [--rot 180] [--maxtex 2048]

- --len    真实全长/机长（米）；按旋转后的 Y 向包围盒缩放到该值
- --axis   导入时船艏/机头所在的水平轴：X/Y/-X/-Y（默认 Y）
- --rot    额外汇率（度），绕 Z 旋转后再检查
- --anchor water|ground|center  原点 z：0（水线）/ -minZ / 中心
"""
import bpy
import math
import os
import sys
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
src, dst = argv[0], argv[1]
opts = {}
i = 2
while i < len(argv):
    k = argv[i].lstrip("-")
    v = argv[i + 1]
    opts[k] = v
    i += 2

REAL_LEN = float(opts.get("len", "0"))
AXIS = opts.get("axis", "Y")
EXTRA_ROT = float(opts.get("rot", "0"))
TARGET_TRIS = int(opts.get("tris", "45000"))
ANCHOR = opts.get("anchor", "water")
MAXTEX = int(opts.get("maxtex", "2048"))
DUMP_AXIS = opts.get("dump", "")


def bbox(objs):
    mn = Vector((1e9,) * 3)
    mx = Vector((-1e9,) * 3)
    for o in objs:
        for c in o.bound_box:
            w = o.matrix_world @ Vector(c)
            mn = Vector((min(mn[i], w[i]) for i in range(3)))
            mx = Vector((max(mx[i], w[i]) for i in range(3)))
    return mn, mx


bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
if not objs:
    raise SystemExit("no meshes")

# 可选：裁掉扫描件自带的底座/展台（按包围盒高度比例从最低处切）。
if "cutz" in opts:
    import bmesh
    frac = float(opts["cutz"])
    mn, mx = bbox(objs)
    zcut = mn.z + (mx.z - mn.z) * frac
    for o in objs:
        bm = bmesh.new()
        bm.from_mesh(o.data)
        bm.faces.ensure_lookup_table()
        mw = o.matrix_world
        doomed = []
        for f in bm.faces:
            c = f.calc_center_median()
            if (mw @ c).z < zcut:
                doomed.append(f)
        bmesh.ops.delete(bm, geom=doomed, context="FACES")
        bm.to_mesh(o.data)
        bm.free()
        o.data.update()
    print("CUTZ", round(frac, 3), flush=True)

# 1) 朝向：把 AXIS 旋到 +Y
axis_angle = {"Y": 0.0, "X": 90.0, "-X": -90.0, "-Y": 180.0}.get(AXIS)
if axis_angle is None:
    raise SystemExit("bad axis " + AXIS)

# 可选：按主轴自动纠偏（车辆/舰船顶层俯视被任意角度摆放时用）。
# fit 模式下 AXIS 只用于确定“哪一端是艏/头”（flip），主轴本身由 PCA 决定。
if opts.get("fit") == "yaw":
    import numpy as np
    pts = []
    total_v = sum(len(o.data.vertices) for o in objs)
    step = max(1, total_v // 20000)
    for o in objs:
        mw = o.matrix_world
        vs = o.data.vertices
        for i in range(0, len(vs), step):
            w = mw @ vs[i].co
            pts.append((w.x, w.y))
    a = np.array(pts, dtype=np.float64)
    a -= a.mean(axis=0)
    cov = np.cov(a.T)
    evals, evecs = np.linalg.eigh(cov)
    v = evecs[:, int(np.argmax(evals))]
    ang = math.degrees(math.atan2(v[1], v[0]))  # 主轴相对 +X 的角度
    axis_angle = 90.0 - ang
    print("FITYAW", round(axis_angle, 2), flush=True)

if opts.get("flip") == "1":
    axis_angle += 180.0

from mathutils import Matrix
R = Matrix.Rotation(math.radians(axis_angle + EXTRA_ROT), 4, "Z")
for o in objs:
    o.matrix_world = R @ o.matrix_world

# 自动翻头：艏/机头通常比艉/机尾窄（船艉有宽封板，机尾有平尾）。
# 车辆则靠炮管端（端部横截面很窄）判断。fit 模式下默认开启。
autof = opts.get("autoflip", "1" if opts.get("fit") == "yaw" else "0")
if autof == "1":
    import numpy as np
    pts = []
    total_v = sum(len(o.data.vertices) for o in objs)
    step = max(1, total_v // 30000)
    for o in objs:
        mw = o.matrix_world
        vs = o.data.vertices
        for i in range(0, len(vs), step):
            w = mw @ vs[i].co
            pts.append((w.x, w.y, w.z))
    a = np.array(pts)
    y0, y1 = a[:, 1].min(), a[:, 1].max()
    span = y1 - y0
    band = max(span * 0.06, 1e-6)
    lo = a[a[:, 1] <= y0 + band]
    hi = a[a[:, 1] >= y1 - band]
    wlo = lo[:, 0].max() - lo[:, 0].min() if len(lo) else 0
    whi = hi[:, 0].max() - hi[:, 0].min() if len(hi) else 0
    if whi > wlo * 1.15:
        R180 = Matrix.Rotation(math.pi, 4, "Z")
        for o in objs:
            o.matrix_world = R180 @ o.matrix_world
        print("AUTOFLIP 180 (wlo=%.2f whi=%.2f)" % (wlo, whi), flush=True)
    else:
        print("AUTOFLIP 0 (wlo=%.2f whi=%.2f)" % (wlo, whi), flush=True)

# 2) 缩放到真实长度（按旋转后 Y 向包围盒）
mn, mx = bbox(objs)
cur_len = mx.y - mn.y
scale = REAL_LEN / cur_len if REAL_LEN > 0 else 1.0
S = Matrix.Scale(scale, 4)
for o in objs:
    o.matrix_world = S @ o.matrix_world

# 3) 原点
mn, mx = bbox(objs)
if ANCHOR == "ground":
    dz = mn.z
elif ANCHOR == "center":
    dz = (mn.z + mx.z) / 2.0
else:
    dz = 0.0
dx = (mn.x + mx.x) / 2.0
for o in objs:
    m = o.matrix_world.copy()
    m.translation.x -= dx
    m.translation.z -= dz
    o.matrix_world = m

# 4) 减面
total = 0
for o in objs:
    o.data.calc_loop_triangles()
    total += len(o.data.loop_triangles)
if TARGET_TRIS > 0 and total > TARGET_TRIS:
    ratio = TARGET_TRIS / float(total)
    for o in objs:
        o.data.calc_loop_triangles()
        n = len(o.data.loop_triangles)
        if n < 200:
            continue
        md = o.modifiers.new("dec", "DECIMATE")
        md.ratio = max(0.02, min(1.0, ratio))
        bpy.context.view_layer.objects.active = o
        try:
            bpy.ops.object.modifier_apply(modifier=md.name)
        except Exception:
            o.modifiers.remove(md)

# 5) 贴图压缩
for im in bpy.data.images:
    w, h = im.size
    if max(w, h) > MAXTEX:
        f = MAXTEX / float(max(w, h))
        im.scale(max(1, int(w * f)), max(1, int(h * f)))
    if im.file_format in ("PNG",) and not im.alpha_mode != "NONE":
        pass

# 6) 记录轴/尺寸
mn, mx = bbox(objs)
total = 0
for o in objs:
    o.data.calc_loop_triangles()
    total += len(o.data.loop_triangles)
print("NORM", os.path.basename(dst), "dims", tuple(round(mx[i] - mn[i], 2) for i in range(3)),
      "origin", tuple(round(v, 2) for v in mn), "tris", total, flush=True)
if DUMP_AXIS:
    print("AXISDUMP", DUMP_AXIS, "min", tuple(round(v, 2) for v in mn),
          "max", tuple(round(v, 2) for v in mx), flush=True)

os.makedirs(os.path.dirname(dst), exist_ok=True)
bpy.ops.export_scene.gltf(filepath=dst, export_format="GLB", export_apply=True,
                          export_animations=True, export_yup=True,
                          export_jpeg_quality=85)
print("SAVED", dst, os.path.getsize(dst), flush=True)
