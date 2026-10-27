# -*- coding: utf-8 -*-
"""
按具体型号自建的目标模型（Blender 4.5 无头）。

约定：1 单位 = 1 米，+Y 为舰艏/机头/车头，+Z 向上；海面/地面目标原点在
水线/地面（z=0）。输出 GLB 后 Cesium 中 scale=1；Cesium 1.145 实测模型
局部前向为 -X（EntityLod.forwardAxis() 返回 '-X'）。

运行：
  blender -b --python build_custom.py -- <输出目录> [型号 ...]

型号：052d 056a 903 panshi tuochiang kidd kj500 hq16 yj62
"""
import bpy
import math
import os
import sys
from mathutils import Vector

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
OUT_DIR = ARGS[0] if ARGS else "/tmp/custommodels"
ONLY = set(ARGS[1:]) if len(ARGS) > 1 else None
os.makedirs(OUT_DIR, exist_ok=True)

MATS = {}
TEX = {}


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    MATS.clear()
    TEX.clear()


def mat(name, rgb, rough=0.65, metal=0.0, alpha=1.0):
    if name in MATS:
        return MATS[name]
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes.get("Principled BSDF")
    if b:
        b.inputs["Base Color"].default_value = (rgb[0], rgb[1], rgb[2], 1.0)
        b.inputs["Roughness"].default_value = rough
        b.inputs["Metallic"].default_value = metal
        if alpha < 1.0:
            b.inputs["Alpha"].default_value = alpha
            m.blend_method = "BLEND"
    MATS[name] = m
    return m


def navy():
    return mat("haze_gray", (0.30, 0.33, 0.35), 0.55)


def deck():
    return mat("deck_gray", (0.20, 0.21, 0.22), 0.8)


def dark():
    return mat("dark", (0.045, 0.05, 0.055), 0.6)


def glass():
    return mat("glass", (0.05, 0.08, 0.10), 0.15, 0.2)


def red():
    return mat("anti_fouling", (0.28, 0.06, 0.05), 0.8)


def white():
    return mat("white", (0.75, 0.76, 0.75), 0.5)


def _tex(name, rgb, seed, panel=128, camo=False):
    """生成 1024² 贴图：船体用板缝/锈迹，车辆用迷彩斑块。返回 Blender 图像。"""
    import numpy as np
    d = "/tmp/modelgen_tex"
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(seed)
    w = h = 1024
    arr = np.ones((h, w, 4), dtype=np.float32)
    arr[..., 0], arr[..., 1], arr[..., 2] = rgb
    if camo:
        # 多尺度斑块：先低分辨噪声再放大，形成迷彩
        small = rng.normal(0, 1, (32, 32))
        big = np.kron(small, np.ones((32, 32)))
        big = (big - big.min()) / (big.max() - big.min() + 1e-6)
        m1 = big > 0.62
        m2 = (big <= 0.62) & (big > 0.45)
        arr[..., 0] = np.where(m1, rgb[0] * 0.45, np.where(m2, rgb[0] * 1.55, rgb[0]))
        arr[..., 1] = np.where(m1, rgb[1] * 0.45, np.where(m2, rgb[1] * 1.35, rgb[1]))
        arr[..., 2] = np.where(m1, rgb[2] * 0.45, np.where(m2, rgb[2] * 1.2, rgb[2]))
    arr[..., :3] += rng.normal(0, 0.012, (h, w, 1)).astype(np.float32)
    if not camo:
        for x in range(0, w, panel):
            arr[:, x:x + 2, :3] *= 0.74
        for y in range(0, h, panel):
            arr[y:y + 2, :, :3] *= 0.80
        # 锈迹/水渍竖条
        for _ in range(60):
            x = int(rng.integers(0, w))
            y0 = int(rng.integers(0, h))
            ln = int(rng.integers(16, 140))
            arr[y0:y0 + ln, x:x + int(rng.integers(1, 4)), :3] *= float(rng.uniform(0.72, 0.9))
    arr[..., 3] = 1.0
    img = bpy.data.images.new(name, width=w, height=h)
    img.pixels = np.clip(arr, 0, 1).ravel().tolist()
    img.filepath_raw = os.path.join(d, name + ".png")
    img.file_format = "PNG"
    img.save()
    return img


def use_texture(m, img, rough=0.6):
    nt = m.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = rough
    return m


def cube_uv(objs, size):
    for o in objs:
        if o.type != "MESH":
            continue
        if not o.data.uv_layers:
            o.data.uv_layers.new(name="UVMap")
        bpy.context.view_layer.objects.active = o
        o.select_set(True)
        try:
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.uv.cube_project(cube_size=size)
            bpy.ops.object.mode_set(mode="OBJECT")
        except Exception:
            try:
                bpy.ops.object.mode_set(mode="OBJECT")
            except Exception:
                pass
        o.select_set(False)


def texture_objects(mat_names, tex_name, rgb, seed, cube=10.0, camo=False, panel=128):
    """把指定材质的物体统一做 cube UV，并给材质挂生成贴图。"""
    if tex_name not in TEX:
        TEX[tex_name] = _tex(tex_name, rgb, seed, panel=panel, camo=camo)
    objs = [o for o in bpy.context.scene.objects
            if o.type == "MESH" and any(m and m.name in mat_names for m in o.data.materials)]
    cube_uv(objs, cube)
    for mn in mat_names:
        m = MATS.get(mn)
        if m and m.name in bpy.data.materials:
            use_texture(m, TEX[tex_name])


def obj_from_mesh(name, verts, faces, material):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.validate()
    me.update()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    if material:
        ob.data.materials.append(material)
    return ob


def box(name, center, size, material, bevel=0.0, rot_z=0.0):
    bpy.ops.mesh.primitive_cube_add(size=1, location=center, rotation=(0, 0, rot_z))
    o = bpy.context.object
    o.name = name
    o.scale = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    o.data.materials.append(material)
    if bevel > 0:
        md = o.modifiers.new("bev", "BEVEL")
        md.width = bevel
        md.segments = 2
        bpy.context.view_layer.objects.active = o
        bpy.ops.object.modifier_apply(modifier=md.name)
    return o


def frustum(name, center, bottom, top, height, material):
    """四棱台：bottom=(x,y) 底尺寸，top=(x,y) 顶尺寸，沿 Z 拉伸，底面在 center.z。"""
    cx, cy, cz = center
    bx, by = bottom[0] / 2.0, bottom[1] / 2.0
    tx, ty = top[0] / 2.0, top[1] / 2.0
    z0, z1 = cz, cz + height
    v = [(-bx, -by, z0), (bx, -by, z0), (bx, by, z0), (-bx, by, z0),
         (-tx, -ty, z1), (tx, -ty, z1), (tx, ty, z1), (-tx, ty, z1)]
    f = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return obj_from_mesh(name, [(x + cx, y + cy, z) for x, y, z in v], f, material)


def wedge(name, center, size, material, front_taper=0.6, back_taper=1.0):
    """船体上层建筑用的楔形块：沿 Y 前后收窄，带斜侧面。"""
    cx, cy, cz = center
    lx, ly, lz = size[0] / 2.0, size[1] / 2.0, size[2]
    fx, bxt = lx * front_taper, lx * back_taper
    z0, z1 = cz, cz + lz
    v = [(-bxt, -ly, z0), (bxt, -ly, z0), (fx, ly, z0), (-fx, ly, z0),
         (-bxt * 0.82, -ly * 0.92, z1), (bxt * 0.82, -ly * 0.92, z1),
         (fx * 0.72, ly * 0.9, z1), (-fx * 0.72, ly * 0.9, z1)]
    f = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return obj_from_mesh(name, [(x + cx, y + cy, z) for x, y, z in v], f, material)


def cyl(name, center, radius, depth, material, axis="Z", verts=24):
    rot = {"Z": (0, 0, 0), "X": (0, math.pi / 2, 0), "Y": (math.pi / 2, 0, 0)}[axis]
    bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=radius, depth=depth,
                                        location=center, rotation=rot)
    o = bpy.context.object
    o.name = name
    o.data.materials.append(material)
    return o


def sphere(name, center, radius, material, seg=20, rings=12):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=seg, ring_count=rings, radius=radius,
                                         location=center)
    o = bpy.context.object
    o.name = name
    o.data.materials.append(material)
    return o


def hull(length, beam, draft, freeboard, name="hull", bow_frac=0.62, transom=0.72):
    """船体放样：从艉到艏，截面为**不闭合**的舷弧（左舷顶->龙骨->右舷顶），
    甲板单独一片，避免与舷侧顶面共面造成 z-fighting。z=0 为水线，甲板高于 0。"""
    n = 20
    half = beam / 2.0
    sections = []
    for i in range(n + 1):
        t = i / n                       # 0=艉 1=艏
        y = -length / 2.0 + length * t
        if t < 0.12:
            f = transom + (1.0 - transom) * (t / 0.12)
        elif t > bow_frac:
            u = (t - bow_frac) / (1.0 - bow_frac)
            f = max(0.035, math.sqrt(max(0.0, 1.0 - u ** 1.7)))
        else:
            f = 1.0
        w = half * f
        fb = freeboard * (0.9 + 0.1 * math.sin(math.pi * min(1.0, t * 1.1)))
        if t > bow_frac:
            fb += 0.7 * ((t - bow_frac) / (1 - bow_frac)) ** 2
        # 龙骨线：艉部浅、舯部深、艏部上翘
        keel = -draft * (0.55 + 0.45 * math.sin(math.pi * min(1.0, 0.05 + t * 0.9)))
        pts = [
            (-w, fb),                       # 左舷甲板边
            (-w, fb - 0.5),
            (-w * 0.97, -draft * 0.30),
            (-w * 0.80, -draft * 0.72),
            (-w * 0.38, keel),
            (w * 0.38, keel),
            (w * 0.80, -draft * 0.72),
            (w * 0.97, -draft * 0.30),
            (w, fb - 0.5),
            (w, fb),                        # 右舷甲板边
        ]
        sections.append((y, pts))
    verts = []
    for y, pts in sections:
        for (x, z) in pts:
            verts.append((x, y, z))
    faces = []
    m = len(sections[0][1])
    for i in range(n):
        for j in range(m - 1):
            a = i * m + j
            b = i * m + j + 1
            c = (i + 1) * m + j + 1
            d = (i + 1) * m + j
            faces.append((a, b, c, d))
    # 艉封板：舷弧 + 甲板边闭合
    faces.append(tuple(range(m - 1, -1, -1)))
    # 甲板面（略高于舷顶 0.04 m，避免共面）
    dverts = []
    for y, pts in sections:
        dverts.append((pts[0][0], y, pts[0][1] + 0.04))
        dverts.append((pts[-1][0], y, pts[-1][1] + 0.04))
    base = len(verts)
    verts.extend(dverts)
    for i in range(n):
        a = base + i * 2
        b = base + i * 2 + 1
        c = base + (i + 1) * 2 + 1
        d = base + (i + 1) * 2
        faces.append((a, b, c, d))
    ob = obj_from_mesh(name, verts, faces, navy())
    # 甲板面单独材质
    ob.data.materials.append(deck())
    didx = len(ob.data.materials) - 1
    for p in ob.data.polygons:
        if p.index >= n * (m - 1) + 1:
            p.material_index = didx
    return ob


def paint_below_waterline(ob, draft):
    """给低于水线的面赋红色材质（按面中心 z 判断）。"""
    redm = red()
    ob.data.materials.append(redm)
    idx = len(ob.data.materials) - 1
    for p in ob.data.polygons:
        z = sum(ob.data.vertices[v].co.z for v in p.vertices) / len(p.vertices)
        if z < -0.15:
            p.material_index = idx


def bulwark(length, beam, freeboard, name="bulwark", bow_frac=0.62,
            transom=0.72, height=0.55):
    """舷墙：沿甲板边缘的薄条，双面，跟船体放样同一条宽度曲线。"""
    n = 20
    half = beam / 2.0
    verts, faces = [], []
    for i in range(n + 1):
        t = i / n
        y = -length / 2.0 + length * t
        if t < 0.12:
            f = transom + (1.0 - transom) * (t / 0.12)
        elif t > bow_frac:
            u = (t - bow_frac) / (1.0 - bow_frac)
            f = max(0.035, math.sqrt(max(0.0, 1.0 - u ** 1.7)))
        else:
            f = 1.0
        w = half * f
        fb = freeboard * (0.9 + 0.1 * math.sin(math.pi * min(1.0, t * 1.1)))
        if t > bow_frac:
            fb += 0.7 * ((t - bow_frac) / (1 - bow_frac)) ** 2
        for sx in (-1, 1):
            verts.append((sx * w, y, fb + 0.04))
            verts.append((sx * w, y, fb + 0.04 + height))
    for i in range(n):
        for side in range(2):
            k = i * 4 + side * 2
            faces.append((k, k + 1, k + 5, k + 4))
            faces.append((k + 4, k + 5, k + 1, k))
    return obj_from_mesh(name, verts, faces, navy())


def gun(prefix, y, z, scale=1.0, caliber=0.13):
    base = cyl(prefix + "_base", (0, y, z + 0.45 * scale), 1.6 * scale, 1.1 * scale, navy(), verts=24)
    tur = frustum(prefix + "_turret", (0, y, z + 1.0 * scale), (3.2 * scale, 4.2 * scale),
                  (2.2 * scale, 3.0 * scale), 1.5 * scale, navy())
    barrel = cyl(prefix + "_barrel", (0, y + 2.6 * scale, z + 1.75 * scale), caliber * scale,
                 4.6 * scale, dark(), axis="Y", verts=16)
    return [base, tur, barrel]


def ciws(prefix, y, z):
    cyl(prefix + "_b", (0, y, z + 0.35), 0.9, 0.7, white(), verts=18)
    cyl(prefix + "_d", (0, y, z + 1.05), 0.55, 0.7, white(), verts=18)
    cyl(prefix + "_g", (0, y - 0.9, z + 1.25), 0.09, 1.6, dark(), axis="Y", verts=10)


def vls(prefix, y, z, cells_x=8, cells_y=8, cell=0.72):
    w = cells_x * cell
    l = cells_y * cell
    box(prefix + "_pad", (0, y, z + 0.08), (w + 0.6, l + 0.6, 0.18), navy())
    for ix in range(cells_x):
        for iy in range(cells_y):
            x = -w / 2 + cell * (ix + 0.5)
            yy = y - l / 2 + cell * (iy + 0.5)
            box(prefix + "_c%d_%d" % (ix, iy), (x, yy, z + 0.2), (cell * 0.82, cell * 0.82, 0.1), dark(), bevel=0.02)


def mast(prefix, y, z, height, yards=()):
    cyl(prefix + "_m", (0, y, z + height / 2), 0.28, height, navy(), verts=12)
    for i, (yy, zz, ln) in enumerate(yards):
        cyl(prefix + "_y%d" % i, (0, yy, zz), 0.12, ln, navy(), axis="X", verts=8)
    # 顶部桅杆
    cyl(prefix + "_top", (0, y, z + height + 1.2), 0.09, 2.4, navy(), verts=8)


def radar_panel(prefix, x, y, z, w, h, thick=0.22):
    """倾斜平板相控阵（左右各一块）。"""
    o = box(prefix, (x, y, z), (thick, w, h), mat("panel", (0.18, 0.21, 0.22), 0.35),
            rot_z=0.0)
    return o


def hull_number(prefix, text, y, z, side=1.0):
    """舷号：用文字网格贴在舷侧（Blender 内置字体）。"""
    try:
        cu = bpy.data.curves.new(prefix, "FONT")
        cu.body = text
        cu.size = 2.2
        cu.align_x = "CENTER"
        cu.extrude = 0.02
        ob = bpy.data.objects.new(prefix, cu)
        bpy.context.scene.collection.objects.link(ob)
        ob.location = (side * 0.0, y, z)
        ob.rotation_euler = (math.pi / 2, 0, math.pi / 2 if side > 0 else -math.pi / 2)
        bpy.context.view_layer.objects.active = ob
        bpy.ops.object.convert(target="MESH")
        ob.data.materials.append(white())
    except Exception:
        pass


def finish(name, origin_mode="water"):
    """应用变换，统一原点：water=水线 z=0，ground=最低点 z=0，center=包围盒中心。"""
    # 贴图：舰体板缝、甲板、车辆迷彩（同材质物体统一 cube UV）
    have = set()
    for o in bpy.context.scene.objects:
        if o.type == "MESH":
            for m in o.data.materials:
                if m:
                    have.add(m.name)
    if "haze_gray" in have:
        texture_objects(["haze_gray"], "navy_panel", (0.30, 0.33, 0.35), 11, cube=10.0, panel=128)
    if "deck_gray" in have:
        texture_objects(["deck_gray"], "deck_panel", (0.16, 0.17, 0.18), 5, cube=8.0, panel=96)
    if "mil_green" in have:
        texture_objects(["mil_green"], "army_camo", (0.20, 0.24, 0.17), 7, cube=3.0, camo=True)
    objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    mn = Vector((1e9,) * 3)
    mx = Vector((-1e9,) * 3)
    for o in objs:
        for c in o.bound_box:
            w = o.matrix_world @ Vector(c)
            mn = Vector((min(mn[i], w[i]) for i in range(3)))
            mx = Vector((max(mx[i], w[i]) for i in range(3)))
    if origin_mode == "ground":
        dz = mn.z
    elif origin_mode == "center":
        dz = (mn.z + mx.z) / 2
    else:
        dz = 0.0
    dx = (mn.x + mx.x) / 2
    dy = 0.0  # 船体沿 y 以 0 为中心；若模型不对称再调
    for o in objs:
        o.location.x -= dx
        o.location.y -= dy
        o.location.z -= dz
        o.select_set(False)
    path = os.path.join(OUT_DIR, name + ".glb")
    bpy.ops.export_scene.gltf(filepath=path, export_format="GLB", export_apply=True,
                              export_animations=True, export_yup=True)
    tris = 0
    for o in objs:
        o.data.calc_loop_triangles()
        tris += len(o.data.loop_triangles)
    print("EXPORTED", name, os.path.getsize(path), "bytes", tris, "tris", flush=True)


# ---------------------------------------------------------------------------
# 052D 型驱逐舰：L 157 m, B 17.2 m, draft 6.2 m
# ---------------------------------------------------------------------------
def build_052d():
    reset()
    L, B, D = 157.0, 17.2, 6.2
    h = hull(L, B, D, 6.2, "052d_hull")
    paint_below_waterline(h, D)
    bulwark(L, B, 6.2, "052d_bulwark")
    deckz = 6.2
    # 艏部：防浪板/锚机/锚孔
    box("052d_breakwater", (0, 66.0, deckz + 0.6), (9.0, 0.4, 1.2), navy())
    for sx in (-1, 1):
        box("052d_windlass_%d" % sx, (sx * 3.0, 68.5, deckz + 0.45), (1.6, 2.2, 0.9), dark())
        box("052d_hawse_%d" % sx, (sx * 3.2, 71.5, deckz - 0.8), (0.6, 1.0, 0.9), dark())
    # 130 mm 主炮
    gun("052d_gun", 61.0, deckz, 1.05, 0.16)
    # 前部 VLS：32 单元（4 列 x 8 行）
    vls("052d_vls_f", 48.5, deckz, 4, 8)
    for sx in (-1, 1):
        sphere("052d_esm_%d" % sx, (sx * 3.9, 34.0, deckz + 10.9), 0.28, white(), seg=14, rings=8)
        box("052d_door_%d_0" % sx, (sx * 6.3, 18.0, deckz + 1.2), (0.3, 1.0, 2.0), dark())
        box("052d_door_%d_1" % sx, (sx * 6.72, -34.0, deckz + 1.6), (0.3, 1.0, 2.0), dark())
    for sx in (-1, 1):
        box("052d_vent_%d" % sx, (sx * 5.6, -6.0, deckz + 4.7), (0.5, 2.0, 0.8), dark())
    # 舰桥：三层阶梯 + 前窗/侧窗
    wedge("052d_bridge1", (0, 26, deckz), (13.5, 30, 4.4), navy(), 0.9, 1.0)
    wedge("052d_bridge2", (0, 27, deckz + 4.4), (11.4, 22, 3.8), navy(), 0.85, 0.98)
    wedge("052d_bridge3", (0, 29, deckz + 8.2), (9.0, 12, 3.2), navy(), 0.72, 0.9)
    box("052d_win1_f", (0, 40.0, deckz + 3.0), (10.4, 0.9, 1.2), glass())
    box("052d_win2_f", (0, 37.4, deckz + 6.8), (8.6, 0.9, 1.1), glass())
    box("052d_win3_f", (0, 34.4, deckz + 10.0), (7.0, 0.9, 1.0), glass())
    for sx in (-1, 1):
        box("052d_win1_s_%d" % sx, (sx * 5.9, 26.0, deckz + 3.0), (0.6, 9.0, 1.0), glass())
    # 346A 相控阵：前部两块（舰桥两舷）+ 后部两块（机库两舷）
    for sx in (-1, 1):
        box("052d_aesa_bp_f_%d" % sx, (sx * 5.12, 30.0, deckz + 6.2), (0.7, 4.7, 3.7), navy())
        box("052d_aesa_f_%d" % sx, (sx * 5.4, 30.0, deckz + 6.2), (0.3, 4.4, 3.4),
            mat("aesa", (0.14, 0.17, 0.18), 0.3))
        box("052d_aesa_bp_a_%d" % sx, (sx * 6.1, -28.0, deckz + 4.8), (0.7, 4.6, 4.0), navy())
        box("052d_aesa_a_%d" % sx, (sx * 6.4, -28.0, deckz + 4.8), (0.3, 4.3, 3.7),
            mat("aesa", (0.14, 0.17, 0.18), 0.3))
    # 前桅/雷达罩/卫通/导航雷达
    mast("052d_fmast", 29, deckz + 11.4, 7.0, yards=[(29, deckz + 16.4, 5.0)])
    cyl("052d_radome", (0, 20.5, deckz + 8.65), 0.8, 0.9, white(), verts=18)
    for sx in (-1, 1):
        sphere("052d_satcom_%d" % sx, (sx * 3.2, 26.5, deckz + 11.95), 0.55, white())
    cyl("052d_navradar", (0, 29.0, deckz + 15.4), 0.7, 0.16, dark(), verts=16)
    # 中部甲板室 + 双烟囱 + 反舰导弹（2x4）
    wedge("052d_mid", (0, -2, deckz), (13.2, 30, 5.0), navy(), 0.95, 1.0)
    for i, yy in enumerate((1.0, -11.0)):
        wedge("052d_funnel%d" % i, (0, yy, deckz + 5.0), (7.2, 10.0, 5.4), navy(), 0.8, 0.92)
        box("052d_funnel_top%d" % i, (0, yy - 0.8, deckz + 10.4), (5.6, 7.4, 0.5), dark())
    for sx in (-1, 1):
        box("052d_assm_base_%d" % sx, (sx * 5.3, 1.5, deckz + 4.9), (1.6, 5.8, 0.5), navy())
        for i in range(4):
            cyl("052d_assm_%d_%d" % (sx, i), (sx * 5.3, 3.5 - i * 1.35, deckz + 5.6),
                0.55, 6.0, mat("assm", (0.24, 0.27, 0.26), 0.5), axis="Y", verts=14)
    # 后部 VLS：32 单元（8 列 x 4 行），抬高的甲板室
    box("052d_avls_house", (0, -21.0, deckz + 1.7), (9.6, 8.0, 3.4), navy())
    vls("052d_vls_a", -21.0, deckz + 3.4, 8, 4)
    # 机库/机库门/侧窗
    wedge("052d_hangar", (0, -36, deckz), (14.0, 20, 7.2), navy(), 0.95, 1.0)
    box("052d_helo_door", (0, -46.2, deckz + 3.2), (7.0, 0.5, 5.2), dark())
    for sx in (-1, 1):
        box("052d_hangar_win_%d" % sx, (sx * 6.05, -36, deckz + 5.3), (0.5, 10.0, 1.0), glass())
    # 后桅
    mast("052d_rmast", -30.0, deckz + 7.2, 5.5, yards=[(-30.0, deckz + 11.0, 4.0)])
    cyl("052d_navradar2", (0, -30.0, deckz + 10.4), 0.6, 0.14, dark(), verts=14)
    # 1130 近防炮（舰桥前上方平台）与 HQ-10（机库顶）
    cyl("052d_ciws_base", (0, 36.5, deckz + 8.55), 0.95, 0.7, white(), verts=18)
    box("052d_ciws_body", (0, 36.5, deckz + 9.35), (1.05, 1.25, 0.9), white())
    for i in range(6):
        cyl("052d_ciws_g%d" % i, (-0.28 + i * 0.112, 37.35, deckz + 9.45),
            0.042, 1.7, dark(), axis="Y", verts=6)
    sphere("052d_ciws_radome", (0, 36.1, deckz + 9.95), 0.3, white(), seg=14, rings=8)
    box("052d_hq10_base", (0, -38.0, deckz + 7.5), (3.2, 3.6, 0.6), navy())
    box("052d_hq10_box", (0, -38.0, deckz + 8.6), (2.8, 3.2, 1.8),
        mat("launcher", (0.18, 0.20, 0.21), 0.55))
    for r in range(3):
        for c in range(8):
            box("052d_hq10_c%d_%d" % (r, c),
                (-1.19 + c * 0.34, -39.1 + r * 1.1, deckz + 9.55),
                (0.24, 0.24, 0.1), dark())
    # 鱼雷发射管（两舷各 3 管）/ 干扰弹发射器
    for sx in (-1, 1):
        box("052d_tt_base_%d" % sx, (sx * 7.2, -14.0, deckz + 1.0), (1.2, 3.4, 0.9), navy())
        for k in range(3):
            cyl("052d_tt_%d_%d" % (sx, k), (sx * 7.3, -15.2 + k * 1.0, deckz + 1.6),
                0.24, 2.4, dark(), axis="Y", verts=8)
        for yy in (12.0, -12.0):
            box("052d_decoy_%d_%d" % (sx, int(yy)), (sx * 7.0, yy, deckz + 1.4),
                (0.8, 1.8, 1.2), mat("launcher", (0.18, 0.20, 0.21), 0.55))
    # 舷侧 RHIB / 救生筏
    for sx in (-1, 1):
        box("052d_rhib_%d" % sx, (sx * 7.6, -20.0, deckz + 0.5), (1.4, 4.0, 0.85),
            mat("rhib", (0.18, 0.20, 0.21), 0.6))
        for i in range(4):
            cyl("052d_raft_%d_%d" % (sx, i), (sx * 7.2, -32.0 - i * 4.0, deckz + 0.45),
                0.45, 0.9, white(), axis="X", verts=10)
    # 飞行甲板标线
    box("052d_flightdeck", (0, -60, deckz + 0.10), (14.0, 28.0, 0.12), deck())
    cyl("052d_helo_circle", (0, -60, deckz + 0.18), 4.0, 0.05, white(), verts=40)
    finish("t_052d", "water")

# ---------------------------------------------------------------------------
# 056A 型轻型护卫舰：L 89 m, B 11.1 m, draft 4.4 m
# ---------------------------------------------------------------------------
def build_056a():
    reset()
    L, B, D = 89.0, 11.1, 4.4
    h = hull(L, B, D, 4.6, "056a_hull")
    paint_below_waterline(h, D)
    deckz = 4.6
    wedge("056a_bridge", (0, 10, deckz), (9.6, 22, 6.4), navy(), 0.78, 1.0)
    box("056a_windows", (0, 17.5, deckz + 5.0), (8.4, 3.2, 0.9), glass())
    for sx in (-1, 1):
        box("056a_aesa_%d" % sx, (sx * 4.95, 12.0, deckz + 4.3), (0.25, 3.4, 3.4),
            mat("aesa", (0.16, 0.19, 0.20), 0.35))
    mast("056a_mast", 4, deckz + 6.4, 9.0, yards=[(4, deckz + 11.0, 4.0)])
    gun("056a_gun", 31.0, deckz, 0.95, 0.076)
    ciws("056a_ciws", 20.0, deckz + 6.4)
    # 反舰导弹 2x2
    for sx in (-1, 1):
        for i in range(2):
            cyl("056a_assm_%d_%d" % (sx, i), (sx * 3.8, -1.0 + i * 1.2, deckz + 4.4), 0.55, 5.0,
                mat("assm", (0.24, 0.27, 0.26), 0.5), axis="Y", verts=12)
    wedge("056a_aft", (0, -16, deckz), (9.4, 20, 4.6), navy(), 0.95, 1.0)
    box("056a_flightdeck", (0, -33, deckz + 0.06), (10.2, 16.0, 0.12), deck())
    cyl("056a_helo_circle", (0, -33, deckz + 0.14), 2.8, 0.04, white(), verts=30)
    # 烟囱（后倾）
    wedge("056a_funnel", (0, -5.0, deckz + 2.4), (5.2, 7.0, 4.0), navy(), 0.82, 0.9)
    box("056a_funnel_top", (0, -5.4, deckz + 6.5), (4.0, 5.4, 0.35), dark())
    # 桅顶雷达罩
    cyl("056a_radome", (0, 4, deckz + 16.2), 0.85, 0.9, white(), verts=18)
    # 舷侧救生筏
    for sx in (-1, 1):
        for i in range(4):
            cyl("056a_raft_%d_%d" % (sx, i), (sx * 4.6, -6.0 + i * 4.0, deckz + 1.6),
                0.42, 0.8, white(), axis="X", verts=10)
    # 舰艉 HQ-10
    box("056a_hq10_base", (0, -23.0, deckz + 4.5), (2.2, 2.2, 0.5), navy())
    for i in range(4):
        box("056a_hq10_%d" % i, (-0.75 + 0.5 * i, -23.0, deckz + 5.0),
            (0.42, 0.42, 0.8), dark())
    # 右舷 RHIB
    box("056a_rhib", (4.3, -8.0, deckz + 1.2), (1.4, 3.2, 0.8), mat("rhib", (0.18, 0.20, 0.21), 0.6))
    finish("t_056a", "water")


# ---------------------------------------------------------------------------
# 903 型综合补给舰：L 178.5 m, B 24.8 m, draft 8.7 m
# ---------------------------------------------------------------------------
def build_903():
    reset()
    L, B, D = 178.5, 24.8, 8.7
    h = hull(L, B, D, 8.0, "903_hull", bow_frac=0.7, transom=0.8)
    paint_below_waterline(h, D)
    deckz = 8.0
    # 艏楼
    wedge("903_forecastle", (0, 62, deckz), (16.0, 30, 5.0), navy(), 0.85, 1.0)
    box("903_bridge_win", (0, 72, deckz + 3.6), (13.0, 4.0, 1.2), glass())
    mast("903_mast", 62, deckz + 5.0, 8.0)
    # 中部大型补给门架
    for i, yy in enumerate((25.0, 2.0, -22.0)):
        box("903_gantry%d" % i, (0, yy, deckz + 9.0), (2.0, 2.0, 18.0), navy())
        cyl("903_boom%d" % i, (0, yy - 6.0, deckz + 15.5), 0.45, 22.0, navy(), axis="Y", verts=12)
    # 上层建筑（后部）
    wedge("903_super", (0, -46, deckz), (20.0, 34, 12.0), navy(), 0.9, 1.0)
    box("903_super_win", (0, -34, deckz + 9.0), (17.0, 4.0, 1.2), glass())
    # 烟囱
    wedge("903_funnel", (0, -58, deckz + 12.0), (7.0, 8.0, 7.0), navy(), 0.85, 0.9)
    box("903_funnel_top", (0, -58, deckz + 19.0), (5.4, 6.0, 0.4), dark())
    # 货舱口盖
    for i, yy in enumerate((40.0, 12.0)):
        box("903_hatch%d" % i, (0, yy, deckz + 0.4), (12.0, 12.0, 0.8), mat("hatch", (0.32, 0.34, 0.33), 0.7))
    # 直升机甲板
    box("903_flightdeck", (0, -76, deckz + 0.08), (20.0, 22.0, 0.16), deck())
    cyl("903_helo_circle", (0, -76, deckz + 0.2), 4.4, 0.05, white(), verts=40)
    # --- 细节补强：翼台/舷窗/横向补给站/近防/救生筏/RHIB ---
    for sx in (-1, 1):
        box("903_bridgewing_%d" % sx, (sx * 7.6, 66.0, deckz + 3.0), (3.4, 7.0, 0.3), navy())
        box("903_deckhouse_win_%d" % sx, (sx * 6.9, 56.0, deckz + 2.2),
            (0.16, 10.0, 0.9), glass())
        box("903_super_side_win_%d" % sx, (sx * 8.2, -46.0, deckz + 9.4),
            (0.16, 14.0, 1.1), glass())
    # 舰桥前窗（原有窗带嵌在体块内，补一条真正露在外面的）
    box("903_bridge_win_f", (0, 75.9, deckz + 3.8), (9.6, 0.8, 1.2), glass())
    box("903_super_win_f", (0, -30.3, deckz + 9.0), (13.6, 0.8, 2.4), glass())
    # 舯部横向补给站：两舷各两组门架 + 外伸吊臂
    for sx in (-1, 1):
        for yy in (16.0, -14.0):
            cyl("903_kingpost_%d_%d" % (sx, int(yy)), (sx * 9.6, yy, deckz + 6.5),
                0.32, 13.0, navy(), verts=10)
            cyl("903_rasboom_%d_%d" % (sx, int(yy)), (sx * 14.2, yy, deckz + 12.4),
                0.22, 8.6, navy(), axis="X", verts=8)
            box("903_rashead_%d_%d" % (sx, int(yy)), (sx * 18.2, yy, deckz + 12.1),
                (0.9, 1.4, 0.7), dark())
    # 第三货舱口盖
    box("903_hatch2", (0, -8.0, deckz + 0.4), (12.0, 8.0, 0.8),
        mat("hatch", (0.32, 0.34, 0.33), 0.7))
    # 艏部防浪板/锚机
    box("903_breakwater", (0, 78.2, deckz + 0.7), (9.0, 0.4, 1.4), navy())
    for sx in (-1, 1):
        box("903_windlass_%d" % sx, (sx * 5.0, 72.0, deckz + 0.9), (1.8, 2.4, 1.0), dark())
    # 30 mm 近防：艏部两舷 + 艉部上层建筑顶
    for sx in (-1, 1):
        cyl("903_gun30_f_%d_b" % sx, (sx * 8.6, 52.0, deckz + 0.5), 0.8, 1.0, white(), verts=14)
        cyl("903_gun30_f_%d_g" % sx, (sx * 8.6, 51.0, deckz + 1.3), 0.12, 1.4, dark(), axis="Y", verts=8)
        cyl("903_gun30_a_%d_b" % sx, (sx * 6.0, -52.0, deckz + 12.6), 0.8, 1.0, white(), verts=14)
        cyl("903_gun30_a_%d_g" % sx, (sx * 6.0, -53.0, deckz + 13.4), 0.12, 1.4, dark(), axis="Y", verts=8)
    # 救生筏
    for sx in (-1, 1):
        for i in range(4):
            cyl("903_raft_%d_%d" % (sx, i), (sx * 10.3, -38.0 - i * 5.0, deckz + 1.8),
                0.45, 1.0, white(), axis="X", verts=8)
    # RHIB
    box("903_rhib", (9.7, -18.0, deckz + 1.2), (1.5, 4.6, 0.9),
        mat("rhib", (0.18, 0.20, 0.21), 0.6))
    # 桅顶导航雷达
    cyl("903_navradar", (0, 62.0, deckz + 13.6), 0.9, 0.16, dark(), verts=16)
    finish("t_903", "water")


# ---------------------------------------------------------------------------
# 磐石级快速战斗支援舰：L 196 m, B 25.9 m
# ---------------------------------------------------------------------------
def build_panshi():
    reset()
    L, B, D = 196.0, 25.9, 8.5
    h = hull(L, B, D, 8.2, "panshi_hull", bow_frac=0.72, transom=0.82)
    paint_below_waterline(h, D)
    deckz = 8.2
    wedge("panshi_bridge", (0, 48, deckz), (18.0, 34, 13.0), navy(), 0.85, 1.0)
    box("panshi_win", (0, 60, deckz + 10.2), (15.0, 4.0, 1.3), glass())
    mast("panshi_mast", 45, deckz + 13.0, 9.0)
    for i, yy in enumerate((18.0, -8.0, -34.0)):
        box("panshi_gantry%d" % i, (0, yy, deckz + 10.0), (2.2, 2.2, 20.0), navy())
        cyl("panshi_boom%d" % i, (0, yy - 6.0, deckz + 17.0), 0.5, 24.0, navy(), axis="Y", verts=12)
    for i, yy in enumerate((30.0, 2.0, -26.0)):
        box("panshi_hatch%d" % i, (0, yy, deckz + 0.45), (13.0, 13.0, 0.9),
            mat("hatch", (0.32, 0.34, 0.33), 0.7))
    wedge("panshi_funnel", (0, -50, deckz + 8.0), (8.0, 9.0, 8.0), navy(), 0.85, 0.9)
    box("panshi_funnel_top", (0, -50, deckz + 16.0), (6.0, 6.5, 0.4), dark())
    box("panshi_flightdeck", (0, -82, deckz + 0.08), (21.0, 24.0, 0.16), deck())
    cyl("panshi_helo", (0, -82, deckz + 0.2), 4.6, 0.05, white(), verts=40)
    # --- 细节补强：机库/烟囱基座/补给站/吊车/近防/救生筏/RHIB ---
    # 烟囱基座（原烟囱悬空于甲板之上）
    box("panshi_funnel_base", (0, -50.0, deckz + 4.0), (9.0, 10.0, 8.0), navy())
    # 艉部机库
    wedge("panshi_hangar", (0, -61.0, deckz), (14.0, 16.0, 5.6), navy(), 0.95, 1.0)
    box("panshi_hangar_door", (0, -68.7, deckz + 2.2), (8.0, 0.5, 3.6), dark())
    for sx in (-1, 1):
        box("panshi_hangar_win_%d" % sx, (sx * 5.75, -61.0, deckz + 4.6),
            (0.16, 10.0, 0.9), glass())
    # 舰桥翼台/侧窗/前窗
    for sx in (-1, 1):
        box("panshi_bridgewing_%d" % sx, (sx * 8.6, 56.0, deckz + 3.2), (3.2, 6.0, 0.3), navy())
        box("panshi_bridge_sidewin_%d" % sx, (sx * 7.1, 48.0, deckz + 10.6),
            (0.16, 12.0, 1.1), glass())
    box("panshi_bridge_win_f", (0, 63.5, deckz + 10.2), (10.0, 0.8, 1.2), glass())
    # 两舷横向补给站
    for sx in (-1, 1):
        for yy in (10.0, -18.0):
            cyl("panshi_kingpost_%d_%d" % (sx, int(yy)), (sx * 10.0, yy, deckz + 7.0),
                0.34, 14.0, navy(), verts=10)
            cyl("panshi_rasboom_%d_%d" % (sx, int(yy)), (sx * 14.8, yy, deckz + 13.4),
                0.24, 9.0, navy(), axis="X", verts=8)
            box("panshi_rashead_%d_%d" % (sx, int(yy)), (sx * 19.4, yy, deckz + 13.0),
                (0.9, 1.4, 0.7), dark())
    # 甲板吊车
    for i, (cx, cy) in enumerate(((7.2, 26.0), (-7.2, -12.0))):
        cyl("panshi_crane%d_base" % i, (cx, cy, deckz + 1.8), 0.7, 3.6, navy(), verts=10)
        box("panshi_crane%d_cab" % i, (cx, cy, deckz + 4.1), (1.5, 1.7, 1.2), navy())
        box("panshi_crane%d_jib" % i, (cx - 2.9 if cx > 0 else cx + 2.9, cy, deckz + 4.6),
            (5.4, 0.5, 0.5), navy())
    # 30 mm 近防（舰桥翼台）
    for sx in (-1, 1):
        cyl("panshi_gun30_%d_b" % sx, (sx * 8.6, 56.0, deckz + 3.65), 0.7, 0.9, white(), verts=14)
        cyl("panshi_gun30_%d_g" % sx, (sx * 8.6, 55.2, deckz + 4.35), 0.11, 1.2, dark(), axis="Y", verts=8)
    # 救生筏
    for sx in (-1, 1):
        for i in range(4):
            cyl("panshi_raft_%d_%d" % (sx, i), (sx * 9.3, 40.0 - i * 5.5, deckz + 1.9),
                0.46, 1.0, white(), axis="X", verts=8)
    # 两舷补给管线
    for sx in (-1, 1):
        box("panshi_pipe_%d" % sx, (sx * 10.8, -8.0, deckz + 0.35),
            (0.18, 96.0, 0.18), dark())
    # 舷侧 RHIB
    for sx in (-1, 1):
        box("panshi_rhib_%d" % sx, (sx * 10.6, 28.0, deckz + 1.2), (1.4, 4.4, 0.85),
            mat("rhib", (0.18, 0.20, 0.21), 0.6))
    # 桅顶导航雷达
    cyl("panshi_navradar", (0, 45.0, deckz + 22.4), 0.9, 0.16, dark(), verts=16)
    finish("t_panshi", "water")


# ---------------------------------------------------------------------------
# 沱江级巡逻舰（双体）：L 60.4 m, B 14 m
# ---------------------------------------------------------------------------
def build_tuochiang():
    reset()
    L, B = 60.4, 14.0
    # 两个细长片体
    for sx in (-1, 1):
        h = hull(L, 4.6, 2.2, 3.2, "tc_hull_%d" % sx, bow_frac=0.55, transom=0.75)
        h.location.x = sx * (B / 2 - 2.3)
        paint_below_waterline(h, 2.2)
    deckz = 4.6
    # 连接桥/主甲板
    box("tc_deck", (0, 0, deckz), (B, L * 0.72, 1.4), navy(), bevel=0.15)
    # 隐身上层建筑
    wedge("tc_super", (0, 4, deckz + 1.4), (10.0, 28, 5.6), navy(), 0.65, 1.0)
    box("tc_win", (0, 14, deckz + 5.6), (8.6, 3.0, 0.9), glass())
    wedge("tc_funnel", (0, -6, deckz + 5.0), (5.0, 6.0, 3.2), navy(), 0.8, 0.85)
    box("tc_funnel_top", (0, -6, deckz + 8.2), (3.8, 4.4, 0.3), dark())
    mast("tc_mast", 8, deckz + 7.0, 8.0)
    gun("tc_gun", 24.0, deckz + 1.4, 0.85, 0.076)
    ciws("tc_ciws", 12.0, deckz + 7.0)
    # 反舰导弹 2x2（舯部）
    for sx in (-1, 1):
        for i in range(2):
            cyl("tc_assm_%d_%d" % (sx, i), (sx * 3.2, -12.0 + i * 1.3, deckz + 3.2), 0.5, 4.6,
                mat("assm", (0.24, 0.27, 0.26), 0.5), axis="Y", verts=12)
    # --- 细节补强：翼台/侧窗/第二组反舰弹/雷达/救生筏/喷水口/艉部 ---
    for sx in (-1, 1):
        box("tc_bridgewing_%d" % sx, (sx * 4.2, 14.0, deckz + 3.0), (2.0, 4.0, 0.25), navy())
        box("tc_sidewin_%d" % sx, (sx * 3.65, 6.0, deckz + 4.4), (0.14, 9.0, 0.7), glass())
    # 舰桥前窗（原有窗带嵌在体块内）
    box("tc_win_f", (0, 17.0, deckz + 5.6), (4.8, 0.8, 1.0), glass())
    # 雷达/桅顶
    cyl("tc_radar", (0, 8.0, deckz + 11.4), 1.1, 0.2, mat("radar", (0.22, 0.24, 0.24), 0.4), verts=20)
    cyl("tc_esm", (0, 8.0, deckz + 14.6), 0.18, 0.9, dark(), verts=8)
    # 反舰导弹：每舷 4 管（第二组 + 共用底座）
    for sx in (-1, 1):
        box("tc_asm_base_%d" % sx, (sx * 3.2, -12.7, deckz + 2.4), (2.6, 7.4, 0.8), navy())
        for i in range(2):
            cyl("tc_assm2_%d_%d" % (sx, i), (sx * 3.2, -16.0 + i * 1.3, deckz + 3.2),
                0.5, 4.6, mat("assm", (0.24, 0.27, 0.26), 0.5), axis="Y", verts=12)
    # 救生筏
    for sx in (-1, 1):
        for i in range(4):
            cyl("tc_raft_%d_%d" % (sx, i), (sx * 6.9, -14.0 + i * 4.4, deckz + 1.3),
                0.34, 1.1, white(), axis="X", verts=8)
    # 喷水推进出口
    for sx in (-1, 1):
        for k in range(2):
            cyl("tc_jet_%d_%d" % (sx, k), (sx * (3.4 + k * 2.4), -30.6, 3.0),
                0.36, 1.2, dark(), axis="Y", verts=10)
    # 艉部低甲板 + RHIB
    box("tc_stern_deck", (0, -26.0, 3.5), (6.0, 6.0, 0.2), deck())
    box("tc_rhib", (0, -26.0, 3.95), (1.3, 3.8, 0.8),
        mat("rhib", (0.18, 0.20, 0.21), 0.6))
    # 锚机
    for sx in (-1, 1):
        box("tc_hawse_%d" % sx, (sx * 4.7, 28.6, 3.6), (0.7, 1.1, 0.8), dark())
    finish("t_tuochiang", "water")


# ---------------------------------------------------------------------------
# 基隆级（Kidd）驱逐舰：L 171.6 m, B 16.8 m
# ---------------------------------------------------------------------------
def build_kidd():
    reset()
    L, B, D = 171.6, 16.8, 9.6
    h = hull(L, B, D, 7.2, "kidd_hull")
    paint_below_waterline(h, D)
    deckz = 7.2
    # 舰桥
    wedge("kidd_bridge", (0, 42, deckz), (13.0, 22, 8.5), navy(), 0.8, 1.0)
    box("kidd_win", (0, 52, deckz + 6.6), (11.0, 3.4, 1.1), glass())
    # 主炮
    gun("kidd_gun", 62.0, deckz, 1.05, 0.127)
    # 前部 Mk26 双臂发射架
    cyl("kidd_mk26_f", (0, 33, deckz + 0.9), 2.6, 1.8, navy(), verts=20)
    box("kidd_arm_f", (0, 34.5, deckz + 2.0), (1.0, 5.0, 0.35), dark(), rot_z=0.0)
    # 前桅/雷达
    mast("kidd_mast", 20, deckz + 8.5, 14.0)
    box("kidd_sps48", (0, 20, deckz + 22.0), (7.0, 0.4, 4.4), mat("radar", (0.22, 0.24, 0.24), 0.4))
    # 两个大烟囱
    for i, yy in enumerate((4.0, -22.0)):
        wedge("kidd_funnel%d" % i, (0, yy, deckz + 6.5), (8.0, 10.0, 8.0), navy(), 0.85, 0.9)
        box("kidd_funnel_top%d" % i, (0, yy - 0.5, deckz + 14.5), (6.0, 7.0, 0.4), dark())
    wedge("kidd_mid", (0, -8, deckz), (14.0, 26, 6.5), navy(), 0.95, 1.0)
    # 后部上层建筑 + 机库
    wedge("kidd_hangar", (0, -46, deckz), (14.0, 30, 7.5), navy(), 0.95, 1.0)
    box("kidd_helo_door", (0, -60, deckz + 3.4), (7.0, 0.5, 5.2), dark())
    # 后部 Mk26
    cyl("kidd_mk26_a", (0, -66, deckz + 1.6), 2.6, 1.8, navy(), verts=20)
    box("kidd_flightdeck", (0, -78, deckz + 0.08), (14.0, 20.0, 0.14), deck())
    cyl("kidd_helo", (0, -78, deckz + 0.2), 4.2, 0.05, white(), verts=36)
    # --- 细节补强：翼台/舷窗/鱼叉/照射雷达/密集阵/鱼雷管/后桅 ---
    for sx in (-1, 1):
        box("kidd_bridgewing_%d" % sx, (sx * 7.4, 40.0, deckz + 3.0), (2.6, 6.0, 0.3), navy())
        box("kidd_mid_win_%d" % sx, (sx * 6.0, -8.0, deckz + 4.8), (0.16, 12.0, 1.0), glass())
        box("kidd_hangar_win_%d" % sx, (sx * 5.95, -46.0, deckz + 5.4), (0.16, 16.0, 1.1), glass())
    # 鱼叉 2x4（中层甲板顶，两舷）
    for sx in (-1, 1):
        box("kidd_harpoon_base_%d" % sx, (sx * 4.2, -9.0, deckz + 6.9), (2.6, 5.6, 0.8), navy())
        for iy in (-1, 0):
            for iz in (0, 1):
                cyl("kidd_harpoon_%d_%d_%d" % (sx, iy, iz),
                    (sx * 4.2 + iy * 0.75, -9.0, deckz + 7.7 + iz * 0.8),
                    0.3, 5.2, mat("harpoon", (0.22, 0.25, 0.25), 0.5), axis="Y", verts=10)
    # SPG-51 照射雷达（前 1 后 2）
    for key, (px, py, pz) in (("f", (0.0, 33.0, deckz + 9.0)),
                              ("a1", (-3.0, -32.0, deckz + 7.6)),
                              ("a2", (3.0, -32.0, deckz + 7.6))):
        cyl("kidd_spg_%s_ped" % key, (px, py, pz), 0.28, 1.8, navy(), verts=8)
        cyl("kidd_spg_%s_dish" % key, (px, py, pz + 1.5), 1.05, 0.26, white(), axis="Y", verts=18)
    # 密集阵
    ciws("kidd_ciws_f", 57.0, deckz + 0.6)
    ciws("kidd_ciws_a", -40.0, deckz + 7.6)
    # 鱼雷发射管（两舷各 3 管）
    for sx in (-1, 1):
        box("kidd_tt_base_%d" % sx, (sx * 5.4, -24.0, deckz + 1.1), (1.2, 3.6, 0.9), navy())
        for k in range(3):
            cyl("kidd_tt_%d_%d" % (sx, k), (sx * 5.5, -23.2 + k * 1.0, deckz + 1.7),
                0.26, 2.6, dark(), axis="Y", verts=8)
    # 后桅 + SPS-49
    mast("kidd_mast2", -34.0, deckz + 7.5, 8.0)
    box("kidd_sps49", (0, -34.0, deckz + 16.4), (5.4, 0.35, 2.6),
        mat("radar", (0.22, 0.24, 0.24), 0.4))
    # Mk26 装填舱
    box("kidd_mk26_f_house", (0, 30.6, deckz + 1.3), (3.6, 3.0, 2.2), navy())
    box("kidd_mk26_a_house", (0, -63.8, deckz + 2.0), (3.6, 3.0, 2.2), navy())
    box("kidd_arm_a", (0, -64.8, deckz + 2.6), (1.0, 5.0, 0.35), dark())
    # 救生筏
    for sx in (-1, 1):
        for i in range(4):
            cyl("kidd_raft_%d_%d" % (sx, i), (sx * 7.2, -38.0 - i * 5.0, deckz + 1.6),
                0.42, 0.9, white(), axis="X", verts=8)
    # 舷侧 RHIB
    box("kidd_rhib", (7.75, -10.0, deckz + 1.2), (1.4, 4.2, 0.85),
        mat("rhib", (0.18, 0.20, 0.21), 0.6))
    # 锚孔
    for sx in (-1, 1):
        box("kidd_hawse_%d" % sx, (sx * 2.6, 79.0, deckz - 0.8), (0.9, 1.3, 1.0), dark())
    # 前桅基座甲板室与 2 号烟囱基座（原结构悬空于甲板之上）
    wedge("kidd_fwd_house", (0, 20.0, deckz), (7.0, 8.0, 8.5), navy(), 0.9, 1.0)
    for sx in (-1, 1):
        box("kidd_fwd_house_win_%d" % sx, (sx * 3.55, 20.0, deckz + 5.7),
            (0.14, 5.0, 1.0), glass())
    box("kidd_funnel2_base", (0, -22.0, deckz + 3.25), (8.0, 9.0, 6.5), navy())
    finish("t_kidd", "water")


# ---------------------------------------------------------------------------
# 空警-500：Y-9 平台 + 圆盘雷达，L 36 m, 翼展 38 m
# ---------------------------------------------------------------------------
def build_kj500():
    reset()
    skin = mat("air_skin", (0.62, 0.64, 0.64), 0.45, 0.1)
    L = 36.0
    # 机身：椭圆截面放样
    n = 14
    verts = []
    for i in range(n + 1):
        t = i / n
        y = -L / 2 + L * t
        # 头尾收细
        f = math.sin(min(1.0, max(0.02, (1 - abs(2 * t - 1) ** 1.5)))) ** 0.55
        r = 2.15 * f + 0.28
        if t < 0.15:
            r *= 0.82 + t
        if t > 0.9:
            r *= 0.9
        ring = []
        for k in range(12):
            a = 2 * math.pi * k / 12
            ring.append((math.cos(a) * r, y, math.sin(a) * r * 0.95 + 2.4))
        verts.extend(ring)
    faces = []
    for i in range(n):
        for k in range(12):
            a = i * 12 + k
            b = i * 12 + (k + 1) % 12
            c = (i + 1) * 12 + (k + 1) % 12
            d = (i + 1) * 12 + k
            faces.append((a, b, c, d))
    faces.append(tuple(range(11, -1, -1)))
    faces.append(tuple(range(n * 12, n * 12 + 12)))
    fus = obj_from_mesh("kj_fuselage", verts, faces, skin)
    # 机翼（后掠 + 上反 0）
    for sx in (-1, 1):
        wing = []
        root_y, tip_y = 2.0, -1.2
        root_x, tip_x = sx * 2.0, sx * 19.0
        chord_r, chord_t = 6.2, 2.6
        z = 2.6
        vs = [(root_x, root_y - chord_r / 2, z), (root_x, root_y + chord_r / 2, z),
              (tip_x, tip_y + chord_t / 2, z + 0.8), (tip_x, tip_y - chord_t / 2, z + 0.8)]
        # 厚度
        th = 0.35
        vs2 = [(x, y, zz - th) for x, y, zz in vs]
        vv = [(x, y, zz + th) for x, y, zz in vs] + vs2
        ff = [(0, 1, 2, 3), (7, 6, 5, 4), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
        obj_from_mesh("kj_wing_%d" % sx, vv, ff, skin)
    # 平尾
    for sx in (-1, 1):
        box("kj_tailplane_%d" % sx, (sx * 5.2, -15.8, 3.6), (8.2, 2.6, 0.28), skin)
    # 垂尾
    box("kj_fin", (0, -15.0, 7.2), (0.5, 6.0, 6.4), skin)
    # 4 台涡桨
    for sx in (-1, 1):
        for k, yy in enumerate((3.2, -2.2)):
            x = sx * 8.2
            cyl("kj_eng_%d_%d" % (sx, k), (x, yy + 2.0, 2.2), 0.75, 4.6, skin, axis="Y", verts=16)
            cyl("kj_prop_%d_%d" % (sx, k), (x, yy + 4.5, 2.2), 2.4, 0.12,
                mat("prop_disk", (0.12, 0.12, 0.13), 0.5), axis="Y", verts=24)
    # 圆盘雷达罩
    cyl("kj_dome", (0, -1.0, 7.8), 3.7, 1.15, mat("dome", (0.55, 0.57, 0.57), 0.5), verts=40)
    box("kj_dome_strut1", (0, 2.5, 6.0), (0.4, 0.4, 3.6), skin)
    box("kj_dome_strut2", (0, -4.5, 6.0), (0.4, 0.4, 3.6), skin)
    # 起落架鼓包
    box("kj_gear_f", (0, 8.0, 0.6), (3.2, 5.0, 1.2), skin)
    finish("t_kj500", "center")


# ---------------------------------------------------------------------------
# 红旗-16 发射车：8x8 + 6 联装
# ---------------------------------------------------------------------------
def build_hq16():
    reset()
    green = mat("mil_green", (0.18, 0.22, 0.16), 0.75)
    rubber = mat("rubber", (0.04, 0.04, 0.045), 0.9)
    # 底盘
    box("hq16_chassis", (0, 0, 1.3), (3.0, 10.5, 0.7), green, bevel=0.05)
    box("hq16_cab", (0, 4.2, 2.4), (2.9, 2.4, 1.9), green, bevel=0.08)
    box("hq16_win", (0, 5.45, 2.9), (2.6, 0.1, 0.8), glass())
    # 车轮 8x
    for sx in (-1, 1):
        for i in range(4):
            cyl("hq16_wheel_%d_%d" % (sx, i), (sx * 1.55, 3.2 - i * 2.1, 0.72), 0.72, 0.42,
                rubber, axis="X", verts=20)
    # 发射架：6 联装，3 列 x 2 层，整体后仰
    box("hq16_base", (0, -2.6, 2.15), (3.2, 6.0, 0.5), green)
    for j in range(2):
        for i in range(3):
            x = (i - 1) * 1.05
            z = 3.35 + j * 1.1
            cyl("hq16_tube_%d_%d" % (i, j), (x, -2.6, z), 0.46, 5.4,
                mat("tube", (0.22, 0.25, 0.19), 0.7), axis="Y", verts=14)
    # 雷达
    box("hq16_radar", (0, -6.5, 3.0), (1.8, 0.2, 1.6), mat("radar", (0.2, 0.23, 0.2), 0.5))
    finish("t_hq16", "ground")


# ---------------------------------------------------------------------------
# 鹰击-62 岸舰导弹发射车
# ---------------------------------------------------------------------------
def build_yj62():
    reset()
    green = mat("mil_green", (0.18, 0.22, 0.16), 0.75)
    rubber = mat("rubber", (0.04, 0.04, 0.045), 0.9)
    box("yj62_chassis", (0, 0, 1.25), (3.2, 12.0, 0.75), green, bevel=0.05)
    box("yj62_cab", (0, 4.8, 2.5), (3.0, 2.6, 2.0), green, bevel=0.08)
    box("yj62_win", (0, 6.15, 3.0), (2.7, 0.12, 0.85), glass())
    for sx in (-1, 1):
        for i in range(4):
            cyl("yj62_wheel_%d_%d" % (sx, i), (sx * 1.65, 3.6 - i * 2.3, 0.75), 0.75, 0.45,
                rubber, axis="X", verts=20)
    # 三联装大发射筒（横向并排）
    box("yj62_base", (0, -3.0, 2.1), (3.4, 7.0, 0.55), green)
    for i in range(3):
        x = (i - 1) * 1.35
        cyl("yj62_tube%d" % i, (x, -3.0, 3.35), 0.62, 7.6,
            mat("tube", (0.22, 0.25, 0.19), 0.7), axis="Y", verts=18)
    finish("t_yj62", "ground")


# ---------------------------------------------------------------------------
# M777 超轻型榴弹炮（牵引状态）：全长 10.2 m，宽 2.8 m
# ---------------------------------------------------------------------------
def build_m777():
    reset()
    green = mat("mil_green", (0.20, 0.24, 0.16), 0.75)
    darkm = mat("dark", (0.05, 0.05, 0.055), 0.7)
    rubber = mat("rubber", (0.04, 0.04, 0.045), 0.9)
    # 炮管（+Y 向前，略上扬）
    cyl("m777_barrel", (0, 3.6, 1.55), 0.09, 5.4, green, axis="Y", verts=16)
    box("m777_muzzle", (0, 6.2, 1.62), (0.34, 0.7, 0.34), darkm, bevel=0.04)
    # 炮闩/摇架
    box("m777_cradle", (0, 1.0, 1.35), (0.7, 1.8, 0.8), green, bevel=0.06)
    cyl("m777_recoil1", (0.45, 1.6, 1.55), 0.12, 2.0, darkm, axis="Y", verts=10)
    cyl("m777_recoil2", (-0.45, 1.6, 1.55), 0.12, 2.0, darkm, axis="Y", verts=10)
    # 防盾（两块内倾板）
    box("m777_shield_l", (-0.85, 2.2, 1.1), (0.08, 1.6, 1.3), green, rot_z=0.0)
    box("m777_shield_r", (0.85, 2.2, 1.1), (0.08, 1.6, 1.3), green, rot_z=0.0)
    # 车轮
    for sx in (-1, 1):
        cyl("m777_wheel_%d" % sx, (sx * 1.25, 0.8, 0.62), 0.62, 0.45, rubber, axis="X", verts=20)
        cyl("m777_hub_%d" % sx, (sx * 1.5, 0.8, 0.62), 0.22, 0.12, green, axis="X", verts=12)
    # 大架（两条向后张开的梁）+ 驻锄
    for sx in (-1, 1):
        o = box("m777_trail_%d" % sx, (sx * 0.9, -1.3, 0.45), (0.22, 4.2, 0.25), green, rot_z=sx * 0.12)
        box("m777_spade_%d" % sx, (sx * 1.4, -3.35, 0.45), (0.5, 0.5, 0.8), darkm, bevel=0.03)
    # 液压/工具箱
    box("m777_box", (0, 0.4, 0.95), (1.1, 0.9, 0.5), green, bevel=0.04)
    box("m777_seat", (-0.7, 0.2, 1.05), (0.4, 0.5, 0.35), darkm)
    finish("t_m777", "ground")



# ---------------------------------------------------------------------------
# 075 型两栖攻击舰：L 237 m, B 36 m, 直通甲板 + 右舷舰岛（近似）
# ---------------------------------------------------------------------------
def build_075():
    reset()
    L, B, D = 237.0, 36.0, 8.1
    h = hull(L, B, D, 15.0, "075_hull", bow_frac=0.74, transom=0.88)
    paint_below_waterline(h, D)
    deckz = 15.0
    # 直通飞行甲板（略宽于船体，模拟外飘）
    box("075_deck", (0, 4, deckz + 0.7), (34.0, 206.0, 1.4), deck())
    box("075_deck_edge", (0, 4, deckz + 1.45), (34.4, 206.4, 0.18), navy())
    # 甲板标线 / 起降点
    for i, yy in enumerate((-78.0, -45.0, -12.0, 21.0, 54.0, 87.0)):
        cyl("075_spot%d" % i, (0, yy, deckz + 1.53), 4.6, 0.06, white(), verts=36)
        box("075_line%d" % i, (0, yy, deckz + 1.54), (8.0, 0.22, 0.04), white())
    # 右舷舰岛
    wedge("075_island", (9.6, 56, deckz + 1.4), (12.0, 48.0, 17.0), navy(), 0.88, 0.96)
    box("075_bridge_win", (9.6, 68, deckz + 15.0), (10.0, 5.0, 1.6), glass())
    box("075_island_win2", (9.6, 50, deckz + 11.0), (10.2, 4.0, 1.3), glass())
    cyl("075_mast", (9.6, 60, deckz + 20.0), 0.25, 6.0, navy(), verts=12)
    cyl("075_yard", (9.6, 60, deckz + 22.8), 0.11, 7.0, navy(), axis="X", verts=8)
    # 烟囱
    wedge("075_funnel", (9.6, 32, deckz + 13.0), (7.0, 9.0, 6.0), navy(), 0.85, 0.9)
    box("075_funnel_top", (9.6, 32, deckz + 19.0), (5.4, 7.0, 0.4), dark())
    # 两台舷侧升降机
    for i, yy in enumerate((54.0, -28.0)):
        box("075_elev%d" % i, (-15.0, yy, deckz + 1.55), (8.0, 18.0, 0.55), navy())
    # 艉部坞舱门与艏门
    box("075_welldeck", (0, -112.0, deckz - 4.5), (20.0, 10.0, 7.0), dark())
    box("075_bowdoor", (0, 112.0, deckz - 2.0), (12.0, 4.0, 8.0), dark())
    # 近防/防空
    ciws("075_ciws_f", 100.0, deckz + 1.6)
    ciws("075_ciws_a", -100.0, deckz + 1.6)
    box("075_hq10", (-12.0, -44.0, deckz + 2.1), (2.4, 2.4, 1.2), navy())
    # 救生筏
    for sx in (-1, 1):
        for i in range(6):
            cyl("075_raft_%d_%d" % (sx, i), (sx * 14.0, -52.0 + i * 14.0, deckz + 1.8),
                0.5, 0.9, white(), axis="X", verts=8)
    finish("t_075", "water")


BUILDERS = {
    "052d": build_052d, "056a": build_056a, "903": build_903, "panshi": build_panshi,
    "tuochiang": build_tuochiang, "kidd": build_kidd, "kj500": build_kj500,
    "hq16": build_hq16, "yj62": build_yj62, "m777": build_m777, "075": build_075,
}

for key, fn in BUILDERS.items():
    if ONLY and key not in ONLY:
        continue
    fn()
print("ALL_DONE")
