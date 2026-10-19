"""
目标 3D 模型资产生成（Blender 无头）。

路线（用户 2026-10-19 决策）：优先 CC0 资产，缺失形态用同风格的低模程序化补足，
全部按真实量级建模/归一，并给需要转动的部件（旋翼、雷达）加基础动画。

本脚本在 Blender 内运行：
    blender -b --python tools/modelgen/build_models.py -- <输出目录> <Mig-15.blend 路径>

产出（写进 situation3d/web/assets/models/）：
    fighter.glb        MiG-15（OpenGameArt CC0，Sketlux/yd），由 .blend 转 glb
    awacs.glb          预警机 + 旋转雷达罩（程序化）
    helicopter.glb     武装直升机 + 主/尾旋翼（程序化）
    apc.glb            轮式装甲车（程序化）
    artillery.glb      自行火炮（程序化）
    sam_launcher.glb   防空导弹发射车（程序化）
    coastal_missile.glb 岸舰导弹发射车（程序化）

约定：Blender 里 Z 向上、机头/车头朝 +Y；导出 glTF 后 Cesium 中 +Y 为北。
程序化模型按**米**建模（1 单位 = 1 米），所以 Entity.model.scale=1。
"""
import bpy
import math
import os
import sys
from mathutils import Vector

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
OUT_DIR = ARGS[0] if ARGS else "/tmp/models"
MIG_BLEND = ARGS[1] if len(ARGS) > 1 else ""

os.makedirs(OUT_DIR, exist_ok=True)

GREEN = None
DARK = None
GRAY = None


def reset():
    global GREEN, DARK, GRAY
    bpy.ops.wm.read_factory_settings(use_empty=True)
    GREEN = mat("mil_green", (0.20, 0.24, 0.17), 0.85)
    DARK = mat("dark", (0.06, 0.07, 0.08), 0.7)
    GRAY = mat("gray", (0.42, 0.45, 0.48), 0.6)
    GLASS = mat("glass", (0.08, 0.14, 0.18), 0.2, 0.6)
    return GLASS


def mat(name, rgb, rough=0.7, metal=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (rgb[0], rgb[1], rgb[2], 1.0)
        bsdf.inputs["Roughness"].default_value = rough
        bsdf.inputs["Metallic"].default_value = metal
    return m


def cube(name, loc, dims, material, rot=(0, 0, 0)):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc, rotation=rot)
    o = bpy.context.object
    o.name = name
    o.scale = dims
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    o.data.materials.append(material)
    return o


def cyl(name, loc, radius, depth, material, rot=(0, 0, 0), verts=16):
    bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=radius, depth=depth,
                                        location=loc, rotation=rot)
    o = bpy.context.object
    o.name = name
    o.data.materials.append(material)
    return o


def sphere(name, loc, dims, material):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=0.5, location=loc)
    o = bpy.context.object
    o.name = name
    o.scale = dims
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    o.data.materials.append(material)
    return o


def empty(name, loc):
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=loc)
    o = bpy.context.object
    o.name = name
    return o


def spin(obj, axis_index=2, frames=60, sign=1.0):
    """绕局部轴匀速旋转的关键帧 + 循环修改器（导出为 glTF 动画）。"""
    obj.rotation_mode = 'XYZ'
    obj.rotation_euler[axis_index] = 0.0
    obj.keyframe_insert(data_path="rotation_euler", index=axis_index, frame=1)
    obj.rotation_euler[axis_index] = sign * 2.0 * math.pi
    obj.keyframe_insert(data_path="rotation_euler", index=axis_index, frame=frames + 1)
    ad = obj.animation_data
    if ad and ad.action:
        for fc in ad.action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = 'LINEAR'
            try:
                fc.modifiers.new('CYCLES')
            except Exception:
                pass


def export(name):
    path = os.path.join(OUT_DIR, name + ".glb")
    bpy.ops.export_scene.gltf(
        filepath=path,
        export_format='GLB',
        export_apply=True,
        export_animations=True,
        export_yup=True,
    )
    print("EXPORTED", path, os.path.getsize(path), flush=True)


# ---------------------------------------------------------------------------
# 预警机（AEW）：机身 + 大圆盘雷达（旋转动画）
# ---------------------------------------------------------------------------
def build_awacs():
    glass = reset()
    cyl("fuselage", (0, 0, 2.2), 2.2, 32, GRAY, rot=(math.radians(90), 0, 0), verts=20)
    sphere("nose", (0, 16.5, 2.2), (4.4, 4.4, 3.6), GRAY)
    cyl("tailcone", (0, -18.5, 2.4), 1.4, 6, GRAY, rot=(math.radians(90), 0, 0))
    cube("wing", (0, 1.5, 1.4), (40, 7.0, 0.55), GRAY)
    cube("stabilizer", (0, -19.0, 2.6), (14, 4.0, 0.4), GRAY)
    cube("fin", (0, -19.5, 5.6), (0.45, 5.0, 6.0), GRAY)
    for x in (-7.5, 7.5, -13.5, 13.5):
        cyl("engine", (x, 3.0, 0.2), 1.0, 5.0, DARK, rot=(math.radians(90), 0, 0))
    # 旋转雷达罩
    rot = empty("Radar", (0, 0, 4.6))
    dish = cyl("radome", (0, 0, 4.6), 3.9, 1.1, DARK, verts=24)
    dish.parent = rot
    spin(rot, axis_index=2, frames=90)
    export("awacs")


# ---------------------------------------------------------------------------
# 武装直升机：主旋翼 + 尾旋翼动画
# ---------------------------------------------------------------------------
def build_helicopter():
    glass = reset()
    cube("fuselage", (0, 0.4, 1.7), (1.8, 8.0, 1.8), GREEN)
    sphere("nose", (0, 4.6, 1.6), (1.8, 2.6, 1.6), GREEN)
    cube("canopy", (0, 3.1, 2.25), (1.35, 1.8, 0.7), glass)
    cyl("boom", (0, -6.4, 1.9), 0.26, 5.6, GREEN, rot=(math.radians(90), 0, 0))
    cube("fin", (0, -8.9, 2.4), (0.18, 1.4, 1.6), GREEN)
    # 武器的短翼与挂架
    cube("stubwing", (0, 0.6, 1.2), (4.6, 0.8, 0.25), DARK)
    for x in (-2.0, 2.0):
        cyl("pylon", (x, 0.6, 1.0), 0.28, 0.9, DARK, rot=(math.radians(90), 0, 0))
        cyl("rocketpod", (x, 0.6, 0.55), 0.34, 1.6, DARK, rot=(math.radians(90), 0, 0))
    # 主旋翼
    hub = empty("MainRotor", (0, 0.2, 3.35))
    cyl("rotorhub", (0, 0.2, 3.3), 0.28, 0.5, DARK)
    for ang in (0, 90, 180, 270):
        bl = cube("blade", (0, 0.2, 3.35), (0.30, 13.0, 0.06), DARK,
                  rot=(0, 0, math.radians(ang)))
        bl.parent = hub
        bl.matrix_parent_inverse = hub.matrix_world.inverted()
    spin(hub, axis_index=2, frames=36)
    # 尾旋翼
    tail = empty("TailRotor", (0.25, -8.9, 2.4))
    for ang in (0, 90):
        bl = cube("tailblade", (0.25, -8.9, 2.4), (0.05, 2.4, 0.16), DARK,
                  rot=(math.radians(ang), 0, 0))
        bl.parent = tail
        bl.matrix_parent_inverse = tail.matrix_world.inverted()
    spin(tail, axis_index=0, frames=24)
    # 起落橇
    for x in (-0.9, 0.9):
        cyl("skid", (x, 0.3, 0.12), 0.08, 5.2, DARK, rot=(math.radians(90), 0, 0))
        for y in (-1.8, 1.8):
            cube("strut", (x, y, 0.7), (0.08, 0.08, 1.3), DARK)
    export("helicopter")


# ---------------------------------------------------------------------------
# 轮式装甲车（APC）
# ---------------------------------------------------------------------------
def build_apc():
    reset()
    cube("hull", (0, 0, 1.45), (3.0, 7.0, 1.5), GREEN)
    cube("glacis", (0, 3.35, 1.45), (2.9, 1.2, 1.4), GREEN, rot=(math.radians(-25), 0, 0))
    cube("turret", (0, 0.8, 2.55), (1.7, 2.2, 0.9), GREEN)
    cyl("gun", (0, 2.6, 2.55), 0.11, 3.2, DARK, rot=(math.radians(90), 0, 0))
    for x in (-1.55, 1.55):
        for y in (-2.4, -1.3, 1.3, 2.4):
            cyl("wheel", (x, y, 0.72), 0.72, 0.42, DARK,
                rot=(0, math.radians(90), 0), verts=14)
    for x in (-0.7, 0.7):
        cube("hatch", (x, -2.0, 2.24), (0.7, 0.7, 0.08), DARK)
    export("apc")


# ---------------------------------------------------------------------------
# 自行火炮（履带底盘 + 长身管）
# ---------------------------------------------------------------------------
def build_artillery():
    reset()
    cube("hull", (0, 0, 1.0), (3.2, 7.6, 1.1), GREEN)
    for x in (-1.55, 1.55):
        cube("track", (x, 0, 0.55), (0.95, 7.8, 1.05), DARK)
        for y in (-3.0, -1.0, 1.0, 3.0):
            cyl("roadwheel", (x, y, 0.5), 0.42, 1.0, GRAY,
                rot=(0, math.radians(90), 0), verts=12)
    cube("turret", (0, -0.6, 2.0), (2.8, 3.4, 1.3), GREEN)
    # 身管沿 +Y 扬起 18°：轴向 (0, cos p, sin p)。圆柱默认轴 +Z，绕 X 转 p-90°
    # 才能得到 +Y 方向（转 90-p 会指向 -Y，实测上一次就是这么把炮管甩到车后的）。
    p = math.radians(18)
    d = (0.0, math.cos(p), math.sin(p))
    start = (0.0, 0.9, 2.3)          # 炮塔前缘
    L = 7.2
    cyl("barrel", (start[0] + d[0] * L / 2,
                   start[1] + d[1] * L / 2,
                   start[2] + d[2] * L / 2), 0.17, L, DARK,
        rot=(p - math.pi / 2, 0, 0), verts=12)
    tip = (start[0] + d[0] * L, start[1] + d[1] * L, start[2] + d[2] * L)
    cyl("muzzle", (tip[0], tip[1], tip[2]), 0.24, 0.7, DARK,
        rot=(p - math.pi / 2, 0, 0), verts=12)
    export("artillery")


# ---------------------------------------------------------------------------
# 防空导弹发射车（轮式 TEL，四联装倾斜发射筒）
# ---------------------------------------------------------------------------
def build_sam():
    reset()
    cube("chassis", (0, 0, 0.95), (2.5, 10.5, 0.5), DARK)
    cube("cab", (0, 4.0, 1.9), (2.4, 2.4, 2.0), GREEN)
    cube("cabglass", (0, 5.25, 2.3), (2.1, 0.15, 0.8), mat("glass", (0.08, 0.14, 0.18), 0.2, 0.6))
    cube("bed", (0, -2.2, 1.2), (2.4, 5.6, 0.3), GRAY)
    for x in (-1.15, 1.15):
        for y in (3.0, -1.6, -3.0):
            cyl("wheel", (x, y, 0.75), 0.75, 0.5, DARK,
                rot=(0, math.radians(90), 0), verts=14)
    # 发射架：底座 + 四根仰起的发射筒（45°）
    cube("mount", (0, -2.2, 1.7), (2.2, 3.2, 0.6), GREEN)
    for dx in (-0.62, 0.62):
        for dz in (0.0, 0.8):
            cyl("canister", (dx, -1.6, 2.9 + dz), 0.28, 4.6, GREEN,
                rot=(math.radians(45), 0, 0), verts=12)
    cube("radar", (-1.0, 2.4, 3.3), (0.8, 0.2, 1.2), DARK)
    export("sam_launcher")


# ---------------------------------------------------------------------------
# 岸舰导弹发射车（两联装大筒，水平运输/发射）
# ---------------------------------------------------------------------------
def build_coastal_missile():
    reset()
    cube("chassis", (0, 0, 0.95), (2.6, 10.0, 0.5), DARK)
    cube("cab", (0, 3.8, 1.95), (2.4, 2.2, 2.0), GREEN)
    cube("bed", (0, -2.6, 1.2), (2.4, 5.4, 0.3), GRAY)
    for x in (-1.2, 1.2):
        for y in (3.0, -1.8, -3.4):
            cyl("wheel", (x, y, 0.75), 0.75, 0.5, DARK,
                rot=(0, math.radians(90), 0), verts=14)
    for dx in (-0.72, 0.72):
        cyl("canister", (dx, -2.2, 2.5), 0.48, 6.4, GREEN,
            rot=(math.radians(90), 0, 0), verts=14)
        cyl("nosecone", (dx, 1.1, 2.5), 0.48, 0.7, DARK,
            rot=(math.radians(90), 0, 0), verts=14)
    cube("mount", (0, -2.2, 1.8), (2.2, 2.2, 0.6), GREEN)
    export("coastal_missile")


# ---------------------------------------------------------------------------
# CC0：MiG-15（.blend → glb）
# ---------------------------------------------------------------------------
def build_fighter():
    if not MIG_BLEND or not os.path.isfile(MIG_BLEND):
        print("SKIP fighter: blend not found", MIG_BLEND, flush=True)
        return
    reset()
    bpy.ops.wm.open_mainfile(filepath=MIG_BLEND)
    # 统一材质粗糙度，避免旧版 Principled 参数在新渲染器里过亮
    bpy.ops.export_scene.gltf(
        filepath=os.path.join(OUT_DIR, "fighter.glb"),
        export_format='GLB',
        export_apply=True,
        export_yup=True,
    )
    print("EXPORTED fighter", os.path.getsize(os.path.join(OUT_DIR, "fighter.glb")),
          flush=True)


if __name__ == "__main__":
    build_fighter()
    build_awacs()
    build_helicopter()
    build_apc()
    build_artillery()
    build_sam()
    build_coastal_missile()
    print("MODELGEN DONE", OUT_DIR)
