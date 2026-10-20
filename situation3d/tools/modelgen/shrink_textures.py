# -*- coding: utf-8 -*-
"""贴图压缩：把已归一化 GLB 的贴图缩到 MAXTEX，并以 JPEG 重编码导出。

用法：blender -b --python shrink_textures.py -- IN.glb OUT.glb 1024
"""
import bpy
import os
import sys

argv = sys.argv[sys.argv.index("--") + 1:]
src, dst = argv[0], argv[1]
maxtex = int(argv[2]) if len(argv) > 2 else 1024

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
for im in bpy.data.images:
    w, h = im.size
    if max(w, h) > maxtex:
        f = maxtex / float(max(w, h))
        im.scale(max(1, int(w * f)), max(1, int(h * f)))
    # 无 alpha 的贴图转 JPEG
    try:
        if im.channels < 4:
            im.file_format = "JPEG"
    except Exception:
        pass

os.makedirs(os.path.dirname(dst), exist_ok=True)
bpy.ops.export_scene.gltf(
    filepath=dst, export_format="GLB", export_apply=True,
    export_animations=True, export_yup=True,
    export_image_format="JPEG", export_jpeg_quality=85,
)
print("SHRINK", os.path.basename(dst), os.path.getsize(dst), flush=True)
