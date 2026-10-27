# -*- coding: utf-8 -*-
"""合并共享贴图的材质与 primitive，压 draw calls（当前用于 ZTD-05 的 LOD0/LOD1）。

原理：ZTD-05 的 5 个 PaletteMaterial 共用同一张调色板贴图，只是 roughness/
metallic 因子不同；把其中 4 个指定给第一个材质、统一 TEXCOORD 类型后 join，
primitive 由 5 降到 2（保留玻璃 BLEND 材质）。外观在渲染对比中一致。

用法（在 situation3d 目录下）：
  python3 tools/modelgen/merge_materials.py              # 演练
  python3 tools/modelgen/merge_materials.py --apply      # 原地替换（先备份）
"""
import argparse
import datetime
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = os.path.join(ROOT, "web", "assets", "models")
NODE_BIN = "/remote-home/lizilong/tools/node-v22.14.0-linux-x64/bin"
NODE_GLOBAL = "/remote-home/lizilong/tools/node-global"
GLTF = os.path.join(NODE_GLOBAL, "bin", "gltf-transform")
BACKUP = "/remote-home/lizilong/model_backups/target-models-v3/merge-materials"

# 资产名 -> 要并入材质 0 的材质序号（逗号分隔）
CFG = {
    "type_ztd05": "1,2,4",
    "lod1_ztd05": "1,2,4",
}

REMAP_JS = r'''
const { NodeIO } = require("@gltf-transform/core");
const { ALL_EXTENSIONS } = require("@gltf-transform/extensions");
(async () => {
  const io = new NodeIO().registerExtensions(ALL_EXTENSIONS);
  const doc = await io.read(process.argv[2]);
  const root = doc.getRoot();
  const mats = root.listMaterials();
  const base = mats[0];
  const idxs = process.argv[4].split(",").map(Number);
  const merge = new Set(idxs.map((i) => mats[i]).filter(Boolean));
  const buffer = root.listBuffers()[0];
  let converted = 0;
  for (const mesh of root.listMeshes()) for (const p of mesh.listPrimitives()) {
    if (merge.has(p.getMaterial())) p.setMaterial(base);
    const uv = p.getAttribute("TEXCOORD_0");
    if (uv && uv.getComponentType() !== 5126) {
      const arr = uv.getArray();
      const f = new Float32Array(arr.length);
      const norm = uv.getNormalized ? uv.getNormalized() : false;
      for (let i = 0; i < arr.length; i++) f[i] = norm ? arr[i] / 65535 : arr[i];
      const acc = doc.createAccessor().setType("VEC2").setArray(f).setBuffer(buffer);
      p.setAttribute("TEXCOORD_0", acc);
      converted++;
    }
  }
  for (const m of merge) m.dispose();
  await io.write(process.argv[3], doc);
  console.log("remapped; uv converted", converted);
})().catch((e) => { console.error(e); process.exit(1); });
'''


def run(cmd):
    env = os.environ.copy()
    env["PATH"] = NODE_BIN + ":" + os.path.dirname(GLTF) + ":" + env.get("PATH", "")
    env["NODE_PATH"] = os.path.join(NODE_GLOBAL, "lib", "node_modules")
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if p.returncode != 0:
        raise RuntimeError("command failed: %s\n%s\n%s" % (
            " ".join(cmd), p.stdout[-1500:], p.stderr[-1500:]))
    return p.stdout


def stats(path):
    data = open(path, "rb").read()
    off = 12
    g = None
    while off < len(data):
        clen, ctype = struct.unpack_from("<II", data, off)
        off += 8
        if ctype == 0x4E4F534A:
            g = json.loads(data[off:off + clen].decode("utf-8"))
        off += clen
    tris = prims = 0
    for mesh in g.get("meshes", []):
        for p in mesh.get("primitives", []):
            prims += 1
            if "indices" in p:
                tris += g["accessors"][p["indices"]]["count"] // 3
            elif "POSITION" in p.get("attributes", {}):
                tris += g["accessors"][p["attributes"]["POSITION"]]["count"] // 3
    return {"bytes": len(data), "tris": tris, "prims": prims,
            "materials": len(g.get("materials", []))}


def process(name, idxs, tmp, apply):
    src = os.path.join(MODELS, name + ".glb")
    work = os.path.join(tmp, name + ".glb")
    shutil.copyfile(src, work)
    before = stats(work)
    run([GLTF, "copy", work, work + ".decoded.glb"])
    js = os.path.join(tmp, "remap.js")
    if not os.path.exists(js):
        open(js, "w", encoding="utf-8").write(REMAP_JS)
    run([os.path.join(NODE_BIN, "node"), js, work + ".decoded.glb",
         work + ".remapped.glb", idxs])
    run([GLTF, "join", work + ".remapped.glb", work + ".joined.glb"])
    run([GLTF, "webp", work + ".joined.glb", work + ".webp.glb", "--quality", "85"])
    run([GLTF, "meshopt", work + ".webp.glb", work + ".final.glb"])
    run([GLTF, "validate", work + ".final.glb"])
    after = stats(work + ".final.glb")
    if apply:
        os.makedirs(BACKUP, exist_ok=True)
        bak = os.path.join(BACKUP, datetime.date.today().isoformat(), name + ".glb")
        os.makedirs(os.path.dirname(bak), exist_ok=True)
        if not os.path.exists(bak):
            shutil.copyfile(src, bak)
        shutil.copyfile(work + ".final.glb", src)
    return {"asset": name, "before": before, "after": after}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    names = [n for n in CFG if not args.only or n in args.only]
    tmp = tempfile.mkdtemp(prefix="merge-mat-")
    try:
        for n in names:
            try:
                r = process(n, CFG[n], tmp, args.apply)
            except Exception as e:
                print("FAIL", n, e, file=sys.stderr)
                return 1
            print("%-12s %6d -> %6d tris   %2d -> %2d prims   %6.1f -> %6.1f KB%s" % (
                n, r["before"]["tris"], r["after"]["tris"],
                r["before"]["prims"], r["after"]["prims"],
                r["before"]["bytes"] / 1024.0, r["after"]["bytes"] / 1024.0,
                "" if args.apply else "  (dry-run)"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
