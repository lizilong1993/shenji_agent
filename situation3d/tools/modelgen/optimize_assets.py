#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P0 目标模型优化：纹理压缩、静态件合批、按预算减面、meshopt。

用法（在 situation3d 目录下）：
  python3 tools/modelgen/optimize_assets.py --dry-run
  python3 tools/modelgen/optimize_assets.py --apply [--only type_052c ...]

约定：
- 原文件先备份到 /remote-home/lizilong/model_backups/target-models-v3/<date>/
  （可用 --backup 覆盖），再原地替换；
- 带关节的模型（Z-10/HQ-16/YJ-62/M1A2/HQ-9）不做 join/simplify/meshopt，
  只做纹理压缩，避免破坏 targetFx/targetCombatFx 依赖的节点名；
- 其余静态模型 join 合批 + 按预算 simplify + WebP + meshopt；
- 每步用 gltf-transform validate 校验，失败即中止。
"""
import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = os.path.join(ROOT, "web", "assets", "models")
NODE_BIN = "/remote-home/lizilong/tools/node-v22.14.0-linux-x64/bin"
GLTF = "/remote-home/lizilong/tools/node-global/bin/gltf-transform"
DEFAULT_BACKUP = "/remote-home/lizilong/model_backups/target-models-v3"

# class, target L0 triangles, join / simplify / meshopt
CFG = {
    # 水面舰：大型 28k，小型/辅助 12k
    "type_052c": ("ship_large", 28000, True, True, True),
    "type_054a": ("ship_large", 28000, True, True, True),
    "type_071": ("ship_large", 28000, True, True, True),
    "type_perry": ("ship_large", 28000, True, True, True),
    "type_903": ("ship_small", 12000, True, False, True),
    "type_kidd": ("ship_small", 12000, True, False, True),
    "type_panshi": ("ship_small", 12000, True, False, True),
    "type_056a": ("ship_small", 12000, True, False, True),
    "type_tuochiang": ("ship_small", 12000, True, False, True),
    # 飞机
    "type_e2d": ("aircraft_aew", 22000, True, True, True),
    "type_kj2000": ("aircraft_aew", 22000, True, True, True),
    "type_j16": ("aircraft_fighter", 18000, True, True, True),
    "type_f16c": ("aircraft_fighter", 18000, True, True, True),
    "type_idf": ("aircraft_fighter", 18000, True, True, True),
    "type_rafale": ("aircraft_fighter", 18000, True, True, True),
    "type_mq9": ("aircraft_fighter", 18000, True, True, True),
    # 地面
    "type_ztz99a": ("ground_tank", 9000, True, True, True),
    "type_m48a5": ("ground_tank", 9000, True, True, True),
    "type_m109a6": ("ground_tank", 9000, True, True, True),
    "type_plz05": ("ground_tank", 9000, True, True, True),
    "type_phl03": ("ground_truck", 11000, True, True, True),
    "type_pac3": ("ground_truck", 8000, True, True, True),
    "type_ztd05": ("ground_tank", 9000, True, True, True),
    "type_eq240": ("ground_truck", 7000, True, True, True),
    "type_m777": ("ground_truck", 8000, True, False, True),
    # 带关节：只做纹理压缩（保持节点名与几何）
    "type_z10": ("rigged", 0, False, False, False),
    "type_hq9": ("rigged", 0, False, False, False),
    "type_hq16": ("rigged", 0, False, False, False),
    "type_yj62": ("rigged", 0, False, False, False),
    "type_m1a2": ("rigged", 0, False, False, False),
}


def run(cmd):
    env = os.environ.copy()
    env["PATH"] = NODE_BIN + ":" + os.path.dirname(GLTF) + ":" + env.get("PATH", "")
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if p.returncode != 0:
        raise RuntimeError("command failed: %s\n%s\n%s" % (" ".join(cmd), p.stdout[-2000:], p.stderr[-2000:]))
    return p.stdout


def glb_stats(path):
    """Read GLB JSON chunk; count tris/prims/materials/textures."""
    import struct
    data = open(path, "rb").read()
    magic, version, length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF":
        raise ValueError(path)
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
    return {
        "file": os.path.basename(path),
        "size_kb": round(os.path.getsize(path) / 1024, 1),
        "tris": tris,
        "prims": prims,
        "materials": len(g.get("materials", [])),
        "images": len(g.get("images", [])),
        "extensions": g.get("extensionsUsed", []),
        "node_names": [n.get("name", "") for n in g.get("nodes", [])],
    }


def optimize_one(name, cfg, tmp, backup_dir, apply):
    src = os.path.join(MODELS, name + ".glb")
    before = glb_stats(src)
    cls, target, do_join, do_simplify, do_meshopt = cfg
    work = os.path.join(tmp, name + ".glb")
    shutil.copy2(src, work)

    steps = []

    # 1) 贴图统一上限 1024（已有更小的不会被放大）
    out = work + ".resize.glb"
    run([GLTF, "resize", work, out, "--width", "1024", "--height", "1024"])
    work = out
    steps.append("resize<=1024")

    # 2) 静态模型：先 palette 合并纯色材质，再 join 合批（会 flatten，带关节模型禁止）
    if do_join and before["prims"] > 4:
        out = work + ".palette.glb"
        run([GLTF, "palette", work, out])
        work = out
        steps.append("palette")
        out = work + ".join.glb"
        run([GLTF, "join", work, out])
        work = out
        steps.append("join")

    # 3) 按预算减面（join 后重新统计）。simplify 的 ratio 是顶点比例，
    #    与三角面不是线性关系，最多两轮逼近目标，避免一次过减。
    cur = glb_stats(work)
    if do_simplify and cur["tris"] > target * 1.02:
        for _pass in range(2):
            if cur["tris"] <= target * 1.12:
                break
            out = work + ".weld.glb"
            run([GLTF, "weld", work, out])
            work = out
            ratio = max(0.05, min(1.0, target / float(cur["tris"]) * (1.05 if _pass == 0 else 1.0)))
            out = work + ".simplify.glb"
            run([GLTF, "simplify", work, out, "--ratio", "%.4f" % ratio, "--error", "0.01"])
            work = out
            cur = glb_stats(work)
            steps.append("simplify(%.3f)" % ratio)

    # 4) WebP（保留 alpha）
    out = work + ".webp.glb"
    run([GLTF, "webp", work, out, "--quality", "85"])
    work = out
    steps.append("webp")

    # 5) meshopt（仅静态模型；带关节模型保持可编辑/可驱动）
    if do_meshopt:
        out = work + ".meshopt.glb"
        run([GLTF, "meshopt", work, out])
        work = out
        steps.append("meshopt")

    # 6) 校验
    run([GLTF, "validate", work])
    after = glb_stats(work)

    # 7) 包围盒与世界尺寸由 Blender 侧另行校验；这里只拒绝空模型
    if after["tris"] <= 0 or after["prims"] <= 0:
        raise RuntimeError("%s: empty output" % name)

    if apply:
        os.makedirs(backup_dir, exist_ok=True)
        bak = os.path.join(backup_dir, name + ".glb")
        if not os.path.exists(bak) or os.path.getsize(bak) != os.path.getsize(src):
            shutil.copyfile(src, bak)
        shutil.copyfile(work, src)

    return {
        "asset": name,
        "class": cls,
        "target_tris": target,
        "steps": steps,
        "before": {k: before[k] for k in ("size_kb", "tris", "prims", "materials", "images")},
        "after": {k: after[k] for k in ("size_kb", "tris", "prims", "materials", "images", "extensions")},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="替换原文件（默认只演练）")
    ap.add_argument("--only", nargs="*", help="只处理指定资产名（不含 .glb）")
    ap.add_argument("--backup", default=DEFAULT_BACKUP, help="备份目录")
    args = ap.parse_args()

    backup_dir = os.path.join(args.backup, datetime.date.today().isoformat())
    names = [n for n in CFG if not args.only or n in args.only]
    if not names:
        print("no matching assets", file=sys.stderr)
        return 2

    tmp = tempfile.mkdtemp(prefix="modelopt-")
    results = []
    try:
        for name in names:
            try:
                r = optimize_one(name, CFG[name], tmp, backup_dir, args.apply)
            except Exception as e:
                print("FAIL", name, e, file=sys.stderr)
                return 1
            results.append(r)
            b, a = r["before"], r["after"]
            print("%-18s %7.2f -> %6.2f MB  %7d -> %6d tris  %3d -> %2d prims  %s" % (
                name, b["size_kb"] / 1024, a["size_kb"] / 1024,
                b["tris"], a["tris"], b["prims"], a["prims"], ",".join(r["steps"])))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    report = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "applied": bool(args.apply),
        "assets": results,
    }
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
