#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""目标模型预算校验与 manifest 生成。

用法：
  python3 tools/model_budget_test.py                 # 只校验
  python3 tools/model_budget_test.py --write-manifest # 校验并写 manifest.json

预算按 specs.json 的 class 取值；tris/体积允许 TOLERANCE 的余量，primitive/
材质按硬上限。带关节模型（rigged）暂时给较宽预算，P2 再收紧。
"""
import argparse
import datetime
import glob
import hashlib
import json
import os
import struct
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "web", "assets", "models")
SPECS = os.path.join(MODELS, "specs.json")
LICENSES = os.path.join(MODELS, "licenses.json")
MANIFEST = os.path.join(MODELS, "manifest.json")
TOLERANCE = 0.08

P0_BUDGETS = {
    # class: (tris, prims, materials, file_mb)
    # P0 验收预算：按 2026-10-26 优化后的可达值 + 余量；P1 重建后应满足
    # P1_TARGETS（与报告 §5.1 一致）。默认用 P0，避免把“P0 未达标”误报为回归。
    "ship_large": (35000, 14, 14, 1.70),
    "ship_small": (14000, 10, 8, 0.75),
    "aircraft_fighter": (21000, 18, 18, 1.60),
    "aircraft_aew": (24000, 10, 9, 1.25),
    "ground_tank": (15000, 18, 18, 1.30),
    "ground_truck": (12000, 8, 8, 0.60),
    "rigged": (31000, 20, 12, 2.20),
}

P1_TARGETS = {
    # 报告的 P1 目标；`--strict` 时使用。
    "ship_large": (28000, 12, 6, 1.70),
    "ship_small": (14000, 10, 4, 0.75),
    "aircraft_fighter": (18000, 12, 6, 0.95),
    "aircraft_aew": (22000, 12, 6, 1.25),
    "ground_tank": (9000, 6, 3, 0.33),
    "ground_truck": (7000, 8, 3, 0.28),
    "rigged": (30000, 15, 10, 2.20),
}

EXPECTED_MAIN_ASSETS = 39

ALLOWED_EXT = {
    "EXT_texture_webp",
    "EXT_meshopt_compression",
    "KHR_mesh_quantization",
    "KHR_materials_ior",
    "KHR_materials_specular",
    "KHR_materials_clearcoat",
    "KHR_materials_transmission",
    "KHR_materials_unlit",
}


def glb_info(path):
    data = open(path, "rb").read()
    magic, version, length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF":
        raise ValueError("not a GLB: " + path)
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
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "mb": round(len(data) / 1048576, 2),
        "tris": tris,
        "prims": prims,
        "materials": len(g.get("materials", [])),
        "images": len(g.get("images", [])),
        "extensions": g.get("extensionsUsed", []),
        "animations": len(g.get("animations", [])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-manifest", action="store_true")
    ap.add_argument("--strict", action="store_true", help="用 P1_TARGETS（报告 §5.1）而不是 P0 验收预算")
    args = ap.parse_args()
    budgets = P1_TARGETS if args.strict else P0_BUDGETS

    specs = json.load(open(SPECS, encoding="utf-8"))["assets"]
    licenses = {}
    if os.path.exists(LICENSES):
        licenses = json.load(open(LICENSES, encoding="utf-8")).get("assets", {})

    failures = []
    rows = []
    manifest_assets = {}
    files = sorted(glob.glob(os.path.join(MODELS, "type_*.glb")))
    for path in files:
        name = os.path.basename(path)
        if name.endswith("_gunfx.glb") or name.endswith("_launcherfx.glb"):
            continue
        meta = specs.get(name)
        if not meta:
            failures.append((name, "specs.json 缺少条目"))
            continue
        cls = meta["class"]
        if cls not in budgets:
            failures.append((name, "未知 class " + cls))
            continue
        bt, bp, bm, bmb = budgets[cls]
        info = glb_info(path)
        bad = []
        if info["tris"] > bt * (1 + TOLERANCE):
            bad.append("tris %d>%d" % (info["tris"], bt))
        if info["prims"] > bp:
            bad.append("prims %d>%d" % (info["prims"], bp))
        if info["materials"] > bm:
            bad.append("mats %d>%d" % (info["materials"], bm))
        if info["mb"] > bmb * (1 + TOLERANCE):
            bad.append("mb %.2f>%.2f" % (info["mb"], bmb))
        unknown = [e for e in info["extensions"] if e not in ALLOWED_EXT]
        if unknown:
            bad.append("ext " + ",".join(unknown))
        rows.append((name, cls, info["tris"], bt, info["prims"], bp, info["mb"], bmb, "OK" if not bad else ";".join(bad)))
        if bad:
            failures.append((name, ";".join(bad)))
        manifest_assets[name] = dict(
            info,
            **{"class": cls, "public": {k: meta.get(k) for k in ("public_length_m", "public_width_m", "approx_of", "notes") if meta.get(k) is not None},
               "license": licenses.get(name, {})},
        )

    if len(rows) != EXPECTED_MAIN_ASSETS:
        failures.append(("__count__", "主模型 %d 个，期望 %d 个" % (len(rows), EXPECTED_MAIN_ASSETS)))

    print("budget: %s" % ("P1_TARGETS" if args.strict else "P0_BUDGETS"))
    print("%-22s %-16s %8s %8s %6s %4s %7s %6s  %s" % ("asset", "class", "tris", "budget", "prims", "max", "MB", "budget", "status"))
    for r in rows:
        print("%-22s %-16s %8d %8d %6d %4d %7.2f %6.2f  %s" % r)

    if args.write_manifest:
        out = {
            "version": 2,
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "note": "由 tools/model_budget_test.py 生成；sha256 对应优化后的 GLB。",
            "assets": manifest_assets,
        }
        open(MANIFEST, "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
        print("wrote", MANIFEST)

    if failures:
        print("\nFAIL %d asset(s):" % len(failures), file=sys.stderr)
        for name, why in failures:
            print("  %s: %s" % (name, why), file=sys.stderr)
        return 1
    print("\nALL PASS (%d assets)" % len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
