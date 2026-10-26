#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 gltf-transform inspect 生成 bboxes.json（支持 meshopt/quantization）。

model_bboxes.py 是 stdlib 读原始 accessor 的实现，遇到 meshopt/draco 会跳过；
P0 之后 25 个静态模型是 meshopt，必须用本脚本。输出 schema 与旧文件一致。

用法：python3 tools/model_bboxes_gltf.py <situation3d 根目录>
"""
import csv
import json
import os
import subprocess
import sys

GLTF = "/remote-home/lizilong/tools/node-global/bin/gltf-transform"
NODE_BIN = "/remote-home/lizilong/tools/node-v22.14.0-linux-x64/bin"


def inspect_bbox(path):
    env = os.environ.copy()
    env["PATH"] = NODE_BIN + ":" + os.path.dirname(GLTF) + ":" + env.get("PATH", "")
    out = subprocess.check_output([GLTF, "inspect", path, "--format", "csv"],
                                  stderr=subprocess.DEVNULL, env=env, text=True)
    lines = out.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("#,name,rootName,bboxMin,bboxMax"):
            row = next(csv.reader([lines[i + 1]]))
            mn = [float(x) for x in row[3].split(",")]
            mx = [float(x) for x in row[4].split(",")]
            return mn, mx
    raise RuntimeError("no SCENES csv row in inspect output: " + path)


def main(root):
    models_dir = os.path.join(root, "web", "assets", "models")
    out_engine = os.path.join(root, "engine", "data", "models", "bboxes.json")
    out_web = os.path.join(models_dir, "bboxes.json")
    models = {}
    for fn in sorted(os.listdir(models_dir)):
        if not fn.startswith("type_") or not fn.endswith(".glb") or "_gunfx" in fn or "_launcherfx" in fn:
            continue
        mn, mx = inspect_bbox(os.path.join(models_dir, fn))
        dims = [mx[i] - mn[i] for i in range(3)]
        models[fn] = {
            "size_m": [round(dims[0], 2), round(dims[1], 2), round(dims[2], 2)],
            "length_m": round(max(dims[0], dims[2]), 2),
            "width_m": round(min(dims[0], dims[2]), 2),
            "height_m": round(dims[1], 2),
            "min_m": [round(x, 2) for x in mn],
            "max_m": [round(x, 2) for x in mx],
        }
        print(fn, models[fn]["length_m"], models[fn]["width_m"], models[fn]["height_m"])
    if len(models) != 30:
        print("expected 30 main models, got %d" % len(models), file=sys.stderr)
        return 1
    doc = {"schema": "situation3d.model_bboxes.v1", "models": models}
    for path in (out_engine, out_web):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
        print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
