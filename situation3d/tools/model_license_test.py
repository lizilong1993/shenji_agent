#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 CREDITS.md 生成/校验 licenses.json。

用法：
  python3 tools/model_license_test.py --write
  python3 tools/model_license_test.py
"""
import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "web", "assets", "models")
CREDITS = os.path.join(MODELS, "CREDITS.md")
OUT = os.path.join(MODELS, "licenses.json")

ROW = re.compile(r"^\|\s*`([^`]+\.glb)`\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    assets = {}
    for line in open(CREDITS, encoding="utf-8"):
        m = ROW.match(line.strip())
        if not m:
            continue
        name, source, author, license, use = m.groups()
        risk = []
        text = (source + " " + author + " " + license).lower()
        if "dcs world" in text:
            risk.append("upstream-game-asset")
        if "free standard" in text:
            risk.append("sketchfab-free-standard-redistribution")
        if "not my model" in text or "来源链待核" in text:
            risk.append("provenance-unclear")
        murl = re.search(r"\((https?://[^)]+)\)", source)
        assets[name] = {
            "source": source,
            "author": author,
            "license": license,
            "use": use,
            "risk": risk,
            "source_url": murl.group(1) if murl else None,
        }

    specs = json.load(open(os.path.join(MODELS, "specs.json"), encoding="utf-8"))["assets"]
    missing = sorted(set(specs) - set(assets))
    extra = sorted(set(assets) - set(specs))
    if missing:
        print("missing license rows:", missing, file=sys.stderr)
        return 1
    if extra:
        print("note: extra license rows (not in specs):", extra)

    risks = {k: v["risk"] for k, v in assets.items() if v["risk"]}
    print("licensed assets:", len(assets))
    print("risk assets:", len(risks))
    for k, v in sorted(risks.items()):
        print("  %-22s %s" % (k, ",".join(v)))

    if args.write:
        open(OUT, "w", encoding="utf-8").write(json.dumps({
            "version": 1,
            "note": "由 tools/model_license_test.py 从 CREDITS.md 生成；risk 仅提示，不阻断入库。",
            "assets": assets,
        }, ensure_ascii=False, indent=2) + "\n")
        print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
