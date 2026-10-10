#!/usr/bin/env python3
"""离线资产打包：清单 + SHA-256 + 包版本号（C1 步骤 1/2）。

为什么需要它
------------
报告 §5.4 指出「资产与依赖是最大可复现性风险」：这些路径被 `.gitignore` 忽略，
**不在版本控制内**，换机/清空后无法凭仓库恢复：
  - `web/tiles/`、`web/vendor/`、`web/data/situations/`、
    `engine/data/roads/`、`engine/data/vector/`、`engine/data/intent_model.json`
故需要一份**带校验和的清单** + 恢复脚本。

实测规模（2026-10-10，远程 situation3d）
--------------------------------------
| 类别 | 文件数 | 体积 |
| --- | --- | --- |
| tiles | 76,946 | **1384.83 MB** |
| vendor | 396 | 22.46 MB |
| situations | 2 | 53.62 MB |
| roads | 2 | 24.92 MB |
| vector | 3 | 0.43 MB |
| intent_model | 1 | 0.05 MB |
| **合计** | **77,350** | **1486.3 MB** |

⚠ **`.gitignore` 里写的是「~10k files, ~200 MB」——实测已是 7 倍。** 注释过期，已记录待改。

设计要点
--------
1. **清单只含相对路径**，凭 `<root>` 解析 —— 包可在任意目录还原，不绑定机器路径；
2. **版本号由内容决定**：`package_version = sha256(所有 (相对路径, 字节数, sha256) 的规范序列)[:16]`。
   任何文件增删改都会改变版本号，**不依赖人工维护**；
3. `intent_model.json` 是**可选**产物（C1 步骤 4），缺失时**不参与版本号**，
   只在清单里标 `optional: true`——否则「缺模型」会把整个包判成不同版本；
4. 输出 JSON 清单（机器读）+ CSV 清单（人读/可 diff）。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

DEFAULT_CATEGORIES: List[Tuple[str, str, bool]] = [
    # (类别名, 相对路径, 是否可选)
    ("tiles", "web/tiles", False),
    ("vendor", "web/vendor", False),
    ("situations", "web/data/situations", False),
    ("roads", "engine/data/roads", False),
    ("vector", "engine/data/vector", False),
    ("intent_model", "engine/data/intent_model.json", True),
]

MANIFEST_SCHEMA = "shenji.offline_assets/1"
CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def iter_category(root: Path, rel: str) -> Iterable[Path]:
    p = root / rel
    if not p.exists():
        return
    if p.is_file():
        yield p
        return
    for f in sorted(p.rglob("*")):
        if f.is_file():
            yield f


def build_manifest(root: Path, categories: Optional[List[Tuple[str, str, bool]]] = None,
                   *, verbose: bool = False) -> Dict[str, Any]:
    cats = categories or DEFAULT_CATEGORIES
    entries: List[Dict[str, Any]] = []
    per_cat: Dict[str, Dict[str, Any]] = {}
    for name, rel, optional in cats:
        files = list(iter_category(root, rel))
        cat_bytes = 0
        for f in files:
            size = f.stat().st_size
            digest = sha256_file(f)
            entries.append({
                "category": name,
                "path": f.relative_to(root).as_posix(),
                "bytes": size,
                "sha256": digest,
                "optional": optional,
            })
            cat_bytes += size
        per_cat[name] = {"files": len(files), "bytes": cat_bytes, "optional": optional}
        if verbose:
            print("  %-14s %7d 文件  %12d B" % (name, len(files), cat_bytes), file=sys.stderr)

    entries.sort(key=lambda e: e["path"])
    # 版本号只由**必需**条目决定：可选产物缺失不应产生新版本
    required = [(e["path"], e["bytes"], e["sha256"]) for e in entries if not e["optional"]]
    h = hashlib.sha256()
    for path, size, digest in required:
        h.update(("%s\0%d\0%s\n" % (path, size, digest)).encode("utf-8"))
    version = h.hexdigest()[:16]

    return {
        "schema": MANIFEST_SCHEMA,
        "package_version": version,
        "root": str(root),
        "categories": per_cat,
        "totals": {
            "files": len(entries),
            "bytes": sum(e["bytes"] for e in entries),
            "required_files": len(required),
        },
        "entries": entries,
    }


def write_manifest(manifest: Dict[str, Any], out_json: Path, out_csv: Optional[Path] = None) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["category", "path", "bytes", "sha256", "optional"])
            for e in manifest["entries"]:
                w.writerow([e["category"], e["path"], e["bytes"], e["sha256"], int(bool(e["optional"]))])


def load_manifest(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Package offline assets into a manifest with checksums (C1).")
    ap.add_argument("--root", default=".", help="situation3d root")
    ap.add_argument("--out", default="dist/offline_assets_manifest.json")
    ap.add_argument("--csv", default="dist/offline_assets_manifest.csv")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    if not (root / "web").exists():
        print("not a situation3d root (no web/): %s" % root, file=sys.stderr)
        return 2
    m = build_manifest(root, verbose=args.verbose)
    write_manifest(m, Path(args.out), Path(args.csv) if args.csv else None)

    if args.json:
        print(json.dumps({k: v for k, v in m.items() if k != "entries"},
                         ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print("package_version: %s" % m["package_version"])
        for name, c in m["categories"].items():
            flag = " (optional)" if c["optional"] else ""
            print("  %-14s %7d files %14d B%s" % (name, c["files"], c["bytes"], flag))
        t = m["totals"]
        print("  %-14s %7d files %14d B (%.1f MB)" % ("TOTAL", t["files"], t["bytes"], t["bytes"] / 1024 / 1024))
        print("  required_files (参与版本号): %d" % t["required_files"])
        print("manifest -> %s" % args.out)
        if args.csv:
            print("csv      -> %s" % args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
