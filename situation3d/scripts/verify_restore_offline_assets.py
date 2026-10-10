#!/usr/bin/env python3
"""离线资产校验与一键恢复（C1 步骤 3）。

与打包脚本**分离**（C1【风险与回滚】要求「校验脚本与恢复脚本分离，便于单独停用」）。

三种模式
--------
  --mode verify    只校验目标路径现有资产与清单是否逐项一致（默认，**不写盘**）
  --mode plan      仅打印将要做什么（不读哈希、不写盘，秒级）
  --mode restore   从 `--source` 复制到 `--target`，**逐项比对 SHA-256**

`--source` 未给且目标缺失时，报明确错误并列出缺哪些项 —— 而不是静默跳过。

为什么要 `--source`
-------------------
资产不入库（`.gitignore`），故「恢复」必然是「从某份已有的拷贝复制」：
可能是同机的另一份检出、外部盘、或从归档解出的目录。脚本不假设来源，
只要求来源里的文件哈希与清单一致。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest(path: Path) -> Dict[str, Any]:
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    if "entries" not in m or "package_version" not in m:
        raise ValueError("not a valid offline-assets manifest: %s" % path)
    return m


def classify(target: Path, manifest: Dict[str, Any], *, want_progress: bool = False) -> Dict[str, List[Dict[str, Any]]]:
    """把每个条目分为 ok / missing / size_mismatch / hash_mismatch。"""
    out: Dict[str, List[Dict[str, Any]]] = {"ok": [], "missing": [], "size_mismatch": [], "hash_mismatch": []}
    entries = manifest["entries"]
    n = len(entries)
    for i, e in enumerate(entries, 1):
        p = target / e["path"]
        if not p.exists():
            out["missing"].append(e)
        elif p.stat().st_size != e["bytes"]:
            out["size_mismatch"].append(e)
        elif sha256_file(p) != e["sha256"]:
            out["hash_mismatch"].append(e)
        else:
            out["ok"].append(e)
        if want_progress and i % 2000 == 0:
            print("  校验中 %d/%d ..." % (i, n), file=sys.stderr)
    return out


def do_restore(target: Path, source: Path, manifest: Dict[str, Any], bad: List[Dict[str, Any]],
               *, dry_run: bool = False) -> Dict[str, Any]:
    """从 source 复制 bad 中列出的条目到 target，并逐项复核哈希。"""
    stats = {"copied": 0, "repaired": 0, "still_bad": [], "source_missing": []}
    for e in bad:
        src = source / e["path"]
        dst = target / e["path"]
        if not src.exists():
            stats["source_missing"].append(e["path"])
            continue
        if sha256_file(src) != e["sha256"]:
            stats["source_missing"].append(e["path"] + " (source hash mismatch)")
            continue
        if dry_run:
            stats["copied"] += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if sha256_file(dst) == e["sha256"]:
            stats["copied"] += 1
            stats["repaired"] += 1
        else:
            stats["still_bad"].append(e["path"])
    return stats


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Verify or restore offline assets from a manifest (C1).")
    ap.add_argument("--manifest", default="dist/offline_assets_manifest.json")
    ap.add_argument("--target", default=".", help="situation3d root to verify/restore into")
    ap.add_argument("--source", default=None, help="root to copy missing assets from (restore mode)")
    ap.add_argument("--mode", choices=["verify", "plan", "restore"], default="verify")
    ap.add_argument("--allow-missing-optional", action="store_true", default=True,
                    help="optional entries (e.g. intent_model.json) missing is not a failure")
    ap.add_argument("--strict-optional", dest="allow_missing_optional", action="store_false")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    target = Path(args.target).resolve()
    manifest = load_manifest(Path(args.manifest))
    print("manifest: %s" % args.manifest)
    print("package_version: %s" % manifest["package_version"])
    print("target: %s" % target)
    print("entries: %d (%d required, %d optional)"
          % (manifest["totals"]["files"], manifest["totals"]["required_files"],
             manifest["totals"]["files"] - manifest["totals"]["required_files"]))

    if args.mode == "plan":
        n_missing = sum(1 for e in manifest["entries"] if not (target / e["path"]).exists())
        print("\n[plan] 将检查 %d 项；按当前目录状态，缺失 %d 项。" % (len(manifest["entries"]), n_missing))
        print("[plan] 不读哈希、不写盘。")
        return 0

    cls = classify(target, manifest, want_progress=not args.json)
    print("\n=== 校验结果 ===")
    for k in ("ok", "missing", "size_mismatch", "hash_mismatch"):
        print("  %-14s %d" % (k, len(cls[k])))

    # 可选条目单独处理：允许缺失，但不允许「存在却哈希错」
    opt_missing = [e for e in cls["missing"] if e.get("optional")]
    req_missing = [e for e in cls["missing"] if not e.get("optional")]
    if opt_missing:
        print("  可选条目缺失（不作为失败）: %d" % len(opt_missing))
        for e in opt_missing[:5]:
            print("    - %s" % e["path"])
    if req_missing and not args.allow_missing_optional:
        pass

    bad = req_missing + cls["size_mismatch"] + cls["hash_mismatch"]
    bad += [e for e in opt_missing if args.strict_optional] if not args.allow_missing_optional else []

    if not bad:
        print("\nRESULT: OK（必需项全部一致）")
        if args.json:
            print(json.dumps({"ok": True, "package_version": manifest["package_version"],
                              "counts": {k: len(v) for k, v in cls.items()}}, ensure_ascii=False, indent=2))
        return 0

    print("\n不合规项 %d：" % len(bad))
    for e in bad[:20]:
        print("    - %s" % e["path"])
    if len(bad) > 20:
        print("    ... 其余 %d 项" % (len(bad) - 20))

    if args.mode == "verify":
        print("\nRESULT: FAIL（用 --mode restore --source <已有拷贝> 修复）")
        return 1

    # restore
    if not args.source:
        print("\nrestore 需要 --source <包含这些资产的目录>", file=sys.stderr)
        return 2
    source = Path(args.source).resolve()
    if not source.exists():
        print("source not found: %s" % source, file=sys.stderr)
        return 2
    print("\n=== 恢复（从 %s） ===" % source)
    st = do_restore(target, source, manifest, bad)
    print("  复制=%d 修复=%d 源缺失=%d 复制后仍不一致=%d"
          % (st["copied"], st["repaired"], len(st["source_missing"]), len(st["still_bad"])))
    for p in st["source_missing"][:10]:
        print("    源缺失: %s" % p)

    # 恢复后**重新完整校验**，而不是相信刚才的复制
    cls2 = classify(target, manifest)
    bad2 = [e for e in (cls2["missing"] + cls2["size_mismatch"] + cls2["hash_mismatch"])
            if not e.get("optional")]
    print("\n=== 恢复后复核 ===")
    for k in ("ok", "missing", "size_mismatch", "hash_mismatch"):
        print("  %-14s %d" % (k, len(cls2[k])))
    ok = not bad2
    print("\nRESULT: %s" % ("OK（哈希逐项一致）" if ok else "FAIL（仍有 %d 项不一致）" % len(bad2)))
    if args.json:
        print(json.dumps({"ok": ok, "package_version": manifest["package_version"],
                          "counts": {k: len(v) for k, v in cls2.items()},
                          "restore": st}, ensure_ascii=False, indent=2, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
