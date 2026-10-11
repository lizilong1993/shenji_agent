#!/usr/bin/env python3
"""检查点与 manifest 一致性验证（I07 步骤 3）。

判据 2：`python3 scripts/verify_checkpoint.py --weights ai/weights` 退出码 0。

检查什么
--------
1. **manifest 齐全性**：六类字段（commit / 数据哈希 / 超参 / 依赖版本 / 环境 / experiment_id）；
2. **权重与声明一致**：
   - **哈希**：`environment.artifact_sha256` 里声明的 sha256 必须等于实算值（逐字节核对）；
   - **维度**：从权重形状反推的维度与 manifest 声明一致；
   - **文件名**：manifest 声明了文件名就**以声明为准**，缺失即失败；
3. **归一化器与模型输入匹配**：`feature_dim` == situation 模型输入维度；
4. 输出每个托管文件的 sha256 与差异明细。

退出码
------
`0` 通过；`1` 有失败项；`2` 用法错误。**任何畸形输入都不得抛未捕获异常** ——
`--json` 与退出码是被机器消费的，崩溃会让调用方拿不到 `checks/failed/warnings`。

历史产物怎么处理（【风险与回滚】要求「新字段全部可选」）
----------------------------------------------------
**逐字段**降级，不是整块降级：
  - 缺某个新字段 → 该字段记为「历史产物，未核验」并**判通过**（带标注）；
  - 缺 `environment.artifact_sha256` → 标注「未核验哈希」而不是失败；
  - 但**声明了却对不上** → 失败。降级与核对是两件事，不能混。

设计上的一条原则
----------------
加载器要「尽量能加载」，**校验器要「声明什么就必须是什么」**。
故别名回退只允许用于「manifest 根本没声明该键」的历史产物场景。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

FIELDS_OF_INTEREST = ("intent_lstm", "situation_gcn", "situation_feature_normalizer", "tactical_model")
# 与 `training/common.py::MANIFEST_NON_PATH_KEYS` 同源：这些键的字符串值**不是**文件路径。
try:  # pragma: no cover - 单文件使用时也允许独立运行
    from training.common import MANIFEST_NON_PATH_KEYS, repo_root
except Exception:  # noqa: BLE001
    MANIFEST_NON_PATH_KEYS = frozenset({"generated_at", "managed_files", "environment"})

# manifest 未声明该键时，历史产物的历史文件名（别名回退**仅**用于这种情况）。
LEGACY_ALIASES = {
    "intent_lstm": ("intent_lstm.pt", "intention_lstm.pth"),
    "situation_gcn": ("situation_gcn.pt", "situation_gcn.pth"),
    "tactical_model": ("tactical_model.pt",),
    "situation_feature_normalizer": ("situation_feature_normalizer.json",),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _torch():
    try:
        import torch  # noqa: PLC0415

        return torch
    except Exception:  # noqa: BLE001
        return None


def inspect_state_dict(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """只读窥探 state_dict 形状，**不构造模型**。"""
    torch = _torch()
    if torch is None:
        return None, "torch unavailable"
    try:
        # `weights_only` 只在 torch>=2.0 支持；本脚本的用途正是核验别人给的权重，
        # 故优先用受限加载，失败再退回默认（兼容老格式与包装型检查点）。
        try:
            obj = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:  # noqa: BLE001
            obj = torch.load(path, map_location="cpu")
    except Exception as exc:  # noqa: BLE001
        return None, "%s: %s" % (type(exc).__name__, exc)
    if not isinstance(obj, dict):
        return None, "not a dict: %s" % type(obj).__name__
    sd = obj["model_state"] if ("model_state" in obj and isinstance(obj["model_state"], dict)) else obj
    shapes = {}
    for k, v in sd.items():
        if hasattr(v, "shape"):
            shapes[k] = tuple(v.shape)
    if not shapes:
        return None, "no tensors with shape found"
    return {"shapes": shapes, "keys": len(shapes),
            "wrapped": "model_state" in obj}, None


def derive_dimensions(shapes: Dict[str, Tuple[int, ...]]) -> Dict[str, Any]:
    """从权重形状反推维度，用于与声明**交叉核对**。

    - `lstm.weight_ih_l0` -> hidden = 行数//4（4 门），input = 列数
    - `fc_intent.weight`  -> num_classes = 行数（legacy 头）
    - `sage1.linear.weight` -> `ai/models.py` 是 `Linear(in_features*2, out)`，
      故 num_features = 列数//2
    """
    out: Dict[str, Any] = {}
    lstm = shapes.get("lstm.weight_ih_l0")
    if lstm and len(lstm) == 2:
        out["hidden_dim"] = int(lstm[0] // 4)
        out["input_dim"] = int(lstm[1])
    fc = shapes.get("fc_intent.weight")
    if fc and len(fc) == 2:
        out["num_classes"] = int(fc[0])
    sage = shapes.get("sage1.linear.weight") or shapes.get("backbone.sage1.linear.weight")
    if sage and len(sage) == 2:
        out["situation_hidden_dim"] = int(sage[0])
        out["situation_num_features"] = int(sage[1] // 2)
    return out


def find_weights(weights_dir: Path, manifest: Dict[str, Any]) -> Tuple[Dict[str, Path], List[str]]:
    """按 manifest 声明找文件。返回 (找到的, 声明与实取不一致的说明)。

    **声明优先**：声明了名字就用它；不存在即**不回退**（由调用方判失败）。
    别名回退**仅**用于 manifest 未声明该键的历史产物。
    """
    found: Dict[str, Path] = {}
    notes: List[str] = []
    for key in FIELDS_OF_INTEREST:
        declared = manifest.get(key)
        if isinstance(declared, str) and declared:
            p = weights_dir / declared
            if p.is_file():
                found[key] = p
            # 声明了但不存在：**不回退**，交给调用方判失败并报出声明名。
            continue
        # 未声明：历史产物，允许别名回退，但要记录
        for name in LEGACY_ALIASES.get(key, ()):
            p = weights_dir / name
            if p.is_file():
                found[key] = p
                notes.append("%s: manifest 未声明文件名，按历史别名取 %s" % (key, name))
                break
    return found, notes


def verify(weights_dir: Path) -> Dict[str, Any]:
    rep: Dict[str, Any] = {"weights_dir": str(weights_dir), "checks": [], "failed": 0,
                           "warnings": [], "ok": False, "notes": []}

    def check(name: str, ok: bool, detail: str = "") -> None:
        rep["checks"].append({"name": name, "ok": bool(ok), "detail": detail})
        if not ok:
            rep["failed"] += 1

    def note(msg: str) -> None:
        rep["warnings"].append(msg)

    # ── manifest 读取（所有畸形输入都必须变成失败项，而不是异常）────────────
    mp = weights_dir / "model_manifest.json"
    if not mp.is_file():
        check("model_manifest.json 存在", False, "not found: %s" % mp)
        return rep
    try:
        manifest = json.loads(mp.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        check("model_manifest.json 可解析", False, "malformed JSON: %s: %s" % (type(exc).__name__, exc))
        return rep
    if not isinstance(manifest, dict):
        check("model_manifest.json 是对象", False, "顶层类型是 %s，应为 object" % type(manifest).__name__)
        return rep
    check("model_manifest.json 可解析", True, "%d 个顶层键" % len(manifest))

    env = manifest.get("environment")
    if env is not None and not isinstance(env, dict):
        check("environment 是对象", False, "类型是 %s" % type(env).__name__)
        env = None
    else:
        check("environment 是对象", True, "缺失（历史产物）" if env is None else "ok")
    legacy = not isinstance(env, dict)
    if legacy:
        note("legacy_artifact: 无 environment 段（历史产物，按降级通过并标注）")
    env = env if isinstance(env, dict) else {}

    # ── ① 六类字段（**逐字段**可选降级）──────────────────────────────────
    check("commit（source_commit_sha）", bool(manifest.get("source_commit_sha")),
          str(manifest.get("source_commit_sha"))[:12] or "缺失（历史产物）")
    check("超参（history_len/horizons）",
          manifest.get("history_len") is not None and manifest.get("horizons") is not None)

    if legacy:
        check("依赖版本/环境/experiment_id（历史产物降级通过）", True, "legacy_artifact")
    else:
        # 逐字段：缺失 → 降级通过并标注；**类型错 → 失败**。
        for fname, label in (("python_version", "python_version"), ("torch_version", "torch_version")):
            v = env.get(fname)
            if v is None:
                note("legacy_field: environment.%s 缺失，未核验（降级通过）" % fname)
                check("依赖版本（%s）" % label, True, "缺失，未核验（历史产物）")
            elif isinstance(v, (str, int, float)) and str(v).strip():
                check("依赖版本（%s）" % label, True, str(v))
            else:
                check("依赖版本（%s）" % label, False, "类型/取值非法: %r" % (v,))
        rl = env.get("requirements_lock_sha256")
        if rl is None:
            # 既无 lock 又无 requirements.txt 时这是**诚实的 None**，不是缺陷 ——
            # 与 torch_version 的处理口径必须一致，否则新产物在别的环境里会让验证器变红。
            note("unverifiable: environment.requirements_lock_sha256 缺失，无法核验依赖哈希")
            check("数据/依赖哈希（requirements lock）", True, "缺失，无法核验（非失败）")
        elif isinstance(rl, str) and len(rl) == 64:
            check("数据/依赖哈希（requirements lock）", True, rl[:16])
        else:
            check("数据/依赖哈希（requirements lock）", False, "应为 64 位十六进制字符串: %r" % (rl,))
        # ── ② 数据哈希（I07 判据 1 六类之②）────────────────────────────
        # 本脚本 docstring 早已声称校验六类（含「数据哈希」），但实现里**没有** ——
        # 本轮补上，使该字段**可被独立重算比对**，而不只是被记录。
        # 注意与 ④ 的区别：`requirements_lock_sha256` 是**依赖锁**哈希，不是数据集哈希。
        data_rec = env.get("data_sha256")
        if not isinstance(data_rec, dict):
            note("legacy_field: environment.data_sha256 缺失，未核验数据哈希")
            check("数据哈希（历史产物降级通过）", True, "legacy_artifact")
        else:
            n_ok = n_bad = n_absent = 0
            for key, rec in data_rec.items():
                if not isinstance(rec, dict):
                    continue
                rel = rec.get("path")
                declared = rec.get("sha256")
                if not rel or not isinstance(declared, str):
                    n_absent += 1
                    note("unverifiable: data_sha256.%s 无路径或哈希（%s）"
                         % (key, rec.get("note", "")))
                    continue
                try:
                    _root = repo_root()
                except Exception:      # noqa: BLE001 - 导入失败时退回 cwd
                    _root = Path.cwd()
                f = _root / rel
                if not f.is_file():
                    n_absent += 1
                    note("data_sha256.%s 指向的文件不存在，无法核验: %s" % (key, rel))
                    continue
                actual = sha256_file(f)
                if actual == declared:
                    n_ok += 1
                else:
                    n_bad += 1
                    check("数据哈希 %s" % key, False,
                          "声明 %s… 实际 %s…" % (declared[:16], actual[:16]))
            if n_bad == 0:
                check("数据哈希（对数据文件重算比对）", True,
                      "ok=%d absent=%d" % (n_ok, n_absent))

        eid = env.get("experiment_id")
        if eid is None:
            note("legacy_field: environment.experiment_id 缺失，未核验")
            check("experiment_id", True, "缺失，未核验（历史产物）")
        elif isinstance(eid, str) and eid.strip():
            check("experiment_id", True, eid)
        else:
            check("experiment_id", False, "类型/取值非法: %r" % (eid,))

    # ── ② 文件：声明优先，声明缺失即失败 ──────────────────────────────────
    found, notes = find_weights(weights_dir, manifest)
    rep["notes"].extend(notes)
    rep["files"] = {}
    declared_hashes = env.get("artifact_sha256")
    if declared_hashes is not None and not isinstance(declared_hashes, dict):
        check("environment.artifact_sha256 是对象", False,
              "类型是 %s" % type(declared_hashes).__name__)
        declared_hashes = None
    if not isinstance(declared_hashes, dict):
        note("unverified_hash: manifest 未声明 artifact_sha256，哈希**未被核验**（仅打印实算值）")

    for key in FIELDS_OF_INTEREST:
        declared_name = manifest.get(key)
        p = found.get(key)
        if p is None:
            optional = (key == "tactical_model" and declared_name is None)
            detail = "missing"
            if isinstance(declared_name, str) and declared_name:
                detail = "manifest 声明 %r 但该文件不存在（不回退别名）" % declared_name
            check("存在 %s" % key, optional, detail + ("（可选）" if optional else ""))
            continue
        rep["files"][key] = {"name": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)}
        detail = "%s (%d B)" % (p.name, p.stat().st_size)
        if isinstance(declared_name, str) and declared_name and p.name != declared_name:
            # 理论不可达（声明名不存在时不回退），留作断言
            check("存在 %s" % key, False, "实取 %s != 声明 %s" % (p.name, declared_name))
            continue
        check("存在 %s" % key, True, detail)

        # 哈希核对：声明了就必须逐字节相符。
        # ⚠ 声明表的键是**文件名**（`artifact_sha256` 由 `_artifact_sha256()` 按文件名建表），
        # 不是 manifest 的类别键。我第一版按类别键查，结果**永远查不到、静默跳过比对** ——
        # 于是伪造哈希仍判通过。评审正是用这个用例证伪的，故这里两种键都试。
        want = None
        if isinstance(declared_hashes, dict):
            for lookup in (p.name, key, manifest.get(key)):
                if isinstance(lookup, str) and lookup in declared_hashes:
                    want = declared_hashes[lookup]
                    break
        if want is not None:
            got = rep["files"][key]["sha256"]
            if not isinstance(want, str) or len(want) != 64:
                check("哈希 %s" % key, False, "声明值非法: %r" % (want,))
            elif want.lower() == got.lower():
                check("哈希 %s" % key, True, "declared=%s… == computed" % want[:16])
            else:
                check("哈希 %s" % key, False,
                      "declared=%s… computed=%s… 不一致（权重不是 manifest 声明的那一份）"
                      % (want[:16], got[:16]))

    # ── ③ 权重形状 vs 声明 ────────────────────────────────────────────────
    intent_path = found.get("intent_lstm")
    if intent_path is not None:
        info, err = inspect_state_dict(intent_path)
        if err:
            check("intent 权重可加载", False, err)
        else:
            check("intent 权重可加载", True, "%d 个张量%s" % (
                info["keys"], "（包装型）" if info["wrapped"] else "（裸 state_dict）"))
            dims = derive_dimensions(info["shapes"])
            rep["derived"] = dims
            if "hidden_dim" in dims:
                check("intent hidden_dim 可推导", dims["hidden_dim"] > 0,
                      "hidden=%d input=%d" % (dims["hidden_dim"], dims.get("input_dim", -1)))
            ncls = dims.get("num_classes")
            if ncls is not None:
                note("intent num_classes=%s（数据侧标签为 7 类 one-hot，大于 7 时多出的类"
                     "无对应标签 —— 报出但不判失败）" % ncls)

    # ── ④ 归一化器 feature_dim == 模型输入维度 ────────────────────────────
    norm_path = found.get("situation_feature_normalizer")
    sit_path = found.get("situation_gcn")
    if norm_path is not None:
        norm = None
        try:
            norm = json.loads(norm_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            check("归一化器可解析", False, "malformed: %s: %s" % (type(exc).__name__, exc))
        if norm is not None:
            if not isinstance(norm, dict):
                check("归一化器是对象", False, "顶层类型是 %s" % type(norm).__name__)
                norm = None
            else:
                check("归一化器是对象", True, "")
        if isinstance(norm, dict):
            fd = norm.get("feature_dim")
            check("归一化器 feature_dim 存在", fd is not None, str(fd))
            declared_fd = manifest.get("situation_normalizer_feature_dim")
            if fd is not None and declared_fd is not None:
                try:
                    same = int(fd) == int(declared_fd)
                    check("归一化器 feature_dim == manifest 声明", same,
                          "文件=%s manifest=%s" % (fd, declared_fd))
                except (TypeError, ValueError) as exc:
                    check("归一化器 feature_dim == manifest 声明", False,
                          "取值非整数，无法比较: %s: %s" % (type(exc).__name__, exc))
            if fd is not None and sit_path is not None:
                info2, err2 = inspect_state_dict(sit_path)
                if not err2:
                    dims2 = derive_dimensions(info2["shapes"])
                    nf = dims2.get("situation_num_features")
                    if nf is not None:
                        try:
                            check("归一化器 feature_dim == 模型输入维度", int(fd) == int(nf),
                                  "norm=%s model=%s" % (fd, nf))
                        except (TypeError, ValueError) as exc:
                            check("归一化器 feature_dim == 模型输入维度", False,
                                  "feature_dim 非整数: %s" % exc)
                    else:
                        note("无法从 situation 权重推导输入维度，跳过该项比对")
                else:
                    note("situation 权重不可加载，跳过维度比对: %s" % err2)

    rep["ok"] = rep["failed"] == 0
    return rep


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Verify checkpoint/manifest consistency (I07).")
    ap.add_argument("--weights", default="ai/weights", help="artifacts directory")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    wd = Path(args.weights)
    if not wd.is_dir():
        print("not a directory: %s" % wd, file=sys.stderr)
        return 2
    try:
        rep = verify(wd)
    except Exception as exc:  # noqa: BLE001 - 兜底：验证器自身异常也必须给出可用输出
        rep = {"weights_dir": str(wd), "checks": [{"name": "verifier internal error", "ok": False,
                                                   "detail": "%s: %s" % (type(exc).__name__, exc)}],
               "failed": 1, "warnings": [], "ok": False, "notes": []}
    if args.json:
        print(json.dumps(rep, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print("weights: %s" % wd)
        for c in rep["checks"]:
            print("  %-4s %-46s %s" % ("ok" if c["ok"] else "FAIL", c["name"], c["detail"][:90]))
        for w in rep["warnings"]:
            print("  warn %s" % w[:110])
        for n in rep.get("notes", []):
            print("  note %s" % n[:110])
        if rep.get("files"):
            print("\n  实算 sha256（供第三方逐字节核对）:")
            for k, v in sorted(rep["files"].items()):
                print("    %-30s %s  %s" % (k, v["sha256"][:32] + "…", v["name"]))
        print("\nRESULT: %s（失败 %d 项）" % ("OK" if rep["ok"] else "FAIL", rep["failed"]))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
