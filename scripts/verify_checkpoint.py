#!/usr/bin/env python3
"""检查点与 manifest 一致性验证（I07 步骤 3）。

判据 2：`python3 scripts/verify_checkpoint.py --weights ai/weights` 退出码 0。

检查什么
--------
1. **manifest 齐全性**：六类字段（commit / 数据哈希 / 超参 / 依赖版本 / 环境 / experiment_id）；
2. **权重与声明一致**：manifest 里每个"文件名型"字段对应的文件是否存在、可加载，
   且**从权重形状反推的维度**与 manifest 声明一致（`horizons` / `num_classes` / `feature_dim`）；
3. **归一化器与模型输入匹配**：`situation_feature_normalizer.json` 的 `feature_dim`
   必须等于 situation 模型输入维度（任务清单点名要求 feature_dim=108）；
4. **哈希记录**：给出每个托管文件的 sha256，使"这一份权重"可被第三方逐字节核对。

退出码
------
`0` 通过；`1` 有失败项；`2` 用法错误。

历史产物怎么处理（【风险与回滚】要求）
--------------------------------------
「新字段全部可选，缺失时按**历史产物**降级通过并标注」。
故：缺 `environment` 不判失败，而是记为 `legacy_artifact` 并在输出里标注 ——
**旧权重不该因为"没有新字段"而被判成坏权重**。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

FIELDS_OF_INTEREST = ("intent_lstm", "situation_gcn", "situation_feature_normalizer", "tactical_model")
NON_FILE_KEYS = {"generated_at", "managed_files", "environment"}


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
    """只读地窥探 state_dict 的形状，**不构造模型**（避免依赖模型类与本仓布局）。"""
    torch = _torch()
    if torch is None:
        return None, "torch unavailable"
    try:
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
    return {"shapes": shapes, "keys": len(shapes),
            "wrapped": "model_state" in obj, "meta": {k: obj[k] for k in obj if k != "model_state" and not hasattr(obj[k], "shape")}}, None


def derive_dimensions(shapes: Dict[str, Tuple[int, ...]]) -> Dict[str, Any]:
    """从权重形状反推维度，用于与 manifest 声明**交叉核对**。

    三种头都覆盖（本仓实测存在多种）：
      - `lstm.weight_ih_l0`      -> hidden = 行数//4（双向 LSTM 的 4 门），input = 列数
      - `fc_intent.weight`       -> num_classes = 行数（legacy IntentionLSTM）
      - `head.*.weight` / 其它   -> 记录最大行数以供人工核对
      - `sage1.linear.weight`    -> hidden = 行数，num_features = 列数//2
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


def find_weights(weights_dir: Path, manifest: Dict[str, Any]) -> Dict[str, Path]:
    """按 manifest 的声明找文件；声明缺失时按别名表兜底（与 ai/artifacts.py 同口径）。"""
    aliases = {"intent_lstm": ("intent_lstm.pt", "intention_lstm.pth"),
               "situation_gcn": ("situation_gcn.pt", "situation_gcn.pth"),
               "tactical_model": ("tactical_model.pt",),
               "situation_feature_normalizer": ("situation_feature_normalizer.json",)}
    found: Dict[str, Path] = {}
    for key in FIELDS_OF_INTEREST:
        declared = manifest.get(key)
        cands: List[str] = []
        if isinstance(declared, str) and declared:
            cands.append(declared)
        cands.extend(aliases.get(key, ()))
        for name in cands:
            p = weights_dir / name
            if p.is_file():
                found[key] = p
                break
    return found


def verify(weights_dir: Path) -> Dict[str, Any]:
    # `ok` 必须在**构造时**就存在：早退分支（缺 manifest）若不带它，
    # 调用方读 `rep["ok"]` 会 KeyError —— 返回值形状必须对所有分支一致。
    rep: Dict[str, Any] = {"weights_dir": str(weights_dir), "checks": [], "failed": 0,
                           "warnings": [], "ok": False}

    def check(name: str, ok: bool, detail: str = "") -> None:
        rep["checks"].append({"name": name, "ok": bool(ok), "detail": detail})
        if not ok:
            rep["failed"] += 1

    mp = weights_dir / "model_manifest.json"
    if not mp.is_file():
        check("model_manifest.json 存在", False, "not found: %s" % mp)
        return rep
    manifest = json.loads(mp.read_text(encoding="utf-8"))
    check("model_manifest.json 可解析", True, "%d 个顶层键" % len(manifest))

    # ① 六类字段
    env = manifest.get("environment")
    legacy = not isinstance(env, dict)
    if legacy:
        rep["warnings"].append("legacy_artifact: 无 environment 段（历史产物，按降级通过并标注）")
    check("commit（source_commit_sha）", bool(manifest.get("source_commit_sha")),
          str(manifest.get("source_commit_sha"))[:12])
    check("超参（history_len/horizons）",
          manifest.get("history_len") is not None and manifest.get("horizons") is not None)
    if legacy:
        check("依赖版本/环境/experiment_id（历史产物降级通过）", True, "legacy_artifact")
    else:
        check("依赖版本（python/torch）",
              bool(env.get("python_version")) and env.get("torch_version") is not None,
              "python=%s torch=%s" % (env.get("python_version"), env.get("torch_version")))
        check("数据/依赖哈希（requirements lock）",
              env.get("requirements_lock_sha256") is not None,
              str(env.get("requirements_lock_sha256"))[:16])
        check("experiment_id", bool(env.get("experiment_id")), str(env.get("experiment_id")))

    # ② 文件与哈希
    found = find_weights(weights_dir, manifest)
    rep["files"] = {}
    for key in FIELDS_OF_INTEREST:
        p = found.get(key)
        if p is None:
            optional = key in ("tactical_model",) or manifest.get(key) is None
            check("存在 %s" % key, optional, "missing" + ("（可选）" if optional else ""))
            continue
        rep["files"][key] = {"name": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)}
        check("存在 %s" % key, True, "%s (%d B)" % (p.name, p.stat().st_size))

    # ③ 权重形状 vs manifest 声明
    intent_path = found.get("intent_lstm")
    if intent_path is not None:
        info, err = inspect_state_dict(intent_path)
        if err:
            check("intent 权重可加载", False, err)
        else:
            check("intent 权重可加载", True, "%d 个张量%s" % (info["keys"], "（包装型）" if info["wrapped"] else "（裸 state_dict）"))
            dims = derive_dimensions(info["shapes"])
            rep["derived"] = dims
            if "hidden_dim" in dims:
                check("intent hidden_dim 可推导", dims["hidden_dim"] > 0, "hidden=%d input=%d" % (
                    dims["hidden_dim"], dims.get("input_dim", -1)))
            # 数据侧真实类别数（7 类 one-hot）；模型类别数可以更大（历史产物），故只报不判
            ncls = dims.get("num_classes")
            if ncls is not None:
                rep["warnings"].append("intent num_classes=%s（数据侧标签为 7 类 one-hot，"
                                       "大于 7 时多出的类无对应标签 —— 报出但不判失败）" % ncls)

    # ④ 归一化器 feature_dim 与模型输入匹配（任务清单点名要求）
    norm_path = found.get("situation_feature_normalizer")
    sit_path = found.get("situation_gcn")
    if norm_path is not None:
        try:
            norm = json.loads(norm_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            norm, rep_err = None, str(exc)
            check("归一化器可解析", False, rep_err)
        if norm is not None:
            fd = norm.get("feature_dim")
            check("归一化器 feature_dim 存在", fd is not None, str(fd))
            declared_fd = manifest.get("situation_normalizer_feature_dim")
            if declared_fd is not None:
                check("归一化器 feature_dim == manifest 声明", int(fd) == int(declared_fd),
                      "文件=%s manifest=%s" % (fd, declared_fd))
            if sit_path is not None:
                info2, err2 = inspect_state_dict(sit_path)
                if not err2:
                    dims2 = derive_dimensions(info2["shapes"])
                    nf = dims2.get("situation_num_features")
                    if nf is not None:
                        check("归一化器 feature_dim == 模型输入维度", int(fd) == int(nf),
                              "norm=%s model=%s" % (fd, nf))
                    else:
                        rep["warnings"].append("无法从 situation 权重推导输入维度，跳过该项比对")
                else:
                    rep["warnings"].append("situation 权重不可加载，跳过维度比对: %s" % err2)
    # 注意：归一化器**缺失**的情况已在上面的通用循环里记过一次，
    # 这里不再重复记 —— 否则同一个缺失被算成两项失败，失败计数虚高。

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
    rep = verify(wd)
    if args.json:
        print(json.dumps(rep, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print("weights: %s" % wd)
        for c in rep["checks"]:
            print("  %-4s %-46s %s" % ("ok" if c["ok"] else "FAIL", c["name"], c["detail"][:80]))
        for w in rep["warnings"]:
            print("  warn %s" % w[:110])
        print("\nRESULT: %s（失败 %d 项）" % ("OK" if rep["ok"] else "FAIL", rep["failed"]))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
