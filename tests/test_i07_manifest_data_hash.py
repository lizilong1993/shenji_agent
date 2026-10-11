"""I07 判据 1 的**最后一类**：manifest 的「② 数据哈希」。

## 判据原文

> 1) manifest 含 commit、**数据哈希**、超参、依赖版本、环境与 `experiment_id` **六类字段**；
> 2) `python3 scripts/verify_checkpoint.py --weights ai/weights` **退出码 0**；
> 3) 同 seed 两次训练权重哈希一致（既有测试保持通过）。

## 本轮修复前的实测

| 类别 | 修前 |
| --- | --- |
| commit / 超参 / 依赖版本 / 环境 / `experiment_id` | ✓ 均在 |
| **数据哈希** | ✗ **缺失** |

**故只补这一项。** 且注意区分：已有的 `environment.requirements_lock_sha256` 是
**依赖锁**哈希，**不是数据集哈希** —— 拿它顶替不成立（独立评审亦指出）。

## 补法

`training/supervised.py:_data_sha256()` 对**训练真正读入的数据文件**算 SHA-256，
并记记录数作为覆盖面证据；文件缺失时记 `null` 并写明原因，**不编造**。
`scripts/verify_checkpoint.py` 新增**重算比对**（此前 docstring 声称校验六类含数据哈希，
但实现里没有）。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

MANIFEST = REPO / "ai" / "weights" / "model_manifest.json"
SUPERVISED = REPO / "training" / "supervised.py"
VERIFY = REPO / "scripts" / "verify_checkpoint.py"

# 判据 1) 的六类（键名按实际层级）
SIX = {
    "commit": ["source_commit_sha"],
    "数据哈希": ["data_sha256"],
    "超参": ["history_len", "horizons"],
    "依赖版本": ["python_version", "torch_version", "requirements_lock_sha256"],
    "环境": ["platform", "cuda_available"],
    "experiment_id": ["experiment_id"],
}


def _leaf_paths(obj, prefix="", depth=0, out=None):
    if out is None:
        out = []
    if isinstance(obj, dict) and depth < 5:
        for k, v in obj.items():
            _leaf_paths(v, (prefix + "." + k) if prefix else k, depth + 1, out)
    else:
        out.append(prefix)
    return out


def _manifest() -> dict:
    if not MANIFEST.is_file():
        pytest.skip("manifest not present")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


# ────────────── 判据 1：六类齐全

def test_manifest_contains_all_six_required_categories() -> None:
    """**判据 1 的核心断言**：六类字段齐备。"""
    d = _manifest()
    paths = _leaf_paths(d)
    missing = []
    for cat, keys in SIX.items():
        # ② 数据哈希必须来自 data_sha256，不能被依赖锁哈希冒充
        if cat == "数据哈希":
            hits = [p for p in paths
                    if p.startswith("environment.data_sha256") and p.endswith(".sha256")]
        else:
            hits = [p for p in paths if p.split(".")[-1] in keys]
        if not hits:
            missing.append(cat)
    assert not missing, "manifest 缺类别: %s" % missing


def test_data_hash_is_not_the_dependency_lock_hash() -> None:
    """**关键区分**：`data_sha256` 与 `requirements_lock_sha256` 必须是**不同的**东西。

    拿依赖锁哈希当数据哈希是评审明确指出的不成立做法。
    """
    d = _manifest()
    env = d.get("environment") or {}
    ds = env.get("data_sha256")
    assert isinstance(ds, dict), "缺 environment.data_sha256"
    lock = env.get("requirements_lock_sha256")
    data_hashes = {v.get("sha256") for v in ds.values()
                   if isinstance(v, dict) and v.get("sha256")}
    assert data_hashes, "data_sha256 里没有任何实际哈希"
    assert lock not in data_hashes, "数据哈希与依赖锁哈希相同 —— 显然是拿错了对象"
    # 且必须指向真实数据文件，而非权重目录
    for k, v in ds.items():
        if isinstance(v, dict) and v.get("path"):
            assert not v["path"].startswith("ai/"), \
                "data_sha256.%s 指向权重目录，不是数据" % k


def test_data_hash_matches_the_actual_data_file() -> None:
    """**可独立重算**：声明的哈希必须等于对真实数据文件重算的结果。"""
    d = _manifest()
    ds = (d.get("environment") or {}).get("data_sha256") or {}
    checked = 0
    for key, rec in ds.items():
        if not isinstance(rec, dict):
            continue
        rel, declared = rec.get("path"), rec.get("sha256")
        if not rel or not isinstance(declared, str):
            continue
        f = REPO / rel
        if not f.is_file():
            pytest.skip("data file absent: %s" % rel)
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        assert h == declared, "%s 声明 %s… 实际 %s…" % (key, declared[:16], h[:16])
        checked += 1
    assert checked > 0, "没有任何可重算的数据哈希"


def test_data_hash_records_record_counts_as_coverage_evidence() -> None:
    """只记哈希无法说明**读了多少** —— 须同时记记录数。"""
    d = _manifest()
    ds = (d.get("environment") or {}).get("data_sha256") or {}
    lab = ds.get("objective_labels") or {}
    assert lab.get("records"), "未记录 objective_labels 的记录数"
    assert lab["records"] > 0
    # 与真实文件行数一致
    f = REPO / lab["path"]
    if f.is_file():
        n = sum(1 for line in f.read_text(encoding="utf-8").splitlines() if line.strip())
        assert lab["records"] == n, "记录数 %s 与实际行数 %d 不符" % (lab["records"], n)


def test_missing_data_file_records_null_not_a_fabricated_hash() -> None:
    """**不编造**：文件缺失时记 `null` 并说明，而不是用一个假哈希"凑齐六类"。"""
    src = SUPERVISED.read_text(encoding="utf-8")
    i = src.find("def _data_sha256(")
    assert i > 0, "缺 _data_sha256"
    body = src[i:i + 2600]
    assert '"sha256": None' in body, "缺失时未记 null"
    assert "file not found" in body, "缺失时未写明原因"
    # 二进制读，避免换行转换改变哈希
    assert '"rb"' in body, "未以二进制读 —— 换行转换会改变哈希"


# ────────────── 判据 2：验证脚本退出码

def test_verify_checkpoint_exits_zero() -> None:
    """判据 2：`verify_checkpoint.py --weights ai/weights` **退出码 0**。"""
    if not VERIFY.is_file():
        pytest.skip("verify_checkpoint.py not present")
    r = subprocess.run([sys.executable, str(VERIFY), "--weights", "ai/weights"],
                       cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, (r.stdout or "")[-800:]


def test_verify_checkpoint_actually_checks_the_data_hash() -> None:
    """**docstring 声称校验六类（含数据哈希），实现里原先没有** —— 须确有该检查项。

    只声称不实现，等于把判据 1 的一条**记成已验而实际未验**。
    """
    if not VERIFY.is_file():
        pytest.skip("verify_checkpoint.py not present")
    r = subprocess.run([sys.executable, str(VERIFY), "--weights", "ai/weights"],
                       cwd=REPO, capture_output=True, text=True)
    out = r.stdout or ""
    assert "数据哈希" in out, "验证脚本输出里没有数据哈希检查项"
    assert "重算比对" in out, "未做重算比对（只读声明值不算核验）"


def test_verify_still_handles_a_legacy_manifest_without_data_hash(tmp_path: Path) -> None:
    """**历史产物降级通过**：老 manifest 无 `data_sha256` 时不得判失败。

    否则每次扩展 manifest 都会让历史检查点失效 —— 那是 `I07`【风险与回滚】要求的。
    """
    src = VERIFY.read_text(encoding="utf-8")
    assert "legacy_field: environment.data_sha256" in src or "legacy_artifact" in src
    i = src.find("data_sha256")
    assert i > 0
    assert "legacy" in src[i:i + 900].lower(), "缺老产物降级分支"
