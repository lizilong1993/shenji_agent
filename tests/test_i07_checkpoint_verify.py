"""I07：检查点验证脚本 + manifest 六类字段。

判据 2 是「`python3 scripts/verify_checkpoint.py --weights ai/weights` 退出码 0」，
故这里**真的跑那个脚本**（子进程），而不是重新实现一遍它的逻辑 ——
重实现会掩盖"脚本本身跑不起来"这类问题。
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
VERIFY = REPO / "scripts" / "verify_checkpoint.py"


def _load_verify():
    spec = importlib.util.spec_from_file_location("verify_checkpoint", VERIFY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ───────────────────────── 判据 2：脚本在真实权重上退出码 0

def test_verify_script_passes_on_repo_weights() -> None:
    if not VERIFY.exists():
        pytest.skip("verify_checkpoint.py not present")
    wd = REPO / "ai" / "weights"
    if not wd.is_dir():
        pytest.skip("ai/weights not present")
    r = subprocess.run([sys.executable, str(VERIFY), "--weights", str(wd), "--json"],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, "退出码应为 0\n%s\n%s" % (r.stdout[-2000:], r.stderr[-2000:])
    rep = json.loads(r.stdout)
    assert rep["ok"] is True and rep["failed"] == 0


def test_verify_script_reports_hash_and_dimensions() -> None:
    if not VERIFY.exists():
        pytest.skip("verify_checkpoint.py not present")
    wd = REPO / "ai" / "weights"
    if not wd.is_dir():
        pytest.skip("ai/weights not present")
    r = subprocess.run([sys.executable, str(VERIFY), "--weights", str(wd), "--json"],
                       capture_output=True, text=True, timeout=300)
    rep = json.loads(r.stdout)
    # 每个托管文件都要有 sha256，第三方才能逐字节核对
    assert rep.get("files"), "未记录任何文件哈希"
    for name, info in rep["files"].items():
        assert len(info["sha256"]) == 64, "%s 的 sha256 长度不对" % name
        assert info["bytes"] > 0
    # 归一化器维度必须与模型输入匹配（清单点名要求 feature_dim=108）
    norm = [c for c in rep["checks"] if "归一化器 feature_dim == 模型输入维度" in c["name"]]
    assert norm and norm[0]["ok"], "归一化器 feature_dim 与模型输入不匹配"


def test_verify_script_exit_codes() -> None:
    if not VERIFY.exists():
        pytest.skip("verify_checkpoint.py not present")
    # 不存在的目录 -> 用法错误 2
    r = subprocess.run([sys.executable, str(VERIFY), "--weights", "/nonexistent-xyz"],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 2


# ───────────────────────── 历史产物降级通过

def test_missing_environment_is_legacy_not_failure(tmp_path: Path) -> None:
    """【风险与回滚】：新字段可选，缺失时按**历史产物**降级通过并标注。

    旧权重不该因为"没有新字段"被判成坏权重 —— 那样验证脚本会永远红着，没人再看它。
    """
    if not VERIFY.exists():
        pytest.skip("verify_checkpoint.py not present")
    v = _load_verify()
    # ⚠ 必须把托管文件造齐：否则这个用例测的是「文件缺失」，而不是「缺 environment」。
    # 我第一版没造，于是它失败了一次 —— 那是**测试写错**，不是被测逻辑错。
    (tmp_path / "intent_lstm.pt").write_bytes(b"placeholder")
    (tmp_path / "situation_gcn.pt").write_bytes(b"placeholder")
    (tmp_path / "situation_feature_normalizer.json").write_text(
        json.dumps({"feature_dim": 108}), encoding="utf-8")
    (tmp_path / "model_manifest.json").write_text(json.dumps({
        "generated_at": "20260428_235813",
        "source_commit_sha": "c3bd538603afe1f020bea7709523c588fb8b69a0",
        "history_len": 8, "horizons": 5,
        "intent_lstm": "intent_lstm.pt",
        "situation_gcn": "situation_gcn.pt",
        "situation_feature_normalizer": "situation_feature_normalizer.json",
        "managed_files": [],
    }, ensure_ascii=False), encoding="utf-8")
    rep = v.verify(tmp_path)
    legacy_checks = [c for c in rep["checks"] if "历史产物降级通过" in c["name"]]
    assert legacy_checks and legacy_checks[0]["ok"], \
        "缺 environment 时应按历史产物**判通过**并标注"
    assert any("legacy_artifact" in w for w in rep["warnings"]), "未标注 legacy_artifact"


def test_missing_manifest_fails(tmp_path: Path) -> None:
    if not VERIFY.exists():
        pytest.skip("verify_checkpoint.py not present")
    v = _load_verify()
    rep = v.verify(tmp_path)
    assert rep["ok"] is False and rep["failed"] >= 1


# ───────────────────────── manifest 构建：六类字段

def test_manifest_has_all_six_field_classes() -> None:
    if not VERIFY.exists():
        pytest.skip("verify_checkpoint.py not present")
    from training.supervised import _build_model_manifest

    cfg = {"intent": {"history_len": 8, "horizons": 5}}
    metrics = {"situation": {"normalizer": {
        "kind": "per_dimension_zscore", "fitted_on": "train",
        "feature_dim": 108, "zero_std_count": 0, "training_applied_normalizer": True}}}
    m = _build_model_manifest(cfg, metrics, tactical_skipped=False)
    env = m["environment"]
    assert m["source_commit_sha"]                     # 1 代码状态
    assert "requirements_lock_sha256" in env          # 2 数据/依赖哈希（值可 None）
    assert m["history_len"] == 8 and m["horizons"] == 5   # 3 超参
    assert env["python_version"] and "torch_version" in env  # 4 依赖版本
    assert env["platform"]                            # 5 环境
    assert env["experiment_id"]                       # 6 实验身份


def test_experiment_id_is_deterministic_and_derived() -> None:
    """同输入必须同 ID（不含随机数），否则第三方无法判断"是不是同一次实验"。"""
    from training.supervised import _experiment_id
    a = _experiment_id("20260101_000000")
    b = _experiment_id("20260101_000000")
    assert a == b, "实验 ID 不确定，幂等/复跑核验会失效"
    assert a.startswith("20260101_000000-")


# ───────────────────────── 保护 copy_publish_artifacts 不被误伤

def test_environment_is_a_dict_not_top_level_scalars() -> None:
    """若把新字段平铺到顶层，`copy_publish_artifacts` 会把字符串值当文件路径。

    该函数只跳过 `generated_at` 与 `managed_files`，对其它**字符串值**一律
    尝试 `run_dir / rel_path` 复制。今天靠"文件不存在就跳过"侥幸不出错，
    但那是巧合式安全，故新信息必须收进一个 dict。
    """
    from training.supervised import _build_model_manifest

    cfg = {"intent": {"history_len": 8, "horizons": 5}}
    metrics = {"situation": {"normalizer": {}}}
    m = _build_model_manifest(cfg, metrics, tactical_skipped=True)
    assert isinstance(m["environment"], dict)
    allowed_str = {"generated_at", "source_commit_sha", "intent_lstm", "situation_gcn",
                   "situation_feature_normalizer", "tactical_model",
                   "situation_normalizer_kind", "situation_normalizer_fitted_on",
                   "managed_files"}
    offenders = [k for k, v in m.items() if isinstance(v, str) and k not in allowed_str]
    assert offenders == [], "顶层出现会被当文件路径的字符串值: %s" % offenders
    # 与 copy_publish_artifacts 的实际跳过集合保持一致
    assert "environment" not in {"generated_at", "managed_files"}, "本测试的前提已变"
