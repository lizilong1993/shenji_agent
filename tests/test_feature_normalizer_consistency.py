from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ai.policy import LearnedPolicy
from scripts.diagnose_model_quality import diagnose
from training.supervised import _apply_feature_normalizer, _build_model_manifest, _fit_feature_normalizer


def test_training_normalizer_fits_train_split_only_and_transforms_val() -> None:
    train = [np.array([[1.0, 10.0, 3.0], [3.0, 10.0, 7.0]], dtype=np.float32)]
    val = torch.tensor([[100.0, 10.0, 11.0]], dtype=torch.float32)

    normalizer = _fit_feature_normalizer(train)
    transformed_train = _apply_feature_normalizer(torch.tensor(train[0]), normalizer)
    transformed_val = _apply_feature_normalizer(val, normalizer)

    assert normalizer["fitted_on"] == "situation_train_split"
    assert normalizer["feature_mean"] == [2.0, 10.0, 5.0]
    assert normalizer["zero_std_count"] == 1
    assert torch.allclose(transformed_train.mean(dim=0), torch.zeros(3), atol=1e-6)
    assert torch.isfinite(transformed_val).all()
    assert transformed_val[0, 0].item() == 98.0


def test_zero_std_dimension_uses_safe_std_without_nan_or_inf() -> None:
    normalizer = _fit_feature_normalizer([np.ones((4, 2), dtype=np.float32)])
    transformed = _apply_feature_normalizer(torch.ones(2, 2), normalizer)

    assert normalizer["feature_std"] == [1.0, 1.0]
    assert torch.isfinite(transformed).all()


def test_manifest_records_training_normalizer_contract(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("training.supervised.now_tag", lambda: "test-tag")
    monkeypatch.setattr("training.supervised.git_head_sha", lambda _cwd: "abc123")
    config = {"intent": {"history_len": 8, "horizons": 5}}
    metrics = {
        "situation": {
            "normalizer": {
                "kind": "per_dimension_zscore",
                "fitted_on": "situation_train_split",
                "feature_dim": 108,
                "zero_std_count": 3,
                "training_applied_normalizer": True,
            }
        }
    }

    manifest = _build_model_manifest(config, metrics, tactical_skipped=False)

    assert manifest["situation_feature_normalizer"] == "situation_feature_normalizer.json"
    assert manifest["situation_training_applied_normalizer"] is True
    assert manifest["situation_normalizer_feature_dim"] == 108
    assert "tactical_model.pt" in manifest["managed_files"]

    # ── I07：manifest 必须含实验 ID / 依赖版本 / 环境记录 ──────────────────
    # 既有键一个都不能动（既有调用方按名字取值），新信息集中在 `environment` 一个 dict 里。
    env = manifest.get("environment")
    assert isinstance(env, dict), "I07 要求 manifest 含 environment 段（整块 dict）"
    assert env["experiment_id"] == "test-tag-abc123", \
        "实验 ID 应为 <generated_at>-<commit8>（monkeypatch 已固定这两个输入）"
    assert env["generated_at"] == "test-tag", "environment.generated_at 应与顶层同源"
    assert env["python_version"], "缺少 python_version"
    assert "naming_rule" in env, "缺少命名规则声明"
    # 依赖哈希可能为 None（本仓无 lock 文件时）—— 但**键必须存在**
    assert "requirements_lock_sha256" in env, "缺少依赖哈希键（值可为 None，但不得缺席）"
    # 关键：不得新增"会被 copy_publish_artifacts 当文件路径复制"的顶层字符串值。
    # **引用生产代码的跳过集合**，不抄白名单快照 —— 抄一份会与实现各自漂移，
    # 而且会在 I06 写入合法字符串字段时误报（评审实测过）。
    # 按**真实 run_dir** 判定（不是 repo_root）：copy_publish_artifacts 复制的是
    # run_dir / <值>。用 repo_root 会把"恰好与仓内某文件同名的合法字段"误判为危害。
    from training.common import MANIFEST_NON_PATH_KEYS

    run_dir = tmp_path
    for key, value in manifest.items():
        if key in MANIFEST_NON_PATH_KEYS or not isinstance(value, str):
            continue
        assert not (run_dir / value).exists(), (
            "顶层字符串值 %r=%r 在 run_dir 下真实存在，会被 copy_publish_artifacts 复制" % (key, value))


class _Backbone(nn.Module):
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x[:, :1])


class _SituationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _Backbone()

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return self.backbone(x, adj).mean(dim=0)


def _policy_with_situation(normalizer: dict | None) -> LearnedPolicy:
    policy = LearnedPolicy()
    policy.faction = 1
    policy.situation_model = _SituationModel()
    policy.intent_model = object()
    policy._feature_normalizer = normalizer or {}
    return policy


def test_policy_applies_normalizer_only_when_feature_dim_matches() -> None:
    normalizer = {"feature_mean": [0.0] * 108, "feature_std": [1.0] * 108, "feature_dim": 108}
    policy = _policy_with_situation(normalizer)
    observation = {"operators": [{"obj_id": 1, "color": 1, "cur_hex": 1000}]}

    policy._score_counterfactual(observation, {1: {"obj_id": 1, "color": 1, "cur_hex": 1001}})
    diagnostics = policy.get_diagnostics()["normalizer_diagnostics"]

    assert diagnostics["normalizer_applied_count"] == 1
    assert diagnostics["normalizer_mismatch_reason"] is None


def test_policy_records_mismatch_reason_for_wrong_feature_dim() -> None:
    normalizer = {"feature_mean": [0.0] * 3, "feature_std": [1.0] * 3, "feature_dim": 3}
    policy = _policy_with_situation(normalizer)
    observation = {"operators": [{"obj_id": 1, "color": 1, "cur_hex": 1000}]}

    policy._score_counterfactual(observation, {})
    diagnostics = policy.get_diagnostics()["normalizer_diagnostics"]

    assert diagnostics["normalizer_applied_count"] == 0
    assert diagnostics["normalizer_mismatch_count"] == 1
    assert "feature_dim=108" in diagnostics["normalizer_mismatch_reason"]


def test_missing_normalizer_is_not_reported_as_normal_success() -> None:
    policy = _policy_with_situation(None)
    observation = {"operators": [{"obj_id": 1, "color": 1, "cur_hex": 1000}]}

    policy._score_counterfactual(observation, {})
    diagnostics = policy.get_diagnostics()

    assert diagnostics["normalizer_loaded"] is False
    assert diagnostics["normalizer_diagnostics"]["normalizer_mismatch_reason"] == "normalizer_missing"


def test_diagnose_reports_training_and_inference_normalizer_contract(tmp_path: Path) -> None:
    artifacts = tmp_path / "weights"
    artifacts.mkdir()
    (artifacts / "model_manifest.json").write_text(
        json.dumps(
            {
                "history_len": 8,
                "horizons": 5,
                "situation_feature_normalizer": "situation_feature_normalizer.json",
                "situation_normalizer_feature_dim": 108,
                "situation_training_applied_normalizer": True,
                "managed_files": ["situation_feature_normalizer.json"],
            }
        ),
        encoding="utf-8",
    )
    (artifacts / "situation_feature_normalizer.json").write_text(
        json.dumps({"feature_mean": [0.0] * 108, "feature_std": [1.0] * 108, "feature_dim": 108}),
        encoding="utf-8",
    )

    result = diagnose(artifacts, datasets_dir=None)

    assert result["normalizer_applied_in_training"] is True
    assert result["normalizer_loaded_in_inference"] is True


def test_diagnose_flags_missing_training_normalizer_application(tmp_path: Path) -> None:
    artifacts = tmp_path / "weights"
    artifacts.mkdir()
    (artifacts / "model_manifest.json").write_text(json.dumps({"managed_files": []}), encoding="utf-8")

    result = diagnose(artifacts, datasets_dir=None)

    assert result["normalizer_applied_in_training"] is False
    assert "normalizer_not_applied_in_training" in result["warnings"]
