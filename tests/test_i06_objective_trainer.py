"""I06 步骤 3：objective 模型训练器（由引擎裁决标签真实训练）。

标准依据：`31007 5.3.3`（奖励函数设计指导）与 `I06` 判据 1 ——
「重跑后 `objective_label_counts.high_confidence>0` 且 `objective_metrics.model_trained=true`」。

第 52 轮查清：此前**训练器根本不存在**（`supervised.py` 里 `objective` 出现 0 次，
全仓无 objective 模型类，`ai/policy.py` 只有 `objective_model = None` 与回退诊断）。
本文件把新增的训练器与加载分支钉住。
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

MANIFEST = REPO / "ai" / "weights" / "model_manifest.json"
LABELS = (REPO / "experiments" / "shenji_agent" / "objective_labels"
          / "latest" / "objective_labels.jsonl")
MODELS = REPO / "ai" / "models.py"
SUP = REPO / "training" / "supervised.py"
POL = REPO / "ai" / "policy.py"
COMMON = REPO / "training" / "common.py"


# ─────────────────── 训练器存在且可用

def test_objective_model_class_exists() -> None:
    """`ai/models.py` 必须有 `ObjectiveModel`（原先不存在）。"""
    if not MODELS.is_file():
        pytest.skip("ai/models.py not present")
    src = MODELS.read_text(encoding="utf-8")
    assert "class ObjectiveModel" in src


def test_trainer_exists_in_supervised() -> None:
    """`supervised.py` 必须有训练器（原先 `objective` 出现 0 次）。"""
    if not SUP.is_file():
        pytest.skip("supervised.py not present")
    src = SUP.read_text(encoding="utf-8")
    assert "def train_objective_model" in src
    assert "def load_objective_labels" in src
    assert "OBJECTIVE_FEATURE_NAMES" in src
    # main() 必须真的调用它（否则函数写了也不会跑）
    assert '"objective": train_objective_model(' in src


def test_trainer_uses_only_existing_label_fields() -> None:
    """特征必须**取自标签既有字段**，不得自造。

    这条是"防把标准往实现上凑"的检查：特征名必须都能在标签 JSON 里找到。
    """
    # ⚠ 不能用 spec_from_file_location 直接加载 supervised.py：
    # 它有 `from .common import ...` 相对导入，脱离包上下文会 ImportError。
    # 故按包路径导入。
    from training import supervised as mod  # noqa: PLC0415
    names = list(mod.OBJECTIVE_FEATURE_NAMES)
    assert names, "特征名列表为空"
    base = {"damage_dealt", "damage_taken", "kill_dealt", "kill_taken",
            "occupy_success", "distance_delta", "city_value", "side"}
    for n in names:
        assert n in base or n in ("has_damage", "has_kill"), \
            "特征 %s 不是标签既有字段/其指示位" % n


def test_trainer_trains_only_on_high_confidence_labels() -> None:
    """只在**高置信**标签上训练 —— 低置信标签特征全 0、目标恒 0，会教模型输出 0。"""
    if not SUP.is_file():
        pytest.skip("supervised.py not present")
    src = SUP.read_text(encoding="utf-8")
    assert 'label_confidence' in src and '"high"' in src


def test_trainer_reports_improvement_over_baseline() -> None:
    """必须给出"是否优于朴素基线"，否则"训练了"不等于"学到东西"。"""
    if not SUP.is_file():
        pytest.skip("supervised.py not present")
    src = SUP.read_text(encoding="utf-8")
    assert "baseline_mae" in src
    assert "improves_over_baseline" in src


# ─────────────────── 训练器可实际运行（真标签）

def test_trainer_runs_on_real_labels(tmp_path: Path) -> None:
    """**端到端**：用真实标签训练一次，断言判据 1 的两半。"""
    if not LABELS.is_file():
        pytest.skip("objective labels not present")
    # ⚠ 不能用 spec_from_file_location 直接加载 supervised.py：
    # 它有 `from .common import ...` 相对导入，脱离包上下文会 ImportError。
    # 故按包路径导入。
    from training import supervised as mod  # noqa: PLC0415
    import torch

    res = mod.train_objective_model({}, LABELS, torch.device("cpu"), tmp_path)
    assert res.get("high_confidence", 0) > 0, "高置信标签为 0"
    assert res.get("model_trained") is True, "未训练成功：%s" % res
    assert (tmp_path / "objective_model.pt").is_file(), "权重未落盘"
    # 方向一致率应显著高于随机（0.5）
    assert res.get("val_sign_agreement", 0.0) > 0.7, \
        "符号方向一致率过低（%s）—— 模型没学到东西" % res.get("val_sign_agreement")


def test_trainer_degrades_gracefully_without_labels(tmp_path: Path) -> None:
    """没有标签时必须**结构化降级**，不得抛异常。"""
    if not SUP.is_file():
        pytest.skip("supervised.py not present")
    # ⚠ 不能用 spec_from_file_location 直接加载 supervised.py：
    # 它有 `from .common import ...` 相对导入，脱离包上下文会 ImportError。
    # 故按包路径导入。
    from training import supervised as mod  # noqa: PLC0415
    import torch

    res = mod.train_objective_model({}, tmp_path / "nope.jsonl",
                                    torch.device("cpu"), tmp_path)
    assert res.get("model_trained") is False
    assert res.get("objective_mode") == "heuristic_policy_score"
    assert res.get("reason")


# ─────────────────── manifest 取证字段

def test_manifest_records_the_two_criterion_fields() -> None:
    """**判据 1 的取证出口**：manifest 必须含这两个字段且为训练后取值。"""
    if not MANIFEST.is_file():
        pytest.skip("manifest not present")
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    olc = m.get("objective_label_counts") or {}
    om = m.get("objective_metrics") or {}
    assert int(olc.get("high_confidence") or 0) > 0, \
        "objective_label_counts.high_confidence 必须 > 0（判据 1 前半）"
    assert om.get("model_trained") is True, \
        "objective_metrics.model_trained 必须为 true（判据 1 后半）"
    assert m.get("objective_mode") == "trained_objective_model"
    assert m.get("objective_model") == "objective_model.pt"
    assert "objective_model.pt" in (m.get("managed_files") or [])


def test_objective_label_source_is_a_relative_metadata_path() -> None:
    """**我自己引入过的缺陷**：`objective_label_source` 一度写成**绝对路径**。

    那会被 `copy_publish_artifacts` 当产物处理（I07 的守卫正是防这个），
    实际也真的把 3.27 MB 的标签文件复制进了 `ai/weights/`。
    现改为**相对仓库**路径，并登记进 `MANIFEST_NON_PATH_KEYS`。
    """
    if not MANIFEST.is_file():
        pytest.skip("manifest not present")
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    src = m.get("objective_label_source") or ""
    assert src, "缺少 objective_label_source"
    assert not src.startswith("/") and ":" not in src.split("/")[0], \
        "必须是相对路径，实际 %r" % src
    assert src.startswith("experiments/")


def test_objective_label_source_is_registered_as_non_path_key() -> None:
    """登记为非产物键，否则发布逻辑会去 run_dir 下找它。"""
    if not COMMON.is_file():
        pytest.skip("training/common.py not present")
    src = COMMON.read_text(encoding="utf-8")
    assert "MANIFEST_NON_PATH_KEYS" in src
    assert "objective_label_source" in src


def test_publish_does_not_copy_the_label_file(tmp_path: Path) -> None:
    """**端到端**：跑一次发布复制，断言标签文件**没有**被复制。"""
    if not (MANIFEST.is_file() and COMMON.is_file()):
        pytest.skip("manifest or common not present")
    spec = importlib.util.spec_from_file_location("common", COMMON)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    run_dir = tmp_path / "run"; run_dir.mkdir()
    pub = tmp_path / "pub"; pub.mkdir()
    for name in m.get("managed_files") or []:
        if name == "model_manifest.json":
            continue
        (run_dir / name).write_bytes(b"x")
    mod.copy_publish_artifacts(run_dir, pub, m)
    copied = sorted(p.name for p in pub.iterdir())
    assert "objective_labels.jsonl" not in copied, "标签文件被误当产物复制"


# ─────────────────── policy 加载链

def test_policy_loads_the_published_objective_model() -> None:
    """policy 必须真的加载它（原代码只有 `objective_model = None`）。"""
    if not POL.is_file():
        pytest.skip("policy.py not present")
    src = POL.read_text(encoding="utf-8")
    assert "_load_objective_model" in src, "缺少加载方法"
    assert "objective_enabled" in src, "缺少加载分支"
    assert "objective_model_missing_using_heuristic" in src, "回退原因被删了"


def test_policy_infers_dims_from_weights_not_hardcoded() -> None:
    """维度必须**从权重形状推导** —— 硬编码会在特征口径变化时静默加载错。"""
    if not POL.is_file():
        pytest.skip("policy.py not present")
    src = POL.read_text(encoding="utf-8")
    i = src.find("def _load_objective_model")
    assert i > 0
    body = src[i:i + 1200]
    assert "shape[1]" in body and "shape[0]" in body
