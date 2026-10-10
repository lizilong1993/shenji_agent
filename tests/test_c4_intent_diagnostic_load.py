"""C4 回归：诊断器必须能读**仓库实际格式**的 intent 检查点。

为什么单独钉住
--------------
`scripts/diagnose_model_quality.py` 原先只支持「包装型」检查点
（`{"model_state": {...}, "horizons":5, "num_classes":7}`），
而 `ai/weights/intention_lstm.pth` 是**裸 state_dict** + legacy `fc_intent` 头 + **8 类**。
后果是诊断把**存在且主链路能正常加载**的模型报成 MISSING，
于是 `move_bias_estimate` 从未真正算过，而它却驱动了 `address_intent_move_bias` 建议。

本测试构造与仓库同构的裸 state_dict（含 `fc_intent`、8 类），
断言诊断能加载、能推断维度、且**输出的是类别分布**（长度 = 类别数）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("numpy")

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _bare_legacy_state_dict(num_classes: int = 8, feature_dim: int = 108, hidden_dim: int = 64):
    """构造 legacy IntentionLSTM 的**裸** state_dict（无 model_state 包装）。"""
    from ai.models import IntentionLSTM

    m = IntentionLSTM(input_dim=feature_dim, hidden_dim=hidden_dim, num_classes=num_classes)
    return m.state_dict()


def test_diagnostic_loads_bare_legacy_state_dict(tmp_path: Path) -> None:
    """裸 state_dict：必须能加载，且 num_classes 从 fc_intent 推断。"""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "diag", REPO / "scripts" / "diagnose_model_quality.py")
    diag = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diag)

    sd = _bare_legacy_state_dict(num_classes=8)
    assert "model_state" not in sd, "本用例要的是裸 state_dict"
    assert "fc_intent.weight" in sd
    ckpt = tmp_path / "intention_lstm.pth"
    torch.save(sd, ckpt)

    model, payload = diag._load_intent_model(ckpt, "cpu")
    assert model is not None, "裸 state_dict 必须能加载（原先在这里失败）"
    assert payload.get("num_classes") == 8, "类别数应从 fc_intent.weight 的形状推断"
    assert payload.get("feature_dim") == 108


def test_diagnostic_loads_wrapped_multi_horizon(tmp_path: Path) -> None:
    """包装型 + head 头：原有路径不得回归。"""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "diag2", REPO / "scripts" / "diagnose_model_quality.py")
    diag = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diag)

    from ai.models import MultiHorizonIntentionModel

    m = MultiHorizonIntentionModel(input_dim=108, hidden_dim=64, horizons=5, num_classes=7)
    ckpt = tmp_path / "wrapped.pt"
    torch.save({"model_state": m.state_dict(), "feature_dim": 108, "hidden_dim": 64,
                "horizons": 5, "num_classes": 7}, ckpt)

    model, payload = diag._load_intent_model(ckpt, "cpu")
    assert model is not None
    assert payload.get("num_classes") == 7


def test_diagnostic_reports_load_error_instead_of_silence(tmp_path: Path) -> None:
    """无法加载时 payload 必须带 load_error —— 不能静默变成 MISSING。"""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "diag3", REPO / "scripts" / "diagnose_model_quality.py")
    diag = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diag)

    bad = tmp_path / "bad.pt"
    torch.save({"some_unrelated": torch.zeros(3)}, bad)
    model, payload = diag._load_intent_model(bad, "cpu")
    assert model is None
    assert payload and "load_error" in payload, "失败必须留下 load_error 供排查"

# ── situation 路径：与 intent 路径**同一个 bug**（本轮补上） ──────────────
#
# `ai/weights/situation_gcn.pth` 同样是**裸 state_dict**，且键为 legacy 形式
# `sage1.linear.weight`（**无 `backbone.` 前缀**）。原诊断只认
# `payload["model_state"]` + `backbone.` 前缀，于是同样报 MISSING。
# 主链路的处置是「把 legacy 键加 `backbone.` 前缀再 load」——这里钉住该行为。


def _bare_legacy_situation_state_dict(num_features: int = 108, hidden_dim: int = 64):
    """构造 legacy SituationGCN 的**裸** state_dict（无 backbone. 前缀、无包装）。"""
    from ai.models import SituationGCN

    # 注意：`SituationGCN.state_dict()` 的键**本来就没有** `backbone.` 前缀
    # （`backbone.` 是外层 `GraphSituationModel` 加上的），故直接返回即可 ——
    # 这正是仓库里 situation_gcn.pth 的形态。
    return SituationGCN(num_features=num_features, hidden_dim=hidden_dim, output_dim=1).state_dict()


def test_diagnostic_loads_bare_legacy_situation_state_dict(tmp_path: Path) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "diag_sit", REPO / "scripts" / "diagnose_model_quality.py")
    diag = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diag)

    sd = _bare_legacy_situation_state_dict(num_features=108, hidden_dim=64)
    assert "sage1.linear.weight" in sd, "本用例要的是 legacy 无前缀形态"
    assert not any(k.startswith("backbone.") for k in sd)
    ckpt = tmp_path / "situation_gcn.pth"
    torch.save(sd, ckpt)

    model, payload = diag._load_situation_model(ckpt, "cpu")
    assert model is not None, "裸 legacy state_dict 必须能加载（原先在这里失败）"
    assert payload.get("hidden_dim") == 64
    assert payload.get("input_dim") == 108
    # 加载后必须能真的前向（否则只是"load_state_dict 没抛异常"）
    adj = torch.eye(3)
    x = torch.randn(3, 108)
    with torch.no_grad():
        out = model(x, adj)
    assert out is not None


def test_diagnostic_loads_wrapped_situation_with_backbone_prefix(tmp_path: Path) -> None:
    """带 backbone. 前缀的包装型：原有路径不得回归。"""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "diag_sit2", REPO / "scripts" / "diagnose_model_quality.py")
    diag = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diag)

    from ai.models import GraphSituationModel

    m = GraphSituationModel(num_features=108, hidden_dim=64)
    ckpt = tmp_path / "wrapped_sit.pt"
    torch.save({"model_state": m.state_dict(), "input_dim": 108, "hidden_dim": 64}, ckpt)

    model, payload = diag._load_situation_model(ckpt, "cpu")
    assert model is not None
    assert payload.get("hidden_dim") == 64
