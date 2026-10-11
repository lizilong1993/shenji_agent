"""I07 / `31007 5.3.6`：检查点的**定期保存**与**触发式保存**。

## 查清的缺口

`docs/checkpoint_policy.md` 已定义四项（命名规范 / 定期保存间隔 / 触发式保存条件 / 保留份数），
但**代码里没有对应实现**：`training/supervised.py` 原本只在训练**结束后**存一次最终权重
（四处 `torch.save`，**均无 epoch 级保存**）。

**即 `31007 5.3.6` 的「定期保存、触发式保存」在本仓库长期是"有规范、无实现"。**

## 本文件验证什么

| 触发 | 期望 |
| --- | --- |
| **定期保存** | 每 `every_n_epochs` 存一档，命名 `<experiment_id>/epoch_{n:04d}.pt` |
| **触发式（指标改善）** | 仅当优于历史最优时存 `best.pt` |
| **触发式（崩溃前）** | 异常路径落 `crash.pt`，且带错误信息 |
| **保留份数** | 超出 `keep_last` 删**最旧定期档**，**不碰** `best.pt`/`crash.pt` |

以及「保存内容」：**必须含 epoch / optimizer / rng / 指标** ——
缺这些的检查点**只能推理、不能恢复**。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from training.checkpoint_writer import (  # noqa: E402
    CHECKPOINT_EPOCH_RE,
    CheckpointWriter,
)


def _torch():
    try:
        import torch  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        pytest.skip("torch not available")
    return torch


def _model_and_opt():
    torch = _torch()
    m = torch.nn.Linear(4, 2)
    return m, torch.optim.Adam(m.parameters())


# ────────────── 定期保存

def test_periodic_save_fires_every_n_epochs(tmp_path: Path) -> None:
    """**定期保存**：间隔为 2 时应只在第 2/4/6 轮落盘。"""
    m, opt = _model_and_opt()
    w = CheckpointWriter(tmp_path, experiment_id="e1", every_n_epochs=2, keep_last=0)
    fired = [e for e in range(1, 7)
             if w.maybe_periodic(epoch=e, model_state=m.state_dict(),
                                 optimizer=opt.state_dict()) is not None]
    assert fired == [2, 4, 6], "定期触发轮次不符: %s" % fired


def test_periodic_usee_the_documented_naming(tmp_path: Path) -> None:
    """命名必须符合 `checkpoint_policy`：`<experiment_id>/epoch_{n:04d}.pt`。"""
    m, opt = _model_and_opt()
    w = CheckpointWriter(tmp_path, experiment_id="expX", every_n_epochs=1, keep_last=0)
    p = w.maybe_periodic(epoch=7, model_state=m.state_dict(),
                         optimizer=opt.state_dict())
    assert p is not None
    assert p.name == CHECKPOINT_EPOCH_RE.format(7) == "epoch_0007.pt", p.name
    assert p.parent == tmp_path / "expX", "未按 experiment_id 分子目录"


def test_periodic_can_be_disabled(tmp_path: Path) -> None:
    """`every_n_epochs <= 0` 应关闭定期保存（政策允许关闭）。"""
    m, opt = _model_and_opt()
    w = CheckpointWriter(tmp_path, experiment_id="e", every_n_epochs=0)
    for e in range(1, 6):
        assert w.maybe_periodic(epoch=e, model_state=m.state_dict(),
                                optimizer=opt.state_dict()) is None


# ────────────── 触发式：指标改善

def test_best_only_fires_on_improvement(tmp_path: Path) -> None:
    """**触发式保存**：只在指标改善时落 `best.pt`，变差时**不落**。"""
    m, opt = _model_and_opt()
    w = CheckpointWriter(tmp_path, experiment_id="e", monitor_name="val_loss",
                         greater_is_better=False)
    assert w.maybe_best(epoch=1, value=0.5, model_state=m.state_dict()) is not None
    assert w.maybe_best(epoch=2, value=0.4, model_state=m.state_dict()) is not None
    assert w.maybe_best(epoch=3, value=0.6, model_state=m.state_dict()) is None, \
        "指标变差却仍触发 best"
    assert w.best_value == 0.4, "best_value 未保持最优"


def test_best_respects_greater_is_better(tmp_path: Path) -> None:
    """方向可配（准确率类指标越大越好）。"""
    m, _ = _model_and_opt()
    w = CheckpointWriter(tmp_path, experiment_id="e", monitor_name="acc",
                         greater_is_better=True)
    assert w.maybe_best(epoch=1, value=0.5, model_state=m.state_dict()) is not None
    assert w.maybe_best(epoch=2, value=0.7, model_state=m.state_dict()) is not None
    assert w.maybe_best(epoch=3, value=0.6, model_state=m.state_dict()) is None
    assert w.best_value == 0.7


# ────────────── 触发式：崩溃前

def test_crash_save_records_the_error(tmp_path: Path) -> None:
    """**崩溃前保存**：落 `crash.pt`，且错误信息可追溯。"""
    m, _ = _model_and_opt()
    w = CheckpointWriter(tmp_path, experiment_id="e")
    p = w.save_on_crash(epoch=5, model_state=m.state_dict(), error="CUDA OOM at step 12")
    assert p.name == "crash.pt"
    meta = json.loads(p.with_suffix(p.suffix + ".json").read_text(encoding="utf-8"))
    assert "CUDA OOM" in meta.get("error", ""), "未记录错误信息"
    assert meta.get("kind") == "crash"


def test_crash_save_does_not_need_a_prior_best(tmp_path: Path) -> None:
    """第一轮就崩也要能存 —— 这正是"崩溃前保存"的价值。"""
    m, _ = _model_and_opt()
    w = CheckpointWriter(tmp_path, experiment_id="e")
    assert w.best_value is None
    p = w.save_on_crash(epoch=1, model_state=m.state_dict(), error="boom")
    assert p.is_file()


# ────────────── 保留份数

def test_keep_last_prunes_only_periodic_archives(tmp_path: Path) -> None:
    """**保留份数**：超出时删**最旧定期档**，`best.pt`/`crash.pt` **不受影响**。

    ⚠ 我连写两版期望都算错了，实测序列说明问题（每存一档即清理一次）：
    `存1→[1]  存2→[1,2]  存3→[2,3]  存4→[3,4]  存5→[4,5]`
    故循环到 5、`keep_last=2` 时，存活的是 **[4, 5]**（最新两份）。
    **实现是对的，错的是我的期望值。**
    """
    m, opt = _model_and_opt()
    w = CheckpointWriter(tmp_path, experiment_id="e", every_n_epochs=1, keep_last=2)
    for e in range(1, 6):
        w.maybe_periodic(epoch=e, model_state=m.state_dict(), optimizer=opt.state_dict())
    w.maybe_best(epoch=1, value=0.1, model_state=m.state_dict())
    w.save_on_crash(epoch=6, model_state=m.state_dict(), error="x")
    names = sorted(p.name for p in (tmp_path / "e").glob("*.pt"))
    periodic = [n for n in names if n.startswith("epoch_")]
    assert len(periodic) == 2, "regular archives not pruned to keep_last: %s" % periodic
    assert periodic == ["epoch_0004.pt", "epoch_0005.pt"], \
        "应保留最新的两份，实际 %s" % periodic
    assert "best.pt" in names and "crash.pt" in names, "误删了 best/crash"


def test_prune_removes_the_oldest_not_an_arbitrary_one(tmp_path: Path) -> None:
    """**语义钉死**：「保留份数」= 保留**最近 N 份**，不是任意 N 份。

    这条独立于上一条，因为"数量对"与"删对了哪个"是两件事。
    """
    m, opt = _model_and_opt()
    w = CheckpointWriter(tmp_path, experiment_id="e", every_n_epochs=1, keep_last=1)
    for e in range(1, 5):
        w.maybe_periodic(epoch=e, model_state=m.state_dict(), optimizer=opt.state_dict())
    left = sorted(p.name for p in (tmp_path / "e").glob("epoch_*.pt"))
    assert left == ["epoch_0004.pt"], "应只留最新一档，实际 %s" % left


def test_keep_last_zero_keeps_everything(tmp_path: Path) -> None:
    """`keep_last <= 0` 不删（保底行为）。"""
    m, opt = _model_and_opt()
    w = CheckpointWriter(tmp_path, experiment_id="e", every_n_epochs=1, keep_last=0)
    for e in range(1, 5):
        w.maybe_periodic(epoch=e, model_state=m.state_dict(), optimizer=opt.state_dict())
    assert len(list((tmp_path / "e").glob("epoch_*.pt"))) == 4


# ────────────── 保存内容（31007 5.3.6）

def test_checkpoint_contains_resume_state(tmp_path: Path) -> None:
    """**保存内容必须足以续跑**：模型 + 优化器 + 轮次 + rng + 指标。

    缺 optimizer/rng 的检查点**只能用于推理，不能用于恢复** ——
    那不算满足 5.3.6 的「保存内容」。
    """
    torch = _torch()
    m, opt = _model_and_opt()
    w = CheckpointWriter(tmp_path, experiment_id="e", monitor_name="val_loss")
    p = w.maybe_best(epoch=3, value=0.25, model_state=m.state_dict(),
                     optimizer=opt.state_dict())
    d = torch.load(p, map_location="cpu")
    assert d.get("model_state") is not None, "无模型权重"
    assert d.get("optimizer") is not None, "无优化器状态 → 不能续跑"
    assert int(d.get("epoch", -1)) == 3, "无轮次"
    rng = d.get("rng_state") or {}
    assert rng.get("torch") is not None, "无 torch rng 状态"
    assert rng.get("python") is not None, "无 python rng 状态"
    assert (d.get("metric") or {}).get("val_loss") == 0.25, "无该档指标"
    assert d.get("experiment_id") == "e"


def test_metadata_json_avoids_loading_the_weights(tmp_path: Path) -> None:
    """指标另存 JSON —— **不加载权重**即可查看某档指标。"""
    m, _ = _model_and_opt()
    w = CheckpointWriter(tmp_path, experiment_id="e", monitor_name="val_loss")
    p = w.maybe_best(epoch=2, value=0.3, model_state=m.state_dict())
    meta = json.loads(p.with_suffix(p.suffix + ".json").read_text(encoding="utf-8"))
    assert meta.get("epoch") == 2
    assert (meta.get("metric") or {}).get("val_loss") == 0.3
    # 元数据**不含**大对象（避免变成第二份权重）
    assert "model_state" not in meta
    assert "rng_state" not in meta


# ────────────── 接线与账目

def test_training_loop_is_wired(tmp_path: Path) -> None:
    """接线必须真的存在（否则只是加了个没人调用的模块）。"""
    src = (REPO / "training" / "supervised.py").read_text(encoding="utf-8")
    assert "CheckpointWriter" in src
    assert "maybe_periodic(" in src
    assert "maybe_best(" in src
    # 检查点失败不得中断训练
    assert "不得中断训练" in src or "skipped at epoch" in src


def test_ledger_reports_what_was_written(tmp_path: Path) -> None:
    """账目须如实列出「存了什么、为何存」，便于训练后核对。"""
    m, _ = _model_and_opt()
    w = CheckpointWriter(tmp_path, experiment_id="e", every_n_epochs=1)
    w.maybe_periodic(epoch=1, model_state=m.state_dict())
    w.maybe_best(epoch=1, value=0.9, model_state=m.state_dict())
    w.save_on_crash(epoch=2, model_state=m.state_dict(), error="x")
    led = w.ledger()
    kinds = [x["kind"] for x in led["written"]]
    assert kinds == ["periodic", "best", "crash"], kinds
    assert led["best_value"] == 0.9
    assert any(f.endswith(".pt") for f in led["files"])
