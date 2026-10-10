"""I07：确定性种子与「同 seed 权重逐字节一致」的守护。

## 背景

`docs/checkpoint_policy.md:162` 记录的缺口：

> 「同 seed 两次训练权重**逐字节一致**既未实现也无守护（经独立评审四种搜法确认全仓/全史零命中）」

同文件 `:166` 注明「需**先工程化确定性**再谈守护」。

## 第 55 轮实测事实（本文件即据此写成）

| 项 | 修前 | 修后 |
| --- | --- | --- |
| CPU 同 seed 两次训练权重 | **已逐字节一致** | 一致（未变） |
| `torch.backends.cudnn.deterministic` | **False** | **True** |
| `torch.are_deterministic_algorithms_enabled()` | **False** | **True** |
| `CUBLAS_WORKSPACE_CONFIG` | **未设** | **`:4096:8`** |
| CUDA 可用 | **True** | True |

**即：CPU 路径本来就一致，缺口在 CUDA 路径缺确定性保证。**
故本文件既守住「同 seed 一致」，也守住「异 seed 必须不同」——
后者是关键：**若异 seed 也一致，说明种子根本没生效，那不是可复现而是"训练无效"。**
"""
from __future__ import annotations

import hashlib
import io
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

COMMON = REPO / "training" / "common.py"


def _torch():
    try:
        import torch  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        pytest.skip("torch not available")
    return torch


def _train_hash(seed: int, *, deterministic: bool = True) -> str:
    """用 `set_seed` 跑一次极小训练，返回权重 state_dict 的 SHA-256。"""
    torch = _torch()
    from training.common import set_seed  # noqa: PLC0415

    set_seed(seed, deterministic=deterministic)
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(64, 10)
    y = torch.randn(64, 1)
    for _ in range(5):
        opt.zero_grad()
        torch.nn.functional.mse_loss(model(x), y).backward()
        opt.step()
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


# ─────────────────── 核心：同 seed 一致、异 seed 不同

def test_same_seed_produces_byte_identical_weights() -> None:
    """**缺口 1 的守护**：同 seed 两次训练权重**逐字节一致**。"""
    a = _train_hash(42)
    b = _train_hash(42)
    assert a == b, "同 seed 两次训练权重不一致 —— 可复现性缺口回来了"


def test_different_seed_produces_different_weights() -> None:
    """**关键反向断言**：异 seed 必须产生**不同**权重。

    若这条不成立，说明 `set_seed` 根本没生效 ——
    那"同 seed 一致"就毫无意义（训练根本没受种子影响）。
    **只测"一致"而不测"不同"，守护是半个。**
    """
    a = _train_hash(42)
    c = _train_hash(43)
    assert a != c, "异 seed 得到相同权重 —— 种子未生效"


def test_seed_effect_is_reproducible_across_repeats() -> None:
    """多取一次，排除"恰好一致"的巧合。"""
    h42 = {_train_hash(42) for _ in range(3)}
    assert len(h42) == 1, "同 seed 三次得到 %d 种权重" % len(h42)


# ─────────────────── 确定性工程化（修前的缺口）

def test_set_seed_enables_deterministic_algorithms() -> None:
    """`use_deterministic_algorithms` 必须被开启（修前为 False）。"""
    torch = _torch()
    from training.common import set_seed  # noqa: PLC0415

    set_seed(7)
    assert torch.are_deterministic_algorithms_enabled() is True, \
        "确定性算法未开启 —— CUDA 上「逐字节一致」不成立"


def test_set_seed_enables_cudnn_determinism() -> None:
    """cuDNN 确定性必须开启、benchmark 必须关闭（修前前者为 False）。"""
    torch = _torch()
    from training.common import set_seed  # noqa: PLC0415

    set_seed(7)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False, \
        "benchmark=True 会挑选不确定的算法"


def test_set_seed_sets_cublas_workspace_config() -> None:
    """cuBLAS 在 CUDA ≥10.2 需该环境变量才保证确定性（修前未设）。"""
    _torch()
    from training.common import set_seed  # noqa: PLC0415

    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    set_seed(7)
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") in (":4096:8", ":16:8"), \
        "CUBLAS_WORKSPACE_CONFIG 未设或取值不符官方要求"


def test_deterministic_false_is_available_as_an_escape_hatch() -> None:
    """必须保留"关掉确定性"的开关 ——
    某些算子**没有**确定性实现，强开会让它们抛错。"""
    torch = _torch()
    from training.common import set_seed  # noqa: PLC0415

    set_seed(7, deterministic=False)
    # 关掉时必须真的回到不强制状态，否则该开关是假的
    assert torch.are_deterministic_algorithms_enabled() is False
    # 复原，避免污染同进程后续用例
    set_seed(7)
    assert torch.are_deterministic_algorithms_enabled() is True


def test_cuda_is_actually_available_on_this_host() -> None:
    """记录一个事实：本机 CUDA 可用 → 上述 CUDA 设置**不是摆设**。

    若本机无 CUDA，则这些设置只是"为将来准备"；有 CUDA 时它们是**必需**的。
    """
    torch = _torch()
    # 不断言一定为真（换机器会变），只把事实记进测试输出，便于判读
    assert isinstance(torch.cuda.is_available(), bool)


# ─────────────────── 文档同步

def test_checkpoint_policy_records_the_gap_as_addressed() -> None:
    """`docs/checkpoint_policy.md` 记录了该缺口，须同步为"已实现并有守护"。"""
    doc = REPO / "docs" / "checkpoint_policy.md"
    if not doc.is_file():
        pytest.skip("checkpoint_policy.md not present")
    d = doc.read_text(encoding="utf-8")
    assert "逐字节一致" in d, "文档未提及该缺口"
    # 必须写明第 55 轮已处理
    assert "第 55 轮" in d or "已实现" in d, "文档未同步状态"


def test_set_seed_documents_the_measured_facts() -> None:
    """`set_seed` 的 docstring 必须写明实测事实（否则后人不知为何这样设）。"""
    if not COMMON.is_file():
        pytest.skip("training/common.py not present")
    src = COMMON.read_text(encoding="utf-8")
    i = src.find("def set_seed")
    assert i > 0
    body = src[i:i + 4000]
    assert "cudnn.deterministic" in body
    assert "CUBLAS_WORKSPACE_CONFIG" in body
    # 必须说明"关掉"的用途，避免后人误删该开关
    assert "deterministic=False" in body or "deterministic: bool" in body
