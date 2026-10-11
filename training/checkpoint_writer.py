"""I07 / `31007 5.3.6`：检查点的**定期保存**与**触发式保存**（此前只有书面策略）。

## 查清的缺口

`docs/checkpoint_policy.md` 已定义四项（命名规范 / 定期保存间隔 / 触发式保存条件 / 保留份数），
但**代码里没有对应实现**：`training/supervised.py` 只在训练**结束后**存一次最终权重
（四处 `torch.save`，均无 epoch 级保存）。

**故 `31007 5.3.6` 的「定期保存、触发式保存」在本仓库是"有规范、无实现"。**

## 本模块做什么

把两者实现为**可独立测试**的组件（`CheckpointWriter`）：

| 能力 | 触发条件 | 落盘名（依 `checkpoint_policy` 命名规范） |
| --- | --- | --- |
| 定期保存 | `every_n_epochs`（`<=0` 关闭） | `<experiment_id>/epoch_{n:04d}.pt` |
| 触发式保存（指标改善） | `monitor_value` 优于历史最优 | `<experiment_id>/best.pt` |
| 触发式保存（崩溃前） | 抛出异常时（`except` 分支） | `<experiment_id>/crash.pt` |
| 保留份数 | 超出 `keep_last` 时删最旧**定期**档 | — |

**为什么独立成模块**：这样可以在**不跑真实训练**的前提下完整验证四种触发条件；
若写进训练循环，验证成本会高到没人愿意跑。

## 保存内容（`31007 5.3.6` 的"保存内容"）

不只权重 —— 还含 `optimizer` 状态、`epoch`、`rng` 状态与**当时的指标**，
使"从中断处续跑"与"追溯该档的指标"都可行。**缺这些的检查点只能用于推理，不能用于恢复。**
"""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

CHECKPOINT_EPOCH_RE = "epoch_{:04d}.pt"


def _rng_state() -> Dict[str, Any]:
    """收集 `rng` 状态（python / numpy / torch），使检查点**可用于续跑**。"""
    state: Dict[str, Any] = {"python": random.getstate()}
    try:
        import numpy as np  # noqa: PLC0415

        state["numpy"] = np.random.get_state()
    except Exception:  # noqa: BLE001
        state["numpy"] = None
    try:
        import torch  # noqa: PLC0415

        state["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
    except Exception:  # noqa: BLE001
        state["torch"] = None
    return state


class CheckpointWriter:
    """按 `docs/checkpoint_policy.md` 落盘检查点。

    Parameters
    ----------
    root:
        检查点根目录（通常是 `run_dir / "checkpoints"`）。
    experiment_id:
        命名规范的前缀（`checkpoint_policy` 第 1 节）。
    every_n_epochs:
        定期保存间隔；`<=0` 表示关闭定期保存。
    keep_last:
        定期档保留份数（超出删最旧）。`<=0` 表示不删。
    monitor_name / greater_is_better:
        触发式保存的指标方向（如 `val_loss` 越小越好）。
    """

    def __init__(self, root: Path, *, experiment_id: str,
                 every_n_epochs: int = 5, keep_last: int = 3,
                 monitor_name: str = "val_loss",
                 greater_is_better: bool = False) -> None:
        self.root = Path(root)
        self.experiment_id = str(experiment_id)
        self.every_n_epochs = int(every_n_epochs)
        self.keep_last = int(keep_last)
        self.monitor_name = str(monitor_name)
        self.greater_is_better = bool(greater_is_better)
        self.best_value: Optional[float] = None
        self.written: List[Dict[str, Any]] = []

    # ── 路径
    @property
    def dir(self) -> Path:
        return self.root / self.experiment_id

    def _path(self, name: str) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        return self.dir / name

    # ── 内部：统一落盘
    def _save(self, name: str, *, kind: str, payload: Dict[str, Any],
              epoch: int, metric: Optional[float]) -> Path:
        p = self._path(name)
        body = dict(payload)
        body.update({"epoch": int(epoch), "kind": kind,
                     "experiment_id": self.experiment_id,
                     "rng_state": _rng_state()})
        if metric is not None:
            body["metric"] = {self.monitor_name: float(metric)}
        # 指标与元数据另存一份 JSON —— 便于**不加载权重**就能查看某档的指标
        meta_p = p.with_suffix(p.suffix + ".json")
        meta = {k: v for k, v in body.items()
                if k not in ("model_state", "optimizer", "rng_state")}
        with open(meta_p, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2, default=str)
        try:
            import torch  # noqa: PLC0415

            torch.save(body, p)
        except Exception:  # noqa: BLE001 - 无 torch 时退化为 JSON（仍可用于测试）
            with open(p.with_suffix(".jsononly"), "w", encoding="utf-8", newline="\n") as fh:
                json.dump({k: str(v) for k, v in body.items()}, fh, ensure_ascii=False)
        self.written.append({"kind": kind, "name": name, "epoch": int(epoch),
                             "metric": metric})
        return p

    # ── 三种触发
    def maybe_periodic(self, *, epoch: int, model_state: Any,
                       optimizer: Any = None) -> Optional[Path]:
        """**定期保存**：每 `every_n_epochs` 个 epoch 存一档，并按 `keep_last` 清理。"""
        if self.every_n_epochs <= 0:
            return None
        if epoch % self.every_n_epochs != 0:
            return None
        p = self._save(CHECKPOINT_EPOCH_RE.format(epoch), kind="periodic",
                       payload={"model_state": model_state, "optimizer": optimizer},
                       epoch=epoch, metric=None)
        self._enforce_keep_last()
        return p

    def maybe_best(self, *, epoch: int, value: float, model_state: Any,
                   optimizer: Any = None) -> Optional[Path]:
        """**触发式保存（指标改善）**：仅当优于历史最优时存 `best.pt`。"""
        v = float(value)
        improved = (self.best_value is None
                    or (v > self.best_value if self.greater_is_better
                        else v < self.best_value))
        if not improved:
            return None
        self.best_value = v
        return self._save("best.pt", kind="best",
                          payload={"model_state": model_state, "optimizer": optimizer},
                          epoch=epoch, metric=v)

    def save_on_crash(self, *, epoch: int, model_state: Any,
                      optimizer: Any = None, error: str = "") -> Path:
        """**触发式保存（崩溃前）**：把当前状态落盘，避免整轮训练白跑。"""
        return self._save("crash.pt", kind="crash",
                          payload={"model_state": model_state, "optimizer": optimizer,
                                   "error": str(error)[:2000]},
                          epoch=epoch, metric=self.best_value)

    def _enforce_keep_last(self) -> List[str]:
        """按 `keep_last` 删**最旧的定期档**（不碰 `best.pt` / `crash.pt`）。"""
        if self.keep_last <= 0 or not self.dir.is_dir():
            return []
        periodic = sorted(self.dir.glob("epoch_*.pt"))
        removed = []
        while len(periodic) > self.keep_last:
            victim = periodic.pop(0)
            for extra in (victim, victim.with_suffix(victim.suffix + ".json")):
                try:
                    extra.unlink()
                    removed.append(extra.name)
                except OSError:
                    pass
        return removed

    # ── 账目
    def ledger(self) -> Dict[str, Any]:
        """落盘账目 —— 供训练与测试核对"存了什么、为何存"。"""
        return {
            "experiment_id": self.experiment_id,
            "root": str(self.root),
            "every_n_epochs": self.every_n_epochs,
            "keep_last": self.keep_last,
            "monitor_name": self.monitor_name,
            "greater_is_better": self.greater_is_better,
            "best_value": self.best_value,
            "written": list(self.written),
            "files": sorted(p.name for p in self.dir.glob("*")) if self.dir.is_dir() else [],
        }


def purge_checkpoint_dir(path: Path) -> None:
    """删除整个检查点目录（仅用于测试与显式清理，**不**由训练自动调用）。"""
    p = Path(path)
    if p.is_dir():
        shutil.rmtree(p)
