"""滑动基线异常检测。

回答的问题只有一个：**此刻的量比平时反常吗**。不是预测，也不是告警决策——
它输出"偏离基线多少个标准差"，是否值得提示由调用方决定。

用 Welford 的在线均值/方差，而不是先攒一批再算：情报是流式进来的，
每来一条就更新一次，不需要保留历史样本，也不会因为样本变多而变慢。

几条刻意的取舍：

* **冷启动不报异常**。样本数低于 ``min_samples`` 时不判定——用两个点算出
  的均值去判断第三个点，报出来的"异常"基本是噪声。宁可晚几条，不可误报。
* **方差下限**。全部样本完全相同时方差为 0，z-score 会除零；给一个下限，
  让"一直稳定在 5，忽然变成 8"仍然能被识别为异常。
* **方向可见**。返回 z 的符号，调用方据此区分"突增"与"骤降"——两者在
  情报上意味着完全不同的事。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# 冷启动门槛：少于这么多样本不判定。
DEFAULT_MIN_SAMPLES = 8
# z 阈值。三个标准差是按正态分布约 0.3% 的双侧概率取的常规值。
DEFAULT_THRESHOLD = 3.0


@dataclass
class Baseline:
    """一个被观测序列的在线统计量。"""

    key: str = ""
    mean: float = 0.0
    m2: float = 0.0
    count: int = 0
    min_value: float = float("inf")
    max_value: float = float("-inf")
    # 方差下限：默认取均值的 1%，均值为 0 时退回一个绝对小量。
    floor_ratio: float = 0.01
    absolute_floor: float = 0.5

    def update(self, value: Any) -> Optional[float]:
        """加入一个样本，返回加入**之前**的 z-score（未知样本时为 None）。

        先算再更新：否则每个点都会把自己的均值往自己身上拉，z 被系统性压低，
        真正的尖峰反而测不出来。
        """
        try:
            x = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(x):
            return None
        z = self.z_score(x)
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (x - self.mean)
        self.min_value = min(self.min_value, x)
        self.max_value = max(self.max_value, x)
        return z

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)

    @property
    def stdev(self) -> float:
        return math.sqrt(max(0.0, self.variance))

    def floor(self) -> float:
        return max(self.absolute_floor, abs(self.mean) * self.floor_ratio)

    def z_score(self, value: Any) -> Optional[float]:
        """给定值相对**当前**基线的 z-score；样本不足时返回 None。"""
        try:
            x = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(x) or self.count < 2:
            return None
        sigma = max(self.stdev, self.floor())
        if sigma <= 0:
            return None
        return (x - self.mean) / sigma

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "count": self.count,
            "mean": round(self.mean, 4),
            "stdev": round(self.stdev, 4),
            "min": None if self.count == 0 else round(self.min_value, 4),
            "max": None if self.count == 0 else round(self.max_value, 4),
        }


@dataclass
class Anomaly:
    key: str
    value: float
    z: float
    direction: str          # "up" | "down"
    baseline: Baseline

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "value": round(self.value, 4),
            "z": round(self.z, 2),
            "direction": self.direction,
            "baseline_mean": round(self.baseline.mean, 4),
            "baseline_stdev": round(self.baseline.stdev, 4),
        }


@dataclass
class BaselineTracker:
    """按 key 维护一组基线（例如"每个网格一个"）。"""

    min_samples: int = DEFAULT_MIN_SAMPLES
    threshold: float = DEFAULT_THRESHOLD
    series: Dict[str, Baseline] = field(default_factory=dict)

    def observe(self, key: str, value: Any) -> Optional[Anomaly]:
        """记录一个样本；若越过阈值返回异常，否则 None。"""
        key = str(key or "")
        baseline = self.series.get(key)
        if baseline is None:
            baseline = Baseline(key=key)
            self.series[key] = baseline
        # 冷启动：样本不足时不判定，但**仍然记录**——否则永远攒不够。
        z = None
        if baseline.count >= self.min_samples:
            z = baseline.z_score(value)
        baseline.update(value)
        if z is None or abs(z) < self.threshold:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return Anomaly(key=key, value=numeric, z=z,
                       direction="up" if z > 0 else "down", baseline=baseline)

    def snapshot(self) -> Dict[str, dict]:
        return {key: base.as_dict() for key, base in sorted(self.series.items())}

    def reset(self) -> None:
        self.series.clear()
