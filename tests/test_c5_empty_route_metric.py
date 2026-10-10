"""C5 步骤 5：空路径率指标（training/evaluate.py 与 training/selfplay.py 同口径）。

为什么单独钉住：`empty_route_count` 这类**原始计数**早就有，但**率**没有 ——
报告里那句「15,506/23,500 = 66%」是**事后手算**的。这条测试保证率出现在产物里，
且分母处理、向后兼容与「先求和再算率」的口径不被以后改回去。
"""
from __future__ import annotations

from training.evaluate import _merge_policy_stats
from training.selfplay import _merge_policy_stats as selfplay_merge


def _match(attempts: int, empty: int, reachable_total: int, by_move_type=None):
    """evaluate 的入参层级：move_route_aggregate 在 candidate_policy_stats 下。"""
    return {
        "candidate_policy_stats": {
            "counts": {"move": {"total": attempts, "learned": attempts},
                       "shoot": {"total": 0, "learned": 0}},
            "move_route_aggregate": {
                "total_attempts": attempts,
                "empty_route_count": empty,
                "reachable_city_total": reachable_total,
                "by_move_type": by_move_type or {},
                "route_failure_reasons": {"empty_route": empty} if empty else {},
            },
        }
    }


def _policy_stats(attempts: int, empty: int, reachable_total: int, by_move_type=None):
    """selfplay 的入参层级：直接吃 policy_stats 本身。"""
    return _match(attempts, empty, reachable_total, by_move_type)["candidate_policy_stats"]


def test_empty_route_rate_matches_report_figure() -> None:
    """报告里的 15,506/23,500 = 66% 应能由产物直接算出。"""
    agg = _merge_policy_stats([_match(23500, 15506, 18994)])["move_route_aggregate"]

    assert agg["total_attempts"] == 23500
    assert agg["empty_route_count"] == 15506
    assert abs(agg["empty_route_rate"] - 15506 / 23500) < 1e-12
    assert round(agg["empty_route_rate"], 2) == 0.66


def test_rate_is_recomputed_from_totals_not_averaged() -> None:
    """多场比赛应先求和再算率；若改成「各场率的平均」会得到不同的值。"""
    agg = _merge_policy_stats([_match(100, 50, 100), _match(300, 30, 300)])["move_route_aggregate"]

    assert agg["total_attempts"] == 400
    assert agg["empty_route_count"] == 80
    assert abs(agg["empty_route_rate"] - 0.2) < 1e-12
    # 各场率的平均会是 (0.5 + 0.1) / 2 = 0.3，与之不同
    assert abs(agg["empty_route_rate"] - 0.3) > 1e-6


def test_rates_are_zero_when_denominator_is_zero() -> None:
    """空输入不得抛异常、不得产生 NaN。"""
    agg = _merge_policy_stats([])["move_route_aggregate"]

    assert agg["total_attempts"] == 0
    assert agg["empty_route_rate"] == 0.0
    assert agg["no_reachable_city_rate"] == 0.0
    assert agg["reachable_city_mean"] == 0.0


def test_existing_keys_are_preserved() -> None:
    """只新增字段：既有子键必须还在（tests 与 generate_paper_prewrite_package.py 依赖它们）。"""
    agg = _merge_policy_stats([_match(10, 4, 9, {"March": {"total": 10, "empty": 4}})])["move_route_aggregate"]

    for key in ("total_attempts", "empty_route_count", "reachable_city_total",
                "by_move_type", "route_failure_reasons"):
        assert key in agg, key
    assert agg["route_failure_reasons"]["empty_route"] == 4


def test_per_move_type_empty_rate() -> None:
    """分移动类型的空路径率用于定位「哪种机动方式在失路」。"""
    agg = _merge_policy_stats([
        _match(10, 4, 10, {"March": {"total": 8, "empty": 4}, "Fly": {"total": 2, "empty": 0}})
    ])["move_route_aggregate"]

    assert abs(agg["by_move_type"]["March"]["empty_rate"] - 0.5) < 1e-12
    assert agg["by_move_type"]["Fly"]["empty_rate"] == 0.0


def test_no_reachable_city_counts_use_attempt_denominator() -> None:
    """`reachable_city_total` 是「每次尝试可达城市数之和」，故与尝试数相减得到无城可达的尝试数。

    这里刻意取一个 > attempts 的值，验证 max(0, ...) 不会产出负数。
    """
    agg = _merge_policy_stats([_match(100, 10, 250)])["move_route_aggregate"]

    assert agg["no_reachable_city_count"] == 0
    assert agg["reachable_city_mean"] == 2.5


def test_selfplay_merge_uses_same_definition() -> None:
    """运行期产物（selfplay）与评估产物（evaluate）必须同口径。"""
    ev = _merge_policy_stats([_match(23500, 15506, 18994)])["move_route_aggregate"]
    sp = selfplay_merge(_policy_stats(23500, 15506, 18994))["move_route_aggregate"]

    assert abs(ev["empty_route_rate"] - sp["empty_route_rate"]) < 1e-12
    assert ev["no_reachable_city_count"] == sp["no_reachable_city_count"]
    assert ev["reachable_city_mean"] == sp["reachable_city_mean"]
