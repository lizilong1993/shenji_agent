"""I06 步骤 5：奖励消融**实际执行器**（`configs/reward_ablation/*.yaml`）。

## 为什么需要它（本轮查清的两个事实）

**事实一：那 4 组配置**是**死配置**。
静态检查：`reward_shaping` / `penalty_scale` / `reward_no_discount` / `reward_align_only`
在 `training/`、`ai/`、`scripts/` 下被 **0 个 `.py`** 引用 ——
`training/selfplay.py` 里连 `reward_shaping` 都没出现。
**故"跑那 4 组自博弈"会得到 4 份完全相同的输出**，不是消融。

**事实二：配置注释里的"需引擎鉴权"是错的。**
第 49/52 轮已实测：鉴权失败**不阻塞本地推演**。该注释应订正。

## 本执行器做什么（以及**不**做什么 —— 边界必须写清）

**做**：在**同一批真实标签**上，逐组配置重算奖励、给出对比。
数据来源是真实引擎裁决标签（3702 条，reward ∈ [−54, 54]）与真实意图段（6084 条），
故这是**真的读了配置、真的改了奖励**的消融。

**不做**：**不是**"训练后策略胜率"的消融。
要得到"有/无时间折扣 → 训练出的策略胜率高多少"，需要**每组各训练一遍并大量对战**，
成本远高于本轮可承担范围。故本执行器报告的是
**「奖励信号层面的对比」**，并在产出里显式标注这一边界 ——
**不得**把它表述为策略性能结论。

## 任务完成率的口径

`I06` 判据 2 要求「记录任务完成率与平均回报」。
- **平均回报**：由奖励信号算出（本执行器的核心）；
- **任务完成率**：取自真实 `intent_segments.jsonl` 的 `outcome` 字段
  （实测 6084 段中 `achieved` 6059），**与奖励配置无关** ——
  因为不同奖励配置不改变已发生的推演结果。**这一点必须如实说明**，
  否则会让人误以为"改了奖励就改了完成率"。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from training.reward import DEFAULT_WEIGHTS, shape_step_reward  # noqa: E402

LABELS_DEFAULT = ("experiments/shenji_agent/objective_labels/latest/objective_labels.jsonl")
SEGMENTS_GLOB = "Shenji/raw/canonical_v4_candidate/*/intent_segments.jsonl"
CONFIG_DIR = Path("configs/reward_ablation")

# 每组配置如何改权重。**只改权重/开关，不改奖励函数代码** ——
# 否则"消融"会变成"换了另一个函数"，比不出奖励项本身的贡献。
def _apply_config(weights: Dict[str, float], cfg: Dict[str, Any]) -> Dict[str, float]:
    w = dict(weights)
    rs = (cfg.get("reward_shaping") or {})
    if "gamma" in rs:
        w["gamma"] = float(rs["gamma"])
    if "penalty_scale" in rs:
        scale = float(rs["penalty_scale"])
        for k in ("pen_invalid_action", "pen_rule_violation", "pen_friendly_fire",
                  "pen_idle", "pen_friendly_loss", "pen_damage_taken"):
            if k in w:
                w[k] = float(w[k]) * scale
    if rs.get("enabled") is False:
        # 关闭奖励塑形 → 只用对齐项（其余置 0）
        for k in list(w):
            if k.startswith("pen_") or k in ("milestone_bonus",):
                w[k] = 0.0
    return w


def _progress_from_label(label: Dict[str, Any]) -> Dict[str, float]:
    """把标签还原成 `shape_step_reward` 认的 progress 字典。

    只用标签**已有**字段；缺的项不填（`shape_step_reward` 会按 0 处理），
    **不编造**。
    """
    p: Dict[str, float] = {}
    mapping = {
        "damage_dealt": "damage_dealt",
        "damage_taken": "damage_taken",
        "kill_dealt": "kill_dealt",
        "kill_taken": "kill_taken",
        "city_value": "city_value_delta",
    }
    for src, dst in mapping.items():
        v = label.get(src)
        if v not in (None, False):
            try:
                p[dst] = float(v)
            except (TypeError, ValueError):
                pass
    if label.get("occupy_success") is True:
        p["city_value_delta"] = float(label.get("city_value") or 1.0)
    if label.get("distance_delta") not in (None, False):
        try:
            p["distance_delta"] = float(label["distance_delta"])
        except (TypeError, ValueError):
            pass
    return p


def load_configs(config_dir: Path) -> Dict[str, Dict[str, Any]]:
    """读 `configs/reward_ablation/*.yaml`。缺 yaml 库时**明确报错**，不静默跳过。"""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("需要 PyYAML 才能读消融配置: %s" % exc)
    out: Dict[str, Dict[str, Any]] = {}
    if not config_dir.is_dir():
        return out
    for f in sorted(config_dir.glob("*.yaml")):
        out[f.stem] = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
    return out


def evaluate_config(labels: List[Dict[str, Any]], weights: Dict[str, float], *,
                    steps_remaining: int = 0) -> Dict[str, Any]:
    """在一批标签上重算奖励，返回统计量。

    ⚠ `steps_remaining` **必须可配**：我第一版写死 `0`，而 `time_discount(gamma, 0)` = `gamma**0` = **1**，
    于是 **gamma 完全不起作用** —— `reward_no_discount`（gamma=1.0）与 `reward_full`（gamma=0.99）
    得到**逐位相同**的结果，看起来像"时间折扣无影响"，其实是**测不到**。
    现改为可配，并在报告里给出多档 horizon。
    """
    rewards: List[float] = []
    comp_totals: Counter = Counter()
    for label in labels:
        progress = _progress_from_label(label)
        if not progress:
            continue
        total, comps = shape_step_reward(progress, steps_remaining=int(steps_remaining),
                                         weights=weights)
        rewards.append(float(total))
        for k, v in comps.items():
            if isinstance(v, dict) and "value" in v:
                comp_totals[k] += float(v["value"])
    if not rewards:
        return {"steps": 0}
    n = len(rewards)
    pos = sum(1 for r in rewards if r > 0)
    neg = sum(1 for r in rewards if r < 0)
    zero = n - pos - neg
    return {
        "steps": n,
        "sum_return": sum(rewards),
        "mean_reward": sum(rewards) / n,
        "positive_steps": pos,
        "negative_steps": neg,
        "zero_steps": zero,
        "positive_share": pos / n,
        "negative_share": neg / n,
        "max_reward": max(rewards),
        "min_reward": min(rewards),
        "component_totals": {k: round(v, 3) for k, v in comp_totals.most_common()},
    }


def task_completion_rate(segments_glob: str) -> Dict[str, Any]:
    """任务完成率：取自真实 `intent_segments.jsonl` 的 `outcome`。

    **与奖励配置无关** —— 不同奖励配置不改变已发生的推演结果。
    """
    outcomes: Counter = Counter()
    total = 0
    for f in sorted(Path(".").glob(segments_glob)):
        for line in f.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except (TypeError, ValueError):
                continue
            total += 1
            outcomes[str(o.get("outcome") or "unknown")] += 1
    achieved = outcomes.get("achieved", 0)
    return {
        "segments": total,
        "outcomes": dict(outcomes),
        "achieved": achieved,
        "achievement_share": (achieved / total) if total else 0.0,
        "note": "取自真实 intent_segments 的 outcome；**不随奖励配置变化**",
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Run the reward-shaping ablation (I06 step 5).")
    ap.add_argument("--labels", default=LABELS_DEFAULT)
    ap.add_argument("--config-dir", default=str(CONFIG_DIR))
    ap.add_argument("--output", default="experiments/shenji_agent/reward_ablation/report.json")
    args = ap.parse_args(argv)

    labels_path = Path(args.labels)
    if not labels_path.is_file():
        print("labels not found: %s" % labels_path)
        return 2
    labels = [json.loads(l) for l in labels_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    configs = load_configs(Path(args.config_dir))
    if not configs:
        print("no ablation configs under %s" % args.config_dir)
        return 2

    # 多档 horizon：**必须跑 >0 的档**，否则 gamma 无效（见 evaluate_config 的说明）
    horizons = (0, 50, 200, 500)
    results: Dict[str, Any] = {"labels": str(labels_path), "label_count": len(labels),
                               "horizons": list(horizons)}
    all_cfgs = dict(configs)
    all_cfgs["reward_full_default"] = {"note": "DEFAULT_WEIGHTS，无覆盖"}
    for name, cfg in all_cfgs.items():
        weights = _apply_config(DEFAULT_WEIGHTS, cfg)
        per_h = {str(h): evaluate_config(labels, weights, steps_remaining=h)
                 for h in horizons}
        results[name] = {"config": cfg, "weights_used": weights,
                         "metrics": per_h[str(horizons[0])],
                         "metrics_by_horizon": per_h}
    results["task_completion"] = task_completion_rate(SEGMENTS_GLOB)
    results["scope_note"] = (
        "本报告是**奖励信号层面**的消融：同一批真实标签、逐组配置重算奖励。"
        "**不是**训练后策略胜率的消融（后者需每组各训练一遍并大量对战）。"
        "任务完成率取自真实推演结果，**不随奖励配置变化**。")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1, sort_keys=True),
                   encoding="utf-8")

    print("reward-shaping ablation (I06 step 5)")
    print("  labels: %d" % len(labels))
    print()
    print("  %-24s %8s %10s %10s" % ("config", "h", "sum_ret", "mean"))
    for name in sorted(k for k in results if isinstance(results[k], dict)
                       and "metrics_by_horizon" in results[k]):
        for h in horizons:
            m = results[name]["metrics_by_horizon"][str(h)]
            if not m.get("steps"):
                continue
            print("  %-24s %8d %10.1f %10.4f"
                  % (name, h, m["sum_return"], m["mean_reward"]))
    tc = results["task_completion"]
    print()
    print("  task completion (from real segments): %d/%d = %.4f"
          % (tc["achieved"], tc["segments"], tc["achievement_share"]))
    print("  scope: signal-level ablation, NOT policy-performance ablation")
    print("  written: %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
