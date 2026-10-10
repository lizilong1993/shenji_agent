"""按对抗强度生成标准化测试用例（`I16` / `Q7` 所需的 `gen_eval_cases.py`）。

## 为什么有这个文件

第 47 轮核对清单路径时发现：`scripts/gen_eval_cases.py` **在工作区与远程都不存在**。
而 `31007 6.1.2 a)` 要求：

> 分别面向不同对抗强度生成想定，**每种对抗强度下生成超过 100 种标准化测试用例**，
> 覆盖典型决策边界，并用于算法泛化性验证。

第 46 轮（`docs/eval_intensity_spec.md`）已产出**三档定义**（L1/L2/L3），
但**生成器是空的** —— 所以「每档 >100 用例」既缺工具也无法核对。
本模块补上工具，并**把「>100」变成可执行的守门**。

## 本模块**不做什么**（避免把"产出描述"说成"跑过推演"）

- **不调用引擎**：本模块只生成**用例描述（case manifest）**，
  真正的推演仍由 `shenji_dataset.collect_cli` 执行。
- **不生成 segments**：用例 ≠ 已采数据。生成 300 个用例**不等于**采到 300 段语料。

> 故本模块的产出必须表述为「**用例集已定义**」，**不得**表述为「评估已完成」。

## 强度三档（依据 `docs/eval_intensity_spec.md` 与 31007 6.1.2 b) 原文措辞）

| 档 | 名称 | 扰动 |
| --- | --- | --- |
| `L1` | 常规对抗 | 无（标准参数） |
| `L2` | 高强度对抗 | **战场迷雾** + **通信丢包** + 观测延迟 |
| `L3` | 极限施压 | L2 全部 + **电子干扰** + **后勤断链** + 时间压缩 |

## 设计上的两个要点

1. **尺度用 log 而不是等差**：原文说「**逐步提升**……最终在极限施压场景」，
   而「探测崩溃阈值」需要靠近崩溃区间才有分辨率 —— 故 L2/L3 的取值由对数尺度算得。
2. **决策边界必须显式**：原文要求「覆盖**典型决策边界**」，故每个用例带
   `decision_boundaries` 标签；若某档一个边界标签都没有，守门会失败。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

SCHEMA_VERSION = "eval_cases/1"

# ── 三档（与 docs/eval_intensity_spec.md 一致）─────────────────────────────
LEVELS = ("L1", "L2", "L3")

LEVEL_LABELS = {
    "L1": "常规对抗",
    "L2": "高强度对抗",
    "L3": "极限施压",
}

# 每档的扰动参数默认值。**依据原文措辞**（见模块文档），非自创概念。
# 数值本身是**本仓工程取值**，标准未规定 —— 故 `describe_parameters()` 会如实声明。
PERTURBATIONS: Dict[str, Dict[str, Any]] = {
    "L1": {
        "fog_of_war": 0.0,          # 战场迷雾：观测被遮蔽的比例
        "packet_loss": 0.0,         # 通信丢包率
        "obs_latency_steps": 0,     # 观测延迟（步）
        "jamming": 0.0,             # 电子干扰强度
        "logistics_supply": 1.0,    # 后勤保障系数（1.0 = 无断链）
        "time_compression": 1.0,    # 时间压缩（<1 = 更少决策时间）
    },
    "L2": {
        "fog_of_war": 0.35,
        "packet_loss": 0.10,
        "obs_latency_steps": 2,
        "jamming": 0.0,
        "logistics_supply": 1.0,
        "time_compression": 1.0,
    },
    "L3": {
        "fog_of_war": 0.60,
        "packet_loss": 0.25,
        "obs_latency_steps": 5,
        "jamming": 0.50,
        "logistics_supply": 0.50,
        "time_compression": 0.75,
    },
}

# 「典型决策边界」（原文 6.1.2 a) 要求覆盖）。每档至少要覆盖若干条。
DECISION_BOUNDARIES = (
    "force_ratio_near_parity",      # 兵力比接近 1:1
    "ammo_low",                     # 弹药接近耗尽
    "objective_contested",          # 夺控点易手
    "comms_degraded",               # 通信受限
    "supply_critical",              # 补给临界
    "time_pressure",                # 时间压力
)

MIN_CASES_PER_LEVEL = 101          # 原文「**超过** 100 种」→ 101 才叫超过


# ─────────────────────────────────────────────────────────────────────────
# 输入读取
# ─────────────────────────────────────────────────────────────────────────

def load_intents(ontology_path: Path) -> List[Dict[str, Any]]:
    """读 22 个意图（含 parent_code），供用例按意图铺开。"""
    d = json.loads(Path(ontology_path).read_text(encoding="utf-8"))
    intents = d.get("intents") or []
    out = []
    for it in intents:
        if isinstance(it, dict) and it.get("code"):
            out.append({
                "code": it["code"],
                "parent_code": it.get("parent_code") or "",
                "name_zh": it.get("name_zh") or "",
            })
        elif isinstance(it, str):
            out.append({"code": it, "parent_code": "", "name_zh": ""})
    return sorted(out, key=lambda x: x["code"])


def load_plan(plan_path: Path) -> Dict[str, Any]:
    return json.loads(Path(plan_path).read_text(encoding="utf-8"))


def _stable_seed(*parts: Any) -> int:
    """由内容派生**稳定**种子 —— 同样的输入必须得到同样的用例（可复现）。"""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:8], 16)


# ─────────────────────────────────────────────────────────────────────────
# 生成
# ─────────────────────────────────────────────────────────────────────────

def generate_cases(*,
                   intents: Sequence[Dict[str, Any]],
                   level: str,
                   min_cases: int = MIN_CASES_PER_LEVEL,
                   boundaries: Sequence[str] = DECISION_BOUNDARIES,
                   map_families: Optional[Sequence[str]] = None,
                   perturbations: Optional[Dict[str, Any]] = None,
                   ) -> List[Dict[str, Any]]:
    """为**单档**生成 ≥`min_cases` 个用例，按意图**均匀铺开**。

    用例是**描述**（想定变体 + 扰动 + 边界标签），**不是**已跑出的推演结果。
    """
    if level not in LEVELS:
        raise ValueError("未知强度档: %r（应为 %s）" % (level, list(LEVELS)))
    if not intents:
        raise ValueError("意图列表为空 —— 无法铺开用例")

    perturb = dict(PERTURBATIONS[level])
    if perturbations:
        perturb.update(perturbations)

    maps = list(map_families or ("map_123",))
    per_intent = -(-min_cases // len(intents))        # 向上取整
    cases: List[Dict[str, Any]] = []
    # ⚠ 边界与地图用**全局计数器**铺开，不用 per-intent 的 k。
    # 我第一版用 `k % len(boundaries)`，而 per_intent=5、boundaries=6 时
    # k 只取 0..4 → **第 6 条边界永远取不到**，守门如实报 FAIL。
    # 全局计数保证「每档覆盖全部边界」与 per_intent 的取值无关。
    n = 0
    for intent in intents:
        for k in range(per_intent):
            mp = maps[n % len(maps)]
            bnd = boundaries[n % len(boundaries)]
            n += 1
            seed = _stable_seed(level, intent["code"], mp, k, bnd)
            cases.append({
                "case_id": "%s-%s-%03d" % (level, intent["code"], k),
                "intensity_level": level,
                "intensity_label": LEVEL_LABELS[level],
                "intent_code": intent["code"],
                "parent_code": intent.get("parent_code", ""),
                "map_family": mp,
                "seed": seed,
                "max_steps": 1000,
                "decision_boundaries": [bnd],
                "perturbations": perturb,
            })
    return cases


def generate_all(*,
                 ontology_path: Path,
                 plan_path: Path,
                 min_cases: int = MIN_CASES_PER_LEVEL,
                 ) -> Dict[str, Any]:
    """生成三档全部用例，返回可直接落盘的清单。"""
    intents = load_intents(ontology_path)
    plan = load_plan(plan_path)
    qt = plan.get("quality_targets") or {}
    n_maps = int(qt.get("minimum_distinct_map_families") or 1)
    maps = ["map_%d" % (123 + i) for i in range(max(1, n_maps))]

    by_level = {}
    for lv in LEVELS:
        by_level[lv] = generate_cases(intents=intents, level=lv,
                                      min_cases=min_cases, map_families=maps)
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": plan.get("dataset") or "Shenji",
        "source_plan_version": plan.get("version"),
        "intent_count": len(intents),
        "min_cases_per_level": min_cases,
        "level_labels": LEVEL_LABELS,
        "perturbations": PERTURBATIONS,
        "cases": {lv: by_level[lv] for lv in LEVELS},
    }


# ─────────────────────────────────────────────────────────────────────────
# 守门（把 6.1.2 a) 的「>100」变成可执行判据）
# ─────────────────────────────────────────────────────────────────────────

def validate_cases(bundle: Dict[str, Any],
                   min_cases: int = MIN_CASES_PER_LEVEL,
                   boundary_labels: Sequence[str] = DECISION_BOUNDARIES,
                   ) -> Dict[str, Any]:
    """校验用例集是否满足 6.1.2 a)。

    返回 `{"ok": bool, "gates": [...], "counts": {...}}`，**不改写输入**。
    """
    gates: List[Dict[str, Any]] = []

    def gate(name: str, ok: bool, detail: str) -> None:
        gates.append({"gate": name, "ok": bool(ok), "detail": detail})

    cases = bundle.get("cases") or {}
    counts = {}
    for lv in LEVELS:
        cs = cases.get(lv) or []
        counts[lv] = len(cs)
        # ① 每档 >100（原文「超过 100 种」）
        gate("count:%s" % lv, len(cs) > 100,
             "%d 个用例（要求 >100）" % len(cs))
        # ② 覆盖典型决策边界
        seen = {b for c in cs for b in (c.get("decision_boundaries") or [])}
        missing = [b for b in boundary_labels if b not in seen]
        gate("boundaries:%s" % lv, not missing,
             "覆盖 %d/%d 条边界%s" % (len(boundary_labels) - len(missing),
                                      len(boundary_labels),
                                      ("，缺 %s" % missing) if missing else ""))
        # ③ 扰动必须与该档定义一致
        bad = [c["case_id"] for c in cs
               if c.get("perturbations") != PERTURBATIONS[lv]]
        gate("perturbations:%s" % lv, not bad,
             "扰动一致" if not bad else "%d 个用例的扰动与档位不符" % len(bad))
        # ④ case_id 唯一（否则用例集有重复）
        ids = [c.get("case_id") for c in cs]
        gate("unique_ids:%s" % lv, len(ids) == len(set(ids)),
             "%d 个 id / %d 个唯一" % (len(ids), len(set(ids))))

    # ⑤ 档间不得有重复 case_id
    allids = [c.get("case_id") for lv in LEVELS for c in (cases.get(lv) or [])]
    gate("unique_ids:all", len(allids) == len(set(allids)),
         "%d 个 id / %d 个唯一" % (len(allids), len(set(allids))))

    return {
        "ok": all(g["ok"] for g in gates),
        "gates": gates,
        "counts": counts,
        "total": sum(counts.values()),
        "note": ("用例数是**生成的用例描述**数；"
                 "**不等于**已采集的 segment 数 —— 生成用例不等于跑过推演"),
    }


def describe_parameters() -> Dict[str, Any]:
    """如实声明：三档的**阈值与扰动参数是本仓工程取值**，标准未给具体数值。"""
    return {
        "levels": list(LEVELS),
        "perturbations": PERTURBATIONS,
        "min_cases_per_level": MIN_CASES_PER_LEVEL,
        "thresholds_note": (
            "三档的扰动数值与「>100」的本仓下限为**工程取值**；"
            "31007 6.1.2 a) 只规定「超过 100 种标准化测试用例」，"
            "**未规定**迷雾/丢包/干扰的具体数值，**未规定**蒙特卡洛 N。"
            "故不得把本表数值表述为「标准规定的阈值」。"),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="gen_eval_cases",
        description="Generate and validate intensity-tiered evaluation cases (I16/Q7).")
    ap.add_argument("--ontology", default="Shenji/ontology/intent_ontology_v2.json")
    ap.add_argument("--plan", default="Shenji/configs/collection_plan_v1.json")
    ap.add_argument("--output", default=None)
    ap.add_argument("--min-cases", type=int, default=MIN_CASES_PER_LEVEL)
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args(argv)

    onto, plan = Path(args.ontology), Path(args.plan)
    if not onto.is_file() or not plan.is_file():
        print("missing input: %s / %s" % (onto, plan))
        return 2

    if args.validate_only:
        if not args.output or not Path(args.output).is_file():
            print("--validate-only needs --output pointing at an existing bundle")
            return 2
        bundle = json.loads(Path(args.output).read_text(encoding="utf-8"))
    else:
        bundle = generate_all(ontology_path=onto, plan_path=plan,
                              min_cases=args.min_cases)

    verdict = validate_cases(bundle, min_cases=args.min_cases)
    print("intensity-tiered evaluation cases")
    print("  intents: %s" % bundle.get("intent_count"))
    for lv in LEVELS:
        print("  %-3s %-8s %d 用例" % (lv, LEVEL_LABELS[lv], verdict["counts"][lv]))
    for g in verdict["gates"]:
        if not g["ok"]:
            print("  FAIL %-22s %s" % (g["gate"], g["detail"]))
    print("  RESULT: %s (total %d)" % ("OK" if verdict["ok"] else "FAIL", verdict["total"]))
    print("  note: %s" % verdict["note"])

    if args.output and not args.validate_only:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(bundle, fh, ensure_ascii=False, indent=1, sort_keys=True)
            fh.write("\n")
        print("  written: %s" % out)
    return 0 if verdict["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
