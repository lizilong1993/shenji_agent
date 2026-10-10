"""评估有效性守门：三层框架 + 平局率 / 区分度前置指标（P1 步骤 3/4/5）。

## 标准依据（**原文**，非索引摘要）

`.scratch/std/s31007.txt`：

- **6.1.1 评估方法**（`:663-677`）——原文明确要求：
  > 「本标准采用**仿真环境验证-实际推演校验-横向对比分析**的三层递进框架，
  >  覆盖从理论假设到实践落地的全链路验证需求。」
  >
  > · 仿真测试：「设置不同的想定和对抗场景……评估其在各种情况下的性能表现」
  > · 实际推演验证：「组织实际的兵棋推演活动，邀请专业的推演人员参与」
  > · 对比评估：「将开发的算法与模型**与已有的同类算法和模型进行对比**……
  >   可从决策速度、准确性、适应性、可扩展性等多个方面进行」

- **6.2.3 决策质量指标评估**（`:734-740`）——原文首项即：
  > 「a） 胜率与任务完成率……**胜率指标应通过足够次数的重复推演获得，
  >  以确保具有统计显著性和稳定性**。」

## 本条要解决的问题（P1【上级报告出处】给的事实）

报告 §7.2/§6 Q9 的证据：**20 轮离线对战 100% 平局**；
正式晋升的 **88.5% / Elo+354.5** 是 **vs 自家 `weights_incumbent_rule`** 的**自博弈配对**，
**不是**官方 Demo AI 或线上成绩。

**→ 一个 100% 平局的评估没有区分度，按 6.2.3「统计显著性」的要求，
它不足以支撑任何对外成绩表述。** 本模块把这句判断变成**可执行的守门**。

## 设计原则

1. **守门，不改写结论**：本模块**不修改**任何既有评估输出，
   只在结论上判定「可采信 / 不可采信」并给出原因。
2. **不虚报达成**：外部对比层拿不到对手时，判定为
   `pending_platform_support`（**待平台方配合**），**不是**"已达成"。
   （`P1`【风险与回滚】原文即要求如此。）
3. **来源必须标注**：6.1.1 区分三层框架，故任何成绩都要带来源标签；
   **自博弈成绩不得表述为对外实战能力**（`P1` 步骤 5）。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

# ── 三层框架（31007 6.1.1 原文的措辞）────────────────────────────────────
LAYER_SIMULATION = "simulation"          # 仿真环境验证
LAYER_LIVE = "live"                      # 实际推演校验
LAYER_COMPARATIVE = "comparative"        # 横向对比分析
ALL_LAYERS = (LAYER_SIMULATION, LAYER_LIVE, LAYER_COMPARATIVE)

LAYER_LABELS = {
    LAYER_SIMULATION: "仿真环境验证",
    LAYER_LIVE: "实际推演校验",
    LAYER_COMPARATIVE: "横向对比分析",
}

# ── 对手来源分类（P1 步骤 1 要求"盘点哪些是自家基线、哪些是外部可获取"）──
SOURCE_SELF = "self_baseline"            # 自家基线（如 weights_incumbent_rule）
SOURCE_EXTERNAL = "external"             # 外部/官方/公开对手
SOURCE_UNKNOWN = "unknown"

SOURCE_LABELS = {
    SOURCE_SELF: "自家基线（自博弈）",
    SOURCE_EXTERNAL: "外部/官方对手",
    SOURCE_UNKNOWN: "来源未标注",
}

# ── 有效性阈值 ─────────────────────────────────────────────────────────────
#
# ⚠ **标准未给具体数值**：6.2.3 只要求"足够次数的重复推演"与"统计显著性和稳定性"，
#    没有规定平局率上限或区分度下限。故下面是**本仓的工程取值**，
#    写在一处便于审计与调整，**不得**表述为"标准规定的阈值"。
DEFAULT_MAX_DRAW_RATE = 0.5      # 平局率高于此值 → 缺乏区分度
DEFAULT_MIN_DISCRIMINATION = 0.2  # 区分度低于此值 → 无法分辨强弱
DEFAULT_MIN_GAMES = 20           # 6.2.3「足够次数」的本仓下限


def summarise_outcomes(outcomes: Sequence[Any]) -> Dict[str, Any]:
    """统计一批对局结果。

    `outcomes` 每项可为：
      - `1` / `-1` / `0`（胜 / 负 / 平）
      - 字符串 `"win"` / `"loss"` / `"draw"`
      - dict 且含 `winner`（`1` = 本方胜、`-1` = 对方胜、`0`/`-1` 之外的 None = 平）
    """
    wins = losses = draws = 0
    for o in outcomes or []:
        v = _outcome_value(o)
        if v is None:
            continue
        if v > 0:
            wins += 1
        elif v < 0:
            losses += 1
        else:
            draws += 1
    total = wins + losses + draws
    return {
        "total": total,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "draw_rate": (draws / total) if total else 0.0,
        "win_rate": (wins / total) if total else 0.0,
        # 区分度：**非平局局里本方胜的比例** —— 它度量"这场评估能不能分出强弱"。
        # 若全是平局，非平局数为 0 → 区分度 0（而非 0.5，避免"看起来还行"）。
        "discrimination": (wins / (wins + losses)) if (wins + losses) else 0.0,
    }


def _outcome_value(o: Any) -> Optional[int]:
    if isinstance(o, bool):
        return None
    if isinstance(o, (int, float)):
        if o > 0:
            return 1
        if o < 0:
            return -1
        return 0
    if isinstance(o, str):
        s = o.strip().lower()
        if s in ("win", "w", "victory"):
            return 1
        if s in ("loss", "l", "defeat"):
            return -1
        if s in ("draw", "d", "tie"):
            return 0
        return None
    if isinstance(o, dict):
        if "outcome" in o:
            return _outcome_value(o["outcome"])
        if "result" in o:
            return _outcome_value(o["result"])
        w = o.get("winner")
        if w is None:
            return 0                     # 无 winner 即平局
        try:
            w = int(w)
        except (TypeError, ValueError):
            return None
        if w == 1:
            return 1
        if w == -1:
            return -1
        return 0
    return None


def evaluate_validity(outcomes: Sequence[Any], *,
                      opponent_source: str = SOURCE_UNKNOWN,
                      layers_present: Optional[Sequence[str]] = None,
                      max_draw_rate: float = DEFAULT_MAX_DRAW_RATE,
                      min_discrimination: float = DEFAULT_MIN_DISCRIMINATION,
                      min_games: int = DEFAULT_MIN_GAMES,
                      ) -> Dict[str, Any]:
    """判定一次评估**是否可采信**。返回结构化结论，**不改写原始数据**。

    Returns
    -------
    dict
        `admissible`（是否可采信）、`summary`（统计）、`gates`（逐条守门结果）、
        `source`（来源标签）、`layers`（三层框架到位情况）、`warnings`。
    """
    stats = summarise_outcomes(outcomes)
    gates: List[Dict[str, Any]] = []
    warnings: List[str] = []

    def gate(name: str, ok: bool, detail: str) -> None:
        gates.append({"gate": name, "ok": bool(ok), "detail": detail})

    # ① 样本量：6.2.3「足够次数的重复推演……统计显著性和稳定性」
    gate("sample_size", stats["total"] >= min_games,
         "%d 局（下限 %d）" % (stats["total"], min_games))

    # ② 平局率：平局率过高则无区分度
    gate("draw_rate", stats["draw_rate"] <= max_draw_rate,
         "%.3f（上限 %.2f）" % (stats["draw_rate"], max_draw_rate))

    # ③ 区分度
    gate("discrimination", stats["discrimination"] >= min_discrimination,
         "%.3f（下限 %.2f）" % (stats["discrimination"], min_discrimination))

    # ④ 来源必须标注：未标注即不可采信（防止把自博弈当对外成绩）
    if opponent_source == SOURCE_UNKNOWN:
        gate("opponent_source_labelled", False, "对手来源未标注")
        warnings.append("对手来源未标注 —— 该成绩**不得**对外引用")
    else:
        gate("opponent_source_labelled", True, SOURCE_LABELS.get(opponent_source, opponent_source))

    # ⑤ 三层框架到位情况（6.1.1）
    have = set(layers_present or [])
    unknown = sorted(have - set(ALL_LAYERS))
    if unknown:
        warnings.append("出现未识别的层名：%s" % unknown)
    for layer in ALL_LAYERS:
        gate("layer:%s" % layer, layer in have,
             LAYER_LABELS[layer] + ("（已有）" if layer in have else "（缺）"))

    # ⑥ 自博弈成绩**不得**表述为对外能力
    if opponent_source == SOURCE_SELF:
        warnings.append(
            "该评估对手为**自家基线**（自博弈）—— 可作内部对比，"
            "**不得**表述为对外/实战能力（P1 步骤 5）")

    comparative_ok = LAYER_COMPARATIVE in have
    admissible = all(g["ok"] for g in gates if g["gate"] != "layer:%s" % LAYER_COMPARATIVE)

    return {
        "admissible": admissible,
        "summary": stats,
        "gates": gates,
        "source": opponent_source,
        "source_label": SOURCE_LABELS.get(opponent_source, opponent_source),
        "layers_present": sorted(have & set(ALL_LAYERS)),
        "layers_missing": [l for l in ALL_LAYERS if l not in have],
        # **不虚报达成**：外部对比层缺席时明确标为待平台方配合
        "comparative_layer_status": ("present" if comparative_ok
                                     else "pending_platform_support"),
        "warnings": warnings,
        "thresholds_note": ("阈值（平局率上限 %.2f / 区分度下限 %.2f / 最少 %d 局）"
                            "为本仓工程取值，**标准未规定具体数值**"
                            % (max_draw_rate, min_discrimination, min_games)),
    }


def label_claim(claim: str, *, opponent_source: str) -> str:
    """给任何对外成绩表述**强制加来源标签**（P1 步骤 5）。

    这是最后一道防线：即使有人直接拿字符串去写报告，
    也必须经过本函数才能得到带标签的版本。
    """
    tag = SOURCE_LABELS.get(opponent_source, SOURCE_LABELS[SOURCE_UNKNOWN])
    if opponent_source == SOURCE_SELF:
        return "%s［口径：%s —— 内部对比，非对外能力］" % (claim, tag)
    if opponent_source == SOURCE_EXTERNAL:
        return "%s［口径：%s］" % (claim, tag)
    return "%s［口径未标注 —— 不得对外引用］" % claim


def inventory_opponents(entries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """盘点对手来源（P1 步骤 1）：把每个对手归入自家基线 / 外部 / 未标注。

    `entries` 每项期望含 `name`，可选 `source`（`self_baseline`/`external`）。
    """
    buckets: Dict[str, List[str]] = {SOURCE_SELF: [], SOURCE_EXTERNAL: [], SOURCE_UNKNOWN: []}
    for e in entries or []:
        if not isinstance(e, dict):
            continue
        name = str(e.get("name") or "(unnamed)")
        src = str(e.get("source") or SOURCE_UNKNOWN)
        if src not in buckets:
            src = SOURCE_UNKNOWN
        buckets[src].append(name)
    return {
        "self_baseline": sorted(buckets[SOURCE_SELF]),
        "external": sorted(buckets[SOURCE_EXTERNAL]),
        "unknown": sorted(buckets[SOURCE_UNKNOWN]),
        "counts": {k: len(v) for k, v in buckets.items()},
        "external_available": bool(buckets[SOURCE_EXTERNAL]),
        "note": ("横向对比层需要 external 非空；为空时应标 'pending_platform_support'，"
                 "**不得**标为已达成"),
    }
