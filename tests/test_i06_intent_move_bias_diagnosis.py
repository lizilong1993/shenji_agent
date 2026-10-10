"""`intent_negative_move_bias` 成因诊断（`I06` 判据 3）。

## 本轮查清的结论

**该告警不是模型缺陷，而是判据口径与数据分布算术不相容 —— 「门」不可能通过。**

模型输出分布与**真实标签分布几乎重合**（逐类差 ≤ 0.022），
即模型**忠实地学到了数据里的类别失衡**；
而 `move_bias_estimate = mean(移动类 1..6) − (不动类 0)`
对任何具有该分布的模型都必然给出强负值。

## 必须先分开的两种口径

| 路径 | 输入 | 样本数 | 行为意义 |
| --- | --- | --- | --- |
| 噪声路径 | `torch.randn` | **4** | **无**（代码自述） |
| **真实数据路径** | `train.json` | **793** | **有** |

判据 3 的 gate 用**真实数据路径** → 故「4 个样本是噪声」**不成立**。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DOC = REPO / "docs" / "intent_move_bias_diagnosis_20261010.md"
DIAG = (REPO / "experiments" / "shenji_agent" / "model_quality"
        / "diagnosis_with_real_data.json")
SCRIPT = REPO / "scripts" / "diagnose_model_quality.py"


def _doc() -> str:
    if not DOC.is_file():
        pytest.skip("diagnosis doc not present")
    return DOC.read_text(encoding="utf-8")


def _diag() -> dict:
    if not DIAG.is_file():
        pytest.skip("diagnosis json not present")
    return json.loads(DIAG.read_text(encoding="utf-8"))


# ─────────────────── 口径：判据用的是真实数据路径

def test_gate_uses_the_real_data_path_not_the_noise_path() -> None:
    """**核心事实**：gate 的 `metric_source` 是 `intent_real_data`（793 样本），
    不是 4 样本的噪声路径 —— 故「样本太少」不能用来解释掉该告警。
    """
    d = _diag()
    gate = d.get("move_bias_gate") or {}
    assert gate.get("metric_source") == "intent_real_data", gate
    rd = d.get("intent_real_data") or {}
    assert int(rd.get("sample_count") or 0) > 500, \
        "真实数据样本数过少（%s）—— 结论需重估" % rd.get("sample_count")
    # 噪声路径确实是 4 个
    assert int(d.get("intent_bias_sample_count") or 0) == 4


def test_gate_is_failing() -> None:
    """判据 3 **仍未通过** —— 不得写成已解决。"""
    d = _diag()
    gate = d.get("move_bias_gate") or {}
    assert gate.get("accepted") is False
    assert abs(float(gate.get("measurement") or 0)) > float(gate.get("threshold") or 1)


# ─────────────────── 成因：模型与数据分布几乎重合

def test_model_output_matches_real_label_distribution() -> None:
    """**成因的直接证据**：模型输出与真实标签分布逐类差很小。

    这条一旦不成立（模型明显偏离数据），结论就要改成"模型确实有偏"。
    """
    d = _diag()
    model = d.get("intent_output_distribution") or {}
    real = (d.get("intent_real_data") or {}).get("class_distribution") or {}
    assert model and real, "缺少分布数据"
    diffs = {}
    for i in range(7):
        k = "class_%d" % i
        if k in model and k in real:
            diffs[k] = abs(float(model[k]) - float(real[k]))
    assert len(diffs) == 7, "类别数不为 7"
    worst = max(diffs.values())
    assert worst < 0.05, (
        "模型与数据分布差异最大 %.4f，过大 —— 不能断言'模型忠实反映数据'" % worst)


def test_dominant_class_is_the_stay_class() -> None:
    """不动类在**真实标签**里就占多数 —— 负 bias 的根源在数据，不在模型。"""
    d = _diag()
    real = (d.get("intent_real_data") or {}).get("class_distribution") or {}
    c0 = float(real.get("class_0") or 0)
    assert c0 > 0.5, "不动类占比 %.4f，不足以解释强负 bias" % c0
    moves = [float(real.get("class_%d" % i) or 0) for i in range(1, 7)]
    assert sum(moves) / 6 < 0.15, "移动类平均概率过高，与负 bias 不符"


def test_gate_threshold_is_arithmetically_incompatible_with_the_data() -> None:
    """**「门」不可能通过**的算术证明。

    通过条件：`|mean(class_1..6) − class_0| <= threshold`
    → `mean(class_1..6) >= class_0 − threshold`。
    用实测值代入，所需的移动类平均概率远超数据实际值。
    """
    d = _diag()
    real = (d.get("intent_real_data") or {}).get("class_distribution") or {}
    thr = float((d.get("move_bias_gate") or {}).get("threshold") or 0.1)
    c0 = float(real.get("class_0") or 0)
    need = c0 - thr                       # 通过所需的最小"移动类平均概率"
    have = sum(float(real.get("class_%d" % i) or 0) for i in range(1, 7)) / 6
    assert need > have * 5, (
        "通过所需的移动类概率 %.4f 仅为实际的 %.1f 倍 —— 不相容性结论需重估"
        % (need, need / have if have else math.inf))


# ─────────────────── 文档：结论与边界

def test_doc_states_the_conclusion_is_not_a_model_defect() -> None:
    d = _doc()
    assert "不是模型缺陷" in d or "不是「模型偏向不动」" in d
    assert "不可能通过" in d or "算术不相容" in d


def test_doc_refuses_to_change_the_criterion_unilaterally() -> None:
    """**关键边界**：改阈值/指标属于**改判据**，须先裁定，不得自行决定。"""
    d = _doc()
    assert "不擅自改" in d or "须先裁定" in d
    assert "A" in d and "B" in d and "C" in d, "未给出 A/B/C 选项"


def test_doc_does_not_recommend_retraining_to_inflate_move_probability() -> None:
    """**不得**把"重训模型抬高 move 概率"当解法 —— 那会让模型偏离训练分布。"""
    d = _doc()
    assert "不建议" in d, "未声明不建议重训以抬高 move 概率"


def test_doc_keeps_criterion_three_as_failed() -> None:
    """判据 3 仍未通过，文档必须保持这一结论。"""
    d = _doc()
    assert "仍未通过" in d
    for claim in ("判据 3 已通过", "已解决", "I06 已完成"):
        assert claim not in d, "出现过度声明 %r" % claim


def test_doc_links_the_root_cause_to_class_imbalance() -> None:
    """须与 D1 的类别失衡、§66.2 的标定问题关联（同一根因）。"""
    d = _doc()
    assert "类别失衡" in d
    assert "D1" in d and "66.2" in d


# ─────────────────── 指标公式（防回归）

def test_move_bias_formula_is_dimensionally_consistent() -> None:
    """公式必须是 `mean(移动类) − mean(不动类)`，不得回到"求和 vs 求均"的量纲错误。"""
    if not SCRIPT.is_file():
        pytest.skip("diagnose script not present")
    src = SCRIPT.read_text(encoding="utf-8")
    i = src.find("def move_bias_estimate_from_probs")
    assert i > 0
    # ⚠ 只检查**可执行代码**，不检查 docstring ——
    # 该函数的 docstring **故意引用**了原来的错误公式来说明修的是什么。
    # 我前两版把整段/整块都拿去匹配，于是被 docstring 里的引用反复误报失败。
    # 正确做法：用 AST 取出函数体里**除 docstring 之外**的语句源码。
    import ast
    import textwrap

    tree = ast.parse(src)
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "move_bias_estimate_from_probs"),
              None)
    assert fn is not None, "找不到 move_bias_estimate_from_probs"
    body_stmts = list(fn.body)
    if body_stmts and isinstance(body_stmts[0], ast.Expr) and \
            isinstance(body_stmts[0].value, ast.Constant) and \
            isinstance(body_stmts[0].value.value, str):
        body_stmts = body_stmts[1:]          # 去掉 docstring
    code = "\n".join(textwrap.dedent(ast.unparse(st))
                     for st in body_stmts)
    assert code.strip(), "函数体为空（解析有误）"
    # 修正后的写法两侧都是均值
    assert "mean(" in code, "函数体里没有 mean(...)"
    # 不得出现量纲错误的原始式
    for bad in ("sum(probs[:, 1:]) / (n - 1)", "sum(probs[:, 1:]) / (n-1)"):
        assert bad not in code, "代码里出现了量纲错误的原始式: %s" % bad
    # 但 docstring **必须**保留对原错误的说明（否则后人不知为何这样写）
    doc = ast.get_docstring(fn) or ""
    assert "量纲" in doc, "docstring 未说明量纲错误"


def test_noise_path_is_documented_as_behaviourally_meaningless() -> None:
    """噪声路径（`torch.randn`）必须仍被标注为"只有结构意义"，防止被当行为证据。"""
    if not SCRIPT.is_file():
        pytest.skip("diagnose script not present")
    src = SCRIPT.read_text(encoding="utf-8")
    assert "torch.randn" in src
    assert "结构意义" in src and "行为意义" in src
