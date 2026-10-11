"""`I17` 判据 2 + 3 的可运行入口测试：CLI 必须能跑、样例必须**与实现一致**。

## 判据

> 2) **规则冲突检测可运行并输出清单**；
> 3) **认知对齐度给出公式与样例计算**。

## 重点

1. **CLI 真的能跑**（不是"存在一个脚本"）；
2. **样例计算与实现逐位一致** —— 否则文档里的"样例"会与实际算法脱节；
3. **样例必须被声明为合成数据** —— 不得让读者以为这就是真实评测结果（那是虚报）。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "run_i17_checks.py"
sys.path.insert(0, str(REPO))

from training.alignment import cognitive_alignment  # noqa: E402
from training.expert_rating import rule_conflicts    # noqa: E402


def _run(*extra):
    if not SCRIPT.is_file():
        pytest.skip("run_i17_checks.py not present")
    return subprocess.run([sys.executable, str(SCRIPT), *extra],
                          cwd=REPO, capture_output=True, text=True)


# ══════════════ 判据 2：可运行 + 输出清单

def test_cli_runs_and_exits_zero() -> None:
    r = _run()
    assert r.returncode == 0, (r.stdout or "") + (r.stderr or "")[-400:]


def test_cli_prints_a_conflict_list() -> None:
    """**判据 2 的核心**：必须**输出清单**（不是只说"检测完成"）。"""
    out = _run().stdout or ""
    assert "规则冲突检测" in out
    assert "冲突数" in out
    assert "attack_vs_hold_fire" in out, "未输出冲突条目"


def test_cli_conflict_matches_the_library() -> None:
    """CLI 的冲突结论必须与库函数**一致**（不能各算各的）。"""
    r = _run("--json")
    assert r.returncode == 0
    d = json.loads(r.stdout)
    n_cli = d["rule_conflict_detection"]["n_conflicts"]
    # 复算：内置样例里只有 actor=2 的 attack/hold_fire 是冲突
    assert n_cli == 1, d["rule_conflict_detection"]


def test_cli_accepts_an_external_action_file(tmp_path: Path) -> None:
    """必须支持**对给定动作清单**跑检测（否则只是个演示脚本）。"""
    f = tmp_path / "acts.json"
    f.write_text(json.dumps([
        {"kind": "engage", "actor": 7, "target": 3},
        {"kind": "disengage", "actor": 7, "target": 3},
        {"kind": "occupy", "actor": 1, "target": "A"},
        {"kind": "withdraw", "actor": 2, "target": "A"},
    ]), encoding="utf-8")
    r = _run("--actions", str(f), "--json")
    assert r.returncode == 0, r.stderr
    d = json.loads(r.stdout)["rule_conflict_detection"]
    assert d["n_actions"] == 4
    assert d["n_conflicts"] == 1, "跨主体误报或漏报"
    assert d["conflicts"][0]["rule"] == "engage_vs_disengage"


def test_cli_rejects_a_missing_file_loudly(tmp_path: Path) -> None:
    """**不静默降级**：动作文件不存在时应报错退出，而不是悄悄用内置样例。"""
    r = _run("--actions", str(tmp_path / "nope.json"))
    assert r.returncode != 0


def test_cli_rejects_malformed_json(tmp_path: Path) -> None:
    f = tmp_path / "bad.json"
    f.write_text("{not json", encoding="utf-8")
    r = _run("--actions", str(f))
    assert r.returncode != 0


# ══════════════ 判据 3：公式 + 样例计算

def test_cli_shows_the_formula_and_a_worked_example() -> None:
    out = _run().stdout or ""
    assert "认知对齐度" in out
    assert "显性行动吻合度" in out and "隐性意图匹配度" in out
    assert "手算复核" in out, "未给出样例计算过程"


def test_worked_example_matches_the_implementation() -> None:
    """**样例数值必须与实现一致** —— 文档样例与实际算法脱节是常见缺陷。"""
    r = _run("--json")
    d = json.loads(r.stdout)["cognitive_alignment"]
    # 用同一批输入独立复算
    exp = cognitive_alignment(
        ["move", "occupy", "attack", "hold_fire", "occupy"],
        ["move", "occupy", "attack", "hold_fire", "occupy", "withdraw"],
        ["ASSAULT", "HOLD", "RECON", "FIRE_SUPPORT"],
        ["ASSAULT", "HOLD", "RECON", "FIRE_SUPPORT"],
        confidences=[0.9, 0.8, 0.7, 0.6],
    )
    assert d["explicit_action_agreement"] == pytest.approx(exp["explicit_action_agreement"])
    assert d["implicit_intent_match"] == pytest.approx(exp["implicit_intent_match"])
    assert d["value"] == pytest.approx(exp["value"])


def test_worked_example_numbers_are_exact() -> None:
    """钉住具体数值：5/6 与 1.0 → 0.9167（便于人工复算）。"""
    d = json.loads(_run("--json").stdout)["cognitive_alignment"]
    assert d["explicit_action_agreement"] == pytest.approx(5.0 / 6.0)
    assert d["implicit_intent_match"] == pytest.approx(1.0)
    assert d["value"] == pytest.approx(0.5 * (5.0 / 6.0) + 0.5)


def test_hand_calculations_appear_in_the_output() -> None:
    d = json.loads(_run("--json").stdout)["cognitive_alignment"]
    we = d["worked_example"]
    for k in ("explicit_hand_calc", "implicit_hand_calc", "combined_hand_calc"):
        assert we.get(k), "缺手算过程: %s" % k
    assert "5/6" in we["explicit_hand_calc"]


# ══════════════ 不得虚报

def test_output_declares_the_sample_is_synthetic() -> None:
    """**不得让读者以为样例是真实评测结果** —— 那是虚报 `I17` 判据 1。"""
    d = json.loads(_run("--json").stdout)
    note = d.get("sample_data_note", "")
    assert "合成" in note
    assert "不构成" in note and "真实" in note


def test_text_output_also_carries_the_caveat() -> None:
    out = _run().stdout or ""
    assert "合成" in out and "不构成" in out


def test_weights_are_declared_in_the_output() -> None:
    d = json.loads(_run("--json").stdout)["cognitive_alignment"]
    assert d["weights_declared"] == {"explicit": 0.5, "implicit": 0.5}
    assert "自定" in d["note"]
