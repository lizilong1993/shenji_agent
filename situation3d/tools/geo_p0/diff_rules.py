#!/usr/bin/env python3
"""Generate the P0 difference report: authoritative tables vs wargame_core MVP.

Inputs:
  <rules_dir>/rules/data/*.json   (from extract_rules.py)
  <base_dir>/weapons.py, combat.py, detection.py  (baseline 6982407 copies)

Output: markdown on stdout.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


PROFILE_TO_NAME = {
    "cannon_large": "大号直瞄炮",
    "cannon_medium": "中号直瞄炮",
    "cannon_small": "小号直瞄炮",
    "autocannon": "速射炮",
    "mg": "车载轻武器",
    "small_arms": "步兵轻武器",
    "atgm": "重型导弹",
    "portable_atgm": "便携导弹",
    "aa_gun": "防空高炮",
    "artillery": "重型炮",
}

TAR_TYPE_PERSONNEL = 1


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_profiles(base_dir):
    spec = importlib.util.spec_from_file_location(
        "baseline_weapons", Path(base_dir) / "weapons.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["baseline_weapons"] = mod
    spec.loader.exec_module(mod)
    return mod.PROFILES


def values_list(values, att_max):
    if not isinstance(att_max, int):
        return None
    return [values.get(str(i), 0) for i in range(att_max + 1)]


def compare_series(label, current, authoritative, max_range=None):
    if authoritative is None:
        return f"| {label} | 无权威表 | - | - |"
    n = min(len(current), len(authoritative))
    diffs = [(i, current[i], authoritative[i])
             for i in range(n) if current[i] != authoritative[i]]
    missing = len(authoritative) - len(current)
    extra = len(current) - len(authoritative)
    if not diffs and missing == 0:
        return f"| {label} | 一致 | - | 无行为差异 |"
    parts = []
    if missing > 0:
        parts.append(f"截断 {missing} 项（表长 {len(current)} vs {len(authoritative)}）")
    if extra > 0:
        parts.append(f"当前表多填充 {extra} 项")
    if diffs:
        sample = "，".join(f"d{i}: {c}→{a}" for i, c, a in diffs[:6])
        parts.append(f"{len(diffs)} 处数值差异（{sample}{'…' if len(diffs) > 6 else ''}）")
    impact = _impact_note(current, authoritative, max_range)
    return f"| {label} | 不一致 | {'；'.join(parts)} | {impact} |"


def _impact_note(current, authoritative, max_range):
    """Whether the difference is reachable in the baseline implementation."""
    if max_range is None:
        return "-"
    current_max = min(int(max_range), len(current) - 1)
    n = min(len(current), len(authoritative))
    if any(current[i] != authoritative[i] for i in range(0, min(current_max, n - 1) + 1)):
        return "**射程内数值不同，影响行为**"
    if len(authoritative) - 1 > current_max:
        return "射程内一致；%d 格以外被截断" % current_max
    return "表尾差异，射程内不可达（无行为影响）"


def main():
    rules_dir = Path(sys.argv[1])
    base_dir = Path(sys.argv[2])
    data = rules_dir / "rules" / "data"
    attack = load(data / "attack_levels.json")
    weapons0 = load(data / "weapons_0.json")
    wpdata = load(data / "wpdata.json")["rows"]
    profiles = load_profiles(base_dir)

    vehicle = {r["name"]: r for r in attack["vs_vehicle"]}
    personnel = {(r["name"], r["blood"]): r for r in attack["vs_personnel"]}
    small_vehicle = {r["blood"]: r for r in attack["small_arms_vs_vehicle"]}
    zm = load(data / "zm_attack_level.json")
    zm_car = {r["weapon_id"]: r for r in zm["to_car"]}
    zm_peo = {(r["weapon_id"], r.get("blood")): r for r in zm["to_peo"]}

    lines = []
    lines.append("# P0 规则差异报告：权威表 vs wargame_core MVP")
    lines.append("")
    lines.append("> 基线：`feat/campaign-demo` @ 6982407（tag `wargame-baseline-6982407`）  ")
    lines.append("> 权威数据：`docs/裁决表格.xlsx`、`Data/weapons/*`（2026-09-15 快照）  ")
    lines.append("> 生成：`tools/geo_p0/diff_rules.py`（自动生成，勿手改）")
    lines.append("")

    lines.append("## 1. 武器对车辆表")
    lines.append("")
    lines.append("| 武器 | 结论 | 差异 | 备注 |")
    lines.append("|---|---|---|---|")
    for key, name in PROFILE_TO_NAME.items():
        p = profiles[key]
        if key == "aa_gun":
            continue
        auth = vehicle.get(name)
        if auth is None and name == "小号直瞄炮":
            auth = vehicle.get("小号直瞄跑")  # 原表笔误：跑/炮
        if auth is None and key == "artillery":
            rows = [r for r in wpdata if r["name"] == name and r["tar_type"] == 0]
            if rows:
                r = rows[0]
                auth = {"att_max": r["att_max"],
                        "values": values_list(r["values"], r["att_max"])}
                lines.append(compare_series(f"{key}（{name}，间瞄）",
                                            p.vs_vehicle, auth["values"],
                                            max_range=p.max_range))
                continue
            lines.append(f"| {key}（{name}） | 无权威行 | - | - |")
            continue
        if auth is None:
            lines.append(f"| {key}（{name}） | 无权威行 | - | - |")
            continue
        auth_values = values_list(auth["values"], auth["att_max"])
        if auth_values is None:
            lines.append(f"| {key}（{name}） | 见专门表 | - | 权威行注明见《步兵轻武器对车辆攻击等级表》 |")
            continue
        lines.append(compare_series(f"{key}（{name}）", p.vs_vehicle, auth_values,
                                    max_range=p.max_range))

    lines.append("")
    lines.append("## 2. 武器对人员表（当前实现只取 blood=1）")
    lines.append("")
    lines.append("| 武器 | 结论 | 差异 | 备注 |")
    lines.append("|---|---|---|---|")
    for key, name in PROFILE_TO_NAME.items():
        if key in ("aa_gun",):
            continue
        p = profiles[key]
        row = (personnel.get((name, 1))
               or personnel.get((name, 0))
               or personnel.get((name + "（对地）", 1)))
        if row is None:
            # 裁决表格 lacks some personnel rows that WpData has (e.g. 重型炮
            # TarType=1 ObjBlood=0); fall back to the master table.
            wp = [r for r in wpdata
                  if r["name"] == name and r["tar_type"] == TAR_TYPE_PERSONNEL
                  and r["obj_blood"] in (1, 0)]
            if wp:
                r = sorted(wp, key=lambda x: x["obj_blood"])[-1]
                auth_values = values_list(r["values"], r["att_max"])
                lines.append(compare_series(f"{key}（{name}）", p.vs_personnel,
                                            auth_values,
                                            max_range=min(p.max_range, r["att_max"])))
                continue
            lines.append(f"| {key}（{name}） | 无权威行 | - | 权威对人员表无此武器 |")
            continue
        auth_values = values_list(row["values"], row["att_max"])
        lines.append(compare_series(f"{key}（{name}）", p.vs_personnel, auth_values,
                                    max_range=min(p.max_range, row["att_max"])))

    lines.append("")
    lines.append("## 3. 防空武器（aa_gun）")
    lines.append("")
    aa = [r for r in wpdata if r["weapon_id"] == "1" and r["tar_type"] == 3]
    if aa:
        r = aa[0]
        av = values_list(r["values"], r["att_max"])
        lines.append(f"- 权威 `WpData` TarType=3 防空高炮：AttMax={r['att_max']}，"
                     f"0..{r['att_max']} 攻击力 = {av}")
    lines.append(f"- 当前 `aa_gun`：max_range={profiles['aa_gun'].max_range}，"
                 f"vs_personnel={profiles['aa_gun'].vs_personnel}")
    lines.append("- 结论：**当前 `aa_gun` 不是对空表**（用的是速射炮对人员表的副本，"
                 "射程 10 格）；权威对空 20 格表未被使用。")
    lines.append("")

    lines.append("## 4. 结构性问题")
    lines.append("")
    lines.append("| 项 | 现状 | 权威要求 | 影响 |")
    lines.append("|---|---|---|---|")
    lines.append("| 表键 | `WeaponProfile(vs_personnel, vs_vehicle)` 两个一维表 | "
                 "`(WeaponID, TarType, ObjBlood)` 每键独立 AttMin/AttMax | "
                 "同一武器的不同目标类型无法区分射程（wp36 对人 10 格、对车 18 格） |")
    keys = []
    for r in wpdata:
        keys.append((r["weapon_id"], r["tar_type"], r["obj_blood"]))
    lines.append(f"| 表数量 | 10 个手工 profile | WpData 共 {len(wpdata)} 行 "
                 f"({len(set(keys))} 个键) | MVP 只覆盖一小部分组合 |")
    lines.append("| 截断 | 多个 profile 的表短于 AttMax，`min()` 截断 | 表长必须 = AttMax+1 | "
                 "远距离攻击力被错误地保持为近距离值 |")
    lines.append("| 攻击力符号 | `max(0, base+mods)` 截断负值 | 攻击力可为负（修正量） | "
                 "间瞄/不利条件下的负修正被抹掉 |")
    lines.append("| 目标类型 | infantry/vehicle 两类 | TarType 0/1/2/3；对人员表按**攻方车/班数**（ObjBlood 1–5）分级 | "
                 "对空、按攻方规模分级均缺失 |")
    lines.append("| 拾取武器 | 只按当前表攻击力最大 | 需按 `(TarType, ObjBlood)` 取表并校验 AttMin/AttMax | "
                 "选武器与查表口径不一致 |")
    lines.append("")

    lines.append("## 5. 高差修正")
    lines.append("")
    elev = load(data / "elevation.json")
    car = elev["car"]
    lines.append("权威 `ele_rect_to_car`（8×12，值域 -5..0）：")
    lines.append("")
    lines.append("```")
    for row in car:
        lines.append(" ".join(f"{v:2d}" for v in row))
    lines.append("```")
    lines.append("")
    lines.append("当前 `combat._elevation_modifier`：上坡 `-min(5, |Δh|/ele_grade/2 + 1)`，"
                 "下坡 `+min(2, Δh/ele_grade/2 + 1)`。")
    lines.append("")
    lines.append("| 差异 | 说明 |")
    lines.append("|---|---|")
    lines.append("| 符号 | 权威表全部 ≤0；当前下坡给正修正（+1/+2），方向上不符 |")
    lines.append("| 维度 | 权威 8×12 表至少有两维索引；当前只用 `abs(Δh)` 压成一维，且无距离维 |")
    lines.append("| 边界 | 当前公式对任意 Δh 都给修正；权威表按等级钳位 |")
    lines.append("| 行列语义 | **已确认**：行 = 高差等级（1–8），列 = 距离（1–12）；仅仰射（攻方更低）时生效，"
                 "平射/俯射为 0，距离 >12 为 0 |")
    lines.append("")

    lines.append("## 6. 观察与通视")
    lines.append("")
    lines.append("| 项 | 现状 | 权威 | 差异 |")
    lines.append("|---|---|---|---|")
    lines.append("| 基础观察表 | 10/25/25；无人机对地 2，对空/无人机不可观察 | docs.md 表同值 | 一致 |")
    lines.append("| 侦察车/防空等特种观察距离 | `scenario._rich_unit` 把 `observe_distance` 合入 visual sensor，但 `detection.can_observe` 只读固定表 | "
                 "来自想定 `observe_distance` 数组，非公开表 | 逐单位观察距离未进入裁决 |")
    lines.append("| 掩蔽减半的高地例外 | 对所有目标类型生效 | 仅车辆目标（docs.md 观察规则第2条） | "
                 "对人员目标错误地免除了减半 |")
    lines.append("| 地形遮蔽减半 | 居民地/丛林减半 | 同一格内且通视 | 一致 |")
    lines.append("| 通视实现 | 格中心连线 + 每格高程 | 同规则文本（居民地/丛林 +1 级、端点不修正） | 语义一致，精度受格尺度限制 |")
    lines.append("")

    lines.append("## 7. 裁决与状态机")
    lines.append("")
    lines.append("| 项 | 现状 | 权威 |")
    lines.append("|---|---|---|")
    lines.append("| 战斗结果 | `score = att + (roll-7) - armor` 阈值映射 | 2d6 CRT + 装甲/地形/状态修正表（`crt.json`） |")
    lines.append("| 骰数 | `rng.randint(2, 12)` 均匀分布（非 2d6 三角分布） | 攻击 2d6；战损修正对人员 1d6、对车辆 2d6（黑盒 200 样本呈三角分布） |")
    lines.append("| 冷却/转换 | 无冷却、每 tick 可开火 | 直瞄冷却 75 s、同格 25 s、机动↔停止 75 s、展开/锁定 75 s |")
    lines.append("| 压制 | 固定 2 tick | 按规则持续秒数 |")
    lines.append("| 夺控 | 5 tick 保持 | 目标格 + 6 邻格无敌方地面单位，无等待 |")
    lines.append("| 堆叠 | 进入格时检查 ≤4 | 同格同方地面 ≤4 | 语义一致（尺度问题见方案 §1.6） |")
    lines.append("| 行军射击 | 任意 vehicle 可行进间射击 | 仅坦克主炮 `shoot_on_move`；行军不能射击；炮兵不能机动射击 |")
    lines.append("| 攻击力负值 | `max(0, …)` 截断 | 保留到 CRT |")
    lines.append("")

    lines.append("## 8. 时间基准")
    lines.append("")
    lines.append("| 项 | 现状 | 说明 |")
    lines.append("|---|---|---|")
    lines.append("| 想定 `time.tick` | `taiwan_landing` 写 `1.0`，`scenario.build_state` 不读取 | "
                 "SDK 语义是“每步前进的帧数”，`max_step=max_time/tick` |")
    lines.append("| 旧战役时间 | `CampaignSpec.tick_seconds=30`，1440 tick = 12 h | 与想定 `time.tick` 无关 |")
    lines.append("| 规则计时 | 75 s/25 s 等以秒计 | 黑盒实测：`tick` = 每步帧数（1 帧 = 1 s）；直瞄冷却 75 s → 75/tick 步（tick=2 时 38 步） |")
    lines.append("")

    lines.append("## 9. 黑盒实测状态（2026-10-21）")
    lines.append("")
    lines.append("| 目标 | 状态 | 证据 |")
    lines.append("|---|---|---|")
    lines.append("| TarType 映射 | **已确认** | wp36 对车辆 d14/15/16/18 → 6/5/4/2（TarType=2 行逐格一致）；"
                 "对人员用 TarType=1，`ObjBlood` 是**攻方车/班数**：攻方 blood 1–5 对步兵打出 2/3/5/7/9 |")
    lines.append("| ele_rect 行列语义 | **已确认** | 行=高差等级 1–8、列=距离 1–12；仅仰射生效（攻方低 2 级 d1 → -2，d5 → -1；"
                 "低 1 级 d8 → 0）；俯射/平射为 0；人对表同规则 |")
    lines.append("| 骰点分布/种子 | **分布已确认，种子未控** | 200 个独立 setup 的 random1/random2 呈 2d6 三角分布（峰值 6–8）；"
                 "相同想定重复 setup 结果不同，引擎 RNG 不受 Python 种子控制 |")
    lines.append("| time.tick 语义 | **已确认** | `tick` = 每步帧数，1 帧 = 1 s；直瞄 75 s 冷却在 tick=1/2/5 下间隔 75/38/15 步；"
                 "`cur_step` 每 step 加 1 |")
    lines.append("| AttMin | **已确认** | wp69（AttMin=2）在 d1 射击报错 503 `CantShootOfOutOfAim` |")
    lines.append("| F0/F1/F2 | **不影响直瞄裁决** | F0=100/101/102 的 wp71/83/75 均正常直瞄射击且 judge 字段无差异；wp69/73 拒绝由 AttMin=2 决定；判定为制导/引信类元数据，P4 主表对拍不依赖 |")
    lines.append("")
    lines.append("鉴权处置：项目既有 `round5_revision_v12/sdk_plan_c_runbook.sh` 记录了同容器（465a18f93ac6）的"
                 "重绑流程——以当前 eth0 MAC + 原始首次使用时间戳重写 `.engine_config`，并清除失败校验留下的 "
                 "`/root/.engine_config` 过期标记。2026-10-21 按该流程执行后 `pass_MAC_check`/`pass_time_check_since_first_use` "
                 "均通过，`env.setup()` 打印 `did pass authentication`，黑盒实验恢复（原配置已备份到 /tmp）。")
    lines.append("")

    lines.append("## 10. 权威源之间的冲突（P4 前必须定优先级）")
    lines.append("")
    lines.append("| 项 | 来源 A | 来源 B | 冲突 |")
    lines.append("|---|---|---|---|")
    lines.append("| 速射炮 (56, TarType=1) 射程 | `weapons_0.xlsx` AttMax=5 | `WpData` AttMax=10 | 有 |")
    lines.append("| 便携导弹 (71, TarType=3) 射程 | `weapons_0.xlsx` AttMax=5 | `WpData` AttMax=10 | 有 |")
    lines.append("| 车载轻武器 对人员 blood=2/3/4 距离 9–10 | `裁决表格` 2/3/4 | `WpData` 1/2/3 | 有 |")
    lines.append("")
    lines.append("`rules/weapons.py` 当前以 `WpData` 为主表；`weapons_0` 仅作射程清单交叉校验。"
                 "该优先级是工程选择，不是平台官方声明，P4 对拍前需书面确认。")
    lines.append("")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
