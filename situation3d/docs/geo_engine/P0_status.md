# P0 基线冻结状态（2026-10-21）

> 基线：`feat/campaign-demo` @ `6982407`，tag `wargame-baseline-6982407`
> 本阶段结论：**P0 数据语义硬门槛已全部关闭**（鉴权按项目既有 runbook 恢复后完成受控实测）；仅 F0/F1/F2 标志语义未专项确认，不影响 P4 的主表/高差/骰点对拍。台海 golden replay 仍受 map_9001 未入库限制（见 §5）。

## 1. 已完成

| 项 | 结果 |
|---|---|
| 基线 tag | `wargame-baseline-6982407` → 6982407730db1ef6613e5ed2e349414fbbbb569e |
| 基线测试 | 41 项全部通过（core 34 + campaign 7），worktree 干净 |
| 规则数据 JSON | `rules/data/*.json`，由 `tools/geo_p0/extract_rules.py` 从 `docs/裁决表格.xlsx` 与 `Data/weapons/*` 生成 |
| 差异报告 | `docs/geo_engine/P0_rule_diff_report.md`（`tools/geo_p0/diff_rules.py` 可重生成） |
| tick>1 回归 fixture | `tests/test_wargame_core_timebase.py`；换算公式集中在 `compat/timebase.py` |
| golden replay | `tests/fixtures/golden_core_1231_seed20261021.json`；`tests/test_golden_replay.py`（`GEORULES_GOLDEN=1` 时重放 300 tick，逐位一致） |
| 黑盒实测 | `tools/geo_p0/probe_env.py` 等 4 个探针；鉴权恢复后完成 TarType/高差/骰点/tick/AttMin 实验 |

## 2. 黑盒实测结论（train_env 4.1.0，2026-10-21）

- **TarType（确认）**：`0` 间瞄、`1` 对人员、`2` 对车辆、`3` 对空。旧数据字典把 1/2 写反。
  - 对车辆：wp36 在 d14/15/16/18 攻击力 6/5/4/2，与 TarType=2 行逐格一致（历史回放 + 受控实验）。
  - 对人员：`ObjBlood` 是**攻方车/班数**，不是目标血量。攻方 blood 1/2/3/4/5 对步兵打出 2/3/5/7/9，与 TarType=1 的 ObjBlood 1..5 行一致；目标 blood 保持 3 不变。
- **ele_rect（确认）**：8×12 表行 = 高差等级（1–8），列 = 距离（1–12）；**仅仰射（攻方更低）生效**，平射/俯射为 0，距离 >12 为 0，Δh>8 按 8 钳位。
  - 验证点：低 2 级 d1 → -2（基值 10→8），低 2 级 d5 → -1（基值 9→8），低 1 级 d8 → 0，低 3 级 d1 → -3；人员表同规则（基值 5→3）。
  - 数据字典“行=射击方、列=目标方”的推测错误；`裁决表格` 表头“高度差/距离”正确。
- **骰点（确认）**：`random1`/`random2` 均为 2d6 三角分布（200 个独立 setup，峰值 6–8），不是 `randint(2,12)` 均匀分布。引擎随机流不受 Python 种子控制，相同想定重复 setup 结果不同。
- **time.tick（确认）**：`tick` = 每步帧数，1 帧 = 1 s；直瞄 75 s 冷却在 tick=1/2/5 下间隔 75/38/15 步；`cur_step` 每 step 加 1。
- **AttMin（确认）**：wp69（AttMin=2）在 d1 射击返回错误 503 `CantShootOfOutOfAim`。
- **F0/F1/F2（未专项）**：导弹制导标志，需另做实验；不影响主表/高差/骰点。

## 3. 鉴权问题的处置（项目既有方案）

项目 `round5_revision_v12/sdk_plan_c_runbook.sh` 记录了同容器（465a18f93ac6，eth0 `02:42:ac:11:00:05`）的处理办法：容器重建会换 MAC，`.engine_config` 需按**当前 MAC + 原始首次使用时间戳**重绑；失败的时间校验会在 `/root/.engine_config` 留下过期标记，重绑前需清除。

2026-10-21 按该流程执行：

1. 备份原配置到 `/tmp/engine_config.bak.20261021_134447`；
2. 写入 `base64(<当前 MAC>,1785636126742546063)`；
3. 删除 `/root/.engine_config` 过期标记；
4. `Authenticator.pass_MAC_check()` 与 `pass_time_check_since_first_use()` 均返回 True；
5. `env.setup()` 打印 `did pass authentication`，`judge_info` 正常产生。

## 4. 未完成项

- F0/F1/F2 制导标志语义（P4 前补）；
- 台海案例 golden replay（见 §5）。

## 5. 台海 golden replay 说明

`campaigns/taiwan_landing.py` 使用的 `map_9001` 由 `scripts/build_campaign_map.py` 现场生成，未入库；重建需要 Natural Earth 陆地 shapefile 与 Terrarium 地形瓦片（仓库内无缓存），本阶段未重建。因此本阶段录制的是**内核级 golden replay**（公开想定 1231 + map_123，300 tick，seed 20261021），用于锁定 6982407 的运动/裁决行为。

台海案例的 golden replay 需在拥有 map_9001 的环境执行：

```bash
python3 scripts/run_campaign.py --case taiwan_landing --out /tmp/tw --seed 20261021
# 对输出 JSON 取 hash 并存档；或在生成 map_9001 后用 run_wargame_core_demo 录制
```
