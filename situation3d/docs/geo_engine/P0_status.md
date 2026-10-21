# P0 基线冻结状态（2026-10-21）

> 基线：`feat/campaign-demo` @ `6982407`，tag `wargame-baseline-6982407`
> 本阶段结论：**P0 未完全关闭**。静态数据与基线回放已冻结；两项规则语义确认被庙算 SDK 的 MAC 鉴权阻断，需运维/平台方解除后才能完成。

## 1. 已完成

| 项 | 结果 |
|---|---|
| 基线 tag | `wargame-baseline-6982407` → 6982407730db1ef6613e5ed2e349414fbbbb569e |
| 基线测试 | 41 项全部通过（core 34 + campaign 7），worktree 干净 |
| 规则数据 JSON | `rules/data/*.json`，由 `tools/geo_p0/extract_rules.py` 从 `docs/裁决表格.xlsx` 与 `Data/weapons/*` 生成 |
| 差异报告 | `docs/geo_engine/P0_rule_diff_report.md`（`tools/geo_p0/diff_rules.py` 可重生成） |
| tick>1 回归 fixture | `tests/test_wargame_core_timebase.py`；换算公式集中在 `compat/timebase.py` |
| golden replay | `tests/fixtures/golden_core_1231_seed20261021.json`；`tests/test_golden_replay.py`（`GEORULES_GOLDEN=1` 时重放 300 tick，逐位一致） |

## 2. 黑盒实测结论

历史回放 `replays/replay_1770632023.2594852.zip` 含 24 条 `judge_info`，据此被动确认：

- **TarType 映射（确认）**：`2` = 对车辆、`1` = 对人员（按 `ObjBlood` 分级）、`0` = 间瞄、`3` = 对空。
  证据：wp36 对车辆在 d14/15/16/18 的攻击力为 6/5/4/2，与 `WpData` TarType=2 行逐格一致；TarType=1 行是常量 2（对人员 blood=1），若按旧字典（1=对车辆）则与实测矛盾。**数据字典里 1/2 的映射写反了。**
- **骰点（部分确认）**：`random1`/`random2` 均落在 2..12；分布与种子控制未验证。
- **ele_rect 行列语义（未确认）**：24 条回放的 `ele_diff` 全为 0，无法区分。
- **F0/F1/F2、time.tick（未确认）**：需要受控实验。

## 3. 受控实验阻塞

`train_env` 鉴权绑定 MAC `02:42:ac:11:00:03`，当前容器 eth0 为 `02:42:ac:11:00:05`；`env.setup()` 打印 `did not pass authentication`，后续 `step()` 不产生 `judge_info`。

按 `docs/deployment_auth.md`，合法路径只有：

1. 由运维以授权 MAC 启动容器（`docker-compose.auth-mac.yml`，需要宿主机权限，本容器无 docker）；
2. 或向平台方申请针对实际 MAC 重新生成 `.engine_config`；
3. 或在官方授权宿主/镜像上运行。

禁止自行构造/修改 `.engine_config`，本阶段未做任何规避尝试。

## 4. 未完成项与解除后的实验脚本

`tools/geo_p0/probe_env.py` 已写好受控实验（`smoke/tartype/elevation/dice/tick`）。鉴权解除后：

```bash
source /remote-home/lizilong/py310env.sh
cd /remote-home/lizilong/shenji_agent/land_wargame_sdk
python /path/to/situation3d/tools/geo_p0/probe_env.py elevation
python /path/to/situation3d/tools/geo_p0/probe_env.py tartype
python /path/to/situation3d/tools/geo_p0/probe_env.py dice --rounds 500
python /path/to/situation3d/tools/geo_p0/probe_env.py tick
```

需要冻结的接口：

- `elevation_modifier(shooter_grade, target_grade)`：在确认 `ele_rect` 行列语义前不写死；
  当前 P2 骨架未实现高差查表，避免把猜测固化进 API。
- `time.tick`：`compat/timebase.py` 先固定 SDK 公式，geo 引擎接入时按 P0 fixture 显式切换。

## 5. 台海 golden replay 说明

`campaigns/taiwan_landing.py` 使用的 `map_9001` 由 `scripts/build_campaign_map.py` 现场生成，未入库；重建需要 Natural Earth 陆地 shapefile 与 Terrarium 地形瓦片（仓库内无缓存），本阶段未重建。因此本阶段录制的是**内核级 golden replay**（公开想定 1231 + map_123，300 tick，seed 20261021），用于锁定 6982407 的运动/裁决行为。

台海案例的 golden replay 需在拥有 map_9001 的环境执行：

```bash
python3 scripts/run_campaign.py --case taiwan_landing --out /tmp/tw --seed 20261021
# 对输出 JSON 取 hash 并存档；或在生成 map_9001 后用 run_wargame_core_demo 录制
```
