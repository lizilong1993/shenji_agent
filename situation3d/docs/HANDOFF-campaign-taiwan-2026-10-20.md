# 台海战役演示案例 — 交付与部署说明（2026-10-20）

## 1. 交付范围

| 项 | 位置 |
|---|---|
| 自研兵棋内核 | `situation3d/wargame_core/`（陆/海/空、传感器、交战矩阵、后勤、换乘） |
| 战役编排 | `situation3d/campaign.py` |
| 战役三级推理 | `situation3d/campaign_inference.py` |
| 战役想定 | `situation3d/campaigns/taiwan_landing.py` |
| 六角地图生成 | `situation3d/scripts/build_campaign_map.py` |
| 案例数据生成 | `situation3d/scripts/run_campaign.py` |
| 前端战役界面 | `situation3d/web/src/campaignUi.js`（+ app.js/index.html/style.css 小补丁） |
| 端到端自检 | `situation3d/tools/campaign_selftest.py` |
| 规则规格 | `situation3d/docs/wargame_core_spec.md` |
| 总体方案 | `situation3d/docs/campaign_demo_plan.md` |

分支：`feat/campaign-demo`（服务器 worktree `/remote-home/lizilong/shenji_agent_campaign`），
已推送 GitHub `lizilong1993/shenji_agent`。

## 2. 台海案例内容

- 7 阶段：集结装载（0–180）→ 航渡/海上拦截（180–420）→ 换乘编波（420–600）→
  抢滩突击（600–780）→ 巩固/反突击（780–1020）→ 向内陆夺控（1020–1320）→ 结局（1320–1440）。
- 1440 帧（30 s/tick，12 h）；33 个单位（红 18 / 蓝 15）。
- 5 个目标点：登陆场、台南、高雄、澎湖、海峡中段。
- 18 个脚本事件（ISR、警戒、电子干扰起止、岸舰齐射、两次换乘、无人机侦察、
  抢滩上陆、直升机支援、登陆艇抵滩、装甲反击、第二梯队、工兵、机场争夺、停火）。
- 登陆兵力从海上出发：两栖战车/步兵在换乘阶段由舰艇释放，抢滩阶段转为陆战单位。
- 结局：`outcome.engine`（内核计算）+ `outcome.script`（虚构设定）+ `match`。

## 3. 前端能力

- 底部**阶段色带**：7 段，当前阶段高亮，点击跳转。
- 右栏**战役态势**：阶段列表、目标点归属、事件流（可按级别筛选、点击定位到 tick/目标/地点）、结局卡。
- 自动生成交火/对峙热点（按 `engage_km`）与控制区（按 `influence_km`），复用既有图层。
- 战役三级推理（L1/L2/L3）写入 `inference_frames`，右侧 TS 面板与热点着色沿用原逻辑。

## 4. 部署（已完成）

线上工作树 `/remote-home/lizilong/shenji_agent`（8848）已刷新：

- `situation3d/web/src/campaignUi.js`（新增）
- `situation3d/web/src/app.js`、`index.html`、`style.css`（+56 行，5 处钩子）
- `situation3d/web/data/situations/taiwan_landing.json`（campaign.v3，约 17 MB）
- `situation3d/web/data/campaign_maps/map_9001/`（`basic.json` / `cost.pickle`）
- `situation3d/tools/campaign_selftest.py`

未重启 8848（静态文件替换即可生效）。旧 v2 数据集备份在
`/tmp/situation3d-deploy-bak-20261020_110450/taiwan_landing.json.v2`
（容器 `/tmp` 重启会清空；如需长期回滚请另行保存）。

**注意**：线上 `app.js` 的 5 处钩子是直接在 main 工作树上打的补丁，尚未提交到 main 分支。
合并 `feat/campaign-demo` 时需确认这几处与分支版本一致，避免覆盖。

## 5. 验证记录

- 引擎/战役单测：41 项全部通过
  （`PYTHONPATH=situation3d python3 -m unittest discover -s situation3d/tests -p 'test_*.py'`）。
- 线上 8848：`python3 situation3d/tools/campaign_selftest.py --port 8848` → ALL CHECKS PASSED
  （阶段色带/阶段列表/事件列表/阶段跳转/事件定位/结局卡/无页面错误）。
- 数据集：1440 帧、237 事件、engine=red / script=red / match=true。

## 6. 已知限制

- 海空武器参数为合成演示值（`approximated`），交战矩阵只覆盖两案例用到的组合。
- 未实现：间瞄校射、指挥链/通信、空中加油、潜艇浮出转换、两栖卸载时间。
- 中印案例尚未建立（`campaigns/india_china_west.py` 待做）。
- 战役数据的 `plots`（时间窗标绘）已生成，但前端尚未按 `from_tick/to_tick` 显示。
