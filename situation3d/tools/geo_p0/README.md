# geo_p0 工具

| 脚本 | 用途 |
|---|---|
| `extract_rules.py` | 从 `docs/裁决表格.xlsx`、`Data/weapons/*` 生成 `rules/data/*.json`（需要 openpyxl；输入数据在 `land_wargame_sdk` 分发包内） |
| `diff_rules.py` | 生成 `docs/geo_engine/P0_rule_diff_report.md`；用法 `python3 tools/geo_p0/diff_rules.py . wargame_core` |
| `probe_env.py` | 庙算黑盒受控实验（TarType / ele_rect / 骰点 / tick / AttMin）；需 py310 + 授权 MAC 环境，见 `docs/geo_engine/P0_status.md` |
| `probe_dice_multi.py` | 每局一发的骰点采样（规避 75 s 冷却与目标阵亡） |
| `probe_tick.py` | `time.tick` 语义：不同 tick 下 75 s 冷却对应的步数 |

`rules/data/*.json` 已入库；只有权威数据更新时才需要重跑 `extract_rules.py`。
