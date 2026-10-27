# 服务器恢复通知（2026-10-27）

> 给所有在 server155/situation3d 上继续工作的会话：先读本页再执行任务。

## 发生了什么

- 原 Docker 服务器被黑客挖矿，管理员重启；Tailscale 节点 `server-10-254-1-155`（100.113.178.69）离线，重新登录后节点变为 `465a18f93ac6`，新地址 **100.105.105.121**。
- 旧项目路径 `/remote-home/lizilong/shenji_agent` 已被另一块盘（/dev/sda2，shenji-v2 项目）覆盖；旧数据盘 /dev/sdb 仍挂载在 `/remote-home`，但其 `/lizilong` 被子挂载遮蔽，暂时无法直接访问。**不要写旧路径。**

## 当前工作副本

- 代码：**`/remote-home/lizilong/shenji_agent_sit`**（从 GitHub `lizilong1993/shenji_agent` 恢复）
- main：`716c134`（已合并 `feat/target-models-v3`、`docs/target-models-v3`），回退标签 `recovered-main-90ddd9c`
- 远程分支已清理：只保留 `main`；原 35 个分支中已合并的 32 个已删除；未合并的旧 UI `feat/taiwan-dense-transfer-v1` 和旧备份分支先打成 `archive/...` 标签后删除。
- `.env` 已按用户提供的 Cesium ion token / 天地图 key 恢复（权限 600，gitignored）。
- `web/vendor/cesium` 已用 `tools/fetch_cesium.sh` 恢复（23 MB）。

## 已恢复的数据

- `situation3d/envpacks/taiwan_tw2_v1`：由 `/tmp` 的 `tw2-gba-canopy-lc` 复制（含 GBA 高度、CHMv2 树冠、WorldCover 地类）。
- `situation3d/var/resources`：904 MB、24 条资源，索引路径已重写，`python3 scripts/resource.py verify` 全过。
- `var/resources/cache/osm_pbf/taiwan-latest.osm.pbf`：313 MB。
- `web/data/situations/taiwan_dense_transfer_v1.json`：用 `/tmp/8848_core.json` + `8848_inference.json` 重组（48 MB，旧生成器数据；validate_case 会提示缺 lane/bbox，非阻塞）。
- 全量测试：`cd situation3d && python3 -m pytest tests -q` → **558 passed / 14 skipped**。

## 仍缺失

- `web/tiles`（约 3 GB 离线底图）、`tiles_db`、`tiles3d_db`、demo worktree、部分 gitignored `engine/data` 资产。
- 旧 `/dev/sdb` 上的历史资产需要管理员重新挂载到 `/remote-home/lizilong` 才能直接恢复；否则用 `tools/fetch_assets.py` 重新抓取或重新生成。
- 8858（测试）和 8848（演示）当前没有常驻服务；冒烟启动 8858 后已停止。

## 如何继续任务

1. SSH：`ssh situ155`（已指向 100.105.105.121，root，密钥不变）。
2. 进入 `cd /remote-home/lizilong/shenji_agent_sit/situation3d`；不要在旧 `/remote-home/lizilong/shenji_agent` 下改代码。
3. 先跑 `python3 -m pytest tests -q` 确认基线；改代码走 feature 分支/worktree，合并 main 并推送后删除远程分支（已合并即删）。
4. 缺 tiles/vendor 时先按需重建：`sh tools/fetch_cesium.sh`（vendor）、`python3 tools/fetch_assets.py ...`（tiles）。不要假设 8858/8848 在运行。
5. 数据/资源 CLI：`python3 scripts/resource.py scan / verify / list`；环境变量 `SITUATION_RESOURCES_DIR` 可指向资源索引目录。

## 各会话的补充

- **目标模型 v3**：已合并进 main；继续项见记忆 `project_target_models_v3`（052D/M777 等待办）。
- **战斗时间轴**：`feat/battle-timeline` 未出现在 GitHub 分支（原会话在旧盘中断）；请先检查本地是否还有 worktree，否则需按记忆/方案重建。
- **任务驱动/资源管理**：主体已在 main；Sentinel-2 适配器的未提交 WIP 在恢复仓库 `git stash@{0}`（STAC 搜索已验证，TCI 抓取/JSON datetime bug 待修），不要覆盖该 stash。
- **演示/8848**：demo worktree 与离线瓦片缺失，先不要尝试发布或回放演示；等资产恢复后再走 `demo_update.sh`。
