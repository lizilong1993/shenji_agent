# 演示 / 开发环境管理

**核心原则：开发环境可以持续迭代，演示环境必须保持稳定；只有达到明确开发里程碑并通过验证、经用户同意后，才更新对外演示版本。**

## 两套环境

| | 演示环境 | 开发环境 |
|---|---|---|
| 端口 | **8848**（Tailscale Funnel 唯一对外端口） | **8858**（仅 tailnet/内网，不接 Funnel） |
| 访问 | https://server-10-254-1-155.tailf6b2dc.ts.net/ | http://100.113.178.69:8858/ |
| worktree | `/remote-home/lizilong/shenji_agent_demo` | `/remote-home/lizilong/shenji_agent`（主工作树） |
| Git 分支 | `demo`（只接受 main 的快进更新） | `main`；新特性在 `feat/*` worktree 内开发 |
| 服务管理 | `ops/situation_service.sh demo …` | `ops/situation_service.sh dev …` |
| 日志 | `/tmp/situation3d_demo_8848.log` | `/tmp/situation3d_dev_8858.log` |

两个服务可同时运行、互不影响：改代码、重启、测试都只作用于各自 worktree。
其他临时 feature worktree 继续用 8850/8851/8856 等空闲端口，**任何情况下不得占用 8848**。

## 演示环境的数据与凭据

演示 worktree 中以下运行时文件**不属于 Git**，首次建立时已配置好，日常不要手改：

- `web/data/situations/*.json`：演示案例数据的**冻结副本**。开发环境对案例数据的改动不会影响演示；里程碑需要更新案例数据时，用 `demo_update.sh --with-data`。
- `web/data/base.json`、`ion.json` 等同级数据文件：同样为副本。
- `.env`：从主工作树复制的凭据副本。
- `web/vendor`、`web/tiles`：软链到主工作树的静态资产（Cesium 包 / 离线底图缓存）。
- 瓦片库 `tiles_db`、`tiles3d_db`：通过启动时的 `TILE_DB_DIR` / `TILES3D_DIR` 环境变量指向主工作树的共享大资产（共 33 GB）。**这两个目录是只读素材库，开发时不要原地重建/覆盖**，如需更新应作为独立的数据变更并同步告知。

## 日常开发

```bash
cd /remote-home/lizilong/shenji_agent/situation3d
ops/situation_service.sh dev start      # 起开发服务 :8858
ops/situation_service.sh dev restart    # 改完服务端代码后重启
ops/situation_service.sh dev smoke      # 首页/app.js/案例索引/案例 core 冒烟
ops/situation_service.sh dev stop
```

新特性按既定流程在 `feat/<name>` 分支 + 独立 worktree 开发，验证通过后合并回 `main` 并推送 GitHub。**合并/推送 main 不会影响演示**，只有执行 `demo_update.sh` 才会。

## 更新演示（里程碑）

前置条件：功能已在 8858 完成测试；`main` 上目标 commit 已验证（主功能正常、演示案例可完整运行）。

```bash
# 1) 预检：只打印将要更新的提交，不做任何改动
situation3d/ops/demo_update.sh <commit>

# 2) 向用户说明更新内容并取得明确同意后执行
situation3d/ops/demo_update.sh <commit> --yes
# 里程碑含案例数据更新时：
situation3d/ops/demo_update.sh <commit> --yes --with-data
```

脚本行为：检查演示工作树干净 → 校验目标在 main 上且为 demo 的后代（只做 fast-forward）→ 打备份 tag `demo-backup-<时间戳>` → 快进合并 →（可选）同步案例数据 → 重启 8848 → 冒烟 → **失败自动回退到备份 tag 并重启**。

## 回退

任一时点回到上一个稳定版本：

```bash
git -C /remote-home/lizilong/shenji_agent_demo tag | grep demo-backup   # 选一个备份
git -C /remote-home/lizilong/shenji_agent_demo checkout -B demo demo-backup-<时间戳>
ops/situation_service.sh demo restart
ops/situation_service.sh demo smoke
```

不要用 `git reset --hard`；不要手工 `git checkout <其他分支>` 演示工作树。

## 约束

1. 8848 只运行 `demo` 分支 worktree 的代码，不接受直接改文件、直接部署未验证代码。
2. 不在演示 worktree 里开发、拉分支、合并；更新只通过 `demo_update.sh`。
3. 每次更新演示前必须征得用户同意；更新前后各做一次 smoke。
4. Tailscale Funnel 配置固定为 443 → 127.0.0.1:8848，不要改动。
