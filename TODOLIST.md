# TODOLIST

本文件是项目唯一的总待办表，**随代码一起维护**（旧的三份清单都已并入这里：`任务总清单_20261103.md`、`本机复现清单与计划_20261106.md`、交接文档第 3 节）。

状态：`[x]` 已完成　`[~]` 进行中　`[ ]` 未开始　`[-]` 明确不做
最近更新：2026-11-06

---

## 0. 环境

- [x] 本机开发环境（Python 3.11.9 + node 24，`situation3d` 全量 624 passed / 18 skipped）
- [x] 本机演示环境：`python situation3d/server/serve.py --port 8848`，demo 模式，3 个案例可选，浏览器冒烟通过（TW1 案例 301 帧、31 个目标、无 JS 报错）
- [x] `.env` 凭据已配置（天地图 key + Cesium ion token），`/api/health` 报 `tianditu_configured: true` / `ion_configured: true`；`/tdt/img/...` 代理实测 200 且天地图影像与注记正常叠加
- [x] 公网访问：Tailscale Funnel 在 8443 暴露本机 8848 —— `https://pc-desktop.tailf6b2dc.ts.net:8443/`（443 已被其他服务占用，故用 8443）。实测公网 URL 能完整渲染案例
- [x] 静态资产重建：`web/vendor/cesium`（1.145.0）、`web/tiles`（低层级影像 z0-8 + 地形 z0-6）、`web/data`（3 个案例数据集）
- [ ] `.env` 凭据同步到服务器（服务器上那份仍未配置，演示 8848 在服务器上跑不起来在线底图）
- [~] 服务器 situ155：容器在线，但 `/remote-home` 挂在本地盘而非 NFS，容器内 mount 返回 EPERM 且无 `/dev/fuse`。目前用 libnfs 用户态工具只读访问
- [ ] 服务器容器重新挂载 NFS（需要宿主机操作，容器内做不到）
- [ ] `web/tiles` 高层级瓦片与 `tiles_db` / `tiles3d_db` 重建（体积大，按需）

## 1. 收尾队列（交接文档既定）

- [x] **K** 标绘迁到地图右键子菜单（bce01fb / 8e3d6c0，已并入 main）
- [x] **I** 情报 7 天缓存 + 按源增量（b40a7d0）
- [x] **J** 情报界面 World-Monitor 化（f2224d9 / c35008f）
- [x] **F** 情报汇聚事件后端（b7252d9 / b6f769e）：`intel/convergence.py`、`GET /api/intel/convergence`、17 个用例
- [x] **B** 时间轴 `feat/battle-timeline` 合并（cba1d17 服务器真实历史 + b2d0b77 补回复审遗漏的三处 hardening，已推送 a74f7b3）
- [x] **A（代码部分）** 资源管理 SQLite：pull、全量测试、API 增删改查与上传
- [ ] **A（验收部分）** 8858 资源面板对真实 19 条数据的人工验收——**阻塞**：数据在 NFS 上，容器不可写
- [ ] **C** intelpipe worktree 清理（detached `aa13be6`）——**阻塞**：需要能写 NFS
- [ ] **D** CM-11 主战坦克拼接——候选件已出（M48A5 车体 + M60A3 炮塔，18,887 面，headless 渲染确认几何正确），剩色差统一、接缝处理、减面（原方案目标 ~6,610 面）、入库（specs/licenses/bboxes/manifest + 两个校验脚本）
- [x] **备份（本机部分）** `_backup_20261106/shenji_agent_local_20261106.bundle`（254 MB，verify 通过）；`_recovered/timeline_from_nfs_20261106.bundle`（从 NFS 取回的服务器真实历史）
- [ ] **备份（服务器部分）** canonical 未提交改动 + `.env`——**阻塞**：容器不可写

## 2. Palantir 借鉴（来源：Titan 机制分析与 UI 分析）

### P0

- [ ] **PAL-DATA-01** 位置不确定性上屏：`_auto_item()` 透传 `geo_uncertainty_m`，卡片显示 ±X km，地图按半径画圆（先圆后椭圆）
- [ ] **PAL-UI-01** 情报卡片固定"完整度"区块（位置/来源/关联/时效），缺项显式写"无"
- [ ] **PAL-UI-02** 对象级时效标签"更新于 X 前"，超半衰期进 event feed 成为告警条目
- [ ] **PAL-UI-03** 动作跟着对象走：情报卡片上加确认/否决/转人工，不再强制切标签页

### P1

- [ ] **PAL-DATA-02** 人工跨源关联（`intel/correlations.py`，覆盖算法 `link_items()` 的结论）
- [ ] **PAL-DATA-03** AI 建议带推理记录（`AISuggestionStore` 加 `reasoning` / `model` / `agent_version`）
- [ ] **PAL-DATA-04** 禁止打击清单（NSL）标记
- [ ] **PAL-UI-04** 情报面板由模态框改为可贴靠右侧停靠栏
- [ ] **PAL-UI-05** 图层加 `intelClass` 元数据分组（不动 z 序与互斥规则）
- [ ] **PAL-UI-06** 标绘加工具栏第二入口（`plotDrawBar` 加类型下拉）

### P2

- [ ] **PAL-FLOW-01** 目标生命周期状态机 + 只读看板（五段：IDENTIFIED / PRIORITIZED / IN COORDINATION / IN EXECUTION / COMPLETE）
- [ ] **PAL-FLOW-02** 毁伤评估（BDA）回写态势（依赖 PAL-FLOW-01）

### 明确不做

- [-] 按门类重排现有图层分组
- [-] 影像参数滑块（色温/对比度等 8 项）
- [-] 对象级密级标签
- [-] 图层拖动重排序
- [-] 看板并发编辑冲突处理

## 3. 调研衍生（未建任务）

- [ ] **G** 数据质量与来源徽标（quality 枚举 + provenance 字段 + 属性面板）
- [ ] **H** 标绘右键编辑菜单（改 UI 需批准；与 PAL-UI-06 共用样式）
- [ ] 基线异常检测（Welford z-score），待 F 的 UI 落地后评估
- [ ] 情报源目录元数据（许可/采集方式/tier）
- [ ] CP-SAT 火力分配（需产品决策）
- [ ] 屏外方向箭头、AI 两段式管线、2D 视图（均需产品决策）

## 4. 本机暴露并已修复的可移植性缺陷

- [x] `serve.py` 四处 `json.load` 未指定编码（中文案例在 Windows 上按 GBK 解码失败）
- [x] `test_p4_rules.py` 读 `crt.json` 未指定编码
- [x] `test_resource_fetch.py` 用盘符拼 `file://`（改用 `pathlib.Path.as_uri()`）
- [x] `serve.py` 的 `send_error()` 传入中文消息 → HTTP 状态行 latin-1 编码失败，打断连接（47ff037）
- [x] `web/data/geo_reports/` 未进 gitignore，生成产物会被误提交

---

## 维护规则

1. 完成一项就改状态并在后面补上 commit 短哈希；合并后不再可能回退的，写进第 1 节。
2. 阻塞项必须写清阻塞原因，恢复后从这条接手。
3. 新增任务先归组（收尾队列 / Palantir / 调研衍生），不要另起清单。
4. 需要改 web UI 的任务在动手前要用户批准。
