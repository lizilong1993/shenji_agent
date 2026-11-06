# TODOLIST

本文件是项目唯一的总待办表，**随代码一起维护**（旧的三份清单都已并入这里：`任务总清单_20261103.md`、`本机复现清单与计划_20261106.md`、交接文档第 3 节）。

状态：`[x]` 已完成　`[~]` 进行中　`[ ]` 未开始　`[!]` 需用户批准　`[-]` 明确不做
最近更新：2026-11-06

---

## 0. 环境

- [x] 本机开发环境（Python 3.11.9 + node 24）；当前全量 **680 passed / 18 skipped**，三个 JS 自检全过
- [x] 演示环境 **8848**（demo 模式，只读）：`python situation3d/server/serve.py --port 8848 --mode demo`
- [x] 测试环境 **8858**（test 模式，可推演可写运行库）：`python situation3d/server/serve.py --port 8858 --mode test --runs-dir <本地盘>`
      端口分工沿用 `交接_情报WM化与后续_20261103.md:73`「8848=demo，8858=main」。
- [x] `.env` 凭据已配置（天地图 key + Cesium ion token），`/tdt/img/...` 代理实测 200，影像与注记正常叠加
- [x] 演示公网入口：Tailscale Funnel 8443 → `https://pc-desktop.tailf6b2dc.ts.net:8443/`（443 已被其他服务占用）。**测试环境不开放公网**
- [x] 静态资产重建：`web/vendor/cesium`、`web/tiles`（低层级）、`web/data`（3 个案例）、`envpacks`
- [ ] `.env` 凭据同步到服务器（服务器容器挂不上 NFS，无法写入）
- [~] 服务器 situ155：容器在线，`/remote-home` 落在本地盘，容器内 mount 返回 EPERM 且无 `/dev/fuse`；用 libnfs 用户态工具只读访问
- [ ] 服务器容器重新挂载 NFS（需宿主机操作，容器内做不到）
- [ ] `web/tiles` 高层级瓦片与 `tiles_db` / `tiles3d_db` 重建（体积大，按需）

## 1. 收尾队列（交接文档既定）

- [x] **K** 标绘迁到地图右键子菜单（bce01fb / 8e3d6c0）
- [x] **I** 情报 7 天缓存 + 按源增量（b40a7d0）
- [x] **J** 情报界面 World-Monitor 化（f2224d9 / c35008f）
- [x] **F** 情报汇聚事件后端（b7252d9 / b6f769e）
- [x] **B** 时间轴合并（cba1d17 服务器真实历史 + b2d0b77 补回复审遗漏的三处 hardening）
- [x] **A（代码部分）** 资源管理 SQLite：pull、全量测试、API 增删改查与上传
- [ ] **A（验收部分）** 8858 资源面板对真实 19 条数据的人工验收——**阻塞**：数据在 NFS 上，容器不可写
- [ ] **C** intelpipe worktree 清理（detached `aa13be6`）——**阻塞**：需要能写 NFS
- [x] **D** CM-11 主战坦克拼接（7764562）：M60A3 车体 + M48A5 炮塔，7,093 面、0.40 MB，
      炮塔按实测亮度比压到车体配色；已入库 specs / licenses / bboxes / manifest / entityLod。
      **交接文档记录的组合方向相反，已按公开资料纠正并在 CREDITS 写明**
- [x] **备份（本机部分）** `_backup_20261106/shenji_agent_local_20261106.bundle`（254 MB，verify 通过）；
      `_recovered/timeline_from_nfs_20261106.bundle`（从 NFS 取回的服务器真实历史）
- [ ] **备份（服务器部分）** canonical 未提交改动 + `.env`——**阻塞**：容器不可写

## 2. Palantir 借鉴（来源：Titan 机制分析与 UI 分析）

### P0

- [x] **PAL-DATA-01（后端）** 位置不确定性透传（26dda43）：`_auto_item()` / `_manual_item()`
      输出 `geo_uncertainty_m` / `is_centroid` / `ingest_time`；缺值与非法值一律归零
- [x] **PAL-DATA-01（前端）** 卡片显示定位精度与质心标注、"定位"按钮在地图上按误差半径画置信圈（5ddf99b）
- [x] **PAL-UI-01** 情报卡片固定"完整度"区块（位置/来源/关联/时效），缺项显式写"无"（5ddf99b）
- [x] **PAL-UI-02** 对象级时效标签"更新于 X 前"，超半衰期标黄并写明半衰期（5ddf99b）

### P1

- [x] **PAL-DATA-02** 人工跨源关联（26dda43）：`intel/correlations.py`，confirm 提升为 manual/0.99，
      reject 直接移除；`/api/intel/correlations` 支持 GET 与 POST
- [x] **PAL-DATA-03** AI 建议带推理记录（26dda43）：`reasoning{text, agent_version, algorithm,
      detected_at}`，评审不改写
- [x] **PAL-DATA-04** 禁止打击清单（26dda43）：`no_strike` / `no_strike_reason`，counts 里有计数
- [x] **PAL-UI-05** 图层加 `intelClass` 元数据（7c721d5）：纯元数据，不动 group / z / 互斥；
      `tools/intel_class_test.js` 守边界
- [x] **PAL-UI-03** 动作跟着对象走：卡片上"转人工"与"标记禁止打击"（须填理由）（5ddf99b）
- [x] **PAL-UI-04** 情报面板可停靠右侧，遮罩透明不吃事件、地图仍可操作（5ddf99b）
- [x] **PAL-UI-06** 标绘工具栏第二入口（与右键菜单共用目录与样式）（5ddf99b）

### P2

- [x] **PAL-FLOW-01（后端）** 目标生命周期状态机（1a05a55）：五段式、只允许表内迁移、
      历史 append-only；overview 输出只读看板，`/api/intel/lifecycle` 支持读写
- [x] **PAL-FLOW-02（后端）** BDA 回写（c525f25）：只接受 in_execution 的目标，
      结果限 destroyed/damaged/functional/unknown，写 assessment 并推进到 complete
- [x] **PAL-FLOW-01/02（前端）** 目标看板标签页（五列 + 允许迁移按钮）与毁伤评估表单（5ddf99b）

### 明确不做

- [-] 按门类重排现有图层分组
- [-] 影像参数滑块（色温/对比度等 8 项）
- [-] 对象级密级标签
- [-] 图层拖动重排序
- [-] 看板并发编辑冲突处理

## 3. 调研衍生

- [x] **G（后端）** 数据质量枚举（16ef223）：`intel/quality.py`，四选一
      measured/interpolated/predicted/stale，理由逐条，半衰期复用 trust.py
- [x] **G（前端）** 卡片显示质量四选一徽章（悬停看逐条理由）与时效/定位（5ddf99b）
- [x] **H** 标绘右键编辑菜单（改名/换归属/定位/删除）（5ddf99b）
- [ ] 基线异常检测（Welford z-score），待 F 的 UI 落地后评估（F 的 UI 已落地，可着手）
- [ ] 情报源目录元数据（许可/采集方式/tier）
- [ ] CP-SAT 火力分配（需产品决策）
- [ ] 屏外方向箭头、AI 两段式管线、2D 视图（均需产品决策）

## 4. 本机暴露并已修复的可移植性缺陷

- [x] `serve.py` 四处 `json.load` 未指定编码（中文案例在 Windows 上按 GBK 解码失败）
- [x] `test_p4_rules.py` 读 `crt.json` 未指定编码
- [x] `test_resource_fetch.py` 用盘符拼 `file://`（改用 `pathlib.Path.as_uri()`）
- [x] `serve.py` 的 `send_error()` 传入中文消息 → HTTP 状态行 latin-1 编码失败（47ff037）
- [x] `web/data/geo_reports/` 未进 gitignore（5f488a6）

---

## 维护规则

1. 完成一项就改状态并在后面补上 commit 短哈希；合并后不再可能回退的，写进第 1 节。
2. 阻塞项必须写清阻塞原因，恢复后从这条接手。
3. 新增任务先归组（收尾队列 / Palantir / 调研衍生），不要另起清单。
4. 需要改 web UI 的任务标 `[!]` 并等用户批准；纯后端先做完。
