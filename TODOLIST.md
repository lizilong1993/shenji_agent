# TODOLIST

本文件是项目唯一的总待办表，**随代码一起维护**（旧的三份清单都已并入这里：`任务总清单_20261103.md`、`本机复现清单与计划_20261106.md`、交接文档第 3 节）。

状态：`[x]` 已完成　`[~]` 进行中　`[ ]` 未开始　`[!]` 需用户批准　`[-]` 明确不做
最近更新：2026-11-18

---

## 0. 环境

- [x] 本机开发环境（Python 3.11.9 + node 24）；当前全量 **810 passed / 10 skipped**，
      **七个 JS 自检**全过。唯一失败 `test_fill_default_heights` 缺 `shapely`
      （本机无此依赖，改动前后同样失败，非代码问题）
- [x] 演示环境 **8848**（demo 模式，只读）：`python situation3d/server/serve.py --port 8848 --mode demo`
      提醒：此端口曾出现两个进程同时 listen（16:46 与 20:27 各起一次），旧进程 `Stop-Process`
      后新进程继续服务。重启前先 `netstat -ano | findstr :8848` 查重，别再叠第二份
- [x] 测试环境 **8858**（test 模式，可推演可写运行库）：`python situation3d/server/serve.py --port 8858 --mode test --runs-dir <本地盘>`
      端口分工沿用 `交接_情报WM化与后续_20261103.md:73`「8848=demo，8858=main」。
- [x] `.env` 凭据已配置（天地图 key + Cesium ion token），`/tdt/img/...` 代理实测 200，影像与注记正常叠加
- [x] 演示公网入口：Tailscale Funnel 8443 → `https://pc-desktop.tailf6b2dc.ts.net:8443/`（443 已被其他服务占用）。**测试环境不开放公网**
- [x] 静态资产重建：`web/vendor/cesium`、`web/tiles`（低层级）、`web/data`（3 个案例）、`envpacks`
- [ ] `.env` 凭据同步到服务器（服务器容器挂不上 NFS，无法写入）
- [~] 服务器 situ155：容器在线，`/remote-home` 落在本地盘，容器内 mount 返回 EPERM 且无 `/dev/fuse`；用 libnfs 用户态工具只读访问。
      **2026-11-15 复查**：situ155 已不在 tailnet 节点表里；server160 可达但它的
      `/remote-home/lizilong` 是**本机 ext4**（`/dev/sda2`）不是共享 NFS，且没有
      `shenji_agent` 目录——换机器取同一份数据这条路走不通，下面三条阻塞照旧
- [ ] 服务器容器重新挂载 NFS（需宿主机操作，容器内做不到）
- [x] `web/tiles` 本地瓦片补齐（c96a48c）：影像到 z11（4002 张）+ WSF3D 建筑高度
      z0-10（2802 张）+ 实测标定的 height_ramp.json。`tiles_db`/`tiles3d_db` 是发布用
      mbtiles 打包产物，按需生成，不做

## 0.5 三条分支的剩余项（2026-11-07 梳理）

用户指出情报 / 时间轴 / 引擎改造三条分支均未完成。逐条核对后的结论：

### 情报分支

- [x] **汇聚事件界面**（3a4c8a0）：后端早就算好 `overview.convergence`，前端此前零引用。
      现在列表顶部按评分列出最多 6 处，可定位到网格
- [x] **汇聚事件进 event feed**（0624fb9）：按事件 id 去重推送，按案例清空集合
- [x] **情报源目录元数据**（313742b）：cadence / cadence_note / tier 三字段，tier 直接由 trust 分档
- [x] **GDELT 扩展**（a01a2ba）：`languages`（sourcelang: 过滤，未知代码丢弃）
      与 `extra_terms`（AND 收窄）；不填等于旧查询
- [-] **情报缓存跨案例去重**：核对后确认**已具备**——`observations` 以 `observation_id` 为主键，
      同一 USGS/GDACS 事件在多个案例下只存一份；只有 GDELT 按 case 隔离（其命中依赖案例关键词，
      跨案例复用会串味），这是有意设计。原清单这条判断有误，撤销

### 时间轴分支

- [x] **实时推演链路**（cba1d17 + b2d0b77）：开战 / 轮询 / 取消 / 切案例 / 终态全部验证过
- [x] **成品回放 `?run=`**：simConsole 的「查看」按钮跳转到该路径，链路通
- [x] **事件/节点面板**（71bbe5e）：右下角事件按钮，七个筛选分类，点条目跳到该 tick，超 400 条截断并说明
- [x] **多 run 对比**（977edd5）：运行表加"对比"按钮，选两条后并排显示
      事件/开火/毁伤/双方损失/耗时/seed 与差值；只比两条
- [x] **回放导出**（2a8bbe4）：单文件 HTML（内联数据 + canvas 播放器），离线双击可开，
      含轨迹/事件/指标；选 HTML 不选视频是为了不引入 ffmpeg 与几百 MB 中间件

### 引擎改造分支

- [x] **中途任务切换**（3a4c8a0）：`MissionRunner.retask()` + `live/retask.json` 文件通道
- [x] **间瞄校射**：已按"是否被己方探测到"接入 `observed` 标志（`sim/world.py:1142`）
- [x] **中印案例**：`campaigns/india_china_west.py` 已存在且可被 `run_campaign.py --case` 加载
- [x] **战役地图**（bb66b22）：补了 NE 10m 陆地矢量与地形 z10，生成 map_9001 / map_9002；
      注意本地地图无 OSM 路网，海岸线与原图不同，两条依赖原图的测试改为按数据差异跳过
- [x] **中印案例插件**（bb66b22）：补齐六个接口，可跑出 1440 帧 / 256 事件
- [~] **指挥链 / 通信**：`sim/commander.py` 有 commander 层（默认关）。**卡在建模口径**：
      中断的是哪一层（上级命令下不来 / 同级协同断）、恢复到什么程度算恢复、
      现有多跳 comms_range_m 怎么算——这三个问题没有文档可依，不自行拍板。
      **进展（2026-11-07）**：great-powers-rts 给出了前两问的候选答案，第三问它也没做，
      详见 §5.4；仍等你定口径，代码未动
- [x] **潜艇浮出转换**（8bea776）：`sim/submarine.py` 三态（潜航/潜望镜/浮出），
      接入感知层；潜航只对声呐可见、浮出后才能用常规手段发现、浮出时不能用鱼雷
- [x] **两栖卸载时机**（0ea8757）：按载体限速，默认 20 秒一批，
      顺序按 uid 固定；此前整船乘客在同一 tick 全部上岸
- [~] **空中加油**：**卡在建模口径**：没有燃油消耗模型，加油机与受油机的 rendezvous、
      加油窗口、加油期间双方机动受限都需要先定规则；凭空造一套数值没有意义
- [x] **交战规则细化**（24115d9）：`sim/roe.py` 三种政策（自由/责任区/还击）+ 确认门槛 + NSL；
      接入 `World._engage`，被拒理由可查；默认值等于旧行为（同 seed 重跑数值不变）

### 手动测试发现并修复的问题（2026-11-07）

按真人用户视角走查网站，逐条记录。三条都已修完并复验。

- [x] **情报卡片读不懂**（33a4f33）：标题原本是 `Weather at 23.0000, 120.2000`，
      三条不同时刻的记录长得一模一样，看不出说了什么。改为按 WMO 码与实测值生成
      「局部多云，22°C，东南风 14 km/h」，并把关键读数（气温/风速/降水/机型/高度/
      地速/震级）直接铺到卡片上；WMO 码到中文的映射独立成 `intel/weather_text.py`
- [x] **切案例后情报面板报错文案误导**（c96a48c）：切到从未采集过情报的案例时显示
      「没有符合条件的情报」，让人以为筛选条件设错了；实际是该案例没数据。
      现在区分两态：无数据写「这个案例还没有情报数据」并说明怎么办，有数据才谈筛选
- [x] **建筑体块图层 feed=failed**（c96a48c）：根因不是缺瓦片而是缺色带——
      `./tiles/wsf3d/height_ramp.json` 404，图层直接标红。DLR 服务给的是调色板 PNG，
      色带必须实测反推。跑 `tools/calibrate_wsf3d.py --zoom 10 --samples 400`
      得到 13 条映射（0.2–10.6 m）后图层恢复 ok
- [x] **案例名与 TW1/TW2 命名不一致**（b1ee04c）：菜单里是长中文名，改成
      `ZY1`（中印西段）/ `TW3`（台海登陆），两个数据集一并重生成。
      菜单现在统一为 TW1 / TW2 / TW3 / ZY1

## 1. 收尾队列

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
- [x] **基线异常检测**（87e9e34）：`intel/baseline.py`，Welford 在线统计，按网格
      统计情报量，偏离 3σ 时输出异常；冷启动 8 样本内不判定；只报偏离不判定告警
- [x] 情报源目录元数据（许可/采集方式/tier）（313742b）
- [ ] CP-SAT 火力分配（需产品决策）
- [ ] 屏外方向箭头、AI 两段式管线、2D 视图（均需产品决策）

## 4. 本机暴露并已修复的可移植性缺陷

- [x] `serve.py` 四处 `json.load` 未指定编码（中文案例在 Windows 上按 GBK 解码失败）
- [x] `test_p4_rules.py` 读 `crt.json` 未指定编码
- [x] `test_resource_fetch.py` 用盘符拼 `file://`（改用 `pathlib.Path.as_uri()`）
- [x] `serve.py` 的 `send_error()` 传入中文消息 → HTTP 状态行 latin-1 编码失败（47ff037）
- [x] `web/data/geo_reports/` 未进 gitignore（5f488a6）
- [x] `test_resource_path_outside_roots_rejected` 用 `/etc/hostname`，Windows 上
      `os.path.isabs` 为 False，拒绝来自 isabs 分支而非 roots 分支，断言 "outside"
      落空（9aceea7 改用 `tempfile.gettempdir()` 下的路径）

## 5. great-powers-rts 调研（2026-11-07）

对象：<https://github.com/chencheng666/great-powers-rts>，快照 `1f8efa4`（2026-11-06），
MIT，155 个 JS 源文件 / 13,361 行，98 个 `node --test` 测试。three.js 浏览器 RTS，
无 Python、无 Cesium、无情报管线。
完整评估见项目根目录 `great-powers-rts_集成评估_20261107.md`（仓库外）。

**先排除一个误解**：它的 `intelligence.js` 不是情报汇聚，是间谍/网络战玩法
（潜入放炸弹、拆弹、断电锁指令），与本项目 `intel/` 无对应关系。真正的重叠在推演规则与前端容错两层。

盘点：**采纳 4 · 待定 3 · 不采纳 7**。
**采纳四项全部完成**（GP-01/02/04 = 7875af8，GP-03 = 9aceea7）。同 seed 复跑
20261023 / 20 tick 仍是 486 事件 / 225 开火 / 8 毁伤——全在前端呈现层，引擎未动。
待定三项里 GP-07 已于 2026-11-18 做完（纯后端），GP-05/GP-06 仍需产品判断。

### 5.1 采纳（4）

- [x] **GP-01** HP 驱动的损伤呈现（7875af8）：`presentationStage(hp, alive)` 四档
      完好 / 冒烟（<98%）/ 起火（<30%）/ 残骸，另接明火池（8 km 内才画，火比烟近）。
      **做的过程中先挖出一个更严重的既有缺陷**：`TargetStatus.at()` 只在
      `provenance.runtime === 'engine_run'` 时读每帧 HP，而两个战役数据集写的是
      字符串 `"fictional_demo"`/`"simulated"`，于是全落到按推理暴露量反推的老启发式——
      **taiwan_campaign 47520 行里 62.3 % 档位错误**（含 28.3 % 已击毁仍判正常，
      直接画满血条），india_china_west 61.6 %。两个写入方量纲还不一样
      （run_situation_case 的 strength 是 0-1 比值，campaign 的 strength 是 0-100
      且战损时不变、blood 才是真的）。修后 HP 与存活判定 100 % 对齐（76320 行
      0 处不一致）。另：`combatEnabled` 原来 gated on `data.combat.model`，
      四个数据集全没有该字段 → 整层战损呈现从未运行过；烟柱池上限 10 但实测
      单帧峰值 19 个冒烟目标，超出部分被静默丢弃，已提到 20
- [x] **GP-02** 血条避让布局（7875af8）：新增 `web/src/healthLayout.js`（纯函数）。
      候选位从默认位向外扩、代价 `|dx|+|dy|*1.1`、取首个空位、超 68 px 或出视口
      丢弃。布局在帧循环里按 150 ms 节流（依赖相机投影，不能只在脏帧跑）。
      可见性**现算**而不是读 syncEntities 的缓存——那个缓存只在脏帧更新，
      实测跳帧后 33 条里 8 条仍判不可见，血条会整片消失。
      **画面实测的局限**：120 km 高度下 23/33 个目标被编队聚合吃掉，真实页面
      只能同时看到 1–6 条血条；密集情形由 `health_layout_dense_test.js` 用合成
      数据覆盖（20 条同点、4 种缩放的真实坐标分布，全部 0 重叠）
- [x] **GP-03** HTTP JSON 体结构防护（9aceea7）：`_json_body_ok()` 限深 24、
      拒非有限浮点、拒 `__proto__`/`prototype`/`constructor`；解析处补抓
      `RecursionError`（它不是 `ValueError` 子类，40000 层体原来会逃成 500）。
      **实测确认的漏洞**：`/api/resources` 把 `notes` 整个 dict 存进条目
      （`ResourceEntry.validate` 只校 bbox 与 quality），POST `{"notes":{"score":NaN}}`
      返回 201 且响应体带字面量 `NaN` —— 那不是 JSON，前端 `JSON.parse` 直接抛，
      坏值同时落盘。新增 11 例（7 判据单测 + 4 真 HTTP），去掉判据后 NaN 与
      原型键两例确实变红，不是空转测试
- [x] **GP-04** WebGL 上下文丢失恢复（7875af8）：监听 canvas 的
      `webglcontextlost`/`webglcontextrestored`。丢失时 preventDefault（不 prevent
      则浏览器不尝试恢复）+ 停时间轴 + 状态报红 + 弹出可关闭的恢复提示
      （切流畅画质 / 知道了）；恢复时重算投影缓存并重新标脏。
      此前全仓无此监听：上下文一丢 rAF 循环继续空转，画面停在最后一帧、
      顶栏状态点仍是绿的

### 5.2 待定（3）

- [ ] **GP-05** 单位级交战姿态（自由/原地警戒/停火待命）：本项目 ROE 是**全场统一**的
      （`sim/roe.py` 三政策整场生效），没有「这队按兵不动、那队自由开火」的表达力。
      技术上可行（约 150 行），但受「默认值等于旧行为」约束须按案例 opt-in；
      且只对「有人在现场下命令」有意义，要不要做需产品判断
- [ ] **GP-06** 队形落点分配：`formationOffsets()` 按运动层分组、⌈√n⌉ 栅格槽位、
      沿重心→目标方向旋转、最近单位优先贪心。本项目 `sim/mission.py:145`
      只做单点 17 候环形试探，多支部队各自选点无整体队形。约 80 行，同样须 opt-in；
      当前 20–33 目标规模下 OBB 碰撞已能推出排队效果，优先级低于 GP-01/02
- [x] **GP-07** 按兵种的战损明细：`_losses_by_type()` 按 `Entity.unit_kind` 记
      `{count, value}`，`value` 直接用引擎的 `Entity.value`（与 `GameState.score()`
      同一字段），不另造造价表。写进 `summary.json` 与 `metrics`（键 `losses_by_type`），
      总数 `losses: {red, blue}` 保持不变。只统计插件 `meta` 里的导入编成，
      运行时生成的单位不在 meta 里，天然排除；本项目无蜂群子单位概念。
      前端不加展示（改 UI 须另行批准）。trajectory.json 结构未变（同 seed 逐键比对 0 处差异）。
      配套的「规则式复盘建议」仍**不做**——与「只提供动作与查询、不做自动决策」冲突

### 5.3 不采纳（7）

- [-] 方向装甲倍率（`armorFacing()` 0.82/1.0/1.18）：**已有且更细**，
      `rules/physics_direct_fire.py:41` 带 front/side/rear/top + 板厚 + 入射角 + 跳弹阈值；移植是降级
- [-] 稳定战斗指纹（`legacyBattleId()` 双 FNV）：已有 `scenario_hash`/`envpack_hash`/`plan_hash`/`run_id`
- [-] InstancedMesh 合批 + 画质阶梯：面向 120–240 单位，本项目最大 80 目标，Cesium 管线尚未成瓶颈
- [-] 雾图 2 bit 打包（`packFog()`）：联机专用，本项目无多人对局
- [-] 空中转向率 + 越肩脱离再进入（`aircraftStep()`）：需连续坐标，我们在 hex 空间（`wargame_core/air.py`），是架构级改动
- [-] AI 反制生产（`aiCounterUnit()`）：本项目无生产/建造系统，需求表无处落地
- [-] `Math.random()` 15 处（开火相位/AI 波次/出生抖动）：反面样本，与「同 seed 逐项一致」直接冲突

另：`entityLod.js:46` 已有 enter/exit 双阈值滞回；`targetClusters.js` + `overview.js:54`
的屏幕聚合与红蓝分离避让比它的 `separateUnits()` 更完整；`sim/collision.py` 的 OBB+SAT
已覆盖它物理位移那部分。

### 5.4 指挥链口径：它给了前两问的候选答案，第三问它也缺

对照 §0.5 里「没有文档可依」的三问：

| 待定问题 | gp-rts 的答案 | 能否沿用 |
|---|---|---|
| 中断的是哪一层 | 拆两层。**网络层**锁「下达新命令」这一个动作（仍自动还击，镜头/暂停/存档不受影响）；**信号层**锁「单位能否收到链路」 | 结构可参考，但断在哪一环要对着 `sim/commander.py` 的「模型建议→带过期时间的 order」自己定 |
| 恢复到什么程度算恢复 | 网络层：源站存活且有电 → 立即恢复（每帧重算，不靠倒计时到期）。信号层：无恢复语义，只看当下有无覆盖 | 只有网络层给了判据 |
| 多跳 `comms_range_m` 怎么算 | **没有多跳**——单跳，中继 260 内或信标 330 内即视为有链路 | **它也没做**，仍须你定 |

另外两条来自它的取舍值得记下：预警期 12 s 内摧毁源站或断电可取消；锁指令**只锁受到攻击的一方**，
AI 侧不受影响（本项目里对应「哪一方被断链」，不对称性需要显式声明）。

状态仍为 `[~]`：以上只是候选，口径未定前不写代码。

---

## 维护规则

1. 完成一项就改状态并在后面补上 commit 短哈希；合并后不再可能回退的，写进第 1 节。
2. 阻塞项必须写清阻塞原因，恢复后从这条接手。
3. 新增任务先归组（收尾队列 / Palantir / 调研衍生 / 外部项目调研），不要另起清单。
4. 需要改 web UI 的任务标 `[!]` 并等用户批准；纯后端先做完。
