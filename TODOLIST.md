# TODOLIST

本文件是项目唯一的总待办表，**随代码一起维护**（旧的三份清单都已并入这里：`任务总清单_20261103.md`、`本机复现清单与计划_20261106.md`、交接文档第 3 节）。

状态：`[x]` 已完成　`[~]` 进行中　`[ ]` 未开始　`[!]` **阻塞/需用户决策**　`[-]` 明确不做
最近更新：2026-11-19

---

## 0. 环境

- [x] 本机开发环境（Python 3.14.6 + node 24）；2026-11-19 全量 **852 passed / 10 skipped
      / 1 failed**。唯一失败 `test_fill_default_heights` 缺 `shapely`
      （本机无此依赖，改动前后同样失败，非代码问题）。
      JS 自检 11 个文件：**11 全过**（2026-11-19 `plot_dynamic_test.js` 按当前设计重写）。
      注意 `python -m pytest -q` 会误收 `tools/*_test.py`（那些脚本按 argv 读文件），
      跑全量要写 `python -m pytest tests -q`
- [x] 演示环境 **8848**（demo 模式，只读）：`python situation3d/server/serve.py --port 8848 --mode demo`
      提醒：此端口曾出现两个进程同时 listen（16:46 与 20:27 各起一次），旧进程 `Stop-Process`
      后新进程继续服务。重启前先 `netstat -ano | findstr :8848` 查重，别再叠第二份
- [x] 测试环境 **8858**（test 模式，可推演可写运行库）：`python situation3d/server/serve.py --port 8858 --mode test --runs-dir <本地盘>`
      端口分工沿用 `交接_情报WM化与后续_20261103.md:73`「8848=demo，8858=main」。
- [x] `.env` 凭据已配置（天地图 key + Cesium ion token），`/tdt/img/...` 代理实测 200，影像与注记正常叠加
- [x] 演示公网入口：Tailscale Funnel 8443 → `https://pc-desktop.tailf6b2dc.ts.net:8443/`（443 已被其他服务占用）。**测试环境不开放公网**
- [x] 静态资产重建：`web/vendor/cesium`、`web/tiles`（低层级）、`web/data`（3 个案例）、`envpacks`
- [x] 远端 SSH 连通性复查（2026-11-19）。SSH 登录及只读检查成功，进程列表无 D 状态。
      同一日在 `/tmp` 隔离检出跑全量：共 830 项，813 项通过，17 项跳过，用时 40 秒。
      检出必须带 `web/`。缺少该目录时 `serve.py` 打印 `web root not found` 并退出，19 个 HTTP 用例无法启动服务。
- [-] 部署凭据同步、正式部署环境、服务器容器挂载 NFS、服务器备份——**2026-11-19 按用户
      迁移口径撤销**：开发与验收默认在本机（`.env` 与 8848/8858 已在本机跑通），
      服务器只在需要算力时临时调用，不再作为部署目标。服务器上的 `remote-home` 挂载
      （`ext4`）与容器 `EPERM` 问题因此不再阻塞本项目；本机备份见第 1 节。
      若今后需要正式对外部署，从本行重新开一条「部署环境」任务，不要沿用旧结论。
- [x] 远端隔离浏览器与 JS 验收环境（2026-11-19）。系统 Python 的 Playwright 1.48 可启动缓存的 Chromium 130，包内 Node 20.18.0 通过影像层顺序与 WebGL 恢复自检。Cesium 从已有检出只读引用，测试文件和运行产物放在 `/tmp`。
- [-] 服务器容器重新挂载 NFS（同上，随本机迁移撤销）。
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
- [x] **指挥链 / 通信**（3fba886）：`sim/command_links.py`。三问按 §5.4 可沿用的部分定成
      案例可写的口径，全部 opt-in（没写 `command_links` 块 = `World.command_links` 为 None，
      路径与旧行为一致）：
      - **断哪一层**：拆 `upper`（窗口内不下发**新** commander 命令，已发命令按原到期规则
        自然失效，自动还击不受影响）与 `peer`（窗口内 `radio_share` 这一跳不成立）。
      - **恢复判据**：`until_s` 与 `restore_when.all_alive` 任一满足即恢复，谁先到谁生效。
      - **多跳**：不做。共享仍是单跳，`test_radio_share_does_not_multihop` 继续守住。
      测试 `tests/test_command_links.py` 11 项；与 `test_event_scheduler` / `test_cot_integration`
      / `test_roe*` 同跑 74 项通过。**未做**：跨案例的链路状态可视化（改 UI 须另批）。
      写法与示例：`situation3d/docs/geo_engine/OPT_IN_RULES.md` §1
- [x] **潜艇浮出转换**（8bea776）：`sim/submarine.py` 三态（潜航/潜望镜/浮出），
      接入感知层；潜航只对声呐可见、浮出后才能用常规手段发现、浮出时不能用鱼雷
- [x] **两栖卸载时机**（0ea8757）：按载体限速，默认 20 秒一批，
      顺序按 uid 固定；此前整船乘客在同一 tick 全部上岸
- [x] **空中加油**（159a381）：`sim/refuel.py`，opt-in，`scenario["refuel"]` 显式配一对 uid。
      - **配对**：一个加油机 + 一个受油机，不做就近自动分配；必须同阵营、同为空域、
        未搭载。
      - **会合**：水平距离 ≤ `range_m`（默认 300 m）且高度差 ≤ `alt_tolerance_m`
        （默认 300 m）同时成立才开窗；`window_s` 到期、任一方脱离、受油机满油或
        加油机余额耗尽即关窗，**关过不再重开**（同一次会合只加一次油）。
      - **输油**：每步 `min(rate_per_s*dt, 受油缺口, supply)`；快进大步长按窗口剩余
        时长截断，不超窗。加油期间不改双方机动（沿用现有航路）。
      - **油耗**：地理引擎仍不引入通用飞行油耗；只新增 `operators[].fuel` 可写
        （默认满油，与旧行为一致），缺油起点由案例显式给出。
      `tests/test_air_refuel.py` 11 项覆盖开窗/速率/脱离/满油/窗口到期/余额耗尽/
      大 dt 截断/跨阵营/缺 uid/非法速率/非空域。全量 852 passed。
      写法与示例：`situation3d/docs/geo_engine/OPT_IN_RULES.md` §3
      背景（2026-11-18 复查仍成立）：`wargame_core/air.py` 有油耗而 `sim/` 没有；
      `wargame_core/logistics.py::consume_fuel()` 全仓无调用方，是死代码，仅记录不删
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
- [x] **A（验收部分）** 资源面板对真实资源库的验收（2026-11-18）：**原描述的
      「19 条 / 数据在 NFS 上 / 容器不可写」已核实为过时**。本机
      `var/resources/resources.db` 有 **15 条**真实条目（case_roads 2 / envpack 1 /
      platforms 10 / rolecards 1 / weapons 1），8858 test 模式 `/api/resources`
      下发同样 15 条且路径未脱敏；NFS 上那份服务器库随 situ155 掉出 tailnet 已不可达。
      新增 `situation3d/tools/verify_resources_panel.py`（Playwright 无头 Chromium）
      做机器验收，10 项全过：入口可开、列表与 API 逐条对齐、kind 计数与 stats 相等、
      搜索按 kind 前缀过滤且清空恢复、test 模式路径明文、无未捕获异常。
      脚本刻意不用 `[data-entry-id]` 计行（那是每行四个操作按钮的标记，会数出 4 倍），
      搜索走「搜索」按钮而不是 `input` 事件（组件只在 click / Enter 时读 `#resSearch`）
- [x] **E** `tools/plot_dynamic_test.js` 重写完成（2026-11-19）。
      删掉全部针对 `target_track` / `plot.track` / 「态势标绘」的旧断言，改为测当前设计：
      标绘类别常驻在册、未声明类别动态成层、显隐意图跨注销保留、导出/导入往返、
      计划轨迹取点（当前位置→剩余航路点 / 无航路点按帧并在转向处截断），
      并静态断言 `predict-` 实体用虚线材质 + 终点箭头。
      `readSrc()` 统一做 CRLF→LF，修掉按字符串切 `app.js` 在 CRLF 检出下必断的问题。
      验：`node tools/plot_dynamic_test.js` PASS，其余 10 个 JS 自检全过。
      **顺带登记两处既有缺陷**（测试如实报 `DEFECT`，不计 fail，也未改产品代码）：
        1. `app.js:7055` `ensurePlotLayer()` 的"已存在"分支仍判 `def.group !== '标绘'`，
           但 `3a60ad7` 起标绘层是 `group:'TS图层', subgroup:'标绘'`；当某一轮恰好有
           图层增删走重建分支时，既有标绘层会被误判为"非标绘类别层"而**不补 provider**
           （菜单里这些层的要素列表变空）。集合一致的 sync 分支不经过这里，所以平时看不见。
        2. `situation.js:677` 导入校验仍用 `l.group === '标绘'`；同样自 `3a60ad7` 起恒 false，
           `plotLayers` 只剩目录默认类别，TS 定义里显式写 `plots[].layer` 的标绘会被
           静默改回默认图层（`warnings` 里提示"不在册"）。
      另：`situation.js:660-661` 还有两行提到 `plot.track` 的注释，是已删设计的残留说明
      （那里实际计算 `systemOnlyLayers`，当前目录无 system 类型，逻辑空转但不致错）。
      两处行为缺陷已分别登记，旧注释随相关修复一起核对。
- [x] **标绘图层重建时补齐要素提供者**（`3d4fea0`）。`ensurePlotLayer` 改判 `subgroup`，重建分支上既有标绘层不再被拒；`plot_dynamic_test` 对应断言转正常断言并通过。
- [x] **导入保留标绘的显式图层归属**（`3d4fea0`）。`situation.js` 导入校验改判 `subgroup`，TS 定义里显式 `plots[].layer` 不再被静默改回默认类别。验证只到源码级断言（`plot_dynamic_test` 检查 `.subgroup === '标绘'` 在场且 `.group === '标绘'` 消失）与 `Plotting.importAll` 往返；未做经 `situation.js` 入口的浏览器往返。
- [-] **C** intelpipe worktree 清理：**核实后不存在这项**。`git worktree list --porcelain`
      只有一个工作树，`.git/worktrees` 不存在，`prune --dry-run` 无输出；
      `git branch -a --contains aa13be6` 显示 main 与 origin/feat/battle-timeline 都已含该提交。
      原清单判断有误，撤销
- [x] **D** CM-11 主战坦克拼接（7764562）：M60A3 车体 + M48A5 炮塔，7,093 面、0.40 MB，
      炮塔按实测亮度比压到车体配色；已入库 specs / licenses / bboxes / manifest / entityLod。
      **交接文档记录的组合方向相反，已按公开资料纠正并在 CREDITS 写明**
- [x] **备份（本机部分）** `_backup_20261106/shenji_agent_local_20261106.bundle`（254 MB，verify 通过）；
      `_recovered/timeline_from_nfs_20261106.bundle`（从 NFS 取回的服务器真实历史）
- [-] **备份（服务器部分）**——随本机迁移撤销。本机 bundle 见上条；服务器历史已由
      `_recovered/timeline_from_nfs_20261106.bundle` 取回过一次。

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
- [!] CP-SAT 火力分配——**阻塞：待产品决策**。本机无 `ortools` 也无 `pulp`，
      真要做还得先选求解器；且「自动分配火力」与项目「只提供动作与查询、
      不做自动决策」的定位存在张力，需先定这条边界。
      2026-11-19 复查补两点：现有引擎已有**逐单位自动选目标 + 交战裁决**
      （`World._engage`），所以「没有自动决策」不能一概而论，真正的边界是
      「不做跨单位的全局火力分配建议」；且求解器是新增外部依赖，
      按最小方案原则本轮不引入
- [!] 屏外方向箭头、AI 两段式管线、2D 视图——**阻塞：待产品决策**（三项各自是独立取舍，
      不能打包批准）。2026-11-19 复查：三项都只改 `web/src`（`app.js` / `overview.js`），
      后端无对应改动，因此本轮迁移口径下的「继续推进」不覆盖它们——照维护规则 4 等批准

### 地图与地理数据剪藏评估（2026-11-19）

- [x] **剪藏评估**（`a6a5328`）。已交付 [地图与地理数据集成评估](goals/webclip_integration_assessment.md)，静态核对了影像时间写入、地图要素来源和瓦片诊断入口。
- [x] **影像采集时间入库**。采集时间区间随栅格传入瓦片登记，`as_of` 取区间末日；旧缓存里用入库日冒充采集日的条目只做读取时投影，不回写库。远端隔离检出 `test_acquisition`、`test_sentinel`、`test_resource_plan`、`test_resources_v1`、`test_resource_fetch`、`test_case_imagery` 共 66 项通过；同一批新测试对未修复代码 6 项失败，确认用例真的覆盖该缺陷。
- [x] **影像时间分项展示**（`c9b86dc`）。资源列表与编辑表单各出一组只读三项：采集区间（单景只报一个时间戳）、检索参考日期、入库时间；没有采集记录（XYZ 源、旧缓存）写"未记录"，不拿入库日或 `as_of` 冒充。采集检索的 `requested` 随栅格写入 `notes.acquisition`。远端 `ui_provenance_test` 15 项通过；把采集时间改成 `as_of`、删掉来源三项、摘掉复核渲染三种改动各自失败。
- [x] **地图要素来源说明**（`c9b86dc`）。标绘编辑菜单顶部只读显示来源/引用/精度，缺失写"未记录"；`getItem` 透出这三项，`exportAll` 仍只带控制点。`plot_dynamic_test` 增两项断言，远端通过。
- [x] **标绘两处缺陷修复**（`3d4fea0`）。`app.js` `ensurePlotLayer` 与 `situation.js` 导入校验改用 `subgroup === '标绘'`；`plot_dynamic_test` 里两条登记的 DEFECT 转成普通断言并通过。
- [x] **地图定位回归样例**。新增 [合成样例自检](situation3d/tools/verify_map_location.py)，在远端 Chromium 的固定视口点击两张民用地点卡片，验证投影中心、零半径不画圈及清除旧圈、正半径圆的位置和双轴、质心标签。五组独立负向对照按预期失败，恢复基线后通过。
- [x] **关联复核状态展示**（`c9b86dc`）。情报列表新增"关联复核"区块：确认/否决各自计数，逐对列出 `a ↔ b`、分析员备注与时间，每条带撤销按钮（POST 到 `/api/intel/correlations/<id>`，撤销后该对退回算法判定）。只用传入对象的 `correlations.items`，跨案例隔离依赖后端 `case_id` 过滤。远端 `ui_provenance_test` 覆盖确认/否决呈现、备注、空集合不渲染、撤销请求路径；服务端语义由 `test_intel_correlations` 守住。
- [!] **双期影像比对**——**外部依赖：民用样区两期影像**。需要同一区域两期日期可信、
      许可明确、可配准的影像才能验；本机 `web/tiles` 只有单期（z11 一期快照）。
      另需按规则 4 批准界面。数据到位前不改代码
- [x] **历史时点查询（只读筛选）**（见 [最小方案](goals/intel_asof_query_design.md)）：
      `IntelManager.overview(case_id, as_of=...)` 与 `GET /api/intel/overview?as_of=`。
      判定：`ingest_time <= T`（存在性）+ `event_time <= T`（可见性）同时成立；
      缺 `ingest_time` 的旧记录按"存在性未知"排除并在 `warnings` 里点名。
      返回体新增 `as_of` / `as_of_applied` / `late_records_skipped` /
      `manual_status_not_historical`；不传 `as_of` 时这些字段固定为
      null/false/0/false，其余逐键不变。`as_of` 进缓存键，否则历史视图会拿到
      当前视图算好的 `links`；历史视图不喂基线统计（`observe=False`），
      免得筛选后的条数污染运行期分布。非法或未来 `as_of` 返回 400。
      `tests/test_intel_as_of.py` 6 项 + 原 intel 用例 78 项通过；全量 858 passed。
      **未做**：UI 入口（改 UI 须另批）；派生结果快照（B 类）
- [-] **人工情报状态历史**——**已核实的硬边界**：撤销只改 `status` 并覆盖 `updated_at`，
      「T 时刻是否 active」还原不了。要还原须给 `manual.py` 加只追加的状态日志，
      属另立任务；当前按当前值呈现并标注 `manual_status_not_historical: true`
- [!] **三维瓦片运行诊断与离线验证**——**外部依赖：代表性授权样本**。本机
      `web/` 下没有 `tiles3d/`（只有 `tiles/` 影像与 WSF3D，10651 个文件），
      印证了这一项缺的确实是样本而不是代码。`?diag=1`
      运行时读数已存在（`web/src/app.js:270`），远端合成 ZIP 的外部引用/缺失内容/
      层级与剪枝/哈希一致性已验过，Chromium 的加载失败、挂载、关闭三态也已验过。
      仍缺的是授权 3D Tiles 样本的首屏、冷/热缓存、内存基线——手头只有合成 ZIP，
      测不出真实流式加载。诊断界面本身按规则 4 需批准

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

- [x] **GP-05** 单位级交战姿态（3fba886）：`scenario["roe"]["unit_postures"]` =
      `{uid: "free" | "hold_fire" | "hold_position"}`。姿态**只替换全场手册的 `policy`
      一项**，责任区、确认门槛、还击窗口不动；`free→weapons_free`，
      `hold_fire/hold_position→hold_fire`。没写这张表时 `World.unit_postures` 为空，
      判定仍只看 `world.roe`，与旧行为一致。人工 `fire` 命令仍优先放行、NSL 仍优先。
      非法姿态/非法 uid 只追加 `world.roe_warning`，不让案例加载失败。
      `tests/test_unit_posture.py` 6 项。**未做**：界面上的姿态设置入口（改 UI 须另批）。
      写法与示例：`situation3d/docs/geo_engine/OPT_IN_RULES.md` §2
- [x] **GP-06** 队形落点分配（5bb21c1）：`mission["formation"] = {"spacing_m": m}` 后按
      ⌈√n⌉ 栅格槽位、以「重心→目标」方向旋转生成逐单位落点，仍走 `_point_free()`
      校验，失败整段回退旧路径。没写 `formation` 或间距非法 → 仍走
      `_passable_point()` 的 17 候选点，与旧行为一致。`retask()` 新建任务会保留该字段。
      与 GPRTS 原方案差一处：它按运动层分组 + 最近单位贪心，这里按 uid 顺序占槽，
      更适合「同批出发」；分批抵达的队形不在本次范围。
      `tests/test_formation_slots.py` 5 项。写法与示例：`situation3d/docs/geo_engine/OPT_IN_RULES.md` §4
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

状态（2026-11-19）：三问已按本表可沿用的部分实现，见 §0.5 的「指挥链 / 通信」（3fba886）。
第三问取「不做多跳」——它与现有单跳实现和既有测试一致，不需要外部口径。

---

## 维护规则

1. 完成一项就改状态并在后面补上 commit 短哈希；合并后不再可能回退的，写进第 1 节。
2. 阻塞项必须写清阻塞原因，恢复后从这条接手。
3. 新增任务先归组（收尾队列 / Palantir / 调研衍生 / 外部项目调研），不要另起清单。
4. 需要改 web UI 的任务标 `[!]` 并等用户批准；纯后端先做完。
