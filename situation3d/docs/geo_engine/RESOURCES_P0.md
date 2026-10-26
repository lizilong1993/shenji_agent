# 资源注册层 P0（契约 + 索引）

> 2026-10-26，分支 `feat/task-resource`。P0 只做“描述与登记”，不抓取、不解析、不接引擎。
> 总体方案见 `docs/engine_task_resource_plan_2026-10-26.md`（本机备份目录）。

## 1. 目标

把引擎实际依赖的输入统一登记成 `ResourceEntry`：内容哈希、来源、数据时效（`as_of`）、
获取时间（`fetched_at`）、许可、质量等级。后续 P2 的预检/计划、P3 的在线获取都读写同一份索引，
避免服务端、runner、脚本各写一套逻辑。

## 2. 契约（`situation3d/resources/model.py`）

| 字段 | 语义 |
|---|---|
| `entry_id` | `kind:name`，例如 `envpack:taiwan_strait` |
| `kind` | 资源类型，见 `resources/catalog.py` |
| `path` | 绝对路径；索引只引用、不复制大文件 |
| `content_hash` / `size` | 文件或目录树的 sha256（目录按相对路径+大小+内容排序计算） |
| `version` | 数据版本；未显式提供时取内容哈希前 12 位 |
| `schema` | 数据契约标识（`msgeo.envpack.v2` 等） |
| `provider` / `source` | 来源适配器与可读来源说明 |
| `as_of` | **数据时效**（数据描述世界的时点），用于想定日期匹配 |
| `fetched_at` | **获取时间**（何时进入本地），用于缓存 TTL 判断 |
| `bbox` | 适用空间范围（可选） |
| `license` / `quality` | 许可；`measured/derived/default/unknown` |
| `pinned` | 用户钉住，删除/替换需显式解除 |

同时冻结了后续阶段使用的 `ResourceGap`（缺口分级与建议）、`ResourceDecision`（用户决策）、
`ResourcePlan`（运行钉住的条目集 + 计划哈希）。

## 3. 类型与默认必要性（`resources/catalog.py`）

| kind | 默认必要性 | 默认时效 | 说明 |
|---|---|---|---|
| `envpack` | critical | 30 天 | 地形/地类/水系/建筑/路网 |
| `case_roads` | critical | — | 案例初始编成与路由 |
| `rolecards` / `platforms` / `weapons` | critical | — | 能力、平台剖面、规则数据 |
| `intel_bundle` | advisory | 1 天 | VGI 线索，只作提示 |
| `bathymetry` / `weather` | advisory | 365 / 1 天 | 水深、天气时间线 |
| `commander_service` | advisory | — | 远程指挥，离线回退 RuleCommander |
| `tiles3d` / `imagery` | cosmetic | 365 天 | 仅显示 |

必要性只是默认值；P2 会按用户任务细化（例如两栖任务把水深提升为 critical）。

## 4. 索引与 CLI

索引位置：`situation3d/var/resources/index.json`（gitignored，按 worktree 各自独立）。
所有写入原子替换并加文件锁。

```bash
python3 scripts/resource.py scan            # 扫描并登记现有 artifact（可重复执行）
python3 scripts/resource.py list [--kind K] [--long] [--json]
python3 scripts/resource.py show ENTRY
python3 scripts/resource.py verify [ENTRY]
python3 scripts/resource.py pin|unpin ENTRY
python3 scripts/resource.py remove ENTRY [--force]
python3 scripts/resource.py kinds
```

`scan` 覆盖：`envpacks/*/manifest.json`、`engine/rolecards/data`、`platforms/*.json`、
`rules/data`、`engine/data/roads/*.json`。缺失目录跳过，坏数据逐项报错不中断。

## 5. 语义要点

- `scan` 以内容哈希判定同一资源：哈希不变只刷新元数据；哈希变化视为新版本，`scan` 会替换登记，
  但 **pinned 条目保持 pinned**。程序内 `register(replace=False)` 遇到哈希变化会抛
  `ResourceConflict`，必须显式替换。
- `verify` 区分 `ok / missing / size_mismatch / hash_mismatch`；运行（P2）在启动前必须全量校验。
- `as_of` 优先取数据自带时点：envpack 取 `built`（或 v3 `as_of`）、路网取 `osm_base_utc`、
  规则取 `source_date`；仓库内其它数据取该文件最后一次 git 提交日期（非 checkout mtime）。
- 当前 TW2 envpack 登记为 `quality=derived`：清单 provenance 显示 `dem=fallback-basic`，
  即地形来自 basic 地图回退，P2 预检应把这一点呈现给用户，而不是当作实测数据。

## 6. 测试

```bash
cd situation3d && python3 -m pytest tests/test_resources_v1.py -q
```

覆盖：元数据解析、哈希随内容变化、索引持久化、篡改/缺失检测、同内容合并、哈希冲突、
pinned 保护、目录哈希与文件创建顺序无关、仓库扫描（五类资源 + 坏数据隔离）、计划哈希。

## 7. 未覆盖（后续阶段）

- P2：`resolver.py` / `preflight.py`，把 `ResourceGap` 接入 `/api/sim/runs` 与 `waiting_resources` 状态。
- P3：`providers.py` 在线获取与缓存写入 `var/resources/cache/`；代理与超时探测。
- 路径为绝对路径，索引不跨机器迁移；换机后重新 `scan` 即可重建。
