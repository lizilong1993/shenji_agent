# 独立评审问题修复记录（P5）

> 2026-10-26。评审对象：`feat/task-resource`。修复后全量 493 passed / 14 skipped，并重跑验收
> （未来日期、历史日期、并发 resolve、provide_file、UI Playwright、确定性对拍）。

## Major

1. **刷新无法消除 stale/历史日期缺口**：时效参考日改为 `min(想定日期, 今天)`；历史日期的 stale
   建议改为 “调整日期/提供历史快照”，不再推荐在线刷新；auto 仅在目标日 ≥ 今天且缺口不是
   `no_point_in_time` 时尝试刷新；同一模板同日刷新复用已有条目，避免重复重建与注册表膨胀。
   仅对世界数据类（envpack/case_roads/bathymetry/weather/intel）做日期过滤，角色卡/平台/武器等
   代码数据不受历史想定日期影响。
2. **任务 bbox 校验未接线**：runner 在解析前用“锚点落入的最新 envpack bbox”校验所有任务点；
   越界坐标直接以 task request rejected 退出，不再伪装成 envpack 资源缺口。
3. **逐条任务 side 绕过覆盖/冲突**：`_unit_override` 与 mixed 冲突检查改为按每条任务自身 side
   分组；修复部分覆盖时 `units` 修剪误用 dict 键的 bug；新增导演视角（逐条 side）测试。
4. **资源引导只按 kind 补齐**：`_ensure_local_resources` 每次扫描仓库产物，对新增/内容变化的条目
   执行 register(replace)，不再因“该 kind 已存在”而漏登记新路网/新平台文件；仅在全部缺失时补默认 envpack。

## Minor

5. `start_s` 现在拒绝 NaN/Inf。
6. resolve 加进程内锁 + `_spawn_run` 存活 pid 守卫；UI 提交期间禁用按钮。并发 resolve 实测
   一个 200、一个 409。
7. runner 进入 waiting 时清空 pid；取消只对存活 pid 发送信号。
8. `provide_file` 路径限制：服务端与 runner 都校验绝对路径、存在性与允许根目录
   （`SITUATION_RESOURCE_ROOTS` + envpacks/var/resources），非法路径返回 400；entry_id 格式错误
   转为 ProviderError。
9. 运行进入 running 时清空 `resource_gaps`；计划摘要与运行摘要分离保留。
10. `ResourceDecision.as_dict` 不再自动填 `decided_at`，相同决策的计划哈希稳定。
11. JSON 请求体上限从 64 KB 提高到 1.2 MB；resolve 校验 action 枚举与 provide 路径。
12. 控制台显示 waiting 的 fetch_error、degraded 的降级数据项数；页面重新可见且用户未改日期时
    刷新默认想定日期；stale 缺口文案改为“数据时效不足”。

## 复验

- 2027-01-01 + offline → completed/ready（不再误判当前快照过期）。
- 2026-09-01 → waiting（仅 envpack no_point_in_time，pid=None）→ provide_file → completed/ready，
  gaps=0。
- 部分注册表（仅 india_china 路网 + envpack）→ 运行前自动补登 taiwan_strait，计划 ready。
- 同 seed 同计划两次运行：payload 哈希 `59ae1fb9…` 一致，plan_hash `b273cf33…` 一致。
- Playwright：无 pageerror；等待→提供路径→完成；资源列显示 degraded 与降级数据提示。
