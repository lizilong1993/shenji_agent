# P4 事件驱动时间、扫描错峰、空间裁剪与 800 实体（2026-10-23）

> 结论先行：传感器/弹着/天气/指挥从"每 tick 轮询"改为**到期才处理的事件门**；
> 空闲世界可用 `World.advance_to()` 直接跳到下一事件（运动仍按固定 dt 积分）。首次全量扫描后，
> 每个观察者按 uid 确定性相位错开，锁步尖峰被打散。新增 800 实体基准档位并纳入门槛。
> 感知概率改为公开的 Johnson/NIIRS 50% 探测距离约定标定（非场试实测）。
> 全量 **314 passed, 3 skipped**；P6 台海 envpack 结果不变（734 事件）。

## 1. 事件驱动时间

- `World._compute_table_seen` / `perception.compute_seen`：无观察者到期且无死亡/登载变化时
  直接返回上一帧视野；只有到期 tick 才构建实体表、cover/elevation 元数据。800 实体、
  30 s 周期下约每 tick 都有一批（约 1/30）观察者到期，因此收益是"只处理该批"，而非
  "完全不构建"；低密度场景可整 tick 跳过。
- `_resolve_pending_impacts`：维护 `_next_impact_s`，未到期直接返回；结算后重算最早到期。
- `_update_environment`：仅在天气时间线条目或昼夜边界（6/8/18/20 时）到期时重算，
  `_next_env_s` 记录下一事件。
- 指挥器已有 `_commander_next_s` 门控；扫描到期表 `_scan_due` / `_phys_scan_due` 同时作为
  事件源。
- `World.next_event_s()` 返回最早事件；`World.idle()` 判断无订单/意图/路线；
  `World.advance_to(end)` 在空闲时段直接跳到下一事件（跳跃期间仍结算补给），到达事件时刻用
  `step(0)` 处理；有运动时按 `dt` 积分。该接口用于批量/无头推进，不替换运动积分器。

## 2. 扫描错峰与子周期掷骰

- 首次扫描仍是 t=0 全量（保持初始态势即时可见）；之后每个观察者维护相位对齐的 due：
  `phase = hash(uid) % interval`，首轮为 `t0 + interval + phase`，此后取不小于
  `max(上一 due, now)` 的下一个 `phase + k·interval`。因此即使多个观察者因时间跳跃或大 dt
  在同 tick 到期，相位仍然保留，不会塌回锁步。
- 仅时间回跳时重锚 due（`_reanchor_scan_due`）；正常前进不改写相位。物理模型同规则。
- 物理掷骰桶从固定 10 s 改为 `min(10 s, scan_interval_s)`（下限 1 s）：周期 <10 s 时每次
  扫描都能得到新的掷骰结果；周期 ≥10 s 语义不变。

## 3. 空间裁剪

- 规则格成员索引 `_cell_members` 在 `_refresh_cell_cache` 中一并构建，`entities_in_cell`
  从 O(N) 全表扫描改为按格取成员（目标/占领判定使用）；`_update_objectives` 的敌方邻格
  检查仍需全表遍历，未纳入索引。
- 碰撞沿用 P0.2 的 100 m OBB 空间桶；感知沿用规则格距离预筛 + 静态 `cell_line_of_sight`
  缓存 + `_detect_cache`，扫描周期内先按格距过滤再算 LOS。
- 未引入多进程/多线程并行：Python GIL 下对当前负载无收益，且会破坏确定性；真正分区并行
  需要独立的事件调度与归约设计，留待后续。

## 4. 800 实体性能（70 tick，warmup 10，shared host）

| 规模 | p50 | p95 | p99 | 事件哈希 |
|---|---|---|---|---|
| 300 | 6.6–7.5 ms | 9.8–11.2 ms | 11–36 ms | `ca6dc98b93194982` |
| 500 | 13.8–15.4 ms | 17.5–20.9 ms | 18–59 ms | `6f92ee20e2982bae` |
| 800 | 30.8–34.9 ms | 36.2–42.6 ms | 58–67 ms | `7d9e9cfbd18758d3` |

门槛仍为 p95 80 ms / p99 150 ms（含共享宿主机噪声余量）。300/500 的探测事件结果未变；
800 为新增基准。t=0 的首个全量扫描在 800 实体约 7.8 s（一次性，bench 预热排除），其后扫描
按相位摊平到约 0.2 s/扫描 tick。

## 5. 公开经验标定（感知）

- 传感器记录的名义距离按公开 Johnson/NIIRS 目标获取约定视为 **50% 探测距离**：
  `p(d)=p0·exp(-d/scale)`，`scale = range / ln(p0/0.5)`，因此裸曲线
  `p(nominal)=0.5`、`p(0)≈p0`、超出名义距离为 0。实际 `detection_probability` 还会乘
  特征/能见度/烟幕/隐蔽/运动因子，故场景内名义距离处的有效概率不是恰好 0.5。
  数据与引用见 `rules/data/detection_calibration.json`。
- `sim/calibration.py` 提供 `nominal_scale_frac`、`probability_at`、`report`；
  `scripts/calibrate_detection.py` 输出各传感器类型的 50%/半程/四分之一程概率。
- 这是**公开经验准则标定**（Johnson 1958；NIIRS/GIQE 1997），不是本项目的场试实测数据；
  真实场景仍需外部探测试验数据替换。

## 6. 测试

- `tests/test_event_scheduler.py`（7 项）：新世界首步、空闲快进、扫描到点唤醒、天气时间线
  在跳跃后生效、弹着只在 due 结算、订单存在时禁用快进、无线电共享不逐跳中继。
- `tests/test_calibration.py`（4 项）：公开准则与引用、名义距离 P=0.5、距离单调衰减、
  报告数值有限。
- `tests/test_p3_acceptance.py` 增加相位/子周期掷骰断言；`test_bench_regression.py` 增加
  800 实体档位。
- 全量 **316 passed, 3 skipped**（319 项）。

## 7. 限制与后续

- 运动积分仍是固定 dt；`advance_to` 只在空闲时快进（跳跃时同步结算冷却/压制/补给），
  不改变有运动时的步长语义；`scan_interval_s=0` 时禁用快进以保留逐 tick 感知。
- 观察者错峰只在首次全量扫描之后生效；首个 t=0 扫描仍是同步尖峰。
- 公开标定是经验约定，不是场试实测；有效探测概率还受特征/环境/隐蔽因子影响。
- 空间并行（多进程/分区线程）未实现；当前通过门控、相位与缓存满足 800 实体 p95 门槛。
- 公开标定是经验约定，非实测；后续应以试验数据校准 p0/scale 与火力穿深。
