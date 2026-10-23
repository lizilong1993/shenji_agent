# P3 规模、确定性与物理化验收（2026-10-23）

> 结论先行：传感器改为**按观察者固定 30 s 扫描节奏**（可由 `engine.scan_interval_s` 调整），
> 探测刷新与兵力数量彻底解耦；表模型/物理模型共享该节奏，未到期的观察者保持上一帧视图。
> 观察者之间为**同步锁步扫描**，每周期有一次扫描尖峰（错峰调度未采用）；基准窗口扩到
> 70 tick（覆盖 ≥2 个周期），300/500 实体 p95 **25–28/34–49 ms**、p99 **31–38/66–69 ms**
> （standalone；p95 不含尖峰，p99/门槛覆盖尖峰），事件哈希不变；新增物理化验收测试：
> 无穿模/瞬移/航向跳变、转弯半径与制动距离双边界、探测概率校准（最大偏差 1.2%）、ROC AUC
> 1.0、同 seed 事件/位置哈希一致，以及扫描节奏语义（保持/到期/跨周期/零周期/物理到期）。
> 全量 **303 项测试通过（3 skipped）**。

## 1. 问题（ARCHITECTURE_GAPS G8 + P3）

- 实体数 >50 时 `_compute_table_seen` 按 5 片轮扫，**探测节奏随兵力规模变化**，不同规模结果
  不可比；性能门槛靠缓存维持。
- 缺少物理化验收：穿模/瞬移计数、转弯半径/制动距离、感知 ROC、复现性、性能门槛。

## 2. 方案

### 2.1 扫描节奏（`sim/world.py`、`sim/perception.py`）

- `scan_interval_s`（默认 30 s，`engine.scan_interval_s` 可配）为每个观察者维护
  `next_scan_s`；只有到期的观察者在本 tick 重新计算视野，其余沿用 `world.seen` 上一帧；
  首次 t=0 全量扫描。表模型与 P1.4 物理模型都使用同一节奏（`_scan_due` / `_phys_scan_due`），
  shadow 模式仍强制全量扫描以便逐 tick 对拍。
- 移除了按实体数选择 1/5 片轮扫的逻辑，刷新周期只由时间与配置决定；规模扩大时每个观察者
  的探测节奏保持不变。所有观察者由同一 `time_s` 驱动，属于锁步扫描：刷新时刻重合，每周期
  出现一次同步尖峰；逐观察者错峰（哈希偏移）未采用，留作后续优化。
- 默认 30 s 节奏改变了小规模（≤50 实体、dt=1）逐 tick 刷新的旧语义；依赖逐 tick 重扫的
  场景/测试需显式设置 `scan_interval_s=0`。负值统一按默认 30 s 处理（与场景配置一致），
  0 表示逐 tick 扫描。

### 2.2 验收测试（`tests/test_p3_acceptance.py`）

- 规模无关节奏：10 vs 70 实体的观察者 `next_scan_s` 完全一致（30/60 s）；
- 节奏语义：未到期观察者保持上一帧视野、到期后按新位置重算；时间跨越多个周期只扫一次并
  重锚 `next_scan_s`；`scan_interval_s=0` 每 tick 重扫；死亡观察者从 due/seen 清理；
  扫描次数 = 周期数×观察者数（时间回跳会重锚 due，shadow 全量模式将 due 钉在
  now+interval，不会累积漂移）；物理模型 `_phys_scan_due` 同步推进；死亡与登载观察者
  均从 due/seen 清理（14 项验收测试）；
- 不变量（40 实体、120 tick、随机行军令）：出生去重叠后初始 OBB 相交对为 0；逐 tick OBB
  相交（margin=0）、位移 `≤ max_speed·dt+2 m`、航向跳变 `≤ max_yaw_rate·dt+3°` 全部为 0；
- 运动学双边界：轮式圆角每顶点按弧长曲率 `κ = 2·sin(Δψ/2)/chord ≤ 1/R_min ×1.1`，单顶点
  转角 `≤35°`；直线制动从首帧减速起测，距离落在 `[0.8, 1.3]×v²/(2·decel)`，逐帧
  `Δv ≤ decel·dt`；路线完成不再瞬时清零速度（残余速度在下一 tick 原地减速）；
- 感知校准：物理模型解析概率 vs 2000 次确定性掷骰，最大偏差 1.2%；距离扫描 ROC
  （3 km 内/外）AUC=1.0；**这是实现自洽性验证，非实测数据标定**；提供
  `scripts/calibrate_perception.py` 输出校准 JSON；
- 复现性：同 seed 120 tick 事件与位置 SHA256 一致；基准 70 tick 300/500 事件哈希稳定。

### 2.3 性能（负载≈40 的共享宿主机）

| 规模 | p50 | p95（70 tick 窗） | p99（含尖峰） | 旧 p95（P1.5，40 tick 窗） |
|---|---|---|---|---|
| 300 | 6.6–6.9 ms | 25.3–25.7 ms | 31–38 ms | 16–31 ms |
| 500 | 12.8–13.2 ms | 30.7–37.4 ms | 66–69 ms | 31–51 ms |

窗口 70 tick（warmup 10）覆盖 ≥2 个扫描周期；60 个样本中 p95 只到第 3 大值，会漏掉两次
锁步扫描尖峰，因此另设 p99 与 150 ms 门槛（全量测试并发下仍有 ≥2× 余量）。哈希与 P2
相同：300 `ca6dc98b93194982`、500 `6f92ee20e2982bae`（本基准为静态世界，哈希不检验节奏
回归；节奏回归由 `CadenceSemanticsTest` 的扫描次数断言守护）。
P6 envpack 1440 tick（dt=30，等于扫描周期，因此每 tick 全量扫描）结果与 P2 一致：734 事件、
323 发射、320 弹着、15 毁伤、直瞄 6；共享宿主负载下 p50 84 ms、p95 219 ms。

## 3. 测试

- 新增 `tests/test_p3_acceptance.py` 14 项（节奏语义/不变量/曲率/制动/校准/ROC/确定性）。
- 既有感知/缓存测试按“固定节奏”语义更新：需要逐 tick 重扫的用例显式
  `scan_interval_s=0` 或推进到下一扫描时刻；`test_envpack_builder` 加 `--no-bath` 避免
  网络依赖（离线缓存环境）。
- 全量 **303 passed, 3 skipped**（306 项）。

## 4. 限制与后续

- 运动积分仍为固定 dt；扫描/弹着/天气/指挥已事件门控，空闲可用 `advance_to` 快进；
  空间并行未实现。详见 `P4_event_spatial.md`。
- 扫描周期为全局单一值，不同传感器（雷达/光学）尚无独立刷新率；spotting 模型仍逐 tick
  计算（未纳入节奏批处理）。首次全量扫描仍同步；其后按 uid 相位错峰，见
  `P4_event_spatial.md`。
- 物理掷骰桶已改为 `min(10 s, scan_interval_s)`（下限 1 s）：周期 <10 s 时每次扫描都有新
  掷骰结果，见 `P4_event_spatial.md`。
- 物理保持期为 `max(60 s, 2×扫描周期)`：周期 >30 s 时保持时间相应加长，避免两周期之间
  目标闪烁。
- ROC 自洽性验证之外，名义探测距离已按公开 Johnson/NIIRS 50% 约定标定
  （`sim/calibration.py`）；仍非场试实测，参数确认需外部试验数据。
- 800 实体已纳入基准（p95 36–43 ms），多进程/空间并行仍未实现；网络类数据源测试在
  离线环境需缓存或降级。

## 5. 复现

```bash
cd situation3d
python3 -m pytest tests/test_p3_acceptance.py -q
python3 scripts/calibrate_perception.py --seeds 2000
python3 -m pytest tests/test_bench_regression.py -q
python3 scripts/run_geo_demo.py --case taiwan_landing --envpack envpacks/anping_v2c \
    --ticks 1440 --dt 30 --out /tmp/tw_p3
```
