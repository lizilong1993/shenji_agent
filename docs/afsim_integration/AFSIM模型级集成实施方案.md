# AFSIM 模型级集成实施方案

> 版本：v1.1（2026-10-22，按独立复核意见修订）
> 执行基线：**geo-engine 分支**（具体分支名在 Phase 0 确认）
> 首期边界：**传感器探测 + 航迹处理 + 武器交战**
> 上游依据：《AFSIM 可集成性调研报告》
> 适用对象：在新会话中执行本任务的工程师/智能体

---

## 0. 这份文档怎么用

1. 先读《AFSIM 可集成性调研报告》，确认路线与合规约束。
2. **不要直接写代码**：先执行 §3 的 Phase 0 仓库勘察，把实际模块路径、帧/子时钟、平台状态、DEM 接口、数据集生成入口、测试命令摸清，写入 `docs/afsim_integration/00_baseline.md`。
3. Phase 0 完成后再按 §11 的任务顺序推进；每个 Phase 有独立验收标准。
4. 本方案是“**模型级重实现（clean-room）**”，不引入 AFSIM 源码、二进制、场景文件或第三方转发的安装包。
5. 全部 P0 完成后做**一次**独立复核（建议 2 名 fresh-context 复核者），不要在每个子任务后重复评审。

### 0.1 执行摘要（先看这个）

- **M0（Phase 0）**：只勘察、不改代码；确认数据集生成入口、基线哈希、帧/子时钟、平台绑定、DEM 与健康字段。
- **M1（Phase 1–5）**：感知层——雷达检测 + 航迹管理 + L1 接入；关闭模型时数据集零差异。
- **M2（Phase 6–7）**：交战层——3DOF + PN + 导引头 + 引信/Pk + LAR + 案例交战脚本；三案例至少各一条 launch→detonate 链。
- **M3（Phase 8–9）**：事件/诊断/文档/性能/独立复核/合并推送。
- 每个阶段必须通过本阶段验收再进入下一阶段；M1/M2 不合并主干，全部完成后一次合并。

---

## 1. 目标与边界

### 1.1 目标

在 geo-engine 的物理化底座上增加“感知—航迹—交战”子系统，使现有“射程圆 + 真实位置”的态势推理升级为：

- 传感器按物理模型产生**检测**（雷达方程、RCS、信噪比、检测概率、扫描周期、地形遮蔽）；
- 检测进入**航迹管理器**（卡尔曼滤波、航迹相关、生命周期、质量评估）；
- 武器按**发射包线 + 3DOF 弹道 + 比例导引 + 导引头 + 引信/Pk**完成交战；
- L1 态势推理改为读取**航迹**而不是真实位置，威胁度不再纯几何；
- 全过程产生**状态变化事件**，可回放、可统计、可做 A/B 对比。

### 1.2 非目标（本期不做）

- 不移植 AFSIM 的 WSF 内核、脚本 VM、插件 ABI、Wizard/Warlock/Mystic。
- 不做进程级 AFSIM 集成（mission/.evt/DIS 桥接）——许可资格未确认前不启动。
- 不做 C2 任务处理器、通信/数据链、电子战、空间/赛博（P1/P2，另立方案）。
- **不改前端 UI**：本期只改仿真侧与数据集；如需调试图层/面板，先征得用户同意。
- 不替换现有运动学/地理权威：传感器与武器只读取平台状态，不另建一套坐标/运动系统。
- 不做 Swerling 0 / 复杂杂波 / 多径 / 大气波导；首期只做 Swerling I 雷达检测。
- 不使用任何密级或来源不明参数；只用公开资料/项目既有型号数据，并记录出处。

### 1.3 完成定义（DoD）

同时满足以下条件才算本期完成：

1. `models.*` 全部关闭时，ZY1/TW1/TW2 三个案例的数据集与**同一生成入口的基线**逐字节一致（或 SHA256 一致）。
2. 开启感知后，同一配置/种子连续两次运行输出**完全一致**（确定性）。
3. L1 推理在感知开启时使用每帧 `perception` 数据；关闭时自动回退旧逻辑。
4. 雷达探测距离与型号标称值在**明确判定条件**（Pd、Pfa、RCS、高度、姿态）下偏差 ≤ ±20%；超出需书面解释，禁止静默调 RCS 反凑。
5. 非机动目标拦截脱靶量 < 5 m；LAR 随目标高度/进入角单调合理。
6. 性能：模型关闭零开销；开启时感知+交战合计 p95 增量 ≤ 8 ms/帧（30 s 基准帧，80 目标/20 传感器/20 活跃弹的压力场景），并单独报告感知/交战耗时。
7. 现有 191 项测试全部通过；新增单测覆盖核心公式与生命周期。
8. 仓库无 AFSIM 源码/二进制/场景文件；`docs/afsim_integration/PROVENANCE.md` 记录全部参数来源。
9. 独立复核通过，合并回 geo-engine 分支并推送 GitHub。

---

## 2. 合规护栏（硬约束）

AFSIM 官方 WSF 文档标注 **CUI（受控非密信息）**，并声明使用、传播或披露受 README/LICENSE 限制；公开中文资料也提示授权对象受限。因此：

- **禁止**把 AFSIM 源码、头文件、DLL、场景 `.txt/.evt/.aer`、第三方“完整源码包”复制进仓库或依赖。
- **禁止**在代码注释/文档中粘贴 AFSIM 源码片段。
- **实现者可见资料**限定为：公开教材/论文（雷达方程、Swerling 检测、卡尔曼滤波、比例导引、发射包线等）、项目既有数据、公开型号资料。AFSIM 文档仅供规格作者做**功能名对照**，不作为实现参考。
- 建立 `docs/afsim_integration/CLEANROOM.md`：记录每位实现者读过的外部资料、未复制代码的声明、复核人签字（或会话记录）。
- 所有参数（RCS、功率、噪声系数、气动系数、Pk）必须记录来源：公开型号资料、公开论文、项目既有数据，或明确标注“工程假设”。
- 许可问题由项目法务/保密部门确认；本方案不构成法律意见。

---

## 3. Phase 0：仓库勘察（必须先做）

### 3.1 要回答的 12 个问题

| # | 问题 | 产出 |
|---|---|---|
| 1 | geo-engine 的包根与目录结构？`engine/` 还是独立包？ | 目录树 + 包根路径 |
| 2 | 仿真主循环/帧入口在哪里？`TICK_SECONDS` 多少？固定帧还是事件驱动？ | `file:line` 与调用链 |
| 3 | 平台状态（位置/速度/姿态/型号/健康）由哪个对象持有？坐标系？ | 数据类/字段清单 |
| 4 | 现有传感器/武器怎么表达？`sensor_range_km`/`weapon_range_km` 从哪来、谁消费？ | 字段与消费点 |
| 5 | DEM/地形如何访问？瓦片格式、采样 API、缓存？若无，是否可由前端资产离线导出？ | 接口签名或“不存在”结论 |
| 6 | ZY1/TW1/TW2 数据集的**权威生成入口**是哪条命令、哪个分支？基线哈希？ | 命令 + SHA256 + 耗时 |
| 7 | `inference.py` 输入输出契约？L1 威胁度公式？每帧结构？ | 函数签名 + 字段 |
| 8 | 测试怎么跑？191 项测试命令与基线耗时？ | 命令 + 基线 |
| 9 | 事件/帧输出在哪里？是否已有 event feed/CZML 写入？ | 模块与格式 |
| 10 | 远端服务器 worktree/分支现状？geo-engine 分支名？ | `git status`/`branch -a` |
| 11 | 平台是否有健康/毁伤字段？`kill` 事件由谁消费？ | 字段/消费链 |
| 12 | 三案例的平台数、传感器数、武器数、典型速度/高度？能否支撑 80/20/20 压力场景？ | 统计表 |

### 3.2 产出

`docs/afsim_integration/00_baseline.md`，包含：

- 上述 12 项的答案，尽量带 `file:line`；
- 基线命令与结果：测试全绿、三案例生成耗时、数据集 SHA256；
- **明确“数据集生成路径”**：若 demo 引擎与 geo-engine 不是同一入口，记录两条路径并增加 parity 任务；
- 目标模块落点建议（见 §4.2 的映射规则）；
- 任何与本文档假设不一致的地方，明确列出。

### 3.3 门禁

Phase 0 只读、不改代码。完成后把 `00_baseline.md` 给用户确认（或在新会话中明确记录“假设已核实”），再进入 Phase 1。若 DEM、健康状态、生成入口等关键前提缺失，先停下报告，不要自行假设。

---

## 4. 总体架构

### 4.1 数据流

```
平台运动学（现有）
      │  位置/速度/姿态/型号
      ▼
[SensorManager] ── 扫描调度 ──► 雷达方程 ──► LOS/地形 ──► 检测
      │                                              │
      │                                              ▼
      │                                     [TrackManager]
      │                                     相关/卡尔曼/生命周期
      │                                              │
      ▼                                              ▼
[Signature/Environment]                       航迹（TrackState）
                                                     │
                                                     ▼
                                          [EngagementManager]
                                     发射决策/中制导（读航迹）
                                     末制导（导引头量测）
                                     弹道/引信/Pk（读真值）
                                                     │
                                                     ▼
                                          交战事件（launch/hit/kill）
                                                     │
                                                     ▼
                                     [Inference L1] + 数据集/事件流
```

### 4.2 模块落点（Phase 0 按实际包根映射）

建议在 geo-engine 包根下新增：

```
<GEO_ENGINE_ROOT>/
  perception/
    __init__.py
    config.py            # 配置 dataclass + 校验 + 平台绑定
    mathutil.py          # dB、插值、稳定哈希、角度约定
    clock.py             # 帧内子时钟/扫描相位/微步调度
    los.py               # 4/3 地球视距 + DEM 采样
    signatures.py        # RCS/IR 表与姿态插值
    sensors/
      base.py            # SensorModel 协议
      radar.py           # 雷达传感器
    detection.py         # Detection 数据类
    track.py             # TrackState/TrackManager/Kalman6
    events.py            # 事件类型与序列化
  engagement/
    __init__.py
    config.py
    launch_computer.py   # LAR 预计算/查询
    flyout.py            # 3DOF 质点弹道 + CPA
    guidance.py          # PN/APN
    seeker.py            # 截获/FOV
    fuze.py              # 引信/Pk
    manager.py           # 交战生命周期
  tests/
    perception/...
    engagement/...
```

映射规则：

- 若仓库已有 `engine/` 包且所有模块都在其下，则落为 `engine/perception`、`engine/engagement`，**不要新建平行包**。
- 若 geo-engine 是独立包（如 `geo_engine/`），则放在 `geo_engine/perception`、`geo_engine/engagement`。
- 测试跟随仓库现有测试布局。

### 4.3 真值与航迹的接口边界（重要）

| 用途 | 输入 | 说明 |
|---|---|---|
| 发射决策 | `TrackState` | 只读航迹质量、位置、速度；不读真值 |
| 中制导 | `TrackState` 外推 | 目标丢失时按最后航迹外推 |
| 末制导/导引头 | 真值 + 导引头量测模型 | 截获/跟踪带噪声、更新率、FOV 限制 |
| 弹道运动学、碰撞、引信、脱靶量 | 真值 | 物理正确性所需；不得用于决策逻辑 |
| 战果评估 | 真值 + Pk | `kill` 仅发事件，由现有规则/后续模块消费 |

禁止把真值泄漏进发射决策；同样禁止用航迹替代碰撞判定。

---

## 5. 数据契约

### 5.1 配置（示例为**示意值，不是标定值**）

```json
{
  "models": {
    "perception": {"enabled": false, "deterministic": true, "seed": 20261022},
    "engagement": {"enabled": false, "pk_mode": "expected"}
  },
  "scene": {"enu_origin": {"lat_deg": 24.0, "lon_deg": 119.0, "h_m": 0.0}},
  "sensors": [
    {
      "sensor_id": "S_R06_RADAR",
      "owner_platform_id": "R06",
      "kind": "radar",
      "mount": {"az_deg": 0.0, "el_deg": 0.0},
      "fov": {"az_half_deg": 60.0, "el_min_deg": -5.0, "el_max_deg": 60.0},
      "scan_period_s": 2.0,
      "scan_phase_s": 0.0,
      "power_w": 1000000.0,
      "gain_db": 30.0,
      "frequency_hz": 3000000000.0,
      "bandwidth_hz": 1000000.0,
      "noise_figure_db": 3.0,
      "loss_db": 6.0,
      "pfa": 1e-6,
      "swerling": 1,
      "min_range_m": 1000.0,
      "max_range_m": 400000.0,
      "sigma_range_m": 50.0,
      "sigma_az_deg": 0.3,
      "sigma_el_deg": 0.3
    }
  ],
  "platform_signatures": [
    {"platform_type": "F-16V", "signature_id": "SIG_F16V"}
  ],
  "signatures": [
    {
      "signature_id": "SIG_F16V",
      "rcs_m2": {
        "default": 5.0,
        "by_aspect_deg": [[0.0, 8.0], [90.0, 5.0], [180.0, 3.0]]
      }
    }
  ],
  "platform_loadouts": [
    {"platform_id": "R12", "weapon_ids": ["W_HQ9B"], "max_rounds": 4}
  ],
  "weapons": [
    {
      "weapon_id": "W_HQ9B",
      "kind": "sam",
      "guidance": "pn",
      "navigation_constant": 4.0,
      "max_g": 30.0,
      "seeker": {"range_m": 25000.0, "fov_deg": 60.0, "update_period_s": 0.1},
      "fuze_radius_m": 15.0,
      "pk": {"by_miss_m": [[0.0, 0.95], [5.0, 0.85], [15.0, 0.5]]},
      "motor": {"thrust_n": 120000.0, "burn_s": 8.0, "mass_kg": 900.0, "fuel_kg": 300.0},
      "aero": {"cd0": 0.35, "area_m2": 0.12}
    }
  ]
}
```

要求：

- 所有数值带单位后缀（`_m`、`_s`、`_db`、`_hz`、`_w`、`_kg`），避免歧义。
- 配置加载时做范围校验（功率 > 0、Pfa ∈ (0,1)、扫描周期 > 0、Pk ∈ [0,1]）。
- **平台绑定校验**：每个 `owner_platform_id` 必须能在三个案例里解析；每个 `platform_type` 必须有 signature 映射；每个 `platform_id` 的 loadout 必须引用存在的 weapon。
- 配置中保留 `source`/`note` 字段，供 PROVENANCE 自动汇总（可选）。

### 5.2 运行时数据类

| 类型 | 关键字段 | 说明 |
|---|---|---|
| `Detection` | `t, sensor_id, target_id, snr_db, pd, detected, range_m, az_deg, el_deg, los_clear` | 每次扫描结果；`detected` 由 Pd 采样 |
| `TrackState` | `track_id, status, t, pos_ecef_m, vel_ecef_mps, cov_pos_3x3, quality, sources, last_update_s, lock_count` | 完整位置协方差用于门限 |
| `TrackUpdate` | `t, track_id, sensor_id, measurement, innovation` | 调试与指标 |
| `LaunchOrder` | `t, shooter_id, weapon_id, target_id` | 由案例脚本/自动规则注入；target_id 解析为当前航迹 |
| `EngagementEvent` | `t, seq, type, actor, target, data` | type: detection_start/detection_end/track_start/track_drop/launch/seeker_lock/detonate/miss/kill |
| `PerceptionSnapshot` | `t, detections, tracks` | 写入每帧 |

坐标：内部统一使用 ECEF（米），滤波在**场景级 ENU**（`scene.enu_origin` 固定）；输出经纬度保留 6 位以上小数。

### 5.3 数据集 schema

- **模型关闭**：不写 `schema_version`、`models`、`perception`、`events` 等任何新字段，保持现有 v1 输出逐字节一致。
- **模型开启**：写 `schema_version: 2` 与 `models`，`frames[].perception` 为可选块。

```json
{
  "schema_version": 2,
  "models": {"perception": true, "engagement": true, "deterministic": true, "seed": 20261022},
  "frames": [
    {
      "t": 120.0,
      "targets": [
        {
          "id": "R06",
          "perception": {
            "detected_by": ["S_R06_RADAR"],
            "track_id": "T0001",
            "track_quality": 0.82
          }
        }
      ]
    }
  ],
  "events": [
    {"t": 118.4, "seq": 1, "type": "detection_start", "sensor": "S_R06_RADAR", "target": "B03", "snr_db": 14.2},
    {"t": 119.0, "seq": 2, "type": "track_start", "track": "T0001", "target": "B03"},
    {"t": 240.5, "seq": 3, "type": "launch", "shooter": "R12", "weapon": "W_HQ9B", "target": "B03"},
    {"t": 268.1, "seq": 4, "type": "detonate", "track": "T0001", "miss_m": 6.3, "pk": 0.78}
  ]
}
```

事件只记录**状态变化**（detection_start/end、track_start/drop、quality 变化 > 0.1、launch/detonate/kill），不逐扫描记录 detection，避免百万级事件。事件按 `(t, seq)` 稳定排序；seq 全局递增。

### 5.4 向后兼容

- `inference.infer(frames, targets, hotspot_names=())` **签名不变**；每帧有 `perception` 时使用，无则走旧逻辑。
- 旧数据集（无 `perception`）仍可加载；新字段一律可选。
- 前端对未知字段应忽略；Phase 0 用现有页面加载一次新数据集验证不报错。

---

## 6. 算法规格

### 6.1 时钟、调度与角度约定

**帧与子时钟**：现有主循环按帧生成（当前 situation3d 为 `TICK_SECONDS=30 s`；geo-engine 以 Phase 0 实测为准）。模型不能按 30 s 步长跑传感器/弹道，因此在每帧内部引入**子事件调度器**：

- 帧边界 `t0, t1 = t0+30`；
- 传感器按 `scan_period_s`（如 2 s）与 `scan_phase_s` 在 `[t0,t1)` 内产生扫描时刻；
- 交战微步 `dt_micro = 0.25 s`，接近目标 5 km 内自适应降到 0.05 s（或解析 CPA）；
- 所有子事件按 `(t, event_type, id)` 排序处理；帧结束时只输出该帧的快照与状态变化事件。

**角度约定**（必须写死并在配置校验/单测覆盖）：

- 方位角 `az`：从真北顺时针，`[0,360)`；
- 俯仰角 `el`：相对当地水平面，向上为正；
- 平台姿态旋转顺序：Z-Y-X（yaw-pitch-roll），与现有运动学一致；
- 传感器 `mount` 与 `fov` 均在该平台局部坐标系下定义；
- 导引头视线角 = 弹体速度方向与 LOS 的夹角（0–180°），与 `fov_deg/2` 比较。

时间用仿真秒 `float`；事件时间保留 3 位小数；经纬度输出保留 ≥ 6 位小数。

### 6.2 雷达方程与检测概率

线性形式：

```
λ = c / f
Pr = Pt · Gt · Gr · λ² · σ / ((4π)³ · R⁴ · L)
SNR = Pr / (k · T0 · B · F)
```

dB 形式（实现用）：

```
Pr_dbm = Pt_dbm + Gt_db + Gr_db + 20log10(λ) + 10log10(σ)
         - 30log10(4π) - 40log10(R) - L_db
N_dbm  = -174 + 10log10(B_hz) + NF_db
SNR_db = Pr_dbm - N_dbm
```

检测概率（Swerling I，单脉冲起伏）：

```
Pd = Pfa^(1 / (1 + SNR_linear))
```

- `Pfa` 默认 `1e-6`；`Pd` 截断到 `[0,1]`。
- 本期不做 Swerling 0；后续需要时用级数/North 近似并给出误差界。
- 距离门限：`R < min_range` 或 `R > max_range` 直接不检测。
- 每次扫描对每个目标只采样一次 `detected`；随机数按稳定哈希派生（见 §8.2）。
- 天线增益：首期常数 `gain_db`；预留 `gain_by_off_boresight_deg` 表。

### 6.3 视距与地形遮蔽

- 有效地球半径：`R_eff = 4/3 · 6371008.8 m`。
- 几何视距：`d_h = sqrt(2·R_eff·h1) + sqrt(2·R_eff·h2)`；超过则不检测。
- 高度基准：`h` 为平台/地形的同一基准（椭球高或海拔，Phase 0 确认）；DEM 与平台高度基准不一致时，统一换算并记录。
- 地形采样：沿视线按距离自适应：
  - `R ≤ 50 km`：步长 250 m，至少 16 点；
  - `50 km < R ≤ 200 km`：步长 1 km；
  - `R > 200 km`：步长 2 km，并在日志记录 `los_resolution_m`。
  - 上限 400 点；超过时按上限等距采样，接受远距离漏检小地形的已知限制。
- 每点：`h_line(s) = (1-s)·h1 + s·h2 - s(1-s)·R²/(2·R_eff)`；若 `h_line(s) < DEM(lon,lat) + margin`（margin 默认 5 m，可配），判定遮蔽。
- DEM 采样点经纬度按大圆插值；使用现有 DEM 接口，缓存按瓦片键 LRU（上限 256 瓦片）。
- 无 DEM 时退化为几何视距，日志 `los_source=geometric`，并在报告中注明验收降级。
- 输出 `los_clear` 与 `blocked_at_m`（首个遮挡点距离）。

### 6.4 扫描调度

- 每个传感器维护 `next_scan_t`，初值 `scan_phase_s`；帧内 `while t < t1: if t >= next_scan_t: scan(); next_scan_t += scan_period_s`。
- 单次扫描覆盖整个 FOV（不做波束级扫描）；后续需要时按方位分扇区、每扇区一个 dwell。
- 目标在 FOV 内、LOS 通、距离门限内，才计算 SNR/Pd。
- 扫描间隙内不更新检测，航迹进入 coasting。
- 传感器挂载在平台局部坐标，朝向随平台姿态旋转；平台姿态缺失时按 yaw=航向、pitch=0、roll=0 处理并记录。

### 6.5 航迹管理（卡尔曼 + 生命周期）

**坐标系**：场景级 ENU，原点 `scene.enu_origin` 固定；状态与协方差都在该 ENU 下，输出时转 ECEF。

状态：

```
x = [e, n, u, ve, vn, vu]ᵀ
F = [[I, dt·I], [0, I]]
Q = q · [[dt⁴/4·I, dt³/2·I], [dt³/2·I, dt²·I]]   # q 为加速度谱密度，按目标类型配置
```

- 测量：`range, az, el` 转 ENU 直角坐标，并用一阶雅可比得 `R`（3×3）；`R` 由 `sigma_range_m / sigma_az_deg / sigma_el_deg` 构造，可按 SNR 缩放（`R *= max(1, (SNR_threshold/SNR)^2)`，可选）。
- **时间对齐**：每条检测带时间戳；更新前先把航迹状态传播到该时刻（`F(Δt)`）。检测按时间排序；乱序检测若早于 `last_update_s` 超过 0.5 s 则丢弃并计数，否则回插处理。
- 关联：马氏距离门限 `χ²(3, 0.99) ≈ 11.34`；最近邻；并列按 `(distance, track_id)` 排序；多传感器顺序更新同一航迹。
- 生命周期（显式表）：

| 当前状态 | 事件 | 新状态 | 条件 |
|---|---|---|---|
| tentative | 命中 | tentative/confirmed | 累计 2 次命中 → confirmed；否则保持 |
| confirmed | 命中 | confirmed | 更新 |
| confirmed | 丢失 | coasting | 超过 1 个“全局航迹更新周期”无更新 |
| coasting | 命中 | confirmed | 重新更新 |
| coasting | 丢失 | dropped | 连续 5 个周期无更新 |
| any | 任务锁定 | 状态不变 | `lock_count > 0` 时禁止 dropped |

- `lock_count` 在 `TrackState` 中；交战任务激活时 +1，结束 -1。
- “全局航迹更新周期”定义为创建该航迹的传感器的 `scan_period_s`（缺省 2 s）；多传感器时取最近一次成功更新的传感器的周期。
- 质量：`quality = clamp(1 / (1 + trace(P_pos)/σ_ref²), 0, 1)`，`σ_ref` 按传感器配置（默认 1000 m）。
- 航迹 ID：`T%04d`，按创建顺序稳定分配。
- `TrackState` 存 `cov_pos_3x3`（位置协方差）与内部完整 6×6；跨航迹关联使用共享场景 ENU，禁止每条航迹各自原点。

### 6.6 3DOF 弹道

状态 `[p, v, m]`；受力：

```
T(t) = thrust_n if t < burn_s else 0
D = 0.5 · ρ(h) · |v|² · Cd · A
a = (T·u + D_opposite + m·g) / m
dm/dt = -fuel_kg / burn_s   (燃烧期间)
```

- 积分：RK4，基准 `dt_micro = 0.25 s`；当弹目距离 < 5 km 时降到 `0.05 s`，或改用解析 CPA 判定。
- 大气密度：首期指数模型 `ρ = 1.225·exp(-h/8500)`；后续可接环境包。
- 初始速度：发射平台速度 + 发射方向；发射方位/俯仰由发射计算机给定。
- 终止：命中（引信）、燃料耗尽后滑翔、超出 `t_max`、落地。

### 6.7 制导、导引头、引信与 Pk

- 制导：比例导引
  `a_cmd = N · Vc · ω`，`Vc = -(r·v_rel)/|r|`，`ω = |r × v_rel| / |r|²`；
  横向加速度限幅 `|a| ≤ max_g·9.80665`；`N` 默认 4。
- 中制导：按航迹外推的预测拦截点飞行；导引头截获后转 PN。
- 导引头：`|r| ≤ seeker.range_m` 且视线与弹轴夹角 `≤ fov_deg/2` → `seeker_lock`；否则继续中制导。
- **命中判定（连续 CPA）**：每个 RK4 子步计算弹目线段-线段最近距离（两者都在动）；若 CPA ≤ `fuze_radius_m`，在该子步内插值出引爆时刻并 `detonate`。`miss_m` 定义为整个交战的最小 CPA，不是某一步的瞬时距离。
- Pk：按 `miss_m` 查表（线性插值，表只覆盖 `[0, fuze_radius_m]`，超出为 0）。
  - `pk_mode=expected`：记录 `pk` 作为期望毁伤比例，**不**发布尔 `kill`；
  - `pk_mode=stochastic`：按稳定种子伯努利采样，采样为真时发 `kill`。
  - 事件字段区分 `pk`（期望）与 `kill`（布尔）。
- 事件：`detection_start/end`、`track_start/drop`、`launch`、`seeker_lock`、`detonate`、`miss`、`kill`；`kill` 只发事件，不直接改平台状态，由现有规则消费（若平台无健康字段，则仅记录，见 §14）。

### 6.8 发射包线（LAR）

- 预计算，不在运行时做蒙特卡洛。
- 网格（每条武器 × 目标类别）：
  - 目标高度 `0–15 km`（1 km 步长）；
  - 目标进入角 `0–360°`（30° 步长）；
  - 发射方高度 `0–15 km`（1 km 步长）；
  - 目标速度与发射方速度：按配置固定（如 250 m/s / 250 m/s），在产物中记录；不做速度轴，避免网格爆炸。
- 每个格点计算：`max_range_m`、`min_range_m`、`nez`（不可逃逸区，定义为目标以最大过载逃逸仍被命中的最远距离）、发射方位/俯仰装订 `launch_az_deg/launch_el_deg`。
- 输出：`data/lar/<weapon_id>_<target_class>.json`，含生成参数与版本号。
- 运行时按最近格点查询；查不到时回退武器标称射程，并记录 `lar_fallback=true`。
- `tools/build_lar.py` 可离线重跑；产物入库（JSON 体积可控）。

### 6.9 交战管理

- **触发源**：
  1. 案例脚本 `data/engagements/<case>.json`：定时 `LaunchOrder`（shooter、weapon、target_id、最早/最晚发射时间）；target_id 在发射时解析为当前**质量最高**的匹配航迹，无航迹则订单失效并记录。
  2. 自动规则（最小版）：敌我识别 → 目标按威胁/距离排序 → 选择在 LAR 内且质量 ≥ 0.6 的航迹 → 每对 `(shooter, target)` 只发射一次。
- 每个 `(shooter, target_track)` 同时最多 1 个活跃交战；全局活跃弹上限可配（默认 20）。
- 状态机：`ordered → launched → midcourse → seeker_lock → terminal → detonate → assessed`。
- 目标航迹丢失：中制导按外推继续；超过 `t_max` 或外推误差过大则 `miss`。
- 发射方被击毁：仅当平台有健康字段时取消未发射订单；否则记录订单并在 `t_max` 后判 `miss`。

---

## 7. 与现有系统的集成点

### 7.1 主循环

在现有帧生成中插入**帧内子调度器**（不改 30 s 帧边界）：

1. 平台运动学按帧更新（现有）；
2. 子调度器在 `[t0,t1)` 内按时间处理：传感器扫描 → 检测 → 航迹更新；
3. 交战微步：发射/中制导/末制导/引信；
4. 帧末：`build_perception_snapshot(t1)`；
5. L1 推理（读取帧内 `perception`）；
6. 帧/事件写出。

不得改变现有帧生成顺序；模型关闭时走原路径，不实例化任何对象。**失败策略**：默认失败即报错（保证确定性）；只有显式 `--degrade` 才允许关闭模型继续，并在数据集记录 `degraded=true` 与原因。

### 7.2 L1 推理

- `infer` 签名不变；每帧读取 `frame.get("perception")`。
- 有感知：
  - “被探测数量”改为 `len(detected_by)` 或 `track_quality ≥ 0.5` 的传感器数；
  - 最近对方目标改为“最近**已跟踪**目标”；
  - 威胁度覆盖度项用 `track_quality` 加权：`Σ 3·(1−d/R)·quality`；
  - 接近度项暂用武器标称射程（LAR 在 Phase 7 接入，接入前保持回退）。
- 所有新增口径写入 `inference_model` 段，前端不复制。

### 7.3 数据集与前端

- 只加字段，不改现有字段；模型关闭保持 v1 字节一致。
- 前端默认忽略未知字段；Phase 0 用现有页面加载一次新数据集验证不报错。
- **不改 UI**；如需 `?models=1` 调试开关或感知图层，先向用户申请。

### 7.4 特性开关

- 默认全关：`models.perception.enabled=false`、`models.engagement.enabled=false`。
- CLI/环境变量：`--models perception,engagement`；数据集记录实际开关。
- 关闭时不得实例化传感器/航迹对象，避免性能与哈希变化。

---

## 8. 性能、确定性与可观测性

### 8.1 性能预算（按 30 s 基准帧；若 Phase 0 帧长不同则按帧内计算量等比调整并记录）

| 项 | 预算（80 目标 / 20 传感器 / 20 活跃弹压力场景） |
|---|---|
| 传感器+航迹（每帧汇总） | p95 ≤ 5 ms |
| 交战（每帧汇总） | p95 ≤ 3 ms |
| 合计 | p95 ≤ 8 ms/帧 |
| 模型关闭 | 与基线完全一致（0 开销） |
| 数据集体积 | 相比基线增加 ≤ 20% |
| 事件数 | ≤ 50k/案例 |

- 传感器-目标候选对用空间网格（cell ≈ 50 km）剪枝，避免 O(N×M) 全遍历。
- 扫描在帧内分批处理，单次扫描最多 200 个检测，超出按 SNR 排序截断。
- DEM 采样按瓦片缓存；LAR 只查表。
- 压力场景单独提供 `tools/stress_scene.json`（80/20/20），不与现有三案例混淆。

### 8.2 确定性

- 随机数：`seed_i = int.from_bytes(sha256(f"{seed}|{domain}|{id1}|{id2}|{index}").digest()[:8], "big")`；用 `random.Random(seed_i)`。
- 并列 tie-break：`(distance, track_id)`；检测按 `(sensor_id, target_id)` 排序；事件按 `(t, seq)` 排序。
- 传感器扫描相位在配置中显式给出（`scan_phase_s`），不依赖对象创建顺序。
- 浮点输出统一格式化（位置 6 位小数、时间 3 位小数）。
- 同一配置连续两次生成，数据集 SHA256 必须一致。

### 8.3 可观测性

- `--diag` 输出：传感器扫描次数、检测数、航迹数、活跃交战数、每帧耗时。
- `--dump-perception <path>` 输出每帧检测/航迹 JSON，便于复盘。
- 事件流可直接用现有 event feed 展示；不改 UI 的情况下先落盘。
- 关键日志：LOS 遮蔽原因与分辨率、Pd 采样值、航迹创建/删除、发射/截获/引爆。

---

## 9. 测试与验收

### 9.1 单元测试（必须）

| 模块 | 用例 | 判据 |
|---|---|---|
| 雷达方程 | 固定 Pt/G/λ/σ/R，手算 SNR | 误差 < 0.01 dB |
| Pd | SNR -10→30 dB | 单调递增，∈[0,1] |
| 视距 | h1=h2=10 m | 与解析值误差 < 1% |
| 地形遮蔽 | 构造 30 km 山脊 | `los_clear=false`，遮挡点合理 |
| 角度约定 | 正北/正东/平台旋转 | az/el 符号与旋转正确 |
| 卡尔曼 | 匀速直线、无噪声 | 速度收敛，位置误差趋 0 |
| 量测对齐 | 2 s 扫描、900 m/s 目标 | 传播后误差 < 50 m |
| 生命周期 | 命中/丢失序列 | 状态表逐行通过 |
| PN | 非机动目标 | 脱靶量 < 5 m |
| CPA/引信 | 1500 m/s 闭合 | 不跨过引信半径；`miss_m` 连续 |
| Pk | 脱靶量 0→15 m | 单调递减，范围外为 0 |
| LAR | 高度/进入角变化 | 最大射程单调合理 |

### 9.2 集成/黄金测试

1. **关闭模型**：生成 ZY1/TW1/TW2，数据集 SHA256 与 Phase 0 基线一致。
2. **开启感知**：同一 seed 连续两次生成，SHA256 一致；每个目标至少被探测到一次；检测状态在案例内至少变化一次，或记录为持续覆盖并解释；全 0 判失败。
3. **开启交战**：案例脚本至少产生 launch/detonate/kill 链；事件时间单调；无孤立事件。
4. **回退**：删除 `perception` 段，旧数据集仍可加载，L1 输出不变。

### 9.3 性能测试

- `tools/bench_models.py`：加载 80/20/20 压力场景，跑 200 帧，输出 p50/p95。
- 对比基线（模型关闭）；超过 §8.1 预算则优化或降级（降低扫描频率/限制活跃弹/关闭地形）。

### 9.4 参数校验

- 探测距离判定条件固定为：`Pd=0.5`、`Pfa=1e-6`、目标 RCS 按配置姿态、目标高度按案例、传感器高度按平台。
- 用公开型号资料核对探测距离/射程，记录偏差；偏差 > ±20% 必须在 `PROVENANCE.md` 说明，禁止静默调 RCS 反凑。
- 对关键参数做敏感性扫描（RCS ×0.5/×2、功率 ×0.5/×2），确认结论不翻转。

### 9.5 最终复核

- 全部 P0 完成后，找 2 名 fresh-context 复核者（只给方案、代码、测试结果，不给推理过程）：
  - 复核 1：模型正确性/数值稳定性/确定性；
  - 复核 2：集成/性能/兼容性/合规。
- 修复确认的问题后重新跑全部测试；不嵌套评审。

---

## 10. 分支、提交与发布

### 10.1 分支

```bash
# 在远端服务器执行；GEO_BRANCH 在 Phase 0 确认
git fetch origin
git worktree add ../situation3d-afsim-models -b feat/afsim-model-integration origin/GEO_BRANCH
cd ../situation3d-afsim-models
```

- 新特性必须独立 worktree/分支；不改主干。
- 不在下载/临时目录建仓库；不提交密钥、构建产物、数据集大文件（按现有 .gitignore）。

### 10.2 提交

- 每个 Phase 至少一个有意义提交，消息示例：
  - `feat(perception): add radar detection model with LOS masking`
  - `feat(perception): add track manager with 6-state Kalman filter`
  - `feat(engagement): add 3DOF flyout and PN guidance`
  - `test(perception): cover detection lifecycle and determinism`
- 不整文件覆盖合并；冲突逐块审查，保留双方修改。
- 失败时记录原因，只回退自己的改动，禁止 `reset --hard`。

### 10.3 合并与发布

- 全部测试 + 独立复核通过后，合并回 geo-engine 分支；推送 `zililong1993/shenji_agent`。
- **演示环境 8848 不自动更新**；按里程碑经用户同意后走 `demo_update.sh`，保留回退版本。
- 合并后在服务器跑一次全量回归（191 项 + 3 案例 + 性能）。

---

## 11. 任务拆解（新会话按序执行）

> 工作量是参考值（1 人、含测试），不是承诺；Phase 0 后按实际情况调整。

### Phase 0：仓库勘察（0.5–1 人日）

- 产出：`docs/afsim_integration/00_baseline.md`；把本方案与调研报告复制到 `docs/afsim_integration/`。
- 验收：§3.1 十二问全部有答案；基线测试/哈希/耗时记录在案；数据集生成入口唯一确定。
- 依赖：无。

### Phase 1：脚手架、配置与事件定义（1–1.5 人日）

- 产出：`perception/`、`engagement/` 包骨架；`config.py`（含平台绑定校验）；`events.py` 事件定义；特性开关；空实现。
- 测试：配置校验（平台/signature/loadout 解析）、开关关闭时零行为。
- 验收：模型关闭时三案例哈希与基线一致；现有测试全绿。
- 依赖：Phase 0。

### Phase 2：数学、地理与时钟（2 人日）

- 产出：`mathutil.py`（dB、插值、稳定哈希、角度约定）、`los.py`、`clock.py`（帧内子调度器）。
- 测试：视距/遮蔽/角度/调度顺序。
- 验收：解析值误差 < 1%；30 s 帧内扫描时刻正确。
- 依赖：Phase 1。

### Phase 2.5：型号参数包与 PROVENANCE（1–2 人日）

- 产出：`data/models/*.json`（传感器/RCS/武器/平台绑定）、`PROVENANCE.md` 初版、参数校验脚本。
- 验收：三案例所有平台都能解析到传感器/武器/signature；探测距离判定条件写死；示例值明确标注“示意”。
- 依赖：Phase 1。

### Phase 3：雷达传感器（3 人日）

- 产出：`sensors/radar.py`、扫描、RCS 表、Pd 采样、检测事件。
- 测试：雷达方程、Pd 单调、扫描周期、确定性。
- 验收：探测距离按 §9.4 口径偏差 ≤ ±20%；同 seed 两次扫描一致。
- 依赖：Phase 2、2.5。

### Phase 4：航迹管理（3 人日）

- 产出：`track.py`（KF6、关联、时间对齐、生命周期、质量）、`TrackManager`。
- 测试：匀速收敛、生命周期、量测对齐、门限、确定性。
- 验收：直线航段航迹连续率 ≥ 90%；误关联率 < 2%（按真值关联标签统计）。
- 依赖：Phase 3。

### Phase 5：感知接入推理与数据集（2 人日）

- 产出：`inference` 读取每帧 `perception`；数据集 v2；`PerceptionSnapshot` 写出。
- 测试：关闭模型哈希不变；开启后 L1 使用航迹；旧数据兼容。
- 验收：DoD 第 1、3 条；L1 输出 schema 不变；每案例至少一个目标的告警等级因感知变化且可解释。
- 依赖：Phase 4。

### Phase 6：武器弹道与制导（4–5 人日）

- 产出：`flyout.py`、`guidance.py`、`seeker.py`、`fuze.py`；3DOF + PN + 截获 + CPA + Pk。
- 测试：PN 脱靶、导引头门限、CPA 1500 m/s、Pk 单调、事件顺序。
- 验收：非机动目标脱靶 < 5 m；事件链完整。
- 依赖：Phase 4。

### Phase 7：LAR、交战脚本与管理（3–4 人日）

- 产出：`launch_computer.py`、`tools/build_lar.py`、`data/lar/*.json`、`data/engagements/<case>.json`、`manager.py`；L1 接近度接入 LAR。
- 测试：LAR 单调、查询回退、交战状态机、案例事件链。
- 验收：LAR 查询 < 0.1 ms；三案例至少各 1 条 launch→detonate 链；无孤立事件。
- 依赖：Phase 6。

### Phase 8：事件序列化、诊断与文档（2 人日）

- 产出：事件流落盘/校验、`--dump-perception`、`--diag`、`README`、`CLEANROOM.md`。
- 测试：事件 schema 校验、序列化稳定、事件体积 ≤ 50k/案例。
- 验收：事件可按 `(t, seq)` 重放；文档含参数来源与 clean-room 记录。
- 依赖：Phase 5/7。

### Phase 9：性能、回归与复核（2–3 人日）

- 产出：`tools/bench_models.py`、`tools/stress_scene.json`、性能报告、回归记录、复核意见与修复。
- 测试：全量 191 项 + 三案例 + 压力场景 + 性能预算。
- 验收：DoD 全部满足；独立复核通过。
- 依赖：Phase 8。

---

## 12. 新会话启动提示词（复制即用）

```text
你将在本项目的 geo-engine 分支上执行“AFSIM 模型级集成”任务。

先读以下材料（按顺序）：
1. 本项目 docs/afsim_integration/AFSIM模型级集成实施方案.md（本文件）
2. 本项目 docs/afsim_integration/AFSIM可集成性调研报告.md
3. memory/project_afsim_assessment.md

如果上述文件缺失：停止并报告，不要自行猜测或从 outputs 目录复制旧版本。

硬约束：
- 模型级 clean-room 重实现，禁止引入 AFSIM 源码/二进制/场景/第三方安装包。
- 不改前端 UI；不改主干；新特性走独立 worktree/分支。
- 所有开发/运行/验证在远程服务器完成，本机只做同步。
- 先做 Phase 0 仓库勘察，产出 docs/afsim_integration/00_baseline.md；未完成前不写业务代码。
- 每个 Phase 按方案中的验收标准自测；全部 P0 完成后做一次独立复核。
- 不整文件覆盖合并，不 reset --hard；失败只回退自己的改动。

本次只执行 Phase 0：回答方案 §3.1 的十二个问题，记录基线测试/哈希/耗时，给出模块落点映射。完成后停下，汇报“已核实/与假设不符”的清单，等待确认后再进入 Phase 1。
```

---

## 13. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| 参数来源不可靠（RCS/功率/气动） | 探测距离/包线失真 | `PROVENANCE.md` 逐项记录；敏感性分析；标注工程假设 |
| 30 s 帧内子调度开销大 | 生成变慢 | 事件驱动微步；空间网格；活跃弹上限；性能预算 |
| 传感器-目标全遍历 | 性能崩塌 | 空间网格剪枝；扫描限流；p95 预算 |
| 与现有运动学重复/冲突 | 状态不一致 | 只读平台状态；坐标转换走现有权威；不做第二套运动学 |
| 数据集 schema 破坏前端 | 页面报错 | 只加可选字段；Phase 0 验证；关闭模型哈希不变 |
| 随机性导致不可复现 | 无法回归 | 稳定哈希种子；实体排序；确定性模式默认开 |
| 命中判定跨过引信半径 | 永不引爆/假命中 | 连续 CPA + 自适应微步 + 1500 m/s 回归用例 |
| 许可/合规污染 | 无法交付 | clean-room；禁止 AFSIM 代码/文件；复核清单 |
| 过早改 UI 引发返工 | 违反用户约束 | 本期只落数据/事件；UI 另行申请 |
| 任务范围膨胀到 C2/通信/EW | 周期失控 | 非目标清单；P1/P2 另立方案 |

---

## 14. 待决问题（Phase 0 或开工前确认）

1. **geo-engine 包根与分支名**：`GEO_ENGINE_ROOT`、`GEO_BRANCH` 具体是什么？
2. **数据集生成入口**：ZY1/TW1/TW2 由哪个命令/分支生成？与 geo-engine 是否同一路径？若不同，需 parity 任务。
3. **DEM 接口**：现有地形瓦片如何按经纬度采样？是否支持批量/缓存？若不存在，地形遮蔽设为可选降级还是新增数据任务？
4. **高度基准**：平台高度是椭球高还是海拔？DEM 是否同一基准？
5. **平台姿态**：现有状态是否包含 heading/pitch/roll？传感器挂载需要。
6. **健康/毁伤字段**：平台是否有 health/damage？`kill` 由谁消费？若无，本期只写事件。
7. **型号参数**：三案例现有 `sensor_range_km`/`weapon_range_km` 的来源与可信度？
8. **Pk 模式**：默认 `expected` 是否满足演示/评估口径？还是需要随机 Monte Carlo？
9. **EOIR**：本期只做雷达是否足够？若需要红外，追加 2–3 人日。
10. **调试图层**：是否需要前端显示检测/航迹/包线？需用户明确批准后才做。
11. **LAR 产物**：JSON 是否入库，还是构建时生成？建议入库（体积小、可复现）。
12. **交战脚本**：三案例的发射指令由谁提供（用户/规则）？若无，Phase 7 先实现最小自动规则。

---

## 15. 参考

### 15.1 规格背景（仅规格作者阅读，不作为实现参考）

1. 《AFSIM 可集成性调研报告》。
2. AFSIM 文档中文翻译（任务处理器/航迹/脚本/传感器工具）：<https://gitee.com/Maicss/afsim-doc-zh>
3. WSF Doxygen（平台/传感器/通信/处理器/Observer，含 CUI 声明）：<https://afismapibook.help/html/index.html>

### 15.2 实现参考（公开教材/论文，实现者可用）

1. 雷达方程与检测理论：Skolnik, *Introduction to Radar Systems*；Swerling 起伏模型。
2. 卡尔曼滤波与航迹管理：Blackman & Popoli, *Design and Analysis of Modern Tracking Systems*。
3. 制导与发射包线：Zarchan, *Tactical and Strategic Missile Guidance*。
4. 坐标与地球模型：WGS-84 / NIMA TR 8350.2；4/3 地球视距模型。
5. 实现时在文件头记录具体公式出处；不得复制教材代码。
