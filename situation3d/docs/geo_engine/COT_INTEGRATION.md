# Claude-of-Tanks 算法集成（侦察 / 直瞄 / 战术指挥）

> 来源：`https://github.com/Kevin-Liu-01/Claude-of-Tanks`，固定提交 `9ae86a7e9fc4a03cc30c40bc8ad2ad3103561e86`。
> 只移植 MIT 路径下的算法（`src/sim/spotting.ts`、`src/sim/ballistics.ts`、`src/sim/damage.ts`、`src/game/ai.ts` 的行为框架），未复制 `src/vehicles/**`、`src/world/**` 等 Reserved Content。
> 代码版权：Copyright © 2026 Kevin B. Liu，MIT License。移植文件头已注明来源。

## 1. 开关

`sim.scenario.build_world` 读取想定顶层的 `engine` 块：

```json
{
  "engine": {
    "detection_model": "table | spotting | shadow | physical",
    "direct_fire_model": "crt | physics | shadow",
    "enable_radio_share": false,
    "enable_tactics": false,
    "enable_commander": false,
    "commander_interval_s": 2.0
  }
}
```

默认值与 P1.5 后的语义：`detection_model=table`、`direct_fire_model=physics`、
`indirect_fire_model=physics`、`enable_radio_share=false`、`enable_tactics=false`。
CRT/散布表可通过 `direct_fire_model=crt`、`indirect_fire_model=table` 切回校准/兼容模式。

`detection_model=physical`（P1.4，可选，默认不启用）：3D 地形/建筑/烟幕射线 + 传感器视场/探测概率 + 分方航迹，详见 `P1_4_perception.md`；该模式按 `enable_radio_share` 决定是否共享。

`scripts/run_geo_demo.py` 也提供命令行开关，便于 A/B：

```bash
python3 scripts/run_geo_demo.py --case taiwan_landing --ticks 300 \
  --detection-model shadow --direct-fire-model shadow --tactics --commander \
  --out /tmp/cot_shadow
```

## 2. 侦察/隐蔽（`sim/spotting.py`）

- 观察者视距优先取 `sensors[]` 的 `range_km × quality`，其次按目标类型取想定 `observe_distance`（格 × 200 m），最后回落到公开观察表。`spotting` 模式会同时保留观察表/传感器对海空目标的发现结果，陆地-陆地才用新模型。
- 目标隐蔽由角色卡式 `concealment`（0–0.9）或 `concealed` 布尔 + `size_class` 得到，再叠加地形隐蔽（森林 +0.35、居民地 +0.25、软土 +0.10）。
- 发现距离沿用 COT 公式 `spotRange = viewRange − (viewRange − 50) × camo`，夹在 `[50 m, min(viewRange, 5000 m)]`；50 m 内无条件发现。
- 开火会按 `0.82` 的比例损失自身隐蔽，并按 `τ = 1.7 s` 指数恢复；炮口焰窗口内（bloom ≥ 0.45）开阔地目标会被直接暴露，附近植被的隐蔽加成减半，对应原版的 15 m 规则。
- `detection_model=spotting` 时，陆地观察完全走该模型；`shadow` 时仍用观察表裁决，但每 tick 把两套结果写入 `world.detection_shadow`（`table_pairs`、`spotting_pairs`、`both`、`only_table`、`only_spotting`、`samples`）。
- `enable_radio_share` 或 `spotting` 模式下，观察者会把接触按 `comms_range_m`（默认 600 m，可为 0 表示无电台）一跳共享给同方同域单位，不做链式中继。

角色卡对接字段（想定算子级，均可选）：

```json
{
  "concealment": 0.35,
  "comms_range_m": 600,
  "armor_profile": {"front": 550, "side": 250, "rear": 150, "top": 60},
  "observe_distance": {"infantry": 10, "vehicle": 25, "uav": 1}
}
```

## 3. 直瞄弹道/穿深（`rules/physics_direct_fire.py`）

- `direct_fire_model=physics` 时，`rules/combat.resolve_direct_fire` 先尝试物理裁决；没有弹道数据的武器回落到庙算 CRT。
- 弹道数据在 `rules/data/ballistics.json`：`caliber_mm`、`shell_type`（AP/APCR/APFSDS/HEAT/HE/HESH）、`velocity_mps`、`pen_100m/1000m/2000m_mm`、`blood_damage`、`dispersion_mrad`、`max_range_m`。
- 命中：按 `dispersion_mrad × 距离` 生成高斯瞄准误差，超过目标尺寸半径（small/medium/large = 1.5/3/5 m）判为 miss。
- 入射角：用目标航向与“目标→射手”方位差得到 0°（正面）→180°（尾部），据此选 `armor_profile` 的 front/side/rear，再把该角度换算成所选装甲面的法向角（侧面 `|θ−90°|`、尾部 `|180°−θ|`）后计算厚度与跳弹。
- 穿深：`penAtDistance` 在 100/1000/2000 m 间线性插值，再乘 ±25% 掷骰；有效厚度为 `RHAe / cos(法向修正后角度)^slopeExp`，AP 法向 5°、跳弹 70°、指数 1.4；APFSDS 法向 2°、跳弹 78°、指数 1.0；HEAT 跳弹 85°、指数 1.0；2 倍口径触发法向增强，3 倍口径动能弹不跳弹；APFSDS 的超口径口径按 `0.6 × 炮口口径` 计。
- 命中：用二维高斯生成横向/纵向偏差，按径向误差与目标尺寸半径比较；这比一维高斯更接近原版的散布圆盘。
- 伤害：`blood_damage × min(2, pen/eff)`，取整后扣 `blood`；HE 未击穿时给 30 s 压制。该映射是演示标定，不是 COT 的 HP 体系。
- `direct_fire_model=shadow` 时仍用 CRT 裁决，但在 CRT 扣弹/扣血之前、用 CRT 同款武器和一份独立的 RNG 状态计算 `shadow_physics`（`dry_run=True`），因此既不影响后续随机流，也不会把“已经开火后”的状态当成对比输入；默认 CRT 事件不增加任何字段。

## 4. 战术层与模型指挥（`sim/tactics.py`、`sim/commander.py`）

- `enable_tactics` 启用确定性战术控制器：低血量（≤35%）向己方目标点撤退；已发现敌人且在森林/居民地则原地据守；进入武器射程则停车交战；否则向最近敌方目标点或最近发现敌人机动。只下 `move_to/stop` 命令，不直接开火。
- `sim/commander.py` 提供与 COT `jevCommander` 同形的“模型下判断、规则层执行”接口：
  - `CommanderOrder(uid, posture, target_uid, lon, lat, expires_s)`；
  - `RuleCommander` 作为离线默认/回退；
  - `RemoteCommander(url, timeout_s, token)` 是可选 HTTP 适配器，任何超时/异常/格式错误返回空列表；
  - `CommanderRegistry` 管理订单过期与回退，`World.set_commander(model, interval_s)` 接入，按 `commander_interval_s` 询问一次；`engine.enable_commander=true` 会挂载离线 `RuleCommander`。
  - 指挥官订单带来源与到期时间：到期会清掉该单位由指挥官下发的路线/停止状态；手动订单（`World.order(..., source="manual")`）优先，指挥官不会覆盖，也不会给敌方单位或跨域目标下单；`RemoteCommander` 的超时/异常会记为 `last_error` 并回退到 `RuleCommander`。
- 外部模型没有打包；军用/离线环境应使用本地模型，只复用协议、限界、过期与回退设计。

## 5. 验证

```bash
cd situation3d
python3 -m unittest tests/test_cot_integration.py -v
python3 -m unittest discover -s tests -q
```

新测试覆盖：视距/隐蔽/开火暴露/影子对比/无线电共享/海空回退、穿深衰减/跳弹/超口径/侧后装甲角/物理事件/影子 RNG 隔离、规则指挥姿态/订单过期/回退/远端失败/引擎配置安全、战术机动与 World 接入。集成后全量 130 项测试通过（10 项跳过）。

## 6. 已知限制

1. 没有 COT 的逐棵树冠圆盘，15 m 规则用“炮口焰窗口内掩体加成减半”近似。
2. 装甲只有 `armor_class` 五档或手填 `armor_profile`，不是 COT 的逐板几何；`rules/data/ballistics.json` 是演示估值，需按真实装备标定。
3. COT 的 HP 体系未照搬，伤害映射到本项目的 `blood`，需要按案例校准。
4. 物理模式的射速仍用庙算武器 `cooldown_s`（默认 75 s），没有采用 COT 的 6–10 s 装填节奏。
5. `RemoteCommander` 只定义最小 JSON 协议，未实现鉴权轮换、限流、批量分片和审计日志；接入真实模型前需补齐。
6. 100 个纯陆上实体的本机基准（单线程 Python 3.8）：`table` p50 11.5 ms / p95 37.7 ms，`spotting` p50 48.7 ms / p95 71.0 ms，`shadow` p50 80.0 ms / p95 110.1 ms。`spotting` 默认每 tick 全量重算；大编队上线前应做跨 tick LOS 缓存或按观察者分片。
