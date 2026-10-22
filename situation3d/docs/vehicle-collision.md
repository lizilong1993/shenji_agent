# 车辆碰撞体积与车道契约

本页说明 TW2 生成器与前端新增的碰撞检测接口；**变道、避让与导航不在此实现**，
由外部算法消费碰撞结果后决定。

## 数据字段

每个 `targets[]` 条目：

```json
{
  "id": "TBV002",
  "collision_bbox": {
    "length_m": 9.49,
    "width_m": 3.68,
    "height_m": 4.88,
    "source": "glb_bbox",
    "asset": "type_ztd05.glb"
  },
  "lane": {
    "id": 0,
    "width_m": 3.75,
    "center_offset_m": 1.875,
    "side": "right"
  }
}
```

- `collision_bbox` 来自归一化 GLB 的真实包围盒，不是演示用的概略尺寸；
  `height_m` 只记录，当前 2D 检测不使用。
- `lane` 只给地面车辆。`center_offset_m` 是车道中心线相对道路中心线的
  右偏距离，等于 `width_m / 2`；航向反向时由生成器自动换到对向车道。
- `physical_profile` 与 `attributes["车辆尺寸"]` 也同步为 GLB 尺寸。

## 检测器

Python：`situation3d/engine/collision.py`

```python
from situation3d.engine import collision
hits = collision.swept_collisions(frames, {t["id"]: t for t in targets})
print(collision.summarize(hits))
```

- `frame_collisions(frame, targets)`：单帧 2D OBB SAT。
- `swept_collisions(frames, targets, substeps=4)`：在 30 s 帧之间再采样，
  避免两车在帧间交换位置时漏检。
- `overlap(a, b)`：返回最小穿透深度（米），分离时返回 `None`。
- 只检测 `domain="land"` 且 `mode != "embarked"` 的目标；甲板上的载员不与
  载舰自身发生碰撞。

前端：`window.TargetCollision`

```js
TargetCollision.checkAt(viewer.clock.currentTime)  // [{a,b,depth_m}, ...]
TargetCollision.debug()                            // 最近一次 tick 的结果
```

`TargetCollision.tick()` 已接入主循环，每 500 ms 更新一次；位置和航向读取
CZML 插值实体，所以暂停/拖动时间轴时结果一致。它只报告重叠，不移动模型。

## 生成与校验

```bash
# 用 GLB 重新生成 bbox 清单（需要 numpy）
PYTHONPATH=. python3 situation3d/tools/model_bboxes.py situation3d

# 重新生成案例（会写 web/data/situations/taiwan_dense_transfer_v1.json）
PYTHONPATH=situation3d python3 situation3d/scripts/gen_campaign_case.py \
  --out situation3d/web/data/situations

# 检查当前数据的重叠对
PYTHONPATH=. python3 situation3d/tools/check_collisions.py \
  --case taiwan_dense_transfer_v1 [--swept] [--json]

# 契约检查（同时验证 60 辆地面车都有 bbox 和 lane）
PYTHONPATH=. python3 situation3d/tools/validate_case.py \
  --case taiwan_dense_transfer_v1
```

当前数据刻意保留原始道路运动，因此检测器会报告路口/跟车重叠；在导航算法
接入前这是预期结果，不是数据损坏。
