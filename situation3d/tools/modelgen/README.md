# 目标 3D 模型资产生成（modelgen）

2026-10-19 建立。路线由用户确定：**优先 CC0 资产，重点型号后续细化，需要基础动画**。

## 资产清单

| 文件 | 形态 | 来源 / 许可 |
| --- | --- | --- |
| `ship-*.glb` / `boat-*.glb` | 驱逐舰/护卫舰/补给舰/导弹艇/拖船 | Kenney Watercraft Kit，CC0 |
| `armor.glb` | 主战坦克（含行进/转向动画） | Quaternius Animated Tanks Pack，CC0 |
| `fighter.glb` | 战斗机（MiG-15） | OpenGameArt "Mig"（Sketlux/yd），CC0 |
| `awacs.glb` | 预警机（雷达罩旋转动画） | 本目录脚本程序化生成，CC0 |
| `helicopter.glb` | 武装直升机（主/尾旋翼动画） | 同上 |
| `apc.glb` | 轮式装甲车 | 同上 |
| `artillery.glb` | 自行火炮 | 同上 |
| `sam_launcher.glb` | 防空导弹发射车 | 同上 |
| `coastal_missile.glb` | 岸舰导弹发射车 | 同上 |

## 重新生成

服务器已装 Blender 4.5.10 LTS（无头，CPU）：

```bash
cd situation3d
/remote-home/lizilong/tools/blender-4.5.10-linux-x64/blender -b \
  --python tools/modelgen/build_models.py -- \
  web/assets/models /remote-home/lizilong/geo_sources/models/Mig-15.blend
```

约定：Blender 内 Z 向上、机头/车头朝 +Y；程序化模型按**米**建模，导出 GLB 时
`export_yup=True`，因此 Cesium 中 +Y 为北、1 单位 = 1 米。旋翼/雷达的旋转用
关键帧 + CYCLES 修改器导出为 glTF 动画，Cesium `Entity.model.runAnimations`
默认为 true，无需额外代码。

`web/src/entityLod.js` 的 `MODEL_MAP` / `MODEL_UNITS` / `REAL_LENGTH_M` 必须与
本目录产出同步；`tools/entity_lod_test.js` 会校验键集合一致且文件真实存在。

## 尺度

`MODEL_UNITS` 是 Blender 导出的世界包围盒**最长轴**（单位），`REAL_LENGTH_M`
是该形态的真实量级（米）。Cesium 按 `REAL_LENGTH_M / MODEL_UNITS` 统一缩放，
所以资产本身不需要预处理成米制。

MiG-15 的 `.blend` 源文件不在仓库内（约 950 KB），保存在服务器
`/remote-home/lizilong/geo_sources/models/Mig-15.blend`；如需重建 fighter.glb
请从 OpenGameArt 重新下载：https://opengameart.org/content/mig
