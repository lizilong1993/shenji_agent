# 目标 3D 模型资产管线（modelgen）

2026-10-20 重构。目标：让每个目标用**具体型号**的高质量模型；无可下载资产的型号
回退到**大类模型**。资产来源与许可见 `web/assets/models/CREDITS.md`。

## 约定

- 1 单位 = 1 米，艏/机头/车头朝 Blender +Y，Z 向上。
- 原点：舰船在水线（z=0），地面车辆在最低点，飞机在包围盒中心。
- 导出 GLB 时 `export_yup=True`；Cesium 1.145 实测模型局部前向为 **-X**
  （`EntityLod.forwardAxis()` 返回 `'-X'`），不是旧版注释里的 +Y。
- 运行时 `EntityLod.scaleFor()` 恒为 1，不再做尺度补偿。

## 现有资产

| 文件 | 型号 | 来源 |
| --- | --- | --- |
| `type_052c.glb` | 052C 驱逐舰（052D / 驱逐舰大类近似） | Sketchfab |
| `type_054a.glb` | 054A 护卫舰 | Sketchfab |
| `type_071.glb` | 071 综合登陆舰 | Sketchfab |
| `type_perry.glb` | 成功级（Perry） | Sketchfab |
| `type_056a.glb` | 056A 轻护 | 自建 |
| `type_903.glb` | 903 补给舰 | 自建 |
| `type_panshi.glb` | 磐石级补给舰 | 自建 |
| `type_tuochiang.glb` | 沱江级巡逻舰 | 自建 |
| `type_kidd.glb` | 基隆级（Kidd） | 自建 |
| `type_ztz99a.glb` | 99A | Sketchfab |
| `type_m1a2.glb` | M1A2 | Sketchfab |
| `type_m48a5.glb` | M48A5（CM-11 近似） | Sketchfab |
| `type_ztd05.glb` | 05 式两栖突击车 | Sketchfab |
| `type_plz05.glb` | PLZ-05 | Sketchfab |
| `type_m109a6.glb` | M109A6 | Sketchfab |
| `type_m777.glb` | M777 超轻型榴弹炮 | 自建 |
| `type_phl03.glb` | PHL-03 | Sketchfab |
| `type_hq9.glb` | 红旗-9 | Sketchfab |
| `type_hq16.glb` | 红旗-16 发射车 | 自建 |
| `type_pac3.glb` | 爱国者 PAC-3 | Sketchfab |
| `type_yj62.glb` | 鹰击-62 发射车 | 自建 |
| `type_j16.glb` | 歼-16（Su-30MKK 近似） | Sketchfab |
| `type_f16c.glb` | F-16C（F-16V 近似） | Sketchfab |
| `type_idf.glb` | IDF 经国号 | Sketchfab |
| `type_rafale.glb` | 阵风 | Sketchfab |
| `type_kj2000.glb` | KJ-2000（空警-500 / 预警机大类） | Sketchfab |
| `type_e2d.glb` | E-2D（E-2K 近似） | Sketchfab |
| `type_z10.glb` | 直-10 | Sketchfab |
| `type_mq9.glb` | MQ-9（无人机） | Sketchfab |
| `type_eq240.glb` | 东风 EQ240（运输车队） | Sketchfab |

## 管线

### 1. 下载（Sketchfab）

下载需要账号 API token（只在服务器 /tmp 使用，不写入仓库）：

```bash
python3 /tmp/sfdl2.py <model_uid> <slug>
```

`glb` 直链由 `GET /v3/models/{uid}/download` 获取。

### 2. 归一化（减面 / 贴图 / 朝向 / 米制）

```bash
blender -b --python tools/modelgen/normalize_glb.py -- IN.glb OUT.glb \
  --len 155 --axis -Y --tris 50000 --anchor water --maxtex 2048
```

- `--len` 公开全长（米）；`--axis` 导入时船艏/机头所在轴（X/Y/-X/-Y）
- `--fit yaw` 按主轴自动纠偏（车辆/舰船），`--autoflip` 自动翻头
- `--cutz 0.06` 裁掉扫描件底座（15 式扫描件仍不可用，已回退到 99A）

### 3. 贴图压缩

```bash
blender -b --python tools/modelgen/shrink_textures.py -- IN.glb OUT.glb 1024
```

地面车辆用 512，舰船/飞机 1024；统一转 JPEG。

### 4. 自建型号

```bash
blender -b --python tools/modelgen/build_custom.py -- <输出目录> [052d 056a ...]
```

自建型号：056A、903、磐石、沱江、基隆、红旗-16、鹰击-62（052D / KJ-500
代码保留但未使用，改用 052C / KJ-2000 近似）。自建船体带板缝/锈迹贴图，
车辆带迷彩贴图。

## 校验

```bash
node tools/entity_lod_test.js
```

用例覆盖：阈值/滞回、精确型号优先、大类回退、文件存在、scale=1、forwardAxis。

## 已知近似

- 052D → 052C；空警-500 → KJ-2000；歼-16 → Su-30MKK；F-16V → F-16C；
  CM-11 → M48A5；15 式 → 99A 大类。
- 15 式 Sketchfab 资产是带木质展台的三维扫描件，裁切后仍无法分离，弃用。

## 旧脚本

`build_models.py` 是 2026-10-19 的 Kenney/Quaternius/程序化低模管线，已被
本管线取代；保留仅供追溯，不再运行。
