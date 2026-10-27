# 目标模型 v3：052D 自建近似

2026-10-27 · 分支 `feat/target-model-052d` · worktree `/remote-home/lizilong/shenji_agent_052d`

P1 遗留的最后一处型号错配：052D 一直用 052C 资产顶替。052D 无可下载的精确 CC 资产，本次把 `build_custom.py` 里的 `build_052d()` 初版重做并正式接入。

## 1. 背景

- 052C 是弧形雷达罩，052D 是四块平板 346A 相控阵，舰桥/VLS 布置也不同，远看即可分辨。
- `build_052d()` 初版未接入资产库，且存在问题：前桅/烟囱悬空、64 单元 VLS 全堆在艏部（实舰为前 32 + 后 32）、HQ-10 用密集阵外形、舰桥窗带嵌在体块内。
- 本机工具链在服务器恢复后重建：Blender 4.5.10 / Node 22.14.0 / @gltf-transform/cli 4.5.1，位于 `/remote-home/lizilong/tools/`。

## 2. 模型改动

尺寸 157.0 × 17.2 m（水线原点，+Y 艏，1 单位 = 1 m）。

- 武备：130 mm 主炮；前 32（4 列 × 8 行）+ 后 32（8 列 × 4 行）单元 VLS；2 × 4 反舰弹发射筒；1130 近防炮（6 管简化 + 雷达罩）；24 单元 HQ-10；两舷各 3 管鱼雷；干扰弹发射器 × 4。
- 上层建筑：三层阶梯舰桥 + 前窗/侧窗；四块 346A 平板（舰桥两舷、机库两舷，带背板）；前桅 + 雷达罩 + 2 具卫通 + 导航雷达；后桅 + 导航雷达；双烟囱落在中部甲板室；机库 + 机库门 + 侧窗；飞行甲板 + 直升机圈。
- 细节：舷墙（沿船体宽度曲线放样，双面）、锚机/锚孔/防浪板、舷门 × 4、通风口、RHIB × 2、救生筏 × 8。
- 修正：前后桅、双烟囱均有基座/甲板室，不再悬空；窗带跨壁厚 0.5–0.6 m，避免离壁漂浮；AESA 面板高度限制在舰桥层高内。

## 3. 资产与集成

- 新资产 `type_052d.glb`：**11,552 面 / 3 prims / 0.35 MB**（1024 WebP + Meshopt），bbox 157.0 × 17.19 × 33.19 m。
- `entityLod.js`：`052D|Luyang III` → `type_052d`；新增 `052C` → `type_052c`；`destroyer` 大类回退仍是 `type_052c`。
- `targetCombatFx.js` 新增 `ship052d` 代理炮塔（复用 052C 100 mm 炮塔外观，offset `[-61, 0, 6.05]`、muzzle `[-66.15, 0, 7.9]`），避免原 052C 代理叠在艏部 VLS 上。
- 052D 暂未配置 LOD1（默认案例只有 2 艘 052D，密集案例不含 052D）。
- `specs.json`、`CREDITS.md`、`licenses.json`、`manifest.json`、两份 `bboxes.json` 均更新；主模型 39 → **40**。

## 4. 验证

- `model_budget_test.py --write-manifest`：**ALL PASS (40 assets)**；合计 **22.11 MB / 597,102 面 / 263 prims**。
- `model_license_test.py`：40 条许可，7 条风险（未新增）。
- `model_bboxes_gltf.py`：40 型，两份 bboxes 同步。
- `entity_lod_test.js`：**221/221**（新增 052D/052C 映射与 052D 无 LOD1 断言）。
- Cesium 1.145 + headless Chromium：052D、052C 同屏加载均 HTTP 200，无 pageerror。
- 全量 `pytest tests -q`：**563 passed / 14 skipped**（无回归）。

## 5. 已知限制

- 程序化近似：贴图密度与细节低于外部 052C 资产，AESA 未做倾角，舰体线型简化。
- 无 LOD1/实例化；若 052D 进入密集案例需补 LOD1。
- 1130 为 6 管简化 + 雷达罩，不是 11 管精确外形；HQ-10 为 24 格平板发射箱。
- 052C 资产（DCS 提取物）仍在库中，仅供 052C 与驱逐舰大类回退使用。
