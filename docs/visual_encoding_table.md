# 视觉编码对照表（I09 步骤 6）

> **本文件由 `scripts/gen_visual_encoding_table.py` 从源码自动生成**，
> 故表里的**文件与行号必然与当前代码一致**。手写这份表会在下次改代码后立刻腐烂，
> 而**错的引用比没有引用更糟**（会让人按错行号去找）。

生成时间戳由调用方决定；本表只描述**结构**，不含运行输出。

## 1. 五个编码维度与**唯一配置来源**

| 维度 | 唯一配置来源 | 说明 |
| --- | --- | --- |
| 阵营 | `web/src/theme.css` 的 `--faction-*` | 阵营主色（填充） |
| 类型 | `web/src/symbology.js` 的形状函数 | 按 GJB 6277 方向规则 |
| 状态 | `--status-border-*` | 状态边框，与选中态区分 |
| 强度 | `--ring-*` / `--strength-*` + `theme.js` 半径规则 | 强度内环 |
| 时效 | `--pulse-*` + `theme.js` 周期规则 | 时效外圈脉冲 |

## 2. 逐项对照（含**真实行号**）

| 维度 | 变量 | 取值 | 文件 | 行号 |
| --- | --- | --- | --- | --- |
| 阵营 | `--faction-red` | `#d0342c` | `situation3d/web/src/theme.css` | 28 |
| 阵营 | `--faction-blue` | `#1f5fbf` | `situation3d/web/src/theme.css` | 29 |
| 阵营 | `--faction-green` | `#2f7d4f` | `situation3d/web/src/theme.css` | 30 |
| 阵营 | `--faction-neutral` | `#7d8a99` | `situation3d/web/src/theme.css` | 31 |
| 阵营 | `--faction-contested` | `#d8a12a` | `situation3d/web/src/theme.css` | 32 |
| 状态 | `--status-border-normal` | `#6b7688` | `situation3d/web/src/theme.css` | 44 |
| 状态 | `--status-border-alert` | `#c8352c` | `situation3d/web/src/theme.css` | 45 |
| 状态 | `--status-border-selected` | `#1668dc` | `situation3d/web/src/theme.css` | 46 |
| 状态 | `--status-border-width` | `2px` | `situation3d/web/src/theme.css` | 43 |
| 强度 | `--ring-strength-base-radius` | `6px` | `situation3d/web/src/theme.css` | 60 |
| 强度 | `--ring-strength-step` | `2px` | `situation3d/web/src/theme.css` | 61 |
| 强度 | `--ring-max-rings` | `4` | `situation3d/web/src/theme.css` | 62 |
| 强度 | `--strength-0` | `#7d8a99` | `situation3d/web/src/theme.css` | 36 |
| 强度 | `--strength-1` | `#4a9c6d` | `situation3d/web/src/theme.css` | 37 |
| 强度 | `--strength-2` | `#d8a12a` | `situation3d/web/src/theme.css` | 38 |
| 强度 | `--strength-3` | `#d9772a` | `situation3d/web/src/theme.css` | 39 |
| 强度 | `--strength-4` | `#c8352c` | `situation3d/web/src/theme.css` | 40 |
| 时效 | `--pulse-period-default` | `1.5s` | `situation3d/web/src/theme.css` | 50 |
| 时效 | `--pulse-period-fast` | `0.8s` | `situation3d/web/src/theme.css` | 51 |
| 时效 | `--pulse-speed-threshold-kmh` | `50` | `situation3d/web/src/theme.css` | 52 |
| 时效 | `--pulse-enabled` | `1` | `situation3d/web/src/theme.css` | 53 |
| 交互 | `--hover-offset` | `8px` | `situation3d/web/src/theme.css` | 56 |
| 交互 | `--hover-transition` | `200ms` | `situation3d/web/src/theme.css` | 57 |

## 3. 规则实现位置（`theme.js` 的纯函数）

| 规则 | 函数 | 行号 |
| --- | --- | --- |
| 脉冲周期选择（>50km/h → 0.8s，否则 1.5s） | `pulsePeriodSeconds` | 109 |
| 强度内环半径（强度 0 不画；1..4 递增；封顶） | `strengthRingRadius` | 118 |
| 读取 CSS 变量 | `（见该行）` | 68 |

## 4. 加载顺序（顺序错了只在运行时报错）

| 顺序 | 脚本 | `index.html` 行号 |
| --- | --- | --- |
| 1 | `style.css` | 8 |
| 2 | `theme.css` | 9 |
| 3 | `theme.js` | 253 |
| 4 | `eventFeed.js` | 254 |
| 5 | `renderGovernor.js` | 260 |
| 6 | `dataHealth.js` | 261 |
| 7 | `statusPanels.js` | 264 |
| 8 | `app.js` | 270 |

**约束**：`theme.js` 必须在 `app.js` 之前（`app.js` 经 `Theme` 取语义色）；
`eventFeed.js` 必须在 `statusPanels.js` / `dataHealth.js` 之前。

## 5. 截图位置（**待补**）

> 判据 1) 要求「视觉编码对照表逐项可在界面复现**且截图评审通过**」。
>
> **本节如实留空**：本环境**没有浏览器**（也无 JS 运行时），
> 无法渲染界面，故**不能**提供截图。请在有前端的机器上按本表逐项核对后补图。

| 维度 | 截图文件 | 状态 |
| --- | --- | --- |
| 阵营 | `docs/shots/阵营.png` | **待补（本环境无浏览器）** |
| 状态 | `docs/shots/状态.png` | **待补（本环境无浏览器）** |
| 强度 | `docs/shots/强度.png` | **待补（本环境无浏览器）** |
| 时效 | `docs/shots/时效.png` | **待补（本环境无浏览器）** |
| 交互 | `docs/shots/交互.png` | **待补（本环境无浏览器）** |

## 6. 本表的验证

`tests/test_i09_visual_encoding_table.py` 断言：

- 本表的**每个变量名**都在 `theme.css` 中真实存在；
- 表中标出的**行号**确实指向该变量（防止生成后代码又改而表未更新）；
- 五个维度**各有唯一配置来源**，且不出现「同一语义两个来源」。

