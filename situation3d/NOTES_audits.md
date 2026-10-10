# 审计记录：哪些轴已扫过、结论是什么

本文只记一件事：**哪些审计轴已经扫过、结论如何**，以及扫描方法。
目的是让下一个人不要从零再扫一遍，而是直接去看“本轮新加的代码”。

**为什么需要它**：这个项目里的缺陷大多“**不报错**”——界面不报、
`pageErrors` 为空、案例照样加载。所以只能靠系统性扫描。而扫描结果不落盘，
下一个人的起点就还是零。

## 已扫过的轴（均已干净，除非注明）

| 轴 | 方法 | 结论 |
|---|---|---|
| 悬空 DOM id | `index.html` 现存 id 全集 对 `app.js` 的 `getElementById`/`bindCheck` | **修了 6 处**（含建筑体块整条链路） |
| 定义但不可达的函数 | 全仓引用计数 | **删了 3 个 setter** + 附带死绑定 |
| 空 `catch` 吞异常 | 正则扫 `catch` 空块 | **9/9 均合理**（都是“移除已消失的实体”） |
| JS 用但 CSS 未定义的类 | 类名集 对 `style.css` | 2 个，属**有意保留的空桩** `renderHotspotTable` |
| 只写不读的 `state.*` | 赋值点 vs 读取点计数 | 3 个无害残留 |
| `LAYERS` 字段是否有人读 | 逐字段在注册表内/消费者里计数 | 唯一惰性声明 `requires`，**已删** |
| 图层是否空转（kind 无 sink） | 直接问 `apply()` 的 `skipped` | **27/27 都有 handler 或 sink** |
| `exclusives` 对称性 | 逐对比 | 两组都双向写了 |
| 声明 vs 盘上事实（瓦片） | Mercator 算该有哪些瓦 + `isfile` | **6/6 零缺**（两战区 × imagery/terrain/wsf3d） |
| URL 钩子是否绕过注册表 | 逐个钩子实测注册表意图 | **4 例已修**，已装通用守卫（16 条 URL） |
| 各模块“效果入口”的调用点 | 逐入口列调用点 | 都只从各自 sink/handler 调用 |
| **事件监听注册/注销配对** | 73 注册 vs 1 注销，逐个看调用位置与 viewer 生命周期 | **无泄漏**（见下） |
| **模块内部状态 vs 注册表意图** | 逐对翻转 `desired`，再读模块自己的 `enabled`/`show` | `places.labels` ✓、`tiles.google3d` ✓、**`wsf3d.overlay` ✗ 已修** |
| **本体副本 vs 单一真源** | 递归结构化 diff（`tools/check_ontology_sync.py`） | 真源 `Shenji/ontology/intent_ontology_v2.json`（22 意图）与副本 `engine/data/intent_ontology_v2.json` **只差 1 个字段**（`/parent_categories[4]/name_zh`：机动态势 vs 机动TS，属术语替换规则的**有意豁免**）；其余**零差异**。已装校验：**除登记豁免外任何差异即失败** |

## 监听器轴的结论依据（2026-10-08）

73 个 `addEventListener` 对 1 个 `removeEventListener` 看着像漏漏，但**逐个看过后不是**：

* 大多数挂在**静态 DOM 节点**上，由 `bindUI()` 在开机时注册一次；
  `bindUI` 全仓**只被调用一次**（app.js:4503）。
* `window.pointermove` / `pointerup`（app.js:2976/2984）看上去像“按一次加一对”，
  实际在 `bindFollowInput()` 里，而它只从 `bindUI` 被调一次（app.js:3890）；
  行为靠 `state.follow.press` 当标志位，不依赖重复注册。
* `situation.js:336` 的 `document.keydown` 与 353 行的 `removeEventListener` **成对**。
* **关键前提**：`new Cesium.Viewer` 全仓**只有一处**（app.js:444），
  `teardownScene()` **不重建 viewer**（只清 entity/primitive）。
  所以开机注册的监听不会随案例切换叠加。

**如果将来真要重建 viewer**，这条结论就不再成立** ——
那时这 70+ 个开机注册的监听会全部挂在旧对象上。
这是本条结论的**前提**，不是一个永久结论。

## 尚未扫的轴

**（无）** —— 2026-10-08 已扫完。

下一步不是“再找一个轴”，而是：

1. **新增代码时对照本清单** —— 历史证明缺陷集中在“新加的图层行 / 新加的 URL 钩子 /
   新加的直接调 setter”。
2. **把 `tools/consistency_selftest.py` 的 `SPECS` 跟上** —— 它现在只盖 4 个图层。
3. 修改后跑下面六套自检。
## 修改后该做什么

每次动过这些轴上的代码后，至少跑：

```
python3 tools/consistency_selftest.py # 意图↔效果（图层行在、效果不发生的那类）
python3 tools/plot_selftest.py        # 标绘端到端
python3 tools/deeplink_selftest.py    # 16 条 URL + 图层意图断言 + 鸟瞰图黄框
python3 tools/road_selftest.py        # 引擎侧
python3 tools/intent_selftest.py
python3 tools/intent_set_selftest.py
python3 tools/check_ontology_sync.py # 本体副本 vs 单一真源（无远程依赖，本地即可跑）
```

若改了 `manifest` 或重抓了瓦片，另跑一遍“声明 vs 盘上事实”
（脚本跟随 `NOTES_offline_coverage.md`）。

## `wsf3d.overlay` 一例的细节（2026-10-08）

**现象**：把 `desired` 置为 true 后，`layersOf('wsf3d.overlay')` **返回空** —— 场景里
根本没有这一层。即**面板上“建筑高度图层（WSF3D）”那个复选框是空转的**。

**根因**：图层的**创建**写在 `setWsf3dOverlay(on)` 里（`if (on && !state.wsf3dOverlayLayer) { ...
attachWsf3dOverlay ... registerImagerySlot ... }`），而那个函数**只有 URL 钩子会调**。
该行在注册表里是 `kind:'imagery'` 且**没有 handler**，所以面板路径走到 imagery sink
时发现“没有已登记的图层”—— 什么都不做，也不报错。

**为什么 URL 钩子看上去是好的**：`?overlay=1` 同时调了 `setSwitch(...)` 与
`setWsf3dOverlay(true)`。那段代码的注释就写着“两者幂等，**且 setter 可能还做了别的事**”
—— 这次就是“别的事”：创建图层。所以只测钩子会得出“正常”的错结论。

**修法**：给该行加一个 handler，只做两件事——需要时创建（并登记槽位）、设 `show`。

⚠ **不能直接把 `setWsf3dOverlay` 挂成 handler**：它末尾调 `applyLayers()`，而 handler
就是在 `apply()` 里被调用的 —— **无限递归**。掉头重排由 `apply()` 在所有 writer
跑完后统一做（`out.imagery = reorderImagery(...)`），不需要在 handler 里做。

**实测**：修后 `desired` 置开 → 图层创建且 `show=true`；置关 → `show=false`；
`places.labels` 与 `tiles.google3d` 仍一致；`pageErrors` 为空。

## `tools/consistency_selftest.py`（意图↔效果，2026-10-08）

逐对翻转图层的 `desired`，再读**场景/模块自己的状态**（不只看 `desired`）。
当前覆盖 4 个图层：`places.labels`、`wsf3d.overlay`、`tianditu.label`、`imagery.offline`。

**它真的会失败 —— 已演示过三阶段：**

| 阶段 | 结果 |
|---|---|
| 正常代码 | ALL CHECKS PASSED |
| 人为破坏：把图层创建条件改成 `if (false)`（复现上一轮那个 bug） | **FAILED (1)**：`wsf3d.overlay 置开后能取到效果读数 读数为 None` |
| 恢复 | ALL CHECKS PASSED |

破坏时**另外 3 个图层仍然通过** —— 失败是定位到具体图层的，没有级联误报。

⚠ 读数为 `None` 没有当成“无法取证”放过 —— 在本自检里它
**就是失败**：图层未创建恰恰是要抓的现象。

新增图层时应把它加进 `SPECS`（每项只需一个 `read()`）。

## 又三条已验证的轴（2026-10-08 补记，均为“只在特定时序/重复下才暴露”的类别）

| 轴 | 方法 | 结论 |
|---|---|---|
| **面板复选框 ↔ 注册表意图** | 翻转 `desired` 后读 `layerRow-<id>.checked` | **一致**；已固化为守卫（10 图层 × 2 案例） |
| **互斥争裁与恢复** | 开 A → 开 B（互斥）→ 关 B，逐步看 `feed` 与场景 | **能恢复且不回退**；已固化 |
| **重复重载下的状态积累** | 各案例重载 2 次，比对 5 项计数 | **逐项恒定不增**；已固化 |
| **场景文档存取往返** | 制造特征状态 → `shareScene()` 出链接 → **打开该链接** | **PASS**；**尚未固化**（见下） |

### 场景文档存取往返：已验证，**尚未固化为守卫**

`shareScene()` → `SceneDoc.shareUrl(viewer, providers)` 产出链接；
`applySceneDoc()` → `SceneDoc.fromUrl()` 从**当前 URL** 读。所以真实用户路径是
**复制链接 → 打开链接**，必须**两次加载**才能验。

制造一个与默认不同的状态（关 `tianditu.label`、开 `places.labels` / `weather.model` /
`imagery.offline`，相机移到 119.7/24.9/210 km），取链接，**再用那个链接开一次**：

```
caseId=taiwan_landing   sceneApplied=True
相机 want=(119.7, 24.9, 210000)  got=(119.7, 24.9, 210000)
图层 want=22  got=22    无缺、无多
=> PASS
```

链接里 `scene=` 是 base64 的 `{v,m,l,c}`：`m`=相机、`l`=28 个图层意图、`c`=案例。

一处细节：文档里 `buildings.blocks` 与 `tiles.google3d` 同时为 `1` —— 它们**互斥**。
这不是 bug：文档记的是**意图**（两方都“想要”），争證交给 `apply()` 时的
`exclusives`，两端行为一致。

**为什么没固化**：要做成守卫，测试框架必须能“先跑一次拿到 URL，
再拿那个 URL 跑第二次”。现有 `run_probe` 只跑单页，加这层需要改框架，
而改框架的风险高于收益。**宁可如实记为“已验证、未固化”。**

## 引擎侧（`engine/`）首次按同样的类别扫（2026-10-08）

此前所有审计都在 `web/src`（前端）。本次把引擎侧（6,700 行 / 15 个模块）
按同样的类别过了一遍：

| 类别 | 方法 | 结果 |
|---|---|---|
| 静默吞异常 | 扫 `except` 后跟 `pass`/`continue`、以及裸 `except:` | **0 处** |
| 模块级常量定义但全仓未用 | 全仓（116 万字符 / 71 个文件）引用计数 | **0 个** |
| 函数定义但全仓未调用 | 同上 | **0 个** |

### ⚠ 第一次扫出了 36 个“死函数”——**全是假阳性**，因为我的正则写错了

我用的是 `(?<![A-Za-z0-9_.])NAME\s*\(`，**排除了前面的点** ——
于是 `scenario.build_dataset(...)` 这类**模块限定调用全被漏掉**，
把实际在用的函数报成了死代码。改成允许点、并把语料扩到
整个项目树（含 `tools/`、`scripts/`、`.md`）之后，结果从 36 变为 **0**。

这与本会话反复出现的主题同源：**审计工具自己也会撒谎**。
若不是因为“36 个死函数”这个数字看起来太夸张而去核实，
我就会去删一批正在被用的代码 —— 那是本会话最危险的一种改动。

**经验**：数量类审计结果如果“多得不像真的”，先怀疑扫描工具。