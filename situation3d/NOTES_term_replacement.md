# 术语替换：态势 → TS、战场 → ZC

本文记录一次全局术语替换的完整依据：规则、逐处清单、豁免与风险。
清单由 `tools/apply_term_rename.py --report --inventory NOTES_term_replacement.md`
生成，位于下方标记之间；标记之外的文字是手写分析，重新生成**不会**覆盖它。

## 替换规则

| 原词 | 替换为 | 说明 |
|---|---|---|
| 态势 | `TS` | 字面替换，不做分词判断 |
| 战场 | `ZC` | 同上 |
| 战场态势（相邻） | `ZC-TS` | 两个缩写黏连成 `ZCTS` 无法辨读，故加连字符 |

ASCII 拼写 `situation` / `Situation`（219 处 / 42 文件）**不动**：它是标识符不是中文
术语；改名会牵动 `SituationHost` / `SituationPanel` 等公开全局名，属另一轮的重命名工作。
替换后代码里会出现「中文文案叫 TS、标识符叫 Situation」的两套称呼，这是已知代价。

## 结构性判定：这是纯文本替换，不是 schema 变更

排查过、均**无**命中：Python/JS 变量名与函数名、JSON **键名**、CSS 类名与选择器、
HTML `id`/`class`。已提交的数据文件（`engine/data/roads/*.json` 76 MB、
`vector/*.geojson`）grep 计数为 0。因此替换不可能改变代码行为或运行期字段。

## 豁免项（有意保留旧词）

| 标记 | 位置 | 保留理由 |
|---|---|---|
| R1 产品名 | `README.md:1`、`web/index.html:6,13`、`web/src/app.js:1,733,853`（共 6 处） | 产品名「三维战场态势生成系统」。机械替换得「三维ZCTS生成系统」，作为对外名称不可读。已在 `index.html` 用注释标注。 |
| R2 本体论术语 | `engine/data/intent_ontology_v2.json:65` `"name_zh": "机动态势"` | 它是 6 个父类名称之一（`MANEUVER_POSTURE`），会显示在「目标意图」栏；且该文件是 `code/shenji_agent/Shenji/ontology/intent_ontology_v2.json` 的**内嵌副本**，只改一边会让两处口径不一致。已在 `engine/intent.py` 的 `_ONTOLOGY_REL` 旁注释。 |

R3「战场态势」相邻**无独立出现**：实测 6 处「战场态势」全部属于产品名，因此
`ZC-TS` 规则实际未被触发（清单里没有 RISK 条目）。

## 风险与不处理项

| 项 | 判定 | 处置 |
|---|---|---|
| `ts-*` CSS 类名前缀 | 与「态势→TS」**同形但无关**（那是"目标状态"栏的命名空间） | 类名不动，在 `style.css` 注释注明歧义 |
| `TS` / `ZC` 既有占用 | 替换前独立词 `TS`/`ZC` 在全部文本文件中均为 **0** 次 | 无冲突 |
| `TS` / `ZC` 行业歧义 | `ZC` 亦可指侦察/中程/指挥 | 属既定要求，仅记录 |
| 生成数据集 | `web/data/*.json` 内嵌引擎字符串（`inference_model.levels[].name`、`case_name`、`overall.note`） | **已重建两个现役案例**，并删除 4 个已删案例的陈旧数据集 |
| `docs/庙算平台资料汇总.md` | 位于 `situation3d` 子树之外，且是第三方资料汇总 | 不处理（不在本仓库跟踪范围内） |
| 目录名 `三维态势系统/` | 仅本地存在，远端仓库对应目录是 ASCII 的 `situation3d/` | 不改（改名的路径连锁代价远超收益） |
| Git 历史提交信息 | 改写会破坏哈希 | 不改；新提交起使用新术语 |
| `tools/shots/`、`web/vendor/`、`web/tiles/` | 二进制与第三方 | 不处理 |
| `tools/apply_term_rename.py`、本文 | 规则定义与清单本身必然含原词 | 预期残留 |

## 验证

* 残留复核：全树 grep 后**只剩上述 7 条豁免** + 规则工具 + 本文。
* 语法：改动过的 7 个 JS 文件 `node --check` 通过；全部 `.py` `py_compile` 通过。
* 运行时：两案例浏览器加载 `pageErrors` 为空；界面文案与数据集均已使用新术语。
* 数据集一致性：断言 `inference_model.levels[].name` 与 `case_name` 已是新术语，
  证明是重建而非新旧混用。

## 逐处清单

<!-- BEGIN AUTO-INVENTORY -->
| 文件:行 | 原文 | 替换后 | 标记 |
|---|---|---|---|
| `README.md:1` | # situation3d — 三维战场态势生成系统 | # situation3d — 三维战场态势生成系统 | KEEP (R1 产品名) |
| `engine/data/intent_ontology_v2.json:65` | "name_zh": "机动态势", | "name_zh": "机动态势", | KEEP (R2 本体论术语) |
| `engine/intent.py:69` | # `机动态势` (MANEUVER_POSTURE) is **deliberately left in the original wording** | # `机动态势` (MANEUVER_POSTURE) is **deliberately left in the original wording** | KEEP (R2 本体论术语) |
| `web/index.html:7` | 产品名「三维战场态势生成系统」按术语替换的豁免规则 R1 **有意保留原样**： | 产品名「三维战场态势生成系统」按术语替换的豁免规则 R1 **有意保留原样**： | KEEP (R1 产品名) |
| `web/index.html:12` | <title>三维战场态势生成系统</title> | <title>三维战场态势生成系统</title> | KEEP (R1 产品名) |
| `web/index.html:19` | <strong>三维战场态势生成系统</strong> | <strong>三维战场态势生成系统</strong> | KEEP (R1 产品名) |
| `web/src/app.js:1` | /* situation3d — 三维战场态势生成系统前端 | /* situation3d — 三维战场态势生成系统前端 | KEEP (R1 产品名) |
| `web/src/app.js:733` | document.title = '三维战场态势生成系统 — ' + data.case_name; | document.title = '三维战场态势生成系统 — ' + data.case_name; | KEEP (R1 产品名) |
| `web/src/app.js:853` | document.title = '三维战场态势生成系统'; | document.title = '三维战场态势生成系统'; | KEEP (R1 产品名) |
<!-- END AUTO-INVENTORY -->
