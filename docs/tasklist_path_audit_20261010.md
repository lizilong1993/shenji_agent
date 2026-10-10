# 任务清单路径引用核对（2026-10-10）

> 目的：把任务清单里**引用的文件路径**逐条核对，订正其中的**事实性错误**。
> 理由：第 45—46 轮已两次因清单路径写错而多花时间（`D1` 的 `collect_cli.py`）。
> 这类错误**每轮都要重新发现一次**，应一次性核清。

**核对方法（可复现）**：
1. 用正则抽出任务清单里所有反引号包裹、且形如路径的 token → **256 条引用 / 180 个唯一 basename**；
2. 在本地三个根（`code\shenji_agent`、`outputs\`、工作区根）建索引（15,818 个文件 / 7,091 个 basename）；
3. 逐个比对，**未命中项再分类**：真错 / 在别处（`.scratch/`、P4 树、远程）/ 从未产出。

**结果：180 个 basename 中 141 个直接命中，34 个未命中** —— 34 个里**只有 7 个是真的路径引用错误**，
其余分别是「在 `.scratch/`」「在 P4 仓库」「在远程」「通配写法」。

---

## 1. 订正表（**7 处真实错误**）

| # | 清单里的写法 | 实际情况 | 订正 |
| --- | --- | --- | --- |
| 1 | `Shenji\collect_cli.py`（`D1`【落点】） | 不存在 | **`Shenji\src\shenji_dataset\collect_cli.py`** |
| 2 | `docs/xxx.md`（`docs\xxx.md`） | 不存在，**是占位符** | 删除该引用 |
| 3 | `pytest tools/test_x.py` | 不存在，**是占位符** | 删除该引用 |
| 4 | `docs\baseline_ledger.md` | 不存在 | 实际为 **`outputs\基线台账_20261010.md`** |
| 5 | `outputs\标准量化指标对标台账_20261010.md` | 文件名写错 | 实际为 **`outputs\标准量化指标基线与对标台账_20261010.md`** |
| 6 | `docs\merge_buckets.md`、`docs\merge_evidence.md` | 不存在 | **从未产出**（属待做产出，不是错路径） |
| 7 | `docs\ui_theme_spec.md`、`docs\expert_rating_protocol.md`、`docs\competition_platform_assessment.md`、`docs\security_data_policy.md`、`docs\repo_reconciliation_20261010.md`、`docs\std_clause_audit.md` | 不存在 | **从未产出**（属待做产出） |

> 第 6/7 类严格说不是「路径写错」，而是「**引用了还没产出的文件**」——
> 但效果相同：**照着清单去找会找不到**，故一并列出。

---

## 2. 未命中但**不是错误**的（分类说明）

| 类别 | 例子 | 说明 |
| --- | --- | --- |
| 在 `.scratch/`（索引时被跳过） | `.scratch\std\s31002.txt`、`.scratch\bz.txt`、`.scratch\proj_code.md`、`.scratch\proj_docs.md`、`.scratch\std_index.md` | **实测均存在**（6 项） |
| 在远程 | `/remote-home/lizilong/py310env.sh` | 远程文件 |
| **在 P4 仓库** | `backend\app\jobs\consistency_check.py`、`backend\app\db\sync.py`、`backend\app\core\ws_auth.py`、`backend\tests\test_sync_consistency.py`、`backend\tests\bench_ws.py`、`backend\bench\bench_sync_latency.py` | 见 §3 —— **这两项在 `demo20260104` 里都不存在** |
| 通配/枚举写法 | `*.json`、`test_*.py`、`supervised_*.yaml`、`{__init__,agent_api,llm_adapter}.py` | 不适用逐条核对 |

**`.scratch` 下实测存在**：`std\s31002.txt`、`std\s31006.txt`、`std\s31007.txt`、`bz.txt`、
`proj_code.md`、`proj_docs.md`、`std_index.md` —— **7 项全在**。

---

## 3. 一项新发现：6 个 P4 文件在**三个** P4 仓库里都不存在

`consistency_check.py` / `sync.py` / `ws_auth.py` / `test_sync_consistency.py` /
`bench_ws.py` / `bench_sync_latency.py` —— 清单把它们写成 `backend\...` 下的路径。

本地有**三个** P4 树：`code\demo20260104`（`P0-2` 明确指定的**主**交付）、
`code\demo20251219`、`code\demo`。**实测：这 6 个文件在三个树里命中数均为 0 / 6。**

| 树 | 命中 |
| --- | --- |
| `demo20260104` | **0 / 6** |
| `demo20251219` | **0 / 6** |
| `demo` | **0 / 6** |

**→ 故这 6 条不是"路径写错到另一个仓库"，而是「引用了不存在的文件」。**
两种可能，本文不臆断：

1. 它们**是某任务要新建**的文件（清单把"应产出"写成了"既有路径"）；
2. 它们属于**尚未纳入工作区**的 P4 版本。

**处置**：在实施相应任务前必须核清，否则照清单去找会找不到。

---

## 4. 最要紧的一条：`gen_eval_cases.py` 全工作区不存在

清单把 `scripts\gen_eval_cases.py` 列为**评测用例生成脚本**。
**实测：整个工作区（含本地三个根与远程仓库）没有 `gen_eval_cases.py`。**

这条与第 46 轮（§59）的结论**直接相连**：

> `Q7` / `I16` 要求「每种对抗强度下生成超过 100 种标准化测试用例」，
> 而**生成用例的脚本不存在** —— 故「每档 >100 用例」目前**不只是缺定义，也缺工具**。

**这让 `Q7` 的缺口比第 46 轮记录的更清楚**：三档定义已产出，
但**用例生成器是空的**，所以"每档 >100 用例"在可预见的下一步也做不出来。

---

## 5. 顺带核对：`D1` 步骤 6 的发布信息

`D1`【验收判据】3 要求「发布包含**版本、哈希、许可证**，`DATA_CARD.md` 更新」。

| 项 | 实测 |
| --- | --- |
| **版本** | ✓ `release.py` 维护 `dataset_version` / `schema_version` / `ontology_version` / `labeler_version` / `split_version` |
| **哈希** | ✓ `release.py:_sha256()` 逐产物算 SHA-256，并把 `ontology_sha256` 记入 |
| **许可证** | **✗ `DATA_CARD.md` 明写「intentionally not selected」** —— 需**项目负责人**选定 |

> `DATA_CARD.md` 原文（`:42`）：
> 「The public data license and code license are **intentionally not selected** by this
> implementation. The project owner must choose them before public publication…」
>
> **故判据 3 的"许可证"一项不是实现缺口，而是待你决定的**——
> 这也解释了为什么它不该被算作"未做"。

---

## 6. 处置建议（本文只核对，不改任务正文语义）

| 建议 | 说明 |
| --- | --- |
| 把 §1 的 7 处订正**写回任务清单** | 避免每轮重复发现 |
| §3 的 6 个 P4 文件 | 先核实归属仓库/是否为待新建，再订正 |
| §4 的 `gen_eval_cases.py` | 应作为 `Q7`/`I16` 的**独立缺口**登记 |
| §5 的许可证 | 转为**待你裁定**项，不计入实现缺口 |

---

## 7. 未做

| 项 | 状态 |
| --- | --- |
| 180 个 basename 的存在性核对 | **✓ 本文** |
| 路径**内容**是否与清单描述相符（如某文件是否真是"迁移脚本"） | **✗ 未做** |
| `demo20251219` / `code\demo` 两个 P4 树的完整核对 | **✗ 未做**（只核了 `demo20260104`） |
| 工作区根下 `进度汇报\` 等目录的存在性 | **✗ 未做** |
| 把订正写回任务清单 | **✗ 未做**（本文只给核对结果） |
