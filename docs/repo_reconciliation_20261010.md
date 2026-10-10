# P3（shenji_agent）本地与远程仓库比对与归并报告

- 日期：2026-10-10
- 任务来源：`outputs\兵棋推演9项团体标准对标任务清单_20261010.v1.1.md` 的 **P0-1**（最高优先门禁）
- 归并提交：`d64ad71`（远程 `feat/3d-situation-cesium` 分支）
- 回滚点：分支 `backup/pre-reconcile-20261010_112226` 与 tag `pre-reconcile-20261010_112226`，均指向 `7dfa0a3`

---

## 1. 结论

**本项不是一次 git 分支合并，而是一次内容级归并。** 实测确认两侧仓库**无共同祖先**：本地 `code\shenji_agent` 不存在远程 HEAD `7dfa0a3`，远程也不存在本地 HEAD `65db518`（`git cat-file -t` 双向查证均为 unknown object）。因此**任何 `git merge` 都无法执行**，只能按文件内容逐项归并。

归并结果：远程新增 **98 个此前未入版本控制的文件**，修改 1 个共享文件（`ai/agent.py`），远程受控文件数由 **1049 → 1147**。归并后 `py_compile` 对项目代码 **2200 通过 / 4 失败**，4 处失败全部位于 `Shenji/dataset/.venv/` 内的第三方库（py3.10 语法），**不属于项目代码**。

---

## 2. 归并前的两侧状态（实测）

| 项 | 本地 `code\shenji_agent` | 远程 `/remote-home/lizilong/shenji_agent` |
| --- | --- | --- |
| 分支 | `main` | `feat/3d-situation-cesium` |
| HEAD | `65db518`（2026-05-02） | `7dfa0a3` |
| 领先自身 upstream | `main` 领先 `origin/main` **65 个提交** | — |
| 是否含对方 HEAD | **否** | **否** |
| 工作区未提交 | **327 项** | **17 项** |
| remote 配置 | **无 remote** | `origin`→GitHub `shenji_agent.git`；`fresh`→`../shenji_agent_fresh` |
| 备份分支 | — | 18 个 `backup/before_*` 历史备份 |

### 2.1 差异量化（按文件内容比对，git blob 哈希）

比对方法：远程用 `git ls-tree -r -l HEAD` 取受控文件与 blob 哈希；本地用 Python 计算同算法（`sha1("blob <len>\0" + content)`）后逐文件比对。为使比对聚焦项目资产，两侧均排除工具与产物目录（`.agent/`、`.reasonix/`、`.codegraph/`、`.claude/`、`logs/`、`output/`、`input/`、`experiments/`、`datasets/`、`round5_revision_v12/`、`selective_tactical_restore/`、`map_id/`、`__pycache__/`）。

| 类别 | 数量 | 说明 |
| --- | --- | --- |
| 内容一致 | **452** | 两侧同内容，无需处理 |
| 内容不同（共享文件） | **2** | `AGENTS.md`、`ai/agent.py`——需逐个人工裁决 |
| 仅远程有 | **85** | 其中 84 个为 `paper/` 子树、1 个为 `docs/wargame_engine_data_dictionary.md`——**本地缺失，必须保留远程版本** |
| 仅本地有 | **191** | 其中 174 个为实质文件、17 个为产物/备份（见第 4 节） |

---

## 3. 两个冲突文件的裁决（逐个人工判定）

### 3.1 `ai/agent.py` → **采用本地版本**

逐行 diff 结论：**本地版是远程版的严格超集，无远程独有逻辑丢失**。

| 差异 | 内容 | 判定 |
| --- | --- | --- |
| 新增 95 行 | 新增 `set_mission(mission_type, target_pos)` 与 `_build_priority()`（四种任务意图 ATTACK/DEFENSE/RECON/SEIZE 对应的动作优先级表） | 采纳（远程无此能力） |
| 修改 5 处 | `get_road_type` 与另一处 `basic_data` 取值由 `[...]` 改为 `.get(..., 默认值)`，防 KeyError | 采纳（更健壮） |
| 重构 | `setup()` 中内联的 `self.priority = [...]` 改为 `self.priority = self._build_priority()`；`step()` 中每次更新任务后重建优先级 | 采纳（行为等价且支持任务切换） |
| 唯一「删除」 | 20 项动作枚举的内联列表（重构为方法内返回，内容一致） | 无损失 |

行数：远程 1286 行 → 本地 1352 行；归并后 `python3 -m py_compile ai/agent.py` 通过。

### 3.2 `AGENTS.md` → **保留远程版本**

| | 行数 | 内容 |
| --- | --- | --- |
| 远程版 | 42 | 项目真值规则：工作方式、项目真值来源、远程与同步约定、sidecar 清单、AI 审稿人入口等 |
| 本地版 | 2 | 仅 `# AGENTS.md` 占位标题 |

本地版为早期桩文件，覆盖远程版会**丢失全部项目规则**，故不采纳。本地桩已另存 `/tmp/p3_skipped/AGENTS.md.local` 备查。

---

## 4. 归并执行过程（可复现步骤）

1. **建立回滚点**：`git branch backup/pre-reconcile-20261010_112226 7dfa0a3` + `git tag pre-reconcile-20261010_112226 7dfa0a3`。
2. **构造精确载荷**：按第 2.1 节的差异清单打包为 `p3_payload.tar.gz`（**0.48 MB**），结构为 `new/<路径>`（174 个实质新文件）与 `modified/<路径>.local`（2 个冲突文件）。
3. **归档排除**：打包时排除两类不应入版本控制的文件——
   - **产物/备份（17 个）**：`ai/weights/*.pt`、`ai/weights_incumbent_rule/*.pt(h)`、`ai/agent.py.bak_bulletproof`、`Shenji/.pytest_cache/*`、`reports/v3g_full_route_coverage/full_eval_run.log`；
   - **超大 grep 转储（2 个）**：`reports/v3g_full_route_coverage/route_full_coverage_grep.txt`（79.2 MB）、`reports/v3f_route_reliability/route_reference_grep.txt`（45.5 MB）——两者合计 125 MB，属调试中间产物。
4. **落盘**：解包到 `/tmp/p3_stage`，`cp -p` 逐个落到工作树。
5. **显式路径暂存**：`git add --pathspec-from-file=/tmp/add_list.txt`（**不使用 `git add -A`**，遵守仓库既有提交纪律）。
6. **冲突文件裁决**：按第 3 节分别处理 `ai/agent.py`（采纳本地）与 `AGENTS.md`（保留远程）。
7. **语法自检后提交**：`py_compile` 通过后一次性提交为 `d64ad71`。

### 4.1 被 `git add` 拒绝的 74 个文件（**未纳入，属项目既定决策**）

74 个候选文件被仓库**自带的 `.gitignore`** 忽略，命中规则与实际内容：

| 规则（`.gitignore` 行） | 被忽略的候选文件数 | 代表路径 |
| --- | --- | --- |
| `reports/`（第 8 行） | 63 | `reports/DATA_QUALITY_AUDIT.md`、`reports/ai_reviewer/*`、`reports/v3f_route_reliability/*` |
| `league/`（第 7 行） | 1 | `league/league_state.json` |
| `datasets/`、`*.pt`、`*.pth`、`ai/weights/*.pt`、`ai/weights/model_manifest.json` | 10 | 权重与数据集 |

**判定：这 74 个文件不合入。** 理由：`.gitignore` 为仓库既有、显式、成体系的决策（连同 `datasets/`、`*.pt` 一并忽略），不是本次归并遗漏；`reports/` 是实验产物目录（272 个文件、含临时审计输出），强制加入会与既有约定冲突并显著膨胀仓库。

> **遗留决策点（建议交用户裁定）**：`reports/` 与 `league/` 内确有一部分内容具备长期价值——例如 `reports/DATA_QUALITY_AUDIT.md`、`reports/DATASET_REPORT_CROSS_AUDIT.md`、`reports/ai_reviewer/*`（评审证据矩阵与分数）、`league/league_state.json`（联赛状态）。若希望对标任务清单的复现性要求（C1/C2/D1），建议单独评估「是否将其中白名单子集纳入版本控制」，届时用 `git add -f` 显式加入并同步修改 `.gitignore`，**不宜在本报告中擅自放开**。

---

## 5. 验证证据

### 5.1 归并结果结构

| 检查 | 结果 |
| --- | --- |
| 提交 | `d64ad71` |
| 相对 `7dfa0a3` 的变更 | 99 files changed, 13075 insertions(+), 28 deletions(-) |
| 其中新增 / 修改 / 删除 | 98 / 1 / 0 |
| 受控文件总数 | 1049 → **1147** |
| 是否含 `Shenji/dataset/.venv/` | **否**（0 个文件） |
| 是否含 `.pyc` | **否**（0 个文件） |

### 5.2 关键文件已入册（抽样核验）

| 文件 | `git cat-file -e HEAD:<path>` |
| --- | --- |
| `REPRODUCE.md` | 是 |
| `Shenji/DATA_CARD.md` | 是 |
| `Shenji/ontology/intent_ontology_v2.json` | 是 |
| `Shenji/src/shenji_dataset/release.py` | 是 |
| `docs/wargame_ref/observation_structure.txt` | 是 |
| `scripts/run_v2_collection.py` | 是 |

### 5.3 内容正确性

| 检查 | 结果 |
| --- | --- |
| `ai/agent.py` 行数 | 1352（= 本地版） |
| `ai/agent.py` 含 `def set_mission` | 1 处 |
| `ai/agent.py` 含 `_build_priority` | 3 处 |
| `AGENTS.md` 行数 | 42（= 远程版，项目规则完整） |
| `python3 -m py_compile ai/agent.py` | 通过 |
| 全项目 `py_compile`（`ai/` `training/` `Shenji/` `scripts/`） | **2200 通过 / 4 失败**；4 处失败均在 `Shenji/dataset/.venv/lib/python3.10/site-packages/` 内的 numpy/scipy 第三方文件（py3.10 语法），非项目代码 |

### 5.4 ⚠ 未能执行的验证：P3 测试套件

**本环境无法运行 `python3 -m pytest tests/ -q`**，原因实测如下：

| 检查 | 结果 |
| --- | --- |
| 系统 python3 | 3.8.10，**无 pytest、无 torch** |
| `/remote-home/lizilong/venvs/shenji/` | 存在但 **site-packages 为空**（0 个包） |
| 其他 8 个 venv（bright_cvprw26、dfukg、geoevidence_*、qwen36_35b_controller 等） | **均无 torch** |
| 系统 pip | `No module named pip` |
| conda / mamba | 不存在 |

因此本次归并**只完成了静态语法验证**，未完成运行时回归。这是**既有环境限制**（归并前同样不具备），不是本次归并引入的问题；但对标任务清单 P0-1 判据 3「合并后 `python3 -m pytest tests/ -q` 可运行」**未满足**，需在补齐 torch/pytest 环境后补跑。

---

## 6. 回滚方法

```bash
cd /remote-home/lizilong/shenji_agent
git reset --hard backup/pre-reconcile-20261010_112226   # 需用户显式同意后执行
```

归并前的 `7dfa0a3` 亦由 tag `pre-reconcile-20261010_112226` 固定；工作区原有 17 项未提交改动**未被本次归并纳入或删除**（归并只 add 了显式路径），如需复原可从 `/tmp/p3_stage`（载荷解包）与各文件自身内容取回。

---

## 7. 残留事项（本报告范围内未处理）

1. **P3 测试套件无法运行**（见 5.4）——需先安装 torch/pytest 环境。
2. **工作区仍有 47 项未提交**（归并前为 17 项）：主要为本地新增的 `Shenji/dataset/`、`Shenji/raw/`、`Shenji/benchmarks/` 等大体积数据集目录，以及归并前既有的 `.gitignore`、`ai/__pycache__` 等改动。其中 `Shenji/dataset/**` 体积达 **138 GB**，**明确不应入 git**，建议在 `.gitignore` 中显式忽略。
3. **`reports/` 与 `league/` 的白名单决策**（见 4.1 遗留决策点）。
4. 本地仓库 `code\shenji_agent` **无 remote 配置**，且其 `main` 领先自身 `origin/main` 65 个提交——若后续仍需本地与远程并行开发，需先明确同步策略，否则会再次分叉。

（报告完，2026-10-10）
