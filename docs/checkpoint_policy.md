# 检查点策略（I07 步骤 1）

> 依据 `31007 5.3.6 模型检查点策略`（定期保存、触发式保存、保存内容、命名规范与版本管理、
> 恢复与验证）与 `31007 5.3.7 实验版本管理与可复现机制`。

---

## 1. 命名规范

| 对象 | 规范 |
| --- | --- |
| 实验 ID | `<generated_at>-<commit8>`，例如 `20261010_180129-bde7df79` |
| `generated_at` | `YYYYmmdd_HHMMSS`（`training/common.py::now_tag()`） |
| 检查点文件 | 沿用既有固定名（`intent_lstm.pt` / `situation_gcn.pt` / `tactical_model.pt`），**不把实验 ID 编进文件名** |

### 为什么实验 ID 不编进文件名

判据要求「任一次训练结果可被第三方复跑核验」，而 `ai/weights/` 是**发布位**（固定名，供推理直接加载）。
把实验 ID 编进文件名会让推理侧必须知道"最新一次是哪个 ID"才能加载 —— 那是把**实验追踪**和
**部署寻址**两件事混在一起。故：**文件名保持稳定，实验身份写进 `manifest.environment.experiment_id`**。

### 为什么上溯到提交号

实验 ID 的后缀是产生它的提交前 8 位。这样"这一次实验"与"代码状态"**绑定**：
只看 ID 就知道该用哪份代码复跑，不必再去翻日志。

---

## 2. 保存触发

### 2.1 本策略**要求**的触发（目标态）

| 触发 | 条件 | 说明 |
| --- | --- | --- |
| **定期** | 每 N 个 epoch 末（建议 N=1，可配置 `save_every_n_epochs`） | 崩溃/抢占时不丢训练进度 |
| **触发式（指标改善）** | 验证指标创新低 | 保留最优权重，供后续选择与发布 |
| **触发式（崩溃前）** | 捕获 SIGTERM/SIGINT 或进入 `finally` 分支 | 异常退出前落最后一次盘 |
| **触发式（发布）** | `config.publish_to_ai_weights = true` | 训练结束把托管文件 + manifest 复制到 `ai/weights/` |

### 2.2 **当前实现的实际行为（与上表的差距，必须如实读）**

> ⚠ 本节此前写成"训练本身已按 epoch 保存""验证损失创新低→保留最优权重"，
> **那是把目标态写成了既有事实**（经独立评审实测证伪）。下面是实测现状。

| 项 | 现状 | 差距 |
| --- | --- | --- |
| 定期保存 | ❌ **没有**。`training/supervised.py` 的 3 处 `torch.save`（intent / situation / tactical）**全部位于 `for _ in range(epochs)` 循环之外**，训练跑完只落**最后一次**盘 | 未实现，且无 `save_every_n_epochs` 之类配置项 |
| 指标改善 | ❌ **没有**。验证只在训练全部结束后跑一次，**无逐 epoch 验证、无 best 跟踪、无 best 回存** | 保存的是最后一个 epoch 的权重，不是最优权重 |
| 崩溃前保存 | ❌ **没有**。无 signal handler，也无 `try/finally` 落盘 | 进程被抢占即丢失本次训练 |
| 发布复制 | ✅ 有。`publish_to_ai_weights=true` 时 `copy_publish_artifacts()` 复制托管文件 + manifest | 无差距；注意配置为 `false` 时**根本不产出 manifest**（也就没有实验 ID） |

**结论**：`31007 5.3.6` 的「定期保存」「触发式保存」在本仓**尚未满足**，
本节其余部分是**要求**而非既成事实。**不得**据 §2.1 判定合规。

> **为什么本轮不改训练循环**：I07 只固化**策略与验证**；改循环会与 I06（同样要碰
> `supervised.py`）冲突，且属独立一轮的工作量（需重跑训练验证）。

---

## 2b. 保留份数

| 项 | 规定 |
| --- | --- |
| **保留策略** | 每个模型保留 **最近 K 次定期检查点**（建议 `keep_last_n = 3`）**加上最优权重 1 份**（`best/`） |
| **最优权重** | 按验证指标选取；指标口径必须与训练 config 一致，且写入 manifest |
| **清理时机** | 成功写入新检查点**之后**清理，避免清理失败导致无盘可回 |
| **清理范围** | **只清理训练输出目录内的检查点**；`ai/weights/`（发布位）**永不自动清理** |
| **命名** | 定期：`<experiment_id>/epoch_{n:04d}.pt`；最优：`<experiment_id>/best.pt`；发布位仍用固定名 |
| **现状** | ❌ **未实现**。当前每个模型只落一份盘，故"保留份数"无从谈起。本节是**要求** |

> 当前既没有多份检查点，也就没有"保留份数"问题；一旦按 §2.1 实现定期保存，
> **必须同时实现本节**，否则检查点会无限堆积。

---

## 3. 保存内容（manifest 六类字段）

`ai/weights/model_manifest.json` 必须含以下六类：

| 类别 | 字段 | 来源 |
| --- | --- | --- |
| 1. **代码状态** | `source_commit_sha` | `git rev-parse HEAD` |
| 2. **数据/依赖哈希** | `environment.requirements_lock_sha256` | `requirements.lock` 或 `requirements.txt` 的 sha256 |
| 3. **超参** | `history_len`、`horizons` | 训练 config |
| 4. **依赖版本** | `environment.python_version`、`environment.torch_version` | 运行时实测 |
| 5. **环境** | `environment.platform`、`cuda_available`、`cuda_device_count` | 运行时实测 |
| 6. **实验身份** | `environment.experiment_id`、`environment.naming_rule` | 由 1 + `generated_at` 派生 |

外加既有的：托管文件名、以及 situation 归一化器的 `kind / fitted_on / feature_dim / zero_std_count`。

### ⚠ 一个必须遵守的约束：新增项要收进 `environment` 这一个 dict

`training/common.py::copy_publish_artifacts()` 会遍历 manifest，
把**任何字符串值**当作文件路径去 `run_dir / rel_path` 复制，**只跳过 `generated_at` 与 `managed_files`**。

它的行为是「`src` 不存在就跳过」，所以今天不会出错 —— 但这只是**巧合式的安全**：
将来若 `run_dir` 下恰好出现同名文件，manifest 里的实验 ID、平台字符串就会被当成文件复制过去。

故新增信息**集中在 `environment` 一个 dict 里**：该函数遇到 dict 值会跳过（非字符串），
**无需修改它**，也就避免了碰 I06 同样会改的文件。

---

## 3b. 检查点载荷规范（31007 5.3.6「保存内容」）

除 manifest 六类字段外，**检查点载荷本身**必须足以支撑"恢复训练"与"精确重现"：

| 载荷项 | 为什么需要 | 现状 |
| --- | --- | --- |
| `model_state` | 权重本体 | ✅ 有 |
| 模型结构参数（`history_len`/`horizons`/`input_dim`/`hidden_dim`/`feature_dim`/`num_classes`） | 无结构参数就无法重建模型 | ✅ 有 |
| `optimizer_state` | 续训需要优化器动量 | ❌ **无** → **不支持断点续训** |
| `epoch` / 全局步数 | 判断从哪继续 | ❌ **无** |
| `rng_state`（python/numpy/torch） | 精确重现要求随机流可复位 | ❌ **无** → 只能"同 seed 从头重跑"，不能"从检查点续跑且逐位一致" |
| `normalizer` 契约 | 推理必须按训练时的归一化口径 | ✅ 部分（`training_applied_normalizer`/`normalizer_kind`） |

**结论**：当前检查点是**推理用权重**，**不是可续训检查点**。
「恢复验证」（§4）因此只能验证"权重与声明一致"，**不能验证"训练可恢复"**。
这一限制必须在符合性声明里写明，不得表述为"已满足 5.3.6 的保存内容要求"。

---

## 4. 恢复与验证

```bash
# 校验权重与 manifest 声明一致（判据 2）
python3 scripts/verify_checkpoint.py --weights ai/weights
```

`scripts/verify_checkpoint.py` 检查：

1. **六类字段齐全性**；
2. **权重文件存在、可加载**（裸 `state_dict` 与包装型都支持）；
3. **从权重形状反推的维度**与 manifest 声明交叉核对（`hidden_dim` / 归一化器 `feature_dim`）；
4. **归一化器 `feature_dim` == situation 模型输入维度**（本仓实测 **108 == 108**）；
5. 输出每个托管文件的 **sha256**，供第三方逐字节核对。

退出码：`0` 通过；`1` 有失败项；`2` 用法错误。

---

## 5. 历史产物如何处置（**重要**）

31007 的检查点策略是**新要求**，而 `ai/weights/` 里已有一份**早于该策略产生的**权重
（其 manifest 无 `environment` 段）。

**处置：按「历史产物」降级通过并标注，不判失败。**

理由：**旧权重不因为"没有新字段"就变坏**。把它判成失败会让"通过"失去意义 ——
验证脚本会永远红着，于是没人再看它。故：

- 缺 `environment` → 记 `warnings: legacy_artifact`，对应检查项**判通过**并在输出里标注；
- 其余能核的项（文件存在性、可加载性、形状、归一化器维度、sha256）**照常核**。

---

## 6. 已知局限（不夸大）

1. **`requirements_lock_sha256` 记的是 `requirements.txt`**（本仓无 `.lock`）；
   而 `requirements.txt` 未必钉死全部传递依赖 —— 故它**不等于完整环境锁**。
   要真正可复现需 `pip freeze` 或 lock 文件，属独立改动。
2. **"同 seed 两次训练权重逐字节一致"既未实现也无守护**（经独立评审四种搜法确认全仓/全史零命中）。
   判据 3 只满足了括号里的"既有测试保持通过"，**实质要求未达成**。
   若要达成，需新增一个小规模双跑比对用例；另注意 `training/common.py::set_seed()`
   **未设** `torch.use_deterministic_algorithms(True)` / cudnn deterministic，
   故在 CUDA 上"逐字节一致"未必成立 —— 需先工程化确定性再谈守护。
3. **未改训练循环**：定期/触发式保存的具体实现沿用既有代码，本文件只固化**策略与验证口径**。
