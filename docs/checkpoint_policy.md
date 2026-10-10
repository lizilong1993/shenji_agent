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

| 触发 | 条件 | 说明 |
| --- | --- | --- |
| **定期** | 每个 epoch 末（由 `training/supervised.py` 的既有循环负责） | 训练本身已按 epoch 保存 |
| **触发式（指标改善）** | 验证损失创新低 | 保留最优权重 |
| **触发式（发布）** | `config.publish_to_ai_weights = true` | 训练结束把托管文件 + manifest 复制到 `ai/weights/` |

> **不在本文件里改训练循环**：I07 只固化**策略与验证**，训练触发逻辑沿用既有实现，
> 避免与 I06（同样会碰 `supervised.py`）互相冲突。

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
2. **未自动校验"同 seed 两次训练权重逐字节一致"**：那是训练耗时动作，
   判据 3 交由**既有测试**守护（本轮未新增该测试，见清单 §44 的说明）。
3. **未改训练循环**：定期/触发式保存的具体实现沿用既有代码，本文件只固化**策略与验证口径**。
