# Reproduction Guide / 复现指南

## 环境要求

- Python 3.10
- PyTorch (with CUDA support recommended)
- NumPy
- 远程服务器: `lizilong@146.56.220.99:21431`

## 项目结构

```
shenji_agent/
├── ai/                    # AI 核心代码
│   ├── agent.py          # Agent 入口 (setup/step/reset)
│   ├── policy.py         # 学习策略 (ensemble scoring)
│   ├── models.py         # 神经网络定义 (LSTM/GCN/Tactical)
│   ├── features.py       # 特征工程
│   ├── map.py            # 地图工具 (如需要)
│   └── weights/          # 模型权重
│       ├── model_manifest.json
│       ├── intent_lstm.pt
│       ├── situation_gcn.pt
│       ├── tactical_model.pt
│       └── situation_feature_normalizer.json
├── training/              # 训练与评估模块
│   ├── supervised.py     # 监督训练
│   ├── selfplay.py       # 自博弈评估
│   ├── evaluate.py       # 评估管线
│   ├── maps.py           # 地图采样
│   └── __init__.py
├── scripts/               # 工具脚本
│   ├── build_objective_labels.py
│   ├── diagnose_model_quality.py
│   ├── run_ai_reviewer.py
│   └── remote_ops.sh     # 远程操作脚本
├── tests/                 # 测试套件
├── configs/               # 配置文件
├── land_wargame_sdk/      # SDK
└── experiments/           # 实验记录
```

## 快速开始

### 1. 同步文件 (首次)

```bash
# 从远程同步所有缺失文件
bash scripts/remote_ops.sh sync
```

### 2. 训练模型

```bash
PYTHONPATH=/remote-home/lizilong/.local/lib/python3.10/site-packages \
python3.10 train_models.py --config configs/supervised_score_quality.yaml
```

### 3. 运行推演 (Smoke Test)

```bash
PYTHONPATH=/remote-home/lizilong/.local/lib/python3.10/site-packages \
python3.10 -m training.selfplay \
    --config configs/selfplay_smoke_learned_engagement.yaml \
    --candidate ai/weights \
    --incumbent ai/weights_incumbent_rule
```

### 4. 正式评估 (Promotion Gate)

```bash
PYTHONPATH=/remote-home/lizilong/.local/lib/python3.10/site-packages \
python3.10 -m training.selfplay \
    --config configs/selfplay_sota_stop.yaml \
    --candidate ai/weights \
    --incumbent ai/weights_incumbent_rule
```

## 远程诊断

```bash
# 检查远程服务器状态
bash scripts/remote_ops.sh diagnose

# 检查 consecutive_windows 详情
bash scripts/remote_ops.sh check_consecutive

# 上传修复后的代码
bash scripts/remote_ops.sh upload

# 远程运行 smoke test
bash scripts/remote_ops.sh smoke
```

---

## 实验 ID 与检查点验证（I07）

> 依据 `31007 5.3.6`（检查点策略）与 `5.3.7`（实验版本管理与可复现机制）。
> 策略细则见 [`docs/checkpoint_policy.md`](docs/checkpoint_policy.md)。

### 实验 ID 生成规则

```
experiment_id = <generated_at>-<commit 前 8 位>      例如 20261010_180129-bde7df79
```

- `generated_at` 为 `YYYYmmdd_HHMMSS`（`training/common.py::now_tag()`）；
- 后缀把「这一次实验」与「产生它的代码状态」**绑定**；
- **完全确定**：同一时刻 + 同一提交 → 同一 ID，**不含随机数**，
  故第三方复跑时能判断「是不是同一次实验」。

写盘位置：`ai/weights/model_manifest.json` 的 `environment` 段
（**不是**文件名 —— `ai/weights/` 是发布位，文件名须稳定以便推理直接加载）。

### manifest 的六类字段

| 类别 | 字段 |
| --- | --- |
| 代码状态 | `source_commit_sha` |
| 数据/依赖哈希 | `environment.requirements_lock_sha256` |
| 超参 | `history_len`、`horizons` |
| 依赖版本 | `environment.python_version`、`environment.torch_version` |
| 环境 | `environment.platform`、`cuda_available`、`cuda_device_count` |
| 实验身份 | `environment.experiment_id`、`environment.naming_rule` |

### 远程验证命令

```bash
cd /remote-home/lizilong/shenji_agent
source /remote-home/lizilong/py310env.sh          # 提供 python3.10 / torch 2.0.1+cu117

# 1) 校验检查点与 manifest 一致（判据：退出码 0）
python3 scripts/verify_checkpoint.py --weights ai/weights

# 2) 机器可读输出
python3 scripts/verify_checkpoint.py --weights ai/weights --json

# 3) 训练侧回归
python3 -m pytest tests/ -q
```

`verify_checkpoint.py` 检查：六类字段齐全性、权重文件存在且可加载、
**从权重形状反推的维度**与 manifest 声明交叉核对、
`Situation_feature_normalizer.json` 的 `feature_dim` 与 situation 模型输入维度一致
（本仓实测 **108 == 108**）、并打印每个托管文件的 **sha256** 供逐字节核对。

退出码：`0` 通过；`1` 有失败项；`2` 用法错误。

### 环境前置（含鉴权限制）

| 项 | 说明 |
| --- | --- |
| Python | **3.10**（`py310env.sh` 提供；裸 `python3` 是 3.8 且无 numpy/torch） |
| torch | **2.0.1+cu117**（与 `requirements.txt` 的 pin 一致，来自 `venvs/shenji`） |
| git | 需可用（`source_commit_sha` 由 `git rev-parse HEAD` 得到）；容器重建后曾缺失，已装 |
| **引擎鉴权** | **`.engine_config` 的 MAC 绑定鉴权在本环境不通过** → 依赖闭源引擎的流程（自博弈、晋升）**无法运行**。详见 [`docs/deployment_auth.md`](docs/deployment_auth.md)。`verify_checkpoint.py` **不依赖引擎**，可独立运行 |
| 网络 | 校验与训练回归**不需要外网** |

### 历史产物

`ai/weights/` 里可能存有**早于**本策略产生的权重（manifest 无 `environment` 段）。
`verify_checkpoint.py` 对其**降级通过并标注 `legacy_artifact`**，不判失败 ——
**旧权重不因为「没有新字段」就变坏**；若把它判成失败，验证脚本会永远红着，于是没人再看它。

## 检查点与恢复（I07 / `31007 5.3.6`）

### 检查点落盘位置与命名

每次训练在 `<output_root>/<run>/checkpoints/<run>/<模型>/` 下落盘：

| 文件 | 触发条件 | 说明 |
| --- | --- | --- |
| `epoch_{n:04d}.pt` | **定期保存**（每 `checkpoint_every_epochs` 轮，默认 5） | 超出 `checkpoint_keep_last`（默认 3）时删**最旧**一份 |
| `best.pt` | **触发式**：监控指标较历史最优改善 | 监控量为该轮训练损失（越小越好） |
| `crash.pt` | **触发式**：训练异常中断前 | 内含错误信息，便于定位 |

`<模型>` 为 `intent` / `tactical` / `situation` / `objective` 之一。
**注意**：四个模型共用同一 run 目录，故路径中**必须**含模型标签，否则同名检查点会互相覆盖。

### 检查点内容（可用于**恢复**，不只推理）

| 键 | 用途 |
| --- | --- |
| `model_state` | 权重 |
| `optimizer` | **恢复所必需** —— 缺它只能推理 |
| `epoch` | 续跑起点 |
| `rng_state` | python / numpy / torch 随机流（**缺它续跑后随机流不一致**） |
| `metric` | 该档的指标，便于追溯 |

同名 `.json` 侧车文件含元数据（不含大对象），**不加载权重**即可查看某档指标。

### 恢复步骤

```bash
# 1) 选择要恢复的检查点（示例：最近的定期档）
CKPT=<output_root>/<run>/checkpoints/<run>/intent/epoch_0020.pt

# 2) 用同一 Python 环境（依赖版本见 manifest.environment）
set -a; . /remote-home/lizilong/py310env.sh; set +a
export PYTHONPATH="$PYTHONPATH:$(pwd)/Shenji/src:$(pwd)"

# 3) 载入并核对内容齐全（缺 optimizer/rng 则**不能**续跑）
python3 - <<'PY'
import torch
d = torch.load("<CKPT>", map_location="cpu")
for k in ("model_state", "optimizer", "epoch", "rng_state", "metric"):
    print(k, "有" if d.get(k) is not None else "**缺**")
PY

# 4) 恢复随机流后再继续训练（顺序不可颠倒：先恢复 rng，再建 DataLoader）
python3 - <<'PY'
import random, numpy as np, torch
d = torch.load("<CKPT>", map_location="cpu")
random.setstate(d["rng_state"]["python"])
np.random.set_state(d["rng_state"]["numpy"])
torch.set_rng_state(d["rng_state"]["torch"])
print("rng restored at epoch", d["epoch"])
PY
```

### 环境前置（含 MAC 绑定鉴权限制）

| 前置 | 说明 |
| --- | --- |
| Python / torch 版本 | 见 `ai/weights/model_manifest.json` 的 `environment.python_version` / `torch_version` |
| 依赖锁 | `environment.requirements_lock_sha256`（对锁文件重算即可核对） |
| **引擎鉴权（MAC 绑定）** | **本环境不通过** → 依赖闭源引擎的流程（自博弈、晋升）无法运行；**检查点保存与恢复不依赖引擎**，可独立使用 |

### 独立校验

```bash
python3 scripts/verify_checkpoint.py --weights ai/weights   # 退出码 0 即通过
```

校验项含：权重哈希与声明一致、`feature_dim` 与模型输入维度匹配、
**数据哈希对数据文件重算比对**、依赖版本 / 环境 / `experiment_id` 齐备。


---

## 交付物核对（一条入口 + 四项）

> **本仓与交付物清单分置两机**：清单（`outputs/兵棋推演9项团体标准对标任务清单_*.md`）
> 在工作站侧，代码库在本仓库。**故没有任何一个位置能跑全部核对** ——
> 下表写明**在哪跑什么**；**在错的一侧运行会看到环境类失败，那不是交付物有问题**。

### 一条入口

```bash
# 仓库侧（指标覆盖 / 落点引用 / 断言反模式）
python3 scripts/check_all.py <含主清单的目录，可选>

# 工作站侧（交付物健康）
python3 scripts/check_all.py <outputs 目录>
```

### 七项核对（`check_all.py` 一次跑完；**先报环境，再报结果**）

```bash
python3 scripts/check_all.py <含主清单的目录>          # 约 42 s（默认跳过慢项）
python3 scripts/check_all.py <含主清单的目录> --slow   # 另含落点引用（约 4m37s）
```

**入口会先打印"我在哪一侧"**（`scripts/which_side.py`），再列结果：

```
这一侧 = **仓库侧（完整）**
· 可跑：指标覆盖 / 待批准就绪度 / 快照断言 / **落点分级** / 断言反模式
· **不可跑**：交付物健康（**需就地在工作站侧**，本侧无主清单）
```

> **⚠ 为什么先报环境**：本地镜像只有约 **2,000** 个文件，而完整仓库约 **95,000**。
> **用镜像判仓库内容，结论会系统性偏向"找不到"** —— 故该警告必须出现在**数字之前**。

**⚠ 两机布局**：`交付物健康` 需**主清单**（工作站侧）；其余六项需**仓库侧**。
**没有任何一个位置能跑全部七项。**

| # | 核对 | 在哪跑 | 退出码 | 耗时 | 是否判合格 |
| --- | --- | --- | --- | --- | --- |
| 0 | `which_side.py`（**环境判明**） | 任意 | 恒 0 | <1 s | **否** —— 只报环境性质 |
| 1 | `check_deliverable_health.py` | **工作站侧** | 0/1/**2** | <1 s | **是**；**rc=2 = 本机没有主清单（环境不符）** |
| 2 | `check_metric_coverage.py` | **仓库侧** | 0/1 | 0.3 s | **是** |
| 3 | `check_pending_readiness.py` | **仓库侧** | 0/1 | 0.4 s | **是** |
| 4 | `check_snapshot_claims.py` | **仓库侧**（需 `docs/_snapshot_for_check.md`） | 0/**2** | 0.2 s | **是**；rc=2 = 未拷入 |
| 5 | **判据证据** `check_criteria_evidence.py` | **仓库侧**（需 `docs/_tasklist_for_check.md`） | 0/**2** | 0.3 s | **否** —— **已知 5 类假阳性**（百分数/阈值符号/千分位/斜杠枚举/序号混判），须人工复核 |
| 6 | `grade_landing_refs.py` | **仓库侧**（需 `docs/_tasklist_for_check.md`） | 0/**2** | 1.5 s | **否** —— 产出是**分级清单**；**仅 `MISS` 需人工** |
| 7 | `audit_tasklist_landing.py` | **仓库侧** | 0/2 | **4m37s** | **否** —— **默认跳过，加 `--slow`** |
| 8 | `check_assertion_antipatterns.py` | **仓库侧** | 0/1 | 0.1 s | **否** —— **rc=1 是正常结果**，启发式 |

**入口的标记含义**（**勿误读**）：

| 标记 | 含义 |
| --- | --- |
| `OK` | 退出码在预期集合内 |
| **`ENV`** | **rc=2 = 本机缺输入（环境不符）** —— **不等于「跑过了」** |
| `跳过` | 慢项未启用（无 `--slow`） |
| `TIMEOUT` | rc=124，超 120 s 被中止 —— **请单独运行该项** |
| `注意` | 退出码不在预期集合内 |

**`check_all.py` 只让第 1—4 项决定整体退出码。**

### 落点引用的**分级**（`grade_landing_refs.py`）

把"找不到的引用"分成五级 —— **只有 `MISS` 需人工判定**：

| 级别 | 含义 |
| --- | --- |
| `OK` | 在仓库或 `outputs/` 侧找到 |
| `NOTE` | 文件名在**本节正文别处**出现过 → **落点过时**，非缺失 |
| `PLAN` | 紧跟「新增/产出/计划」→ **计划产出**，未创建正常 |
| `EXT` | 路径含 `backend/`/`code/demo` → **外部依赖**（P4 源码不在本检出） |
| `MISS` | 以上都不是 → **需人工判定** |

**在真实仓库上实测：158 条引用 → OK 101 · NOTE 39 · EXT 5 · MISS 13。**

### 另有两项自证测试

```bash
pytest tests/test_assertion_antipattern_detector.py -q   # 检测器必须真的会报
pytest tests/test_i0809_runtime_js.py -q                 # 需先 bash scripts/setup_js_engine.sh
```

### 全量套件基线

```bash
python3 -m pytest tests/ -q
```

**基线：`23 failed / 1105 passed / 11 skipped`** —— 那 **23 条**的**性质已逐条定性**
（见交付物清单 §107）：**12 条指向一个未实现的功能（`tactical_mode`）**，
其余 **11 条为测试侧问题**（窗口/字面量/桩过时/权限位）。
**"23 failed"不是 23 个散落的实现缺陷。**
