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
