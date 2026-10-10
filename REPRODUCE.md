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
