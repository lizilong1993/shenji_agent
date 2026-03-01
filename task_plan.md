# Task Plan: Model Training with Real Datasets

## Goal
Train `IntentionLSTM` and `SituationGCN` using the newly identified real datasets (`WG-StateGraph`, `WargameData_mini01`, and `陆战兵棋态势特征数据集`) in the Docker environment, then run a full evaluation.

## Phases

### Phase 1: Data Inspection & Loading
- [ ] Inspect `datasets/意图识别研究数据集WargameData_mini01/` (JSON format).
- [ ] Inspect `datasets/[重庆大学] 分队级陆战兵棋态势表征图数据集 WG-StateGraph/` (likely custom format or images/JSON).
- [ ] Inspect `datasets/陆战兵棋态势特征数据集/` (CSV format).
- [ ] Create/Update `ai/dataset_loaders.py` to handle these specific formats.

### Phase 2: Model Training (Real Data)
- [ ] Update `ai/train.py` (or `train_models.py`) to use the new loaders.
- [ ] Train `IntentionLSTM` using `WargameData_mini01`.
- [ ] Train `SituationGCN` using `WG-StateGraph` or `陆战兵棋态势特征数据集`.
- [ ] Save new weights to `ai/weights/`.

### Phase 3: Full Evaluation
- [ ] Run `run_evaluation.py` inside the Docker container with the new weights.
- [ ] Analyze results (expecting potentially non-zero win rate or different behavior).
- [ ] Update `test_results.md` and `进度报告_20260301.md`.

## Current Phase
Phase 1: Data Inspection & Loading
