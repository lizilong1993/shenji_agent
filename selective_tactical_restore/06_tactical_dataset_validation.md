# Tactical Dataset Validation

Date: 2026-05-14

## Result

**TACTICAL_DATASET_COMPATIBLE**

## Details

| Property | Value |
|---|---|
| File | `datasets/陆战兵棋态势特征数据集/Wargame_Feature_Dataset.csv` |
| Exists | Yes |
| Size | 409,055,700 bytes (~391 MB) |
| Encoding | GBK (Chinese) |
| Rows (est.) | 24,080 (data) + 1 header |
| Columns | 18 |

### Schema

| Index | Column | Type |
|---|---|---|
| 0 | ID | identifier |
| 1 | 算子类型 (Operator type) | categorical |
| 2 | 算子所属方 (Operator affiliation) | categorical |
| 3 | **样本标签 (Label)** | target (binary) |
| 4-17 | Feature columns | mixed (numeric + list-like) |

Label column: **样本标签** — FOUND
Feature count: 17 (will be padded to TACTICAL_FEATURE_DIM=32 by loader)
Expected feature dim: 32 (via `training/data.py` TacticalPositionDataset padding)

### Compatibility note

The dataset has 17 feature columns. The `TacticalPositionDataset` loader in `training/data.py`
pads or truncates to exactly `TACTICAL_FEATURE_DIM=32`:
```python
if features.shape[1] < TACTICAL_FEATURE_DIM:
    features = np.pad(features, ((0, 0), (0, TACTICAL_FEATURE_DIM - features.shape[1])))
```

This is compatible with the existing data pipeline.

### Verdict

Dataset is fully compatible with the current training/data pipeline.
