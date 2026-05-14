# Tactical Dataset Validation

Command:

```bash
/remote-home/lizilong/venvs/shenji/bin/python scripts/validate_tactical_dataset.py --dataset "datasets/陆战兵棋态势特征数据集/Wargame_Feature_Dataset.csv" --out reports/selective_tactical_restore/tactical_dataset_validation.json --sample-rows 5000
```

Result: `TACTICAL_DATASET_COMPATIBLE`

Evidence:

- File exists at `datasets/陆战兵棋态势特征数据集/Wargame_Feature_Dataset.csv`.
- Size: 409,055,700 bytes.
- Header has 18 columns.
- Label column `样本标签` is present.
- Sample rows checked: 5,000.
- Parsed feature matrix shape: `[5000, 32]`.
- Expected tactical feature dimension: 32.
- Missing values in sample: 6, handled by current dataset loader.

The validator uses the current `training.data.TacticalPositionDataset` path, so this checks the current training/feature expectation rather than only the raw CSV header.
