# Selected Copy Log

Date: 2026-05-14

## Summary

Both primary assets were already present in the current repo at runtime paths. No copy operation was needed.

## Asset verification

| Asset | Source (quarantine) | Destination (current repo) | Size | SHA256 match | Action |
|---|---|---|---|---|---|
| `tactical_model.pt` | `.../candidate_shenji_agent_backup_20260514_hardlink_041215/ai/weights/tactical_model.pt` | `ai/weights/tactical_model.pt` | 53,013 | ✅ `9fc261e5...` | Already in place |
| `Wargame_Feature_Dataset.csv` | `.../datasets/陆战兵棋态势特征数据集/Wargame_Feature_Dataset.csv` | `datasets/陆战兵棋态势特征数据集/Wargame_Feature_Dataset.csv` | 409,055,700 | ✅ `824dbb0f...` | Already in place |

## Checksums recorded

```text
9fc261e560112e33c4a79c7b40a47a0ebf1314595ad37970cb3b6eef6209cb04  ai/weights/tactical_model.pt
824dbb0f81ef4510d762a0d616c4cb3ed45be8b54ec4117c47b86d516ee4e718  datasets/陆战兵棋态势特征数据集/Wargame_Feature_Dataset.csv
```
