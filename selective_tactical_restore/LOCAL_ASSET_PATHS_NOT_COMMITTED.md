# Local Asset Paths (Not Committed)

The following files are git-ignored and will not be committed. They are at expected runtime paths.

| Asset | Path | Git-ignored by |
|---|---|---|
| Tactical checkpoint | `ai/weights/tactical_model.pt` | `.gitignore`: `ai/weights/*.pt` |
| Tactical dataset | `datasets/陆战兵棋态势特征数据集/Wargame_Feature_Dataset.csv` | `.gitignore`: `datasets/` |

These files must exist at these paths for the tactical module to work at runtime.
If cloning this repo fresh, copy these files from the quarantine backup:
```
QUARANTINE="/remote-home/lizilong/recovery/shenji_backup_audit/candidate_shenji_agent_backup_20260514_hardlink_041215"
cp -an "$QUARANTINE/ai/weights/tactical_model.pt" ai/weights/tactical_model.pt
mkdir -p "datasets/陆战兵棋态势特征数据集"
cp -an "$QUARANTINE/datasets/陆战兵棋态势特征数据集/Wargame_Feature_Dataset.csv" \
  "datasets/陆战兵棋态势特征数据集/Wargame_Feature_Dataset.csv"
```
