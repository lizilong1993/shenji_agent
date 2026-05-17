# Runtime Path Connection

Date: 2026-05-14

## Current code references

### Checkpoint (`ai/weights/tactical_model.pt`)

- `ai/policy.py:253-258`: Loads via manifest resolution or direct path `tactical_model.pt`, uses `torch.load()` and `_load_tactical_model()`
- `ai/policy.py:136-144`: `_load_tactical_model()` builds `TacticalScorer` and calls `load_state_dict(payload["model_state"])`
- `ai/models.py:100-113`: `TacticalScorer(input_dim=32, hidden_dim=128, output_dim=1)` — simple 3-layer MLP
- `ai/weights/model_manifest.json`: Already references `"tactical_model": "tactical_model.pt"` in managed_files

### Dataset (`datasets/陆战兵棋态势特征数据集/Wargame_Feature_Dataset.csv`)

- `training/data.py:26`: `discover_dir(datasets_root, required_file="Wargame_Feature_Dataset.csv")`
- `training/data.py:209-267`: `TacticalPositionDataset` loads CSV expecting label column "样本标签"
- `training/supervised.py:187-188`: `train_tactical_model()` loads via `TacticalPositionDataset`

## Connection status

| File | Runtime path | Git-ignored? | Referenced by code? | Loader expects it? | Status |
|---|---|---|---|---|---|
| `tactical_model.pt` | `ai/weights/tactical_model.pt` | YES (`*.pt`) | `ai/policy.py`, `model_manifest.json` | YES | Already at runtime path |
| `Wargame_Feature_Dataset.csv` | `datasets/陆战兵棋态势特征数据集/Wargame_Feature_Dataset.csv` | YES (`datasets/`) | `training/data.py` | YES | Already at runtime path |

## Manifest update

No update needed. The manifest already references `tactical_model.pt`. The file exists on disk and is git-ignored.

## Conclusion

Both assets are at their expected runtime paths. No symlinks, copies, or manifest updates are needed.
