# Runtime Connection

Runtime files were connected by no-overwrite copy:

- `ai/weights/tactical_model.pt`
- `datasets/陆战兵棋态势特征数据集/Wargame_Feature_Dataset.csv`

No symlinks were used.

Manifest/code findings:

- `ai/weights/model_manifest.json` already declares `"tactical_model": "tactical_model.pt"` and includes it in `managed_files`.
- `ai/policy.py` resolves `tactical_model` through the manifest and loads `TacticalScorer`.
- `training/data.py` discovers `Wargame_Feature_Dataset.csv` under `datasets/` and expects label column `样本标签`.
- No manifest update was needed.

Both runtime files are ignored by git and must not be committed. See `LOCAL_ASSET_PATHS_NOT_COMMITTED.md`.
