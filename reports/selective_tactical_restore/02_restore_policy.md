# Restore Policy

Policy: copy selected assets only. Do not restore source files, `paper/`, `experiments/`, full `ai/`, full `training/`, full `scripts/`, or full `tests/` from backup.

Selected assets were first copied into ignored local storage:

- `external_assets_recovered/shenji_backup_20260514/ai/weights/tactical_model.pt`
- `external_assets_recovered/shenji_backup_20260514/ai/weights/tactical_model_from_experiment_20260327_141522.pt`
- `external_assets_recovered/shenji_backup_20260514/datasets/陆战兵棋态势特征数据集/Wargame_Feature_Dataset.csv`

Runtime copies were then connected with no-overwrite `cp -an` because current code and manifest expect:

- `ai/weights/tactical_model.pt`
- `datasets/陆战兵棋态势特征数据集/Wargame_Feature_Dataset.csv`

Git handling:

- `.gitignore:16` ignores `ai/weights/*.pt`, including `ai/weights/tactical_model.pt`.
- `.gitignore:3` ignores `datasets/`.
- `.gitignore:10` ignores `*.pt` under `external_assets_recovered/`.
- `git ls-files` reported no tracked tactical checkpoint or tactical dataset path before restore.

Large binary/data files must remain local and uncommitted.
