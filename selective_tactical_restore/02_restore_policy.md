# Restore Policy

Date: 2026-05-14

## Decision

Both tactical assets were already in place before this task started (verified by checksum match to quarantine backup). Therefore:

1. **No copy needed** — assets already reside at expected runtime paths.
2. **External_assets_recovered directory**: Not created since assets are already in repo at runtime paths.
3. **Git tracking**: Both files are properly git-ignored (`ai/weights/*.pt` and `datasets/` in `.gitignore`). No large binaries will be committed.
4. **No symlinks needed** — files are at direct runtime paths.

## Tracking status

| Path | Ignored? | Tracked? | Source |
|---|---|---|---|
| `ai/weights/tactical_model.pt` | YES (`.gitignore` rule `ai/weights/*.pt`) | NO | Backup quarantine (verified) |
| `datasets/陆战兵棋态势特征数据集/Wargame_Feature_Dataset.csv` | YES (`.gitignore` rule `datasets/`) | NO | Backup quarantine (verified) |

## Policy compliance

- ✅ No whole-backup restore
- ✅ No current source files overwritten
- ✅ Large binaries git-ignored
- ✅ Assets verified against quarantine checksums
