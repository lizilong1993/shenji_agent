# Restore Decision

State: BACKUP_PARTIAL_REVIEW_REQUIRED

## Backup audited

- Source backup path: `/remote-home/lizilong/shenji_agent_backup`
- Quarantine path: `/remote-home/lizilong/recovery/shenji_backup_audit/candidate_shenji_agent_backup_20260514_hardlink_041215`
- Current project path: `/remote-home/lizilong/shenji_agent`

## Decision

The backup is plausibly a Shenji_agent backup and contains useful missing assets, but it is not proven valid for direct restoration. It has no `.git` history, lacks the current `paper/` tree, lacks many current reports and experiment artifacts, and differs from the current source layout. Therefore no overwrite or broad merge is recommended.

## Candidate files/assets for human-reviewed selected restore

- `ai/weights/tactical_model.pt` (53,013 bytes; present in backup, missing in current)
- `experiments/shenji_agent/20260327_141522/tactical_model.pt` (53,013 bytes; present in backup, missing in current)
- `datasets/陆战兵棋态势特征数据集/Wargame_Feature_Dataset.csv` (409,055,700 bytes; present in backup, missing in current)
- Backup-only source snapshots under `ai/`, `training/`, `scripts/`, and `tests/` should be reviewed manually before any restore because current repo versions are newer/different.

## Files/assets that must not be restored automatically

- Do not overwrite current `ai/*.py`, `training/*.py`, `scripts/*.py`, `tests/*.py`, or config files.
- Do not overwrite current `land_wargame_sdk/Data/Data/`; current and backup SDK counts match for scenarios/maps.
- Do not overwrite incumbent checkpoints already present in current repo.
- Do not restore `__pycache__/`, logs, temporary vendor folders, or generated slide/tmp outputs.
- Do not copy large datasets/checkpoints into the git index unless explicitly intended.

## Backup handling recommendation

Keep `/remote-home/lizilong/shenji_agent_backup` and the quarantine copy for review. Do not delete or rename until the user confirms which large assets should be restored and how they should be tracked or externalized.

## Next recommended action

Human review should confirm whether to non-destructively copy only the missing tactical checkpoint and tactical dataset, preferably with a large-file policy first. Until then, keep the current project unchanged and do not resume V2/V3 experiments.
