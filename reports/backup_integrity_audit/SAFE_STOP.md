# Safe Stop

Reason: backup is useful but not proven valid for restoration.

## What was checked

- Located candidate directory `/remote-home/lizilong/shenji_agent_backup`.
- Created quarantine copy at `/remote-home/lizilong/recovery/shenji_backup_audit/candidate_shenji_agent_backup_20260514_hardlink_041215`.
- Audited identity markers and expected Shenji_agent directories.
- Searched for missing checkpoints, datasets, SDK assets, and experiments.
- Compared current and backup file manifests.
- Generated checksums for critical backup binaries/datasets.

## Why audit cannot continue to restore

The backup lacks `.git` provenance and `paper/`, and current repo contains many files absent from the backup. The backup also contains large binary/dataset candidates whose tracking policy is ambiguous. Restoration would require human selection and must remain non-destructive.

## Human action needed

Approve selected restore of specific missing assets, especially `ai/weights/tactical_model.pt` and `datasets/陆战兵棋态势特征数据集/Wargame_Feature_Dataset.csv`, or upload a complete git-backed backup.

## Preserve current project

Yes. Keep the current project unchanged. No overwrite, destructive sync, or restore was performed.
