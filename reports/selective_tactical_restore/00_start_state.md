# Start State

Date: 2026-05-14

## Repository

- Path: `/remote-home/lizilong/shenji_agent`
- Branch: `main`
- HEAD: `0eafcbe7c66a8d121156dbda91d952e6e361d5e7`
- Worktree before restore: dirty only by expected untracked goal/report workspace from this task.
- Audit commit: `0eafcbe`
- V2 SAFE_STOP commit: `33fc11f3eec0993b61494a60bd37e7e09e9f38f1`
- Backup branch created: `backup/before_selective_tactical_asset_restore_20260514_134407`

## Tactical Blocker State

- `ai/weights/tactical_model.pt`: missing before selected restore.
- `datasets/陆战兵棋态势特征数据集/Wargame_Feature_Dataset.csv`: missing before selected restore.
- V2 state: `NOT_READY_SAFE_STOP`; tactical contribution claim forbidden until executable diagnostics prove the branch loads and produces meaningful scores.

The audited backup is used only as a selective asset source. No whole-backup restore is permitted.
