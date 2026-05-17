# Start State — Selective Tactical Asset Restore

Date: 2026-05-14

## Repository

- **Path**: `/remote-home/lizilong/shenji_agent`
- **Branch**: `main`
- **HEAD**: `0eafcbe7c66a8d121156dbda91d952e6e361d5e7`
- **Worktree**: Clean (one untracked file: `goals/codex_goal_selective_tactical_asset_restore.md`)

## Reference commits

- **Audit commit**: `0eafcbe` — "audit Shenji_agent backup integrity before v2"
- **V2 SAFE_STOP commit**: `33fc11f3eec0993b61494a60bd37e7e09e9f38f1` — "upgrade evidence toward top-tier submission candidate"

## Tactical blocker state

- `ai/weights/tactical_model.pt`: **MISSING** from current repo
- `datasets/陆战兵棋态势特征数据集/Wargame_Feature_Dataset.csv`: **MISSING** from current repo (datasets/ directory is empty)
- V2 readiness: `NOT_READY_SAFE_STOP` — tactical contribution is forbidden
- No prior selective restore attempt recorded

## Backup branch

Created: `before_selective_tactical_asset_restore_20260514_050548`

## Pre-restore state summary

| Item | Status |
|---|---|
| Worktree clean | YES (only goal file untracked) |
| Backup branch created | YES |
| Quarantine backup accessible | YES |
| Tactical checkpoint candidate | FOUND (52K, SHA256 9fc261e5...) |
| Tactical dataset candidate | FOUND (391M, SHA256 824dbb0f...) |
| Current repo has tactical checkpoint | NO |
| Current repo has tactical dataset | NO |
| Large binary policy | NOT YET DECIDED |
