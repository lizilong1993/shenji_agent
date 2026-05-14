# Phase 0 — Start State

**Date:** 2026-05-14 15:16

## Current state

| Property | Value |
|----------|-------|
| Repo path | `/remote-home/lizilong/shenji_agent` |
| Branch | `main` |
| HEAD | `7ee858b7a1d9d3a725f8c3e15d7d21e86e507ea0` |
| Worktree | clean except modified scripts from prior restore task |
| Commit 7ee858b in history | Yes |
| Backup branch | `backup/before_v3a_tactical_ablation_20260514_151649` |

## Local tactical assets

| Asset | Path | Exists | Size | SHA256 | Ignored |
|-------|------|--------|------|--------|---------|
| Tactical checkpoint | `ai/weights/tactical_model.pt` | Yes | 52K | `9fc261e5...` | Yes (`.gitignore` line 16) |
| Tactical dataset | `datasets/陆战兵棋态势特征数据集/Wargame_Feature_Dataset.csv` | Yes | 391M | `824dbb0f...` | Yes (`.gitignore` line 3) |

## Prior validation state (from commit 7ee858b)

| Validation | Status |
|------------|--------|
| Checkpoint compatibility | `TACTICAL_CHECKPOINT_COMPATIBLE` |
| Dataset compatibility | `TACTICAL_DATASET_COMPATIBLE` |
| Branch diagnostic | `TACTICAL_BRANCH_LOADS_AND_PRODUCES_NONZERO_SCORES` |
| Compatibility smoke | `TACTICAL_ENABLED_COMPATIBILITY_SMOKE_PASS` |
| Tests | 74 OK |
| `tests/test_tactical_asset_restore.py` | 4 passed |
| compileall | Passed |

## Current modified/untracked files

```
 M scripts/run_tactical_branch_diagnostic.py
 M scripts/validate_tactical_checkpoint.py
 M scripts/validate_tactical_dataset.py
 M tests/test_tactical_asset_restore.py
?? goals/codex_goal_v3a_tactical_contribution_ablation.md
?? selective_tactical_restore/
```

These are expected artifacts from the prior selective restore task. No conflicts.

## Important boundary

Compatibility is not contribution. This task must determine whether tactical produces measurable action-level and/or outcome-level contribution.
