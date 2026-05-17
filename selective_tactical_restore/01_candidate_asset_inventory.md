# Candidate Tactical Asset Inventory

Date: 2026-05-14

## Quarantine source

```
/remote-home/lizilong/recovery/shenji_backup_audit/candidate_shenji_agent_backup_20260514_hardlink_041215
```

## Asset inventory

| Asset | Candidate found? | Current repo has it? | Size | SHA256 | Restore needed? | Status |
|---|---|---|---|---|---|---|
| `ai/weights/tactical_model.pt` | YES | YES | 53,013 bytes | `9fc261e5...` | Already in place | Identical to quarantine |
| experiment tactical checkpoint | YES | N/A (same file) | 53,013 bytes | `9fc261e5...` (same) | Not needed — identical to weights copy | Duplicate of weights copy |
| `Wargame_Feature_Dataset.csv` | YES | YES | 409,055,700 bytes | `824dbb0f...` | Already in place | Identical to quarantine |

## Notes

- Both `tactical_model.pt` and `Wargame_Feature_Dataset.csv` already exist in the current repo with checksums matching the quarantine backup.
- The experiment checkpoint at `experiments/shenji_agent/20260327_141522/tactical_model.pt` is byte-identical to `ai/weights/tactical_model.pt` (same SHA256).
- Both files are git-ignored (`ai/weights/*.pt` and `datasets/` in `.gitignore`).
- No copy from quarantine is needed — proceed directly to validation.
