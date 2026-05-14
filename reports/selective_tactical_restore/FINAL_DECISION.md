# Final Decision

Final state: `TACTICAL_ASSETS_VALIDATED_RESUME_V3`

The selected tactical assets from the audited quarantine backup were copied without restoring the backup as a project source and without overwriting current source files. The current repository can now load the tactical checkpoint and parse the tactical dataset through current code paths.

## Evidence

- Checkpoint copied: `ai/weights/tactical_model.pt`.
- Dataset copied: `datasets/陆战兵棋态势特征数据集/Wargame_Feature_Dataset.csv`.
- Checkpoint validation: `TACTICAL_CHECKPOINT_COMPATIBLE`.
- Dataset validation: `TACTICAL_DATASET_COMPATIBLE`.
- Tactical branch diagnostic: `TACTICAL_BRANCH_LOADS_AND_PRODUCES_NONZERO_SCORES`.
- Minimal compatibility smoke: `TACTICAL_ENABLED_COMPATIBILITY_SMOKE_PASS`.
- Large files are ignored and not committed.

## V3 Resume Decision

V3 experimental rebuild may resume from an asset-compatibility standpoint.

Do not claim tactical contribution yet. The next V3 lane must run tactical-on vs tactical-off ablation and formal route/baseline/statistical gates before making contribution claims.
