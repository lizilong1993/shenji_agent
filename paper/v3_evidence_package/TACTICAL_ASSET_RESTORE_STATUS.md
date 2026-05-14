# Tactical Asset Restore Status

State: `TACTICAL_ASSETS_VALIDATED_RESUME_V3`

The tactical checkpoint and tactical dataset were selectively restored from the audited quarantine backup and validated against the current Shenji_agent code.

Evidence:

- `reports/selective_tactical_restore/tactical_checkpoint_validation.json`: checkpoint compatible with `ai.models.TacticalScorer`.
- `reports/selective_tactical_restore/tactical_dataset_validation.json`: dataset schema compatible with `training.data.TacticalPositionDataset`.
- `reports/selective_tactical_restore/tactical_branch_diagnostic.json`: tactical branch loads and produces nonzero scores.
- `reports/selective_tactical_restore/tactical_enabled_smoke.json`: compatibility smoke passed and is not formal evidence.

V3 experiments may resume only as experiments. Tactical contribution remains unclaimed until tactical-on vs tactical-off ablation proves an effect.
