# V3B Phase 0 — Start State

**Date:** 2026-05-15

## Current state

| Property | Value |
|----------|-------|
| Repo path | `/remote-home/lizilong/shenji_agent` |
| Branch | `main` |
| HEAD | `4ad76e3d2af9862de7f2b00ceb8a98d2bfe88d42` |
| Worktree | clean except untracked goal file |
| Git stash | 3 stashes (pre-test pycache, pre-fresh-merge) |

### Recent commits

```
4ad76e3 add intent checkpoint fix reports and evidence updates
54fde8f fix intent checkpoint compatibility for full-agent replay
77f2c26 evaluate tactical contribution with on-off ablation
7ee858b Validate recovered tactical assets for V3 resume
0eafcbe audit Shenji_agent backup integrity before v2
```

### Backup branches

Existing backup branches:

- `backup/before_backup_integrity_audit_20260513_082959`
- `backup/before_backup_integrity_audit_20260514_034744`
- `backup/before_paper_prewrite_package_20260513_045909`
- `backup/before_restore_to_5e414f7_20260513_033829`
- `backup/before_round007_asset_restore_20260512_072421`
- `backup/before_selective_tactical_asset_restore_20260514_134407`
- `backup/before_v2_top_tier_upgrade_20260513_072051`
- `backup/before_v3a_intent_checkpoint_fix_20260514_184612`
- `backup/before_v3a_tactical_ablation_20260514_151649`

## Tactical checkpoint availability

| Asset | Path | Exists | Size |
|-------|------|--------|------|
| Tactical model | `ai/weights/tactical_model.pt` | Yes | 52K |
| Intention LSTM | `ai/weights/intention_lstm.pth` | Yes | — |
| Situation GCN | `ai/weights/situation_gcn.pth` | Yes | — |
| Backup tactical model | `external_assets_recovered/.../tactical_model.pt` | Yes | — |
| Backup tactical model (experiment) | `external_assets_recovered/.../tactical_model_from_experiment_20260327_141522.pt` | Yes | — |

## Intent checkpoint compatibility

| Checkpoint | Status | Source |
|------------|--------|--------|
| `intention_lstm.pth` | Compatible (confirmed V3A) | `ai/weights/` |
| Legacy format checkpoints | Fixed in `54fde8f` | Full-agent replay compatible |

## SDK availability

| Component | Path | Status |
|-----------|------|--------|
| SDK wheel | `land_wargame_sdk/land_wargame_train_env-4.1.0-cp310-cp310-...whl` | Present |
| SDK data | `land_wargame_sdk/Data/` | Present |
| Runner script | `land_wargame_sdk/run_offline_games.py` | Present |
| Python | `/usr/bin/python3` (3.10.12) | Available |
| PyTorch | 2.2.2+cu118 | Available |

## Replay environment status

| Check | Status |
|-------|--------|
| Python 3.10 | OK |
| PyTorch 2.2.2 | OK |
| SDK wheel installed | Not yet — needs `pip install` |
| Replay pairing infra | Not yet built |
| Game-level metrics | Not yet available |

## Current evaluation scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_v3a_tactical_ablation_smoke.py` | V3A tactical on/off smoke (synthetic) |
| `scripts/run_v3a_tactical_ablation_multiseed.py` | V3A multi-seed ablation (synthetic) |
| `scripts/run_v3a_tactical_action_diagnostic.py` | V3A action-level diagnostic |
| `scripts/run_v3a_full_agent_compatibility_smoke.py` | Full-agent compatibility smoke |
| `scripts/run_v3a_outcome_ablation_unblocked_smoke.py` | Outcome ablation unblocked smoke |
| `scripts/run_tactical_enabled_smoke.py` | Tactical-enabled smoke |
| `scripts/run_tactical_branch_diagnostic.py` | Tactical branch diagnostic |
| `scripts/validate_tactical_checkpoint.py` | Checkpoint validation |
| `scripts/validate_tactical_dataset.py` | Dataset validation |
| `land_wargame_sdk/run_offline_games.py` | SDK offline game runner |

**Note:** All existing V3A scripts operate on synthetic candidates, not formal game replay.

## Current route-empty metrics (from prior work)

Previous route-empty rate: 15641 / 23580 ≈ 0.6633 (from V3A records).
This must be re-evaluated under formal replay conditions.

## Current known blockers

| Blocker | Status |
|---------|--------|
| SDK pip install needed | Needs verification |
| Replay pairing infra | Needs building |
| Game-level metric collection | Needs building |
| Formal statistical aggregation | Needs building |
| Replay scenario availability | Needs verification |
| Route-empty measurement | Needs evaluation |

## V3A prior evidence summary

Previous phases achieved:

| Phase | Status | Note |
|-------|--------|------|
| V3A Phase 0 (Start) | Done | |
| V3A Phase 1 (Criteria) | Done | Three-level contribution framework |
| V3A Phase 2 (Action diagnostic) | Done | Synthetic, not game-based |
| V3A Phase 3 (Smoke ablation) | Done | Synthetic candidates |
| V3A Phase 4 (Multiseed) | Done | 5 seeds, synthetic, bootstrap CIs |
| V3A Phase 5 (Decision) | Done | Action-level supported, outcome-level unproven |
| V3A Validation/Tests | Done | 74 tests OK |

**Key gap:** All V3A evaluations used synthetic candidate sets, not actual game replay. V3B closes this by running formal SDK-backed paired replay.

## Worktree dirtiness

Only untracked files present:
- `goals/V3B_Formal_Tactical_Outcome_Ablation_via_SDK_Replay.goal.md`
- `selective_tactical_restore/`

No modified tracked files. No dirty assets. SAFE to proceed.
