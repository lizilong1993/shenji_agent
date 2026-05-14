# Phase 8 — Validation Log

**Date:** 2026-05-14

## Commands and exit codes

| Command | Exit code | Result |
|---------|-----------|--------|
| `pytest tests/test_v3a_tactical_switch.py tests/test_tactical_asset_restore.py -v` | 0 | 22 passed |
| `python3 -m compileall scripts ai tests training -q` | 0 | All compiled |

## Test results

- `test_v3a_tactical_switch.py`: 15/15 passed
- `test_tactical_asset_restore.py`: 7/7 passed
- Total: 22/22 passed

## Compilation results

All scripts compiled without errors:
- `scripts/run_v3a_tactical_action_diagnostic.py`
- `scripts/run_v3a_tactical_ablation_smoke.py`
- `scripts/run_v3a_tactical_ablation_multiseed.py`
- `scripts/validate_tactical_checkpoint.py`
- `scripts/validate_tactical_dataset.py`
- `scripts/run_tactical_branch_diagnostic.py`
- `scripts/run_tactical_enabled_smoke.py`
- All `ai/` modules
- All `tests/` modules
- All `training/` modules

## Fixes applied during validation

1. **Pickle parser fix:** Added `SHORT_BINUNICODE` support to `_parse_pt_pickle_keys` in `validate_tactical_checkpoint.py`. Python 3.10 uses `SHORT_BINUNICODE` for strings under 256 characters, but the parser only checked for `BINUNICODE`.

## Known pre-existing issue (not fixed)

- `tests/test_policy_score_tuning.py` fixture fails due to `_load_intent_model` KeyError (model architecture mismatch between saved weights and current code). This is a pre-existing issue outside the scope of this task.
- `training/selfplay.py` and `training/evaluate.py` cannot load full policy for the same reason.
