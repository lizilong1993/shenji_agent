
# Restore to Confirmed Complete Commit

## Objective

The previous audit confirmed that destructive sync commit `8c7067c` deleted required project files/assets.

The confirmed restore target is:

`5e414f7`

This commit was identified as the pre-sync parent / complete state containing:

- training source files;
- AI source files;
- tests;
- scripts;
- candidate model weights:
  - `ai/weights/intention_lstm.pth`
  - `ai/weights/situation_gcn.pth`
- SDK scenarios;
- SDK maps;
- archived experiment artifacts.

Later audit/recovery commits are not important for this task.

The goal is to restore the server project to `5e414f7` and verify that the project is materially complete again.

## Hard constraints

- You may hard reset the current branch to `5e414f7`.
- Create a backup branch and tag before reset.
- Do not train.
- Do not run long evaluation.
- Do not edit archived league artifacts manually.
- Do not fabricate missing files.
- Do not attempt to preserve later Round-005/006/007 audit commits.
- Do not claim tactical contribution or incumbent comparison.
- Do not claim promotion success.
- Only verify file completeness and basic non-destructive checks.

## Phase 1 — Pre-reset safety snapshot

Run from repo root:

```bash
cd /remote-home/lizilong/shenji_agent
pwd
git status --short
git branch --show-current
git log --oneline --decorate -n 20
````

Create a backup branch and tag:

```bash
git branch backup/before_restore_to_5e414f7_$(date +%Y%m%d_%H%M%S)
git tag backup-before-restore-to-5e414f7-$(date +%Y%m%d-%H%M%S) HEAD || true
```

Write a short local snapshot before reset if possible:

```bash
mkdir -p /remote-home/lizilong/shenji_restore_logs

{
  echo "## pwd"
  pwd
  echo
  echo "## branch"
  git branch --show-current
  echo
  echo "## status"
  git status --short
  echo
  echo "## recent log"
  git log --oneline --decorate -n 30
} > /remote-home/lizilong/shenji_restore_logs/pre_reset_snapshot_$(date +%Y%m%d_%H%M%S).txt
```

## Phase 2 — Verify restore target

Before reset, verify that `5e414f7` exists and inspect it:

```bash
git rev-parse 5e414f7
git show --stat --oneline 5e414f7
git ls-tree -r --name-only 5e414f7 | head -100
```

Check that key paths exist in `5e414f7`:

```bash
git ls-tree -r --name-only 5e414f7 training | head -50
git ls-tree -r --name-only 5e414f7 scripts | head -50
git ls-tree -r --name-only 5e414f7 ai | head -100
git ls-tree -r --name-only 5e414f7 tests | head -50
git ls-tree -r --name-only 5e414f7 land_wargame_sdk/Data/Data/scenarios | head -20
git ls-tree -r --name-only 5e414f7 land_wargame_sdk/Data/Data/maps | head -50
```

If `5e414f7` cannot be resolved, stop with HARD_BLOCK.

## Phase 3 — Hard reset to confirmed complete commit

Run:

```bash
git reset --hard 5e414f7
```

After reset:

```bash
git status --short
git log --oneline --decorate -n 10
```

Do not run `git clean -fdx` automatically.

If untracked files remain, only list them:

```bash
git clean -nd
```

Do not delete untracked files unless they clearly interfere with validation.

## Phase 4 — Verify source completeness

Run:

```bash
echo "## training files"
find training -type f 2>/dev/null | sort

echo "## scripts"
find scripts -maxdepth 2 -type f 2>/dev/null | sort

echo "## ai files"
find ai -maxdepth 3 -type f 2>/dev/null | sort

echo "## tests"
find tests -maxdepth 3 -type f 2>/dev/null | sort

echo "## paper files"
find paper -maxdepth 3 -type f 2>/dev/null | sort | head -200
```

Required source areas:

* `training/`
* `ai/`
* `scripts/`
* `tests/`
* `paper/`
* `experiments/`

If `training/*.py` is still missing after reset, write SAFE_STOP.

## Phase 5 — Verify model/checkpoint assets

Run:

```bash
echo "## ai/weights"
find ai/weights -type f -printf '%p\t%s bytes\n' 2>/dev/null | sort || true

echo "## ai/weights_incumbent_rule"
find ai/weights_incumbent_rule -type f -printf '%p\t%s bytes\n' 2>/dev/null | sort || true

echo "## all checkpoint-like files"
find . -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name '*.onnx' \) \
  -not -path './.git/*' \
  -printf '%p\t%s bytes\n' 2>/dev/null | sort || true
```

Expected candidate files:

* `ai/weights/intention_lstm.pth`
* `ai/weights/situation_gcn.pth`

Known limitation from previous audit:

* tactical checkpoint may not exist;
* incumbent checkpoints may not exist.

Do not treat missing tactical/incumbent checkpoints as reset failure if they never existed in git history.

## Phase 6 — Verify SDK data

Run:

```bash
echo "scenario_count=$(find land_wargame_sdk/Data/Data/scenarios -type f -name '*.json' 2>/dev/null | wc -l)"
echo "map_basic_count=$(find land_wargame_sdk/Data/Data/maps -type f -name 'basic.json' 2>/dev/null | wc -l)"
echo "map_cost_count=$(find land_wargame_sdk/Data/Data/maps -type f -name 'cost.pickle' 2>/dev/null | wc -l)"
echo "map_see_count=$(find land_wargame_sdk/Data/Data/maps -type f -name '*see.npz' 2>/dev/null | wc -l)"

find land_wargame_sdk/Data/Data/scenarios -type f -name '*.json' 2>/dev/null | sort | head -100
find land_wargame_sdk/Data/Data/maps -type f -name 'basic.json' 2>/dev/null | sort
find land_wargame_sdk/Data/Data/maps -type f -name 'cost.pickle' 2>/dev/null | sort
find land_wargame_sdk/Data/Data/maps -type f -name '*see.npz' 2>/dev/null | sort
```

Expected approximate counts:

* `scenario_count = 50`
* `map_basic_count = 16`
* `map_cost_count = 16`
* `map_see_count = 16`

If these are missing or much lower, write SAFE_STOP.

## Phase 7 — Non-destructive validation

Run:

```bash
python -m unittest discover -v
python -m compileall paper/scripts scripts ai tests training
```

If available, run smoke/preflight only:

```bash
python scripts/run_round005_smoke_pipeline.py --out reports/presync_restore_validation/smoke_preflight 2>&1 || true
python scripts/round005_tactical_diagnostic.py 2>&1 || true
python scripts/round005_route_diagnostic.py 2>&1 || true
```

Do not train.

Do not run long evaluation.

Do not modify archived league artifacts.

## Phase 8 — Write restore decision report

Create:

```bash
mkdir -p reports/presync_restore_validation
```

Write:

`reports/presync_restore_validation/RESTORE_DECISION.md`

The report must include:

* current HEAD after reset;
* backup branch/tag created;
* whether source directories are present;
* checkpoint inventory summary;
* SDK scenario/map counts;
* unittest result;
* compileall result;
* smoke/preflight result if available;
* remaining blockers.

Use exactly one final state:

### PRESYNC_RESTORE_COMPLETE

Use if:

* current HEAD is `5e414f7`;
* source directories are present;
* candidate checkpoints are present;
* SDK maps/scenarios are present;
* unittest and compileall pass, or only fail due to known environment/runtime limitations;
* project is materially complete again compared with the post-sync broken state.

### PRESYNC_RESTORE_PARTIAL

Use if:

* reset succeeded;
* some files returned;
* but important source/assets are still missing.

### PRESYNC_RESTORE_SAFE_STOP

Use if:

* `5e414f7` cannot be resolved;
* reset fails;
* source remains incomplete after reset;
* SDK data remains missing after reset;
* repository cannot be inspected.

## Phase 9 — Commit restore validation report only if useful

After `git reset --hard 5e414f7`, the branch is already restored.

Do not commit anything unless the validation report should be preserved.

If committing the validation report is desired:

```bash
git add reports/presync_restore_validation
git commit -m "validate restore to confirmed pre-sync complete state"
```

Otherwise leave the report uncommitted.

## SUCCESS criteria

SUCCESS if:

* backup branch/tag created;
* hard reset to `5e414f7` completed;
* source completeness checked;
* checkpoint assets checked;
* SDK maps/scenarios checked;
* unittest and compileall attempted;
* restore decision report written;
* no training run;
* no archived artifacts manually edited.

## HARD_BLOCK criteria

HARD_BLOCK if:

* `5e414f7` cannot be resolved;
* git reset fails due to repository corruption;
* git history is unavailable.

## SAFE_STOP criteria

SAFE_STOP if:

* reset succeeds but source remains incomplete;
* reset succeeds but SDK maps/scenarios remain missing;
* reset succeeds but candidate checkpoints remain missing;
* repository cannot run basic Python checks.

If SAFE_STOP, write:

`reports/presync_restore_validation/SAFE_STOP.md`

Include:

* exact failed command;
* observed missing files/assets;
* next human action.

