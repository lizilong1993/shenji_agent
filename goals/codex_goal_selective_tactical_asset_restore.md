
# Selective Tactical Asset Restore & Validation — Shenji_agent

## Objective

Pause V2/V3 top-tier evidence upgrade work.

The backup integrity audit reached:

```text
SAFE_STOP_NEED_CORRECT_BACKUP
````

but the audited backup contains important missing tactical assets:

```text
/remote-home/lizilong/shenji_agent_backup/ai/weights/tactical_model.pt
/remote-home/lizilong/shenji_agent_backup/experiments/shenji_agent/20260327_141522/tactical_model.pt
/remote-home/lizilong/shenji_agent_backup/datasets/陆战兵棋态势特征数据集/Wargame_Feature_Dataset.csv
```

The backup is NOT proven safe for full restore because it lacks `.git`, lacks current `paper/`, and is missing many current repo artifacts.

Therefore this task must:

1. treat the backup only as a selective asset source;
2. restore only tactical checkpoint and tactical dataset candidates;
3. avoid overwriting current source files;
4. validate compatibility with the current correct repository;
5. decide whether tactical evidence can now be regenerated;
6. update V3 readiness state.

Do not write the manuscript in this task.

Do not resume full V3 experiments until selected tactical assets are validated.

---

## Source of truth

Read first:

```text
reports/backup_integrity_audit/RESTORE_DECISION.md
reports/backup_integrity_audit/FINAL_COMPLETENESS_REPORT.md
reports/backup_integrity_audit/SAFE_STOP.md
reports/backup_integrity_audit/09_completion_audit.md
reports/v2_top_tier_upgrade/SAFE_STOP.md
reports/v2_top_tier_upgrade/FINAL_DECISION.md
paper/v2_evidence_package/V2_TOP_TIER_READINESS.md
paper/prewrite_package/PAPER_WRITING_READY.md
ai/
training/
scripts/
tests/
```

Known audit commit:

```text
0eafcbe audit Shenji_agent backup integrity before v2
```

Known V2 SAFE_STOP commit:

```text
33fc11f3eec0993b61494a60bd37e7e09e9f38f1
```

---

## Hard constraints

Do not:

* restore the whole backup;
* overwrite current source code from backup;
* overwrite current `paper/`;
* overwrite current `experiments/`;
* overwrite current `ai/`, `training/`, `scripts/`, or `tests/` from backup;
* hand-edit formal sidecars;
* claim tactical contribution without executable diagnostics;
* run destructive sync commands;
* commit large binary files unless explicitly intended and already acceptable under repo policy.

Forbidden destructive commands:

```text
git reset --hard
git clean -fdx
rsync --delete
rsync --delete-excluded
rsync --delete-missing-args
robocopy /MIR
rm -rf current_repo_paths
```

Allowed:

* copy selected checkpoint/dataset files from backup;
* store copied assets under an ignored external-assets directory;
* create symlinks if safe;
* copy to expected runtime paths if needed, without staging large binaries;
* compute checksums;
* inspect dataset schema;
* run fail-closed diagnostics;
* add small validator scripts/tests;
* commit reports, manifests, code, and tests.

---

## Backup paths

Audited backup:

```text
/remote-home/lizilong/shenji_agent_backup
```

Quarantine copy:

```text
/remote-home/lizilong/recovery/shenji_backup_audit/candidate_shenji_agent_backup_20260514_hardlink_041215
```

Prefer the quarantine copy for reading assets.

Candidate tactical assets:

```text
<QUARANTINE>/ai/weights/tactical_model.pt
<QUARANTINE>/experiments/shenji_agent/20260327_141522/tactical_model.pt
<QUARANTINE>/datasets/陆战兵棋态势特征数据集/Wargame_Feature_Dataset.csv
```

---

## Phase 0 — Freeze current repo

Run:

```bash
pwd
git status --short
git branch --show-current
git rev-parse HEAD
git log --oneline --decorate -n 30
```

Create backup branch:

```bash
git branch backup/before_selective_tactical_asset_restore_$(date +%Y%m%d_%H%M%S)
```

Create:

```text
reports/selective_tactical_restore/00_start_state.md
```

Record:

* current repo path;
* branch;
* HEAD;
* worktree clean/dirty;
* audit commit `0eafcbe`;
* V2 SAFE_STOP commit `33fc11f3eec0993b61494a60bd37e7e09e9f38f1`;
* current tactical blocker state.

If worktree is dirty, do not restore assets. Write SAFE_STOP unless dirty files are clearly expected and non-conflicting.

---

## Phase 1 — Candidate tactical asset inventory

Create:

```text
reports/selective_tactical_restore/01_candidate_asset_inventory.md
reports/selective_tactical_restore/candidate_asset_inventory.json
```

Inspect candidate files:

```bash
mkdir -p reports/selective_tactical_restore

QUARANTINE="/remote-home/lizilong/recovery/shenji_backup_audit/candidate_shenji_agent_backup_20260514_hardlink_041215"

for f in \
  "$QUARANTINE/ai/weights/tactical_model.pt" \
  "$QUARANTINE/experiments/shenji_agent/20260327_141522/tactical_model.pt" \
  "$QUARANTINE/datasets/陆战兵棋态势特征数据集/Wargame_Feature_Dataset.csv"
do
  echo "=== $f ==="
  if [ -f "$f" ]; then
    ls -lh "$f"
    sha256sum "$f"
    file "$f" || true
  else
    echo "MISSING"
  fi
done | tee reports/selective_tactical_restore/candidate_asset_listing.txt
```

Also inspect current repo state:

```bash
find ai/weights ai/weights_incumbent_rule -type f -printf '%p\t%s bytes\n' 2>/dev/null \
  | tee reports/selective_tactical_restore/current_weights_listing.txt

find datasets -maxdepth 4 -type f -printf '%p\t%s bytes\n' 2>/dev/null \
  | tee reports/selective_tactical_restore/current_dataset_listing.txt || true
```

The report must answer:

| Asset                          | Candidate found? | Current repo has it? | Size | SHA256 | Restore needed? |
| ------------------------------ | ---------------: | -------------------: | ---: | ------ | --------------- |
| `ai/weights/tactical_model.pt` |                  |                      |      |        |                 |
| experiment tactical checkpoint |                  |                      |      |        |                 |
| `Wargame_Feature_Dataset.csv`  |                  |                      |      |        |                 |

---

## Phase 2 — Decide restore target policy

Create:

```text
reports/selective_tactical_restore/02_restore_policy.md
```

Decide how to place assets.

Preferred policy:

```text
Do not commit large backup binaries.
Copy selected assets into an ignored local asset area and, only if required by current loader, copy or symlink to expected runtime paths.
```

Recommended local ignored area:

```text
external_assets_recovered/shenji_backup_20260514/
```

Potential runtime paths:

```text
ai/weights/tactical_model.pt
datasets/陆战兵棋态势特征数据集/Wargame_Feature_Dataset.csv
```

Before copying, check ignore/tracking behavior:

```bash
git check-ignore -v ai/weights/tactical_model.pt || true
git check-ignore -v "datasets/陆战兵棋态势特征数据集/Wargame_Feature_Dataset.csv" || true
git ls-files ai/weights/tactical_model.pt || true
git ls-files "datasets/陆战兵棋态势特征数据集/Wargame_Feature_Dataset.csv" || true
```

Record whether these files would be tracked or ignored.

If they would be staged as large new binaries, do not commit them unless intentionally approved. Commit only manifests/reports.

---

## Phase 3 — Selective asset copy

Only copy selected files. Do not overwrite existing files.

Create restore directory:

```bash
mkdir -p external_assets_recovered/shenji_backup_20260514/ai/weights
mkdir -p "external_assets_recovered/shenji_backup_20260514/datasets/陆战兵棋态势特征数据集"
```

Copy candidates with no overwrite:

```bash
QUARANTINE="/remote-home/lizilong/recovery/shenji_backup_audit/candidate_shenji_agent_backup_20260514_hardlink_041215"

cp -an "$QUARANTINE/ai/weights/tactical_model.pt" \
  external_assets_recovered/shenji_backup_20260514/ai/weights/tactical_model.pt

cp -an "$QUARANTINE/experiments/shenji_agent/20260327_141522/tactical_model.pt" \
  external_assets_recovered/shenji_backup_20260514/ai/weights/tactical_model_from_experiment_20260327_141522.pt

cp -an "$QUARANTINE/datasets/陆战兵棋态势特征数据集/Wargame_Feature_Dataset.csv" \
  "external_assets_recovered/shenji_backup_20260514/datasets/陆战兵棋态势特征数据集/Wargame_Feature_Dataset.csv"
```

Create checksums:

```bash
find external_assets_recovered/shenji_backup_20260514 -type f -print0 \
  | xargs -0 sha256sum \
  > reports/selective_tactical_restore/restored_asset_sha256.txt
```

Create:

```text
reports/selective_tactical_restore/03_selected_copy_log.md
```

Record:

* source path;
* destination path;
* whether copied or skipped;
* file size;
* checksum.

---

## Phase 4 — Runtime path connection

If current code expects `ai/weights/tactical_model.pt`, connect the restored asset carefully.

First inspect current manifest/code:

```bash
cat ai/weights/model_manifest.json 2>/dev/null | tee reports/selective_tactical_restore/current_model_manifest.txt || true

grep -RInE 'tactical_model|Wargame_Feature_Dataset|tactical.*\.pt|tactical.*\.pth|feature.*dataset' \
  ai training scripts tests paper 2>/dev/null \
  | tee reports/selective_tactical_restore/tactical_reference_grep.txt || true
```

If current code expects `ai/weights/tactical_model.pt` and the file does not exist, copy without overwrite:

```bash
cp -an external_assets_recovered/shenji_backup_20260514/ai/weights/tactical_model.pt \
  ai/weights/tactical_model.pt
```

If current code expects dataset at a specific path and it does not exist, copy without overwrite:

```bash
mkdir -p "datasets/陆战兵棋态势特征数据集"
cp -an "external_assets_recovered/shenji_backup_20260514/datasets/陆战兵棋态势特征数据集/Wargame_Feature_Dataset.csv" \
  "datasets/陆战兵棋态势特征数据集/Wargame_Feature_Dataset.csv"
```

Do not overwrite existing files.

Create:

```text
reports/selective_tactical_restore/04_runtime_connection.md
```

Record:

* whether runtime files were copied;
* whether symlinks were used;
* whether manifest needs update;
* whether code references these paths;
* whether files are tracked/ignored.

If manifest must be updated to include tactical checkpoint, update it only if the current loader expects manifest-based resolution. Otherwise document that current loader uses direct path.

---

## Phase 5 — Tactical checkpoint compatibility test

Create or update a lightweight validator:

```text
scripts/validate_tactical_checkpoint.py
```

The validator should:

1. resolve tactical checkpoint path;
2. report whether file exists;
3. report file size and checksum;
4. attempt safe PyTorch load if torch is available;
5. report top-level object type;
6. compare keys/shapes against expected tactical model class if possible;
7. fail closed with explicit reason.

Use safe loading where possible:

```python
torch.load(path, map_location="cpu", weights_only=True)
```

If the local PyTorch version does not support `weights_only=True`, document that and load only if the project trusts the backup as local user-owned data.

Create:

```text
reports/selective_tactical_restore/05_tactical_checkpoint_validation.md
reports/selective_tactical_restore/tactical_checkpoint_validation.json
```

Run:

```bash
python scripts/validate_tactical_checkpoint.py \
  --checkpoint ai/weights/tactical_model.pt \
  --out reports/selective_tactical_restore/tactical_checkpoint_validation.json
```

If the checkpoint cannot load, compare with the experiment checkpoint:

```bash
python scripts/validate_tactical_checkpoint.py \
  --checkpoint external_assets_recovered/shenji_backup_20260514/ai/weights/tactical_model_from_experiment_20260327_141522.pt \
  --out reports/selective_tactical_restore/tactical_checkpoint_validation_experiment.json
```

Final states:

```text
TACTICAL_CHECKPOINT_COMPATIBLE
TACTICAL_CHECKPOINT_PRESENT_BUT_INCOMPATIBLE
TACTICAL_CHECKPOINT_UNREADABLE
TACTICAL_CHECKPOINT_MISSING
```

---

## Phase 6 — Tactical dataset compatibility test

Create or update:

```text
scripts/validate_tactical_dataset.py
```

The validator should:

1. locate `Wargame_Feature_Dataset.csv`;
2. report row count;
3. report column count;
4. list columns;
5. identify label columns if present;
6. identify feature columns;
7. check missing values;
8. compare columns to tactical training code expectations;
9. fail closed with explicit reason.

Run:

```bash
python scripts/validate_tactical_dataset.py \
  --dataset "datasets/陆战兵棋态势特征数据集/Wargame_Feature_Dataset.csv" \
  --out reports/selective_tactical_restore/tactical_dataset_validation.json
```

Create:

```text
reports/selective_tactical_restore/06_tactical_dataset_validation.md
reports/selective_tactical_restore/tactical_dataset_validation.json
```

Final states:

```text
TACTICAL_DATASET_COMPATIBLE
TACTICAL_DATASET_PRESENT_BUT_SCHEMA_MISMATCH
TACTICAL_DATASET_UNREADABLE
TACTICAL_DATASET_MISSING
```

---

## Phase 7 — Rerun tactical diagnostics

Use existing diagnostic scripts if present. If not present, create a fail-closed diagnostic:

```text
scripts/run_tactical_branch_diagnostic.py
```

The diagnostic should report:

```text
tactical_checkpoint_resolved
tactical_checkpoint_loaded
tactical_model_class
tactical_feature_builder_available
tactical_candidate_count
tactical_reached_count
tactical_nonzero_score_count
tactical_changed_action_count if replay/evaluation is possible
reason_if_zero
```

Run:

```bash
python scripts/run_tactical_branch_diagnostic.py \
  --out reports/selective_tactical_restore/tactical_branch_diagnostic.json
```

Create:

```text
reports/selective_tactical_restore/07_tactical_branch_diagnostic.md
```

Allowed outcomes:

```text
TACTICAL_BRANCH_LOADS_AND_PRODUCES_NONZERO_SCORES
TACTICAL_BRANCH_LOADS_BUT_ZERO_SCORES
TACTICAL_BRANCH_PRESENT_BUT_NOT_REACHED
TACTICAL_BRANCH_BLOCKED_BY_FEATURE_SCHEMA
TACTICAL_BRANCH_BLOCKED_BY_CHECKPOINT_INCOMPATIBILITY
TACTICAL_BRANCH_MISSING
```

Only if the first state is achieved may later V3 tasks test tactical contribution.

Even then, do not claim contribution yet. Nonzero scores are not enough. Contribution requires tactical-on vs tactical-off ablation.

---

## Phase 8 — Minimal smoke with tactical enabled

If and only if the checkpoint and dataset validators pass, run a minimal candidate smoke with tactical enabled.

Use existing smoke scripts if present. Otherwise create:

```text
scripts/run_tactical_enabled_smoke.py
```

Output:

```text
experiments/shenji_agent/tactical_restore_smoke/
reports/selective_tactical_restore/08_tactical_enabled_smoke.md
```

Required metrics:

```text
episodes/games attempted
episodes/games completed
tactical_loaded
tactical_reached_count
tactical_nonzero_score_count
tactical_changed_action_count if available
route_empty_count
route_empty_rate
errors
```

Do not call this formal evidence.

This is only a compatibility smoke.

---

## Phase 9 — Update V3 readiness

Create:

```text
reports/selective_tactical_restore/FINAL_DECISION.md
paper/v3_evidence_package/TACTICAL_ASSET_RESTORE_STATUS.md
```

Use one final state:

## TACTICAL_ASSETS_VALIDATED_RESUME_V3

Use only if:

```text
tactical checkpoint copied
tactical dataset copied
checkpoint validates against current code
dataset validates against current training/feature expectations
tactical branch diagnostic runs
tactical branch produces nonzero scores or reaches a meaningful next diagnostic state
tests pass
compileall passes
```

## TACTICAL_ASSETS_RESTORED_BUT_NEED_RETRAIN_OR_ADAPTER

Use if:

```text
assets are present
but checkpoint is incompatible
or dataset schema does not match
or current code cannot load them directly
but there is enough information to retrain or build an adapter
```

## TACTICAL_ASSETS_PRESENT_BUT_NOT_USABLE

Use if:

```text
files exist but cannot be read
checkpoint format is incompatible
dataset is unusable
or no current code can consume them
```

## SELECTIVE_RESTORE_SAFE_STOP

Use if:

```text
copy failed
files disappeared
worktree unsafe
large binary handling ambiguous
or validation cannot run
```

---

## Phase 10 — Tests and compile

Run:

```bash
python -m unittest discover -v
python -m compileall paper/scripts scripts ai tests training
```

If validators were added, add tests for them:

```text
tests/test_tactical_asset_restore.py
```

Required test coverage:

```text
checkpoint validator detects existing tactical checkpoint
dataset validator detects dataset schema
missing checkpoint produces fail-closed error
missing dataset produces fail-closed error
```

---

## Commit policy

Do not accidentally commit large binaries.

Before staging:

```bash
git status --short
git diff --cached --name-only | grep -E '\.(pt|pth|pickle|npz|zip|tar|tgz|whl|csv)$' || true
```

Stage reports and code/tests:

```bash
git add reports/selective_tactical_restore
git add paper/v3_evidence_package/TACTICAL_ASSET_RESTORE_STATUS.md 2>/dev/null || true
git add scripts/validate_tactical_checkpoint.py scripts/validate_tactical_dataset.py scripts/run_tactical_branch_diagnostic.py scripts/run_tactical_enabled_smoke.py 2>/dev/null || true
git add tests/test_tactical_asset_restore.py 2>/dev/null || true
git add goals/codex_goal_selective_tactical_asset_restore.md
```

If binary/data files appear in staging, unstage them unless intentionally approved:

```bash
git restore --staged ai/weights/tactical_model.pt 2>/dev/null || true
git restore --staged "datasets/陆战兵棋态势特征数据集/Wargame_Feature_Dataset.csv" 2>/dev/null || true
git restore --staged external_assets_recovered 2>/dev/null || true
```

Commit:

```bash
git commit -m "selectively restore and validate tactical assets"
```

If asset files are needed for local runtime but not committed, document their paths in:

```text
reports/selective_tactical_restore/LOCAL_ASSET_PATHS_NOT_COMMITTED.md
```

---

## SUCCESS criteria

SUCCESS if one of these final states is reached and committed:

```text
TACTICAL_ASSETS_VALIDATED_RESUME_V3
TACTICAL_ASSETS_RESTORED_BUT_NEED_RETRAIN_OR_ADAPTER
TACTICAL_ASSETS_PRESENT_BUT_NOT_USABLE
```

and all of the following are true:

* backup was used only as selective asset source;
* no whole-backup restore occurred;
* no current source files were overwritten by backup source files;
* tactical checkpoint candidates were copied or explicitly rejected;
* tactical dataset was copied or explicitly rejected;
* checksums were recorded;
* checkpoint validation was run;
* dataset validation was run;
* tactical branch diagnostic was run or failed closed with exact reason;
* tests and compileall pass, or failures are precisely documented;
* large binary/data commit policy was respected;
* final decision explains whether V3 can resume.

---

## SAFE_STOP criteria

SAFE_STOP if:

* worktree is unsafe;
* candidate files are missing from quarantine;
* copying would overwrite current files;
* large-binary policy is ambiguous;
* checkpoint validation cannot be implemented;
* dataset validation cannot be implemented;
* current repo cannot safely reference restored assets.

In SAFE_STOP, create:

```text
reports/selective_tactical_restore/SAFE_STOP.md
```

It must state:

* what was checked;
* what failed;
* exact files involved;
* whether current repo is preserved unchanged;
* what human action is needed.
只能当作 tactical 资产仓库。**

