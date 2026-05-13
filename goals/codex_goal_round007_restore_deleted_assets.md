
# Round-007 Restore Assets Deleted by Sync Commit

## Objective

The missing assets from Round-005/Round-006 may not be truly unrecoverable. Git history shows that sync commit `8c7067c` deleted model and SDK map assets.

Round-007 must precisely restore deleted assets from `8c7067c^` without reverting unrelated commits.

This is an asset recovery and validation round, not a paper-polishing round.

## Known deletion commit

Deletion commit:

`8c7067cf459e4e87f0c3781fc72890cce6106f00`

Commit message:

`sync: push from Windows 2026-05-12 11:06:09`

Known deleted examples:

- `ai/weights/intention_lstm.pth`
- `ai/weights/situation_gcn.pth`
- `land_wargame_sdk/Data/Data/maps/map_*/basic.json`
- `land_wargame_sdk/Data/Data/maps/map_*/cost.pickle`
- `land_wargame_sdk/Data/Data/maps/map_*/*see.npz`

Round-005/Round-006 preflight expected possibly different checkpoint names:

- `ai/weights/intent_lstm.pt`
- `ai/weights/situation_gcn.pt`
- `ai/weights/tactical_model.pt`
- `ai/weights_incumbent_rule/intent_lstm.pt`
- `ai/weights_incumbent_rule/situation_gcn.pt`

Therefore, this round must distinguish:

1. truly missing assets;
2. assets deleted by sync and restorable;
3. naming mismatch between validator expectations and actual manifest/checkpoint names;
4. tactical checkpoint genuinely absent.

## Hard constraints

- Do not revert the whole `8c7067c` commit unless explicitly necessary.
- Do not reset history.
- Do not overwrite archived formal league artifacts.
- Do not edit old sidecars to make claims pass.
- Do not claim tactical contribution unless a fresh executable diagnostic produces nonzero tactical contribution.
- Do not claim second passing window unless newly produced by executable evaluation.
- Do not claim baseline improvement unless newly produced.
- Do not copy unrelated large binaries into git.
- Do not hide the fact that assets were deleted by Windows sync.
- Preserve all provenance in reports.

## Phase 1 — Safety snapshot

Run:

```bash
git status
git log --oneline --decorate -n 30
git branch backup/before_round007_asset_restore_$(date +%Y%m%d_%H%M%S)
````

If the worktree is dirty, document it and avoid destructive operations.

Create:

`reports/round007_asset_restore/`

## Phase 2 — Audit deleted paths from sync commit

Create deletion audit files:

```bash
mkdir -p reports/round007_asset_restore

git diff --name-status --diff-filter=D 8c7067c^ 8c7067c \
  > reports/round007_asset_restore/deleted_by_sync_8c7067c_name_status.txt

git diff --name-only --diff-filter=D 8c7067c^ 8c7067c \
  > reports/round007_asset_restore/deleted_by_sync_8c7067c_paths.txt
```

Generate a filtered recovery list:

```bash
grep -E '^(ai/weights/|ai/weights_incumbent_rule/|land_wargame_sdk/Data/Data/maps/|land_wargame_sdk/Data/Data/scenarios/)' \
  reports/round007_asset_restore/deleted_by_sync_8c7067c_paths.txt \
  > reports/round007_asset_restore/asset_paths_to_restore.txt || true
```

Write:

`reports/round007_asset_restore/deletion_audit.md`

It must summarize:

* how many files were deleted by `8c7067c`;
* how many are model/checkpoint files;
* how many are SDK map files;
* how many are SDK scenario files;
* whether scenarios were deleted in this commit;
* whether `tactical_model.pt` ever appears in history.

## Phase 3 — Restore only deleted assets from parent commit

Restore paths listed in:

`reports/round007_asset_restore/asset_paths_to_restore.txt`

Use:

```bash
while IFS= read -r p; do
  [ -n "$p" ] && git restore --source=8c7067c^ -- "$p"
done < reports/round007_asset_restore/asset_paths_to_restore.txt
```

If `git restore` is unavailable, use:

```bash
while IFS= read -r p; do
  [ -n "$p" ] && git checkout 8c7067c^ -- "$p"
done < reports/round007_asset_restore/asset_paths_to_restore.txt
```

Do not restore unrelated source files from `8c7067c^`.

## Phase 4 — Inventory restored assets

Create:

`reports/round007_asset_restore/restored_asset_manifest.md`

Include:

* restored path;
* file size;
* source commit `8c7067c^`;
* whether currently tracked;
* whether it is a checkpoint/map/scenario/other;
* whether it matches Round-005/Round-006 expected path;
* whether it is referenced by manifest or code.

Run and record output:

```bash
find ai/weights ai/weights_incumbent_rule -type f -printf '%p\t%s bytes\n' 2>/dev/null \
  > reports/round007_asset_restore/restored_weights_listing.txt || true

find land_wargame_sdk/Data/Data/maps -type f -printf '%p\t%s bytes\n' 2>/dev/null \
  > reports/round007_asset_restore/restored_maps_listing.txt || true

find land_wargame_sdk/Data/Data/scenarios -type f -printf '%p\t%s bytes\n' 2>/dev/null \
  > reports/round007_asset_restore/restored_scenarios_listing.txt || true
```

Also run:

```bash
git ls-tree -r 8c7067c^ --name-only ai/weights ai/weights_incumbent_rule \
  > reports/round007_asset_restore/weights_in_parent_commit.txt || true

git rev-list --all --objects | grep -Ei '(tactical|intent|intention|situation).*\.(pt|pth)$' \
  > reports/round007_asset_restore/checkpoint_objects_in_history.txt || true
```

## Phase 5 — Diagnose naming mismatch

Inspect manifests and code:

```bash
cat ai/weights/model_manifest.json > reports/round007_asset_restore/candidate_model_manifest.txt 2>/dev/null || true
cat ai/weights_incumbent_rule/model_manifest.json > reports/round007_asset_restore/incumbent_model_manifest.txt 2>/dev/null || true

grep -RInE 'intent_lstm|intention_lstm|situation_gcn|tactical_model|\.pt|\.pth' \
  ai training scripts tests paper 2>/dev/null \
  > reports/round007_asset_restore/checkpoint_reference_grep.txt || true
```

Write:

`reports/round007_asset_restore/naming_mismatch_diagnosis.md`

Classify each expected asset:

| Expected by validator | Actual restored file       | Manifest reference | Status                  |
| --------------------- | -------------------------- | ------------------ | ----------------------- |
| `intent_lstm.pt`      | maybe `intention_lstm.pth` | check manifest     | mismatch / missing / OK |
| `situation_gcn.pt`    | maybe `situation_gcn.pth`  | check manifest     | mismatch / missing / OK |
| `tactical_model.pt`   | check history              | check manifest     | missing / OK            |
| incumbent intent      | check history              | check manifest     | missing / OK            |
| incumbent situation   | check history              | check manifest     | missing / OK            |

If the asset exists under a different legitimate name, fix validator/config logic to use the model manifest rather than hard-coded stale filenames.

Do not rename checkpoint files unless the manifest and loader require it.

## Phase 6 — Rerun validators

Run fresh validation into new directories.

Do not overwrite Round-005 or Round-006 reports.

```bash
python scripts/run_round005_smoke_pipeline.py --out reports/round007_restored_smoke
```

Then run:

```bash
python scripts/round005_tactical_diagnostic.py
python scripts/round005_route_diagnostic.py
python -m unittest discover -v
python -m compileall paper/scripts scripts ai tests training
```

Capture outputs in:

`reports/round007_asset_restore/validation_log.md`

If commands fail, record exact exit code and blocker.

## Phase 7 — Update previous SAFE_STOP interpretation

Create:

`reports/round007_asset_restore/ROUND007_DECISION.md`

Use one of the following states:

### ASSETS_RESTORED_READY_FOR_REVALIDATION

Use only if:

* deleted model/map/scenario assets were restored;
* Round-005/006 smoke preflight no longer blocks on those restored asset classes;
* tests and compile pass;
* remaining blockers, if any, are different and precisely documented.

### PARTIAL_ASSET_RESTORE

Use if:

* some assets were restored;
* but smoke preflight still blocks on missing scenarios, tactical checkpoint, incumbent checkpoints, naming mismatch, or SDK data.

### RESTORE_FAILED_SAFE_STOP

Use if:

* recovery list is empty;
* restore fails;
* assets are absent from `8c7067c^`;
* restored files are insufficient for any validator progress.

## Phase 8 — Documentation and provenance

Append a short entry to:

* `paper/REVIEW_LOG.md`
* `paper/CHANGELOG.md`
* `paper/ACCEPTANCE_CRITERIA.md`

State that Round-006 `UNRECOVERABLE_ASSETS_SAFE_STOP` is superseded or qualified only if assets are actually restored.

Do not strengthen manuscript claims.

If assets are restored, say:

“Round-006 external-asset conclusion was based on the post-sync repository state. Round-007 traced missing assets to sync commit `8c7067c` and restored the deleted subset from `8c7067c^`. Fresh validation status is recorded in `reports/round007_asset_restore/ROUND007_DECISION.md`.”

## Commit

If restoration and reports are complete, commit with:

```bash
git add reports/round007_asset_restore
git add reports/round007_restored_smoke 2>/dev/null || true
git add ai/weights ai/weights_incumbent_rule land_wargame_sdk/Data/Data/maps land_wargame_sdk/Data/Data/scenarios 2>/dev/null || true
git add paper/REVIEW_LOG.md paper/CHANGELOG.md paper/ACCEPTANCE_CRITERIA.md 2>/dev/null || true
git commit -m "restore assets deleted by sync and revalidate blockers"
```

If large binary policy says these assets should not be in git, do not commit binaries. Instead:

* restore them into working tree;
* document they must be managed by Git LFS/DVC/external asset storage;
* commit only reports and asset contract updates.

## SUCCESS criteria

SUCCESS only if:

* deleted assets from `8c7067c` are audited;
* relevant deleted assets are restored from `8c7067c^`;
* restored asset manifest is written;
* naming mismatch is diagnosed;
* Round-005/006 smoke preflight is rerun into a new output directory;
* tactical and route diagnostics are rerun or fail with precise new blockers;
* tests and compile pass;
* no old formal league artifact is edited;
* no fresh claim is made without executable evidence;
* all changes are committed or SAFE_STOP report is committed.

## SAFE_STOP criteria

SAFE_STOP if:

* assets are not present in `8c7067c^`;
* restore fails;
* restored assets are insufficient for smoke preflight;
* missing tactical checkpoint remains unresolved;
* missing scenario dataset remains unresolved;
* SDK map data remains incomplete;
* validator/config mismatch cannot be resolved safely.

In SAFE_STOP, write:

`reports/round007_asset_restore/SAFE_STOP.md`

It must include:

* what was restored;
* what remains missing;
* whether Round-006 should be revised from “unrecoverable assets” to “assets deleted by sync, partially recoverable”;
* exact next human action.

