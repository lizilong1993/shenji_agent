
# Server File Completeness Audit Goal

## Objective

The local project has been overwritten/synced onto the remote server. The purpose of this round is to audit whether the remote server project is now complete.

Current remote project path:

`/remote-home/lizilong/shenji_agent`

This is an audit-only round.

Do not train.  
Do not repair.  
Do not delete.  
Do not revert.  
Do not edit archived league artifacts.  
Do not strengthen manuscript claims.

The final output should answer:

1. Are source files complete?
2. Are model/checkpoint assets complete?
3. Are SDK maps and scenarios complete?
4. Are manifests consistent with actual files?
5. Can existing smoke/preflight validators proceed further than before?
6. What remains missing, mismatched, or externally blocked?

## Background

Previous rounds found these states:

- Round-005: source partially recovered, but fresh rebuild blocked by missing assets.
- Round-006: assets appeared unrecoverable from current server state.
- Round-007: sync commit `8c7067c` was identified as deleting assets; partial restore recovered:
  - candidate checkpoint files;
  - SDK maps;
  - SDK scenarios.
- Remaining blockers after Round-007:
  - checkpoint naming mismatch;
  - restored candidate checkpoint files were `.pth`, while some validators expected `.pt`;
  - tactical checkpoint may still be missing;
  - incumbent checkpoint may still be missing.

Now the user has overwritten/synced local project contents onto the remote server. This round must re-audit the current server state from scratch.

## Hard constraints

- Do not run training.
- Do not run long evaluations.
- Do not modify archived league sidecars.
- Do not edit old formal artifacts to make them pass.
- Do not delete files.
- Do not revert commits.
- Do not perform another sync.
- Do not copy large binaries into git.
- Do not claim tactical contribution, promotion success, second passing window, or baseline superiority.
- Only write audit reports under `reports/server_file_completeness_audit/`.
- If small helper scripts are absolutely necessary, create them under `scripts/` and document them, but prefer existing validators first.

## Phase 1 — Safety snapshot

Run from repo root:

```bash
pwd
git status --short
git log --oneline --decorate -n 30
````

Create audit directory:

```bash
mkdir -p reports/server_file_completeness_audit
```

Write:

`reports/server_file_completeness_audit/environment_snapshot.md`

Include:

* current path;
* current branch;
* current HEAD commit;
* git status;
* whether the worktree is clean;
* whether untracked files exist;
* current timestamp;
* Python version;
* basic disk visibility.

Commands to collect:

```bash
{
  echo "## pwd"
  pwd
  echo
  echo "## git status"
  git status --short
  echo
  echo "## git branch"
  git branch --show-current
  echo
  echo "## git HEAD"
  git log --oneline --decorate -n 5
  echo
  echo "## python"
  python --version || true
  echo
  echo "## disk"
  df -h . || true
} > reports/server_file_completeness_audit/environment_snapshot.txt
```

## Phase 2 — Full file inventory

Create a complete file inventory without modifying files.

```bash
find . -type f \
  -not -path './.git/*' \
  -printf '%p\t%s bytes\n' \
  | sort > reports/server_file_completeness_audit/full_file_inventory.txt
```

Create focused inventories:

```bash
find ai -type f -printf '%p\t%s bytes\n' 2>/dev/null | sort \
  > reports/server_file_completeness_audit/ai_inventory.txt || true

find training -type f -printf '%p\t%s bytes\n' 2>/dev/null | sort \
  > reports/server_file_completeness_audit/training_inventory.txt || true

find scripts -type f -printf '%p\t%s bytes\n' 2>/dev/null | sort \
  > reports/server_file_completeness_audit/scripts_inventory.txt || true

find tests -type f -printf '%p\t%s bytes\n' 2>/dev/null | sort \
  > reports/server_file_completeness_audit/tests_inventory.txt || true

find land_wargame_sdk -type f -printf '%p\t%s bytes\n' 2>/dev/null | sort \
  > reports/server_file_completeness_audit/sdk_inventory.txt || true

find experiments -type f -printf '%p\t%s bytes\n' 2>/dev/null | sort \
  > reports/server_file_completeness_audit/experiments_inventory.txt || true
```

## Phase 3 — Check required source files

Audit whether these source components exist:

### Core AI/runtime

* `ai/agent.py`
* `ai/policy.py`
* `ai/models.py`
* `ai/__init__.py`

### Training/evaluation

* `training/*.py`
* supervised training entrypoint if present
* self-play / league runner if present
* evaluation runner if present
* map utilities if present

### Scripts

* `scripts/run_round005_smoke_pipeline.py`
* `scripts/round005_tactical_diagnostic.py`
* `scripts/round005_route_diagnostic.py`
* `scripts/validate_assets.py` if present
* any promotion/evaluation/baseline scripts

### Tests

* `tests/`
* paper reviewer scoring tests
* asset/preflight tests if present

Create:

`reports/server_file_completeness_audit/source_completeness.md`

Use this table format:

| Component | Expected path/pattern | Exists? | File count | Notes |
| --------- | --------------------- | ------: | ---------: | ----- |

Commands:

```bash
{
  echo "## Core files"
  for p in ai/agent.py ai/policy.py ai/models.py ai/__init__.py; do
    [ -e "$p" ] && echo "OK $p" || echo "MISSING $p"
  done

  echo
  echo "## Training files"
  find training -maxdepth 2 -type f 2>/dev/null | sort || true

  echo
  echo "## Scripts"
  find scripts -maxdepth 2 -type f 2>/dev/null | sort || true

  echo
  echo "## Tests"
  find tests -maxdepth 2 -type f 2>/dev/null | sort || true
} > reports/server_file_completeness_audit/source_file_check.txt
```

## Phase 4 — Check model/checkpoint assets

Audit all checkpoint-like files.

Expected possible names include both old and restored variants:

Candidate checkpoints:

* `ai/weights/intent_lstm.pt`
* `ai/weights/intention_lstm.pth`
* `ai/weights/situation_gcn.pt`
* `ai/weights/situation_gcn.pth`
* `ai/weights/tactical_model.pt`
* `ai/weights/tactical_model.pth`
* `ai/weights/model_manifest.json`
* `ai/weights/situation_feature_normalizer.json`

Incumbent checkpoints:

* `ai/weights_incumbent_rule/intent_lstm.pt`
* `ai/weights_incumbent_rule/intention_lstm.pth`
* `ai/weights_incumbent_rule/situation_gcn.pt`
* `ai/weights_incumbent_rule/situation_gcn.pth`
* `ai/weights_incumbent_rule/model_manifest.json`

Run:

```bash
{
  echo "## ai/weights"
  find ai/weights -type f -printf '%p\t%s bytes\n' 2>/dev/null | sort || true
  echo
  echo "## ai/weights_incumbent_rule"
  find ai/weights_incumbent_rule -type f -printf '%p\t%s bytes\n' 2>/dev/null | sort || true
  echo
  echo "## all checkpoint-like files"
  find . -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name '*.onnx' \) \
    -not -path './.git/*' \
    -printf '%p\t%s bytes\n' 2>/dev/null | sort || true
} > reports/server_file_completeness_audit/checkpoint_inventory.txt
```

Inspect manifests:

```bash
{
  echo "## ai/weights/model_manifest.json"
  cat ai/weights/model_manifest.json 2>/dev/null || echo "MISSING"
  echo
  echo "## ai/weights_incumbent_rule/model_manifest.json"
  cat ai/weights_incumbent_rule/model_manifest.json 2>/dev/null || echo "MISSING"
} > reports/server_file_completeness_audit/model_manifests.txt
```

Create:

`reports/server_file_completeness_audit/checkpoint_completeness.md`

Classify:

| Checkpoint class           | Manifest expects | Existing actual file | Size | Loadable? | Status |
| -------------------------- | ---------------- | -------------------- | ---: | --------: | ------ |
| candidate intent/intention |                  |                      |      |           |        |
| candidate situation        |                  |                      |      |           |        |
| candidate tactical         |                  |                      |      |           |        |
| incumbent intent/intention |                  |                      |      |           |        |
| incumbent situation        |                  |                      |      |           |        |

If possible, run a safe CPU-only `torch.load` diagnostic on every `.pt` / `.pth` file and write JSON:

`reports/server_file_completeness_audit/checkpoint_loadability.json`

Do not mutate checkpoints.

## Phase 5 — Check SDK maps and scenarios

Audit SDK data presence.

Expected:

* scenario JSON files under:

  * `land_wargame_sdk/Data/Data/scenarios/*.json`
* map folders under:

  * `land_wargame_sdk/Data/Data/maps/map_*`
* per-map files:

  * `basic.json`
  * `cost.pickle`
  * `*see.npz` or visibility arrays

Run:

```bash
{
  echo "scenario_count=$(find land_wargame_sdk/Data/Data/scenarios -type f -name '*.json' 2>/dev/null | wc -l)"
  echo "map_basic_count=$(find land_wargame_sdk/Data/Data/maps -type f -name 'basic.json' 2>/dev/null | wc -l)"
  echo "map_cost_count=$(find land_wargame_sdk/Data/Data/maps -type f -name 'cost.pickle' 2>/dev/null | wc -l)"
  echo "map_see_count=$(find land_wargame_sdk/Data/Data/maps -type f -name '*see.npz' 2>/dev/null | wc -l)"
  echo
  echo "## scenarios"
  find land_wargame_sdk/Data/Data/scenarios -type f -name '*.json' 2>/dev/null | sort | head -200
  echo
  echo "## map basic"
  find land_wargame_sdk/Data/Data/maps -type f -name 'basic.json' 2>/dev/null | sort
  echo
  echo "## map cost"
  find land_wargame_sdk/Data/Data/maps -type f -name 'cost.pickle' 2>/dev/null | sort
  echo
  echo "## map see"
  find land_wargame_sdk/Data/Data/maps -type f -name '*see.npz' 2>/dev/null | sort
} > reports/server_file_completeness_audit/sdk_data_check.txt
```

Create:

`reports/server_file_completeness_audit/sdk_data_completeness.md`

It must state:

* scenario count;
* map basic count;
* cost count;
* see array count;
* whether counts match previous Round-007 recovery:

  * scenario_count expected around 50;
  * map_basic_count expected around 16;
  * map/cost/see expected consistent per map;
* missing map folders if any;
* missing per-map files if any.

## Phase 6 — Check archived experiment artifacts

Audit whether formal archived league artifacts still exist and were not overwritten.

Important directory:

`experiments/shenji_agent/league_20260501_170204/`

Expected sidecars may include:

* `summary.md`
* `rating.json`
* `promotion.json`
* `league_state.json`
* `participants.json`
* `exploitability.json`

Run:

```bash
{
  echo "## league dirs"
  find experiments/shenji_agent -maxdepth 2 -type d 2>/dev/null | sort || true
  echo
  echo "## latest known league sidecars"
  find experiments/shenji_agent/league_20260501_170204 -maxdepth 2 -type f 2>/dev/null | sort || true
} > reports/server_file_completeness_audit/experiment_artifact_check.txt
```

Create:

`reports/server_file_completeness_audit/experiment_artifact_completeness.md`

State:

* whether archived league directory exists;
* whether required sidecars exist;
* whether any sidecars appear missing;
* do not edit them.

## Phase 7 — Run non-destructive validators

Run existing validators only. Do not train.

Recommended commands:

```bash
python scripts/run_round005_smoke_pipeline.py --out reports/server_file_completeness_audit/smoke_preflight_recheck
```

Then:

```bash
python scripts/round005_tactical_diagnostic.py
python scripts/round005_route_diagnostic.py
python -m unittest discover -v
python -m compileall paper/scripts scripts ai tests training
```

Capture outputs and exit codes in:

`reports/server_file_completeness_audit/validation_results.md`

If `run_round005_smoke_pipeline.py` still blocks, inspect:

`reports/server_file_completeness_audit/smoke_preflight_recheck/smoke_preflight.json`

Classify blockers as:

* missing maps;
* missing scenarios;
* missing candidate checkpoint;
* missing incumbent checkpoint;
* missing tactical checkpoint;
* naming mismatch;
* manifest mismatch;
* SDK import/runtime failure;
* other.

Do not force validators to pass by weakening them unless explicitly instructed.

## Phase 8 — Compare against previous blockers

Read prior reports if present:

* `reports/round005_source_recovery/SAFE_STOP.md`
* `reports/round006_asset_recovery/SAFE_STOP.md`
* `reports/round007_asset_restore/ROUND007_DECISION.md`
* `reports/round007_restored_smoke/smoke_preflight.json`

Create:

`reports/server_file_completeness_audit/blocker_comparison.md`

Use table:

| Blocker              | Round-006 status | Round-007 status   | Current server status | Resolved? |
| -------------------- | ---------------- | ------------------ | --------------------- | --------- |
| candidate checkpoint | missing          | partially restored |                       |           |
| maps                 | missing          | restored           |                       |           |
| scenarios            | missing          | restored           |                       |           |
| tactical checkpoint  | missing          | missing            |                       |           |
| incumbent checkpoint | missing          | missing            |                       |           |
| naming mismatch      | present          | present            |                       |           |

## Phase 9 — Final decision report

Create:

`reports/server_file_completeness_audit/SERVER_FILE_COMPLETENESS_DECISION.md`

Use exactly one of these states:

### COMPLETE_FOR_CANDIDATE_SMOKE

Use if:

* source code exists;
* maps exist;
* scenarios exist;
* candidate intent/intention checkpoint exists and is loadable;
* candidate situation checkpoint exists and is loadable;
* candidate-only smoke or preflight can proceed past missing-file checks;
* tests and compile pass.

This does not mean promotion-ready.

### COMPLETE_FOR_PROMOTION_REBUILD

Use only if:

* everything for candidate smoke exists;
* incumbent checkpoints exist and are loadable;
* promotion preflight no longer blocks on missing files;
* tests and compile pass.

This does not mean promotion has succeeded.

### COMPLETE_FOR_TACTICAL_DIAGNOSTIC

Use only if:

* candidate smoke assets exist;
* tactical checkpoint exists and is loadable;
* tactical diagnostic can run past missing-file checks.

This does not mean tactical contribution is nonzero.

### PARTIALLY_COMPLETE

Use if:

* maps/scenarios/source exist;
* some candidate checkpoints exist;
* but promotion/tactical or naming/manifest mismatches still block validators.

### INCOMPLETE

Use if:

* major source directories are missing;
* maps or scenarios are missing;
* candidate checkpoints are missing;
* validators cannot even reach meaningful preflight.

### HARD_BLOCK

Use if:

* repository cannot be inspected;
* filesystem permission blocks file inventory;
* Python cannot run;
* git state is corrupted;
* audit cannot complete safely.

## Phase 10 — Commit audit reports only

Commit reports, not large binaries.

```bash
git add reports/server_file_completeness_audit
git commit -m "audit server file completeness after sync"
```

If the worktree includes unrelated untracked files such as `goals/`, do not commit them unless intentionally requested.

## SUCCESS criteria

SUCCESS if:

* full file inventory is written;
* source completeness is audited;
* checkpoint completeness is audited;
* SDK map/scenario completeness is audited;
* archived experiment artifacts are audited;
* non-destructive validators are run or exact blockers are recorded;
* blocker comparison is written;
* final decision report is written;
* audit reports are committed;
* no training is run;
* no archived league artifacts are edited.

## SAFE_STOP criteria

SAFE_STOP if:

* repo cannot be inspected;
* current path is not the expected project;
* filesystem permission blocks inventory;
* git status is unsafe and prevents audit;
* Python environment is unusable;
* scripts cannot be imported or executed at all.

If SAFE_STOP, write:

`reports/server_file_completeness_audit/SAFE_STOP.md`

It must include:

* exact command that failed;
* exact error;
* whether this is a filesystem, git, Python, or project-structure blocker;
* what human action is needed next.
