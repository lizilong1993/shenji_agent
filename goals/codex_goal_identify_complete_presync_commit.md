
# Identify Last Complete Pre-Sync Commit

## Objective

The user is not sure whether `8c7067c` is the correct destructive sync commit.

This task must identify the exact last complete commit before file loss.

Do not hard reset immediately.

The goal is to answer:

1. Which commit deleted or lost important files?
2. Which parent or earlier commit contains the most complete project state?
3. Does that commit include:
   - training source files;
   - validator scripts;
   - AI source files;
   - candidate checkpoints;
   - tactical checkpoint if it ever existed;
   - incumbent checkpoints if they ever existed;
   - SDK scenarios;
   - SDK maps;
   - experiment sidecars?
4. What commit should be used as the safe restore target?

## Hard constraints

- Do not run `git reset --hard` until a complete restore target is proven.
- Do not run `git clean -fdx`.
- Do not delete files.
- Do not train.
- Do not modify archived league artifacts.
- Do not fabricate missing files.
- Prefer temporary `git worktree` probes over destructive checkout/reset.
- Commit only audit reports, not restored binaries.

## Phase 1 — Snapshot current state

Run from repo root:

```bash
cd /remote-home/lizilong/shenji_agent
pwd
git status --short
git branch --show-current
git log --oneline --decorate -n 50
````

Create:

```bash
mkdir -p reports/presync_commit_identification
```

Write:

```bash
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
  git log --oneline --decorate -n 80
  echo
  echo "## reflog"
  git reflog --date=iso -n 80
} > reports/presync_commit_identification/current_state.txt
```

Create backup branch:

```bash
git branch backup/before_presync_commit_identification_$(date +%Y%m%d_%H%M%S)
```

## Phase 2 — Find deletion commits

Search for commits that deleted source/assets.

Run:

```bash
git log --all --diff-filter=D --summary -- \
  training \
  scripts/run_round005_smoke_pipeline.py \
  scripts/round005_tactical_diagnostic.py \
  scripts/round005_route_diagnostic.py \
  ai/weights \
  ai/weights_incumbent_rule \
  land_wargame_sdk/Data/Data/scenarios \
  land_wargame_sdk/Data/Data/maps \
  '*intent_lstm.pt' \
  '*intention_lstm.pth' \
  '*situation_gcn.pt' \
  '*situation_gcn.pth' \
  '*tactical_model.pt' \
  '*tactical_model.pth' \
  '*cost.pickle' \
  '*basic.json' \
  '*see.npz' \
  > reports/presync_commit_identification/deletion_commits_summary.txt
```

Also run:

```bash
git log --all --name-status -- \
  training \
  scripts/run_round005_smoke_pipeline.py \
  scripts/round005_tactical_diagnostic.py \
  scripts/round005_route_diagnostic.py \
  ai/weights \
  ai/weights_incumbent_rule \
  land_wargame_sdk/Data/Data/scenarios \
  land_wargame_sdk/Data/Data/maps \
  > reports/presync_commit_identification/relevant_path_history.txt
```

Write:

`reports/presync_commit_identification/deletion_commit_analysis.md`

Include:

| Commit | Date | Message | Deleted source? | Deleted weights? | Deleted SDK maps? | Deleted scenarios? | Candidate restore parent |
| ------ | ---- | ------: | --------------: | ---------------: | ----------------: | -----------------: | ------------------------ |

## Phase 3 — Identify candidate restore commits

Collect candidate commits:

1. Parent of each deletion commit.
2. Commits immediately before Windows sync commits.
3. Commits from reflog before sync.
4. Any known good commits from Round reports.
5. `8c7067c^` only as one candidate, not as an assumption.

Generate candidate list:

```bash
{
  echo "## Parents of deletion commits"
  awk '/^commit / {print $2}' reports/presync_commit_identification/deletion_commits_summary.txt | while read c; do
    git rev-parse "$c^" 2>/dev/null || true
  done

  echo "## Recent commits"
  git log --all --format='%H %ad %s' --date=iso -n 100

  echo "## Reflog commits"
  git reflog --format='%H %gd %ad %gs' --date=iso -n 100
} > reports/presync_commit_identification/candidate_restore_commits_raw.txt
```

Normalize a short candidate list manually or with shell commands and save:

`reports/presync_commit_identification/candidate_restore_commits.txt`

The list should include at least:

* `8c7067c^`
* parent commits of other deletion commits found
* latest commit before the local-to-remote overwrite, if visible in reflog

## Phase 4 — Probe candidate commits using temporary worktrees

Do not checkout/reset the main working tree.

Create a probe root:

```bash
mkdir -p /remote-home/lizilong/shenji_commit_probes
```

For each candidate commit, create a temporary worktree:

```bash
git worktree add --detach /remote-home/lizilong/shenji_commit_probes/probe_<short_commit> <commit>
```

In each worktree, collect completeness stats:

```bash
cd /remote-home/lizilong/shenji_commit_probes/probe_<short_commit>

{
  echo "commit=<commit>"
  echo "short_commit=<short_commit>"
  echo
  echo "## source counts"
  echo "training_py_count=$(find training -type f -name '*.py' 2>/dev/null | wc -l)"
  echo "scripts_py_count=$(find scripts -type f -name '*.py' 2>/dev/null | wc -l)"
  echo "ai_py_count=$(find ai -type f -name '*.py' 2>/dev/null | wc -l)"
  echo "tests_py_count=$(find tests -type f -name '*.py' 2>/dev/null | wc -l)"
  echo
  echo "## key validator scripts"
  for p in \
    scripts/run_round005_smoke_pipeline.py \
    scripts/round005_tactical_diagnostic.py \
    scripts/round005_route_diagnostic.py
  do
    [ -e "$p" ] && echo "OK $p" || echo "MISSING $p"
  done
  echo
  echo "## checkpoints"
  find ai/weights ai/weights_incumbent_rule -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.json' \) -printf '%p\t%s bytes\n' 2>/dev/null | sort || true
  echo
  echo "## SDK data counts"
  echo "scenario_count=$(find land_wargame_sdk/Data/Data/scenarios -type f -name '*.json' 2>/dev/null | wc -l)"
  echo "map_basic_count=$(find land_wargame_sdk/Data/Data/maps -type f -name 'basic.json' 2>/dev/null | wc -l)"
  echo "map_cost_count=$(find land_wargame_sdk/Data/Data/maps -type f -name 'cost.pickle' 2>/dev/null | wc -l)"
  echo "map_see_count=$(find land_wargame_sdk/Data/Data/maps -type f -name '*see.npz' 2>/dev/null | wc -l)"
  echo
  echo "## experiment sidecars"
  find experiments/shenji_agent -maxdepth 3 -type f 2>/dev/null | sort | head -200 || true
} > /remote-home/lizilong/shenji_agent/reports/presync_commit_identification/probe_<short_commit>.txt
```

Optionally run non-destructive compile check in each probe if dependencies allow:

```bash
python -m compileall paper/scripts scripts ai tests training \
  > /remote-home/lizilong/shenji_agent/reports/presync_commit_identification/probe_<short_commit>_compileall.log 2>&1 || true
```

After probing, remove worktrees:

```bash
cd /remote-home/lizilong/shenji_agent
git worktree remove /remote-home/lizilong/shenji_commit_probes/probe_<short_commit> --force
```

## Phase 5 — Score candidate completeness

Create:

`reports/presync_commit_identification/candidate_completeness_matrix.md`

Use this table:

| Candidate commit | Date/message | training source | validator scripts | candidate ckpt | incumbent ckpt | tactical ckpt | scenarios | maps | compileall | Overall |
| ---------------- | ------------ | --------------: | ----------------: | -------------: | -------------: | ------------: | --------: | ---: | ---------: | ------- |

Classification:

* `COMPLETE`
* `MOST_COMPLETE`
* `PARTIAL`
* `INCOMPLETE`
* `UNKNOWN`

Rules:

### COMPLETE

Use only if the commit has:

* training source;
* key validator/evaluation scripts;
* AI source;
* SDK scenarios;
* SDK maps;
* candidate checkpoints;
* no obvious source deletion;
* compileall passes or only fails on environment dependency.

### MOST_COMPLETE

Use if no commit is fully complete, but one commit clearly contains the maximum recoverable source/assets.

### PARTIAL

Use if only some sources/assets exist.

## Phase 6 — Decide recommended restore target

Create:

`reports/presync_commit_identification/RESTORE_TARGET_DECISION.md`

Use exactly one state:

### RESTORE_TARGET_CONFIRMED

Use if one commit is clearly the last complete or most complete pre-loss state.

Must include:

* recommended commit hash;
* why it is recommended;
* whether it is `8c7067c^` or another commit;
* what files/assets it contains;
* what remains missing, if anything;
* exact command for user-approved reset:

```bash
git branch backup/before_restore_to_confirmed_target_$(date +%Y%m%d_%H%M%S)
git reset --hard <recommended_commit>
```

Do not execute reset unless explicitly authorized by the user or by a separate goal.

### MULTIPLE_CANDIDATES_NEED_HUMAN_CHOICE

Use if two or more commits look plausible.

Must include comparison and recommendation.

### NO_COMPLETE_COMMIT_FOUND

Use if no commit contains complete source/assets.

Must state what is missing from all history.

### HARD_BLOCK

Use if git history/reflog cannot be inspected.

## Phase 7 — Optional restore only if proven and allowed

This goal is identification-first.

Do not hard reset unless all are true:

1. `RESTORE_TARGET_CONFIRMED` is written.
2. The target is clearly complete or most complete.
3. The goal runner is explicitly allowed to restore after confirmation.

If restoration is allowed within this run, perform:

```bash
git branch backup/before_restore_to_confirmed_target_$(date +%Y%m%d_%H%M%S)
git reset --hard <recommended_commit>
```

Then run:

```bash
git status --short
python -m unittest discover -v || true
python -m compileall paper/scripts scripts ai tests training || true
```

Write:

`reports/presync_commit_identification/POST_RESTORE_VALIDATION.md`

## Phase 8 — Commit audit reports

Commit audit reports only:

```bash
git add reports/presync_commit_identification
git commit -m "identify complete pre-sync restore target"
```

If the branch was reset to an earlier commit, do not force commit unless the report is intentionally being added on top.

## SUCCESS criteria

SUCCESS if:

* deletion commits are audited;
* candidate restore commits are identified;
* temporary worktree probes are used;
* candidate completeness matrix is written;
* recommended restore target decision is written;
* no destructive reset is performed unless target is proven and allowed;
* no training is run;
* no archived artifacts are manually edited.

## SAFE_STOP criteria

SAFE_STOP if:

* git history is unavailable;
* deletion commits cannot be inspected;
* worktree probes cannot be created;
* no commit contains enough source/assets to be useful;
* the repository state is too inconsistent to safely identify a restore target.

If SAFE_STOP, write:

`reports/presync_commit_identification/SAFE_STOP.md`

Include exact blocker and next human action.

