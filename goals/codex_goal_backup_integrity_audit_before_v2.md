
# Backup Integrity Audit Before V2/V3 — Shenji_agent

## Objective

Pause all V2/V3 top-tier evidence upgrade work.

The user discovered that a previously uploaded `shenji_agent_backup` on the server may contain the wrong project version. A new backup may be uploaded and may contain the correct missing project assets.

This task must audit the backup safely before any merge, restore, overwrite, experiment, or manuscript work.

Main goal:

1. identify the newly uploaded backup location;
2. inspect it in isolation;
3. determine whether it contains missing Shenji_agent files;
4. compare it against the current repository;
5. produce a restore recommendation;
6. only restore files if the backup is proven valid and restoration is non-destructive.

Do not continue V2/V3 experiments until this audit is complete.

---

## Hard constraints

Do not:

- run V2/V3 experiments;
- write or rewrite the paper;
- overwrite the current `shenji_agent` repository;
- delete the current repo;
- run destructive sync commands;
- use `rsync --delete`;
- use `robocopy /MIR`;
- use `git reset --hard`;
- use `git clean -fdx`;
- blindly copy backup files into the repo;
- commit large binary files unless explicitly confirmed as intended.

Forbidden unless operating only inside a disposable quarantine directory:

```bash
rm -rf
````

Allowed:

* inspect files;
* create manifests;
* compare directories;
* compute checksums;
* copy candidate backup into an isolated recovery directory;
* run read-only `find`, `du`, `tree`, `git log`, `git status`;
* restore selected files only after validation;
* write audit reports.

---

## Known current blockers to search for

The current project still needs or may benefit from recovering:

```text
ai/weights/tactical_model.pt
ai/weights/tactical_model.pth
Wargame_Feature_Dataset.csv
ai/weights_incumbent_rule/intent_lstm.pt
ai/weights_incumbent_rule/intent_lstm.pth
ai/weights_incumbent_rule/intention_lstm.pt
ai/weights_incumbent_rule/intention_lstm.pth
ai/weights_incumbent_rule/situation_gcn.pt
ai/weights_incumbent_rule/situation_gcn.pth
land_wargame_sdk/Data/Data/scenarios/*.json
land_wargame_sdk/Data/Data/maps/map_*/basic.json
land_wargame_sdk/Data/Data/maps/map_*/cost.pickle
land_wargame_sdk/Data/Data/maps/map_*/*see.npz
experiments/shenji_agent/league_*
training/*.py
scripts/*.py
ai/*.py
tests/*.py
paper/prewrite_package/
paper/v2_evidence_package/
reports/v2_top_tier_upgrade/
```

---

## Phase 0 — Freeze current state

Run in current repo:

```bash
pwd
git status --short
git branch --show-current
git rev-parse HEAD
git log --oneline --decorate -n 30
```

Create backup marker:

```bash
git branch backup/before_backup_integrity_audit_$(date +%Y%m%d_%H%M%S)
```

Create report directory:

```bash
mkdir -p reports/backup_integrity_audit
```

Write:

```text
reports/backup_integrity_audit/00_current_repo_state.md
```

Record:

* current repo path;
* branch;
* HEAD;
* clean/dirty status;
* latest relevant commits:

  * V1 commit;
  * V2 SAFE_STOP commit;
  * current HEAD;
* current known blockers.

If worktree is dirty, do not proceed with restore. Only inspect backup and write SAFE_STOP if needed.

---

## Phase 1 — Locate candidate backup

Search likely locations:

```bash
find /remote-home/lizilong -maxdepth 5 -type d -iname '*shenji*backup*' -print 2>/dev/null
find /remote-home/lizilong -maxdepth 5 -type d -iname '*shenji_agent*' -print 2>/dev/null
find /remote-home/lizilong -maxdepth 5 -type f \( \
  -iname '*shenji*backup*.zip' -o \
  -iname '*shenji*.tar' -o \
  -iname '*shenji*.tar.gz' -o \
  -iname '*shenji*.tgz' -o \
  -iname '*shenji*.7z' \
\) -printf '%p\t%s bytes\n' 2>/dev/null
```

Save output:

```bash
find /remote-home/lizilong -maxdepth 5 -type d -iname '*shenji*backup*' -print 2>/dev/null \
  > reports/backup_integrity_audit/candidate_backup_dirs.txt

find /remote-home/lizilong -maxdepth 5 -type f \( \
  -iname '*shenji*backup*.zip' -o \
  -iname '*shenji*.tar' -o \
  -iname '*shenji*.tar.gz' -o \
  -iname '*shenji*.tgz' -o \
  -iname '*shenji*.7z' \
\) -printf '%p\t%s bytes\n' 2>/dev/null \
  > reports/backup_integrity_audit/candidate_backup_archives.txt
```

Write:

```text
reports/backup_integrity_audit/01_backup_candidates.md
```

Include:

* all candidate backup paths;
* modified time;
* size;
* whether it appears to contain a git repo;
* whether it appears to be the correct project.

If multiple backups exist, audit all plausible ones. Do not assume newest is correct.

---

## Phase 2 — Quarantine extraction / copy

Create isolated recovery area:

```bash
mkdir -p /remote-home/lizilong/recovery/shenji_backup_audit
```

For archive backup, extract into a unique directory:

```bash
mkdir -p /remote-home/lizilong/recovery/shenji_backup_audit/candidate_$(date +%Y%m%d_%H%M%S)
```

Use safe extraction only. Do not extract over the current repo.

For directory backup, do not modify it directly. Copy read-only if needed:

```bash
rsync -a --info=progress2 <BACKUP_PATH>/ /remote-home/lizilong/recovery/shenji_backup_audit/<candidate_name>/
```

Important: do not use `--delete`.

Create:

```text
reports/backup_integrity_audit/02_quarantine_location.md
```

Record:

* source backup path;
* quarantine path;
* copy/extract command used;
* whether copy succeeded.

---

## Phase 3 — Backup identity audit

Inside each candidate backup/quarantine directory, check:

```bash
pwd
find . -maxdepth 3 -type f \( -name 'program.md' -o -name 'AGENTS.md' -o -name 'main.tex' -o -name 'model_manifest.json' \) -print
find . -maxdepth 3 -type d \( -name 'ai' -o -name 'training' -o -name 'scripts' -o -name 'tests' -o -name 'paper' -o -name 'experiments' -o -name 'land_wargame_sdk' \) -print
```

If `.git` exists:

```bash
git -C <candidate_dir> status --short
git -C <candidate_dir> log --oneline --decorate -n 30
git -C <candidate_dir> branch --show-current
git -C <candidate_dir> rev-parse HEAD
```

Create:

```text
reports/backup_integrity_audit/03_backup_identity.md
```

Classify candidate as:

```text
LIKELY_CORRECT_SHENJI_AGENT
POSSIBLY_CORRECT_BUT_INCOMPLETE
WRONG_PROJECT_OR_WRONG_VERSION
UNREADABLE_BACKUP
```

Criteria for likely correct:

* contains `ai/`, `training/`, `scripts/`, `tests/`, `paper/`;
* contains `land_wargame_sdk/`;
* contains `experiments/shenji_agent/`;
* contains files mentioning Shenji_agent / land_wargame;
* git history or files match known project names and commits.

---

## Phase 4 — Missing-file search inside backup

For each candidate backup, search for missing critical assets.

Run from candidate root:

```bash
mkdir -p <CURRENT_REPO>/reports/backup_integrity_audit/candidate_searches

find . -type f \( \
  -iname 'tactical_model.pt' -o \
  -iname 'tactical_model.pth' -o \
  -iname 'Wargame_Feature_Dataset.csv' -o \
  -iname '*tactical*' -o \
  -iname '*feature*dataset*.csv' -o \
  -iname 'intent_lstm.pt' -o \
  -iname 'intent_lstm.pth' -o \
  -iname 'intention_lstm.pt' -o \
  -iname 'intention_lstm.pth' -o \
  -iname 'situation_gcn.pt' -o \
  -iname 'situation_gcn.pth' \
\) -printf '%p\t%s bytes\n' 2>/dev/null
```

Also count SDK assets:

```bash
find . -path '*/land_wargame_sdk/Data/Data/scenarios/*.json' -type f 2>/dev/null | wc -l
find . -path '*/land_wargame_sdk/Data/Data/maps/map_*/basic.json' -type f 2>/dev/null | wc -l
find . -path '*/land_wargame_sdk/Data/Data/maps/map_*/cost.pickle' -type f 2>/dev/null | wc -l
find . -path '*/land_wargame_sdk/Data/Data/maps/map_*/*see.npz' -type f 2>/dev/null | wc -l
find . -path '*/experiments/shenji_agent/league_*' -type d 2>/dev/null | wc -l
```

Create:

```text
reports/backup_integrity_audit/04_missing_file_search.md
reports/backup_integrity_audit/missing_file_search.json
```

Must explicitly answer:

| File / asset class             | Found in current repo? | Found in backup? | Backup path | Size | Restore candidate? |
| ------------------------------ | ---------------------: | ---------------: | ----------- | ---: | ------------------ |
| tactical checkpoint            |                        |                  |             |      |                    |
| tactical dataset               |                        |                  |             |      |                    |
| incumbent intent checkpoint    |                        |                  |             |      |                    |
| incumbent situation checkpoint |                        |                  |             |      |                    |
| scenarios                      |                        |                  |             |      |                    |
| maps basic.json                |                        |                  |             |      |                    |
| maps cost.pickle               |                        |                  |             |      |                    |
| see arrays                     |                        |                  |             |      |                    |
| league artifacts               |                        |                  |             |      |                    |

---

## Phase 5 — Compare current repo vs backup

From current repo, generate file manifests.

Current repo:

```bash
find . -type f \
  -not -path './.git/*' \
  -printf '%P\t%s\n' | sort \
  > reports/backup_integrity_audit/current_file_manifest.tsv
```

Backup candidate:

```bash
cd <candidate_dir>
find . -type f \
  -not -path './.git/*' \
  -printf '%P\t%s\n' | sort \
  > <CURRENT_REPO>/reports/backup_integrity_audit/backup_file_manifest.tsv
```

Compare:

```bash
comm -23 \
  reports/backup_integrity_audit/backup_file_manifest.tsv \
  reports/backup_integrity_audit/current_file_manifest.tsv \
  > reports/backup_integrity_audit/files_in_backup_not_current.tsv || true

comm -13 \
  reports/backup_integrity_audit/backup_file_manifest.tsv \
  reports/backup_integrity_audit/current_file_manifest.tsv \
  > reports/backup_integrity_audit/files_in_current_not_backup.tsv || true
```

Also compare only important paths:

```bash
grep -E '^(ai/|training/|scripts/|tests/|paper/|experiments/shenji_agent/|land_wargame_sdk/)' \
  reports/backup_integrity_audit/files_in_backup_not_current.tsv \
  > reports/backup_integrity_audit/important_files_in_backup_not_current.tsv || true
```

Create:

```text
reports/backup_integrity_audit/05_current_vs_backup_diff.md
```

Summarize:

* files present in backup but missing in current;
* files present in current but missing in backup;
* whether backup appears older/newer;
* whether backup contains critical missing assets;
* whether backup is safe to merge.

---

## Phase 6 — Checksum manifest for critical binaries

For critical binaries and datasets in backup, compute checksums:

```bash
find <candidate_dir> -type f \( \
  -iname '*.pt' -o \
  -iname '*.pth' -o \
  -iname '*.pickle' -o \
  -iname '*.npz' -o \
  -iname '*.csv' -o \
  -iname '*.whl' \
\) -printf '%p\n' 2>/dev/null \
  | while read -r f; do
      sha256sum "$f"
    done \
  > reports/backup_integrity_audit/backup_critical_sha256.txt
```

Create:

```text
reports/backup_integrity_audit/06_backup_checksums.md
```

Do not commit large binaries automatically, but do commit the checksum report.

---

## Phase 7 — Restoration decision

Create:

```text
reports/backup_integrity_audit/RESTORE_DECISION.md
reports/backup_integrity_audit/RESTORE_DECISION.json
```

Use one of these states:

## BACKUP_VALID_RESTORE_RECOMMENDED

Use only if:

* backup is clearly correct Shenji_agent;
* contains one or more currently missing critical assets;
* asset paths and sizes are plausible;
* no evidence it is a wrong project;
* restoring selected files would improve completeness.

## BACKUP_PARTIAL_REVIEW_REQUIRED

Use if:

* backup contains some useful files;
* but version mismatch or path ambiguity exists;
* human should review before restoring.

## BACKUP_WRONG_VERSION_DO_NOT_RESTORE

Use if:

* backup is wrong project/version;
* does not contain needed missing files;
* would overwrite newer valid project files.

## BACKUP_UNREADABLE_SAFE_STOP

Use if:

* archive cannot be read;
* extraction failed;
* permission denied;
* file is corrupted.

The decision must include:

* exact backup path;
* exact candidate files to restore;
* exact files that must not be restored;
* whether backup should be kept, renamed, or deleted;
* next recommended action.

---

## Phase 8 — Optional selected restore

Only perform this phase if `RESTORE_DECISION.md` says:

```text
BACKUP_VALID_RESTORE_RECOMMENDED
```

Restore selected files only.

Do not overwrite newer source files blindly.

Create a restore list:

```text
reports/backup_integrity_audit/selected_restore_paths.txt
```

For each restore candidate, use a safe command like:

```bash
mkdir -p "$(dirname <target_path>)"
cp -an <backup_file> <target_path>
```

Use `cp -an` to avoid overwriting existing files.

If overwrite is needed, create a `.before_restore` backup first and document it.

After selected restore, run:

```bash
git status --short
python -m unittest discover -v
python -m compileall paper/scripts scripts ai tests training
```

If validators exist:

```bash
python scripts/validate_v3_assets.py || true
python scripts/validate_v3_evidence.py || true
```

Create:

```text
reports/backup_integrity_audit/08_selected_restore_log.md
```

Record:

* files restored;
* files skipped;
* tests run;
* remaining blockers.

---

## Phase 9 — Final completeness report

Create:

```text
reports/backup_integrity_audit/FINAL_COMPLETENESS_REPORT.md
```

It must answer:

1. Is the backup correct?
2. Does it contain missing tactical checkpoint?
3. Does it contain tactical dataset?
4. Does it contain incumbent checkpoints?
5. Does it contain complete SDK scenarios/maps?
6. Does it contain useful experiment artifacts?
7. Does it contain useful source files newer/different from current?
8. Should anything be restored?
9. Is the current project now complete enough to resume V2/V3?
10. What is the exact next goal?

Final state must be one of:

```text
PROJECT_COMPLETENESS_CONFIRMED_RESUME_V3
PROJECT_PARTIALLY_IMPROVED_RESUME_WITH_BLOCKERS
BACKUP_INVALID_KEEP_CURRENT_PROJECT
SAFE_STOP_NEED_CORRECT_BACKUP
```

---

## Commit

Commit only audit reports and small text manifests by default.

```bash
git add reports/backup_integrity_audit goals/codex_goal_backup_integrity_audit_before_v2.md 2>/dev/null || true
git status --short
git diff --cached --name-only | grep -E '\.(pt|pth|pickle|npz|zip|tar|tgz|whl)$' || true
git commit -m "audit Shenji_agent backup integrity before v2"
```

If large binaries are staged unexpectedly, unstage them and document:

```bash
git restore --staged <large_file>
```

Only commit large binaries if they were already tracked before or explicitly intended.

---

## SUCCESS criteria

SUCCESS if all are true:

* current repo state was frozen and documented;
* candidate backups were located;
* backup was inspected in quarantine;
* missing-file search was completed;
* current-vs-backup diff was generated;
* critical checksums were generated;
* restore decision was written;
* final completeness report was written;
* no destructive restore or overwrite occurred;
* tests/compile were run if selected restore occurred;
* audit reports were committed.

SUCCESS does not require that the backup contains missing files.
SUCCESS means the backup status is known and the next action is clear.

---

## SAFE_STOP criteria

SAFE_STOP if:

* no candidate backup is visible;
* backup is corrupted;
* backup cannot be read due permissions;
* current repo is dirty and unsafe to compare;
* backup identity cannot be determined;
* restore would require destructive overwrite;
* large-binary handling is ambiguous.

In SAFE_STOP, write:

```text
reports/backup_integrity_audit/SAFE_STOP.md
```

It must state:

* what was checked;
* why audit cannot continue;
* exact human action needed;
* whether current project should be preserved unchanged.

````

