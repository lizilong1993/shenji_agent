
# Paper Prewrite Package Goal — Shenji_agent

## Objective

Prepare everything needed for the next task to write the Shenji_agent manuscript.

This task is NOT the manuscript-writing task. Do not substantially rewrite `paper/main.tex` yet.

The goal is to produce a complete, honest, manuscript-ready evidence package that contains:

1. restored project state verification;
2. asset provenance and protection records;
3. executable code / validator inventory;
4. claim-to-evidence matrix;
5. result tables extracted from formal artifacts;
6. figure and table source data;
7. method description materials;
8. limitation and risk register;
9. reviewer-facing gap analysis;
10. final readiness decision for paper writing.

The next task should be able to start writing the paper from this package without rediscovering evidence.

---

## Current context

The project was previously blocked by missing assets after a Windows sync operation.

Important known history:

- `8c7067c` was a destructive Windows sync commit that deleted model and SDK map/scenario assets.
- A later restore/hard reset returned the repository to the pre-sync complete state:
  - expected current restore target: `5e414f7`
  - full hash: `5e414f794d6ac5ec31adf433e73cb46e4bb64c81`
- Required source areas were confirmed present after restore:
  - `training/`
  - `ai/`
  - `scripts/`
  - `tests/`
  - `paper/`
  - `experiments/`
- Candidate checkpoints were confirmed present:
  - `ai/weights/intention_lstm.pth`
  - `ai/weights/situation_gcn.pth`
- SDK assets were confirmed present:
  - scenarios
  - maps
  - `basic.json`
  - `cost.pickle`
  - `*see.npz`

However, remaining research blockers may still exist:

- tactical checkpoint may still be absent;
- incumbent checkpoint may still be absent;
- checkpoint naming may be inconsistent between manifest, loader, and validators;
- tactical contribution may still be zero or unvalidated;
- route empty-rate issue may still exist;
- two-window promotion success must not be claimed unless formal sidecars prove it;
- baseline/ablation evidence may still be insufficient.

This task must prepare the paper materials honestly around these facts.

---

## Hard constraints

Do not:

- fabricate results;
- hand-edit formal league artifacts;
- overwrite archived sidecars;
- claim tactical contribution unless executable artifacts prove nonzero contribution;
- claim second passing window unless formal sidecars prove it;
- claim SOTA or public leaderboard superiority;
- claim algorithmic novelty for standard LSTM / GraphSAGE alone;
- train expensive new formal experiments unless explicitly required and safe;
- commit large binary assets unless they are already tracked and intentionally part of the restored repository;
- delete or clean files destructively;
- use `git reset --hard`, `git clean`, `rsync --delete`, `robocopy /MIR`, or equivalent destructive sync commands.

Allowed:

- run tests;
- run lightweight validators;
- run smoke diagnostics;
- inspect formal artifacts;
- create tracked paper-preparation documents;
- create small JSON/CSV/Markdown evidence summaries;
- recover non-destructive validator scripts from backup branches/history if clearly useful;
- update paper support files such as `paper/source_evidence.md`, but do not rewrite the main manuscript.

---

## Output locations

Create a tracked paper-preparation package under:

```text
paper/prewrite_package/
````

Use this as the main handoff directory for the next writing task.

You may also create local reports under:

```text
reports/paper_prewrite_package/
```

But because `reports/` may be ignored by git, anything required for the next writing task must also be copied or summarized under:

```text
paper/prewrite_package/
```

Required final handoff file:

```text
paper/prewrite_package/PAPER_WRITING_READY.md
```

---

## Phase 0 — Safety and baseline verification

Run:

```bash
git status --short
git log --oneline --decorate -n 30
git rev-parse HEAD
git branch --show-current
```

Create:

```text
paper/prewrite_package/00_repository_state.md
```

Record:

* current branch;
* current HEAD hash;
* whether HEAD is exactly or descends from `5e414f7`;
* whether worktree is clean;
* whether there are untracked goal files;
* whether `reports/` is git-ignored;
* backup branch/tag availability if visible.

If current HEAD is not the restored pre-sync state or a clean descendant, do not continue blindly. Write a clear blocker and SAFE_STOP.

Create a backup branch:

```bash
git branch backup/before_paper_prewrite_package_$(date +%Y%m%d_%H%M%S)
```

Do not reset or clean.

---

## Phase 1 — Asset provenance and loss-prevention package

Create:

```text
paper/prewrite_package/01_asset_provenance.md
paper/prewrite_package/asset_manifest.json
paper/prewrite_package/asset_loss_prevention.md
```

Inventory these assets:

```text
ai/weights/
ai/weights_incumbent_rule/
land_wargame_sdk/land_wargame_train_env-*.whl
land_wargame_sdk/Data/Data/scenarios/
land_wargame_sdk/Data/Data/maps/
experiments/shenji_agent/
```

For each asset class, record:

* exists / missing / partial;
* path;
* file count;
* total size;
* representative filenames;
* whether tracked by git;
* whether ignored by git;
* whether needed for training/evaluation/paper evidence;
* whether it was previously deleted by sync commit `8c7067c`;
* whether safe backup instructions are needed.

Run and save concise outputs:

```bash
find ai/weights ai/weights_incumbent_rule -type f -printf '%p\t%s bytes\n' 2>/dev/null
find land_wargame_sdk/Data/Data/scenarios -type f -name '*.json' 2>/dev/null | wc -l
find land_wargame_sdk/Data/Data/maps -type f -name 'basic.json' 2>/dev/null | wc -l
find land_wargame_sdk/Data/Data/maps -type f -name 'cost.pickle' 2>/dev/null | wc -l
find land_wargame_sdk/Data/Data/maps -type f -name '*see.npz' 2>/dev/null | wc -l
```

Also create a loss-prevention note that explicitly warns against:

```text
rsync --delete
rsync --delete-excluded
rsync --delete-missing-args
robocopy /MIR
git reset --hard
git clean -fdx
```

If dangerous sync scripts exist, create:

```text
paper/prewrite_package/dangerous_sync_audit.md
```

Search for dangerous patterns:

```bash
grep -RInE -- '--delete|--delete-excluded|--delete-missing-args|robocopy .* /MIR|rm -rf|git reset --hard|git clean -fdx|Remove-Item' \
  . 2>/dev/null | head -200
```

Do not modify user sync scripts unless they are inside this repo and clearly safe to patch. If patching, remove destructive defaults and document the change.

---

## Phase 2 — Selective validator recovery audit

The reset to `5e414f7` may have removed useful Round-005/Round-006/Round-007 validator scripts.

Audit git history and backup branches for non-destructive validators such as:

```text
scripts/run_round005_smoke_pipeline.py
scripts/round005_tactical_diagnostic.py
scripts/round005_route_diagnostic.py
asset validators
sidecar validators
promotion evidence validators
```

Create:

```text
paper/prewrite_package/02_validator_inventory.md
```

For each validator, record:

* current path;
* exists at HEAD: yes/no;
* found in git history: yes/no;
* candidate source commit;
* whether it is safe to restore;
* whether it writes formal artifacts;
* whether it only reads and reports;
* whether it should be restored now.

Allowed restoration:

* only restore read-only or fail-closed validators;
* do not restore scripts that hand-edit formal sidecars;
* do not restore outdated SAFE_STOP conclusions as source truth;
* if restoring from history, document exact commit and reason.

If useful and safe, restore validator scripts from history with a path-specific restore only, for example:

```bash
git restore --source=<commit> -- scripts/<validator>.py
```

Do not restore whole old commits.

---

## Phase 3 — Formal artifact inventory

Create:

```text
paper/prewrite_package/03_formal_artifact_inventory.md
paper/prewrite_package/formal_artifacts_index.json
```

Inventory formal experiment artifacts under:

```text
experiments/shenji_agent/
```

Look for:

```text
summary.md
rating.json
promotion.json
league_state.json
participants.json
exploitability.json
```

For each league directory, record:

* path;
* timestamp / name;
* all required sidecars present or missing;
* candidate identity;
* incumbent identity;
* map/scenario count if available;
* win rate / rating fields if available;
* promotion result if available;
* consecutive window count if available;
* route diagnostics if available;
* tactical diagnostics if available;
* whether this artifact is suitable for manuscript evidence.

Important:

Use sidecars as source of truth, not prose summaries.

If `summary.md` conflicts with JSON sidecars, record the conflict and prefer sidecars.

---

## Phase 4 — Claim-to-evidence matrix

Create:

```text
paper/prewrite_package/04_claim_to_evidence_matrix.md
paper/prewrite_package/claim_to_evidence_matrix.json
```

Build a table with columns:

```text
Claim ID
Candidate manuscript claim
Status: supported / partially supported / unsupported / forbidden
Evidence path
Evidence type
Exact metric or observation
Known caveat
Allowed wording
Forbidden wording
```

At minimum include these claims:

1. closed-loop land-wargame SDK evaluation was implemented;
2. candidate agent uses learned intent/situation components;
3. candidate was evaluated against a rule incumbent;
4. promotion gate or league-style evaluation exists;
5. one verified promotion window exists, if supported;
6. two-window hard-stop exists, if supported;
7. tactical module contributes, if supported;
8. route planner is reliable, if supported;
9. route empty-rate limitation exists, if supported;
10. baselines/ablations exist, if supported;
11. method is novel;
12. evaluation is reproducible;
13. public SOTA is achieved;
14. manuscript is ready for top-tier submission.

Be conservative.

Examples of required classifications:

* If two-window success is not proven by sidecars, mark it forbidden.
* If tactical checkpoint is missing or tactical nonzero count is zero, mark tactical contribution unsupported or forbidden.
* If route empty-rate is high, mark route reliability unsupported and limitation supported.
* If no protocol-matched public leaderboard exists, mark public SOTA forbidden.
* If only smoke-level diagnostics exist, mark formal claims unsupported but smoke diagnostics allowed.

---

## Phase 5 — Result extraction for paper tables

Create:

```text
paper/prewrite_package/05_results_for_tables.md
paper/prewrite_package/results_tables.json
paper/prewrite_package/results_tables.csv
```

Extract only sidecar-supported metrics.

Potential paper tables:

### Table 1 — System components

Rows may include:

```text
Intent model
Situation model
Tactical scorer
Rule fallback
Route planner
Promotion gate
League evaluator
Artifact validator
```

Columns:

```text
Component
Path
Input
Output
Evidence status
Manuscript-safe claim
Limitation
```

### Table 2 — Formal league / promotion evidence

Columns:

```text
League artifact path
Candidate
Incumbent
Scenario/map scope
Win rate / rating metric
Promotion status
Consecutive windows
Evidence status
Caveat
```

### Table 3 — Failure / limitation diagnostics

Columns:

```text
Issue
Observed evidence
Artifact path
Severity
Impact on claim
Required future fix
```

Must include:

```text
tactical contribution not proven
route empty-rate issue
two-window hard-stop not proven
baseline/ablation insufficiency
single SDK limitation
limited scenario/map coverage if true
```

### Table 4 — Reproducibility assets

Columns:

```text
Asset
Path
Exists
Tracked
Needed for
Risk
Protection recommendation
```

Do not invent numbers. If a metric is not available, use `not_available` and explain.

---

## Phase 6 — Figure preparation package

Create:

```text
paper/prewrite_package/06_figure_plan.md
paper/prewrite_package/figure_data/
```

Prepare source data or figure specifications for manuscript figures.

Do not spend time making polished figures unless the data are already clean. The next writing task can render figures.

Required figure candidates:

### Figure 1 — Shenji_agent architecture

Should show:

```text
SDK environment state
legal action candidates
intent model
situation model
tactical scorer if available
rule/objective fallback
route planner
selected action
league/promotion evaluator
sidecar artifacts
```

Output:

```text
paper/prewrite_package/figure_data/fig1_architecture_spec.md
```

### Figure 2 — Evidence-gated evaluation loop

Should show:

```text
candidate generation
battle/evaluation
rating
promotion gate
sidecar validation
claim boundary
SAFE_STOP if evidence insufficient
```

Output:

```text
paper/prewrite_package/figure_data/fig2_evidence_gate_spec.md
```

### Figure 3 — Result summary

Only prepare if sidecar metrics support it.

Output:

```text
paper/prewrite_package/figure_data/fig3_result_summary_data.csv
paper/prewrite_package/figure_data/fig3_result_summary_notes.md
```

### Figure 4 — Diagnostic limitations

Should show:

```text
tactical missing/zero contribution
route empty-rate
two-window gap
baseline gap
asset provenance risk
```

Output:

```text
paper/prewrite_package/figure_data/fig4_diagnostic_limitations.md
```

---

## Phase 7 — Methods package

Create:

```text
paper/prewrite_package/07_methods_package.md
```

This should be a manuscript-ready methods scaffold, not polished prose.

Include:

1. environment and SDK setup;
2. scenario/map assets used;
3. agent architecture;
4. learned model components;
5. fallback logic;
6. route planner logic;
7. evaluation protocol;
8. promotion gate definition;
9. sidecar validation;
10. reviewer/audit process;
11. reproducibility commands;
12. known unavailable components.

Include exact code paths and command examples.

Commands should be real, tested if possible, or marked untested.

Do not claim the commands produce formal paper results unless they actually do.

---

## Phase 8 — Related-work and novelty framing notes

Create:

```text
paper/prewrite_package/08_novelty_and_positioning.md
```

The novelty framing must be conservative.

Allowed framing may include:

```text
fail-closed evidence-gated evaluation workflow
closed-loop SDK-based land-wargame agent evaluation
learned-path/action-diagnostic attribution
promotion-gate sidecar validation
reproducibility and claim-boundary discipline
```

Forbidden framing unless new evidence exists:

```text
new LSTM algorithm
new GraphSAGE algorithm
public SOTA
general land-wargame benchmark superiority
tactical module proven useful
route planner proven reliable
fully reproducible training/evaluation if required assets remain missing
```

Add a reviewer-facing novelty risk analysis:

```text
Risk: standard model components
Risk: single SDK
Risk: limited scenario set
Risk: formal promotion incomplete
Risk: missing ablations
Risk: route/tactical unresolved
```

For each risk, propose manuscript wording that is honest.

---

## Phase 9 — Reviewer response preparation

Create:

```text
paper/prewrite_package/09_reviewer_risk_register.md
```

Use five dimensions:

```text
Soundness
Clarity
Novelty
Significance
Evidence
```

For each dimension:

* current expected weakness;
* evidence available;
* evidence missing;
* manuscript strategy;
* do-not-claim list;
* required future experiment.

Base this on the known Round 3 / Round 4 review findings.

Do not run AI reviewer as a substitute for real evidence. If running reviewer, label it as a local heuristic diagnostic only.

---

## Phase 10 — Paper source evidence update

Update or create:

```text
paper/source_evidence.md
```

Also create:

```text
paper/prewrite_package/10_source_evidence_delta.md
```

`paper/source_evidence.md` should map major future manuscript claims to exact code and artifact paths.

Required mappings:

```text
agent implementation
model definitions
training entrypoints
evaluation entrypoints
promotion sidecars
candidate checkpoints
SDK maps/scenarios
route diagnostics
tactical diagnostics
baseline/ablation status
known limitations
asset provenance
```

Do not make `paper/source_evidence.md` stronger than the evidence allows.

---

## Phase 11 — Minimal validation run

Run:

```bash
python -m unittest discover -v
python -m compileall paper/scripts scripts ai tests training
```

If restored validators exist, run read-only/fail-closed validators and capture outputs.

Potential commands, only if scripts exist:

```bash
python scripts/run_round005_smoke_pipeline.py --out paper/prewrite_package/validation/smoke_preflight
python scripts/round005_tactical_diagnostic.py
python scripts/round005_route_diagnostic.py
```

Create:

```text
paper/prewrite_package/11_validation_log.md
paper/prewrite_package/validation/
```

Record:

* command;
* exit code;
* whether expected pass/fail;
* output path;
* blocker if any.

If a command fails due to missing tactical/incumbent assets, record that as a limitation, not as a crash to hide.

---

## Phase 12 — Manuscript readiness decision

Create the final handoff:

```text
paper/prewrite_package/PAPER_WRITING_READY.md
```

It must contain:

1. current repository state;
2. current evidence state;
3. what claims are safe;
4. what claims are forbidden;
5. which figures can be drawn;
6. which tables can be written;
7. which methods details are ready;
8. unresolved blockers;
9. recommended manuscript positioning;
10. exact next `/goal` command for paper writing.

Use one of these final states:

### READY_FOR_CONSERVATIVE_MANUSCRIPT_WRITING

Use only if:

* source, asset, and artifact inventories are complete;
* claim-to-evidence matrix is complete;
* figure/table plans are complete;
* methods package is complete;
* tests and compile checks pass;
* all unsupported claims are clearly marked forbidden;
* next writing task can proceed without more evidence discovery.

### PARTIAL_READY_WITH_BLOCKERS

Use if:

* most materials are prepared;
* but important evidence is missing;
* next writing task can still write a conservative manuscript or technical report with limitations.

### NOT_READY_SAFE_STOP

Use if:

* restored state is inconsistent;
* formal artifacts are unreadable;
* asset state is unclear;
* tests fail in a way that invalidates code evidence;
* claim boundaries cannot be determined.

---

## Documentation updates

Append concise entries to:

```text
paper/CHANGELOG.md
paper/REVIEW_LOG.md
paper/ACCEPTANCE_CRITERIA.md
```

The entry should say this was a prewriting evidence-package task, not a paper revision.

Do not claim manuscript improvement unless `paper/main.tex` was actually changed.

---

## Commit

At the end, if successful or safe-stopped with useful reports, commit tracked artifacts.

Suggested commit message:

```bash
git add goals/codex_goal_paper_prewrite_package.md
git add paper/prewrite_package
git add paper/source_evidence.md paper/CHANGELOG.md paper/REVIEW_LOG.md paper/ACCEPTANCE_CRITERIA.md 2>/dev/null || true
git add scripts tests ai training 2>/dev/null || true
git status --short
git commit -m "prepare manuscript prewrite evidence package"
```

Before committing, ensure large binaries are not newly added accidentally:

```bash
git diff --cached --name-only | grep -E '\.(pt|pth|pickle|npz|zip|tar|tgz|whl)$' || true
```

If large binaries appear unexpectedly, stop and explain before committing.

---

## SUCCESS criteria

SUCCESS only if all are true:

* repository baseline is verified;
* asset provenance is documented;
* dangerous sync deletion risk is documented;
* formal artifacts are inventoried;
* claim-to-evidence matrix is complete;
* paper result tables are prepared from real artifacts only;
* figure plan and source data/specs are prepared;
* methods package is prepared;
* novelty/positioning notes are prepared;
* reviewer risk register is prepared;
* source evidence mapping is updated;
* tests and compile checks pass, or failures are precisely justified;
* final `PAPER_WRITING_READY.md` exists;
* final decision is `READY_FOR_CONSERVATIVE_MANUSCRIPT_WRITING` or `PARTIAL_READY_WITH_BLOCKERS`;
* no unsupported manuscript claim is introduced;
* all important outputs needed by the next writing task are under `paper/prewrite_package/`;
* changes are committed or SAFE_STOP report is committed.

---

## SAFE_STOP criteria

SAFE_STOP if:

* current HEAD is not the restored complete project state or a clean descendant;
* worktree contains unexplained destructive sync damage;
* core directories are missing again;
* candidate checkpoints or SDK assets are missing again;
* formal artifacts cannot be parsed;
* tests fail in a way that invalidates code evidence;
* claim-to-evidence matrix cannot be completed;
* large binary commit risk cannot be resolved safely.

In SAFE_STOP, create:

```text
paper/prewrite_package/SAFE_STOP.md
```

It must state:

* what was checked;
* what failed;
* exact paths;
* whether the next task can write a conservative paper;
* what human input is needed.

Do not proceed to manuscript writing in this task.

