
# V3A Tactical Contribution Ablation — Shenji_agent

## Objective

Validate whether the recovered tactical assets can support a real tactical contribution claim.

The previous selective restore task reached:

```text
TACTICAL_ASSETS_VALIDATED_RESUME_V3
````

Commit:

```text
7ee858b Validate recovered tactical assets for V3 resume
```

Known validated facts:

* `ai/weights/tactical_model.pt` exists as an ignored local asset.
* `datasets/陆战兵棋态势特征数据集/Wargame_Feature_Dataset.csv` exists as an ignored local asset.
* checkpoint validation: `TACTICAL_CHECKPOINT_COMPATIBLE`.
* dataset validation: `TACTICAL_DATASET_COMPATIBLE`.
* tactical branch diagnostic: `TACTICAL_BRANCH_LOADS_AND_PRODUCES_NONZERO_SCORES`.
* compatibility smoke: `TACTICAL_ENABLED_COMPATIBILITY_SMOKE_PASS`.
* tests passed: 74 OK.
* `tests/test_tactical_asset_restore.py`: 4 passed.
* compileall passed.

Important boundary:

Compatibility is not contribution.

This task must determine whether tactical produces measurable action-level and/or outcome-level contribution.

Do not write the final manuscript in this task.

---

## Source of truth

Read first:

```text
reports/selective_tactical_restore/FINAL_DECISION.md
reports/selective_tactical_restore/09_completion_audit.md
paper/v3_evidence_package/TACTICAL_ASSET_RESTORE_STATUS.md
reports/v2_top_tier_upgrade/SAFE_STOP.md
paper/v2_evidence_package/V2_TOP_TIER_READINESS.md
paper/prewrite_package/PAPER_WRITING_READY.md
ai/
training/
scripts/
tests/
experiments/shenji_agent/
```

---

## Hard constraints

Do not:

* fabricate tactical contribution;
* hand-edit formal sidecars;
* overwrite archived experiments;
* claim win-rate improvement from compatibility smoke alone;
* claim tactical contribution from nonzero score alone;
* commit large binary tactical assets unless explicitly intended;
* restore whole backup;
* modify paper claims before evidence is generated;
* run destructive sync or cleanup commands.

Forbidden destructive commands:

```text
git reset --hard
git clean -fdx
rsync --delete
rsync --delete-excluded
rsync --delete-missing-args
robocopy /MIR
rm -rf current project paths
```

Allowed:

* add tactical-on/off config flags;
* add ablation scripts;
* add diagnostics;
* run smoke evaluation;
* run multi-seed tactical-on/off evaluation if feasible;
* update V3 evidence package;
* update claim matrix conservatively.

---

## Phase 0 — Freeze current state and verify local assets

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
git branch backup/before_v3a_tactical_ablation_$(date +%Y%m%d_%H%M%S)
```

Create:

```text
reports/v3a_tactical_ablation/00_start_state.md
```

Record:

* current branch;
* HEAD;
* whether commit `7ee858b` is in history;
* worktree status;
* local tactical checkpoint path;
* local tactical dataset path;
* whether these assets are ignored or tracked.

Verify local assets:

```bash
ls -lh ai/weights/tactical_model.pt
ls -lh "datasets/陆战兵棋态势特征数据集/Wargame_Feature_Dataset.csv"
sha256sum ai/weights/tactical_model.pt
sha256sum "datasets/陆战兵棋态势特征数据集/Wargame_Feature_Dataset.csv"
git check-ignore -v ai/weights/tactical_model.pt || true
git check-ignore -v "datasets/陆战兵棋态势特征数据集/Wargame_Feature_Dataset.csv" || true
```

If local tactical assets are missing, SAFE_STOP and tell the user to restore them from the audited backup again.

---

## Phase 1 — Define tactical contribution criteria

Create:

```text
reports/v3a_tactical_ablation/01_tactical_contribution_criteria.md
reports/v3a_tactical_ablation/tactical_contribution_criteria.json
```

Define three levels of evidence:

## Level 1 — Compatibility only

Supported if:

```text
checkpoint loads
dataset validates
tactical branch produces nonzero scores
compatibility smoke passes
```

This is already achieved.

Allowed claim:

```text
The tactical branch can be loaded and produces nonzero scores under compatibility diagnostics.
```

Forbidden claim:

```text
The tactical branch improves agent performance.
```

## Level 2 — Action-level contribution

Supported only if tactical-on vs tactical-off replay/evaluation shows:

```text
tactical_reached_count > 0
tactical_nonzero_score_count > 0
tactical_changed_action_count > 0
changed_action_rate is measured
changed actions are legal
route empty-rate does not worsen substantially
```

Allowed claim:

```text
The tactical branch measurably changes selected actions in a bounded diagnostic setting.
```

## Level 3 — Outcome-level contribution

Supported only if tactical-on vs tactical-off evaluation shows:

```text
win rate improves, or
rating improves, or
promotion metric improves, or
diagnostic objective improves
```

and the effect is shown across multiple seeds or with a confidence interval.

Allowed claim:

```text
The tactical branch improves the evaluated metric under the tested protocol.
```

Only Level 3 can support strong tactical contribution in a paper.

---

## Phase 2 — Add tactical on/off switch

Inspect current code:

```bash
grep -RInE 'tactical|tactical_model|tactical_score|use_tactical|disable_tactical' ai training scripts tests 2>/dev/null
```

Implement or confirm a runtime switch:

```text
--tactical-mode on
--tactical-mode off
--tactical-mode missing_fail_closed
```

or equivalent config.

Requirements:

* tactical-off must use the same candidate agent except tactical scores disabled;
* tactical-on must use same seed/scenario/map protocol;
* all non-tactical settings must be identical;
* diagnostics must record tactical mode;
* fallback behavior must be identical except tactical contribution.

Create or update tests:

```text
tests/test_v3a_tactical_switch.py
```

Test:

```text
tactical-on resolves checkpoint
tactical-off does not load or use tactical score
same non-tactical config is preserved
diagnostic output records tactical_mode
```

---

## Phase 3 — Action-level tactical diagnostic

Create or update:

```text
scripts/run_v3a_tactical_action_diagnostic.py
```

Output directory:

```text
reports/v3a_tactical_ablation/action_diagnostic/
```

The diagnostic should compare tactical-on vs tactical-off on the same states/candidates where possible.

Required metrics:

```text
states_examined
candidate_actions_examined
tactical_reached_count
tactical_nonzero_score_count
tactical_changed_action_count
changed_action_rate
changed_action_types
changed_action_legality_rate
changed_action_route_empty_rate
reason_if_no_change
```

Run:

```bash
python scripts/run_v3a_tactical_action_diagnostic.py \
  --tactical-checkpoint ai/weights/tactical_model.pt \
  --out reports/v3a_tactical_ablation/action_diagnostic
```

Create:

```text
reports/v3a_tactical_ablation/02_action_level_diagnostic.md
reports/v3a_tactical_ablation/action_level_diagnostic.json
```

Final action-level states:

```text
ACTION_LEVEL_TACTICAL_CONTRIBUTION_SUPPORTED
TACTICAL_NONZERO_BUT_NO_ACTION_CHANGE
TACTICAL_BRANCH_NOT_REACHED_IN_ACTION_SELECTION
TACTICAL_ACTION_DIAGNOSTIC_BLOCKED
```

Do not proceed to strong tactical claim unless action-level contribution is supported.

---

## Phase 4 — Tactical-on/off smoke evaluation

Create or update:

```text
scripts/run_v3a_tactical_ablation_smoke.py
```

Output:

```text
experiments/shenji_agent/v3a_tactical_ablation_smoke/
reports/v3a_tactical_ablation/03_smoke_ablation.md
reports/v3a_tactical_ablation/smoke_ablation.json
```

Run paired smoke evaluation:

```bash
python scripts/run_v3a_tactical_ablation_smoke.py \
  --seeds 0 1 2 \
  --tactical-checkpoint ai/weights/tactical_model.pt \
  --out experiments/shenji_agent/v3a_tactical_ablation_smoke
```

If the script supports only one seed initially, run one seed and label it smoke.

Required paired metrics:

```text
seed
scenario/map subset
games attempted
games completed
tactical_mode
win_rate
rating if available
route_empty_count
route_empty_rate
illegal_action_count
tactical_reached_count
tactical_nonzero_score_count
tactical_changed_action_count
errors
```

Compute paired differences:

```text
win_rate_delta = tactical_on - tactical_off
route_empty_rate_delta = tactical_on - tactical_off
changed_action_rate
```

Do not call smoke results formal.

---

## Phase 5 — Multi-seed tactical ablation if smoke passes

Only if Phase 4 smoke completes without runtime blockers, run a broader but bounded evaluation.

Create or update:

```text
scripts/run_v3a_tactical_ablation_multiseed.py
```

Output:

```text
experiments/shenji_agent/v3a_tactical_ablation_multiseed/
reports/v3a_tactical_ablation/04_multiseed_ablation.md
reports/v3a_tactical_ablation/multiseed_ablation.json
```

Suggested command:

```bash
python scripts/run_v3a_tactical_ablation_multiseed.py \
  --seeds 0 1 2 \
  --tactical-checkpoint ai/weights/tactical_model.pt \
  --out experiments/shenji_agent/v3a_tactical_ablation_multiseed
```

If feasible, use 5 seeds.

Required statistics:

```text
per-seed tactical-on metrics
per-seed tactical-off metrics
mean difference
standard deviation
bootstrap CI or simple confidence interval
number of games
scenario/map coverage
failure modes
```

If full evaluation is too expensive, keep it smoke-level and state that formal contribution remains unproven.

---

## Phase 6 — Tactical contribution decision

Create:

```text
reports/v3a_tactical_ablation/TACTICAL_CONTRIBUTION_DECISION.md
reports/v3a_tactical_ablation/tactical_contribution_decision.json
paper/v3_evidence_package/TACTICAL_CONTRIBUTION_STATUS.md
```

Use one state:

## TACTICAL_OUTCOME_CONTRIBUTION_SUPPORTED

Allowed only if:

```text
tactical-on improves outcome metric over tactical-off
effect is shown across seeds or with uncertainty interval
action-level diagnostics show tactical changes actions
route/illegal action metrics do not show unacceptable degradation
```

## TACTICAL_ACTION_LEVEL_ONLY_SUPPORTED

Use if:

```text
tactical changes selected actions
but outcome improvement is absent, weak, or only smoke-level
```

Allowed paper claim:

```text
Tactical branch changes action selection under diagnostics, but outcome contribution remains limited or unproven.
```

## TACTICAL_COMPATIBLE_BUT_NO_CONTRIBUTION_SHOWN

Use if:

```text
checkpoint loads
scores are nonzero
but actions or outcomes do not change meaningfully
```

## TACTICAL_ABLATION_BLOCKED

Use if:

```text
smoke/evaluation cannot run
branch cannot be compared
SDK/runtime blocks paired evaluation
```

---

## Phase 7 — Update claim matrix and V3 readiness

Update:

```text
paper/v3_evidence_package/V3_READINESS.md
paper/v3_evidence_package/claim_to_evidence_matrix_v3.md
paper/v3_evidence_package/results_tables_v3.csv
paper/v3_evidence_package/results_tables_v3.json
paper/source_evidence.md
```

Classify tactical claim based on the decision.

Allowed claim upgrades:

* If outcome-level supported:

  * tactical contribution can become a bounded contribution claim.
* If action-level only:

  * tactical can be a diagnostic/action-selection contribution, not performance contribution.
* If no contribution:

  * tactical remains implementation detail/future work.
* If blocked:

  * tactical remains forbidden.

Do not upgrade route reliability, two-window, public SOTA, or top-tier readiness in this task.

---

## Phase 8 — Validation

Run:

```bash
python -m unittest discover -v
python -m pytest tests/test_tactical_asset_restore.py tests/test_v3a_tactical_switch.py -q
python -m compileall paper/scripts scripts ai tests training
```

If V3 validators exist:

```bash
python scripts/validate_v3_evidence.py || true
```

Create:

```text
reports/v3a_tactical_ablation/08_validation_log.md
```

Record all commands, exit codes, and outputs.

---

## Phase 9 — Final decision

Create:

```text
reports/v3a_tactical_ablation/FINAL_DECISION.md
```

Use one state:

## RESUME_V3_WITH_TACTICAL_CONTRIBUTION

Use if tactical outcome contribution is supported.

Next task should be:

```text
V3B route repair and baseline expansion
```

## RESUME_V3_WITH_TACTICAL_AS_DIAGNOSTIC_ONLY

Use if action-level contribution exists but outcome contribution is weak or unproven.

Next task should decide whether tactical remains a secondary analysis.

## RESUME_V3_WITHOUT_TACTICAL_MAIN_CLAIM

Use if tactical compatibility exists but contribution is not shown.

Next task should focus on route-aware action selection / evidence-gated policy composition.

## SAFE_STOP_TACTICAL_ABLATION_BLOCKED

Use if paired tactical-on/off evaluation cannot run.

---

## Commit policy

Do not commit tactical checkpoint or dataset unless explicitly intended.

Before staging:

```bash
git status --short
git diff --cached --name-only | grep -E '\.(pt|pth|pickle|npz|zip|tar|tgz|whl|csv)$' || true
```

Stage code/reports/tests only:

```bash
git add reports/v3a_tactical_ablation
git add paper/v3_evidence_package paper/source_evidence.md 2>/dev/null || true
git add scripts/run_v3a_tactical_action_diagnostic.py scripts/run_v3a_tactical_ablation_smoke.py scripts/run_v3a_tactical_ablation_multiseed.py 2>/dev/null || true
git add tests/test_v3a_tactical_switch.py 2>/dev/null || true
git add goals/codex_goal_v3a_tactical_contribution_ablation.md
```

Unstage large assets if accidentally staged:

```bash
git restore --staged ai/weights/tactical_model.pt 2>/dev/null || true
git restore --staged "datasets/陆战兵棋态势特征数据集/Wargame_Feature_Dataset.csv" 2>/dev/null || true
git restore --staged external_assets_recovered 2>/dev/null || true
```

Commit:

```bash
git commit -m "evaluate tactical contribution with on-off ablation"
```

---

## SUCCESS criteria

SUCCESS if:

* tactical-on/off switch exists or is confirmed;
* action-level tactical diagnostic runs;
* tactical-on/off smoke evaluation runs or fails closed with exact reason;
* multiseed ablation runs if feasible;
* tactical contribution decision is written;
* V3 tactical claim matrix is updated;
* tests pass;
* compileall passes;
* no large tactical assets are committed;
* final decision states how V3 should resume.

SUCCESS does not require tactical contribution to be positive.

SUCCESS means the tactical contribution question is answered honestly.

---

## SAFE_STOP criteria

SAFE_STOP if:

* local tactical assets disappear;
* tactical-on/off comparison cannot be implemented safely;
* SDK/runtime prevents even smoke paired evaluation;
* diagnostics cannot determine whether tactical changes actions;
* worktree or asset state is unsafe;
* large binary policy blocks required runtime use.

In SAFE_STOP, create:

```text
reports/v3a_tactical_ablation/SAFE_STOP.md
```

It must state:

* what was attempted;
* what failed;
* whether tactical should remain forbidden;
* what human input is needed.
