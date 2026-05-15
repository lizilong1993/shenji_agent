
# V3E Baseline, Route, and Claim-Freeze Audit

## Objective

V3D completed authenticated tactical outcome ablation.

Known V3D facts:

- Commit: `0f5bd0d`
- Final decision: `RESUME_V3_WITH_TACTICAL_DIAGNOSTIC_OR_WEAK_CLAIM`
- Tactical branch invoked in all tactical-ON runs
- Tactical nonzero scores: 351
- Tactical OFF runs: 0 tactical invocations
- Route empty rate: 0.0 across 30 games
- Mean score delta: -1.27
- Confidence interval: [-50.07, 34.82]
- Cohen's d: -0.015
- Claim gate: `TACTICAL_OUTCOME_CONTRIBUTION_WEAK_OR_INCONCLUSIVE`

Therefore, tactical must not be claimed as a reliable outcome-level performance contribution.

V3E must freeze the V3 evidence boundary and decide whether the project can proceed to manuscript rewrite, or whether more baseline/route/promotion evidence is required.

This is an audit and claim-freeze round.

Do not train.
Do not run long evaluations.
Do not edit archived league artifacts by hand.
Do not fabricate baselines.
Do not strengthen tactical, route, promotion, SOTA, or top-tier claims beyond executable evidence.

## Source of truth

Read first:

- `reports/v3d_authenticated_tactical_ablation/FINAL_DECISION.md`
- `reports/v3d_authenticated_tactical_ablation/claim_gate.json`
- `reports/v3d_authenticated_tactical_ablation/statistics.json`
- `reports/v3d_authenticated_tactical_ablation/multiseed_ablation.json`
- `paper/v3_evidence_package/TACTICAL_CONTRIBUTION_STATUS.md`
- `paper/v3_evidence_package/V3_READINESS.md`
- `paper/v3_evidence_package/claim_to_evidence_matrix_v3.md`
- `paper/v3_evidence_package/results_tables_v3.csv`
- `paper/v3_evidence_package/results_tables_v3.json`
- `paper/source_evidence.md`
- archived league sidecars under `experiments/shenji_agent/`
- latest formal promotion / rating / league_state / participants sidecars
- `program.md`
- `paper/prewrite_package/PAPER_WRITING_READY.md` if present

## Hard constraints

Do not:

- claim tactical improves outcome performance;
- claim tactical improves win rate;
- claim tactical is a main contribution;
- claim route reliability is solved unless denominator and move-route evidence prove it;
- claim two-window promotion success unless sidecars prove it;
- claim public SOTA;
- claim top-tier readiness;
- hand-edit formal sidecars;
- run destructive git commands;
- reset or revert commits;
- modify model checkpoints;
- commit auth/license-sensitive files.

Allowed:

- audit existing evidence;
- run small non-destructive validators;
- run existing report-generation scripts;
- update V3 evidence package conservatively;
- create claim-freeze reports;
- add tests for claim consistency;
- add small helper scripts if needed.

## Phase 0 — Start state and safety

Run:

```bash
pwd
git status --short
git log --oneline --decorate -n 30
git branch --show-current
````

Create:

```text
reports/v3e_claim_freeze/00_start_state.md
```

Record:

* current branch;
* HEAD;
* V3D commit `0f5bd0d`;
* worktree status;
* whether reports from V3D exist;
* whether V3 evidence package exists;
* known tactical outcome conclusion.

Create backup branch:

```bash
git branch backup/before_v3e_claim_freeze_$(date +%Y%m%d_%H%M%S)
```

## Phase 1 — Tactical claim freeze

Create:

```text
reports/v3e_claim_freeze/01_tactical_claim_freeze.md
reports/v3e_claim_freeze/tactical_claim_freeze.json
```

Audit V3A, V3B, V3C, and V3D evidence.

Required classification:

```text
TACTICAL_ACTION_LEVEL_SUPPORTED
TACTICAL_OUTCOME_WEAK_OR_INCONCLUSIVE
TACTICAL_OUTCOME_NOT_SUPPORTED
TACTICAL_OUTCOME_BLOCKED
```

For current evidence, expected result should be:

```text
TACTICAL_ACTION_LEVEL_SUPPORTED
TACTICAL_OUTCOME_WEAK_OR_INCONCLUSIVE
```

Allowed claim:

```text
The tactical branch is invoked and produces nonzero diagnostic/action-level effects under authenticated replay, but paired multi-seed outcome ablation does not show a reliable performance benefit.
```

Forbidden claims:

```text
Tactical improves win rate.
Tactical improves outcome performance.
Tactical is a main performance contribution.
Tactical contribution is formally proven.
```

Update, if needed:

```text
paper/v3_evidence_package/TACTICAL_CONTRIBUTION_STATUS.md
paper/v3_evidence_package/claim_to_evidence_matrix_v3.md
paper/source_evidence.md
```

## Phase 2 — Route reliability audit

V3D reported route empty rate = 0.0 across 30 games.

This must be audited carefully.

Create:

```text
reports/v3e_claim_freeze/02_route_reliability_audit.md
reports/v3e_claim_freeze/route_reliability_audit.json
```

Required questions:

1. Was route empty rate computed over actual move attempts?
2. How many move attempts occurred?
3. How many route-planned moves occurred?
4. How many route-empty events occurred?
5. Did battle phase progress enough for route diagnostics to be meaningful?
6. Is V3D route_empty_rate=0.0 comparable to earlier high empty-route evidence?
7. Did earlier high empty-rate come from different runner, different phase, different sidecar, or different denominator?
8. Can route reliability be claimed as repaired, or only “no degradation observed in V3D tactical ablation”?

Search and inspect:

```bash
grep -RInE 'route_empty|empty_route|move_route|route_failure|no_valid_move_candidate|move_attempt' \
  reports experiments paper scripts ai tests 2>/dev/null \
  > reports/v3e_claim_freeze/route_reference_grep.txt || true
```

If existing scripts exist, run a non-destructive route diagnostic:

```bash
python scripts/round005_route_diagnostic.py \
  > reports/v3e_claim_freeze/route_diagnostic_recheck.log 2>&1 || true
```

Classify route status as exactly one:

```text
ROUTE_RELIABILITY_SUPPORTED
ROUTE_NO_DEGRADATION_OBSERVED_ONLY
ROUTE_RELIABILITY_STILL_UNPROVEN
ROUTE_RELIABILITY_BLOCKED_BY_MISSING_DENOMINATOR
```

Use `ROUTE_RELIABILITY_SUPPORTED` only if there is enough denominator evidence and authenticated replay covers meaningful move-route decisions.

Otherwise, prefer:

```text
ROUTE_NO_DEGRADATION_OBSERVED_ONLY
```

or

```text
ROUTE_RELIABILITY_STILL_UNPROVEN
```

Forbidden claim unless proven:

```text
Route reliability is fixed.
Route planner is robust.
Empty-route issue is solved.
```

Allowed conservative wording:

```text
In the authenticated V3D tactical ablation runs, no route-empty degradation was observed; however, broader route reliability remains claim-bounded unless move-route coverage and denominator evidence support a stronger statement.
```

## Phase 3 — Baseline and ablation coverage audit

Create:

```text
reports/v3e_claim_freeze/03_baseline_ablation_audit.md
reports/v3e_claim_freeze/baseline_ablation_audit.json
```

Inventory available evidence for:

* rule incumbent baseline;
* learned candidate vs rule incumbent;
* tactical on vs tactical off;
* intent-only ablation;
* situation-only ablation;
* no-learned fallback;
* objective-only baseline;
* random/legal baseline if meaningful;
* route-aware / route-disabled variants;
* promotion windows;
* exploiter / robustness evaluation;
* static training metrics, if any;
* authenticated replay smoke;
* authenticated multi-seed replay.

Search:

```bash
grep -RInE 'baseline|ablation|incumbent|candidate|promotion|rating|win_rate|win-rate|exploitability|tactical|intent-only|situation-only|objective-only|random' \
  reports experiments paper scripts tests 2>/dev/null \
  > reports/v3e_claim_freeze/baseline_reference_grep.txt || true
```

For each baseline or ablation, record:

```text
Name
Exists?
Executable?
Evidence path
Formal or smoke?
Number of games / pairs / seeds
Outcome metric
Limitations
Can support manuscript claim?
```

Classify baseline status as exactly one:

```text
BASELINE_COVERAGE_STRONG_ENOUGH_FOR_CONSERVATIVE_REPORT
BASELINE_COVERAGE_SMOKE_ONLY
BASELINE_COVERAGE_INSUFFICIENT_FOR_STRONG_PAPER
BASELINE_COVERAGE_BLOCKED
```

Expected likely result:

```text
BASELINE_COVERAGE_INSUFFICIENT_FOR_STRONG_PAPER
```

unless there is strong formal baseline evidence already present.

Allowed claim:

```text
The project includes a rule-incumbent comparison and authenticated tactical-on/off ablation, but broader baseline and component-ablation coverage remains limited.
```

Forbidden claim unless proven:

```text
The method outperforms strong baselines.
The method is SOTA.
The method is broadly validated across baseline families.
```

## Phase 4 — Promotion and two-window audit

Create:

```text
reports/v3e_claim_freeze/04_promotion_window_audit.md
reports/v3e_claim_freeze/promotion_window_audit.json
```

Inspect latest formal sidecars:

```bash
find experiments/shenji_agent -maxdepth 3 -type f \
  \( -name 'promotion.json' -o -name 'league_state.json' -o -name 'rating.json' -o -name 'participants.json' -o -name 'summary.md' \) \
  | sort > reports/v3e_claim_freeze/formal_sidecar_paths.txt
```

If possible, summarize the latest sidecars without editing them.

Required questions:

1. How many promotion windows are formally verified?
2. Is there a two-window hard-stop success?
3. Are candidate/incumbent participants distinct?
4. Are rating/promotion sidecars complete?
5. Are latest sidecars consistent with paper claims?
6. Is any manuscript/report prose overstating promotion status?

Classify:

```text
PROMOTION_SINGLE_WINDOW_ONLY
PROMOTION_TWO_WINDOW_SUPPORTED
PROMOTION_NOT_SUPPORTED
PROMOTION_STATUS_INCONSISTENT
PROMOTION_BLOCKED
```

Use `PROMOTION_TWO_WINDOW_SUPPORTED` only if formal sidecars prove two valid consecutive windows.

Forbidden unless proven:

```text
Two-window hard-stop success.
Promotion criteria fully passed.
Top-tier readiness.
```

## Phase 5 — Claim-to-evidence matrix freeze

Create:

```text
reports/v3e_claim_freeze/05_claim_matrix_freeze.md
reports/v3e_claim_freeze/claim_matrix_freeze.json
```

Audit every major claim in:

```text
paper/v3_evidence_package/claim_to_evidence_matrix_v3.md
paper/v3_evidence_package/V3_READINESS.md
paper/source_evidence.md
paper/main.tex 2>/dev/null || true
```

Each claim must be classified as exactly one:

```text
SUPPORTED
PARTIALLY_SUPPORTED
SMOKE_ONLY
DIAGNOSTIC_ONLY
WEAK_OR_INCONCLUSIVE
UNSUPPORTED
FORBIDDEN
```

Minimum claims to audit:

1. closed-loop SDK evaluation exists;
2. authenticated replay is available;
3. learned candidate uses intent/situation components;
4. tactical branch is invoked;
5. tactical changes action selection;
6. tactical improves outcome performance;
7. route empty issue is solved;
8. route degradation absent in V3D;
9. candidate beats rule incumbent;
10. promotion gate exists;
11. one promotion window exists;
12. two-window success exists;
13. public SOTA exists;
14. top-tier readiness exists;
15. baseline coverage is strong;
16. manuscript-ready evidence package exists.

Expected conservative statuses:

```text
closed-loop SDK evaluation exists: SUPPORTED
authenticated replay is available: SUPPORTED
learned candidate uses intent/situation components: SUPPORTED or PARTIALLY_SUPPORTED
tactical branch is invoked: SUPPORTED
tactical changes action selection: DIAGNOSTIC_ONLY or PARTIALLY_SUPPORTED
tactical improves outcome performance: WEAK_OR_INCONCLUSIVE or UNSUPPORTED
route empty issue is solved: UNSUPPORTED unless denominator proves otherwise
route degradation absent in V3D: SUPPORTED if metrics prove it
candidate beats rule incumbent: depends on formal sidecars
promotion gate exists: SUPPORTED
one promotion window exists: depends on sidecars
two-window success exists: FORBIDDEN unless sidecars prove it
public SOTA exists: FORBIDDEN
top-tier readiness exists: FORBIDDEN
baseline coverage is strong: likely UNSUPPORTED
manuscript-ready evidence package exists: PARTIAL_READY_WITH_BLOCKERS
```

Update:

```text
paper/v3_evidence_package/claim_to_evidence_matrix_v3.md
paper/v3_evidence_package/V3_READINESS.md
paper/source_evidence.md
```

Do not make claims stronger than the matrix.

## Phase 6 — Manuscript readiness decision

Create:

```text
reports/v3e_claim_freeze/06_manuscript_readiness_decision.md
reports/v3e_claim_freeze/manuscript_readiness_decision.json
```

Decide which path is currently appropriate:

```text
READY_FOR_CONSERVATIVE_TECHNICAL_REPORT
READY_FOR_CONSERVATIVE_SDK_EVALUATION_PAPER_WITH_LIMITATIONS
NEEDS_BASELINE_EXPANSION_BEFORE_MANUSCRIPT
NEEDS_ROUTE_RELIABILITY_EVIDENCE_BEFORE_MANUSCRIPT
NEEDS_PROMOTION_EVIDENCE_BEFORE_MANUSCRIPT
NOT_READY_FOR_MANUSCRIPT
```

Use conservative criteria.

`READY_FOR_CONSERVATIVE_TECHNICAL_REPORT` is allowed if:

* claim matrix is honest;
* tactical is bounded as diagnostic/weak;
* route/promotion/baseline limitations are explicit;
* no unsupported SOTA/top-tier/two-window claims remain.

`READY_FOR_CONSERVATIVE_SDK_EVALUATION_PAPER_WITH_LIMITATIONS` requires stronger baseline/promotion/route evidence.

`NEEDS_BASELINE_EXPANSION_BEFORE_MANUSCRIPT` should be used if the main remaining blocker is lack of stronger baseline/component ablation evidence.

`NOT_READY_FOR_MANUSCRIPT` should be used if evidence is internally inconsistent or core claims cannot be supported.

## Phase 7 — Optional small consistency tests

Add or update tests only if useful.

Suggested file:

```text
tests/test_v3e_claim_boundaries.py
```

Tests should verify:

* tactical outcome contribution is not marked as supported when V3D claim gate is weak/inconclusive;
* public SOTA is forbidden;
* two-window success is not claimed unless sidecars support it;
* route reliability is not marked solved without denominator evidence;
* V3_READINESS and claim matrix use consistent tactical status;
* source_evidence references V3D evidence.

Do not make brittle tests that depend on exact prose unless necessary.

## Phase 8 — Validation

Run:

```bash
python -m unittest discover -v
python -m compileall paper/scripts scripts ai tests training
```

If specific V3 tests exist, run them explicitly:

```bash
python -m pytest tests/test_v3e_claim_boundaries.py -q 2>/dev/null || true
```

Create:

```text
reports/v3e_claim_freeze/08_validation_log.md
```

Record:

* command;
* exit code;
* output summary;
* warnings;
* failures if any.

## Phase 9 — Final decision

Create:

```text
reports/v3e_claim_freeze/FINAL_DECISION.md
reports/v3e_claim_freeze/final_decision.json
```

Use exactly one final state:

```text
CLAIMS_FROZEN_READY_FOR_CONSERVATIVE_REPORT
```

Use if:

* tactical claim is frozen as diagnostic/weak;
* route and baseline limitations are clear;
* unsupported claims are forbidden;
* evidence package is internally consistent;
* manuscript can proceed only as conservative technical report.

```text
CLAIMS_FROZEN_NEEDS_BASELINE_EXPANSION
```

Use if:

* claims are consistent;
* but baseline/ablation evidence is insufficient for paper-level contribution.

```text
CLAIMS_FROZEN_NEEDS_ROUTE_EVIDENCE
```

Use if:

* route reliability remains a central unresolved blocker.

```text
CLAIMS_FROZEN_NEEDS_PROMOTION_EVIDENCE
```

Use if:

* promotion/two-window evidence is the central blocker.

```text
SAFE_STOP_CLAIM_EVIDENCE_INCONSISTENT
```

Use if:

* reports, sidecars, and evidence package contradict each other in a way that cannot be resolved safely.

The final decision must state the recommended next goal:

* `V3F baseline expansion`
* `V3F route reliability formal audit`
* `V3F conservative manuscript rewrite`
* `V3F promotion evidence rebuild`
* or `SAFE_STOP evidence reconciliation`

## Phase 10 — Commit

Before staging:

```bash
git status --short
git diff --name-only
git diff --cached --name-only | grep -Ei 'engine_config|license|auth|key|secret|token|\.pt$|\.pth$|\.zip$|\.tar$|\.tgz$|\.whl$' || true
```

Stage only reports, tests, scripts, and evidence text:

```bash
git add reports/v3e_claim_freeze
git add paper/v3_evidence_package paper/source_evidence.md 2>/dev/null || true
git add tests/test_v3e_claim_boundaries.py 2>/dev/null || true
git add goals/codex_goal_v3e_baseline_route_claim_freeze.md 2>/dev/null || true
```

Unstage sensitive or large files if needed:

```bash
git restore --staged '.engine_config' 2>/dev/null || true
git restore --staged '*license*' 2>/dev/null || true
git restore --staged '*secret*' 2>/dev/null || true
git restore --staged '*.pt' '*.pth' '*.zip' '*.tar' '*.tgz' '*.whl' 2>/dev/null || true
```

Commit:

```bash
git commit -m "freeze v3 claims after tactical ablation"
```

## SUCCESS criteria

SUCCESS if:

* tactical claim is frozen as diagnostic/weak or stronger only if already proven;
* route reliability is audited with denominator awareness;
* baseline/ablation coverage is audited;
* promotion/two-window status is audited;
* claim-to-evidence matrix is updated conservatively;
* manuscript readiness decision is written;
* final decision recommends the next V3 path;
* tests and compileall pass;
* no archived formal sidecars are edited by hand;
* no unsupported claim is introduced.

## SAFE_STOP criteria

SAFE_STOP if:

* V3D reports are missing or inconsistent;
* claim matrix contradicts formal sidecars and cannot be reconciled;
* promotion sidecars are unreadable;
* route metrics lack denominators and are central to current claims;
* validation fails in a way that invalidates claim-freeze consistency;
* sensitive SDK auth files would need to be committed.

If SAFE_STOP, create:

```text
reports/v3e_claim_freeze/SAFE_STOP.md
```

It must state:

* what was audited;
* exact inconsistency or blocker;
* which claims must remain forbidden;
* what human action is needed next.


