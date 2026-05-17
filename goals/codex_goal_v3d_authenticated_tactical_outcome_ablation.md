
# V3D Authenticated Tactical Outcome Ablation

## Objective

V3C recovered SDK authentication in the current server environment.

Known V3C facts:

- Final state: `SDK_REPLAY_AUTH_RECOVERED`
- Commit: `729b1d0`
- `.engine_config` was rebound to current MAC `02:42:ac:11:00:07`
- Minimal 1-pair smoke replay passed authentication
- Deployment and battle phases were entered
- Tactical invocations were nonzero in smoke: tactical-on = 3, tactical-off = 0
- 87/87 tests passed
- compileall passed
- `LegacySituationModel.backbone` alias was added so policy code can access legacy/current situation models uniformly

V3B previously concluded:

- `TACTICAL_OUTCOME_CONTRIBUTION_WEAK_OR_INCONCLUSIVE`
- Main blocker was SDK MAC mismatch causing replay to remain in deployment phase
- Formal ablation infrastructure already exists:
  - `scripts/run_v3b_formal_tactical_ablation.py`
  - `scripts/v3b_pairing.py`
  - `scripts/v3b_metrics.py`

V3D must rerun formal tactical-on/off outcome ablation under the authenticated environment.

Do not train.
Do not hand-edit archived league artifacts.
Do not claim tactical outcome contribution unless paired multi-seed evidence supports it.
Do not claim promotion, two-window success, route reliability, public SOTA, or top-tier readiness.

## Phase 0 — Safety and commit hygiene

Run:

```bash
pwd
git status --short
git log --oneline --decorate -n 30
git show --name-only --stat 729b1d0
git show --name-only 729b1d0 | grep -Ei 'engine_config|license|auth|key|secret|token|mac' || true
````

Create:

```text
reports/v3d_authenticated_tactical_ablation/00_start_state.md
```

Record:

* current branch;
* HEAD;
* V3C commit `729b1d0`;
* worktree status;
* whether auth/license-sensitive files were committed;
* SDK auth state;
* current MAC;
* V3B prior blocker;
* V3C smoke result.

Create backup branch:

```bash
git branch backup/before_v3d_authenticated_tactical_ablation_$(date +%Y%m%d_%H%M%S)
```

If sensitive SDK auth files were committed accidentally, document it and SAFE_STOP unless the user explicitly approves remediation.

## Phase 1 — Confirm replay environment

Run a minimal authenticated replay smoke before formal ablation.

Use the smallest existing command, for example:

```bash
python scripts/run_v3b_formal_tactical_ablation.py \
  --seeds 0 \
  --pairs 1 \
  --out reports/v3d_authenticated_tactical_ablation/preflight_smoke
```

If the script uses different arguments, inspect `--help` and run the equivalent smallest tactical-on/off paired smoke.

Create:

```text
reports/v3d_authenticated_tactical_ablation/01_preflight_smoke.md
reports/v3d_authenticated_tactical_ablation/preflight_smoke.json
```

Required checks:

* authentication passes;
* deployment phase entered;
* battle phase entered;
* game completes or progresses meaningfully;
* tactical-on invokes tactical branch;
* tactical-off does not invoke tactical branch;
* route diagnostics are recorded if available;
* outcome metric is emitted if available.

If replay regresses to auth/deployment-only blocker, stop with SAFE_STOP.

## Phase 2 — Formal paired protocol

Create or update:

```text
reports/v3d_authenticated_tactical_ablation/02_protocol.md
reports/v3d_authenticated_tactical_ablation/protocol.json
```

Protocol must specify:

* seed list;
* scenario/map selection;
* paired tactical-on/off design;
* identical non-tactical configuration;
* outcome metric;
* battle progression metric;
* tactical invocation metrics;
* route empty-rate metrics;
* illegal action metrics;
* confidence interval or bootstrap method;
* stopping rules;
* claim gates.

Minimum recommended run:

```text
5 seeds × 3 scenario/map pairs = 15 paired comparisons
```

If feasible, use more seeds, but do not run an uncontrolled long job.

## Phase 3 — Run authenticated multi-seed tactical ablation

Run the existing V3B formal runner under authenticated SDK.

Suggested command, adapted to actual CLI:

```bash
python scripts/run_v3b_formal_tactical_ablation.py \
  --seeds 0 1 2 3 4 \
  --pairs 3 \
  --out experiments/shenji_agent/v3d_authenticated_tactical_ablation
```

Outputs should be copied or summarized into:

```text
reports/v3d_authenticated_tactical_ablation/03_multiseed_ablation.md
reports/v3d_authenticated_tactical_ablation/multiseed_ablation.json
```

Required per-pair metrics:

* seed;
* scenario/map;
* tactical mode;
* deployment entered;
* battle entered;
* game completed;
* outcome metric;
* win/loss/draw if available;
* score/rating/objective if available;
* tactical invocation count;
* tactical nonzero score count;
* tactical changed action count if available;
* route empty count;
* route empty rate;
* illegal action count;
* runtime errors.

Required paired deltas:

* outcome delta = tactical-on minus tactical-off;
* win-rate delta if available;
* objective/score delta if available;
* route empty-rate delta;
* tactical invocation delta;
* changed-action delta if available.

## Phase 4 — Statistical summary

Use or extend:

```text
scripts/v3b_metrics.py
```

Create:

```text
reports/v3d_authenticated_tactical_ablation/04_statistics.md
reports/v3d_authenticated_tactical_ablation/statistics.json
```

Required:

* number of paired comparisons;
* number completed;
* completion rate;
* mean paired outcome delta;
* median paired outcome delta;
* standard deviation;
* bootstrap confidence interval if available;
* per-seed breakdown;
* failure modes;
* route metric comparison;
* tactical invocation comparison.

Do not overinterpret tiny samples.

## Phase 5 — Claim gate decision

Create:

```text
reports/v3d_authenticated_tactical_ablation/05_claim_gate.md
reports/v3d_authenticated_tactical_ablation/claim_gate.json
```

Use these gates:

### Outcome contribution supported

Allowed only if:

* authenticated paired replay completes;
* tactical-on shows positive outcome delta;
* effect is visible across multiple seeds or confidence interval is meaningfully positive;
* tactical branch is actually invoked;
* route/illegal action metrics do not degrade unacceptably.

State:

```text
TACTICAL_OUTCOME_CONTRIBUTION_SUPPORTED
```

### Weak or inconclusive

Use if:

* paired replay completes;
* tactical is invoked;
* but outcome delta is small, unstable, mixed, or confidence interval crosses zero.

State:

```text
TACTICAL_OUTCOME_CONTRIBUTION_WEAK_OR_INCONCLUSIVE
```

### No contribution shown

Use if:

* tactical is invoked;
* but outcome delta is zero or negative with no useful diagnostic benefit.

State:

```text
TACTICAL_OUTCOME_CONTRIBUTION_NOT_SHOWN
```

### Blocked

Use if:

* authenticated replay fails;
* battle phase cannot be entered;
* runner cannot produce paired outcomes.

State:

```text
TACTICAL_OUTCOME_ABLATION_BLOCKED
```

## Phase 6 — Update V3 evidence package

Update only evidence/claim-boundary files:

```text
paper/v3_evidence_package/TACTICAL_CONTRIBUTION_STATUS.md
paper/v3_evidence_package/V3_READINESS.md
paper/v3_evidence_package/claim_to_evidence_matrix_v3.md
paper/v3_evidence_package/results_tables_v3.csv
paper/v3_evidence_package/results_tables_v3.json
paper/source_evidence.md
```

Allowed wording depends on result.

If supported:

```text
Authenticated paired tactical-on/off evaluation shows bounded outcome-level tactical contribution under the tested protocol.
```

If weak/inconclusive:

```text
Authenticated paired tactical-on/off evaluation executes successfully and invokes the tactical branch, but outcome-level contribution remains weak or inconclusive under the tested protocol.
```

If not shown:

```text
Authenticated paired tactical-on/off evaluation invokes the tactical branch, but no outcome-level benefit is shown.
```

Forbidden unless proven:

```text
Tactical improves win rate generally.
Tactical is a main performance contribution.
The method is promotion-ready.
Two-window success is achieved.
Route reliability is solved.
```

## Phase 7 — Validation

Run:

```bash
python -m unittest discover -v
python -m compileall paper/scripts scripts ai tests training
```

If specific V3B/V3C/V3D tests exist, run them explicitly.

Create:

```text
reports/v3d_authenticated_tactical_ablation/07_validation_log.md
```

Record command outputs and exit codes.

## Phase 8 — Final decision

Create:

```text
reports/v3d_authenticated_tactical_ablation/FINAL_DECISION.md
reports/v3d_authenticated_tactical_ablation/final_decision.json
```

Use exactly one:

```text
RESUME_V3_WITH_TACTICAL_OUTCOME_CLAIM
```

Only if outcome contribution is supported.

```text
RESUME_V3_WITH_TACTICAL_DIAGNOSTIC_OR_WEAK_CLAIM
```

Use if outcome evidence is weak/inconclusive but tactical invocation/action-level effect remains valid.

```text
RESUME_V3_WITHOUT_TACTICAL_OUTCOME_CLAIM
```

Use if no outcome contribution is shown.

```text
SAFE_STOP_OUTCOME_ABLATION_BLOCKED
```

Use if authenticated formal ablation cannot run.

## Phase 9 — Commit

Before staging, check for large/sensitive files:

```bash
git status --short
git diff --cached --name-only | grep -Ei 'engine_config|license|auth|key|secret|token|\.pt$|\.pth$|\.zip$|\.tar$|\.tgz$|\.whl$' || true
```

Stage only reports, scripts, tests, and evidence package updates:

```bash
git add reports/v3d_authenticated_tactical_ablation
git add experiments/shenji_agent/v3d_authenticated_tactical_ablation 2>/dev/null || true
git add paper/v3_evidence_package paper/source_evidence.md 2>/dev/null || true
git add scripts tests 2>/dev/null || true
```

Unstage sensitive or large assets if accidentally staged:

```bash
git restore --staged '.engine_config' 2>/dev/null || true
git restore --staged '*license*' 2>/dev/null || true
git restore --staged '*secret*' 2>/dev/null || true
git restore --staged '*.pt' '*.pth' '*.zip' '*.tar' '*.tgz' '*.whl' 2>/dev/null || true
```

Commit:

```bash
git commit -m "rerun tactical outcome ablation with authenticated replay"
```

## SUCCESS criteria

SUCCESS if:

* authenticated replay preflight runs;
* formal paired tactical-on/off ablation is attempted;
* tactical invocation metrics are recorded;
* outcome deltas are computed;
* route and illegal-action metrics are recorded if available;
* claim gate decision is written;
* V3 evidence package is updated conservatively;
* tests and compileall pass;
* no unsupported tactical outcome claim is made.

## SAFE_STOP criteria

SAFE_STOP if:

* SDK authentication regresses;
* replay cannot enter battle phase;
* paired ablation cannot produce outcome metrics;
* scripts fail in a way that invalidates the experiment;
* sensitive auth files would need to be committed.

If SAFE_STOP, create:

```text
reports/v3d_authenticated_tactical_ablation/SAFE_STOP.md
```

It must state:

* what was attempted;
* exact blocker;
* whether V3C remains valid;
* what human action is needed.


