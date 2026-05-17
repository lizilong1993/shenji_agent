
# V3F Route Reliability Repair and Evaluation

## Objective

V3E completed claim freeze and ended with:

`CLAIMS_FROZEN_NEEDS_ROUTE_EVIDENCE`

Current evidence conflict:

- V3D authenticated tactical ablation:
  - 590 genuine route/move attempts
  - 0 route-empty events
  - 0% route-empty rate
  - no route degradation observed

- Archived league sidecar evidence:
  - 15641 / 23580 empty routes
  - route-empty rate = 66.3%
  - reason included `no_valid_move_candidate`
  - route reliability remains unsupported as a general claim

V3F must diagnose and, if possible, repair route reliability by identifying whether the earlier 66.3% empty-route evidence came from:

1. old SDK auth / deployment-phase artifact;
2. old runner bug;
3. different denominator;
4. invalid candidate generation;
5. legal-move filtering issue;
6. route planner bug;
7. policy fallback bug;
8. scenario/map-specific issue;
9. archived league protocol that remains reproducible today.

The task must produce executable route evidence under authenticated replay.

Do not train.
Do not change tactical claims.
Do not edit archived league artifacts by hand.
Do not fabricate route repair.
Do not claim route reliability is solved unless authenticated route evaluation supports it with denominator evidence.

## Source of truth

Read first:

- `reports/v3e_claim_freeze/FINAL_DECISION.md`
- `reports/v3e_claim_freeze/02_route_reliability_audit.md`
- `reports/v3e_claim_freeze/route_reliability_audit.json`
- `reports/v3e_claim_freeze/05_claim_matrix_freeze.md`
- `paper/v3_evidence_package/V3_READINESS.md`
- `paper/v3_evidence_package/claim_to_evidence_matrix_v3.md`
- `paper/source_evidence.md`
- `reports/v3d_authenticated_tactical_ablation/multiseed_ablation.json`
- `reports/v3d_authenticated_tactical_ablation/statistics.json`
- archived league sidecars under `experiments/shenji_agent/`
- `ai/policy.py`
- `ai/agent.py`
- route-related code in `training/`, `scripts/`, and `tests/`

## Hard constraints

Do not:

- train models;
- edit archived league sidecars;
- rewrite historical route evidence;
- claim route reliability is solved without fresh authenticated denominator evidence;
- claim tactical outcome contribution;
- claim two-window promotion success;
- claim public SOTA;
- claim top-tier readiness;
- run destructive git commands;
- modify SDK auth files unless required and documented;
- commit auth/license-sensitive files;
- commit large checkpoint files.

Allowed:

- inspect archived sidecars;
- add route diagnostics;
- add route evaluation scripts;
- run bounded authenticated route evaluation;
- repair route candidate generation or route diagnostic accounting if a bug is found;
- add tests;
- update evidence package conservatively.

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
reports/v3f_route_reliability/00_start_state.md
```

Record:

* current branch;
* HEAD;
* V3E commit `82e4ba0`;
* worktree status;
* V3E final decision;
* known route conflict;
* SDK auth status;
* whether V3D reports exist;
* whether archived league sidecars exist.

Create backup branch:

```bash
git branch backup/before_v3f_route_reliability_$(date +%Y%m%d_%H%M%S)
```

## Phase 1 — Route evidence inventory

Create:

```text
reports/v3f_route_reliability/01_route_evidence_inventory.md
reports/v3f_route_reliability/route_evidence_inventory.json
```

Inventory all route-related evidence from:

* V3D authenticated tactical ablation;
* archived league sidecars;
* Round-005 route diagnosis;
* V3E route audit;
* tests;
* scripts.

Run:

```bash
grep -RInE 'route_empty|empty_route|move_route|route_failure|no_valid_move_candidate|move_attempt|route_empty_rate|pathfinder|route planner|reachable|candidate' \
  reports experiments paper scripts ai training tests 2>/dev/null \
  > reports/v3f_route_reliability/route_reference_grep.txt || true
```

For each route evidence source, record:

```text
Evidence source
Path
Protocol
Authenticated replay?
Deployment phase or battle phase?
Move attempts
Route-planned attempts
Empty routes
Empty rate
Failure reasons
Scenario/map coverage
Can support general route claim?
Limitations
```

## Phase 2 — Reconcile V3D vs archived league discrepancy

Create:

```text
reports/v3f_route_reliability/02_route_discrepancy_analysis.md
reports/v3f_route_reliability/route_discrepancy_analysis.json
```

Answer these questions:

1. Did V3D and archived league use the same route diagnostic definitions?
2. Did V3D count all move attempts or only route-eligible moves?
3. Did archived league count route-empty for non-move actions?
4. Did archived league include deployment-phase or invalid-state artifacts?
5. Did archived league use the same policy code as current authenticated replay?
6. Did archived league use the same map/scenario set?
7. Did archived league fail due to `no_valid_move_candidate` before route planning?
8. Was the earlier 66.3% empty rate a candidate-generation issue rather than route-search failure?
9. Is the current 0% V3D evidence sufficient, or too narrow?

Classify discrepancy as exactly one:

```text
ROUTE_DISCREPANCY_OLD_RUNNER_ARTIFACT
ROUTE_DISCREPANCY_DENOMINATOR_MISMATCH
ROUTE_DISCREPANCY_CANDIDATE_GENERATION_FAILURE
ROUTE_DISCREPANCY_SCENARIO_COVERAGE_GAP
ROUTE_DISCREPANCY_UNRESOLVED
```

Do not force a favorable classification.

## Phase 3 — Route diagnostic instrumentation audit

Inspect route-related code.

Likely files:

* `ai/policy.py`
* `ai/agent.py`
* `training/maps.py`
* self-play / evaluation scripts
* V3B/V3D runner scripts
* route diagnostic tests

Create:

```text
reports/v3f_route_reliability/03_route_instrumentation_audit.md
reports/v3f_route_reliability/route_instrumentation_audit.json
```

Required audit points:

* where route attempts are counted;
* where empty routes are counted;
* where failure reasons are assigned;
* where `no_valid_move_candidate` is generated;
* whether denominator excludes non-route actions;
* whether route diagnostics reset correctly per game;
* whether tactical-on/off affects route counting;
* whether fallback actions are counted separately;
* whether deployment-phase actions are excluded;
* whether illegal/no-op actions are separated from route-empty events.

If instrumentation is wrong, fix it with tests.

Do not change historical sidecars.

## Phase 4 — Add or update route reliability evaluator

Create or update:

```text
scripts/run_v3f_route_reliability_evaluation.py
```

The evaluator must run authenticated replay with route metrics focused on route reliability.

It must support at least:

```bash
python scripts/run_v3f_route_reliability_evaluation.py \
  --seeds 0 1 2 3 4 \
  --pairs 3 \
  --out experiments/shenji_agent/v3f_route_reliability
```

Adapt CLI to project conventions if needed.

Required output:

```text
experiments/shenji_agent/v3f_route_reliability/
reports/v3f_route_reliability/04_route_eval.md
reports/v3f_route_reliability/route_eval.json
```

Required per-game metrics:

```text
seed
scenario_id
map_id
authenticated
deployment_entered
battle_entered
game_completed
turns
total_actions
move_actions
route_eligible_move_actions
route_planned_actions
route_empty_count
route_empty_rate
route_failure_reasons
no_valid_move_candidate_count
fallback_move_count
illegal_action_count
selected_by_learned_count
selected_by_rule_or_fallback_count
candidate_count_distribution
reachable_candidate_count_distribution
```

The evaluator must fail closed if:

* SDK auth fails;
* battle phase is not entered;
* route denominator is missing;
* scenario/map data is missing;
* policy diagnostics are unavailable.

## Phase 5 — Reproduce archived league route issue if possible

Create:

```text
reports/v3f_route_reliability/05_archived_route_reproduction.md
reports/v3f_route_reliability/archived_route_reproduction.json
```

Try to reproduce the archived 66.3% empty-route issue under current authenticated environment.

Use either:

* same scenario/map IDs from archived league if recoverable;
* same candidate/incumbent configuration if recoverable;
* closest current authenticated replay configuration.

Do not run a huge evaluation. Use a bounded reproduction set.

Suggested command:

```bash
python scripts/run_v3f_route_reliability_evaluation.py \
  --seeds 0 1 2 \
  --pairs 5 \
  --scenario-source archived_league \
  --out experiments/shenji_agent/v3f_route_reliability_archived_repro
```

If CLI differs, implement equivalent behavior.

Classify reproduction result:

```text
ARCHIVED_ROUTE_ISSUE_REPRODUCED
ARCHIVED_ROUTE_ISSUE_NOT_REPRODUCED
ARCHIVED_ROUTE_ISSUE_PARTIALLY_REPRODUCED
ARCHIVED_ROUTE_REPRODUCTION_BLOCKED
```

If reproduced, identify exact cause.

If not reproduced, explain why current authenticated runner differs.

## Phase 6 — Repair route reliability if a code bug is found

Only repair if evidence identifies a specific bug.

Possible bugs:

* valid move candidates filtered out too aggressively;
* route candidate generation ignores reachable cities;
* move type fallback incorrectly records route-empty;
* route planner called with wrong map ID;
* route diagnostics count no-candidate as empty-route incorrectly;
* deployment-phase actions contaminate route metrics;
* fallback no-op treated as route-empty;
* stale map/cost data path.

Allowed files to modify:

* `ai/policy.py`
* `ai/agent.py`
* `training/maps.py`
* route diagnostic scripts
* tests

Do not change model checkpoints.

For each repair, create:

```text
reports/v3f_route_reliability/06_route_repair.md
reports/v3f_route_reliability/route_repair.json
```

Document:

* bug;
* file changed;
* before behavior;
* after behavior;
* test added;
* whether archived issue is expected to change.

If no bug is found, write:

```text
NO_ROUTE_CODE_REPAIR_APPLIED
```

and explain why.

## Phase 7 — Route reliability evaluation after repair or confirmation

Run bounded authenticated evaluation after any repair, or after confirming no repair is needed.

Suggested command:

```bash
python scripts/run_v3f_route_reliability_evaluation.py \
  --seeds 0 1 2 3 4 \
  --pairs 5 \
  --out experiments/shenji_agent/v3f_route_reliability_final
```

Create:

```text
reports/v3f_route_reliability/07_final_route_evaluation.md
reports/v3f_route_reliability/final_route_evaluation.json
```

Required aggregate metrics:

```text
games_attempted
games_completed
battle_entered_count
total_move_actions
route_eligible_move_actions
route_planned_actions
route_empty_count
route_empty_rate
no_valid_move_candidate_count
fallback_move_count
illegal_action_count
mean_route_empty_rate
max_route_empty_rate_by_game
scenario_count
map_count
failure_reasons
confidence interval if feasible
```

## Phase 8 — Route claim gate

Create:

```text
reports/v3f_route_reliability/08_route_claim_gate.md
reports/v3f_route_reliability/route_claim_gate.json
```

Use exactly one route status:

```text
ROUTE_RELIABILITY_SUPPORTED_AUTHENTICATED
```

Use only if:

* authenticated replay enters battle;
* denominator is meaningful;
* route-empty rate is low under bounded evaluation;
* archived high empty-rate is explained or not reproduced;
* no major route failure modes remain in tested protocol.

```text
ROUTE_NO_DEGRADATION_OBSERVED_ONLY
```

Use if:

* current evaluation shows low/zero empty-route;
* but coverage is too narrow for general route reliability.

```text
ROUTE_RELIABILITY_PARTIALLY_SUPPORTED
```

Use if:

* route reliability improves or is bounded in some scenarios;
* but failures remain in specific maps/scenarios/protocols.

```text
ROUTE_RELIABILITY_NOT_SUPPORTED
```

Use if:

* high empty-route persists;
* candidate generation remains broken;
* route metrics remain unreliable.

```text
ROUTE_EVALUATION_BLOCKED
```

Use if:

* authenticated route evaluation cannot run.

Allowed claim only if supported:

```text
Authenticated route evaluation shows bounded route-empty behavior under the tested protocol.
```

Forbidden unless strongly supported:

```text
Route reliability is solved generally.
The route planner is robust across the SDK.
The archived 66.3% route failure is irrelevant.
```

## Phase 9 — Update V3 evidence package

Update conservatively:

```text
paper/v3_evidence_package/V3_READINESS.md
paper/v3_evidence_package/claim_to_evidence_matrix_v3.md
paper/v3_evidence_package/results_tables_v3.csv
paper/v3_evidence_package/results_tables_v3.json
paper/source_evidence.md
```

Required updates:

* route reliability status;
* relation between V3D 0% empty and archived 66.3% empty;
* route claim gate;
* whether manuscript readiness improved;
* remaining blockers.

Do not change tactical status except to preserve V3E/V3D conclusion.

## Phase 10 — Tests

Add or update tests:

```text
tests/test_v3f_route_reliability.py
```

Tests should cover:

* route denominator is nonzero when move actions occur;
* route-empty is not counted for non-move actions;
* no-valid-candidate is separated from route planner empty path;
* fallback move type is recorded separately;
* V3D route evidence is not generalized without claim gate;
* route claim gate cannot mark general support without denominator evidence.

Run:

```bash
python -m unittest discover -v
python -m compileall paper/scripts scripts ai tests training
```

Create:

```text
reports/v3f_route_reliability/10_validation_log.md
```

## Phase 11 — Final decision

Create:

```text
reports/v3f_route_reliability/FINAL_DECISION.md
reports/v3f_route_reliability/final_decision.json
```

Use exactly one final state:

```text
ROUTE_RELIABILITY_REPAIRED_READY_FOR_BASELINE_EXPANSION
```

Use if route reliability is supported or sufficiently bounded under authenticated evaluation.

```text
ROUTE_RELIABILITY_PARTIAL_NEEDS_MORE_COVERAGE
```

Use if current route results are improved but not broad enough.

```text
ROUTE_RELIABILITY_NOT_FIXED
```

Use if high empty-route persists or route failure remains central.

```text
ROUTE_EVIDENCE_RECONCILED_NO_CODE_REPAIR_NEEDED
```

Use if archived high route issue is explained as old runner/denominator/auth artifact and current authenticated route evaluation is bounded.

```text
SAFE_STOP_ROUTE_EVALUATION_BLOCKED
```

Use if route evaluation cannot run.

The final decision must recommend the next goal:

* `V3G baseline expansion`
* `V3G conservative manuscript rewrite`
* `V3G route coverage expansion`
* `SAFE_STOP route blocker`
* or `V3G promotion evidence rebuild`

## Phase 12 — Commit

Before staging:

```bash
git status --short
git diff --name-only
git diff --cached --name-only | grep -Ei 'engine_config|license|auth|key|secret|token|\.pt$|\.pth$|\.zip$|\.tar$|\.tgz$|\.whl$' || true
```

Stage only code, tests, reports, evidence text, and goal:

```bash
git add reports/v3f_route_reliability
git add experiments/shenji_agent/v3f_route_reliability 2>/dev/null || true
git add experiments/shenji_agent/v3f_route_reliability_archived_repro 2>/dev/null || true
git add experiments/shenji_agent/v3f_route_reliability_final 2>/dev/null || true
git add scripts/run_v3f_route_reliability_evaluation.py 2>/dev/null || true
git add ai/policy.py ai/agent.py training/maps.py 2>/dev/null || true
git add tests/test_v3f_route_reliability.py 2>/dev/null || true
git add paper/v3_evidence_package paper/source_evidence.md 2>/dev/null || true
git add goals/codex_goal_v3f_route_reliability_repair_and_evaluation.md 2>/dev/null || true
```

Unstage sensitive or large files:

```bash
git restore --staged '.engine_config' 2>/dev/null || true
git restore --staged '*license*' 2>/dev/null || true
git restore --staged '*secret*' 2>/dev/null || true
git restore --staged '*.pt' '*.pth' '*.zip' '*.tar' '*.tgz' '*.whl' 2>/dev/null || true
```

Commit:

```bash
git commit -m "evaluate and repair route reliability evidence"
```

## SUCCESS criteria

SUCCESS if:

* V3D vs archived route discrepancy is analyzed;
* route instrumentation is audited;
* route evaluator exists or current evaluator is extended;
* authenticated route evaluation is run or blocked with exact reason;
* archived high route issue is reproduced, not reproduced, or precisely blocked;
* code repair is applied only if a real bug is identified;
* route claim gate is written;
* V3 evidence package is updated conservatively;
* tests and compileall pass;
* no unsupported route claim is introduced.

## SAFE_STOP criteria

SAFE_STOP if:

* SDK auth regresses;
* route evaluation cannot enter battle phase;
* route diagnostics cannot produce denominator evidence;
* archived route sidecars are unreadable;
* route metrics are internally inconsistent;
* validation fails in a way that invalidates the route audit;
* sensitive SDK auth files would need to be committed.

If SAFE_STOP, create:

```text
reports/v3f_route_reliability/SAFE_STOP.md
```

It must state:

* what was attempted;
* exact blocker;
* current route claim boundary;
* whether route reliability remains forbidden;
* human action needed next.


